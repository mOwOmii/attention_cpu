"""Preview/launch fixed-node jobs; optional short capacity search on all 16 nodes.

Default prints commands. --execute authorizes SSH execution; --node-rank launches
one agent locally (also usable under a scheduler). Inventory must be shared.
"""
import argparse
import json
import ipaddress
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_inventory(path, suite):
    data=json.loads(Path(path).read_text())
    expected=2 if suite=='correctness-performance' else 16
    nodes=data['nodes'][:expected]
    if len(nodes)!=expected or len({n['host'] for n in nodes})!=expected:
        raise ValueError(f'inventory requires {expected} distinct nodes')
    for node in nodes:
        if not node['host'] or node['host'].startswith('-') or not node['interface'] or 'REPLACE' in node['interface'] or 'REPLACE' in node['host']:
            raise ValueError('fill real host/interface names before launching')
    for node in nodes:
        if node.get('ip'):
            ipaddress.ip_address(node['ip'])
        if node['interface']=='auto' and not node.get('ip'):
            raise ValueError('interface=auto requires the node IP')
    addresses=[n.get('ip',n['host']) for n in nodes]
    if len(set(addresses))!=expected:
        raise ValueError('duplicate node connection address')
    for field in ['python','workspace']:
        if not Path(data[field]).is_absolute():
            raise ValueError(f'{field} must be absolute')
    if not data.get('master_addr') or 'REPLACE' in data['master_addr']:
        raise ValueError('set master_addr to a routable master-node address')
    return data,nodes


def resolve_interface(node):
    """Resolve only an interface owning this node's configured communication IP."""
    if node['interface'] != 'auto':
        return node['interface']
    result=subprocess.run(['ip','-j','-4','addr','show'],capture_output=True,text=True,check=True)
    matches={row['ifname'] for row in json.loads(result.stdout)
             if any(addr.get('local')==node['ip'] for addr in row.get('addr_info',[]))}
    if len(matches)!=1:
        raise ValueError(f"expected exactly one local interface owning {node['ip']}; found {sorted(matches)}")
    return matches.pop()


def node_command(args, data, nodes, workers, output, probe):
    # All nodes use one shared environment path; no reliance on SSH shell activation.
    python=data['python']
    script=str(Path(data['workspace'])/'scripts/bind_cpu_worker.sh')
    return [python,'-m','torch.distributed.run',f'--nnodes={len(nodes)}',f'--nproc-per-node={workers}',
            f'--node-rank={args.node_rank}',f'--master-addr={data["master_addr"]}',
            f'--master-port={args.port}','--max-restarts=0','--no-python',script,
            '--threads',str(args.threads),'--',python,'-m','tools.acceptance_worker',
            '--suite',args.suite,'--output',str(output),'--threads',str(args.threads),
            '--warmup',str(args.warmup),'--steps',str(args.steps),'--timeout',str(args.collective_timeout)
            ]+(['--probe-only'] if probe else [])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True)
    p.add_argument('--suite',choices=['correctness-performance','max-parallel'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers-per-node',type=int)
    p.add_argument('--threads',type=int)
    p.add_argument('--port',type=int,default=29500)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--collective-timeout',type=int,default=180)
    p.add_argument('--job-timeout',type=int,default=1800)
    p.add_argument('--node-rank',type=int)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--transport',choices=['ssh','jexec'],default='ssh')
    p.add_argument('--ssh-config',help='optional SSH config path; /dev/null skips broken system config')
    p.add_argument('--capacity-search',action='store_true',help='halving short probes, then formal run at highest observed success')
    p.add_argument('--probe-only',action='store_true')
    args=p.parse_args()
    data,nodes=load_inventory(args.inventory,args.suite)
    if args.transport=='jexec':
        from tools.jsub_acceptance import allocation, validate_allocation
        if not os.environ.get('JH_JOBID'):
            raise ValueError('jexec transport requires an active jsub allocation')
    workers=args.workers_per_node or (2 if args.suite=='correctness-performance' else 96)
    args.threads=args.threads or (44 if args.suite=='correctness-performance' else 1)
    if workers<=0 or args.threads<=0 or not 1<=args.port<=65535:
        raise ValueError('invalid workers/threads/port')
    if args.suite=='correctness-performance' and (workers!=2 or args.capacity_search or args.probe_only):
        raise ValueError('first suite uses two workers per node, no capacity probes')
    if args.suite=='max-parallel' and (workers%2 or workers>96 or args.threads!=1):
        raise ValueError('maximum suite uses even P<=96 and threads=1')
    if args.transport=='jexec':
        validate_allocation(nodes,allocation(os.environ.get('JH_HOSTS','')),slots=workers*args.threads)
    root=Path(args.output).resolve()
    if args.node_rank is not None:
        if not 0<=args.node_rank<len(nodes):
            raise ValueError('node rank out of range')
        from tools.bind_acceptance_worker import physical_cpus
        if len(physical_cpus()) < workers*args.threads:
            raise ValueError('insufficient disjoint physical cores for requested workers/threads')
        env=os.environ.copy()
        env.update(ACCEPTANCE_PYTHON=data['python'],GLOO_SOCKET_IFNAME=resolve_interface(nodes[args.node_rank]),
                   OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads))
        process=subprocess.Popen(node_command(args,data,nodes,workers,root,args.probe_only),
                                 cwd=data['workspace'],env=env,start_new_session=True)
        def interrupted(signum,frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        code=1
        try:
            code=process.wait(timeout=args.job_timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            code=124
        finally:
            # Only this node's job process group; never broad pkill by executable.
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise SystemExit(code)
    if args.execute:
        root.mkdir(parents=True,exist_ok=False)
    attempts=[]
    def launch(pcount,directory,probe):
        commands=[]
        for rank,node in enumerate(nodes):
            remote=[data['python'],'-m','tools.launch_acceptance','--inventory',str(Path(args.inventory).resolve()),
                    '--suite',args.suite,'--output',str(directory),'--workers-per-node',str(pcount),
                    '--threads',str(args.threads),'--node-rank',str(rank),'--port',str(args.port),
                    '--steps',str(args.steps),'--warmup',str(args.warmup),
                    '--collective-timeout',str(args.collective_timeout),'--job-timeout',str(args.job_timeout)]
            if probe:
                remote+=['--probe-only']
            # timeout bounds remote process lifetime even if the SSH controller disappears.
            shell='cd '+shlex.quote(data['workspace'])+' && '+shlex.join(
                ['timeout','--signal=TERM','--kill-after=30',str(args.job_timeout+30)]+remote)
            if args.transport=='jexec':
                from tools.jsub_acceptance import scheduler_name
                guard=[data['python'],'-m','tools.scheduler_node_guard','--lease',str(directory/'controller_lease.json'),'--']+remote
                guarded='cd '+shlex.quote(data['workspace'])+' && '+shlex.join(guard)
                jexec=str(Path(os.environ.get('JHSCHEDULER_BIN','/opt/jhinno/unischeduler/linux-x86_64/bin'))/'jexec')
                commands.append([jexec,'-t',str(args.job_timeout+120),'-m',scheduler_name(node),'/bin/bash','-c',guarded])
            else:
                commands.append(['ssh']+(['-F',args.ssh_config] if args.ssh_config else [])
                                +['-o','BatchMode=yes','-o','ConnectTimeout=15',node.get('ip',node['host']),shell])
        if not args.execute:
            for cmd in commands:
                print(shlex.join(cmd))
            return False
        directory.mkdir(parents=True,exist_ok=False)
        logs=[]; processes=[]
        lease=directory/'controller_lease.json'
        def heartbeat(active=True):
            temp=lease.with_suffix('.tmp')
            save(temp,dict(active=active,tick=time.monotonic_ns()))
            temp.replace(lease)
        if args.transport=='jexec':
            heartbeat()
        started=time.monotonic()
        try:
            for rank,cmd in enumerate(commands):
                stream=open(directory/f'node_{rank}.log','w')
                logs.append(stream)
                processes.append(subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT))
            deadline=started+args.job_timeout+90
            if args.transport=='jexec':
                while any(proc.poll() is None for proc in processes):
                    heartbeat(active=not any(proc.poll() not in (None,0) for proc in processes))
                    if time.monotonic()>deadline:
                        raise subprocess.TimeoutExpired('jexec job',args.job_timeout)
                    time.sleep(2)
            else:
                for proc in processes:
                    proc.wait(timeout=max(1,deadline-time.monotonic()))
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            for proc in processes:
                proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise  # Do not launch another topology while old remote workers may remain.
        finally:
            if args.transport=='jexec':
                heartbeat(active=False)
            for stream in logs:
                stream.close()
        if any(proc.returncode in (124,137,255) for proc in processes):
            for rank,proc in enumerate(processes):
                if proc.returncode:
                    log=directory/f'node_{rank}.log'
                    print(f'Node {nodes[rank]["host"]} ({nodes[rank].get("ip",nodes[rank]["host"])}) '
                          f'exit={proc.returncode}; log={log}',file=sys.stderr)
                    print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]),file=sys.stderr)
            attempts.append(dict(workers_per_node=pcount,world_size=len(nodes)*pcount,probe=probe,
                                 output=str(directory),passed=False,reason='remote_timeout_or_ssh_failure',
                                 exit_codes=[proc.returncode for proc in processes],seconds=time.monotonic()-started))
            save(root/'capacity_attempts.json',attempts)
            save(root/'launch_summary.json',dict(suite=args.suite,passed=False,
                 maximum_observed_world_size=None,proven_maximum=False,attempts=attempts))
            raise SystemExit('Remote timeout/SSH failure: inspect node logs before retrying.')
        ok=all(proc.returncode==0 for proc in processes)
        reason='completed' if ok else 'worker_or_launch_failure'
        if ok:
            result=subprocess.run([data['python'],'-m','tools.aggregate_acceptance',str(directory)],cwd=data['workspace'])
            ok=result.returncode==0
            reason='validated' if ok else 'evidence_validation_failed'
        attempts.append(dict(workers_per_node=pcount,world_size=len(nodes)*pcount,probe=probe,
                             output=str(directory),passed=ok,reason=reason,
                             exit_codes=[proc.returncode for proc in processes],seconds=time.monotonic()-started))
        save(root/'capacity_attempts.json',attempts)
        return ok
    if args.capacity_search:
        while workers>=2:
            success=launch(workers,root/f'probe_p{workers}',True)
            if not args.execute:
                print('Preview only: failed resource probes may halve even P; final selected P uses a formal run.')
                return
            if success:
                break
            # Only explicitly recognizable resource failures allow automatic fallback.
            combined='\n'.join(p.read_text(errors='replace') for p in (root/f'probe_p{workers}').glob('node_*.log'))
            resource_failure=any(marker in combined for marker in ['insufficient disjoint physical cores',
                'Cannot allocate memory','Resource temporarily unavailable','Too many open files'])
            if not resource_failure:
                raise SystemExit('Probe failed without clear capacity evidence; inspect logs before retrying.')
            workers=(workers//2)//2*2
        if workers<2:
            raise SystemExit('No successful capacity probe')
    success=launch(workers,root/'formal',args.probe_only)
    if args.execute:
        save(root/'launch_summary.json',dict(suite=args.suite,selected_workers_per_node=workers,
             maximum_observed_world_size=len(nodes)*workers if success else None,
             proven_maximum=False,passed=success,attempts=attempts))
        raise SystemExit(0 if success else 1)


if __name__=='__main__':
    main()
