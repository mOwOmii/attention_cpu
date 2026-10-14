"""Submit two dependent jsub jobs; run only on their allocated JH_HOSTS."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from tools.launch_acceptance import load_inventory


def allocation(text):
    """The site's getJobHosts.sh defines JH_HOSTS as host/slot pairs."""
    fields=text.split()
    if not fields or len(fields)%2:
        raise ValueError('JH_HOSTS must be host/slot pairs (see site getJobHosts.sh)')
    result={}
    for host,count in zip(fields[::2],fields[1::2]):
        if host in result or not count.isdigit() or int(count)<=0:
            raise ValueError('invalid/duplicate JH_HOSTS allocation')
        result[host]=int(count)
    return result


def scheduler_name(node):
    # Explicit inventory field; never derive the allocation from communication IPs.
    return node.get('scheduler_host',node['host'])


def validate_allocation(nodes, allocated, slots=96):
    expected={scheduler_name(n) for n in nodes}
    if set(allocated)!=expected:
        raise ValueError(f'allocated hosts {sorted(allocated)} != requested hosts {sorted(expected)}')
    if any(v<slots for v in allocated.values()):
        raise ValueError(f'at least {slots} allocated slots required on each host')


def resource_threads(suite,slots,threads=None):
    if slots<2:
        raise ValueError('at least two slots per node are required')
    if suite=='max-parallel':
        if slots!=96 or threads not in (None,1):
            raise ValueError('maximum suite reserves 96 slots/node and uses one thread/worker')
        return 1
    threads=min(44,slots//2) if threads is None else threads
    if threads<1 or 2*threads>slots:
        raise ValueError('two workers must fit within allocated CPU slots')
    return threads


def job_script(data,nodes,suite,inventory,output,queue,slots=96,threads=None):
    threads=resource_threads(suite,slots,threads)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',queue):
        raise ValueError('invalid queue name')
    for node in nodes:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',scheduler_name(node)):
            raise ValueError('invalid scheduler hostname')
    command=[data['python'],'-m','tools.jsub_acceptance','run','--suite',suite,
             '--inventory',str(inventory),'--output',str(output),'--slots-per-node',str(slots),'--threads',str(threads)]
    return '\n'.join(['#!/bin/bash',f'#JSUB -J attention_{suite.replace("-","_")}',
        f'#JSUB -q {queue}',f'#JSUB -n {slots*len(nodes)}',f'#JSUB -R "span[ptile={slots}]"',
        '#JSUB -m "'+' '.join(scheduler_name(n) for n in nodes)+'"',
        '#JSUB -W 120',f'#JSUB -cwd {shlex.quote(data["workspace"])}',
        f'#JSUB -o {shlex.quote(str(output.parent/(suite+".%J.out")))}',
        f'#JSUB -e {shlex.quote(str(output.parent/(suite+".%J.err")))}',
        'set -euo pipefail',f'cd {shlex.quote(data["workspace"])}',shlex.join(command),''])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','run'])
    p.add_argument('--inventory',default='configs/acceptance_cluster.json')
    p.add_argument('--output')
    p.add_argument('--queue',default='normal')
    p.add_argument('--slots-per-node',type=int,default=96)
    p.add_argument('--threads',type=int)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--suite',choices=['correctness-performance','max-parallel'])
    args=p.parse_args()
    inventory=Path(args.inventory).resolve()
    if args.action=='run':
        if not args.suite or not args.output or not os.environ.get('JH_JOBID'):
            raise ValueError('run requires suite, output and a real JH_JOBID allocation')
        data,nodes=load_inventory(inventory,args.suite)
        allocated=allocation(os.environ.get('JH_HOSTS',''))
        threads=resource_threads(args.suite,args.slots_per_node,args.threads)
        validate_allocation(nodes,allocated,slots=args.slots_per_node)
        root=Path(args.output).resolve()
        root.parent.mkdir(parents=True,exist_ok=True)
        (root.parent/f'{args.suite}_allocation.json').write_text(json.dumps(dict(
            job_id=os.environ['JH_JOBID'],allocation=allocated,queue=os.environ.get('JH_QUEUE')),indent=2)+'\n')
        command=[data['python'],'-m','tools.launch_acceptance','--inventory',str(inventory),
                 '--suite',args.suite,'--output',str(root),'--transport','jexec','--execute',
                 '--port',str(20000+int(os.environ['JH_JOBID'])%20000),'--threads',str(threads)]
        if args.suite=='max-parallel':
            command+=['--workers-per-node','96','--threads','1','--capacity-search']
        raise SystemExit(subprocess.call(command,cwd=data['workspace']))
    root=Path(args.output or 'runs/jsub_acceptance_'+datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    root.mkdir(parents=True,exist_ok=False)
    suites=[args.suite] if args.suite else ['correctness-performance','max-parallel']
    scripts=[]
    for suite in suites:
        data,nodes=load_inventory(inventory,suite)
        path=root/(suite+'.jsub')
        path.write_text(job_script(data,nodes,suite,inventory,root/suite,args.queue,args.slots_per_node,args.threads))
        scripts.append(path)
    if args.dry_run:
        print(f'Prepared jsub scripts: {root}; no jobs submitted')
        return
    jsub=shutil.which('jsub') or '/opt/jhinno/unischeduler/linux-x86_64/bin/jsub'
    results=[]
    dependency=None
    for suite,path in zip(suites,scripts):
        command=[jsub]+(['-w',f'done({dependency})'] if dependency else [])
        with path.open() as script:
            try:
                proc=subprocess.run(command,stdin=script,capture_output=True,text=True,timeout=60)
            except subprocess.TimeoutExpired as exc:
                partial=''.join(v.decode(errors='replace') if isinstance(v,bytes) else (v or '') for v in (exc.stdout,exc.stderr))
                (root/(suite+'.submit.log')).write_text(partial)
                results.append(dict(suite=suite,status='SUBMISSION_UNKNOWN',error='jsub timed out; inspect jjobs before retrying'))
                (root/'submission.json').write_text(json.dumps(results,indent=2)+'\n')
                raise SystemExit(str(exc))
        (root/(suite+'.submit.log')).write_text(proc.stdout+proc.stderr)
        match=re.search(r'\bJob\s*<(\d+)>',proc.stdout,re.I)
        results.append(dict(suite=suite,returncode=proc.returncode,job_id=match.group(1) if match else None,
                            dependency=dependency,status='SUBMITTED' if proc.returncode==0 and match else 'SUBMISSION_FAILED_OR_UNKNOWN'))
        (root/'submission.json').write_text(json.dumps(results,indent=2)+'\n')
        print(proc.stdout+proc.stderr,flush=True)
        if proc.returncode or not match:
            raise SystemExit('Submission not confirmed; inspect logs/jjobs before retrying. No dependent job submitted.')
        dependency=match.group(1)
    print(f'{len(results)} job(s) submitted; records: {root}')


if __name__=='__main__':
    main()
