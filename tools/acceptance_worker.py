"""torchrun entry: two acceptance suites; --local-smoke never claims cluster PASS."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.distributed as dist

from attention_cpu.config import AttentionConfig
from attention_cpu.parallel import initialize_parallel, ParallelContext
from attention_cpu.acceptance.common import (atomic_json, barrier, digest, gather, identity, MeasuredContext)
from attention_cpu.acceptance.runner import benchmark, numerical


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['correctness-performance','max-parallel'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/acceptance_performance.json')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--local-smoke', action='store_true')
    p.add_argument('--probe-only', action='store_true')
    p.add_argument('--no-profile', action='store_true', help='diagnostic only; max-parallel formal PASS requires traces')
    return p


def main():
    args = parser().parse_args()
    root = Path(args.output).resolve()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if min(args.threads,args.steps,args.timeout) <= 0 or args.warmup < 0:
        raise ValueError('threads/steps/timeout must be positive, warmup nonnegative')
    if args.local_smoke and world != 1:
        raise ValueError('--local-smoke requires one process')
    if args.suite=='max-parallel' and args.threads!=1:
        raise ValueError('maximum parallel suite requires one compute thread per worker')
    if args.probe_only and args.suite != 'max-parallel':
        raise ValueError('--probe-only belongs to max-parallel')
    topologies = ([(1,1)] if args.local_smoke else [(world,1),(world//2,2)]
                  if args.suite == 'correctness-performance' else [(32,world//32)])
    if not args.local_smoke and (world not in (2,4,8,16,32) if args.suite == 'correctness-performance'
                                 else world < 32 or world % 32):
        raise ValueError('correctness-performance requires W in {2,4,8,16,32}; max-parallel requires W divisible by 32')
    root.mkdir(parents=True,exist_ok=True)
    report = dict(rank=rank, suite=args.suite, numerical=[], benchmarks=[], status='FAIL')
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        started = time.perf_counter()
        ctx = MeasuredContext(initialize_parallel(*topologies[0], timeout_seconds=args.timeout))
        if rank == 0:
            (root/'.worker_claim').mkdir()  # reject reused result directories
        barrier()
        me = identity()
        peers = gather(me)
        expected_nodes = 1 if args.local_smoke else 2 if args.suite == 'correctness-performance' else 16
        if len({r['hostname'] for r in peers}) != expected_nodes:
            raise ValueError(f'expected {expected_nodes} distinct physical hostnames')
        if len({r['source_sha256'] for r in peers}) != 1 or len({r['torch'] for r in peers}) != 1:
            raise ValueError('source or PyTorch version differs across ranks')
        cfg = AttentionConfig.from_json(args.config)
        if cfg.dtype != 'float32' or cfg.input_layout != 'BSE' or cfg.dropout != 0 or cfg.num_heads != 32 or cfg.embed_dim != 512:
            raise ValueError('performance config requires float32/BSE/dropout=0/E=512/H=32')
        signature = dict(suite=args.suite,config=cfg.to_dict(),threads=args.threads,warmup=args.warmup,
                         steps=args.steps,seed=args.seed,output=str(root),probe=args.probe_only,
                         profile=not args.no_profile,local_smoke=args.local_smoke)
        if any(v != signature for v in gather(signature)):
            raise ValueError('worker arguments/config differ across ranks')
        # Verify physical core disjointness for local worker affinity sets.
        if args.threads > len(me['affinity']):
            raise ValueError('threads exceed worker affinity')
        for i,left in enumerate(peers):
            for right in peers[i+1:]:
                if left['hostname']==right['hostname'] and set(left['affinity']) & set(right['affinity']):
                    raise ValueError('worker CPU bindings overlap; use the provided launcher')
        atomic_json(root/f'preflight_rank_{rank}.json',me)
        barrier()
        shared_ok = ((all((root/f'preflight_rank_{r}.json').exists() for r in range(world)))
                     if rank==0 else (root/'preflight_rank_0.json').exists())
        if not all(gather(shared_ok)):
            raise ValueError('output directory is not shared across all workers')
        # Real collective probe, without opening another distributed job.
        total = torch.tensor(rank,dtype=torch.int64)
        if dist.is_initialized():
            dist.all_reduce(total)
        if total.item() != world*(world-1)//2:
            raise ValueError('global collective rank sum mismatch')
        manifest = dict(schema=1, **signature, world_size=world, expected_nodes=expected_nodes,
                        topologies=topologies, peers=peers, config_sha256=digest(args.config),
                        transport='local_single_process' if world==1 else 'gloo',
                        reference_performance_required=args.suite=='correctness-performance',
                        initialization_seconds=time.perf_counter()-started)
        if rank==0:
            atomic_json(root/'manifest.json',manifest)
        report.update(identity=me,manifest_sha256=None)
        barrier()
        report['manifest_sha256']=digest(root/'manifest.json')
        if args.suite=='correctness-performance':
            # Independent timing baseline uses a context with no distributed calls.
            if rank==0:
                torch.set_num_threads(args.threads)
                baseline=benchmark(cfg,MeasuredContext(ParallelContext()),root/'reference_performance',
                                   warmup=args.warmup,steps=args.steps,profile=False,seed=args.seed)
                atomic_json(root/'reference_performance.json',baseline)
            barrier()
        for index,(tp,dp) in enumerate(topologies):
            if index:
                ctx=MeasuredContext(initialize_parallel(tp,dp,timeout_seconds=args.timeout))
            name=f'tp{tp}_dp{dp}'
            report.setdefault('groups',[]).append(ctx.metadata())
            if not args.probe_only:
                torch.set_num_threads(1)
                if args.suite=='correctness-performance':
                    for dtype in ['float64','float32']:
                        for dropout in [0.0,0.2]:
                            numcfg=AttentionConfig(embed_dim=128,num_heads=32,rope=True,dtype=dtype,dropout=dropout)
                            result=numerical(numcfg,ctx,root/'references'/name/f'{dtype}_dropout{dropout}',seed=args.seed)
                            report['numerical'].append(dict(case=f'{name}_{dtype}_{dropout}',**result))
                            if not result['all_ranks_passed']:
                                raise AssertionError('numerical comparison failed; see rank report')
                else:
                    result=numerical(cfg,ctx,root/'references'/name,batch=96,seq=128,
                                     accumulation=1,seed=args.seed,variable=False)
                    report['numerical'].append(dict(case=name,**result))
                    if not result['all_ranks_passed']:
                        raise AssertionError('maximum parallel numerical comparison failed')
            torch.set_num_threads(args.threads)
            row=benchmark(cfg,ctx,root/name,batch=96 if args.suite=='max-parallel' else 8,
                          warmup=0 if args.probe_only else args.warmup,
                          steps=3 if args.probe_only else args.steps,seed=args.seed,probe=args.probe_only,
                          profile=args.suite=='max-parallel' and not args.no_profile and not args.probe_only)
            report['benchmarks'].append(dict(case=name,**row))
            if not row['all_ranks_finite']:
                raise AssertionError('nonfinite benchmark result')
        report['status']='LOCAL_ONLY' if args.local_smoke else 'PROBE_PASS' if args.probe_only else 'PASS'
        atomic_json(root/'rank_reports'/f'rank_{rank}.json',report)
        barrier()
        if rank==0:
            print(f'{args.suite}: {report["status"]}; reports: {root}',flush=True)
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc())
        atomic_json(root/'rank_reports'/f'rank_{rank}.json',report)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':
    main()
