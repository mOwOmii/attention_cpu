"""Bind each worker to disjoint physical cores before importing PyTorch."""
import argparse
import os
from pathlib import Path


def physical_cpus(allowed=None):
    result={}
    for cpu in sorted(os.sched_getaffinity(0) if allowed is None else allowed):
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=(int((base/'physical_package_id').read_text()),int((base/'core_id').read_text()))
        result.setdefault(key,cpu)
    return [result[k] for k in sorted(result)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    rank=int(os.environ.get('LOCAL_RANK',0))
    workers=int(os.environ.get('LOCAL_WORLD_SIZE',1))
    cores=physical_cpus()
    first=rank*len(cores)//workers
    last=(rank+1)*len(cores)//workers
    assigned=cores[first:last][:args.threads]
    if args.threads<=0 or len(assigned)!=args.threads or not command:
        raise SystemExit('insufficient disjoint physical cores or missing command')
    os.sched_setaffinity(0,assigned)
    os.environ.update(OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
                      OPENBLAS_NUM_THREADS=str(args.threads))
    # Allocations occur after affinity; Linux first-touch gives local placement.
    # No strict membind claim: that would require numactl/libnuma and policy checks.
    os.execvp(command[0],command)


if __name__=='__main__':
    main()
