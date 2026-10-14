import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import platform
import sys
import time

import torch
import torch.distributed as dist

from ..analyze import write_json
from ..parallel.groups import ParallelContext
from ..profiling import scope


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    paths=[]
    for directory in ['attention_cpu','tools','scripts']:
        paths.extend(p for p in (root/directory).rglob('*') if p.suffix in ('.py','.sh'))
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    write_json(tmp, value)
    tmp.replace(path)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather(value):
    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = None
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                rss = int(line.split()[1]) * 1024
    except OSError:
        pass
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':')[0] in ('MemTotal','MemAvailable'):
            memory[line.split(':')[0]]=int(line.split()[1])*1024
    return dict(node_memory=memory, rss_bytes=rss, peak_rss_bytes=usage.ru_maxrss * 1024,
                cpu_seconds=usage.ru_utime + usage.ru_stime,
                voluntary_switches=usage.ru_nvcsw, involuntary_switches=usage.ru_nivcsw,
                os_threads=len(list(Path('/proc/self/task').iterdir())))


def identity():
    return dict(hostname=socket.gethostname(), rank=int(os.environ.get('RANK', 0)),
                local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                node_rank=int(os.environ.get('GROUP_RANK', 0)), pid=os.getpid(),
                affinity=sorted(os.sched_getaffinity(0)), torch=str(torch.__version__),
                python=platform.python_version(), executable=sys.executable,
                limits={k:list(resource.getrlimit(v)) for k,v in [('nofile',resource.RLIMIT_NOFILE),('nproc',resource.RLIMIT_NPROC)]},
                source_sha256=source_digest())


class MeasuredContext(ParallelContext):
    def __init__(self, context):
        super().__init__(**vars(context))
        self.events = []
        self.measure = False

    def sum(self, tensor, domain='tp'):
        size = self.tp_size if domain == 'tp' else self.dp_size
        start = time.perf_counter()
        with scope(domain + '_all_reduce'):
            result = super().sum(tensor, domain)
        if self.measure and size > 1:
            self.events.append(dict(domain=domain, seconds=time.perf_counter()-start,
                                    logical_bytes=tensor.numel()*tensor.element_size()))
        return result


def ddp_hook(context, bucket):
    # Delegate to PyTorch's default averaging hook; the returned Future remains
    # asynchronous. Duration includes completion waiting, NOT isolated wire time.
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
    start = time.perf_counter()
    size = bucket.buffer().numel() * bucket.buffer().element_size()
    measure = context.measure
    with scope('dp_all_reduce_submit'):
        future = allreduce_hook(context.dp_group, bucket)
    def done(fut):
        if measure:
            context.events.append(dict(domain='dp', seconds=time.perf_counter()-start, logical_bytes=size))
        return fut.value()
    return future.then(done)
