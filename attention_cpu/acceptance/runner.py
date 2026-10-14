"""Numerical comparisons and steady-state measurements using production TP/DDP."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from ..attention import CPUSelfAttention
from ..analyze import compare_tensors
from ..parallel.attention import TensorParallelSelfAttention
from ..parallel.train import batch_range, make_training_batch, normalized_loss, step_dropout_mask
from ..profiling import scope, enabled
from .common import atomic_json, barrier, ddp_hook, digest, gather, resources


def executor_for(config, context, state=None, seed=7):
    torch.manual_seed(seed)
    model = TensorParallelSelfAttention(config, context)
    if state is not None:
        model.load_full_state_dict(state)
    executor = model
    if context.dp_size > 1:
        executor = DistributedDataParallel(model, device_ids=None, process_group=context.dp_group,
                                          broadcast_buffers=False)
        executor.register_comm_hook(context, ddp_hook)
    return model, executor


def reference(config, batch, seq, seed, variable, root, dp_size):
    """Only rank zero builds the reference, in chunks to bound activation memory."""
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = CPUSelfAttention(config)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x, target, valid = make_training_batch(config, batch, seq, seed, variable)
    mask = step_dropout_mask(config, batch, seq, seed, 0)
    count = valid.sum().item() * config.embed_dim
    outputs, dx, total = [], [], 0.0
    for first in range(0, batch, 8):
        last = min(first + 8, batch)
        data = x[first:last].clone().requires_grad_(True)
        y = model(data, key_padding_mask=~valid[first:last],
                  dropout_mask=None if mask is None else mask[first:last])
        loss, _ = normalized_loss(y, target[first:last], valid[first:last], global_count=count, dp_size=1)
        loss.backward()
        total += loss.item()
        outputs.append(y.detach())
        dx.append(data.grad.detach())
    outputs, dx = torch.cat(outputs), torch.cat(dx)
    gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters()}
    torch.save(dict(state=state, gradients=gradients, loss=total, count=count), root/'weights.pt')
    hashes = {'weights.pt': digest(root/'weights.pt')}
    for dp in range(dp_size):
        a, b = batch_range(batch, dp_size, dp)
        path = root/f'data_dp_{dp}.pt'
        torch.save(dict(x=x[a:b], target=target[a:b], valid=valid[a:b], output=outputs[a:b],
                        dx=dx[a:b], mask=None if mask is None else mask[a:b], begin=a, end=b), path)
        hashes[path.name] = digest(path)
    atomic_json(root/'manifest.json', dict(config=config.to_dict(), batch=batch, seq=seq,
                                         seed=seed, hashes=hashes))


def numerical(config, context, root, *, batch=9, seq=17, accumulation=2, seed=7, variable=True):
    import json
    if batch < context.dp_size * accumulation:
        raise ValueError('batch must cover every DP microbatch')
    if config.input_layout != 'BSE':
        raise ValueError('acceptance profiles currently require BSE')
    if context.rank == 0:
        reference(config, batch, seq, seed, variable, root, context.dp_size)
    barrier()
    manifest = json.loads((root/'manifest.json').read_text())
    paths = [root/'weights.pt', root/f'data_dp_{context.dp_rank}.pt']
    for path in paths:
        if digest(path) != manifest['hashes'][path.name]:
            raise ValueError(f'reference checksum mismatch: {path}')
    weights, data = [torch.load(path, weights_only=True) for path in paths]
    model, executor = executor_for(config, context, weights['state'], seed=seed)
    source = data['x'].clone().requires_grad_(True)
    checks = {}
    tol = dict(atol=1e-10, rtol=1e-8) if config.dtype == 'float64' else dict(atol=1e-5, rtol=1e-4)
    numerator = torch.zeros((), dtype=torch.float64)
    for micro in range(accumulation):
        a, b = batch_range(len(source), accumulation, micro)
        sync = nullcontext() if micro == accumulation-1 or context.dp_size == 1 else executor.no_sync()
        with sync:
            y = executor(source[a:b], key_padding_mask=~data['valid'][a:b],
                         dropout_mask=None if data['mask'] is None else data['mask'][a:b])
            checks[f'output.micro_{micro}'] = compare_tensors(y, data['output'][a:b], **tol)
            loss, num = normalized_loss(y, data['target'][a:b], data['valid'][a:b],
                                       global_count=weights['count'], dp_size=context.dp_size)
            loss.backward()
            numerator += num.double()
    context.sum(numerator, 'dp')
    checks['loss'] = compare_tensors(numerator/weights['count'], torch.tensor(weights['loss'], dtype=torch.float64), **tol)
    checks['dInput'] = compare_tensors(source.grad/context.dp_size, data['dx'], **tol)
    wanted = model.shard_state_dict(weights['gradients'])
    for name, p in model.named_parameters():
        checks['gradient.'+name] = compare_tensors(p.grad, wanted[name], **tol)
    torch.optim.SGD(model.parameters(), lr=0.03).step()
    updated = model.shard_state_dict({k: v-0.03*weights['gradients'][k] for k,v in weights['state'].items()})
    for name, p in model.named_parameters():
        checks['updated.'+name] = compare_tensors(p, updated[name], **tol)
    result = dict(config=config.to_dict(), batch=batch, seq=seq, accumulation=accumulation,
                  begin=data['begin'], end=data['end'], tolerance=tol, optimizer="SGD", lr=0.03,
                  reference_sha256=digest(root/'manifest.json'), checks=checks,
                  passed=all(c['passed'] for c in checks.values()))
    # All ranks finish the same collectives before acting on a numeric failure.
    statuses = gather(result['passed'])
    result['all_ranks_passed'] = all(statuses)
    return result


def benchmark(config, context, root, *, batch=8, seq=128, warmup=10, steps=50,
              profile=True, seed=7, probe=False):
    sync = barrier if context.world_size > 1 else lambda: None
    collect = gather if context.world_size > 1 else lambda value: [value]
    if batch < context.dp_size:
        raise ValueError('batch must be >= DP')
    model, executor = executor_for(config, context, seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    x, target, valid = make_training_batch(config, batch, seq, seed, False)
    a,b = batch_range(batch, context.dp_size, context.dp_rank)
    x, target, valid = x[a:b], target[a:b], valid[a:b]
    count = batch * seq * config.embed_dim
    def step():
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with scope('forward'):
            y = executor(x, key_padding_mask=~valid)
        t1 = time.perf_counter()
        with scope('loss'):
            loss,_ = normalized_loss(y, target, valid, global_count=count, dp_size=context.dp_size)
        t2 = time.perf_counter()
        with scope('backward'):
            loss.backward()
        t3 = time.perf_counter()
        with scope('optimizer'):
            optimizer.step()
        t4 = time.perf_counter()
        return dict(forward=t1-t0, loss=t2-t1, backward=t3-t2, optimizer=t4-t3,
                    step=t4-t0), loss.detach()
    for _ in range(warmup):
        step()
    sync()
    before = resources()
    context.events.clear()
    context.measure = not probe
    start = time.perf_counter()
    times = []
    for _ in range(steps):
        timing, loss = step()
        times.append(timing)
    duration = time.perf_counter()-start
    context.measure = False
    after = resources()
    finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p).all()) and p.grad is not None
                   and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    row = dict(rank=context.rank, times=times, duration_seconds=duration, before=before, after=after,
               finite=finite, communication=list(context.events), global_batch=batch, seq=seq,
               local_batch=b-a, threads=torch.get_num_threads(), optimizer='SGD', lr=0.001, profile=None)
    # Gather just durations for trace selection; detailed timings are written per rank.
    durations = collect(duration)
    slowest = max(range(len(durations)), key=durations.__getitem__)
    import os
    selected = int(os.environ.get('LOCAL_RANK', '0')) == 0 or context.rank == slowest
    if profile and not probe:
        sync()
        manager = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                   schedule=torch.profiler.schedule(wait=1,warmup=1,active=3,repeat=1),
                   record_shapes=True, profile_memory=True, with_stack=False) if selected else nullcontext())
        with manager as prof:
            with enabled() if selected else nullcontext():
                for _ in range(5):
                    step()
                    if prof is not None:
                        prof.step()
        if selected:
            directory = root/'traces'
            directory.mkdir(parents=True, exist_ok=True)
            trace = directory/f'rank_{context.rank}.json'
            prof.export_chrome_trace(str(trace))
            operators = []
            for event in prof.key_averages(group_by_input_shape=True):
                operators.append(dict(name=event.key, shapes=event.input_shapes, count=event.count,
                    self_cpu_ms=event.self_cpu_time_total/1000, total_cpu_ms=event.cpu_time_total/1000,
                    self_cpu_memory_bytes=event.self_cpu_memory_usage))
            operators.sort(key=lambda op:op['self_cpu_ms'], reverse=True)
            row['profile'] = dict(trace=str(trace.resolve()), sha256=digest(trace), operators=operators)
    sync()
    row['all_ranks_finite'] = all(collect(finite))
    return row
