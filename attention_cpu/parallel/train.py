"""CPU TP + DDP training with globally normalized, accumulated masked losses."""

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..analyze import write_json
from ..capture import CaptureConfig, CaptureWriter
from .attention import TensorParallelSelfAttention
from .checkpoint import gather_full_state_dict, restore_parallel_checkpoint, save_parallel_checkpoint


def batch_range(size, parts, rank):
    start = rank * (size // parts) + min(rank, size % parts)
    return start, start + size // parts + (rank < size % parts)


def make_training_batch(config, batch_size, seq_len, seed, variable_lengths=False):
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(batch_size, seq_len, config.embed_dim, generator=generator, dtype=config.torch_dtype)
    lengths = 1 + torch.arange(batch_size) % seq_len if variable_lengths else torch.full((batch_size,), seq_len)
    valid = torch.arange(seq_len)[None, :] < lengths[:, None]
    x = x * valid[..., None]
    target = (x.cumsum(1) / torch.arange(1, seq_len+1, dtype=x.dtype)[None,:,None] if config.causal else
              (x.sum(1, keepdim=True)/lengths[:,None,None]).expand_as(x).clone())
    return x, target, valid


def step_dropout_mask(config, batch_size, seq_len, seed, step):
    if not config.dropout:
        return None
    generator = torch.Generator().manual_seed(seed + 100000 + step)
    return torch.rand(batch_size, config.num_heads, seq_len, seq_len, generator=generator) >= config.dropout


def normalized_loss(prediction, target, valid, *, global_count, dp_size):
    # DDP averages ranks: compensate by dp_size. Summing microbatch losses then
    # yields the global valid-element mean even for unequal DP batches/token counts.
    numerator = (((prediction - target) ** 2) * valid[..., None]).sum()
    return numerator * (dp_size / global_count), numerator.detach()


def train_parallel(config, context, *, steps=20, batch_size=4, seq_len=16, seed=7, lr=0.01,
                   output="runs/parallel", resume=None, threads=1, grad_accum_steps=1,
                   variable_lengths=False, capture=None):
    for name, value in (("steps",steps),("batch_size",batch_size),("seq_len",seq_len),
                        ("threads",threads),("grad_accum_steps",grad_accum_steps)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if batch_size < context.dp_size * grad_accum_steps:
        raise ValueError("global batch_size must be >= dp_size * grad_accum_steps")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = TensorParallelSelfAttention(config, context)
    train_config = {"batch_size":batch_size, "seq_len":seq_len, "seed":seed, "lr":lr, "threads":threads,
                    "grad_accum_steps":grad_accum_steps, "variable_lengths":variable_lengths,
                    "optimizer":"AdamW", "weight_decay":0.0, "task":"global fixed batch masked mean regression"}
    if dist.is_initialized():
        signatures = [None] * context.world_size
        dist.all_gather_object(signatures, (config.to_dict(),train_config,str(output),str(resume)))
        if any(s != signatures[0] for s in signatures):
            raise ValueError("all ranks must agree on training configuration and shared output/resume paths")
    executor = (DistributedDataParallel(model, device_ids=None, process_group=context.dp_group,
                                       broadcast_buffers=False) if context.dp_size > 1 else model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    start, losses = (restore_parallel_checkpoint(resume, model=model, optimizer=optimizer, train_config=train_config)
                     if resume else (0,[]))
    if steps <= start:
        raise ValueError("steps must exceed the restored optimizer step")
    x, target, valid = make_training_batch(config,batch_size,seq_len,seed,variable_lengths)
    begin, end = batch_range(batch_size, context.dp_size, context.dp_rank)
    count = valid.sum().item() * config.embed_dim
    capture = capture or CaptureConfig()
    writer = CaptureWriter(capture)
    captured = []
    model.train()
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True)
        mask = step_dropout_mask(config,batch_size,seq_len,seed,step)
        local_numerator = torch.zeros((),dtype=torch.float64)
        for micro in range(grad_accum_steps):
            first,last = batch_range(end-begin,grad_accum_steps,micro)
            first,last = first+begin,last+begin
            data, expected, tokens = x[first:last],target[first:last],valid[first:last]
            if config.input_layout == "SBE":
                data,expected,tokens = data.transpose(0,1),expected.transpose(0,1),tokens.transpose(0,1)
            synchronize = micro == grad_accum_steps-1
            scope = nullcontext() if synchronize or context.dp_size == 1 else executor.no_sync()
            with scope:
                with writer.record(model, step=step+1, rank=context.rank, micro=micro,
                                   forward_model=executor, dp_synchronized=synchronize) as record:
                    prediction = record.forward(data, key_padding_mask=~valid[first:last],
                                                dropout_mask=mask[first:last] if mask is not None else None)
                    loss,numerator = normalized_loss(prediction,expected,tokens,global_count=count,dp_size=context.dp_size)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"rank {context.rank}: nonfinite loss at step {step+1}")
                    loss.backward()
                if record.path is not None:
                    captured.append(str(record.path))
            local_numerator += numerator.double()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"rank {context.rank}: missing/nonfinite gradient at step {step+1}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"rank {context.rank}: nonfinite parameter at step {step+1}")
        context.sum(local_numerator,"dp")
        losses.append(local_numerator.item()/count)
    model.eval()
    with torch.no_grad():
        data,expected,tokens = x[begin:end],target[begin:end],valid[begin:end]
        if config.input_layout == "SBE":
            data,expected,tokens = data.transpose(0,1),expected.transpose(0,1),tokens.transpose(0,1)
        prediction = model(data,key_padding_mask=~valid[begin:end])
        _, numerator = normalized_loss(prediction,expected,tokens,global_count=count,dp_size=context.dp_size)
        evaluation = context.sum(numerator.double(),"dp").item()/count
    root = Path(output)
    root.mkdir(parents=True,exist_ok=True)
    checkpoint = save_parallel_checkpoint(root/"checkpoints"/f"step_{steps}",model=model,optimizer=optimizer,
                                         step=steps,train_config=train_config,losses=losses)
    full_state = gather_full_state_dict(model)
    report = {**context.metadata(),"device":"cpu","steps":steps,"resumed_from_step":start,
              "model_config":config.to_dict(),"train_config":train_config,"losses":losses,
              "initial_training_loss":losses[0],"final_training_loss":losses[-1],"evaluation_loss":evaluation,
              "checkpoint":str(checkpoint),"captured_records":captured,
              "capture":{**asdict(capture),"root":str(capture.root)}}
    write_json(root/f"training_report_rank_{context.rank}.json",report)
    if context.rank == 0:
        torch.save({"config":config.to_dict(),"model":full_state},root/"full_model.pt")
        write_json(root/"training_report.json",report)
    return report
