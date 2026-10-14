"""Deterministic TP/DP group construction; TP varies fastest in global rank."""

from dataclasses import dataclass
from datetime import timedelta
import os

import torch.distributed as dist


@dataclass
class ParallelContext:
    tp_size: int = 1
    dp_size: int = 1
    rank: int = 0
    tp_group: object = None
    dp_group: object = None

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.tp_size, self.dp_size)):
            raise ValueError("tp_size and dp_size must be positive integers")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid global rank")

    @property
    def world_size(self):
        return self.tp_size * self.dp_size

    @property
    def tp_rank(self):
        return self.rank % self.tp_size

    @property
    def dp_rank(self):
        return self.rank // self.tp_size

    @property
    def tp_ranks(self):
        return list(range(self.dp_rank * self.tp_size, (self.dp_rank + 1) * self.tp_size))

    @property
    def dp_ranks(self):
        return list(range(self.tp_rank, self.world_size, self.tp_size))

    def sum(self, tensor, domain="tp"):
        size = self.tp_size if domain == "tp" else self.dp_size
        if domain not in ("tp", "dp"):
            raise ValueError("domain must be tp or dp")
        if size > 1:
            group = self.tp_group if domain == "tp" else self.dp_group
            if group is None or not dist.is_initialized():
                raise RuntimeError(f"{domain} requires an initialized process group")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    def metadata(self):
        return {"rank": self.rank, "tp_rank": self.tp_rank, "dp_rank": self.dp_rank,
                "tp_size": self.tp_size, "dp_size": self.dp_size,
                "tp_ranks": self.tp_ranks, "dp_ranks": self.dp_ranks}


def initialize_parallel(tp_size=1, dp_size=1, *, timeout_seconds=60, init_method="env://"):
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    context = ParallelContext(tp_size, dp_size, rank)
    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != context.world_size:
        raise ValueError("WORLD_SIZE must equal tp_size * dp_size")
    if world_size == 1 and not dist.is_initialized():
        return context
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=timeout_seconds))
    if dist.get_backend() != "gloo":
        raise ValueError("CPU training requires the Gloo backend")
    specifications = [None] * world_size
    dist.all_gather_object(specifications,(tp_size,dp_size))
    if any(spec != specifications[0] for spec in specifications):
        raise ValueError("all ranks must agree on TP/DP topology")
    # Every rank creates every group in the same order, including singleton groups.
    for dp in range(dp_size):
        ranks = list(range(dp * tp_size, (dp + 1) * tp_size))
        group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=timeout_seconds))
        if rank in ranks:
            context.tp_group = group
    for tp in range(tp_size):
        ranks = list(range(tp, world_size, tp_size))
        group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=timeout_seconds))
        if rank in ranks:
            context.dp_group = group
    return context
