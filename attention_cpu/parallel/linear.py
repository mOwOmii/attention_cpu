"""QKV column shards and output projection column slices in [out,in] layout."""

import torch
from torch import nn
from torch.nn import functional as F

from .mappings import copy_to_tp, reduce_from_tp


def feature_indices(config, parallel):
    width = config.embed_dim // parallel.tp_size
    return list(range(parallel.tp_rank * width, (parallel.tp_rank + 1) * width))


def qkv_indices(config, parallel):
    return [offset + i for offset in (0, config.embed_dim, 2 * config.embed_dim)
            for i in feature_indices(config, parallel)]


class ColumnParallelLinear(nn.Module):
    def __init__(self, full_layer, indices, parallel):
        super().__init__()
        self.parallel = parallel
        self.weight = nn.Parameter(full_layer.weight.detach()[indices].clone())
        self.bias = nn.Parameter(full_layer.bias.detach()[indices].clone()) if full_layer.bias is not None else None

    def forward(self, value):
        return F.linear(copy_to_tp(value, self.parallel), self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, full_layer, indices, parallel):
        super().__init__()
        self.parallel = parallel
        self.weight = nn.Parameter(full_layer.weight.detach()[:, indices].clone())
        self.bias = nn.Parameter(full_layer.bias.detach().clone()) if full_layer.bias is not None else None

    def forward(self, value):
        partial = F.linear(value, self.weight, None)
        output = reduce_from_tp(partial, self.parallel)
        return output + self.bias if self.bias is not None else output
