"""Head-sharded standard self-attention with replicated full module outputs."""

import torch

from ..attention import CPUSelfAttention
from .linear import ColumnParallelLinear, RowParallelLinear, feature_indices, qkv_indices
from .mappings import copy_to_tp


class TensorParallelSelfAttention(CPUSelfAttention):
    def __init__(self, config, parallel):
        if config.num_heads % parallel.tp_size:
            raise ValueError("num_heads must be divisible by tp_size")
        super().__init__(config)
        self.parallel = parallel
        self.local_num_heads = config.num_heads // parallel.tp_size
        self.qkv_proj = ColumnParallelLinear(self.qkv_proj, qkv_indices(config, parallel), parallel)
        self.out_proj = RowParallelLinear(self.out_proj, feature_indices(config, parallel), parallel)

    def load_full_state_dict(self, state):
        return self.load_state_dict(self.shard_state_dict(state))

    def shard_state_dict(self, state):
        qi, fi = qkv_indices(self.config, self.parallel), feature_indices(self.config, self.parallel)
        result = {"qkv_proj.weight": state["qkv_proj.weight"][qi], "out_proj.weight": state["out_proj.weight"][:, fi]}
        if self.config.bias:
            result.update({"qkv_proj.bias": state["qkv_proj.bias"][qi], "out_proj.bias": state["out_proj.bias"]})
        return result

    def _head_slice(self, tensor):
        if tensor is None or tensor.ndim < 3 or tensor.shape[-3] == 1:
            return tensor
        if tensor.shape[-3] != self.config.num_heads:
            raise ValueError("head-specific masks/bias must use global num_heads")
        return tensor.narrow(-3, self.parallel.tp_rank * self.local_num_heads, self.local_num_heads)

    def forward(self, x, *, attn_mask=None, key_padding_mask=None, positional_bias=None,
                position_ids=None, dropout_mask=None, return_intermediates=False):
        # All TP ranks must run input-gradient all_reduce, even if capture selects only one rank.
        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.detach().requires_grad_(True)
        if self.training and self.config.dropout and dropout_mask is None:
            batch, seq = x.shape[:2] if self.config.input_layout == "BSE" else (x.shape[1], x.shape[0])
            # TP peers must have matching RNG states; training seeds them identically per DP replica.
            dropout_mask = torch.rand(batch, self.config.num_heads, seq, seq) >= self.config.dropout
        def mask(value):
            if value is not None and value.is_floating_point() and value.requires_grad:
                value = copy_to_tp(value, self.parallel)
            return self._head_slice(value)
        return super().forward(x, attn_mask=mask(attn_mask), key_padding_mask=key_padding_mask,
            positional_bias=mask(positional_bias), position_ids=position_ids,
            dropout_mask=self._head_slice(dropout_mask), return_intermediates=return_intermediates)

    def weight_partitions(self):
        e = self.config.embed_dim
        qi, fi = qkv_indices(self.config, self.parallel), feature_indices(self.config, self.parallel)
        result = {"qkv_proj.weight": {"kind": "indexed", "global_shape": [3*e, e], "axis": 0, "indices": qi},
                  "out_proj.weight": {"kind": "indexed", "global_shape": [e, e], "axis": 1, "indices": fi}}
        if self.config.bias:
            result.update({"qkv_proj.bias": {"kind": "indexed", "global_shape": [3*e], "axis": 0, "indices": qi},
                           "out_proj.bias": {"kind": "replicated", "global_shape": [e]}})
        return result

    def capture_partitions(self, values):
        result = {}
        weights = self.weight_partitions()
        head_names = {"query", "key", "value", "query_before_rope", "key_before_rope", "scores", "logits",
                      "probabilities", "dropped_probabilities", "context", "valid_rows", "dropout_mask"}
        for name, value in values.items():
            phase, tensor = name.split(".", 1)
            shape = list(value.shape)
            spec = {"kind": "replicated", "global_shape": shape}
            if tensor.startswith("param."):
                spec = weights[tensor[6:]]
            elif tensor in head_names and value.ndim == 4 and value.shape[1] == self.local_num_heads:
                shape[1] = self.config.num_heads
                spec = {"kind": "indexed", "global_shape": shape, "axis": 1,
                        "indices": list(range(self.parallel.tp_rank*self.local_num_heads,
                                              (self.parallel.tp_rank+1)*self.local_num_heads))}
            elif tensor in ("qkv_projection", "merged_context"):
                shape[-1] = self.config.embed_dim * (3 if tensor == "qkv_projection" else 1)
                spec = {"kind": "indexed", "global_shape": shape, "axis": value.ndim-1,
                        "indices": qkv_indices(self.config, self.parallel) if tensor == "qkv_projection"
                                   else feature_indices(self.config, self.parallel)}
            result[name] = spec
        return result
