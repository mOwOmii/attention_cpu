"""Trainable CPU multi-head self-attention with separate Q/K/V weight segments."""

import torch
from torch import nn

from .profiling import scope
from .config import AttentionConfig
from .core import CPUAttentionCore, retain_intermediate_gradients
from .positional import apply_rope


class CPUSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        kwargs = {"bias": config.bias, "device": "cpu", "dtype": config.torch_dtype}
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim, **kwargs)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, **kwargs)
        self.core = CPUAttentionCore(dropout=config.dropout, causal=config.causal,
                                     scale=config.scale, softmax_dtype=config.softmax_dtype)

    def forward(self, x, *, attn_mask=None, key_padding_mask=None, positional_bias=None,
                position_ids=None, dropout_mask=None, return_intermediates=False):
        cfg = self.config
        if x.device.type != "cpu" or x.dtype != cfg.torch_dtype:
            raise ValueError("input must be a CPU tensor with the configured dtype")
        if any(p.device.type != "cpu" or p.dtype != cfg.torch_dtype for p in self.parameters()):
            raise ValueError("parameters must remain on CPU with the configured dtype")
        if x.ndim != 3 or any(n <= 0 for n in x.shape) or x.shape[-1] != cfg.embed_dim:
            raise ValueError("input must have nonempty [B,S,E] or [S,B,E] shape with E=embed_dim")
        if position_ids is not None and not cfg.rope:
            raise ValueError("position_ids requires rope=True")
        canonical = x if cfg.input_layout == "BSE" else x.transpose(0, 1)
        batch, seq, _ = canonical.shape
        with scope("qkv_projection"):
            projected = self.qkv_proj(canonical)
        chunks = projected.chunk(3, dim=-1)
        heads = getattr(self, "local_num_heads", cfg.num_heads)
        q, k, v = [t.reshape(batch, seq, heads, cfg.head_dim).transpose(1, 2)
                   for t in chunks]
        q_before, k_before = q, k
        if cfg.rope:
            kwargs = {"rotary_dim": cfg.rotary_dim, "base": cfg.rope_base}
            with scope("rope"):
                q = apply_rope(q, position_ids, **kwargs)
                k = apply_rope(k, position_ids, **kwargs)
        result = self.core(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           positional_bias=positional_bias, dropout_mask=dropout_mask,
                           return_intermediates=return_intermediates)
        context, details = result if return_intermediates else (result, None)
        merged = context.transpose(1, 2).reshape(batch, seq, heads * cfg.head_dim)
        with scope("output_projection"):
            output = self.out_proj(merged)
        if cfg.input_layout == "SBE":
            output = output.transpose(0, 1)
        if not return_intermediates:
            return output
        details.update({"input": x, "qkv_projection": projected, "query_before_rope": q_before,
                        "key_before_rope": k_before, "merged_context": merged, "output": output})
        retain_intermediate_gradients(details)
        return output, details
