"""Explicit CPU scaled dot-product attention; no fused attention operators."""

import math

import torch
from torch import nn

from .profiling import scope
from .config import DTYPES


def _broadcast_tensor(tensor, shape, name):
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    try:
        result = torch.broadcast_shapes(tuple(tensor.shape), tuple(shape))
    except RuntimeError as exc:
        raise ValueError(f"{name} cannot broadcast to {tuple(shape)}") from exc
    if tuple(result) != tuple(shape):
        raise ValueError(f"{name} cannot broadcast to {tuple(shape)}")
    return tensor


def retain_intermediate_gradients(details):
    for value in details.values():
        if isinstance(value, torch.Tensor) and value.requires_grad:
            value.retain_grad()


class CPUAttentionCore(nn.Module):
    """MHA core with bool True=blocked, additive float masks, and safe empty rows.

    Inputs: Q/K [B,H,Sq/Skv,D], V [B,H,Skv,Dv]. Float masks and
    positional bias use the input dtype. dropout_mask uses True=keep.
    A completely blocked row has zero probability, context, and core gradient.
    """

    def __init__(self, *, dropout=0.0, causal=True, scale=None, softmax_dtype=None):
        super().__init__()
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        if type(causal) is not bool:
            raise ValueError("causal must be boolean")
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("scale must be finite and positive")
        if softmax_dtype not in (*DTYPES, None):
            raise ValueError("softmax_dtype must be float32, float64, or None")
        self.dropout, self.causal, self.scale = dropout, causal, scale
        self.softmax_dtype = softmax_dtype

    def forward(self, query, key, value, *, attn_mask=None, key_padding_mask=None,
                positional_bias=None, dropout_mask=None, return_intermediates=False):
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.device.type != "cpu" or tensor.dtype not in (torch.float32, torch.float64):
                raise ValueError(f"{name} must be float32/float64 on CPU")
            if tensor.ndim != 4 or any(n <= 0 for n in tensor.shape):
                raise ValueError(f"{name} must have nonempty shape [B,H,S,D]")
        if not query.dtype == key.dtype == value.dtype:
            raise ValueError("query, key, value must have the same dtype")
        if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
            raise ValueError("batch/head sizes must match; key/value sequence lengths must match")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("query/key head dimensions must match")
        batch, heads, sq, dim = query.shape
        sk = key.shape[-2]
        shape = (batch, heads, sq, sk)
        scale = dim ** -0.5 if self.scale is None else self.scale
        with scope("qk_matmul"):
            scores = torch.matmul(query, key.transpose(-2, -1))
        with scope("mask_scale"):
            scores = scores * scale
            logits = scores
            if positional_bias is not None:
                _broadcast_tensor(positional_bias, shape, "positional_bias")
                if positional_bias.dtype != query.dtype or not torch.isfinite(positional_bias).all():
                    raise ValueError("positional_bias must be finite and have the input dtype")
                logits = logits + positional_bias

            blocked = torch.zeros((sq, sk), dtype=torch.bool)
            if self.causal:
                blocked = torch.arange(sk)[None, :] > torch.arange(sq)[:, None]
            if attn_mask is not None:
                _broadcast_tensor(attn_mask, shape, "attn_mask")
                if attn_mask.dtype == torch.bool:
                    blocked = blocked | attn_mask
                elif attn_mask.dtype == query.dtype:
                    if torch.isnan(attn_mask).any() or torch.isposinf(attn_mask).any():
                        raise ValueError("float attn_mask cannot contain NaN or positive infinity")
                    logits = logits + attn_mask
                else:
                    raise ValueError("attn_mask must be boolean or have the input dtype")
            if key_padding_mask is not None:
                if (key_padding_mask.device.type != "cpu" or key_padding_mask.dtype != torch.bool
                        or tuple(key_padding_mask.shape) != (batch, sk)):
                    raise ValueError("key_padding_mask must be boolean CPU [B,Skv]")
                blocked = blocked | key_padding_mask[:, None, None, :]
            logits = logits.masked_fill(blocked, float("-inf"))
            # Avoid softmax(-inf,...,-inf); the zero-row branch also blocks its gradient.
            valid_rows = ~torch.isneginf(logits).all(dim=-1, keepdim=True)
            safe_logits = torch.where(valid_rows, logits, torch.zeros_like(logits))
        with scope("softmax"):
            sm_dtype = query.dtype if self.softmax_dtype is None else DTYPES[self.softmax_dtype]
            probabilities = torch.softmax(safe_logits, dim=-1, dtype=sm_dtype).to(query.dtype)
            probabilities = torch.where(valid_rows, probabilities, torch.zeros_like(probabilities))
        active_dropout = self.training and self.dropout > 0
        if dropout_mask is not None:
            _broadcast_tensor(dropout_mask, shape, "dropout_mask")
            if dropout_mask.dtype != torch.bool or not active_dropout:
                raise ValueError("dropout_mask must be boolean and requires training dropout > 0")
        if active_dropout:
            if dropout_mask is None:
                dropout_mask = torch.rand(shape, device="cpu") >= self.dropout
            dropped = probabilities * dropout_mask / (1 - self.dropout)
        else:
            dropped = probabilities
        with scope("pv_matmul"):
            context = torch.matmul(dropped, value)
        if not return_intermediates:
            return context
        details = {"query": query, "key": key, "value": value, "scores": scores,
                   "logits": logits, "probabilities": probabilities,
                   "dropped_probabilities": dropped, "context": context,
                   "valid_rows": valid_rows, "dropout_mask": dropout_mask}
        retain_intermediate_gradients(details)
        return context, details
