"""Interleaved (adjacent even/odd pair) rotary position embeddings."""

import math

import torch


def apply_rope(x, position_ids=None, *, rotary_dim=None, base=10000.0):
    """Rotate [B,H,S,D], preserving any non-rotary suffix and its gradient."""
    if x.device.type != "cpu" or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("RoPE requires float32/float64 CPU tensors")
    if x.ndim != 4 or any(n <= 0 for n in x.shape):
        raise ValueError("RoPE input must have nonempty shape [B,H,S,D]")
    batch, _, seq, dim = x.shape
    rd = dim if rotary_dim is None else rotary_dim
    if type(rd) is not int or rd <= 0 or rd > dim or rd % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("RoPE base must be finite and positive")
    if position_ids is None:
        position_ids = torch.arange(seq)
    if position_ids.device.type != "cpu" or position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("position_ids must be an integer CPU tensor")
    if tuple(position_ids.shape) not in ((seq,), (batch, seq)):
        raise ValueError("position_ids must have shape [S] or [B,S]")
    if (position_ids < 0).any():
        raise ValueError("position_ids must be nonnegative")
    positions = position_ids.to(x.dtype).reshape(-1, 1, seq, 1)
    inverse_frequency = base ** (-torch.arange(0, rd, 2, dtype=x.dtype) / rd)
    angles = positions * inverse_frequency
    cosine, sine = angles.cos(), angles.sin()
    even, odd = x[..., :rd:2], x[..., 1:rd:2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1)
    return torch.cat((rotated.flatten(-2), x[..., rd:]), dim=-1)
