"""The versioned single-process MHA specification."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch


DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class AttentionConfig:
    embed_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.0
    causal: bool = True
    bias: bool = True
    rope: bool = False
    rope_base: float = 10000.0
    rotary_dim: int | None = None
    scale: float | None = None
    dtype: str = "float32"
    softmax_dtype: str | None = None
    input_layout: str = "BSE"

    def __post_init__(self):
        for name in ("embed_dim", "num_heads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        for name in ("causal", "bias", "rope"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        if self.scale is not None and (not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("scale must be finite and positive")
        if self.dtype not in DTYPES or self.softmax_dtype not in (*DTYPES, None):
            raise ValueError("this version supports only float32 and float64")
        if self.input_layout not in ("BSE", "SBE"):
            raise ValueError("input_layout must be BSE or SBE")
        rd = self.head_dim if self.rotary_dim is None else self.rotary_dim
        if self.rotary_dim is not None or self.rope:
            if type(rd) is not int or rd <= 0 or rd > self.head_dim or rd % 2:
                raise ValueError("rotary_dim must be positive, even, and <= head_dim")

    @property
    def head_dim(self):
        return self.embed_dim // self.num_heads

    @property
    def torch_dtype(self):
        return DTYPES[self.dtype]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
