"""Standard, observable CPU attention with a single-process training example."""

from .config import AttentionConfig
from .core import CPUAttentionCore
from .attention import CPUSelfAttention

__version__ = "0.1.0"
__all__ = ["AttentionConfig", "CPUAttentionCore", "CPUSelfAttention"]
