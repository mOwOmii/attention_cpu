"""Opt-in semantic ranges; normal training does not emit profiler events."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from torch.profiler import record_function

_enabled = ContextVar('attention_cpu_profile', default=False)


def scope(name):
    return record_function('attention::' + name) if _enabled.get() else nullcontext()


@contextmanager
def enabled():
    token = _enabled.set(True)
    try:
        yield
    finally:
        _enabled.reset(token)
