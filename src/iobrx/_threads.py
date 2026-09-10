"""Thread-count policy for the iobrx public API.

Every public function accepts ``n_threads=None``. Resolution order:

1. an explicit ``n_threads`` argument wins;
2. otherwise the value set by :func:`iobrx.set_threads` (if ever called);
3. otherwise ``min(8, available_cpu_count())``.

The cap of 8-by-default (rather than "all cores") keeps desktop machines
responsive while still exploiting the embarrassingly parallel solvers; the
official benchmarks used up to 224 threads on a 224-core node.
"""
from __future__ import annotations

import os

__all__ = ["set_threads", "resolve_threads", "get_threads", "available_cpu_count"]

_user_default: int | None = None


def set_threads(n: int | None) -> None:
    """Set the process-wide default thread count used when ``n_threads=None``.

    Parameters
    ----------
    n : int or None
        Positive thread count, or ``None`` to restore the built-in default
        ``min(8, available_cpu_count())``.
    """
    global _user_default
    if n is None:
        _user_default = None
        return
    n = int(n)
    if n < 1:
        raise ValueError(f"n_threads must be >= 1, got {n}")
    _user_default = n


def resolve_threads(n_threads: int | None = None) -> int:
    """Resolve the effective thread count for one call (see module docstring)."""
    if n_threads is not None:
        n = int(n_threads)
        if n < 1:
            raise ValueError(f"n_threads must be >= 1, got {n_threads}")
        return n
    if _user_default is not None:
        return _user_default
    return min(8, available_cpu_count())


def get_threads() -> int:
    """The thread count a call with ``n_threads=None`` would use right now."""
    return resolve_threads(None)


def available_cpu_count() -> int:
    """CPUs available to this process, respecting affinity where supported.

    This is not a cgroup CPU-time quota or an allocation from a job scheduler.
    Pass an explicit thread budget for those limits. Explicit/user defaults
    remain unchanged, including when they exceed the detected affinity.
    """
    counts = []
    for get_count in (getattr(os, "process_cpu_count", None),
                      lambda: len(os.sched_getaffinity(0))):
        if get_count is not None:
            try:
                count = get_count()
                if count is not None and count > 0:
                    counts.append(count)
            except (AttributeError, NotImplementedError, OSError):
                pass
    return min(counts) if counts else (os.cpu_count() or 1)
