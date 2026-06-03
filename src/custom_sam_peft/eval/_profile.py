"""TEMPORARY eval profiler (issue #250, Phase 1). REMOVED in Phase 2.

Env-gated, CUDA-synchronized bucket timer + metadata capture. When
CSP_EVAL_PROFILE is unset/0, every public call is a no-op so normal eval runs
pay nothing. This is spike-only instrumentation — NOT a permanent --profile
feature (spec §9). All call sites and this file are reverted in Phase 2.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("CSP_EVAL_PROFILE", "0") not in ("", "0", "false", "False")

# Bucket name -> accumulated seconds.
_BUCKETS: dict[str, float] = defaultdict(float)
# Free-form metadata (N, n_classes, image sizes, forwards count, ...).
_META: dict[str, object] = {}


def enabled() -> bool:
    return _ENABLED


@contextmanager
def bucket(name: str) -> Iterator[None]:
    """CUDA-synchronized timer for one bucket. No-op when disabled.

    Synchronizes BEFORE starting and BEFORE stopping so async CUDA kernels are
    attributed to the bucket that launched them, not the next one.
    """
    if not _ENABLED:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _BUCKETS[name] += time.perf_counter() - t0


def note(**kwargs: object) -> None:
    """Record metadata (last value wins per key). No-op when disabled."""
    if not _ENABLED:
        return
    _META.update(kwargs)


def incr(key: str, by: int = 1) -> None:
    """Increment an integer counter (e.g. forwards). No-op when disabled."""
    if not _ENABLED:
        return
    _META[key] = int(_META.get(key, 0)) + by  # type: ignore[arg-type]


def snapshot() -> tuple[dict[str, float], dict[str, object]]:
    """Return (buckets_seconds, metadata) copies for reporting."""
    return dict(_BUCKETS), dict(_META)


def reset() -> None:
    _BUCKETS.clear()
    _META.clear()
