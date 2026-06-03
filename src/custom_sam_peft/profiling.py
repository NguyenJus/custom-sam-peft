"""Permanent general-purpose env-gated bucket-timer facility (issue #255).

Generalizes the removed #250 spike profiler (``eval/_profile.py``).  When
disabled every public call is a strict no-op — no perf_counter, no CUDA call,
no dict write.  Enable via ``CSP_PROFILE=1`` (or ``enable()`` at runtime).

Disabled values for ``CSP_PROFILE``: ``""``, ``"0"``, ``"false"``, ``"False"``.
Any other non-empty value enables the profiler.

Greppable dot-namespaced bucket names::

    eval.forward          eval.coco_aggregate   eval.mask_upsample
    eval.transfer_binarize  eval.rle_encode     eval.proxy_iou
    train.forward         train.loss            train.backward
    train.optim_step
    predict.forward       predict.postprocess   predict.write
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-global state — runtime-mutable so the CLI and tests can toggle
# without restarting the process.
# ---------------------------------------------------------------------------

_ENABLED: bool = os.environ.get("CSP_PROFILE", "0") not in ("", "0", "false", "False")

_BUCKETS: dict[str, float] = defaultdict(float)
_META: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Control API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True if the profiler is currently collecting data."""
    return _ENABLED


def enable() -> None:
    """Enable the profiler at runtime."""
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    """Disable the profiler at runtime (all calls become strict no-ops)."""
    global _ENABLED
    _ENABLED = False


# ---------------------------------------------------------------------------
# Collection API
# ---------------------------------------------------------------------------


@contextmanager
def bucket(name: str) -> Iterator[None]:
    """CUDA-synchronized timer for one named bucket.

    When enabled: synchronizes BEFORE starting and BEFORE stopping so async
    CUDA kernels are attributed to the bucket that launched them, not the next
    one.  Accumulates elapsed seconds into ``name``; re-entrant (additive).

    When disabled: ``yield`` only — no perf_counter, no CUDA call, no dict
    write.
    """
    if not _ENABLED:
        yield
        return

    # Lazy import — no mandatory torch import at module level (call-site guard).
    try:
        import torch as _torch

        _cuda = _torch.cuda.is_available()
    except ImportError:
        _cuda = False

    if _cuda:
        import torch

        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if _cuda:
            import torch

            torch.cuda.synchronize()
        _BUCKETS[name] += time.perf_counter() - t0


def note(**kwargs: object) -> None:
    """Record free-form metadata (last value wins per key).  No-op when disabled."""
    if not _ENABLED:
        return
    _META.update(kwargs)


def incr(key: str, by: int = 1) -> None:
    """Increment an integer counter.  No-op when disabled."""
    if not _ENABLED:
        return
    prev = _META.get(key, 0)
    _META[key] = (prev if isinstance(prev, int) else 0) + by


# ---------------------------------------------------------------------------
# Read / export API
# ---------------------------------------------------------------------------


def snapshot() -> tuple[dict[str, float], dict[str, object]]:
    """Return ``(buckets_seconds, metadata)`` copies.

    Safe to call when disabled — returns empty dicts.  Copies are independent
    of internal state; mutating them does not affect the profiler.
    """
    return dict(_BUCKETS), dict(_META)


def snapshot_json(indent: int = 2) -> str:
    """Return ``json.dumps({"buckets": ..., "meta": ...}, default=str)``."""
    buckets, meta = snapshot()
    return json.dumps({"buckets": buckets, "meta": meta}, default=str, indent=indent)


def dump(path: Path | str) -> Path:
    """Write ``snapshot_json()`` to *path* (parents created); flush; return Path.

    Durable: the file is flushed and the fd is closed before returning so a
    SIGKILL or crash after the call leaves a valid JSON file on disk.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = snapshot_json()
    with out.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    return out


def reset() -> None:
    """Clear all buckets and metadata."""
    _BUCKETS.clear()
    _META.clear()
