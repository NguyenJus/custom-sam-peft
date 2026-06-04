"""Part 2 tests: eval-scoped image-read cache (spec §Design — Part 2).

Covers:
- A counting fake read_image proves a second eval over the same indices,
  while the cache is active, triggers zero new reads.
- The store is not consulted or populated when inactive (reads outside the
  context manager always hit the decode path).
- The read-only flag rejects in-place mutation.
- Cross-eval persistence: reads on eval #2..N over the same indices are
  served from the cache (store survives context-manager exit).
- Training read path is unaffected: reads issued while the cache is inactive
  behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_io_module() -> Any:
    """Import a fresh copy of data.io so module globals are pristine."""
    import custom_sam_peft.data.io as io_mod

    # Reset global state between tests.
    io_mod._cache_active = False
    io_mod._store._store.clear()
    return io_mod


def _make_png(tmp_path: Path, name: str = "img.png") -> Path:
    """Write a tiny real PNG and return its path."""
    from PIL import Image

    p = tmp_path / name
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]
    Image.fromarray(img).save(p)
    return p


# ---------------------------------------------------------------------------
# Test: second pass over same indices = zero new reads while active
# ---------------------------------------------------------------------------


def test_second_pass_zero_new_reads(tmp_path: Path) -> None:
    """While the cache is active a second read of the same file hits the store."""
    import custom_sam_peft.data.io as io_mod

    io_mod._cache_active = False
    io_mod._store._store.clear()

    p = _make_png(tmp_path, "img.png")
    decode_calls: list[int] = []

    original_decode = io_mod._decode_image

    def counting_decode(path: Path, channels: int) -> np.ndarray[Any, Any]:
        decode_calls.append(1)
        return original_decode(path, channels)

    with (
        mock.patch.object(io_mod, "_decode_image", side_effect=counting_decode),
        io_mod.cached_image_reads(maxsize=8),
    ):
        # First read — cache miss, should decode once.
        io_mod.read_image(str(p), 3)
        assert len(decode_calls) == 1, "first read must decode exactly once"

        # Second read — cache hit, decode must NOT be called again.
        io_mod.read_image(str(p), 3)
        assert len(decode_calls) == 1, (
            f"second read while active must be served from cache (decode_calls={decode_calls})"
        )


# ---------------------------------------------------------------------------
# Test: inactive = always decode (store not consulted or populated)
# ---------------------------------------------------------------------------


def test_inactive_always_decodes(tmp_path: Path) -> None:
    """Outside cached_image_reads the decode path is always used; store is untouched."""
    import custom_sam_peft.data.io as io_mod

    io_mod._cache_active = False
    io_mod._store._store.clear()

    p = _make_png(tmp_path, "img.png")
    decode_calls: list[int] = []

    original_decode = io_mod._decode_image

    def counting_decode(path: Path, channels: int) -> np.ndarray[Any, Any]:
        decode_calls.append(1)
        return original_decode(path, channels)

    with mock.patch.object(io_mod, "_decode_image", side_effect=counting_decode):
        # Both reads happen outside any context manager.
        io_mod.read_image(str(p), 3)
        io_mod.read_image(str(p), 3)

    assert len(decode_calls) == 2, (
        f"inactive path must always decode; expected 2 calls, got {len(decode_calls)}"
    )
    assert len(io_mod._store._store) == 0, "store must remain empty when cache is inactive"


# ---------------------------------------------------------------------------
# Test: read-only flag rejects in-place mutation
# ---------------------------------------------------------------------------


def test_cached_array_is_read_only(tmp_path: Path) -> None:
    """Arrays returned from the cache have flags.writeable == False."""
    import custom_sam_peft.data.io as io_mod

    io_mod._cache_active = False
    io_mod._store._store.clear()

    p = _make_png(tmp_path, "img.png")

    with io_mod.cached_image_reads(maxsize=8):
        arr = io_mod.read_image(str(p), 3)

    assert not arr.flags.writeable, "cached array must be read-only"
    with pytest.raises(ValueError, match="read-only"):
        arr[0, 0, 0] = 99  # must raise


# ---------------------------------------------------------------------------
# Test: cross-eval persistence (store survives context-manager exit)
# ---------------------------------------------------------------------------


def test_cross_eval_persistence(tmp_path: Path) -> None:
    """The store is not cleared on context-manager exit; eval #2 hits the cache."""
    import custom_sam_peft.data.io as io_mod

    io_mod._cache_active = False
    io_mod._store._store.clear()

    p = _make_png(tmp_path, "img.png")
    decode_calls: list[int] = []

    original_decode = io_mod._decode_image

    def counting_decode(path: Path, channels: int) -> np.ndarray[Any, Any]:
        decode_calls.append(1)
        return original_decode(path, channels)

    with mock.patch.object(io_mod, "_decode_image", side_effect=counting_decode):
        # Eval #1: miss, decode once.
        with io_mod.cached_image_reads(maxsize=8):
            io_mod.read_image(str(p), 3)

        assert len(decode_calls) == 1, "eval #1 must decode exactly once"

        # Store must have survived exit.
        assert len(io_mod._store._store) == 1, "store must persist after context-manager exit"

        # Eval #2: the store is still warm; decode must not be called again.
        with io_mod.cached_image_reads(maxsize=8):
            io_mod.read_image(str(p), 3)

        assert len(decode_calls) == 1, (
            f"eval #2 must be served from the persisted store; got {len(decode_calls)} decode calls"
        )


# ---------------------------------------------------------------------------
# Test: training path (inactive) is unaffected
# ---------------------------------------------------------------------------


def test_training_path_unaffected(tmp_path: Path) -> None:
    """Reads issued while the cache is inactive always go through the decode path."""
    import custom_sam_peft.data.io as io_mod

    io_mod._cache_active = False
    io_mod._store._store.clear()

    p = _make_png(tmp_path, "img.png")
    decode_calls: list[int] = []

    original_decode = io_mod._decode_image

    def counting_decode(path: Path, channels: int) -> np.ndarray[Any, Any]:
        decode_calls.append(1)
        return original_decode(path, channels)

    with mock.patch.object(io_mod, "_decode_image", side_effect=counting_decode):
        # Simulate training: reads happen outside any eval context manager.
        for _ in range(3):
            io_mod.read_image(str(p), 3)

    assert len(decode_calls) == 3, (
        f"training (inactive) path must decode every call; got {len(decode_calls)}"
    )
    # Verify the training reads did not pollute the store.
    assert len(io_mod._store._store) == 0, (
        "training reads must not populate the store when cache is inactive"
    )


# ---------------------------------------------------------------------------
# Test: LRU eviction keeps store bounded
# ---------------------------------------------------------------------------


def test_lru_evicts_when_over_maxsize(tmp_path: Path) -> None:
    """When more than maxsize distinct keys are inserted the oldest entries are evicted."""
    import custom_sam_peft.data.io as io_mod

    io_mod._cache_active = False
    io_mod._store._store.clear()

    # Create maxsize+1 distinct image files.
    maxsize = 4
    # Force the store back to this maxsize so previous tests don't inflate it.
    io_mod._store._maxsize = maxsize
    paths = [_make_png(tmp_path, f"img_{i}.png") for i in range(maxsize + 1)]

    with io_mod.cached_image_reads(maxsize=maxsize):
        for p in paths:
            io_mod.read_image(str(p), 3)

    assert len(io_mod._store._store) <= maxsize, (
        f"store must not exceed maxsize={maxsize}; got {len(io_mod._store._store)} entries"
    )
