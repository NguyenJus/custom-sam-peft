"""Channel-aware image reader (spec §6). Keys off channel COUNT only —
channel_semantics never reaches this module."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# ---------------------------------------------------------------------------
# Eval-scoped image-read cache (Part 2 spec §Design — Part 2)
# ---------------------------------------------------------------------------
# The store is a module-level bounded LRU keyed on (resolved_path_str, channels).
# It is long-lived: it persists across cached_image_reads() context manager
# exits so that the same ~64 val images remain resident across all ~160
# periodic evals.  The active flag enables/disables consult + populate;
# when inactive read_image behaves exactly as before (no overhead).


class _LRUStore:
    """Bounded LRU cache that allows explicit insertion.

    Unlike functools.lru_cache this is a plain dict-backed store so callers can
    insert pre-computed values without re-routing through a callable.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[tuple[str, int], np.ndarray[Any, Any]] = OrderedDict()

    def get(self, key: tuple[str, int]) -> np.ndarray[Any, Any] | None:
        if key not in self._store:
            return None
        # Move to end (most-recently used).
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: tuple[str, int], value: np.ndarray[Any, Any]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if len(self._store) >= self._maxsize:
            self._store.popitem(last=False)  # evict LRU
        self._store[key] = value

    def resize(self, maxsize: int) -> None:
        """Grow maxsize; never shrinks (avoids evicting warm entries)."""
        if maxsize > self._maxsize:
            self._maxsize = maxsize


_store = _LRUStore(maxsize=64)
_cache_active: bool = False


@contextmanager
def cached_image_reads(maxsize: int) -> Iterator[None]:
    """Activate the eval-scoped image-read cache for the duration of this block.

    The underlying LRU store is long-lived and is **not** cleared on exit, so
    the hot val images stay resident across all periodic evals.  Only the
    active flag is toggled; reads outside this block always hit the decode path.

    Scoping rationale: periodic eval runs synchronously in the main process (no
    DataLoader workers).  Training image reads happen only in worker processes.
    Because the cache is activated only around the main-process eval call —
    never during DataLoader iteration or across a worker fork — training workers
    never see the active flag and never populate the cache.
    """
    global _cache_active
    _store.resize(maxsize)
    previous = _cache_active
    _cache_active = True
    try:
        yield
    finally:
        _cache_active = previous


def _coerce_to_channels(obj: object, channels: int) -> np.ndarray[Any, Any]:
    """Coerce a PIL image OR an ndarray to (H, W, C) with C == channels.

    PIL path uses mode conversion (1->L, 3->RGB, 4->RGBA). Array path accepts
    2-D (H,W -> triplicate/keep), (H,W,C), or (C,H,W); validates C == channels.
    """
    from PIL import Image as PILImage

    if isinstance(obj, PILImage.Image):
        mode = _PIL_MODE.get(channels)
        if mode is None:
            raise ValueError(
                f"read_image: PIL/raster input cannot produce channels={channels} "
                f"(PIL supports 1=L, 3=RGB, 4=RGBA only). Use a .npy/.npz/.tif source."
            )
        out = np.asarray(obj.convert(mode))
        if out.ndim == 2:  # mode "L"
            out = out[:, :, None]
        return out

    arr = np.asarray(obj)
    if arr.ndim == 2:
        if channels == 1:
            return arr[:, :, None]
        return np.repeat(arr[:, :, None], channels, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"read_image: expected 2-D or 3-D array, got ndim={arr.ndim}")
    # Resolve channel axis: prefer HWC; transpose CHW when the leading dim matches.
    if arr.shape[2] == channels:
        hwc = arr
    elif arr.shape[0] == channels:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        found = arr.shape[2] if arr.shape[2] <= arr.shape[0] else arr.shape[0]
        raise ValueError(f"read_image: array has {found} channels but data.channels={channels}")
    return np.ascontiguousarray(hwc)


def _read_tiff_rasterio(path: Path, channels: int) -> tuple[np.ndarray[Any, Any], Any]:
    """Read a TIFF with rasterio, returning (pixels, SpatialMeta|None).

    Geo TIFFs carry CRS/affine metadata; plain TIFFs return meta=None.
    Nodata pixels are zero-filled before the model (spec §6.3).
    """
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()  # always (C, H, W)
        pixels = _coerce_to_channels(arr, channels)  # preserves C == channels validation
        if src.crs is None and src.transform.is_identity:
            return pixels, None  # plain non-geo TIFF — identical behaviour to old tifffile path
        nodata = src.nodata
        nodata_mask = None
        if nodata is not None:
            nodata_mask = np.any(arr == nodata, axis=0)
            # nodata pixels zero-filled before the model — matches PadIfNeeded fill=0; spec §6.3.
            pixels = pixels.copy()
            pixels[nodata_mask] = 0
        from custom_sam_peft.data.spatial_meta import SpatialMeta

        return pixels, SpatialMeta(
            kind="geo",
            crs=src.crs,
            affine=src.transform,
            nodata=nodata,
            nodata_mask=nodata_mask,
        )


def read_image_with_meta(
    path: str | Path,
    channels: int,
    *,
    dicom_voi_window: tuple[float, float] | None = None,
) -> tuple[np.ndarray[Any, Any], Any]:
    """Read pixels + optional SpatialMeta (spec §6.2).

    Pixels-first; meta is None for plain images.
    ``read_image`` is a thin wrapper returning pixels only.

    dicom_voi_window: optional (center, width) override for DICOM VOI windowing.
    None (default) → use each file's own window or skip VOI. Non-DICOM paths
    ignore this parameter.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        return _read_tiff_rasterio(path, channels)
    if ext == ".dcm":
        from custom_sam_peft.data.dicom_io import read_dcm_with_meta

        return read_dcm_with_meta(path, channels, voi_window=dicom_voi_window)
    return read_image(path, channels), None  # plain raster/npy: meta None


def read_image(path: str | Path, channels: int) -> np.ndarray[Any, Any]:
    """Read an image file to (H, W, C) with C == channels. Dispatch on extension.

    When the eval-scoped cache is active (inside ``cached_image_reads``), a
    fast path checks the module-level LRU store first.  On a hit the cached
    read-only array is returned directly (zero new disk I/O).  On a miss the
    normal decode runs, the result is made read-only, and it is inserted into
    the store.  When inactive the function behaves exactly as before.
    """
    path = Path(path)

    # --- eval-scoped cache fast path ---
    if _cache_active:
        cache_key = (str(path), channels)
        cached = _store.get(cache_key)
        if cached is not None:
            return cached
        # Miss: decode, mark read-only, insert, return.
        arr = _decode_image(path, channels)
        arr.flags.writeable = False
        _store.put(cache_key, arr)
        return arr

    return _decode_image(path, channels)


def _decode_image(path: Path, channels: int) -> np.ndarray[Any, Any]:
    """Internal: unconditionally decode *path* without consulting the cache."""
    ext = path.suffix.lower()
    if ext in _RASTER_EXTS:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            return _coerce_to_channels(im, channels)
    if ext in {".npy", ".npz"}:
        loaded = np.load(path)
        if ext == ".npz":
            loaded = loaded[loaded.files[0]]
        return _coerce_to_channels(loaded, channels)
    if ext in {".tif", ".tiff"}:
        return _read_tiff_rasterio(path, channels)[0]
    raise ValueError(f"read_image: unsupported file extension {ext!r} for {path}")
