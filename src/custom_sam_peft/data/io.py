"""Channel-aware image reader (spec §6). Keys off channel COUNT only —
channel_semantics never reaches this module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


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


def read_image_with_meta(path: str | Path, channels: int) -> tuple[np.ndarray[Any, Any], Any]:
    """Read pixels + optional SpatialMeta (spec §6.2).

    Pixels-first; meta is None for plain images.
    ``read_image`` is a thin wrapper returning pixels only.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        return _read_tiff_rasterio(path, channels)
    if ext == ".dcm":
        from custom_sam_peft.data.dicom_io import read_dcm_with_meta  # Phase 3 (not yet present)

        return read_dcm_with_meta(path, channels)
    return read_image(path, channels), None  # plain raster/npy: meta None


def read_image(path: str | Path, channels: int) -> np.ndarray[Any, Any]:
    """Read an image file to (H, W, C) with C == channels. Dispatch on extension."""
    path = Path(path)
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
