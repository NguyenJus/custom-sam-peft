"""DICOM reads behind the optional [dicom] extra (spec §8). pydicom/nibabel are
lazy-imported so base install/import never requires them."""

from __future__ import annotations

import types
from typing import Any

_MISSING = "DICOM support requires the optional extra: pip install custom-sam-peft[dicom]"


def _require_pydicom() -> types.ModuleType:
    """Import pydicom or raise an actionable RuntimeError."""
    try:
        import pydicom

        return pydicom
    except ImportError as exc:
        raise RuntimeError(_MISSING) from exc


def read_dcm_with_meta(
    path: object,
    channels: int,
    *,
    voi_window: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Read a DICOM file -> (pixels (H,W,C), SpatialMeta(kind='dicom')).

    Decode order — DICOM PS3.3 §C.11.2 (corrects spec §8.1 stated order):
      1. Modality LUT (apply_modality_lut) — always; converts stored to real units (e.g. HU).
      2. VOI windowing — only if voi_window override given OR file carries WindowCenter/WindowWidth.
         Override wins over file window.  apply_voi_lut accepts MONOCHROME1 PI, so PI need not
         be changed before calling it.
      3. MONOCHROME1 inversion — only when PhotometricInterpretation == "MONOCHROME1".
         Per DICOM PS3.3 §C.11.2, VOI windowing operates on the Modality-LUT (HU-space) output;
         MONOCHROME1 inversion is a display-time step and MUST come AFTER VOI.  Applying the
         window on already-inverted values (the old order) corrupts MONOCHROME1+window output.
      4. Coerce to (H,W,C) via _coerce_to_channels.
    """
    pydicom = _require_pydicom()

    # Lazy import LUT helpers from pydicom 3.x location.
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    ds = pydicom.dcmread(path)

    # 1. Modality LUT — always (slope/intercept → real units e.g. HU).
    arr = apply_modality_lut(ds.pixel_array, ds)

    # 2. VOI windowing — only when there's something to apply.
    # Must precede MONOCHROME1 inversion (DICOM PS3.3 §C.11.2): the window operates on
    # HU-space values; inversion is display-time and comes after.
    has_file_window = hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth")
    if voi_window is not None or has_file_window:
        if voi_window is not None:
            # Override wins: patch the dataset's window tags before calling apply_voi_lut.
            ds.WindowCenter = voi_window[0]
            ds.WindowWidth = voi_window[1]
        arr = apply_voi_lut(arr, ds)

    # 3. MONOCHROME1 inversion (min=white convention) — display-time, always after VOI.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    # 4. Coerce to (H, W, C).
    from custom_sam_peft.data.io import _coerce_to_channels

    pixels = _coerce_to_channels(arr, channels)

    # Build SpatialMeta — use getattr with None defaults for optional geometry tags.
    from custom_sam_peft.data.spatial_meta import SpatialMeta

    raw_spacing = getattr(ds, "PixelSpacing", None)
    pixel_spacing = (
        (float(raw_spacing[0]), float(raw_spacing[1])) if raw_spacing is not None else None
    )

    raw_orient = getattr(ds, "ImageOrientationPatient", None)
    orientation = [float(v) for v in raw_orient] if raw_orient is not None else None

    raw_pos = getattr(ds, "ImagePositionPatient", None)
    position = [float(v) for v in raw_pos] if raw_pos is not None else None

    fref = str(ds.FrameOfReferenceUID) if hasattr(ds, "FrameOfReferenceUID") else None

    slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else None
    intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else None
    rescale = (slope, intercept) if slope is not None and intercept is not None else None

    # Effective VOI window: override if given, else file window if present, else None.
    if voi_window is not None:
        effective_voi: tuple[float, float] | None = voi_window
    elif has_file_window:
        # WindowCenter/WindowWidth may be multi-valued (list); take first element.
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        wc_val = float(wc[0]) if hasattr(wc, "__len__") and not isinstance(wc, str) else float(wc)
        ww_val = float(ww[0]) if hasattr(ww, "__len__") and not isinstance(ww, str) else float(ww)
        effective_voi = (wc_val, ww_val)
    else:
        effective_voi = None

    series_uid = str(ds.SeriesInstanceUID) if hasattr(ds, "SeriesInstanceUID") else None
    sop_uid = str(ds.SOPInstanceUID) if hasattr(ds, "SOPInstanceUID") else None

    meta = SpatialMeta(
        kind="dicom",
        pixel_spacing=pixel_spacing,
        orientation=orientation,
        position=position,
        frame_of_reference_uid=fref,
        rescale=rescale,
        voi_window=effective_voi,
        series_uid=series_uid,
        sop_uid=sop_uid,
    )
    return pixels, meta
