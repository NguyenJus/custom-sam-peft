"""DICOM reads behind the optional [dicom] extra (spec §8). pydicom/nibabel are
lazy-imported so base install/import never requires them."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from os import PathLike

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


def _slice_normal(orientation: Any) -> np.ndarray:
    """Slice-normal unit vector = cross product of the two ImageOrientationPatient
    direction-cosine triplets (row dir cross column dir)."""
    iop = np.asarray(orientation, dtype=float).reshape(2, 3)
    row_dir, col_dir = iop[0], iop[1]
    return np.cross(row_dir, col_dir)


def group_series(paths: list[str | PathLike[str]]) -> dict[str, list[Any]]:
    """Bucket `.dcm` files by SeriesInstanceUID and order each bucket geometrically.

    Each bucket is sorted by the projection of ``ImagePositionPatient`` onto the
    slice-normal (cross product of the two ``ImageOrientationPatient`` direction
    cosines) — the standard geometric slice ordering (nibabel DICOM-orientation
    convention, https://nipy.org/nibabel/dicom/dicom_orientation.html). Returns
    ``{series_uid: [ordered pydicom datasets]}``.
    """
    pydicom = _require_pydicom()

    buckets: dict[str, list[Any]] = {}
    for path in paths:
        ds = pydicom.dcmread(path)
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        buckets.setdefault(uid, []).append(ds)

    ordered: dict[str, list[Any]] = {}
    for uid, datasets in buckets.items():
        # Sort by projection of ImagePositionPatient onto the slice normal. Datasets
        # missing geometry fall back to projection 0.0 (degenerate, but never crashes
        # grouping itself — series_affine raises the §11.4 error for multi-slice).
        def _proj(ds: Any) -> float:
            pos = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            if pos is None or iop is None:
                return 0.0
            return float(np.dot(np.asarray(pos, dtype=float), _slice_normal(iop)))

        ordered[uid] = sorted(datasets, key=_proj)
    return ordered


def series_affine(ordered_datasets: list[Any]) -> np.ndarray:
    """Build the 4x4 voxel->world affine for an ordered DICOM series (spec §8.2).

    Follows nibabel's documented DICOM->world convention
    (https://nipy.org/nibabel/dicom/dicom_orientation.html): DICOM patient
    coordinates are LPS+ (x->Left, y->Posterior, z->Superior) while NIfTI/nibabel
    world space is RAS+, so the L and P axes are negated (flip the sign of the
    first two rows of the rotation/translation) to convert LPS->RAS.

    Construction (nibabel "Defining the affine"): columns of the 3x3 are the
    direction cosines scaled by spacing -
      - col 0 = row-direction cosine * column-spacing (PixelSpacing[1], between cols),
      - col 1 = column-direction cosine * row-spacing (PixelSpacing[0], between rows),
      - col 2 = slice-normal * inter-slice spacing (distance between consecutive
        ImagePositionPatient along the normal),
    and the translation is the first slice's ImagePositionPatient - all in LPS,
    then negated on the L/P axes for RAS+.

    Raises ValueError (spec §11.4) when a multi-slice series lacks the geometry tags
    required to stack into a volume. A single slice without geometry degrades to a
    2D mask upstream (the runner handles that) and never reaches this function.
    """
    if not ordered_datasets:
        raise ValueError("series_affine requires at least one slice")

    first = ordered_datasets[0]
    iop = getattr(first, "ImageOrientationPatient", None)
    ipp = getattr(first, "ImagePositionPatient", None)
    spacing = getattr(first, "PixelSpacing", None)
    if iop is None or ipp is None or spacing is None:
        raise ValueError(
            "DICOM series is missing required geometry tags "
            "(ImageOrientationPatient / ImagePositionPatient / PixelSpacing); "
            "cannot stack into a NIfTI volume. A single slice without geometry "
            "degrades to a 2D mask."
        )

    iop_arr = np.asarray(iop, dtype=float).reshape(2, 3)
    row_dir, col_dir = iop_arr[0], iop_arr[1]
    normal = np.cross(row_dir, col_dir)
    row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
    pos0 = np.asarray(ipp, dtype=float)

    # Inter-slice spacing: signed distance between the first two slices along the
    # normal. Single-slice series default to row-spacing (no z extent to measure).
    if len(ordered_datasets) >= 2:
        ipp_next = getattr(ordered_datasets[1], "ImagePositionPatient", None)
        if ipp_next is None:
            raise ValueError(
                "DICOM series slice is missing ImagePositionPatient; cannot derive "
                "inter-slice spacing for a multi-slice NIfTI volume."
            )
        slice_spacing = float(np.dot(np.asarray(ipp_next, dtype=float) - pos0, normal))
    else:
        slice_spacing = row_spacing

    affine = np.eye(4, dtype=float)
    affine[:3, 0] = row_dir * col_spacing
    affine[:3, 1] = col_dir * row_spacing
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = pos0
    # LPS (DICOM) → RAS+ (nibabel/NIfTI): negate the L and P (x, y) axes.
    affine[0, :] *= -1.0
    affine[1, :] *= -1.0
    return affine
