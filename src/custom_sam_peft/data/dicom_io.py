"""DICOM reads behind the optional [dicom] extra (spec §8). pydicom/nibabel are
lazy-imported so base install/import never requires them."""

from __future__ import annotations

import types

_MISSING = "DICOM support requires the optional extra: pip install custom-sam-peft[dicom]"


def _require_pydicom() -> types.ModuleType:
    """Import pydicom or raise an actionable RuntimeError."""
    try:
        import pydicom

        return pydicom
    except ImportError as exc:
        raise RuntimeError(_MISSING) from exc


def read_dcm_with_meta(path: object, channels: int) -> tuple[object, object]:
    """Read a DICOM file -> (pixels (H,W,C), SpatialMeta(kind='dicom')).

    Body filled in Task 3.2; for now it just enforces the [dicom] extra is present.
    """
    _require_pydicom()
    raise NotImplementedError("DICOM decode lands in Task 3.2")  # replaced next task
