"""Optional pixels-first spatial-metadata sidecar (spec §6.2). Tagged union by
source `kind`. NEVER reaches the model — carried read->dataset->writers for
output reconstruction only. Default None for plain images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SpatialMeta:
    kind: Literal["geo", "dicom"]
    # geo (rasterio)
    crs: Any = None
    affine: Any = None
    nodata: float | None = None
    nodata_mask: Any = None  # bool ndarray | None
    # dicom (pydicom)
    pixel_spacing: Any = None
    orientation: Any = None  # ImageOrientationPatient
    position: Any = None  # ImagePositionPatient
    frame_of_reference_uid: str | None = None
    rescale: tuple[float, float] | None = None  # (slope, intercept)
    voi_window: tuple[float, float] | None = None  # (center, width)
    series_uid: str | None = None
    sop_uid: str | None = None
