"""CPU-only: GeoTIFF mask writer (spec §7.1; C7, C9)."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from custom_sam_peft.data.spatial_meta import SpatialMeta
from custom_sam_peft.predict.writers import write_geotiff_mask


def test_C7_C9_geotiff_mask_roundtrips_crs_affine_and_nodata(tmp_path):
    transform = from_origin(500000, 4600000, 10, 10)
    nodata_mask = np.zeros((20, 24), bool)
    nodata_mask[0, :] = True  # top row is nodata
    meta = SpatialMeta(
        kind="geo", crs="EPSG:32633", affine=transform, nodata=0, nodata_mask=nodata_mask
    )
    mask = np.ones((20, 24), np.uint8)
    out = tmp_path / "mask.tif"
    write_geotiff_mask(mask, meta, out)
    with rasterio.open(out) as src:
        assert "32633" in str(src.crs)
        assert tuple(src.transform)[:6] == tuple(transform)[:6]
        read = src.read(1)
        assert (read[0, :] == 0).all()  # nodata re-marked to 0 in the output
