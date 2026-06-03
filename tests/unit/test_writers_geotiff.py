"""CPU-only: GeoTIFF mask writer (spec §7.1; C7, C9)."""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import rasterio
from rasterio.transform import from_origin

from custom_sam_peft.data.spatial_meta import SpatialMeta
from custom_sam_peft.predict.writers import write_geotiff_mask, write_geotiff_masks


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
        assert (read[1:, :] == 1).all()  # non-nodata pixels remain 1
        assert src.nodata == 0


def _make_rle(binary_mask: np.ndarray) -> dict:
    """Encode a binary uint8 mask to a pycocotools RLE dict with ASCII counts."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def test_write_geotiff_masks_emission_loop(tmp_path):
    """Geo-emission loop: correct file naming, instance counter, CRS, nodata, and
    skips image_ids not present in id_to_spatial_meta."""
    H, W = 20, 24
    transform = from_origin(500000, 4600000, 10, 10)
    nodata_mask = np.zeros((H, W), bool)
    nodata_mask[0, :] = True  # top row nodata
    meta = SpatialMeta(
        kind="geo", crs="EPSG:32633", affine=transform, nodata=0, nodata_mask=nodata_mask
    )

    # Two entries for image_id=1, same category → instance counter 0 then 1.
    binary = np.ones((H, W), np.uint8)
    entry_a = {"image_id": 1, "category_id": 3, "segmentation": _make_rle(binary), "score": 0.9}
    entry_b = {"image_id": 1, "category_id": 3, "segmentation": _make_rle(binary), "score": 0.8}
    # Entry for image_id=2 — NOT in id_to_spatial_meta → must produce no file.
    entry_c = {"image_id": 2, "category_id": 3, "segmentation": _make_rle(binary), "score": 0.7}

    all_predictions = [entry_a, entry_b, entry_c]
    id_to_spatial_meta = {1: meta}
    id_to_stem = {1: "scene", 2: "other"}
    originals = {1: (H, W), 2: (H, W)}

    write_geotiff_masks(all_predictions, id_to_spatial_meta, id_to_stem, originals, tmp_path)

    # First instance: scene_3_0.tif
    tif0 = tmp_path / "masks" / "scene_3_0.tif"
    assert tif0.exists(), "scene_3_0.tif must exist"
    with rasterio.open(tif0) as src:
        assert "32633" in str(src.crs)
        assert tuple(src.transform)[:6] == tuple(transform)[:6]
        read = src.read(1)
        assert (read[0, :] == 0).all()  # nodata row zeroed
        assert (read[1:, :] == 1).all()  # non-nodata pixels are 1
        assert src.nodata == 0

    # Second instance of the same (image, cat): scene_3_1.tif
    tif1 = tmp_path / "masks" / "scene_3_1.tif"
    assert tif1.exists(), "scene_3_1.tif must exist (counter increments)"

    # image_id=2 not in id_to_spatial_meta → no file written
    assert not (tmp_path / "masks" / "other_3_0.tif").exists(), (
        "image_id=2 not in id_to_spatial_meta — no file should be written"
    )


def test_write_geotiff_masks_noop_on_empty_meta(tmp_path):
    """write_geotiff_masks must be a no-op when id_to_spatial_meta is empty."""
    binary = np.ones((10, 12), np.uint8)
    entry = {"image_id": 1, "category_id": 1, "segmentation": _make_rle(binary), "score": 0.9}
    write_geotiff_masks([entry], {}, {1: "scene"}, {1: (10, 12)}, tmp_path)
    assert not (tmp_path / "masks").exists()
