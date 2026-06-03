"""Geo-TIFF reading tests: C7, C13, C14 (spec §6.2, §6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from custom_sam_peft.data.io import read_image, read_image_with_meta


def _write_geotiff(path, arr_hwc, crs="EPSG:32633", nodata=None):
    """Write a geo-referenced TIFF with CRS + affine transform."""
    import rasterio
    from rasterio.transform import from_origin

    h, w, c = arr_hwc.shape
    transform = from_origin(500000, 4600000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=arr_hwc.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for b in range(c):
            dst.write(arr_hwc[:, :, b], b + 1)
    return transform


def test_C7_geotiff_read_carries_crs_affine(tmp_path):
    """C7: geo TIFF → pixels (H,W,C) + meta.kind=='geo' with correct CRS/affine."""
    arr = (np.random.rand(20, 24, 3) * 255).astype(np.uint8)
    p = tmp_path / "geo.tif"
    transform = _write_geotiff(p, arr)
    pixels, meta = read_image_with_meta(p, 3)
    assert pixels.shape == (20, 24, 3)
    assert meta is not None
    assert meta.kind == "geo"
    assert "32633" in str(meta.crs)
    assert tuple(meta.affine)[:6] == pytest.approx(tuple(transform)[:6])


def test_C13_plain_tiff_returns_none_meta(tmp_path):
    """C13: plain (non-geo) TIFF → pixels OK, meta is None."""
    import tifffile  # dev fixture only — writes a plain non-geo TIFF

    arr = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, arr)
    pixels, meta = read_image_with_meta(p, 3)
    assert pixels.shape == (8, 10, 3)
    assert meta is None


def test_C14_band_count_mismatch_raises(tmp_path):
    """C14: requesting channels≠band count raises ValueError naming the requested count."""
    arr = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = tmp_path / "rgb.tif"
    _write_geotiff(p, arr)
    with pytest.raises(ValueError, match=r"channels=4"):
        read_image(p, 4)


def test_C9_nodata_pixels_zero_filled(tmp_path):
    """C9: nodata pixels are zero-filled; nodata_mask marks their locations."""
    arr = (np.ones((8, 10, 3)) * 128).astype(np.uint8)
    # Inject a nodata sentinel value (200) in one pixel
    arr[3, 5, :] = 200
    p = tmp_path / "nodata.tif"
    _write_geotiff(p, arr, nodata=200)
    pixels, meta = read_image_with_meta(p, 3)
    assert pixels.shape == (8, 10, 3)
    assert meta is not None
    assert meta.nodata == 200
    assert meta.nodata_mask is not None
    # The nodata pixel should be zero-filled
    assert np.all(pixels[3, 5, :] == 0)
    # Non-nodata pixels should be unchanged
    assert np.all(pixels[0, 0, :] == 128)
    # Nodata mask correctly identifies the sentinel pixel
    assert meta.nodata_mask[3, 5] is True or bool(meta.nodata_mask[3, 5])
    assert not meta.nodata_mask[0, 0]


def test_read_image_tiff_pixels_only(tmp_path):
    """read_image on a geo TIFF returns pixels only (no meta), same shape as read_image_with_meta."""  # noqa: E501
    arr = (np.random.rand(12, 16, 3) * 255).astype(np.uint8)
    p = tmp_path / "geo2.tif"
    _write_geotiff(p, arr)
    pixels_only = read_image(p, 3)
    pixels_meta, _ = read_image_with_meta(p, 3)
    assert pixels_only.shape == (12, 16, 3)
    np.testing.assert_array_equal(pixels_only, pixels_meta)
