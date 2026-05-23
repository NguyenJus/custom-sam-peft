import numpy as np
import pytest
from PIL import Image

from custom_sam_peft.data.io import _coerce_to_channels, read_image


def _save_png(tmp_path, arr, name="x.png"):
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def test_pil_grayscale_rgb_rgba(tmp_path):
    rgb = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = _save_png(tmp_path, rgb)
    assert read_image(p, 3).shape == (8, 10, 3)
    assert read_image(p, 1).shape == (8, 10, 1)
    assert read_image(p, 4).shape == (8, 10, 4)


def test_pil_unsupported_channel_count_errors(tmp_path):
    rgb = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = _save_png(tmp_path, rgb)
    with pytest.raises(ValueError, match=r"channels=5"):
        read_image(p, 5)  # PIL caps at RGBA


def test_npy_hwc_and_chw(tmp_path):
    hwc = (np.random.rand(8, 10, 5)).astype(np.float32)
    chw = np.transpose(hwc, (2, 0, 1)).copy()
    p_hwc = tmp_path / "hwc.npy"
    p_chw = tmp_path / "chw.npy"
    np.save(p_hwc, hwc)
    np.save(p_chw, chw)
    assert read_image(p_hwc, 5).shape == (8, 10, 5)
    assert read_image(p_chw, 5).shape == (8, 10, 5)


def test_npy_channel_mismatch_errors(tmp_path):
    p = tmp_path / "m.npy"
    np.save(p, np.zeros((8, 10, 3), np.float32))
    with pytest.raises(ValueError, match=r"has 3 channels but .*channels=4"):
        read_image(p, 4)


def test_tiff_multiband(tmp_path):
    import tifffile

    arr = (np.random.rand(6, 8, 7)).astype(np.float32)  # H,W,C=7
    p = tmp_path / "mb.tif"
    tifffile.imwrite(p, np.transpose(arr, (2, 0, 1)))  # tifffile writes C,H,W as pages
    out = read_image(p, 7)
    assert out.shape == (6, 8, 7)


def test_npz_dispatch(tmp_path):
    arr = (np.random.rand(8, 10, 4)).astype(np.float32)
    p = tmp_path / "x.npz"
    np.savez(p, arr)  # single array stored under key "arr_0"
    assert read_image(p, 4).shape == (8, 10, 4)


def test_coerce_pil_2d_array_triplicate_and_keep1():
    arr2d = (np.random.rand(8, 10) * 255).astype(np.uint8)
    assert _coerce_to_channels(arr2d, 3).shape == (8, 10, 3)
    assert _coerce_to_channels(arr2d, 1).shape == (8, 10, 1)
