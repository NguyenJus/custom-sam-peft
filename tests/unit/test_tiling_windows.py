import numpy as np

from custom_sam_peft.data.tiling import Window, iter_windows, tiling_engaged
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE


def test_small_image_single_window():
    # C1/C5: max(edge) <= tile -> exactly one full-image window (direct-path equivalent)
    ws = iter_windows(800, 1008, tile=SAM3_IMAGE_SIZE, overlap=0.25)
    assert ws == [Window(y0=0, x0=0, h=800, w=1008)]
    assert tiling_engaged(800, 1008) is False
    assert tiling_engaged(1008, 1008) is False  # boundary: exactly 1008 -> direct


def test_oversized_engages_and_covers():
    # C5: > 1008 engages tiling
    assert tiling_engaged(1500, 1500) is True
    ws = iter_windows(1500, 1500, tile=1008, overlap=0.25)
    # Cover the whole image: every pixel is inside at least one window.
    cover = np.zeros((1500, 1500), bool)
    for w in ws:
        assert w.h <= 1008 and w.w <= 1008
        cover[w.y0 : w.y0 + w.h, w.x0 : w.x0 + w.w] = True
    assert cover.all()


def test_edge_windows_clamp_flush():
    # C1: last window flush to the edge, no margin dropped
    ws = iter_windows(1009, 1009, tile=1008, overlap=0.25)
    assert max(w.x0 + w.w for w in ws) == 1009
    assert max(w.y0 + w.h for w in ws) == 1009
    # last column window is flush-right (x0 + w == 1009)
    assert any(w.x0 + w.w == 1009 and w.x0 > 0 for w in ws)


def test_overlap_band_width_matches_fraction():
    # C1: step = round(tile * (1 - overlap)); adjacent windows overlap by ~ tile*overlap
    ws = iter_windows(2000, 1008, tile=1008, overlap=0.25)
    ys = sorted({w.y0 for w in ws})
    step = ys[1] - ys[0]
    assert step == round(1008 * (1 - 0.25))  # 756


def test_non_square_oversized_covers():
    # Both axes oversized and unequal: guards the 2-D window product.
    ws = iter_windows(1500, 2000, tile=1008, overlap=0.25)
    cover = np.zeros((1500, 2000), bool)
    for w in ws:
        assert w.h <= 1008 and w.w <= 1008
        cover[w.y0 : w.y0 + w.h, w.x0 : w.x0 + w.w] = True
    assert cover.all()
    assert max(w.y0 + w.h for w in ws) == 1500
    assert max(w.x0 + w.w for w in ws) == 2000
