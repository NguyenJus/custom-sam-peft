import numpy as np

from custom_sam_peft.data.tiling import Fragment, iter_windows, run_windows


def test_run_windows_offsets_fragments_to_canvas():
    img = np.zeros((1500, 1500, 3), np.uint8)
    windows = iter_windows(1500, 1500, tile=1008, overlap=0.25)

    def fn(crop, window):
        # A fake per-tile "forward": one fragment, a 5x5 box at tile-local (0,0).
        m = np.zeros(crop.shape[:2], bool)
        m[0:5, 0:5] = True
        return [Fragment(mask=m, score=1.0, category_id=1, window_id=window.y0 * 99999 + window.x0)]

    frags = run_windows(img, windows, fn)
    assert len(frags) == len(windows)
    # Each returned fragment mask must be full-canvas sized, offset by window origin.
    for f, win in zip(frags, windows, strict=True):
        assert f.mask.shape == (1500, 1500)
        assert f.mask[win.y0, win.x0]  # box landed at the window origin on the canvas


def test_run_windows_offset_math_nonzero_origin():
    img = np.zeros((1500, 1500, 3), np.uint8)
    windows = iter_windows(1500, 1500, tile=1008, overlap=0.25)
    # pick a window whose origin is NOT (0, 0)
    win = next(w for w in windows if w.y0 > 0 and w.x0 > 0)

    def fn(crop, window):
        m = np.zeros(crop.shape[:2], bool)
        m[3:5, 4:6] = True  # tile-local box at (3,4)
        return [Fragment(mask=m, score=1.0, category_id=1, window_id=7)]

    frags = run_windows(img, [win], fn)
    assert len(frags) == 1
    placed = frags[0].mask
    assert placed[win.y0 + 3, win.x0 + 4]  # offset by window origin
    assert not placed[3, 4]  # NOT dumped at canvas origin
    assert not placed[win.y0 + 5, win.x0 + 5]  # outside the box
