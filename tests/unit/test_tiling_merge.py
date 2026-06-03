import numpy as np

from custom_sam_peft.data.tiling import Fragment, merge_fragments


def _box(canvas, y0, y1, x0, x1, cat=1, score=0.9, wid=0):
    m = np.zeros(canvas, bool)
    m[y0:y1, x0:x1] = True
    return Fragment(mask=m, score=score, category_id=cat, window_id=wid)


def test_C2_seam_object_merges_to_one():
    canvas = (40, 40)
    # one object split across a seam at x=20, overlapping in [18,22)
    left = _box(canvas, 10, 30, 5, 22, score=1.0, wid=0)
    right = _box(canvas, 10, 30, 18, 35, score=0.6, wid=1)
    merged = merge_fragments([left, right], canvas)
    assert len(merged) == 1
    ref = left.mask | right.mask
    assert np.array_equal(merged[0].mask, ref)  # logical OR within the component
    # area-weighted mean of scores (left larger -> closer to 1.0)
    la, ra = left.mask.sum(), right.mask.sum()
    expected = (1.0 * la + 0.6 * ra) / (la + ra)
    assert abs(merged[0].score - expected) < 1e-6


def test_C3_three_tile_transitive_merge():
    canvas = (20, 90)
    a = _box(canvas, 5, 15, 0, 35, wid=0)
    b = _box(canvas, 5, 15, 30, 65, wid=1)  # overlaps a in [30,35)
    c = _box(canvas, 5, 15, 60, 90, wid=2)  # overlaps b in [60,65); NOT a
    merged = merge_fragments([a, b, c], canvas)
    assert len(merged) == 1  # transitive A-B-C via union-find


def test_C4_distinct_and_cross_category_stay_separate():
    canvas = (20, 60)
    # same category, NO band overlap -> separate
    o1 = _box(canvas, 5, 15, 0, 20, cat=1, wid=0)
    o2 = _box(canvas, 5, 15, 30, 50, cat=1, wid=1)
    assert len(merge_fragments([o1, o2], canvas)) == 2
    # overlapping but DIFFERENT category -> never merge
    a = _box(canvas, 5, 15, 0, 30, cat=1, wid=0)
    b = _box(canvas, 5, 15, 20, 50, cat=2, wid=1)
    assert len(merge_fragments([a, b], canvas)) == 2


def test_threshold_boundary_strict_greater_and_near_miss():
    canvas = (20, 60)
    # below threshold: IoM = 8/100 = 0.08 -> incidental adjacency stays separate
    a = _box(canvas, 0, 10, 0, 10, cat=1, wid=0)  # 100 px
    b = _box(canvas, 0, 8, 9, 40, cat=1, wid=1)  # inter = 8 px -> 0.08
    assert len(merge_fragments([a, b], canvas)) == 2
    # exactly at threshold: IoM = 10/100 = 0.10; strict '>' keeps them separate
    a2 = _box(canvas, 0, 10, 0, 10, cat=1, wid=0)  # 100 px
    b2 = _box(canvas, 0, 10, 9, 40, cat=1, wid=1)  # inter = 10 px -> 0.10
    assert len(merge_fragments([a2, b2], canvas)) == 2
    # above threshold: IoM = 20/100 = 0.20 -> merge
    a3 = _box(canvas, 0, 10, 0, 10, cat=1, wid=0)  # 100 px
    b3 = _box(canvas, 0, 11, 8, 40, cat=1, wid=1)  # inter = 20 px -> 0.20
    assert len(merge_fragments([a3, b3], canvas)) == 1
