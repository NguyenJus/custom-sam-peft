"""Unit tests for the lite-mode GPU dense-IoU AP proxy (eval/proxy_map.py, #269).

CPU-only — the IoU matmul runs on CPU tensors here, the same code path that runs
on GPU in production. Run without the cov gate / GPU suite:

    pytest tests/unit/test_proxy_map.py -o "addopts=" -p no:cacheprovider
"""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from pycocotools.coco import COCO

from custom_sam_peft.eval.metrics import compute_coco_map
from custom_sam_peft.eval.proxy_map import (
    ProxyEntry,
    dense_iou_matrix,
    proxy_map_from_iou,
)

# --------------------------------------------------------------------------- #
# dense_iou_matrix
# --------------------------------------------------------------------------- #


def _box(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_dense_iou_identical_mask_is_one():
    m = _box(8, 8, 1, 5, 1, 5).unsqueeze(0)
    iou = dense_iou_matrix(m, m)
    assert iou.shape == (1, 1)
    assert iou.dtype == torch.float32
    assert iou[0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_dense_iou_disjoint_is_zero():
    a = _box(8, 8, 0, 4, 0, 4).unsqueeze(0)
    b = _box(8, 8, 4, 8, 4, 8).unsqueeze(0)
    iou = dense_iou_matrix(a, b)
    assert iou[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_dense_iou_half_overlap_hand_value():
    # pred: rows 0..4 (4 rows) cols 0..4 -> 16 px; gt: rows 2..6 cols 0..4 -> 16 px.
    # intersection rows 2..4 (2 rows) x 4 cols = 8; union = 16 + 16 - 8 = 24; iou = 1/3.
    pred = _box(8, 8, 0, 4, 0, 4).unsqueeze(0)
    gt = _box(8, 8, 2, 6, 0, 4).unsqueeze(0)
    iou = dense_iou_matrix(pred, gt)
    assert iou[0, 0].item() == pytest.approx(8.0 / 24.0, abs=1e-6)


def test_dense_iou_rectangular_matrix_shape():
    preds = torch.stack([_box(8, 8, 0, 4, 0, 4), _box(8, 8, 4, 8, 4, 8)])
    gts = torch.stack([_box(8, 8, 0, 4, 0, 4), _box(8, 8, 0, 4, 0, 4), _box(8, 8, 4, 8, 4, 8)])
    iou = dense_iou_matrix(preds, gts)
    assert iou.shape == (2, 3)
    assert iou[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert iou[1, 2].item() == pytest.approx(1.0, abs=1e-6)
    assert iou[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_dense_iou_empty_pred_or_gt_returns_correct_shape():
    pred = _box(8, 8, 0, 4, 0, 4).unsqueeze(0)
    empty_gt = torch.zeros((0, 8, 8), dtype=torch.bool)
    iou = dense_iou_matrix(pred, empty_gt)
    assert iou.shape == (1, 0)
    empty_pred = torch.zeros((0, 8, 8), dtype=torch.bool)
    iou2 = dense_iou_matrix(empty_pred, pred)
    assert iou2.shape == (0, 1)


def test_dense_iou_fp32_accum_matches_pycocotools_on_large_mask():
    # fp32 accumulation must stay exact for areas far above 2048 px (fp16 would drift).
    rng = np.random.default_rng(0)
    h = w = 200
    pred_np = (rng.random((h, w)) > 0.5).astype(np.uint8)
    gt_np = (rng.random((h, w)) > 0.5).astype(np.uint8)
    pred_t = torch.from_numpy(pred_np.astype(bool)).unsqueeze(0)
    gt_t = torch.from_numpy(gt_np.astype(bool)).unsqueeze(0)
    ours = dense_iou_matrix(pred_t, gt_t)[0, 0].item()
    pred_rle = mask_utils.encode(np.asfortranarray(pred_np))
    gt_rle = mask_utils.encode(np.asfortranarray(gt_np))
    ref = float(mask_utils.iou([pred_rle], [gt_rle], [0])[0, 0])
    assert ours == pytest.approx(ref, abs=1e-6)


# --------------------------------------------------------------------------- #
# proxy_map_from_iou — AP correctness (the five ordering-critical rules)
# --------------------------------------------------------------------------- #

THRS = [0.5, 0.75]


def _entry(image_id: int, cat: int, iou_rows: list[list[float]], scores: list[float]) -> ProxyEntry:
    iou = np.asarray(iou_rows, dtype=np.float64).reshape(len(scores), -1)
    return ProxyEntry(image_id=image_id, category_id=cat, iou=iou, scores=np.asarray(scores))


def _empty_gt_entry(image_id: int, cat: int, scores: list[float]) -> ProxyEntry:
    # m preds, 0 GT for this (image, category) -> (m, 0) matrix.
    iou = np.zeros((len(scores), 0), dtype=np.float64)
    return ProxyEntry(image_id=image_id, category_id=cat, iou=iou, scores=np.asarray(scores))


def test_single_perfect_pred_map_one():
    entries = [_entry(1, 1, [[1.0]], [0.9])]
    out = proxy_map_from_iou(entries, THRS, max_dets=100)
    assert out["mAP"] == pytest.approx(1.0, abs=1e-6)
    assert out["mAP_50"] == pytest.approx(1.0, abs=1e-6)
    assert out["mAP_75"] == pytest.approx(1.0, abs=1e-6)


def test_high_scoring_fp_before_tp_halves_ap():
    # one image, one category, one GT; an FP (iou 0) outscores the TP (iou 1.0).
    # dets in score order: FP(0.95), TP(0.9). recall maxes only after the FP.
    # AP = 0.5 at every threshold the TP clears (iou 1.0 clears both 0.5 and 0.75).
    entries = [_entry(1, 1, [[0.0], [1.0]], [0.95, 0.9])]
    out = proxy_map_from_iou(entries, THRS, max_dets=100)
    assert out["mAP"] == pytest.approx(0.5, abs=1e-6)


def test_per_category_pooling_not_per_image_mean():
    # Rule (b): pool across images into ONE PR curve, do NOT average per-image APs.
    # Image 1: GT + perfect TP (score 0.9). Image 2: GT + FP iou 0 (score 0.95).
    # Per-image mean would be (1.0 + 0.0)/2 = 0.5.
    # Pooled (npig=2): order FP(0.95), TP(0.9) -> recall [0, .5], prec env [.5,.5]
    #   -> AP = 0.5 * 0.5 = 0.25 at each threshold the TP clears.
    entries = [
        _entry(1, 1, [[1.0]], [0.9]),
        _entry(2, 1, [[0.0]], [0.95]),
    ]
    out = proxy_map_from_iou(entries, THRS, max_dets=100)
    assert out["mAP"] == pytest.approx(0.25, abs=1e-6)
    assert out["mAP"] != pytest.approx(0.5, abs=1e-3)  # not the per-image mean


def test_matching_is_independent_per_threshold():
    # Rule (c): a pred at IoU 0.6 is a TP at t=0.5 but an FP at t=0.7.
    entries = [_entry(1, 1, [[0.6]], [0.9])]
    out = proxy_map_from_iou(entries, [0.5, 0.7], max_dets=100)
    # t=0.5 -> AP 1.0 ; t=0.7 -> AP 0.0 ; mean = 0.5
    assert out["mAP"] == pytest.approx(0.5, abs=1e-6)


def test_no_gt_category_masked_before_mean():
    # Rule (d): a category with predictions but ZERO GT is excluded from the mean,
    # not counted as AP 0 (which would drag mAP to 0.5 here).
    entries = [
        _entry(1, 1, [[1.0]], [0.9]),  # cat 1: 1 GT, perfect -> AP 1.0
        _empty_gt_entry(1, 2, [0.9]),  # cat 2: pred but no GT -> excluded
    ]
    out = proxy_map_from_iou(entries, THRS, max_dets=100)
    assert out["mAP"] == pytest.approx(1.0, abs=1e-6)


def test_category_with_gt_but_no_tp_counts_zero():
    # cat with GT but only an FP -> AP 0, INCLUDED in the mean (drags it down).
    entries = [
        _entry(1, 1, [[1.0]], [0.9]),  # AP 1.0
        _entry(1, 2, [[0.0]], [0.9]),  # GT present (M=1), pred misses -> AP 0.0
    ]
    out = proxy_map_from_iou(entries, THRS, max_dets=100)
    assert out["mAP"] == pytest.approx(0.5, abs=1e-6)


def test_top_100_truncation_drops_low_scored_tp():
    # Rule (e): re-truncate to top-max_dets by score per (image, category) BEFORE
    # matching. 100 high-scoring FPs (iou 0) + 1 low-scoring TP (iou 1.0) = 101 dets.
    # With max_dets=100 the TP (rank 101) is dropped -> no TP -> AP 0.
    iou_rows = [[0.0]] * 100 + [[1.0]]
    scores = [0.9 - 0.001 * i for i in range(100)] + [0.1]
    entries = [_entry(1, 1, iou_rows, scores)]
    out = proxy_map_from_iou(entries, THRS, max_dets=100)
    assert out["mAP"] == pytest.approx(0.0, abs=1e-6)


def test_top_100_keeps_tp_when_cap_allows():
    # Same dets, but a cap of 101 keeps the TP -> AP > 0.
    iou_rows = [[0.0]] * 100 + [[1.0]]
    scores = [0.9 - 0.001 * i for i in range(100)] + [0.1]
    entries = [_entry(1, 1, iou_rows, scores)]
    out = proxy_map_from_iou(entries, THRS, max_dets=101)
    assert out["mAP"] > 0.0


def test_map_50_75_keys_present_iff_thresholds_present():
    entries = [_entry(1, 1, [[1.0]], [0.9])]
    out_50 = proxy_map_from_iou(entries, [0.5], max_dets=100)
    assert "mAP_50" in out_50 and "mAP_75" not in out_50
    out_60 = proxy_map_from_iou(entries, [0.6], max_dets=100)
    assert "mAP_50" not in out_60 and "mAP_75" not in out_60
    assert out_60["mAP"] == pytest.approx(1.0, abs=1e-6)


def test_empty_subset_returns_zero_map():
    out = proxy_map_from_iou([], THRS, max_dets=100)
    assert out["mAP"] == 0.0


# --------------------------------------------------------------------------- #
# Monotone-agreement vs pycocotools (toy rank-order gate, spike §5.3)
# --------------------------------------------------------------------------- #


def _rle(mask: np.ndarray) -> dict:
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def _toy_gt() -> COCO:
    # one image, one category, one small GT box (rows 2..6, cols 2..6) on a 16x16 grid.
    gt_mask = np.zeros((16, 16), dtype=np.uint8)
    gt_mask[2:6, 2:6] = 1
    gt = COCO()
    gt.dataset = {
        "images": [{"id": 1, "height": 16, "width": 16}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [2, 2, 4, 4],
                "area": 16,
                "segmentation": _rle(gt_mask),
            }
        ],
    }
    gt.createIndex()
    return gt, gt_mask


def _coco_pred(mask: np.ndarray, score: float) -> dict:
    ys, xs = np.where(mask)
    if len(ys):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
    else:
        bbox = [0, 0, 0, 0]
    return {
        "image_id": 1,
        "category_id": 1,
        "bbox": bbox,
        "score": score,
        "segmentation": _rle(mask),
    }


def test_proxy_preserves_exact_ranking_precise_vs_flooding():
    gt, gt_mask = _toy_gt()

    precise_mask = np.zeros((16, 16), dtype=np.uint8)
    precise_mask[2:6, 2:7] = 1  # tight, high IoU with GT

    flood_mask = np.ones((16, 16), dtype=np.uint8)  # whole-image flood, tiny IoU

    # --- exact pycocotools ---
    exact_precise = compute_coco_map(
        [_coco_pred(precise_mask, 0.9)], gt, [0.5, 0.75], include_per_class=False
    ).overall["mAP"]
    gt2, _ = _toy_gt()
    exact_flood = compute_coco_map(
        [_coco_pred(flood_mask, 0.9)], gt2, [0.5, 0.75], include_per_class=False
    ).overall["mAP"]
    assert exact_precise > exact_flood  # sanity: exact ranks precise above flooding

    # --- proxy on the SAME masks ---
    gt_t = torch.from_numpy(gt_mask.astype(bool)).unsqueeze(0)
    iou_precise = dense_iou_matrix(
        torch.from_numpy(precise_mask.astype(bool)).unsqueeze(0), gt_t
    ).numpy()
    iou_flood = dense_iou_matrix(
        torch.from_numpy(flood_mask.astype(bool)).unsqueeze(0), gt_t
    ).numpy()
    proxy_precise = proxy_map_from_iou(
        [ProxyEntry(1, 1, iou_precise, np.asarray([0.9]))], [0.5, 0.75], max_dets=100
    )["mAP"]
    proxy_flood = proxy_map_from_iou(
        [ProxyEntry(1, 1, iou_flood, np.asarray([0.9]))], [0.5, 0.75], max_dets=100
    )["mAP"]

    # The load-bearing claim: proxy reproduces the exact ordering.
    assert proxy_precise > proxy_flood
