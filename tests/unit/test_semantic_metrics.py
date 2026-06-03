"""compute_semantic_metrics on hand-built confusion matrices (§8.1)."""

from __future__ import annotations

import numpy as np

from custom_sam_peft.eval.metrics import compute_semantic_metrics


def test_perfect_prediction_miou_one():
    # 3 classes (bg + 2 concepts), diagonal confusion -> mIoU == 1.
    conf = np.diag([10, 20, 30]).astype(np.int64)
    m = compute_semantic_metrics(conf, class_names=["road", "tree"])
    assert m.mean_iou == 1.0
    assert m.pixel_accuracy == 1.0
    assert set(m.per_class_iou.keys()) == {"background", "road", "tree"}
    assert m.per_class_iou["road"] == 1.0


def test_iou_formula_tp_fp_fn():
    # class 1: TP=8, FP=2 (col1 row0), FN=2 (row1 col0). IoU = 8/(8+2+2)=0.666...
    conf = np.array([[10, 2, 0], [2, 8, 0], [0, 0, 5]], dtype=np.int64)
    m = compute_semantic_metrics(conf, class_names=["a", "b"])
    assert abs(m.per_class_iou["a"] - 8 / 12) < 1e-9


def test_no_gt_class_skipped_from_miou():
    # class 2 has zero GT pixels (row all-zero) -> omitted from mIoU.
    conf = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=np.int64)
    m = compute_semantic_metrics(conf, class_names=["a", "b"])
    # only bg + a have GT; mIoU = mean over those two.
    assert "b" in m.per_class_iou  # reported but...
    # mIoU computed over classes-with-GT only.
    assert abs(m.mean_iou - 1.0) < 1e-9


def test_pixel_accuracy_is_trace_over_total():
    conf = np.array([[8, 2], [0, 10]], dtype=np.int64)
    m = compute_semantic_metrics(conf, class_names=["x"])  # bg + 1 concept
    assert abs(m.pixel_accuracy - 18 / 20) < 1e-9
