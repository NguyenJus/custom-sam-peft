"""Unit coverage for the Evaluator output schema (MetricsReport).

C1 per spec §6.2. Pinned on CPU because the schema invariants do not depend
on real model output — see spec §7 for why metric *values* (T2/T3) need GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from custom_sam_peft.eval.metrics import MetricsReport, compute_coco_map


def _load_gt(tiny_coco_dir: Path) -> COCO:
    return COCO(str(tiny_coco_dir / "annotations.json"))


def _perfect_prediction_from_first_gt(gt: COCO) -> list[dict[str, object]]:
    """Return a single COCO-results entry that exactly matches the first GT annotation."""
    ann_ids = gt.getAnnIds()
    assert ann_ids, "tiny_coco has no annotations — fixture broken"
    ann = gt.loadAnns(ann_ids[:1])[0]
    img = gt.loadImgs([ann["image_id"]])[0]
    # Synthesize an RLE mask matching the bbox extent.
    h, w = img["height"], img["width"]
    bin_mask = np.zeros((h, w), dtype=np.uint8, order="F")
    x, y, bw, bh = ann["bbox"]
    x0, y0 = int(x), int(y)
    x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
    bin_mask[y0:y1, x0:x1] = 1
    rle = mask_utils.encode(bin_mask)
    rle["counts"] = rle["counts"].decode("ascii")  # COCO results expect str
    return [
        {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "segmentation": rle,
            "score": 1.0,
        }
    ]


def test_empty_predictions_returns_zeroed_report(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    report = compute_coco_map(
        predictions=[],
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75, 0.95],
        include_per_class=True,
    )
    assert isinstance(report, MetricsReport)
    assert report.overall == {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    assert report.per_class == {}
    assert report.n_predictions == 0
    assert report.n_images == len(gt.imgs)  # tiny_coco has 2


def test_iou_thresholds_pick_only_50(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5],
        include_per_class=False,
    )
    assert "mAP" in report.overall
    assert "mAP_50" in report.overall
    assert "mAP_75" not in report.overall


def test_iou_thresholds_pick_only_75(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.75],
        include_per_class=False,
    )
    assert "mAP" in report.overall
    assert "mAP_75" in report.overall
    assert "mAP_50" not in report.overall


def test_overall_keys_finite(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75, 0.95],
        include_per_class=True,
    )
    for k, v in report.overall.items():
        assert isinstance(v, (int, float, np.floating)), f"{k} not numeric: {type(v)}"
        assert math.isfinite(v), f"{k} not finite: {v}"
        assert 0.0 <= v <= 1.0, f"{k} outside [0,1]: {v}"


def test_per_class_skips_classes_without_gt(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75],
        include_per_class=True,
    )
    # Every per_class row is keyed by a category name (str) and has a finite "AP".
    assert report.per_class, "per_class empty despite include_per_class=True with valid GT"
    for cat_name, row in report.per_class.items():
        assert isinstance(cat_name, str)
        assert "AP" in row
        assert math.isfinite(row["AP"])


def test_include_per_class_false_returns_empty_per_class(tiny_coco_dir: Path) -> None:
    gt = _load_gt(tiny_coco_dir)
    preds = _perfect_prediction_from_first_gt(gt)
    report = compute_coco_map(
        predictions=preds,
        ground_truth=gt,
        iou_thresholds=[0.5, 0.75],
        include_per_class=False,
    )
    assert report.per_class == {}
    assert "mAP" in report.overall


# ---------------------------------------------------------------------------
# EvalConfig.batch_size knob (T10)
# ---------------------------------------------------------------------------


def test_eval_config_batch_size_default_auto() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    cfg = EvalConfig()
    assert cfg.batch_size == "auto"


def test_eval_config_batch_size_accepts_positive_int() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    assert EvalConfig(batch_size=4).batch_size == 4


def test_eval_config_batch_size_rejects_zero() -> None:
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import EvalConfig

    with pytest.raises(ValidationError):
        EvalConfig(batch_size=0)
