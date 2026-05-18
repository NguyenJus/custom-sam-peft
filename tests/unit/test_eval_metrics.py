"""pycocotools-wrapper tests for eval/metrics.py."""

from __future__ import annotations

import logging

import numpy as np
import pycocotools.mask as mask_utils
import pytest
from pycocotools.coco import COCO

from esam3.eval.metrics import compute_coco_map


def _build_gt(images: list[dict], categories: list[dict], anns: list[dict]) -> COCO:
    gt = COCO()
    gt.dataset = {"images": images, "categories": categories, "annotations": anns}
    gt.createIndex()
    return gt


def _rle(mask: np.ndarray) -> dict:
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def _full_mask_rle(h: int, w: int) -> dict:
    return _rle(np.ones((h, w), dtype=np.uint8))


def test_perfect_predictions_get_map_one():
    images = [{"id": 1, "height": 8, "width": 8}]
    categories = [{"id": 1, "name": "cat"}]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 8, 8],
            "area": 64,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    gt = _build_gt(images, categories, anns)
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 8, 8],
            "score": 1.0,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    report = compute_coco_map(preds, gt, [0.5, 0.75], include_per_class=True)
    assert report.overall["mAP"] == pytest.approx(1.0, abs=1e-6)
    assert report.overall["mAP_50"] == pytest.approx(1.0, abs=1e-6)
    assert report.overall["mAP_75"] == pytest.approx(1.0, abs=1e-6)
    assert "cat" in report.per_class
    assert report.per_class["cat"]["AP"] == pytest.approx(1.0, abs=1e-6)
    assert report.per_class["cat"]["AP_50"] == pytest.approx(1.0, abs=1e-6)
    assert report.n_predictions == 1


def test_zero_predictions_returns_zeroed_report(caplog: pytest.LogCaptureFixture):
    images = [{"id": 1, "height": 8, "width": 8}]
    categories = [{"id": 1, "name": "cat"}]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 8, 8],
            "area": 64,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    gt = _build_gt(images, categories, anns)
    with caplog.at_level(logging.WARNING, logger="esam3.eval.metrics"):
        report = compute_coco_map([], gt, [0.5], include_per_class=True)
    assert report.overall == {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    assert report.per_class == {}
    assert report.n_predictions == 0
    assert any("no predictions" in rec.message.lower() for rec in caplog.records)


def test_class_with_zero_gt_filtered_from_per_class():
    images = [{"id": 1, "height": 8, "width": 8}]
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 8, 8],
            "area": 64,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    gt = _build_gt(images, categories, anns)
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 8, 8],
            "score": 1.0,
            "segmentation": _full_mask_rle(8, 8),
        },
        {
            "image_id": 1,
            "category_id": 2,
            "bbox": [0, 0, 8, 8],
            "score": 1.0,
            "segmentation": _full_mask_rle(8, 8),
        },
    ]
    report = compute_coco_map(preds, gt, [0.5], include_per_class=True)
    assert "cat" in report.per_class
    assert "dog" not in report.per_class


def test_include_per_class_false_skips_per_class():
    images = [{"id": 1, "height": 8, "width": 8}]
    categories = [{"id": 1, "name": "cat"}]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 8, 8],
            "area": 64,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    gt = _build_gt(images, categories, anns)
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 8, 8],
            "score": 1.0,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    report = compute_coco_map(preds, gt, [0.5], include_per_class=False)
    assert report.per_class == {}
    assert report.overall["mAP_50"] == pytest.approx(1.0, abs=1e-6)


def test_iou_threshold_passthrough():
    """mAP_50/mAP_75 appear iff 0.5/0.75 are in iou_thresholds; mAP is mean over the given list."""
    images = [{"id": 1, "height": 8, "width": 8}]
    categories = [{"id": 1, "name": "cat"}]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 8, 8],
            "area": 64,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 8, 8],
            "score": 1.0,
            "segmentation": _full_mask_rle(8, 8),
        }
    ]

    # --- iou_thresholds=[0.5] only: mAP_50 present, mAP_75 absent ---
    gt = _build_gt(images, categories, anns)
    report_50 = compute_coco_map(preds, gt, [0.5], include_per_class=False)
    assert "mAP_50" in report_50.overall
    assert "mAP_75" not in report_50.overall
    assert report_50.overall["mAP_50"] == pytest.approx(1.0, abs=1e-6)
    assert report_50.overall["mAP"] == pytest.approx(1.0, abs=1e-6)

    # --- non-standard list [0.3, 0.5, 0.7]: neither convenience key for 0.75 ---
    gt2 = _build_gt(images, categories, anns)
    report_custom = compute_coco_map(preds, gt2, [0.3, 0.5, 0.7], include_per_class=False)
    assert "mAP_50" in report_custom.overall  # 0.5 is present
    assert "mAP_75" not in report_custom.overall  # 0.75 is absent
    # mAP is the mean over the 3 thresholds — with perfect preds it should be 1.0
    assert report_custom.overall["mAP"] == pytest.approx(1.0, abs=1e-6)
    assert report_custom.overall["mAP_50"] == pytest.approx(1.0, abs=1e-6)

    # --- iou_thresholds=[0.6]: neither mAP_50 nor mAP_75 ---
    gt3 = _build_gt(images, categories, anns)
    report_60 = compute_coco_map(preds, gt3, [0.6], include_per_class=False)
    assert "mAP_50" not in report_60.overall
    assert "mAP_75" not in report_60.overall
    assert report_60.overall["mAP"] == pytest.approx(1.0, abs=1e-6)
