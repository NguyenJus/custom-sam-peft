"""COCO mAP + per-class AP via pycocotools."""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass, field

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsReport:
    """Result of an Evaluator.evaluate() call."""

    overall: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    n_images: int = 0
    n_predictions: int = 0


def _silent_evaluate(coco_eval: COCOeval) -> None:
    """Run COCOeval.evaluate/accumulate/summarize without pycocotools' prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def _zero_overall() -> dict[str, float]:
    return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}


def compute_coco_map(
    predictions: list[dict[str, object]],
    ground_truth: COCO,
    iou_thresholds: list[float],
    include_per_class: bool,
) -> MetricsReport:
    """Score predictions against ground_truth and return a MetricsReport.

    Predictions are COCO results entries (one per query, with RLE mask). The
    function uses segmentation IoU (not box IoU) for matching. ``mAP`` is the
    mean over ``iou_thresholds``; ``mAP_50`` and ``mAP_75`` are slices at the
    corresponding thresholds (only when those thresholds appear in
    ``iou_thresholds``; otherwise the slice is omitted).
    """
    n_images = len(ground_truth.imgs)
    if not predictions:
        _LOG.warning("compute_coco_map: no predictions; returning zeroed report")
        return MetricsReport(
            overall=_zero_overall(),
            per_class={},
            n_images=n_images,
            n_predictions=0,
        )

    coco_dt = ground_truth.loadRes(predictions)
    coco_eval = COCOeval(ground_truth, coco_dt, iouType="segm")
    coco_eval.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
    _silent_evaluate(coco_eval)

    precision = coco_eval.eval["precision"]  # (T, R, K, A, M)
    iou_list = list(iou_thresholds)

    # mAP: mean over all T thresholds, all categories, area="all", maxDets last
    valid_all = precision[:, :, :, 0, -1]
    valid_all = valid_all[valid_all > -1]
    overall: dict[str, float] = {"mAP": float(valid_all.mean()) if valid_all.size else 0.0}

    if 0.5 in iou_list:
        idx50 = iou_list.index(0.5)
        p50 = precision[idx50, :, :, 0, -1]
        v50 = p50[p50 > -1]
        overall["mAP_50"] = float(v50.mean()) if v50.size else 0.0
    if 0.75 in iou_list:
        idx75 = iou_list.index(0.75)
        p75 = precision[idx75, :, :, 0, -1]
        v75 = p75[p75 > -1]
        overall["mAP_75"] = float(v75.mean()) if v75.size else 0.0

    per_class: dict[str, dict[str, float]] = {}
    if include_per_class:
        cat_ids = coco_eval.params.catIds
        for k, cat_id in enumerate(cat_ids):
            p = precision[:, :, k, 0, -1]
            valid = p[p > -1]
            if valid.size == 0:
                continue  # class with no GT — skip
            ap = float(valid.mean())
            row: dict[str, float] = {"AP": ap}
            if 0.5 in iou_list:
                idx = iou_list.index(0.5)
                p50_k = precision[idx, :, k, 0, -1]
                v50_k = p50_k[p50_k > -1]
                if v50_k.size:
                    row["AP_50"] = float(v50_k.mean())
            cat_name = ground_truth.cats[cat_id]["name"]
            per_class[cat_name] = row

    return MetricsReport(
        overall=overall,
        per_class=per_class,
        n_images=n_images,
        n_predictions=len(predictions),
    )
