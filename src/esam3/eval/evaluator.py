"""Evaluator — runs a model over a dataset and returns a MetricsReport.

See docs/superpowers/specs/2026-05-17-eval-design.md for the contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pycocotools.mask as mask_utils
import torch
from pycocotools.coco import COCO

from esam3.config.schema import EvalConfig
from esam3.data.base import Dataset, Example, TextPrompts
from esam3.eval.metrics import MetricsReport, compute_coco_map
from esam3.eval.postprocess import queries_to_coco_results

_LOG = logging.getLogger(__name__)


def _int_image_id(image_id: str) -> int:
    """Stable int hash of a string image_id (blake2s, 8-byte digest)."""
    return int(hashlib.blake2s(image_id.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _mask_to_rle(mask: torch.Tensor) -> Any:
    """Convert a (H, W) bool tensor to a pycocotools RLE dict."""
    arr = mask.cpu().numpy().astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(arr))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _build_coco_gt_from_examples(
    examples: Sequence[Example], dataset: Dataset
) -> tuple[COCO, dict[str, int]]:
    """Build an in-memory COCO ground-truth from a pre-fetched list of Examples.

    Returns the COCO object and a ``str_image_id -> int_image_id`` map.
    Raises RuntimeError on int-id collision.
    """
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    seen_ids: dict[int, str] = {}
    str_to_int: dict[str, int] = {}
    ann_id = 1

    for ex in examples:
        int_id = _int_image_id(ex.image_id)
        prior = seen_ids.get(int_id)
        if prior is not None and prior != ex.image_id:
            raise RuntimeError(
                f"image_id hash collision: {ex.image_id!r} and {prior!r} both hash to {int_id}"
            )
        seen_ids[int_id] = ex.image_id
        str_to_int[ex.image_id] = int_id
        h, w = ex.image.shape[-2:]
        images.append({"id": int_id, "height": int(h), "width": int(w)})
        for inst in ex.instances:
            rle = _mask_to_rle(inst.mask)
            area = int(mask_utils.area(rle))
            x1, y1, x2, y2 = (float(v) for v in inst.box.tolist())
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int_id,
                    "category_id": int(inst.class_id) + 1,  # 1-indexed for COCO
                    "iscrowd": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": area,
                    "segmentation": rle,
                }
            )
            ann_id += 1

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(dataset.class_names)]
    gt = COCO()
    gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    gt.createIndex()
    return gt, str_to_int


class Evaluator:
    """Compute COCO metrics for a model on a dataset."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self._last_predictions: list[dict[str, object]] = []

    @overload
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: Literal[False] = False,
    ) -> MetricsReport: ...

    @overload
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: Literal[True],
    ) -> tuple[MetricsReport, list[float]]: ...

    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: bool = False,
    ) -> MetricsReport | tuple[MetricsReport, list[float]]:
        """Run the model over the dataset and return a MetricsReport.

        Pure compute — no disk I/O. Restores the model's training/eval state
        after the forward loop.

        When ``return_per_example_iou=True``, also returns a list of per-example
        MEAN IoU values across ``cfg.iou_thresholds`` aligned with dataset indices.
        The default ``False`` preserves the previous return type for backward
        compatibility (e.g. `esam3 eval` CLI, mid-training eval).
        """
        # Reset predictions at the start so evaluate_and_save never writes
        # stale data from a prior call that may have failed mid-run.
        self._last_predictions = []

        cfg = self.cfg
        n_total = len(dataset)
        n = n_total if cfg.mode == "full" else min(cfg.lite_max_images, n_total)
        indices = range(n)

        # Fetch each example exactly once — used for both GT construction and
        # model inference, avoiding a double dataset traversal.
        examples = [dataset[i] for i in indices]

        gt, _ = _build_coco_gt_from_examples(examples, dataset)

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()

        predictions: list[dict[str, object]] = []
        try:
            with torch.no_grad():
                for ex in examples:
                    original_hw = (int(ex.image.shape[-2]), int(ex.image.shape[-1]))
                    int_id = _int_image_id(ex.image_id)
                    # Note: mode="lite" bounds only the image dimension; the
                    # class dimension is intentionally unbounded (per spec's
                    # documented compute cost O(images * classes)).
                    for cat_idx, class_name in enumerate(dataset.class_names):
                        cat_id = cat_idx + 1
                        outputs = model(
                            ex.image.unsqueeze(0),
                            [TextPrompts(classes=[class_name])],
                            box_hints=None,
                        )
                        entries = queries_to_coco_results(
                            outputs,
                            int_id,
                            cat_id,
                            original_hw,
                            cfg.mask_threshold,
                        )
                        predictions.extend(entries)
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        report = compute_coco_map(
            predictions=predictions,
            ground_truth=gt,
            iou_thresholds=cfg.iou_thresholds,
            include_per_class=(cfg.mode == "full"),
        )

        if cfg.mode == "full":
            skipped = sum(1 for name in dataset.class_names if name not in report.per_class)
            if skipped:
                _LOG.info(
                    "eval: skipped %d/%d classes with no GT instances",
                    skipped,
                    len(dataset.class_names),
                )

        self._last_predictions = predictions

        if not return_per_example_iou:
            return report

        per_example_iou = self._compute_per_example_iou(examples, predictions, gt)
        return report, per_example_iou

    def _compute_per_example_iou(
        self,
        examples: Sequence[Example],
        predictions: list[dict[str, object]],
        gt: COCO,
    ) -> list[float]:
        """Compute mean IoU per example across self.cfg.iou_thresholds.

        The 'IoU' here is segmentation IoU between the best-matched predicted
        mask and any GT mask in the same image (greedy match, max IoU). For an
        example with no GT instances, IoU is 0.0 if it has predictions, else 1.0
        (vacuous match — consistent with COCO's empty-image handling). Examples
        skipped during model inference are marked NaN; pick_samples treats NaN
        as -inf for ranking and they are eligible only as 'worst' picks.
        """
        out: list[float] = []
        # Group predictions by image_id for cheap lookup.
        preds_by_image: dict[int, list[dict[str, object]]] = {}
        for entry in predictions:
            preds_by_image.setdefault(int(entry["image_id"]), []).append(entry)  # type: ignore[call-overload]

        for ex in examples:
            int_id = _int_image_id(ex.image_id)
            gt_anns = gt.imgToAnns.get(int_id, [])
            ex_preds = preds_by_image.get(int_id, [])

            if not gt_anns and not ex_preds:
                out.append(1.0)  # vacuous match
                continue
            if not gt_anns or not ex_preds:
                out.append(0.0)
                continue

            # Build (n_pred, n_gt) IoU matrix for this example.
            pred_rles = [p["segmentation"] for p in ex_preds]
            gt_rles = [a["segmentation"] for a in gt_anns]
            iscrowd = [0] * len(gt_rles)
            iou_mat = mask_utils.iou(pred_rles, gt_rles, iscrowd)
            # max-IoU greedy: for each GT, the best predicted IoU; mean over thresholds.
            # Spec §6.1: "the MEAN IoU across the eval's IoU thresholds [0.5, …, 0.95]".
            # We compute the per-GT best-pred IoU once, then average across thresholds:
            # at threshold t, the per-GT-IoU is the best-pred IoU if >= t else 0, so the
            # threshold-mean reduces to mean_t(best_iou >= t) which is the cdf at the
            # discrete thresholds. Use that as the example score.
            if iou_mat.size == 0:
                out.append(0.0)
                continue
            best_per_gt = np.asarray(iou_mat).max(axis=0)  # (n_gt,)
            thresholds = np.asarray(self.cfg.iou_thresholds)
            # Mean over (gt, thresholds) of (best_per_gt[g] >= thresholds[t]).
            hit = best_per_gt[:, None] >= thresholds[None, :]
            out.append(float(hit.mean()))

        return out

    def evaluate_and_save(self, model: Any, dataset: Dataset, output_dir: Path) -> MetricsReport:
        """Call ``evaluate``, write ``metrics.json``, and optionally ``predictions.json``.

        ``predictions.json`` is written only when ``cfg.save_predictions=True``
        AND ``cfg.mode == "full"``. In lite mode, predictions are never persisted
        regardless of ``cfg.save_predictions``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.evaluate(model, dataset)

        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "overall": report.overall,
                    "per_class": report.per_class,
                    "n_images": report.n_images,
                    "n_predictions": report.n_predictions,
                },
                indent=2,
            )
        )

        if self.cfg.save_predictions and self.cfg.mode == "full":
            (output_dir / "predictions.json").write_text(json.dumps(self._last_predictions))

        return report
