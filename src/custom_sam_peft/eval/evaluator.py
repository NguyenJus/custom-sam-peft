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

from custom_sam_peft.cli._progress import progress as P
from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.data.base import Dataset, Example, TextPrompts
from custom_sam_peft.eval.metrics import MetricsReport, compute_coco_map
from custom_sam_peft.eval.postprocess import queries_to_coco_results
from custom_sam_peft.paths import predictions_path
from custom_sam_peft.runtime import Runtime, to_device

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

    # ------------------------------------------------------------------
    # Private helpers (decomposed from evaluate)
    # ------------------------------------------------------------------

    def _iter_predictions(
        self, model: Any, examples: Sequence[Example], dataset: Dataset
    ) -> list[dict[str, object]]:
        """Run the forward loop and return raw COCO-format prediction entries.

        Puts the model into eval mode for the duration of the loop and restores
        its training state on exit. Moves dataset images to the model's device
        before each forward.
        """
        cfg = self.cfg

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()

        # Resolve the model's device once and move dataset images onto it
        # before each forward via runtime.to_device (§3 seam discipline).
        # The dataset yields CPU tensors; passing them straight to a
        # CUDA-resident model raises
        # `Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type)
        #  should be the same` inside the first Conv2d. Falls back to CPU
        # for parameterless / non-nn.Module test stubs.
        try:
            param_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            param_device = torch.device("cpu")
        eval_runtime = Runtime(device=param_device, dtype=torch.float32)

        log_every_n = max(1, len(examples) // 50)
        predictions: list[dict[str, object]] = []
        P.reset_inner(total=len(examples))
        try:
            with torch.no_grad():
                for img_idx, ex in enumerate(examples):
                    original_hw = (int(ex.image.shape[-2]), int(ex.image.shape[-1]))
                    int_id = _int_image_id(ex.image_id)
                    # Note: mode="lite" bounds only the image dimension; the
                    # class dimension is intentionally unbounded (per spec's
                    # documented compute cost O(images * classes)).
                    for cat_idx, class_name in enumerate(dataset.class_names):
                        cat_id = cat_idx + 1
                        outputs = model(
                            to_device(ex.image.unsqueeze(0), eval_runtime),
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
                    P.advance_inner()
                    if (img_idx + 1) % log_every_n == 0:
                        P.update_postfix(it_s=float(img_idx + 1))
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        return predictions

    def _aggregate_metrics(
        self,
        predictions: list[dict[str, object]],
        gt: COCO,
        dataset: Dataset,
    ) -> MetricsReport:
        """Compute a MetricsReport from raw predictions and ground-truth COCO data."""
        cfg = self.cfg

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

        return report

    def _maybe_save_predictions(
        self,
        preds: list[dict[str, object]],
        run_dir: Path | None,
        *,
        split: str = "val",
    ) -> None:
        """Write predictions to disk when configured and ``run_dir`` is given.

        Uses ``paths.predictions_path`` for the canonical output path.
        Skipped in lite mode regardless of ``cfg.save_predictions``.
        """
        if run_dir is None:
            return
        if not (self.cfg.save_predictions and self.cfg.mode == "full"):
            return
        out_path = predictions_path(run_dir, split=split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(preds))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: Literal[False] = False,
    ) -> MetricsReport:
        pass

    @overload
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: Literal[True],
    ) -> tuple[MetricsReport, list[float]]:
        pass

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
        compatibility (e.g. `custom_sam_peft eval` CLI, mid-training eval).
        """
        # Reset predictions at the start so evaluate_and_save never writes
        # stale data from a prior call that may have failed mid-run.
        self._last_predictions = []

        cfg = self.cfg
        n_total = len(dataset)
        n = n_total if cfg.mode == "full" else min(cfg.lite_max_images, n_total)
        examples = [dataset[i] for i in range(n)]
        gt, _ = _build_coco_gt_from_examples(examples, dataset)

        predictions = self._iter_predictions(model, examples, dataset)
        report = self._aggregate_metrics(predictions, gt, dataset)
        self._maybe_save_predictions(predictions, run_dir=None)
        self._last_predictions = predictions

        if not return_per_example_iou:
            return report
        return report, self._compute_per_example_iou(examples, predictions, gt)

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
        """Call ``evaluate``, write ``metrics.json``, and optionally save predictions.

        Predictions are written via ``_maybe_save_predictions`` using the canonical
        path from ``paths.predictions_path`` — only when ``cfg.save_predictions=True``
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

        self._maybe_save_predictions(self._last_predictions, run_dir=output_dir)

        return report
