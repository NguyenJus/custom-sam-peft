"""SemanticEvaluator — streaming mIoU over a semantic-segmentation dataset (§8).

Mirrors the instance Evaluator's public surface so run_eval and mid-training
eval can dispatch on task with no caller rewrite.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
import torch
import torch.nn.functional as F

from custom_sam_peft import profiling
from custom_sam_peft.cli._progress import progress as P
from custom_sam_peft.config.schema import EvalConfig, SemanticLossConfig
from custom_sam_peft.data.base import Dataset, Example, TextPrompts
from custom_sam_peft.eval.metrics import MetricsReport, compute_semantic_metrics
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
from custom_sam_peft.models.semantic import (
    build_semantic_logits,
    marginalize_group,
    semantic_argmax,
)
from custom_sam_peft.oom import OomDecision, OomLadder, is_cuda_oom
from custom_sam_peft.runtime import Runtime, to_device

_LOG = logging.getLogger(__name__)


class SemanticEvaluator:
    """Compute mIoU + pixel accuracy for a model on a semantic dataset.

    Constructor takes only EvalConfig, mirroring the instance Evaluator's
    signature so callers can swap evaluators without change.

    Eval uses schema-default marginalize knobs; threading a run's overridden
    train.semantic_loss through run_eval is a documented follow-up
    (v1 validates defaults only).
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        # Use schema defaults for the marginalize knobs (query_reduce, source,
        # background_logit). See class docstring for the v1 limitation note.
        self._sem_cfg = SemanticLossConfig()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_batch_size(self) -> int:
        """Return an int batch size; coerce 'auto' -> 1 for the semantic path.

        The instance Evaluator relies on run_eval resolving 'auto' externally;
        for the semantic path we default to 1 so unit tests with EvalConfig()
        work without a run_eval wrapper.
        """
        bs = self.cfg.batch_size
        if isinstance(bs, int):
            return bs
        return 1  # 'auto' -> 1 for semantic eval

    def _iter_semantic_predictions(
        self,
        model: Any,
        examples: list[Example],
        dataset: Dataset,
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int64]], list[float]]:
        """Run the forward loop; return (confusion, per_example_ious).

        Mirrors the instance Evaluator._iter_predictions loop structure:
        image-chunk x K-group nested loops with OomLadder recovery.

        Returns:
          confusion: (K+1, K+1) int64 accumulated across all examples.
          per_example_ious: mean_iou per image (list aligned with examples).
        """
        sem_cfg = self._sem_cfg
        K = len(dataset.class_names)

        confusion = np.zeros((K + 1, K + 1), dtype=np.int64)
        per_example_ious: list[float] = []

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()

        try:
            param_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            param_device = torch.device("cpu")
        eval_runtime = Runtime(device=param_device, dtype=torch.float32)

        bs = self._resolve_batch_size()
        ladder = OomLadder(
            micro_batch_size=bs,
            effective_K=min(MULTIPLEX_CAP, K) if K else 1,
        )

        _meta_noted = False  # emit profiling.note once for the whole run
        try:
            with torch.no_grad(), P.push_subtask("sem_eval", total=len(examples)) as sub:
                i = 0
                img_idx_global = 0
                while i < len(examples):
                    with profiling.bucket("semantic_eval.total"):
                        chunk_bs = ladder.micro_batch_size
                        image_chunk = list(examples[i : i + chunk_bs])
                        images_t = to_device(
                            torch.stack([ex.image for ex in image_chunk]), eval_runtime
                        )
                        if not _meta_noted:
                            profiling.note(
                                n_images=len(examples),
                                K=K,
                                sem_forward_dtype=str(images_t.dtype),
                            )
                            _meta_noted = True
                        # slices[image_idx] accumulates (k_g, H_m, W_m) per K-group
                        chunk_slices: list[list[torch.Tensor]] = [[] for _ in image_chunk]
                        restart_chunk = False
                        j = 0  # class index into dataset.class_names
                        while j < K:
                            K_g = min(ladder.effective_K, K - j)
                            group = dataset.class_names[j : j + K_g]
                            prompts_g = [TextPrompts(classes=list(group)) for _ in image_chunk]
                            try:
                                with profiling.bucket("semantic_eval.forward"):
                                    outputs = cast(
                                        "dict[str, torch.Tensor]",
                                        model(images_t, prompts_g, support=None),
                                    )
                                profiling.incr("semantic_eval.forwards")
                            except RuntimeError as oom_exc:
                                if not is_cuda_oom(oom_exc):
                                    raise
                                decision = ladder.on_oom()
                                if decision is OomDecision.RETRY_B:
                                    # Discard buffer and restart at smaller B.
                                    chunk_slices = [[] for _ in image_chunk]
                                    restart_chunk = True
                                    break
                                if decision is OomDecision.RETRY_K:
                                    # Resume from same class index at smaller K_g.
                                    continue
                                if decision is OomDecision.FLOOR_RETRY:
                                    continue
                                raise RuntimeError(
                                    "semantic eval OOM at batch_size=1 and "
                                    "classes_per_forward=1; use a larger GPU or "
                                    "smaller image_size."
                                ) from None

                            b = len(image_chunk)
                            # marginalize_group returns (b, K_g, H_m, W_m)
                            logit_slice = marginalize_group(
                                outputs,
                                b,
                                K_g,
                                query_reduce=sem_cfg.query_reduce,
                                source=sem_cfg.source,
                            )
                            # Distribute per-image slices into chunk_slices.
                            for ii in range(b):
                                chunk_slices[ii].append(
                                    logit_slice[ii : ii + 1]
                                )  # (1, K_g, H_m, W_m)

                            j += K_g

                        if restart_chunk:
                            continue  # re-enter outer while at smaller B; i unchanged

                        # All K-groups done for this chunk. Build full semantic logits
                        # and accumulate confusion per image.
                        for ii, ex in enumerate(image_chunk):
                            if ex.semantic is None:
                                _LOG.warning(
                                    "semantic eval: example %s has no SemanticTarget; skipping",
                                    ex.image_id,
                                )
                                per_example_ious.append(float("nan"))
                                continue

                            # chunk_slices[ii] is already a list of (1, k_g, H_m, W_m)
                            # tensors — exactly the shape build_semantic_logits expects.
                            sem_logits_b = build_semantic_logits(
                                chunk_slices[ii],
                                background_logit=sem_cfg.background_logit,
                            )  # (1, K+1, H_m, W_m)

                            # Upsample to GT label resolution BEFORE argmax (§6.3). Read
                            # the target HW from the label map itself so the confusion
                            # comparison is shape-aligned regardless of the loader.
                            H, W = (int(d) for d in ex.semantic.labels.shape[-2:])
                            with profiling.bucket("semantic_eval.upsample"):
                                sem_logits_up = F.interpolate(
                                    sem_logits_b,
                                    size=(H, W),
                                    mode="bilinear",
                                    align_corners=False,
                                )  # (1, K+1, H, W)
                                pred_labels = semantic_argmax(sem_logits_up)  # (1, H, W)

                            with profiling.bucket("semantic_eval.transfer"):
                                pred = pred_labels[0].cpu().numpy().astype(np.int64)  # (H, W)
                                gt = ex.semantic.labels.cpu().numpy().astype(np.int64)  # (H, W)
                            ignore = ex.semantic.ignore_index

                            # One per-image confusion block (ignore_index excluded),
                            # reused for the global accumulate and the per-example IoU.
                            with profiling.bucket("semantic_eval.confusion"):
                                valid = gt != ignore
                                idx = (K + 1) * gt[valid] + pred[valid]
                                ex_conf = np.bincount(idx, minlength=(K + 1) ** 2).reshape(
                                    K + 1, K + 1
                                )
                                confusion += ex_conf

                                ex_metrics = compute_semantic_metrics(
                                    ex_conf, list(dataset.class_names)
                                )
                                per_example_ious.append(ex_metrics.mean_iou)

                    i += len(image_chunk)
                    img_idx_global += len(image_chunk)
                    for _ in range(len(image_chunk)):
                        sub.advance()
                    sub.update_postfix(it_s=float(img_idx_global))
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        return confusion, per_example_ious

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

        Pure compute — no disk I/O. Restores the model's training/eval state.

        When ``return_per_example_iou=True``, also returns a list of per-image
        mean IoU values aligned with dataset indices.
        """
        cfg = self.cfg
        n_total = len(dataset)
        n = n_total if cfg.mode == "full" else min(cfg.lite_max_images, n_total)
        examples = [dataset[i] for i in range(n)]

        confusion, per_example_ious = self._iter_semantic_predictions(model, examples, dataset)

        metrics = compute_semantic_metrics(confusion, list(dataset.class_names))

        # n_predictions repurposed as pixels-scored for semantic (not mask count).
        total_pixels = int(confusion.sum())

        report = MetricsReport(
            overall={"mIoU": metrics.mean_iou, "pixel_acc": metrics.pixel_accuracy},
            per_class={name: {"IoU": iou} for name, iou in metrics.per_class_iou.items()},
            n_images=n,
            n_predictions=total_pixels,
        )

        if not return_per_example_iou:
            return report
        return report, per_example_ious

    def evaluate_and_save(self, model: Any, dataset: Dataset, output_dir: Path) -> MetricsReport:
        """Call ``evaluate``, write ``metrics.json``, return the report.

        ``metrics.json`` carries a ``"task": "semantic"`` field to distinguish
        it from the instance path's output.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.evaluate(model, dataset)

        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "task": "semantic",
                    "overall": report.overall,
                    "per_class": report.per_class,
                    "n_images": report.n_images,
                    "n_predictions": report.n_predictions,
                },
                indent=2,
            )
        )

        return report
