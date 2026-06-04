"""Best-as-final close-out (spec §7).

Restore run_dir/best/ into the model, run ONE full eval
(return_per_example_iou=True), and write run_dir/adapter (+ optional
run_dir/merged + metrics.json) — all on the BEST weights. Falls back to the
current in-memory (last-step) weights when no best/ exists, or when restoring
best/ raises (swallow-and-continue, mirroring _maybe_save_best).

Called on early stop, normal completion, and the finalize entry. Never called
for a _TimeLimitReached pause (spec §9.1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
from custom_sam_peft.train.checkpoint import load_adapter, save_adapter, save_merged

if TYPE_CHECKING:
    from custom_sam_peft.config.schema import TrainConfig
    from custom_sam_peft.data.base import Dataset
    from custom_sam_peft.models.sam3 import Sam3Wrapper
    from custom_sam_peft.train.ladder import LadderEvents
    from custom_sam_peft.train.loop import OomState

_LOG = logging.getLogger(__name__)


def close_out(
    run_dir: Path,
    model: Sam3Wrapper,
    cfg: TrainConfig,
    *,
    evaluator_val_ds: Dataset | None,
    oom_state: OomState | None,
    final_step: int,
    final_epoch: int,
    ladder_events: LadderEvents | None = None,
) -> EvalArtifacts:
    # 1. Restore best/ (or keep last-step weights on absence/failure).
    best_adapter = run_dir / "best" / "adapter"
    final_weights = "last_step"
    if best_adapter.is_dir():
        try:
            load_adapter(model, best_adapter)
            final_weights = "best"
        except Exception:
            _LOG.warning(
                "close_out: failed to restore best/ — finalizing on last-step weights.",
                exc_info=True,
            )

    # 2. Write the (best, or last-step) adapter.
    save_adapter(model, run_dir / "adapter")

    # 3. Optional merged (soft-fail — mirrors the old orchestrator behaviour).
    merged_export_error: str | None = None
    if cfg.export.merge:
        try:
            save_merged(model, run_dir / "merged")
        except Exception as exc:
            _LOG.warning("close_out: export-merge failed: %s", exc, exc_info=True)
            merged_export_error = str(exc)

    # 4. Single full eval on the restored weights (return_per_example_iou=True).
    report: Any = None
    per_example_iou: list[float] | None = None
    gt_counts: list[int] | None = None
    if evaluator_val_ds is not None:
        full_eval_cfg = cfg.eval
        if full_eval_cfg.batch_size == "auto":
            from custom_sam_peft.presets import decide_eval_batch_size

            bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
            if oom_state is not None and bs > oom_state.micro_batch_size:
                bs = oom_state.micro_batch_size
            full_eval_cfg = full_eval_cfg.model_copy(update={"batch_size": bs})
        report, per_example_iou, gt_counts = Evaluator(full_eval_cfg).evaluate(
            model, evaluator_val_ds, return_per_example_iou=True
        )

    # 5. metrics.json (best mAP), final_weights, ladder events.
    metrics: dict[str, Any] = {
        "final_weights": final_weights,
        "global_step": final_step,
        "epoch": final_epoch,
    }
    if report is not None:
        metrics.update(
            {
                "overall": report.overall,
                "per_class": report.per_class,
                "n_images": report.n_images,
                "n_predictions": report.n_predictions,
            }
        )
    else:
        metrics["note"] = "no validation set provided"
    if ladder_events is not None:
        metrics["ladder_events"] = {
            "stop_reason": ladder_events.stop_reason,
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 6. Optional visualization pass (soft-fail — metrics.json is already written).
    if evaluator_val_ds is not None and cfg.eval.visualize and per_example_iou is not None:
        try:
            from custom_sam_peft.eval.visualize import write_eval_visualizations

            write_eval_visualizations(
                model,
                evaluator_val_ds,
                run_dir,
                per_example_iou=per_example_iou,
                count=cfg.eval.visualize_count,
                mask_threshold=cfg.eval.mask_threshold,
                model_name=cfg.model.name,
                normalize=cfg.data.normalize,
                channel_semantics=cfg.data.channel_semantics,
                gt_counts=gt_counts,
            )
        except Exception:
            _LOG.warning("close_out visualize pass failed; metrics are persisted.", exc_info=True)

    return EvalArtifacts(
        checkpoint_path=run_dir / "adapter",
        peft_method=cfg.peft.method,
        run_dir=run_dir,
        final_metrics=report,
        oom_events=tuple(oom_state.pending_oom_events) if oom_state is not None else (),
        per_example_iou=per_example_iou,
        final_weights=final_weights,
        ladder_events=ladder_events,
        merged_export_error=merged_export_error,
    )
