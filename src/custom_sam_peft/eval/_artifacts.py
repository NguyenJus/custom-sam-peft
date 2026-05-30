"""EvalArtifacts — the single value object the evaluator consumes from the trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.eval.metrics import MetricsReport
    from custom_sam_peft.train.types import OomEvent


@dataclass(frozen=True)
class TimeLimitStop:
    """Set when Trainer.fit stopped early on a wall-clock budget. None otherwise.

    Carried on EvalArtifacts as an optional field the evaluator never reads;
    the CLI uses it to print the resume message and exit 0. Spec §4.7.
    """

    stop_step: int
    stop_epoch: int  # zero-based epoch index at the stop
    total_epochs: int  # cfg.train.epochs
    checkpoint_dir: Path  # run_dir/checkpoints/step_<N>/
    duration_label: str  # format_seconds(budget_seconds), e.g. "2h30m"
    best_dir: Path | None  # run_dir/best/ if it exists, else None
    best_map: float | None  # best.json "value" if best/ exists, else None


@dataclass(frozen=True)
class EvalArtifacts:
    """Hand-off object returned by Trainer.fit, consumed by Evaluator.

    These are the ONLY fields the evaluator may read from training
    output. The evaluator does not reach into trainer internals beyond
    this object. Tests in tests/integration/test_trainer_evaluator_seam.py
    enforce this.
    """

    checkpoint_path: Path
    peft_method: str
    run_dir: Path
    # End-of-run eval report; None when val_ds is None (no-val mode) or eval raised.
    final_metrics: MetricsReport | None = field(default=None)
    # OOM recovery events accumulated by the trainer's per-step retry loop.
    oom_events: tuple[OomEvent, ...] = field(default=())
    # Set when training stopped early on a wall-clock budget; None on the
    # normal path. The evaluator never reads it (seam-safe optional field).
    time_limit_stop: TimeLimitStop | None = field(default=None)
