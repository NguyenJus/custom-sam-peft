"""EvalArtifacts — the single value object the evaluator consumes from the trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.eval.metrics import MetricsReport
    from custom_sam_peft.train.types import OomEvent


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
