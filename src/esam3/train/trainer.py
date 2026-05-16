"""Trainer — public training entrypoint. Loop body lives in train/loop.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esam3.config.schema import TrainConfig
from esam3.data.base import Dataset
from esam3.eval.metrics import MetricsReport
from esam3.tracking.base import Tracker


@dataclass(frozen=True)
class RunResult:
    """Returned from Trainer.fit() — what a run produced on disk."""

    run_dir: Path
    adapter_path: Path
    merged_path: Path | None
    final_metrics: MetricsReport | None


class Trainer:
    """Drive a finetuning run end-to-end.

    Implementation deferred to spec/training-loop.
    """

    def __init__(
        self,
        model: Any,
        train_ds: Dataset,
        val_ds: Dataset,
        tracker: Tracker,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tracker = tracker
        self.cfg = cfg

    def fit(self) -> RunResult:
        raise NotImplementedError("filled in by spec: spec/training-loop")
