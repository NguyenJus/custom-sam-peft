"""End-to-end train pipeline. CLI is a thin wrapper over `run_training`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from esam3._registry import lookup
from esam3.config.schema import TrackingConfig, TrainConfig
from esam3.data.base import Dataset
from esam3.models.sam3 import load_sam31
from esam3.tracking.base import Tracker
from esam3.train.trainer import RunResult, Trainer


def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Returns the created path."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataset(cfg: TrainConfig, pipeline: str) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    result = builder(cfg.data.model_dump(), model_name=cfg.model.name, pipeline=pipeline)
    return cast(Dataset, result)


def build_tracker(cfg: TrackingConfig, run_dir: Path) -> Tracker:
    """Dispatch the registered tracker factory with backend-specific kwargs."""
    factory = lookup("tracker", cfg.backend)
    if cfg.backend == "tensorboard":
        return cast(Tracker, factory({"log_dir": str(run_dir / "tb")}))
    if cfg.backend == "wandb":
        return cast(
            Tracker,
            factory(
                {
                    "project": cfg.wandb.project,
                    "entity": cfg.wandb.entity,
                }
            ),
        )
    return cast(Tracker, factory({}))


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    """Build datasets, load model + PEFT, build tracker, run Trainer.fit."""
    run_dir = make_run_dir(cfg)
    train_ds = _build_dataset(cfg, "train")
    val_ds = _build_dataset(cfg, "eval")
    wrapper: Any = load_sam31(cfg.model)
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)
    tracker = build_tracker(cfg.tracking, run_dir)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)
