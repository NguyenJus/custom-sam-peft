"""End-to-end train pipeline. CLI is a thin wrapper over `run_training`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.val_source import (
    _log_val_source,
    resolve_val_source,
    save_val_source,
)
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import RunResult, Trainer


def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Returns the created path."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataset_from_dict(
    data_cfg_dict: dict[str, Any], cfg: TrainConfig, pipeline: str
) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline=pipeline))


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    """Build datasets, load model + PEFT, build tracker, run Trainer.fit.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §6.4.
    """
    run_dir = make_run_dir(cfg)

    # On resume, look for val_source.json in the run dir that owns the
    # checkpoint (checkpoints live at <run_dir>/checkpoints/step_N/).
    resume_run_dir = resume_from.parent.parent if resume_from is not None else None
    vs = resolve_val_source(cfg, run_dir=resume_run_dir)
    save_val_source(vs, run_dir)
    _log_val_source(vs)

    data_cfg_dict = cfg.data.model_dump()
    if vs.mode == "auto_split":
        assert vs.train_ids is not None and vs.val_ids is not None
        data_cfg_dict["_resolved_image_ids"] = {
            "train": list(vs.train_ids),
            "eval": list(vs.val_ids),
        }

    train_ds = _build_dataset_from_dict(data_cfg_dict, cfg, "train")
    val_ds: Dataset | None = (
        None if vs.mode == "none" else _build_dataset_from_dict(data_cfg_dict, cfg, "eval")
    )

    wrapper: Any = load_sam31(cfg.model)
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)
