"""End-to-end train pipeline. CLI is a thin wrapper over `run_training`."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.subset import SubsetDataset, resolve_subset_indices
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import RunResult, Trainer

_LOG = logging.getLogger(__name__)


def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Returns the created path."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataset(cfg: TrainConfig, pipeline: str) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    inner = cast(
        Dataset, builder(cfg.data.model_dump(), model_name=cfg.model.name, pipeline=pipeline)
    )

    lim_cfg = cfg.data.limit
    limit_val = lim_cfg.train if pipeline == "train" else lim_cfg.val
    if limit_val is None:
        return inner

    labels = None
    if lim_cfg.strategy == "stratified":
        labels = getattr(inner, "image_class_labels", None)

    indices = resolve_subset_indices(
        len(inner),
        limit_val,
        seed=lim_cfg.seed,
        strategy=lim_cfg.strategy,
        image_class_labels=labels,
    )
    _LOG.info(
        "data.limit applied: %s=%d/%d (strategy=%s, seed=%d)",
        pipeline,
        len(indices),
        len(inner),
        lim_cfg.strategy,
        lim_cfg.seed,
    )
    return SubsetDataset(inner, indices)


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    """Build datasets, load model + PEFT, build tracker, run Trainer.fit."""
    run_dir = make_run_dir(cfg)
    train_ds = _build_dataset(cfg, "train")
    val_ds = _build_dataset(cfg, "eval")

    # Write subset.json when at least one side has a limit applied
    lim_cfg = cfg.data.limit
    if lim_cfg.train is not None or lim_cfg.val is not None:
        _write_subset_manifest(run_dir, train_ds, val_ds, cfg)

    wrapper: Any = load_sam31(cfg.model)
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)


def _write_subset_manifest(
    run_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
) -> None:
    """Write <run_dir>/subset.json recording resolved indices per side."""
    lim_cfg = cfg.data.limit
    manifest: dict[str, Any] = {
        "limit": {
            "train": lim_cfg.train,
            "val": lim_cfg.val,
            "seed": lim_cfg.seed,
            "strategy": lim_cfg.strategy,
        }
    }
    if lim_cfg.train is not None and isinstance(train_ds, SubsetDataset):
        inner_len = len(train_ds._inner)
        manifest["train"] = {
            "n_total": inner_len,
            "n_kept": len(train_ds),
            "indices": train_ds._indices,
        }
    if lim_cfg.val is not None and isinstance(val_ds, SubsetDataset):
        inner_len = len(val_ds._inner)
        manifest["val"] = {
            "n_total": inner_len,
            "n_kept": len(val_ds),
            "indices": val_ds._indices,
        }
    (run_dir / "subset.json").write_text(json.dumps(manifest, indent=2))
