"""Resolve the validation source for a run: explicit, auto_split, or none.

The resolver is the single seam between schema and the splitter. It also
owns persistence (`save_val_source` / `load_val_source`) of the resolved
record to `<run_dir>/val_source.json`. Trainer hparams logging and tracker
hparams injection both read the saved record, so the resolver writes the
authoritative copy once per run (before Trainer.fit begins).

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §5.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from custom_sam_peft.config.schema import DataConfig, TrainConfig
from custom_sam_peft.data.splitter import SplittableItem, stratified_split

_LOG = logging.getLogger(__name__)

ValMode = Literal["explicit", "auto_split", "none"]


@dataclass(frozen=True)
class ValSource:
    mode: ValMode
    train_ids: tuple[str, ...] | None
    val_ids: tuple[str, ...] | None
    realized_fraction: float | None
    per_class_counts: dict[int, tuple[int, int]] | None
    missing_in_val: tuple[int, ...] | None
    fraction_requested: float | None
    seed_used: int | None


def resolve_val_source(cfg: TrainConfig, *, run_dir: Path | None = None) -> ValSource:
    """Resolve which validation source to use for this run.

    Dispatch (spec §5.2):
      1. run_dir/val_source.json exists → load_val_source(run_dir) (resume).
      2. cfg.data.val_split is not None → enumerate + stratify.
      3. cfg.data.val is not None → mode='explicit'.
      4. else → mode='none'.
    """
    if run_dir is not None:
        saved = load_val_source(run_dir)
        if saved is not None:
            _LOG.info("val_source: resumed from %s (mode=%s)", run_dir, saved.mode)
            return saved

    if cfg.data.val_split is not None:
        seed_used = cfg.data.val_split.seed if cfg.data.val_split.seed is not None else cfg.run.seed
        items = _enumerate_items(cfg.data)
        result = stratified_split(items, cfg.data.val_split.fraction, seed_used)
        return ValSource(
            mode="auto_split",
            train_ids=result.train_ids,
            val_ids=result.val_ids,
            realized_fraction=result.realized_fraction,
            per_class_counts=result.per_class_counts,
            missing_in_val=result.missing_in_val,
            fraction_requested=cfg.data.val_split.fraction,
            seed_used=seed_used,
        )

    if cfg.data.val is not None:
        return ValSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            fraction_requested=None,
            seed_used=None,
        )

    return ValSource(
        mode="none",
        train_ids=None,
        val_ids=None,
        realized_fraction=None,
        per_class_counts=None,
        missing_in_val=None,
        fraction_requested=None,
        seed_used=None,
    )


def save_val_source(vs: ValSource, run_dir: Path) -> None:
    """Write `<run_dir>/val_source.json`. Atomic via tmp + os.replace."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "mode": vs.mode,
        "fraction_requested": vs.fraction_requested,
        "seed_used": vs.seed_used,
        "realized_fraction": vs.realized_fraction,
        "n_train": (len(vs.train_ids) if vs.train_ids is not None else None),
        "n_val": (len(vs.val_ids) if vs.val_ids is not None else None),
        "per_class_counts": (
            {str(k): list(v) for k, v in vs.per_class_counts.items()}
            if vs.per_class_counts is not None
            else None
        ),
        "missing_in_val": (list(vs.missing_in_val) if vs.missing_in_val is not None else None),
        "train_ids": (list(vs.train_ids) if vs.train_ids is not None else None),
        "val_ids": (list(vs.val_ids) if vs.val_ids is not None else None),
    }
    tmp = run_dir / "val_source.json.tmp"
    final = run_dir / "val_source.json"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, final)


def load_val_source(run_dir: Path) -> ValSource | None:
    """Read `<run_dir>/val_source.json`. Returns None if missing."""
    p = run_dir / "val_source.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    per_class_raw = raw.get("per_class_counts")
    per_class: dict[int, tuple[int, int]] | None = (
        {int(k): (int(v[0]), int(v[1])) for k, v in per_class_raw.items()}
        if per_class_raw is not None
        else None
    )
    train_ids_raw = raw.get("train_ids")
    val_ids_raw = raw.get("val_ids")
    missing_raw = raw.get("missing_in_val")
    return ValSource(
        mode=raw["mode"],
        train_ids=(tuple(train_ids_raw) if train_ids_raw is not None else None),
        val_ids=(tuple(val_ids_raw) if val_ids_raw is not None else None),
        realized_fraction=raw.get("realized_fraction"),
        per_class_counts=per_class,
        missing_in_val=(tuple(missing_raw) if missing_raw is not None else None),
        fraction_requested=raw.get("fraction_requested"),
        seed_used=raw.get("seed_used"),
    )


def _log_val_source(vs: ValSource) -> None:
    """Emit the INFO/WARN log lines documented in spec §4.5 / §5.3."""
    if vs.mode == "explicit":
        _LOG.info("val source: explicit (cfg.data.val)")
        return
    if vs.mode == "auto_split":
        assert vs.train_ids is not None and vs.val_ids is not None  # noqa: S101
        assert vs.realized_fraction is not None and vs.fraction_requested is not None  # noqa: S101
        assert vs.per_class_counts is not None  # noqa: S101
        n_train, n_val = len(vs.train_ids), len(vs.val_ids)
        pct = 100.0 * vs.realized_fraction
        covered = sum(1 for (_t, v) in vs.per_class_counts.values() if v > 0)
        total_classes = len(vs.per_class_counts)
        _LOG.info(
            "val source: auto-split fraction=%.2f, realized=train=%d/val=%d (%.2f%%); "
            "coverage=%d/%d classes in val",
            vs.fraction_requested,
            n_train,
            n_val,
            pct,
            covered,
            total_classes,
        )
        if vs.missing_in_val:
            _LOG.warning(
                "auto-split: %d classes missing from val: %s",
                len(vs.missing_in_val),
                list(vs.missing_in_val),
            )
        if (
            abs(vs.realized_fraction - vs.fraction_requested) / vs.fraction_requested > 0.2
            or n_val < 8
        ):
            _LOG.warning("auto-split: realized fraction deviates from requested or val is small")
        return
    # mode == "none"
    _LOG.warning(
        "training without validation set; eval_every is a no-op, end-of-run "
        "eval and bundle samples are skipped. Use data.val to provide one or "
        "data.val_split to auto-split."
    )


def _enumerate_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Dispatch to the right per-format enumerator."""
    if data_cfg.format == "coco":
        return _enumerate_coco_items(data_cfg)
    if data_cfg.format == "hf":
        return _enumerate_hf_items(data_cfg)
    raise ValueError(f"unknown data.format: {data_cfg.format!r}")


def _enumerate_coco_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Walk data.train COCO annotations and return SplittableItems.

    Reuses _load_coco_index / _build_category_remap / _drop_crowd_only_images
    from data/coco.py. Each SplittableItem.image_id is str(int_image_id);
    class_ids is the frozenset of dense ids present after crowd filtering.
    """
    from custom_sam_peft.data.coco import (
        _build_category_remap,
        _drop_crowd_only_images,
        _load_coco_index,
    )

    coco = _load_coco_index(data_cfg.train.annotations)
    _sparse, sparse_to_dense, _names = _build_category_remap(coco)
    kept, ann_index, _dropped = _drop_crowd_only_images(coco)
    items: list[SplittableItem] = []
    for image_id in kept:
        anns = ann_index[image_id]
        class_ids = frozenset(sparse_to_dense[int(a["category_id"])] for a in anns)
        items.append(SplittableItem(image_id=str(image_id), class_ids=class_ids))
    return items


def _enumerate_hf_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Walk data.hf.split_train and return SplittableItems.

    image_id = str(row_index). class_ids is the frozenset of int category ids
    in the row's data.hf.field_map.category field.
    """
    from datasets import load_dataset

    from custom_sam_peft.data.hf import _resolve_field

    assert data_cfg.hf is not None  # noqa: S101 — schema validator guarantees this
    ds = load_dataset(data_cfg.hf.name, split=data_cfg.hf.split_train)
    items: list[SplittableItem] = []
    for i in range(len(ds)):
        row = ds[i]
        try:
            classes_raw = _resolve_field(row, data_cfg.hf.field_map.category)
        except KeyError:
            classes_raw = []
        class_ids = frozenset(int(c) for c in classes_raw)
        items.append(SplittableItem(image_id=str(i), class_ids=class_ids))
    return items
