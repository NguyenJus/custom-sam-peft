"""Resolve the validation/test source for a run: explicit, auto_split, or none.

The resolver is the single seam between schema and the splitter. It also
owns persistence (`save_split_source` / `load_split_source`) of the resolved
record to `<run_dir>/split_source.json`. Trainer hparams logging and tracker
hparams injection both read the saved record, so the resolver writes the
authoritative copy once per run (before Trainer.fit begins).

Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §6.
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

SplitMode = Literal["explicit", "auto_split", "none"]


@dataclass(frozen=True)
class SplitSource:
    mode: SplitMode  # val-side mode (unchanged semantics)
    train_ids: tuple[str, ...] | None
    val_ids: tuple[str, ...] | None
    test_ids: tuple[str, ...] | None  # populated when split.test set
    realized_fraction: tuple[float, float] | None  # (val, test); auto_split only
    per_class_counts: dict[int, tuple[int, int, int]] | None  # (train, val, test)
    missing_in_val: tuple[int, ...] | None
    missing_in_test: tuple[int, ...] | None
    val_fraction_requested: float | None  # auto_split only
    test_fraction_requested: float | None  # auto_split only
    seed_used: int | None


def resolve_split_source(cfg: TrainConfig, *, run_dir: Path | None = None) -> SplitSource:
    """Resolve which validation/test source to use for this run.

    Dispatch (spec §6.2):
      1. run_dir/split_source.json exists → load_split_source(run_dir) (resume).
      2. cfg.data.split is not None → enumerate + stratify (auto_split or none).
      3. cfg.data.val is not None → mode='explicit'.
      4. cfg.data.format=='hf' and cfg.data.hf.split_val is not None → mode='explicit' (HF).
      5. else → mode='none'.
    """
    if run_dir is not None:
        saved = load_split_source(run_dir)
        if saved is not None:
            _LOG.info("split_source: resumed from %s (mode=%s)", run_dir, saved.mode)
            return saved

    if cfg.data.split is not None:
        split_cfg = cfg.data.split
        seed_used = split_cfg.seed if split_cfg.seed is not None else cfg.run.seed
        items = _enumerate_items(cfg.data)
        result = stratified_split(
            items,
            split_cfg.val or 0.0,
            split_cfg.test or 0.0,
            seed_used,
        )
        mode: SplitMode = "auto_split" if split_cfg.val is not None else "none"
        return SplitSource(
            mode=mode,
            train_ids=result.train_ids,
            val_ids=result.val_ids,
            test_ids=result.test_ids if split_cfg.test is not None else None,
            realized_fraction=result.realized_fraction,
            per_class_counts=result.per_class_counts,
            missing_in_val=result.missing_in_val,
            missing_in_test=result.missing_in_test if split_cfg.test is not None else None,
            val_fraction_requested=split_cfg.val,
            test_fraction_requested=split_cfg.test,
            seed_used=seed_used,
        )

    if cfg.data.val is not None:
        return SplitSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            test_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            missing_in_test=None,
            val_fraction_requested=None,
            test_fraction_requested=None,
            seed_used=None,
        )

    if cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None:
        return SplitSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            test_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            missing_in_test=None,
            val_fraction_requested=None,
            test_fraction_requested=None,
            seed_used=None,
        )

    return SplitSource(
        mode="none",
        train_ids=None,
        val_ids=None,
        test_ids=None,
        realized_fraction=None,
        per_class_counts=None,
        missing_in_val=None,
        missing_in_test=None,
        val_fraction_requested=None,
        test_fraction_requested=None,
        seed_used=None,
    )


def save_split_source(vs: SplitSource, run_dir: Path) -> None:
    """Write `<run_dir>/split_source.json`. Atomic via tmp + os.replace."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "mode": vs.mode,
        "val_fraction_requested": vs.val_fraction_requested,
        "test_fraction_requested": vs.test_fraction_requested,
        "seed_used": vs.seed_used,
        "realized_fraction": (
            list(vs.realized_fraction) if vs.realized_fraction is not None else None
        ),
        "n_train": (len(vs.train_ids) if vs.train_ids is not None else None),
        "n_val": (len(vs.val_ids) if vs.val_ids is not None else None),
        "n_test": (len(vs.test_ids) if vs.test_ids is not None else None),
        "per_class_counts": (
            {str(k): list(v) for k, v in vs.per_class_counts.items()}
            if vs.per_class_counts is not None
            else None
        ),
        "missing_in_val": (list(vs.missing_in_val) if vs.missing_in_val is not None else None),
        "missing_in_test": (list(vs.missing_in_test) if vs.missing_in_test is not None else None),
        "train_ids": (list(vs.train_ids) if vs.train_ids is not None else None),
        "val_ids": (list(vs.val_ids) if vs.val_ids is not None else None),
        "test_ids": (list(vs.test_ids) if vs.test_ids is not None else None),
    }
    tmp = run_dir / "split_source.json.tmp"
    final = run_dir / "split_source.json"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, final)


def load_split_source(run_dir: Path) -> SplitSource | None:
    """Read `<run_dir>/split_source.json`. Returns None if missing.

    Note: does NOT read the old val_source.json — §6.6 explicitly forbids
    backward-read compat.
    """
    p = run_dir / "split_source.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    per_class_raw = raw.get("per_class_counts")
    per_class: dict[int, tuple[int, int, int]] | None = (
        {int(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in per_class_raw.items()}
        if per_class_raw is not None
        else None
    )
    train_ids_raw = raw.get("train_ids")
    val_ids_raw = raw.get("val_ids")
    test_ids_raw = raw.get("test_ids")
    missing_val_raw = raw.get("missing_in_val")
    missing_test_raw = raw.get("missing_in_test")
    realized_raw = raw.get("realized_fraction")
    return SplitSource(
        mode=raw["mode"],
        train_ids=(tuple(train_ids_raw) if train_ids_raw is not None else None),
        val_ids=(tuple(val_ids_raw) if val_ids_raw is not None else None),
        test_ids=(tuple(test_ids_raw) if test_ids_raw is not None else None),
        realized_fraction=(
            (float(realized_raw[0]), float(realized_raw[1])) if realized_raw is not None else None
        ),
        per_class_counts=per_class,
        missing_in_val=(
            tuple(int(x) for x in missing_val_raw) if missing_val_raw is not None else None
        ),
        missing_in_test=(
            tuple(int(x) for x in missing_test_raw) if missing_test_raw is not None else None
        ),
        val_fraction_requested=raw.get("val_fraction_requested"),
        test_fraction_requested=raw.get("test_fraction_requested"),
        seed_used=raw.get("seed_used"),
    )


def _log_split_source(vs: SplitSource) -> None:
    """Emit the INFO/WARN log lines documented in spec §6.4 and §6.5."""
    if vs.mode == "explicit":
        _LOG.info("split source: explicit (cfg.data.val or data.hf.split_val)")
        return
    if vs.mode == "auto_split":
        if vs.train_ids is None or vs.val_ids is None:
            return
        if vs.realized_fraction is None or vs.val_fraction_requested is None:
            return
        if vs.per_class_counts is None:
            return
        n_train, n_val = len(vs.train_ids), len(vs.val_ids)
        val_realized, test_realized_or_none = vs.realized_fraction
        pct_val = 100.0 * val_realized
        covered_val = sum(1 for (_t, v, _te) in vs.per_class_counts.values() if v > 0)
        total_classes = len(vs.per_class_counts)
        _LOG.info(
            "split source: auto-split val=%.2f, realized=train=%d/val=%d (%.2f%%); "
            "coverage=%d/%d classes in val",
            vs.val_fraction_requested,
            n_train,
            n_val,
            pct_val,
            covered_val,
            total_classes,
        )
        # val WARN: missing classes
        if vs.missing_in_val:
            _LOG.warning(
                "auto-split: %d classes missing from val: %s",
                len(vs.missing_in_val),
                list(vs.missing_in_val),
            )
        # val WARN: deviation / small bucket (per §6.5 applied per present bucket)
        if vs.val_fraction_requested > 0.0 and (
            abs(val_realized - vs.val_fraction_requested) / vs.val_fraction_requested > 0.2
            or n_val < 8
        ):
            _LOG.warning(
                "auto-split: val realized fraction deviates from requested or val is small"
            )
        # test INFO + WARN lines (when test_ids populated)
        if vs.test_ids is not None:
            n_test = len(vs.test_ids)
            test_realized = test_realized_or_none if test_realized_or_none is not None else 0.0
            pct_test = 100.0 * test_realized
            covered_test = sum(1 for (_t, _v, te) in vs.per_class_counts.values() if te > 0)
            _LOG.info(
                "auto-split test: realized=test=%d (%.2f%%); coverage=%d/%d classes in test",
                n_test,
                pct_test,
                covered_test,
                total_classes,
            )
            # test WARN: missing classes
            if vs.missing_in_test:
                _LOG.warning(
                    "auto-split: %d classes missing from test: %s",
                    len(vs.missing_in_test),
                    list(vs.missing_in_test),
                )
            # test WARN: deviation / small bucket
            if (
                vs.test_fraction_requested is not None
                and vs.test_fraction_requested > 0.0
                and (
                    abs(test_realized - vs.test_fraction_requested) / vs.test_fraction_requested
                    > 0.2
                    or n_test < 8
                )
            ):
                _LOG.warning(
                    "auto-split: test realized fraction deviates from requested or test is small"
                )
        return
    # mode == "none"
    _LOG.warning(
        "training without validation set; eval_every is a no-op, end-of-run "
        "eval and bundle samples are skipped. Use data.val to provide one or "
        "data.split to auto-split."
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

    if data_cfg.hf is None:  # narrow structurally (ruff S101: no bare assert in src/)
        raise ValueError("data.hf is None but format=='hf'")
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
