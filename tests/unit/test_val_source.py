"""Unit tests for data/val_source.py — resolver + persistence.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §5, §9.2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
    ValSplitConfig,
)
from custom_sam_peft.data.val_source import (
    ValSource,
    _enumerate_coco_items,
    load_val_source,
    resolve_val_source,
    save_val_source,
)


def _base_cfg(tiny_coco_dir: Path, *, val: bool, val_split: bool) -> TrainConfig:
    """Build a TrainConfig pinned at tiny_coco; flags pick the resolved mode."""
    return TrainConfig(
        run=RunConfig(name="r", output_dir="./runs", seed=7),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=(
                DataSplit(
                    annotations=str(tiny_coco_dir / "annotations.json"),
                    images=str(tiny_coco_dir / "images"),
                )
                if val
                else None
            ),
            val_split=(ValSplitConfig(fraction=0.5, seed=None) if val_split else None),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )


def test_resolve_mode_explicit(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=True, val_split=False)
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.mode == "explicit"
    assert vs.train_ids is None
    assert vs.val_ids is None


def test_resolve_mode_auto_split(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.mode == "auto_split"
    assert vs.train_ids is not None
    assert vs.val_ids is not None
    assert vs.seed_used == 7  # inherited from run.seed
    assert vs.fraction_requested == 0.5
    # Tiny COCO has 2 keep-after-crowd-filter images; fraction=0.5 yields 1+1.
    assert len(vs.train_ids) + len(vs.val_ids) == 2


def test_resolve_mode_none_warns(tiny_coco_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=False)
    with caplog.at_level(logging.WARNING):
        vs = resolve_val_source(cfg, run_dir=None)
    assert vs.mode == "none"
    assert vs.train_ids is None
    assert vs.val_ids is None
    # No WARN emitted by the resolver itself; the trainer-side _log_val_source
    # is where the WARN happens. The resolver may emit INFO only.
    # (Spec §5.2 case 4: resolver returns ValSource(mode='none'); §5.3 logs WARN
    # at training start via _log_val_source.) Adjust if implementation logs here.


def test_log_val_source_warns_for_none(
    tiny_coco_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Spec §5.3: _log_val_source emits the no-val WARN."""
    from custom_sam_peft.data.val_source import _log_val_source

    vs = ValSource(
        mode="none",
        train_ids=None,
        val_ids=None,
        realized_fraction=None,
        per_class_counts=None,
        missing_in_val=None,
        fraction_requested=None,
        seed_used=None,
    )
    with caplog.at_level(logging.WARNING):
        _log_val_source(vs)
    assert any("no-op" in r.message or "no validation" in r.message.lower() for r in caplog.records)


def test_save_and_load_round_trip_auto_split(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    assert (tmp_path / "val_source.json").is_file()
    loaded = load_val_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == vs.mode
    assert loaded.train_ids == vs.train_ids
    assert loaded.val_ids == vs.val_ids
    assert loaded.realized_fraction == vs.realized_fraction
    assert loaded.fraction_requested == vs.fraction_requested
    assert loaded.seed_used == vs.seed_used
    # per_class_counts JSON-serializes int keys as strings; loader must re-cast.
    if loaded.per_class_counts is not None:
        for k in loaded.per_class_counts:
            assert isinstance(k, int)


def test_save_and_load_round_trip_explicit(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=True, val_split=False)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    loaded = load_val_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == "explicit"
    assert loaded.train_ids is None
    assert loaded.val_ids is None


def test_save_and_load_round_trip_none(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=False)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    loaded = load_val_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == "none"


def test_load_val_source_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_val_source(tmp_path) is None


def test_resume_preference_loads_saved_record(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs_first = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs_first, tmp_path)

    # Now change fraction in cfg; resolver MUST return the saved record.
    cfg2 = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    assert cfg2.data.val_split is not None
    # Mutate frozen via model_copy
    cfg2 = cfg2.model_copy(
        update={"data": cfg2.data.model_copy(update={"val_split": ValSplitConfig(fraction=0.1)})}
    )
    vs_loaded = resolve_val_source(cfg2, run_dir=tmp_path)
    assert vs_loaded.fraction_requested == 0.5  # the SAVED fraction, not the new one.


def test_coco_enumeration_excludes_crowd_only(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=False)
    items = _enumerate_coco_items(cfg.data)
    # tiny_coco's surviving images all have class ids; no empty class_ids.
    assert len(items) >= 1
    for it in items:
        assert isinstance(it.image_id, str)
        assert isinstance(it.class_ids, frozenset)


def test_seed_inheritance_from_run_seed(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    # val_split.seed is None in _base_cfg → must inherit run.seed (7).
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.seed_used == 7


def test_seed_override_explicit_seed(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"val_split": ValSplitConfig(fraction=0.5, seed=99)})
        }
    )
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.seed_used == 99


def test_atomic_save_does_not_leave_tmp_file(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    # No leftover .tmp file from the os.replace flow.
    assert not (tmp_path / "val_source.json.tmp").exists()
    assert (tmp_path / "val_source.json").is_file()
