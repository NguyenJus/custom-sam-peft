"""Unit tests for data/split_source.py — resolver + persistence.

Spec: docs/superpowers/specs/2026-06-04-train-val-test-split-design.md §6, §10.3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    SplitConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.split_source import (
    SplitSource,
    _enumerate_coco_items,
    _log_split_source,
    load_split_source,
    resolve_split_source,
    save_split_source,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _base_cfg(
    tiny_coco_dir: Path,
    *,
    val: bool,
    split_val: float | None = None,
    split_test: float | None = None,
    split_seed: int | None = None,
    run_seed: int = 7,
) -> TrainConfig:
    """Build a TrainConfig pinned at tiny_coco; flags pick the resolved mode."""
    split: SplitConfig | None = None
    if split_val is not None or split_test is not None:
        split = SplitConfig(val=split_val, test=split_test, seed=split_seed)
    return TrainConfig(
        run=RunConfig(name="r", output_dir="./runs", seed=run_seed),
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
            split=split,
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )


# ---------------------------------------------------------------------------
# §10.3 case 1: Mode dispatch
# ---------------------------------------------------------------------------


def test_resolve_mode_auto_split_when_val_set(tiny_coco_dir: Path) -> None:
    """split.val set → mode='auto_split'."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5)
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.mode == "auto_split"
    assert vs.train_ids is not None
    assert vs.val_ids is not None
    assert vs.seed_used == 7  # inherited from run.seed
    assert vs.val_fraction_requested == 0.5
    assert vs.test_ids is None  # no test fraction set
    # Tiny COCO has 2 keep-after-crowd-filter images; fraction=0.5 yields 1+1.
    assert len(vs.train_ids) + len(vs.val_ids) == 2


def test_resolve_mode_none_test_ids_populated_when_test_only(tiny_coco_dir: Path) -> None:
    """split.test only (no val) → mode='none' + test_ids populated."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_test=0.5)
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.mode == "none"
    assert vs.test_ids is not None
    # val_ids should be an empty tuple (val_fraction=0.0)
    assert vs.val_ids is not None
    assert len(vs.val_ids) == 0


def test_resolve_mode_explicit_when_val_set(tiny_coco_dir: Path) -> None:
    """Explicit data.val → mode='explicit'."""
    cfg = _base_cfg(tiny_coco_dir, val=True)
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.mode == "explicit"
    assert vs.train_ids is None
    assert vs.val_ids is None
    assert vs.test_ids is None


def test_resolve_mode_none_when_no_val_source(
    tiny_coco_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Neither val nor split → mode='none'."""
    cfg = _base_cfg(tiny_coco_dir, val=False)
    with caplog.at_level(logging.WARNING):
        vs = resolve_split_source(cfg, run_dir=None)
    assert vs.mode == "none"
    assert vs.train_ids is None
    assert vs.val_ids is None
    assert vs.test_ids is None


# ---------------------------------------------------------------------------
# §10.3 case 2: Save/load round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip_3way(tiny_coco_dir: Path, tmp_path: Path) -> None:
    """save_split_source → load_split_source returns structurally equal SplitSource."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.3, split_test=0.3)
    vs = resolve_split_source(cfg, run_dir=None)
    save_split_source(vs, tmp_path)
    assert (tmp_path / "split_source.json").is_file()
    loaded = load_split_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == vs.mode
    assert loaded.train_ids == vs.train_ids
    assert loaded.val_ids == vs.val_ids
    assert loaded.test_ids == vs.test_ids
    assert loaded.realized_fraction == vs.realized_fraction
    assert loaded.val_fraction_requested == vs.val_fraction_requested
    assert loaded.test_fraction_requested == vs.test_fraction_requested
    assert loaded.seed_used == vs.seed_used
    assert loaded.missing_in_val == vs.missing_in_val
    assert loaded.missing_in_test == vs.missing_in_test
    # per_class_counts JSON-serializes int keys as strings; loader must re-cast.
    if loaded.per_class_counts is not None:
        for k, v in loaded.per_class_counts.items():
            assert isinstance(k, int)
            assert isinstance(v, tuple)
            assert len(v) == 3  # (train, val, test) 3-tuple


def test_save_and_load_round_trip_val_only(tiny_coco_dir: Path, tmp_path: Path) -> None:
    """2-way split (no test): test_ids=None round-trips correctly."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5)
    vs = resolve_split_source(cfg, run_dir=None)
    save_split_source(vs, tmp_path)
    loaded = load_split_source(tmp_path)
    assert loaded is not None
    assert loaded.test_ids is None
    assert loaded.test_fraction_requested is None
    assert loaded.missing_in_test is None


def test_save_and_load_round_trip_none(tiny_coco_dir: Path, tmp_path: Path) -> None:
    """mode='none' persists and reloads cleanly."""
    cfg = _base_cfg(tiny_coco_dir, val=False)
    vs = resolve_split_source(cfg, run_dir=None)
    save_split_source(vs, tmp_path)
    loaded = load_split_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == "none"


def test_load_split_source_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_split_source(tmp_path) is None


def test_atomic_save_does_not_leave_tmp_file(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5)
    vs = resolve_split_source(cfg, run_dir=None)
    save_split_source(vs, tmp_path)
    assert not (tmp_path / "split_source.json.tmp").exists()
    assert (tmp_path / "split_source.json").is_file()


# ---------------------------------------------------------------------------
# §10.3 case 3: Resume preference — stored json returned verbatim
# ---------------------------------------------------------------------------


def test_resume_preference_loads_saved_record(tiny_coco_dir: Path, tmp_path: Path) -> None:
    """Pre-existing split_source.json returned verbatim even when cfg fractions differ."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5)
    vs_first = resolve_split_source(cfg, run_dir=None)
    save_split_source(vs_first, tmp_path)

    # Change fractions in cfg; resolver MUST return the saved record.
    cfg2 = _base_cfg(tiny_coco_dir, val=False, split_val=0.1, split_test=0.1)
    vs_loaded = resolve_split_source(cfg2, run_dir=tmp_path)
    # The SAVED val_fraction_requested (0.5), not the new one (0.1), is returned.
    assert vs_loaded.val_fraction_requested == 0.5
    assert vs_loaded.test_fraction_requested is None  # from the saved record


# ---------------------------------------------------------------------------
# §10.3 case 4: Shared-vocab dense ids across all three buckets
# ---------------------------------------------------------------------------


def test_coco_enumeration_shared_vocab_dense_ids(tiny_coco_dir: Path) -> None:
    """One COCO annotations file → identical dense id space across 3 buckets."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.3, split_test=0.3)
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.per_class_counts is not None
    # All class ids in per_class_counts are dense (int) — same remap applied to all.
    for class_id in vs.per_class_counts:
        assert isinstance(class_id, int)
    # Counts sum correctly: train + val + test == total for each class.
    items = _enumerate_coco_items(cfg.data)
    class_totals: dict[int, int] = {}
    for it in items:
        for c in it.class_ids:
            class_totals[c] = class_totals.get(c, 0) + 1
    for c, (t, v, te) in vs.per_class_counts.items():
        assert t + v + te == class_totals.get(c, 0), (
            f"class {c}: counts {t}+{v}+{te} != total {class_totals.get(c, 0)}"
        )


def test_coco_enumeration_excludes_crowd_only(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False)
    items = _enumerate_coco_items(cfg.data)
    assert len(items) >= 1
    for it in items:
        assert isinstance(it.image_id, str)
        assert isinstance(it.class_ids, frozenset)


# ---------------------------------------------------------------------------
# §10.3 case 5: Seed inheritance
# ---------------------------------------------------------------------------


def test_seed_inheritance_from_run_seed(tiny_coco_dir: Path) -> None:
    """split.seed=None → seed_used inherits run.seed."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5, split_seed=None, run_seed=7)
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.seed_used == 7


def test_seed_explicit_overrides_run_seed(tiny_coco_dir: Path) -> None:
    """split.seed set → seed_used uses it, not run.seed."""
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5, split_seed=99, run_seed=7)
    vs = resolve_split_source(cfg, run_dir=None)
    assert vs.seed_used == 99


# ---------------------------------------------------------------------------
# §10.3 case 6: WARN policy (§6.5)
# ---------------------------------------------------------------------------


def test_warn_policy_small_bucket_emits_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Small realized bucket (< 8) → WARN emitted by _log_split_source."""
    vs = SplitSource(
        mode="auto_split",
        train_ids=tuple(str(i) for i in range(90)),
        val_ids=("1",),  # size=1 < 8 → WARN
        test_ids=None,
        realized_fraction=(0.01, None),
        per_class_counts={0: (90, 1, 0)},
        missing_in_val=(),
        missing_in_test=None,
        val_fraction_requested=0.01,
        test_fraction_requested=None,
        seed_used=0,
    )
    with caplog.at_level(logging.WARNING):
        _log_split_source(vs)
    warn_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("small" in m.lower() or "deviat" in m.lower() for m in warn_messages), (
        f"Expected WARN about small bucket; got: {warn_messages}"
    )


def test_warn_policy_missing_in_test_emits_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Non-empty missing_in_test → WARN emitted by _log_split_source."""
    vs = SplitSource(
        mode="auto_split",
        train_ids=tuple(str(i) for i in range(50)),
        val_ids=tuple(str(i) for i in range(50, 60)),
        test_ids=tuple(str(i) for i in range(60, 70)),
        realized_fraction=(0.1, 0.1),
        per_class_counts={0: (50, 10, 10), 1: (50, 0, 0)},
        missing_in_val=(1,),
        missing_in_test=(1,),
        val_fraction_requested=0.1,
        test_fraction_requested=0.1,
        seed_used=0,
    )
    with caplog.at_level(logging.WARNING):
        _log_split_source(vs)
    warn_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("missing" in m.lower() and "test" in m.lower() for m in warn_messages), (
        f"Expected WARN about missing_in_test; got: {warn_messages}"
    )


def test_warn_policy_deviation_emits_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Large fraction deviation (>20%) → WARN emitted by _log_split_source."""
    # requested=0.1 but realized=0.5 → deviation >> 20%
    vs = SplitSource(
        mode="auto_split",
        train_ids=tuple(str(i) for i in range(50)),
        val_ids=tuple(str(i) for i in range(50, 100)),  # 50 out of 100 = 50%
        test_ids=None,
        realized_fraction=(0.5, None),
        per_class_counts={0: (50, 50, 0)},
        missing_in_val=(),
        missing_in_test=None,
        val_fraction_requested=0.1,
        test_fraction_requested=None,
        seed_used=0,
    )
    with caplog.at_level(logging.WARNING):
        _log_split_source(vs)
    warn_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("deviat" in m.lower() or "small" in m.lower() for m in warn_messages), (
        f"Expected WARN about deviation; got: {warn_messages}"
    )


def test_log_split_source_warns_for_none(caplog: pytest.LogCaptureFixture) -> None:
    """_log_split_source emits a no-val WARN for mode='none'."""
    vs = SplitSource(
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
    with caplog.at_level(logging.WARNING):
        _log_split_source(vs)
    assert any("no-op" in r.message or "no validation" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# HF split_val explicit branch
# ---------------------------------------------------------------------------


def _hf_cfg(split_val: str | None) -> TrainConfig:
    hf: dict[str, object] = {"name": "tiny/ds"}
    if split_val is not None:
        hf["split_val"] = split_val
    return TrainConfig.model_validate(
        {
            "run": {"name": "r"},
            "model": {},
            "data": {
                "format": "hf",
                "train": {"annotations": "unused", "images": "unused"},
                "val": None,
                "hf": hf,
            },
            "peft": {"method": "lora"},
            "train": {"epochs": 1},
        }
    )


def test_resolve_hf_split_val_is_explicit() -> None:
    vs = resolve_split_source(_hf_cfg("myval"), run_dir=None)
    assert vs.mode == "explicit"
    assert vs.train_ids is None
    assert vs.val_ids is None
    assert vs.test_ids is None


def test_resolve_hf_no_split_val_is_none() -> None:
    vs = resolve_split_source(_hf_cfg(None), run_dir=None)
    assert vs.mode == "none"


# ---------------------------------------------------------------------------
# §6.6: No backward-read of val_source.json
# ---------------------------------------------------------------------------


def test_no_backward_read_of_val_source_json(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """A val_source.json in run_dir is NOT loaded — §6.6 explicitly forbids it."""
    # Write a val_source.json with explicit mode (old artifact)
    (tmp_path / "val_source.json").write_text(json.dumps({"mode": "explicit"}))
    # Resolver should NOT find a split_source.json and should recompute.
    cfg = _base_cfg(tiny_coco_dir, val=False, split_val=0.5)
    vs = resolve_split_source(cfg, run_dir=tmp_path)
    # Should be auto_split (recomputed from cfg), not explicit (from old json)
    assert vs.mode == "auto_split"
