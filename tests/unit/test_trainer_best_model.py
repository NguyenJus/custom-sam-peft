"""Tests for Trainer best-model tracking: _eval_epoch / _maybe_save_best.

TDD spec: #174 — trainer must save the best mAP adapter to run_dir/best/ during
periodic lite evals, not just the final-step adapter.

Mocking conventions match test_eval_batch_size_cap.py (Evaluator + save_adapter
monkeypatched; Trainer built with MagicMock model/val_ds + NoopTracker).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    EvalConfig,
    PEFTConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer


def _make_trainer() -> Trainer:
    """Build a CPU-safe Trainer with a stubbed model and non-None val_ds."""
    cfg = TrainConfig(
        run=RunConfig(name="best-test", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(epochs=1, batch_size=1),
        eval=EvalConfig(batch_size=1),
        tracking=TrackingConfig(backend="none"),
    )
    model = MagicMock()
    val_ds = MagicMock()
    return Trainer(model, val_ds, val_ds, NoopTracker(), cfg)


# ---------------------------------------------------------------------------
# Helper: build a fake Evaluator class that returns a preset mAP sequence.
# ---------------------------------------------------------------------------


def _fake_evaluator_factory(map_sequence: list[float]) -> type:
    """Return an Evaluator *class* (callable → instance) that yields mAP values
    from ``map_sequence`` in order each time .evaluate() is called.
    """
    call_counter = {"n": 0}
    maps = map_sequence

    class _FakeEvaluator:
        def __init__(self, _cfg: object) -> None:
            pass

        def evaluate(self, _model: object, _ds: object) -> object:
            i = call_counter["n"]
            call_counter["n"] += 1
            report = MagicMock()
            report.overall = {"mAP": maps[i % len(maps)]}
            return report

    return _FakeEvaluator


# ---------------------------------------------------------------------------
# Core test: rising-then-falling mAP; best/adapter tracks the peak step.
# ---------------------------------------------------------------------------


def test_maybe_save_best_tracks_peak_map_across_eval_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_eval_epoch/_maybe_save_best saves best adapter at peak mAP step.

    mAP sequence: 0.1 (step 1) → 0.5 (step 2, new best) → 0.3 (step 3, no update).
    Asserts:
    - run_dir/best/adapter exists after the sequence.
    - run_dir/best/best.json has value=0.5 and global_step=2.
    """
    trainer = _make_trainer()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    map_seq = [0.1, 0.5, 0.3]
    monkeypatch.setattr(
        "custom_sam_peft.train.trainer.Evaluator",
        _fake_evaluator_factory(map_seq),
    )

    saved_calls: list[Path] = []

    def _stub_save_adapter(wrapper: object, path: Path) -> None:
        saved_calls.append(path)
        # Actually create the directory to mimic real save_adapter behaviour.
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("custom_sam_peft.train.trainer.save_adapter", _stub_save_adapter)

    # Drive three eval steps with rising-then-falling mAP.
    trainer._eval_epoch(step=1, run_dir=run_dir)
    trainer._eval_epoch(step=2, run_dir=run_dir)
    trainer._eval_epoch(step=3, run_dir=run_dir)

    # best/adapter must exist.
    assert (run_dir / "best" / "adapter").exists(), (
        "run_dir/best/adapter was not created despite a new-best mAP being observed."
    )

    # best.json must record the peak.
    best_json_path = run_dir / "best" / "best.json"
    assert best_json_path.exists(), "run_dir/best/best.json was not written."
    payload = json.loads(best_json_path.read_text())
    assert payload["metric"] == "mAP", f"Expected metric='mAP', got {payload['metric']!r}"
    assert payload["value"] == pytest.approx(0.5), (
        f"Expected value=0.5 (peak mAP), got {payload['value']}"
    )
    assert payload["global_step"] == 2, (
        f"Expected global_step=2 (peak step), got {payload['global_step']}"
    )


def test_maybe_save_best_does_not_save_when_metric_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No best/ directory is created when the report has no 'mAP' key."""
    trainer = _make_trainer()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class _NoMapEvaluator:
        def __init__(self, _cfg: object) -> None:
            pass

        def evaluate(self, _model: object, _ds: object) -> object:
            report = MagicMock()
            report.overall = {"precision": 0.9}  # no mAP key
            return report

    monkeypatch.setattr("custom_sam_peft.train.trainer.Evaluator", _NoMapEvaluator)

    saved_calls: list[Path] = []

    def _stub_save_adapter(wrapper: object, path: Path) -> None:
        saved_calls.append(path)
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("custom_sam_peft.train.trainer.save_adapter", _stub_save_adapter)

    trainer._eval_epoch(step=1, run_dir=run_dir)

    assert not (run_dir / "best").exists(), (
        "run_dir/best/ should not be created when 'mAP' is absent from report."
    )


def test_maybe_save_best_only_saves_when_strictly_better(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_adapter is called exactly once for the peak, not on equal or lower mAP."""
    trainer = _make_trainer()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Sequence: 0.5, 0.5 (equal — no second save), 0.3 (lower — no save)
    monkeypatch.setattr(
        "custom_sam_peft.train.trainer.Evaluator",
        _fake_evaluator_factory([0.5, 0.5, 0.3]),
    )

    best_save_calls: list[Path] = []

    def _stub_save_adapter(wrapper: object, path: Path) -> None:
        best_save_calls.append(path)
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("custom_sam_peft.train.trainer.save_adapter", _stub_save_adapter)

    trainer._eval_epoch(step=1, run_dir=run_dir)
    trainer._eval_epoch(step=2, run_dir=run_dir)
    trainer._eval_epoch(step=3, run_dir=run_dir)

    # Only one best-save call (for step=1, the first best).
    best_saves = [p for p in best_save_calls if "best" in str(p)]
    assert len(best_saves) == 1, (
        f"Expected exactly 1 best-save call; got {len(best_saves)}: {best_saves}"
    )


def test_maybe_save_best_no_val_ds_creates_no_best_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When val_ds is None, _eval_epoch returns early and no best/ dir is created."""
    cfg = TrainConfig(
        run=RunConfig(name="no-val", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=None,
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(epochs=1),
        eval=EvalConfig(batch_size=1),
        tracking=TrackingConfig(backend="none"),
    )
    model = MagicMock()
    trainer = Trainer(model, MagicMock(), None, NoopTracker(), cfg)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    called = []

    def _stub_save_adapter(wrapper: object, path: Path) -> None:
        called.append(path)

    monkeypatch.setattr("custom_sam_peft.train.trainer.save_adapter", _stub_save_adapter)

    trainer._eval_epoch(step=1, run_dir=run_dir)

    assert not (run_dir / "best").exists(), (
        "best/ dir must not be created when val_ds is None (eval is skipped)."
    )
    assert called == [], "save_adapter must not be called when val_ds is None."


def test_best_json_metric_key_is_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """best.json must have metric='mAP' (exact string)."""
    trainer = _make_trainer()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "custom_sam_peft.train.trainer.Evaluator",
        _fake_evaluator_factory([0.42]),
    )

    def _stub_save_adapter(wrapper: object, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("custom_sam_peft.train.trainer.save_adapter", _stub_save_adapter)

    trainer._eval_epoch(step=5, run_dir=run_dir)

    payload = json.loads((run_dir / "best" / "best.json").read_text())
    assert payload == {"metric": "mAP", "value": pytest.approx(0.42), "global_step": 5}


def test_maybe_save_best_does_not_crash_when_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing save_adapter must be swallowed — _eval_epoch returns normally.

    This is the core safety property: a disk-write failure during best-model
    saving must never propagate and crash training.
    """
    trainer = _make_trainer()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "custom_sam_peft.train.trainer.Evaluator",
        _fake_evaluator_factory([0.5]),
    )

    def _failing_save_adapter(wrapper: object, path: Path) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr("custom_sam_peft.train.trainer.save_adapter", _failing_save_adapter)

    # Must not raise — the exception is logged and swallowed.
    trainer._eval_epoch(step=1, run_dir=run_dir)

    # Best was never committed to disk, so a later equal-mAP eval can retry.
    assert trainer._best_metric_value == float("-inf")
