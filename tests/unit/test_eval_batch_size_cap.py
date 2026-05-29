"""Unit tests: eval auto-batch is capped by the train batch size at all three callsites.

Bug: eval auto-batch could exceed the train batch size (e.g. 4x larger), causing
eval OOM even when training was stable. Fix: min(picked, train_cap) at each site.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.train.loop import OomState

# ---------------------------------------------------------------------------
# Callsite 1: Trainer._eval_epoch
# ---------------------------------------------------------------------------


def _make_trainer_with_val_ds() -> object:
    """Build a Trainer whose val_ds is a non-None stub, CPU-safe."""
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

    cfg = TrainConfig(
        run=RunConfig(name="t", output_dir="./runs", seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(epochs=1, batch_size=2),
        eval=EvalConfig(batch_size="auto"),
        tracking=TrackingConfig(backend="none"),
    )
    model = MagicMock()
    val_ds = MagicMock()
    return Trainer(model, val_ds, val_ds, NoopTracker(), cfg)


def test_eval_epoch_caps_auto_batch_by_oom_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_eval_epoch must cap decide_eval_batch_size result by oom_state.micro_batch_size."""
    trainer = _make_trainer_with_val_ds()

    # predictor would pick 8, but oom_state has halved train batch to 2
    # The import is lazy (inside the if-block), so patch at the presets module.
    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        lambda *_a, **_kw: (8, 1, "analytic"),
    )

    captured: list[object] = []

    def _fake_evaluator(cfg: object) -> object:
        captured.append(cfg)
        ev = MagicMock()
        ev.evaluate = MagicMock(return_value=MagicMock(overall={}))
        return ev

    monkeypatch.setattr("custom_sam_peft.train.trainer.Evaluator", _fake_evaluator)

    oom_state = OomState(micro_batch_size=2)
    trainer._eval_epoch(step=1, run_dir=tmp_path, oom_state=oom_state)  # type: ignore[attr-defined]

    assert len(captured) == 1
    assert captured[0].batch_size == 2, (
        f"Expected batch_size=2 (capped), got {captured[0].batch_size}"
    )


def test_eval_epoch_no_cap_when_predictor_within_train_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When predictor picks ≤ oom_state.micro_batch_size, the value is unchanged."""
    trainer = _make_trainer_with_val_ds()

    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        lambda *_a, **_kw: (2, 1, "analytic"),
    )

    captured: list[object] = []

    def _fake_evaluator(cfg: object) -> object:
        captured.append(cfg)
        ev = MagicMock()
        ev.evaluate = MagicMock(return_value=MagicMock(overall={}))
        return ev

    monkeypatch.setattr("custom_sam_peft.train.trainer.Evaluator", _fake_evaluator)

    oom_state = OomState(micro_batch_size=4)
    trainer._eval_epoch(step=1, run_dir=tmp_path, oom_state=oom_state)  # type: ignore[attr-defined]

    assert len(captured) == 1
    assert captured[0].batch_size == 2


def test_eval_epoch_no_cap_when_oom_state_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When oom_state is None (legacy / no-train path), no cap is applied."""
    trainer = _make_trainer_with_val_ds()

    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        lambda *_a, **_kw: (8, 1, "analytic"),
    )

    captured: list[object] = []

    def _fake_evaluator(cfg: object) -> object:
        captured.append(cfg)
        ev = MagicMock()
        ev.evaluate = MagicMock(return_value=MagicMock(overall={}))
        return ev

    monkeypatch.setattr("custom_sam_peft.train.trainer.Evaluator", _fake_evaluator)

    trainer._eval_epoch(step=1, run_dir=tmp_path, oom_state=None)  # type: ignore[attr-defined]

    assert len(captured) == 1
    assert captured[0].batch_size == 8


# ---------------------------------------------------------------------------
# Callsite 2: Trainer.fit post-train eval
# ---------------------------------------------------------------------------


def test_fit_post_train_eval_caps_auto_batch() -> None:
    """fit() post-train eval caps via _cap_eval_batch_size — test the helper directly.

    The post-train block in fit() calls self._cap_eval_batch_size(bs, oom_state.micro_batch_size).
    Exercising the helper is the correct unit for this callsite.
    """
    trainer = _make_trainer_with_val_ds()

    # predictor picks 8; sticky micro-batch is 2 (as if OOM halved it)
    result = trainer._cap_eval_batch_size(8, 2)  # type: ignore[attr-defined]
    assert result == 2, f"Expected capped batch_size=2, got {result}"


def test_fit_post_train_no_cap_when_within_micro_batch() -> None:
    """_cap_eval_batch_size returns the original value when bs <= cap."""
    trainer = _make_trainer_with_val_ds()

    result = trainer._cap_eval_batch_size(2, 4)  # type: ignore[attr-defined]
    assert result == 2, f"Expected uncapped batch_size=2, got {result}"


# ---------------------------------------------------------------------------
# Callsite 3: run_eval in eval/runner.py
# ---------------------------------------------------------------------------


def _make_run_eval_cfg(train_batch_size: int = 2) -> MagicMock:
    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {
        "format": "coco",
        "train": {"annotations": "t.json", "images": "t/"},
        "val": {"annotations": "v.json", "images": "v/"},
        "test": None,
    }
    cfg.data.val = MagicMock()
    cfg.data.val_split = None
    cfg.data.test = None
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.train.batch_size = train_batch_size

    from custom_sam_peft.config.schema import EvalConfig

    cfg.eval = EvalConfig(batch_size="auto", visualize=False)
    return cfg


def test_run_eval_caps_auto_batch_by_train_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval must cap decide_eval_batch_size result by cfg.train.batch_size."""
    from custom_sam_peft.eval.runner import run_eval

    cfg = _make_run_eval_cfg(train_batch_size=2)

    # Lazy import: patch at the presets module so the local binding picks it up.
    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        lambda *_a, **_kw: (8, 1, "analytic"),
    )

    captured: list[object] = []

    def _fake_evaluator(eval_cfg: object) -> object:
        captured.append(eval_cfg)
        ev = MagicMock()
        ev.evaluate_and_save = MagicMock(return_value=MagicMock())
        return ev

    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", _fake_evaluator)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_adapter", lambda *_a, **_kw: None)

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert len(captured) == 1
    assert captured[0].batch_size == 2, (
        f"Expected batch_size=2 (capped), got {captured[0].batch_size}"
    )


def test_run_eval_no_cap_when_predictor_within_train_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When predictor picks ≤ cfg.train.batch_size, the value is unchanged."""
    from custom_sam_peft.eval.runner import run_eval

    cfg = _make_run_eval_cfg(train_batch_size=4)

    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        lambda *_a, **_kw: (3, 1, "analytic"),
    )

    captured: list[object] = []

    def _fake_evaluator(eval_cfg: object) -> object:
        captured.append(eval_cfg)
        ev = MagicMock()
        ev.evaluate_and_save = MagicMock(return_value=MagicMock())
        return ev

    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", _fake_evaluator)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_adapter", lambda *_a, **_kw: None)

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert len(captured) == 1
    assert captured[0].batch_size == 3


def test_run_eval_cap_logs_info_when_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An INFO line must be logged when the cap actually lowers the batch size."""
    import logging

    from custom_sam_peft.eval.runner import run_eval

    cfg = _make_run_eval_cfg(train_batch_size=2)

    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        lambda *_a, **_kw: (8, 1, "analytic"),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=MagicMock())),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_adapter", lambda *_a, **_kw: None)

    with caplog.at_level(logging.INFO, logger="custom_sam_peft.eval.runner"):
        run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    log_text = " ".join(r.message for r in caplog.records)
    assert "capped" in log_text.lower(), f"Expected cap log message; got: {log_text!r}"
