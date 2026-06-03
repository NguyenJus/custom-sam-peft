"""run_eval dispatches to SemanticEvaluator under task: semantic (§10.2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.eval.runner import run_eval


def _make_semantic_cfg(format_: str = "coco") -> MagicMock:
    """Build a mock TrainConfig with task='semantic'."""
    cfg = MagicMock()
    cfg.task = "semantic"
    cfg.data.format = format_
    cfg.data.model_dump.return_value = {
        "format": format_,
        "train": {"annotations": "t.json", "images": "t/"},
        "val": {"annotations": "v.json", "images": "v/"},
        "test": None,
    }
    cfg.data.val = MagicMock()
    cfg.data.val_split = None
    cfg.data.test = None
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.eval.model_copy = lambda update=None: cfg.eval
    cfg.eval.visualize = False
    return cfg


def test_run_eval_uses_semantic_evaluator_under_semantic_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval must construct a SemanticEvaluator (not Evaluator) when cfg.task='semantic'."""
    cfg = _make_semantic_cfg()

    fake_report = MagicMock()
    seen: dict[str, bool] = {}

    def _fake_semantic_evaluator(eval_cfg: object) -> object:
        seen["constructed"] = True
        ev = MagicMock()
        ev.evaluate_and_save = MagicMock(return_value=fake_report)
        return ev

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.SemanticEvaluator", _fake_semantic_evaluator
    )
    # Evaluator must NOT be constructed for the semantic path.
    instance_constructed: list[object] = []
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: instance_constructed.append(_cfg) or MagicMock(),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert seen.get("constructed"), "SemanticEvaluator must be constructed for task='semantic'"
    assert instance_constructed == [], "Evaluator must NOT be constructed for task='semantic'"
    assert result is fake_report


def test_run_eval_uses_instance_evaluator_under_instance_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: run_eval must still construct Evaluator when cfg.task='instance'."""
    cfg = _make_semantic_cfg()
    cfg.task = "instance"

    fake_report = MagicMock()
    seen: dict[str, bool] = {}

    def _fake_evaluator(eval_cfg: object) -> object:
        seen["constructed"] = True
        ev = MagicMock()
        ev.evaluate_and_save = MagicMock(return_value=fake_report)
        return ev

    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", _fake_evaluator)
    semantic_constructed: list[object] = []
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.SemanticEvaluator",
        lambda _cfg: semantic_constructed.append(_cfg) or MagicMock(),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert seen.get("constructed"), "Evaluator must be constructed for task='instance'"
    assert semantic_constructed == [], "SemanticEvaluator must NOT be constructed for task='instance'"
    assert result is fake_report
