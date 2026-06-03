"""run_eval dispatches to SemanticEvaluator under task: semantic (§10.2)."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# spec §8.2: task-tagged inline metrics.json writers
# ---------------------------------------------------------------------------


def _patch_common_semantic(
    monkeypatch: pytest.MonkeyPatch,
    fake_report: MagicMock,
    per_example_iou: list[float],
) -> None:
    """Patch the parts of run_eval common to the inline-writer tests."""
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["sky"]),
    )
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

    ev = MagicMock()

    def _evaluate(
        model: object,
        dataset: object,
        *,
        return_per_example_iou: bool = False,
    ) -> object:
        if return_per_example_iou:
            return fake_report, per_example_iou
        return fake_report

    ev.evaluate = _evaluate
    ev.evaluate_and_save = MagicMock(return_value=fake_report)

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.SemanticEvaluator",
        lambda _cfg: ev,
    )
    # Evaluator must not be constructed for semantic task.
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("Evaluator must not be used for semantic")),
    )


def test_inline_return_per_example_iou_semantic_tags_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """return_per_example_iou=True (inline branch 1) must write metrics.json with
    'task': 'semantic' when cfg.task == 'semantic' (spec §8.2)."""
    cfg = _make_semantic_cfg()
    fake_report = MagicMock(
        overall={"mIoU": 0.6, "pixel_acc": 0.9},
        per_class={"sky": {"IoU": 0.6}},
        n_images=2,
        n_predictions=1000,
    )
    _patch_common_semantic(monkeypatch, fake_report, [0.6, 0.7])

    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path, return_per_example_iou=True)

    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload.get("task") == "semantic", (
        f"Expected 'task': 'semantic' in metrics.json; got keys: {list(payload)}"
    )
    # task must be the FIRST key (ordering consistency with evaluate_and_save).
    assert list(payload.keys())[0] == "task", (
        f"'task' must be the first key; got ordering: {list(payload.keys())}"
    )


def test_inline_visualize_on_semantic_tags_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """visualize=True (inline branch 2) must write metrics.json with 'task': 'semantic'
    when cfg.task == 'semantic' (spec §8.2). viz pass is monkeypatched to a no-op."""
    cfg = _make_semantic_cfg()
    cfg.eval.visualize = True
    cfg.eval.visualize_count = 3
    cfg.eval.mask_threshold = 0.0
    cfg.eval.save_predictions = False
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"

    fake_report = MagicMock(
        overall={"mIoU": 0.55, "pixel_acc": 0.85},
        per_class={"sky": {"IoU": 0.55}},
        n_images=1,
        n_predictions=500,
    )
    _patch_common_semantic(monkeypatch, fake_report, [0.55])

    # Stub out write_eval_visualizations so the viz pass succeeds silently.
    monkeypatch.setattr(
        "custom_sam_peft.eval.visualize.write_eval_visualizations",
        lambda *a, **k: [],
    )

    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)

    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload.get("task") == "semantic", (
        f"Expected 'task': 'semantic' in metrics.json (visualize branch); got keys: {list(payload)}"
    )
    assert list(payload.keys())[0] == "task", (
        f"'task' must be the first key; got ordering: {list(payload.keys())}"
    )


def test_inline_return_per_example_iou_instance_no_task_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: return_per_example_iou=True with task='instance' must write
    metrics.json with NO 'task' key (byte-identical invariance, spec §12)."""
    cfg = _make_semantic_cfg()
    cfg.task = "instance"

    fake_report = MagicMock(
        overall={"mAP": 0.5},
        per_class={},
        n_images=3,
        n_predictions=3,
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_lora", lambda *_a, **_kw: None)

    ev = MagicMock()
    ev.evaluate.return_value = (fake_report, [0.5, 0.4, 0.6])
    ev._last_predictions = []

    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _cfg: ev)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.SemanticEvaluator",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("SemanticEvaluator must not be used for instance")),
    )

    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path, return_per_example_iou=True)

    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert "task" not in payload, (
        f"Instance path must NOT include 'task' key; got keys: {list(payload)}"
    )
