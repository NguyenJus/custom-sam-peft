"""run_eval builds dataset via registry, loads adapter, calls Evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_sam_peft.eval.runner import run_eval


def _make_cfg(
    format_: str = "coco", peft_method: str = "lora", has_test: bool = False
) -> MagicMock:
    cfg = MagicMock()
    cfg.data.format = format_
    cfg.data.model_dump.return_value = {
        "format": format_,
        "train": {"annotations": "t.json", "images": "t/"},
        "val": {"annotations": "v.json", "images": "v/"},
        "test": ({"annotations": "te.json", "images": "te/"} if has_test else None),
        "prompt_mode": "text",
        "image_size": 1008,
    }
    cfg.data.val = MagicMock()
    cfg.data.test = MagicMock() if has_test else None
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = peft_method
    cfg.eval.model_copy = lambda update=None: cfg.eval
    return cfg


def test_run_eval_rejects_non_lora_peft(tmp_path: Path) -> None:
    cfg = _make_cfg(peft_method="qlora")
    with pytest.raises(ValueError, match="lora"):
        run_eval(cfg, checkpoint=tmp_path, split="val")


def test_run_eval_rejects_test_split_when_data_test_none(tmp_path: Path) -> None:
    cfg = _make_cfg(has_test=False)
    with pytest.raises(ValueError, match=r"data\.test"):
        run_eval(cfg, checkpoint=tmp_path, split="test")


def test_run_eval_dispatches_dataset_via_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Format 'hf' must reach the @register('dataset', 'hf') factory."""
    cfg = _make_cfg(format_="hf")

    calls: list[tuple[str, str]] = []

    builder_mock = MagicMock(return_value=MagicMock(__len__=lambda self: 0, class_names=[]))

    def fake_lookup(kind: str, name: str) -> object:
        calls.append((kind, name))
        return builder_mock

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", fake_lookup)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert ("dataset", "hf") in calls
    assert result is fake_report
    # Verify builder was called with the expected shape.
    builder_mock.assert_called_once()
    call_args = builder_mock.call_args
    assert call_args.kwargs.get("pipeline") == "eval"
    assert call_args.kwargs.get("model_name") == "facebook/sam3.1"
    assert isinstance(call_args.args[0], dict)


def test_run_eval_accepts_prebuilt_val_dataset_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If val_dataset/model are provided, runner MUST NOT call lookup('dataset', …)
    or load_sam31."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    forbidden: list[str] = []

    def _forbidden_lookup(kind: str, name: str) -> object:
        forbidden.append(f"{kind}:{name}")
        return lambda *a, **kw: None

    def _forbidden_load(_m: object) -> object:
        forbidden.append("load_sam31")
        return None

    monkeypatch.setattr("custom_sam_peft.eval.runner.lookup", _forbidden_lookup)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", _forbidden_load)
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=3, n_predictions=3)
    captured: dict[str, object] = {}

    def _fake_evaluator_init(_cfg: object) -> object:
        ev = MagicMock()

        def _evaluate(
            model: object,
            dataset: object,
            *,
            return_per_example_iou: bool = False,
        ) -> object:
            captured["model"] = model
            captured["dataset"] = dataset
            captured["return_per_example_iou"] = return_per_example_iou
            if return_per_example_iou:
                return fake_report, [0.1, 0.5, 0.9]
            return fake_report

        ev.evaluate = _evaluate
        ev.evaluate_and_save = MagicMock(return_value=fake_report)
        return ev

    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", _fake_evaluator_init)

    fake_ds = MagicMock(__len__=lambda self: 3, class_names=["a"])
    fake_model = MagicMock()
    report, per_ex = run_eval(
        cfg,
        checkpoint=tmp_path,
        split="val",
        output_dir=tmp_path,
        val_dataset=fake_ds,
        model=fake_model,
        return_per_example_iou=True,
    )
    assert report is fake_report
    assert per_ex == [0.1, 0.5, 0.9]
    assert captured["dataset"] is fake_ds
    assert captured["model"] is fake_model
    assert captured["return_per_example_iou"] is True
    assert forbidden == []  # neither lookup nor load_sam31 should have been called


def test_run_eval_return_per_example_iou_default_false_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default kwarg path returns MetricsReport (not tuple) — existing CLI contract."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    _empty_ds = MagicMock(__len__=lambda self: 0, class_names=[])
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: _empty_ds,
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.0})
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    out = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert out is fake_report
    assert not isinstance(out, tuple)
