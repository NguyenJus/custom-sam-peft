"""run_eval builds dataset via registry, loads adapter, calls Evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from esam3.eval.runner import run_eval


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

    monkeypatch.setattr("esam3.eval.runner.lookup", fake_lookup)
    monkeypatch.setattr("esam3.eval.runner.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("esam3.eval.runner.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock()
    monkeypatch.setattr(
        "esam3.eval.runner.Evaluator",
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
