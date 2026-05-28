"""Tests for the shared interactive module (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from custom_sam_peft.cli import _interactive as itv


def test_prompt_primitives_importable() -> None:
    assert callable(itv.ask_text)
    assert callable(itv.ask_choice)
    assert callable(itv.ask_confirm)
    assert callable(itv.run_wizard)
    assert hasattr(itv, "WizardStep")
    assert hasattr(itv, "Ctx")


def test_ask_choice_reasks_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["bogus", "coco"])
    monkeypatch.setattr(itv.typer, "prompt", lambda *a, **k: next(answers))
    out: list[str] = []
    monkeypatch.setattr(itv.typer, "echo", lambda msg="", *a, **k: out.append(str(msg)))
    assert itv.ask_choice("Format?", ["coco", "hf"], default="coco") == "coco"
    assert any("choose one of" in line for line in out)


def test_deep_merge_nested() -> None:
    dst = {"data": {"format": "coco"}}
    itv._deep_merge(dst, {"data": {"val_split": {"fraction": 0.1}}})
    assert dst == {"data": {"format": "coco", "val_split": {"fraction": 0.1}}}


def test_shared_steps_return_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv, "ask_choice", lambda *a, **k: "coco")
    answers = iter(["ann.json", "imgs/"])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(answers))
    ctx = itv.Ctx(answers={}, cuda_available=False)
    frag = itv._ask_dataset_source(ctx)
    assert frag == {
        "data": {"format": "coco", "train": {"annotations": "ann.json", "images": "imgs/"}}
    }


def test_require_tty_non_tty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv.sys.stdin, "isatty", lambda: False)
    with pytest.raises(typer.BadParameter, match="TTY"):
        itv.require_tty()


def test_require_tty_tty_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itv.sys.stdin, "isatty", lambda: True)
    assert itv.require_tty() is None


def test_validate_checkpoint_dir(tmp_path: Path) -> None:
    good = tmp_path / "ckpt"
    good.mkdir()
    (good / "adapter_config.json").write_text("{}")
    assert itv.validate_checkpoint_dir(str(good)) is None
    bad = tmp_path / "empty"
    bad.mkdir()
    assert itv.validate_checkpoint_dir(str(bad)) is not None
    assert itv.validate_checkpoint_dir(str(tmp_path / "missing")) is not None


def test_validate_config_with_eval_split(tmp_path: Path) -> None:
    import textwrap

    def _write(body: str) -> Path:
        p = tmp_path / f"{abs(hash(body))}.yaml"
        p.write_text(textwrap.dedent(body))
        return p

    base = """
    run: {name: r}
    model: {name: facebook/sam3.1, local_dir: models/sam3.1, checkpoint_file: c.pt}
    data:
      format: coco
      train: {annotations: t.json, images: t/}
      VAL_BLOCK
    peft: {method: lora, r: 16, alpha: 32, dropout: 0.05}
    train:
      epochs: 1
      loss: {preset: natural, class_imbalance: balanced}
    """
    with_val = _write(base.replace("VAL_BLOCK", "val: {annotations: v.json, images: v/}"))
    assert itv.validate_config_with_eval_split(str(with_val)) is None
    no_val = _write(base.replace("      VAL_BLOCK\n", ""))
    assert itv.validate_config_with_eval_split(str(no_val)) is not None
    assert itv.validate_config_with_eval_split(str(tmp_path / "nope.yaml")) is not None


def test_peek_adapter_lora(tmp_path: Path) -> None:
    import json

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "facebook/sam3.1"})
    )
    pretty, base = itv.peek_adapter(tmp_path)
    assert pretty == "LoRA"
    assert base == "facebook/sam3.1"


def test_peek_adapter_qlora(tmp_path: Path) -> None:
    import json

    (tmp_path / "adapter_config.json").write_text(json.dumps({}))
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    pretty, base = itv.peek_adapter(tmp_path)
    assert pretty == "QLoRA"
    assert base is None
