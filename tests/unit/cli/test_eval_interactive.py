"""Tests for the `eval -i` helper (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from custom_sam_peft.cli import _interactive as itv
from custom_sam_peft.config.loader import load_config


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text(
        textwrap.dedent("""
        run: {name: r, output_dir: ./runs, seed: 42}
        model:
          name: facebook/sam3.1
          local_dir: models/sam3.1
          checkpoint_file: c.pt
          dtype: bfloat16
        data:
          format: coco
          train: {annotations: t.json, images: t/}
          val: {annotations: v.json, images: v/}
        peft: {method: lora, r: 16, alpha: 32, dropout: 0.05}
        train:
          epochs: 1
          batch_size: 1
          grad_accum_steps: 8
          loss: {preset: natural, class_imbalance: balanced}
        """)
    )
    return p


def _lora_ckpt(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "facebook/sam3.1"})
    )
    return d


def _baseline_choice(prompt: str, choices: list[str], **k: object) -> str:
    """Route ask_choice for the baseline wizard flow."""
    mapping = {"Evaluate": "baseline", "Dataset": "coco", "Validation": "auto-split"}
    for kw, val in mapping.items():
        if kw in prompt:
            return val
    return choices[0]


def test_reuse_prints_command_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    cfg = _write_cfg(tmp_path)
    ckpt = _lora_ckpt(tmp_path)
    text_answers = iter([str(cfg), str(ckpt)])

    def _choice(prompt: str, choices: list[str], **k: object) -> str:
        return "reuse" if "Evaluate" in prompt else "val"

    monkeypatch.setattr(itv, "ask_choice", _choice)
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    before = set(tmp_path.iterdir())
    itv.run_eval_interactive(output=None, force=False)
    out = capsys.readouterr().out
    assert f"custom-sam-peft eval --config {cfg} --checkpoint {ckpt} --split val" in out
    assert set(tmp_path.iterdir()) == before  # nothing new written


def test_reuse_peek_prints_method(tmp_path, monkeypatch, capsys) -> None:
    cfg = _write_cfg(tmp_path)
    ckpt = _lora_ckpt(tmp_path)
    (ckpt / "custom_sam_peft_qlora.json").write_text("{}")  # make it qlora
    text_answers = iter([str(cfg), str(ckpt)])

    def _choice(prompt: str, choices: list[str], **k: object) -> str:
        return "reuse" if "Evaluate" in prompt else "val"

    monkeypatch.setattr(itv, "ask_choice", _choice)
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    itv.run_eval_interactive(output=None, force=False)
    out = capsys.readouterr().out
    assert "QLoRA" in out
    assert "facebook/sam3.1" in out


def test_baseline_emits_reloadable_config(tmp_path, monkeypatch, capsys) -> None:
    out_cfg = tmp_path / "baseline.yaml"
    monkeypatch.setattr(itv, "ask_choice", _baseline_choice)
    text_answers = iter(["ann.json", "imgs/", "0.1", ""])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    itv.run_eval_interactive(output=out_cfg, force=False)
    assert out_cfg.is_file()
    cfg = load_config(out_cfg)
    assert cfg.data.val_split is not None
    assert cfg.run.name == "baseline-eval"
    out = capsys.readouterr().out
    assert f"custom-sam-peft eval --config {out_cfg} --split val" in out
    assert "--checkpoint" not in out


def test_output_exists_without_force(tmp_path, monkeypatch) -> None:
    out_cfg = tmp_path / "baseline.yaml"
    out_cfg.write_text("existing\n")
    monkeypatch.setattr(itv, "ask_choice", _baseline_choice)
    text_answers = iter(["ann.json", "imgs/", "0.1", ""])
    monkeypatch.setattr(itv, "ask_text", lambda *a, **k: next(text_answers))
    monkeypatch.setattr(itv, "ask_confirm", lambda *a, **k: True)
    import typer

    with pytest.raises(typer.BadParameter, match="refusing to overwrite"):
        itv.run_eval_interactive(output=out_cfg, force=False)
    assert out_cfg.read_text() == "existing\n"


def test_ctrl_c_writes_nothing(tmp_path, monkeypatch) -> None:
    def _boom(*a: object, **k: object) -> str:
        raise KeyboardInterrupt

    def _choice(prompt: str, choices: list[str], **k: object) -> str:
        return "baseline" if "Evaluate" in prompt else choices[0]

    monkeypatch.setattr(itv, "ask_choice", _choice)
    monkeypatch.setattr(itv, "ask_text", _boom)
    before = set(tmp_path.iterdir())
    with pytest.raises(KeyboardInterrupt):
        itv.run_eval_interactive(output=tmp_path / "x.yaml", force=False)
    assert set(tmp_path.iterdir()) == before
