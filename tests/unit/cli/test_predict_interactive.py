"""Tests for the `predict -i` helper (CPU-only; prompts monkeypatched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from custom_sam_peft.cli import _interactive as itv


def _lora_ckpt(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "facebook/sam3.1"})
    )
    return d


def _drive(
    monkeypatch,
    *,
    checkpoint="",
    channels="3",
    semantics="rgb",
    merge=True,
    threshold="0.3",
    save_masks="rle",
    visualize=False,
    images="imgs/",
    prompts="cat,dog",
    output="out/",
):
    text_iter = iter([checkpoint, channels, threshold, images, prompts, output])

    def _ask_choice(prompt, choices, **k):
        if "semantics" in prompt:
            return semantics
        if "Mask output" in prompt:
            return save_masks
        return choices[0]

    def _ask_text(prompt, **k):
        return next(text_iter)

    monkeypatch.setattr(itv, "ask_choice", _ask_choice)
    monkeypatch.setattr(itv, "ask_text", _ask_text)
    monkeypatch.setattr(
        itv, "ask_confirm", lambda prompt, **k: merge if "Merge" in prompt else visualize
    )


def test_command_assembly_baseline_rgb(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", channels="3", semantics="rgb")
    itv.run_predict_interactive(force=False)
    out = capsys.readouterr().out
    assert "--images imgs/" in out
    assert "--prompts cat,dog" in out
    assert "--output out/" in out
    assert "--checkpoint" not in out
    assert "--merge-adapter" not in out
    assert "--config" not in out
    assert "--visualize" not in out
    assert not (tmp_path / "predict-config.yaml").exists()


def test_command_assembly_with_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    ckpt = _lora_ckpt(tmp_path)
    _drive(monkeypatch, checkpoint=str(ckpt), channels="3", semantics="rgb", merge=True)
    itv.run_predict_interactive(force=False)
    out = capsys.readouterr().out
    assert f"--checkpoint {ckpt}" in out
    assert "--merge-adapter" in out
    assert "LoRA" in out  # peek output


def test_thin_config_emitted_for_non_rgb(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", channels="4", semantics="rgba")
    itv.run_predict_interactive(force=False)
    thin = tmp_path / "predict-config.yaml"
    assert thin.is_file()
    raw = yaml.safe_load(thin.read_text())
    assert raw["data"]["channels"] == 4
    assert raw["data"]["channel_semantics"] == "rgba"
    assert raw["model"]["name"] == "facebook/sam3.1"
    out = capsys.readouterr().out
    assert "--config" in out and "predict-config.yaml" in out


def test_thin_config_not_emitted_for_rgb(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", channels="3", semantics="rgb")
    itv.run_predict_interactive(force=False)
    assert not (tmp_path / "predict-config.yaml").exists()
    assert "--config" not in capsys.readouterr().out


def test_visualize_flag_emitted_when_yes(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _drive(monkeypatch, checkpoint="", visualize=True)
    itv.run_predict_interactive(force=False)
    assert "--visualize" in capsys.readouterr().out


def test_thin_config_overwrite_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "predict-config.yaml").write_text("existing\n")
    _drive(monkeypatch, checkpoint="", channels="4", semantics="rgba")
    import typer

    with pytest.raises(typer.BadParameter, match="refusing to overwrite"):
        itv.run_predict_interactive(force=False)
    assert (tmp_path / "predict-config.yaml").read_text() == "existing\n"
