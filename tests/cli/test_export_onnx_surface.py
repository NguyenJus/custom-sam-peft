"""`csp export` ONNX-path flags: validation + dispatch (spec §4.1, §4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _ckpt_with_config(tmp_path: Path) -> Path:
    """A run-dir-shaped tree: {run_dir}/config.yaml + {run_dir}/adapter/."""
    run_dir = tmp_path / "run"
    ckpt = run_dir / "adapter"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (run_dir / "config.yaml").write_text(
        "run:\n  name: ex\ndata:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\ntrain:\n  epochs: 1\n"
    )
    return ckpt


def _patch_onnx(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Stub run_export_onnx so dispatch tests never trace a real model."""
    import custom_sam_peft.export.onnx as onnx_mod

    def _fake(cfg: Any, checkpoint: Any, **kw: Any) -> Any:
        captured.update(kw)
        return kw["output"]

    monkeypatch.setattr(onnx_mod, "run_export_onnx", _fake)


def _patch_pytorch(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Stub run_export so the pytorch dispatch never loads a real model."""
    import custom_sam_peft.cli.export_cmd as export_cmd

    def _fake(cfg: Any, checkpoint: Any, *, merge: bool = False, output: Any = None) -> Any:
        captured["called"] = True
        captured["merge"] = merge
        return output

    monkeypatch.setattr(export_cmd, "run_export", _fake)


def test_invalid_to_rejected(tmp_path: Path) -> None:
    """--to outside {pytorch,onnx} is a BadParameter."""
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(
        app,
        ["export", "--checkpoint", str(ckpt), "--output", str(tmp_path / "o"), "--to", "tflite"],
    )
    assert result.exit_code != 0
    assert "--to" in result.output


def test_invalid_include_rejected(tmp_path: Path) -> None:
    """--include outside {encoder,decoder,all} is a BadParameter."""
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(tmp_path / "o"),
            "--to",
            "onnx",
            "--include",
            "vision",
        ],
    )
    assert result.exit_code != 0
    assert "--include" in result.output


def test_opset_floor_rejected(tmp_path: Path) -> None:
    """--opset below 17 is a BadParameter naming the floor."""
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(tmp_path / "o"),
            "--to",
            "onnx",
            "--opset",
            "16",
        ],
    )
    assert result.exit_code != 0
    assert "--opset floor is 17." in result.output


def test_quantize_reserved_rejected(tmp_path: Path) -> None:
    """--quantize non-none raises the reserved-not-implemented BadParameter (§9)."""
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(tmp_path / "o"),
            "--to",
            "onnx",
            "--quantize",
            "int8-dynamic",
        ],
    )
    assert result.exit_code != 0
    assert "reserved" in result.output.lower()


def test_pytorch_path_ignores_onnx_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--to pytorch with an ONNX-only flag set non-default: INFO log, no error."""
    ckpt = _ckpt_with_config(tmp_path)
    captured: dict[str, Any] = {}
    _patch_pytorch(monkeypatch, captured)
    result = runner.invoke(
        app,
        ["export", "--checkpoint", str(ckpt), "--output", str(tmp_path / "o"), "--fp16"],
    )
    assert result.exit_code == 0, result.output
    assert captured["called"] is True


def test_dispatch_onnx_calls_run_export_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--to onnx routes to run_export_onnx with the ONNX flags and prints the bundle line."""
    ckpt = _ckpt_with_config(tmp_path)
    captured: dict[str, Any] = {}
    _patch_onnx(monkeypatch, captured)
    out = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "export",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out),
            "--to",
            "onnx",
            "--fp16",
            "--include",
            "encoder",
            "--no-dynamic-axes",
            "--check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["output"] == out
    assert captured["opset"] == 17
    assert captured["fp16"] is True
    assert captured["include"] == "encoder"
    assert captured["dynamic_axes"] is False
    assert captured["check"] is True
    assert "onnx bundle" in result.output
