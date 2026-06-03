"""CLI integration for --override / --name / --output-dir (Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: tl\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def test_run_override_reaches_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import run_cmd

    seen: dict[str, Any] = {}

    def fake_orchestrate(cfg: Any, *a: Any, **k: Any) -> int:
        seen["epochs"] = cfg.train.epochs
        return 0

    monkeypatch.setattr(run_cmd, "_orchestrate", fake_orchestrate)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg), "--override", "train.epochs=7"])
    assert result.exit_code == 0, result.output
    assert seen["epochs"] == 7


def test_train_name_synthesizes_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.eval._artifacts import EvalArtifacts

    seen: dict[str, Any] = {}

    def fake_run_train(cfg: Any, **k: Any) -> EvalArtifacts:
        seen["name"] = cfg.run.name
        return EvalArtifacts(
            checkpoint_path=tmp_path / "adapter",
            peft_method="lora",
            run_dir=tmp_path,
            final_metrics=None,
        )

    monkeypatch.setattr(train_cmd, "run_train", fake_run_train)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg), "--name", "my-run"])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "my-run"


def test_train_name_conflict_raises(tmp_path: Path) -> None:
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(
        app, ["train", "--config", str(cfg), "--name", "foo", "--override", "run.name=bar"]
    )
    assert result.exit_code != 0
    assert "conflict" in result.output.lower()
