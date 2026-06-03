"""CLI --dry-run preview for train/run/eval (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: dr\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def test_train_dry_run_skips_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import train_cmd

    called = {"run": False}
    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **k: called.__setitem__("run", True))
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["run"] is False


def test_run_dry_run_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import run_cmd

    called = {"orch": False}
    monkeypatch.setattr(
        run_cmd, "_orchestrate", lambda *a, **k: called.__setitem__("orch", True) or 0
    )
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["orch"] is False


def test_eval_dry_run_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import eval_cmd

    called = {"eval": False}
    monkeypatch.setattr(eval_cmd, "run_eval", lambda *a, **k: called.__setitem__("eval", True))
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["eval"] is False
