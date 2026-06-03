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
