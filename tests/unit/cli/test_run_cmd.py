"""Tests for the `csp run` CLI option surface (CPU-only)."""

from __future__ import annotations

from pathlib import Path

import pytest

import custom_sam_peft.cli.run_cmd as run_cmd


def test_run_folds_visualize_into_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run() folds --visualize/--no-visualize into cfg.eval.visualize before
    calling _orchestrate. close_out (inside run_training) is the consumer of
    cfg.eval.visualize; _orchestrate no longer threads it into a separate
    run_eval call (run_eval was removed in the close_out refactor)."""
    captured: dict[str, object] = {}

    def _fake_orchestrate(cfg, resume, mode, *, config_path):  # type: ignore[no-untyped-def]
        captured["visualize"] = cfg.eval.visualize

    monkeypatch.setattr(run_cmd, "_orchestrate", _fake_orchestrate)

    from typer.testing import CliRunner

    from custom_sam_peft.cli.main import app

    runner = CliRunner()

    # Write a minimal valid config so run() doesn't trigger auto-init.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
run: {{name: t, output_dir: {tmp_path / "runs"}, seed: 0}}
data:
  format: coco
  train: {{annotations: t.json, images: t/}}
  val: {{annotations: v.json, images: v/}}
peft: {{method: lora}}
train: {{epochs: 1}}
export: {{merge: false}}
"""
    )

    # --no-visualize → cfg.eval.visualize is False
    result = runner.invoke(app, ["run", "--config", str(cfg_path), "--no-visualize"])
    assert result.exit_code == 0, result.output
    assert captured.get("visualize") is False

    # --visualize (default True) → cfg.eval.visualize is True
    result = runner.invoke(app, ["run", "--config", str(cfg_path), "--visualize"])
    assert result.exit_code == 0, result.output
    assert captured.get("visualize") is True
