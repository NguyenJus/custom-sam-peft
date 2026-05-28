"""Tests for the `csp eval` CLI option surface (CPU-only)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_eval_checkpoint_optional_invokes_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --checkpoint must NOT be a CLI usage error; run_eval is called
    with checkpoint=None (baseline)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("placeholder")
    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.load_config", lambda p: MagicMock())
    captured: dict[str, object] = {}

    def _fake_run_eval(cfg, **kw):
        captured.update(kw)
        report = MagicMock()
        report.overall = {}
        return report

    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.run_eval", _fake_run_eval)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "val"])
    assert result.exit_code == 0, result.output
    assert captured["checkpoint"] is None


def test_eval_config_required_non_interactive() -> None:
    result = runner.invoke(app, ["eval", "--split", "val"])
    assert result.exit_code != 0
    assert "config" in result.output.lower()


def test_eval_interactive_dispatches_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    # CliRunner replaces sys.stdin; patch require_tty directly to simulate TTY.
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.require_tty",
        lambda: None,
    )
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_eval_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(app, ["eval", "--interactive"])
    assert result.exit_code == 0, result.output
    assert len(called) == 1


def test_eval_interactive_non_tty_hard_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # CliRunner replaces sys.stdin with a non-TTY; require_tty should fire.
    # Do NOT patch require_tty here — the real one should raise BadParameter.
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_eval_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(app, ["eval", "--interactive"])
    assert result.exit_code != 0
    assert "tty" in result.output.lower()
    assert called == []


def test_eval_export_requires_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--export without --checkpoint must exit non-zero with a message mentioning
    export and checkpoint."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("placeholder")
    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.load_config", lambda p: MagicMock())

    def _fake_run_eval(cfg, **kw):
        report = MagicMock()
        report.overall = {}
        return report

    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.run_eval", _fake_run_eval)
    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.run_export", lambda *a, **kw: None)

    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "val", "--export"])
    assert result.exit_code != 0
    output_lower = result.output.lower()
    assert "export" in output_lower or "checkpoint" in output_lower
