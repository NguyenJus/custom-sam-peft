"""Tests for the `csp predict` CLI option surface (CPU-only)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_predict_interactive_dispatches_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[dict] = []
    monkeypatch.setattr("custom_sam_peft.cli._interactive.require_tty", lambda: None)
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_predict_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(
        app,
        ["predict", "--interactive", "--images", "x", "--prompts", "y", "--output", "z"],
    )
    assert result.exit_code == 0, result.output
    assert len(called) == 1


def test_predict_interactive_non_tty_hard_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[dict] = []
    monkeypatch.setattr(
        "custom_sam_peft.cli._interactive.run_predict_interactive",
        lambda **kw: called.append(kw),
    )
    result = runner.invoke(
        app,
        ["predict", "--interactive", "--images", "x", "--prompts", "y", "--output", "z"],
    )
    assert result.exit_code != 0
    assert "tty" in result.output.lower()
    assert called == []
