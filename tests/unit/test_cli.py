"""Tests for the Typer CLI skeleton."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from esam3.cli.main import app

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s: str) -> str:
    """Strip ANSI escape sequences so substring asserts are terminal-independent."""
    return _ANSI.sub("", s)


runner = CliRunner()


def test_root_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in _plain(result.stdout)
    assert "eval" in _plain(result.stdout)
    assert "export" in _plain(result.stdout)
    assert "init" in _plain(result.stdout)
    assert "doctor" in _plain(result.stdout)


def test_train_help_exits_zero() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in _plain(result.stdout)


def test_eval_help_exits_zero() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--config" in _plain(result.stdout)
    assert "--checkpoint" in _plain(result.stdout)


def test_export_help_exits_zero() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint" in _plain(result.stdout)


def test_init_help_exits_zero() -> None:
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


def test_doctor_runs_and_prints_not_implemented() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "not yet implemented" in _plain(result.stdout).lower()


def test_train_with_valid_config_prints_not_implemented(tmp_path: object) -> None:
    # Depends on Task 16 example configs being committed.
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "configs" / "examples" / "coco_text_lora.yaml"
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "not yet implemented" in _plain(result.stdout).lower()
