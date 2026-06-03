"""-v on doctor/init/calibrate and -y on init are surfaced (Phase 2)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("cmd", ["doctor", "init", "calibrate"])
def test_verbose_in_help(cmd: str) -> None:
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "-v" in result.output


def test_init_short_yes_in_help() -> None:
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "-y" in result.output
