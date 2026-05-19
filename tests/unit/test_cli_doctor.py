"""esam3 doctor formats run_doctor output."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from esam3.cli.main import app

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


runner = CliRunner()


def test_doctor_table_output_includes_torch() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "torch" in text.lower()
    assert "python" in text.lower()


def test_doctor_json_round_trips() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "torch_version" in blob
    assert "optional_deps" in blob
    assert "core_versions" in blob
