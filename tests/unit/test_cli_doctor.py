"""esam3 doctor formats run_doctor output."""

from __future__ import annotations

import json
import re

import pytest
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


# ---------------------------------------------------------------------------
# spec/hf-utils — HuggingFace auth rendering
# ---------------------------------------------------------------------------


def test_doctor_table_includes_hf_auth_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.diagnostics.huggingface_hub.get_token", lambda: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "HuggingFace auth" in text or "token source" in text


def test_doctor_json_reports_env_token_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert blob["hf_auth"]["token_source"] == "env"
    assert blob["hf_auth"]["has_token"] is True


def test_doctor_json_reports_cache_token_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.diagnostics.huggingface_hub.get_token", lambda: "cache-tok")
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert blob["hf_auth"]["token_source"] == "cache"


def test_doctor_json_reports_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.diagnostics.huggingface_hub.get_token", lambda: None)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert blob["hf_auth"]["token_source"] == "none"
    assert blob["hf_auth"]["has_token"] is False
    assert any("no HuggingFace token" in i for i in blob["issues"]), blob["issues"]


def test_doctor_json_round_trips_hf_auth_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--json` always includes the hf_auth field with both sub-keys."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.diagnostics.huggingface_hub.get_token", lambda: None)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "hf_auth" in blob
    assert set(blob["hf_auth"].keys()) == {"token_source", "has_token"}
