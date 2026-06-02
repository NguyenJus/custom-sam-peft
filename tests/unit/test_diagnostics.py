"""run_doctor reports environment info without importing heavy deps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_sam_peft.diagnostics import DoctorReport, run_doctor


def test_report_has_expected_fields() -> None:
    r = run_doctor()
    assert isinstance(r, DoctorReport)
    assert r.python_version.startswith(("3.12", "3.13"))
    assert r.torch_version
    assert "bitsandbytes" in r.optional_deps
    assert "wandb" in r.optional_deps
    assert "tensorboard" in r.optional_deps
    assert "peft" in r.core_versions
    assert "transformers" in r.core_versions
    assert r.sam3_weights.path.name == "sam3.1_multiplex.pt"


def test_weights_path_override_existing_file(tmp_path: Path) -> None:
    fake = tmp_path / "weights.pt"
    fake.write_bytes(b"x" * 1024)
    r = run_doctor(weights_path=fake)
    assert r.sam3_weights.exists is True
    assert r.sam3_weights.size_bytes == 1024


def test_weights_path_override_missing_file(tmp_path: Path) -> None:
    r = run_doctor(weights_path=tmp_path / "absent.pt")
    assert r.sam3_weights.exists is False
    assert r.sam3_weights.size_bytes is None


def test_report_is_json_serializable() -> None:
    import dataclasses

    r = run_doctor()
    blob = json.dumps(dataclasses.asdict(r), default=str)
    assert "torch_version" in blob


# ---------------------------------------------------------------------------
# spec/hf-utils — hf_auth field
# ---------------------------------------------------------------------------


def test_run_doctor_reports_env_token_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    r = run_doctor()
    assert r.hf_auth.token_source == "env"
    assert r.hf_auth.has_token is True


def test_run_doctor_reports_cache_token_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "custom_sam_peft.diagnostics.huggingface_hub.get_token", lambda: "cache-tok"
    )
    r = run_doctor()
    assert r.hf_auth.token_source == "cache"
    assert r.hf_auth.has_token is True


def test_run_doctor_reports_no_token_and_appends_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("custom_sam_peft.diagnostics.huggingface_hub.get_token", lambda: None)
    r = run_doctor()
    assert r.hf_auth.token_source == "none"
    assert r.hf_auth.has_token is False
    assert any("no HuggingFace token" in issue for issue in r.issues), r.issues
