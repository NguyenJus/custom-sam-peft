"""run_doctor reports environment info without importing heavy deps."""

from __future__ import annotations

import json
from pathlib import Path

from esam3.diagnostics import DoctorReport, run_doctor


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
