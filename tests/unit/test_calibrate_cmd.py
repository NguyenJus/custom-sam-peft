"""Tests for src/custom_sam_peft/cli/calibrate_cmd.py — calibration probe CLI.

All `models.sam3.load_sam31`, `peft_adapters.lora.apply_lora`, and
`torch.cuda.max_memory_allocated` are monkeypatched — these tests run on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

_GB = 1024**3
runner = CliRunner()


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    peak: int = int(38 * _GB),
    gpu_name: str = "NVIDIA A100-SXM4-40GB",
    total: int = int(40 * _GB),
    sha: str = "deadbeef",
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = MagicMock(total_memory=total)
    props.name = gpu_name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: gpu_name)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peak)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd._run_probe",
        lambda: peak,
    )
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd._sam3_checkpoint_sha",
        lambda: sha,
    )


def test_calibrate_writes_cache_with_schema_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    assert cache.is_file()
    data = json.loads(cache.read_text())
    expected_keys = {
        "schema_version",
        "calibrated_at",
        "gpu_name",
        "gpu_total_memory_bytes",
        "sam3_checkpoint_sha",
        "torch_version",
        "custom_sam_peft_version",
        "activation_bytes_per_example",
        "peak_memory_bytes_at_probe",
    }
    assert expected_keys.issubset(data.keys())
    assert data["schema_version"] == 2
    assert "image_size" not in data
    assert data["sam3_checkpoint_sha"] == "deadbeef"


def test_calibrate_cache_fresh_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "calibrated_at": "2026-05-22T00:00:00+00:00",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "gpu_total_memory_bytes": int(40 * _GB),
                "sam3_checkpoint_sha": "deadbeef",
                "torch_version": "2.4.0",
                "custom_sam_peft_version": "0.0.0",
                "activation_bytes_per_example": 1,
                "peak_memory_bytes_at_probe": 2,
            }
        )
    )
    mtime_before = cache.stat().st_mtime
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    assert "cache fresh" in result.output
    assert cache.stat().st_mtime == mtime_before  # not rewritten


def test_calibrate_force_overwrites_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"stale": true}')
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads(cache.read_text())
    assert data.get("schema_version") == 2


def test_calibrate_non_cuda_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 2
    assert "CUDA" in result.output


def test_calibrate_negative_activation_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # peak much smaller than model+adapter+opt → negative raw activation.
    _patch_probe(monkeypatch, peak=10 * 1024**2)  # 10 MiB peak — tiny
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["activation_bytes_per_example"] == 0
    # The warning lands on stderr; CliRunner merges it into .output when mix_stderr=True (default).
    assert "negative" in result.output.lower() or "clamp" in result.output.lower()


def test_calibrate_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"prior": true}')
    # Force the os.replace step to fail; the prior cache must still exist.
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd.os.replace",
        lambda _src, _dst: (_ for _ in ()).throw(OSError("disk full")),
    )
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 6
    # The original file content survives the failed write.
    assert json.loads(cache.read_text()) == {"prior": True}
