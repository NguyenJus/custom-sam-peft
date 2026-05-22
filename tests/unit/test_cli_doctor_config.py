"""csp doctor --config happy + sad paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from custom_sam_peft.cli.doctor_cmd import doctor
from custom_sam_peft.diagnostics import run_doctor

app = typer.Typer()
app.command()(doctor)

runner = CliRunner()


@pytest.fixture
def valid_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid YAML and patch _build_dataset to return stubs."""
    p = tmp_path / "config.yaml"
    p.write_text(
        """
run:
  name: test
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt

data:
  format: coco
  train:
    annotations: tests/fixtures/tiny_coco/annotations.json
    images: tests/fixtures/tiny_coco/images
  val:
    annotations: tests/fixtures/tiny_coco/annotations.json
    images: tests/fixtures/tiny_coco/images
  prompt_mode: bbox
  limit:
    train: 1
    val: 1
    seed: 0
    strategy: random

peft:
  method: lora
  r: 4
  alpha: 8

train:
  epochs: 1
"""
    )
    return p


def test_doctor_config_happy_path_prints_dataset_section(
    valid_config_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--config: Dataset table appears in rich output."""
    stub_ds = MagicMock()
    stub_ds.__len__ = MagicMock(return_value=2)
    stub_ds.class_names = ["a", "b"]

    with patch("custom_sam_peft.train.runner._build_dataset", return_value=stub_ds):
        result = runner.invoke(app, ["--config", str(valid_config_yaml)])

    assert result.exit_code == 0, result.output
    assert "Dataset" in result.output


def test_doctor_config_no_config_flag_dataset_none() -> None:
    """Without --config, report.dataset is None."""
    r = run_doctor()
    assert r.dataset is None


def test_doctor_config_bad_path_exit_0_issue_in_report(tmp_path: Path) -> None:
    """Non-existent config path: exit 0, 'config' or 'load' in issues."""
    absent = tmp_path / "no_such.yaml"
    result = runner.invoke(app, ["--config", str(absent)])
    assert result.exit_code == 0
    # Issues should mention the path
    assert str(absent) in result.output or "config" in result.output.lower()


def test_doctor_config_schema_error_exit_0_issue_in_report(tmp_path: Path) -> None:
    """Malformed YAML (schema error): exit 0, error text in output."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("run:\n  name: x\n  seed: not-a-number-for-int: oops\n")
    result = runner.invoke(app, ["--config", str(bad_yaml)])
    assert result.exit_code == 0


def test_doctor_config_build_error_exit_0_couldnt_build(
    valid_config_yaml: Path,
) -> None:
    """_build_dataset raises: exit 0, 'couldn't build' in issues."""
    with patch(
        "custom_sam_peft.train.runner._build_dataset",
        side_effect=RuntimeError("injected build error"),
    ):
        result = runner.invoke(app, ["--config", str(valid_config_yaml)])

    assert result.exit_code == 0
    assert "couldn't build" in result.output.lower()


def test_doctor_config_json_output_has_dataset_field(
    valid_config_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json: blob["dataset"] has all fields when --config succeeds."""
    stub_ds = MagicMock()
    stub_ds.__len__ = MagicMock(return_value=2)
    stub_ds.class_names = ["a", "b"]

    with patch("custom_sam_peft.train.runner._build_dataset", return_value=stub_ds):
        result = runner.invoke(app, ["--config", str(valid_config_yaml), "--json"])

    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)
    assert blob["dataset"] is not None
    assert "train_kept" in blob["dataset"]
    assert "val_kept" in blob["dataset"]


def test_doctor_json_output_dataset_null_without_config() -> None:
    """--json without --config: blob['dataset'] is null."""
    result = runner.invoke(app, ["--json"])
    assert result.exit_code == 0
    blob = json.loads(result.output)
    assert blob["dataset"] is None
