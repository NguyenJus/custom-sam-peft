"""csp doctor --config task-aware output — semantic branch tests.

Mirrors test_cli_doctor_config.py. Covers:
  - Task row shown in table output for semantic config
  - Resolved semantic losses table shown (and instance table suppressed)
  - Head indicator line shown
  - --json carries task + semantic_loss, no loss key
  - Regression: instance config still renders instance losses + task
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from custom_sam_peft.cli.doctor_cmd import doctor
from custom_sam_peft.config.schema import TrainConfig

app = typer.Typer()
app.command()(doctor)

runner = CliRunner()


def _make_semantic_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid semantic (mask_png) config YAML."""
    p = tmp_path / "semantic_config.yaml"
    p.write_text(
        """
run:
  name: sem-test
  output_dir: ./runs

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt

task: semantic

data:
  format: mask_png
  train:
    images: tests/fixtures/tiny_coco/images
    annotations: tests/fixtures/tiny_mask_png/labels
  semantic:
    class_map: tests/fixtures/tiny_mask_png/class_map.json
    ignore_index: 255
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


def _make_instance_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid instance (coco) config YAML."""
    p = tmp_path / "instance_config.yaml"
    p.write_text(
        """
run:
  name: inst-test
  output_dir: ./runs

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


@pytest.fixture
def semantic_config_yaml(tmp_path: Path) -> Path:
    return _make_semantic_yaml(tmp_path)


@pytest.fixture
def instance_config_yaml(tmp_path: Path) -> Path:
    return _make_instance_yaml(tmp_path)


# ---------------------------------------------------------------------------
# Verify the YAML parses to a valid semantic TrainConfig before running CLI
# ---------------------------------------------------------------------------


def test_semantic_yaml_parses_correctly(semantic_config_yaml: Path) -> None:
    """Sanity: the fixture YAML produces a valid semantic TrainConfig."""
    from custom_sam_peft.config.loader import load_config

    # load_config may reach out to the filesystem for class_map, so patch it
    with patch(
        "custom_sam_peft.config.loader.load_config",
        return_value=TrainConfig.model_validate(
            {
                "run": {"name": "sem-test", "output_dir": "./runs"},
                "model": {
                    "name": "facebook/sam3.1",
                    "local_dir": "models/sam3.1",
                    "checkpoint_file": "sam3.1_multiplex.pt",
                },
                "task": "semantic",
                "data": {
                    "format": "mask_png",
                    "train": {
                        "images": "tests/fixtures/tiny_coco/images",
                        "annotations": "tests/fixtures/tiny_mask_png/labels",
                    },
                    "semantic": {
                        "class_map": "tests/fixtures/tiny_mask_png/class_map.json",
                    },
                    "limit": {"train": 1, "val": 1, "seed": 0, "strategy": "random"},
                },
                "peft": {"method": "lora", "r": 4, "alpha": 8},
                "train": {"epochs": 1},
            }
        ),
    ):
        cfg = load_config(semantic_config_yaml)
    assert cfg.task == "semantic"
    assert cfg.data.semantic is not None


# ---------------------------------------------------------------------------
# Helper: invoke doctor with a semantic config, patching load_config +
# dataset build so we never touch the real filesystem or torch.
# ---------------------------------------------------------------------------


def _semantic_cfg() -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "run": {"name": "sem-test", "output_dir": "./runs"},
            "model": {
                "name": "facebook/sam3.1",
                "local_dir": "models/sam3.1",
                "checkpoint_file": "sam3.1_multiplex.pt",
            },
            "task": "semantic",
            "data": {
                "format": "mask_png",
                "train": {
                    "images": "tests/fixtures/tiny_coco/images",
                    "annotations": "tests/fixtures/tiny_mask_png/labels",
                },
                "semantic": {
                    "class_map": "tests/fixtures/tiny_mask_png/class_map.json",
                },
                "limit": {"train": 1, "val": 1, "seed": 0, "strategy": "random"},
            },
            "peft": {"method": "lora", "r": 4, "alpha": 8},
            "train": {"epochs": 1},
        }
    )


def _instance_cfg() -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "run": {"name": "inst-test", "output_dir": "./runs"},
            "model": {
                "name": "facebook/sam3.1",
                "local_dir": "models/sam3.1",
                "checkpoint_file": "sam3.1_multiplex.pt",
            },
            "data": {
                "format": "coco",
                "train": {
                    "images": "tests/fixtures/tiny_coco/images",
                    "annotations": "tests/fixtures/tiny_coco/annotations.json",
                },
                "limit": {"train": 1, "val": 1, "seed": 0, "strategy": "random"},
            },
            "peft": {"method": "lora", "r": 4, "alpha": 8},
            "train": {"epochs": 1},
        }
    )


# ---------------------------------------------------------------------------
# Test 1: table output for semantic config
# ---------------------------------------------------------------------------


def test_doctor_semantic_renders_task_and_semantic_loss_table(
    semantic_config_yaml: Path,
) -> None:
    """Table output: Task row, Resolved semantic losses table, Head line, NO instance Resolved losses."""
    sem_cfg = _semantic_cfg()

    with (
        patch("custom_sam_peft.cli.doctor_cmd.load_config", return_value=sem_cfg),
        patch(
            "custom_sam_peft.diagnostics.run_doctor",
            return_value=_minimal_report(),
        ),
    ):
        result = runner.invoke(app, ["--config", str(semantic_config_yaml)])

    assert result.exit_code == 0, result.output

    # Task row present
    assert "semantic" in result.output

    # Resolved semantic losses table present
    assert "Resolved semantic losses" in result.output

    # At least one resolved field visible
    assert "sem_family" in result.output
    assert "w_ce" in result.output
    assert "w_region" in result.output

    # Head indicator line present
    assert "marginalize" in result.output or "semantic_seg" in result.output

    # Instance "Resolved losses" table SUPPRESSED
    assert "Resolved losses" not in result.output.replace("Resolved semantic losses", "")


def test_doctor_semantic_head_indicator_marginalize(
    semantic_config_yaml: Path,
) -> None:
    """Head indicator shows 'marginalization (head-free)' for source=marginalize."""
    from custom_sam_peft.config.schema import SemanticLossConfig

    sem_cfg = _semantic_cfg()
    # Default source is "marginalize"
    assert sem_cfg.train.semantic_loss.source == "marginalize"

    with (
        patch("custom_sam_peft.cli.doctor_cmd.load_config", return_value=sem_cfg),
        patch("custom_sam_peft.diagnostics.run_doctor", return_value=_minimal_report()),
    ):
        result = runner.invoke(app, ["--config", str(semantic_config_yaml)])

    assert result.exit_code == 0, result.output
    assert "marginalization (head-free)" in result.output


# ---------------------------------------------------------------------------
# Test 2: --json output for semantic config
# ---------------------------------------------------------------------------


def test_doctor_json_carries_task_and_semantic_loss(
    semantic_config_yaml: Path,
) -> None:
    """--json: resolved_config has task=semantic, semantic_loss key, NO loss key."""
    sem_cfg = _semantic_cfg()

    with (
        patch("custom_sam_peft.cli.doctor_cmd.load_config", return_value=sem_cfg),
        patch("custom_sam_peft.diagnostics.run_doctor", return_value=_minimal_report()),
    ):
        result = runner.invoke(app, ["--config", str(semantic_config_yaml), "--json"])

    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)

    # Top-level task in resolved_config
    assert "resolved_config" in blob
    rc = blob["resolved_config"]
    assert rc["task"] == "semantic"

    # semantic_loss present with expected structure
    assert "semantic_loss" in rc
    sl = rc["semantic_loss"]
    assert "preset" in sl
    assert "class_imbalance" in sl
    assert "resolved" in sl
    assert "sem_family" in sl["resolved"]

    # No loss key under semantic task
    assert "loss" not in rc


# ---------------------------------------------------------------------------
# Test 3: regression — instance config still renders instance Resolved losses
# ---------------------------------------------------------------------------


def test_doctor_instance_still_renders_resolved_losses_and_task(
    instance_config_yaml: Path,
) -> None:
    """Regression: instance config shows 'Resolved losses' table + task=instance in --json."""
    inst_cfg = _instance_cfg()

    with (
        patch("custom_sam_peft.cli.doctor_cmd.load_config", return_value=inst_cfg),
        patch("custom_sam_peft.diagnostics.run_doctor", return_value=_minimal_report()),
    ):
        table_result = runner.invoke(app, ["--config", str(instance_config_yaml)])
        json_result = runner.invoke(
            app, ["--config", str(instance_config_yaml), "--json"]
        )

    # Table: instance Resolved losses present, semantic table absent
    assert table_result.exit_code == 0, table_result.output
    assert "Resolved losses" in table_result.output
    assert "Resolved semantic losses" not in table_result.output

    # JSON: task=instance, loss key present, no semantic_loss key
    assert json_result.exit_code == 0, json_result.output
    blob = json.loads(json_result.output)
    rc = blob["resolved_config"]
    assert rc["task"] == "instance"
    assert "loss" in rc
    assert "semantic_loss" not in rc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_report():
    """Build a minimal DoctorReport with no gpus and default fields."""
    from custom_sam_peft.diagnostics import (
        DoctorReport,
        HuggingFaceAuthInfo,
        WeightsInfo,
    )

    return DoctorReport(
        python_version="3.12.0",
        platform="linux",
        torch_version="2.0.0",
        cuda_build=None,
        cuda_available=False,
        gpus=[],
        optional_deps={},
        core_versions={"custom_sam_peft": "0.1.0"},
        sam3_weights=WeightsInfo(
            path=Path("models/sam3.1/sam3.1_multiplex.pt"),
            exists=False,
            size_bytes=None,
        ),
        hf_auth=HuggingFaceAuthInfo(token_source="none", has_token=False),
        dataset=None,
        issues=[],
    )
