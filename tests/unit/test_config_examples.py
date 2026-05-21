"""Every YAML under configs/examples/ must validate against TrainConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.config.loader import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "examples"


@pytest.mark.parametrize(
    "yaml_path",
    sorted(CONFIG_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_example_config_validates(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    assert cfg.run.name  # smoke: schema parsed and produced a populated TrainConfig
