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


def test_coco_text_lora_subset_yaml_validates() -> None:
    """coco_text_lora_subset.yaml must parse cleanly with a non-None limit."""
    import yaml

    from custom_sam_peft.config.schema import TrainConfig

    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "configs" / "examples" / "coco_text_lora_subset.yaml"
    raw = yaml.safe_load(p.read_text())
    cfg = TrainConfig.model_validate(raw)
    assert cfg.data.limit.train == 64
    assert cfg.data.limit.val == 16
    assert cfg.data.limit.seed == 42
    assert cfg.data.limit.strategy == "random"
