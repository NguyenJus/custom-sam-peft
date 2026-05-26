"""Unit tests for ModelConfig schema additions in spec/model-loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import ModelConfig


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.name == "facebook/sam3.1"
    assert cfg.local_dir == "models/sam3.1"
    assert cfg.checkpoint_file == "sam3.1_multiplex.pt"
    assert cfg.revision is None
    assert cfg.dtype == "bfloat16"
    assert cfg.device is None


def test_model_config_overrides() -> None:
    cfg = ModelConfig(local_dir=None, device="cpu")
    assert cfg.local_dir is None
    assert cfg.device == "cpu"


def test_model_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(unknown_field="x")  # type: ignore[call-arg]


def test_model_config_rejects_gradient_checkpointing() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(gradient_checkpointing=False)  # type: ignore[call-arg]
