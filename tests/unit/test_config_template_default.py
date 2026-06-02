"""The config_full.yaml template defaults tracking.backend to local."""

from __future__ import annotations

from importlib import resources


def test_config_full_template_defaults_backend_local() -> None:
    text = resources.files("custom_sam_peft.cli.templates").joinpath("config_full.yaml").read_text()
    assert "backend: local" in text
    assert "backend: tensorboard" not in text
