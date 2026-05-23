"""Smoke test for the new Pydantic LossConfig.

Full coverage in test_loss_presets and test_config_schema.
"""

from __future__ import annotations

from custom_sam_peft.config.schema import LossConfig, LossOverrides


def test_loss_config_smoke() -> None:
    cfg = LossConfig()
    assert cfg.preset == "natural"
    assert cfg.class_imbalance == "balanced"
    assert isinstance(cfg.overrides, LossOverrides)


def test_loss_config_overrides_smoke() -> None:
    cfg = LossConfig(
        preset="medical", class_imbalance="moderate", overrides=LossOverrides(focal_gamma=3.5)
    )
    assert cfg.overrides.focal_gamma == 3.5
