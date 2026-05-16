"""Unit tests for LossConfig + MatcherWeights schemas in spec/model-loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import LossConfig, MatcherWeights, TrainConfig


def test_matcher_weights_defaults() -> None:
    w = MatcherWeights()
    assert w.lambda_cls == 2.0
    assert w.lambda_l1 == 5.0
    assert w.lambda_giou == 2.0
    assert w.lambda_mask == 5.0


def test_loss_config_defaults() -> None:
    cfg = LossConfig()
    assert cfg.w_mask == 1.0
    assert cfg.w_box == 5.0
    assert cfg.w_obj == 1.0
    assert cfg.w_cls == 2.0
    assert cfg.focal_gamma == 2.0
    assert cfg.focal_alpha == 0.25
    assert isinstance(cfg.matcher_weights, MatcherWeights)


def test_loss_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LossConfig(unknown=1.0)  # type: ignore[call-arg]


def test_train_config_includes_loss() -> None:
    """TrainConfig must expose `loss: LossConfig` with defaults."""
    from esam3.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrainHyperparams,
    )

    tc = TrainConfig(
        run=RunConfig(name="x"),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a", images="b"),
            val=DataSplit(annotations="a", images="b"),
            prompt_mode="bbox",
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )
    assert isinstance(tc.train.loss, LossConfig)
