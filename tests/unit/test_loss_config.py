"""Unit tests for LossConfig + MatcherWeights internal configs (audit Section G).

LossConfig and MatcherWeights have been moved to config._internal as plain
dataclasses (no Pydantic). They are re-exported from config.schema for backward
compatibility. New code should import from custom_sam_peft.config._internal.
"""

from __future__ import annotations

import pytest

# Import via schema (backward compat re-export) to verify re-export works.
from custom_sam_peft.config.schema import LossConfig, MatcherWeights, TrainConfig


def test_matcher_weights_defaults() -> None:
    w = MatcherWeights()
    assert w.lambda_mask == 5.0
    # No lambda_cls — open-vocab head has no per-class classification.
    assert not hasattr(w, "lambda_cls")
    # lambda_l1 / lambda_giou were demoted to inline literals in losses.py (#92).
    assert not hasattr(w, "lambda_l1")
    assert not hasattr(w, "lambda_giou")


def test_matcher_weights_rejects_extra_fields() -> None:
    # MatcherWeights is now a dataclass (not Pydantic) — extra fields raise TypeError.
    with pytest.raises(TypeError):
        MatcherWeights(lambda_cls=2.0)  # type: ignore[call-arg]


def test_loss_config_defaults() -> None:
    cfg = LossConfig()
    assert cfg.w_mask == 1.0
    # w_box was demoted earlier (audit Section E).
    assert cfg.w_box == 0.0
    assert cfg.w_obj == 1.0
    assert cfg.w_presence == 1.0
    assert isinstance(cfg.matcher_weights, MatcherWeights)
    # focal_gamma / focal_alpha demoted to module-level constants in losses.py (#93).
    assert not hasattr(cfg, "focal_gamma")
    assert not hasattr(cfg, "focal_alpha")
    # No w_cls — open-vocab head has no per-class classification.
    assert not hasattr(cfg, "w_cls")


def test_loss_config_rejects_extra_fields() -> None:
    # LossConfig is now a dataclass (not Pydantic) — extra fields raise TypeError.
    with pytest.raises(TypeError):
        LossConfig(w_cls=2.0)  # type: ignore[call-arg]


def test_train_config_includes_loss() -> None:
    from custom_sam_peft.config.schema import (
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


def test_loss_config_docstring_drops_legacy_claim() -> None:
    from custom_sam_peft.config._internal import LossConfig

    doc = LossConfig.__doc__ or ""
    assert "one forward pass per class prompt" not in doc
    assert "multiplex" in doc.lower()
