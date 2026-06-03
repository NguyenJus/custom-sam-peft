# tests/config/test_semantic_loss_config.py
"""SemanticLossConfig schema coverage (§7.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import SemanticLossConfig


def test_defaults():
    c = SemanticLossConfig()
    assert c.preset == "natural"
    assert c.class_imbalance == "balanced"
    assert c.background_logit == 0.0
    assert c.background_class_name is None
    assert c.query_reduce == "max"
    assert c.source == "marginalize"


def test_strict_extra_rejected():
    with pytest.raises(ValidationError):
        SemanticLossConfig(unknown_knob=1)


def test_query_reduce_literal():
    assert SemanticLossConfig(query_reduce="sum").query_reduce == "sum"
    with pytest.raises(ValidationError):
        SemanticLossConfig(query_reduce="mean")


def test_source_literal():
    assert SemanticLossConfig(source="semantic_seg").source == "semantic_seg"
    with pytest.raises(ValidationError):
        SemanticLossConfig(source="head")


def test_overrides_sem_family_literal():
    c = SemanticLossConfig(overrides={"sem_family": "focal_tversky"})
    assert c.overrides.sem_family == "focal_tversky"
    with pytest.raises(ValidationError):
        SemanticLossConfig(overrides={"sem_family": "bce"})  # not a SemMaskFamily
