"""Tests for BoxHintSchedule + the widened Optimizer literal + new TrainHyperparams fields."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from esam3.config.schema import (
    BoxHintSchedule,
    LossConfig,
    MatcherWeights,
    TrainHyperparams,
)


def test_box_hint_schedule_defaults() -> None:
    s = BoxHintSchedule()
    assert s.p_start == 1.0
    assert s.p_end == 0.0
    assert s.decay_steps == 5000
    assert s.early_stop_p_threshold == 0.05


def test_box_hint_schedule_rejects_non_monotone() -> None:
    with pytest.raises(ValidationError, match="must decay"):
        BoxHintSchedule(p_start=0.2, p_end=0.8)


def test_box_hint_schedule_accepts_equal_endpoints() -> None:
    """p_start == p_end is a constant schedule, allowed."""
    s = BoxHintSchedule(p_start=0.3, p_end=0.3)
    assert s.p_start == s.p_end == 0.3


def test_train_hyperparams_new_fields() -> None:
    h = TrainHyperparams(epochs=1)
    assert isinstance(h.box_hint, BoxHintSchedule)
    assert h.log_every == 50
    assert h.nan_abort_after == 20
    assert h.num_workers == min(4, os.cpu_count() or 1)


def test_train_hyperparams_optimizer_default_is_auto() -> None:
    h = TrainHyperparams(epochs=1)
    assert h.optimizer == "auto"


def test_train_hyperparams_optimizer_accepts_explicit_values() -> None:
    for opt in ("adamw", "adamw8bit", "auto"):
        h = TrainHyperparams(epochs=1, optimizer=opt)
        assert h.optimizer == opt


def test_loss_config_default_w_box_is_zero() -> None:
    """v0 text-only training drops box supervision."""
    assert LossConfig().w_box == 0.0


def test_matcher_weights_default_box_terms_are_zero() -> None:
    """v0 matcher is mask-only by default."""
    w = MatcherWeights()
    assert w.lambda_l1 == 0.0
    assert w.lambda_giou == 0.0
    assert w.lambda_mask == 5.0  # unchanged
