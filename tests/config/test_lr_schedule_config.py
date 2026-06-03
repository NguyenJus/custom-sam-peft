"""Schema tests for the LR schedule + early-stop config blocks (#264)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import (
    EarlyStopConfig,
    TrainHyperparams,
)


def test_lr_schedule_default_is_poly() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.lr_schedule == "poly"


def test_lr_schedule_accepts_supported_modes() -> None:
    for mode in ("constant", "cosine", "linear", "poly"):
        assert TrainHyperparams(epochs=1, lr_schedule=mode).lr_schedule == mode


def test_lr_schedule_rejects_removed_plateau() -> None:
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, lr_schedule="plateau")  # type: ignore[arg-type]


def test_lr_schedule_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, lr_schedule="bogus")  # type: ignore[arg-type]


def test_lr_decay_on_plateau_field_removed() -> None:
    """The plateau LR-decay block no longer exists on TrainHyperparams (#264)."""
    hp = TrainHyperparams(epochs=1)
    assert not hasattr(hp, "lr_decay_on_plateau")
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, lr_decay_on_plateau={"patience": 5})  # type: ignore[call-arg]


def test_early_stop_defaults() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.early_stop.enabled is True
    assert hp.early_stop.monitor == "mAP"
    assert hp.early_stop.min_delta == 0.001
    assert hp.early_stop.stop_patience == 10


def test_early_stop_warmup_floor_steps_default() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.early_stop.warmup_floor_steps == 1000


def test_early_stop_warmup_floor_steps_allows_zero() -> None:
    """0 disables the backstop (adaptive-baseline-only grace) (#264)."""
    assert EarlyStopConfig(warmup_floor_steps=0).warmup_floor_steps == 0


def test_early_stop_warmup_floor_steps_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        EarlyStopConfig(warmup_floor_steps=-1)


def test_early_stop_monitor_is_single_valued() -> None:
    with pytest.raises(ValidationError):
        EarlyStopConfig(monitor="DSC")  # type: ignore[arg-type]


def test_blocks_reject_extra_keys() -> None:
    with pytest.raises(ValidationError):
        EarlyStopConfig(bogus=1)  # type: ignore[call-arg]
