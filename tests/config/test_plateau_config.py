"""Schema tests for the plateau ladder config blocks (spec §5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import (
    EarlyStopConfig,
    LrDecayOnPlateauConfig,
    TrainHyperparams,
)


def test_lr_schedule_default_is_plateau() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.lr_schedule == "plateau"


def test_lr_schedule_accepts_legacy_modes() -> None:
    for mode in ("constant", "cosine", "linear", "plateau"):
        assert TrainHyperparams(epochs=1, lr_schedule=mode).lr_schedule == mode


def test_lr_schedule_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, lr_schedule="poly")


def test_lr_decay_on_plateau_defaults() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.lr_decay_on_plateau.patience == 5
    assert hp.lr_decay_on_plateau.factor == 0.1
    assert hp.lr_decay_on_plateau.min_lr == 1.0e-6


def test_lr_decay_factor_must_shrink() -> None:
    with pytest.raises(ValidationError):
        LrDecayOnPlateauConfig(factor=1.0)
    with pytest.raises(ValidationError):
        LrDecayOnPlateauConfig(factor=1.5)


def test_early_stop_defaults() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.early_stop.enabled is True
    assert hp.early_stop.monitor == "mAP"
    assert hp.early_stop.min_delta == 0.001
    assert hp.early_stop.stop_patience == 10


def test_early_stop_monitor_is_single_valued() -> None:
    with pytest.raises(ValidationError):
        EarlyStopConfig(monitor="DSC")


def test_blocks_reject_extra_keys() -> None:
    with pytest.raises(ValidationError):
        LrDecayOnPlateauConfig(bogus=1)
    with pytest.raises(ValidationError):
        EarlyStopConfig(bogus=1)
