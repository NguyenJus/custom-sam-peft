"""Tests for the loss-preset resolver (spec §11.1)."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.config.schema import (
    ClassImbalance,
    LossConfig,
    LossOverrides,
    Preset,
)
from custom_sam_peft.models.losses.presets import (
    _LEGACY_DEFAULTS,
    PRESET_TABLE,
    dump_loss_bundle,
    resolve,
)

_REAL_PRESETS: list[Preset] = ["natural", "medical", "satellite", "microscopy"]
_TIERS: list[ClassImbalance] = ["balanced", "moderate", "severe"]


# -- Table exact values (spec §5) ----------------------------------------------


@pytest.mark.parametrize("preset", _REAL_PRESETS)
@pytest.mark.parametrize("tier", _TIERS)
def test_resolve_table_exact_values(preset: Preset, tier: ClassImbalance) -> None:
    cfg = LossConfig(preset=preset, class_imbalance=tier)
    r = resolve(cfg)
    row = PRESET_TABLE[(preset, tier)]
    for fname, expected in row.items():
        assert getattr(r, fname) == expected, (preset, tier, fname, expected, getattr(r, fname))


# -- Short-circuit presets ------------------------------------------------------


@pytest.mark.parametrize("tier", _TIERS)
def test_resolve_none_uses_legacy_defaults(tier: ClassImbalance) -> None:
    cfg = LossConfig(preset="none", class_imbalance=tier)
    r = resolve(cfg)
    for fname, expected in _LEGACY_DEFAULTS.items():
        assert getattr(r, fname) == expected, (fname, expected, getattr(r, fname))


def test_resolve_custom_seeds_with_natural_balanced() -> None:
    cfg = LossConfig(preset="custom")
    r = resolve(cfg)
    row = PRESET_TABLE[("natural", "balanced")]
    for fname, expected in row.items():
        assert getattr(r, fname) == expected, (fname, expected, getattr(r, fname))


# -- Override layering ----------------------------------------------------------


def test_resolve_override_wins_over_table() -> None:
    cfg = LossConfig(
        preset="natural",
        class_imbalance="balanced",
        overrides=LossOverrides(focal_gamma=5.0),
    )
    r = resolve(cfg)
    assert r.focal_gamma == 5.0
    # other fields untouched
    assert r.mask_family == PRESET_TABLE[("natural", "balanced")]["mask_family"]


def test_resolve_override_zero_is_valid() -> None:
    cfg = LossConfig(
        preset="natural",
        class_imbalance="balanced",
        overrides=LossOverrides(w_obj=0.5),  # 0 rejected by PositiveFloat; use 0.5
    )
    r = resolve(cfg)
    assert r.w_obj == 0.5
    # w_box is the only override that allows zero (ge=0.0)
    cfg2 = LossConfig(overrides=LossOverrides(w_box=0.0))
    assert resolve(cfg2).w_box == 0.0


def test_resolve_matcher_weights_override() -> None:
    cfg = LossConfig(overrides=LossOverrides(matcher_weights=MatcherWeights(lambda_mask=9.0)))
    r = resolve(cfg)
    assert r.matcher_weights.lambda_mask == 9.0


def test_resolve_matcher_weights_dict_coerced() -> None:
    cfg = LossConfig(overrides={"matcher_weights": {"lambda_mask": 11.0}})  # type: ignore[arg-type]
    r = resolve(cfg)
    assert r.matcher_weights.lambda_mask == 11.0


# -- LOCKED_OFF warns -----------------------------------------------------------


def test_resolve_locked_off_warns_medical_mask_family(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(
        preset="medical",
        class_imbalance="moderate",
        overrides=LossOverrides(mask_family="dice_bce"),
    )
    r = resolve(cfg)
    assert r.mask_family == "dice_bce"  # override wins
    msgs = [rec.message for rec in caplog.records]
    assert any("mask_family" in m and "medical" in m and "rare positives" in m for m in msgs), msgs


def test_resolve_locked_off_warns_natural_mask_family(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(
        preset="natural",
        class_imbalance="balanced",
        overrides=LossOverrides(mask_family="focal_tversky"),
    )
    resolve(cfg)
    msgs = [rec.message for rec in caplog.records]
    assert any("mask_family" in m and "natural" in m and "unusual" in m for m in msgs), msgs


def test_resolve_locked_off_no_warn_when_override_matches_seed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overriding to the table's existing value is a no-op; no warn."""
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    seed = PRESET_TABLE[("medical", "balanced")]["mask_family"]
    cfg = LossConfig(
        preset="medical",
        class_imbalance="balanced",
        overrides=LossOverrides(mask_family=seed),
    )
    resolve(cfg)
    assert not caplog.records, [rec.message for rec in caplog.records]


def test_resolve_none_skips_locked_off(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(preset="none", overrides=LossOverrides(mask_family="focal_tversky"))
    resolve(cfg)
    assert not caplog.records, [rec.message for rec in caplog.records]


def test_resolve_custom_skips_locked_off(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(preset="custom", overrides=LossOverrides(mask_family="boundary"))
    resolve(cfg)
    assert not caplog.records, [rec.message for rec in caplog.records]


# -- ResolvedLosses immutability ------------------------------------------------


def test_resolved_losses_frozen() -> None:
    r = resolve(LossConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.w_mask = 2.0  # type: ignore[misc]
    # replace works
    r2 = dataclasses.replace(r, w_mask=2.0)
    assert r2.w_mask == 2.0


# -- Sidecar helper -------------------------------------------------------------


def test_dump_loss_bundle_shape() -> None:
    cfg = LossConfig(preset="medical", class_imbalance="moderate")
    d = dump_loss_bundle(cfg)
    assert set(d.keys()) == {
        "preset",
        "class_imbalance",
        "resolved",
        "term_classes",
        "library_version",
    }
    assert d["preset"] == "medical"
    assert d["class_imbalance"] == "moderate"
    assert set(d["resolved"].keys()) == {
        "mask_family",
        "box_family",
        "obj_family",
        "presence_family",
        "w_mask",
        "w_box",
        "w_obj",
        "w_presence",
        "focal_gamma",
        "focal_alpha",
        "tversky_alpha",
        "tversky_gamma",
        "boundary_weight",
    }
    assert d["term_classes"] == {
        "mask": "FocalTverskyLoss",
        "box": "L1GIoULoss",
        "obj": "FocalBCELoss",
        "presence": "BCELoss",
    }
    assert isinstance(d["library_version"], str) and d["library_version"]
    # round-trip through JSON
    assert json.loads(json.dumps(d)) == d


def test_dump_loss_bundle_for_none_preset() -> None:
    d = dump_loss_bundle(LossConfig(preset="none"))
    assert d["resolved"]["mask_family"] == "dice_bce"
    assert d["term_classes"]["mask"] == "DiceBCELoss"


# -- Microscopy alias contract --------------------------------------------------


@pytest.mark.parametrize("tier", _TIERS)
def test_microscopy_equals_medical(tier: ClassImbalance) -> None:
    assert PRESET_TABLE[("microscopy", tier)] == PRESET_TABLE[("medical", tier)]
