"""Tests for custom_sam_peft.data.aug_presets — resolver, locked-off WARN, sidecar dump."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from custom_sam_peft.config.schema import AugmentationOverrides, AugmentationsConfig
from custom_sam_peft.data.aug_presets import (
    _STEP_NAMES_FOR,
    PRESET_TABLE,
    dump_augmentation_pipeline,
    resolve,
)

_LOGGER = "custom_sam_peft.data.aug_presets"


@pytest.mark.parametrize(
    "preset,intensity",
    sorted(PRESET_TABLE.keys()),
)
def test_resolve_table_exact_values(preset: str, intensity: str) -> None:
    """Every (preset, intensity) cell resolves to its table row."""
    cfg = AugmentationsConfig(preset=preset, intensity=intensity)  # type: ignore[arg-type]
    resolved = resolve(cfg)
    expected = PRESET_TABLE[(preset, intensity)]  # type: ignore[index]
    for k, v in expected.items():
        assert getattr(resolved, k) == v, (preset, intensity, k, v, getattr(resolved, k))


@pytest.mark.parametrize("intensity", ["safe", "medium", "aggressive"])
def test_resolve_none_zeroes_all_knobs(intensity: str) -> None:
    cfg = AugmentationsConfig(preset="none", intensity=intensity)  # type: ignore[arg-type]
    resolved = resolve(cfg)
    assert resolved.hflip is False
    assert resolved.vflip is False
    assert resolved.rotate90 is False
    assert resolved.rotate_arbitrary == 0.0
    assert resolved.color_jitter == 0.0
    assert resolved.stain_jitter == 0.0
    assert resolved.blur == 0.0
    assert resolved.gauss_noise == 0.0


def test_resolve_custom_zeroes_then_overrides_apply() -> None:
    cfg = AugmentationsConfig(
        preset="custom",
        intensity="aggressive",  # ignored
        overrides=AugmentationOverrides(hflip=True, stain_jitter=0.05),
    )
    resolved = resolve(cfg)
    assert resolved.hflip is True
    assert resolved.stain_jitter == 0.05
    # Everything else stays at the all-zero seed.
    assert resolved.vflip is False
    assert resolved.color_jitter == 0.0
    assert resolved.gauss_noise == 0.0


def test_resolve_override_wins_over_table() -> None:
    cfg = AugmentationsConfig(
        preset="natural",
        intensity="medium",
        overrides=AugmentationOverrides(color_jitter=0.5),
    )
    resolved = resolve(cfg)
    assert resolved.color_jitter == 0.5  # override
    # Other fields preserved from the table row.
    assert resolved.hflip is True
    assert resolved.rotate_arbitrary == 0.0  # natural/medium row
    assert resolved.blur == 0.0


def test_resolve_override_zero_disables_table_knob() -> None:
    """Zero is a valid override, not 'inherit'."""
    cfg = AugmentationsConfig(
        preset="natural",
        intensity="medium",
        overrides=AugmentationOverrides(color_jitter=0.0),
    )
    resolved = resolve(cfg)
    assert resolved.color_jitter == 0.0


@pytest.mark.parametrize(
    "preset,knob,value,expected_substr",
    [
        ("medical", "hflip", True, "laterality"),
        ("natural", "rotate90", True, "up"),
        ("microscopy", "color_jitter", 0.1, "fluorescence"),
        ("satellite", "stain_jitter", 0.05, "H&E"),
    ],
)
def test_resolve_locked_off_warns(
    caplog: pytest.LogCaptureFixture,
    preset: str,
    knob: str,
    value: object,
    expected_substr: str,
) -> None:
    cfg = AugmentationsConfig(
        preset=preset,  # type: ignore[arg-type]
        intensity="medium",
        overrides=AugmentationOverrides(**{knob: value}),  # type: ignore[arg-type]
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolved = resolve(cfg)

    # Override applied as-is (not stripped).
    assert getattr(resolved, knob) == value

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1, [r.getMessage() for r in caplog.records]
    msg = warns[0].getMessage()
    assert knob in msg
    assert preset in msg
    assert expected_substr in msg


def test_resolve_locked_off_no_warn_when_disabling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """False/0 override on a locked-off knob does not warn (disabling is always fine)."""
    cfg = AugmentationsConfig(
        preset="medical",
        intensity="medium",
        overrides=AugmentationOverrides(hflip=False),
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolve(cfg)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolve_none_skips_locked_off_check(caplog: pytest.LogCaptureFixture) -> None:
    cfg = AugmentationsConfig(
        preset="none",
        overrides=AugmentationOverrides(hflip=True),
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolved = resolve(cfg)
    assert resolved.hflip is True
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolve_custom_skips_locked_off_check(caplog: pytest.LogCaptureFixture) -> None:
    cfg = AugmentationsConfig(
        preset="custom",
        overrides=AugmentationOverrides(hflip=True, stain_jitter=0.1),
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolved = resolve(cfg)
    assert resolved.hflip is True
    assert resolved.stain_jitter == 0.1
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolved_augmentations_frozen() -> None:
    r = resolve(AugmentationsConfig(preset="none"))
    # replace works.
    r2 = dataclasses.replace(r, hflip=True)
    assert r2.hflip is True
    # Direct mutation forbidden.
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.hflip = True  # type: ignore[misc]


def test_dump_augmentation_pipeline_shape_medical_medium() -> None:
    """Spec §10 literal example."""
    cfg = AugmentationsConfig(preset="medical", intensity="medium")
    d = dump_augmentation_pipeline(cfg)
    assert d["preset"] == "medical"
    assert d["intensity"] == "medium"
    assert d["resolved"] == {
        "hflip": False,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 5.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.03,
        "blur": 0.0,
        "gauss_noise": 0.01,
    }
    assert d["steps"] == [
        "LongestMaxSize",
        "PadIfNeeded",
        "Affine",
        "GaussNoise",
        "StainJitter",
        "Normalize",
        "ToTensorV2",
    ]
    assert isinstance(d["library_version"], str) and d["library_version"]


def test_dump_augmentation_pipeline_steps_empty_for_none() -> None:
    d = dump_augmentation_pipeline(AugmentationsConfig(preset="none"))
    assert d["steps"] == ["LongestMaxSize", "PadIfNeeded", "Normalize", "ToTensorV2"]


def test_step_names_for_natural_aggressive() -> None:
    """Representative non-trivial cell: every knob fires."""
    cfg = AugmentationsConfig(preset="natural", intensity="aggressive")
    resolved = resolve(cfg)
    assert _STEP_NAMES_FOR(resolved) == [
        "LongestMaxSize",
        "PadIfNeeded",
        "HorizontalFlip",
        "VerticalFlip",
        "Affine",
        "GaussNoise",
        "GaussianBlur",
        "ColorJitter",
        "Normalize",
        "ToTensorV2",
    ]
