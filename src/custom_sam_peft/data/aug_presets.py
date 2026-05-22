"""Domain-aware augmentation presets — resolver and run-metadata helpers.

Pure-Python (numpy at most); this module itself does not import albumentations.
Note that `csp doctor` still pulls albumentations transitively via
`data.transforms.resolve_normalization_with_path`, so the practical isolation
benefit is limited to keeping `aug_presets` cheap to import in tests.

Public API:
  - PRESET_TABLE: dict[(Preset, Intensity), dict[str, bool | float]]
  - LOCKED_OFF:   dict[str, dict[str, str]]
  - ResolvedAugmentations: frozen dataclass with 8 knobs
  - resolve(cfg) -> ResolvedAugmentations
  - dump_augmentation_pipeline(cfg) -> dict  (sidecar helper)
  - _STEP_NAMES_FOR(resolved) -> list[str]   (module-private; consumed by
    trainer + doctor for run-metadata + table display)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from custom_sam_peft.config.schema import AugmentationsConfig, Intensity, Preset

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset × intensity table — spec §5
# ---------------------------------------------------------------------------

# Twelve cells for the four real domains. `none` and `custom` are short-circuited.
PRESET_TABLE: dict[tuple[Preset, Intensity], dict[str, bool | float]] = {
    ("natural", "safe"): {
        "hflip": True,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.05,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("natural", "medium"): {
        "hflip": True,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.1,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("natural", "aggressive"): {
        "hflip": True,
        "vflip": True,
        "rotate90": False,
        "rotate_arbitrary": 10.0,
        "color_jitter": 0.2,
        "stain_jitter": 0.0,
        "blur": 0.05,
        "gauss_noise": 0.02,
    },
    ("medical", "safe"): {
        "hflip": False,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("medical", "medium"): {
        "hflip": False,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 5.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.03,
        "blur": 0.0,
        "gauss_noise": 0.01,
    },
    ("medical", "aggressive"): {
        "hflip": False,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 10.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.07,
        "blur": 0.03,
        "gauss_noise": 0.03,
    },
    ("satellite", "safe"): {
        "hflip": True,
        "vflip": True,
        "rotate90": True,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("satellite", "medium"): {
        "hflip": True,
        "vflip": True,
        "rotate90": True,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.05,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("satellite", "aggressive"): {
        "hflip": True,
        "vflip": True,
        "rotate90": True,
        "rotate_arbitrary": 15.0,
        "color_jitter": 0.1,
        "stain_jitter": 0.0,
        "blur": 0.05,
        "gauss_noise": 0.02,
    },
    ("microscopy", "safe"): {
        "hflip": False,
        "vflip": True,
        "rotate90": True,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("microscopy", "medium"): {
        "hflip": False,
        "vflip": True,
        "rotate90": True,
        "rotate_arbitrary": 0.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.0,
        "blur": 0.0,
        "gauss_noise": 0.0,
    },
    ("microscopy", "aggressive"): {
        "hflip": False,
        "vflip": True,
        "rotate90": True,
        "rotate_arbitrary": 15.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.0,
        "blur": 0.05,
        "gauss_noise": 0.02,
    },
}


# ---------------------------------------------------------------------------
# Locked-off knob map — spec §6
# ---------------------------------------------------------------------------

LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "hflip": "laterality (left vs right) is clinically meaningful in most medical modalities (CXR, mammography, derm)",
        "vflip": "laterality (superior vs inferior) is clinically meaningful in most medical modalities",
        "rotate90": "laterality is clinically meaningful; arbitrary 90° rotation breaks canonical orientation",
        "color_jitter": "color carries diagnostic signal (e.g. melanoma); use stain_jitter for H&E instead",
    },
    "natural": {
        "rotate90": "arbitrary 90° rotation breaks 'up' for natural photography; use rotate_arbitrary for mild tilt",
    },
    "microscopy": {
        "hflip": "horizontal flip can break channel-ordering conventions in multiplexed microscopy",
        "color_jitter": "color identifies fluorescence channels and must be preserved",
    },
    "satellite": {
        "stain_jitter": "stain_jitter is H&E-specific (HED color deconvolution); satellite imagery is not H&E",
    },
}


# ---------------------------------------------------------------------------
# Resolved view — spec §7
# ---------------------------------------------------------------------------

_ZERO_BASE: dict[str, bool | float] = {
    "hflip": False,
    "vflip": False,
    "rotate90": False,
    "rotate_arbitrary": 0.0,
    "color_jitter": 0.0,
    "stain_jitter": 0.0,
    "blur": 0.0,
    "gauss_noise": 0.0,
}


@dataclass(frozen=True)
class ResolvedAugmentations:
    """Immutable 8-knob view consumed by build_train_transforms and the sidecar."""

    hflip: bool
    vflip: bool
    rotate90: bool
    rotate_arbitrary: float
    color_jitter: float
    stain_jitter: float
    blur: float
    gauss_noise: float


def _is_enabled(v: bool | float | None) -> bool:
    """True if v is a non-False bool or a strictly positive float."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) > 0.0
    return False


def resolve(cfg: AugmentationsConfig) -> ResolvedAugmentations:
    """Resolve (preset, intensity, overrides) into the 8-knob immutable view.

    - For `preset` in {"none", "custom"}: seed all-zero; intensity ignored.
    - Otherwise: seed from PRESET_TABLE[(preset, intensity)].
    - Apply overrides on top. Locked-off knob enabled under a real preset → WARN
      (user override wins; warn is the entire contract — spec §6).
    """
    if cfg.preset in ("none", "custom"):
        base: dict[str, bool | float] = dict(_ZERO_BASE)
    else:
        base = dict(PRESET_TABLE[(cfg.preset, cfg.intensity)])

    for field, override in cfg.overrides.model_dump().items():
        if override is None:
            continue
        base[field] = override
        if cfg.preset in ("none", "custom"):
            continue
        if field in LOCKED_OFF.get(cfg.preset, {}) and _is_enabled(override):
            reason = LOCKED_OFF[cfg.preset][field]
            _LOG.warning(
                "You enabled %s=%s under preset=%s; %s. The override will be applied as-is.",
                field,
                override,
                cfg.preset,
                reason,
            )

    return ResolvedAugmentations(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step-name list — spec §8 assembly mirrored for sidecar + doctor display
# ---------------------------------------------------------------------------


def _STEP_NAMES_FOR(resolved: ResolvedAugmentations) -> list[str]:
    """Ordered Albumentations class-name list produced by build_train_transforms.

    MUST match the conditional emission in
    `custom_sam_peft.data.transforms.build_train_transforms` step-for-step.
    """
    steps: list[str] = ["LongestMaxSize", "PadIfNeeded"]
    if resolved.hflip:
        steps.append("HorizontalFlip")
    if resolved.vflip:
        steps.append("VerticalFlip")
    if resolved.rotate90:
        steps.append("RandomRotate90")
    if resolved.rotate_arbitrary > 0.0:
        steps.append("Affine")
    if resolved.gauss_noise > 0.0:
        steps.append("GaussNoise")
    if resolved.blur > 0.0:
        steps.append("GaussianBlur")
    if resolved.color_jitter > 0.0:
        steps.append("ColorJitter")
    if resolved.stain_jitter > 0.0:
        steps.append("StainJitter")
    steps += ["Normalize", "ToTensorV2"]
    return steps


def dump_augmentation_pipeline(cfg: AugmentationsConfig) -> dict[str, Any]:
    """Build the JSON-shaped sidecar dict for a resolved augmentation config.

    See spec §10 for the exact dict shape. Consumed by the trainer to write
    `run_dir/augmentation_pipeline.json` and by `csp doctor --config` for the
    `resolved_config.augmentations` JSON block.

    For strict reproducibility across library versions, copy the returned
    `resolved` dict verbatim into `overrides:` under `preset: custom` —
    the resolver then returns identical values regardless of future
    PRESET_TABLE shifts.
    """
    try:
        from custom_sam_peft import __version__ as lib_version
    except (ImportError, AttributeError):
        lib_version = "unknown"

    resolved = resolve(cfg)
    return {
        "preset": cfg.preset,
        "intensity": cfg.intensity,
        "resolved": {
            "hflip": resolved.hflip,
            "vflip": resolved.vflip,
            "rotate90": resolved.rotate90,
            "rotate_arbitrary": resolved.rotate_arbitrary,
            "color_jitter": resolved.color_jitter,
            "stain_jitter": resolved.stain_jitter,
            "blur": resolved.blur,
            "gauss_noise": resolved.gauss_noise,
        },
        "steps": _STEP_NAMES_FOR(resolved),
        "library_version": lib_version,
    }
