"""Domain-aware loss-function presets — resolver and run-metadata helpers.

Pure-Python module: does NOT import torch. The resolver can be imported into
`csp doctor` without dragging torch into the doctor import graph.

Public API:
  - PRESET_TABLE: dict[(Preset, ClassImbalance), dict[str, str | float]]
  - LOCKED_OFF:   dict[str, dict[str, str]]
  - ResolvedLosses: frozen dataclass with 13 knobs + matcher_weights
  - resolve(cfg) -> ResolvedLosses
  - dump_loss_bundle(cfg) -> dict  (sidecar helper)

Citation tags (see spec §5.3):
  (A) #112 issue body            — cell lifted verbatim from the issue's draft table
  (B) preserved pre-#112         — matches today's hardcoded trainer behavior in losses.py
  (C) Lin et al. 2017 (RetinaNet/focal loss)         — γ=2.0, α=0.25 from Table 1
  (D) Abraham & Khan 2019 (focal Tversky)            — γ=0.75 best on ISIC
  (E) Salehi et al. 2017 (Tversky loss)              — β=0.7 (FN weight) best on MS lesions
  (F) degenerate-case identity                       — α=0.5 reduces Tversky to Dice;
                                                       γ=1.0 reduces Focal-Tversky to Tversky
  (G) alias-of-medical                               — microscopy copies medical (unsourced
                                                       alias; tbd #191 — see #120)
  (H) Kervadec et al. 2019 (boundary loss)           — blend coefficient ~0.2 representative

All cells now resolve to a legend letter, `# tbd: #191`, or `# cite: empirical`.
γ=2.5/3.0 escalations (moderate/severe) have no external literature source and
no recorded internal run → tagged `# tbd: #191`. `tversky_alpha=0.7` is Salehi
et al. 2017's best FN-penalization weight (their β=0.7) → `# cite: (A,E)`.
`tversky_alpha=0.8` has no external source → `# tbd: #191`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    from custom_sam_peft._version import __version__ as _LIB_VERSION
except ImportError:
    _LIB_VERSION = "unknown"
from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.config.schema import (
    ClassImbalance,
    LossConfig,
    Preset,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Term-class names (used by dump_loss_bundle to avoid importing compose.py
# from this pure-Python module). Kept in lockstep with compose.py's term
# registries via a sync-check test (see test_loss_compose.py::
# test_term_class_names_match_compose_registry).
# ---------------------------------------------------------------------------

_TERM_CLASS_NAMES: dict[str, dict[str, str]] = {
    "mask": {
        "bce": "BCELoss",
        "dice": "DiceLoss",
        "dice_bce": "DiceBCELoss",
        "focal_bce": "FocalBCELoss",
        "focal_dice": "FocalDiceLoss",
        "focal_tversky": "FocalTverskyLoss",
        "boundary": "BoundaryLoss",
    },
    "box": {
        "l1_giou": "L1GIoULoss",
        "giou_only": "GIoUOnlyLoss",
        "ciou": "CIoULoss",
    },
    "obj": {
        "focal_bce": "FocalBCELoss",
        "bce": "BCELoss",
    },
    "presence": {
        "bce": "BCELoss",
        "focal_bce": "FocalBCELoss",
    },
}


# ---------------------------------------------------------------------------
# Preset × class_imbalance table — spec §5
# ---------------------------------------------------------------------------

# Twelve cells for the four real domains. Microscopy is byte-equal to medical
# in v1 (alias-of-medical; spec §5.2). `none` and `custom` are short-circuited
# in resolve(), not stored here.
PRESET_TABLE: dict[tuple[Preset, ClassImbalance], dict[str, Any]] = {
    # ----- natural -----
    ("natural", "balanced"): {
        "mask_family": "dice_bce",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 2.0,  # cite: (A,C)
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,
    },
    ("natural", "moderate"): {
        "mask_family": "dice_bce",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 2.5,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,
    },
    ("natural", "severe"): {
        "mask_family": "focal_dice",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 3.0,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.6,  # cite: (A,E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,
    },
    # ----- medical -----
    ("medical", "balanced"): {
        "mask_family": "focal_dice",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 2.0,  # cite: (A,C)
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.6,  # cite: (A,E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,
    },
    ("medical", "moderate"): {
        "mask_family": "focal_tversky",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 2.5,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.7,  # cite: (A,E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,
    },
    ("medical", "severe"): {
        "mask_family": "boundary",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 3.0,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.8,  # tbd: #191
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.2,  # cite: (A,H)
    },
    # ----- satellite -----
    ("satellite", "balanced"): {
        "mask_family": "dice_bce",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 2.0,  # cite: (A,C)
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,
    },
    ("satellite", "moderate"): {
        "mask_family": "focal_dice",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 2.5,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.6,  # cite: (A,E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,
    },
    ("satellite", "severe"): {
        "mask_family": "focal_tversky",  # cite: (A)
        "box_family": "l1_giou",  # cite: (B)
        "obj_family": "focal_bce",  # cite: (B)
        "presence_family": "bce",  # cite: (B)
        "w_mask": 1.0,  # cite: (B)
        "w_box": 0.0,  # cite: (B)
        "w_obj": 1.0,  # cite: (B)
        "w_presence": 1.0,  # cite: (B)
        "focal_gamma": 3.0,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (A,C)
        "tversky_alpha": 0.7,  # cite: (A,E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,
    },
}

# Microscopy = strict alias of medical (spec §5.2). Reuse the same dicts.
PRESET_TABLE[("microscopy", "balanced")] = dict(PRESET_TABLE[("medical", "balanced")])  # cite: (G)
PRESET_TABLE[("microscopy", "moderate")] = dict(PRESET_TABLE[("medical", "moderate")])  # cite: (G)
PRESET_TABLE[("microscopy", "severe")] = dict(PRESET_TABLE[("medical", "severe")])  # cite: (G)


# ---------------------------------------------------------------------------
# _LEGACY_DEFAULTS — values used when preset == "none" (preserves pre-#112).
# ---------------------------------------------------------------------------

_LEGACY_DEFAULTS: dict[str, Any] = {
    "mask_family": "dice_bce",
    "box_family": "l1_giou",
    "obj_family": "focal_bce",
    "presence_family": "bce",
    "w_mask": 1.0,
    "w_box": 0.0,
    "w_obj": 1.0,
    "w_presence": 1.0,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    "tversky_alpha": 0.5,  # neutral — Dice-equivalent; ignored by dice_bce
    "tversky_gamma": 1.0,  # neutral — Tversky-equivalent; ignored by dice_bce
    "boundary_weight": 0.0,
}


# ---------------------------------------------------------------------------
# LOCKED_OFF — knob overrides that emit a WARN under specific presets.
# ---------------------------------------------------------------------------

LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "mask_family": (
            "the medical preset chose focal_dice/focal_tversky/boundary to handle "
            "rare positives; overriding to dice_bce or bce may underweight them"
        ),
    },
    "natural": {
        "mask_family": (
            "the natural preset chose dice_bce/focal_dice; overriding to "
            "focal_tversky or boundary is unusual for balanced natural-image data"
        ),
    },
    # satellite, microscopy: no locked-off entries in v1 (revisit after real users).
}


# ---------------------------------------------------------------------------
# Resolved view (frozen) and resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedLosses:
    mask_family: str
    box_family: str
    obj_family: str
    presence_family: str
    w_mask: float
    w_box: float
    w_obj: float
    w_presence: float
    focal_gamma: float
    focal_alpha: float
    tversky_alpha: float
    tversky_gamma: float
    boundary_weight: float
    matcher_weights: MatcherWeights = field(default_factory=MatcherWeights)


def _override_triggers_warn(
    field_name: str, value: object, preset: Preset, class_imbalance: ClassImbalance
) -> bool:
    """Spec §6.2: warn only when the override changes the locked-off knob away
    from the table's seed value."""
    if preset not in LOCKED_OFF:
        return False
    if field_name not in LOCKED_OFF[preset]:
        return False
    if value is None:
        return False
    seed = PRESET_TABLE[(preset, class_imbalance)][field_name]
    return bool(value != seed)


def resolve(cfg: LossConfig) -> ResolvedLosses:
    """Spec §7. Returns a frozen ResolvedLosses with all 13 knobs populated."""
    # 1. Seed from the preset table (or short-circuit for none/custom).
    if cfg.preset == "none":
        base = dict(_LEGACY_DEFAULTS)
        seed_matcher = MatcherWeights()
    elif cfg.preset == "custom":
        base = dict(PRESET_TABLE[("natural", "balanced")])
        seed_matcher = MatcherWeights()
    else:
        base = dict(PRESET_TABLE[(cfg.preset, cfg.class_imbalance)])
        seed_matcher = MatcherWeights()

    # 2. Apply overrides; warn if a locked-off knob is overridden.
    ov = cfg.overrides.model_dump(exclude_unset=False)
    for fname, override in ov.items():
        if override is None:
            continue
        if fname == "matcher_weights":
            seed_matcher = MatcherWeights(**override) if isinstance(override, dict) else override
            continue
        if cfg.preset not in ("none", "custom") and _override_triggers_warn(
            fname, override, cfg.preset, cfg.class_imbalance
        ):
            reason = LOCKED_OFF[cfg.preset][fname]
            _LOG.warning(
                "You overrode %s=%s under preset=%s; %s. The override will be applied as-is.",
                fname,
                override,
                cfg.preset,
                reason,
            )
        base[fname] = override

    return ResolvedLosses(**base, matcher_weights=seed_matcher)


# ---------------------------------------------------------------------------
# Sidecar helper — spec §9
# ---------------------------------------------------------------------------


def dump_loss_bundle(cfg: LossConfig) -> dict[str, Any]:
    """Return the JSON-serializable dict written to run_dir/loss_bundle.json."""
    resolved = resolve(cfg)
    term_classes = {
        "mask": _TERM_CLASS_NAMES["mask"][resolved.mask_family],
        "box": _TERM_CLASS_NAMES["box"][resolved.box_family],
        "obj": _TERM_CLASS_NAMES["obj"][resolved.obj_family],
        "presence": _TERM_CLASS_NAMES["presence"][resolved.presence_family],
    }
    return {
        "preset": cfg.preset,
        "class_imbalance": cfg.class_imbalance,
        "resolved": {
            "mask_family": resolved.mask_family,
            "box_family": resolved.box_family,
            "obj_family": resolved.obj_family,
            "presence_family": resolved.presence_family,
            "w_mask": resolved.w_mask,
            "w_box": resolved.w_box,
            "w_obj": resolved.w_obj,
            "w_presence": resolved.w_presence,
            "focal_gamma": resolved.focal_gamma,
            "focal_alpha": resolved.focal_alpha,
            "tversky_alpha": resolved.tversky_alpha,
            "tversky_gamma": resolved.tversky_gamma,
            "boundary_weight": resolved.boundary_weight,
        },
        "term_classes": term_classes,
        "library_version": _LIB_VERSION or "unknown",
    }
