"""Domain-aware SEMANTIC loss presets — torch-free resolver + run-metadata helpers.

Mirrors models/losses/presets.py. Pure-Python (no torch) so `csp doctor` and
schema tests import it without dragging torch in.

Citation legend (spec §7.3):
  (S) SAMed (Zhang & Liu 2023, arXiv:2304.13785) §3.3 — CE/region = 0.2/0.8.
  (C) Lin et al. 2017 (focal) — gamma=2.0, alpha=0.25.
  (D) Abraham & Khan 2019 (Focal-Tversky) — gamma=0.75 best on ISIC.
  (E) Salehi et al. 2017 (Tversky) — beta/alpha=0.7 (FN weight).
  (H) Kervadec et al. 2019 (boundary) — blend ~0.2.
  (F) degenerate identity (alpha=0.5 -> Dice; gamma=1.0 -> Tversky).
  (G) alias-of-medical (microscopy copies medical).
gamma escalations beyond a cited source -> `# tbd: #191`; unsourced tversky_alpha -> `# tbd: #191`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from custom_sam_peft._version import __version__ as _LIB_VERSION
except ImportError:
    _LIB_VERSION = "unknown"
from custom_sam_peft.config.schema import (
    ClassImbalance,
    Preset,
    SemanticLossConfig,
)

_LOG = logging.getLogger(__name__)

# Region-term class names per sem_family (avoids importing semantic_compose here).
# Kept in lockstep with semantic_compose via a sync-check test (B3).
_SEM_TERM_CLASS_NAMES: dict[str, str] = {
    "ce_dice": "SemCEDiceLoss",
    "focal_dice": "SemFocalDiceLoss",
    "focal_tversky": "SemFocalTverskyLoss",
    "boundary": "SemBoundaryLoss",
    "ce": "SemCELoss",
    "dice": "SemDiceLoss",
}

SEMANTIC_PRESET_TABLE: dict[tuple[Preset, ClassImbalance], dict[str, Any]] = {
    # ----- natural -----
    ("natural", "balanced"): {
        "sem_family": "ce_dice",  # cite: (S)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("natural", "moderate"): {
        "sem_family": "focal_dice",  # cite: (S,C)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("natural", "severe"): {
        "sem_family": "focal_dice",  # cite: (S)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 3.0,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    # ----- medical -----
    ("medical", "balanced"): {
        "sem_family": "focal_dice",  # cite: (S,C)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("medical", "moderate"): {
        "sem_family": "focal_tversky",  # cite: (S,E,D)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.7,  # cite: (E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("medical", "severe"): {
        "sem_family": "boundary",  # cite: (S,H)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.7,  # cite: (E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.2,  # cite: (H)
    },
    # ----- satellite -----
    ("satellite", "balanced"): {
        "sem_family": "ce_dice",  # cite: (S)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("satellite", "moderate"): {
        "sem_family": "boundary",  # cite: (S,H)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.2,  # cite: (H)
    },
    ("satellite", "severe"): {
        "sem_family": "focal_tversky",  # cite: (S,E,D)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.7,  # cite: (E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,  # cite: (F)
    },
}

# Microscopy = strict alias of medical (§7.3 (G)).
SEMANTIC_PRESET_TABLE[("microscopy", "balanced")] = dict(
    SEMANTIC_PRESET_TABLE[("medical", "balanced")]
)  # cite: (G)
SEMANTIC_PRESET_TABLE[("microscopy", "moderate")] = dict(
    SEMANTIC_PRESET_TABLE[("medical", "moderate")]
)  # cite: (G)
SEMANTIC_PRESET_TABLE[("microscopy", "severe")] = dict(
    SEMANTIC_PRESET_TABLE[("medical", "severe")]
)  # cite: (G)


LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "sem_family": (
            "the medical preset chose focal/tversky/boundary to handle rare positives; "
            "overriding to ce or dice may underweight them"
        ),
    },
    "natural": {
        "sem_family": (
            "the natural preset chose ce_dice/focal_dice; overriding to focal_tversky or "
            "boundary is unusual for balanced natural-image data"
        ),
    },
}


@dataclass(frozen=True)
class ResolvedSemanticLoss:
    sem_family: str
    w_ce: float
    w_region: float
    focal_gamma: float
    focal_alpha: float
    tversky_alpha: float
    tversky_gamma: float
    boundary_weight: float


def _override_triggers_warn(
    field_name: str, value: object, preset: Preset, class_imbalance: ClassImbalance
) -> bool:
    if preset not in LOCKED_OFF or field_name not in LOCKED_OFF[preset] or value is None:
        return False
    seed = SEMANTIC_PRESET_TABLE[(preset, class_imbalance)][field_name]
    return bool(value != seed)


def resolve(cfg: SemanticLossConfig) -> ResolvedSemanticLoss:
    base = dict(SEMANTIC_PRESET_TABLE[(cfg.preset, cfg.class_imbalance)])
    ov = cfg.overrides.model_dump(exclude_unset=False)
    for fname, override in ov.items():
        if override is None:
            continue
        if _override_triggers_warn(fname, override, cfg.preset, cfg.class_imbalance):
            _LOG.warning(
                "You overrode %s=%s under preset=%s; %s. The override will be applied as-is.",
                fname,
                override,
                cfg.preset,
                LOCKED_OFF[cfg.preset][fname],
            )
        base[fname] = override
    return ResolvedSemanticLoss(**base)


def dump_semantic_loss_bundle(cfg: SemanticLossConfig) -> dict[str, Any]:
    resolved = resolve(cfg)
    return {
        "preset": cfg.preset,
        "class_imbalance": cfg.class_imbalance,
        "resolved": {
            "sem_family": resolved.sem_family,
            "w_ce": resolved.w_ce,
            "w_region": resolved.w_region,
            "focal_gamma": resolved.focal_gamma,
            "focal_alpha": resolved.focal_alpha,
            "tversky_alpha": resolved.tversky_alpha,
            "tversky_gamma": resolved.tversky_gamma,
            "boundary_weight": resolved.boundary_weight,
        },
        "term_classes": {"region": _SEM_TERM_CLASS_NAMES[resolved.sem_family]},
        "source": cfg.source,
        "query_reduce": cfg.query_reduce,
        "background_logit": cfg.background_logit,
        "library_version": _LIB_VERSION or "unknown",
    }
