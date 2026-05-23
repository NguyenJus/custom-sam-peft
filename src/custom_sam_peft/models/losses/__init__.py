"""SAM 3.1 training losses — domain-aware preset-driven loss bundle.

Public API:
  - LossBundle, build_loss_bundle  (composer; spec §8)
  - resolve, ResolvedLosses        (resolver; spec §7)
  - PRESET_TABLE, LOCKED_OFF       (preset table; spec §5/§6)
  - dump_loss_bundle               (sidecar helper; spec §9)
  - total_loss                     (back-compat shim — see spec §8.6)
"""

from __future__ import annotations

from typing import Any

from custom_sam_peft.models.losses.compose import (
    LossBundle,
    build_loss_bundle,
)
from custom_sam_peft.models.losses.presets import (
    LOCKED_OFF,
    PRESET_TABLE,
    ResolvedLosses,
    dump_loss_bundle,
    resolve,
)


def total_loss(outputs: dict[str, Any], targets: Any, cfg: Any) -> dict[str, Any]:
    """Back-compat shim. Spec §8.6 — builds a fresh bundle per call.

    The two call sites in `train/loop.py` (lines 257, 278) continue to work
    unmodified. Phase D may replace this shim with a long-lived bundle on the
    trainer; if it does, this function becomes dead code and is removed.
    """
    return build_loss_bundle(resolve(cfg)).forward(outputs, targets)


__all__ = [
    "LOCKED_OFF",
    "PRESET_TABLE",
    "LossBundle",
    "ResolvedLosses",
    "build_loss_bundle",
    "dump_loss_bundle",
    "resolve",
    "total_loss",
]
