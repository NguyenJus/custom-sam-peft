"""Patch: enable sam3 ViT-Det per-block activation checkpointing.

Config-gated (cfg.gradient_checkpointing) — NOT registered in _ALL_PATCHES.
Invoked conditionally from models/sam3.py::_construct_raw_model (static entry
point) and from train/loop.py's OOM ladder (dynamic entry point).

This module currently implements the FLAG-FLIP half only: it sets
``use_act_checkpoint=True`` on every submodule that already exposes that
attribute (sam3's ViT-Det blocks, vitdet.py:982). sam3 self-checkpoints inside
the block forward via ``checkpoint.checkpoint(blk, x, use_reentrant=False)``,
which raised a recompute-metadata CheckpointError on T4 (issue #60 / #89). The
deterministic-autocast wrap that resolves that mismatch (Fix A) is added by the
Phase-1 task after the Phase-0 Colab T4 diagnostic classifies the divergence.

See models/_patches/README.md "When SAM-3 bumps" and the spec
docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

from torch import nn

logger = logging.getLogger(__name__)

# sam3's per-ViT-Det-block flag (vitdet.py:982). If a sam3 bump renames or
# removes this, the patch flips zero blocks and logs a loud warning.
_ACT_CHECKPOINT_ATTR = "use_act_checkpoint"
_SENTINEL_ATTR = "_custom_sam_peft_act_checkpoint_patched"


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Enable activation checkpointing on every exposing ViT-Det block.

    Idempotent via a per-module sentinel (mirrors module_input_dtype.py:46-49).
    Only flips the flag where sam3 already declared it; never injects the
    attribute onto unrelated modules.

    The ``runtime`` argument is unused by the flag-flip half but is part of the
    patch contract and is consumed by the deterministic-autocast wrap added in
    the Phase-1 fix task.
    """
    patched_count = 0
    for submodule in model.modules():
        if not hasattr(submodule, _ACT_CHECKPOINT_ATTR):
            continue
        if getattr(submodule, _SENTINEL_ATTR, False):
            continue
        setattr(submodule, _ACT_CHECKPOINT_ATTR, True)
        setattr(submodule, _SENTINEL_ATTR, True)
        patched_count += 1

    if patched_count == 0:
        logger.warning(
            "vit_act_checkpoint: found ZERO modules exposing %r. Either the "
            "model has no ViT-Det blocks (wrong model) or sam3 renamed the "
            "attribute (see vitdet.py:982 and _patches/README.md 'When SAM-3 "
            "bumps').",
            _ACT_CHECKPOINT_ATTR,
        )
    else:
        logger.info(
            "Enabled activation checkpointing on %d ViT-Det block(s).",
            patched_count,
        )
