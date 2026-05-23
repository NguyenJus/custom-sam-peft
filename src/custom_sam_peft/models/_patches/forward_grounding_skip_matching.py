"""Patch: skip sam3's training-mode matching side-effect when find_target is None.

See src/custom_sam_peft/models/sam3.py::_patch_forward_grounding_skip_matching_on_none_target
for full rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Neutralize sam3's training-mode matching side-effect when ``find_target`` is ``None``.

    Per-instance MethodType rebind on back_convert and _compute_matching; idempotent.
    See ``_patch_forward_grounding_skip_matching_on_none_target`` in models/sam3.py.
    """
    from types import MethodType

    if getattr(model, "_custom_sam_peft_skip_matching_on_none_target_patched", False):
        return
    if not hasattr(model, "back_convert") or not hasattr(model, "_compute_matching"):
        return

    orig_back_convert = model.back_convert
    orig_compute_matching = model._compute_matching

    def _back_convert_none_safe(self, targets, _orig=orig_back_convert):  # type: ignore[no-untyped-def]
        if targets is None:
            return None
        return _orig(targets)

    def _compute_matching_none_safe(self, out, targets, _orig=orig_compute_matching):  # type: ignore[no-untyped-def]
        if targets is None:
            return None
        return _orig(out, targets)

    model.back_convert = MethodType(_back_convert_none_safe, model)  # type: ignore[assignment]
    model._compute_matching = MethodType(_compute_matching_none_safe, model)  # type: ignore[assignment]
    model._custom_sam_peft_skip_matching_on_none_target_patched = True  # type: ignore[assignment]
    logger.info(
        "Patched sam3.Sam3Image.{back_convert,_compute_matching} to short-circuit "
        "on find_target=None; training-mode forward_grounding now bypasses sam3's "
        "internal matching side-effect (we run our own matcher in losses.total_loss)."
    )
