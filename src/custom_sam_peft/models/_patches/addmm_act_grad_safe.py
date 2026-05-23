"""Patch: make sam3's addmm_act fused kernel grad-aware for LoRA training.

See src/custom_sam_peft/models/sam3.py::_patch_addmm_act_grad_safe for full rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Make sam3's ``addmm_act`` fused kernel grad-aware so LoRA training works.

    Process-wide patch (does not use *model* or *runtime*); idempotent.
    See ``_patch_addmm_act_grad_safe`` in models/sam3.py for the full rationale.
    """
    import sam3.model.vitdet as _vd
    import sam3.perflib.fused as _pf
    import torch
    import torch.nn.functional as F

    from custom_sam_peft.models.sam3 import _apply_activation, _is_linear4bit

    if getattr(_pf, "_custom_sam_peft_addmm_act_grad_safe_patched", False):
        return

    _orig = _pf.addmm_act

    def _addmm_act_grad_safe(activation, linear, mat1):  # type: ignore[no-untyped-def]
        if not torch.is_grad_enabled():
            if _is_linear4bit(linear):
                out = linear(mat1)
                return _apply_activation(activation, out)
            return _orig(activation, linear, mat1)
        x = linear(mat1)
        if activation in (nn.ReLU, F.relu):
            return F.relu(x)
        if activation in (nn.GELU, F.gelu):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    _pf.addmm_act = _addmm_act_grad_safe
    _vd.addmm_act = _addmm_act_grad_safe
    _pf._custom_sam_peft_addmm_act_grad_safe_patched = True
    logger.info(
        "Patched sam3.perflib.fused.addmm_act (and vitdet binding) for "
        "grad-aware forward; LoRA backbone fine-tuning now works on this "
        "sam3 commit."
    )
