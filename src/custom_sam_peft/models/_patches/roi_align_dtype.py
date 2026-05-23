"""Patch: wrap torchvision.ops.roi_align to handle bf16 and dtype skew.

See src/custom_sam_peft/models/sam3.py::_patch_roi_align_dtype for full rationale.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

import torch
from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Wrap ``torchvision.ops.roi_align`` to handle bf16-incompatible kernels.

    Process-wide patch (does not use *model* or *runtime*); idempotent.
    See ``_patch_roi_align_dtype`` in models/sam3.py for the full rationale.
    """
    import torchvision.ops as tvo  # type: ignore[import-untyped]

    tvo_ra_mod = sys.modules["torchvision.ops.roi_align"]

    if getattr(tvo, "_custom_sam_peft_roi_align_dtype_patched", False):
        return
    _original = tvo.roi_align

    def _roi_align_dtype_aware(input, boxes, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_dtype = input.dtype
        if original_dtype == torch.bfloat16:
            input = input.float()
            if isinstance(boxes, (list, tuple)):
                boxes = type(boxes)(b.float() for b in boxes)
            elif hasattr(boxes, "float"):
                boxes = boxes.float()
        else:
            if isinstance(boxes, (list, tuple)):
                boxes = type(boxes)(
                    b.to(dtype=input.dtype) if b.dtype != input.dtype else b for b in boxes
                )
            elif hasattr(boxes, "dtype") and boxes.dtype != input.dtype:
                boxes = boxes.to(dtype=input.dtype)
        out = _original(input, boxes, *args, **kwargs)
        if out.dtype != original_dtype:
            out = out.to(dtype=original_dtype)
        return out

    tvo.roi_align = _roi_align_dtype_aware
    tvo_ra_mod.roi_align = _roi_align_dtype_aware  # type: ignore[attr-defined]
    tvo._custom_sam_peft_roi_align_dtype_patched = True
    logger.info(
        "Patched torchvision.ops.roi_align (functional + RoIAlign class) "
        "for bf16-safe execution; bf16 inputs run via fp32 upcast."
    )
