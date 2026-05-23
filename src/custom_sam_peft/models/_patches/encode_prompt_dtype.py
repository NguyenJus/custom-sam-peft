"""Patch: cast _encode_prompt's returned prompt to the model's parameter dtype.

See src/custom_sam_peft/models/sam3.py::_patch_encode_prompt_dtype for full rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Cast ``_encode_prompt``'s returned ``prompt`` to the model's parameter dtype.

    Per-instance MethodType rebind; idempotent.
    See ``_patch_encode_prompt_dtype`` in models/sam3.py for the full rationale.
    """
    from types import MethodType

    if not hasattr(model, "_encode_prompt"):
        return
    if getattr(model, "_custom_sam_peft_encode_prompt_dtype_patched", False):
        return
    original = model._encode_prompt
    target_dtype = next(model.parameters()).dtype

    def _encode_prompt_dtype_aware(self, *args, _orig=original, _dtype=target_dtype, **kwargs):  # type: ignore[no-untyped-def]
        prompt, prompt_mask, backbone_out = _orig(*args, **kwargs)
        if prompt.dtype != _dtype:
            prompt = prompt.to(dtype=_dtype)
        return prompt, prompt_mask, backbone_out

    model._encode_prompt = MethodType(_encode_prompt_dtype_aware, model)  # type: ignore[assignment]
    model._custom_sam_peft_encode_prompt_dtype_patched = True  # type: ignore[assignment]
    logger.info(
        "Patched SAM3Image._encode_prompt for dtype awareness (prompt cast to %s).",
        target_dtype,
    )
