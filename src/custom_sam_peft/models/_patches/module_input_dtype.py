"""Patch: generic fp-input-dtype backstop on every dtype-sensitive submodule.

See src/custom_sam_peft/models/sam3.py::_patch_module_input_dtype for full rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

import torch
from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Install a generic fp-input-dtype backstop on every dtype-sensitive submodule.

    Per-instance register_forward_pre_hook; idempotent.
    See ``_patch_module_input_dtype`` in models/sam3.py for the full rationale.
    """
    from custom_sam_peft.models.sam3 import _DTYPE_SENSITIVE_MODULE_TYPES

    def _input_dtype_hook(module: nn.Module, args: tuple[Any, ...]):  # type: ignore[no-untyped-def]
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None
        try:
            target_dtype = next(module.parameters()).dtype
        except StopIteration:
            return None
        if x.dtype == target_dtype:
            return None
        return (x.to(dtype=target_dtype), *args[1:])

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, _DTYPE_SENSITIVE_MODULE_TYPES):
            continue
        if getattr(submodule, "_custom_sam_peft_module_input_dtype_patched", False):
            continue
        submodule.register_forward_pre_hook(_input_dtype_hook)
        submodule._custom_sam_peft_module_input_dtype_patched = True  # type: ignore[assignment]
        patched_count += 1

    logger.info(
        "Patched %d dtype-sensitive modules (Linear/LayerNorm/Conv) with input-dtype hook.",
        patched_count,
    )
