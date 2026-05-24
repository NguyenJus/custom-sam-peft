"""Patch: cast query/key/value/attn_mask of every MHA module to the MHA's weight dtype.

See src/custom_sam_peft/models/sam3.py::_patch_mha_input_dtype for full rationale.

The hook also casts a FLOAT additive ``attn_mask`` to the module's weight dtype.
Under float16 (e.g. GTX 1080, CC 6.1), SAM3's decoder cross-attention
(``sam3/model/decoder.py:166``) passes a float32 additive ``attn_mask`` while the
query/key/value are fp16; ``F.scaled_dot_product_attention``
(``sam3/model/model_misc.py:397``) then raises
``RuntimeError: invalid dtype for bias - should match query's dtype``. The
``is_floating_point()`` guard casts the float additive bias but leaves BOOLEAN
masks untouched (bool masks are a valid SDPA input).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

import torch
from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Install a query/key/value input-dtype hook on every MHA submodule of *model*.

    Per-instance register_forward_pre_hook; idempotent.
    See ``_patch_mha_input_dtype`` in models/sam3.py for the full rationale.
    """
    mha_types: tuple[type[nn.Module], ...] = (nn.MultiheadAttention,)
    try:
        from sam3.model.model_misc import MultiheadAttention as _Sam3CustomMHA

        mha_types = (*mha_types, _Sam3CustomMHA)
    except ImportError:
        # sam3's custom MHA absent; fall back to torch's MultiheadAttention only.
        pass

    def _mha_input_dtype_hook(module, args, kwargs):  # type: ignore[no-untyped-def]
        try:
            target_dtype = next(module.parameters()).dtype
        except StopIteration:
            return None
        new_args = list(args)
        for i in range(min(3, len(new_args))):
            t = new_args[i]
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != target_dtype:
                new_args[i] = t.to(dtype=target_dtype)
        new_kwargs = dict(kwargs)
        for name in ("query", "key", "value", "attn_mask"):
            t = new_kwargs.get(name)
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != target_dtype:
                new_kwargs[name] = t.to(dtype=target_dtype)
        return tuple(new_args), new_kwargs

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, mha_types):
            continue
        if getattr(submodule, "_custom_sam_peft_mha_input_dtype_patched", False):
            continue
        submodule.register_forward_pre_hook(_mha_input_dtype_hook, with_kwargs=True)
        submodule._custom_sam_peft_mha_input_dtype_patched = True  # type: ignore[assignment]
        patched_count += 1

    logger.info(
        "Patched %d MultiheadAttention modules with query/key/value/attn_mask input-dtype hook.",
        patched_count,
    )
