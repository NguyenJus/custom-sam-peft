"""Patch: replace sam3's text-pooling helpers to honor prompt dtype instead of fp32.

See src/custom_sam_peft/models/sam3.py::_patch_text_pool_dtype for full rationale.
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
    """Replace sam3's text-pooling helpers to honor prompt dtype instead of fp32.

    Process-wide patch (does not use *model* or *runtime*); idempotent.
    See ``_patch_text_pool_dtype`` in models/sam3.py for the full rationale.
    """
    # encoder.pool_text_feat (module-level function)
    try:
        import sam3.model.encoder as _encoder_mod

        if not getattr(_encoder_mod, "_custom_sam_peft_pool_text_feat_dtype_patched", False):

            def _pool_text_feat_dtype_aware(prompt, prompt_mask, pool_with_mask):  # type: ignore[no-untyped-def]
                if not pool_with_mask:
                    return prompt.mean(dim=0)
                if prompt_mask.dim() != 2:
                    raise ValueError(
                        f"pool_text_feat: prompt_mask.dim() must be 2; got {prompt_mask.dim()}"
                    )
                is_valid = (~prompt_mask).to(dtype=prompt.dtype).permute(1, 0)[..., None]
                num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
                pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
                return pooled_text

            _encoder_mod.pool_text_feat = _pool_text_feat_dtype_aware
            _encoder_mod._custom_sam_peft_pool_text_feat_dtype_patched = True
            logger.info("Patched sam3.model.encoder.pool_text_feat for dtype-aware text pooling.")
    except ImportError:
        # sam3 not installed in this environment; nothing to patch.
        pass

    # DotProductScoring.mean_pool_text (method on a class)
    try:
        import sam3.model.model_misc as _mm_mod

        DotProductScoring = _mm_mod.DotProductScoring
        if not getattr(DotProductScoring, "_custom_sam_peft_mean_pool_text_dtype_patched", False):

            def _mean_pool_text_dtype_aware(self, prompt, prompt_mask):  # type: ignore[no-untyped-def]
                is_valid = (~prompt_mask).to(dtype=prompt.dtype).permute(1, 0)[..., None]
                num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
                pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
                return pooled_prompt

            DotProductScoring.mean_pool_text = _mean_pool_text_dtype_aware
            DotProductScoring._custom_sam_peft_mean_pool_text_dtype_patched = True
            logger.info(
                "Patched sam3.model.model_misc.DotProductScoring.mean_pool_text "
                "for dtype-aware text pooling."
            )
    except ImportError:
        # sam3 not installed in this environment; nothing to patch.
        pass
