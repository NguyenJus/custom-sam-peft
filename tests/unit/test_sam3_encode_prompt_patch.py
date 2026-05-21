"""CPU unit tests for _patch_encode_prompt_dtype.

Covers the bf16/fp32 mismatch fix described at
src/custom_sam_peft/models/sam3.py::_patch_encode_prompt_dtype:
when sam3's _encode_prompt cats an fp32 zero-length visual_prompt_embed with
bf16 txt_feats/geo_feats, torch.cat type-promotes the result to fp32 and the
downstream cross-attn KV projection (bf16 weight) explodes.

These tests use a fake stand-in for sam3.Sam3Image that exposes the exact
attribute name (`_encode_prompt`) and dtype semantics the patch relies on,
so they run on CPU without the gated checkpoint.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from custom_sam_peft.models.sam3 import _patch_encode_prompt_dtype


class _FakeSam3Image(nn.Module):
    """Minimal stand-in for sam3.Sam3Image with the attributes the patch reads."""

    def __init__(self, param_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        # _patch_encode_prompt_dtype reads `next(model.parameters()).dtype`.
        self.linear = nn.Linear(4, 4).to(dtype=param_dtype)

    def _encode_prompt(
        self, *_args: Any, **_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Reproduce sam3's bug: cat a bf16 + fp32-zero-length tensor → result is fp32.
        txt = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
        geo = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
        visual = torch.zeros(0, 1, 4)  # NO dtype= → fp32
        prompt = torch.cat([txt, geo, visual], dim=0)
        prompt_mask = torch.zeros(1, prompt.shape[0], dtype=torch.bool)
        return prompt, prompt_mask, {}


def test_unpatched_prompt_is_fp32() -> None:
    """Pin the bug: without the patch, _encode_prompt returns fp32 prompt."""
    m = _FakeSam3Image(param_dtype=torch.bfloat16)
    prompt, _, _ = m._encode_prompt()
    assert prompt.dtype == torch.float32


def test_patched_prompt_matches_model_dtype_bf16() -> None:
    m = _FakeSam3Image(param_dtype=torch.bfloat16)
    _patch_encode_prompt_dtype(m)
    prompt, _, _ = m._encode_prompt()
    assert prompt.dtype == torch.bfloat16


def test_patched_prompt_is_noop_in_fp32() -> None:
    """When the model is fp32 the cast is a no-op (prompt was already fp32)."""
    m = _FakeSam3Image(param_dtype=torch.float32)
    _patch_encode_prompt_dtype(m)
    prompt, _, _ = m._encode_prompt()
    assert prompt.dtype == torch.float32


def test_idempotency() -> None:
    """Calling the patcher twice rebinds at most once."""
    m = _FakeSam3Image(param_dtype=torch.bfloat16)
    _patch_encode_prompt_dtype(m)
    bound1 = m._encode_prompt
    _patch_encode_prompt_dtype(m)
    bound2 = m._encode_prompt
    assert bound1 is bound2
    assert getattr(m, "_custom_sam_peft_encode_prompt_dtype_patched", False) is True


def test_missing_encode_prompt_is_silent() -> None:
    """Models without _encode_prompt are skipped, not errored."""
    m = nn.Linear(4, 4)
    _patch_encode_prompt_dtype(m)  # must not raise
    assert not hasattr(m, "_custom_sam_peft_encode_prompt_dtype_patched")


def test_prompt_mask_passthrough() -> None:
    """The mask returned by the wrapper is the same object as the original."""
    m = _FakeSam3Image(param_dtype=torch.bfloat16)
    _patch_encode_prompt_dtype(m)
    _, mask, _ = m._encode_prompt()
    assert mask.dtype == torch.bool


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_patch_handles_half_precision_dtypes(dtype: torch.dtype) -> None:
    """The patch picks up the correct target dtype from model parameters."""
    m = _FakeSam3Image(param_dtype=dtype)
    _patch_encode_prompt_dtype(m)
    prompt, _, _ = m._encode_prompt()
    assert prompt.dtype == dtype
