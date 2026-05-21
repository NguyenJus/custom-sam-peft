"""Unit tests for _patch_text_pool_dtype — CPU-only, exercises real sam3 modules."""

from __future__ import annotations

import pytest

# Real sam3 is in the test venv (required by the project at runtime).
import sam3.model.encoder as _encoder_mod
import sam3.model.model_misc as _mm_mod
import torch

from custom_sam_peft.models.sam3 import _patch_text_pool_dtype


@pytest.fixture(autouse=True)
def _restore_text_pool_patches():
    """Restore both patched names + sentinels after each test."""
    original_pool_fn = _encoder_mod.pool_text_feat
    original_pool_sentinel = getattr(
        _encoder_mod, "_custom_sam_peft_pool_text_feat_dtype_patched", False
    )
    original_mean_pool_method = _mm_mod.DotProductScoring.mean_pool_text
    original_mean_pool_sentinel = getattr(
        _mm_mod.DotProductScoring, "_custom_sam_peft_mean_pool_text_dtype_patched", False
    )
    yield
    _encoder_mod.pool_text_feat = original_pool_fn
    _encoder_mod._custom_sam_peft_pool_text_feat_dtype_patched = original_pool_sentinel
    _mm_mod.DotProductScoring.mean_pool_text = original_mean_pool_method
    _mm_mod.DotProductScoring._custom_sam_peft_mean_pool_text_dtype_patched = (
        original_mean_pool_sentinel
    )


def test_unpatched_pool_text_feat_promotes_bf16_to_fp32() -> None:
    """Pin the bug: without the patch, pool_text_feat returns fp32 for bf16 prompt."""
    prompt = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
    out = _encoder_mod.pool_text_feat(prompt, prompt_mask, pool_with_mask=True)
    assert out.dtype == torch.float32, (
        "pre-patch: sam3's hardcoded .float() is supposed to promote bf16 to fp32"
    )


def test_patched_pool_text_feat_preserves_bf16() -> None:
    """After patching, bf16 prompt yields bf16 pooled output."""
    _patch_text_pool_dtype()
    prompt = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
    out = _encoder_mod.pool_text_feat(prompt, prompt_mask, pool_with_mask=True)
    assert out.dtype == torch.bfloat16


def test_patched_pool_text_feat_fp32_passthrough() -> None:
    """fp32 prompt stays fp32; the cast to prompt.dtype is a no-op."""
    _patch_text_pool_dtype()
    prompt = torch.zeros(2, 1, 4, dtype=torch.float32)
    prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
    out = _encoder_mod.pool_text_feat(prompt, prompt_mask, pool_with_mask=True)
    assert out.dtype == torch.float32


def test_patched_pool_text_feat_pool_with_mask_false_branch() -> None:
    """The `pool_with_mask=False` branch still returns prompt.mean(dim=0)."""
    _patch_text_pool_dtype()
    prompt = torch.arange(8, dtype=torch.bfloat16).view(2, 1, 4)
    prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
    out = _encoder_mod.pool_text_feat(prompt, prompt_mask, pool_with_mask=False)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out, prompt.mean(dim=0))


def test_unpatched_mean_pool_text_promotes_bf16_to_fp32() -> None:
    """Pin the bug for DotProductScoring.mean_pool_text."""
    # Build a DotProductScoring without invoking __init__ (its constructor
    # creates nn.Linear modules we don't need for testing this method).
    instance = _mm_mod.DotProductScoring.__new__(_mm_mod.DotProductScoring)
    prompt = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
    out = instance.mean_pool_text(prompt, prompt_mask)
    assert out.dtype == torch.float32


def test_patched_mean_pool_text_preserves_bf16() -> None:
    """After patching, bf16 prompt yields bf16 pooled output."""
    _patch_text_pool_dtype()
    instance = _mm_mod.DotProductScoring.__new__(_mm_mod.DotProductScoring)
    prompt = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
    out = instance.mean_pool_text(prompt, prompt_mask)
    assert out.dtype == torch.bfloat16


def test_patches_numerically_equivalent_in_fp32() -> None:
    """For fp32 prompt, patched and original outputs must be byte-identical
    (the patch only changes the dtype of the cast, not the math)."""
    # Capture original BEFORE patching.
    pool_orig = _encoder_mod.pool_text_feat
    mean_pool_orig = _mm_mod.DotProductScoring.mean_pool_text

    torch.manual_seed(0)
    prompt_fp32 = torch.randn(8, 2, 16, dtype=torch.float32)
    prompt_mask = torch.tensor(
        [
            [False, False, False, True, True, True, True, True],
            [False, False, False, False, False, True, True, True],
        ],
        dtype=torch.bool,
    )

    out_pool_orig = pool_orig(prompt_fp32, prompt_mask, pool_with_mask=True)
    inst = _mm_mod.DotProductScoring.__new__(_mm_mod.DotProductScoring)
    out_mean_orig = mean_pool_orig(inst, prompt_fp32, prompt_mask)

    _patch_text_pool_dtype()

    out_pool_patched = _encoder_mod.pool_text_feat(prompt_fp32, prompt_mask, pool_with_mask=True)
    out_mean_patched = _mm_mod.DotProductScoring.mean_pool_text(inst, prompt_fp32, prompt_mask)

    assert torch.equal(out_pool_orig, out_pool_patched)
    assert torch.equal(out_mean_orig, out_mean_patched)


def test_idempotency() -> None:
    """Re-applying the patch does not double-wrap either site."""
    _patch_text_pool_dtype()
    first_pool_fn = _encoder_mod.pool_text_feat
    first_method = _mm_mod.DotProductScoring.mean_pool_text

    _patch_text_pool_dtype()
    second_pool_fn = _encoder_mod.pool_text_feat
    second_method = _mm_mod.DotProductScoring.mean_pool_text

    assert first_pool_fn is second_pool_fn
    assert first_method is second_method
    assert getattr(_encoder_mod, "_custom_sam_peft_pool_text_feat_dtype_patched", False)
    assert getattr(
        _mm_mod.DotProductScoring, "_custom_sam_peft_mean_pool_text_dtype_patched", False
    )
