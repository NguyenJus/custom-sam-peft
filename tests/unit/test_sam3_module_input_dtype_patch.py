"""CPU unit tests for _patch_module_input_dtype.

Covers the generic fp-input-dtype backstop described at
src/custom_sam_peft/models/sam3.py::_patch_module_input_dtype.  This hook is the
consumer-side defense for the cascading fp32/bf16 collisions seen on Colab
T4 when sam3 internal helpers (e.g. ``gen_sineembed_for_position``,
``get_valid_ratio``) leak fp32 tensors into bf16-weight nn.Linear /
LayerNorm / Conv modules and raise
``RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16``.

The tests construct minimal nn.Module stand-ins (no sam3, no checkpoint)
so they run on CPU without GPU or the gated SAM 3.1 weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from custom_sam_peft.models.sam3 import _patch_module_input_dtype


def test_unpatched_linear_raises_on_fp32_input_to_bf16_weight() -> None:
    """Pin the bug: without the patch, fp32 input + bf16 weight raises."""
    layer = nn.Linear(4, 4).to(dtype=torch.bfloat16)
    x = torch.zeros(2, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dtype"):
        layer(x)


def test_patched_linear_accepts_fp32_input() -> None:
    """After patching, fp32 input is silently cast to the weight dtype."""
    model = nn.Sequential(nn.Linear(4, 4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    x = torch.zeros(2, 4, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.bfloat16
    assert y.shape == (2, 4)


def test_patched_layernorm_accepts_fp32_input() -> None:
    """LayerNorm is also covered — fp32 input is cast to its bf16 weight dtype."""
    model = nn.Sequential(nn.LayerNorm(4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    x = torch.zeros(2, 4, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.bfloat16


def test_patched_conv2d_accepts_fp32_input() -> None:
    """Conv2d is also covered."""
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    x = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.bfloat16


def test_patch_is_noop_when_dtypes_already_match() -> None:
    """No cast happens when input dtype already matches weight dtype."""
    model = nn.Sequential(nn.Linear(4, 4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    y = model(x)
    assert y.dtype == torch.bfloat16


def test_patch_is_noop_in_fp32_model() -> None:
    """In an fp32 model the hook is a no-op for fp32 inputs."""
    model = nn.Sequential(nn.Linear(4, 4))  # default fp32
    _patch_module_input_dtype(model)
    x = torch.zeros(2, 4, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.float32


def test_idempotency_does_not_double_hook() -> None:
    """Calling the patcher twice attaches the hook at most once per module."""
    model = nn.Sequential(nn.Linear(4, 4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    linear = model[0]
    hooks_after_first = len(linear._forward_pre_hooks)
    _patch_module_input_dtype(model)
    hooks_after_second = len(linear._forward_pre_hooks)
    assert hooks_after_first == hooks_after_second
    assert getattr(linear, "_custom_sam_peft_module_input_dtype_patched", False) is True


def test_embedding_is_skipped() -> None:
    """nn.Embedding takes integer input; we must not hook it (would crash on .to(dtype=)).

    We don't include Embedding in the module-type set, so an Embedding inside the
    model should remain unhooked and continue to accept LongTensor input.
    """
    model = nn.Sequential(nn.Embedding(10, 4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    embedding = model[0]
    assert getattr(embedding, "_custom_sam_peft_module_input_dtype_patched", False) is False
    # And it still works on LongTensor input.
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    y = model(idx)
    assert y.dtype == torch.bfloat16


def test_integer_input_is_not_cast() -> None:
    """Even if a Linear is given an int tensor (shouldn't happen but defensive),
    the hook should pass it through without casting, letting the underlying
    op raise its native error rather than silently re-typing user data."""
    model = nn.Sequential(nn.Linear(4, 4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    x = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(RuntimeError):
        model(x)


def test_hook_survives_to_dtype_recast() -> None:
    """The hook is attached to the module, not parameters, so a subsequent
    ``.to(dtype=...)`` does not strip it.  This is important because real
    sam3 use re-casts the model after building."""
    model = nn.Sequential(nn.Linear(4, 4)).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    # Re-cast to fp16 — the hook now targets fp16 (derived at call time).
    model = model.to(dtype=torch.float16)
    x = torch.zeros(2, 4, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.float16


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_handles_half_precision_dtypes(dtype: torch.dtype) -> None:
    """Both bf16 and fp16 model dtypes are honored — target dtype is derived
    from the module's first parameter, not hardcoded."""
    model = nn.Sequential(nn.Linear(4, 4)).to(dtype=dtype)
    _patch_module_input_dtype(model)
    x = torch.zeros(2, 4, dtype=torch.float32)
    y = model(x)
    assert y.dtype == dtype


def test_parameterless_module_in_tree_is_skipped() -> None:
    """A bare nn.ReLU has no parameters; it's not in the iterated type-set
    either, so it should not be touched."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).to(dtype=torch.bfloat16)
    _patch_module_input_dtype(model)
    relu = model[1]
    assert getattr(relu, "_custom_sam_peft_module_input_dtype_patched", False) is False


def test_reproduces_gen_sineembed_failure_mode() -> None:
    """End-to-end mimic of the Colab failure: a producer builds an fp32
    tensor (sin/cos via fp32 arange) and feeds an MLP of bf16 Linears,
    which mirrors ``ref_point_head(query_sine_embed)`` in sam3 decoder.py:516.
    Without the patch this raises; with the patch it succeeds.
    """
    pos_tensor = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    num_feats = 4
    dim_t = torch.arange(num_feats, dtype=torch.float32)  # fp32, like sam3
    fp32_embed = (pos_tensor[..., :1] / dim_t).flatten(2)  # promotes to fp32

    ref_point_head = nn.Sequential(
        nn.Linear(fp32_embed.shape[-1], 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    ).to(dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="dtype"):
        ref_point_head(fp32_embed)

    _patch_module_input_dtype(ref_point_head)
    y = ref_point_head(fp32_embed)
    assert y.dtype == torch.bfloat16
