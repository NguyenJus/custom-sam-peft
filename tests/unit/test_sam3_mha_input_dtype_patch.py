"""CPU unit tests for _patch_mha_input_dtype.

Covers the MHA input-dtype backstop described at
src/custom_sam_peft/models/sam3.py::_patch_mha_input_dtype.

The patch hooks every nn.MultiheadAttention (and sam3's custom MHA, if
present) and casts query/key/value to the module's first-parameter dtype
before forward, preventing upstream fp32 leaks from crashing into the
bf16 in_proj_weight inside the MHA's F.linear path.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
from torch import nn

from custom_sam_peft.models.sam3 import _patch_mha_input_dtype


def _make_bf16_mha(embed_dim: int = 8, num_heads: int = 2) -> nn.MultiheadAttention:
    """Build an nn.MultiheadAttention whose params are bf16."""
    m = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    m = m.to(dtype=torch.bfloat16)
    return m


def test_positional_fp32_query_key_value_cast_to_bf16() -> None:
    """All three positional tensor args are cast to the MHA's bf16 weight dtype."""
    mha = _make_bf16_mha()
    _patch_mha_input_dtype(mha)
    seen: dict[str, torch.dtype] = {}

    original_forward = mha.forward

    def _capturing_forward(q, k, v, *args, **kwargs):
        seen["q"] = q.dtype
        seen["k"] = k.dtype
        seen["v"] = v.dtype
        return original_forward(q, k, v, *args, **kwargs)

    mha.forward = _capturing_forward  # type: ignore[assignment]

    q = torch.zeros(1, 4, 8, dtype=torch.float32)
    k = torch.zeros(1, 4, 8, dtype=torch.float32)
    v = torch.zeros(1, 4, 8, dtype=torch.float32)
    mha(q, k, v)

    assert seen["q"] == torch.bfloat16
    assert seen["k"] == torch.bfloat16
    assert seen["v"] == torch.bfloat16


def test_keyword_qkv_also_cast() -> None:
    """When q/k/v are passed by keyword, they're cast just like positional."""
    mha = _make_bf16_mha()
    _patch_mha_input_dtype(mha)
    seen: dict[str, torch.dtype] = {}

    original_forward = mha.forward

    def _capturing_forward(query, key, value, *args, **kwargs):
        seen["q"] = query.dtype
        seen["k"] = key.dtype
        seen["v"] = value.dtype
        return original_forward(query, key, value, *args, **kwargs)

    mha.forward = _capturing_forward  # type: ignore[assignment]

    q = torch.zeros(1, 4, 8, dtype=torch.float32)
    k = torch.zeros(1, 4, 8, dtype=torch.float32)
    v = torch.zeros(1, 4, 8, dtype=torch.float32)
    mha(query=q, key=k, value=v)

    assert seen["q"] == torch.bfloat16
    assert seen["k"] == torch.bfloat16
    assert seen["v"] == torch.bfloat16


def test_matching_dtype_passthrough() -> None:
    """When input already matches module dtype, the hook is a no-op (no copy)."""
    mha = _make_bf16_mha()
    _patch_mha_input_dtype(mha)
    seen: dict[str, Any] = {}

    original_forward = mha.forward

    def _capturing_forward(q, k, v, *args, **kwargs):
        seen["q_id"] = id(q)
        return original_forward(q, k, v, *args, **kwargs)

    mha.forward = _capturing_forward  # type: ignore[assignment]

    q = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    orig_id = id(q)
    mha(q, q, q)
    assert seen["q_id"] == orig_id, "no-op path must NOT clone tensors"


def test_masks_and_scalars_left_untouched() -> None:
    """attn_mask / need_weights and other non-qkv args are not cast."""
    mha = _make_bf16_mha()
    _patch_mha_input_dtype(mha)
    captured: dict[str, Any] = {}

    original_forward = mha.forward

    def _capturing_forward(q, k, v, *args, **kwargs):
        captured.update(kwargs)
        return original_forward(q, k, v, *args, **kwargs)

    mha.forward = _capturing_forward  # type: ignore[assignment]

    q = torch.zeros(1, 4, 8, dtype=torch.float32)
    attn_mask = torch.zeros(4, 4, dtype=torch.bool)
    mha(q, q, q, attn_mask=attn_mask, need_weights=False)

    assert captured["attn_mask"].dtype == torch.bool, "bool mask must NOT be cast"
    assert captured["need_weights"] is False


def test_idempotency() -> None:
    """Applying the patch twice installs at most one hook per module."""
    mha = _make_bf16_mha()
    _patch_mha_input_dtype(mha)
    hooks_after_first = len(mha._forward_pre_hooks)
    _patch_mha_input_dtype(mha)
    hooks_after_second = len(mha._forward_pre_hooks)
    assert hooks_after_first == hooks_after_second
    assert getattr(mha, "_custom_sam_peft_mha_input_dtype_patched", False) is True


def test_module_without_mha_is_silent() -> None:
    """Models with no MHA modules are processed without error."""
    m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    _patch_mha_input_dtype(m)  # must not raise
    for submodule in m.modules():
        assert not getattr(submodule, "_custom_sam_peft_mha_input_dtype_patched", False)


def test_sam3_custom_mha_also_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """sam3.model.model_misc.MultiheadAttention is also covered.

    The patch imports sam3's custom MHA lazily and adds it to the
    isinstance check alongside torch's built-in MHA. We fake-install a
    sam3.model.model_misc with a sentinel class to verify the contract.
    """
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3_model = types.ModuleType("sam3.model")
    fake_sam3_model_misc = types.ModuleType("sam3.model.model_misc")

    class _Sam3CustomMHA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_weight = nn.Parameter(torch.zeros(24, 8, dtype=torch.bfloat16))
            self.out_proj = nn.Linear(8, 8).to(dtype=torch.bfloat16)
            self.received: dict[str, torch.dtype] = {}

        def forward(self, query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.received["q"] = query.dtype
            self.received["k"] = key.dtype
            self.received["v"] = value.dtype
            return query

    fake_sam3_model_misc.MultiheadAttention = _Sam3CustomMHA
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model", fake_sam3_model)
    monkeypatch.setitem(sys.modules, "sam3.model.model_misc", fake_sam3_model_misc)

    sam3_mha = _Sam3CustomMHA()
    _patch_mha_input_dtype(sam3_mha)

    q = torch.zeros(1, 4, 8, dtype=torch.float32)
    sam3_mha(q, q, q)

    assert sam3_mha.received["q"] == torch.bfloat16
    assert sam3_mha.received["k"] == torch.bfloat16
    assert sam3_mha.received["v"] == torch.bfloat16


def test_degrades_to_torch_only_without_sam3(monkeypatch: pytest.MonkeyPatch) -> None:
    """When sam3.model.model_misc is unimportable, torch's MHA is still patched."""
    monkeypatch.setitem(sys.modules, "sam3.model.model_misc", None)

    mha = _make_bf16_mha()
    _patch_mha_input_dtype(mha)
    assert getattr(mha, "_custom_sam_peft_mha_input_dtype_patched", False) is True
