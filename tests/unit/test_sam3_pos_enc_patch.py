"""CPU unit test for the PositionEmbeddingSine._encode_xy dtype-cast patch.

The patch lives in src/custom_sam_peft/models/sam3.py::_patch_pos_enc_dtype. It rebinds
the bound method on each PositionEmbeddingSine instance so its (pos_x, pos_y)
outputs are cast to the input tensor's dtype before returning.
"""

from __future__ import annotations

import pytest
import torch
from sam3.model.position_encoding import PositionEmbeddingSine
from torch import nn

from custom_sam_peft.models.sam3 import _patch_pos_enc_dtype


def test_pos_enc_patch_casts_outputs_to_input_dtype() -> None:
    pos_enc = PositionEmbeddingSine(num_pos_feats=8)

    # Pre-patch: outputs are fp32 regardless of input dtype.
    x = torch.randn(3, dtype=torch.bfloat16)
    y = torch.randn(3, dtype=torch.bfloat16)
    px_pre, py_pre = pos_enc._encode_xy(x, y)
    assert px_pre.dtype == torch.float32
    assert py_pre.dtype == torch.float32

    # Apply the patch via a wrapping nn.Module.
    holder = nn.Sequential(pos_enc)
    _patch_pos_enc_dtype(holder)

    # Post-patch: outputs honor the input dtype.
    px_post, py_post = pos_enc._encode_xy(x, y)
    assert px_post.dtype == torch.bfloat16, f"expected bf16, got {px_post.dtype}"
    assert py_post.dtype == torch.bfloat16, f"expected bf16, got {py_post.dtype}"


def test_pos_enc_patch_is_idempotent() -> None:
    pos_enc = PositionEmbeddingSine(num_pos_feats=8)
    holder = nn.Sequential(pos_enc)
    _patch_pos_enc_dtype(holder)
    first_bound = pos_enc._encode_xy
    _patch_pos_enc_dtype(holder)  # second call: must be a no-op
    second_bound = pos_enc._encode_xy
    assert first_bound is second_bound, (
        "second _patch_pos_enc_dtype call must not re-wrap an already-patched instance"
    )


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pos_enc_patch_preserves_numerical_content_for_fp32(
    input_dtype: torch.dtype,
) -> None:
    """For fp32 input, post-patch output must equal pre-patch output bitwise.

    For lower-precision dtypes we only check the dtype is honored
    (numerical content differs by the bf16/fp16 truncation, which is the
    intended behavior).
    """
    pos_enc = PositionEmbeddingSine(num_pos_feats=8)
    x = torch.tensor([0.1, 0.5, 0.9], dtype=input_dtype)
    y = torch.tensor([0.2, 0.6, 0.8], dtype=input_dtype)
    px_pre, py_pre = pos_enc._encode_xy(x, y)

    holder = nn.Sequential(pos_enc)
    _patch_pos_enc_dtype(holder)
    px_post, py_post = pos_enc._encode_xy(x, y)

    assert px_post.dtype == input_dtype
    assert py_post.dtype == input_dtype
    if input_dtype == torch.float32:
        assert torch.equal(px_pre, px_post)
        assert torch.equal(py_pre, py_post)
