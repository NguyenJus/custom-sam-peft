"""Faithful, non-coerced bf16 numerics — the gpu_bf16 tier (RTX 5070 Ti, CC >= 8.0).

This is the test a Tesla T4 (CC 7.5) CANNOT run: below CC 8.0,
coerce_dtype_for_capability turns bfloat16 into float16, so bf16 numerics are
never exercised faithfully there (documents #139's finding). On CC >= 8.0 the
dtype is preserved and a real bf16 kernel path runs.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu_bf16


def test_bf16_not_coerced_on_cc_ge_80():
    """On CC >= 8.0 the dtype is NOT coerced; a real bf16 tensor stays bf16."""
    from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability

    cap = torch.cuda.get_device_capability()
    assert cap >= (8, 0)
    assert coerce_dtype_for_capability(torch.bfloat16, capability=cap) == torch.bfloat16
    x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    y = x @ x  # a real bf16 kernel path
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
