"""Unit tests for _patch_roi_align_dtype — CPU-only, no GPU required."""

import sys

import pytest
import torch
import torchvision.ops

from custom_sam_peft.models.sam3 import _patch_roi_align_dtype

# torchvision.ops.__init__ re-exports `roi_align` as the FUNCTION under the
# same name as the submodule; reach the submodule via sys.modules.
_tvo_ra_mod = sys.modules["torchvision.ops.roi_align"]


@pytest.fixture(autouse=True)
def _restore_roi_align():
    """Restore torchvision.ops.roi_align (both names) and the sentinel after each test."""
    original_fn = torchvision.ops.roi_align
    original_submod_fn = _tvo_ra_mod.roi_align
    original_sentinel = getattr(torchvision.ops, "_custom_sam_peft_roi_align_dtype_patched", False)
    yield
    torchvision.ops.roi_align = original_fn
    _tvo_ra_mod.roi_align = original_submod_fn
    torchvision.ops._custom_sam_peft_roi_align_dtype_patched = original_sentinel


def test_list_rois_dtype_mismatch_real_kernel() -> None:
    """fp16 rois (list form) are cast to fp32 input dtype before the kernel call."""
    _patch_roi_align_dtype()
    input_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float16)]
    out = torchvision.ops.roi_align(input_fp32, boxes, output_size=2)
    assert out.dtype == torch.float32


def test_tensor_rois_dtype_mismatch_real_kernel() -> None:
    """fp16 rois (tensor form, shape (N,5)) are cast to fp32 input dtype."""
    _patch_roi_align_dtype()
    input_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    # (N, 5) form: [batch_idx, x1, y1, x2, y2]
    boxes_fp16 = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0]], dtype=torch.float16)
    out = torchvision.ops.roi_align(input_fp32, boxes_fp16, output_size=2)
    assert out.dtype == torch.float32


def test_same_dtype_passthrough() -> None:
    """Patched roi_align produces identical output to the unpatched version when dtypes match."""
    input_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)]

    # Capture unpatched result BEFORE installing the patch
    unpatched_out = torchvision.ops.roi_align(input_fp32, boxes, output_size=2)

    _patch_roi_align_dtype()
    patched_out = torchvision.ops.roi_align(input_fp32, boxes, output_size=2)

    assert torch.allclose(unpatched_out, patched_out)


def test_idempotency() -> None:
    """Calling _patch_roi_align_dtype twice does not double-wrap roi_align."""
    _patch_roi_align_dtype()
    after_first = torchvision.ops.roi_align

    _patch_roi_align_dtype()
    after_second = torchvision.ops.roi_align

    assert after_first is after_second


def test_bf16_input_upcast_to_fp32_then_cast_back() -> None:
    """bf16 input MUST be upcast to fp32 for the kernel and downcast on return.

    Regression: torchvision's CUDA roi_align kernel raises
    NotImplementedError: "roi_align_forward_kernel" not implemented for 'BFloat16'.
    The patch upcasts to fp32 for the kernel call, then casts the output back
    to the caller's original dtype so downstream Linear layers (whose
    `_patch_module_input_dtype` hook expects compute_dtype) don't see a fp32
    leak.
    """
    _patch_roi_align_dtype()
    input_bf16 = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)]
    out = torchvision.ops.roi_align(input_bf16, boxes, output_size=2)
    assert out.dtype == torch.bfloat16, "roi_align output must be cast back to caller's bf16 dtype"


def test_bf16_input_with_bf16_boxes_still_runs() -> None:
    """bf16 boxes also get upcast to fp32 alongside bf16 input."""
    _patch_roi_align_dtype()
    input_bf16 = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    boxes_bf16 = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.bfloat16)]
    out = torchvision.ops.roi_align(input_bf16, boxes_bf16, output_size=2)
    assert out.dtype == torch.bfloat16


def test_RoIAlign_class_routed_through_patch() -> None:
    """torchvision.ops.RoIAlign.forward must pick up the patched bf16-safe roi_align.

    sam3/model/decoder.py:289 instantiates `RoIAlign(...)` directly. Its
    forward looks up `roi_align` in its own module namespace, not in
    torchvision.ops. Patching only the package-level name leaves
    RoIAlign.forward bound to the original kernel; the submodule patch in
    `_patch_roi_align_dtype` covers it.
    """
    from torchvision.ops import RoIAlign

    _patch_roi_align_dtype()
    layer = RoIAlign(output_size=2, spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    input_bf16 = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)]
    # Pre-patch this would raise NotImplementedError for BFloat16.
    out = layer(input_bf16, boxes)
    assert out.dtype == torch.bfloat16


def test_fp16_input_kept_in_fp16_kernel_path() -> None:
    """fp16 input is supported by the torchvision kernel; don't upcast unnecessarily."""
    _patch_roi_align_dtype()
    input_fp16 = torch.zeros(1, 1, 4, 4, dtype=torch.float16)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)]
    out = torchvision.ops.roi_align(input_fp16, boxes, output_size=2)
    assert out.dtype == torch.float16


def test_idempotency_covers_both_namespaces() -> None:
    """Re-applying must not double-wrap either the package name or the submodule name."""
    _patch_roi_align_dtype()
    after_first_pkg = torchvision.ops.roi_align
    after_first_submod = _tvo_ra_mod.roi_align

    _patch_roi_align_dtype()
    after_second_pkg = torchvision.ops.roi_align
    after_second_submod = _tvo_ra_mod.roi_align

    assert after_first_pkg is after_second_pkg
    assert after_first_submod is after_second_submod
    assert after_first_pkg is after_first_submod, (
        "package-level and submodule-level rebinds must point at the same wrapper"
    )
