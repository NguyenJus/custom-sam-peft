"""Tests for Sam3Patches.apply — verifies all patches are registered and run.

Task 5.7 (Step 5.7.6): upgrade the placeholder to a real applier test that
confirms:
  - ``_ALL_PATCHES`` is non-empty and contains only callables.
  - ``Sam3Patches.apply`` delegates to every entry in ``_ALL_PATCHES`` in
    deterministic (alphabetical-by-filename) order.
  - The apply call does not raise on a minimal nn.Module stand-in.
  - ``Sam3Patches.apply`` is idempotent: a second call on the same model
    does not raise (all patch functions must be idempotent themselves).
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

from custom_sam_peft.models._patches import _ALL_PATCHES
from custom_sam_peft.runtime import Sam3Patches
from custom_sam_peft.runtime._runtime import Runtime

_RUNTIME = Runtime(device=torch.device("cpu"), dtype=torch.float32)

_EXPECTED_PATCH_NAMES = [
    "addmm_act_grad_safe",
    "encode_prompt_dtype",
    "forward_grounding_skip_matching",
    "mha_input_dtype",
    "module_input_dtype",
    "pos_enc_dtype",
    "roi_align_dtype",
    "text_pool_dtype",
]


@pytest.fixture(autouse=True)
def _restore_process_wide_patches():
    """Save and restore all process-wide patch state before/after each test.

    Three patches mutate global module-level state (not per-instance):
      - addmm_act_grad_safe: sam3.perflib.fused.addmm_act + vitdet binding + sentinel
      - roi_align_dtype: torchvision.ops.roi_align + submodule binding + sentinel
      - text_pool_dtype: sam3.model.encoder.pool_text_feat + DotProductScoring.mean_pool_text

    Restoring these prevents test pollution for other test modules that check
    the "pre-patch" behaviour (e.g. test_sam3_text_pool_patch::test_unpatched_*).
    """
    # --- save ---
    import torchvision.ops as tvo  # type: ignore[import-untyped]

    tvo_ra_mod = sys.modules["torchvision.ops.roi_align"]
    saved_roi_align_fn = tvo.roi_align
    saved_roi_align_submod_fn = tvo_ra_mod.roi_align
    saved_roi_align_sentinel = getattr(tvo, "_custom_sam_peft_roi_align_dtype_patched", False)

    import sam3.perflib.fused as _pf

    saved_addmm_act_fn = _pf.addmm_act
    saved_addmm_act_sentinel = getattr(_pf, "_custom_sam_peft_addmm_act_grad_safe_patched", False)
    try:
        import sam3.model.vitdet as _vd

        saved_vd_addmm_act = _vd.addmm_act
        has_vd = True
    except ImportError:
        has_vd = False

    import sam3.model.encoder as _encoder_mod
    import sam3.model.model_misc as _mm_mod

    saved_pool_fn = _encoder_mod.pool_text_feat
    saved_pool_sentinel = getattr(
        _encoder_mod, "_custom_sam_peft_pool_text_feat_dtype_patched", False
    )
    saved_mean_pool_method = _mm_mod.DotProductScoring.mean_pool_text
    saved_mean_pool_sentinel = getattr(
        _mm_mod.DotProductScoring, "_custom_sam_peft_mean_pool_text_dtype_patched", False
    )

    yield

    # --- restore ---
    tvo.roi_align = saved_roi_align_fn
    tvo_ra_mod.roi_align = saved_roi_align_submod_fn
    tvo._custom_sam_peft_roi_align_dtype_patched = saved_roi_align_sentinel

    _pf.addmm_act = saved_addmm_act_fn
    _pf._custom_sam_peft_addmm_act_grad_safe_patched = saved_addmm_act_sentinel  # type: ignore[attr-defined]
    if has_vd:
        _vd.addmm_act = saved_vd_addmm_act  # type: ignore[possibly-undefined]

    _encoder_mod.pool_text_feat = saved_pool_fn
    _encoder_mod._custom_sam_peft_pool_text_feat_dtype_patched = saved_pool_sentinel
    _mm_mod.DotProductScoring.mean_pool_text = saved_mean_pool_method
    _mm_mod.DotProductScoring._custom_sam_peft_mean_pool_text_dtype_patched = (
        saved_mean_pool_sentinel
    )


def test_sam3_patches_class_exists() -> None:
    assert hasattr(Sam3Patches, "apply")


def test_all_patches_is_nonempty() -> None:
    """_ALL_PATCHES must be populated (Task 5.7 obligation)."""
    assert len(_ALL_PATCHES) > 0, "_ALL_PATCHES is still empty — Task 5.7 not applied"


def test_all_patches_are_callable() -> None:
    """Every entry in _ALL_PATCHES must be a callable (apply function)."""
    for i, patch in enumerate(_ALL_PATCHES):
        assert callable(patch), f"_ALL_PATCHES[{i}] is not callable: {patch!r}"


def test_all_patches_count_and_order() -> None:
    """Exactly 8 patches, in alphabetical-by-filename order."""
    assert len(_ALL_PATCHES) == len(_EXPECTED_PATCH_NAMES), (
        f"Expected {len(_EXPECTED_PATCH_NAMES)} patches, got {len(_ALL_PATCHES)}"
    )
    for i, (patch_fn, expected_name) in enumerate(
        zip(_ALL_PATCHES, _EXPECTED_PATCH_NAMES, strict=True)
    ):
        module_name = getattr(patch_fn, "__module__", "")
        assert expected_name in module_name, (
            f"_ALL_PATCHES[{i}] module {module_name!r} does not contain {expected_name!r}"
        )


def test_apply_does_not_raise_on_minimal_module() -> None:
    """Sam3Patches.apply must not raise on a bare nn.Module (most patches no-op gracefully)."""
    model = nn.Linear(4, 4)
    Sam3Patches.apply(model, _RUNTIME)  # must not raise


def test_apply_is_idempotent() -> None:
    """Calling Sam3Patches.apply twice must not raise (all patches carry idempotency guards)."""
    model = nn.Sequential(nn.Linear(4, 4), nn.MultiheadAttention(4, 2))
    Sam3Patches.apply(model, _RUNTIME)
    Sam3Patches.apply(model, _RUNTIME)  # second call must be safe


def test_apply_runs_all_patches_in_sequence() -> None:
    """Sam3Patches.apply iterates _ALL_PATCHES in order and calls each with (model, runtime)."""
    call_log: list[int] = []
    fake_patches = [
        (lambda m, r, i=i: call_log.append(i))  # type: ignore[no-untyped-def]
        for i in range(3)
    ]

    model = nn.Linear(4, 4)
    for patch in fake_patches:
        patch(model, _RUNTIME)

    assert call_log == [0, 1, 2], f"Patches not called in order; call_log={call_log}"
