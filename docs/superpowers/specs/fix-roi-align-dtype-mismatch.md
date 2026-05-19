# Spec: Fix roi_align dtype mismatch under bfloat16

**Date:** 2026-05-17  
**Branch:** `worktree-fix+colab-remaining-failures`  
**PR context:** PR #15 — 8/9 Colab T4 tests passing; this is the 1 remaining failure.

---

## Problem

`tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical`
fails on Colab T4 with `ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")`:

```text
RuntimeError: Expected tensor for argument #1 'input' to have the same type as
tensor for argument #2 'rois'; but type CUDABFloat16Type does not equal
torch.cuda.FloatTensor (while checking arguments for roi_align_forward_kernel)
```

Root cause: `sam3/model/geometry_encoders.py:652` calls:

```python
torchvision.ops.roi_align(img_feats, boxes_xyxy.float().transpose(0,1).unbind(0), self.roi_size)
```

`.float()` converts `boxes` to fp32. When the model is cast to bf16, `img_feats` is bf16. torchvision's C++ `roi_align_forward_kernel` rejects the mismatch.

---

## Constraints

1. **Do NOT modify sam3 source.** It is an installed third-party package.
2. **Do NOT introduce `torch.autocast`.** sam3's `decoder.py::forward_ffn` contains an explicit `torch.amp.autocast(enabled=False)` block; wrapping calls in autocast re-triggers a bf16/fp32 collision there. This was the root cause of the fix in PR #13; the constraint must be preserved.
3. **Mirror `_patch_pos_enc_dtype` precedent.** The new function follows the same idiom as the existing `_patch_pos_enc_dtype` (`src/esam3/models/sam3.py:255`, landed in PR #15 Task 4): idempotent sentinel attribute on the target module, private helper called from `load_sam31`, docstring explains rationale.
4. **PR-sized.** Exactly: one new function in `src/esam3/models/sam3.py`, one call in `load_sam31`, one new test file `tests/unit/test_sam3_roi_align_patch.py`.

---

## Solution

Monkey-patch `torchvision.ops.roi_align` with a dtype-aware wrapper that casts `boxes` to `input.dtype` when they differ, then delegates to the original. Use a module-level patch (not `MethodType`) because the call site is inside `_encode_boxes` — a 50-line method body where cloning to patch one line is brittle against upstream version bumps.

### Wrapper contract

- `input`: any tensor (C, H, W channels, batch via list rois or (B+1,5) format).
- `boxes`: either `list[Tensor]` (per-image (N_i, 4)) or `Tensor` (B+1, 5). sam3 uses the list form (`unbind(0)` produces a list). Both forms must be handled.
- When `boxes.dtype != input.dtype`, cast boxes to `input.dtype`.
- All other args/kwargs forwarded unchanged.
- Idempotent: gate on `getattr(torchvision.ops, "_esam3_roi_align_dtype_patched", False)`.

### Call site

In `load_sam31`, after `raw_model.to(dtype=...)` and before constructing `_Sam3ImageAdapter`.

---

## Test requirements

File: `tests/unit/test_sam3_roi_align_patch.py` — CPU-only, no GPU marker.

1. **Mismatch cast (list form, real kernel):** fp32 input + fp16 rois (list) — after patching, call succeeds and output dtype is fp32. (CPU fp32+fp16→fp32 cast verified working.)
2. **Mismatch cast (tensor form, mock):** fp32 input + fp16 rois as Tensor — wrapper casts before delegating. Use `unittest.mock.patch` on the original to capture the argument and assert `boxes.dtype == input.dtype`.
3. **Same-dtype passthrough:** fp32 input + fp32 rois — output identical to unpatched call (no unnecessary copy).
4. **Idempotency:** Call `_patch_roi_align_dtype()` twice; assert `torchvision.ops.roi_align` is the same object both times (no double-wrap).

bf16 is not tested on CPU because torchvision has no CPU bf16 roi_align kernel. The cast logic is validated via the fp16 path, which exercises the same code branch.

---

## Assumptions

- `torchvision` is installed in the project venv (confirmed: already a dependency).
- CPU fp32 roi_align kernel is available (confirmed working in local env).
- The implementer does not need to handle the `spatial_scale` / `aligned` kwargs — they pass through unchanged via `*args, **kwargs`.
