# SAM-3 Patches

This directory holds in-process monkey-patches we apply to the upstream SAM-3
codebase. Each patch is narrow, targeted, and exists because a real failure
mode (dtype mismatch, autograd shape error, wrong dispatch path) surfaced in
training or eval against the pinned upstream checkpoint.

`models/sam3.py::load_sam31` wires each patch into the wrapper's
`_apply_patches` step. The patches are import-side-effect-free until that
function calls them.

## Patch index

| File | What it patches |
| --- | --- |
| `addmm_act_grad_safe.py` | Guards `addmm` autograd path against an upstream activation-grad shape mismatch. |
| `encode_prompt_dtype.py` | Forces prompt-encoder activations to the wrapper's compute dtype to prevent fp16/bf16 cast mismatches. |
| `forward_grounding_skip_matching.py` | Skips the upstream grounding matcher path that we replace with our own Hungarian matcher. |
| `mha_input_dtype.py` | Casts MHA inputs to a consistent dtype across Q/K/V projections. |
| `module_input_dtype.py` | Generic input-dtype harmonizer for modules that drop kwargs through. |
| `pos_enc_dtype.py` | Aligns positional-encoding dtype with the surrounding activation dtype. |
| `roi_align_dtype.py` | Forces ROI-Align inputs to fp32 (kernel only supports fp32; see `2026-05-22-fix-roi-align-dtype-mismatch.md`). |
| `text_pool_dtype.py` | Aligns text-pool projection dtype with the text-encoder output. |

## When SAM-3 bumps

Whenever the pinned SAM-3 checkpoint or vendored source version changes,
walk through this checklist before merging the bump:

1. Re-run `tests/gpu/` against the new SAM-3 checkpoint.
2. For each patch in this directory: open the corresponding upstream source
   file (`vendor/sam3/...` or the pinned pip dep), confirm the line numbers
   and function signatures the patch targets still exist.
3. If a target moved: update the patch's line / signature reference. If a
   target was removed: open an issue tagged `sam3-bump` to delete the patch.
4. Confirm `models/sam3.py::load_sam31` still wires each patch into the
   wrapper's `_apply_patches` step.
5. Update the SAM-3 checkpoint SHA pin in
   `src/custom_sam_peft/presets.py::_current_sam3_checkpoint_sha` (the
   analytic VRAM cache uses this to invalidate prior calibrations).
