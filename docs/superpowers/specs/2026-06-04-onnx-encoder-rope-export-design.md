# ONNX Export: Encoder RoPE Complex-Op Swap — Design Spec

- **Issue:** #279 — `csp export --to onnx` traces against `TinySam3Stub` but
  fails on the real SAM 3.1 model (encoder RoPE complex ops).
- **Stacks on:** PR #280 (branch `worktree-spec+onnx-export-77`, adds
  `csp export --to onnx`). This work extends
  `src/custom_sam_peft/export/onnx.py` and **merges into PR #280's branch.**
- **Worktree:** `/home/justin/projects/custom-sam-peft/.claude/worktrees/279-encoder-rope-export`
  (branch `279-encoder-rope-export`, already carries PR #280's commits
  `8100dd8`/`2e0e6e1`/`5527655`).
- **Status:** design LOCKED. This document captures it for implementation. No
  redesign.

---

## Summary

The SAM 3.1 ViT image-encoder's rotary position embedding (RoPE) uses
complex-tensor ops (`torch.view_as_complex` / `torch.view_as_real`) that the
TorchScript ONNX exporter cannot lower at **any** opset — ONNX has no complex
type. Export aborts with:

```
UnsupportedOperatorError: Exporting the operator 'aten::view_as_complex'
to ONNX opset version 17 is not supported
```

scoped to `sam3.model.vitdet.Attention._apply_rope -> apply_rotary_enc`.

sam3 already ships a **real-valued, mathematically-equivalent** RoPE path
(`apply_rotary_enc_real`, selected by `use_rope_real=True`). RoPE is
parameter-free, so swapping the encoder's `Attention` modules onto that path
changes **no learned weights** and is bit-exact (validated at fp32, see Spike
Findings). This spec adds an export-time swap that regenerates the RoPE tables
on the real path **before** the fp16 cast, guards equivalence two ways
(per-module + whole-encoder) **before** any trace/cast, fails loud on the
untraceable VE-RoPE variant, and ships CPU-only tests against the real
`vitdet.Attention`. GPU end-to-end parity (`--check`, `--use-onnx` round-trip)
is a **deferred, user-approved manual gate** documented as a runbook.

---

## Root cause (precise)

- The ViT encoder instantiates ~32 `sam3.model.vitdet.Attention` modules with
  `use_rope=True, use_rope_real=False, use_ve_rope=False`.
- `Attention._apply_rope(q, k)` (sam3 `model/vitdet.py:554`) asserts
  `freqs_cis is not None`, then branches on `use_rope_real`:
  - `False` (encoder default) → the **complex** `apply_rotary_enc(q, k, freqs_cis)`
    (`sam/rope.py:58`), whose body calls
    `torch.view_as_complex(...)` / `torch.view_as_real(...)` → **unsupported op.**
  - `True` → the **real** `apply_rotary_enc_real(q, k, freqs_cis_real, freqs_cis_imag)`
    (`sam/rope.py:92`), which uses only `* - + torch.stack` via
    `complex_mult` (`sam/rope.py:83`) — **no** `view_as_complex` / `torch.complex`.
    Confirmed by reading `sam/rope.py:83-116`.
- The decoder / multiplex transformer already defaults to `use_rope_real=True`
  (real-valued), so **the failure is ENCODER-ONLY.**
- `_setup_rope_freqs()` (sam3 `model/vitdet.py:480`) registers the complex
  `freqs_cis` buffer, and **when `use_rope_real` is set** *additionally*
  registers `freqs_cis_real = freqs_cis.real` and `freqs_cis_imag = freqs_cis.imag`
  (= cosθ / sinθ; `vitdet.py:549-552`). Because RoPE is parameter-free, calling
  `_setup_rope_freqs()` again with `use_rope_real=True` set is the correct,
  weight-preserving way to materialise those buffers.
- **Why the naive patch fails:** "set `use_rope_real=True`; register
  `freqs_cis.real`/`.imag`" is wrong if it runs *after* the existing fp16 cast
  (`merged.half()` at `onnx.py:206`) — by then `freqs_cis` has already been cast
  and the real/imag extraction is off. The correct approach **regenerates** the
  table via the module's own `_setup_rope_freqs()` with `use_rope_real=True`,
  **before** the cast, so the complex `freqs_cis` is intact when `.real`/`.imag`
  are extracted.

---

## Spike findings (already run on CPU — established facts)

Constructed a small real
`sam3.model.vitdet.Attention(dim=64, num_heads=4, use_rope=True,
use_rope_real=False, use_ve_rope=False, input_size=(8,8))`.

- Original `freqs_cis` is `complex64`, shape `(64, 8)`.
- After setting `module.use_rope_real = True` and calling `_setup_rope_freqs()`,
  `freqs_cis_real` / `freqs_cis_imag` register as `float32`, and `_apply_rope(q, k)`
  output is **BIT-EXACT** vs the pre-patch complex path: `max|Δq| = max|Δk| = 0.0`
  at fp32. The per-module guard tolerance can therefore be tight
  (atol/rtol ≈ 1e-5 is generous).
- Calling `.half()` *after* the patch leaves the now-inert `freqs_cis` buffer as
  `complex64` (NOT converted) with **no error**; `freqs_cis_real` / `_imag`
  become `float16`. The real `_apply_rope` branch never reads `freqs_cis` values
  (only asserts non-None), and the tracer does not reference it. → **Keep the
  complex buffer as-is.** It must NOT be set to None (the `_apply_rope` assert
  would fail) and needs no disposal.

These are stated as facts the implementation relies on, not items to re-verify.

---

## Approach

1. **The swap** — a new helper regenerates the RoPE table on sam3's real path
   for every in-scope encoder `Attention`, running inside `_merge_and_cast`
   **after** `merge_lora` and **before** `if fp16: merged.half()`.
2. **Fail-loud guard** — any module using the VE-RoPE variant
   (`use_rope and use_ve_rope`) raises `VeRopeUnsupportedError`; that variant has
   no real-valued equivalent and must not be silently left untraceable.
3. **Equivalence guards (both, fp32, pre-cast)** — a per-module original-vs-real
   forward check (load-bearing, isolates exactly the swapped op) and a
   whole-encoder check through the real attention stack (belt-and-suspenders).
   Both abort export **before** any trace or cast.

---

## Components

All additions live in `src/custom_sam_peft/export/onnx.py`. sam3 imports stay
**lazy/local** inside the helpers (consistent with onnx.py's existing
lazy-import pattern, e.g. `from custom_sam_peft.peft_adapters.lora import merge_lora`
inside `_merge_and_cast`).

### `_patch_encoder_rope_for_export(merged: nn.Module) -> int`

New module-level helper.

- Walk `merged.modules()`. Lazily import `sam3.model.vitdet.Attention` and select
  instances where `module.use_rope and not module.use_rope_real and not
  module.use_ve_rope`.
- For each in-scope module:
  1. **Snapshot** the complex `freqs_cis`:
     `freqs_cis_snapshot = module.freqs_cis.detach().clone()` (still `complex64`,
     pre-cast). Used by the per-module guard.
  2. `module.use_rope_real = True`.
  3. `module._setup_rope_freqs()` — sam3's own code re-registers the complex
     `freqs_cis` **and** the `freqs_cis_real` / `freqs_cis_imag` buffers
     (`vitdet.py:549-552`). **No learned weights are touched.**
  4. Run the **per-module equivalence guard** (see Equivalence guards) using
     `freqs_cis_snapshot`; raise `RopeEquivalenceError` on mismatch.
- Leave the leftover complex `freqs_cis` buffer in place (it is inert, survives
  `.half()`, and **cannot** be `None` — `_apply_rope` asserts non-None).
- Count and return the number of modules patched. **No `count > 0` assertion** —
  modules already on `use_rope_real=True` (decoder), with `use_rope=False`, or
  the `TinySam3Stub` (no vitdet RoPE attention → count `0`) are correctly
  skipped, and a `count==0` assert would break the stub path. Emit a
  debug/info log line with the count.

### `VeRopeUnsupportedError(RuntimeError)`

New error type, mirroring the style of the existing
`ExportParityError(RuntimeError)` (`onnx.py:241`). Raised by
`_patch_encoder_rope_for_export` (VE-RoPE detection) — see Error handling.

### `RopeEquivalenceError(RuntimeError)`

New error type, same style. Raised by either equivalence guard on tolerance
mismatch.

### Hook points in `_merge_and_cast`

Current relevant body (`onnx.py:201-210`):

```python
merge_lora(wrapper)                       # 201
adapter = wrapper.model                   # 202
merged = cast(nn.Module, adapter.model)   # 203
                                          #
if fp16:                                  # 205
    merged.half()                         # 206
    export_dtype = torch.float16          # 207
else:                                     # 208
    merged.float()                        # 209
    export_dtype = torch.float32          # 210
```

Insert, **between line 203 and line 205** (after `merged` is bound, before the
`if fp16:` cast block, while `freqs_cis` is still complex):

1. **VE-RoPE detection** (may raise `VeRopeUnsupportedError`).
2. **Whole-encoder guard, pre-half**: run one deterministic synthetic image
   through `_torch_encoder_feats(wrapper, img)` and cache the output
   (`encoder_ref_pre_patch`).
3. `n = _patch_encoder_rope_for_export(merged)` — performs the swap and the
   per-module guard for each module.
4. **Whole-encoder guard, post-patch**: re-run `_torch_encoder_feats(wrapper, img)`
   and compare against `encoder_ref_pre_patch` at the fp32 `_PARITY_TOL` band;
   raise `RopeEquivalenceError` on mismatch.

All four steps run **before** `merged.half()` / `merged.float()`. The existing
qlora/device handling below (`onnx.py:211-219`) is unchanged.

> Implementation note: the VE-RoPE scan and the whole-encoder
> reference-capture may live directly in `_merge_and_cast` or factor into a
> small private helper — the locked requirement is only the *ordering*
> (after `merge_lora`, before the cast) and that both equivalence guards run on
> fp32 buffers. The VE-RoPE detection MAY also be folded into the
> `_patch_encoder_rope_for_export` walk (single pass over `merged.modules()`),
> provided it is checked and raised before any module is mutated.

---

## Equivalence guards

The issue mandates an original-vs-real RoPE equivalence assertion with a
documented tolerance. We implement **both** of the following; both abort export
**before** any trace or cast.

### Why a separate guard is mandatory (patched-vs-patched rationale)

PR #280's existing `--check` parity (`_run_parity_check`, `onnx.py:245`) compares
the **patched** torch model against the **patched** ORT bundle
(patched-vs-patched). It therefore would **NOT** catch a RoPE-semantics
regression introduced by the swap itself — both sides would share the same
error. An **original-vs-regenerated** check is required, and it must run on the
unpatched complex path before the cast. Hence the two guards below.

### Per-module guard (load-bearing — isolates exactly the swapped op)

Inside `_patch_encoder_rope_for_export`'s loop, after `_setup_rope_freqs()`:

- Build shared random `q, k` of the module's head shape
  `(B, H, L, head_dim)` (deterministic generator). Lazily import
  `apply_rotary_enc` and `apply_rotary_enc_real` from `sam3.sam.rope`.
- **Reference (original complex path):**
  `apply_rotary_enc(q, k, freqs_cis=freqs_cis_snapshot)` — the pre-patch
  `complex64` table from the snapshot. Signature confirmed:
  `apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False)` (`rope.py:58`).
- **Candidate (real path):** the post-patch real branch — either
  `module._apply_rope(q, k)` (now routed through `use_rope_real=True`) or
  `apply_rotary_enc_real(q, k, freqs_cis_real=module.freqs_cis_real,
  freqs_cis_imag=module.freqs_cis_imag)` directly. Signature confirmed:
  `apply_rotary_enc_real(xq, xk, freqs_cis_real, freqs_cis_imag,
  repeat_freqs_k=False)` (`rope.py:92`).
- Assert `torch.allclose` on both the q and k outputs at a **tight** tolerance,
  **atol = rtol = 1e-5** (spike showed bit-exact at fp32; 1e-5 is generous).
- On mismatch raise `RopeEquivalenceError` with the module identity (e.g. its
  qualified name) and the observed max abs delta.

This guard runs on the fp32 complex `freqs_cis` snapshot vs the freshly
regenerated real buffers, isolating **exactly** the one op the swap changes.

### Whole-encoder guard (belt-and-suspenders)

In `_merge_and_cast`, around the patch call:

- Capture a deterministic synthetic image (fixed seed/shape — reuse the
  `_run_parity_check` input conventions: raw floats, `SAM3_IMAGE_SIZE`,
  `cfg.data.channels`, batch `_TRACE_B`).
- **Before** patching: `encoder_ref = _torch_encoder_feats(wrapper, img)` →
  cache the named boundary tensors (dict).
- **After** patching: `encoder_out = _torch_encoder_feats(wrapper, img)`.
- Compare every key with `torch.allclose` (or numpy `allclose`) at the **fp32**
  `_PARITY_TOL` band (`(atol, rtol) = (1e-3, 1e-3)`; `onnx.py:60`). This guard
  always runs pre-cast on fp32 buffers, so the fp32 band applies regardless of
  the export `--fp16` flag.
- On mismatch raise `RopeEquivalenceError` naming the drifting key and the band.

`_torch_encoder_feats` (`onnx.py:315`) is the existing helper that runs the
torch backbone to the encoder↔decoder boundary; it is reused unchanged. This
guard confirms equivalence end-to-end through the real attention stack.

---

## Error handling

| Condition | Error | Where |
|---|---|---|
| Module with `use_rope and use_ve_rope` | `VeRopeUnsupportedError` | VE-RoPE scan in `_merge_and_cast` / `_patch_encoder_rope_for_export` |
| Per-module original-vs-real mismatch (atol/rtol 1e-5) | `RopeEquivalenceError` | patch loop |
| Whole-encoder pre-vs-post mismatch (fp32 `_PARITY_TOL`) | `RopeEquivalenceError` | `_merge_and_cast` |

- **`VeRopeUnsupportedError` message** must state clearly that
  `VisionRotaryEmbeddingVE` (the `use_ve_rope=True` path) has **no real-valued
  variant**, is a **separate, harder blocker**, and that the SAM 3.1 default is
  `use_ve_rope=False`. We **refuse to silently** leave an untraceable op rather
  than skip it.
- Both error classes subclass `RuntimeError` and mirror the docstring/style of
  `ExportParityError` (`onnx.py:241`).
- All three errors abort export **before** any trace or cast — no partial bundle
  is written.

---

## Testing — CPU only, no GPU, no checkpoint

File: `tests/export/test_rope_export_patch.py`.

Built on the **real** `sam3.model.vitdet.Attention`, which is CPU-constructible
with `input_size=(8,8)` (per the spike). Explicit note for the reader: the
`TinySam3Stub` has **no real vitdet RoPE attention**, so CPU stub tests
**structurally cannot** exercise this path — hence the real-`Attention` CPU
tests below.

Four cases:

- **(a) Patch correctness / bit-exactness.** Construct an in-scope `Attention`
  (`use_rope=True, use_rope_real=False, use_ve_rope=False, input_size=(8,8)`).
  Capture the pre-patch `_apply_rope(q, k)` output. Run the swap (via
  `_patch_encoder_rope_for_export` or the per-module steps). Assert: the patch
  flips `use_rope_real` to `True`, registers `freqs_cis_real` / `freqs_cis_imag`
  buffers, and the post-patch `_apply_rope(q, k)` output is **bit-exact** vs the
  pre-patch complex output (`max|Δ| == 0.0` at fp32; assert `allclose` at
  atol/rtol 1e-5 as the durable gate).
- **(b) ONNX export succeeds with no complex op.** After patching,
  `torch.onnx.export(patched_module, ..., opset_version=17)` **succeeds** —
  no `UnsupportedOperatorError` and **no `view_as_complex`** in the graph (the
  exact regression #279 is about). Assert export does not raise, and inspect the
  exported graph for the absence of `view_as_complex` / complex ops.
- **(c) VE-RoPE fails loud.** A module with `use_ve_rope=True` → the patch
  raises `VeRopeUnsupportedError`.
- **(d) Guard is not vacuous.** Deliberately corrupt the regenerated real table
  (e.g. zero or scramble `freqs_cis_real`) so the real path diverges from the
  complex reference → the per-module guard raises `RopeEquivalenceError`. Proves
  the equivalence guard actually fires.

These tests run on CPU, need no GPU and no checkpoint, and gate this PR.

---

## GPU validation runbook (deferred, manual gate — user-approved)

Acceptance criteria #3 and #4 inherently need a GPU run with the real SAM 3.1
checkpoint and are **not** part of this PR's automated gate. They are verified by
a GPU run the **user approves before merge** (per the standing "ask before any
GPU run" rule).

Runbook:

1. **Patch the run config first** (tracked in **#278** — stale run configs).
   Export needs a config compatible with the merged real model.
2. Repro command:

   ```bash
   csp export --to onnx --fp16 --check \
     --checkpoint runs/<run>/adapter \
     --config <patched config> \
     --output /tmp/onnx-real
   ```

3. Verify `--check` parity passes against the real merged model (criterion #3).
4. Verify a real-model `--use-onnx` round-trip matches the torch path
   (criterion #4).
5. **Caveat:** once RoPE is solved, **further** unsupported ops in the
   attention / grounding stack may surface. Expect **iterative** GPU validation
   — solving RoPE unblocks but does not guarantee a clean full-model trace.

---

## Acceptance criteria (verbatim from #279; split PR-vs-deferred)

- [ ] **Encoder traces to ONNX with no `view_as_complex` / complex ops.**
  → CPU test (b) lands in **this PR**; full-encoder confirmation is the
  **deferred GPU gate**.
- [ ] **Original-vs-real RoPE forward equivalence asserted (documented
  tolerance).** → **this PR** — both equivalence guards (per-module atol/rtol
  1e-5; whole-encoder fp32 `_PARITY_TOL` `(1e-3, 1e-3)`).
- [ ] **`--fp16 --check` parity passes against the real merged model on GPU.**
  → **deferred GPU gate** (runbook step 3).
- [ ] **Real-model `--use-onnx` round-trip matches the torch path.**
  → **deferred GPU gate** (runbook step 4).

**Lands in this PR:** the swap helper `_patch_encoder_rope_for_export`, the
`VeRopeUnsupportedError` fail-loud guard, both equivalence guards
(`RopeEquivalenceError`), and the CPU tests.

---

## Out of scope

- **A real-valued VE-RoPE variant** (`VisionRotaryEmbeddingVE` /
  `use_ve_rope=True`). No real equivalent exists; we fail loud. Separate, harder
  follow-up.
- **#278 — run-config patching.** Prerequisite for the GPU runbook; tracked
  separately.
- **Further unsupported ops** in the attention / grounding stack that may surface
  on the real model once RoPE is solved. Follow-up; expect iterative GPU
  validation.
