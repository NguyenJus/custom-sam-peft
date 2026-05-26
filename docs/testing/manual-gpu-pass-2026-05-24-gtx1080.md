# Manual GPU Test Pass — 2026-05-24 (GTX 1080, §4.3 hard-gate milestone)

Operational tracker for the §4.3 Pascal hard-gate milestone and the local
`gpu_local` tier. Companion to
[`local-pascal-gpu-testing.md`](local-pascal-gpu-testing.md) and
[`gpu-test-policy.md`](gpu-test-policy.md).

Hardware: **NVIDIA GeForce GTX 1080**, compute capability 6.1 (sm_61), 8 GB VRAM,
driver 582.28, WSL2.

## How to run

Provision the Pascal env first (cu118 wheel + bitsandbytes):

```bash
uv sync --extra gpu-pascal
```

Then run the gpu_local test tier:

```bash
bash scripts/run_gpu_tests.sh local
```

Or directly:

```bash
uv run --extra gpu-pascal pytest -m gpu_local tests/gpu/ tests/integration/ tests/predict/
```

Single proofs (as used in this milestone):

```bash
uv run --extra gpu-pascal python <proof-script.py>
```

Restore the dev env when done:

```bash
uv sync --extra dev
```

## Test checklist

### §4.3 milestone proofs (2026-05-24)

- [x] **Step 1** — sm_61 CUDA matmul via PTX JIT (cu118 torch 2.7.1)
- [x] **Step 2** — bnb `Linear4bit` NF4 forward on sm_61, float16

### gpu_local tier (populated by later tasks)

- [x] **C-2 / Phase-0 trace** — Blocker 1 (float16 cross-attn `attn_mask` cast)
  FIXED; Blocker 2 (#127 grad-ckpt recompute mismatch) **BLOCKED** on
  Pascal/float16 (sam3-internal save-count divergence; Fix A/B both fail, Fix C
  ruled out). See the session log below.

## Session log

### §4.3 milestone — 2026-05-24 — PASS

#### Context

Task A-1 landed (commit eb523ab) adding an opt-in `gpu-pascal` uv extra that
resolves `torch 2.7.1+cu118` + `bitsandbytes 0.49.2`, isolated from the default
`2.12.0+cu130`. Resolution facts:

| env | torch | bnb |
| --- | --- | --- |
| bare `uv sync` / `uv sync --extra dev` | 2.12.0+cu130 (NO sm_61 cubin) | — |
| `uv sync --extra gpu-pascal` | **2.7.1+cu118** (sm_60..sm_90 + PTX) | **0.49.2** |

The cu130 default torch ships no sm_61 cubin; cu118 covers sm_61 via PTX JIT
from `compute_60`.

#### Step 1 — sm_61 CUDA matmul kernel (cu118 PTX JIT)

Command:

```bash
uv run --extra gpu-pascal python - <<'PY'
import torch
print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = a @ b                      # forces a CUDA matmul kernel launch -> PTX JIT compute_60 -> sm_61
torch.cuda.synchronize()
ref = (a.cpu() @ b.cpu())
err = (c.cpu() - ref).abs().max().item()
print("matmul max abs err", err)
assert err < 1e-2, "sm_61 matmul produced wrong results"
print("SM_61 KERNEL OK")
PY
```

Verbatim output:

```text
torch 2.7.1+cu118
device NVIDIA GeForce GTX 1080 cc (6, 1)
matmul max abs err 0.00011444091796875
SM_61 KERNEL OK
```

Result: **PASS** — no `no kernel image is available` error; PTX JIT compiled
`compute_60` → sm_61 successfully; matmul error 1.14e-4 < 1e-2.

#### Step 2 — bnb Linear4bit NF4 forward on sm_61, float16

Command:

```bash
uv run --extra gpu-pascal python - <<'PY'
import torch, bitsandbytes as bnb
print("bnb", bnb.__version__)
lin = bnb.nn.Linear4bit(256, 128, bias=False, quant_type="nf4", compute_dtype=torch.float16)
lin = lin.to("cuda")           # quantization fires on .to(cuda)
x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
y = lin(x)                     # NF4 4-bit kernel forward on sm_61
torch.cuda.synchronize()
print("out", tuple(y.shape), y.dtype, "finite", bool(torch.isfinite(y).all()))
assert y.shape == (4, 128) and torch.isfinite(y).all()
print("BNB LINEAR4BIT OK")
PY
```

Verbatim output:

```text
bnb 0.49.2
out (4, 128) torch.float16 finite True
BNB LINEAR4BIT OK
```

Result: **PASS** — NF4 4-bit kernel launched on sm_61; output shape (4, 128),
dtype float16, all values finite.

#### Gate decision

Both §4.3 proofs passed. The Pascal track is **unblocked**. Downstream tasks
B/C and D's `gpu_local` calibration may proceed.

---

### Phase-0 trace + fix classification — 2026-05-24 (C-2)

Hardware: GTX 1080 (sm_61), cu118 torch 2.7.1, float16 (bf16 emulated → coerced).
Repro: QLoRA fast-smoke (`configs/examples/gpu_smoke_qlora.yaml`,
`model.dtype=float16`, `peft.qlora.compute_dtype=float16`,
`gradient_checkpointing=true`) under
`torch.utils.checkpoint.set_checkpoint_debug_enabled(True)` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

#### Blocker 1 (PREREQUISITE — FIXED) — float16 cross-attention attn_mask gap

Before this work, the float16 QLoRA forward crashed inside SAM3's decoder
cross-attention (`sam3/model/decoder.py:166` → `model_misc.py:397`) with:

```text
RuntimeError: invalid dtype for bias - should match query's dtype
```

The MHA input-dtype hook
(`src/custom_sam_peft/models/_patches/mha_input_dtype.py::_mha_input_dtype_hook`)
cast `query`/`key`/`value` to the module's weight dtype (fp16) but left the
float additive `attn_mask` as fp32, so `F.scaled_dot_product_attention` rejected
the mismatched bias.

**Fix (landed):** extend the hook's kwarg-cast loop to also cast `attn_mask`:
`for name in ("query", "key", "value", "attn_mask")`. The existing
`is_floating_point()` guard casts the float additive bias and correctly LEAVES
boolean masks untouched (bool masks are valid SDPA inputs). Two CPU unit tests
added in `tests/unit/test_sam3_mha_input_dtype_patch.py`
(`test_float_attn_mask_kwarg_cast_to_module_dtype`,
`test_bool_attn_mask_kwarg_left_unchanged`) — 10/10 pass under cu118
(`uv run --no-sync pytest tests/unit/test_sam3_mha_input_dtype_patch.py --no-cov`).
With this fix the float16 forward completes and training reaches the backward
pass — exposing Blocker 2.

#### Blocker 2 (#127 target — BLOCKED) — recompute-metadata mismatch

With Blocker 1 fixed, the QLoRA grad-ckpt backward raises the deferred #127
`CheckpointError`. **Failing checkpoint SITE:** SAM3's
`model_misc.MultiheadAttention.forward` self-checkpoint at
`sam3/model/model_misc.py:655` (the `_qkv_same_embed_dim` branch wrapping
`multi_head_attention_forward`), reached from the decoder cross-attention
(`decoder.py:166`). The metadata table (flag-flip only, fp16):

```text
position 1: saved [256,512]      fp16   → recomp [256,256]    fp16
position 2: saved [2,8,5184,34]  fp16   → recomp [256,256]    fp16
position 3: saved [2,8,34,32]    fp16   → recomp [2,8,5184,34] fp16
position 4: saved [2,8,5184,32]  fp16   → recomp [2,8,34,32]  fp16
position 5: saved [2,8,34,32]    fp16   → recomp [2,8,5184,32] fp16
position 6: saved [2,8,5184]     fp32   → recomp [2,8,34,32]  fp16
position 7: saved [2,8,5184,32]  fp16   → recomp [2,8,5184]   fp32
position 8: saved []             int64 CPU → recomp [2,8,5184,32] fp16
position 10: saved [256,256]     fp16   → recomp []           int64 CPU
```

Shapes are the cross-attention internals: batch 2, 8 heads, **34 decoder queries
× 5184 image tokens**, head_dim 32; `[2,8,5184]` fp32 is the softmax stat.
First divergent op is position 1 (the in-projection / first saved tensor). The
fingerprint is a **shift-by-one** (recompute slot N carries the forward's slot
N-1) PLUS an **int64 CPU scalar** that materializes at a SHIFTED slot in the
recompute (saved pos 8 vs recomputed pos 10) — i.e. the recompute saves a
DIFFERENT NUMBER of tensors than the forward inside
`multi_head_attention_forward`.

**Classification attempted: Fix A (autocast-only) → did NOT hold; escalated to
Fix B → did NOT hold; Fix C ruled out.** Evidence:

1. **Fix A — deterministic per-MHA `torch.autocast(float16)` wrap** (dtype via
   `coerce_dtype_for_capability` → fp16 on sm_61), scoped to the
   `model_misc.MultiheadAttention` modules only (wrapping the ViT-Det trunk
   blocks instead breaks the fp16 `ConvTranspose2d` neck at `necks.py:117` —
   the trunk emits fp32 under autocast). Result: the **same shift-by-one
   persists**, only the dtypes shift fp16→fp32. Autocast pinning does NOT
   resolve the structural save-count divergence.
2. **Fix A+ — additionally pin `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`**
   inside the wrap (math is the only real SDPA backend on sm_61). An isolated
   repro of the exact failing shapes was metadata-consistent with MATH pinned,
   but in the FULL model the **same shift-by-one persists**. Backend selection
   is not the (sole) divergence.
3. **Fix B — own the checkpoint** (`use_act_checkpoint=False` on the MHA so SAM3
   does not self-checkpoint; wrap calls
   `torch.utils.checkpoint.checkpoint(orig_forward, …, use_reentrant=False,
   context_fn=<pinned autocast pair>)` with the metadata check ON). Result: the
   **same shift-by-one** (now on a self-attention MHA: `[256,768]` qkv in_proj,
   2 queries, same recurring int64 CPU scalar). Owning the checkpoint with a
   pinned autocast context does NOT change the recompute save structure.
4. **Fix C — `determinism_check="none"`** on the owned checkpoint. The metadata
   check is bypassed, but the backward then raises
   `RuntimeError: Expected all tensors to be on the same device, but found at
   least two devices, cuda:0 and cpu!` (in `wrapper_CUDA_mm`). This PROVES the
   divergent tensors are NOT benign/non-differentiable — the recompute genuinely
   produces a CPU int64 scalar at a shifted slot that corrupts the gradient mm.
   Fix C is therefore disallowed (its precondition — provable non-differentiable
   divergence — is FALSE).

**Root cause:** a structural, save-count divergence INSIDE SAM3's
`multi_head_attention_forward` recompute (a CPU int64 scalar materializing at a
non-deterministic position in the autograd save list). It is reproducible only
in the full-model backward, not in an isolated MHA checkpoint (an isolated
`model_misc.MultiheadAttention` checkpoints + back-props cleanly with
`embed_dim` as a plain int on both passes). Resolving it would require editing
`sam3/model/model_misc.py` — **FORBIDDEN** (sam3 is external; we only
monkeypatch via `_patches/`). Autocast (Fix A), SDPA-backend pinning, and
context_fn-pinned own-checkpoint (Fix B) all leave the mismatch intact; Fix C
yields a corrupted backward.

**Outcome: BLOCKED on Pascal/float16 per spec §6.8 graceful degradation.** Fix A
and Fix B were both genuinely attempted on real hardware and neither resolves
the recompute mismatch without an sam3 source edit. Per §6.8 the recommended
degradation is for the orchestrator to reclassify the QLoRA training smoke to
`gpu_t4` and keep `gpu_local` for forward-only / inspection tests — but that is
the orchestrator's call, not C-2's. Blocker 1 (the attn_mask cast) is an
independent, verified win and is landed regardless.

`vit_act_checkpoint.py` was left at its flag-flip-only state (the Fix A/B wraps
were reverted; shipping a wrap that does not resolve the `CheckpointError` would
be misleading).

#### `tests/gpu/test_grad_checkpointing.py` on the 1080 — both paths FAILED

```text
FAILED tests/gpu/test_grad_checkpointing.py::test_lora_no_checkpoint_error_and_vram_lower
FAILED tests/gpu/test_grad_checkpointing.py::test_qlora_no_checkpoint_error_and_vram_lower
================== 2 failed, 4 warnings in 2233.48s (0:37:13) ==================
```

- **QLoRA path (load-bearing per plan):** fails — this is the #127 merge gate.
  The on-run (`gradient_checkpointing=true`) raises the
  `torch.utils.checkpoint.CheckpointError` documented above (verbatim trace
  captured 3× across the Fix-A / Fix-A+SDPA / flag-flip Phase-1 runs). The merge
  gate ("no `CheckpointError` on the QLoRA grad-ckpt path on the 1080") is NOT
  met — the fix could not be landed without editing sam3.
- **Memory pressure:** the card is tight (~7 GB usable of 8 GB). The test runs
  the off-reference AND the on-run sequentially in one test; a single-test
  re-run immediately after the 37-min full suite OOM'd in 64 s
  (`RuntimeError: CUDA driver error: out of memory`) from residual fragmentation.
  Per the plan this does NOT independently block (the QLoRA `CheckpointError` is
  the load-bearing failure), but it reinforces the §6.8 degradation case — the
  off+on double-run does not comfortably fit the 1080.
- **LoRA path:** also FAILED (same suite). The LoRA path is not the load-bearing
  gate per the plan; its failure is consistent with the same recompute mismatch
  (the cross-attention checkpoint is PEFT-method-independent) and/or the same
  memory tightness.

Neither PEFT path passes the grad-ckpt GPU test on the 1080.

---

### Phase-3 calibration numbers

See the full write-up:
[QLoRA 8 GB fit investigation (#137)](../research/2026-05-24-issue-137-qlora-8gb-feasibility.md).

**Verdict: FIT** — one QLoRA fwd+bwd+optim step of SAM 3.1 multiplex fits on the
GTX 1080 at **5.018 GB** peak (≤ ~7.0 GB usable), finite loss, using non-offload
levers only. The decisive lever is decoder-only trainable scope (excluding the
frozen trunk): the as-is `vision_decoder` smoke OOMs on the first backward
because trunk LoRA forces autograd to retain trunk activations, whereas freezing
the trunk frees them. The cost is **activation/weight-bound** — not
optimizer/grad-bound (training adds ~0.03 GB over the forward-only peak) and not
attention-kernel-bound (forcing MATH SDPA moves the peak 0.00 GB). The Pascal
config lives at `configs/examples/min_gpu_qlora.yaml`. This informs (does not
change here) the `gpu_t4` tier; a `gpu_local` test + reclassification is a
follow-up.
