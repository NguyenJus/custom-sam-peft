# QLoRA 8 GB Fit Investigation — GTX 1080 (sm_61)

> Research write-up for [issue #137](https://github.com/NguyenJus/custom-sam-peft/issues/137)
> Date: 2026-05-24
> Spec: `docs/superpowers/specs/2026-05-24-qlora-8gb-fit-investigation-design.md`
> Plan: `docs/superpowers/plans/2026-05-24-qlora-8gb-fit-investigation-plan.md`

## TL;DR

**Verdict: FIT.** One QLoRA forward + backward + optimizer step of SAM 3.1
multiplex fits on the real GTX 1080 in **5.018 GB** peak — well under the
~7.0 GB usable ceiling — with finite loss, using **non-offload, non-checkpointing
levers only**. The minimum achieved peak across the protocol is **5.018 GB**.

The decisive lever is **narrowing the trainable scope to exclude the frozen
trunk** (decoder-only target modules instead of `vision_decoder`). The as-is
smoke config (`vision_decoder` scope) OOMs on the first backward pass because
making trunk LoRA trainable forces autograd to *retain* the trunk's feature-map
activations for the backward; freezing the trunk frees them. The cost is
**activation/weight-bound, not optimizer-bound and not attention-kernel-bound**:

- The training-step peak (5.018 GB) is barely above the forward-only peak
  (4.99 GB), so retained gradient + optimizer state adds only ~0.03 GB.
- Forcing the MATH SDPA kernel for *every* attention (vs the default
  memory-efficient kernel) does **not** move the peak (zero delta), so attention-
  matrix materialization is not the binding constraint.

Plain (non-quantized) **LoRA** is even lighter — a fp16-base decoder-only step
peaks at **3.58 GB** (§9), because on a base this small the 4-bit quantization's
load and on-the-fly dequant transients cost more than the fp16 weights they
replace.

This **informs** (does not change here) the `gpu_t4` classification of
`configs/examples/gpu_smoke_qlora.yaml` (currently `VRAM_CEIL_GB = 10.0`).
A `gpu_local` test + tier reclassification is a follow-up (see §8).

---

## §1 — Background & constraints

These hard facts bound the whole protocol (spec §3):

- **Hardware.** GTX 1080, compute capability **6.1 (sm_61)**, 8 GB physical,
  **~7 GB usable** (WSL/Xwayland holds ~1 GB). Target ceiling **~7.0 GB**, not
  the 8 GB nameplate.
- **Environment.** GPU reached **only** via the `gpu-pascal` uv extra
  (`uv run --extra gpu-pascal …`), resolving **torch 2.7.1+cu118** (sm_61 via PTX
  JIT from `compute_60`) + **bitsandbytes 0.49.2**. The default cu130 torch ships
  no sm_61 cubin and cannot run on this card.
- **No hardware bf16.** Pascal has no bf16 hardware; everything is **fp16**.
  `coerce_dtype_for_capability` (`src/custom_sam_peft/runtime/_runtime.py`)
  coerces a `bfloat16` config to `float16` on sm_61 — confirmed firing in every
  run (`WARNING … coercing to float16`). No GradScaler is added; the QLoRA path
  runs fp16 directly at the `Linear4bit` compute dtype.
- **`image_size` fixed at 1008×1008.** SAM 3.1 resizes internally; not a lever.
- **Gradient checkpointing is dead and out of scope** (#127/#89/#60 closed; the
  sam3 non-reentrant-checkpoint × fused-SDPA-RNG mismatch breaks on sm_61).
- **Non-offload levers only.** CPU offload is documented as a fallback (spec §5),
  not executed — an explicit user decision.
- **Measurement.** Peak VRAM measured **only** with
  `torch.cuda.reset_peak_memory_stats()` then
  `torch.cuda.max_memory_allocated() / 1e9` (GB). **No `nvidia-smi`.** Every run
  set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Run isolation.** Four serialized GPU runs (A–D), each its own process/file,
  the ~3.3 GB checkpoint released between files (fragmentation guard).
- **Current classification.** `configs/examples/gpu_smoke_qlora.yaml` is a
  `gpu_t4` test with `VRAM_CEIL_GB = 10.0`. The gap to close was ~10 GB → ~7 GB.

The protocol was **pre-registered** in the spec (the runs, the early-exit rule,
the decision gate, and the ablated lever) before any number was observed.

---

## §2 — Run A: SDPA backend + static floor

### §2.1 — A(a) SDPA backend (resolves a documented contradiction)

The evidence was contradictory before this run: issue lever #8 assumes the
memory-efficient SDPA backend is selected on sm_61 (a free activation win);
`docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` (Fix A+) claims **"MATH is
the only real SDPA backend on sm_61."** Synthetic probes at the representative
shapes (fp16, cuda) settle it:

| Shape | EFFICIENT | MATH | FLASH | Default (unpinned) |
| --- | --- | --- | --- | --- |
| Decoder cross-attn `q=(1,8,34,64)`, `k=v=(1,8,5184,64)` | RAN ok | RAN ok | RAISED `No available kernel` | **EFFICIENT** (`fmha_cutlassF_f16_aligned_64x64_rf_sm50`) |
| ViT-trunk self-attn `q=k=v=(1,8,5184,64)` | RAN ok | RAN ok | — | **EFFICIENT** (same cutlass kernel) |

**Resolution: the manual-pass doc is wrong; issue #8 is right.** The
memory-efficient (xFormers CutlassF) kernel is compiled for sm50+ and executes
on sm_61; it is the *default* unpinned backend. FLASH is the only backend
blocked (requires Ampere sm_80+) — consistent with sam3's own startup notice
(`Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA
capability`). Because EFFICIENT is the default *and* MATH also runs, the SDPA
backend is **togglable**, which (per spec §4.4) pre-selects the SDPA backend as
Run D's ablated lever.

### §2.2 — A(b) static floor + forward-only peak

Built the QLoRA model (4-bit base + LoRA) exactly as `run_training` does
(`load_sam31` → `lookup("peft", …)` → `peft_factory`), measuring peak with no
forward, then one `no_grad` forward at a representative input
(1×3×1008×1008 fp16, one text class prompt):

- **Static post-load floor (no forward): 3.58 GB** — the irreducible
  weight/quant-constant resident cost.
- **Forward-only peak (one no_grad forward): 4.99 GB** — ~2.0 GB headroom.

**Early-exit gate (pre-registered):** the static floor (3.58 GB) is well below
~7.0 GB, so the protocol continued to Run B (no early exit).

---

## §3 — Peak-VRAM table (Runs A–D)

| Run | Configuration | Peak VRAM | Status | Loss finite |
| --- | --- | --- | --- | --- |
| A — static floor | 4-bit base + LoRA, **no forward** | **3.58 GB** | — | n/a |
| A — forward-only | one `no_grad` forward | **4.99 GB** | completed | n/a |
| B — baseline | as-is smoke: `vision_decoder`, `adamw8bit`, no double-quant | **≥ 6.4 GB** (lower bound) | **OOM** on first `.backward()` | n/a (OOM'd before logging) |
| C — stacked | decoder-only, `r=8`, double-quant NF4, `PagedAdamW8bit` | **5.018 GB** | **completed (FIT)** | **True** |
| D — SDPA ablation | Run C + MATH SDPA forced for all attention | **5.018 GB** | completed | **True** |

Notes:

- **Run B** OOM'd at `train/loop.py:105` (`(loss / n_micro).backward()`) on the
  first step — the forward completed; backward ran the card dry
  (`RuntimeError: CUDA driver error: out of memory`). The 6.4 GB is a lower bound
  (the *failed* allocation exceeded what remained). The as-is `vision_decoder`
  config does **not** fit on the 1080.
- **Run C / Run D** ran 2 epochs over the tiny fixture (4 logged steps); the peak
  is the max across all steps (the fragmentation guard — it plateaued, no creep).
  The decoder-only `target_modules` adapted **221,184** trainable parameters.
- The `paged_adamw_8bit` optimizer is not a config-schema literal; Runs C/D wired
  it via a measurement-only monkeypatch of `_build_optimizer` (not a `src/`
  change). The shipped config (§7) uses the schema-supported `adamw8bit` — at
  this parameter count the paged variant is immaterial to peak (see §5).

---

## §4 — Run D ablation & attribution

**Ablated lever (pre-registered by Run A(a)): the SDPA backend.** Run D re-ran
the Run C config but forced **MATH-only** attention. sam3 selects its backend via
direct `torch.backends.cuda.enable_*_sdp()` global-setter calls (not a context
manager), so an outer `sdpa_kernel([MATH])` context is ignored. MATH was forced
by intercepting those setters and converting every `enable_mem_efficient_sdp(True)`
/ `enable_flash_sdp(True)` to a disable. Hard verification from the run:

```text
enable_mem_efficient_sdp(True) intercepted: 128 times → forced to False each time
enable_flash_sdp(True) intercepted: 128 times → forced to False each time
CONFIRMED: sam3 tried to enable non-MATH backends; all were blocked.
backend state: flash_sdp=False, math_sdp=True, mem_efficient_sdp=False
```

**Result: zero delta.** Run D peak = Run C peak = **5.018 GB**
(delta = 0.000 GB), loss finite.

**Attribution: activation/weight-bound — not optimizer-bound, not
attention-kernel-bound.**

- *Not optimizer/grad-bound.* Training peak (5.018 GB) − forward-only peak
  (4.99 GB) ≈ **0.03 GB**. The retained gradient + 8-bit optimizer state for
  221 K trainable params is negligible.
- *Not attention-matrix-bound.* Forcing the explicit-materialization MATH kernel
  for every attention (vs the default mem-efficient kernel) did not raise the
  peak. The attention-score matrices never set the global peak.
- *What is binding:* the static quantized-weight floor (3.58 GB) plus the
  transient forward feature-map activations (~1.4 GB), and — for the baseline —
  the **retention** of trunk feature-map activations when trunk LoRA is
  trainable. The baseline OOMs in backward; the decoder-only config does not.

---

## §5 — The §4.5 hypothesis, adjudicated

The spec (§4.5) posed a two-part hypothesis to *test*, not assume:

1. **The dominant retained cost is decoder-side; the frozen, upstream trunk's
   activations are not retained by autograd.** **CONFIRMED.** Forward-only
   (4.99 GB) ≈ full training step (5.018 GB) shows the retained training state is
   tiny. And the baseline (Run B, `vision_decoder` → trunk LoRA trainable → trunk
   activations *must* be retained for backward) OOMs, while the decoder-only
   config (Run C, trunk frozen → trunk activations freed) fits. Trunk-activation
   retention is precisely the difference between OOM and fit.

2. **The trunk's transient forward peak hinges on the SDPA backend (MATH → large
   5184² matrices; mem-efficient → small).** **REFUTED.** Run D forced MATH for
   all attention and the peak was identical (zero delta). The transient forward
   peak does not depend on the attention kernel. The most likely mechanism: the
   SAM 3.1 image encoder is a hierarchical (Hiera-style) trunk whose
   global-attention stages operate on pooled/downsampled token counts, so no
   dense 5184² attention matrix is ever materialized at full resolution — the
   peak is set by feature-map activations and the static weight floor, not by
   attention scores. (Stated as interpretation; the *measured* fact is the zero
   delta under verified MATH forcing.)

**Net:** the QLoRA training step is activation/weight-bound. The single decisive
lever is trainable-scope narrowing (exclude the frozen trunk). Double-quant and
smaller rank were stacked but not individually isolated; mechanistically they
shrink the static weight/adapter footprint, not the retained activation volume
that drives the baseline OOM.

---

## §6 — fp16 finiteness (distinct from OOM)

Pascal fp16 has narrow dynamic range, so loss can go **non-finite** — a failure
mode treated as **distinct from OOM** (spec §5, §8). In this investigation loss
stayed **finite** across all logged steps:

- Run C losses: `0.3895, 0.5412, 0.4127, 1.0160` — all finite.
- Run D losses: `0.4565, 0.5190, 0.3918, 0.9545` — all finite.

The training loop's existing NaN-skip / `nan_abort_after` guards apply; none
fired. This is a *finiteness* result over a few steps, not a convergence claim
(convergence / throughput / loss curves are out of scope, spec §2.2).

---

## §7 — Pascal-tuned config (`min_gpu_qlora`)

Because the verdict is **FIT**, the reduced config ships at
`configs/examples/min_gpu_qlora.yaml`. It bakes in the Run C levers using only
schema-supported fields:

- `model.dtype: float16` (Pascal-required; sm_61 has no bf16).
- `peft.r: 8` (down from 16).
- `peft.target_modules`: the two decoder-only patterns (excludes the frozen
  trunk). Under QLoRA the MHA-wrapped `out_proj` stays `nn.Linear` (not
  `Linear4bit`), so the `out_proj` pattern matches nothing and only the decoder
  MLP `linear1/linear2` are adapted (221 K params) — this is the narrowest
  available scope.
- `peft.qlora.compute_dtype: float16`, `peft.qlora.use_double_quant: true` (NF4).
- `train.optimizer: adamw8bit` — schema-supported. The FIT was measured with the
  paged variant (`PagedAdamW8bit`), but at 221 K trainable params the paged
  behavior is immaterial to peak; a YAML comment records this.
- `train.batch_size: 1`, `train.grad_accum_steps: 1` (grad-accum is VRAM-neutral
  at batch 1).

---

## §8 — Verdict & follow-ups

**Verdict: FIT.** A QLoRA fwd+bwd+optim step of SAM 3.1 multiplex fits on the
GTX 1080 at **5.018 GB** peak (minimum achieved peak across the protocol),
≤ ~7.0 GB usable, with finite loss, under `expandable_segments:True`, using
non-offload, non-checkpointing levers. The binding lever is decoder-only
trainable scope.

This **informs** (does not change here) the `gpu_t4` classification. Follow-ups
(spec §9), each a separate issue:

- Add a `gpu_local` QLoRA training test for the `min_gpu_qlora` config and
  reclassify the tier in `gpu-test-policy.md` (the `gpu_t4` tier is documented as
  provisional pending #137).
- CPU-offload feasibility (only if a broader scope is wanted and does not fit).
- Convergence / throughput / loss-curve characterization on Pascal fp16.
- bf16-faithful validation remains a T4 concern (existing `gpu_t4` ceilings).

---

## §9 — Addendum: plain LoRA (non-quantized) comparison

This measurement is a **post-hoc follow-up**, not part of the pre-registered A–D
protocol. Question: does a plain (non-quantized, fp16-base) **LoRA** training step
fit, and how does it compare to QLoRA? Same harness and decoder-only scope as
Run C (`r=8`, batch 1, 2–3 steps, `adamw8bit`), only `peft.method: lora`.

| Run | Method | Base | Trainable params | Peak VRAM | Status |
| --- | --- | --- | --- | --- |
| C | qlora | 4-bit NF4 | 221,184 | 5.018 GB | FIT, finite |
| E | lora | fp16 | 294,912 | **3.576 GB** | **FIT, finite** |

Plain LoRA peaks **1.44 GB lower** than QLoRA and fits with ~3.4 GB headroom.
This is expected for a *small* base model: the SAM 3.1 base is only ~1.75 GB in
fp16 (the 3.5 GB checkpoint is fp32 on disk, halved on load — Run E's resident
post-load footprint was ~1.88 GB). QLoRA's whole-run peak additionally pays a
one-time 4-bit quantization transient at load plus on-the-fly dequant transients
during the forward (each `Linear4bit` dequantizes its weight to fp16 per matmul —
QLoRA forward-only alone was 4.99 GB, above LoRA's *entire* training peak). For a
base this small those overheads exceed the resident-weight savings from going
4-bit; QLoRA's memory advantage only materializes on much larger bases. (LoRA
adapts ~33% more params here because the MHA `out_proj` is a plain `nn.Linear`
and thus adaptable, unlike under QLoRA where it is excluded from `Linear4bit`.)

Both PEFT paths fit comfortably, so the local 1080 can debug either training path.
Loss stayed finite across all logged steps (LoRA: `0.4437, 0.6583, 0.3781,
0.6332`). This is a *finiteness* result, not a convergence claim. Like the rest of
this report it concerns the **training step's** VRAM; it does **not** speak to the
full-scope `gpu_t4` tests (large multiplex forwards, fp32 calibration, or
bf16-faithful numerics), which remain T4 concerns.
