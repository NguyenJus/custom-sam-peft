# QLoRA 8 GB Fit Investigation — Design Spec

**Status:** Draft (2026-05-24)
**Tracking:** [#137](https://github.com/NguyenJus/custom-sam-peft/issues/137)
**Scope:** An empirical, GPU-executed investigation answering a single
binary-with-a-number question: does one QLoRA fwd+bwd+optim step of SAM 3.1
multiplex fit in ~7 GB usable on the GTX 1080 (sm_61, fp16) using
**non-checkpointing** memory levers — or, with measurements, does it not (and at
what minimum VRAM does it)? The deliverables are a committed findings report
(`docs/research/`), a Pascal-tuned config (committed **only if it fits**), and a
link from the manual-GPU-pass document's waiting Phase-3 placeholder. No
`src/`, test-marker, or `gpu-test-policy.md` tier changes ship in this work.

---

## 1. Context & motivation

The local 8 GB GTX 1080 exists so non-interactive agents can debug the QLoRA
training code path locally rather than assume correctness and wait on manual
Colab T4 runs. That payoff requires a QLoRA training **step** (forward +
backward + optimizer step) to actually *run* within the card's ~7 GB usable
budget.

Gradient checkpointing — the usual lever for a tight VRAM budget — is
**abandoned and out of scope** (#127 closed unmerged; #89 / #60 closed
not-planned). The Phase-0 diagnostic on the real 1080 proved sam3's
non-reentrant `torch.utils.checkpoint` × fused-SDPA-RNG interaction raises a
structural `CheckpointError` on both sm_61 and sm_75, unfixable without editing
the external sam3 source (full trace in
`docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md`, "Phase-0 trace + fix
classification"). With that lever gone, fitting a 1008² QLoRA training step in
~7 GB is an **open question** — this investigation answers it with measurements.

The QLoRA training smoke (`configs/examples/gpu_smoke_qlora.yaml`) is currently
a `gpu_t4` test with `VRAM_CEIL_GB = 10.0`
(`tests/gpu/test_real_train_qlora.py:35`). So the gap to close is roughly
**10 GB → ~7 GB** (~30%) using non-offload levers only. The owner has stated up
front that "the answer could very well be 'no, not possible'" — a rigorously
measured "no, and here is the floor" is a first-class outcome of this work.

---

## 2. Goal & non-goals

### 2.1 Goal

Run a pre-registered, staged GPU protocol on the real GTX 1080 that produces:

- The SDPA-backend that actually executes on sm_61 for the representative
  attention shapes (resolving a documented contradiction — see §4, Run A).
- A baseline / stacked / static-floor peak-VRAM table.
- A single-lever ablation attributing the dominant memory cost.
- An fp16-finiteness note (distinct from OOM).
- A **verdict** (FIT / NO-FIT) plus the **minimum achieved peak VRAM**, which
  *informs* (does not change, here) the `gpu_t4` tier classification.

### 2.2 Non-goals (explicit out-of-scope)

These are deliberately excluded; route any of them to a follow-up issue, **not**
this work:

- **Adding a new `gpu_local` pytest test or editing the tier classification in
  `gpu-test-policy.md`.** The issue says findings *inform* the tier, not change
  it here. No test marker moves, no policy-doc tier edits ship in this PR.
- **Running CPU offload.** Offload is documented as a known fallback only (§5);
  it is excluded from execution by an explicit user decision (non-offload
  verdict).
- **Re-enabling or "fixing" gradient checkpointing** (#127 / #89 / #60 closed).
  Do not propose re-enabling it, owning the checkpoint, or editing sam3.
- **Convergence, throughput, or loss-curve characterization.** This is a *fit*
  question. The report asserts step completion and loss *finiteness*, never
  convergence. (Loss-ratio / mAP assertions belong to the existing `gpu_t4`
  smokes, not here.)
- **Lowering `image_size`.** Fixed at 1008×1008 by SAM 3.1 (the owner confirmed
  on the issue that the model resizes internally); it is **not** a VRAM lever
  and must not be proposed.
- **Editing the external sam3 package.** Any monkeypatching is via `_patches/`
  only — and no such patch is anticipated by this investigation.

---

## 3. Background & constraints (hard facts)

State these in the report's background section; they bound the whole protocol.

- **Hardware.** GTX 1080, compute capability **6.1 (sm_61)**, 8 GB physical,
  **~7 GB usable** (WSL/Xwayland holds ~1 GB). The target ceiling is **~7.0 GB**,
  not the 8 GB nameplate.
- **Environment.** Reach the GPU **only** via the `gpu-pascal` uv extra:
  `uv run --extra gpu-pascal …`. It resolves **torch 2.7.1+cu118** (sm_61 via PTX
  JIT from `compute_60`) + **bitsandbytes 0.49.2**. The default cu130 torch ships
  **no sm_61 cubin** and cannot run on this card.
- **No hardware bf16.** Pascal has no bf16 hardware; everything is **fp16**. The
  codebase coerces dtype via `coerce_dtype_for_capability`, so a `bfloat16`
  config is coerced to `float16` on sm_61. bnb 4-bit NF4 forward **and** backward
  are confirmed working on sm_61/fp16 (the §4.3 milestone, manual-pass doc).
- **`image_size` is fixed at 1008×1008.** SAM3 resizes internally; lowering
  resolution is **not possible** (owner-confirmed). Do not change it.
- **Gradient checkpointing is dead.** Out of scope (see §1, §2.2). Do not propose
  re-enabling or fixing it.
- **Do not edit sam3.** `_patches/` monkeypatching only.
- **Current classification.** `configs/examples/gpu_smoke_qlora.yaml` is a
  `gpu_t4` test with `VRAM_CEIL_GB = 10.0`. The gap to close is ~10 GB → ~7 GB.
- **Measurement.** Peak VRAM is measured **only** with
  `torch.cuda.reset_peak_memory_stats()` then `torch.cuda.max_memory_allocated()`
  (the pattern in `tests/gpu/test_real_train_qlora.py:60–62` and
  `tests/gpu/test_multiplex_vram.py:40–43`). **No `nvidia-smi`.** Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- **Run isolation.** GPU runs are expensive and OOM-fragmentation-prone: **one
  GPU run per process/file, serialized.** The ~3.3 GB checkpoint must be released
  between files. The SAM 3.1 checkpoint is reached via a symlink from the main
  repo (the `gpu-pascal` env on this worktree links to the main repo's
  `models/sam3.1/`).
- **Step-count independence.** A single training step's peak VRAM ≈ a 3-step
  fast-smoke peak — peak is step-count-independent once activations + optimizer
  state are allocated. So the existing `test_qlora_smoke_fast` (~3 steps,
  `tests/gpu/test_real_train_qlora.py:89`) mechanics validly stand in for "one
  step" measurement, and a 2–3-step window doubles as the fragmentation guard
  (§4, Run C).

---

## 4. The protocol (Stages 0–3 — the core of this work)

Four serialized GPU runs, each in its **own process/file**, each early-exiting
pessimistically. Each Run is a **self-contained, independently-runnable
measurement task** — the orchestrator session executes them one at a time on the
1080, releasing the checkpoint between files. Every Run sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and uses only
`reset_peak_memory_stats()` + `max_memory_allocated()` for VRAM.

The protocol is **pre-registered**: the Runs, the early-exit rule, the decision
gate, and the ablated lever are all fixed here, before any number is observed,
so the verdict is not retrofitted to the data.

### 4.1 Run A — Stage 0: cheap probes (sub-few-minute, no training loop)

**Purpose.** Cheaply settle the two facts that decide whether the protocol is
even worth continuing — the SDPA backend (which drives the trunk's transient
activation peak) and the irreducible static floor — before spending a full
training run.

**A(a) — SDPA backend probe.** Determine which `torch.nn.attention` SDPA backend
*actually executes* on sm_61 for the representative attention shapes. This
resolves a live contradiction in the existing evidence:

- Issue lever #8 assumes the **memory-efficient** SDPA backend is selected — a
  free activation win, because it avoids materializing the full attention
  matrix.
- `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` (Fix A+, line ~220)
  claims **"MATH is the only real SDPA backend on sm_61."** If MATH is forced,
  the ViT trunk materializes full **5184×5184** attention matrices across many
  layers — a large, possibly fatal transient activation cost.

What to measure / record:

- Which backend the dispatcher selects for the representative shapes (decoder
  cross-attention runs **34 decoder queries × 5184 image tokens** over 8 heads
  per the Blocker-2 trace; the ViT-trunk self-attention is the 5184-token
  self-attention case).
- Whether `EFFICIENT_ATTENTION` runs **without error** on sm_61 when explicitly
  requested via `torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`
  (does it execute, or does it raise / silently fall back to MATH?).
- The selected backend for the default (unpinned) dispatch path that the real
  forward will take.

This probe loads no checkpoint if the shapes can be exercised with synthetic
tensors of the representative dimensions; that keeps it sub-minute. (If a
faithful backend-selection answer requires the real module, it may load the
4-bit base — but it remains forward-shape-only, no training loop.)

**A(b) — Static post-load floor.** Load the 4-bit base + LoRA on the 1080 and
measure `max_memory_allocated()` with **NO forward** — the irreducible
weight/quant-constant floor. Then run a **forward-only** pass and capture that
peak too (the transient forward peak, whose size hinges on the A(a) finding).

What to measure / record:

- Static post-load peak (no forward) — the irreducible floor, in GB.
- Forward-only peak (one no-grad forward) — in GB.

**Early-exit rule (pre-registered).** If the **static floor alone already
exceeds ~7 GB**, the verdict is **NO-FIT** — stop the protocol here, record the
floor, and proceed straight to the report. (Runs B–D add only training state on
top of a floor that already overflows; running them would waste GPU time.)

### 4.2 Run B — Stage 1: baseline

**Purpose.** Establish the as-is peak the smoke currently incurs, to anchor the
gap and confirm the ~10 GB / OOM starting point on the 1080.

**Config.** One training step (verified via the ~3-step fast-smoke mechanics) at
the **as-is** smoke config: fp16 (coerced from the config's `bfloat16` on
sm_61), scope `vision_decoder`, the current optimizer (`adamw8bit`), **no**
double-quant. I.e. `configs/examples/gpu_smoke_qlora.yaml` with only the
fixture/output overrides and the fast-smoke `train.epochs`/`train.log_every`
overrides — no lever changes.

What to measure / record:

- Peak `max_memory_allocated()` in GB (expected ~10 GB or OOM).
- Whether the step completed, OOM'd, or produced non-finite loss.

If Run B OOMs on the 1080 (plausible at the ~10 GB T4 ceiling on a ~7 GB card),
record "baseline OOM at ~7 GB ceiling" and continue to Run C regardless — the
baseline OOM is itself a recorded datum, not an early-exit.

### 4.3 Run C — Stage 2: all non-offload levers stacked

**Purpose.** The decisive run. Measure peak with every non-offload lever ON and
decide FIT vs NO-FIT.

**Config (Pascal-tuned, name `min_gpu_qlora`).** All non-offload levers from §5
ON: fp16 model + compute_dtype, `use_double_quant=true` (NF4),
`paged_adamw_8bit` optimizer, narrowest available trainable scope (e.g.
decoder-only rather than the current `vision_decoder`), smaller LoRA rank `r` and
fewer target modules. `batch_size: 1`; grad-accum is **not** used (VRAM-neutral
at batch 1 — see §5).

**Measurement.** Run **2–3 consecutive steps** and take the max peak across them
(the fragmentation guard — peak should plateau; a rising peak across steps flags
fragmentation rather than a true allocation ceiling).

**Decision gate (pre-registered).**

- **FIT** ⇔ peak `max_memory_allocated()` **≤ ~7.0 GB** AND **no OOM** AND
  **finite loss** across the 2–3 steps, under `expandable_segments:True`.
- Otherwise **NO-FIT** — and record the **floor** (the minimum achieved peak),
  whether limited by OOM, by a >7 GB peak, or by non-finite loss.

Either way, record the minimum achieved peak in GB.

### 4.4 Run D — Stage 3: single-lever ablation

**Purpose.** Attribute the gap — distinguish **activation-bound** from
**optimizer/grad-bound** — by toggling OFF exactly **one** pre-registered
dominant lever and re-measuring against Run C.

**Which lever (pre-registered, decided by Run A(a)).**

- **If the SDPA backend is togglable** (i.e. A(a) found EFFICIENT_ATTENTION runs
  on sm_61 and can be forced off to MATH, or vice-versa): ablate the SDPA
  backend. A large peak swing ⇒ the cost is **activation-bound** (the trunk's
  transient attention-matrix materialization dominates).
- **Else** (SDPA backend is not togglable on sm_61 — e.g. MATH is forced
  regardless): ablate the **optimizer-state lever** (`paged_adamw_8bit` →
  `adamw8bit`, or the 8-bit optimizer → a 32-bit one). A large peak swing ⇒ the
  cost is **optimizer/grad-bound**.

**Measurement.** Same 2–3-step peak as Run C, with the single lever toggled. The
**delta** between Run C and Run D is the attribution.

What to measure / record:

- The ablated lever (named).
- Run-D peak in GB and its delta from Run C.
- The attribution conclusion (activation-bound vs optimizer/grad-bound).

If Run C already early-exited via Run A's static-floor rule, Run D is skipped and
the report notes it as not-run (the floor verdict needs no attribution).

### 4.5 Analytical point the report MUST make (a hypothesis to test, not assume)

Because the ViT trunk is **frozen** and sits **upstream** of the only trainable
params (decoder LoRA), autograd should **not** retain trunk activations — no
trainable param is in or before the trunk, so there is nothing to backprop into
there. The expected consequence:

- The dominant **retained** cost is **decoder-side** (LoRA grads + optimizer
  state + retained decoder activations).
- The trunk contributes only a **transient forward peak**, whose size hinges on
  the Run A(a) SDPA-backend finding (MATH → large 5184² matrices; mem-efficient
  → small).

The report MUST state this as a **hypothesis the measurements test**, not an
assumption baked into the verdict. Run A(a) + Run D together either confirm it
(small retained footprint, large transient trunk peak under MATH) or refute it
(retained footprint dominates), and the report says which.

---

## 5. Levers table (non-offload only)

The user's decision is **non-offload levers only**; offload is documented as a
fallback, not executed.

| Lever | Action | Note to encode |
|---|---|---|
| dtype | `float16` model + `compute_dtype` | Pascal-**required** (coerced via `coerce_dtype_for_capability`), not optional. |
| double-quant | `use_double_quant=true` (NF4) | Already wired (`QLoRAConfig.use_double_quant`, honored at the `Linear4bit` site). |
| optimizer | `adamw8bit` → `paged_adamw_8bit` | Caps optimizer-state spikes (paged state). |
| trainable scope | narrowest available (e.g. decoder-only vs current `vision_decoder`) | Mainly shrinks optimizer/grad state, **NOT** the frozen-trunk forward peak. |
| LoRA | smaller rank `r`, fewer target modules | Shrinks adapter grad/state. |
| grad-accum | **excluded** | VRAM-neutral at batch=1; does **not** help the per-step ceiling — state this explicitly. |
| CPU offload | **excluded from execution** | Documented as a known fallback only; the user chose a non-offload verdict. |

**fp16 caveat (a distinct failure mode).** Pascal fp16 has narrow dynamic range;
loss may go **non-finite**. Treat non-finite loss as a failure mode **distinct
from OOM** and require the report to note it explicitly. The smoke asserts loss
*finiteness*, not convergence (convergence / throughput / loss curves are out of
scope per §2.2). There is no GradScaler in `src/` and none is added; the QLoRA
path disables outer autocast and runs fp16 directly at the `Linear4bit` compute
dtype, and the training loop's existing NaN-skip / `nan_abort_after` guards
apply.

---

## 6. Success criteria (precise)

A **FIT** verdict requires, on the real GTX 1080 under
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

- A single fwd+bwd+optim step (**verified over 2–3 consecutive steps** for the
  fragmentation guard) completes, AND
- peak `max_memory_allocated()` **≤ ~7.0 GB**, AND
- **finite loss** across those steps.

Regardless of pass or fail, the report **MUST** record the **minimum achieved
peak VRAM** (in GB). A precise NO-FIT-with-a-floor ("does not fit; minimum
achieved peak was N GB, limited by <OOM | >7 GB peak | non-finite loss>") is a
complete, first-class result that informs the `gpu_t4` classification.

---

## 7. Deliverables & file paths

YAGNI on anything beyond these three.

1. **Findings report** — committed at
   `docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`. Contents:
   - The SDPA-backend result (Run A(a)).
   - A peak-VRAM table: static floor + forward-only (Run A(b)), baseline
     (Run B), stacked (Run C), and Run-D ablation rows.
   - The Run-D ablation attribution (activation-bound vs optimizer/grad-bound).
   - The fp16-finiteness note (whether loss stayed finite; distinct from OOM).
   - The §4.5 hypothesis stated and adjudicated against the measurements.
   - A **verdict + minimum-achieved VRAM** that *informs* the `gpu_t4`
     classification (does not change it here).
2. **Pascal-tuned config — conditional on the verdict.**
   - **If it FITS:** commit `configs/examples/min_gpu_qlora.yaml` (fp16,
     `use_double_quant`, `paged_adamw_8bit`, narrowest scope, smaller LoRA rank).
   - **If it does NOT fit:** the same reduced config appears instead as a
     **non-shipped appendix inside the findings report** — **not** committed
     under `configs/`. Use the name `min_gpu_qlora` in **both** cases.
3. **Wire the manual-GPU-pass placeholder.** Populate / link the report into the
   waiting "Phase-3 calibration numbers" placeholder in
   `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` (currently the empty
   `<!-- filled by C-4 -->` block at the bottom of that file). Replace the
   placeholder text with a link to (and short summary of) the findings report.

---

## 8. Risks & caveats

1. **fp16 non-finite loss.** Pascal fp16's narrow dynamic range may push loss to
   NaN/Inf. **Mitigation:** treat it as a distinct failure mode from OOM; the
   loop's NaN-skip / `nan_abort_after` guards apply; the report notes finiteness
   explicitly. A non-finite loss does **not** by itself mean "doesn't fit" — it
   is recorded separately from the VRAM verdict.
2. **OOM fragmentation.** The card is tight (~7 GB usable); a 37-min suite re-run
   OOM'd in 64 s from residual fragmentation (manual-pass doc). **Mitigation:**
   one GPU run per process/file, serialized; release the ~3.3 GB checkpoint
   between files; `expandable_segments:True`; the 2–3-step window in Run C
   distinguishes a true ceiling from a fragmentation creep.
3. **SDPA-backend uncertainty.** The issue (lever #8) and the manual-pass doc
   (Fix A+) **contradict** each other on whether mem-efficient SDPA runs on
   sm_61. **Mitigation:** Run A(a) is pre-registered specifically to settle this
   *first*; the Run-D ablation choice and the §4.5 attribution both hinge on its
   answer. If MATH is forced, the trunk's transient 5184² activation peak may
   alone exceed ~7 GB — which Run A(b)'s forward-only peak will reveal.
4. **Baseline OOM masks the gap size.** If Run B OOMs, the exact baseline peak is
   unknown (only "> ~7 GB"). **Mitigation:** Run A(b)'s forward-only and static
   floors bound it from below; the report states the baseline as "OOM at ~7 GB
   ceiling" rather than a fabricated number.
5. **The verdict may be NO-FIT.** The owner anticipated this. **Mitigation:** a
   measured floor is a complete deliverable; the report's value is the number
   that informs `gpu_t4`, not a forced "yes."

---

## 9. Out of scope & follow-ups

Restating §2.2 as the routing list. Each item below is a **follow-up issue**, not
part of this work:

- **Add a `gpu_local` QLoRA training test + reclassify the tier in
  `gpu-test-policy.md`.** Only if Run C returns FIT (or a floor close enough to
  motivate it). This work *informs* the tier; the actual test addition and policy
  edit are a separate issue. The `gpu_t4` tier is documented as **provisional
  pending #137** in `gpu-test-policy.md:69–72` — this investigation supplies the
  number that resolves that provisional note, in a later PR.
- **CPU offload feasibility.** Documented here as a fallback only; if NO-FIT and
  the user wants to pursue offload, that is a new issue.
- **Convergence / throughput / loss-curve characterization** on Pascal fp16 — a
  separate concern from the fit question.
- **bf16-faithful validation** remains a T4 concern (the existing `gpu_t4`
  ceilings), untouched here.

**Execution note.** This spec is produced in a Brainstormer-Planner session; the
actual GPU numbers are produced **later**, when an Implementation-Orchestrator
session executes the protocol on the 1080 (serialized, one run per file, the
checkpoint released between files). Each Run in §4 is written to be a
self-contained, independently-runnable measurement task so the orchestrator can
dispatch them one at a time.
