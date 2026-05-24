# Minimal GPU Testing on a GTX 1080 — Design Spec

> Date: 2026-05-24
> **AMENDED 2026-05-24:** gradient checkpointing (GC) is **abandoned** — see the
> note immediately below and §6. PR #127 closed unmerged; #89 and #60 closed
> not-planned; the 8 GB QLoRA *training* recipe is deferred to **#137**.
> Source issues: #79 (research), #68 (8 GB VRAM floors), #117 (CPU/GPU test
> audit), #116 (notebook coverage audit), #124 (cloud auto-provision),
> #137 (8 GB QLoRA training without GC).
> Related specs: `docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md`
> and its plan `docs/superpowers/plans/2026-05-23-gradient-checkpointing-t4.md`;
> `docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md`;
> `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md`.
>
> **Gradient checkpointing is abandoned (scope-defining amendment).** This spec
> was originally built on "complete PR #127's deferred GC fix on the 1080." The
> Phase-0 diagnostic on the real GTX 1080 (recorded in
> `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md`, "Phase-0 trace + fix
> classification") proved the `CheckpointError` is a **structural save-count
> divergence INSIDE sam3's `multi_head_attention_forward` recompute** (a CPU
> int64 scalar materializing at a non-deterministic autograd-save position). Fix
> A (deterministic-autocast wrap), Fix A+ (SDPA-MATH pinning), and Fix B (owning
> the checkpoint with a pinned `context_fn`) were ALL attempted on real hardware
> and NONE resolve it; Fix C (`determinism_check="none"`) yields a corrupted
> backward (proving the divergence is not benign). The only resolution would edit
> `sam3/model/model_misc.py`, which is **FORBIDDEN** (sam3 is external; we
> monkeypatch only via `_patches/`). **Outcome:** PR #127 closed unmerged; #89
> and #60 closed not-planned; `origin/main` already ships GC OFF (config default
> `false` + a no-op guard, templates `false`), so there is nothing to revert. The
> 8 GB QLoRA *training* recipe — which assumed GC as the dominant
> activation-memory lever — is spun off to **#137**. What survives and is
> independently valuable (GC-independent, already landed on this branch): the
> `gpu-pascal` cu118 extra (sm_61 + bnb, §4.3 PASSED), the CC-6.0 compat floor,
> bf16→fp16 coercion + fp16 preset labeling, the three-tier
> `gpu_local`/`gpu_t4`/`gpu_xl` taxonomy, `use_double_quant` (task C-1), and the
> independent fp16 cross-attention `attn_mask` MHA-hook fix (Blocker 1 in the
> evidence doc). The `gpu_local` **training** tier is **PROVISIONAL pending
> #137**: it currently holds forward-only / structural-inspection tests (no
> backward graph), not a training smoke.

## 1. Overview and goals

The dev box (this WSL2 machine) holds a GTX 1080: compute capability 6.1
(sm_61), 8 GB VRAM with roughly 1 GB already held by WSL/Xwayland, leaving an
**effective budget of about 7 GB**. `nvidia-smi` works (driver 582.28) and
torch sees the card, but the installed `torch 2.12.0+cu130` ships cubins for
sm_75/80/86/90/100/120 and **none for sm_61**, so no kernel runs today.

Research (#79, `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`)
establishes that both the LoRA and QLoRA paths lower to **CC 6.0** on the cu118
torch wheel (which ships sm_60..sm_90) and bitsandbytes ≥ 0.43 NF4/FP4 kernels
(CC 6.0+). Only `LLM.int8()` — which this repo never uses — needs CC 7.5.
The one caveat is that **bf16 is emulated below CC 8.0**, so Pascal must train
in **float16**.

Gradient checkpointing was originally the linchpin of this spec: PR #127
re-enabled it (flag-flip-only) and deferred the actual recompute-metadata fix to
a Phase-0 diagnostic that needed real hardware. That diagnostic was run on the
1080 and proved the fix **infeasible without editing forbidden sam3 source**
(see the abandonment note above and §6). GC is therefore **abandoned**: PR #127
closed unmerged, #89/#60 closed not-planned, and `origin/main` already ships GC
OFF. This spec is consequently **reduced** to the GC-independent work.

**Goal.** Make the GTX 1080 a first-class GPU test target — provision its sm_61
kernels, enable CC-6.0/float16 code paths, and rationalize the GPU test suite
into a three-tier hardware taxonomy (#117/#116) — without a Colab T4. One spec,
one PR, the GC-free workstreams. The `gpu_local` **training** tier is
**PROVISIONAL pending #137**: with GC abandoned, fitting a 1008² QLoRA training
step in ~7 GB is unproven, so `gpu_local` currently holds forward-only /
structural-inspection tests (no backward graph), and the 8 GB *training* recipe
is deferred to #137.

**Guiding principle.** *Any testing runnable on this branch gets run here.* The
1080 is live in WSL, so implementers run the real GPU validations in-session
(sm_61/bnb sanity, the forward-only / inspection `gpu_local` calibration) rather
than deferring them. (The Phase-0 GC diagnostic was likewise run in-session — it
is what established the abandonment.)

## 2. Non-goals

- **No bf16-on-Pascal support.** bf16 is emulated below CC 8.0; we coerce to
  float16 and warn. We do not attempt to make bf16 numerically faithful on
  Pascal.
- **No `LLM.int8()` enablement.** It is unused in this repo and is the only bnb
  feature that genuinely needs CC 7.5. We never lower its floor.
- **No T4-tier or XL-tier execution in this PR.** We *define* and *wire* the
  `gpu_t4` and `gpu_xl` tiers, but the work that genuinely needs >8 GB or
  bf16-faithful numerics is deferred to a follow-on issue (§9) and to #124.
- **No change to the default cu130 install.** The cu118 wheel is reachable only
  through an opt-in `gpu-pascal` extra; the bare `uv sync` still resolves cu130.
- **No raising of the existing 14 GB / 10 GB T4 VRAM ceilings.** Those remain
  the T4-tier release gates, untouched here.
- **No GradScaler introduction.** See the float16-stability risk (§10, Risk 3)
  for why the primary QLoRA path sidesteps the need for one.
- **No 8 GB QLoRA *training* recipe (deferred to #137).** With gradient
  checkpointing abandoned (§6.1), fitting a 1008² training step in ~7 GB is
  unproven; the recipe, its training smoke, and the on-1080 calibration are
  spun off to #137. The `gpu_local` training tier is PROVISIONAL.
- **`image_size` is not a VRAM lever.** SAM 3.1's input resolution is fixed at
  1008×1008 by the model itself (`sam3.py:749,753`; `schema.py:390`); it is not
  tunable for memory reduction. (#137 inherits this constraint.)

## 3. The three-tier GPU test taxonomy (central organizing principle)

We replace the cost-based markers (`gpu_inspection`, `gpu`) with **three
mutually-exclusive hardware tiers, partitioned by the smallest sufficient
hardware**, so each runner executes only its own tier with no recompute overlap.
Every GPU test carries **exactly one** tier marker.

| Tier marker | Hardware envelope | Runner | Selection |
| --- | --- | --- | --- |
| `gpu_local` | Fits the GTX 1080: ≤ ~7 GB effective VRAM, CC 6.0+, NF4 + float16. **Baseline members are forward-only / structural-inspection tests** (no backward graph). A QLoRA *training* smoke is **NOT** a member yet — with GC abandoned, training-fit in ~7 GB is unproven and deferred to **#137** (training tier PROVISIONAL). | Dev box via `uv sync --extra gpu-pascal`. | `run_gpu_tests.sh local`; notebook local cells. |
| `gpu_t4` | Needs > 8 GB but ≤ 16 GB, **or** requires bf16-representative numerics. | Colab T4 notebook. | `run_gpu_tests.sh t4`; notebook T4 cells. |
| `gpu_xl` | Beyond a T4: > 16 GB or a larger architecture. | Cloud auto-provision (#124). | `run_gpu_tests.sh xl`; gated with a clear "needs #124" skip reason. Likely near-empty initially. |

**Canonical partition.** The hardware tier is the *single* selection axis.
Cadence and cost (the old "run on every GPU PR" vs "run before release"
guidance) become **documentation in `gpu-test-policy.md`, not a selection
mechanism**. The `gpu_inspection` / `gpu` markers are **retired**: each test
that carried one is reclassified into exactly one hardware tier (§7, Workstream
D produces the per-test mapping). Test files keep `requires_compatible_gpu`,
`requires_checkpoint`, and `requires_bnb` as orthogonal capability gates.

**Why this is the spine.** Classifying every existing GPU test into a tier *is*
the #117 audit, and the classification is **calibrated empirically on the 1080**
(a test is `gpu_local` only if it actually runs within the ~7 GB budget on the
real card, not by estimate). The same tier set drives the notebook coverage
matrix (#116) and the runner script.

### 3.1 Marker mechanics

- Three pytest markers registered in `pyproject.toml`
  `[tool.pytest.ini_options].markers` and in `tests/conftest.py::pytest_configure`
  (which currently registers only `gpu_inspection`): `gpu_local`, `gpu_t4`,
  `gpu_xl`.
- `tests/conftest.py::pytest_collection_modifyitems` autoskips per tier against
  the **live hardware** of the current runner: a `gpu_t4`/`gpu_xl` test is
  skipped on the 1080 with a reason naming the required tier; all three are
  skipped on CPU-only CI exactly as today (they also carry
  `requires_compatible_gpu`). `gpu_xl` carries a "needs #124" skip reason when
  no XL runner is present.
- The legacy `integration` marker (CPU stub end-to-end tests) is **unaffected**.

## 4. Workstream A — Pascal environment

**Goal.** Reach the 1080's sm_61 kernels from an opt-in dependency group without
disturbing the default cu130 install.

### 4.1 Rationale

`pyproject.toml` pins `torch>=2.4` with no CUDA index, so `uv sync` resolves the
cu130 wheel (no sm_61). #79 proves the cu118 wheel covers sm_60..sm_90; PTX from
`compute_60` JIT-compiles to sm_61 at first kernel launch. bnb ≥ 0.43 NF4/FP4
kernels likewise target sm_60+. We need a *parallel*, opt-in resolution so the
dev box installs cu118 + bitsandbytes while everyone else stays on cu130.

### 4.2 Interfaces

- New optional-dependency group **`gpu-pascal`** in
  `[project.optional-dependencies]` (alongside the existing `qlora`,
  `tensorboard`, `wandb`, `jupyter`, `dev`), pinning `torch` to the cu118 index
  and including `bitsandbytes>=0.43`.
- New uv config blocks (none exist in `pyproject.toml` today):
  - `[[tool.uv.index]]` declaring the cu118 PyTorch index as **explicit**
    (`explicit = true`) so it is consulted only for packages routed to it.
  - `[tool.uv.sources]` routing `torch` (under the `gpu-pascal` extra) to that
    cu118 index.
  - A `[tool.uv]` conflicting-extras declaration marking `gpu-pascal` as
    mutually exclusive with the default cu130 torch resolution, so the two never
    co-resolve.
- Result contract:
  - `uv sync` (no extra) → cu130 torch, unchanged.
  - `uv sync --extra gpu-pascal` → cu118 torch + bitsandbytes, resolved
    independently.
  - `uv sync --extra dev` (the existing test-env path) → unchanged.
  - The exact uv table shapes are an implementation detail for the planner;
    this spec fixes only the contract above.

### 4.3 HARD-GATED FIRST MILESTONE (blocks all downstream work)

Before any other workstream proceeds, **empirically prove on the real 1080**:

1. A real torch CUDA kernel executes on sm_61 under the cu118 wheel — i.e. PTX
   JIT from `compute_60` to sm_61 succeeds at launch (a trivial CUDA matmul /
   element-wise op returns correct results, no "no kernel image is available"
   error).
2. A bitsandbytes `Linear4bit` (NF4) forward runs on the 1080 under float16
   compute dtype.

This milestone is run in-session on the dev box. Its evidence (commands +
output) is recorded in the manual-pass document (§6).

> **Note (historical):** this milestone **PASSED** on the real 1080 (§4.3
> evidence in the manual-pass doc); the fallback below was not invoked. It is
> retained for completeness.

**Fallback if the milestone fails** (e.g. the cu118 wheel turns out to ship no
sm_61-compatible PTX, or bnb's prebuilt wheel rejects sm_61): stop the Pascal
track, document the negative result in the manual-pass record and
`gpu-test-policy.md`, and ship the `gpu_local` tier empty (defined and wired, but
with no member tests). Workstreams D and E still complete on the tiers that do
have hardware. The PR remains shippable; only the Pascal-specific validations
convert to deferred follow-ups.

### 4.4 Acceptance criteria

- `uv sync` with no extra still resolves a cu130 torch (verified: resolution
  log shows a `+cu130` wheel).
- `uv sync --extra gpu-pascal` resolves a `+cu118` torch and installs
  `bitsandbytes`.
- The two milestone proofs (§4.3) pass on the real 1080, with output captured in
  the manual-pass record — **or** the §4.3 fallback is invoked and documented.

### 4.5 Dependencies

None upstream. **Everything in B/C downstream depends on §4.3 passing.**

## 5. Workstream B — Code enablement

**Goal.** Let the existing training and test code run on CC 6.0 hardware in
float16, and wire the three tier markers.

### 5.1 GPU-compatibility gate (closes a #79 follow-up)

- `tests/conftest.py::_has_compatible_gpu` currently returns
  `(major, minor) >= (7, 5)`. Lower the floor to **`>= (6, 0)`**.
- Rewrite the skip reasons (currently the literal
  `"real SAM 3.1 forward requires a CUDA GPU with CC >= 7.5"`) to distinguish:
  - NF4 QLoRA + LoRA work from **CC 6.0** (Pascal), and
  - `LLM.int8()` needs CC 7.5 but **is unused here**.
- This directly closes the #79 follow-up "Tighten CC-7.5 skip message in
  `tests/conftest.py`".

### 5.2 Device-aware dtype coercion

A helper (new; see below for placement) coerces `bfloat16 → float16` when the
target device has **CC < 8.0**, emitting a **one-time** warning. It is invoked at
two resolution points:

1. **Autocast.** `src/custom_sam_peft/train/loop.py::_autocast_ctx` currently
   maps any non-`bfloat16` dtype to `torch.float16` and otherwise selects
   `torch.bfloat16`. It must instead route through the coercion helper so that a
   `bfloat16` config on a CC<8.0 device autocasts in float16. (Note: this path
   is only reached when the PEFT method does **not** disable outer autocast —
   QLoRA disables it, see §5.4.)
2. **QLoRA compute dtype.** `apply_qlora`
   (`src/custom_sam_peft/peft_adapters/qlora.py`) builds each `bnb.nn.Linear4bit`
   with `compute_dtype=_torch_dtype(qcfg.compute_dtype)` inside
   `_replace_with_bnb_linear4bit`. On a CC<8.0 device, a `compute_dtype` of
   `bfloat16` must coerce to `float16` (with the same one-time warning) before
   the `Linear4bit` is constructed.

The helper's home module is an implementation choice for the planner; the
contract is: **input** (requested `Dtype`, target device or its compute
capability) → **output** (`Dtype`, coerced to `float16` iff CC < 8.0 and the
request was `bfloat16`), with at most one warning per process run.

### 5.3 Preset decision dtype fidelity

`src/custom_sam_peft/presets.py` currently hardcodes float dtype:

- `PresetDecision.dtype` is annotated `Literal["bfloat16"]` (line 83) and
  constructed with the literal `dtype="bfloat16"` in `decide_preset` (line 335).
- `PresetDecision.label()` emits the hardcoded token `bf16` (line 120).
- `_bytes_per_param_for_method` (line 142) comments `# bf16 vs NF4`; the byte
  count (2.0 B/param) is identical for float16, so the memory model is unchanged,
  but the comment should acknowledge float16.

Changes:

- Widen `PresetDecision.dtype` to **`Literal["bfloat16", "float16"]`** (matching
  the schema's `Dtype`).
- `decide_preset` selects `float16` when the detected device CC < 8.0, else
  `bfloat16`, and sets `PresetDecision.dtype` accordingly. (CC is already
  obtainable; `decide_preset` reads `torch.cuda.get_device_properties(0)` for
  total memory at line 295 — extend it to read capability.)
- `label()` renders the **real** dtype token (`fp16` or `bf16`) instead of the
  hardcoded `bf16`.
- `config_patch` already emits `self.dtype` into `model.dtype`, so a float16
  decision flows into the generated config automatically once `dtype` is
  correct.

### 5.4 Tier-marker wiring

- Register `gpu_local`, `gpu_t4`, `gpu_xl` (§3.1) in `pyproject.toml` and
  `tests/conftest.py::pytest_configure`; retire `gpu_inspection`/`gpu`
  registrations once all tests are reclassified (Workstream D).
- Extend `pytest_collection_modifyitems` with the per-tier hardware autoskip
  (§3.1).
- Update `scripts/run_gpu_tests.sh` to accept `{local, t4, xl}` (replacing
  `{inspection, release, all}`), mapping each to its marker and path set, and to
  **add `tests/predict/`** to the collected paths (the current script omits it,
  so `tests/predict/test_gpu_predict.py` is never collected by the runner). Fix
  the script's stale header counts. Preserve the `--deselect` convention and the
  `gpu-deselect-check` CI guard.

Note on QLoRA autocast: `_autocast_ctx` returns `nullcontext()` when
`peft_method.disables_outer_autocast()` is true, which QLoRA does (the qlora
adapter deliberately avoids outer autocast — see the extended rationale in
`qlora.py::_freeze_non_adapter`). So for the QLoRA path (and any #137
training-fit work), dtype routing happens at the `Linear4bit` `compute_dtype`
(§5.2 point 2), not at autocast.

### 5.5 Acceptance criteria

- On the 1080, `_has_compatible_gpu()` returns `True`; on CPU it returns
  `False`; the skip reasons read correctly (NF4/CC-6.0 vs unused-LLM.int8).
- A `bfloat16` config requested on the 1080 trains in float16 with exactly one
  coercion warning per process.
- `decide_preset` on the 1080 returns `dtype="float16"`; `label()` shows `fp16`;
  the generated `config_patch` carries `model.dtype: float16`.
- All existing CPU unit tests for `presets.py` still pass (the float16 widening
  must not break the `Literal` round-trip in `to_json`/`from_json`).
- `run_gpu_tests.sh local|t4|xl` each collect only their tier; the script no
  longer accepts the retired tier names; `tests/predict/` is collected.

### 5.6 Dependencies

Depends on A's milestone (§4.3) for the on-1080 acceptance checks. The dtype
helper and preset changes have CPU-testable logic (capability passed as a value)
that can be unit-tested without the GPU.

## 6. Workstream C — `use_double_quant`; GC abandoned; 8 GB training deferred to #137

**Goal (reduced).** Ship the GC-independent QLoRA improvement (`use_double_quant`,
C-1). The two GC-dependent threads originally in this workstream — the #127 GC
fix and the calibrated 8 GB QLoRA *training* recipe — are **abandoned** and
**deferred to #137** respectively (see §6.1 and §6.3 below). The `gpu_local`
**training** tier is **PROVISIONAL pending #137**; its current members are
forward-only / structural-inspection tests (no backward graph), classified in
Workstream D.

### 6.1 GC fix — ABANDONED (not in scope)

The original C-1 (Phase-0 diagnostic) and C-2 (Phase-1 fix) completed PR #127's
deferred gradient-checkpointing fix on the 1080. **This is abandoned.** The
Phase-0 diagnostic was genuinely run on the real GTX 1080 (under float16, with
`torch.utils.checkpoint.set_checkpoint_debug_enabled(True)`) and captured the
`CheckpointError` as a **structural save-count divergence INSIDE sam3's
`multi_head_attention_forward` recompute** — a CPU int64 scalar materializing at
a non-deterministic position in the autograd save list. Fix A
(deterministic-autocast wrap), Fix A+ (SDPA-MATH pinning), and Fix B (owning the
checkpoint with a pinned `context_fn`) were all attempted on real hardware and
none resolve it; Fix C (`determinism_check="none"`) yields a corrupted backward
(proving the divergence is not benign). The only resolution would edit
`sam3/model/model_misc.py`, which is **FORBIDDEN** (sam3 is external; we
monkeypatch only via `_patches/`). The full trace, fix-by-fix evidence, and root
cause are recorded in
`docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` ("Phase-0 trace + fix
classification"). **Outcome:** PR #127 closed unmerged; #89 and #60 closed
not-planned; `origin/main` already ships GC OFF (config default `false` + a
no-op guard, templates `false`), so **no GC fix ships and there is nothing to
revert.** `vit_act_checkpoint.py` (the merged flag-flip-only patch) is removed by
the GC-free rebuild of this branch.

> The independent fp16 cross-attention `attn_mask` MHA-hook fix (Blocker 1 in
> the evidence doc) is **not** GC work — it is a standalone float16 correctness
> win and **landed** regardless.

### 6.2 `use_double_quant` (C-1) — KEEP (GC-independent, landed)

Add a **`use_double_quant`** field to `QLoRAConfig`
(`src/custom_sam_peft/config/schema.py`, originally only `quant_type` and
`compute_dtype`) and wire it into the 4-bit construction. The actual bnb call
site is `_replace_with_bnb_linear4bit` in `qlora.py`, which constructs
`bnb.nn.Linear4bit(...)` **directly** (the repo does not use a
`BitsAndBytesConfig` object). Double-quant is enabled by passing the bnb
double-quant flag (`compress_statistics`) to that `Linear4bit` constructor (and
persisted in the `custom_sam_peft_qlora.json` metadata if the round-trip needs
it — its `format_version` bumps if the metadata shape changes). This is
GC-independent and **already landed** on the branch (task C-1).

### 6.3 8 GB QLoRA *training* recipe — DEFERRED to #137

The calibrated 8 GB QLoRA *training* recipe — originally a new
`configs/examples/gpu_smoke_qlora_8gb.yaml`, a `gpu_local`-tagged
`tests/gpu/test_real_train_qlora_8gb.py`, and a Phase-3 on-1080 calibration — was
**contingent on gradient checkpointing being the dominant activation-memory
lever**. With GC abandoned, fitting a 1008² QLoRA *training* step (a full
backward graph over the ViT-Det activations) within the ~7 GB effective budget is
an **open question**, spun off to **issue #137** ("Investigate non-checkpointing
approaches to fit QLoRA training in an 8 GB VRAM budget", OPEN). #137 owns the
config, the training smoke, and the calibration. Neither the config nor the test
was created on this branch.

> `data.image_size` is **fixed at 1008** by SAM 3.1 (`sam3.py:749,753`;
> `schema.py:390`) and is **not** a VRAM lever; #137 inherits this constraint.
> The GC-independent levers that survive — NF4 + `use_double_quant`, the 8-bit
> optimizer (`adamw8bit`, `schema.py:97`), low LoRA rank/scope,
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `batch_size: 1` +
> grad-accum, and the `paged_adamw8bit` escape hatch — are #137's starting
> material.

### 6.4 `gpu_local` training tier — PROVISIONAL pending #137

Because the training recipe is deferred, the `gpu_local` tier **holds no QLoRA
training smoke**. Its baseline members are:

- **Forward-only / inference tests** — a single SAM 3.1 forward at 1008 in
  NF4 / float16 carries no backward graph, so activation memory is far lower and
  fits ~7 GB.
- **Structural-inspection tests** — model loading, LoRA application,
  weight-shape checks, and similar introspection that does not require a
  training step.

When #137 establishes a GC-free training-fit, a training smoke may be added to
`gpu_local` (or land as `gpu_t4` if it exceeds ~7 GB). Until then the **training
tier is PROVISIONAL**. Workstream D classifies the surviving GPU tests into the
three tiers (calibrated on the 1080).

### 6.5 float16-stability note (unchanged, GC-independent)

There is **no GradScaler anywhere in `src/`**, and none is added. The primary
QLoRA path **disables outer autocast** (§5.4), so it sidesteps the
mixed-precision-scaling concern entirely — the `Linear4bit` compute dtype is
float16 directly. The loop already has a NaN-skip policy and `nan_abort_after`.
If a non-QLoRA float16 path is later found to need scaling, that is a follow-up,
not in scope here.

### 6.6 Acceptance criteria

- `QLoRAConfig.use_double_quant` exists, defaults to a value that preserves
  current behavior for existing configs, and is honored at the `Linear4bit`
  construction site (C-1, landed).
- No GC fix ships; `vit_act_checkpoint.py` and `tests/gpu/test_grad_checkpointing.py`
  are removed by the GC-free rebuild. The Phase-0 abandonment is recorded in the
  manual-pass document.
- The `gpu_local` tier (Workstream D) holds forward-only / structural-inspection
  tests only; the training tier is documented as PROVISIONAL pending #137.

### 6.7 Dependencies

C-1 (`use_double_quant`) depends on B (§5.1–5.2) for the dtype-coercion seam it
shares with QLoRA construction; it is otherwise CPU-testable. The GC fix and the
8 GB training recipe are out of scope (abandoned / deferred to #137).

## 7. Workstream D — #117 full (CPU/GPU split audit)

**Goal.** Inventory **every** GPU-gated test, classify it into a hardware tier,
decide keep-GPU vs move-to-CPU vs delete, perform the CPU moves, and report the
coverage delta.

### 7.1 Inventory (verified against the current tree, GC-free)

After the GC-free rebuild there are **12 GPU-tagged test files**.
`test_grad_checkpointing.py` (a GC-branch file) is **removed** by the rebuild,
and `test_real_train_qlora_8gb.py` was **never created** (deferred to #137), so
neither is in the audit inventory:

- `tests/gpu/` (8): `test_calibrate_real.py`, `test_channel_adapter_gpu.py`,
  `test_multiplex_vram.py`, `test_predict_nchannel_gpu.py`,
  `test_real_train_overfits.py`, `test_real_train_qlora.py`,
  `test_real_train_qlora_resume.py`, `test_run_end_to_end_gpu.py`.
- `tests/integration/` (3): `test_load_sam31_real.py`, `test_peft_lora_real.py`,
  `test_peft_qlora_real.py`.
- `tests/predict/` (1): `test_gpu_predict.py`.

### 7.2 Interfaces (the audit deliverable)

A new audit document (or a dedicated section of the refreshed
`gpu-test-policy.md` — §9 placement is the planner's call) that, for **every**
GPU-gated test (file::test granularity), records:

- the assigned **hardware tier** (`gpu_local` / `gpu_t4` / `gpu_xl`), calibrated
  on the 1080 where the test can run there;
- a **keep-GPU / move-to-CPU / delete** decision with rationale;
- for move-to-CPU decisions, the replacement mechanism (`TinySam3Stub` — already
  in `tests/fixtures/tiny_sam3_stub.py` and used via the `stub_model` fixture —
  synthetic tensors, or mocks).

Then **perform** the move-to-CPU refactors the audit calls for, and **report
coverage before/after** (the suite enforces an 80% gate on the full pytest run,
so moved tests must not regress total coverage).

### 7.3 Acceptance criteria

- Every one of the **12** GPU-tagged tests appears in the audit with a tier +
  decision + rationale.
- Each remaining GPU test carries exactly one tier marker and no legacy
  `gpu`/`gpu_inspection` marker.
- The move-to-CPU refactors land and pass; the full-suite coverage number is
  reported before and after and does not drop below 80%.

### 7.4 Dependencies

Depends on B (§5.4) for the tier markers. The tier *calibration* for
`gpu_local` candidates depends on A (§4.3).

## 8. Workstream E — #116 full (notebook coverage)

**Goal.** Guarantee every GPU test is reachable from a notebook cell (or
documented as intentionally excluded), with the runner script and notebooks
selecting by the three tiers.

### 8.1 Rationale

`notebooks/colab_gpu_tests.ipynb` historically ran tiers via
`scripts/run_gpu_tests.sh`. Several GPU tests added since the original audit are
**not referenced** by any cell: `test_real_train_qlora_resume.py`,
`test_channel_adapter_gpu.py`, `test_multiplex_vram.py`,
`test_predict_nchannel_gpu.py`, `test_calibrate_real.py`, and
`tests/predict/test_gpu_predict.py`.

### 8.2 Interfaces

- A **coverage matrix**: every GPU test ↔ its notebook cell ↔ its hardware tier.
- For each currently-unreferenced test, **either** add a notebook cell **or**
  document an intentional exclusion (with reason) in the matrix.
- Sync `scripts/run_gpu_tests.sh`: fix the stale counts, add `tests/predict/` to
  the path set, and select by the three tiers (this is the same script edit as
  §5.4 — listed here as the #116-facing acceptance surface).
- The notebook selects tiers by calling `run_gpu_tests.sh {local,t4,xl}`. Local
  cells are informational on Colab (the 1080 is not the Colab runtime), but the
  matrix documents which tier each cell targets.

### 8.3 Acceptance criteria

- The coverage matrix covers all **12** GPU tests with no gaps: each is either
  cell-referenced or explicitly excluded with a reason.
- `run_gpu_tests.sh` collects `tests/predict/` and selects by `{local,t4,xl}`;
  header counts are accurate.

### 8.4 Dependencies

Depends on B (§5.4) and D (§7) for the final tier assignments.

## 9. Cross-cutting documentation

- **Update `docs/testing/gpu-test-policy.md`**: state the CC 6.0 floor;
  float16-on-Pascal; replace/augment the cost tiers with the three-tier hardware
  taxonomy (cost/cadence demoted to guidance); refresh the inventory to the
  **12** GC-free GPU-tagged tests (correcting the stale "12 tests"/per-tier
  counts to the GC-free roster).
- **New `docs/testing/local-pascal-gpu-testing.md`**: how to provision the
  `gpu-pascal` uv extra, how to run the `gpu_local` tier on the 1080, and the
  float16 caveat (bf16 is emulated below CC 8.0).
- **Manual-pass record for the 1080 run** (`manual-gpu-pass-2026-05-24-gtx1080.md`),
  mirroring `docs/testing/manual-gpu-pass-2026-05-19.md`: the §4.3 milestone
  evidence (PASS) and the **Phase-0 trace + fix classification that established
  the GC abandonment** (this is the load-bearing evidence cited throughout §6).
  There is no Phase-3 GC calibration — that work is deferred to #137.

## 10. Risks

1. **sm_61 under cu118 is unproven.** PTX-JIT from compute_60 to sm_61, and bnb
   `Linear4bit` on the 1080, have not been run. **Mitigation:** the §4.3
   hard-gated milestone proves both before any downstream work; the §4.3
   fallback (revert to the T4 plan; ship `gpu_local` empty) keeps the PR
   shippable if it fails.
2. **The #127 GC fix proved infeasible on Pascal — RESOLVED as abandoned.** The
   Phase-0 diagnostic on the real 1080 showed the `CheckpointError` is a
   structural save-count divergence INSIDE sam3's `multi_head_attention_forward`
   recompute, unfixable without editing forbidden sam3 source (Fix A/A+/B all
   fail; Fix C corrupts the backward). **Resolution:** GC is abandoned (PR #127
   closed unmerged; #89/#60 not-planned; `origin/main` already ships GC OFF —
   nothing to revert), so this is no longer a live risk to the spec. The
   knock-on — gradient checkpointing was the dominant activation-memory lever for
   a 1008² training step, and there is no smaller image_size to fall back to —
   means fitting QLoRA *training* in ~7 GB is **unproven and deferred to #137**.
   The `gpu_local` **training** tier is consequently PROVISIONAL; its current
   members are forward-only / inspection tests (no backward graph). Full evidence:
   `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md`.
3. **float16 stability without a GradScaler.** No GradScaler exists or is added.
   **Mitigation:** the primary QLoRA path disables outer autocast and runs
   float16 directly at the `Linear4bit` compute dtype; the loop's existing
   NaN-skip policy / `nan_abort_after` guards it. (Any future training-fit work
   that exercises this at scale belongs to #137.)
4. **float16 on Pascal ≠ bf16 on T4.** Numerics calibrated on the 1080 in
   float16 do not certify the bf16 T4 release path. **Mitigation:** the bf16 T4
   confirmation is deferred to the §12 follow-on (`gpu_t4` tier) and to the
   existing 14/10 GB T4 ceilings, which this PR does not touch.
5. **Scope.** Five workstreams in a single PR (reduced: the GC-completion thread
   is dropped — see Risk 2). **Mitigation:** strict sequencing (A milestone gates
   everything; B before C/D/E; D before E); the §4.3 fallback bounds the blast
   radius if the environment work fails.
6. **Effective VRAM is ~7 GB, not 8.** WSL/Xwayland holds ~1 GB.
   **Mitigation:** all `gpu_local` tier classification targets the **measured
   ~7 GB effective budget**, not the 8 GB nameplate. (The 8 GB *training*-recipe
   ceiling is #137's concern, not this PR's.)

## 11. Open-but-resolved decisions

- **Hardware tier is the canonical partition; cost/cadence is documentation.**
  Resolved: the old `gpu`/`gpu_inspection` cost markers are retired, not kept in
  parallel, to avoid a two-axis selection model.
- **Pascal trains in float16, never bf16.** Resolved per #79: coerce + warn; no
  attempt at faithful bf16 on CC < 8.0.
- **Default install stays cu130.** Resolved: cu118 is reachable only via the
  opt-in `gpu-pascal` extra, isolated through uv explicit-index +
  conflicting-extras.
- **Double-quant is wired at the `Linear4bit` constructor**, not via a
  `BitsAndBytesConfig` (the repo constructs `Linear4bit` directly). Resolved
  against the actual `qlora.py` call site (C-1, landed).
- **Gradient checkpointing is abandoned.** Originally a deliberate Phase-0 branch
  point (Fix A the default expectation). The on-1080 trace proved every fix tier
  (A/A+/B) leaves the sam3-internal recompute divergence intact and Fix C
  corrupts the backward — unfixable without editing forbidden sam3 source.
  Resolved: GC dropped (PR #127 closed unmerged; #89/#60 not-planned); 8 GB
  *training* deferred to #137.
- **`gpu_xl` ships near-empty.** Resolved: defined and wired now, populated when
  #124 lands.

## 12. Follow-ups (filed, not done here)

- **#137 — "Investigate non-checkpointing approaches to fit QLoRA training in an
  8 GB VRAM budget"** (OPEN). Owns the deferred 8 GB QLoRA *training* recipe
  (`gpu_smoke_qlora_8gb.yaml`, the training smoke, and on-1080 calibration) now
  that gradient checkpointing — the dominant activation-memory lever — is
  abandoned. The `gpu_local` training tier is PROVISIONAL until #137 resolves.
- **"Operationalize the `gpu_t4` tier on Colab"** — the 14 GB / 10 GB
  release-ceiling gate plus the bf16 confirmation that cannot run on the 8 GB
  Pascal card. Captures Risk 4's deferred bf16-vs-float16 validation.
- **`gpu_xl` tier population** references the existing **#124** (cloud
  auto-provision) as its runner.

**Closed not-planned (no follow-up):** the #127 gradient-checkpointing fix
(#89, #60). The Phase-0 diagnostic proved it unfixable without editing forbidden
sam3 source; see §6.1 and the manual-pass evidence doc.

## 13. Sequencing summary

```
A (env + §4.3 milestone)  ── gate ──▶  B (code enablement)
                                            │
                                            ├─▶ C (use_double_quant / C-1 only; GC abandoned, 8 GB training → #137)
                                            ├─▶ D (#117 audit + CPU moves)
                                            └─▶ E (#116 notebook coverage)   [after D]
Docs (§9) updated alongside the workstream that produces each fact.
```
