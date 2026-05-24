# Minimal GPU Testing on a GTX 1080 — Design Spec

> Date: 2026-05-24
> Builds on: PR #127 (#89, gradient checkpointing) — merged into this branch.
> Source issues: #79 (research), #68 (8 GB VRAM floors), #117 (CPU/GPU test
> audit), #116 (notebook coverage audit), #124 (cloud auto-provision).
> Related specs: `docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md`
> and its plan `docs/superpowers/plans/2026-05-23-gradient-checkpointing-t4.md`;
> `docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md`;
> `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md`.

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

PR #127 re-enabled gradient checkpointing but **deferred the actual fix**. The
flag-flip half ships in `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py`,
which only sets `use_act_checkpoint=True` on each ViT-Det block. sam3
self-checkpoints via `checkpoint.checkpoint(blk, x, use_reentrant=False)`, which
raised a recompute-metadata `CheckpointError` on T4. The deterministic-autocast
wrap that resolves the mismatch was deferred pending a Phase-0 diagnostic that
needed a T4 we did not have.

**Goal.** Make the GTX 1080 a first-class GPU test target so PR #127's deferred
fix can be diagnosed, implemented, and verified on real hardware without a Colab
T4, and rationalize the GPU test suite (#117/#116). One spec, one PR, five
sequenced workstreams, built on the merged #127 code.

**Guiding principle.** *Any testing runnable on this branch gets run here.* The
1080 is live in WSL, so implementers run the real GPU validations in-session
(sm_61/bnb sanity, the Phase-0/1/3 grad-checkpointing flow, the 8 GB
calibration) rather than deferring them.

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
- **No GradScaler introduction.** See the float16-stability risk (§8) for why
  the primary QLoRA path sidesteps the need for one.
- **`image_size` is not a VRAM lever.** SAM 3.1's input resolution is fixed at
  1008×1008 by the model itself (`sam3.py:749,753`; `schema.py:390`); it is not
  tunable for memory reduction and is explicitly excluded from the 8 GB recipe's
  calibration knobs.

## 3. The three-tier GPU test taxonomy (central organizing principle)

We replace the cost-based markers (`gpu_inspection`, `gpu`) with **three
mutually-exclusive hardware tiers, partitioned by the smallest sufficient
hardware**, so each runner executes only its own tier with no recompute overlap.
Every GPU test carries **exactly one** tier marker.

| Tier marker | Hardware envelope | Runner | Selection |
| --- | --- | --- | --- |
| `gpu_local` | Fits the GTX 1080: ≤ ~7 GB effective VRAM, CC 6.0+, NF4 + float16. | Dev box via `uv sync --extra gpu-pascal`. | `run_gpu_tests.sh local`; notebook local cells. |
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

**Fallback if the milestone fails** (e.g. the cu118 wheel turns out to ship no
sm_61-compatible PTX, or bnb's prebuilt wheel rejects sm_61): stop the Pascal
track, document the negative result in the manual-pass record and
`gpu-test-policy.md`, and **fall back to the existing T4 plan** — i.e. PR #127's
deferred fix is diagnosed on Colab T4 per the original
`2026-05-23-gradient-checkpointing-t4-design.md`, and the `gpu_local` tier ships
empty (defined and wired, but with no member tests). Workstreams D and E still
complete on the tiers that do have hardware. The PR remains shippable; only the
Pascal-specific validations convert to deferred follow-ups.

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
`qlora.py::_freeze_non_adapter`). So for the primary 8 GB QLoRA recipe (§6),
dtype routing happens at the `Linear4bit` `compute_dtype` (§5.2 point 2), not at
autocast.

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

## 6. Workstream C — Complete #127 + the 8 GB recipe (core)

**Goal.** Finish PR #127's deferred grad-checkpointing fix on the real 1080 and
ship a calibrated 8 GB QLoRA recipe.

### 6.1 Phase 0 — diagnostic on the 1080

Run the QLoRA fast smoke with gradient checkpointing **on** and
`torch.utils.checkpoint.set_checkpoint_debug_enabled(True)`, under **float16**,
on the 1080. Capture the `CheckpointError` (or its absence) and classify the
divergence per the fix taxonomy already defined in
`2026-05-23-gradient-checkpointing-t4-design.md`:

- **Fix A** — deterministic-autocast wrap (the default expectation): wrap each
  checkpointed block forward so the recompute pass runs under the same autocast
  state as the forward, eliminating the recompute-metadata mismatch.
- **Fix B / Fix C** — the lower/higher-effort alternatives enumerated in the T4
  spec, selected only if the trace justifies them.

This replaces the deferred Colab-T4 Phase-0 diagnostic with an on-1080 run. The
diagnostic is **the branch point**: the fix tier implemented in Phase 1 is
whatever the captured trace justifies.

### 6.2 Phase 1 — implement the justified fix

Implement the lowest-tier fix the Phase-0 trace justifies in
`src/custom_sam_peft/models/_patches/vit_act_checkpoint.py`. The patch's `apply`
already receives a `runtime` argument that is "unused by the flag-flip half but
is part of the patch contract and is consumed by the deterministic-autocast wrap
added in the Phase-1 fix task" — i.e. the seam already exists. Default
expectation is **Fix A**.

Add a **`use_double_quant`** field to `QLoRAConfig`
(`src/custom_sam_peft/config/schema.py`, currently only `quant_type` and
`compute_dtype`) and wire it into the 4-bit construction. The actual bnb call
site is `_replace_with_bnb_linear4bit` in `qlora.py`, which constructs
`bnb.nn.Linear4bit(...)` **directly** (the repo does not use a
`BitsAndBytesConfig` object). Double-quant is enabled by passing the bnb
double-quant flag to that `Linear4bit` constructor (and persisted in the
`custom_sam_peft_qlora.json` metadata if the planner judges the round-trip needs
it — its `format_version` must bump if the metadata shape changes).

### 6.3 New config and test

- New `configs/examples/gpu_smoke_qlora_8gb.yaml`, modeled on the existing
  `configs/examples/gpu_smoke_qlora.yaml` but tuned for the ~7 GB budget:
  NF4 + double-quant, `model.dtype: float16`, `peft.qlora.compute_dtype:
  float16`, `model.gradient_checkpointing: true`, low rank (`peft.r ≈ 8`),
  `optimizer: adamw8bit`, `batch_size: 1` with grad-accum. Note that
  `data.image_size` is **not** a knob here: `load_sam31` hardcodes 1008 into
  both `_Sam3ImageAdapter` and `Sam3Wrapper` (`sam3.py:749,753`), and the
  wrapper contract assumes 1008² inputs with box-scaling built at that
  resolution (`schema.py:390`). Activation memory at 1008² is therefore fixed.
  The empirically-calibrated knobs are **LoRA rank and scope** (and optionally
  optimizer paging — see §6.x).
- New `tests/gpu/test_real_train_qlora_8gb.py`, tagged **`gpu_local`**,
  `requires_compatible_gpu`, `requires_checkpoint`, `requires_bnb`. It runs the
  8 GB recipe end-to-end and asserts: grad-ckpt on completes without
  `CheckpointError`, first-step loss is finite, and peak VRAM stays within the
  empirically-calibrated ceiling (see Phase 3). It reuses the `tiny_coco`
  fixture per the data-size policy.

### 6.x VRAM-minimization levers (image_size is fixed at 1008)

Because `data.image_size` cannot be reduced, the 8 GB recipe reaches the ~7 GB
effective budget through the following levers, listed in rough impact order:

1. **Gradient checkpointing ON (#127)** — the dominant activation-memory lever.
   Without it, the backward graph over 1008² ViT-Det activations almost certainly
   will not fit ~7 GB. This makes the #127 fix **load-bearing** for the training
   recipe; see §6.8 (graceful-degradation) if it cannot be made to work on Pascal.
2. **NF4 4-bit base + double-quant** — compresses base model weight memory by
   roughly 4× vs float16 and adds a second level of quantization for the
   quantization constants themselves.
3. **8-bit optimizer (`optimizer: adamw8bit`, already available via
   `schema.py:97`)** — halves optimizer-state memory vs 32-bit Adam without
   requiring any new schema or trainer changes.
4. **Low LoRA rank (r=4 or 8) + narrow scope (attention-only vs
   `vision_decoder`)** — fewer trainable parameters means a smaller retained
   backward graph AND less optimizer state.
5. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — mitigates allocator
   fragmentation on a tight budget; set in the runner environment, not in config.
6. **`batch_size: 1` + gradient accumulation** — already the minimal batch;
   noted for completeness.

**Additive fallback lever (invoke only if calibration shows the above do not
fit):** a paged / CPU-offloaded optimizer such as `bnb.optim.PagedAdamW8bit`.
This would require adding a `paged_adamw8bit` value to the `Optimizer` literal
in `schema.py:97` (currently `adamw | adamw8bit | auto`) and wiring the new
branch in `train/trainer.py::_build_optimizer`. It is an **escape hatch**, not a
default — attempt the recipe with the levers above before reaching for it.

### 6.4 Phase 3 — verify on the 1080

On the real 1080:

- grad-ckpt **on** → no `CheckpointError` (the fix holds).
- **first-step loss parity** between ckpt-on and ckpt-off (recompute is
  numerically exact) — mirroring the existing
  `tests/gpu/test_grad_checkpointing.py` assertion style.
- **peak VRAM lower** with ckpt on than off.
- **Empirically calibrate the 8 GB ceiling** on the real card against the
  effective ~7 GB budget: find the largest (rank, scope) that fits — and whether
  the paged-optimizer fallback (§6.x) is needed — set the VRAM assertion in
  `test_real_train_qlora_8gb.py` to the measured peak plus a small margin (same
  philosophy as the T4 ceilings), and bake the calibrated values into
  `gpu_smoke_qlora_8gb.yaml`. `image_size` is not tuned; it remains 1008.

### 6.5 float16-stability note

There is **no GradScaler anywhere in `src/`**, and none is added. The primary 8
GB path is QLoRA, which **disables outer autocast** (§5.4), so it sidesteps the
mixed-precision-scaling concern entirely — the `Linear4bit` compute dtype is
float16 directly. Phase 3 must **verify no NaN** appears across the smoke run
(the loop already has a NaN-skip policy and `nan_abort_after`; confirm it does
not trip). If a non-QLoRA float16 path is later found to need scaling, that is a
follow-up, not in scope here.

### 6.6 Acceptance criteria

- Phase-0 trace captured and the chosen fix tier (A/B/C) recorded in the
  manual-pass document, with the rationale.
- With the Phase-1 fix in place, `tests/gpu/test_grad_checkpointing.py` and the
  new `tests/gpu/test_real_train_qlora_8gb.py` pass on the 1080.
- `QLoRAConfig.use_double_quant` exists, defaults to a value that preserves
  current behavior for existing configs, and is honored at the `Linear4bit`
  construction site.
- `gpu_smoke_qlora_8gb.yaml` trains to completion on the 1080 within the
  calibrated ceiling, no NaN, loss moving.

### 6.7 Dependencies

Depends on A (§4.3) and B (§5.1–5.2). Phase 1 depends on Phase 0's
classification (diagnostic-driven branch). The training smoke (`gpu_local`)
is contingent on grad-checkpointing fitting within ~7 GB; see §6.8 for the
graceful-degradation path if it does not.

### 6.8 Graceful degradation — if training cannot fit ~7 GB

The 8 GB **training** recipe is contingent on the #127 grad-checkpointing fix
actually fitting a training step within the ~7 GB effective budget. If it
cannot — either because the fix proves infeasible on Pascal (CC 6.1 / float16
recompute complications), or because peak VRAM still exceeds ~7 GB after every
lever in §6.x including the paged-optimizer fallback — then:

- The QLoRA **training** smoke (`test_real_train_qlora_8gb.py`) is reclassified
  **`gpu_t4`**: it runs on Colab T4, not on the 1080.
- **`gpu_local` is NOT empty.** It retains:
  - **Forward-only / inference tests** — a single SAM 3.1 forward at 1008 in
    NF4 / float16 carries no backward graph, so activation memory is far lower
    and likely fits ~7 GB.
  - **Structural-inspection tests** — model loading, LoRA application,
    weight-shape checks, and similar introspection that does not require a
    training step.

This mirrors the §4.3 fallback pattern: the PR remains shippable either way;
only the on-1080 training validation converts to a deferred T4 follow-up.

## 7. Workstream D — #117 full (CPU/GPU split audit)

**Goal.** Inventory **every** GPU-gated test, classify it into a hardware tier,
decide keep-GPU vs move-to-CPU vs delete, perform the CPU moves, and report the
coverage delta.

### 7.1 Inventory (verified against the current tree)

There are **13 GPU-tagged test files** today (the `gpu-test-policy.md` "12 tests"
inventory is stale and predates several additions):

- `tests/gpu/` (9): `test_calibrate_real.py`, `test_channel_adapter_gpu.py`,
  `test_grad_checkpointing.py`, `test_multiplex_vram.py`,
  `test_predict_nchannel_gpu.py`, `test_real_train_overfits.py`,
  `test_real_train_qlora.py`, `test_real_train_qlora_resume.py`,
  `test_run_end_to_end_gpu.py`. (Plus the new `test_real_train_qlora_8gb.py`
  from Workstream C.)
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

- Every one of the 13 (then 14, with the new 8 GB test) GPU-tagged tests appears
  in the audit with a tier + decision + rationale.
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

- The coverage matrix covers all GPU tests (including the new 8 GB test) with no
  gaps: each is either cell-referenced or explicitly excluded with a reason.
- `run_gpu_tests.sh` collects `tests/predict/` and selects by `{local,t4,xl}`;
  header counts are accurate.

### 8.4 Dependencies

Depends on B (§5.4) and D (§7) for the final tier assignments.

## 9. Cross-cutting documentation

- **Update `docs/testing/gpu-test-policy.md`**: state the CC 6.0 floor;
  float16-on-Pascal; replace/augment the cost tiers with the three-tier hardware
  taxonomy (cost/cadence demoted to guidance); refresh the inventory to the 13
  (then 14) tests and correct the "12 tests" claim and the stale per-tier counts.
- **New `docs/testing/local-pascal-gpu-testing.md`**: how to provision the
  `gpu-pascal` uv extra, how to run the `gpu_local` tier on the 1080, and the
  float16 caveat (bf16 is emulated below CC 8.0).
- **New manual-pass record for the 1080 run**, mirroring the structure of
  `docs/testing/manual-gpu-pass-2026-05-19.md`: the §4.3 milestone evidence, the
  Phase-0 trace + fix classification, and the Phase-3 calibration numbers.

## 10. Risks

1. **sm_61 under cu118 is unproven.** PTX-JIT from compute_60 to sm_61, and bnb
   `Linear4bit` on the 1080, have not been run. **Mitigation:** the §4.3
   hard-gated milestone proves both before any downstream work; the §4.3
   fallback (revert to the T4 plan; ship `gpu_local` empty) keeps the PR
   shippable if it fails.
2. **The #127 fix is diagnostic-driven and load-bearing.** Phase 1's fix tier
   (A/B/C) is unknown until Phase 0 captures the trace on the 1080. More
   critically, gradient checkpointing is the **dominant activation-memory lever**
   for the training recipe: without it, 1008² ViT-Det activations almost
   certainly do not fit ~7 GB, and — unlike T4-tier work — there is no smaller
   image_size to fall back to (SAM 3.1's input resolution is fixed at 1008).
   **Mitigation:** the branch point is explicit (§6.1); Fix A is the default
   expectation and the patch seam already accepts the `runtime` it needs. If the
   fix proves infeasible on Pascal, or training still OOMs after every lever in
   §6.x including paging, the §6.8 graceful-degradation path applies: the
   training smoke is reclassified `gpu_t4`, and `gpu_local` retains
   forward-only / inspection tests (which carry no backward graph).
3. **float16 stability without a GradScaler.** No GradScaler exists or is added.
   **Mitigation:** the primary QLoRA path disables outer autocast and runs
   float16 directly at the `Linear4bit` compute dtype; Phase 3 verifies no NaN
   via the existing NaN-skip policy.
4. **float16 on Pascal ≠ bf16 on T4.** Numerics calibrated on the 1080 in
   float16 do not certify the bf16 T4 release path. **Mitigation:** the bf16 T4
   confirmation is deferred to the §11 follow-on (`gpu_t4` tier) and to the
   existing 14/10 GB T4 ceilings, which this PR does not touch.
5. **Scope.** Five workstreams plus finishing a deferred PR in a single PR.
   **Mitigation:** strict sequencing (A milestone gates everything; B before
   C/D/E; D before E); the §4.3 fallback bounds the blast radius if the
   environment work fails.
6. **Effective VRAM is ~7 GB, not 8.** WSL/Xwayland holds ~1 GB.
   **Mitigation:** all `gpu_local` calibration (the 8 GB recipe ceiling, the tier
   classification) targets the **measured ~7 GB effective budget**, not the 8 GB
   nameplate.

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
  against the actual `qlora.py` call site.
- **Default expected fix is Fix A** (deterministic-autocast wrap), but the
  actual fix is whatever Phase 0 justifies. Resolved as a deliberate branch
  point.
- **`gpu_xl` ships near-empty.** Resolved: defined and wired now, populated when
  #124 lands.

## 12. Follow-ups (filed, not done here)

- **"Operationalize the `gpu_t4` tier on Colab"** — the 14 GB / 10 GB
  release-ceiling gate plus the bf16 confirmation that cannot run on the 8 GB
  Pascal card. Captures Risk 4's deferred bf16-vs-float16 validation.
- **`gpu_xl` tier population** references the existing **#124** (cloud
  auto-provision) as its runner.
- If the §4.3 milestone fails, file the Pascal-track-blocked follow-ups named in
  §4.3 (diagnose #127 on T4; `gpu_local` empty).

## 13. Sequencing summary

```
A (env + §4.3 milestone)  ── gate ──▶  B (code enablement)
                                            │
                                            ├─▶ C (#127 fix + 8 GB recipe)  [Phase 0 → branch → Phase 1 → Phase 3]
                                            ├─▶ D (#117 audit + CPU moves)
                                            └─▶ E (#116 notebook coverage)   [after D]
Docs (§9) updated alongside the workstream that produces each fact.
```
