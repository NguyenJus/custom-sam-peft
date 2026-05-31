<!-- markdownlint-disable MD013 -->

# GPU test migration: re-architect testing around the RTX 5070 Ti

**Status:** locked design, single PR, no back-compat shims.
**Closes:** #142, #139, #193, #195, #83 (all OPEN). Files one new issue (gpu_xl tier).
**Supersedes:** `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md` (the `gpu_inspection`/`gpu` two-tier policy and its Pascal-anchored framing).
**Hardware context:** the dev box GPU was upgraded from a **GTX 1080** (Pascal, sm_61 / CC 6.1, 8 GB) to an **RTX 5070 Ti** (Blackwell, sm_120 / CC 12.0, 16 GB).

This spec re-architects the tiered GPU-test policy around the new card. It drops
Pascal support, sets the **Tesla T4 (CC 7.5)** as the minimum supported GPU, names
the tiers by capability (not by a specific dev card), auto-detects tiers from live
hardware, adds an empirical "won't fit a small card" warning to the train path,
hardens CPU/stub coverage for the classes of GPU bug that previously escaped, and
adds a non-blocking GPU-evidence gate. It is written to be implemented cold: every
changed file and the key symbols are named with line anchors.

---

## §0 Background and the decision that frames this spec

### The old tier system (being replaced)

- **Markers** registered in `pyproject.toml` (~lines 128–134) and `tests/conftest.py`:
  - `gpu_local` — "fits the GTX 1080" (≤ ~7 GB, CC 6.0+, NF4 + float16).
  - `gpu_t4` — ">8 and ≤16 GB, or bf16-representative numerics, or a real training loop" (Colab T4).
  - `gpu_xl` — ">16 GB"; empty.
- **Policy doc:** `docs/testing/gpu-test-policy.md`. **Audit:** `docs/testing/gpu-audit-2026-05-24.md`. **Pascal notes:** `docs/testing/local-pascal-gpu-testing.md`, `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md`.
- **Runner:** `scripts/run_gpu_tests.sh [local|t4|xl]` — `local` runs one pytest process per file (checkpoint memory release); maps tiers to marker filters; has a stateful `--deselect` convention; CI job `gpu-deselect-check` greps for leftover `--deselect`.
- **Gating** in `tests/conftest.py`: `_has_compatible_gpu()` (CC ≥ 6.0 + a real kernel-launch probe `_torch_can_launch_kernel`), `_current_tier()` (returns only `gpu_local` from live HW — see `tests/conftest.py:53-69`; t4/xl asserted out-of-band), `pytest_collection_modifyitems` skips tests above the runner's tier via `_TIER_ORDER`. Autouse `_free_cuda_after_gpu_test` fixture frees CUDA cache after each GPU test (gated on `requires_compatible_gpu`). Markers `requires_compatible_gpu` (CC ≥ 6.0 — `tests/conftest.py:45-50`), `requires_checkpoint` (skip unless `models/sam3.1/sam3.1_multiplex.pt` exists), `requires_bnb`.

### Two facts that invalidate the old taxonomy's premises

1. **bf16 is emulated below CC 8.0 — so the T4 was never a faithful-bf16 target.**
   `coerce_dtype_for_capability(...)` at `src/custom_sam_peft/runtime/_runtime.py:61`
   coerces `bfloat16 → float16` when `capability < (8, 0)` (the early return at
   line 83: `if capability >= (8, 0): return dtype`). The Tesla T4 is CC 7.5 < 8.0,
   so it **coerces bf16 → fp16**. The old `gpu_t4` "bf16-representative numerics"
   claim was always false. The 5070 Ti (CC 12.0) is the project's **first**
   native-bf16 test card.
2. **Flash-attention requires CC ≥ 8.0 — so the T4 was never Flash-capable either.**
   `src/custom_sam_peft/presets.py:225` gates Flash availability on `cc >= (8, 0)`;
   below that, SDPA falls back to the MATH backend (materializes the N×N score
   matrix, heavier memory). The 5070 Ti is the first Flash-capable test card; T4 /
   Pascal use math SDPA.

### Decision authority (the locked call)

The user explicitly decided to **drop Pascal support** and set the **Tesla T4
(CC 7.5)** as the minimum supported GPU, overriding an earlier "Pascal predict-only"
framing. This spec reflects the final decision only. `coerce_dtype_for_capability`
**runtime behavior is unchanged** (the T4 still legitimately coerces bf16 → fp16);
only the *test gate* moves.

### GPU-gated test inventory (27 tests, 12 files)

17 `gpu_local` + 10 `gpu_t4` + 0 `gpu_xl`. Two files carry per-test markers:

- `tests/integration/test_load_sam31_real.py` — 2 `gpu_local` + 1 `gpu_t4` (K8 multiplex forward).
- `tests/predict/test_gpu_predict.py` — 2 `gpu_local` (base model + vram_hint) + 2 `gpu_t4` (LoRA predict + QLoRA predict, which train an adapter as setup).

Other GPU files: `tests/integration/test_peft_lora_real.py`, `test_peft_qlora_real.py`;
`tests/gpu/test_calibrate_real.py`, `test_channel_adapter_gpu.py`,
`test_multiplex_vram.py`, `test_predict_nchannel_gpu.py`, `test_real_train_overfits.py`,
`test_real_train_qlora.py`, `test_real_train_qlora_resume.py`, `test_run_end_to_end_gpu.py`.

---

## §1 Goals and Non-Goals

### Goals

1. Replace the dev-card-named tiers (`gpu_local`/`gpu_t4`/`gpu_xl`) with
   **capability-named, auto-detected** tiers keyed on compute capability + VRAM.
2. Drop Pascal; make the **T4 (CC 7.5) the floor**. Move the gate from CC ≥ 6.0 to CC ≥ 7.5.
3. Add an **empirical** (never analytic) "trained model may not fit a small card"
   warning at model-ready time in the train path.
4. Prove an **8 GB / CC 7.5 card supports BOTH train and predict** of the small config.
5. **Maximize CPU coverage**: audit `tests/integration/` and add bounded CPU/stub
   regression tests for the *classes* of GPU bug that previously escaped.
6. Add a **non-blocking, permissioned** GPU-evidence gate; keep `gpu-deselect-check`.
7. Make the **5070 Ti the primary, automatable** test environment; keep a **minimal
   Colab T4** surface that proves "Colab works for this repo."
8. Honor the **cite-or-tbd** rule for every new numeric default/threshold.
9. Close #142, #139, #193, #195, #83 with concrete deliverables and close criteria.

### Non-Goals

- Cloud auto-provision (#125 — remains open; the right xref for deferred work).
- Tests requiring > 16 GB VRAM — deferred to the new `gpu_xl` issue (§10); `gpu_xl` stays empty here.
- Changing `coerce_dtype_for_capability` **runtime** behavior (only the test gate moves).
- Retuning **production** hyperparameters.
- **Any analytic VRAM model in the warning path** — the warning is empirical-only.
- Building a hosted/self-hosted GPU CI runner. The 5070 Ti runs are local; the Colab surface is user-triggered.

---

## §2 Tier taxonomy — capability-named, auto-detected (R1–R4)

### The three replacement tiers

| Tier | Gate | Cards | Contents | Numerics |
|------|------|-------|----------|----------|
| **`gpu_t4`** | CC ≥ 7.5 **AND** total VRAM ≤ 16 GB | real T4 **and** 5070 Ti | all predict + forward/structural tests; all training smokes that fit ≤16 GB (the current 10 `gpu_t4`); **PLUS** the current 17 `gpu_local` (merged in — the 1080 distinction is gone); **PLUS** #142's new 8 GB-ceiling QLoRA training smoke (§4). | fp16 (bf16 coerced here). |
| **`gpu_bf16`** | CC ≥ 8.0 **AND** total VRAM ≤ 16 GB | 5070 Ti only | the bf16-faithful-numerics test(s) the T4 cannot certify (small: 1–2 tests, §2 R4). Answers #139's "is bf16 faithful?" honestly. | native bf16 (NOT coerced). |
| **`gpu_xl`** | total VRAM > 16 GB | deferred | empty in this PR; populated only via the new issue (§10). | n/a |

- **`gpu_local` is deleted entirely** — its reason (the Pascal 1080) is retired. Its 17 tests merge into `gpu_t4`.
- Tiers are **no longer a strict linear order**: `gpu_bf16` is a *capability superset* of `gpu_t4` on CC ≥ 8.0 cards, but it is **NOT a VRAM superset of `gpu_xl`** (both `gpu_t4` and `gpu_bf16` are ≤16 GB bands; `gpu_xl` is the >16 GB band). The skip predicate (§3 R6) is therefore expressed as **capability-subset checks**, not an integer `_TIER_ORDER` comparison.

**R1 — Marker definitions.** `gpu_t4`, `gpu_bf16`, `gpu_xl` are registered (in
`pyproject.toml` markers ~line 128–134 **and/or** `tests/conftest.py` to match the
existing dual-registration convention; `--strict-markers` is on, so they MUST be
registered before any test using them is collected). Each docstring states its gate
(CC + VRAM band) and links `docs/testing/gpu-test-policy.md`. `gpu_local` is removed
from every registration site.
**Acceptance:** `pytest --collect-only` raises no strict-marker error; `grep -rn "gpu_local"` over `tests/`, `scripts/`, `pyproject.toml`, `docs/` returns zero live references (only dated historical docs per §3 R8 may mention it).

**R2 — Complete reclassification mapping (all 27 tests → new tier).** Default rule:
`gpu_local → gpu_t4`, `gpu_t4 → gpu_t4`. The mapping table below is the contract; the
implementer applies it test-by-test (per-test markers on the two mixed-tier files;
module-level `pytestmark` elsewhere where every test in the file shares a tier).

| File | Test(s) | Old tier | New tier |
|------|---------|----------|----------|
| `tests/integration/test_load_sam31_real.py` | `test_load_sam31_returns_wrapper` | gpu_local | gpu_t4 |
| | `test_load_sam31_forward_to_canonical` | gpu_local | gpu_t4 |
| | K8 multiplex forward test | gpu_t4 | gpu_t4 |
| `tests/integration/test_peft_lora_real.py` | all (module `pytestmark`) | gpu_local | gpu_t4 |
| `tests/integration/test_peft_qlora_real.py` | all (module `pytestmark`) | gpu_local/gpu_t4 (per file) | gpu_t4 |
| `tests/predict/test_gpu_predict.py` | base model | gpu_local | gpu_t4 |
| | vram_hint | gpu_local | gpu_t4 |
| | LoRA predict | gpu_t4 | gpu_t4 |
| | QLoRA predict | gpu_t4 | gpu_t4 |
| `tests/gpu/test_calibrate_real.py` | all | gpu_local/gpu_t4 | gpu_t4 |
| `tests/gpu/test_channel_adapter_gpu.py` | all | gpu_local | gpu_t4 |
| `tests/gpu/test_multiplex_vram.py` | all | gpu_t4 | gpu_t4 |
| `tests/gpu/test_predict_nchannel_gpu.py` | all | gpu_local | gpu_t4 |
| `tests/gpu/test_real_train_overfits.py` | `test_overfits_in_50_steps` | gpu_t4 | gpu_t4 |
| `tests/gpu/test_real_train_qlora.py` | `test_qlora_overfits_in_50_steps` | gpu_t4 | gpu_t4 |
| `tests/gpu/test_real_train_qlora_resume.py` | all | gpu_t4 | gpu_t4 |
| `tests/gpu/test_run_end_to_end_gpu.py` | all | gpu_t4 | gpu_t4 |

> Implementer note: this table accounts for the 27 known tests; if a live `grep -rn "gpu_local\|gpu_t4\|gpu_xl"` over `tests/` surfaces a marker not listed here (test counts drift across merges), apply the default rule (`gpu_local → gpu_t4`, `gpu_t4 → gpu_t4`) and add it to the mapping in the policy doc.

**R3 — #142's new 8 GB-ceiling QLoRA training smoke** is added to **`gpu_t4`** (full spec in §4).

**R4 — A minimal NEW bf16-faithful test for `gpu_bf16`.** Because **no current test is
bf16-faithful** (all run under cards that coerce, or assert nothing about the dtype
path), `gpu_bf16` is populated by a **new minimal test** (1–2 cases) that asserts a
**real, non-coerced bf16** forward/train numeric path executes on CC ≥ 8.0. Concretely:
build/load the model with `dtype=bfloat16` on the live card, assert
`coerce_dtype_for_capability(bfloat16, cap)` returns `bfloat16` (not coerced) on
CC ≥ 8.0, and that a single forward (or a 1–2 step train) produces finite outputs in
true bf16 (e.g. a representative parameter/activation tensor has
`.dtype == torch.bfloat16`). This is the test that the T4 cannot run and that
documents #139's finding.
**Acceptance:** `pytest --collect-only -m gpu_bf16` collects exactly the new test(s); on the 5070 Ti it runs and asserts a non-coerced bf16 tensor dtype; on a T4 it is skipped at the gate (CC 7.5 < 8.0).

**R-counts — collection contract.** `pytest --collect-only -m gpu_t4` collects the
27 reclassified tests + #142's new smoke (= 28). `-m gpu_bf16` collects only the new
bf16 test(s). `-m gpu_xl` collects 0. On a CPU box all are auto-skipped at collection.
**Acceptance:** the three collect-only counts match; CPU `pytest` stays green.

---

## §3 Capability auto-detection, gate move, Pascal removal (R5–R9)

**R5 — `_current_tier()` becomes a set-returning capability probe.** Replace the
single-tier `_current_tier()` (`tests/conftest.py:53-69`, today returns only
`"gpu_local"`) with a probe (rename to `_satisfied_tiers()` or keep the name but
change the return type) that reads **both**:

- compute capability via `torch.cuda.get_device_capability()`, and
- total VRAM via `torch.cuda.get_device_properties(0).total_memory`,

and returns the **SET** of satisfied tiers:

```text
cc = get_device_capability(); total = get_device_properties(0).total_memory
tiers = set()
if _has_compatible_gpu():                      # CC >= 7.5 + kernel-launch probe (R7)
    if cc >= (7, 5) and total <= 16 * _GB: tiers.add("gpu_t4")
    if cc >= (8, 0) and total <= 16 * _GB: tiers.add("gpu_bf16")
    if total > 16 * _GB:                   tiers.add("gpu_xl")
return tiers
```

Expected results: 5070 Ti → `{gpu_t4, gpu_bf16}`; T4 → `{gpu_t4}`; CC < 7.5 → `{}`
(everything skipped at the gate); a >16 GB card → `{gpu_xl}` (plus `gpu_bf16`/`gpu_t4`
only if it is also ≤16 GB, which by definition it is not). Preserve "function not
constant" so tests can monkeypatch the probe.

> Edge note (state explicitly in code comment): the 16 GB band is a **closed** upper
> bound (`<= 16 * _GB`); a card reporting *slightly less* than a marketing "16 GB"
> (driver-reserved memory) still satisfies `gpu_t4`/`gpu_bf16`. A card reporting
> > 16 GB satisfies only `gpu_xl` and is intentionally **not** auto-run for
> `gpu_t4`/`gpu_bf16` here (those VRAM-ceiling assertions are pinned to the ≤16 GB
> band; running them on a bigger card could mask a small-card OOM). The runner (§8)
> may still force a tier explicitly.

**R6 — Skip predicate via capability-subset, not integer order.** Rewrite
`pytest_collection_modifyitems` so a test marked tier `T` runs iff `T ∈ satisfied_tiers`
for the active selection (the runner's forced tier when set, else the live
`_satisfied_tiers()`). **Delete `_TIER_ORDER`** and any `>=`/index comparison over
tiers — the bands are not linearly ordered (§2). The skip reason names the unmet gate
(e.g. "requires gpu_bf16 (CC ≥ 8.0, ≤16 GB); have CC 7.5").
**Acceptance:** on a stubbed `{gpu_t4}` card, `gpu_bf16` tests skip and `gpu_t4` run; on a stubbed `{gpu_t4, gpu_bf16}` card both run; on `{}` all skip. Unit-test the predicate by monkeypatching the probe (mirror existing `tests/unit/test_presets.py` `_stub_gpu` style — capability + `total_memory` stubs).

**R7 — Move the compatibility gate CC 6.0 → CC 7.5.** In `_has_compatible_gpu()`
(`tests/conftest.py:45-50`) change `capability >= (6, 0)` to `capability >= (7, 5)`.
Update the `requires_compatible_gpu` marker docstring (CC ≥ 6.0 → CC ≥ 7.5) at its
registration site. **Preserve** `_torch_can_launch_kernel` (the real kernel probe)
and the autouse `_free_cuda_after_gpu_test` fixture unchanged. Update the
conftest module docstring (it currently names "the GTX 1080 dev box" and "Colab T4"
as the only smoke tier — reframe to T4-floor / 5070 Ti-primary).
**Acceptance:** `_has_compatible_gpu()` returns `False` on a stubbed CC 6.1 card and `True` on stubbed CC 7.5 / 12.0 cards (with the kernel probe stubbed to pass); the `requires_compatible_gpu` docstring says CC ≥ 7.5.

**R8 — Delete Pascal support artifacts.**

- Delete the `gpu-pascal` (cu118) **uv extra** from `pyproject.toml` and its
  index/source routing; the **default cu130 wheel covers both T4 and 5070 Ti**.
  `grep -rn "gpu-pascal\|cu118\|pascal"` over `pyproject.toml`, `uv.lock` references,
  CI workflows, and `docs/` to confirm no other extra/doc/source references it.
- Delete `docs/testing/local-pascal-gpu-testing.md`.
- **Keep** `docs/testing/manual-gpu-pass-2026-05-24-gtx1080.md` as a dated historical
  record; add a one-line banner at the top: "Superseded by the RTX 5070 Ti; Pascal is
  no longer supported as of this PR (min supported GPU: Tesla T4, CC 7.5)."
- **Acceptance:** `grep -rn "gpu-pascal\|cu118"` returns zero live references (config/docs/CI); `local-pascal-gpu-testing.md` is removed; the gtx1080 doc carries the banner; `uv sync` still resolves on the default extra.

**R9 — Runtime dtype coercion unchanged.** `coerce_dtype_for_capability`
(`src/custom_sam_peft/runtime/_runtime.py:61-83`) is **not** touched. The T4 still
legitimately coerces bf16 → fp16; only the test gate (R7) moves.
**Acceptance:** `grep` shows no diff in `_runtime.py`'s coercion branch; existing runtime-coercion unit tests pass unchanged.

---

## §4 #142: 8 GB CC 7.5 support = BOTH train and predict (R10–R12)

The minimum-supported card is a **CC 7.5 / 8 GB** class machine. This PR proves both
training and prediction of the small config fit that envelope, validated on the 16 GB
5070 Ti via an 8 GB-ceiling assertion.

**R10 — 8 GB-ceiling QLoRA training smoke (new `gpu_t4` test).** Add a training smoke
on `configs/examples/min_gpu_qlora.yaml` (the decoder-only, r=8, fp16, adamw8bit
config measured at **~5.0 GB peak** on the 1080 per
`docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`). The test runs the
2-image `tests/fixtures/tiny_coco/` overfit (epochs=25, batch=1, grad_accum=1 →
50 updates, matching the existing smokes), measures real peak VRAM via
`torch.cuda.max_memory_allocated()` around a `reset_peak_memory_stats()`, and asserts
`peak <= QLORA_8GB_CEIL_GB`.

- **Ceiling constant.** `QLORA_8GB_CEIL_GB = 8.0` (GB). **Provenance (cite-or-tbd):**
  the measured peak is ~5.0 GB (`docs/research/2026-05-24-issue-137-qlora-8gb-feasibility.md`,
  GTX 1080, fp16); 8.0 GB is the **target minimum-card envelope** with ~3 GB margin
  over the measured peak, chosen so the test proves the config fits the *8 GB
  minimum-supported card* (not just the 16 GB dev card). The constant carries a
  comment: GPU + config + measured-peak source + date + the margin rationale. After
  the §9 5070 Ti run measures the real peak on the new card, the comment is updated
  with that figure (keep the 8.0 envelope; record the measured peak alongside).
  Until the 5070 Ti figure is recorded, tag `# tbd: #142` on the measured-peak line
  of the comment.
- **min_gpu_qlora.yaml de-Pascal'ing.** Keep `dtype: float16` (it is required to
  model the **CC 7.5 / 8 GB minimum-supported card**, which coerces bf16 → fp16
  anyway), but **rewrite the rationale comments**: the current "Pascal-required /
  sm_61" wording is now wrong. Reframe as: "fp16 to model the CC 7.5 / 8 GB
  minimum-supported card (bf16 is coerced to fp16 below CC 8.0 — see
  `runtime/_runtime.py:61`); fp16 keeps the 8 GB-envelope assertion honest."
- **Acceptance:** `pytest --collect-only -m gpu_t4` includes the new test; on the 5070 Ti it runs, overfits (loss-drop assertion consistent with the existing smokes), and asserts `peak <= 8.0 GB`; `min_gpu_qlora.yaml` contains no "Pascal"/"sm_61" rationale; the new ceiling constant carries a provenance comment.

**R11 — predict-fits-8GB validation.** A test (in `tests/predict/`, `gpu_t4`) builds
the `min_gpu_qlora`-class adapter/model and runs a **batch=1, K=1 predict path**,
measures real peak VRAM, and asserts it is `<= PREDICT_8GB_BUDGET_GB` (the same budget
the §5 warning uses, R13). This is the predict half of "8 GB card supports BOTH train
and predict."
**Acceptance:** on the 5070 Ti the predict peak for the small config is `<= PREDICT_8GB_BUDGET_GB`; the test reuses `tests/predict/fixtures/qlora_adapter` where possible.

**R12 — provenance-table row.** Add a row to `docs/defaults-provenance.md` for any
trust-bearing new constant introduced here (`QLORA_8GB_CEIL_GB`, and the §5
`PREDICT_8GB_BUDGET_GB`) — constant name, value, source, date, GPU/config context.
**Acceptance:** `docs/defaults-provenance.md` has rows for both constants; no constant is shipped as a silent guess (each is cited or `# tbd:`-tagged).

---

## §5 Empirical "trained model may not fit a small card" warning (R13–R17)

The product path today has **only an upward** VRAM hint:
`src/custom_sam_peft/predict/runner.py:352-358` logs "free VRAM is >12 GB; consider
--batch-size 4 or 8" when free VRAM > 12 GB and `batch_size == 1`. There is **no
downward** "won't fit a small card" warning. This section adds one — **empirically**,
never analytically.

**R13 — The budget constant.** Define `PREDICT_8GB_BUDGET_GB` — the usable predict
budget on an 8 GB / CC 7.5 card. **Value + provenance (cite-or-tbd):** a CC 7.5 / 8 GB
card has ~8.0 GB total; subtract a driver/runtime reservation to get a *usable*
budget. Set `PREDICT_8GB_BUDGET_GB = 7.0` (GB) — 8.0 GB nominal minus ~1.0 GB
driver/CUDA-context reservation, consistent with the ~1.0 GiB headroom convention
used elsewhere in the repo's sizing (`presets.py` `_headroom_bytes`, line 340).
Carry a provenance comment (nominal capacity, reservation rationale, date). If the
1.0 GB reservation is not directly measured on a real 8 GB card here, tag
`# tbd: #142` on the reservation line. Add a `docs/defaults-provenance.md` row (R12).

**R14 — Empirical probe at model-ready, pre-loop.** As soon as the model+adapter are
**built/loaded and ready for training but BEFORE the training loop starts**, run a
**single empirical batch=1, K=1 predict-path probe**:

```text
torch.cuda.reset_peak_memory_stats()
<run the minimal forward / predict path once at batch=1, K=1>
measured_predict_peak = torch.cuda.max_memory_allocated()
```

Then compare `measured_predict_peak` to `PREDICT_8GB_BUDGET_GB`. **No analytic model
is consulted** — the empirical measurement is the single source of truth.

**R15 — Warn-not-block, after the probe.** If `measured_predict_peak >
PREDICT_8GB_BUDGET_GB`, emit a clear warning **before training continues** (e.g. via
the existing logger): "the trained model's predict footprint is ~X GB; it may not be
usable for prediction on 8 GB / CC 7.5 GPUs (budget ~7.0 GB)." The warning fires
**after** the probe completes; it **does not block** training. (Below budget → no
warning.)

**R16 — Hook location, CPU no-op, overhead.** The probe + decision hooks **early in
the training entrypoint, after model+adapter construction, before the training loop**.
Locate the train entrypoint (`src/custom_sam_peft/train/loop.py` / `train/trainer.py`
— the implementer confirms the exact post-construction / pre-loop seam). It MUST be a
clean **no-op on CPU** (skip when `not torch.cuda.is_available()`), and its overhead
is exactly **one forward at batch=1 / K=1** (cheap; no training step). The
**decision + message-formatting logic is factored into a pure function** taking a
measured byte value + budget and returning (warn?: bool, message: str), so it is
**unit-testable on CPU** by injecting a measured value; only the measurement itself
needs a GPU.
**Acceptance:** on CPU, the train entrypoint runs with the probe cleanly skipped (no CUDA call, no error); the pure decision function returns `warn=True` for an injected over-budget value and `warn=False` under budget, with the expected message text — covered by a CPU unit test.

**R17 — GPU test for the warning.** A `gpu_t4` test asserts (a) the probe fires the
warning when the measured peak exceeds the budget (drive a config whose predict peak
is over `PREDICT_8GB_BUDGET_GB`, or inject), and (b) a real predict peak on the 16 GB
card for the small config is **consistent with the 8 GB projection** (i.e. the small
config's measured predict peak ≤ budget → no warning, matching R11).
**Acceptance:** the GPU test demonstrates both the warn and no-warn branches against real measurements on the 5070 Ti.

**R18 — Upward hint untouched.** The existing upward hint at
`predict/runner.py:352-358` (and its test `tests/predict/test_gpu_predict.py:226+`)
is **not** changed by this spec.

---

## §6 Maximize CPU coverage; integration-test audit (R19–R22)

GPU testing was previously **optional**, so GPU bugs escaped — and several were
**contract bugs a CPU stub could have caught**. The escaped bugs documented in
`docs/testing/gpu-audit-2026-05-24.md`:

1. **channel_adapter Conv2d dtype mismatch** (`src/custom_sam_peft/models/sam3.py`) — a dtype-consistency contract on the adapter/forward path.
2. **`_row_outputs` KeyError on non-tensor `forward_grounding` entries** (`src/custom_sam_peft/eval/evaluator.py`) — non-tensor entries in forward outputs.
3. **`_BUILTIN_DEFAULT_IMAGE_SIZE` 1024 vs `load_sam31`'s 1008 RoPE assertion** (`src/custom_sam_peft/predict/runner.py`) — an image-size / default-resolution contract.

**R19 — Audit `tests/integration/` for comprehensiveness.** `tests/integration/` is
already mostly CPU/stub-based; the real-GPU files there are
`test_load_sam31_real.py`, `test_peft_lora_real.py`, `test_peft_qlora_real.py`. The
audit records, for each escaped-bug **class**, whether a CPU/stub test now guards it
(and where), using `tests/fixtures/tiny_sam3_stub.py::TinySam3Stub` where possible.
**Acceptance:** the policy doc (§8) gains an audit subsection listing the three bug classes and the CPU test that now guards each (or the follow-up issue if deferred, R21).

**R20 — Add bounded CPU/stub regression tests for the three bug classes** (this PR),
using `TinySam3Stub` where possible:

- **dtype-consistency** on the adapter/forward path (mirror the channel_adapter Conv2d mismatch class): a stub forward with a mismatched input dtype surfaces a clear error / coerces, asserting the contract the GPU bug violated.
- **non-tensor forward-output entries**: feed a stub `forward_grounding` output containing a non-tensor entry through the `_row_outputs` path and assert it no longer KeyErrors (handles or skips non-tensor entries).
- **image-size / default-resolution contract**: assert the default image size used by the predict path is consistent with `load_sam31`'s 1008 RoPE expectation (catch the 1024-vs-1008 class on CPU).

**Bounded:** these are targeted contract tests, not a full GPU-equivalent suite.
**Acceptance:** the three CPU tests exist, are NOT GPU-gated (run in CI on `ubuntu-latest`), pass, and each references the audit bug class it guards.

**R21 — Defer large net-new areas to a follow-up issue.** Any large net-new coverage
area surfaced by the audit (beyond the three bounded tests) is filed as a follow-up
`gh issue create --assignee @me --label testing` rather than bloating this PR.
**Acceptance:** if the audit surfaces an out-of-scope area, an issue number is recorded in the policy doc; otherwise the audit states "no further coverage gaps."

**R22 — Strengthen the CPU-first policy as a review gate.** Add to
`docs/testing/gpu-test-policy.md` the explicit principle: **"Test on CPU by default;
a test earns a GPU tier ONLY when it needs real weights / kernels / quant that a stub
cannot reproduce."** Frame it as a review gate (a reviewer rejects a new GPU-tiered
test that a `TinySam3Stub` could cover).
**Acceptance:** the policy doc states the principle verbatim and lists it in the "adding a new GPU test" checklist.

---

## §7 GPU evidence gate — non-blocking, permissioned (R23–R25, R33)

**R23 — Non-blocking GPU-evidence CI check.** Add a CI check that **warns, never
fails** the PR, reporting whether a committed GPU-run evidence artifact (a results/log
file, e.g. under `docs/testing/`) is present and current for branch HEAD. "Current"
means the artifact references the HEAD commit (or is newer than the last source
change to GPU paths — the implementer picks the simplest checkable signal and
documents it). The check's conclusion is **neutral/warning** (e.g. a non-failing job
or an annotation), never a required-status failure.
**Acceptance:** with no evidence artifact, CI shows a warning annotation but the check does not fail the PR; with an artifact present and current, the check is green.

**R24 — Track the light GPU subset as a PR-description checklist, not a hard gate.**
The PR description carries a checklist item "ran the light GPU subset on the 5070 Ti
(see evidence artifact)". Merge proceeds on the **user's explicit approval**,
regardless of the check's state.
**Acceptance:** the PR template / description includes the checklist item; close-out does not block on the evidence check.

**R25 — Define the "light" GPU subset** (run after the PR is opened, only after the
user grants permission). A small curated set, NOT the full suite:

1. one load + forward (`tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical`),
2. one short QLoRA training smoke (the #142 8 GB-ceiling smoke, R10),
3. the predict-fits-8GB probe test (R11) + the §5 warning GPU test (R17).

**Keep the existing `gpu-deselect-check` CI job** unchanged.
**Acceptance:** the policy doc and the runner (§8) name this exact subset; `gpu-deselect-check` still runs and still greps for leftover `--deselect`.

**R33 — Provable non-blocking guarantee + merge stays approval-gated.** The R23 check
is **additive** to `gpu-deselect-check` (both run; neither replaces the other) and its
non-blocking nature must be *mechanically provable*, not just asserted in prose:

- The CI job (or step script) **always exits 0** — including when the evidence
  artifact is **missing** OR **stale** (e.g. it references an old SHA / fails the
  freshness check). It surfaces the missing/stale state only as a warning annotation
  or a neutral conclusion, never as a failing/required status.
- The job is **not** added to any branch-protection required-status set, and is **not**
  marked `required` in the workflow; merge proceeds solely on the **user's explicit
  approval** ("ship anyway if I approve"), independent of this check's state.
- This guarantee is locked by a **test**: a unit/CI-config assertion that runs the
  evidence-check script against (a) no artifact, (b) a stale artifact, and (c) a
  present-and-current artifact, asserting exit code 0 in all three and a green/neutral
  report only in case (c). Where the check is a standalone script (preferred over an
  inline workflow step, so it is unit-testable off-CI), this test invokes the script
  directly; it also asserts the workflow YAML declares no `required`/blocking status
  for the job.

**Acceptance:** the evidence-check script exits 0 for missing, stale, AND
present-and-current artifacts (proving non-blocking); only the present-and-current case
reports green; a test asserts all three exit codes and that the workflow declares the
job non-required; close-out / merge is gated on the user's explicit approval, never on
this check.

---

## §8 Migration: runners, notebook, docs (R26–R29)

**R26 — Primary environment = 5070 Ti, local.** `scripts/run_gpu_tests.sh` default
runs the bulk of all tests locally on the 5070 Ti (automatable in-session). The
`local` one-pytest-process-per-file behavior (checkpoint memory release) and the
stateful `--deselect` convention are preserved on the default path.

**R27 — Rewrite runner tier args to the new taxonomy.** Replace the `local` selector;
keep/adjust `t4`, `xl`; add a **colab-min** selector (and a **light** selector for the
§7 R25 subset). Mapping:

| Selector | Marker filter | Cards / purpose |
|----------|---------------|-----------------|
| (default) | `gpu_t4 or gpu_bf16` | 5070 Ti local — bulk run, one process per file |
| `t4` | `gpu_t4` | real T4 / forcing the ≤16 GB fp16 band |
| `bf16` | `gpu_bf16` | 5070 Ti — bf16-faithful tests |
| `xl` | `gpu_xl` | >16 GB (empty here) |
| `colab-min` | the §8 R28 curated subset | minimal Colab T4 surface |
| `light` | the §7 R25 subset | post-PR permissioned evidence run |

`grep -rn "run_gpu_tests.sh"` over `docs/`, `notebooks/`, CI to update every caller
of the removed `local` selector.
**Acceptance:** `bash scripts/run_gpu_tests.sh <selector>` dispatches per the table; `run_gpu_tests.sh garbage` exits non-zero with a usage line; on a CPU box every selector runs to clean autoskip (exit 0); no caller still passes `local`.

**R28 — Minimal Colab T4 surface.** `notebooks/colab_gpu_tests.ipynb` runs a SMALL
curated subset proving "Colab works for this repo" — **NOT** the full suite:
**install + load real SAM 3.1 + one forward + one short training smoke**. The
`colab-min` selector (R27) maps to exactly: the load+forward test
(`test_load_sam31_real.py::test_load_sam31_forward_to_canonical`) + one short training
smoke (the existing `gpu_t4` QLoRA overfit smoke, e.g.
`test_real_train_qlora.py::test_qlora_overfits_in_50_steps`). The notebook's cell that
invokes the runner is updated to call `colab-min` (and to capture a cheap few-step T4
timing sample for #193, §9).
**Acceptance:** the notebook's runner cell calls `run_gpu_tests.sh colab-min`; the subset is exactly the load+forward + one short smoke; the notebook documents that bf16 is coerced on the T4 (#139 finding).

**R29 — Update the policy doc.** `docs/testing/gpu-test-policy.md` is rewritten to the
new taxonomy: the three capability-named tiers + gates, the auto-detection model, the
T4-floor / Pascal-dropped decision, the per-tier counts and runner selectors, the
CPU-first review gate (R22), the integration-audit subsection (R19), the light-subset
definition (R25), and the non-blocking evidence gate (R23). Update the per-tier
counts/runtimes table.
**Acceptance:** the policy doc names no `gpu_local`/Pascal as a live tier (only as superseded history), lists the three new tiers with gates, and passes the project markdown linter (the CI lint job's exact `markdownlint-cli2` invocation — discover from the workflow, do not assume).

---

## §9 Per-issue resolution (R30 — all five close at PR merge)

| Issue | Deliverable | Close criterion |
|-------|-------------|-----------------|
| **#142** | 8 GB-ceiling QLoRA training smoke added to `gpu_t4` (R10); tier reclassified (R2); `min_gpu_qlora.yaml` Pascal wording de-Pascal'd (R10); predict-fits-8GB validation (R11). | The smoke + predict validation pass on the 5070 Ti with peak ≤ ceiling/budget; YAML carries no Pascal rationale. |
| **#139** | Minimal Colab-T4 surface wired + verified (R28); documented finding that **bf16 IS coerced on a T4 (CC 7.5)** (in the policy doc + notebook); faithful bf16 lives in `gpu_bf16` on the 5070 Ti (R4). | The Colab surface runs install+load+forward+smoke; the policy doc records the coercion finding; `gpu_bf16` has the bf16-faithful test. **Manual dependency:** closes once the **user confirms the one Colab run** (§10). |
| **#193** | Per-step wall-clock measured on the 5070 Ti (primary) + a cheap few-step T4 sample from the minimal Colab surface; resolve the `# tbd: #193` per-step figure in `docs/defaults-provenance.md` "Reference Training Profile". (The "≤30 min on T4" budget claim is already dropped in that doc — only the unmeasured per-step `# tbd: #193` remains.) | The `# tbd: #193` is replaced with both figures (5070 Ti + T4 sample), each cited (GPU + date + command). **Manual dependency:** the T4 sample needs the user's Colab run (§10). |
| **#195** | 2-image `tiny_coco` overfit speed/convergence measured on the 5070 Ti; confirm or retune the 25/50-step budgets in `tests/gpu/test_real_train_qlora_resume.py` / `test_qlora_overfits_in_50_steps` and the docstring claims. | The step budgets and docstring claims are confirmed-or-retuned against the 5070 Ti measurement; any retune carries provenance. |
| **#83** | **Branch on a 5070 Ti probe.** During implementation, probe all-scope (regex `.*`) LoRA peak VRAM on the 5070 Ti. **(a)** If ≤ ~15 GB with margin → add a `gpu_t4` all-scope LoRA smoke and close #83 **DONE**. **(b)** If > 16 GB → move all-scope to the new `gpu_xl` issue (§10) with the measured number and close #83 as **superseded**. (CPU wiring already covered by `tests/unit/test_peft_scope_coverage.py`.) | Whichever branch the measured peak selects is executed and recorded in the evidence artifact; #83 closed accordingly. |

**Acceptance (R30):** all five issues are referenced in the PR body with their concrete deliverable; the #83 branch taken is recorded with the measured peak.

---

## §10 New issue, manual dependency, sequencing (R31–R32)

**R31 — File the gpu_xl issue.** Create
`gh issue create --assignee @me --label testing` titled **"gpu_xl tier: GPU tests
requiring > 16 GB VRAM"**, body cross-referencing **#125** (cloud auto-provision,
OPEN) and noting it holds any #83 all-scope overflow + future >16 GB tests. (Run
`gh label list`; if `testing` is absent, create it inline.)
**Acceptance:** the issue exists, is assigned `@me`, labeled, and xrefs #125; if #83 takes branch (b), its measured all-scope number is in this issue.

**R32 — Manual dependency + phase sequencing.** The 5070 Ti runs are automatable
in-session, but **#193's T4 sample and #139's surface verification need a manual
Colab run the USER triggers.** Sequence: the **5070 Ti-evidenced items land first**
(#142, #195, #83, plus the PR itself); **#139 and #193 close once the user confirms
the one Colab run.** The plan must phase this so that the Colab-dependent closures are
the last step, gated on user confirmation, and do not block the PR or the 5070 Ti
deliverables.
**Acceptance:** the PR description separates "5070 Ti-evidenced (landed)" from "Colab-confirmation-pending (#139, #193)"; close-out of #139/#193 waits on the user's Colab confirmation.

---

## §11 Cross-cutting requirements

**X1 — Cite-or-tbd for every new/changed numeric.** Every new or changed numeric
default/threshold introduced here — `QLORA_8GB_CEIL_GB` (R10), `PREDICT_8GB_BUDGET_GB`
(R13), the 16 GB band boundary (R5), any margin — MUST carry a rigorous citation OR an
explicit `# tbd: <issue>` tag. Trust-bearing defaults also get a
`docs/defaults-provenance.md` row (R12). Never ship a silent guess.

**X2 — Blast-radius discipline.** The marker-schema change (deleting `gpu_local`,
adding `gpu_bf16`, moving the CC gate) has blast radius beyond the named files.
Before "done":

- `grep -rn "gpu_local\|@pytest.mark.gpu_local"` — every usage retired/reclassified.
- `grep -rn "run_gpu_tests.sh\b"` — every caller (docs, notebook, CI) updated for the removed `local` selector.
- `grep -rn "gpu_t4\|gpu_bf16\|gpu_xl"` — every doc referencing tiers reflects the new taxonomy.
- `grep -rn "gpu-pascal\|cu118\|Pascal\|sm_61\|CC >= 6\|capability >= (6"` — Pascal references retired (except dated history).
- Run the **FULL** CPU test suite (`uv run pytest`) green — adding/removing markers changes collection across the whole suite.

**X3 — Lint gates.** Before any commit landing on a ready PR:
`ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft && uv run pytest`
green; `shellcheck scripts/run_gpu_tests.sh` clean; the CI markdown linter clean on
every touched `.md` (including this spec and the policy doc — discover the exact
`markdownlint-cli2` config from the workflow).

---

## §12 Phasing guidance (for the planner)

The requirement IDs map cleanly onto sequential, independently-reviewable phases.
Suggested grouping (the planner decides final phase boundaries):

1. **Phase A — Taxonomy + conftest + pyproject (R1–R9, R29 doc-stub).** Tier markers,
   set-returning probe, capability-subset skip predicate, CC-gate move, Pascal
   removal. *Interface exposed:* the three markers + `_satisfied_tiers()` contract +
   reclassified tests. *Consumes:* nothing prior.
2. **Phase B — Warning feature (R13–R18, X1).** Empirical model-ready probe + pure
   decision function + GPU test. *Exposes:* `PREDICT_8GB_BUDGET_GB`, the pure
   decision function, the train-entrypoint hook. *Consumes:* none of Phase A's code
   (parallelizable with A, but shares the cite-or-tbd table).
3. **Phase C — #142 train smoke + #83 probe (R10–R12, R30 for #142/#83/#195).** New
   8 GB-ceiling smoke, predict-fits-8GB test, min_gpu_qlora de-Pascal, #83 branch
   probe, #195 budget confirm. *Consumes:* Phase A markers (tags the new tests
   `gpu_t4`) and Phase B's `PREDICT_8GB_BUDGET_GB` (R11 reuses it).
4. **Phase D — CPU integration audit (R19–R22).** Three bounded CPU/stub regression
   tests + audit subsection + CPU-first review gate. *Consumes:* nothing GPU; pure
   CPU, parallelizable.
5. **Phase E — Notebook/runner/docs + evidence gate + issue closure (R23–R33, R29
   doc-final).** Runner rewrite, colab-min notebook, policy-doc rewrite, non-blocking
   evidence CI check + its provable-non-blocking test (R33), gpu_xl issue, per-issue
   closures, Colab manual-dependency sequencing. *Consumes:* Phases A–D (it documents
   and wires everything; the Colab closures are last, gated on user confirmation).

Each phase boundary exposes a clear contract (markers, the probe function, the budget
constant, the new tests) so a later phase's fresh session builds on it without
re-reading prior code.

---

## §13 Spec self-review

- **No untagged placeholders.** Every numeric (`QLORA_8GB_CEIL_GB = 8.0`,
  `PREDICT_8GB_BUDGET_GB = 7.0`, the 16 GB band, the ~1 GB reservation, the ~3 GB
  smoke margin) carries either a citation (the issue-137 feasibility doc, the
  `presets.py` headroom convention) or an explicit `# tbd: #142` tag where a real
  8 GB-card measurement is still pending. No silent guesses.
- **Internal consistency.** The T4-floor decision (CC ≥ 7.5) is applied uniformly:
  the gate (R7), the `gpu_t4` band (R5), the bf16-coercion finding (#139), and the
  Flash-CC-8.0 fact all agree that the T4 is fp16-only and the 5070 Ti is the first
  faithful-bf16 / Flash card. Tier bands are non-linear (R5/R6 use capability-subset
  checks, not `_TIER_ORDER`). The 27-test inventory and the 28-collection count
  (27 + #142's new smoke) are consistent across §2.
- **Single-PR scope.** All five issues close at this PR's merge (with #139/#193's
  final close gated on the user's one Colab run, §10). Deferred work (>16 GB tests,
  cloud auto-provision) is pushed to the new gpu_xl issue and #125 — not this PR.
  CPU coverage additions are bounded (three contract tests); larger areas become
  follow-up issues.
- **Runtime untouched.** `coerce_dtype_for_capability` runtime behavior is explicitly
  out of scope (R9); only the test gate moves. No analytic model enters the warning
  path (empirical-only, §5).
- **GPU testing is non-optional but never merge-blocking.** The §7 evidence gate makes
  the light GPU run a tracked, mechanically-provable expectation (R23/R33: the check
  always exits 0 for missing/stale/current artifacts and is never a required status)
  while keeping it permissioned and additive to `gpu-deselect-check`. The light subset
  runs only after the PR is opened and the user grants permission (R25), and **merge
  proceeds solely on the user's explicit approval** (R24/R33) — "ship anyway if I
  approve." This closes the §6 "GPU testing was previously optional, so GPU bugs
  escaped" problem without a hosted GPU runner and without blocking merges.
