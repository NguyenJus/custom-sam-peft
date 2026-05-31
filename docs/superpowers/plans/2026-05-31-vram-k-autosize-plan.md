# VRAM K-autosize: split activation model + calibrate-and-climb — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the VRAM auto-sizing over-count (#203) by splitting the activation
memory model into a K-invariant encoder term and a K-scaling decoder term, adding
K to the auto-sizing search, and reworking `calibrate` into a model-guided,
OOM-safe confirm-and-climb flow shared by the CLI and the wizard.

**Architecture:** The single lumped `BASE_ACTIVATION_AT_1024` constant is replaced
by two seeds `A_FIXED` (encoder, per image) + `A_PER_CLASS` (decoder, per
image×class), so `activation = (A_FIXED + A_PER_CLASS*K) * batch` — no image-size
scale term (SAM 3.1 is fixed at 1008px). `decide_preset` searches
`methods × rs × batches × Ks` and treats `k` as an upper bound; `calibrate` derives
the split from two cheap probes, aims analytically, then confirm-climbs to maximal
card usage without an unguarded OOM.

**Tech Stack:** Python 3.12, PyTorch (CUDA), Typer CLI, pydantic config schema,
pyyaml line-surgery rewrites, pytest with monkeypatched CUDA stubs.

---

## Amendment 1 (2026-05-31): overhead-model recalibration

**Why.** Phase 4's §9 real-GPU gate (RTX 5070 Ti, 16 GiB, sm_120, driver 610.47)
caught a genuine modeling defect at the first step: the original derivation
`A_FIXED = peak_K1 − fixed_overhead − A_PER_CLASS` trusted an analytic
`fixed_overhead` that included a **fictional SDPA attention term**
(`_attention_bytes_per_example` ≈ 1.6 GiB). Real SAM 3.1 uses flash / mem-efficient
SDPA (no materialized N×N matrix; the whole K=1 forward activation is ≈0.96 GiB), so
the analytic `fixed_overhead` (4.248 GiB) **exceeded** the measured `peak_K1`
(3.049 GiB) and drove `A_FIXED` to −2.36 GiB — physically meaningless, un-landable.

**What changed (spec §2/§2.1/§6/§9).** The decomposition is now **self-consistent**:
predictor and derive script share one overhead `STATIC = _model_bytes +
_adapter_bytes + _optimizer_bytes + WORKSPACE_BYTES` with **no attention term**.

- The `_attention_bytes_per_example(image) * batch` term is **removed from the
  `_predicted_bytes` train branch** (real attention is folded into the empirical
  split). The symbol is **retained** and still feeds the `decide_eval_batch_size`
  SDPA ceiling, which is an independent conservative eval cap (only lowers
  `best_bs`) — kept unchanged.
- `A_FIXED = clamp(peak_K1 − STATIC − A_PER_CLASS, min=0)`. On the dev GPU the
  residual is negative and clamps to **0**, landed as a **cited measured result**
  (not `# tbd:`), the encoder activation being below the model-weight conservatism
  margin in `STATIC`.
- `A_PER_CLASS` (two-point differential, ≈1.163 GiB) is **unchanged** — the
  validated core of #203.
- **Safety inequality** (must over-predict at probe points): `predicted_peak(K=1) =
  2.646 + 1.163 = 3.81 GiB ≥ 3.049` measured; `predicted_peak(K=4) = 2.646 + 4.652 =
  7.30 GiB ≥ 6.54` measured. Both ~0.76 GiB conservative; ~2.4 GiB less
  over-conservative than the broken status quo.

**Phase impact.** Phase 1 is already implemented; the amended Tasks **1.3** and
**1.6** below **MODIFY already-landed `presets.py` + tests** (drop the train-branch
attention term; keep the eval SDPA cap). Task **3.4** (derive script) and Phase 4
Task **4.1** are amended to the recalibrated `STATIC`/clamp formula so a negative no
longer blocks landing. Task 2.1's `_derive_split` and the `test_calibrate_cmd.py`
synthetic-peak helper are corrected to drop the attention term from their overhead.

---

## Phasing overview

Four sequential phases, each a coherent reviewable block. Phases 1–3 are CPU-only
(monkeypatched CUDA) and land in CI. Phase 4 fills the `# tbd:` seed constants from
a real GPU run and executes the mandatory real-silicon acceptance gate.

- **Phase 1 — Core memory model + search (`presets.py`).** Split activation model,
  K in the candidate grid, new `_sort_key`, `decide_preset` with `k` as upper bound,
  `PresetDecision.classes_per_forward`, v3 cache read, `decide_eval_batch_size`
  consuming the split. Seeds land as `# tbd:`-tagged values. CPU-mocked tests.
- **Phase 2 — Calibration core (`calibrate_cmd.py`).** The three-stage
  `run_calibration(...) -> PresetDecision` (derive → aim → confirm-and-climb), v3
  cache write, config rewrite threading `classes_per_forward`, preserved exit codes.
  CPU-mocked tests with a synthetic `_run_probe`.
- **Phase 3 — Wizard + caller integration.** `_ask_peft_sizing` consent →
  `run_calibration` with graceful fallback; `init_cmd`/`run_cmd` pass the K cap;
  `_rewrite_sizing_block` writes `classes_per_forward`; `_derive_preset_constants.py`
  emits the two-point split. Tests for wizard consent + fallback.
- **Phase 4 — Seed derivation + MANDATORY real-GPU acceptance gate (§9).**
  GPU-REQUIRED. Run the derive script on the RTX 5070 Ti, fill `A_FIXED`/`A_PER_CLASS`
  with a rigorous citation, run the §9 end-to-end gate. Not runnable in CPU CI.

---

## Conventions every phase follows

**Test subset run (bypassing the global `--cov-fail-under=80` gate):**

```bash
uv run pytest tests/unit/test_presets.py -o "addopts=" -q
```

The `-o "addopts="` clears the global pytest addopts (which include
`--cov-fail-under=80`); `--no-cov` does NOT work in this repo. Substitute the
phase's test path.

**Lint gate before every commit (CI runs both separately):**

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

If `ruff format --check` reports diffs, run `uv run ruff format src tests scripts`
and re-stage.

**Eager-import guard.** `src/custom_sam_peft/__init__.py` eagerly imports the train
chain, so removing a symbol (e.g. `BASE_ACTIVATION_AT_1024`) can un-import the whole
package mid-refactor. After any symbol-removal step, verify the package still
imports before claiming success:

```bash
uv run python -c "import custom_sam_peft"
uv run python -m py_compile src/custom_sam_peft/presets.py
```

**Cite-new-hyperparams.** Every new/changed default hyperparam carries a rigorous
citation OR an explicit `# tbd:` tag. `A_FIXED`/`A_PER_CLASS` ship as `# tbd:` in
Phase 1 and become cited constants in Phase 4. Never a silent guess.

---

## Phase 1 — Core memory model + search (`presets.py`)

**Goal:** Replace the lumped activation model with the split formula, add K to the
search space, change `_sort_key` and `decide_preset`'s `k` semantics, add
`PresetDecision.classes_per_forward`, read v3 caches, and thread K through
`decide_eval_batch_size`. Seeds land as conservative `# tbd:` values.

**Files:**

- Modify: `src/custom_sam_peft/presets.py`
- Test: `tests/unit/test_presets.py`

**Anchors (current line numbers, from spec §8):** `BASE_ACTIVATION_AT_1024` ~57;
`CACHE_SCHEMA_VERSION` 67; `PresetDecision` 73; `config_patch` 97; `label` 108;
`_activation_per_example`/`_activation_bytes` 188–201; `_predicted_bytes` 204;
`_load_cache` 248; `_candidates` 304; `_sort_key` 311; `decide_preset` 323;
`decide_eval_batch_size` 401.

---

### Task 1.1: Replace the lumped seed constant with the split seeds (`# tbd:`)

**Files:**

- Modify: `src/custom_sam_peft/presets.py:57`

- [ ] **Step 1: Replace `BASE_ACTIVATION_AT_1024` with `A_FIXED` + `A_PER_CLASS`**

Replace this line (~57):

```python
# cite: empirical (#148/#179 VRAM calibration)
BASE_ACTIVATION_AT_1024 = int(1.5 * _GB)  # seed; superseded by calibration cache.
```

with the split seeds, conservatively over-attributing to the encoder (a larger
`A_FIXED` can only *raise* the predicted floor, keeping sizing safe):

```python
# Split activation seeds, measured natively at SAM 3.1's fixed SAM3_IMAGE_SIZE=1008
# (no image-size scale term — image size is constant). Spec §2/§6.
#   activation(method, batch, K) = (A_FIXED + A_PER_CLASS * K) * batch
# A_FIXED   — K-invariant vision-encoder (hiera-large) activation, per image.
# A_PER_CLASS — decoder / mask-head activation, per (image × class).
# tbd: conservative split of the prior BASE_ACTIVATION_AT_1024 (~1.45 GiB at 1008px),
#   bulk attributed to the encoder. RE-DERIVE on the dev-env RTX 5070 Ti before merge:
#   uv run python scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1
#   (§9 real-GPU gate enforces this; replace this tag with a GPU+cc / commit SHA /
#    date / command citation when landing the measured values).
A_FIXED = int(1.30 * _GB)  # tbd: encoder-dominant share of ~1.45 GiB at 1008px
A_PER_CLASS = int(0.15 * _GB)  # tbd: decoder per-class share at 1008px
```

- [ ] **Step 2: Fix the two stale comment references to the removed constant**

The block comment above `_SAM3_PATCH` (~141–146) references
`BASE_ACTIVATION_AT_1024`. Replace its body:

```python
# SAM 3.1 vision backbone (hiera-large), from sam3/model_builder.py. Shared by
# the train-branch formula and decide_eval_batch_size's SDPA ceiling so both
# cite one definition (spec §3.2).
# _attention_bytes_per_example is the dominant activation term at SAM 3.1's
# 1008px image; k_eff scales BASE_ACTIVATION_AT_1024 in the train branch
# (see _activation_bytes).
```

with:

```python
# SAM 3.1 vision backbone (hiera-large), from sam3/model_builder.py. Shared by
# the train-branch formula and decide_eval_batch_size's SDPA ceiling so both
# cite one definition (spec §3.2).
# _attention_bytes_per_example is the dominant activation term at SAM 3.1's
# 1008px image; only the A_PER_CLASS term scales with k_eff in the train branch
# (see _activation_bytes). Spec §2.
```

- [ ] **Step 3: Verify the package still imports (eager-import guard)**

```bash
uv run python -c "import custom_sam_peft" && uv run python -m py_compile src/custom_sam_peft/presets.py
```

Expected: no output, exit 0. (Helpers in the next task still reference the old
names until 1.2 — but only `_activation_per_example`/`_activation_bytes`, which are
rewritten in 1.2; the module still imports because those are function bodies, not
module-level evaluation. If import fails here, complete 1.2 before re-checking.)

- [ ] **Step 4: Bump the cache schema version**

Change `presets.py:67`:

```python
CACHE_SCHEMA_VERSION = 2  # index-only (internal cache versioning; not trust-bearing)
```

to:

```python
CACHE_SCHEMA_VERSION = 3  # v3: split activation (A_fixed/A_per_class); drops activation_bytes_per_example
```

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/presets.py
git commit -m "refactor(presets): split activation seeds (A_FIXED/A_PER_CLASS), bump cache v3"
```

---

### Task 1.2: Rewrite the activation helpers to the split formula

**Files:**

- Modify: `src/custom_sam_peft/presets.py:188-201`
- Test: `tests/unit/test_presets.py`

- [ ] **Step 1: Write the failing test for the split formula**

Add to `tests/unit/test_presets.py` (imports: add `_activation_bytes` to the
`from custom_sam_peft.presets import ...` line, plus `A_FIXED, A_PER_CLASS`):

```python
from custom_sam_peft.presets import A_FIXED, A_PER_CLASS, _activation_bytes


def test_activation_bytes_split_is_linear_in_k_no_analytic_cache() -> None:
    # Encoder term (A_FIXED) does NOT scale with K; only A_PER_CLASS * K does.
    at_k1 = _activation_bytes(batch=1, cache=None, k_eff=1)
    at_k16 = _activation_bytes(batch=1, cache=None, k_eff=16)
    assert at_k1 == A_FIXED + A_PER_CLASS * 1
    assert at_k16 == A_FIXED + A_PER_CLASS * 16
    # The #203 regression guard: K=1 vs K=16 differ by exactly 15 * A_PER_CLASS.
    assert at_k16 - at_k1 == 15 * A_PER_CLASS


def test_activation_bytes_scales_with_batch() -> None:
    assert _activation_bytes(batch=4, cache=None, k_eff=2) == (
        (A_FIXED + A_PER_CLASS * 2) * 4
    )


def test_activation_bytes_reads_split_cache() -> None:
    cache = {"A_fixed": 1000, "A_per_class": 7}
    assert _activation_bytes(batch=2, cache=cache, k_eff=3) == (1000 + 7 * 3) * 2
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/unit/test_presets.py::test_activation_bytes_split_is_linear_in_k_no_analytic_cache -o "addopts=" -q
```

Expected: FAIL — `_activation_bytes` has the old `image_size` signature / lumped
formula, or `A_FIXED`/`A_PER_CLASS` import differs.

- [ ] **Step 3: Replace the two helpers with a single split helper**

Replace `presets.py:188-201` (the `_activation_per_example` + `_activation_bytes`
pair):

```python
def _activation_per_example(image_size: int, cache: dict[str, Any] | None) -> int:
    if cache is not None:
        return int(cache["activation_bytes_per_example"])
    return int(BASE_ACTIVATION_AT_1024 * (image_size / 1024) ** 2)


def _activation_bytes(
    image_size: int, batch: int, cache: dict[str, Any] | None, k_eff: int = 1
) -> int:
    # The SAM 3.1 multiplex forward materializes per-class mask/box decoder
    # activations within a group, so per-example activation scales with k_eff
    # (the per-group class count). Spec §3.1.
    per = _activation_per_example(image_size, cache)
    return int(per * batch * k_eff)
```

with the split formula (drop `image_size` — image size is fixed at 1008px, so the
activation helpers carry no scale term):

```python
def _activation_bytes(batch: int, cache: dict[str, Any] | None, k_eff: int = 1) -> int:
    """Split activation bytes: (A_FIXED + A_PER_CLASS * K) * batch.

    A_FIXED (K-invariant vision-encoder activation) does NOT scale with K; only the
    A_PER_CLASS decoder term does. Measured natively at SAM 3.1's fixed 1008px, so
    there is no image-size scale term. Reads the split from a v3 cache when present.
    Spec §2.
    """
    if cache is not None:
        a_fixed = int(cache["A_fixed"])
        a_per_class = int(cache["A_per_class"])
    else:
        a_fixed = A_FIXED
        a_per_class = A_PER_CLASS
    return int((a_fixed + a_per_class * k_eff) * batch)
```

Note: `_activation_per_example` is **removed** — its only callers are
`_activation_bytes` (rewritten here) and `decide_eval_batch_size` (rewritten in
Task 1.6). No dead helper remains.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_presets.py -k activation_bytes -o "addopts=" -q
```

Expected: 3 PASS. (Other tests in the file may now fail because callers still pass
`image_size` — those callers are fixed in 1.3 and 1.6. That is expected mid-phase.)

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): split activation helper (A_FIXED + A_PER_CLASS*K), drop image_size scale"
```

---

### Task 1.3: Thread the split through `_predicted_bytes` (train + eval)

> **AMENDED (Amendment 1, overhead-model recalibration).** This task **MODIFIES the
> already-landed `_predicted_bytes`** (Phase 1 is implemented). The change vs. the
> original task: the train branch **drops** the `_attention_bytes_per_example(image)
> * batch` term — real SDPA is folded into the empirical split, and keeping a
> separate analytic attention term double-counts (spec §2.1). The train branch
> becomes `STATIC + split` only. The eval branch is unchanged from the original task
> (it never had a separate attention term). A new test asserts the train branch
> carries **no** attention term.

**Files:**

- Modify: `src/custom_sam_peft/presets.py:218-244` (already-landed `_predicted_bytes`)
- Test: `tests/unit/test_presets.py`

- [ ] **Step 1: Write/refresh the failing tests for train/eval K-threading + no-attention**

```python
from custom_sam_peft.presets import (
    WORKSPACE_BYTES,
    _adapter_bytes,
    _model_bytes,
    _optimizer_bytes,
    _predicted_bytes,
)


def test_predicted_bytes_train_threads_k() -> None:
    # K=16 minus K=1 equals exactly 15 * A_PER_CLASS * batch (encoder unchanged).
    img = 1008
    pb_k1 = _predicted_bytes("qlora", 4, 1, img, None, mode="train", k_eff=1)
    pb_k16 = _predicted_bytes("qlora", 4, 1, img, None, mode="train", k_eff=16)
    assert pb_k16 - pb_k1 == 15 * A_PER_CLASS * 1


def test_predicted_bytes_train_has_no_attention_term() -> None:
    # Recalibration (spec §2.1): the train branch is STATIC + split, with NO
    # separate _attention_bytes_per_example term. At K=1 batch=1 it must equal
    # STATIC + (A_FIXED + A_PER_CLASS) exactly.
    img = 1008
    static = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4) + WORKSPACE_BYTES
    pb = _predicted_bytes("qlora", 4, 1, img, None, mode="train", k_eff=1)
    assert pb == static + (A_FIXED + A_PER_CLASS * 1)


def test_predicted_bytes_eval_threads_k() -> None:
    img = 1008
    pb_k1 = _predicted_bytes("lora", 4, 1, img, None, mode="eval", k_eff=1)
    pb_k4 = _predicted_bytes("lora", 4, 1, img, None, mode="eval", k_eff=4)
    assert pb_k4 > pb_k1  # eval activation now scales with K via the split
```

(`A_FIXED`/`A_PER_CLASS` are already imported in the file from Task 1.2.)

- [ ] **Step 2: Run to verify the new no-attention test fails**

```bash
uv run pytest tests/unit/test_presets.py -k predicted_bytes -o "addopts=" -q
```

Expected: `test_predicted_bytes_train_has_no_attention_term` FAILs — the
already-landed train branch still adds `_attention_bytes_per_example(image_size) *
batch`, so `pb` exceeds `static + (A_FIXED + A_PER_CLASS)` by that term. (The
K-delta and eval tests already pass against the landed code; the new assertion is
the recalibration guard.)

- [ ] **Step 3: Update both branches of `_predicted_bytes`**

Replace the already-landed `_predicted_bytes` body (`presets.py:218-244`). The
signature keeps `image_size` (only so the eval-cap caller's signature stays stable;
the activation calls do not use it). **The train branch drops the
`_attention_bytes_per_example(image_size) * batch` term** (spec §2.1 — folded into
the empirical split):

```python
def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    image_size: int,
    cache: dict[str, Any] | None,
    mode: Literal["train", "eval"] = "train",
    k_eff: int = 1,
) -> int:
    if mode == "train":
        # STATIC + split. NO separate _attention_bytes_per_example term — real
        # SDPA (flash/mem-efficient) is folded into the empirical split, and adding
        # an analytic attention figure double-counts (spec §2.1 recalibration). The
        # derive script subtracts the SAME STATIC, so this reproduces the measured
        # peak at the probe points.
        return (
            _model_bytes(method)
            + _adapter_bytes(r)
            + _optimizer_bytes(r)
            + _activation_bytes(batch, cache, k_eff=k_eff)
            + WORKSPACE_BYTES
        )
    # mode == "eval": no optimizer, no adapter bytes; activations x forward_only_factor.
    # K is threaded through the split; decide_eval_batch_size passes its
    # classes_per_forward as k_eff. The SDPA attention CAP stays in
    # decide_eval_batch_size (independent eval ceiling). Spec §2.1/§6.
    activations = int(
        _activation_bytes(batch, cache, k_eff=k_eff) * forward_only_factor
    )
    return _model_bytes(method) + activations + WORKSPACE_BYTES
```

`image_size` is now unused inside `_predicted_bytes`. Keep the parameter (callers
pass it positionally) and silence the linter by prefixing the param with `_` **or**
adding a `# noqa`-free no-op — simplest: rename the parameter to `image_size` but
mark it explicitly unused with a leading underscore is NOT possible without touching
callers, so instead keep the name and add `del image_size` as the first line of the
body, or leave it and confirm `ruff` does not flag an unused **parameter** (ruff's
default ruleset does not flag unused function parameters). Verify with
`ruff check`; if a future rule flags it, add `# noqa` on the signature line.

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_presets.py -k predicted_bytes -o "addopts=" -q
```

Expected: 3 PASS (K-delta, no-attention, eval-threads-K).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "fix(presets): drop fictional attention term from _predicted_bytes train branch (recalibration)"
```

---

### Task 1.4: Add `classes_per_forward` to `PresetDecision` (field, patch, label)

**Files:**

- Modify: `src/custom_sam_peft/presets.py:73-122`
- Test: `tests/unit/test_presets.py`

- [ ] **Step 1: Write the failing test for the new field + patch + label + round-trip**

```python
def _make_decision(**over) -> PresetDecision:
    base = dict(
        method="lora",
        r=16,
        batch_size=4,
        grad_accum_steps=4,
        classes_per_forward=8,
        dtype="bfloat16",
        headroom_bytes=0,
        predicted_bytes=0,
        budget_bytes=0,
        gpu_name="StubGPU",
        provenance="analytic",
        cache_path=None,
        calibrated_at=None,
    )
    base.update(over)
    return PresetDecision(**base)


def test_preset_decision_config_patch_carries_classes_per_forward() -> None:
    d = _make_decision(classes_per_forward=8)
    patch = d.config_patch
    assert patch["train"]["multiplex"]["classes_per_forward"] == 8
    assert patch["train"]["batch_size"] == 4
    assert patch["train"]["grad_accum_steps"] == 4


def test_preset_decision_label_surfaces_k() -> None:
    d = _make_decision(classes_per_forward=8)
    assert "K=8" in d.label()


def test_preset_decision_json_round_trip_carries_k() -> None:
    d = _make_decision(classes_per_forward=8)
    back = PresetDecision.from_json(d.to_json())
    assert back.classes_per_forward == 8
    assert back == d
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_presets.py -k preset_decision -o "addopts=" -q
```

Expected: FAIL — `PresetDecision.__init__` got an unexpected keyword
`classes_per_forward`.

- [ ] **Step 3: Add the field after the train-sizing fields**

In the `PresetDecision` dataclass (~83-94), add `classes_per_forward: int` directly
after `grad_accum_steps`:

```python
    method: Literal["lora", "qlora"]
    r: int
    batch_size: int
    grad_accum_steps: int
    classes_per_forward: int
    dtype: Literal["bfloat16", "float16"]
```

- [ ] **Step 4: Write `classes_per_forward` into `config_patch`**

In `config_patch` (~99-106), replace the `"train"` section:

```python
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
```

with:

```python
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "multiplex": {"classes_per_forward": self.classes_per_forward},
            },
```

- [ ] **Step 5: Surface `K=` in `label()`**

In `label()` (~118-122), update the f-string's first line:

```python
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"grad_accum={self.grad_accum_steps} {dtype_token} — "
            f"fits in {used_gib:.1f}/{total_gib:.1f} GiB on {self.gpu_name} {suffix}"
        )
```

to insert `K=`:

```python
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"K={self.classes_per_forward} grad_accum={self.grad_accum_steps} "
            f"{dtype_token} — "
            f"fits in {used_gib:.1f}/{total_gib:.1f} GiB on {self.gpu_name} {suffix}"
        )
```

`to_json`/`from_json` need no change — `asdict` and the `known`-fields filter carry
the new field automatically (test in Step 1 confirms the round-trip).

- [ ] **Step 6: Run to verify pass**

```bash
uv run pytest tests/unit/test_presets.py -k preset_decision -o "addopts=" -q
```

Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): PresetDecision.classes_per_forward + config_patch + label"
```

---

### Task 1.5: K in the candidate grid, new `_sort_key`, `decide_preset` upper-bound `k`

**Files:**

- Modify: `src/custom_sam_peft/presets.py:304-398`
- Test: `tests/unit/test_presets.py`

- [ ] **Step 1: Write the failing tests for grid, sort, and cap semantics**

```python
from custom_sam_peft.presets import _candidates, _sort_key


def test_candidates_are_4_tuples_with_ks() -> None:
    cands = _candidates()
    assert all(len(c) == 4 for c in cands)
    ks = {c[3] for c in cands}
    assert ks == {1, 2, 4, 8, 16}


def test_sort_key_protects_k_over_batch() -> None:
    # At fixed method/r, (K=8, batch=1) sorts ahead of (K=1, batch=8).
    assert _sort_key(("lora", 16, 1, 8)) < _sort_key(("lora", 16, 8, 1))


def test_sort_key_protects_r_over_k_and_batch() -> None:
    assert _sort_key(("lora", 32, 1, 1)) < _sort_key(("lora", 16, 16, 16))


def test_sort_key_prefers_lora_over_qlora() -> None:
    assert _sort_key(("lora", 16, 1, 1)) < _sort_key(("qlora", 16, 1, 1))


def test_decide_preset_k_is_upper_bound(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset(k=4)
    assert d.classes_per_forward <= 4


def test_decide_preset_k_zero_and_negative_raise(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    with pytest.raises(ValueError):
        decide_preset(k=0)
    with pytest.raises(ValueError):
        decide_preset(k=-1)


def test_decide_preset_24gib_sizes_successfully(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # #203 regression: a 24 GiB card must size successfully, not raise.
    _stub_gpu(monkeypatch, int(24 * _GB))
    d = decide_preset()
    assert isinstance(d, PresetDecision)
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_big_card_picks_high_k(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset()
    assert d.classes_per_forward >= 8
    assert d.batch_size >= 2
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_presets.py -k "candidates or sort_key or decide_preset_k or decide_preset_24 or decide_preset_big" -o "addopts=" -q
```

Expected: FAIL — `_candidates` returns 3-tuples, `_sort_key` takes a 3-tuple,
`decide_preset` doesn't set `classes_per_forward`.

- [ ] **Step 3: Extend `_candidates` to 4-tuples with Ks**

Replace `presets.py:304-308`:

```python
def _candidates() -> list[tuple[str, int, int]]:
    methods = ("lora", "qlora")
    rs = (8, 16, 24, 32, 48, 64)
    batches = tuple(range(1, 17))
    return [(m, r, b) for m in methods for r in rs for b in batches]
```

with:

```python
def _candidates() -> list[tuple[str, int, int, int]]:
    methods = ("lora", "qlora")
    rs = (8, 16, 24, 32, 48, 64)
    batches = tuple(range(1, 17))
    ks = (1, 2, 4, 8, 16)
    return [(m, r, b, k) for m in methods for r in rs for b in batches for k in ks]
```

- [ ] **Step 4: Replace `_sort_key` with the 4-tuple key**

Replace `presets.py:311-317`:

```python
def _sort_key(c: tuple[str, int, int]) -> tuple[int, int, int]:
    method, r, batch = c
    return (
        0 if method == "lora" else 1,
        -r,
        -batch,
    )
```

with:

```python
def _sort_key(c: tuple[str, int, int, int]) -> tuple[int, int, int, int]:
    # Priority highest-first: LoRA over QLoRA -> highest r -> highest K -> highest
    # batch. Tail-to-head = sacrifice order (give up batch, then K, then r, then
    # LoRA->QLoRA). Matches the runtime ladder and design priority (protect
    # accuracy levers method/r; spend throughput-only K and memory-only batch
    # first). Spec §3.
    method, r, batch, k = c
    return (0 if method == "lora" else 1, -r, -k, -batch)
```

- [ ] **Step 5: Rework `decide_preset` to search 4-tuples with `k` as an upper bound**

In `decide_preset`, change the docstring `k` description and the `k_eff` derivation
(~327-346). Replace:

```python
    image_size = SAM3_IMAGE_SIZE
    k_eff = MULTIPLEX_CAP if k is None else min(k, MULTIPLEX_CAP)
    if k_eff < 1:
        raise ValueError(f"k must be >= 1 when provided; got {k}")
```

with:

```python
    image_size = SAM3_IMAGE_SIZE
    # `k` is the UPPER BOUND on the K search (default MULTIPLEX_CAP). A user who
    # pins a lower classes_per_forward is respected as a cap. Spec §3.
    k_cap = MULTIPLEX_CAP if k is None else min(k, MULTIPLEX_CAP)
    if k_cap < 1:
        raise ValueError(f"k must be >= 1 when provided; got {k}")
```

Also update the docstring `k:` paragraph (~327-329) to read:

```python
      k: upper bound on the K (classes-per-forward) search. When None, uses
         MULTIPLEX_CAP. Callers with a config in scope pass
         cfg.train.multiplex.classes_per_forward as the cap. Spec §3.
```

Replace the feasible-search loop (~365-369):

```python
    feasible = []
    for method, r, batch in _candidates():
        pb = _predicted_bytes(method, r, batch, image_size, cache, k_eff=k_eff)
        if pb <= budget:
            feasible.append((method, r, batch, pb))
```

with:

```python
    feasible = []
    for method, r, batch, k_cand in _candidates():
        if k_cand > k_cap:
            continue
        pb = _predicted_bytes(method, r, batch, image_size, cache, k_eff=k_cand)
        if pb <= budget:
            feasible.append((method, r, batch, k_cand, pb))
```

Replace the no-feasible error block (~371-379):

```python
    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache, k_eff=k_eff)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1. Use a larger GPU."
        )
```

with (recompute the floor at K=1, the cheapest config):

```python
    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache, k_eff=1)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1 K=1. Use a larger GPU."
        )
```

Replace the selection + return (~381-398):

```python
    feasible.sort(key=lambda t: _sort_key(t[:3]))
    method, r, batch, predicted = feasible[0]
    grad_accum = max(1, 16 // batch)

    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        dtype=decided_dtype,
        headroom_bytes=headroom,
        predicted_bytes=predicted,
        budget_bytes=budget,
        gpu_name=gpu_name,
        provenance=provenance,
        cache_path=cache_path,
        calibrated_at=calibrated_at,
    )
```

with:

```python
    feasible.sort(key=lambda t: _sort_key(t[:4]))
    method, r, batch, k_chosen, predicted = feasible[0]
    grad_accum = max(1, 16 // batch)

    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        classes_per_forward=k_chosen,
        dtype=decided_dtype,
        headroom_bytes=headroom,
        predicted_bytes=predicted,
        budget_bytes=budget,
        gpu_name=gpu_name,
        provenance=provenance,
        cache_path=cache_path,
        calibrated_at=calibrated_at,
    )
```

- [ ] **Step 6: Run to verify pass**

```bash
uv run pytest tests/unit/test_presets.py -k "candidates or sort_key or decide_preset_k or decide_preset_24 or decide_preset_big" -o "addopts=" -q
```

Expected: all PASS.

- [ ] **Step 7: Update the pre-existing per-tier tests for the corrected model**

The old per-tier tests (`test_decide_preset_32gib_chooses_qlora`,
`..._40gib_chooses_lora_low_rank`, `..._65gib_chooses_lora_high_rank`) assert
behavior under the BUGGY K=16 worst-case lumped model — they will now break because
a 32 GiB card no longer needs QLoRA. Update their assertions to the corrected model:

- `test_decide_preset_32gib_chooses_qlora` → rename to
  `test_decide_preset_32gib_sizes_lora` and assert a `PresetDecision` is returned
  with `predicted_bytes <= budget_bytes` (the split model fits LoRA at 32 GiB; drop
  the `== "qlora"` assertion and the stale K=16/23-GiB comment).
- `test_decide_preset_40gib_chooses_lora_low_rank` and `..._65gib_..._high_rank`:
  keep `method == "lora"`; relax exact-rank comments to reflect the split model
  (they may now select higher r/K/batch). Assert `predicted_bytes <= budget_bytes`.

Run the whole file to confirm no stale assertion survives:

```bash
uv run pytest tests/unit/test_presets.py -o "addopts=" -q
```

Expected: all PASS. (`decide_eval_batch_size` tests are handled in Task 1.6.)

- [ ] **Step 8: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): K in candidate grid + (lora,-r,-K,-batch) sort; k = upper bound"
```

---

### Task 1.6: Thread K through `decide_eval_batch_size` + v3 cache-round-trip tests

> **AMENDED (Amendment 1).** This task **MODIFIES already-landed code**. Under the
> recalibration, `decide_eval_batch_size` **keeps its SDPA attention ceiling
> unchanged** — `_attention_bytes_per_example` is retained as the eval cap's
> `_attn_per_example` (spec §2.1 / §7). The recalibration only removed the attention
> term from the **train** branch (Task 1.3); the eval cap is an independent
> conservative ceiling that can only lower `best_bs`, so it stays as written below.
> No new test is needed beyond the existing K-threading / no-regression test; the
> code block is unchanged from the original task.

**Files:**

- Modify: `src/custom_sam_peft/presets.py:428-529` (already-landed `decide_eval_batch_size`)
- Test: `tests/unit/test_presets.py`

- [ ] **Step 1: Write the failing tests**

```python
from custom_sam_peft.presets import decide_eval_batch_size


def test_decide_eval_batch_size_threads_k_no_regression(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(24 * _GB))
    bs1, _, _ = decide_eval_batch_size(classes_per_forward=1)
    bs16, _, _ = decide_eval_batch_size(classes_per_forward=16)
    # Higher K can only LOWER (or hold) best_bs — never raise it (no regression).
    assert bs16 <= bs1
    assert bs16 >= 1


def test_decide_preset_consumes_v3_cache(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc"
    )
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({
        "schema_version": 3,
        "calibrated_at": "2026-05-31T00:00:00+00:00",
        "gpu_name": "StubGPU",
        "sam3_checkpoint_sha": "abc",
        "A_fixed": 1_000_000_000,
        "A_per_class": 50_000_000,
        "peak_memory_bytes_at_probe": 6_000_000_000,
    }))
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "calibrated"


def test_decide_preset_ignores_v2_cache(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc"
    )
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({
        "schema_version": 2,
        "gpu_name": "StubGPU",
        "sam3_checkpoint_sha": "abc",
        "activation_bytes_per_example": 1_000_000_000,
    }))
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "analytic"  # stale v2 dropped
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_presets.py -k "eval_batch_size_threads or consumes_v3 or ignores_v2" -o "addopts=" -q
```

Expected: FAIL — `decide_eval_batch_size` calls `_predicted_bytes(... mode="eval")`
without `k_eff` (K=1 hard-coded), and `_activation_per_example` (removed in 1.2) is
still referenced at ~468, raising `NameError`.

- [ ] **Step 3: Update `decide_eval_batch_size` to thread `classes_per_forward`**

In `decide_eval_batch_size`, the `_predicted_bytes(..., mode="eval")` calls
(~443-445, 447-449, 481-483) currently omit `k_eff`. Add
`k_eff=classes_per_forward` to all three. Replace the initial best block (~442-453):

```python
    best_bs = 1
    best_predicted = _predicted_bytes(
        "lora", r=4, batch=1, image_size=image_size, cache=cache, mode="eval"
    )
    for batch in range(1, 65):  # B in [1, 64]
        pb = _predicted_bytes(
            "lora", r=4, batch=batch, image_size=image_size, cache=cache, mode="eval"
        )
        if pb <= budget:
            best_bs = batch
            best_predicted = pb
```

with:

```python
    best_bs = 1
    best_predicted = _predicted_bytes(
        "lora", r=4, batch=1, image_size=image_size, cache=cache, mode="eval",
        k_eff=classes_per_forward,
    )
    for batch in range(1, 65):  # B in [1, 64]
        pb = _predicted_bytes(
            "lora", r=4, batch=batch, image_size=image_size, cache=cache,
            mode="eval", k_eff=classes_per_forward,
        )
        if pb <= budget:
            best_bs = batch
            best_predicted = pb
```

Replace the `_act_per_example` cap term (~468) — `_activation_per_example` is gone,
so compute the per-example forward activation from `_activation_bytes(batch=1)`
threaded with K:

```python
    _act_per_example = int(_activation_per_example(image_size, cache) * forward_only_factor)
```

with:

```python
    _act_per_example = int(
        _activation_bytes(batch=1, cache=cache, k_eff=classes_per_forward)
        * forward_only_factor
    )
```

Replace the final re-predict after the cap (~481-483):

```python
        best_predicted = _predicted_bytes(
            "lora", r=4, batch=best_bs, image_size=image_size, cache=cache, mode="eval"
        )
```

with:

```python
        best_predicted = _predicted_bytes(
            "lora", r=4, batch=best_bs, image_size=image_size, cache=cache,
            mode="eval", k_eff=classes_per_forward,
        )
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_presets.py -k "eval_batch_size_threads or consumes_v3 or ignores_v2" -o "addopts=" -q
```

Expected: all PASS.

- [ ] **Step 5: Run the full presets suite + import + lint**

```bash
uv run pytest tests/unit/test_presets.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Expected: all PASS / clean / no output. If any pre-existing
`decide_eval_batch_size` test asserts old K-insensitive numbers, update it to assert
"K can only lower best_bs" per spec §6.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): decide_eval_batch_size threads K through the split (no regression)"
```

---

### Phase 1 — definition of done

- `presets.py` imports cleanly; no reference to `BASE_ACTIVATION_AT_1024` or
  `_activation_per_example` remains anywhere in `src/` or `scripts/`
  (`uv run python -c "import custom_sam_peft"` passes; `grep -rn` finds nothing).
- `tests/unit/test_presets.py` passes in full under `-o "addopts="`.
- `ruff check` and `ruff format --check` clean.
- A 24 GiB stub card returns a `PresetDecision` (no `RuntimeError`).

### Phase 1 — outgoing interface contract

Downstream phases consume (do NOT re-read `presets.py` internals):

- `A_FIXED: int`, `A_PER_CLASS: int` — module constants in `presets.py`
  (`# tbd:`-tagged seeds until Phase 4).
- `_activation_bytes(batch: int, cache: dict | None, k_eff: int = 1) -> int` —
  split formula `(A_fixed + A_per_class*k_eff)*batch`; reads cache keys
  `"A_fixed"`/`"A_per_class"` when `cache` is not None. **No `image_size` param.**
- `_predicted_bytes(method, r, batch, image_size, cache, mode="train"|"eval",
  k_eff=1) -> int` — both branches consume the split; eval threads `k_eff`.
  **Train branch = `STATIC + (A_FIXED + A_PER_CLASS*k_eff)*batch` with NO separate
  `_attention_bytes_per_example` term** (Amendment 1, spec §2.1). `STATIC =
  _model_bytes + _adapter_bytes + _optimizer_bytes + WORKSPACE_BYTES`. `image_size`
  is accepted but unused inside the function.
- `_attention_bytes_per_example(image_size) -> int` — **retained**; consumed ONLY by
  the `decide_eval_batch_size` SDPA ceiling, never by `_predicted_bytes`.
- `decide_preset(k: int | None = None, cache_path: Path | None = None) ->
  PresetDecision` — `k` is the **upper bound** on the K search; `k < 1` raises
  `ValueError`; searches `methods × rs × batches × Ks(1,2,4,8,16)`; sort key
  `(lora?, -r, -K, -batch)`.
- `PresetDecision` gains `classes_per_forward: int` (after `grad_accum_steps`).
  `config_patch["train"]["multiplex"]["classes_per_forward"]` carries it;
  `label()` shows `K=<n>`; `to_json`/`from_json` round-trip it.
- `CACHE_SCHEMA_VERSION == 3`. v3 cache schema: keys `schema_version`,
  `calibrated_at`, `gpu_name`, `gpu_total_memory_bytes`, `sam3_checkpoint_sha`,
  `torch_version`, `custom_sam_peft_version`, **`A_fixed`**, **`A_per_class`**,
  `peak_memory_bytes_at_probe`, plus the OPTIONAL `chosen_method`/`chosen_r`/
  `chosen_batch`/`chosen_classes_per_forward` keys (written by Phase 2 only on the
  post-confirm cache; absent here). **No `activation_bytes_per_example`.** `_load_cache`
  drops non-v3 caches automatically.
- `decide_eval_batch_size(classes_per_forward: int = 16)` threads K through the
  split; the SDPA cap still only lowers `best_bs`.

---

## Phase 2 — Calibration core (`calibrate_cmd.py`)

**Goal:** Replace the single-probe `calibrate` with the three-stage
`run_calibration(...) -> PresetDecision` (derive split → analytic aim →
confirm-and-climb), write the v3 cache, thread `classes_per_forward` into the config
rewrite, and preserve exit codes. The Typer `calibrate` command becomes a thin
wrapper.

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py`
- Test: `tests/unit/test_calibrate_cmd.py`

**Consumes (Phase 1 contract):** `decide_preset`, `PresetDecision`,
`_activation_bytes`, `A_FIXED`/`A_PER_CLASS`, v3 cache schema, `_run_probe`
(unchanged), `_model_bytes`/`_adapter_bytes`/`_optimizer_bytes`/`WORKSPACE_BYTES`/
`_attention_bytes_per_example`.

**Note for the implementer — `_rewrite_sizing_block` signature.** As of Phase 2 it
does NOT yet accept `classes_per_forward` (Phase 3 Task 3.1 adds it). So in Phase 2,
`_apply_config_rewrite` (rewritten in Task 2.1 to take an explicit `PresetDecision`,
Correction B) calls `_rewrite_sizing_block` with the **current** 6-keyword signature
(no `classes_per_forward`). Phase 3 threads the K arg through everywhere in lockstep.
Do not pre-emptively pass `classes_per_forward` to `_rewrite_sizing_block` in
Phase 2 — it would `TypeError`.

---

### Task 2.1: Add `run_calibration` Stage 1 (derive the split) + helpers

> **AMENDED (Amendment 1).** Phase 2 is already implemented; this task's
> `_derive_split` is **modified** so its inverted overhead `STATIC` carries **no
> `_attention_bytes_per_example` term** (matching the predictor, spec §2.1), and a
> clamped-to-zero `A_fixed` is silent/expected (warn only on negative
> `A_per_class`). The Stage-1 test helper `_synthetic_peak` (Step 2) is likewise
> corrected to drop the attention term from its overhead, so the synthetic peak and
> the derive solve stay self-consistent.

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py` (already-landed `_derive_split`)
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Import the split seeds + grid constants into `calibrate_cmd.py`**

Extend the `from custom_sam_peft.presets import (...)` block (~24-33) to add:

```python
    A_FIXED,
    A_PER_CLASS,
    _headroom_bytes,
```

(`_headroom_bytes` is used by `run_calibration` to build the empirical
`PresetDecision`'s `headroom_bytes`/`budget_bytes` in Task 2.2.)

(keep the existing names). Add module constants for the grid the climb walks (cite
spec §3 / §4). These mirror `presets._candidates`' Ks/batches/rs (the rs ordering
matches `presets.py` `rs = (8, 16, 24, 32, 48, 64)` — confirm before landing):

```python
# Search grid mirrors presets._candidates Ks/batches/rs (spec §3/§4). The climb is
# bounded by these so a model error cannot loop (spec §4 "bounded probe count").
# The full sacrifice order on OOM is batch -> K -> r -> method (LoRA->QLoRA), so the
# climb walks _RS down (and flips method) to keep training fitting the GPU (spec §4).
_KS: tuple[int, ...] = (1, 2, 4, 8, 16)
_BATCHES: tuple[int, ...] = tuple(range(1, 17))
_RS: tuple[int, ...] = (8, 16, 24, 32, 48, 64)
```

- [ ] **Step 2: Write the failing test for Stage-1 derivation**

In `tests/unit/test_calibrate_cmd.py`, add a synthetic probe helper and a
derivation test. The synthetic probe encodes the split so derivation is exact:

```python
def _synthetic_peak(*, method: str, r: int, k_eff: int, batch: int) -> int:
    """Deterministic peak following the split model, for confirm-and-climb tests.

    Overhead is STATIC with NO attention term, matching the recalibrated predictor
    and _derive_split (Amendment 1 / spec §2.1), so the Stage-1 solve is exact.
    """
    from custom_sam_peft.presets import (
        WORKSPACE_BYTES,
        _adapter_bytes,
        _model_bytes,
        _optimizer_bytes,
    )

    a_fixed = 1_000_000_000
    a_per_class = 50_000_000
    overhead = (
        _model_bytes(method)
        + _adapter_bytes(r)
        + _optimizer_bytes(r)
        + WORKSPACE_BYTES
    )
    activation = (a_fixed + a_per_class * k_eff) * batch
    return int(overhead + activation)


def test_run_calibration_stage1_solves_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)  # sets cuda stubs + writes config
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    data = json.loads(out.read_text())
    assert data["schema_version"] == 3
    # A_per_class solved from the two synthetic K=1/K=4 peaks (closed form).
    assert abs(data["A_per_class"] - 50_000_000) < 1_000_000
    assert "activation_bytes_per_example" not in data
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest tests/unit/test_calibrate_cmd.py::test_run_calibration_stage1_solves_split -o "addopts=" -q
```

Expected: FAIL — `run_calibration` does not exist.

- [ ] **Step 4: Implement Stage 1 inside a new `run_calibration` skeleton**

Add `run_calibration` above the `calibrate` Typer command. This step implements
Stage 1 (derive) + a placeholder that writes the cache directly from the analytic
aim; Stages 2–3 are filled in Tasks 2.2–2.3. Define module-level exit-code-bearing
exceptions so the thin wrapper can map them:

```python
class _CalibrationError(Exception):
    """Base for calibration failures; carries the CLI exit code."""

    exit_code = 4  # default: probe failure


class _GpuTooSmall(_CalibrationError):
    exit_code = 5


class _CheckpointMissing(_CalibrationError):
    exit_code = 3


class _CacheWriteFailed(_CalibrationError):
    exit_code = 6


def _derive_split(method: str, r: int, batch: int) -> tuple[int, int]:
    """Stage 1: two cheap probes (K=1, K=4) -> (A_fixed, A_per_class).

    Raises _GpuTooSmall iff the K=1 probe OOMs. A K=4-probe OOM degrades to the
    analytic A_PER_CLASS seed (single-point A_fixed from peak_K1). Spec §4 Stage 1.

    STATIC (the inverted overhead) carries NO attention term — it must match the
    _predicted_bytes train branch exactly so the split reproduces the measured peak
    (Amendment 1, spec §2.1). A_fixed clamps to >=0; a clamped-to-zero A_fixed is the
    EXPECTED dev-GPU outcome (encoder activation < model-weight conservatism margin),
    not an error.
    """
    try:
        peak_k1 = _run_probe(method="qlora", r=4, k_eff=1, batch=1)
    except torch.cuda.OutOfMemoryError as exc:
        raise _GpuTooSmall("K=1 probe OOMed — GPU too small") from exc

    # STATIC: model + adapter + optimizer + workspace. NO _attention_bytes_per_example
    # term (Amendment 1 / spec §2.1) — same STATIC the predictor adds.
    static = (
        _model_bytes("qlora")
        + _adapter_bytes(4)
        + _optimizer_bytes(4)
        + WORKSPACE_BYTES
    )
    try:
        peak_k4 = _run_probe(method="qlora", r=4, k_eff=4, batch=1)
        a_per_class = int((peak_k4 - peak_k1) / (4 - 1))
    except torch.cuda.OutOfMemoryError:
        typer.echo(
            "WARNING: K=4 probe OOMed; falling back to analytic A_per_class seed",
            err=True,
        )
        a_per_class = A_PER_CLASS
    a_fixed = int(peak_k1 - static - a_per_class)

    # Warn ONLY on a negative A_per_class (a genuinely broken differential). A
    # negative A_fixed clamps to 0 silently — it is the expected recalibrated
    # outcome and must NOT block the cache write (Amendment 1 / spec §2.1).
    if a_per_class < 0:
        typer.echo(
            f"WARNING: clamped negative A_per_class={a_per_class}; "
            "two-point differential looks broken — re-derive on a real GPU",
            err=True,
        )
        a_per_class = max(0, a_per_class)
    a_fixed = max(0, a_fixed)
    return a_fixed, a_per_class
```

`_run_probe`, `_model_bytes`, `_adapter_bytes`, `_optimizer_bytes`,
`WORKSPACE_BYTES` are already imported. (`_attention_bytes_per_example` is **not**
used by `_derive_split` under Amendment 1 — drop it from this function's overhead;
it remains imported for any eval-cap use.) Add
`run_calibration` returning a `PresetDecision`; for now it derives the split, writes
a v3 cache with the Stage-1 split and `peak_memory_bytes_at_probe = peak_k1`, and
calls `decide_preset` for the decision (Stage 2/3 climb added next tasks):

```python
def run_calibration(*, config: Path, output: Path, force: bool) -> PresetDecision:
    """Three-stage model-guided calibration. Returns the chosen PresetDecision.

    Stage 1 derive -> Stage 2 analytic aim -> Stage 3 confirm-and-climb. Writes the
    v3 cache and rewrites the config sizing block. Spec §4.
    """
    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
    from custom_sam_peft.presets import PresetDecision, decide_preset

    if not config.exists():
        from custom_sam_peft.cli.init_cmd import run_init

        typer.echo(
            f"WARNING: {config} not initialized — auto-init (formula, no probe) then probe.",
            err=True,
        )
        run_init("coco-text-lora", config, force=False)

    cfg = load_config(config)
    method = cfg.peft.method
    r = cfg.peft.r
    k_cap = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    batch = cfg.train.batch_size

    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)

    if not force and _cache_is_fresh(output, gpu_name):
        # No probe runs this invocation. Prefer the EMPIRICAL record persisted by a
        # prior confirm-and-climb (the `chosen_*` cache keys) so a re-run never
        # reverts a probe-reduced config back to the OOM-prone analytic aim. Only when
        # the cache holds no empirical record (placeholder-only / legacy cache) does
        # analytic `decide_preset` become the correct/only source (Correction B path b).
        typer.echo("cache fresh — exiting")
        decision = _decision_from_cache(output, k_cap)
        if decision is None:
            decision = decide_preset(k=k_cap, cache_path=output)
        _apply_config_rewrite(config, decision=decision)
        return decision

    try:
        a_fixed, a_per_class = _derive_split(method, r, batch)
    except FileNotFoundError as exc:
        raise _CheckpointMissing(str(exc)) from exc

    # Stage 2 + Stage 3 (filled in Tasks 2.2-2.3). Placeholder: write the split
    # cache from Stage 1 and aim analytically. NOTE: this analytic return is replaced
    # in Task 2.2 by the EMPIRICAL PresetDecision built from the confirm-and-climb
    # result (Correction B) — do not keep the analytic decide_preset return here past
    # Task 2.2.
    peak = a_fixed + a_per_class  # placeholder; replaced by Stage-3 measured peak
    _write_cache_v3(output, gpu_name=gpu_name, total=total,
                    a_fixed=a_fixed, a_per_class=a_per_class, peak=peak)
    decision = decide_preset(k=k_cap, cache_path=output)  # placeholder; Task 2.2 makes empirical
    _apply_config_rewrite(config, decision=decision)
    return decision
```

Add the cache writer (replaces the inline payload at the old `calibrate:199`):

```python
def _write_cache_v3(
    output: Path,
    *,
    gpu_name: str,
    total: int,
    a_fixed: int,
    a_per_class: int,
    peak: int,
    method: str | None = None,
    r: int | None = None,
    batch: int | None = None,
    classes_per_forward: int | None = None,
) -> None:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "calibrated_at": datetime.now(UTC).isoformat(),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total,
        "sam3_checkpoint_sha": _sam3_checkpoint_sha(),
        "torch_version": torch.__version__,
        "custom_sam_peft_version": _PKG_VERSION,
        "A_fixed": int(a_fixed),
        "A_per_class": int(a_per_class),
        "peak_memory_bytes_at_probe": int(peak),
    }
    # The empirically-chosen sizing (Correction B). ADDITIVE optional v3 keys: ABSENT
    # on the Stage-2 pre-probe placeholder write (peak=0, no chosen_* args); PRESENT
    # on the FINAL post-confirm write (all four passed from the _confirm_and_climb
    # tuple). Persisting them lets the cache-fresh early-return reconstruct the
    # authoritative empirical decision instead of re-deriving the analytic aim.
    if method is not None:
        payload["chosen_method"] = method
    if r is not None:
        payload["chosen_r"] = int(r)
    if batch is not None:
        payload["chosen_batch"] = int(batch)
    if classes_per_forward is not None:
        payload["chosen_classes_per_forward"] = int(classes_per_forward)
    try:
        _atomic_write_json(output, payload)
    except OSError as exc:
        raise _CacheWriteFailed(str(exc)) from exc


def _decision_from_cache(output: Path, k_cap: int) -> PresetDecision | None:
    """Reconstruct the AUTHORITATIVE empirical decision from a confirmed v3 cache.

    Returns the PresetDecision recorded by the last confirm-and-climb (provenance
    "calibrated") when the cache carries the `chosen_*` keys. Returns None when they
    are absent — a placeholder-only (pre-probe) or legacy cache holds no empirical
    record, so the caller must fall back to the analytic `decide_preset`. This is the
    cache-fresh dual of Correction B: a probe's empirical result, once written, stays
    authoritative across re-runs and never reverts to the analytic aim.
    """
    try:
        data = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "chosen_method" not in data:
        return None
    method = data["chosen_method"]
    r = int(data["chosen_r"])
    batch = int(data["chosen_batch"])
    k = min(int(data["chosen_classes_per_forward"]), k_cap)
    peak = int(data["peak_memory_bytes_at_probe"])
    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)
    cc = torch.cuda.get_device_capability(0)
    dtype = "float16" if cc < (8, 0) else "bfloat16"
    headroom = _headroom_bytes()
    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=max(1, 16 // batch),
        classes_per_forward=k,
        dtype=dtype,  # type: ignore[arg-type]
        headroom_bytes=headroom,
        predicted_bytes=peak,
        budget_bytes=total - headroom,
        gpu_name=gpu_name,
        provenance="calibrated",
        cache_path=output,
        calibrated_at=_cache_calibrated_at(output),
    )
```

Add `PresetDecision`, `MULTIPLEX_CAP`, `decide_preset` to imports as needed (they're
imported lazily inside the functions above).

**Change `_apply_config_rewrite` to take an explicit `PresetDecision` (Correction B).**
The rewrite must persist the SAME sizing values the decision carries — never
re-derive them from the cache. Replace the current
`_apply_config_rewrite(config, *, k_eff, cache_path)` body (which internally calls
`decide_preset(k=k_eff, cache_path=cache_path)` and discards the caller's intent)
with a signature that accepts the already-chosen decision:

```python
def _apply_config_rewrite(config: Path, *, decision: PresetDecision) -> None:
    """Rewrite the config's sizing block from an already-chosen PresetDecision.

    The caller passes the authoritative decision: the EMPIRICAL confirm-and-climb
    result for `calibrate` (Correction B), or the analytic decide_preset result for
    the cache-fresh path / init_cmd. This helper no longer re-derives sizing from the
    cache — it persists exactly what `decision` carries. Emits a WARNING on failure
    (OSError/ValueError/RuntimeError) and silently returns — the cache stays the
    authoritative output.
    """
    try:
        from custom_sam_peft.cli._config_rewrite import _rewrite_sizing_block

        annotation = f"# calibrated {datetime.now(UTC).date().isoformat()}"
        _rewrite_sizing_block(
            config,
            method=decision.method,
            r=decision.r,
            batch_size=decision.batch_size,
            grad_accum_steps=decision.grad_accum_steps,
            dtype=decision.dtype,
            annotation=annotation,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(
            f"WARNING: config rewrite failed (cache intact, config unchanged): {exc}",
            err=True,
        )
```

Note: in Phase 2 the `_rewrite_sizing_block` call still uses the **6-keyword**
signature (no `classes_per_forward`) — Phase 3 Task 3.1 adds
`classes_per_forward=decision.classes_per_forward`. The `init_cmd` caller (Phase 3)
constructs the analytic `decide_preset(...)` result and passes it through this same
`decision=` signature; `calibrate` passes the empirical one. Every caller threads a
`PresetDecision`, never `(k_eff, cache_path)`.

- [ ] **Step 5: Run to verify pass**

```bash
uv run pytest tests/unit/test_calibrate_cmd.py::test_run_calibration_stage1_solves_split -o "addopts=" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(calibrate): run_calibration Stage 1 (derive split) + v3 cache writer"
```

---

### Task 2.2: Stage 2 analytic aim + Stage 3 confirm-and-climb/shrink

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py`
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Write the failing climb + shrink + r/method + empirical-authority +
  bounded-count tests**

The synthetic `_synthetic_peak` already varies peak with `method`/`r`/`k_eff`/`batch`
(Task 2.1), so probe tests can force a shrink down the full sacrifice order
`batch -> K -> r -> method`.

```python
def test_run_calibration_climbs_k_then_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    # Big card: synthetic peaks fit a wide grid -> climb should grow K then batch.
    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="BigGPU", total=int(80 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    assert decision.classes_per_forward >= 8
    data = json.loads(out.read_text())
    # Recorded peak is the final fitting probe's measured value, not the placeholder.
    assert data["peak_memory_bytes_at_probe"] > 10 * _GB


def test_run_calibration_shrinks_on_injected_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)
    calls: list[dict] = []

    def _probe(**kw):
        calls.append(kw)
        # OOM whenever batch>1 or K>2 (forces shrink batch-first then K).
        if kw["batch"] > 1 or kw["k_eff"] > 2:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    # Empirical (method, r, batch, k, peak) tuple drives the decision.
    assert decision.batch_size == 1
    assert decision.classes_per_forward <= 2
    assert decision.method == "lora"  # r/method not yet sacrificed here


def test_run_calibration_reduces_r_on_under_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    def _probe(**kw):
        # OOM for EVERY (batch, K) at the aimed r; fits only at a lower r.
        if kw["r"] > 16:
            raise torch.cuda.OutOfMemoryError("synthetic")
        if kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    assert decision.r <= 16  # r reduced to fit (full sacrifice order)
    import yaml

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["peft"]["r"] == decision.r  # written config matches, not the aimed r


def test_run_calibration_flips_to_qlora_when_lora_exhausts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)

    def _probe(**kw):
        # Every LoRA config OOMs (even r=_RS[0], batch=1, K=ks[0]); QLoRA fits.
        if kw["method"] == "lora":
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    assert decision.method == "qlora"


def test_run_calibration_decision_is_empirical_not_analytic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct regression guard for Correction B: when the real probe under-fits the
    analytic aim, the returned decision AND the written config equal the empirically
    fitting config, NOT the analytic aim."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    # The analytic aim (config A) over-predicts headroom and picks a high r/K/batch
    # that the real probe rejects; only a lower config (B) fits empirically.
    def _probe(**kw):
        if kw["r"] > 8 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    # Decision is the empirically-fitting config B, not the analytic aim A.
    assert decision.r == 8
    assert decision.batch_size == 1
    assert decision.classes_per_forward == 1
    import yaml

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["peft"]["r"] == 8
    assert cfg["train"]["batch_size"] == 1
    # Recorded peak is the real measured peak of config B.
    data = json.loads(out.read_text())
    assert data["peak_memory_bytes_at_probe"] == _synthetic_peak(
        method="lora", r=8, k_eff=1, batch=1
    )


def test_run_calibration_cache_fresh_returns_empirical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache-fresh re-run preserves the prior probe's empirical config (Correction B
    for the cached path): it must NOT revert to the OOM-prone analytic aim, and must
    NOT re-probe."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    # First call (force=True): the probe UNDER-fits the analytic aim, so the empirical
    # config is lower-r/batch/K than the aim would pick.
    def _probe(**kw):
        if kw["r"] > 8 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    first = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    data = json.loads(out.read_text())
    # The confirmed cache persists the empirically-chosen sizing.
    assert data["chosen_method"] == first.method == "lora"
    assert data["chosen_r"] == first.r == 8
    assert data["chosen_batch"] == first.batch_size == 1
    assert data["chosen_classes_per_forward"] == first.classes_per_forward == 1

    # Second call (force=False, cache fresh): must NOT probe and must return the SAME
    # empirical config — never the analytic aim (r=64/...).
    def _raise_if_probed(**kw):
        raise AssertionError("cache-fresh path must not re-probe")

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _raise_if_probed)
    monkeypatch.setattr(
        calibrate_cmd, "_derive_split",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cache-fresh path must not derive")
        ),
    )
    second = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=False
    )
    assert (second.method, second.r, second.batch_size, second.classes_per_forward) == (
        "lora", 8, 1, 1,
    )
    assert second.provenance == "calibrated"


def test_run_calibration_probe_count_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="BigGPU", total=int(80 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)
    count = {"n": 0}

    def _probe(**kw):
        count["n"] += 1
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=tmp_path / "c.json", force=True
    )
    # New bound covers the larger walk (batch + K + r + method flip + the 2 derive
    # probes); mirror the _confirm_and_climb max_probes formula.
    assert count["n"] <= (
        len(calibrate_cmd._BATCHES)
        + len(calibrate_cmd._KS)
        + len(calibrate_cmd._RS)
        + 2  # derive probes
        + 2  # method flip + slack
    )


def test_run_calibration_k1_oom_raises_gpu_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)

    def _probe(**kw):
        raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(calibrate_cmd._GpuTooSmall):
        calibrate_cmd.run_calibration(
            config=tmp_path / "config.yaml", output=tmp_path / "c.json", force=True
        )
```

Note on the bounded-count assertion: it must match whatever `max_probes` formula
Step 3 lands (`len(_BATCHES) + len(_KS) + len(_RS) + 2`). Keep the test's bound and
the helper's `max_probes` in sync; the `+ 2` derive probes are counted separately
from `_confirm_and_climb`'s internal cap.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_calibrate_cmd.py -k "climbs_k or shrinks_on or reduces_r or flips_to_qlora or decision_is_empirical or cache_fresh_returns_empirical or probe_count or k1_oom" -o "addopts=" -q
```

Expected: FAIL — climb/shrink/full-sacrifice-order not implemented; the placeholder
analytic return discards the empirical tuple (so `decision_is_empirical`,
`reduces_r`, `flips_to_qlora` fail); placeholder peak too small;
`cache_fresh_returns_empirical` fails because the cache lacks `chosen_*` keys and the
early-return still re-derives the analytic aim.

- [ ] **Step 3: Implement Stages 2–3 in `run_calibration`**

Replace the placeholder block in `run_calibration` (the
`# Stage 2 + Stage 3 (filled ...)` comment through the `return` line) with the
analytic aim + bounded confirm-and-climb. Add a `_confirm_and_climb` helper that
shrinks down the **full sacrifice order** `batch -> K -> r -> method` (Correction A)
and returns the full empirical tuple `(method, r, batch, k, measured_peak)`
(Correction B):

```python
def _confirm_and_climb(
    *, method: str, r: int, batch: int, k: int, budget: int, k_cap: int
) -> tuple[str, int, int, int, int]:
    """Stage 3: probe the aim, then climb (K then batch, at the fitting method/r) on
    headroom, or shrink down the FULL sacrifice order on OOM. Returns the empirical
    (method, r, batch, k, measured_peak). Bounded by the grid.

    Shrink order on OOM/over-budget, one probe per step (spec §4):
      1. batch > 1            -> batch -= 1
      2. else k > ks[0]       -> k = ks[ks.index(k) - 1]
      3. else r > _RS[0]      -> r = _RS[_RS.index(r) - 1]  (batch/K at their mins)
      4. else method == lora  -> method = "qlora"; r = _RS[-1]
         (QLoRA's NF4 base is far cheaper; re-try from the highest r and let the loop
          shrink r to the best-fitting QLoRA r — preserving accuracy where possible)
      5. else (qlora, r=_RS[0], batch=1, K=ks[0]) still OOM -> raise _GpuTooSmall.

    Climb-up NEVER raises r and NEVER flips method on a probe — accuracy levers are
    not raised on the strength of a probe (spec §4). It grows K to the next grid
    value first, then batch, at the fitting method/r only.
    """
    ks = [x for x in _KS if x <= k_cap]
    # Cover the full walk: batch down, K down, r down, plus the method flip + slack.
    max_probes = len(_BATCHES) + len(ks) + len(_RS) + 2
    probes = 0

    def _probe_fits(m: str, rr: int, b: int, kk: int) -> tuple[bool, int]:
        nonlocal probes
        probes += 1
        try:
            peak = _run_probe(method=m, r=rr, k_eff=kk, batch=b)
        except torch.cuda.OutOfMemoryError:
            return False, 0
        return peak <= budget, peak

    # Confirm the Stage-2 aim; shrink down the full sacrifice order until it fits.
    fits, peak = _probe_fits(method, r, batch, k)
    while not fits and probes < max_probes:
        if batch > 1:
            batch -= 1
        elif k > ks[0]:
            k = ks[ks.index(k) - 1]
        elif r > _RS[0]:
            r = _RS[_RS.index(r) - 1]  # batch and K already at their minimums
        elif method == "lora":
            method = "qlora"  # cheaper NF4 base; retry from the highest r
            r = _RS[-1]
        else:
            # qlora, r=_RS[0], batch=1, K=ks[0] and still OOM -> GPU too small.
            raise _GpuTooSmall(
                "no config fits down to (qlora, r="
                f"{_RS[0]}, batch=1, K={ks[0]}) — candidate space exhausted"
            )
        fits, peak = _probe_fits(method, r, batch, k)

    # Climb: grow K to the next grid value first, then batch, at the fitting
    # method/r only (never raise r or flip method on a probe).
    best = (method, r, batch, k, peak)
    while probes < max_probes:
        if k < ks[-1]:
            cand_b, cand_k = batch, ks[ks.index(k) + 1]
        elif batch < _BATCHES[-1]:
            cand_b, cand_k = batch + 1, k
        else:
            break  # grid max reached
        fits, peak = _probe_fits(method, r, cand_b, cand_k)
        if not fits:
            break
        batch, k, best = cand_b, cand_k, (method, r, cand_b, cand_k, peak)
    return best
```

Then in `run_calibration`, replace the placeholder with the analytic aim + the
empirical confirm-and-climb, and build the returned `PresetDecision` from the
EMPIRICAL `(method, r, batch, k, peak)` tuple (Correction B) — never re-derive it
analytically from the cache:

```python
    # Stage 2 — analytic aim over the full grid using the derived split. This is the
    # ONLY analytic use in the probe path (the aim that the probe then confirms).
    _write_cache_v3(output, gpu_name=gpu_name, total=total,
                    a_fixed=a_fixed, a_per_class=a_per_class, peak=0)
    aim = decide_preset(k=k_cap, cache_path=output)
    budget = aim.budget_bytes  # decide_preset already computed total - headroom

    # Stage 3 — confirm + climb/shrink down the full sacrifice order (bounded).
    # Returns the EMPIRICAL config; _GpuTooSmall is raised inside on full exhaustion.
    method, r, batch, k, peak = _confirm_and_climb(
        method=aim.method, r=aim.r, batch=aim.batch_size,
        k=aim.classes_per_forward, budget=budget, k_cap=k_cap,
    )

    # Persist the measured peak AND the empirically-chosen sizing (Correction B). The
    # chosen_* keys make this confirmed config authoritative on every later cache-fresh
    # read, so a re-run never reverts to the analytic aim.
    _write_cache_v3(output, gpu_name=gpu_name, total=total,
                    a_fixed=a_fixed, a_per_class=a_per_class, peak=peak,
                    method=method, r=r, batch=batch, classes_per_forward=k)

    # Build the AUTHORITATIVE decision from the empirical tuple (Correction B): the
    # config rewrite and the returned PresetDecision both use THESE values, not a
    # re-derived analytic decide_preset.
    cc = torch.cuda.get_device_capability(0)
    dtype = "float16" if cc < (8, 0) else "bfloat16"
    headroom = _headroom_bytes()
    decision = PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=max(1, 16 // batch),
        classes_per_forward=k,
        dtype=dtype,  # type: ignore[arg-type]
        headroom_bytes=headroom,
        predicted_bytes=peak,  # the real measured peak
        budget_bytes=total - headroom,
        gpu_name=gpu_name,
        provenance="calibrated",
        cache_path=output,
        calibrated_at=_cache_calibrated_at(output),
    )
    _apply_config_rewrite(config, decision=decision)
    return decision
```

Add `_headroom_bytes` to the `presets` import (it's a module-level helper at
`presets.py:286`). Add a tiny `_cache_calibrated_at(output)` reader that returns the
just-written cache's `calibrated_at` field (the v3 cache stores it), e.g.:

```python
def _cache_calibrated_at(output: Path) -> str | None:
    try:
        return json.loads(output.read_text()).get("calibrated_at")
    except (OSError, json.JSONDecodeError):
        return None
```

Pure analytic `decide_preset(cache)` now remains used ONLY for (a) the Stage-2 aim
above, and (b) the `cache_fresh` early-return path in Task 2.1 *as a fallback* — and
only when `_decision_from_cache` finds no `chosen_*` empirical record (placeholder-only
/ legacy cache). Everywhere a probe ran — this invocation or a prior cached one — the
empirical tuple is authoritative.

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_calibrate_cmd.py -k "climbs_k or shrinks_on or reduces_r or flips_to_qlora or decision_is_empirical or cache_fresh_returns_empirical or probe_count or k1_oom" -o "addopts=" -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(calibrate): Stage 3 full-sacrifice shrink (batch->K->r->method); empirical decision"
```

---

### Task 2.3: Make `calibrate` a thin wrapper preserving exit codes

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py:129-227`
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Write the failing exit-code + v3-cache CLI tests**

Update the existing `test_calibrate_writes_cache_with_schema_v2` to v3 (rename and
fix assertions), and add OOM-mapping tests:

```python
def test_calibrate_writes_cache_with_schema_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["schema_version"] == 3
    assert {"A_fixed", "A_per_class"}.issubset(data.keys())
    assert "activation_bytes_per_example" not in data


def test_calibrate_k1_oom_exits_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        calibrate_cmd, "_run_probe",
        lambda **kw: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")),
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 5
    assert "GPU too small" in result.output
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_calibrate_cmd.py -k "schema_v3 or k1_oom_exits_5" -o "addopts=" -q
```

Expected: FAIL — `calibrate` still uses the old single-probe body.

- [ ] **Step 3: Replace the `calibrate` command body with the thin wrapper**

Replace the entire body of `calibrate` (everything after the docstring,
~134-227) with:

```python
    if not torch.cuda.is_available():
        typer.echo(f"ERROR: {_CUDA_HINT}", err=True)
        raise typer.Exit(code=2)
    try:
        decision = run_calibration(config=config, output=output, force=force)
    except _GpuTooSmall as exc:
        typer.echo(
            f"ERROR: {exc} — calibration probe OOMed; GPU too small", err=True
        )
        raise typer.Exit(code=5) from exc
    except _CheckpointMissing as exc:
        typer.echo(f"ERROR: SAM 3.1 checkpoint not found: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except _CacheWriteFailed as exc:
        typer.echo(f"ERROR: cache write failed: {exc}", err=True)
        raise typer.Exit(code=6) from exc
    except _CalibrationError as exc:
        typer.echo(f"ERROR: probe failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"ERROR: probe failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    typer.echo(decision.label())
```

Add `_CUDA_HINT` to the `presets` import block if not already present (it is
imported at module top — confirm). Remove the now-dead helpers/imports from the old
body: `_run_probe` stays (used by Stage 1/3); the old inline `overhead`/`activation`/
`payload` computation is gone (moved into `_derive_split`/`_write_cache_v3`);
`SAM3_IMAGE_SIZE`/`MULTIPLEX_CAP` lazy imports in the old body are gone. Run ruff to
catch unused imports.

- [ ] **Step 4: Run to verify pass + full calibrate suite + import + lint**

```bash
uv run pytest tests/unit/test_calibrate_cmd.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: PASS / clean. Update any surviving v2-era test
(`test_calibrate_probes_at_config_r_and_k`, `..._negative_activation_warns`,
`..._rewrites_config_in_place_annotated`, `..._non_default_output_...`) to the new
flow: probes now run inside `run_calibration` (Stage 1 fixed at qlora/r4/K=1,4;
Stage 3 at the aimed config), the cache is v3, and the negative-activation warning
now fires from `_derive_split`'s clamp. Where a test asserted "probes at config's
r and K", re-target it to assert the Stage-3 confirm probe ran at the decision's
`(method, r, batch, K)`.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(calibrate): thin Typer wrapper over run_calibration; preserve exit codes"
```

---

### Phase 2 — definition of done

- `tests/unit/test_calibrate_cmd.py` passes in full under `-o "addopts="`.
- `calibrate` exit codes preserved: 2 no-CUDA, 3 checkpoint missing, 4 probe
  failure, 5 GPU-too-small (K=1 probe OOM, or full sacrifice-order exhaustion at
  qlora/r=_RS[0]/batch=1/K=ks[0]), 6 cache write failure.
- v3 cache written with `A_fixed`/`A_per_class`, no `activation_bytes_per_example`;
  `peak_memory_bytes_at_probe` is the Stage-3 measured peak.
- Stage 3 shrinks down the FULL sacrifice order `batch -> K -> r -> method`
  (LoRA→QLoRA), never just giving up at batch+K exhaustion (Correction A).
- The returned `PresetDecision` AND the rewritten config are built from the EMPIRICAL
  `(method, r, batch, k, peak)` tuple, not a re-derived analytic `decide_preset`
  (Correction B).
- The confirmed v3 cache persists the chosen sizing as `chosen_method`/`chosen_r`/
  `chosen_batch`/`chosen_classes_per_forward` (present only on the post-confirm write,
  absent on the Stage-2 placeholder). The `cache_fresh` early-return reconstructs the
  empirical decision via `_decision_from_cache` and only falls back to analytic
  `decide_preset` when those keys are absent — a re-run never reverts a probe-reduced
  config to the analytic aim.
- Probe count bounded by `len(_BATCHES) + len(_KS) + len(_RS) + 2`.
- `ruff check` + `ruff format --check` clean; package imports.

### Phase 2 — outgoing interface contract

- `run_calibration(*, config: Path, output: Path, force: bool) -> PresetDecision` —
  full three-stage flow; writes v3 cache + rewrites config sizing block; returns the
  EMPIRICAL `PresetDecision` (built from the confirm-and-climb result, not analytic
  `decide_preset`). Raises
  `_GpuTooSmall`/`_CheckpointMissing`/`_CacheWriteFailed`/`_CalibrationError`
  (each carries `.exit_code`). Wizard (Phase 3) calls this and catches
  `FileNotFoundError, torch.cuda.OutOfMemoryError, RuntimeError, ValueError,
  _CalibrationError`.
- `_confirm_and_climb(*, method, r, batch, k, budget, k_cap) ->
  (method, r, batch, k, measured_peak)` — shrinks down the full sacrifice order
  `batch -> K -> r -> method` on OOM (raising `_GpuTooSmall` only at full
  exhaustion); climbs K-then-batch on headroom, never raising r or flipping method on
  a probe. Returns the full empirical 5-tuple.
- `_apply_config_rewrite(config, *, decision: PresetDecision)` — takes an explicit
  already-chosen `PresetDecision` and persists exactly its sizing values; it no
  longer re-derives from the cache. `calibrate` passes the empirical decision; the
  cache-fresh path passes the reconstructed empirical one (or analytic fallback).
- `_write_cache_v3(..., method=None, r=None, batch=None, classes_per_forward=None)` —
  the four `chosen_*` params are ADDITIVE optional v3 keys written only on the
  post-confirm cache (absent on the Stage-2 placeholder, peak=0). Schema stays v3.
- `_decision_from_cache(output: Path, k_cap: int) -> PresetDecision | None` — rebuilds
  the authoritative `PresetDecision` (provenance "calibrated") from a confirmed cache's
  `chosen_*` keys; returns `None` for a placeholder-only / legacy cache so the caller
  falls back to analytic `decide_preset`.
- `calibrate` Typer command is a thin wrapper (no behavior beyond option parsing +
  exit-code mapping + printing `decision.label()`).
- `calibrate_cmd._KS`, `calibrate_cmd._BATCHES`, `calibrate_cmd._RS` are the bounded
  grid.
- The `_rewrite_sizing_block` call inside `_apply_config_rewrite` still uses the
  **6-keyword** signature (no `classes_per_forward` yet) — Phase 3 Task 3.1 adds the
  K arg.

---

## Phase 3 — Wizard + caller integration

**Goal:** Route wizard consent through `run_calibration` with graceful fallback;
make `init_cmd`/`run_cmd` pass the K cap; thread `classes_per_forward` through
`_rewrite_sizing_block` and all its callers; extend `_derive_preset_constants.py` to
emit the two-point split; confirm schema bounds.

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py:338-352`
- Modify: `src/custom_sam_peft/cli/init_cmd.py:175-187`
- Modify: `src/custom_sam_peft/cli/run_cmd.py:48-50`
- Modify: `src/custom_sam_peft/cli/_config_rewrite.py:23-134`
- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py` (rewrite call gains K)
- Modify: `scripts/_derive_preset_constants.py`
- Test: `tests/unit/cli/test_setup_wizard.py`, `tests/unit/test_calibrate_cmd.py`

**Consumes (Phases 1–2 contracts):** `decide_preset(k=...)`, `PresetDecision`
(`.classes_per_forward`, `.config_patch`, `.label()`), `run_calibration(...)`,
`A_FIXED`/`A_PER_CLASS`, split helpers.

---

### Task 3.1: Thread `classes_per_forward` through `_rewrite_sizing_block`

**Files:**

- Modify: `src/custom_sam_peft/cli/_config_rewrite.py`
- Test: add `tests/unit/cli/test_config_rewrite.py` (or extend existing rewrite test
  file if present — search `tests/ -name '*config_rewrite*'` first).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from custom_sam_peft.cli._config_rewrite import _rewrite_sizing_block

_CFG = """\
model:
  dtype: float16
peft:
  method: lora
  r: 8
train:
  batch_size: 1
  grad_accum_steps: 16
  multiplex:
    classes_per_forward: 16
"""


def test_rewrite_writes_classes_per_forward(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(_CFG)
    _rewrite_sizing_block(
        p, method="qlora", r=16, batch_size=4, grad_accum_steps=4,
        classes_per_forward=8, dtype="bfloat16", annotation="# calibrated 2026-05-31",
    )
    import yaml

    data = yaml.safe_load(p.read_text())
    assert data["train"]["multiplex"]["classes_per_forward"] == 8
    assert data["peft"]["r"] == 16
    assert data["train"]["batch_size"] == 4
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/cli/test_config_rewrite.py::test_rewrite_writes_classes_per_forward -o "addopts=" -q
```

Expected: FAIL — `_rewrite_sizing_block` got an unexpected keyword
`classes_per_forward`.

- [ ] **Step 3: Add the parameter and a nested-key replacement**

`classes_per_forward` lives at `train.multiplex.classes_per_forward` — a nested
child, not a direct child of `train`. The current helper only rewrites direct
children. Add the keyword and a targeted nested rewrite. In the signature (~23-32),
add `classes_per_forward: int` after `grad_accum_steps`:

```python
def _rewrite_sizing_block(
    config_path: Path,
    *,
    method: str,
    r: int,
    batch_size: int,
    grad_accum_steps: int,
    classes_per_forward: int,
    dtype: str,
    annotation: str,
) -> None:
```

The 5 direct-child targets stay in `replacements`. For the nested
`train.multiplex.classes_per_forward`, add a separate single-line surgery pass after
the existing loop (before the "Validate all 5 expected targets" block). Insert:

```python
    # Nested target: train.multiplex.classes_per_forward (one extra indent level
    # under train). Match the deepest-indented `classes_per_forward:` line and
    # rewrite its value; if absent, raise so the caller knows the config predates
    # the multiplex block. Spec §3.
    _cpf_pat = re.compile(r"^(\s+)classes_per_forward:\s+(\S+)(.*)$")
    cpf_done = False
    for i, line in enumerate(lines):
        m = _cpf_pat.match(line.rstrip("\n"))
        if m:
            indent, _old, tail = m.groups()
            staged.append((i, f"{indent}classes_per_forward: {classes_per_forward}{tail}\n"))
            touched_indices.append(i)
            cpf_done = True
            break
    if not cpf_done:
        raise ValueError(
            "_rewrite_sizing_block: config missing train.multiplex.classes_per_forward"
        )
```

This appends to `staged`/`touched_indices` before the annotation-insertion block, so
the nested rewrite participates in the same atomic write. Confirm `staged_map =
dict(staged)` (~139) still picks it up (it does — it's built after this loop).

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/cli/test_config_rewrite.py -o "addopts=" -q
```

Expected: PASS. Run any pre-existing rewrite tests to confirm none break (they will
now need the new keyword — fix their call sites in the same commit).

- [ ] **Step 5: Update `_apply_config_rewrite` and the wizard/init callers**

In `calibrate_cmd.py:_apply_config_rewrite` (rewritten in Task 2.1 to take an
explicit `decision: PresetDecision`, Correction B), add the decision's K to its inner
`_rewrite_sizing_block` call — its sizing values already come straight from
`decision`, so only the new keyword is added:

```python
        annotation = f"# calibrated {datetime.now(UTC).date().isoformat()}"
        _rewrite_sizing_block(
            config,
            method=decision.method,
            r=decision.r,
            batch_size=decision.batch_size,
            grad_accum_steps=decision.grad_accum_steps,
            classes_per_forward=decision.classes_per_forward,
            dtype=decision.dtype,
            annotation=annotation,
        )
```

(`calibrate` already passes the empirical decision and the cache-fresh path the
analytic one through this same `decision=` signature — no caller-shape change here,
just the extra `classes_per_forward` keyword threaded into `_rewrite_sizing_block`.)

In `init_cmd.py` (~179-187), `init_cmd` legitimately passes the analytic
`decide_preset(...)` result (no probe runs at init). Add
`classes_per_forward=decision.classes_per_forward`:

```python
            decision = decide_preset(k=k)
            _rewrite_sizing_block(
                output,
                method=decision.method,
                r=decision.r,
                batch_size=decision.batch_size,
                grad_accum_steps=decision.grad_accum_steps,
                classes_per_forward=decision.classes_per_forward,
                dtype=decision.dtype,
                annotation="# formula-derived",
            )
```

- [ ] **Step 6: Run all rewrite + init + calibrate tests + lint**

```bash
uv run pytest tests/unit/cli/test_config_rewrite.py tests/unit/test_calibrate_cmd.py -o "addopts=" -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: PASS / clean. (`init_cmd` tests that exercise the rewrite run under CUDA
guards; CPU-mocked ones should still pass.)

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/_config_rewrite.py src/custom_sam_peft/cli/calibrate_cmd.py src/custom_sam_peft/cli/init_cmd.py tests/unit/cli/test_config_rewrite.py
git commit -m "feat(rewrite): write train.multiplex.classes_per_forward; thread through callers"
```

---

### Task 3.2: `run_cmd._fallback_preset` passes the K cap

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py:48-50`
- Test: `tests/` — search for an existing `run_cmd`/`_fallback_preset` test; if none,
  add a focused unit test.

- [ ] **Step 1: Write the failing test**

```python
def test_fallback_preset_passes_k_cap(monkeypatch) -> None:
    import custom_sam_peft.cli.run_cmd as run_cmd
    from custom_sam_peft.config.loader import load_config_from_str  # or build a cfg

    captured = {}

    def _fake_decide_preset(k=None, cache_path=None):
        captured["k"] = k
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(run_cmd, "decide_preset", _fake_decide_preset)
    cfg = ...  # a TrainConfig with train.multiplex.classes_per_forward = 4
    try:
        run_cmd._fallback_preset(cfg)
    except RuntimeError:
        pass
    assert captured["k"] == 4
```

Implementer: construct `cfg` via the project's standard test config builder (mirror
how other `run_cmd`/`trainer` tests build a `TrainConfig` — search
`tests/ -name 'test_run_cmd*'` and `tests/helpers/`). If no ergonomic builder
exists, load a minimal YAML via `load_config` with `classes_per_forward: 4`.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/ -k fallback_preset_passes_k_cap -o "addopts=" -q
```

Expected: FAIL — `_fallback_preset` calls `decide_preset()` with no args, so
`captured["k"]` is `None`, not `4`.

- [ ] **Step 3: Pass the K cap**

Replace `run_cmd.py:48-50`:

```python
def _fallback_preset(cfg: TrainConfig) -> PresetDecision:
    """No sidecar — synthesize one from cfg + decide_preset(). Spec §11.4."""
    return decide_preset()
```

with:

```python
def _fallback_preset(cfg: TrainConfig) -> PresetDecision:
    """No sidecar — synthesize one from cfg + decide_preset(). Spec §11.4.

    Passes the config's classes_per_forward as the K upper bound (spec §3).
    """
    return decide_preset(k=cfg.train.multiplex.classes_per_forward)
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/ -k fallback_preset_passes_k_cap -o "addopts=" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/
git commit -m "fix(run): _fallback_preset passes cfg classes_per_forward as K cap"
```

---

### Task 3.3: Wizard `_ask_peft_sizing` routes consent through `run_calibration`

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py:338-352`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write the failing consent + fallback tests**

Update the two existing tests (`test_vram_autosize_*` at ~470-501) and add a
fallback-on-probe-failure test. The consent path now calls `run_calibration`:

```python
def test_vram_autosize_runs_calibration_on_consent(monkeypatch, tmp_path) -> None:
    import custom_sam_peft.cli.setup_wizard as sw
    from custom_sam_peft.presets import PresetDecision

    decision = PresetDecision(
        method="qlora", r=16, batch_size=2, grad_accum_steps=8,
        classes_per_forward=8, dtype="bfloat16", headroom_bytes=0,
        predicted_bytes=0, budget_bytes=0, gpu_name="StubGPU",
        provenance="calibrated", cache_path=None, calibrated_at="2026-05-31T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd.run_calibration",
        lambda **kw: decision,
    )
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: True)
    ctx = sw.Ctx(answers={}, cuda_available=True)
    frag = sw._ask_peft_sizing(ctx)
    assert frag == decision.config_patch
    assert frag["train"]["multiplex"]["classes_per_forward"] == 8


def test_vram_autosize_falls_back_to_analytic_then_manual(monkeypatch) -> None:
    import custom_sam_peft.cli.setup_wizard as sw

    def _boom(**kw):
        raise RuntimeError("probe failed")

    monkeypatch.setattr("custom_sam_peft.cli.calibrate_cmd.run_calibration", _boom)
    # analytic fallback also fails -> manual
    monkeypatch.setattr("custom_sam_peft.presets.decide_preset", lambda **kw: (_ for _ in ()).throw(RuntimeError("nothing fits")))
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: True)
    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: "qlora")
    ctx = sw.Ctx(answers={}, cuda_available=True)
    frag = sw._ask_peft_sizing(ctx)
    assert frag == {"peft": {"method": "qlora"}}
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/cli/test_setup_wizard.py -k "autosize_runs_calibration or falls_back_to_analytic" -o "addopts=" -q
```

Expected: FAIL — `_ask_peft_sizing` calls `decide_preset()` (analytic-only), never
`run_calibration`.

- [ ] **Step 3: Rewrite `_ask_peft_sizing` to run calibration on consent**

Replace `setup_wizard.py:338-352`:

```python
def _ask_peft_sizing(ctx: Ctx) -> dict[str, Any]:
    from custom_sam_peft.presets import decide_preset

    if ctx.cuda_available and ask_confirm(
        "Auto-size the PEFT config to your GPU's VRAM?", default=True
    ):
        try:
            decision = decide_preset()
        except RuntimeError as exc:
            typer.echo(f"could not auto-size: {exc}; falling back to manual")
        else:
            typer.echo(decision.label())
            return decision.config_patch
    method = ask_choice("PEFT method?", ["lora", "qlora"], default="lora")
    return {"peft": {"method": method}}
```

with the calibrate-flow version + graceful fallback. The wizard derives the K cap
from in-progress answers (default `MULTIPLEX_CAP` when unset), writes/uses a temp
config the calibrate flow needs, and degrades to analytic `decide_preset(k=...)`
then manual:

```python
def _ask_peft_sizing(ctx: Ctx) -> dict[str, Any]:
    if ctx.cuda_available and ask_confirm(
        "Auto-size the PEFT config to your GPU's VRAM?", default=True
    ):
        patch = _calibrate_or_analytic(ctx)
        if patch is not None:
            return patch
    method = ask_choice("PEFT method?", ["lora", "qlora"], default="lora")
    return {"peft": {"method": method}}


def _calibrate_or_analytic(ctx: Ctx) -> dict[str, Any] | None:
    """Consent path: run the calibrate confirm-and-climb flow; on any probe failure
    degrade to analytic decide_preset; return None to fall through to manual.
    Spec §5."""
    import torch

    from custom_sam_peft.cli.calibrate_cmd import run_calibration
    from custom_sam_peft.presets import CACHE_FILENAME, decide_preset

    answers = ctx.answers
    k_cap = answers.get("train", {}).get("multiplex", {}).get("classes_per_forward")

    # The calibrate flow needs a config on disk; the wizard renders the final
    # config later, so write a throwaway minimal config to a temp dir for the probe.
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text(_wizard_probe_config(answers))
            decision = run_calibration(
                config=cfg_path, output=Path(td) / CACHE_FILENAME, force=True
            )
        typer.echo(decision.label())
        return decision.config_patch
    except (
        FileNotFoundError,
        torch.cuda.OutOfMemoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        typer.echo(f"live GPU probe unavailable ({exc}); using analytic estimate")

    try:
        decision = decide_preset(k=k_cap)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"could not auto-size: {exc}; falling back to manual")
        return None
    typer.echo(decision.label())
    return decision.config_patch
```

Add a small `_wizard_probe_config(answers)` helper that renders a minimal valid
TrainConfig string (method/r/batch/K + the model block) for the probe. Reuse the
existing `render(...)` if it can run pre-completion; otherwise hand-write a minimal
YAML mirroring `tests/unit/test_calibrate_cmd.py::_write_config` (method default
"lora", r default the schema default, batch 1, K = `k_cap or 16`, model block from
`_model_block(answers)`). Keep it minimal — the probe only reads peft/train/model.

> Implementer note: the temp-config approach avoids needing the fully-rendered
> wizard config (which isn't assembled until after this step). If the existing
> `render()` requires answers this step doesn't have, the minimal hand-written YAML
> is the correct path. Mirror `_write_config`'s shape exactly so `load_config`
> accepts it.

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/cli/test_setup_wizard.py -k "autosize_runs_calibration or falls_back_to_analytic" -o "addopts=" -q
```

Expected: PASS.

- [ ] **Step 5: Run the full wizard suite + lint + import**

```bash
uv run pytest tests/unit/cli/test_setup_wizard.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: PASS / clean. Fix any pre-existing wizard test that monkeypatched the old
`decide_preset` consent path (re-point to `run_calibration` or assert the analytic
fallback).

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): consent runs run_calibration with analytic/manual fallback"
```

---

### Task 3.4: Extend `_derive_preset_constants.py` to emit the two-point split

**Files:**

- Modify: `scripts/_derive_preset_constants.py`

> **AMENDED (Amendment 1).** This script is already committed (commit `8fa12e0`)
> with the **broken** attention-term overhead — it is exactly what produced the
> −2.36 GiB `A_FIXED` on the 5070 Ti. This task now **corrects the landed script**:
> `fixed_overhead` becomes `STATIC` with **no `_attention_bytes_per_example` term**
> (matching the predictor and `_derive_split`, spec §2.1), and `A_FIXED` clamps to
> `>=0` so a negative residual prints `0` instead of a meaningless negative number.

This script is maintainer-only, not imported by the package or tests, so it has no
unit test. It is exercised by hand in Phase 4. The edits below must still pass
`ruff`.

- [ ] **Step 1: Update the module docstring**

Replace the docstring (~1-12) to drop the `BASE_ACTIVATION_AT_1024`/`design §3.3`
references and the attention term:

```python
"""Re-derive presets.py split activation seeds from probes on the local GPU.

Maintainer-only. Run on the 16 GB dev card:

    uv run python scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1

Runs two cheap probes (K=1, K=4) and prints the two-point split:
    A_per_class = (peak_K4 - peak_K1) / (4 - 1)
    A_fixed     = clamp(peak_K1 - STATIC - A_per_class, min=0)
where STATIC = model + adapter + optimizer + workspace (NO attention term — real
SDPA is folded into the empirical split; same STATIC the predictor adds, spec §2.1).
Measured natively at SAM 3.1's fixed 1008px (no image-size scale term). A clamped
A_FIXED=0 is the expected dev-GPU result. Prints copy-paste-ready
`A_FIXED = ...` / `A_PER_CLASS = ...` lines for presets.py.
Not imported by the package or the test suite. Spec §2.1/§6.
"""
```

- [ ] **Step 2: Replace `main()` body to run the two-point split**

Replace the probe + print section (~47-85) so it loops K over (1, 4), reuses the
single-probe body, computes the split, and prints the constants. Keep the argparse
(but `--k` becomes unused — drop it or leave it ignored with a note). Replace from
`image_size = SAM3_IMAGE_SIZE` through the final `print(...)`:

```python
    image_size = SAM3_IMAGE_SIZE

    def _probe_peak(k_eff: int) -> int:
        wrapper = load_sam31(ModelConfig(), channels=3, channel_semantics="rgb")
        apply_lora(wrapper, PEFTConfig(method=args.method, r=args.r))
        device = next(wrapper.parameters()).device
        images = torch.zeros(
            args.batch, 3, image_size, image_size, dtype=torch.bfloat16, device=device
        )
        prompts = [
            TextPrompts(classes=[f"class_{j}" for j in range(k_eff)])
            for _ in range(args.batch)
        ]
        torch.cuda.reset_peak_memory_stats()
        out = wrapper(images, prompts, support=None)
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for t in out.values():
            if isinstance(t, torch.Tensor):
                loss = loss + t.float().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return int(torch.cuda.max_memory_allocated())

    peak_k1 = _probe_peak(min(1, MULTIPLEX_CAP))
    peak_k4 = _probe_peak(min(4, MULTIPLEX_CAP))

    # STATIC: model + adapter + optimizer + workspace. NO attention term — real SDPA
    # is folded into the empirical split; this is the SAME STATIC the predictor adds
    # (spec §2.1). Inverting it makes the printed seeds reproduce the measured peak.
    static = (
        _model_bytes(args.method)
        + _adapter_bytes(args.r)
        + _optimizer_bytes(args.r)
        + WORKSPACE_BYTES
    )
    a_per_class = int((peak_k4 - peak_k1) / (4 - 1))
    a_fixed = int(peak_k1 - static - a_per_class)

    if args.batch != 1:
        # Split is per-image; normalize the activation by batch before clamping.
        a_per_class = int(a_per_class / args.batch)
        a_fixed = int((peak_k1 - static) / args.batch - a_per_class)

    # Clamp A_FIXED to >=0. A negative residual (encoder activation below the
    # model-weight conservatism margin in STATIC) clamps to 0 — the expected,
    # cited dev-GPU outcome (spec §2.1/§6), not an error.
    clamped = a_fixed < 0
    a_fixed = max(0, a_fixed)

    print(f"peak K=1:        {peak_k1 / _GB:.2f} GiB")  # noqa: T201
    print(f"peak K=4:        {peak_k4 / _GB:.2f} GiB")  # noqa: T201
    print(f"STATIC overhead: {static / _GB:.2f} GiB")  # noqa: T201
    if clamped:
        print(  # noqa: T201
            "note: A_FIXED residual was negative -> clamped to 0 (encoder activation "
            "below STATIC model-weight margin; expected, spec §2.1)"
        )
    print(  # noqa: T201
        f"A_FIXED = {a_fixed}  # {a_fixed / _GB:.3f} GiB (encoder, per image @1008px)"
    )
    print(  # noqa: T201
        f"A_PER_CLASS = {a_per_class}  # {a_per_class / _GB:.3f} GiB (decoder, per class @1008px)"
    )
```

The imports already include `_model_bytes`, `_adapter_bytes`, `_optimizer_bytes`,
`WORKSPACE_BYTES`, and `MULTIPLEX_CAP` is imported in the GPU branch (~44).
`_attention_bytes_per_example` is **no longer used** by this script under Amendment
1 — remove it from the script's import block (and any now-unused `image_size`
reference) so `ruff` stays clean. `image_size` is still needed by `_probe_peak` for
the input tensor shape, so keep `image_size = SAM3_IMAGE_SIZE`.

- [ ] **Step 3: Lint the script**

```bash
uv run ruff check scripts/_derive_preset_constants.py
uv run ruff format --check scripts/_derive_preset_constants.py
uv run python -m py_compile scripts/_derive_preset_constants.py
```

Expected: clean. (Cannot run end-to-end without a GPU; that happens in Phase 4.)

- [ ] **Step 4: Commit**

```bash
git add scripts/_derive_preset_constants.py
git commit -m "feat(scripts): derive two-point activation split (A_FIXED/A_PER_CLASS)"
```

---

### Task 3.5: Confirm schema bounds + full-suite green

**Files:**

- Read-only: `src/custom_sam_peft/config/schema.py:517` (no change expected)
- Test: full unit suite

- [ ] **Step 1: Confirm `MultiplexConfig.classes_per_forward` bounds**

`schema.py:517` is `classes_per_forward: int = Field(default=16, ge=1, le=16)`. The
rewrite writes K in `[1, 16]` (grid is `(1,2,4,8,16)`, capped at `MULTIPLEX_CAP=16`),
so every written value is within bounds. No schema change needed. Add an assertion
test confirming a rewritten config still validates:

```python
def test_rewritten_config_validates_within_bounds(tmp_path: Path) -> None:
    # A rewrite of K=16 (grid max) must still load_config cleanly (le=16).
    ...
```

Implementer: reuse the `_write_config` helper plus `_rewrite_sizing_block(...,
classes_per_forward=16, ...)`, then `load_config(path)` and assert it succeeds.

- [ ] **Step 2: Run the FULL unit suite under the coverage-bypass flag**

```bash
uv run pytest tests/unit -o "addopts=" -q
uv run python -c "import custom_sam_peft"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Expected: all PASS / clean.

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: confirm rewritten config validates within MultiplexConfig bounds"
```

---

### Phase 3 — definition of done

- Wizard consent runs `run_calibration`; CUDA-absent / probe-failure / checkpoint-
  missing degrade to analytic `decide_preset(k=...)` then manual, never crashing.
- `_rewrite_sizing_block` writes `train.multiplex.classes_per_forward`; all callers
  (`calibrate_cmd`, `init_cmd`) thread it in lockstep.
- `run_cmd._fallback_preset` passes `cfg.train.multiplex.classes_per_forward` as the
  K cap.
- `_derive_preset_constants.py` emits the two-point split; lints clean.
- Full `tests/unit` suite green; package imports; ruff check + format clean.

### Phase 3 — outgoing interface contract

- `_rewrite_sizing_block(..., classes_per_forward: int, ...)` — now a required
  keyword; writes the nested `train.multiplex.classes_per_forward`.
- `setup_wizard._ask_peft_sizing` consent → `run_calibration` →
  `decision.config_patch` (carrying `classes_per_forward`); graceful fallback chain.
- `scripts/_derive_preset_constants.py` prints `A_FIXED = <int>` / `A_PER_CLASS =
  <int>` lines (two-point split at 1008px) — consumed by hand in Phase 4.
- All CPU-mocked behavior is locked; Phase 4 only fills the seed values and runs the
  real-GPU gate.

---

## Phase 4 — Seed derivation + MANDATORY real-GPU acceptance gate (§9)

> **GPU-REQUIRED. NOT runnable in a CPU CI lane.** This phase is the human /
> orchestrator-run confirmation before merge. It runs on the dev-env **RTX 5070 Ti
> (16 GiB, sm_120, driver 610.47)**. The prior calibration regression shipped
> because it was never run on a real GPU — this gate is a hard merge blocker.

**Goal:** Replace the Phase-1 `# tbd:` seeds with measured, rigorously-cited
constants from the 5070 Ti, then execute the §9 acceptance gate end-to-end and
capture output.

**Files:**

- Modify: `src/custom_sam_peft/presets.py` (the `A_FIXED`/`A_PER_CLASS` block only)

**Checkpoint note.** The worktree has no `models/sam3.1/`. Before any probe, point
the checkpoint at the main checkout's copy (the 3.3 GB `sam3.1_multiplex.pt`). Either
set the `local_dir` env override `ModelConfig` reads, or symlink:

```bash
ln -s /home/justin/projects/custom-sam-peft/models/sam3.1 \
  /home/justin/projects/custom-sam-peft/.claude/worktrees/fix-vram-k-autosize/models/sam3.1
```

Confirm `models/sam3.1/sam3.1_multiplex.pt` resolves before probing.

---

### Task 4.1: Derive and land the measured seed constants

> **AMENDED (Amendment 1).** A first run of this task on the 5070 Ti caught the
> overhead-model defect (`A_FIXED = −2.36 GiB`). **Prerequisite:** Task 3.4's
> corrected derive script (STATIC, no attention term, clamp ≥0) must be landed
> first. With the correction the **measured** result is `A_FIXED = 0` (clamped
> residual) and `A_PER_CLASS = 1_248_840_021` (≈1.163 GiB). `A_FIXED = 0` lands as a
> **cited measured value** (not `# tbd:`), with a comment noting the clamp rationale
> (spec §2.1). The safety inequality holds: predicted K=1 = 3.81 GiB ≥ 3.05 measured;
> K=4 = 7.30 GiB ≥ 6.54 measured.

- [ ] **Step 1: Run the CORRECTED derive script on the 5070 Ti, capture output**

```bash
uv run python scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1
```

Confirm both cheap probes (K=1, K=4) complete without OOM and record the printed
`A_FIXED = ...` / `A_PER_CLASS = ...` lines plus the peak / STATIC figures. Expect
`A_FIXED = 0` (with the "clamped to 0" note) and `A_PER_CLASS ≈ 1_248_840_021`.
Capture the full stdout for the citation. If `A_FIXED` prints **negative**, the
script still has the old attention-term overhead — go back and finish Task 3.4
before landing.

- [ ] **Step 2: Replace the `# tbd:` seeds with cited measured constants**

In `presets.py`, replace the `# tbd:`-tagged block (from Phase 1 Task 1.1) with the
measured values and a rigorous citation. Use the exact format below, substituting
the live commit SHA and the actual integers from Step 1 (shown here with the
measured dev-GPU values):

```python
# Split activation seeds, measured natively at SAM 3.1's fixed SAM3_IMAGE_SIZE=1008
# (no image-size scale term). Spec §2/§2.1/§6.
#   predicted_peak = STATIC + (A_FIXED + A_PER_CLASS * K) * batch   (NO attention term)
# A_FIXED   — K-invariant vision-encoder (hiera-large) activation, per image.
# A_PER_CLASS — decoder / mask-head activation, per (image × class), two-point split.
# cite: measured on NVIDIA GeForce RTX 5070 Ti (16 GiB, sm_120, cc=12.0, driver
#   610.47), commit <SHA>, 2026-05-31, via:
#   uv run python scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1
#   (split is model+dtype driven, not card-driven → applies to all GPUs).
# A_FIXED clamps to 0: the K-invariant encoder activation sits below the analytic
#   model-weight conservatism margin in STATIC (residual peak_K1 - STATIC -
#   A_PER_CLASS ≈ -0.76 GiB), so STATIC already absorbs it. Predicted peak stays
#   conservative (K=1: 3.81 ≥ 3.05 measured; K=4: 7.30 ≥ 6.54 measured GiB). Spec §2.1.
A_FIXED = 0  # 0.00 GiB encoder activation per image @1008px (clamped residual)
A_PER_CLASS = 1_248_840_021  # 1.163 GiB decoder activation per class @1008px
```

Replace `<SHA>` with `git rev-parse --short HEAD`. If Step 1's run yields different
integers (probe noise / driver update), use those instead — but `A_FIXED` must be
the clamped (`max(0, …)`) value and `A_PER_CLASS` the two-point differential. **No
`# tbd:` tag may remain** anywhere in `presets.py` after this step.

- [ ] **Step 3: Confirm no `# tbd:` remains and the suite still passes**

```bash
grep -rn "tbd:" src/custom_sam_peft/presets.py && echo "TBD STILL PRESENT — FAIL" || echo "clean"
uv run pytest tests/unit/test_presets.py -o "addopts=" -q
uv run ruff check src/custom_sam_peft/presets.py
uv run ruff format --check src/custom_sam_peft/presets.py
```

Expected: "clean", tests PASS, ruff clean. The Phase-1 tests assert the split's
*structure* (linearity in K), not the seed magnitudes, so they remain green with the
new values.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/presets.py
git commit -m "feat(presets): land measured A_FIXED/A_PER_CLASS from RTX 5070 Ti (cite)"
```

---

### Task 4.2: Execute the §9 real-GPU acceptance gate (capture all output)

No code changes — this is the confirmation gate. Run each step on the 5070 Ti and
capture stdout/stderr into the PR description. A *caught* OOM during the climb is
expected and fine; an *unguarded* OOM crash is a gate failure.

- [ ] **Step 1: Seed-derivation re-confirm**

```bash
uv run python scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1
```

Gate: both K=1 and K=4 probes complete (no OOM); printed split matches the landed
constants (within probe noise).

- [ ] **Step 2: `calibrate` end-to-end**

```bash
uv run custom-sam-peft calibrate --config config.yaml --force
cat .custom_sam_peft_calibration.json
```

Gate: exit 0; cache `schema_version == 3` with `A_fixed`/`A_per_class` and **no**
`activation_bytes_per_example`; the confirm-and-climb walk terminated; the chosen
config's `peak_memory_bytes_at_probe` ≤ the 16 GiB budget; no unguarded OOM crash in
the output.

- [ ] **Step 3: Wizard auto-calibrate path on GPU**

Drive `_ask_peft_sizing` consent non-interactively (scripted) and confirm it returns
a fitting `PresetDecision` via `run_calibration`:

```bash
uv run python -c "
from custom_sam_peft.cli import setup_wizard as sw
ctx = sw.Ctx(answers={}, cuda_available=True)
import custom_sam_peft.cli._interactive as it
it.ask_confirm = lambda *a, **k: True  # auto-consent
print(sw._ask_peft_sizing(ctx))
"
```

Gate: prints a `config_patch` carrying `train.multiplex.classes_per_forward`; with
the checkpoint absent it degrades to analytic without crashing.

- [ ] **Step 4: Sanity vs. reality — a few real training steps**

```bash
uv run custom-sam-peft train --config config.yaml --time-limit 90s
```

(or the project's smallest real-train invocation). Gate: training runs a few steps
at the calibrated config without an OOM crash, closing the loop the analytic-only
ship failed to.

- [ ] **Step 5: Record the gate results in the PR**

Paste the captured output from Steps 1–4 into the PR description under a "Real-GPU
acceptance gate (§9)" heading. The gate is satisfied only when all four steps pass.

---

### Phase 4 — definition of done

- `A_FIXED`/`A_PER_CLASS` are measured, cited constants (GPU+cc, commit SHA, date,
  command); no `# tbd:` tag remains in `presets.py`. Recalibrated outcome:
  `A_FIXED = 0` (clamped, cited rationale), `A_PER_CLASS ≈ 1_248_840_021`.
- The derive script prints a non-negative `A_FIXED` (no negative residual blocks
  landing) — the Amendment 1 self-consistency fix is in effect.
- The §9 gate (derive, calibrate e2e, wizard path, real-train sanity) passes on the
  RTX 5070 Ti with captured output in the PR; the safety inequality (predicted ≥
  measured) holds at both probe points (K=1: 3.81 ≥ 3.05; K=4: 7.30 ≥ 6.54 GiB).
- Full `tests/unit` suite still green with the measured constants.

---

## Self-review — spec coverage map

| Spec section | Covered by |
|--------------|-----------|
| §0 root cause (whole-forward × K over-count) | Task 1.2 split helper; regression test 1.2/1.3/1.5 |
| §2 split model `(A_FIXED + A_PER_CLASS*K)*batch`, no scale | Tasks 1.1, 1.2, 1.3 |
| §2.1 self-consistent `STATIC`, no train attention term, A_FIXED clamp, safety inequality (Amendment 1) | Tasks 1.3 (drop attention), 2.1 (`_derive_split` STATIC), 3.4 (derive script STATIC+clamp), 4.1 (cited A_FIXED=0) |
| §2.1 `_attention_bytes_per_example` retained as eval SDPA cap only | Task 1.6 (kept), 1.3 (removed from train) |
| §3 candidate grid + Ks | Task 1.5 (`_candidates`) |
| §3 sort key `(lora?, -r, -K, -batch)` | Task 1.5 (`_sort_key`) |
| §3 `k` = upper bound; `k<1` raises | Task 1.5 (`decide_preset`) |
| §3 `PresetDecision.classes_per_forward` + patch + label + round-trip | Task 1.4 |
| §3 callers (init, run, calibrate) | Tasks 3.1, 3.2 |
| §4 Stage 1 derive split | Task 2.1 (`_derive_split`) |
| §4 Stage 2 analytic aim | Task 2.2 |
| §4 Stage 3 confirm-and-climb, full sacrifice-order shrink `batch->K->r->method`, bounded | Task 2.2 (`_confirm_and_climb`) |
| §4 empirical confirm-climb result authoritative for written config + returned PresetDecision | Task 2.2 (`run_calibration` builds decision from the empirical tuple; `_apply_config_rewrite(decision=...)`) |
| §4 cache-fresh re-run preserves the empirical config (no revert to analytic aim) | Task 2.2 (`chosen_*` cache keys + `_decision_from_cache`; test `cache_fresh_returns_empirical`) |
| §4 "GPU too small" only on K=1 OOM | Tasks 2.1 (`_GpuTooSmall`), 2.3 (exit 5) |
| §4 cache schema v3 | Tasks 1.1 (version), 2.1 (`_write_cache_v3`) |
| §4 reusable `run_calibration` core | Tasks 2.1–2.3 |
| §4 exit codes preserved | Task 2.3 |
| §5 wizard `_ask_peft_sizing` → calibrate + fallback | Task 3.3 |
| §6 derive script two-point split + docstring | Task 3.4 |
| §6 land cited constants (`# tbd:` then GPU) | Tasks 1.1 (tbd), 4.1 (cited) |
| §6/§7 `decide_eval_batch_size` threads K, no regression | Task 1.6 |
| §7 `_load_cache` v3 / v2-drop | Tasks 1.1, 1.6 (round-trip tests) |
| §8 file-by-file change map | all tasks |
| §9 unit tests (split, sort, climb, wizard, eval) | Tasks 1.2–1.6, 2.1–2.3, 3.3 |
| §9 real-GPU acceptance gate | Phase 4 (Tasks 4.1–4.2) |
| §10 Q1 asymmetric r climb/shrink (climb never raises r/method; shrink follows full order `batch->K->r->method`) | Task 2.2 (`_confirm_and_climb`) |
| §10 Q2 no scale conversion | Tasks 1.2, 3.4 |
| §10 Q3 `_rewrite_sizing_block` signature confirmed + threaded | Task 3.1 |
