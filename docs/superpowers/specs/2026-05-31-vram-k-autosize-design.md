# VRAM auto-sizing: split activation model, K in the search, calibrate-and-climb

**Issue:** [#203 — PEFT auto-sizing wrongly rejects 24 GiB GPUs; calibrate OOMs](https://github.com/NguyenJus/custom-sam-peft/issues/203)
**Supersedes/extends:** `docs/superpowers/specs/2026-05-28-vram-calibration-reassess-design.md` (commit `eda9fc3`).
**Release:** pre-1.0 minor bump (corrected sizing behavior + new calibrate flow → MINOR).
**Status:** locked design, single PR, no back-compat shims.

This spec corrects a sizing regression introduced by the prior spec and reworks
calibration into a model-guided, empirically-confirmed flow. It is written to be
implemented cold: every changed file and symbol is named with a line anchor, and
the memory model is specified to the formula.

---

## §0 Problem and root cause

### Symptom

Accepting PEFT auto-sizing in the interactive wizard fails on a Titan RTX
(24 GiB):

```text
could not auto-size: pick_preset(): GPU has 22.5 GiB after 1.0 GiB headroom —
SAM 3.1 needs ~27.5GiB even at QLoRA r=4 batch=1. Use a larger GPU.;
falling back to manual
```

Separately, `custom-sam-peft calibrate` OOMs before writing a cache:

```text
ERROR: calibration probe OOMed at config's sizing — GPU too small
```

Both are wrong: real training succeeds on ≤16 GiB GPUs (RTX 5070 Ti, Colab T4).
The estimate, not the hardware, is the failure.

### Root cause

The prior spec (commit `eda9fc3`, "VRAM calibration reassessment") made the
**entire** forward activation scale with `k_eff`. In
`src/custom_sam_peft/presets.py` (`_activation_per_example` / `_activation_bytes`,
~lines 188–201):

```python
per = BASE_ACTIVATION_AT_1024 * (image_size / 1024) ** 2   # ~1.45 GiB at img 1008
activation = per * batch * k_eff                            # <-- whole forward × k_eff
```

`BASE_ACTIVATION_AT_1024` is dominated by the **K-invariant** vision encoder
(hiera-large; 5184 tokens at patch=14, image=1008). Only the decoder / mask-head
activations scale with K (the number of class prompts per multiplex forward).
Multiplying the whole forward by K over-counts the encoder ~Kx.

The wizard path (`setup_wizard.py:_ask_peft_sizing` → `decide_preset()` with no
`k`) hits the worst case: `decide_preset(k=None)` sets `k_eff = MULTIPLEX_CAP =
16` (`presets.py:344`). At K=16 the lumped activation balloons:

| Config (qlora, r=4, batch=1) | K | Predicted train bytes | Activation share |
|------------------------------|---|-----------------------|------------------|
| K=1                          | 1 | **5.70 GiB**          | ~1.45 GiB        |
| K=16                         | 16 | **27.50 GiB**         | **~23.26 GiB** (bug) |

The 23.26 GiB "activation" at K=16 is the entire over-count. It exceeds a 24 GiB
card's 22.5 GiB budget, so `decide_preset` finds nothing feasible and raises.

### Why real training survives

`train_step` (`src/custom_sam_peft/train/loop.py`, ~lines 222–386) has a runtime
OOM ladder that, on an actual `OutOfMemoryError`, **halves the micro-batch first**
(inner rung) then **halves `effective_K`** (outer rung), replaying the step. The
sizing path never models this ladder, so it predicts a peak the runtime would
never actually hit.

### Why K is the lowest-priority lever

`classes_per_forward` (K) is **throughput-only**, not an accuracy lever. In
`loop.py`, per-group losses are summed and scaled by `1 / (G * grad_accum_steps)`
(see `group_scaled = group_losses["total"] / (G * cfg.train.grad_accum_steps)`,
~line 335). Class grouping changes how many forwards run per step, not the
gradient. Under this codebase's design priority — **final accuracy >
user-facing simplicity >> training speed** — K is the cheapest thing to sacrifice
under VRAM pressure, below the accuracy levers (`method`, `r`) and below batch.

### What the prior spec got wrong, precisely

The prior spec correctly added an SDPA attention term and a `k_eff` concept, but
applied `k_eff` as an **unconditional multiplier on the full lumped activation**
(`BASE_ACTIVATION_AT_1024`), instead of splitting the K-invariant encoder
activation from the K-scaling decoder activation. That single modeling error is
the entirety of #203. Everything else from the prior spec (the SDPA ceiling, the
config rewrite wiring, the runtime ladder) stays.

---

## §1 Goals and non-goals

### Goals

1. Correct the activation memory model so a 24 GiB card sizes successfully and a
   ≤16 GiB card is never falsely rejected.
2. Put K in the auto-sizing search space as the lowest-priority lever, so big
   cards get high K (throughput) and small cards trade K away before accuracy
   levers.
3. Make `calibrate` model-guided and OOM-safe: cheap probes derive the model,
   then a confirmation probe climbs to maximal card usage without an unguarded
   OOM.
4. Have the wizard run the *same* calibrate flow on consent, so wizard users get
   empirical confirm-and-climb, not just a conservative analytic estimate.
5. Honor the cite-new-hyperparams rule for every new seed constant.

### Non-goals

- No change to the runtime OOM ladder in `loop.py` (§6). It remains the safety
  net for data-dependent class counts exceeding the sized K.
- No eval behavioral regression. `decide_eval_batch_size` keeps its SDPA ceiling;
  it only consumes the new split (§6).
- No unrelated refactors. Touch only the symbols named in §7.

---

## §2 The corrected memory model (split activation)

Replace the single lumped `BASE_ACTIVATION_AT_1024` constant with two seed
constants:

- **`A_FIXED`** — K-invariant vision-encoder activations, per image (per batch
  element), measured at SAM 3.1's fixed 1008px (`SAM3_IMAGE_SIZE`).
- **`A_PER_CLASS`** — decoder / mask-head activations, per (image × class),
  measured at SAM 3.1's fixed 1008px.

SAM 3.1 always rescales to `SAM3_IMAGE_SIZE = 1008`, so image size is a constant,
not a variable. The old `(image_size / 1024) ** 2` factor was an artifact from
before 1008px was confirmed; it is **removed entirely**. The constants are defined
natively at 1008px and the formula carries no scale term:

```text
activation(method, batch, K) = (A_FIXED + A_PER_CLASS * K) * batch
```

Notes:

- `A_FIXED` no longer multiplies by K. Only the `A_PER_CLASS * K` term scales
  with class count. This removes the ~Kx over-count.
- The cached form stores both constants (§3, schema v3) and reconstructs the same
  formula: `(cache["A_fixed"] + cache["A_per_class"] * K) * batch` — no scale term.

### Symbols to change in `presets.py`

- **Remove** `BASE_ACTIVATION_AT_1024` (~line 57). Replace its references.
- **Add** module constants `A_FIXED` and `A_PER_CLASS` (seed values + citation,
  §5).
- **`_activation_per_example(image_size, cache)`** (~line 188): replace its single
  lumped return. Because the per-example activation now depends on K, fold this
  into the K-aware path or re-shape its signature to take K. Recommended: replace
  it with a single helper

  ```python
  def _activation_bytes(batch, cache, k_eff=1) -> int:
      if cache is not None:
          a_fixed = int(cache["A_fixed"])
          a_per_class = int(cache["A_per_class"])
      else:
          a_fixed = A_FIXED
          a_per_class = A_PER_CLASS
      return int((a_fixed + a_per_class * k_eff) * batch)
  ```

  The `image_size` parameter is gone from the activation helpers — with image size
  fixed it carries no information. `_attention_bytes_per_example(image_size)` still
  reads `SAM3_IMAGE_SIZE` to derive token count; that term is unchanged. Keep a
  thin `_activation_per_example` only if a caller still needs a per-image figure;
  collapse to the minimal set of helpers the call sites (`_predicted_bytes`,
  `decide_eval_batch_size`) need, with no dead helper left behind.

- **`_predicted_bytes(..., mode, k_eff)`** (~line 204):
  - **train** branch — `_model_bytes + _adapter_bytes + _optimizer_bytes +
    _activation_bytes(image, batch, cache, k_eff) +
    _attention_bytes_per_example(image) * batch + WORKSPACE_BYTES`. Unchanged
    except `_activation_bytes` now uses the split. Attention term is unchanged.
  - **eval** branch — `eval activation = _activation_bytes(image, batch, cache,
    k_eff) * forward_only_factor`. The eval branch must now accept and thread
    `k_eff` (today it hard-codes K=1 by calling `_activation_bytes` without
    `k_eff`). `decide_eval_batch_size` passes its `classes_per_forward` through as
    `k_eff` (§6).

`_attention_bytes_per_example`, `_model_bytes`, `_adapter_bytes`,
`_optimizer_bytes`, `forward_only_factor` are unchanged.

---

## §3 Search space and objective (`decide_preset`)

### Candidate grid

`_candidates()` (`presets.py:304`) currently returns `methods × rs × batches`.
Extend to `methods × rs × batches × Ks`:

- `methods = ("lora", "qlora")`
- `rs = (8, 16, 24, 32, 48, 64)` (unchanged)
- `batches = range(1, 17)` (unchanged)
- `Ks = (1, 2, 4, 8, 16)`, each capped at `MULTIPLEX_CAP`

A candidate tuple becomes `(method, r, batch, K)`.

### Objective / sort key

`_sort_key` selects the **largest-fitting** config as `feasible[0]` after sorting
ascending. The new key encodes the user's sacrifice order:

```python
def _sort_key(c):  # c = (method, r, batch, K)
    method, r, batch, K = c
    return (0 if method == "lora" else 1, -r, -K, -batch)
```

Priority, highest first: **LoRA over QLoRA → highest r → highest K → highest
batch**. Reading the tail-to-head as "what we give up first": **batch first, then
K, then r, then LoRA→QLoRA last**. This matches the runtime ladder (batch inner
rung, K outer rung) and the design priority (protect accuracy levers `method`/`r`;
spend the throughput-only lever K, and the memory-only lever batch, first).

### Meaning change for the `k` parameter

`decide_preset(k=...)` changes meaning from "representative K for the activation
term" to **"upper bound on the K search"** (default `MULTIPLEX_CAP`):

- The K candidates are filtered to `K <= min(k or MULTIPLEX_CAP, MULTIPLEX_CAP)`.
- A user who pins a lower `classes_per_forward` is respected as a cap — the search
  will not exceed it.
- `k < 1` still raises `ValueError` (preserve the existing guard at `presets.py:345`).

### `PresetDecision` gains a field

Add `classes_per_forward: int` to the `PresetDecision` dataclass (~line 73), after
`batch_size`/`grad_accum_steps` (place it adjacent to the other train-sizing
fields). Update:

- **`config_patch`** (~line 97): the `"train"` section writes
  `multiplex={"classes_per_forward": self.classes_per_forward}` alongside
  `batch_size`/`grad_accum_steps`. The deep-merge consumer maps it to
  `train.multiplex.classes_per_forward`.
- **`label()`** (~line 108): surface K, e.g. `auto: lora r=16 batch=4 K=8
  grad_accum=4 bf16 — fits in …`.
- **`to_json` / `from_json`** (~lines 124–138): no code change needed (they use
  `asdict` / field-filtering), but the new field rides along automatically;
  confirm round-trip in tests.

`decide_preset` (~line 382) reads `method, r, batch, K, predicted = feasible[0]`
and passes `classes_per_forward=K` into the returned `PresetDecision`.

### Callers to update

| Call site | Change |
|-----------|--------|
| `cli/init_cmd.py:178` | already passes `k=cfg.train.multiplex.classes_per_forward`; now this is an upper bound. The `_rewrite_sizing_block` call must also write the chosen `classes_per_forward` (see §7 for the rewrite-helper change). |
| `cli/run_cmd.py:50` (`_fallback_preset`) | call `decide_preset(k=cfg.train.multiplex.classes_per_forward)` so the fallback respects the config's K cap (today it calls `decide_preset()` with no args). |
| `cli/calibrate_cmd.py:191` area (`_apply_config_rewrite` → `decide_preset(k=k_eff, …)`) | unchanged call shape; `k_eff` is the cap. The rewrite must persist the decision's `classes_per_forward`. |

`decide_eval_batch_size` is **not** a K-search consumer; it keeps its own loop
(§6).

---

## §4 Calibration flow (`calibrate_cmd.py`)

Replace the single-probe `calibrate` with a three-stage model-guided flow. Factor
the core into a reusable function (§5 / §7) shared by the CLI and the wizard.

### Stage diagram

```text
Stage 1 — DERIVE (two cheap probes, both fit a 16 GiB T4)
  probe @ (qlora, r=4, batch=1, K=1)  -> peak_K1   (~5.7 GiB)
  probe @ (qlora, r=4, batch=1, K=4)  -> peak_K4   (~10 GiB)
  A_per_class = (peak_K4 - peak_K1) / (4 - 1)
  A_fixed     = peak_K1 - fixed_overhead - A_per_class
                (fixed_overhead = model + adapter + optimizer + workspace
                 + attention, as computed today at calibrate_cmd.py:180-186)
        |
        v
Stage 2 — AIM (pure analytic, no GPU work)
  Build the split cache dict {A_fixed, A_per_class}; call decide_preset over the
  full grid (K<=16, batch<=16) to pick the best-fitting (method, r, batch, K).
  Big cards (40/80 GiB) naturally land on high K and high batch here.
        |
        v
Stage 3 — CONFIRM + CLIMB/SHRINK (bidirectional, OOM-safe)
  Probe the Stage-2 choice. Then walk the sorted candidate list:
    fits w/ headroom & within budget -> step UP one rung (grow K first, then
                                          batch; never r) and re-probe; keep
                                          climbing while it fits.
    OOMs                             -> step DOWN the full sacrifice order
                                          (batch -> K -> r -> LoRA→QLoRA) and
                                          re-probe until it fits.
  Record the REAL measured peak of the final fitting config.
  Cap the climb at grid max (K=16, batch=16) or first OOM.
```

### Stage 1 — derive the split

Run two cheap probes via the existing `_run_probe(method, r, k_eff, batch)`
(`calibrate_cmd.py:65`), both at `(qlora, r=4, batch=1)`:

- `peak_K1 = _run_probe(method="qlora", r=4, k_eff=1, batch=1)`
- `peak_K4 = _run_probe(method="qlora", r=4, k_eff=4, batch=1)`

Both fit a 16 GiB T4 (≈5.7 / ≈10 GiB). Solve:

```text
A_per_class = (peak_K4 - peak_K1) / (4 - 1)
fixed_overhead = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4)
                 + WORKSPACE_BYTES + _attention_bytes_per_example(IMG) * 1
A_fixed = peak_K1 - fixed_overhead - A_per_class
```

Clamp `A_fixed` and `A_per_class` to `>= 0`; warn (mirroring today's
negative-activation warning at `calibrate_cmd.py:188`) if either clamps, since it
signals the overhead model needs re-derivation.

**"GPU too small" fires only if the K=1 probe itself OOMs.** Catch
`torch.cuda.OutOfMemoryError` around the `peak_K1` probe and emit the existing
exit-code-5 error there; the K=4 probe OOMing instead degrades to a conservative
default (treat `A_per_class` from a single point unavailable — fall back to
analytic seeds for `A_per_class` and use `peak_K1` to fix `A_fixed`), with a
warning.

### Stage 2 — aim analytically

Assemble the split-cache dict in memory and call `decide_preset` over the full
grid to choose `(method, r, batch, K)`. This is the analytic "aim" — no GPU
allocation. On a 40/80 GiB card it selects high K and high batch; on 24 GiB it
backs K/batch down to fit.

### Stage 3 — confirm and climb/shrink

Walk the **sorted candidate list** (same `_sort_key` order). The two directions
are deliberately asymmetric in `r`:

- **Climb-up** (using headroom on a big card) grows only the throughput dims, K
  then batch, at the chosen method/r. It never probe-bumps `r` — raising an
  accuracy lever on the strength of a probe is what the design priority forbids.
- **Shrink-down** (the chosen config OOMs because the analytic aim was optimistic)
  follows the full sacrifice order, so training always fits the environment's GPU.

Starting from the Stage-2 choice:

- **Probe** the current `(method, r, batch, K)`. Wrap in `try/except
  torch.cuda.OutOfMemoryError`.
- **Fits, with headroom, within budget** → step **UP** one rung (grow K to the
  next grid value first; when K is at its max for this card, grow batch; never
  raise r) and re-probe. Keep climbing while it fits and stays within `budget =
  total - headroom`.
- **OOMs** → step **DOWN** the full sacrifice order: shrink batch first; when
  batch is at 1, shrink K; when K is at 1, drop r to the next-lower grid value;
  when r is at its minimum, flip LoRA→QLoRA. Re-probe until it fits. A clean
  implementation re-runs the Stage-2 analytic aim with the r-cap lowered after a
  batch/K exhaustion, then re-confirms; either way the invariant is **training
  fits the GPU whenever any config does**.
- **Stop** at grid max (K=16, batch=16) when climbing, at the first fit when
  shrinking, or when the candidate space is exhausted (then surface the existing
  "GPU too small" path).

Record the **real measured peak** of the final fitting config. This is the
"maximize card usage / be willing to safely OOM the card" behavior #203 calls
for — push until the card actually refuses, but never leave an unguarded OOM
crash.

**Bounded probe count.** Each rung is exactly one probe. The climb is capped at
the grid (`Ks` has 5 values, `batches` has 16), so worst case is a handful of
probes — a pathological model error cannot loop. State the cap explicitly in the
loop (e.g. a `max_probes` guard equal to `len(Ks) + len(batches)`).

### Cache schema v3

Bump `CACHE_SCHEMA_VERSION` 2 → 3 (`presets.py:67`). The cache payload
(`calibrate_cmd.py:199`) changes:

- **Add** `"A_fixed"` and `"A_per_class"` (ints, measured natively at SAM 3.1's
  fixed 1008px — no scaling conversion, since image size is constant).
- **Remove** `"activation_bytes_per_example"`.
- Keep `schema_version`, `calibrated_at`, `gpu_name`, `gpu_total_memory_bytes`,
  `sam3_checkpoint_sha`, `torch_version`, `custom_sam_peft_version`,
  `peak_memory_bytes_at_probe` (now the Stage-3 confirmed peak).

`_load_cache` (`presets.py:248`) already ignores caches whose `schema_version`
mismatches, so stale v2 caches are dropped automatically. `_cache_is_fresh`
(`calibrate_cmd.py:39`) is unchanged (it compares against
`CACHE_SCHEMA_VERSION`).

### Reusable core

Extract a function (e.g. `run_calibration(*, config, output, force) ->
PresetDecision`) holding stages 1–3 + cache write + config rewrite, so both the
`calibrate` CLI command and the wizard (§4) call it. The `calibrate` Typer command
becomes a thin wrapper that parses options, calls `run_calibration`, prints the
summary, and maps failures to exit codes (preserve codes: 2 no-CUDA, 3 checkpoint
missing, 4 probe failure, 5 GPU-too-small/K1-OOM, 6 cache-write failure).

---

## §5 Wizard integration (`setup_wizard.py:_ask_peft_sizing`)

Currently `_ask_peft_sizing` (~lines 338–352) calls `decide_preset()` (pure
analytic, worst-case K) on consent. Change it so consent runs the **calibrate
probe flow** (§4 stages 1–3) on the in-progress config values (method / r / batch,
and the K upper bound), giving wizard users the same confirm-and-climb
utilization.

- On consent and `ctx.cuda_available`: call the shared `run_calibration(...)`
  (§4) against the wizard's in-progress sizing values, returning a
  `PresetDecision`; echo `decision.label()`; return `decision.config_patch`.
- **Graceful fallback** to analytic `decide_preset(k=...)` (then to manual) when
  CUDA is unavailable, the SAM 3.1 checkpoint is not loadable, or the probe
  otherwise fails — with a clear one-line message naming the reason. Catch the
  same exception classes the CLI maps (FileNotFoundError, OutOfMemoryError,
  RuntimeError, ValueError) and degrade rather than crash the wizard.

Pure analytic `decide_preset` remains in two roles: (a) the wizard/CLI fallback,
and (b) the internal Stage-2 "aim" inside calibration.

---

## §6 Analytic seed derivation and citation (cite-new-hyperparams — MANDATORY)

The analytic path needs `A_FIXED` and `A_PER_CLASS` seeds for when no calibration
cache exists. Every new/changed default hyperparam in this repo must carry a
rigorous citation **or** an explicit `# tbd:` tag — never a silent guess.

### Extend `scripts/_derive_preset_constants.py`

The script (~90 lines) currently prints a single per-K activation figure. Extend
it to emit the **two-point split**:

- Run two probes at `(method, r, batch=1)` for `K=1` and `K=4` (reuse its
  existing single-probe body in a loop over K).
- Compute `A_per_class = (peak_K4 - peak_K1) / 3` and `A_fixed = peak_K1 -
  fixed_overhead - A_per_class` at SAM 3.1's native 1008px — no scaling
  conversion (the prior `* (1024 / image_size) ** 2` artifact is removed).
- Print copy-paste-ready `A_FIXED = ...` and `A_PER_CLASS = ...` lines.
- Update its module docstring (it references `BASE_ACTIVATION_AT_1024` and "design
  §3.3").

### Land the constants

Run the script **once** on the reference GPU — the dev-env **RTX 5070 Ti (16 GiB,
sm_120, driver 610.47)** — and hardcode the result as `A_FIXED` / `A_PER_CLASS` in
`presets.py` with a rigorous citation comment containing: GPU name + compute
capability, commit SHA, date, and the exact derivation command (`uv run python
scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1`). The split is
model-driven — activation bytes per example depend on the model and dtype, not the
card — so constants derived on the 5070 Ti apply to all GPUs.

**If the implementing session lacks GPU access:** land the constants as
`# tbd:`-tagged seeds — a documented, conservative split of the current
`BASE_ACTIVATION_AT_1024` evaluated at 1008px (~1.45 GiB): attribute the bulk to
`A_FIXED` since the encoder dominates, a small slice to `A_PER_CLASS` — with the
exact derivation recipe recorded in the comment. A conservative split slightly
over-attributes to `A_FIXED`, which can only *raise* the predicted floor, keeping
sizing safe. **Never ship a silent guess.** The `# tbd:` tag is the trigger for
re-derivation on the 5070 Ti before merge — which the §9 real-GPU gate enforces.

---

## §7 Out of scope / unchanged

- **Runtime OOM ladder** (`loop.py`, ~222–386): untouched. It stays the safety
  net for data-dependent class counts that exceed the sized K (batch inner rung →
  K outer rung → hard fail).
- **`decide_eval_batch_size`** (`presets.py:401`; consumed by
  `train/trainer.py:304,538`): keeps its SDPA attention ceiling and its own batch
  loop. It only *consumes* the new split — `eval activation = (A_fixed +
  A_per_class * K) * forward_only_factor` — by threading its `classes_per_forward`
  arg into `_predicted_bytes(..., mode="eval", k_eff=classes_per_forward)` and
  into the `_act_per_example` term used for the cap. No eval behavioral regression
  intended; verify the cap still only lowers `best_bs`.
- No unrelated refactors.

---

## §8 File-by-file change map

| File | Symbol / anchor | Change |
|------|-----------------|--------|
| `src/custom_sam_peft/presets.py` | `BASE_ACTIVATION_AT_1024` (~57) | **Remove.** |
| | `A_FIXED`, `A_PER_CLASS` (new, near ~57) | Add seed constants + citation/`# tbd:` (§6). |
| | `CACHE_SCHEMA_VERSION` (67) | 2 → 3. |
| | `PresetDecision` (73) | Add `classes_per_forward: int` field. |
| | `PresetDecision.config_patch` (97) | Write `train.multiplex.classes_per_forward`. |
| | `PresetDecision.label` (108) | Surface `K=`. |
| | `_activation_per_example` / `_activation_bytes` (188–201) | Replace with split formula `(A_fixed + A_per_class*K)*batch*scale`; read split from cache. |
| | `_predicted_bytes` (204) | train + eval branches consume the split; eval threads `k_eff`. |
| | `_load_cache` (248) | reads `A_fixed`/`A_per_class` keys (schema v3); stale-version drop already handled. |
| | `_candidates` (304) | add `Ks=(1,2,4,8,16)`; return `(method, r, batch, K)`. |
| | `_sort_key` (311) | key `(lora?, -r, -K, -batch)` over 4-tuple. |
| | `decide_preset` (323) | `k` = upper bound on K search; iterate 4-tuples; raise message recomputed via split; return `classes_per_forward`. |
| | `decide_eval_batch_size` (401) | thread `classes_per_forward` into the split via `_predicted_bytes(mode="eval", k_eff=...)` and the cap term. |
| `src/custom_sam_peft/cli/calibrate_cmd.py` | `_run_probe` (65) | reused as-is for the multi-probe stages (no signature change required). |
| | `run_calibration` (new) | stages 1–3 core; cache write (v3); config rewrite; returns `PresetDecision`. |
| | `calibrate` (129) | thin wrapper over `run_calibration`; preserve exit codes; new "GPU too small" only on K=1-probe OOM. |
| | `_apply_config_rewrite` (98) | persist `classes_per_forward` via the rewrite helper. |
| | cache payload (199) | add `A_fixed`/`A_per_class`; remove `activation_bytes_per_example`. |
| `src/custom_sam_peft/cli/setup_wizard.py` | `_ask_peft_sizing` (338) | consent → `run_calibration(...)`; graceful fallback to analytic `decide_preset(k=...)` then manual. |
| `src/custom_sam_peft/cli/init_cmd.py` | `run_init` (~178) | `decide_preset(k=cfg…classes_per_forward)`; rewrite must persist chosen `classes_per_forward`. |
| `src/custom_sam_peft/cli/run_cmd.py` | `_fallback_preset` (48–50) | pass `k=cfg.train.multiplex.classes_per_forward`. |
| `src/custom_sam_peft/cli/_config_rewrite.py` | `_rewrite_sizing_block` | extend to write `train.multiplex.classes_per_forward` (verify current signature; add a `classes_per_forward` arg threaded from all callers). |
| `scripts/_derive_preset_constants.py` | `main` | emit two-point split; update docstring. |
| `src/custom_sam_peft/config/schema.py` | `MultiplexConfig.classes_per_forward` (517) | no change (default 16, `ge=1, le=16`); confirm the rewrite writes within bounds. |

> Implementer note: `_rewrite_sizing_block` is referenced by both `init_cmd.py`
> and `calibrate_cmd.py`. Read it first to confirm its current parameters before
> threading `classes_per_forward`; update every call site in lockstep.

---

## §9 Testing strategy

Existing conventions to mirror:

- `tests/unit/test_presets.py` stubs CUDA via monkeypatch helpers
  (`_force_cuda_available`, `_stub_gpu` setting `total_memory` / device name /
  capability) and asserts on `decide_preset()` outputs per VRAM tier. These run on
  CPU.
- `tests/unit/test_calibrate_cmd.py` monkeypatches
  `torch.cuda.max_memory_allocated`, `reset_peak_memory_stats`, and patches
  `calibrate_cmd._run_probe` to return a synthetic peak (`_patch_probe`), then
  invokes `calibrate` and inspects the written cache JSON. These run on CPU.
- `tests/gpu/test_calibrate_real.py` holds the real-GPU probe coverage (out of
  scope to run in CI; keep its shape).

### New / updated unit tests

**Split model (`test_presets.py`).**

- `_activation_bytes` / `_predicted_bytes` produce the correct value at several K:
  assert linearity in K (e.g. value at K=8 equals `A_fixed + 8*A_per_class`
  scaled), and that K=1 vs K=16 differ by exactly `15 * A_per_class * batch *
  scale` — i.e. the encoder term does **not** scale with K. This is the direct
  regression guard for #203.
- A 24 GiB card now sizes successfully (no `RuntimeError`): add a tier test at
  `_stub_gpu(int(24 * _GB))` asserting a `PresetDecision` is returned (replacing
  the false-rejection behavior).
- Cache round-trip: a v3 split cache is consumed by `decide_preset`
  (provenance="calibrated"); a v2 cache is ignored (stale).
- `PresetDecision` JSON round-trip carries `classes_per_forward`.

**Sort / objective (`test_presets.py`).**

- At fixed r, `(K=8, batch=1)` sorts ahead of `(K=1, batch=8)` (K protected over
  batch).
- r is protected over both K and batch: `(r=32, K=1, batch=1)` sorts ahead of
  `(r=16, K=16, batch=16)`.
- LoRA sorts ahead of QLoRA when both fit.
- Big-card tiers (40/80 GiB) select high K and high batch; small card (24 GiB)
  backs K/batch down but keeps the highest feasible r and LoRA.
- `decide_preset(k=4)` never returns K>4 (cap honored); `k=0`/`k=-1` raise.

**Calibrate confirm-and-climb (`test_calibrate_cmd.py`).**

- Mock `_run_probe` to return synthetic peaks as a function of `(method, r, k_eff,
  batch)` so the rung walk is deterministic. Cover:
  - Stage 1 derivation: assert `A_fixed`/`A_per_class` solved from the two
    synthetic K=1/K=4 peaks match the closed form, and land in the v3 cache.
  - Climb: peaks that fit until a threshold → assert the walk grows K first, then
    batch, and stops at the last fitting config; assert recorded peak is the
    final fitting probe's value.
  - Injected OOM: have the mock raise `torch.cuda.OutOfMemoryError` at a chosen
    rung → assert the walk steps **down** (batch first, then K) until it fits, and
    records the fitting config.
  - Bounded probe count: assert the total number of `_run_probe` calls is `<=
    len(Ks) + len(batches) + 2` (the two derivation probes plus the bounded
    walk), proving no unbounded loop.
  - K=1-probe OOM → exit code 5 with the "GPU too small" message; K=4-probe OOM →
    degraded-but-successful cache with a warning.
- Cache schema is v3 with `A_fixed`/`A_per_class` and **no**
  `activation_bytes_per_example`.

**Wizard (`tests/.../test_setup_wizard*` — locate existing wizard tests and mirror).**

- Consent + CUDA available → `run_calibration` invoked; `config_patch` carries
  `classes_per_forward`.
- Consent + probe failure (checkpoint missing / OOM) → falls back to analytic
  `decide_preset`, then manual, without raising.

**Eval (`test_presets.py`).**

- `decide_eval_batch_size` consumes the split and threads its
  `classes_per_forward`; the SDPA cap still only lowers `best_bs` (no regression).

### Real-GPU acceptance gate (MANDATORY before merge)

The prior calibration regression shipped because it was never run on a real GPU.
This is a **hard merge gate**, not optional: the work is not done until calibrate
is confirmed working on the dev-env **RTX 5070 Ti (16 GiB, sm_120)** — the
constrained 16 GiB card where the split model and the climb/shrink flow matter
most. The implementer (or the implementation-orchestrator before opening the PR)
must run and capture output for:

1. **Seed derivation** — `uv run python scripts/_derive_preset_constants.py
   --method qlora --r 4 --batch 1`; record `A_FIXED` / `A_PER_CLASS` and confirm
   the two cheap probes (K=1, K=4) both fit (no OOM).
2. **`calibrate` end-to-end** — `uv run custom-sam-peft calibrate` on a real
   config: assert it writes a v3 cache, the confirm-and-climb walk terminates,
   and the chosen config's measured peak is ≤ the 16 GiB budget (no unguarded OOM
   crash; a *caught* OOM during the climb is expected and fine).
3. **Wizard auto-calibrate path** — drive `_ask_peft_sizing` consent on GPU
   (scripted / non-interactive) and confirm it returns a fitting `PresetDecision`
   via `run_calibration`, with graceful fallback when the checkpoint is absent.
4. **Sanity vs. reality** — train a few steps at the chosen config and confirm it
   does not OOM at runtime, closing the loop the analytic-only ship failed to.

Checkpoint note: the worktree has no `models/sam3.1/`; point
`ModelConfig.local_dir` (or the env override) at the main checkout's
`models/sam3.1` (the 3.3 GB `sam3.1_multiplex.pt`) so the probe can load. The GPU
run happens in the implementation session, not at plan time. CI keeps the
CPU-mocked tests above as the always-on guard; this gate is the human-run,
real-silicon confirmation.

---

## §10 Open questions

1. **Resolved — Stage-3 confirmation is asymmetric in `r`.** Climb-up grows
   **K then batch only**, never probe-bumping `r`. Shrink-down follows the full
   sacrifice order **batch → K → r → (LoRA→QLoRA)** so training always fits the
   environment's GPU even when the analytic aim under-predicts. See §3 / §4
   Stage 3.
2. **Resolved — no reference-scale conversion.** SAM 3.1 always rescales to the
   fixed `SAM3_IMAGE_SIZE = 1008`, so image size is constant. The `(image_size /
   1024) ** 2` factor was an artifact from before 1008px was confirmed and is
   removed everywhere (model, cache, derive script). `A_FIXED` / `A_PER_CLASS` are
   defined and cached natively at 1008px.
3. **`_rewrite_sizing_block` signature.** The exact parameters are unread here;
   the implementer must confirm whether it takes keyword sizing args or a
   `PresetDecision`, then thread `classes_per_forward` consistently across
   `init_cmd`, `calibrate_cmd`, and any other caller.
