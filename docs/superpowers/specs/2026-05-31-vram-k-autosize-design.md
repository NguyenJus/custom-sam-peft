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

> **Amendment 2 (2026-05-31, cc-aware attention model — SUPERSEDES Amendment 1).**
> Amendment 1 dropped the analytic attention term **universally** based on a real-GPU
> probe on the dev-env RTX 5070 Ti. That direction is **wrong** and is superseded
> here. The 5070 Ti is **compute capability 12.0 (sm_120)**, which receives
> **FlashAttention-2** — a best-case card where encoder self-attention never
> materializes the N×N score matrix (the whole K=1 forward activation is only ≈0.96
> GiB). **Older GPUs do not get flash:** FlashAttention-2 in PyTorch SDPA requires
> **sm_80+ (cc ≥ 8.0)**; Turing (cc 7.5: T4, RTX 2080) and Pascal (cc 6.1: GTX 1080)
> do not. The mem-efficient SDPA backend often covers Turing, but **Pascal commonly
> falls back to the math backend, which materializes the full `H·N²` fp32 score
> matrix** (≈1.6 GiB/image at SAM 3.1's 1008px / N=5184 tokens / 16 heads). cc<8.0 is
> a first-class supported target (`presets.py` already branches dtype on `cc < (8,0)`
> for fp16; GTX 1080 / issue-79 older-CUDA testing is in scope). Dropping the
> materialized-attention term **universally** would re-introduce the exact OOM-causing
> **under-count** that `_attention_bytes_per_example` was added to fix on no-flash
> cards.
>
> **What Amendment 2 keeps from Amendment 1 (the genuine fix).** The attention term
> is **K-invariant** in the predictor (`* batch`, never `* k_eff`); only `A_PER_CLASS`
> scales with K, so the split fix already cures #203's ×K over-count regardless of the
> attention term. The only real defect Amendment 1 reacted to is that the *derivation*
> subtracted a worst-case (materialized) attention overhead from a best-case (flash)
> measured peak → a negative `A_FIXED`. Amendment 2 fixes that by making the attention
> term **conditional on the card's SDPA regime**, not by deleting it.
>
> **The cc-aware model.** A helper `_flash_attention_available(cc) -> bool` returns
> `cc >= (8, 0)`. The train predictor adds `_attention_bytes_per_example(image) *
> batch` **only when flash is NOT available** (cc < 8.0, or cc unknown → conservative
> "no flash" over-estimate). On flash cards the real forward attention is already
> folded into the empirically-derived `A_FIXED`/`A_PER_CLASS` (so no separate term);
> on no-flash cards the materialized term is re-added for safety. The derivation
> subtracts **exactly the terms the predictor adds for the card's regime**
> (`overhead_to_subtract = STATIC + (attention_materialized if not flash else 0)`), so
> the stored seeds are a **portable flash-baseline**: seeds derived on the 5070 Ti
> (flash) are valid for ALL cards, and the predictor re-adds the materialized
> attention only for cc<8.0. No re-derivation on a Pascal/Turing card is needed.
> See §2.1 for the per-regime formulas, the `STATIC` definition, and the safety
> inequalities for both a flash and a no-flash card.

Replace the single lumped `BASE_ACTIVATION_AT_1024` constant with two seed
constants:

- **`A_FIXED`** — K-invariant vision-encoder activation, per image (per batch
  element), measured at SAM 3.1's fixed 1008px (`SAM3_IMAGE_SIZE`), defined as the
  **non-negative residual** `clamp(peak_K1 − STATIC − A_PER_CLASS, min=0)` (see
  §2.1 for `STATIC`). On the dev-env 5070 Ti this residual is ≤ 0 and clamps to
  **0**: the K-invariant encoder activation is below the model-weight conservatism
  margin baked into the analytic `STATIC`, so `STATIC` already absorbs it. `A_FIXED
  = 0` is a measured/clamped result, cited as such (§6), not a guess.
- **`A_PER_CLASS`** — decoder / mask-head activation, per (image × class), measured
  at SAM 3.1's fixed 1008px via the two-point differential
  `(peak_K4 − peak_K1) / (4 − 1)` (no analytic overhead enters this term).

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
- **Conditional (cc-aware) attention term — Amendment 2.** The full predicted train
  peak is regime-dependent:

  ```text
  flash_available(cc) := cc >= (8, 0)          # cc unknown -> False (conservative)
  attn(batch)         := _attention_bytes_per_example(SAM3_IMAGE_SIZE) * batch
                         # K-INVARIANT: scales with batch, NEVER with K
  predicted_peak(method, r, batch, K) =
        STATIC(method, r) + (A_FIXED + A_PER_CLASS * K) * batch
      + (attn(batch) if not flash_available(cc) else 0)
  ```

  On a **flash card (cc ≥ 8.0)** the forward attention is already inside the measured
  activation that the empirical split fits, so adding a separate analytic term would
  double-count — no attention term. On a **no-flash card (cc < 8.0)** SDPA commonly
  falls back to the math backend, which materializes the full `H·N²` fp32 score
  matrix; that worst case is NOT in the flash-baseline split, so the materialized
  `_attention_bytes_per_example` term is **re-added** for safety. `STATIC` and
  `_attention_bytes_per_example` are defined in §2.1.
- **The attention term is K-invariant.** It multiplies `batch`, never `k_eff`. Only
  `A_PER_CLASS * K` scales with class count. So re-adding the attention term on
  no-flash cards does **not** re-trigger #203's ×K over-count — the split fix already
  cured that, and a 24 GiB card still sizes with the term present (§2.1 confirms).
- **Self-consistency requirement (regime-matched).** The derive script (§6) inverts
  the **same overhead the predictor adds for the deriving card's regime**:
  `overhead_to_subtract = STATIC + (attn(1) if not flash_available(cc) else 0)`. So
  `predicted_peak` reproduces the measured peak at the probe points (up to the clamp),
  and the stored `A_FIXED`/`A_PER_CLASS` are a **portable flash-baseline** — derived
  on the 5070 Ti (flash, subtract STATIC only) they apply to all cards; the predictor
  re-adds the materialized attention only when cc < 8.0. At K=1 on a flash card,
  `predicted_peak = STATIC + A_FIXED + A_PER_CLASS`; with `A_FIXED` clamped to 0 this
  **over-predicts** by exactly the clamped residual magnitude — a safety margin,
  never an under-prediction.

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
  fixed it carries no information. `_attention_bytes_per_example(image_size)` is
  **retained** (it still reads `SAM3_IMAGE_SIZE` to derive token count) and is now
  called by **(a)** the `decide_eval_batch_size` SDPA cap and **(b)** the
  `_predicted_bytes` train branch **conditionally**, only when flash is unavailable
  (Amendment 2, §2.1). Keep a thin `_activation_per_example` only if a caller still
  needs a per-image figure; collapse to the minimal set of helpers the call sites
  (`_predicted_bytes`, `decide_eval_batch_size`) need, with no dead helper left
  behind.

- **`_flash_attention_available(cc)`** (NEW helper, Amendment 2) — returns
  `cc >= (8, 0)`. cc ≥ 8.0 (Ampere/Ada/Hopper/Blackwell) gets FlashAttention-2 /
  mem-efficient SDPA → no separate materialized attention term. cc < 8.0
  (Turing 7.5, Pascal 6.1) does not → assume the math backend materializes the
  `H·N²` score matrix → include the attention term. **Conservative default:** when
  `cc` is `None`/unreadable, return `False` (assume no flash → include the term →
  safe over-estimate). Deliberate conservatism: Turing (7.5) is treated as no-flash
  even though it usually gets mem-efficient SDPA — over-estimating is always safe,
  and Pascal genuinely needs the term.

- **`_predicted_bytes(..., mode, k_eff, flash_available)`** (~line 218): gains a
  `flash_available: bool = True` parameter (default `True` keeps existing bare unit
  calls in the flash regime). Callers that know the card — `decide_preset`,
  `decide_eval_batch_size` — compute `_flash_attention_available(cc)` from the
  device capability and pass it through.
  - **train** branch — `STATIC + split + (attn(batch) if not flash_available else 0)`.
    Here `split = _activation_bytes(batch, cache, k_eff)`;
    `STATIC = _model_bytes + _adapter_bytes + _optimizer_bytes + WORKSPACE_BYTES`;
    `attn(batch) = _attention_bytes_per_example(image_size) * batch` (K-invariant). On
    a flash card the attention is folded into the empirical split (no term); on a
    no-flash card the materialized term is re-added for safety (§2 / §2.1). This
    supersedes Amendment 1's universal drop.
  - **eval** branch — `_model_bytes(method) + _activation_bytes(batch, cache,
    k_eff) * forward_only_factor + WORKSPACE_BYTES + (attn(batch) if not
    flash_available else 0)`. The eval branch must accept and thread `k_eff` (today
    it hard-codes K=1 by calling `_activation_bytes` without `k_eff`), and now also
    honors `flash_available`: under no-flash it adds the same materialized attention
    term so eval stays SAFE (never under-predicts). `decide_eval_batch_size` passes
    its `classes_per_forward` as `k_eff` and `_flash_attention_available(cc)` as
    `flash_available` (§6). The independent SDPA *ceiling* inside
    `decide_eval_batch_size` is unchanged (§2.1 / §7) — it is an additional cap, not
    the same term.

`_model_bytes`, `_adapter_bytes`, `_optimizer_bytes`, `forward_only_factor`,
`WORKSPACE_BYTES` are unchanged. `_attention_bytes_per_example` is **retained** and
is now called by `_predicted_bytes` **only on the no-flash branch** (cc < 8.0) and by
the eval SDPA ceiling in `decide_eval_batch_size` (§2.1 / §7).

### §2.1 The regime-matched overhead, the cc-aware attention term, the SDPA cap, and the safety inequalities

> **Amendment 2.** The attention term is now **conditional on the card's SDPA
> regime** (flash vs. math backend), not universally dropped. The predictor adds, and
> the derive script subtracts, **exactly the same regime-matched overhead**, so the
> stored seeds form a portable flash-baseline. This section gives `STATIC`, the
> `_flash_attention_available` proxy, the per-regime `overhead_to_subtract`, and the
> safety inequalities for **both** a flash and a no-flash card.

**`STATIC` definition (shared by predictor and derive script).**

```text
STATIC(method, r) = _model_bytes(method) + _adapter_bytes(r)
                    + _optimizer_bytes(r) + WORKSPACE_BYTES
```

`_model_bytes` stays analytic — it is deliberately conservative (over-estimates
model weights), which keeps sizing safe and remains portable to arbitrary method/r.

**The `_flash_attention_available(cc)` proxy and the regime-matched overhead.**

```text
_flash_attention_available(cc) := (cc is not None) and cc >= (8, 0)
attn(batch)                    := _attention_bytes_per_example(SAM3_IMAGE_SIZE) * batch  # K-invariant

# Predictor (train) adds:
overhead_added      = STATIC + (attn(batch) if not flash_available else 0)
# Derive script (§6) subtracts, at batch=1:
overhead_to_subtract = STATIC + (attn(1)     if not flash_available(cc) else 0)
A_FIXED = clamp(peak_K1 − overhead_to_subtract − A_PER_CLASS, min=0)
```

- On a **flash card** (the 5070 Ti dev box, cc 12.0): `overhead_to_subtract = STATIC`
  only → `A_FIXED` is the small flash residual (clamps to 0 on the dev GPU; cited as
  measured, §6).
- On a **no-flash card** (cc < 8.0): `overhead_to_subtract = STATIC + attn(1)` → the
  measured peak (which on the math backend *includes* the materialized attention) has
  that attention subtracted off, so `A_FIXED` is regime-normalized to the **same
  flash-baseline quantity** the predictor stores.

**Portability of the seeds.** Because the derive script subtracts exactly the term
the predictor re-adds for the same regime, one `A_FIXED`/`A_PER_CLASS` pair is a
portable flash-baseline: derived on the 5070 Ti (flash), it is valid for **all**
cards. The predictor re-adds the materialized attention only when cc < 8.0. **No
re-derivation on a Pascal/Turing card is needed.** This rests on one accepted
approximation: `A_PER_CLASS` (the per-class *decoder* term, taken over a few class
queries) is small and robust across SDPA backends relative to the dominant *encoder*
self-attention captured by the conditional `attn` term — so a single flash-derived
`A_PER_CLASS` is reused across regimes. Flagged as an accepted approximation with a
follow-up (§9) to validate on a real cc < 8.0 card if one becomes available.

Measured on the dev-env RTX 5070 Ti (qlora, r=4, batch=1, sm_120, driver 610.47):

| Quantity | Value (GiB) |
|----------|-------------|
| `STATIC("qlora", 4)` = 2.391 + 0.001 + 0.004 + 0.250 | **2.646** |
| `A_PER_CLASS` = (peak_K4 − peak_K1)/3 = (6.54 − 3.049)/3 | **1.163** (1,248,840,021 B) |
| residual `peak_K1 − STATIC − A_PER_CLASS` = 3.049 − 2.646 − 1.163 | **−0.760** → clamp **0** |
| `A_FIXED` (clamped) | **0** |

`A_PER_CLASS` is a pure empirical differential (no analytic overhead enters it) and
is unchanged from the original design — it is the validated core of #203.

**Fate of `_attention_bytes_per_example` (Amendment 2).** The symbol is **retained**
and now has **two** consumers: the conditional no-flash branch of the train/eval
predictor (above), and the SDPA ceiling inside `decide_eval_batch_size`. The eval
SDPA ceiling (`_attn_per_example`, `attn_budget`) is kept as an **unconditional**
conservative cap — it stays regime-independent on purpose: that cap models the math
backend's full `B·H·N²` fp32 score matrix, only ever *lowers* `best_bs` (issue #162),
and can therefore never cause an under-prediction. Keeping it unconditional (rather
than making it cc-aware) preserves a safe eval ceiling on every card with zero OOM
risk; making it cc-aware could only *raise* `best_bs` on flash cards, which is the
unsafe direction, so it is deliberately left as-is. The predictor's *train/eval
attention term* is the cc-aware one; the eval SDPA *ceiling* is the always-on guard.

**Safety inequality A — flash card (RTX 5070 Ti, cc 12.0).** The predictor must
over-predict (≥ measured); under-prediction is forbidden (OOM risk). On a flash card,
no attention term is added (`flash_available = True`), and `A_FIXED = 0`:

```text
predicted_peak(qlora, 4, 1, K=1) = STATIC + (0 + A_PER_CLASS·1)            # no attn term
                                 = 2.646 + 1.163 = 3.809 GiB  ≥  measured 3.049 GiB   (+0.760)
predicted_peak(qlora, 4, 1, K=4) = STATIC + (0 + A_PER_CLASS·4)
                                 = 2.646 + 4.652 = 7.298 GiB  ≥  measured 6.540 GiB   (+0.758)
```

Both probe points are **safely conservative** (~0.76 GiB margin each). The clamp can
only *raise* the predicted floor (residual was negative), so it strictly preserves
over-prediction at every point.

**Safety inequality B — synthetic no-flash card (math-backend SDPA, cc < 8.0).** The
seeds are the same portable flash-baseline (`A_FIXED = 0`, `A_PER_CLASS = 1.163
GiB`), but the predictor **re-adds** the materialized attention term `attn(batch) =
_attention_bytes_per_example(1008) · batch ≈ 1.60 GiB · batch` (`H=16`, `N=5184`,
`16·5184²·4 B ≈ 1.717e9 B ≈ 1.60 GiB`). At batch=1:

```text
predicted_peak_noflash(qlora, 4, 1, K=1) = STATIC + (0 + A_PER_CLASS·1) + attn(1)
                                         = 2.646 + 1.163 + 1.60 = 5.41 GiB
predicted_peak_noflash(qlora, 4, 1, K=4) = STATIC + (0 + A_PER_CLASS·4) + attn(1)
                                         = 2.646 + 4.652 + 1.60 = 8.90 GiB
```

A plausible **math-backend** peak on such a card is the flash measured peak *plus*
the materialized score matrix it cannot avoid: `plausible_noflash_peak(K=1) ≈ 3.049 +
1.60 = 4.65 GiB` and `(K=4) ≈ 6.540 + 1.60 = 8.14 GiB`. So:

```text
K=1:  predicted 5.41 GiB  ≥  plausible math-backend 4.65 GiB   (+0.76, safe)
K=4:  predicted 8.90 GiB  ≥  plausible math-backend 8.14 GiB   (+0.76, safe)
```

The materialized ≈1.6 GiB/image term is re-added and the prediction stays **above** a
plausible math-backend peak — exactly the under-count `_attention_bytes_per_example`
was added to fix. The ~0.76 GiB margin (from the clamped flash residual) carries over
to the no-flash regime, so dropping the term universally (Amendment 1) would have
under-predicted by ≈1.6 GiB on a no-flash card — the forbidden direction.

**#203 non-regression with the term present.** The attention term is K-invariant
(`· batch`, never `· k_eff`), so it adds a single constant to every K candidate and
cannot reintroduce the ×K over-count. A 24 GiB card still sizes successfully with the
term present (the existing `test_decide_preset_24gib_sizes_successfully` runs at
`cc=(8,0)` → flash regime → no term; and even a forced no-flash 24 GiB card adds only
≈1.6 GiB · batch, which leaves QLoRA r=4 batch=1 K=1 ≈ 5.4 GiB ≪ 22.5 GiB budget).

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
Stage 1 — DERIVE (two cheap probes, both fit a 16 GiB card)
  probe @ (qlora, r=4, batch=1, K=1)  -> peak_K1   (~3.05 GiB measured)
  probe @ (qlora, r=4, batch=1, K=4)  -> peak_K4   (~6.54 GiB measured)
  A_per_class = (peak_K4 - peak_K1) / (4 - 1)
  A_fixed     = clamp(peak_K1 - overhead_to_subtract - A_per_class, min=0)
                (overhead_to_subtract = STATIC + (attn(1) if not flash else 0);
                 STATIC = model + adapter + optimizer + workspace — the SAME
                 regime-matched overhead the predictor adds, §2.1. On the cc=12.0
                 dev box flash=True -> subtract STATIC only.)
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

Both fit a 16 GiB card (≈3.05 / ≈6.54 GiB measured). Solve by subtracting the **same
regime-matched overhead the predictor adds** (§2.1), reading the live card's compute
capability to decide the flash regime:

```text
A_per_class = (peak_K4 - peak_K1) / (4 - 1)
cc          = torch.cuda.get_device_capability(0)
flash       = _flash_attention_available(cc)            # cc >= (8, 0)
STATIC      = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4)
              + WORKSPACE_BYTES
overhead    = STATIC + (_attention_bytes_per_example(SAM3_IMAGE_SIZE) if not flash else 0)
A_fixed     = clamp(peak_K1 - overhead - A_per_class, min=0)
```

On the cc=12.0 dev box `flash=True`, so `overhead = STATIC` (no attention subtracted)
and the derived seeds are the portable flash-baseline. On a cc<8.0 card `flash=False`,
so the materialized attention is subtracted off the (math-backend) measured peak,
normalizing `A_fixed` to the **same** flash-baseline quantity — seeds derived on
either regime are interchangeable (§2.1). Clamp `A_fixed` and `A_per_class` to `>= 0`.
On the dev-env 5070 Ti `A_fixed`'s pre-clamp residual is negative (≈ −0.76 GiB) and
clamps to 0 — this is the **expected, cited** outcome (§2.1 / §6), not an error: the
K-invariant encoder activation is below the conservative model-weight margin in
`STATIC`. Warn only on a **negative `A_per_class`** (mirroring today's warning at
`calibrate_cmd.py:188`),
which would signal a genuinely broken differential. A clamped-to-zero `A_fixed` must
**not** block the derivation or the cache write; emit at most an informational note.

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
- Compute `A_per_class = (peak_K4 - peak_K1) / 3` and `A_fixed = clamp(peak_K1 -
  overhead - A_per_class, min=0)` at SAM 3.1's native 1008px — no scaling conversion
  (the prior `* (1024 / image_size) ** 2` artifact is removed). `overhead` is the
  **regime-matched** quantity `STATIC + (attn(1) if not flash else 0)` (Amendment 2,
  §2.1), where `STATIC = _model_bytes + _adapter_bytes + _optimizer_bytes +
  WORKSPACE_BYTES`, `flash = _flash_attention_available(cc)` from the live card's
  `torch.cuda.get_device_capability(0)`, and `attn(1) =
  _attention_bytes_per_example(SAM3_IMAGE_SIZE)`. This is the **same** regime-matched
  overhead the predictor adds, so the printed seeds reproduce the measured peak and
  are a portable flash-baseline (on the cc=12.0 dev box `flash=True` → subtract STATIC
  only).
- Print copy-paste-ready `A_FIXED = ...` and `A_PER_CLASS = ...` lines. A
  clamped-to-zero `A_FIXED` is a valid, expected result on the dev GPU (the residual
  is negative because the encoder activation sits under the model-weight
  conservatism margin); the script must print it without erroring.
- Update its module docstring (it references `BASE_ACTIVATION_AT_1024` and "design
  §3.3").

### Land the constants

Run the script **once** on the reference GPU — the dev-env **RTX 5070 Ti (16 GiB,
sm_120, driver 610.47)** — and hardcode the result as `A_FIXED` / `A_PER_CLASS` in
`presets.py` with a rigorous citation comment containing: GPU name + compute
capability, commit SHA, date, and the exact derivation command (`uv run python
scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1`). The split is
model-driven — activation bytes per example depend on the model and dtype, not the
card — and because the derivation subtracts the regime-matched overhead (Amendment 2,
§2.1), the constants are a **portable flash-baseline** that applies to all GPUs: the
predictor re-adds the materialized attention only on cc < 8.0 cards. **No
re-derivation on a Pascal/Turing card is required.**

**If the implementing session lacks GPU access:** land the constants as
`# tbd:`-tagged seeds — a documented, conservative split of the current
`BASE_ACTIVATION_AT_1024` evaluated at 1008px (~1.45 GiB): attribute the bulk to
`A_FIXED` since the encoder dominates, a small slice to `A_PER_CLASS` — with the
exact derivation recipe recorded in the comment. A conservative split slightly
over-attributes to `A_FIXED`, which can only *raise* the predicted floor, keeping
sizing safe. **Never ship a silent guess.** The `# tbd:` tag is the trigger for
re-derivation on the 5070 Ti before merge — which the §9 real-GPU gate enforces.

> **Amendment 2 (2026-05-31, supersedes the prior note).** The §9 gate was run on the
> 5070 Ti (cc 12.0, **flash** regime). The **measured** outcome is `A_FIXED = 0`
> (clamped residual, `overhead = STATIC` since flash → no attention subtracted) and
> `A_PER_CLASS = 1,248,840,021 B` (≈1.163 GiB). These are the **portable
> flash-baseline** seeds. `A_FIXED = 0` is landed as a **cited measured result**
> (GPU+cc / commit SHA / date / command), noting the K-invariant encoder activation is
> below the model-weight conservatism margin in `STATIC` (§2.1). It is **not** a
> `# tbd:` guess. The safety inequalities (§2.1) hold at both probe points on a flash
> card (A: 3.81 ≥ 3.05; 7.30 ≥ 6.54 GiB) and on a synthetic no-flash card with the
> re-added attention term (B: 5.41 ≥ 4.65; 8.90 ≥ 8.14 GiB). The seeds are valid for
> all cards — **no Pascal/Turing re-derivation needed.**

---

## §7 Out of scope / unchanged

- **Runtime OOM ladder** (`loop.py`, ~222–386): untouched. It stays the safety
  net for data-dependent class counts that exceed the sized K (batch inner rung →
  K outer rung → hard fail).
- **`decide_eval_batch_size`** (`presets.py:401`; consumed by
  `train/trainer.py:304,538`): keeps its **unconditional** SDPA attention ceiling and
  its own batch loop. It *consumes* the new split — `eval activation = (A_fixed +
  A_per_class * K) * forward_only_factor` — by threading its `classes_per_forward`
  arg into `_predicted_bytes(..., mode="eval", k_eff=classes_per_forward,
  flash_available=_flash_attention_available(cc))` (Amendment 2: it now also passes
  the regime flag so the eval predictor adds the materialized attention on no-flash
  cards) and into the `_act_per_example` term used for the cap. The SDPA *ceiling*
  itself stays regime-independent (always-on guard, §2.1). No eval behavioral
  regression intended; verify the cap still only lowers `best_bs`.
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
| | `_activation_per_example` / `_activation_bytes` (188–201) | Replace with split formula `(A_fixed + A_per_class*K)*batch` (no scale term); read split from cache. |
| | `_flash_attention_available` (NEW, near ~190) | **Add** (Amendment 2): `cc >= (8, 0)`; `None`/unreadable cc → `False` (conservative). Keyed on `torch.cuda.get_device_capability`. |
| | `_attention_bytes_per_example` (190) | **Retained**; called CONDITIONALLY by `_predicted_bytes` (only when `not flash_available`, cc < 8.0) AND by the `decide_eval_batch_size` SDPA cap (§2.1, Amendment 2). |
| | `_predicted_bytes` (218) | Gains `flash_available: bool = True`. train branch = `STATIC + split + (attn(batch) if not flash_available else 0)`; eval branch = `model + split·ff + workspace + (attn(batch) if not flash_available else 0)`; both thread `k_eff`. Attention term is **cc-aware & K-invariant** (Amendment 2, supersedes Amendment 1's universal drop). |
| | `_load_cache` (248) | reads `A_fixed`/`A_per_class` keys (schema v3); stale-version drop already handled. |
| | `_candidates` (304) | add `Ks=(1,2,4,8,16)`; return `(method, r, batch, K)`. |
| | `_sort_key` (311) | key `(lora?, -r, -K, -batch)` over 4-tuple. |
| | `decide_preset` (323) | `k` = upper bound on K search; iterate 4-tuples; pass `flash_available=_flash_attention_available(cc)` into `_predicted_bytes` (it already reads `cc`, §line 380); raise message recomputed via split; return `classes_per_forward`. |
| | `decide_eval_batch_size` (401) | thread `classes_per_forward` into the split via `_predicted_bytes(mode="eval", k_eff=..., flash_available=_flash_attention_available(cc))`; SDPA cap stays unconditional (§2.1, Amendment 2). |
| `src/custom_sam_peft/cli/calibrate_cmd.py` | `_run_probe` (65) | reused as-is for the multi-probe stages (no signature change required). |
| | `run_calibration` (new) | stages 1–3 core; cache write (v3); config rewrite; returns `PresetDecision`. |
| | `calibrate` (129) | thin wrapper over `run_calibration`; preserve exit codes; new "GPU too small" only on K=1-probe OOM. |
| | `_apply_config_rewrite` (98) | persist `classes_per_forward` via the rewrite helper. |
| | cache payload (199) | add `A_fixed`/`A_per_class`; remove `activation_bytes_per_example`. |
| `src/custom_sam_peft/cli/setup_wizard.py` | `_ask_peft_sizing` (338) | consent → `run_calibration(...)`; graceful fallback to analytic `decide_preset(k=...)` then manual. |
| `src/custom_sam_peft/cli/init_cmd.py` | `run_init` (~178) | `decide_preset(k=cfg…classes_per_forward)`; rewrite must persist chosen `classes_per_forward`. |
| `src/custom_sam_peft/cli/run_cmd.py` | `_fallback_preset` (48–50) | pass `k=cfg.train.multiplex.classes_per_forward`. |
| `src/custom_sam_peft/cli/_config_rewrite.py` | `_rewrite_sizing_block` | extend to write `train.multiplex.classes_per_forward` (verify current signature; add a `classes_per_forward` arg threaded from all callers). |
| `src/custom_sam_peft/cli/calibrate_cmd.py` | `_derive_split` | subtract **regime-matched** overhead `STATIC + (attn(1) if not flash else 0)` (Amendment 2); clamp `A_fixed` ≥ 0 silently. |
| `scripts/_derive_preset_constants.py` | `main` | emit two-point split using regime-matched overhead (Amendment 2); update docstring. |
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
- **cc-aware attention term (Amendment 2).** Two regime tests: with
  `_predicted_bytes(..., flash_available=True)` the train value carries **no**
  attention term (equals `STATIC + split`); with `flash_available=False` it equals
  `STATIC + split + _attention_bytes_per_example(1008)·batch`. Drive the regime
  through `decide_preset` by monkeypatching `torch.cuda.get_device_capability` to
  `(12, 0)` (no term) and `(7, 5)` (term present). Under cc<8.0 the train predictor
  **grows with image_size** via attention; under cc≥8.0 it does not. The term is
  K-invariant: K=1 vs K=16 still differ by exactly `15·A_per_class·batch` in BOTH
  regimes.
- A 24 GiB card now sizes successfully (no `RuntimeError`): add a tier test at
  `_stub_gpu(int(24 * _GB))` asserting a `PresetDecision` is returned (replacing
  the false-rejection behavior). Confirm it sizes in the flash regime
  (`_stub_gpu` default `cc=(8,0)`) AND when forced to `cc=(7,5)` (no-flash term
  present).
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
   the two cheap probes (K=1, K=4) both fit (no OOM). A clamped `A_FIXED = 0` is the
   expected, valid result on this card (§2.1) — the derive script prints it without
   erroring and it lands as a cited measured value, not a `# tbd:` guess.
2. **`calibrate` end-to-end** — `uv run custom-sam-peft calibrate` on a real
   config: assert it writes a v3 cache, the confirm-and-climb walk terminates,
   and the chosen config's measured peak is ≤ the 16 GiB budget (no unguarded OOM
   crash; a *caught* OOM during the climb is expected and fine).
3. **Wizard auto-calibrate path** — drive `_ask_peft_sizing` consent on GPU
   (scripted / non-interactive) and confirm it returns a fitting `PresetDecision`
   via `run_calibration`, with graceful fallback when the checkpoint is absent.
4. **Sanity vs. reality** — train a few steps at the chosen config and confirm it
   does not OOM at runtime, closing the loop the analytic-only ship failed to.
5. **cc<8.0 prediction path exercised (Amendment 2).** No Pascal/Turing card is on
   hand, so the no-flash branch is exercised by the monkeypatched unit test (cc set
   to `(7, 5)` → attention term present; §9 split-model tests), not on real silicon.
   The gate confirms this test passes; the seeds are a portable flash-baseline, so a
   real cc<8.0 run is a follow-up validation (§10 Q4), not a merge blocker.

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
4. **Follow-up — validate the flash-baseline on a real cc<8.0 card (Amendment 2).**
   The portable-seed claim rests on two assumptions worth confirming on real Pascal
   (GTX 1080, cc 6.1) or Turing (T4, cc 7.5) silicon if one becomes available: (a)
   that SDPA actually falls back to the math backend there (so the re-added
   materialized term is real, not over-conservative), and (b) that `A_PER_CLASS` (the
   per-class decoder term over a few class queries) is small and backend-robust
   relative to the encoder self-attention captured by the conditional `attn` term, so
   reusing the flash-derived `A_PER_CLASS` across regimes holds. Until then the
   monkeypatched cc<8.0 unit test (§9) is the guard, and the conservative default
   (unknown cc → no flash) keeps any mis-detection on the safe over-estimate side.
   Not a merge blocker.
