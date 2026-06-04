# Pin LoRA `method`/`r`/`alpha`; autosize only batch and K

**Issue:** to be filed (`gh issue create --assignee @me --label …`) — "VRAM
autosize raises LoRA rank above the configured value (`r=64, alpha=32`),
violating the locked `alpha=2r` invariant."
**Release:** pre-1.0 — behavior/contract change to `decide_preset` and a cache
schema bump → MINOR.
**Status:** locked design, single PR, no back-compat shims.

The autosizer is supposed to fill the card with *memory-only* (`batch`) and
*throughput-only* (`classes_per_forward`, "K") levers while leaving the
*accuracy* levers (`method`, `r`, `alpha`) exactly as the template or config
pinned them. It does not. `decide_preset` maximizes LoRA rank: on a 16 GB card
it emits `r=64`, and because `PresetDecision.alpha` is a hardcoded field default
of `32` it ships the impossible pair `r=64, alpha=32` — silently breaking the
`alpha=2r` invariant that commit `2453421` (#242) locked into
`docs/defaults-provenance.md:84,85,239`. The original #179/#148 design already
specified "probe at `cfg.peft.r`, not fixed `r=4`"
(`docs/superpowers/specs/2026-05-28-vram-calibration-reassess-design.md:204`)
and "no dynamic LoRA-rank downgrade in the ladder … degrade B then K"
(`…:49`). This spec **enforces that documented-but-unenforced contract** at the
maximization source.

---

## §1 Scope & non-goals

### In scope

| File | Change |
|------|--------|
| `src/custom_sam_peft/presets.py` | `decide_preset` contract change: `method`/`r`/`alpha` become **pinned inputs** (default to the cited `PEFTConfig` defaults `lora`/`r=16`/`alpha=32`). Drop the `-r` dimension from `_candidates`/`_sort_key` (search only `b`/`k`). `PresetDecision.alpha` is set from the pinned alpha (co-scaled only on r-reduction), never stranded at the field default. Bump `CACHE_SCHEMA_VERSION` `3→4`. |
| `src/custom_sam_peft/cli/init_cmd.py` (~181-206) | Pin the template's `method`/`r`/`alpha` into the rewritten sizing block; autosize only `b`/`k`. Pass `num_classes` (best-effort) to the sizer. |
| `src/custom_sam_peft/cli/setup_wizard.py` (~354-369) | Same: the analytic `decide_preset` path pins the chosen `method`/`r`/`alpha`; autosize only `b`/`k`. |
| `src/custom_sam_peft/cli/calibrate_cmd.py` (~419-546) | Stage-2 aim pins `cfg.peft.method`/`cfg.peft.r`/`cfg.peft.alpha`; Stage-3 `_confirm_and_climb` climbs `b` then runs the shared shrink ladder. The existing reduction-only alpha co-scale guard (~491-504) is kept (now provably correct because `r` never climbs). Update the `2*r` fallback in `_decision_from_cache` (~316). |
| `src/custom_sam_peft/cli/run_cmd.py` (~60-65) | No logic change beyond the new `decide_preset` signature: the provenance label now reflects the pinned `r` for free. |
| `src/custom_sam_peft/data/` (`base.py`, `hf.py`, `mask_png.py`, `_semantic_encode.py`) | New best-effort `num_classes` helper off `class_names` (§5). Read-only; no behavioral change to data loading. |
| `docs/defaults-provenance.md` (rows 84, 85, 239) | Update the `PresetDecision.alpha` row wording (§7) so the CI no-uncited-default gate stays green. |
| Tests (§8) | New/updated CPU unit coverage for the pin invariant, alpha-never-stranded, ladder order, num_classes, init/wizard, and cache schema. |

### Non-goals

- **No new accuracy knobs and no auto-raising of `r`.** Hardware never raises
  `method`/`r`/`alpha`. The only `r` movement is the warned reduction fallback.
- **No change to the train-loop OOM ladder** (`train/loop.py` B→K halving from
  the #148 spec). This fix is upstream, at config-generation/calibration time.
- **No change to the memory model** (`_predicted_bytes`, split-activation
  constants, attention term, eval batch sizer). The `r` arg to `_predicted_bytes`
  stays; only the *search dimension* over `r` is removed.
- **No GPU-keyed lookup table** (already dropped in the #148 spec §10).
- **No back-compat shim for v3 caches.** The schema bump auto-invalidates them
  (§7); they re-probe under the new logic.
- Anything else surfaced during implementation → file a follow-up GitHub issue;
  do not widen this spec.

---

## §2 Root cause

`decide_preset` (`presets.py:358`) maximizes rank along three coupled axes:

1. `_candidates()` (`presets.py:337-342`) enumerates the full product including
   `rs = (8, 16, 24, 32, 48, 64)`.
2. `_sort_key` (`presets.py:345-352`) sorts feasible candidates by
   `(method, -r, -k, -batch)` — highest `r` wins after method.
3. `feasible.sort(...); … = feasible[0]` (`presets.py:425-426`) takes the
   highest-rank candidate that fits the VRAM budget.

So `aim.r` = **max-fitting r** (= 64 on a 16 GB card). This rank-maximization
leaks into four config-writing/labeling consumers:

- **calibrate** Stage-2 aim — `aim = decide_preset(...)` (`calibrate_cmd.py:474`)
  seeds `_confirm_and_climb` with `aim.r=64`.
- **init** — `decide_preset(...).r` is rewritten straight into `config.yaml`
  (`init_cmd.py:186-194`).
- **wizard** — `decide_preset(...).config_patch` is returned as the chosen
  sizing (`setup_wizard.py:364-369`).
- **run** provenance label — `_fallback_preset` → `decide_preset(...)`
  (`run_cmd.py:65`) renders `r=64` into the bundle label.

Separately, **alpha is decoupled from r**: `PresetDecision.alpha` is a hardcoded
field default `= 32` (`presets.py:96`), never recomputed as `2r`. Calibrate's
co-scale guard fires **only on reduction** (`if r < cfg_r:`,
`calibrate_cmd.py:491`); on an upward climb the `else` branch
(`calibrate_cmd.py:499-504`) strands `alpha` at the configured `32`. Net effect:
`r=64, alpha=32` — `alpha:r = 0.5` instead of the locked `2.0`.

**Provenance evidence.** Commit `2453421` (#242) documented the
reduction-only co-scale contract and locked `alpha=2r` into
`docs/defaults-provenance.md:84` (`PEFTConfig.r`), `:85` (`PEFTConfig.alpha`),
and `:239` (`PresetDecision.alpha` — "co-scales it alongside `r` on VRAM-driven
rank reduction"). But the same commit never fixed the **upstream
maximization** in `decide_preset`, so the documented invariant was never
enforced. The #148 reassess spec already stated the intended contract twice
(`2026-05-28-…:49`, `:204`); this spec makes it real.

---

## §3 Principle (state up front)

`method`, `r`, and `alpha` are **pinned accuracy choices** — from the template
(`init`/wizard) or from the config (`calibrate`). **Hardware never raises
them.** Only:

- `b` (batch) — *memory-only*: grows to fill the card.
- `k` (`classes_per_forward`) — *throughput-only*: capped, sized down under
  pressure.

are auto-sized. `r` is **reduced only as a warned last resort**, and when it is,
`alpha` is co-scaled to preserve the `alpha:r` ratio. The sacrifice order, from
cheapest to most damaging, is:

```text
b↓  →  k↓ (b held at 1)  →  lora→qlora @ same r  →  r↓ (+WARNING, alpha co-scaled)
```

`r` is the **last** lever touched; `b` is **not** re-grown while `k` shrinks.

---

## §4 The shared sizing ladder

A single ladder, parameterized by a `fits?(method, r, b, k) -> bool` oracle, so
the analytic and empirical paths walk identical logic. Inputs: pinned
`(method, r, alpha)`, `cfg_r`/`cfg_alpha` snapshots, `k_cap`, `num_classes`.

### 4.1 Ladder algorithm

```text
k_start = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP=16, num_classes)

# 1. At pinned (method, r): largest b in [1..16] that fits at k_start.
for b in 16..1:
    if fits?(method, r, b, k_start): choose (method, r, b, k_start); DONE

# 2. b=1 doesn't fit at k_start: step k DOWN the grid (16,8,4,2,1), b held at 1.
for k in <grid values <= k_start, descending>:
    if fits?(method, r, 1, k): choose (method, r, 1, k); DONE   # do NOT re-grow b

# 3. b=1, k=1 still doesn't fit: flip lora -> qlora at the SAME r (NF4 base is far
#    cheaper and PRESERVES rank), retry the b/k search (steps 1-2) at qlora.
if method == "lora":
    method = "qlora"; restart steps 1-2

# 4. Still infeasible: reduce r DOWN the grid (… 32,24,16,8) WITH A USER WARNING,
#    co-scaling alpha = max(1, round(cfg_alpha * r / cfg_r)); retry b/k at the new r.
for r_next in <grid values < r, descending>:
    r = r_next; alpha = max(1, round(cfg_alpha * r / cfg_r)); WARN; restart steps 1-2

# 5. Exhausted (qlora, smallest r, b=1, k=1) still infeasible -> raise GpuTooSmall.
```

### 4.2 Locked forks (do NOT reorder)

- **qlora-flip happens BEFORE r-reduction.** Quantizing the base is far cheaper
  per byte and preserves rank — sacrifice the base dtype before the rank.
- **`b` stays pinned at 1 while dropping `k`** in step 2. Do **not** re-maximize
  `b` at a lower `k` — that re-introduces memory pressure the k-drop just
  relieved.
- **`r` is the last lever**, only ever reduced (never raised), always with a
  warning and a co-scaled `alpha`.
- **calibrate's empirical strategy is analytic-aim + ±1 climb/shrink**, NOT a
  top-down `b` scan. The existing `_confirm_and_climb` (`calibrate_cmd.py:343`)
  probes the analytic aim, then steps one grid value per probe. This keeps the
  probe count bounded against the sm_120/WSL "device not ready" dirty-OOM
  accumulation (`_run_probe` teardown comment, `calibrate_cmd.py:144-159`; memory
  note "CUDA OOM = device not ready"). The bound stays `max_probes = len(_BATCHES)
  + len(_KS) + 2*len(_RS) + 2` (`calibrate_cmd.py:368`); the climb half walks K
  then B at the fitting `(method, r)` only and never raises `r` or flips method
  on a probe (`calibrate_cmd.py:402-416`).
- **`num_classes` IS auto-derived** (§5) and feeds `k_start` — it caps K from the
  dataset vocabulary, not just from `classes_per_forward`.

### 4.3 Two oracles over the identical ladder

- **ANALYTIC** (`init`, wizard, `decide_preset`): `fits?` = `_predicted_bytes(…)
  ≤ budget` (`presets.py:224`, `mode="train"`, the pinned `r`, candidate `b`/`k`,
  card `flash_available`). No GPU probe.
- **EMPIRICAL** (`calibrate`): `fits?` = `_run_probe(…) ≤ budget`
  (`calibrate_cmd.py:108`). Live forward+backward at the candidate.

Both are pure swaps of the `fits?` predicate over the same `(method, r, b, k)`
walk. `decide_preset`'s candidate loop (`presets.py:404-411`) collapses to
the analytic ladder: enumerate only `b × k` at the pinned `(method, r)`, with the
qlora-flip and r-reduction fallbacks appended in the locked order.

---

## §5 `num_classes` derivation

A best-effort helper off the dataset's `class_names` property
(`data/base.py:90-92`):

- **`mask_png`**: name-count from the `class_map` JSON via `build_value_to_label`
  (`data/_semantic_encode.py:57`; `mask_png.py:57-58`, `class_names` at
  `mask_png.py:182`). Cheap — JSON read, no model load.
- **`hf`**: `len(ClassLabel.names)` via `_resolve_class_names`
  (`data/hf.py:74-92`).

**Failure handling (must never hard-fail sizing).** On ANY failure — data absent
at `init` time, unresolved vocabulary, missing `class_map`, HF feature absent —
fall back to the configured `classes_per_forward` (i.e. `num_classes = k_cap`)
and emit a single WARNING. `k_start = min(cfg K, 16, num_classes)` then simply
reduces to `min(cfg K, 16)`, identical to today's behavior. The helper returns
`int | None`; the sizer treats `None` as "use `k_cap`".

This only ever *lowers* `k_start` (a card with 4 classes never probes K=16), so
it can only reduce memory/probe cost — never introduce a regression.

---

## §6 Per-call-site changes

### 6.1 `decide_preset` (`presets.py:358`) — contract change

- **Signature:** add pinned inputs with cited defaults:

  ```python
  def decide_preset(
      k: int | None = None,
      cache_path: Path | None = None,
      *,
      method: Literal["lora", "qlora"] = "lora",   # cite: PEFTConfig.method default
      r: int = 16,        # cite: PEFTConfig.r default (defaults-provenance.md:84)
      alpha: int = 32,    # cite: PEFTConfig.alpha default (defaults-provenance.md:85)
      num_classes: int | None = None,
  ) -> PresetDecision:
  ```

  Defaults reproduce the cited `PEFTConfig` defaults so a caller with no config
  in scope (e.g. the `run_cmd` fallback) still gets the shipped pinned values.
- **`_candidates`/`_sort_key`:** drop the `r` dimension. `_candidates` enumerates
  `(b, k)` only; `_sort_key` sorts by `(-k, -batch)` (method/r are fixed inputs).
  Keep the tail-to-head = sacrifice-order comment but scoped to b/k.
- **Ladder:** replace the single feasibility sweep with the §4.1 ladder —
  pinned `(method, r)` b/k search, then qlora-flip, then warned r-reduction with
  alpha co-scale. `GpuTooSmall`/`RuntimeError` on exhaustion (preserve the
  existing message shape, `presets.py:413-423`).
- **`PresetDecision.alpha`:** set from the pinned/co-scaled `alpha` — never the
  field default. See §7.
- **Blast radius (required-field risk).** Per the memory note "Required-field
  blast radius," grep ALL `decide_preset(` call sites and ALL `PresetDecision(`
  constructors before landing, and run the FULL suite. Verified call sites:
  - `decide_preset(`: `calibrate_cmd.py:460,474`; `setup_wizard.py:364`;
    `run_cmd.py:65`; `init_cmd.py:186`; plus `tests/unit/test_presets.py` (many)
    and `tests/unit/cli/test_run_cmd.py:62` (a `_fake_decide_preset(k=None,
    cache_path=None)` monkeypatch — must accept the new keyword-only args, e.g.
    `**_`).
  - `PresetDecision(`: `presets.py:429`; `calibrate_cmd.py:325,529`.

### 6.2 `init` (`init_cmd.py:181-206`) and wizard (`setup_wizard.py:354-369`)

- Pin the template/chosen `method`/`r`/`alpha` into the `decide_preset` call
  (read them from the just-loaded `cfg.peft` at `init_cmd.py:182`; from
  `ctx.answers` / template default in the wizard). Pass best-effort
  `num_classes` (§5).
- The rewritten sizing block (`_rewrite_sizing_block`, `init_cmd.py:187-196`) and
  the wizard's `config_patch` therefore carry the **template** `r`/`alpha` (16/32
  at the shipped default), differing from the template only in `b`/`k`.
- CPU-only fallback (`init_cmd.py:202-206`) is unchanged.

### 6.3 `calibrate` (`calibrate_cmd.py`)

- **Stage-2 aim** (`calibrate_cmd.py:474`): pass the pinned `cfg.peft.method`,
  `cfg.peft.r`, `cfg.peft.alpha` (already in scope as `method`/`r`/`cfg_alpha`,
  `calibrate_cmd.py:439-444`) into `decide_preset`. The aim's `r` is now the
  configured `r`, not 64.
- **Stage-3 `_confirm_and_climb`** (`calibrate_cmd.py:343`): unchanged in
  structure — it already climbs K-then-B at the fitting `(method, r)` and never
  raises `r` (`calibrate_cmd.py:402-416`); its shrink half already walks the full
  sacrifice order including the qlora-flip-before-r-reduction
  (`calibrate_cmd.py:384-400`). With the aim now pinned at `cfg.peft.r`, the climb
  never starts from 64.
- **Co-scale guard** (`calibrate_cmd.py:491-504`): kept verbatim. It is now
  provably a no-op on the common path (`r == cfg_r`) because `r` never climbs
  above `cfg_r`; it only ever fires on the warned r-reduction. The `else` branch's
  `r > cfg_r` comment is now dead (the analytic aim can no longer exceed `cfg_r`),
  but leaving the guard defensive is harmless — keep it, and update the comment to
  note `r > cfg_r` is unreachable post-pin.
- **`_decision_from_cache` `2*r` fallback** (`calibrate_cmd.py:316`): change
  `alpha = int(data.get("chosen_alpha", 2 * r))` so a cache without
  `chosen_alpha` reconstructs the pinned config alpha rather than `2*r`. Since v3
  caches are now stale (§7), the only `chosen_alpha`-less caches are the Stage-2
  placeholder write (`peak=0`, no `chosen_*`) which already returns `None`
  (`calibrate_cmd.py:312-313`) — so in practice this fallback is defensive. Set
  it to the loaded `cfg.peft.alpha` when available; keep `2*r` only as the
  last-ditch default and add a `# defensive` comment.

### 6.4 `run` (`run_cmd.py:65`)

`_fallback_preset` calls `decide_preset(k=…)`; with the pinned defaults
(`lora`/16/32) the synthesized provenance label now reads the correct `r`. No
logic change — falls out for free. Confirm the `test_run_cmd.py:62` monkeypatch
signature accommodates the new keyword-only args.

---

## §7 alpha + cache schema

### 7.1 alpha (never stranded)

- `PresetDecision.alpha` **must equal the pinned config/template alpha** so that
  `r == cfg.r ⇒ alpha == cfg.alpha` exactly. At the shipped default that is the
  valid `2r` pair `r=16, alpha=32`. The `r=64, alpha=32` signature becomes
  impossible.
- On the warned **r-reduction** path only, `alpha = max(1, round(cfg_alpha * r /
  cfg_r))` (the existing formula, `calibrate_cmd.py:492-493`), preserving the
  `alpha:r` ratio and clamping to the `PositiveInt` invariant.
- The field default `alpha: int = 32` (`presets.py:96`) stays as a *fallback
  only*; every code path that constructs a `PresetDecision` now passes an explicit
  `alpha`. Update the field comment to "set explicitly by every constructor;
  default is a defensive fallback."

### 7.2 cache schema bump `3 → 4`

- `CACHE_SCHEMA_VERSION = 4` (`presets.py:67`). Existing `r=64` v3 caches
  auto-invalidate via the existing version check (`presets.py:298-304`,
  `calibrate_cmd.py:90`) and re-probe under the pinned logic.
- The repo's current `.custom_sam_peft_calibration.json` (`chosen_r=64`) becomes
  stale on bump — expected; the next `calibrate` re-probes at the pinned `r`.
- v4 round-trips through `_write_cache_v3`/`_decision_from_cache` unchanged (the
  bump is purely an invalidation lever; no new keys). Rename is optional and out
  of scope; keep `_write_cache_v3` as-is.

### 7.3 provenance doc

- `docs/defaults-provenance.md:239` (`PresetDecision.alpha`): update the wording
  from "defaults to the schema `PEFTConfig.alpha` value" to reflect that alpha is
  now **pinned to the config/template alpha and co-scaled DOWN only on the warned
  r-reduction** — the field default is a defensive fallback, never the shipped
  value. Keep the `cross-link` cite to `PEFTConfig.alpha` (`:85`).
- Rows `:84`/`:85` (`PEFTConfig.r=16`, `alpha=32`) are unchanged — they remain the
  authoritative pinned values this spec now enforces.
- `presets.py:CACHE_SCHEMA_VERSION` row (`:240`) is `index-only` — update the
  integer mention `3` → `4` if the row names the value; the row is not
  trust-bearing.

---

## §8 Test plan (adversarial)

All CPU unit tests. CUDA, the VRAM budget, and `_predicted_bytes` are
monkeypatched (the existing `tests/unit/test_presets.py` fixtures already
monkeypatch `torch.cuda` device props / capability / memory — reuse them).
Each bullet names the assertion.

### 8.1 Pin invariant (headline) — property-style across budgets

For budgets sweeping tiny→huge (8 / 16 / 24 / 48 / 80 GB), parametrize
`decide_preset`/the analytic sizer at the default `cfg` (`r=16`):

- **`test_pin_never_raises_r`**: for EVERY budget, `decision.r <= 16`. The
  sizer NEVER emits `r > cfg.r` for ANY budget (the `r=64` outcome is impossible).
- **`test_pin_holds_r_when_config_fits`**: whenever the pinned config fits at all
  (any budget ≥ the `r=16, b=1, k=1` floor), `decision.r == 16`. Larger budgets
  do not bump `r`.

### 8.2 alpha never stranded

- **`test_alpha_equals_cfg_alpha_when_r_unchanged`**: whenever `decision.r ==
  cfg.r`, `decision.alpha == cfg.alpha` exactly (the `64/32` signature is
  impossible). Assert the default pair `r=16 ⇒ alpha=32` holds **end-to-end**
  through `init` (rewritten YAML), `calibrate` (`run_calibration` decision), and
  `decide_preset` (returned decision).

### 8.3 batch maximized, monotone

- **`test_b_grows_with_budget_r_alpha_method_fixed`**: a strictly larger budget
  yields `b` ≥ the smaller-budget `b`, while `r`/`alpha`/`method` are
  byte-identical across budgets.
- **`test_k_start_is_min_cfg_cap_numclasses`**: `decision.classes_per_forward`'s
  starting point `== min(cfg K, 16, num_classes)` (assert via a budget large
  enough that no k-down step fires).

### 8.4 Ladder order under shrinking budget

- **`test_sacrifice_sequence_exact`**: drive the budget down through each
  threshold and assert the exact sequence `b↓ → k↓ (b held at 1) → lora→qlora
  @same r → r↓ (+warning)`. Concretely: (i) at a budget where `b=1, k_start`
  fits, `b` shrinks first; (ii) at a tighter budget, `k` steps down the grid
  while `b` stays at 1 (assert `b == 1` and is NOT re-grown at the lower k);
  (iii) tighter still, `method` flips `lora→qlora` at the SAME `r` (assert
  `r` unchanged across the flip); (iv) only at the tightest budget does `r` drop.
- **`test_r_is_last_lever`**: across the descent, `r` is the LAST field to change
  and only ever decreases.

### 8.5 alpha co-scale on reduction

- **`test_alpha_coscaled_on_r_reduction`**: at a budget forcing r-reduction,
  `alpha == max(1, round(cfg_alpha * r_new / cfg_r))` and the `alpha:r` ratio
  matches `cfg_alpha/cfg_r` within rounding; assert a WARNING is emitted (capture
  stderr / `caplog`).
- **`test_no_warning_no_coscale_on_nonreduction`**: on ANY non-reduction path
  (b-down, k-down, qlora-flip), NO warning is emitted and `alpha == cfg_alpha`.

### 8.6 Regression (golden)

- **`test_regression_16gb_r16_stays_r16`**: the exact reported case — `cfg.r=16`,
  16 GB card, probe peak ~12.5 GB < budget → `decision.r == 16`,
  `decision.alpha == 32`, NOT `r=64`. Reference the stale cache's `chosen_r=64`
  as the pre-fix value being corrected.

### 8.7 num_classes

- **`test_numclasses_mask_png_class_map_count`**: a synthetic `class_map` JSON of
  N entries → helper returns N → `k_start == min(cfg K, 16, N)`.
- **`test_numclasses_hf_classlabel_names`**: a stub HF dataset with
  `ClassLabel.names` of length N → helper returns N.
- **`test_numclasses_fallback_warns`**: data absent / unresolved vocabulary →
  helper returns `None`, sizer falls back to `k_cap`, a WARNING is emitted, and
  sizing succeeds (never hard-fails).

### 8.8 init / wizard

- **`test_init_writes_template_r_on_big_card`**: `csp init` on an 80 GB card
  writes the template `r=16`/`alpha=32` into `config.yaml` (NOT 64); only `b`/`k`
  differ from the template defaults. Assert by parsing the rewritten YAML.
- **`test_wizard_patch_pins_r_alpha`**: the wizard's `config_patch` carries
  `peft.r == 16`, `peft.alpha == 32`.

### 8.9 Cache schema

- **`test_v3_cache_is_stale`**: a v3 cache with `chosen_r=64` is treated as stale
  (`_cache_is_fresh` / `_load_cache` reject it on the version mismatch) → re-probe
  path.
- **`test_v4_cache_roundtrips`**: a v4 cache written by `_write_cache_v3`
  round-trips through `_decision_from_cache` with the pinned `r`/`alpha`.

### 8.10 Provenance gate stays green

- **`test_defaults_provenance.py` passes** unchanged after the `:239` (and
  `:240` integer) wording update; `_provenance_check.py` finds a cite/cross-link
  for every changed default — no `# tbd:` regressions.

### 8.11 GPU (existing harness — requirement, not new tests)

- `scripts/run_gpu_tests.sh` real-probe coverage must still pass and must honor
  the pinned `r` on a real 5070 Ti card (the empirical ladder must not climb `r`).
  Per the memory note "GPU runs: ask first," do **not** author new GPU tests or
  kick off a real run as part of this spec — only note the requirement. The
  bare-`pytest tests/` full-GPU-suite freeze risk applies; use the script.

---

## §9 Constraints & conventions

- **Cite every new/changed default** (memory note "Cite new hyperparams"). The
  pinned `decide_preset` defaults (`method=lora`, `r=16`, `alpha=32`) carry inline
  `# cite:` pointing at the `PEFTConfig` rows. Authority for the pin contract:
  commit `2453421` (#242); `docs/defaults-provenance.md:84,85,239`; prior spec
  `2026-05-28-vram-calibration-reassess-design.md:49,204`. No silent guesses; any
  truly undecided value gets an explicit `# tbd: #<issue>`.
- **ruff `S101`**: no bare `assert` in `src/` (bandit). Narrow `num_classes`
  return types structurally (`if isinstance(...) and ...`), not via `assert`.
- **mypy is a SEPARATE CI gate** (`mypy src/custom_sam_peft`): full type args on
  any new signatures (`int | None`, `Literal[...]`). The new keyword-only
  `decide_preset` params and the `num_classes` helper need complete annotations.
- **`ruff format --check`** is a separate gate from `ruff check` — both must pass.
- **Eager-import constraint**: `__init__.py` eagerly imports the train chain;
  if any symbol moves, verify via `ruff`/`py_compile` that the package still
  imports.
- **Required-field blast radius** (§6.1): grep all `decide_preset(`/
  `PresetDecision(` call sites and run the FULL suite — the keyword-only-with-
  defaults design avoids breaking positional callers, but the test monkeypatches
  must accept the new args.
- **This spec markdown** must pass the repo's markdownlint gate (the PreToolUse
  lint-gate hook).

---

## §10 Rollback

Single PR, no migrations beyond the cache version bump. To roll back:

1. Revert the PR. `decide_preset` returns to rank-maximizing; `init`/wizard/
   calibrate/run revert to emitting the max-fitting `r`.
2. Set `CACHE_SCHEMA_VERSION` back to `3` (or leave at `4` — a `3↔4` mismatch
   only forces a harmless re-probe; no data loss, the cache is a derived
   artifact).
3. No schema/data migration: `config.yaml` keeps whatever `r`/`alpha` was last
   written; users on the pinned PR who want the old behavior re-run
   `init`/`calibrate` post-revert.

The only durable side effect is config files written with the pinned `r=16`
while the PR was live — those are valid configs under both behaviors and need no
fix-up.
