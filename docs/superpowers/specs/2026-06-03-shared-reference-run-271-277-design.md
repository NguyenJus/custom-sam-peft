# Shared reference run feeding split #271 and #277 harvests

- **Date:** 2026-06-03
- **Issues:** #271, #277 (remain SEPARATE issues and SEPARATE PRs — not merged)
- **Status:** design / awaiting approval

## Summary

#271 and #277 each need data that only a real-data training run can produce, but
they need it from the **same** run. The single expensive prerequisite is **one
reference training run** that banks checkpoints across the cold → steeply-climbing
mAP trajectory and logs per-eval `mAP` / `mAP_50`. Both downstream harvests are
cheap offline analysis on that run's artifacts:

- **#271** reads the metrics log to find the `mAP_50` wake-up step and refine the
  `early_stop.warmup_floor_steps` default (currently `1000`).
- **#277** sweeps the banked checkpoints through lite eval twice (exact vs proxy)
  to clear the spike §8.2 pre-enablement validation gate for the dense-IoU proxy.

This spec defines the shared run (Phase 0) and the two independent harvests
(Phase A = #271, Phase B = #277). **The issues stay split**; each gets a note
pointing at the shared run's artifacts. No GitHub issue merge.

## Motivation & shared-prerequisite rationale

Both follow-ups were deferred for the *identical* reason: **no non-zero-mAP
checkpoint existed**, and the only way to produce one is a multi-hour real-data
GPU run (crash-risky on the 16 GB sm_120 box). That run is the sole costly step.

- #271 (`gh issue view 271`): "Run one clean … reference run on real data … Record
  the optimizer step where `mAP_50` first goes non-zero." Pure log-reading once
  the run exists.
- #277 (`gh issue view 277`): the §8.2 gate "Requires a non-zero-mAP checkpoint
  (none existed at implementation time) and heavy GPU evals." Pure offline sweep
  once checkpoints exist.

Running two separate 160-epoch reference runs to feed two cheap analyses is
wasteful. **One run amortizes both.** The issues stay split because their
deliverables are unrelated (a default-value + citation change vs. a validation
harness + gate decision) and touch disjoint code; merging them would couple two
independently reviewable PRs to one branch with no benefit.

## Phase 0 — shared reference run

> **HARD GATE — explicit user go-ahead required before launch.** This run is
> multi-hour and crash-risky on the 16 GB box (see Risks). It is **not launched
> without explicit user confirmation.** Per project policy, GPU runs are always
> asked-first. The spec defines the run; it does not authorize starting it.

### Run configuration

- **Real data**, **poly LR schedule** — the current default
  (`config/schema.py:646`, `lr_schedule: LRSchedule = "poly"`; per #264 which
  decoupled LR and removed the plateau scheduler).
- **In-loop validation via the landed fast dense-IoU proxy** — default behavior
  after #276 (`eval/proxy_map.py`; lite⇒proxy / full⇒exact split at
  `eval/evaluator.py:958`). The proxy drives the in-loop `mAP` signal; this is
  fine for harvesting because Phase B reconstructs the **exact** values offline.
- **Log `mAP` and `mAP_50` per eval** so Phase A can read the `mAP_50` wake-up
  step and the headline-`mAP` climb-onset step from the run's own metrics log.

### Checkpoint cadence

Checkpoint densely enough to bank **>= 8 checkpoints in the NON-ZERO-mAP region**
(not just the cold dead zone). The dead zone yields tied zeros where rank
correlation is degenerate (spike §4.3), so the eight well-spread checkpoints the
gate needs must land **after** `mAP_50` first wakes. Cadence is a tunable knob
(see Open questions).

### Harvest-gated early stop (the runtime-reduction decision)

Stop the run as soon as **BOTH** hold:

1. **Quality observed:** the `mAP_50` wake-up step has been seen **AND** headline
   `mAP@[0.5:0.95]` has climbed for **N consecutive evals** (suggested **N = 3**,
   a tunable knob — not a hard requirement).
2. **Coverage banked:** **>= 8 well-spread non-zero-mAP checkpoints** are saved.

Expected truncation: **~20–40 epochs vs. the full ~160** (#264/#271 estimate:
`mAP_50` non-zero within ~1–5 epochs / ~200–1000 steps; headline mAP climbs over
~5–20 epochs — wide error bars, hence the measurement). This early stop is a
manual harvest decision, independent of the trained-in `early_stop` mechanism.

### Scoping note (cite explicitly — do not read "cold → converged" literally)

Truncating means #277's gate is validated over a **cold → steeply-climbing** span
rather than **cold → fully-converged**. This is defensible: the proxy's ranking
faithfulness is an **analytic, scale-invariant property of the AP rules** (spike
§2, §4), not an empirical property of any particular convergence stage. The
Spearman gate needs a **representative quality spread** across distinguishable
checkpoints, not the asymptotic best model. The spike's §8.2 step 1 phrases the
span as "cold → converged"; this spec deliberately relaxes it to
"cold → steeply-climbing" and records the relaxation so a reviewer does not read
the literal phrase as a requirement.

## Phase A — #271 harvest (cheap, no GPU; independent of Phase B)

From the reference run's **own metrics log** (no re-eval needed):

1. Read the optimizer step where `mAP_50` first goes non-zero (the wake-up step).
2. Read the step where headline `mAP@[0.5:0.95]` begins climbing.
3. If the measured wake-up step differs **materially** from the current default
   `1000`, update the default and its citation:
   - `config/schema.py:634` — `warmup_floor_steps: int = Field(default=1000, ge=0)`
     and its citation comment (`config/schema.py:635`, currently
     `# cite: Detectron2 SOLVER.WARMUP_ITERS`).
   - `docs/defaults-provenance.md:96` — the `warmup_floor_steps` provenance row.
   - Per project rule (cite new/changed defaults): any change carries a citation
     or an explicit `# tbd:` tag; a measured value cites the reference run.

**Note on safety (from #271):** `warmup_floor_steps` is only a **backstop**; the
load-bearing cold-start guard is the adaptive baseline (the no-improvement
counter does not accrue until mAP first clears `0.0` — `train/ladder.py:69`,
`grace_lifted = self.woken and step >= warmup_floor_steps`). `0` disables the
backstop. So a "no material difference → no change" outcome is fully acceptable.

**Deliverable:** the #271 PR (default + citation update, or a documented
no-change finding).

## Phase B — #277 harvest (offline sweep; the only substantial new code)

~16 lite evals total = 2 modes × ~8 checkpoints. The new code is a
**checkpoint-sweep harness/script**.

### Sweep

For each banked checkpoint, run lite eval **TWICE on the SAME lite val subset**:

- **exact:** `CSP_LITE_EXACT_MAP=1` set (forces lite → exact pycocotools mAP via
  `_lite_exact_map_hatch`, `eval/evaluator.py:45-52`; consumed at
  `eval/evaluator.py:958`).
- **proxy:** env var unset (the default lite proxy path,
  `eval/evaluator.py:960-964`).

**Same subset both times is essential** — a different subset breaks the paired
rank comparison. Fix the lite subset (`eval.lite_max_images`,
`config/schema.py:720`, default `64`) and the seed across both passes.

### Gate (spike §8.2 step 3)

Compute **Spearman rank-correlation** across the sweep, **restricted to
checkpoints with non-zero EXACT mAP** (the §4.3 dead zone yields tied zeros where
rank-corr is degenerate). **GATE:**

- **ρ >= 0.95**, AND
- **no adjacent-checkpoint inversion within `min_delta`** (an inversion smaller
  than the control threshold the consumers actually use).

### `min_delta` scale check (spike §8.2 step 4 / §7b — GATING)

Measure the proxy's **absolute scale and dynamic range** against exact COCO mAP.
If `min_delta=0.001` (`config/schema.py:632`, `threshold_mode="abs"`) maps to a
**materially different fraction** of the proxy's dynamic range than of exact
mAP's, **recalibrate the default before trusting the proxy as a control input**.
Do **not** ship a silently rescaled plateau/early-stop sensitivity. Cite or
`# tbd:`-tag any change.

> **Current-code note (spike predates #264's LR decoupling).** The spike (§1,
> §7b, §8.2 step 4) describes `min_delta` as feeding BOTH `ReduceLROnPlateau`
> **and** the early-stop test. In the current tree the plateau scheduler has
> been **removed** (#264); `min_delta` now feeds **only** the early-stop
> improvement test (`train/ladder.py:71-72`, `improved = mAP > self.best +
> min_delta`) plus best-checkpoint selection (which is a **strict `>`** with no
> `min_delta`, `train/trainer.py:397`). The scale check still gates — it just
> guards one live consumer (early-stop), not two. State this so the reviewer is
> not surprised the plateau path is gone.

### Re-profile + memory (spike §8.2 step 5)

- Re-profile the proxy path **on real data** to confirm the §5 estimate
  (~3.4×–4.0× lite-validation speedup; headline conservative ~3.4×).
- Capture a **real-checkpoint memory profile** to confirm §6 (matmul path fits
  16 GB with large headroom; worst-case ~328 MB).

### Fallback

Until this gate passes, the `CSP_LITE_EXACT_MAP=1` escape hatch
(`eval/evaluator.py:45-52`, `958`) remains the production fallback.

**Deliverable:** the #277 PR (sweep harness + gate results + any recalibration,
or a documented pass).

## Scoping decision & risks

- **Truncated-span justification** (restated for the reviewer): the gate is run
  over cold → steeply-climbing, not cold → converged. Faithfulness is an analytic
  property of the AP rules (spike §2, §4); a representative quality spread is what
  the Spearman gate needs, not the asymptotic best model.
- **16 GB OOM risk.** Heavy GPU evals and a multi-hour run on the sm_120 box are
  crash-risky on 16 GB (both #277's body and the spike §4.4 flag this). On this
  box CUDA OOM can surface as a generic "device not ready" rather than
  `OutOfMemoryError`. The harness should keep the lite subset small
  (`lite_max_images=64`) and run checkpoints sequentially.
- **Session-crash risk on long runs.** Long sessions on this WSL box risk
  crashes (Bun segfault / disk-I/O saturation, not RAM/VRAM OOM). Keep the run
  monitorable and resumable; bank checkpoints frequently so a crash loses minimal
  progress.
- **Wasted-run risk.** If checkpoint cadence is too coarse, fewer than 8 non-zero
  checkpoints land and Phase B cannot run — cadence is the lever (Open questions).

## Deliverables & PR mapping

| Phase | Issue | Deliverable | GPU? |
| --- | --- | --- | --- |
| 0 | (shared) | One reference run: banked checkpoints + per-eval `mAP`/`mAP_50` log | Yes — **gated on user go-ahead** |
| A | #271 | PR: `warmup_floor_steps` default + citation update (or no-change finding) | No |
| B | #277 | PR: checkpoint-sweep harness + §8.2 gate results (ρ, scale check, profile/memory) | Yes (offline evals) |

Critical path: **Phase 0 is the sole blocking/expensive step and gates BOTH
harvests.** Phases A and B are independent of each other once checkpoints +
metrics exist. Each issue gets a note pointing at the shared run's artifacts.

## Open questions / tunables

- **N (consecutive-climb evals for harvest stop):** suggested `3`; tune against
  the observed `mAP` curve's noise.
- **Checkpoint cadence:** must bank >= 8 non-zero-mAP checkpoints in the
  steeply-climbing region; set after the `mAP_50` wake-up step is observed, or
  conservatively dense from the start.
- **`lite_max_images`:** default `64` (`config/schema.py:720`); same value for
  both sweep passes. Smaller reduces OOM risk but coarsens the lite signal.
- **"Materially different" thresholds:** the numeric bar for "material" wake-up
  shift (Phase A) and "material" scale-fraction divergence (Phase B) — set during
  harvest from the measured values.
