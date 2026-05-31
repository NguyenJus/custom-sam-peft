# spec/plateau-response-ladder — Plateau-response LR decay, early stopping, and best-as-final close-out (issue #197)

**Status:** Draft (2026-05-30)
**Tracking:** [#197](https://github.com/NguyenJus/custom-sam-peft/issues/197) — *Plateau-response ladder: LR decay before early stopping; early stop on by default*
**Scope:** Add a two-rung **plateau-response ladder** to training — (1) reduce-on-plateau LR decay, (2) metric early stop — both driven by the monitored validation metric `mAP` at each `eval_every` boundary. Introduce a new `lr_schedule` mode `plateau` that **becomes the default** (was `cosine`) and replaces the per-step cosine schedule with warmup → `ReduceLROnPlateau`. Generalize end-of-run close-out to restore **best weights as the final adapter** (early stop, normal completion, and an explicit finalize entry all close out on `best/`). Persist ladder state across `--resume` and fix a best-clobber bug. Add a lightweight CLI **finalize** entry that productionizes a paused (time-limited) run with no training.

**Builds on / cross-links:**
[`2026-05-30-train-time-limit-design.md`](2026-05-30-train-time-limit-design.md) (#198) — this spec consumes that feature's `_TimeLimitReached` pause, `EvalArtifacts.time_limit_stop` seam, `_time_limited_artifacts`, and the documented #197 close-out handoff. Voice and structure match that spec.
[`2026-05-17-training-loop-design.md`](2026-05-17-training-loop-design.md) — the `run_epoch` step loop, `on_eval`/`on_checkpoint` wiring, scheduler-stepping precedent, and epoch-boundary resume granularity.
[`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) — the thin-shell command boundary the finalize entry honors.
[`2026-05-18-simplify-ux-design.md`](2026-05-18-simplify-ux-design.md) — the `run` orchestration phases this spec's close-out reuses.

**Research basis (all hyperparameter citations):**
[`docs/research/2026-05-30-issue-197-plateau-lr-decay-early-stopping-lit-review.md`](../../research/2026-05-30-issue-197-plateau-lr-decay-early-stopping-lit-review.md). Every default below cross-links a section of that write-up; do not re-derive citations inline.

---

## 1. Summary

Today training runs a fixed-horizon `cosine` LR schedule (per-step `LambdaLR`) for the full `epochs` count, periodically lite-evaluates, saves `run_dir/best/` on each new best `mAP`, and at the end exports the **last-step** weights. There is no mechanism to react to a validation plateau, and no early stop.

This feature adds a **plateau-response ladder** evaluated at each `eval_every` boundary on `mAP`, using one shared "did it improve?" test — `mAP > best + min_delta`:

1. **Rung 1 — LR decay.** Every `lr_decay_on_plateau.patience` non-improving evals, multiply LR by `factor`, floored at `min_lr`. Implemented by `torch.optim.lr_scheduler.ReduceLROnPlateau` (`mode="max"`).
2. **Rung 2 — early stop.** After `early_stop.stop_patience` non-improving evals — a **separate** counter that resets only on genuine improvement, **not** on an LR cut — halt training.

A new `lr_schedule` mode `plateau` becomes the **default** (was `cosine`). In `plateau` mode, warmup is unchanged (linear ramp over `warmup_steps`), then the LR holds at base, then `ReduceLROnPlateau` drives per-eval cuts. This **replaces** the per-step cosine `LambdaLR`. The per-step `LambdaLR` path stays intact for `cosine`/`linear`/`constant`. The research notes (§3–§4) establish that plateau decay and cosine are **alternatives, not stacked**, and that for this repo's ~160-epoch SAMed-anchored convergence runs that metric-early-stop, plateau decay is the accuracy-reliable choice (a horizon-calibrated cosine forfeits its low-LR endgame when the stop fires early).

When training ends — by early stop **or** by normal full-epoch completion — the run **closes out on the best weights**: a reusable `close_out(run_dir, model, cfg, ...)` restores `run_dir/best/` into the model, runs the full eval, and writes `run_dir/adapter` (+ optional `run_dir/merged` + `metrics.json` + bundle), all on the **best** checkpoint. This generalizes today's normal-completion finalize, which exports last-step weights. A new **finalize** CLI entry rebuilds the model from a paused run's checkpoint and calls `close_out` with **no training**, closing the #198 handshake "a user decides a paused, time-limited run is done."

Ladder state (the `ReduceLROnPlateau` internals, the early-stop counter, the best-metric value) is persisted into the resumable `training_state` checkpoint, and `--resume` restores it. Resume also re-seeds `_best_metric_value` from `run_dir/best/best.json`, fixing a clobber bug where it currently resets to `-inf` each `fit()`.

The #198 wall-clock `time_limit` path is **unchanged**: a time-limited stop stays a pure pause (no eval/export/bundle, prints a resume message) — it is **never** routed through `close_out`. Only early stop, normal completion, and the explicit finalize entry reach `close_out`.

---

## 2. Motivation

The standing design priority is **final accuracy ≫ user-facing simplicity ≫ training speed** (project memory). Two gaps follow from that:

1. **No plateau response.** A fixed-horizon cosine schedule earns its accuracy in the low-LR *endgame* at the horizon. Adding metric early stopping (which #197 wants on by default) means the run can quit *before* the horizon — leaving cosine at a relatively high LR and forfeiting the endgame. `ReduceLROnPlateau` reacts to the same stall that drives the stop, so a low-LR fine-tuning probe always happens before the run gives up. This is the canonical reason to pair reduce-on-plateau with early stopping (research §4).
2. **Last-step, not best, as final.** Today's end-of-run finalize exports the last-step adapter, even though `run_dir/best/` already holds the best-`mAP` checkpoint. With early stopping on, the last step is by construction *worse* than the best (that is why the run stopped). Best-as-final makes the shipped adapter the best one observed, on every termination path.

The companion #198 feature adds a resumable wall-clock pause but deliberately defers "productionize a paused run" to this issue. The finalize entry here closes that handshake.

---

## 3. Goals & Non-goals

**Goals.**

- New `lr_schedule` mode `plateau`, the new default; `cosine`/`linear`/`constant` remain available.
- Two config blocks on `TrainHyperparams`: `lr_decay_on_plateau` (rung 1) and `early_stop` (rung 2), with the exact shape and defaults in §5.
- In `plateau` mode: warmup (unchanged) → hold → `ReduceLROnPlateau` (`mode="max"` on `mAP`), stepped per-eval. The per-step `LambdaLR` path stays for non-plateau modes.
- Two counters fed the same `mAP` at each eval: rung 1 reuses `ReduceLROnPlateau`'s internal bad-eval counter; rung 2 is an independent counter that resets **only** on genuine improvement.
- Single-sourced improvement definition: `min_delta` is the `ReduceLROnPlateau` `threshold` *and* the early-stop delta; `mAP` is the single monitored metric.
- Val-required fallback: `plateau` with no val set falls back to `cosine` with a logged warning. Early stop is a no-op without val (eval already is).
- Best-as-final close-out: a reusable `close_out(run_dir, model, cfg, ...)` restoring `best/`, running full eval, writing `adapter` + optional `merged` + `metrics.json` + bundle on the best weights. Called on early stop, normal completion, and finalize.
- Resume persists ladder state in `training_state`; re-seeds `_best_metric_value` from `best.json` (clobber-bug fix).
- A finalize CLI entry: rebuild from a paused run's checkpoint, call `close_out`, no training.
- Every new/changed default cited or `# tbd:`-tagged; docs updated.

**Non-goals.**

- **Stacking plateau decay on cosine.** The two are alternatives (research §3); `plateau` *replaces* the per-step schedule. Not composed.
- **Other decay methods (SGDR warm restarts, weight-decay scheduling).** Deferred with the research §6 rationale: they fire on a *fixed schedule*, not a plateau/validation trigger, so they do not belong in a plateau-*response* rung. Rung 1 is `LR × factor` only.
- **Configurable monitor metrics beyond `mAP`.** A `monitor` config field is exposed defaulting to `mAP` as a clean seam, but only `mAP` is validated/wired.
- **Cumulative cross-resume time tracking / changing the #198 per-run budget.** The wall-clock budget still restarts per invocation; only the *ladder* state persists.
- **Routing a time-limited stop through close-out.** A `_TimeLimitReached` stays a pure pause (§9). Finalizing a paused run is the explicit, user-initiated finalize entry.

---

## 4. Architecture & data flow

### 4.1 The eval-boundary tick

The ladder advances **only** at an `eval_every` boundary, and **only** when a real eval produced an `mAP`. The seam is the existing `on_eval(step)` closure (`trainer.py:540`), wired to `_eval_epoch`, which is invoked from `run_epoch` at each `eval_every` boundary (`loop.py:519-524`).

A new trainer-owned object, `LadderState`, holds the two counters and the shared improvement test. `_eval_epoch` already computes the lite-eval `report` and calls `_maybe_save_best`. The ladder hooks in **right after** `_maybe_save_best`, fed the same `mAP`:

```text
run_epoch micro-step loop
  └─ eval_every boundary → on_eval(step) → _eval_epoch(step, run_dir, oom_state)
        ├─ report = Evaluator(lite_cfg).evaluate(...)        # may raise → swallowed
        ├─ tracker.log_scalars(step, report.overall)
        ├─ _maybe_save_best(report, step, run_dir)           # saves best/, updates _best_metric_value
        └─ ladder.observe(mAP, step, scheduler) -> StopDecision
              ├─ rung 1: scheduler.step(mAP)  (ReduceLROnPlateau) → maybe LR cut (logged)
              └─ rung 2: early-stop counter; if ≥ stop_patience → signal stop
```

Crucially, `_eval_epoch` swallows eval failures (`trainer.py:317-328`): on a failed/OOM eval it returns **without** calling `_maybe_save_best` or `ladder.observe`. A skipped eval therefore advances **neither** counter (error handling, §10). `ladder.observe` is reached **only** on a successful eval that yielded an `mAP`.

### 4.2 Signalling a stop to `fit()`

Early stop must unwind the epoch loop the same way #198's time-limit does, but it is **not** a pause — it proceeds to close-out. Two equivalent mechanisms; this spec picks the **flag** form to keep the control flow flat and avoid confusing early stop with the `_TimeLimitReached` exception:

- `LadderState.observe(...)` returns a `StopDecision` (a small frozen dataclass: `should_stop: bool`, `reason: str`, `triggering_step: int`, `triggering_map: float`).
- `_eval_epoch` stores the decision on the trainer (`self._early_stop: StopDecision | None`).
- `run_epoch` checks `on_eval`'s side effect via a trainer-provided predicate after each `on_eval` call and, when set, raises a dedicated internal exception `_EarlyStop(step, epoch, reason)` (mirroring `_TimeLimitReached`'s shape, in `train/loop.py`). This unwinds `run_epoch → _train_epoch → fit()`'s epoch loop.

To keep `run_epoch`'s signature seam minimal, the predicate is passed as one more optional callback parameter, `should_stop_early: Callable[[], _EarlyStop | None] | None = None`, evaluated immediately after `on_eval(step)`. When it returns a non-`None` `_EarlyStop`, `run_epoch` raises it. This keeps `run_epoch` free of ladder knowledge — it only knows "after an eval, ask whether to stop."

`fit()` catches `_EarlyStop` around the epoch loop (the same `try` that already catches `_TimeLimitReached` at `trainer.py:577-579`):

```text
try:
    for epoch in range(start_epoch, cfg.train.epochs):
        global_step, nan_streak = self._train_epoch(..., deadline=deadline)
        P.advance_outer()
except _TimeLimitReached as e:
    stop = e               # pure pause (unchanged)
except _EarlyStop as e:
    early = e              # NEW — proceeds to close-out
finally:
    ...
```

Then `fit()` branches:

- `stop is not None` (time limit): return `_time_limited_artifacts(...)` — **unchanged**.
- otherwise (early stop **or** normal completion): call `close_out(...)` and return its `EvalArtifacts`.

Because both early stop and normal completion funnel into the same `close_out`, there is a single best-as-final finalize path.

### 4.3 Component map

| Component | Role | Location |
| --- | --- | --- |
| `lr_schedule="plateau"` enum value | new default schedule mode | `config/schema.py` `LRSchedule` |
| `lr_decay_on_plateau` block (`LrDecayOnPlateauConfig`) | rung-1 knobs | `config/schema.py` (new `_Strict` model) |
| `early_stop` block (`EarlyStopConfig`) | rung-2 knobs + shared monitor/min_delta | `config/schema.py` (new `_Strict` model) |
| scheduler split (warmup → `ReduceLROnPlateau`; per-eval vs per-step stepping) | build + step the right scheduler per mode | `train/trainer.py` (`_build_scheduler`, `fit`), `train/loop.py` (`run_epoch`, `train_step`) |
| `LadderState` | two counters, shared improvement test, LR-cut/stop telemetry | `train/ladder.py` (new) |
| `_EarlyStop` | internal stop signal | `train/loop.py` |
| `StopDecision` | observe() return value | `train/ladder.py` |
| `close_out(run_dir, model, cfg, ...)` | best-as-final finalize | `train/close_out.py` (new) |
| ladder state in `training_state` | persistence | `train/checkpoint.py` (`save_full_state`/`load_full_state`/`ResumeState`) |
| `_best_metric_value` re-seed | clobber-bug fix | `train/trainer.py` (`fit`) |
| finalize CLI | rebuild + close_out, no training | `cli/run_cmd.py` (new `--finalize` flag) + helper |
| ladder/close-out reflection in artifacts | best-as-final semantics | `eval/_artifacts.py`, `runs/bundle.py` (summary/edge-notes) |

---

## 5. Config schema additions

Two new blocks mount on `TrainHyperparams` (`config/schema.py`, after `eval_every` in the `# --- advanced ---` section, before `loss`). The new `lr_schedule` default is a **common-field** change. The exact YAML shape:

```yaml
train:
  lr_schedule: plateau          # NEW default (was cosine)
  warmup_steps: 100             # unchanged
  lr_decay_on_plateau:          # rung 1 — active only in plateau mode
    patience: 5
    factor: 0.1
    min_lr: 1.0e-6
  early_stop:                   # rung 2 — always active (any schedule)
    enabled: true
    monitor: mAP
    min_delta: 0.001
    stop_patience: 10
```

### 5.1 `LRSchedule` extension

```python
LRSchedule = Literal["constant", "cosine", "linear", "plateau"]
```

And the field default flips:

```python
lr_schedule: LRSchedule = "plateau"
# cite: ReduceLROnPlateau (PyTorch/Keras) + the canonical early-stop pairing
#       (research §2–§4); # tbd: #197 — the cosine→plateau default flip.
```

`cosine`/`linear`/`constant` keep today's per-step `LambdaLR` behavior verbatim.

### 5.2 `LrDecayOnPlateauConfig` (rung 1)

```python
class LrDecayOnPlateauConfig(_Strict):
    """Rung-1 reduce-on-plateau knobs. Active only when lr_schedule == "plateau"."""

    patience: PositiveInt = 5
    # cite: Keras ReduceLROnPlateau example 5 (low end of cited 5–10 range);
    #       research §2, §7.
    factor: PositiveFloat = 0.1
    # cite: PyTorch ReduceLROnPlateau default 0.1; research §2, §7.
    min_lr: PositiveFloat = 1.0e-6
    # cite: PyTorch default 0; # tbd: floored at learning_rate/100 to avoid a dead LR;
    #       research §7.
```

Validation: `factor < 1.0` (a reduce factor must shrink the LR). `min_lr` is a positive float floor.

### 5.3 `EarlyStopConfig` (rung 2 + shared improvement definition)

```python
class EarlyStopConfig(_Strict):
    """Rung-2 early-stop knobs. monitor/min_delta are the SHARED improvement
    definition consumed by rung 1 too (see the wart note below)."""

    enabled: bool = True
    # issue: on by default (research §7, issue acceptance criteria).
    monitor: Literal["mAP"] = "mAP"
    # existing best-metric key (trainer.py _best_metric_key). Exposed as a seam;
    # only mAP is validated/wired for now.
    min_delta: PositiveFloat = 0.001
    # cite: early-stop min_delta range 0.001–0.01 (Keras/practitioner);
    #       # tbd: low end chosen for a noisy mAP; research §5, §7.
    stop_patience: PositiveInt = 10
    # cite: patience 5–10 (PyTorch ReduceLROnPlateau default 10 / Prechelt 1998);
    #       # tbd: high end chosen — accuracy ≫ speed; research §5, §7.
```

`monitor` is a single-value `Literal["mAP"]` so the seam is present but no unsupported value can be configured. Widening it later (e.g. to `"DSC"`) is a one-line `Literal` change plus a metric lookup.

### 5.4 The documented wart

`monitor` and `min_delta` live in the `early_stop` block but are the **shared** improvement definition consumed by **both** rungs. Concretely: if `early_stop.enabled = false` while `lr_schedule = plateau`, those two fields still configure the rung-1 LR-decay threshold (they feed `ReduceLROnPlateau`'s `threshold` and the monitored metric). This is the one config wart: the rung-1 decay's "what counts as improvement" is sourced from the `early_stop` block even when early stop is disabled. It is documented in the field docstrings and `config-schema.md`. The alternative — duplicating `monitor`/`min_delta` into `lr_decay_on_plateau` — was rejected because two sources for one "improvement" definition is the worse footgun (they could drift, giving the two rungs *different* notions of improvement, which contradicts the single-sourced design in §6.3).

### 5.5 Demoted `early_stop_p_threshold` seam

The issue references a demoted `early_stop_p_threshold` field. That field lived on `BoxHintSchedule` (never on `TrainHyperparams`) and was **already removed in #88** (CHANGELOG-confirmed: "`BoxHintSchedule.early_stop_p_threshold` — was unused; removed pending" this follow-up); a grep of the current tree finds it only in the CHANGELOG. **Assumption A1:** there is no such field to remove — Phase 1 only confirms its absence (a no-op). The issue's `config/schema.py:530` line reference is stale (line numbers shifted post-#88). No config consumes it.

### 5.6 Defaults table

Every value is cited or `# tbd:`-tagged per the cite-new-hyperparams rule; cross-reference the research notes §7.

| Knob | Value | Tag |
| --- | --- | --- |
| `lr_schedule` default | `plateau` | `# cite:` ReduceLROnPlateau (PyTorch/Keras) + early-stop pairing (research §2–§4); `# tbd: #197` for the cosine→plateau flip |
| `lr_decay_on_plateau.factor` | `0.1` | `# cite:` PyTorch ReduceLROnPlateau default `0.1` (research §2, §7) |
| `lr_decay_on_plateau.patience` | `5` | `# cite:` Keras ReduceLROnPlateau example `5` (within cited 5–10) (research §2, §7) |
| `lr_decay_on_plateau.min_lr` | `1e-6` | `# cite:` PyTorch default `0`; `# tbd:` floored at `learning_rate/100` (research §7) |
| `early_stop.enabled` | `true` | `# issue:` on by default |
| `early_stop.monitor` | `mAP` | existing best-key (`trainer.py`) |
| `early_stop.min_delta` | `0.001` | `# cite:` `0.001`–`0.01` (Keras/practitioner); `# tbd:` low end for noisy mAP (research §5, §7) |
| `early_stop.stop_patience` | `10` | `# cite:` `5`–`10` (PyTorch default `10` / Prechelt 1998); `# tbd:` high end, accuracy ≫ speed (research §5, §7) |

**Resulting ladder** (base LR `1e-4`, one eval/epoch): non-improving evals 1–5 → one 10× cut to `1e-5`; evals 6–10 → halt at eval 10. One deep endgame drop plus a 5-eval low-LR probe before the run gives up. Respecting both cited ranges (`patience ≥ 5`, `stop_patience ≤ 10`) yields **exactly one cut before stop**; a multi-step staircase would require pushing `stop_patience` past the cited range (research §7).

---

## 6. Scheduler-mode mechanics

### 6.1 The two stepping disciplines

The crux: `LambdaLR` is stepped **per training step**; `ReduceLROnPlateau` is stepped **per eval** with the `mAP` value. The trainer must branch on schedule type for both *building* and *stepping*. `ReduceLROnPlateau` is also **not** an `LRScheduler` subclass and has no `get_last_lr()`, so logging must branch too.

| Mode | Scheduler | Stepped | Step argument | LR read for logging |
| --- | --- | --- | --- | --- |
| `cosine` / `linear` / `constant` | `LambdaLR` (today) | per optimizer step (`train_step`, `loop.py:389`) | none | `scheduler.get_last_lr()[0]` |
| `plateau` | warmup→`ReduceLROnPlateau` | warmup ramp per step; plateau cut per eval | `mAP` (per eval) | `optimizer.param_groups[0]["lr"]` |

### 6.2 Warmup → ReduceLROnPlateau handoff (concrete mechanism)

In `plateau` mode the warmup must remain a per-step linear ramp (unchanged behavior over the first `warmup_steps` global steps), and the plateau cuts must be per-eval. PyTorch's `SequentialLR` cannot chain a per-step scheduler into a per-eval `ReduceLROnPlateau` cleanly (the two are stepped on different cadences and `SequentialLR` does not support `ReduceLROnPlateau` as a child). This spec therefore uses a **manual two-phase** mechanism, which is explicit and testable:

1. **Build (in `_build_scheduler`, branching on `cfg.train.lr_schedule`):**
   - Non-plateau modes: unchanged — return today's `LambdaLR` (per-step). The function's return type widens to `LRScheduler | ReduceLROnPlateau` (a `train/_scheduler.py` type alias `PlateauOrLambda`), or it returns a small wrapper (see below).
   - `plateau` mode: return a `ReduceLROnPlateau(optimizer, mode="max", factor=cfg.train.lr_decay_on_plateau.factor, patience=cfg.train.lr_decay_on_plateau.patience, threshold=cfg.train.early_stop.min_delta, threshold_mode="abs", min_lr=cfg.train.lr_decay_on_plateau.min_lr)`. **Note** `threshold_mode="abs"` so the improvement test is `mAP > best + min_delta` in absolute mAP units (matching the early-stop test exactly); the PyTorch default `"rel"` would make the threshold a *fraction* of `best`, diverging from the early-stop semantics.

2. **Warmup ramp (manual, in `train_step`):** the LR is set explicitly during warmup. The per-step warmup multiplier is the existing `(step + 1) / max(warmup_steps, 1)` ramp. In `plateau` mode, `train_step` applies the warmup factor by writing `optimizer.param_groups[*]["lr"]` directly for `global_step < warmup_steps` (scaled from `cfg.train.learning_rate`), and does **not** call `scheduler.step()` for the plateau scheduler. After warmup, the LR holds at base until the first plateau cut. `ReduceLROnPlateau` owns the LR from the first eval onward; it reads/writes `param_groups[*]["lr"]`, so a `min_lr` floor and the warmup writes share the same surface coherently.

   Implementation seam: `train_step` already calls `scheduler.step()` unconditionally at `loop.py:389`. This becomes mode-aware. The cleanest encapsulation is a tiny `_scheduler.py` helper, `step_per_train_step(scheduler, *, global_step, base_lr, warmup_steps, mode)`:
   - non-plateau mode: `scheduler.step()` (per-step `LambdaLR`), exactly as today.
   - plateau mode: if `global_step < warmup_steps`, set `param_groups` LR to `base_lr * (global_step + 1) / max(warmup_steps, 1)`; else **no-op** (the plateau scheduler is stepped only at evals).

   `train_step` calls this helper instead of the bare `scheduler.step()`. This keeps the branch in one place and keeps `run_epoch`/`train_step` otherwise unchanged.

3. **Per-eval plateau step (in `LadderState.observe`, §6.4):** at each successful eval, if the scheduler is a `ReduceLROnPlateau`, call `scheduler.step(mAP)`. This is rung 1.

4. **LR logging (in `run_epoch`'s scalar window, `loop.py:507`):** `scheduler.get_last_lr()` does not exist on `ReduceLROnPlateau`. Read the current LR via `optimizer.param_groups[0]["lr"]` for **all** modes (a `LambdaLR` keeps `param_groups` LR in sync after `step()`, so this read is correct for non-plateau modes too and removes the `get_last_lr` dependency entirely). This is a one-line change at the window-update call site.

### 6.3 The two counters

Both counters are fed the **same** `mAP` at **each** successful eval, using the **same** `min_delta` threshold and the **same** `mAP` monitor — single-sourced improvement.

- **Rung 1 — reuse `ReduceLROnPlateau`'s internal bad-eval counter** (`num_bad_epochs`). Its built-in behavior: increment on a non-improving eval; reset to 0 on improvement; reset to 0 **and** apply a cooldown after a cut. This reset-on-cut is *exactly* the staircase rung 1 wants — after a cut it gives the lower LR a fresh `patience` window before cutting again. We do not re-implement this; we just `scheduler.step(mAP)`.
- **Rung 2 — an independent early-stop counter in `LadderState`** (`evals_without_improvement: int`). It resets to 0 **only** on genuine improvement (`mAP > best + min_delta`) and increments otherwise. It is **not** reset on an LR cut. When it reaches `early_stop.stop_patience`, `observe` returns `should_stop=True`.

Both share one improvement test, computed once per eval:

```text
improved = mAP > ladder.best + cfg.train.early_stop.min_delta
if improved:
    ladder.best = mAP
    ladder.evals_without_improvement = 0
else:
    ladder.evals_without_improvement += 1
# rung 1: scheduler.step(mAP) handles num_bad_epochs + cuts internally (plateau mode)
# rung 2: stop if evals_without_improvement >= stop_patience (when early_stop.enabled)
```

`ladder.best` and `_best_metric_value` track the same quantity but serve different roles: `_best_metric_value` (in `_maybe_save_best`) gates **saving** `best/`; `ladder.best` gates the **counters**. They are kept consistent (both updated from the same eval), and both are re-seeded on resume (§8). **Assumption A2:** `_maybe_save_best` uses a strict `metric > _best_metric_value` test (no `min_delta`), whereas the ladder uses `mAP > best + min_delta`. These can disagree on a tiny improvement (saves a new best but counts as non-improvement for patience). This is intentional and correct: we always want to *save* a strictly-better checkpoint, but only *reset patience* on a meaningfully-better one. Documented in `LadderState`.

### 6.4 `LadderState` interface (Phase-1 contract out)

```python
@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str          # "early_stop: N evals without mAP improvement (>= stop_patience)"
    triggering_step: int
    triggering_map: float

class LadderState:
    best: float                      # best mAP seen by the ladder
    evals_without_improvement: int   # rung-2 counter
    # rung-1 counter lives inside the ReduceLROnPlateau (not duplicated here)

    def observe(self, mAP, step, scheduler, cfg) -> StopDecision: ...
    def state_dict(self) -> dict: ...        # {best, evals_without_improvement}
    def load_state_dict(self, d) -> None: ...
```

`observe` performs the improvement test, updates rung-2, steps the plateau scheduler (rung 1, plateau mode only), logs an LR cut when one fired (detected by comparing pre/post `param_groups[0]["lr"]`), and returns the `StopDecision`. The `ReduceLROnPlateau`'s own `state_dict()`/`load_state_dict()` carry the rung-1 counter (`best`, `num_bad_epochs`, `cooldown_counter`) and are persisted via the existing `scheduler` slot in `training_state` (§8) — no duplication.

### 6.5 Val-fallback

`plateau` **requires** a validation metric. With `val_ds is None`, there is no plateau signal. The trainer **automatically falls back to `cosine`** for the LR schedule, with a logged warning, and early stop is a no-op (eval already is). Concretely, in `fit()` before `_build_scheduler`:

```text
effective_schedule = cfg.train.lr_schedule
if cfg.train.lr_schedule == "plateau" and self.val_ds is None:
    _LOG.warning(
        "lr_schedule=plateau requires a validation set for the plateau signal; "
        "no val set provided — falling back to lr_schedule=cosine. Early stop is a no-op."
    )
    effective_schedule = "cosine"
```

`_build_scheduler` is then built against `effective_schedule` (passed explicitly, not re-reading `cfg.train.lr_schedule`). The **effective schedule kind** — the scheduler actually used after this fallback — is persisted to `training_state` and rebuilt verbatim on `--resume` (§8.1, §8.3), so a no-val fallback run resumes as the same `cosine` scheduler **even if a val set is present on the resume invocation**. This closes the trap where resume would otherwise rebuild `plateau` from `cfg` and fail to load the persisted `cosine` `LambdaLR` state. The written `run_dir/config.yaml` echoes the user's requested `plateau` (the request is preserved; the fallback is explicit in logs and in the persisted `scheduler_kind`). `LadderState` is still constructed but `observe` is never called (no eval → no tick). **Assumption A3:** the fallback affects only the LR schedule and ladder; all other `plateau`-mode behavior (which is val-gated anyway) is naturally inert without val.

---

## 7. Close-out extraction & call sites

### 7.1 Today's normal-completion finalize

The post-loop block (`trainer.py:581-621`, the `if stop is None:` branch) currently: `save_adapter(model, run_dir/adapter)` on **last-step** weights; optional `save_merged`; full eval on `val_ds`; `metrics.json`. This is the logic to generalize.

### 7.2 `close_out(run_dir, model, cfg, ...)`

Extract a reusable function in `train/close_out.py`:

```python
def close_out(
    run_dir: Path,
    model: Sam3Wrapper,
    cfg: TrainConfig,
    *,
    evaluator_val_ds: Dataset | None,
    oom_state: OomState | None,
    final_step: int,
    final_epoch: int,
    ladder_events: LadderEvents | None = None,
) -> EvalArtifacts:
    """Restore best/ into model, run one full eval (return_per_example_iou=True),
    write adapter + optional merged + metrics.json — all on the BEST weights.
    Falls back to the current (last-step) weights only when no best/ exists.
    Returns EvalArtifacts (final_metrics + per_example_iou reflect the BEST
    checkpoint's single eval); callers build any bundle from per_example_iou."""
```

Steps:

1. **Restore best.** If `run_dir/best/adapter` exists, `load_adapter(model, run_dir/best/adapter)`. The model now holds the best weights. If `best/` is absent (no val, or no eval boundary reached — e.g. a sub-`eval_every` run), keep the current in-memory (last-step) weights and note "best/ absent — finalized on last-step weights" in the artifacts/summary.
2. **Write adapter.** `save_adapter(model, run_dir/adapter)` — now the **best** adapter (or last-step in the fallback).
3. **Optional merged.** If `cfg.export.merge`, `save_merged(model, run_dir/merged)`. (Mirrors today.)
4. **Full eval on the restored weights — exactly once.** Same auto-batch-cap logic as today (`trainer.py:588-596`), run on `evaluator_val_ds` with `return_per_example_iou=True`. This single eval measures the **best** checkpoint — so `metrics.json` is the *best* mAP — and its `per_example_iou` is returned on `EvalArtifacts` for the bundle (step 6), so the `run` orchestrator never re-evaluates (§7.4).
5. **`metrics.json`.** Same shape as today (`overall`, `per_class`, `n_images`, `n_predictions`, `global_step`, `epoch`), plus a `"final_weights": "best" | "last_step"` field and the ladder events (cuts, stop reason) when present.
6. **Bundle data, not the bundle itself.** `close_out` does **not** assemble or write the bundle — that stays the caller's job (the `run` orchestrator, or the `train`/finalize post-steps). `close_out` **returns** the step-4 eval's `per_example_iou` on `EvalArtifacts` (a new optional field, default `None`). Callers that build a bundle read this field and assemble `BundleContext` from it, so the bundle is built from `close_out`'s single eval with **no second eval**. This keeps `close_out` free of `BundleContext` assembly while collapsing the run path to one eval.

`LadderEvents` is a small frozen record (`cuts: tuple[LrCut, ...]`, `stop_reason: str | None`) the trainer accumulates during the run and threads into `close_out` for surfacing in `metrics.json`/summary.

### 7.3 Call sites

1. **Early stop** (`fit()`, after catching `_EarlyStop`): call `close_out(...)` with `final_step`/`final_epoch` from the stop, `ladder_events` populated, and return its `EvalArtifacts`.
2. **Normal completion** (`fit()`, epoch loop finished without any stop): call the **same** `close_out(...)`. This *replaces* the inline `trainer.py:581-621` block. `final_epoch = cfg.train.epochs - 1`.
3. **Finalize entry** (§8 / §11, no training): rebuild model, then `close_out(...)`.

`close_out` is **not** called for a `_TimeLimitReached` stop (§9).

### 7.4 `run` orchestrator interaction

`_orchestrate` (`run_cmd.py:72`) today runs train → eval → export-merge → bundle as separate phases. With best-as-final + single-eval, the train phase's `close_out` (inside `fit()`) already does the export-merge **and** the one full eval (`return_per_example_iou=True`) on the best weights, returning `final_metrics` **and** `per_example_iou` on `EvalArtifacts`. So the orchestrator **drops its own eval and export-merge phases** for the normal path and builds the bundle directly from `train_result.final_metrics` + `train_result.per_example_iou`; `checkpoint_path` is `run_dir/adapter` (best weights). **Net: exactly one eval on the `run` path** (in `close_out`), down from two. The time-limit short-circuit (`run_cmd.py:91`) is unchanged — a paused run still skips close-out entirely, and `per_example_iou` is `None` there.

### 7.5 `train` path interaction

`train_cmd.py`'s `train(...)` calls `run_train` then optionally `--eval`/`--export`. With best-as-final, `result.checkpoint_path` is the best adapter and `result.final_metrics` is the best eval. `--eval`/`--export` already operate on `result.checkpoint_path`, so they transparently use the best adapter. No change needed beyond the time-limit short-circuit (already present).

### 7.6 Bundle / summary reflection

`runs/bundle.py` surfaces best-as-final and ladder events:

- The summary "Outputs" section notes the final adapter is the **best** checkpoint (e.g. `- Adapter: adapter/ (best checkpoint, mAP <best_map> at step <best_step>)`), sourced from `best.json`. When `close_out` fell back to last-step weights, it reads `- Adapter: adapter/ (last-step weights — no best/ produced)`.
- A new `## Training` (or an addition to `## Edge cases`) line surfaces ladder events: each LR cut (`LR cut ×0.1 → 1e-5 at eval step <N> (mAP <m>)`) and the stop reason (`early stop: 10 evals without mAP improvement at step <N>`). These come from `LadderEvents`, threaded through `BundleContext` (one new optional field `ladder_events: LadderEvents | None = None`, defaulting to `None` so no-val / non-plateau runs render unchanged).

---

## 8. Resume & persistence

### 8.1 What is added to `training_state`

`save_full_state`/`load_full_state` (`checkpoint.py:147,178`) gain ladder state. The `scheduler` slot already round-trips `ReduceLROnPlateau.state_dict()` (which carries `best`, `num_bad_epochs`, `cooldown_counter` — the rung-1 state) because `save_full_state` already calls `scheduler.state_dict()` generically. So **rung 1 persists for free** once the scheduler is a `ReduceLROnPlateau`. Three new fields are added explicitly to the payload:

```python
payload["ladder"] = ladder.state_dict()      # {best, evals_without_improvement}
payload["best_metric_value"] = best_metric_value   # mirrors trainer._best_metric_value
payload["scheduler_kind"] = effective_schedule     # effective LR schedule actually used (post val-fallback)
```

`ResumeState` gains the restored values:

```python
@dataclass(frozen=True)
class ResumeState:
    start_step: int
    start_epoch: int
    nan_streak: int
    ladder: dict | None = None          # NEW — None for pre-#197 checkpoints
    best_metric_value: float | None = None  # NEW
    scheduler_kind: str | None = None   # NEW — effective LR schedule actually used; governs resume rebuild
```

`save_full_state` takes the new `ladder` and `best_metric_value` arguments. `load_full_state` reads them with `.get(...)` defaults so **old checkpoints load** (the format version stays `1`; the new keys are additive and optional — a missing `ladder` key restores `None`, and `fit()` treats `None` as "fresh ladder + re-seed from best.json").

### 8.2 The clobber-bug fix

Today `_best_metric_value` is set to `-inf` in `__init__` (`trainer.py:181`) and is **never re-seeded on resume**. So after `--resume`, the **first** post-resume eval overwrites `run_dir/best/` even if its mAP is *worse* than the already-saved best (any finite mAP beats `-inf`). The fix, in `fit()` after `load_full_state`:

```text
if resume_from is not None:
    # 1. Prefer the persisted value from training_state (exact).
    if rs.best_metric_value is not None:
        self._best_metric_value = rs.best_metric_value
    # 2. ALSO re-seed from best/best.json (authoritative on-disk best), taking
    #    the max — guards a checkpoint saved before best/ was last updated.
    best_json = (resume_run_dir / "best" / "best.json")
    if best_json.is_file():
        self._best_metric_value = max(self._best_metric_value, float(read best.json["value"]))
    # 3. Re-seed the ladder.best from the same source.
    ladder.best = self._best_metric_value
    if rs.ladder is not None:
        ladder.load_state_dict(rs.ladder)   # restores evals_without_improvement (+ best)
```

The `resume_run_dir` is the run dir owning the checkpoint (`resume_from.parent.parent`, as `run_training` already computes at `runner.py:91`). Re-seeding both `_best_metric_value` and `ladder.best` means the first post-resume eval cannot clobber a better `best/`, and patience counting continues from the correct baseline.

### 8.3 Resume invariants

- On `--resume`, the scheduler is rebuilt from the **persisted `scheduler_kind`** (the effective schedule the run actually used), **not** re-derived from `cfg.train.lr_schedule`. This guarantees the rebuilt scheduler type matches the persisted `state_dict`, regardless of val presence or `cfg` drift on the resume invocation. For `plateau`, `_build_scheduler` rebuilds the `ReduceLROnPlateau` and `load_full_state` restores its `state_dict()` — so `num_bad_epochs`/`cooldown_counter`/`best` continue exactly. The current `param_groups` LR (post-cut) is restored via the optimizer state. (Pre-#197 checkpoints carry no `scheduler_kind`; the trainer falls back to `cfg.train.lr_schedule` for them.)
- `evals_without_improvement` continues from the persisted value.
- The #198 wall-clock `time_limit` budget still **restarts** per invocation (only the ladder state persists). This is intentional and matches #198.
- A **user-initiated** schedule change on resume (editing `lr_schedule` in `--config` to a different family) does not rebuild a different scheduler: the persisted `scheduler_kind` governs construction, so the edited value is ignored for the scheduler and a clear warning is logged (persisted kind wins, mirroring the `cfg_hash`/`peft_method` mismatch guards at `checkpoint.py:231`). **Assumption A5:** the scheduler kind is fixed at the first (fresh) invocation — after val-fallback resolution — and immutable across resumes. This is also what lets the no-val→`cosine` fallback resume cleanly even if a val set is later added.

---

## 9. Interaction with #198 time-limit and the OOM ladder

Both must stay green; this section states **why** they are unaffected.

### 9.1 #198 time-limit

- A `_TimeLimitReached` is raised inside `run_epoch` at the micro-step boundary (`loop.py:525-539`), flushes a checkpoint, and unwinds to `fit()` where it is caught and routed to `_time_limited_artifacts` — a **pure pause**: no eval, no `close_out`, prints a resume message, exit 0.
- **`close_out` is never reached on a time-limited stop.** `fit()` branches: `stop is not None` → `_time_limited_artifacts` (unchanged); else → `close_out`. The two are mutually exclusive.
- The ladder ticks only at eval boundaries (`on_eval`), which the time-limit path does not invoke at stop time — so a time-limited stop never advances a patience counter or fires a cut.
- The ladder state **is** persisted in the time-limit flush (because the flush calls `save_full_state`, which now writes `ladder`/`best_metric_value`). So resuming a time-limited pause restores the ladder correctly — the two features compose cleanly through the shared checkpoint.
- The finalize entry (§11) is the explicit, user-initiated way to turn a time-limited pause into a best-as-final shippable artifact, closing the #198 handshake without coupling the pause to close-out.

### 9.2 OOM ladder

- The OOM ladder (`OomState`/`OomLadder`, `loop.py`) operates **inside** `train_step`, halving micro-batch/effective-K on `torch.cuda.OutOfMemoryError`. It is orthogonal to the eval-boundary plateau ladder.
- An eval-time OOM (`_eval_epoch`'s `RuntimeError "eval OOM"` branch, `trainer.py:317-324`) is swallowed and **does not** call `_maybe_save_best` or `ladder.observe` — so an eval OOM advances **neither** patience counter (§10). The eval is simply skipped, exactly as today.
- `oom_state.pending_oom_events` continue to flow through `EvalArtifacts.oom_events` unchanged; `close_out` threads them into the returned artifacts/bundle exactly as the inline block does today.

---

## 10. Error handling

The governing rule: **a skipped or failed eval must not advance either patience counter, and must not fire an LR cut.** `_eval_epoch` already swallows eval failures (`trainer.py:317-328`) and returns **before** `_maybe_save_best` when `report` is unavailable. The ladder hook sits **after** `_maybe_save_best`, inside the same successful-eval path. Concretely:

- An eval that raises (OOM at bs=1, or any `Exception`) is logged and swallowed *inside the `try`*; control never reaches `_maybe_save_best` or `ladder.observe`. Counters are untouched; the plateau scheduler is not stepped. This is the existing behavior — the ladder must not change it. The implementer must place the `ladder.observe(...)` call **inside** the `try`, *after* `_maybe_save_best`, so a `report`-producing eval is required to tick the ladder, and any exception short-circuits the tick.
- A `report` that lacks an `mAP` key (`report.overall.get("mAP") is None`) is treated as a skipped tick: `_maybe_save_best` already early-returns on a missing metric; `ladder.observe` must likewise no-op (return `StopDecision(should_stop=False, ...)`) when `mAP` is `None`. No counter advances.
- `close_out`'s `load_adapter(best)` failure: if restoring `best/` raises, log a warning and finalize on the current (last-step) in-memory weights, noting it in the summary — never crash the close-out. (Mirrors `_maybe_save_best`'s swallow-and-continue discipline.)
- The finalize entry's model rebuild failure (missing `config.yaml`, missing checkpoint, peft-method mismatch) exits non-zero with a clear message (consistent with `find_latest_checkpoint`/`load_full_state` error UX).

---

## 11. Finalize-a-paused-run entry

### 11.1 CLI surface — decision

**Decision:** add a `--finalize` flag to the existing `run` command (not a new `finalize` subcommand). Justification:

- `run` already owns the "produce the full set of shippable artifacts (adapter, merged, metrics, bundle)" responsibility and the phase composition in `_orchestrate`. Finalize is exactly that pipeline **minus training**, so it belongs on `run` as a mode, not a sibling command.
- `run --finalize --resume __latest__` reads naturally as "finalize the latest paused run," reusing the existing `--resume <ckpt>`/`__latest__` resolution (`run_cmd.py:250-260`) verbatim — no new resolution logic.
- A separate `finalize` subcommand would duplicate the `--config`/`--resume`/`--progress` option surface and the model-rebuild wiring. The cli-design boundary (thin command shell, logic in a helper) is honored by routing `--finalize` to a `_finalize` helper alongside `_orchestrate`.

The flag:

```python
finalize: bool = typer.Option(
    False, "--finalize",
    help="Finalize a paused (time-limited) run: rebuild the model from --resume's "
         "checkpoint, restore the best weights, run eval, and write adapter/merged/"
         "metrics/bundle. Runs NO training. Requires --resume.",
),
```

Validation: `--finalize` requires `--resume` (a checkpoint or `__latest__`); error out clearly if absent. `--time-limit` is rejected with `--finalize` (no training to time-box).

### 11.2 `_finalize` helper

In `run_cmd.py`, parallel to `_orchestrate`:

```text
def _finalize(cfg, resume: Path, mode, *, visualize, config_path) -> int:
    run_dir = resume.parent.parent          # checkpoints/step_N → run_dir
    # 1. Load the run's own config.yaml for fidelity to the paused run.
    saved_cfg = load_config(run_dir / "config.yaml")   # authoritative for this run
    # 2. Rebuild base model + adapter.
    wrapper = load_sam31(saved_cfg.model, channels=..., channel_semantics=...)
    #    Prefer best/, else the resumed checkpoint's adapter.
    adapter = run_dir / "best" / "adapter" if (run_dir/"best"/"adapter").is_dir() else resume / "adapter"
    load_adapter(wrapper, adapter)
    # 3. Rebuild val dataset from the saved val_source.json (same as _orchestrate).
    vs = load_val_source(run_dir); val_ds = _build_val_dataset(saved_cfg, vs) if vs.mode != "none" else None
    # 4. close_out on best weights → writes adapter/merged/metrics; returns artifacts.
    artifacts = close_out(run_dir, wrapper, saved_cfg, evaluator_val_ds=val_ds,
                          oom_state=None, final_step=<from ckpt>, final_epoch=<from ckpt>,
                          ladder_events=None)
    # 5. Bundle from artifacts.per_example_iou (reuse _orchestrate's BundleContext assembly) + done message.
    return 0
```

Key points:

- **Consumes Phase-2 `close_out`** — no finalize-specific finalize logic; it is the same best-restoration + eval + write path.
- The run's **own** `config.yaml` is the source of truth (so the finalized artifacts match the paused run's config, not whatever `--config` the user passes — `--config` is still required by `run` for option parsing but the saved config governs model/eval shape). **Assumption A6:** when the passed `--config` and the saved `config.yaml` disagree, the saved config wins for model/eval/export; a warning is logged. This mirrors the `cfg_hash` mismatch philosophy.
- `final_step`/`final_epoch` come from the resumed checkpoint's `training_state` (`global_step`/`epoch`), read via a lightweight load (or from `best.json`'s `global_step` when finalizing on best).
- No optimizer/scheduler is built; no `Trainer.fit` is called — strictly inference + write.

### 11.3 Cuttability

This entry is its **own plan phase (Phase 3)** and is **cuttable**: Phases 1–2 deliver the ladder, best-as-final close-out, and resume persistence — a complete feature. The finalize entry is the convenience that turns a #198 pause into a shippable artifact without a manual resume-to-completion. If cut, a user can still finalize a paused run by resuming it to completion (which now closes out on best). The handshake is *served* by Phase 3 but not *blocked* by its absence.

---

## 12. Plan shape — phases & interface contracts

Three feature blocks, each independently reviewable, with explicit contracts at the boundaries.

### Phase 1 — Ladder in the trainer

**In scope:** `lr_schedule: plateau` enum + flip; `lr_decay_on_plateau` + `early_stop` config blocks in `schema.py` (with the exact `# cite:`/`# tbd:` comments); scheduler split (warmup→`ReduceLROnPlateau`, per-eval vs per-step stepping, `param_groups` LR read for logging); `LadderState` + `StopDecision` + `_EarlyStop`; rung-1 cuts (via `scheduler.step(mAP)`) + rung-2 stop counter; val-fallback to cosine; the eval-tick wiring in `_eval_epoch` (after `_maybe_save_best`, inside the `try`); logging/telemetry (log each LR cut and the stop with the triggering eval step + mAP); demoted-seam absence check (§5.5).

**Interface out (consumed by Phase 2):**

- `_EarlyStop(step, epoch, reason)` raised from `run_epoch`; `fit()` catches it around the epoch loop (alongside `_TimeLimitReached`).
- `LadderState` with `state_dict()`/`load_state_dict()` and a `best` field; `LadderEvents` accumulator (cuts + stop reason).
- `best/` saving via `_maybe_save_best` is **unchanged** (still strict `>`); the ladder reads the same eval but owns its own `best` baseline.
- A stop signal the `fit()` loop honors, distinct from the time-limit pause.

### Phase 2 — Resumable state + best-as-final close-out

**In scope:** persist/restore `ladder` + `best_metric_value` in `training_state` (`save_full_state`/`load_full_state`/`ResumeState`); the clobber-bug fix (re-seed `_best_metric_value` and `ladder.best` from `best.json` + persisted value); extract `close_out(run_dir, model, cfg, ...)`; wire it into early stop + normal completion (replacing `trainer.py:581-621`); reflect best-as-final in `EvalArtifacts` (final_metrics now = best eval; add a `final_weights` indicator if needed), plus `per_example_iou` from close_out's single eval (so the `run` bundle reuses it — no second eval), `metrics.json` (`final_weights`, ladder events), and the bundle/`summary.md` (final adapter is the best checkpoint; surface cuts + stop reason via `BundleContext.ladder_events`).

**Interface in (from Phase 1):** `_EarlyStop`, `LadderState`, `LadderEvents`.

**Interface out (consumed by Phase 3):**

- `close_out(run_dir, model, cfg, *, evaluator_val_ds, oom_state, final_step, final_epoch, ladder_events=None) -> EvalArtifacts` — the single best-restoration + eval + write function; its one eval (`return_per_example_iou=True`) populates `EvalArtifacts.per_example_iou` for callers' bundles.
- `EvalArtifacts` semantics: `checkpoint_path = run_dir/adapter` now holds the **best** weights; `final_metrics` is the **best** eval (or `None` no-val); `per_example_iou` carries the bundle's IoU data (or `None` no-val).

### Phase 3 — Finalize-a-paused-run entry

**In scope:** `run --finalize` flag + `_finalize` helper: model rebuild from a run's checkpoint (saved `config.yaml`, base model, `load_adapter` of best/latest), val rebuild from `val_source.json`, `close_out` call, bundle, done message. No training. Validation (`--finalize` requires `--resume`, rejects `--time-limit`).

**Interface in (from Phase 2):** `close_out(...)`.

---

## 13. Module & call-site summary

| Change | Location |
| --- | --- |
| `LRSchedule` += `"plateau"`; `lr_schedule` default → `plateau` | `config/schema.py` |
| `LrDecayOnPlateauConfig`, `EarlyStopConfig` models + mount on `TrainHyperparams` | `config/schema.py` |
| Confirm demoted `early_stop_p_threshold` absent (removed in #88) | `config/schema.py` |
| `LadderState`, `StopDecision`, `LadderEvents`, `LrCut` | `train/ladder.py` (new) |
| `_EarlyStop` exception | `train/loop.py` |
| `should_stop_early` callback param; raise `_EarlyStop` after `on_eval` | `train/loop.py` (`run_epoch`) |
| mode-aware per-step stepping helper; `param_groups[0]["lr"]` LR read | `train/loop.py` (`train_step`, scalar window), `train/_scheduler.py` (new) |
| `_build_scheduler` branch (plateau → `ReduceLROnPlateau`); effective-schedule val-fallback | `train/trainer.py` |
| ladder construction; eval-tick in `_eval_epoch`; catch `_EarlyStop`; call `close_out` on early-stop + normal completion; clobber-bug re-seed | `train/trainer.py` (`fit`, `_eval_epoch`) |
| `close_out(run_dir, model, cfg, ...)` | `train/close_out.py` (new) |
| `ladder` + `best_metric_value` in payload; `ResumeState` fields; new args | `train/checkpoint.py` (`save_full_state`/`load_full_state`/`ResumeState`) |
| `EvalArtifacts` best-as-final semantics (+ optional `final_weights`, `per_example_iou`) | `eval/_artifacts.py` |
| `BundleContext.ladder_events`; summary best-adapter + ladder-event lines | `runs/bundle.py` |
| `run --finalize` flag + `_finalize` helper | `cli/run_cmd.py` |
| Doc rows | `docs/config-schema.md`, `docs/defaults-provenance.md` |

---

## 14. Testing strategy

All tests CPU-only, consistent with the repo's GPU-vs-CPU policy; exercise on the tiny-stub fixtures the seam/time-limit tests use.

### 14.1 Counter & staircase (`tests/train/test_ladder.py`)

- **Improvement resets both counters.** Feed mAPs `[0.5, 0.6, 0.7]` (each > prev + `min_delta`): `evals_without_improvement` stays 0; no cut; no stop.
- **Rung-1 staircase.** With `patience=5`, feed 5 non-improving evals → exactly one LR cut (`×factor`); assert `param_groups[0]["lr"]` dropped by `factor`. Assert the `ReduceLROnPlateau` reset its `num_bad_epochs` after the cut (next non-improving eval restarts the rung-1 window).
- **Rung-2 independence.** Rung-2 counter does **not** reset on the rung-1 cut; with `stop_patience=10`, the stop fires at the 10th non-improving eval regardless of the cut at the 5th.
- **One-cut-before-stop.** With the shipped defaults (`patience=5`, `stop_patience=10`, base LR `1e-4`), assert exactly one cut to `1e-5` then a stop at eval 10 (the §5.6 ladder).
- **`min_lr` floor.** Repeated cuts never push LR below `min_lr`.
- **`min_delta` boundary.** A change of exactly `min_delta` is **not** an improvement (strict `>`); a change just above it is.
- **Shared improvement single-sourcing.** With `early_stop.enabled=false` but `lr_schedule=plateau`, rung-1 cuts still use `min_delta`/`mAP` from `early_stop` (the wart, §5.4) — assert a cut fires on the same plateau a stop-enabled run would cut on.

### 14.2 Val-fallback (`tests/train/test_plateau_val_fallback.py`)

- `lr_schedule=plateau` + `val_ds=None` → `_build_scheduler` builds a `cosine` `LambdaLR` (assert scheduler type), a fallback warning is logged, and the run completes normally (no ladder ticks, no early stop). `config.yaml` still echoes `plateau`.

### 14.3 Scheduler mechanics (`tests/train/test_plateau_scheduler.py`)

- **Per-step warmup, per-eval cut.** In plateau mode, over the first `warmup_steps` steps the LR ramps linearly (assert `param_groups` LR at a mid-warmup step); after warmup it holds; `scheduler.step(mAP)` is called only at eval boundaries (assert via a spy that `train_step` does **not** call the plateau scheduler's `step`).
- **Non-plateau unchanged.** `cosine`/`linear`/`constant` still step the `LambdaLR` per step; LR read via `param_groups[0]["lr"]` matches `get_last_lr()` (regression that the logging change is value-preserving).

### 14.4 Resume round-trip of ladder state (`tests/train/test_ladder_resume.py`)

- Run a few evals to advance `evals_without_improvement` and fire one cut; checkpoint; `load_full_state` restores `evals_without_improvement`, `ladder.best`, the `ReduceLROnPlateau`'s `num_bad_epochs`/`best`/`cooldown_counter`, and `_best_metric_value`. Assert continuing the run cuts/stops at the same absolute eval count as an uninterrupted run.
- **Old-checkpoint compatibility.** A `training_state` payload lacking `ladder`/`best_metric_value` (pre-#197) loads without error; `rs.ladder is None`; the trainer re-seeds from `best.json`.

### 14.5 Clobber-bug regression (`tests/train/test_best_clobber_regression.py`)

- Save `best/best.json` with mAP `0.7`. Resume; the first post-resume eval reports mAP `0.5`. Assert `best/` is **not** overwritten (its mAP stays `0.7`) and `_best_metric_value == 0.7` (re-seeded). The pre-fix behavior (`-inf` reset → `0.5` clobbers `0.7`) must fail this test.

### 14.6 close_out best-restoration (`tests/train/test_close_out.py`)

- With a `best/adapter` holding distinguishable weights and the in-memory model holding different (last-step) weights, `close_out` restores `best/` into the model, writes `run_dir/adapter` equal to `best/adapter`, runs eval on the restored weights, and `metrics.json` carries `final_weights="best"` + ladder events.
- **Fallback.** No `best/` → `close_out` finalizes on last-step weights, `metrics.json` carries `final_weights="last_step"`, summary notes it; no crash.
- **`load_adapter(best)` failure** → swallowed; finalize on last-step weights; warning logged.

### 14.7 Early-stop integration (`tests/train/test_early_stop_integration.py`)

- A run whose injected eval mAPs plateau triggers `_EarlyStop` at the expected step; `fit()` returns `EvalArtifacts` with `checkpoint_path = run_dir/adapter` equal to the best adapter, `final_metrics` = the best eval, `metrics.json` present with the stop reason. The epoch loop stopped before `cfg.train.epochs`.
- **Eval-failure does not tick.** An eval that raises (monkeypatched `Evaluator.evaluate` to raise) does **not** advance either counter — a run that would stop after N real evals does not stop early when M of them failed (assert the stop fires N real evals later).

### 14.8 Finalize entry (`tests/cli/test_finalize.py`)

- `run --finalize --resume <ckpt>` (patched model/eval) rebuilds from the run's `config.yaml`, restores best, calls `close_out` (assert via patch), writes adapter/metrics/bundle, runs **no** training (assert `run_train`/`Trainer.fit` not called), exits 0.
- `--finalize` without `--resume` exits non-zero with a clear message.
- `--finalize` with `--time-limit` is rejected.
- `run --finalize --resume __latest__` resolves via `find_latest_checkpoint`.

### 14.9 #198 + OOM non-regression

- The existing time-limit tests stay green: a `_TimeLimitReached` still routes to `_time_limited_artifacts` (no `close_out`, no eval), and the flushed `training_state` now also carries `ladder`/`best_metric_value` (assert present, but the pause path does not read them).
- Existing OOM-ladder tests stay green: eval-time OOM is swallowed and does not tick the ladder.
- The `EvalArtifacts` seam test (`tests/integration/test_trainer_evaluator_seam.py`) stays green (any new optional field defaults safely).

---

## 15. Documentation

- **`docs/config-schema.md`** — `train.lr_schedule` row: add `"plateau"` to the type union and change the default to `"plateau"` (note plateau is the new default; cosine/linear/constant remain). Add a `train.lr_decay_on_plateau` sub-block (rows for `patience`, `factor`, `min_lr`) and a `train.early_stop` sub-block (rows for `enabled`, `monitor`, `min_delta`, `stop_patience`) under **Advanced fields**, each with the cite/`# tbd:` provenance summarized. Document the §5.4 wart (`monitor`/`min_delta` shared by both rungs even when `early_stop.enabled=false`). Note that `plateau` requires a val set and falls back to `cosine` without one.
- **`docs/defaults-provenance.md`** — update the `train.lr_schedule` row (line 89): default `cosine` → `plateau`, basis += the ReduceLROnPlateau/early-stop pairing + the research §3–§4 horizon-mismatch argument + `# tbd: #197` for the flip; keep the SGDR cite for the (still-available) cosine shape. Add rows for the six new knobs (`lr_decay_on_plateau.{patience,factor,min_lr}`, `early_stop.{min_delta,stop_patience}`; `enabled`/`monitor` need only a brief row) with the §5.6 citations cross-linking the research notes. Update the `config_full.yaml:train.lr_schedule` cross-link row (line 206) to `plateau`.
- **CLI docs** — document `run --finalize` (purpose, requires `--resume`, runs no training, writes best-as-final artifacts) wherever the `run`/`train` flags are documented (`docs/config-schema.md` CLI section or the README). Note that the normal `run`/`train` paths now close out on the **best** checkpoint.

---

## 16. Acceptance criteria

Mapped from issue #197:

1. **Early stop on by default.** `early_stop.enabled` defaults to `true`, monitoring `mAP` at every `eval_every` boundary, and the best checkpoint is restored as the final adapter (`run_dir/adapter`) on stop. (§5.3, §7)
2. **LR-decay rung implemented.** `lr_decay_on_plateau` rung 1 reduces LR by `factor` after `patience` non-improving evals, floored at `min_lr`, via `ReduceLROnPlateau` in the new default `plateau` schedule mode. (§5.2, §6)
3. **Two-counter semantics.** Rung 1 (reset on cut → staircase) and rung 2 (reset only on genuine improvement) are independent, fed one shared `mAP`/`min_delta` improvement test. (§6.3)
4. **plateau is the default schedule**, replacing the per-step cosine; cosine/linear/constant remain; `plateau` without a val set falls back to cosine with a warning. (§5.1, §6.5)
5. **Best-as-final close-out.** A reusable `close_out` restores `best/`, runs eval, writes adapter + optional merged + `metrics.json` (+ bundle) on the best weights — on early stop, normal completion, and finalize; falls back to last-step only when no `best/` exists. (§7)
6. **Finalize entry.** `run --finalize --resume <ckpt>` productionizes a paused run with no training, closing the #198 handshake. (§11)
7. **Resume + clobber fix.** Ladder state (rung-1 `ReduceLROnPlateau` internals, rung-2 counter, best value) persists in `training_state` and restores on `--resume`; `_best_metric_value` is re-seeded from `best.json` so a post-resume eval cannot clobber a better `best/`. (§8)
8. **All new defaults cited or `# tbd:`-tagged**, each with a doc row cross-linking the research notes. (§5.6, §15)
9. **Resume, OOM ladder, and eval seams remain green.** Time-limited stop stays a pure pause (never close-out); eval-time OOM/failure ticks no counter; the OOM ladder is orthogonal. (§9, §10)
10. **Docs updated** for the new config knobs and the `--finalize` CLI surface. (§15)

---

## 17. Assumptions

- **A1 — demoted seam already gone.** `early_stop_p_threshold` lived on the removed `BoxHintSchedule` (never on `TrainHyperparams`) and was deleted in #88; Phase 1 only confirms its absence (no-op). The issue's `schema.py:530` reference is stale (lines shifted post-#88). (§5.5)
- **A2 — save-best vs patience thresholds differ intentionally.** `_maybe_save_best` saves on strict `>` (no `min_delta`); the ladder counts improvement on `> best + min_delta`. A tiny improvement can save a new best yet still count as non-improvement for patience. Intentional. (§6.3)
- **A3 — val-fallback scope.** The plateau→cosine fallback affects only the LR schedule and the (inert) ladder; all other behavior is unchanged without val. (§6.5)
- **A4 — single eval on the `run` path.** `close_out` runs exactly one eval (`return_per_example_iou=True`) on the best adapter; that result serves both `metrics.json` and the bundle's per-example IoU (returned on `EvalArtifacts`). The orchestrator runs no second eval and drops its export-merge phase (close_out merges). (§7.2, §7.4)
- **A5 — scheduler kind is fixed at the first invocation.** The effective schedule (after val-fallback) is persisted as `scheduler_kind` and rebuilt verbatim on every `--resume`; `cfg.train.lr_schedule` edits on resume are ignored for scheduler construction (persisted kind wins, with a warning). This lets the no-val→`cosine` fallback resume cleanly even if a val set is later added. (§6.5, §8.3)
- **A6 — finalize uses the run's saved config.** `_finalize` governs model/eval/export shape from `run_dir/config.yaml`, not the `--config` passed; a mismatch logs a warning. Mirrors the `cfg_hash` philosophy. (§11.2)

End of spec.
