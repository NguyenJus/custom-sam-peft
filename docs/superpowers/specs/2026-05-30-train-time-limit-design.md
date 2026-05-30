# spec/train-time-limit — Wall-clock time limit with resumable graceful stop (issue #198)

**Status:** Draft (2026-05-30)
**Tracking:** [#198](https://github.com/NguyenJus/custom-sam-peft/issues/198) — *Add a wall-clock time limit for train/run with resumable graceful stop*
**Scope:** Add an opt-in `train.time_limit` config knob and a `--time-limit` CLI flag (on both `train` and `run`) that cap a training invocation by wall-clock time. When the budget is hit, training finishes the current micro-step, flushes a resumable full-state checkpoint, skips all end-of-run finalization (no eval, no `adapter/` rewrite, no `metrics.json`), prints a resume message, and exits 0. Unset = unlimited (today's behavior, unchanged).

**Builds on:**
[`2026-05-17-training-loop-design.md`](2026-05-17-training-loop-design.md) (the `run_epoch` step loop, `on_checkpoint` wiring, and `nan_abort_after` abort precedent this spec extends);
[`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) (the thin-shell command shape and runner seam);
[`2026-05-18-simplify-ux-design.md`](2026-05-18-simplify-ux-design.md) (the `run` orchestration phases this spec short-circuits).

---

## 1. Summary

This is a "wire up a clock + a clean exit" feature; all the heavy lifting already exists. A new opt-in budget lets a user say "train for at most 2h30m" instead of guessing an epoch count that fits a rented-GPU window.

The mechanism:

1. `Trainer.fit()` records a monotonic start time and computes an absolute `deadline`.
2. The `deadline` flows down into `run_epoch`, which checks elapsed wall-clock **after each completed micro-step** — the same hook point as the existing `nan_abort_after` check.
3. On expiry, the current step finishes, a full-state checkpoint is flushed immediately (regardless of `save_every`), and a graceful internal exception unwinds back to `fit()`.
4. `fit()` catches it, **skips the entire post-loop finalize block** (no `save_adapter` rewrite, no full eval, no `metrics.json`), and returns time-limited artifacts.
5. The CLI prints a resume message and exits **0**.

The stop is a **pure pause** with **zero evaluation**. The user resumes later (`--resume __latest__`) or takes `run_dir/best/` as-is. `run_dir/best/` is whatever the last periodic lite eval produced — valid when present, never refreshed at stop time.

The budget is **per-run**: each invocation gets a fresh wall-clock budget. `--resume` restarts the clock. There is no checkpoint-format change and no persisted elapsed counter.

---

## 2. Motivation

Today the only stopping controls are `train.epochs` (a hard count) and `train.nan_abort_after` (a failure abort). There is no way to say "train for at most 2 hours." Users on time-boxed GPU rentals have to guess an epoch count that fits their budget; overshoot either wastes money or gets the process killed by the host mid-write, corrupting the in-progress checkpoint.

A self-imposed wall-clock budget that always flushes a clean, resumable checkpoint before exiting solves both: the user controls cost directly, and the stop is never mid-`backward()` or mid-checkpoint-write.

---

## 3. Goals & Non-goals

**Goals.**

- `train.time_limit` config knob on `TrainHyperparams`, accepting a human duration string or bare integer seconds; `None`/unset = unlimited.
- `--time-limit <dur>` CLI flag on **both** `train` and `run`, overriding the config value (dedicated option, mirroring `--resume`; not routed through `--override`).
- A reusable duration-parsing utility (`parse_duration_to_seconds`, `format_seconds`).
- Stop within ~one micro-step of the deadline, even mid-epoch.
- A resumable full-state checkpoint flushed at the stop step regardless of `save_every`.
- `--resume __latest__` cleanly continues from the flushed checkpoint.
- Exit 0 with a clear resume message; `run` short-circuits before its eval/export/bundle phases.
- Zero behavior change when `time_limit` is unset.

**Non-goals.**

- **Finalization of a time-limited stop (→ #197 close-out handoff).** Full eval, export, merged export, bundle, and "take best as-is" finalization are explicitly *not* performed on a time-limited stop. The graceful pause writes only the resumable checkpoint. Productionizing a paused run (turning `best/` into a shippable artifact, or running a final eval/bundle) is tracked by #197 and lives behind the scope handshake already posted there.
- **Cumulative cross-resume time tracking.** The budget is per-invocation. Counting time already spent across resumes would require a persisted elapsed counter in the checkpoint; out of scope (see §13).
- **Host SIGTERM/SIGKILL handling.** This is a self-imposed soft budget, not a signal handler. (The flush-before-exit logic would help a future signal handler, but that wiring is out of scope.)
- **Per-epoch or per-eval budgets.** Only a single overall wall-clock cap.
- **A new default hyperparameter.** The feature is opt-in; unset = unlimited. There is no new default to justify or cite.

---

## 4. Design

### 4.1 Config knob & validation

`train.time_limit` mounts on `TrainHyperparams` in `src/custom_sam_peft/config/schema.py` (alongside the existing opt-in `save_every` / `eval_every` fields at lines 527–545).

```python
time_limit: str | int | None = Field(
    default=None,
    description=(
        "Wall-clock budget for this invocation. Accepts a human duration "
        '("2h30m", "90m", "3600s") or bare seconds (3600). None (default) '
        "means unlimited. The budget is per-run: --resume restarts the clock."
    ),
)
```

It is an **advanced** field (place it after `eval_every`, in the `# --- advanced ---` section).

**Validation.** A `@field_validator("time_limit")` (analogous to the existing validators in this module) runs `parse_duration_to_seconds` (§4.2) on any non-`None` value purely to validate it:

- It must parse to a **strictly positive** integer number of seconds. `0`, negative values, empty string, and malformed strings raise `ValueError` with a clear message naming the bad value (e.g. `time_limit: '10x' is not a valid duration (use e.g. "2h30m", "90m", "3600s", or bare seconds)`).
- **The validator does not rewrite the field.** It keeps the original value exactly as written (`"2h30m"` stays `"2h30m"`, `3600` stays `3600`), so `run_dir/config.yaml` echoes what the user typed. Parsing to seconds for enforcement happens later, in `fit()` (§4.4).

The `_Strict` base forbids extra keys, so a typo like `time_limt` is already rejected.

### 4.2 Duration parsing utility

A new small module `src/custom_sam_peft/config/_duration.py` exposes two pure functions. No such utility exists anywhere in `src/` today.

```python
def parse_duration_to_seconds(value: str | int) -> int:
    """Parse a duration to a strictly-positive integer number of seconds.

    Accepts:
      - bare int (3600) or bare numeric string ("3600")  -> seconds
      - h/m/s combos: "1h", "45m", "30s", "2h30m", "1h5m30s"
    Whitespace around the value is tolerated; units are lowercase h/m/s.

    Raises:
      ValueError: on non-positive results (0, negative), empty/whitespace-only
        strings, or any string that doesn't match the grammar (e.g. "abc",
        "10x", "2h30"  -- a trailing number with no unit is rejected).
    """


def format_seconds(seconds: int) -> str:
    """Render a positive second count as a canonical human string.

    Drops zero components; e.g. 9000 -> "2h30m", 3600 -> "1h", 90 -> "1m30s",
    45 -> "45s". Used so the exit message renders consistently whether the
    user wrote "2h30m" or 9000.
    """
```

**Grammar.** A bare integer or an all-digits string is seconds. Otherwise the value must be a non-empty sequence of `<number><unit>` groups where `unit ∈ {h, m, s}`, each unit appearing at most once and in `h`→`m`→`s` order (matching all accepted forms: `"1h"`, `"45m"`, `"30s"`, `"2h30m"`, `"1h5m30s"`). Any malformed remainder — leftover characters, an out-of-order or repeated unit, or a trailing number with no unit (`"2h30"`) — is rejected. The matched components are summed as `h*3600 + m*60 + s`, and the result must be `> 0`.

**Canonical formatting.** `format_seconds` computes `h, m, s` via integer division and joins the non-zero components in `h`,`m`,`s` order. A zero total never reaches `format_seconds` (rejected upstream), so it always returns a non-empty string.

Both functions are pure (no I/O, no logging) and independently unit-tested (§11).

### 4.3 CLI flags

A `--time-limit <dur>` Typer option is added to **both** commands, as a dedicated option mirroring `--resume`. It is **not** routed through `--override`.

`src/custom_sam_peft/cli/train_cmd.py` — `train(...)`:

```python
time_limit: str | None = typer.Option(
    None,
    "--time-limit",
    help='Wall-clock budget for this run (e.g. "2h30m", "90m", "3600s", or bare seconds). '
    "Overrides train.time_limit. The budget is per-run; --resume restarts the clock.",
    metavar="DURATION",
),
```

`src/custom_sam_peft/cli/run_cmd.py` — `run(...)`: the identical option.

**Resolution.** When `--time-limit` is provided, it **overrides** `cfg.train.time_limit` after `load_config`:

```python
if time_limit is not None:
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"time_limit": time_limit})})
```

The override value flows through the same `TrainHyperparams` validator (§4.1), because `model_copy(update=...)` re-validates only if revalidation is enabled; to guarantee validation of a CLI-supplied string, the CLI calls `parse_duration_to_seconds(time_limit)` up front (inside a `try`) for an early, clear error:

```python
if time_limit is not None:
    try:
        parse_duration_to_seconds(time_limit)
    except ValueError as e:
        rprint(f"[red]error[/red] invalid --time-limit: {e}")
        raise typer.Exit(code=1) from e
    cfg = cfg.model_copy(...)  # as above
```

On a parse/validation error from `--time-limit`, the command exits **1** with a clear message (consistent with how `--resume __latest__` failures exit 1 today).

### 4.4 Clock & enforcement

The clock lives in `Trainer.fit()` (`src/custom_sam_peft/train/trainer.py`, around line 396). After the schedule is resolved and immediately before the epoch loop (currently line ~497), `fit()` computes the deadline:

```python
deadline: float | None = None
budget_seconds: int | None = None
if cfg.train.time_limit is not None:
    budget_seconds = parse_duration_to_seconds(cfg.train.time_limit)
    deadline = time.monotonic() + budget_seconds
```

`time.monotonic()` is used (not `time.time()`) so the budget is immune to wall-clock adjustments. `deadline` is an absolute monotonic instant; `None` means unlimited. `budget_seconds` is hoisted to function scope (initialized `None`) so the time-limited artifacts helper (§4.7) can read it; it is non-`None` exactly when `deadline` is non-`None`, i.e. on every path where a `_TimeLimitReached` can fire.

`deadline` is threaded down through the existing call chain as a new keyword argument:

- `Trainer._train_epoch(..., deadline: float | None = None)` (line 245) passes it to `run_epoch`.
- `run_epoch(..., deadline: float | None = None)` (`src/custom_sam_peft/train/loop.py`, line 443) performs the check.

The check sits at the **micro-step boundary** in `run_epoch`'s loop — the same hook point as the `nan_abort_after` per-step check — *after* the step completes and `global_step` has been incremented (after line 484, alongside the existing `save_every`/`eval_every` boundary checks at lines 495–502):

```python
if deadline is not None and time.monotonic() >= deadline:
    # flush a resumable checkpoint, then raise _TimeLimitReached  (see §4.5, §4.6)
    ...
```

Checking after the step (not before `backward()`) guarantees the current step always finishes cleanly — never an interrupted `backward()` and never a half-applied optimizer step.

### 4.5 Stop mechanism & control flow

A new internal exception signals a graceful stop, analogous to the `nan_abort_after` `RuntimeError` raise at `loop.py:362–363` but **graceful** (exit 0, not 1):

```python
class _TimeLimitReached(Exception):
    """Internal signal: the wall-clock budget expired. Carries the stop point.

    Graceful (exit 0), in contrast to the nan_abort_after RuntimeError which is
    a user-facing failure (exit 1). Never propagates past Trainer.fit().
    """
    def __init__(self, step: int, epoch: int) -> None:
        super().__init__(f"time limit reached at step {step} (epoch {epoch})")
        self.step = step
        self.epoch = epoch
```

It lives in `src/custom_sam_peft/train/loop.py` (next to `run_epoch`), or a small shared `train/types.py` if that reads cleaner; either is acceptable.

**Control flow.**

1. `run_epoch` detects expiry at the micro-step boundary (§4.4).
2. It flushes a full-state checkpoint at the current `global_step` (§4.6).
3. It raises `_TimeLimitReached(global_step, epoch)`. This unwinds out of `run_epoch` → `_train_epoch` → the `for epoch in ...` loop in `fit()`.
4. `fit()` wraps the epoch loop (the body currently spanning lines 497–556, inside the existing `try:` whose `finally:` closes the tracker) so the catch is **around the epoch loop and before the post-loop finalize block**:

```python
stop: _TimeLimitReached | None = None
try:
    for epoch in range(start_epoch, cfg.train.epochs):
        ...
        global_step, nan_streak = self._train_epoch(..., deadline=deadline)
        P.advance_outer()
except _TimeLimitReached as e:
    stop = e
    global_step = e.step  # the flushed checkpoint's step
finally:
    self.tracker.close()

if stop is not None:
    return self._time_limited_artifacts(run_dir, stop, budget_seconds)  # §4.7

# ... existing post-loop finalize block runs ONLY when stop is None ...
```

When `stop is not None`, the entire post-loop finalize block (`save_adapter` to `run_dir/adapter`, optional merged export, the end-of-run full eval, and the `metrics.json` write — currently lines 516–556) is **skipped**. The `finally` still runs (`tracker.close()`), keeping tracker lifecycle correct.

**Explicitly NOT done on a time-limited stop:**

- No lite eval and no full eval (no `Evaluator(...).evaluate(...)` call).
- No rewrite of `run_dir/adapter` (`save_adapter` is skipped — the resumable checkpoint under `checkpoints/step_<N>/` is the only fresh artifact).
- No `metrics.json` write.
- No merged export.
- For `run`: no eval/export/bundle phases (§4.9).

**Contrast with `nan_abort_after`.** Both break out of the loop via an exception raised at the same micro-step hook point. The difference is the disposition: `nan_abort_after` raises a user-facing `RuntimeError` that propagates to the CLI and exits **1** (a failure); `_TimeLimitReached` is caught inside `fit()`, finalization is skipped, and the process exits **0** (a graceful pause).

### 4.6 Checkpoint flush

At the stop point, `run_epoch` flushes a full-state checkpoint immediately, **regardless of `save_every`**, so the stop is always resumable even mid-epoch.

The flush writes the same `checkpoints/step_<N>/` full-state directory that `save_full_state` (`checkpoint.py:147`) produces and `find_latest_checkpoint` (`checkpoint.py:244`) discovers. Two acceptable implementations:

- **Preferred — minimal flush.** Call `on_checkpoint(global_step, epoch, nan_streak)` only if a leaner path is awkward; otherwise call `save_full_state(...)` directly from inside `run_epoch` where `optimizer`, `scheduler`, `global_step`, `epoch`, `nan_streak`, and `cfg` are all in scope. The minimal flush writes the checkpoint **without** the image-panel render (`_log_image_panel`) to keep the stop lean. To do this directly, `run_epoch` needs `optimizer`/`scheduler`/`cfg`/`model` (already parameters) and the `run_dir` (already a parameter), so a direct `save_full_state` call to `paths.checkpoint_path(run_dir, step=global_step).parent / f"step_{global_step}"` is self-contained.
- **Acceptable — reuse `on_checkpoint`.** Calling the existing `on_checkpoint(step, epoch, nan_streak)` closure (`trainer.py:487–490`) also flushes a full-state checkpoint; it additionally renders the image panel via `_maybe_checkpoint` → `_log_image_panel`. This is acceptable but slightly heavier at the stop instant.

The spec prefers the minimal flush (checkpoint only, skip the panel). Either way, the resulting `checkpoints/step_<N>/` directory is byte-for-byte the format `load_full_state` restores from, so resume works unchanged.

**Idempotence note.** If the deadline trips exactly on a `save_every` boundary, the same step may already have been checkpointed by the periodic path moments earlier. `save_full_state` writes into a `step_<N>/` directory keyed by step, so a re-flush at the same step overwrites the same directory harmlessly. No special-casing is needed.

### 4.7 Return / signaling shape

The CLI needs stop metadata to print the message: whether the stop was time-limited, the stop `global_step`, the stop `epoch` and `total_epochs`, the flushed checkpoint dir, the `best/` dir + its mAP (only if `best/` exists), and the formatted duration string.

This is carried **without breaking the `EvalArtifacts` trainer→evaluator seam** by adding a single optional nested field to `EvalArtifacts` (`src/custom_sam_peft/eval/_artifacts.py`) that the evaluator never reads:

```python
@dataclass(frozen=True)
class TimeLimitStop:
    """Set when Trainer.fit stopped early on a wall-clock budget. None otherwise."""
    stop_step: int
    stop_epoch: int          # zero-based epoch index at the stop
    total_epochs: int        # cfg.train.epochs
    checkpoint_dir: Path     # run_dir/checkpoints/step_<N>/
    duration_label: str      # format_seconds(budget_seconds), e.g. "2h30m"
    best_dir: Path | None    # run_dir/best/ if it exists, else None
    best_map: float | None   # best.json "value" if best/ exists, else None


@dataclass(frozen=True)
class EvalArtifacts:
    checkpoint_path: Path
    peft_method: str
    run_dir: Path
    final_metrics: MetricsReport | None = field(default=None)
    oom_events: tuple[OomEvent, ...] = field(default=())
    time_limit_stop: TimeLimitStop | None = field(default=None)   # NEW
```

**Seam-test compatibility (verified).** The seam test (`tests/integration/test_trainer_evaluator_seam.py`) does **not** enumerate or freeze the field set; it never iterates `dataclasses.fields(...)` and never asserts "no extra fields." It asserts on specific fields (`run_dir`, `checkpoint_path`, `peft_method`) and stands up an `Evaluator` independently, confirming the evaluator consumes only `EvalArtifacts` data — it reads none of the new field. Adding an **optional** field with a default therefore keeps every existing seam test green: `Trainer.fit` still returns an `EvalArtifacts`, all asserted fields are unchanged, and the new field defaults to `None` on the normal (non-time-limited) path. No alternative carrier is needed.

**Population.** `Trainer._time_limited_artifacts(run_dir, stop, budget_seconds)` (a new private helper, called from §4.5) builds the `EvalArtifacts`:

- `checkpoint_path` = the flushed `run_dir/checkpoints/step_<stop.step>/adapter` (the resumable artifact; note `run_dir/adapter` is intentionally NOT written on a time-limited stop, so `checkpoint_path` points at the step checkpoint's adapter dir).
- `peft_method`, `run_dir` = as today.
- `final_metrics = None` (no eval ran), `oom_events` = accumulated as today.
- `time_limit_stop` = a `TimeLimitStop` with `stop_step`, `stop_epoch`, `total_epochs=cfg.train.epochs`, `checkpoint_dir=run_dir/checkpoints/step_<stop.step>/`, `duration_label=format_seconds(budget_seconds)`.
  - `best_dir` / `best_map`: if `run_dir/best/` exists, read `run_dir/best/best.json` (written by `_maybe_save_best`, `trainer.py:340–348`, shape `{"metric": "mAP", "value": <float>, "global_step": <int>}`) and set `best_dir = run_dir/best/`, `best_map = value`. If `best/` is absent or `best.json` is unreadable, both stay `None`.

### 4.8 Exit UX & message

Both `train` and `run` print a clear message and exit **0** on a time-limited stop. The CLI detects the stop via `result.time_limit_stop is not None`.

When `best/` exists:

```text
⏱  Time limit (2h30m) reached at step 4120 (epoch 3/10).
   Checkpoint saved: runs/<id>/checkpoints/step_4120/
   Best so far:      runs/<id>/best/ (mAP 0.612)

   • Resume:            custom-sam-peft <train|run> --config <cfg> --resume __latest__
   • Use best as-is:    runs/<id>/best/adapter/
```

When `best/` does **not** exist (budget tripped before the first `eval_every` boundary, or no val set), the "Best so far" and "Use best as-is" lines are **omitted** — only the resume command shows:

```text
⏱  Time limit (90m) reached at step 312 (epoch 0/10).
   Checkpoint saved: runs/<id>/checkpoints/step_312/

   • Resume:            custom-sam-peft train --config configs/run.yaml --resume __latest__
```

**Message construction rules.**

- `(2h30m)` is `time_limit_stop.duration_label` (from `format_seconds`), so a config of `9000` and a config of `"2h30m"` render identically.
- `step 4120` is `stop_step`; `(epoch 3/10)` is `stop_epoch + 1` of `total_epochs` (the zero-based epoch index rendered 1-based for the user, matching the `nan_abort`/progress convention of human-facing 1-based epochs).
- The **resume command names the invoked subcommand** (`train` vs `run`) and the **actual `--config` path** the user passed. The CLI command knows both: each command formats its own subcommand token and substitutes the `config` argument it received.
- `mAP 0.612` is `best_map` formatted to 3 decimals; the line is present only when `best_dir is not None`.
- Paths are rendered relative to the cwd when under it (matching the existing `done run_dir=...` style), else absolute.

A small shared formatter (e.g. `cli/_time_limit.py::format_time_limit_message(stop, *, subcommand, config_path) -> str`) keeps `train` and `run` consistent and is unit-testable without invoking Typer. The CLI prints it via `rprint` and returns/exits 0.

### 4.9 `run` short-circuit

`_orchestrate` (`src/custom_sam_peft/cli/run_cmd.py:70`) runs phases train → eval → export-merge → bundle. After the train phase (line 85, `train_result = run_training(...)`), it must detect a time-limited stop and **short-circuit before any other phase**:

```python
train_result = run_training(cfg, resume_from=resume)
if train_result.time_limit_stop is not None:
    rprint(format_time_limit_message(train_result.time_limit_stop, subcommand="run", config_path=config))
    return 0
```

This runs before `load_val_source`, `load_sam31`/`load_adapter`, the eval phase, the export-merge phase, and `write_bundle` — so a time-limited stop produces **no** bundle, **no** eval, **no** merged export. `run` exits 0 with the same message shape as `train` (subcommand token = `run`).

`train_cmd.py`'s `train(...)` detects the stop right after `run_train` returns and before the optional `--eval` / `--export` post-steps, printing the message and exiting 0 (so `--eval`/`--export` are also skipped on a time-limited stop).

---

## 5. Module & call-site summary

| Change | Location |
| --- | --- |
| `parse_duration_to_seconds`, `format_seconds` | `src/custom_sam_peft/config/_duration.py` (new) |
| `time_limit` field + validator | `src/custom_sam_peft/config/schema.py` (`TrainHyperparams`) |
| `--time-limit` option + override + early-validate | `src/custom_sam_peft/cli/train_cmd.py`, `cli/run_cmd.py` |
| `deadline` computation; catch `_TimeLimitReached`; skip finalize; `_time_limited_artifacts` | `src/custom_sam_peft/train/trainer.py` (`fit`, `_train_epoch`) |
| `deadline` param; post-step check; flush + raise `_TimeLimitReached` | `src/custom_sam_peft/train/loop.py` (`run_epoch`) |
| `_TimeLimitReached` exception | `src/custom_sam_peft/train/loop.py` (or `train/types.py`) |
| `TimeLimitStop` dataclass + `time_limit_stop` field | `src/custom_sam_peft/eval/_artifacts.py` |
| Message formatter | `src/custom_sam_peft/cli/_time_limit.py` (new) |
| `run` short-circuit | `src/custom_sam_peft/cli/run_cmd.py` (`_orchestrate`) |
| Doc rows | `docs/config-schema.md`, `docs/defaults-provenance.md`, CLI-flags doc |

---

## 6. Edge cases

- **`best/` absent at stop.** Budget tripped before the first `eval_every` boundary, or no val set → `_maybe_save_best` never ran. `best_dir`/`best_map` are `None`; the message omits the "Best so far" and "Use best as-is" lines (§4.8).
- **Deadline trips on a `save_every` boundary.** The periodic checkpoint and the stop flush target the same `step_<N>/` dir; the re-flush overwrites it harmlessly (§4.6).
- **Deadline already past at loop entry** (tiny budget, or slow setup). The first micro-step still completes (the check is post-step), then the stop fires on the first check. The stop is therefore always at `global_step >= start_step + 1` — at least one step runs. This is the "stop within ~one step" guarantee.
- **Unset budget.** `deadline is None`; the per-step check is a single `is not None` short-circuit with no `time.monotonic()` call on the unlimited path. Strict no-op: byte-for-byte the current behavior.
- **`--time-limit` plus `train.time_limit` both set.** The CLI flag wins (overrides the config value). The config-written `config.yaml` reflects the override because the CLI mutates `cfg.train.time_limit` before `fit()` writes `config.yaml`.
- **Pending un-stepped grad-accum micro-batches at stop.** Discarded (§7). Safe because resume re-walks the epoch.
- **Resume after a time-limited stop.** `--resume __latest__` finds the flushed `step_<N>/`, restores optimizer/scheduler/RNG, restarts at `start_epoch`, re-walks the epoch with RNG restored, and gets a **fresh** budget (per-run). No persisted elapsed counter exists.

---

## 7. Granularity rationale

The check fires at the **micro-step boundary**, not at an optimizer-step (grad-accumulation) boundary. With gradient accumulation, the stop may land between optimizer steps, discarding any pending (un-stepped) accumulated gradients. This is **safe and intentional**, and it resolves issue #198's open question #4:

Resume is **epoch-boundary** (`checkpoint.py` docstring lines 7–9): on resume, `fit()` restarts at `start_epoch` and re-walks the *entire* epoch from the top with RNG restored. The exact micro-step (or optimizer-step) at which the stop occurred is therefore irrelevant to correctness — the resumed run reconstructs the same data order and reprocesses the epoch from its start. Discarded partial gradients are simply never-committed work that the resumed epoch redoes. There is no need to stop on an optimizer-step boundary, and no need to persist sub-epoch progress.

This is the same property that lets a mid-epoch `nan_abort_after` checkpoint resume cleanly.

---

## 8. Acceptance criteria

Adapted from issue #198, adjusted for the locked zero-eval pause (so `run_dir/best/` is "valid when present" rather than guaranteed refreshed at stop time):

1. **Config + CLI.** `train.time_limit` exists on `TrainHyperparams`; `--time-limit <dur>` exists on **both** `train` and `run`, overriding the config value (dedicated option, not via `--override`).
2. **Duration formats.** `"2h30m"`, `"90m"`, `"3600s"`, bare `3600`, bare `"3600"`, and combos like `"1h5m30s"` all parse; `0`, negatives, `""`, `"abc"`, `"10x"` are rejected with a clear error.
3. **Unset = unlimited.** No `time_limit` → today's behavior, unchanged (no new default to justify).
4. **Responsive stop.** Training stops within ~one micro-step of the deadline, even mid-epoch; the current step always finishes cleanly (never mid-`backward()`).
5. **Resumable flush.** A full-state checkpoint is flushed at the stop step **regardless of `save_every`**, under `run_dir/checkpoints/step_<N>/`.
6. **Resume continues.** `--resume __latest__` cleanly continues from the flushed checkpoint (fresh per-run budget).
7. **Best usable when present.** `run_dir/best/` (+ `best.json` mAP) is valid and usable as-is when it exists; it is not refreshed at stop time and may be absent.
8. **Graceful exit.** Exit code **0** with a clear message naming the resume command (and the best path/mAP when `best/` exists; omitting both best lines when absent). The resume command names the invoked subcommand and the actual `--config`.
9. **`run` short-circuits.** A time-limited stop produces no bundle, no eval, no export.
10. **No finalize on stop.** No full eval, no `metrics.json`, no `run_dir/adapter` rewrite, no merged export on a time-limited stop.
11. **Seam intact.** The `EvalArtifacts` seam test stays green with the added optional field.
12. **Tests.** All behaviors in §11 are covered; `ruff`, `mypy --strict`, and `pytest` are clean.

---

## 9. Exit UX message — reference

Full template (both lines present when `best/` exists):

```text
⏱  Time limit (2h30m) reached at step 4120 (epoch 3/10).
   Checkpoint saved: runs/<id>/checkpoints/step_4120/
   Best so far:      runs/<id>/best/ (mAP 0.612)

   • Resume:            custom-sam-peft <train|run> --config <cfg> --resume __latest__
   • Use best as-is:    runs/<id>/best/adapter/
```

Reduced template (`best/` absent — omit both best lines):

```text
⏱  Time limit (90m) reached at step 312 (epoch 0/10).
   Checkpoint saved: runs/<id>/checkpoints/step_312/

   • Resume:            custom-sam-peft train --config configs/run.yaml --resume __latest__
```

---

## 10. Documentation

- `docs/config-schema.md`: add a `train.time_limit` row to the **Advanced fields** table — Type `str | int | None`, Default `None` (unlimited), Layer `advanced`, with the human-duration / bare-seconds note. No "YAGNI rationale" citation is needed beyond "opt-in; unset = unlimited" because there is no default value to justify.
- `docs/defaults-provenance.md`: no new default-value row is required (the default is `None`/unset, not a chosen number). Optionally note in the relevant section that `time_limit` is intentionally opt-in with no default.
- CLI-flags doc (the file updated in commit "docs: document eval.visualize / visualize_count knobs and CLI flags"): document `--time-limit` on `train` and `run`, including the per-run budget semantics and that it overrides `train.time_limit`.

---

## 11. Testing

All tests CPU-only, consistent with the repo's GPU-vs-CPU testing policy. The stop path is exercised with a near-immediate injected deadline on the existing tiny-stub fixtures.

### 11.1 Duration parser (`tests/config/test_duration.py`)

`parse_duration_to_seconds`:

- Accepts and returns correct seconds: `"2h30m"` → 9000, `"90m"` → 5400, `"3600s"` → 3600, bare `3600` → 3600, bare `"3600"` → 3600, `"1h"` → 3600, `"45m"` → 2700, `"30s"` → 30, `"1h5m30s"` → 3930.
- Rejects with `ValueError`: `0`, `-1`, `""`, whitespace-only, `"abc"`, `"10x"`, `"2h30"` (trailing number, no unit), `"-2h"`.

`format_seconds`:

- `9000` → `"2h30m"`, `3600` → `"1h"`, `5400` → `"1h30m"` (canonical collapses to the largest units, so a `"90m"` input renders as `"1h30m"`), `90` → `"1m30s"`, `45` → `"45s"`.
- Round-trip sanity: `parse_duration_to_seconds(format_seconds(n)) == n` for a representative set of positive `n`.

### 11.2 Config schema (`tests/config/test_schema_time_limit.py`)

- `time_limit=None` is valid (default).
- `time_limit="2h30m"`, `=3600`, `="3600"` are valid and **stored verbatim** (the field equals exactly what was passed; not normalized to seconds).
- `time_limit=0`, `=-5`, `="abc"`, `="10x"`, `=""` raise `ValidationError` with a message naming the bad value.

### 11.3 Stop trigger & checkpoint (`tests/train/test_time_limit_stop.py`)

Uses the tiny-stub wrapper + `_TinyDataset` pattern from the seam test, with a config carrying a near-immediate budget (e.g. inject a deadline already in the past, or `time_limit` of effectively one step via monkeypatched `time.monotonic`).

- The epoch loop breaks within ~one micro-step; `global_step` advanced by at least 1.
- A full-state checkpoint exists at `run_dir/checkpoints/step_<N>/` (with `adapter/` + training-state file), even though `save_every` is large (no periodic checkpoint would have fired).
- `Trainer.fit` returns `EvalArtifacts` with `time_limit_stop is not None`, correct `stop_step`, `total_epochs`, `duration_label`.
- `metrics.json` is **not** written; `run_dir/adapter` is **not** written (only the step checkpoint's adapter exists).
- `final_metrics is None`.

### 11.4 Resumability (`tests/train/test_time_limit_resume.py`)

- After a time-limited stop, `find_latest_checkpoint(cfg)` returns the flushed `step_<N>/`.
- A second `Trainer.fit(resume_from=<that dir>)` (no/large budget) loads cleanly, restarts at `start_epoch`, and runs to completion — assert it finishes without error and produces `run_dir/adapter` + `metrics.json` on the resumed run.

### 11.5 Unset budget no-op (`tests/train/test_time_limit_noop.py`)

- A config without `time_limit` (the default) runs the full epoch loop and post-loop finalize exactly as today — assert `run_dir/adapter`, `metrics.json` written, `time_limit_stop is None`. (Existing trainer/seam tests, which never set `time_limit`, must stay green unchanged.)

### 11.6 Exit message (`tests/cli/test_time_limit_message.py`)

- `format_time_limit_message(stop, subcommand="train", config_path=...)` contains the resume command `custom-sam-peft train --config <cfg> --resume __latest__`.
- With `best_dir`/`best_map` set: contains `Best so far`, the `best/` path, the formatted mAP, and `Use best as-is`.
- With `best_dir=None`: omits both best lines; still contains the resume command.
- `subcommand="run"` renders `custom-sam-peft run ...`.
- Duration rendering is from `format_seconds` (a `9000`-second stop and a `"2h30m"` stop produce identical message text).

### 11.7 CLI integration (`tests/cli/test_time_limit_cli.py`)

- `--time-limit "10x"` (malformed) on both `train` and `run` exits **1** with a clear error and never starts training (assert via monkeypatched `run_train`/`run_training` not called).
- `--time-limit "2h"` overrides `cfg.train.time_limit` (assert the cfg passed to the patched runner has `train.time_limit == "2h"`).
- `train` and `run`: when the patched runner returns an `EvalArtifacts` with `time_limit_stop` set, the command prints the message and exits **0**; for `train`, `--eval`/`--export` post-steps are skipped; for `run`, no eval/export/bundle functions are invoked (assert via patched phase functions).

### 11.8 Seam test (`tests/integration/test_trainer_evaluator_seam.py`)

- Existing tests stay green unchanged (the new field defaults to `None`).
- Add one assertion: on the normal (non-time-limited) path, `artifacts.time_limit_stop is None`.

---

## 12. Out of scope

- **#197 close-out (finalization of a paused run):** full eval, export, merged export, bundle, and "take best as-is" finalization. The time-limited pause writes only the resumable checkpoint; productionizing it is #197.
- **Cumulative cross-resume time tracking:** the budget is per-invocation; no persisted elapsed counter (would be a checkpoint-format change).
- **Host SIGTERM/SIGKILL handling:** self-imposed soft budget only.
- **Per-epoch / per-eval budgets:** single overall wall-clock cap only.

(See the scope-handshake comment already posted on #197.)

---

## 13. Assumptions

1. **Per-run budget is the locked semantics.** `--resume` restarts the clock; there is deliberately no persisted elapsed counter and therefore no checkpoint-format change. This matches "I have N hours right now" and keeps the checkpoint format frozen.
2. **The added `EvalArtifacts.time_limit_stop` field is seam-safe.** Verified against `tests/integration/test_trainer_evaluator_seam.py`: the test asserts on named fields and runs an independent `Evaluator`; it does not freeze the field set or forbid extras. An optional field with a default is tolerated. No alternative carrier is needed. *(Double-check: if a future seam test is added that does enumerate `dataclasses.fields(...)`, it must allowlist `time_limit_stop` — flag this when touching the seam test.)*
3. **`checkpoint_path` on a time-limited stop points at the step checkpoint's adapter** (`run_dir/checkpoints/step_<N>/adapter`), not `run_dir/adapter` — because `run_dir/adapter` is deliberately not written on a stop. Callers that assume `checkpoint_path == run_dir/adapter` are only the eval/bundle phases, which the CLI short-circuits past on a time-limited stop, so this is consistent. *(Worth a glance during implementation to confirm no other consumer dereferences `checkpoint_path` on the time-limited path.)*
4. **The duration grammar rejects a trailing number with no unit** (`"2h30"` is invalid; the user must write `"2h30m"`). A bare all-digits string is the *only* accepted unit-less form and means seconds. This keeps the grammar unambiguous.
5. **`format_seconds` collapses to the largest applicable units** (e.g. `5400 → "1h30m"`, not `"90m"`). The exit message therefore may not echo the user's literal string when they wrote a non-canonical form like `"90m"`; it echoes the canonical equivalent (`"1h30m"`). This is intentional for consistency between string and bare-seconds inputs. *(Confirm this is acceptable UX — the alternative is to echo the raw config string, but then `9000` and `"2h30m"` would render differently.)*
6. **The minimal flush is preferred over reusing `on_checkpoint`** to keep the stop instant lean (skip the image-panel render). Reusing `on_checkpoint` is explicitly allowed if the direct `save_full_state` call proves awkward to wire from inside `run_epoch`.

End of spec.
