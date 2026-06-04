# Progress bar: unify rich + plain handles onto an epoch-derived global model

## Background / Problem

The default TTY progress bar (rich, mode `ON`) for training is broken in three
ways. The root cause is that the rich handle (`_ProgressHandle`) uses a
**per-epoch** position model with two separate rich tasks (an outer "epoch" bar
and an inner "step" bar), while the plain handle (`_PlainHandle`) already uses a
**global** model. The fix unifies both handles onto a single epoch-derived
global model and collapses the rich display to one task.

All file references are to this worktree
(`/home/justin/projects/custom-sam-peft/.claude/worktrees/progress-bar-global-model`).

### Bug 1 — Epoch bar never starts (frozen at `0/N`, `-:--:--`)

The outer epoch task only advances once per *completed* epoch, via
`P.advance_outer()` at `src/custom_sam_peft/train/trainer.py:826` (called after
`_train_epoch` returns). During a single long or time-limited epoch the outer
bar sits frozen at `0/N` with no fill and no ETA. A run that is time-limited and
stops before the first epoch completes never moves the outer bar at all.

### Bug 2 — Step bar stops working after epoch 1 (rich finished-latch)

Verified with rich 15.0.0. At each epoch end the inner task reaches
`completed == total` (one full epoch of `total_batches`), so rich records
`finished_time` and latches `finished = True`. The next-epoch
`reset_inner` (`src/custom_sam_peft/cli/_progress.py:156`) sets `completed = 0`
but passes the **same** `total`. Rich only calls the internal `Task._reset()`
(which clears `finished_time`) when `total` *changes*; an unchanged `total`
leaves `finished` stuck `True` forever. From epoch 1 onward the inner bar shows:
spinner stopped, ETA frozen at `0:00:00`, finished-style coloring — even though
`completed` has restarted at 0 and is climbing.

### Bug 3 — Resume restarts at `1/total`; outer total stays at max

On resume the trainer calls `P.set_start(start_epoch, global_step)`
(`src/custom_sam_peft/train/trainer.py:751`). The rich `set_start`
(`_progress.py:174-177`) updates **only** the outer task's `completed` and
ignores `start_step` for the inner bar. Combined with the per-epoch
`reset_inner`, the inner bar restarts at `1/total_batches` each epoch instead of
reflecting the resumed global position. (The plain handle honors `start_step`, so
only the rich bar exhibits this.)

### Root-cause synthesis

`_ProgressHandle` is per-epoch (two tasks, position resets every epoch);
`_PlainHandle` is global (one synthesized global step). Unify both onto a single
**epoch-derived global model** and reduce the rich display to one task.

## Goals

- Rich (`ON`) training bar fills continuously across an entire run, with a live
  ETA, from the first batch through the last batch of the last epoch.
- The rich bar never enters the finished/latched state until the run's true final
  step; `reset_inner` at each epoch boundary must not freeze it.
- Resume positions the bar at the correct global offset
  (`start_epoch * batches_per_epoch`) instead of restarting at the epoch's first
  batch.
- Both handles compute position from the same epoch-derived formula, eliminating
  the per-epoch/global divergence.
- Eval / predict / export (`total_epochs is None`) keep their current behavior
  byte-for-byte (single indeterminate-then-determinate bar, no epoch label,
  `push_subtask` unchanged).

## Non-Goals

- **Fixing the `global_step` re-walk double-count** — tracked as GitHub issue
  **#308**. The resume re-walk increments `global_step` again over batches it has
  already counted, so `global_step` over-counts. This design deliberately derives
  bar position from `epoch * batches_per_epoch + step_in_epoch`, which is
  independent of `global_step`, so the bar is correct regardless of #308. Do not
  attempt to fix #308 here.
- No new columns. The user explicitly chose **not** to add an elapsed-time
  column; ETA-only (`TimeRemainingColumn`) stays.
- No changes to `resolve_mode`, SIGINT handling, log routing, or
  `_silence_third_party_progress`.

## Design

### Core position model (both handles)

- Position = `epoch * batches_per_epoch + step_in_epoch`.
- Total = `total_epochs * batches_per_epoch` (when `total_epochs` is known).
- `epoch` and `step_in_epoch` are tracked on the handle. The trainer supplies the
  authoritative `epoch` to `reset_inner` (see below); `step_in_epoch` resets to 0
  at each `reset_inner` and increments on each `advance_inner`.
- This is **independent of the raw `global_step` counter** (which double-counts on
  resume — #308). Epoch-derived position is monotonic, never overshoots `total`,
  and resumes exactly at the epoch boundary.

### `reset_inner` signature change (Protocol + all handles + proxy)

Add an optional `epoch` parameter:

```
reset_inner(self, total: int | None = None, epoch: int | None = None) -> None
```

Apply the new signature to **all** of:

- `SubTaskHandle`? No — `SubTaskHandle` has no `reset_inner`; leave it.
- `_NoOpHandle.reset_inner` — accept and ignore both args.
- `_ProgressHandle.reset_inner` — see below.
- `_PlainHandle.reset_inner` — see below.
- `_ProgressProxy.reset_inner` — forward both args to `_state.handle.reset_inner`.

When `epoch` is provided, the handle sets its internal `_epoch` to that value
(authoritative). When `epoch is None` (eval/predict/export call sites), `_epoch`
is left unchanged.

### Trainer call-site change

Change `src/custom_sam_peft/train/trainer.py:809` from
`P.reset_inner(total=total_batches)` to
`P.reset_inner(total=total_batches, epoch=epoch)`, where `epoch` is the loop
variable from `for epoch in range(start_epoch, cfg.train.epochs)`. This makes the
loop's `epoch` the single source of truth for the handle's epoch baseline rather
than relying on `advance_outer` increments staying in lockstep.

`P.advance_outer()` at trainer.py:826 stays (it refreshes the label and keeps
`_epoch` advancing for the description), but it no longer drives bar position.
`P.set_start(start_epoch, global_step)` at trainer.py:751 stays unchanged at the
call site.

### `_ProgressHandle` (rich, mode `ON`)

Collapse to a **single** rich task. Drop the separate outer epoch bar entirely.

Stored state:

- `_total_epochs: int | None`
- `_total_batches: int` (per epoch)
- `_epoch: int` (0-indexed, within-run epoch)
- `_step: int` (within current epoch)
- `_postfix: dict[str, Any]`
- the single rich task id (rename from `_inner`; the `_outer` field is removed)

Columns — **unchanged** order and set:
`SpinnerColumn, TextColumn("[progress.description]{task.description}"),
BarColumn, MofNCompleteColumn, TimeRemainingColumn`.

Constructor: now also receives `total_epochs: int | None` (currently it does
not). Remove `outer_task_id`; keep a single task id parameter. Drop the unused
`log_every` only if it is genuinely unused for rich — keep it if other code reads
it; otherwise leave it to minimize churn (it is currently stored but unused —
leaving it is acceptable).

`_render_description() -> str` (new private helper):

- When `_total_epochs is not None`: `f"{self._kind.value} {self._epoch + 1}/{self._total_epochs}"`,
  followed by a space and the formatted postfix when the postfix is non-empty.
- When `_total_epochs is None`: just the formatted postfix (no epoch label) —
  preserving current eval/predict description behavior.
- Postfix formatting: the existing `" ".join(f"{k}={v}" for k, v in ...)` style
  over `_postfix`. (No special numeric formatting is required for rich; the
  trainer passes already-formatted values today via `update_postfix`.)

Methods:

- `reset_inner(total=batches, epoch=None)`:
  - If `epoch is not None`: `self._epoch = epoch`.
  - If `total is not None`: `self._total_batches = total`.
  - Compute the rich task `total`:
    - if `_total_epochs is not None`: `_total_epochs * _total_batches`
    - else: `_total_batches`
  - Compute the rich task `completed` (epoch baseline):
    - if `_total_epochs is not None`: `_epoch * _total_batches`
    - else: `0`
  - `self._step = 0`.
  - Update the rich task with the computed `total`, `completed`, and the refreshed
    `description` from `_render_description()`.
  - **Why this fixes bug 2:** task `completed` only equals task `total` on the
    final epoch's last step, so rich never latches `finished` mid-run, so a later
    `reset_inner` never needs `Task._reset` to un-stick it.
- `advance_inner(n=1)`: `self._step += n`; advance the rich task by `n` (so the
  single bar moves every batch → fixes bug 1).
- `advance_outer(n=1)`: `self._epoch += n`; refresh description. Does **not**
  move the bar (steps move the bar).
- `set_start(start_epoch, start_step)`: `self._epoch = start_epoch`; refresh
  description. `start_step` is intentionally **unused** for position — the
  subsequent `reset_inner(total=batches, epoch=start_epoch)` sets the bar's
  `completed` to `start_epoch * batches` (→ fixes bug 3). Keep the `start_step`
  param for signature compatibility.
- `update_postfix(**kwargs)`: merge into `self._postfix`; refresh description via
  `_render_description()`.
- `push_subtask(label, total)`: **unchanged** — adds a transient rich task and
  removes it on exit.

### `_PlainHandle` (mode `PLAIN`)

Switch the global-step computation in `_emit` from
`cumulative_step_offset + step` to `epoch * total_batches + step`
(drift-immune; no running accumulator to fall out of sync).

- Retire `self._cumulative_step_offset` entirely (the field, its init, the
  accumulation line in `reset_inner`, and both uses in `_emit`).
- `reset_inner(total=None, epoch=None)`:
  - if `total is not None`: `self._total_batches = total`
  - if `epoch is not None`: `self._epoch = epoch`
  - `self._step = 0` (no accumulation step).
- `_emit`: where it previously read `self._cumulative_step_offset + self._step`
  for both the printed step and the ETA `current`, use
  `self._epoch * self._total_batches + self._step`. The `global_total` /
  `total_for_eta` computation (`_total_epochs * _total_batches`) is unchanged.
- `set_start(start_epoch, start_step)`: set `self._epoch = start_epoch`. The
  `start_step` param becomes unused for position (the epoch baseline now derives
  position); keep the param. Do not set `_cumulative_step_offset` (removed).
- `advance_outer` / `advance_inner` (the `% log_every` emit trigger) /
  `update_postfix` / `push_subtask`: unchanged except for the removed accumulator.

**Snapshot contract preserved:** `test_plain_line_snapshot` sets `_epoch = 2`,
`_total_batches = 4530`, `_step = 1240` and expects `step=10300/45300`. The new
formula gives `2 * 4530 + 1240 = 10300` and `10 * 4530 = 45300` — identical. The
test currently also sets `_cumulative_step_offset = 2 * 4530`; that line becomes
inert/removed (see Testing).

### `progress_session`

For mode `ON`:

- Stop creating the separate outer epoch task. Create **exactly one** rich task.
- Initialize that task with `total=None` (indeterminate / pulsing bar), because
  `total_batches_per_epoch` is `0`/unknown at session open for train (the trainer
  fills the real total via the first `reset_inner`). The initial description can
  be the kind label (e.g. `f"{kind.value}"`); it is overwritten on the first
  `reset_inner`/`update_postfix`.
- Pass `total_epochs` into the `_ProgressHandle` constructor (currently omitted).
- Remove the `outer_id` / `add_task(f"{kind.value} epoch", ...)` block.

For modes `PLAIN` and `OFF`: unchanged (other than the `_PlainHandle` internals
above).

### Compatibility requirements (explicit)

- **eval / predict / export** open `progress_session` with `total_epochs`
  omitted (→ `None`):
  - eval (`src/custom_sam_peft/cli/eval_cmd.py:101`) uses `push_subtask`;
    `total_batches_per_epoch=0`.
  - predict (`src/custom_sam_peft/cli/predict_cmd.py:233`) calls
    `P.reset_inner(total=len(image_paths))` (no `epoch`).
  - export (`src/custom_sam_peft/cli/export_cmd.py:41`) opens a session with no
    per-step bar.
  - For all three, `total_epochs is None` ⇒ single task with `total=batches` (or
    indeterminate until `reset_inner`), **no epoch label**, `completed` baseline
    0. Behavior must remain identical to today.
- The `_ProgressProxy.reset_inner` and the `SubTaskHandle` / `_NoOpHandle`
  Protocol surface must stay in sync with the new `reset_inner` signature.
  `SubTaskHandle` itself does not gain `reset_inner`.

## Affected files

- `src/custom_sam_peft/cli/_progress.py` — primary changes:
  - `_NoOpHandle.reset_inner`: add `epoch` param (ignored).
  - `_ProgressHandle`: single-task model; new `_total_epochs`, `_epoch`, `_step`,
    `_postfix`; remove `_outer`; add `_render_description`; rewrite
    `reset_inner` / `advance_inner` / `advance_outer` / `set_start` /
    `update_postfix` per Design; constructor gains `total_epochs`, drops
    `outer_task_id`.
  - `_PlainHandle`: remove `_cumulative_step_offset`; `reset_inner` gains `epoch`;
    `_emit` uses `epoch * total_batches + step`; `set_start` sets only `_epoch`.
  - `_ProgressProxy.reset_inner`: forward `epoch`.
  - `progress_session` (ON branch): single task, `total=None`, pass
    `total_epochs` to `_ProgressHandle`; remove outer-task creation.
- `src/custom_sam_peft/train/trainer.py` — line 809: pass `epoch=epoch` to
  `reset_inner`. (Lines 751 `set_start` and 826 `advance_outer` unchanged.)
- `src/custom_sam_peft/train/loop.py` — no change required; it already calls
  `P.advance_inner()` and `P.update_postfix(...)` per batch (~744-752). Confirm
  it does not call `reset_inner` itself (it does not).
- `tests/unit/test_progress_module.py` — add unit tests (below); keep
  `test_plain_line_snapshot` green.
- `tests/unit/test_progress_integration.py` — update `test_fake_trainer_smoke`
  to the real call order.
- `tests/unit/test_progress_resolve.py` — no change expected (pure
  `resolve_mode`); confirm it still passes.
- CLI command modules (`train_cmd.py`, `run_cmd.py`, `eval_cmd.py`,
  `predict_cmd.py`, `export_cmd.py`) — **no source changes**; they are listed
  only as the call sites whose behavior must be preserved/verified.

## Testing

Run the progress tests bypassing the global coverage gate (a subset would fail
`--cov-fail-under=80`; `--no-cov` does not bypass it — use `-o "addopts="`):

```
.venv/bin/python -m pytest \
  tests/unit/test_progress_module.py \
  tests/unit/test_progress_integration.py \
  -o "addopts="
```

`ruff format --check`, `ruff check`, and `mypy src/custom_sam_peft` are separate
CI gates — run all three locally before claiming done.

### Update existing tests

- `test_fake_trainer_smoke` (`test_progress_integration.py`): the test currently
  calls `handle.advance_outer()` **before** `handle.reset_inner()`. Align to the
  real trainer order: `reset_inner(total=..., epoch=epoch)` →
  `advance_inner()` × N → `advance_outer()`. Pass `epoch=` to `reset_inner` from
  the loop variable. Keep the existing count assertions (outer advances ==
  `total_epochs`, inner advances == `total_epochs * batches_per_epoch`).
- `test_plain_line_snapshot` (`test_progress_module.py`): must stay green under
  the new formula. Verify it still produces `step=10300/45300`. The test's
  `handle._cumulative_step_offset = 2 * 4530` line references a removed field —
  remove that line (the new formula uses `_epoch` and `_total_batches`, both
  already set in the test).

### New unit tests (CPU-only, no GPU markers; match the existing style/naming in `test_progress_module.py`)

Construct `_ProgressHandle` directly inside an `ON` `progress_session` (or via
`progress_session` + `_state.handle`) so a real `rich.Progress` task backs it;
inspect `handle._progress.tasks[...]` for `completed`, `total`, and `finished`.

- **(a) No finished-latch across epoch boundaries.** Drive the handle across 2+
  epoch boundaries: `reset_inner(total=B, epoch=0)`, `advance_inner()`×B,
  `advance_outer()`, `reset_inner(total=B, epoch=1)`, … Assert the rich task is
  **not** `finished` until the final epoch's last `advance_inner`. (Directly
  exercises bug 2.)
- **(b) Resume baseline.** After `set_start(start_epoch=E, start_step=...)` then
  `reset_inner(total=B, epoch=E)`, assert the rich task `completed == E * B` and
  `total == total_epochs * B`. (Directly exercises bug 3.)
- **(c) No overshoot.** Walk all epochs to completion; assert `completed` is never
  `> total` at any point, and equals `total` only at the very end.
- **(d) Description composition.** With `total_epochs` set, after
  `reset_inner(epoch=2)` + `update_postfix(loss="0.5")`, assert the task
  description contains both the epoch label (e.g. `3/160`-style) and `loss=`.
  With `total_epochs is None`, assert the description contains **no** epoch label
  (just the postfix).

## Risks / edge cases

- **Indeterminate bar before the first batch.** With `total=None` at session
  open, rich renders a pulsing bar and `MofNCompleteColumn` shows no fixed
  denominator until the first `reset_inner` sets the real `total`. This is the
  intended pre-first-batch state for train; confirm it renders without raising
  (rich supports `total=None`).
- **eval/predict/export compatibility.** These never pass `epoch`, so `_epoch`
  stays 0 and there is no epoch label; `reset_inner(total=N)` yields a single
  determinate bar `0..N`. Verify the eval `push_subtask` path and the predict
  `reset_inner(total=len(image_paths))` path are visually/behaviorally identical
  to today.
- **rich `Task._reset` semantics.** The fix relies on `completed` reaching
  `total` only once (final step), so `finished` never latches mid-run and the
  per-epoch `reset_inner` (which keeps `total` constant) never needs a reset to
  un-stick the bar. If a future change makes `reset_inner` pass a *changing*
  `total` mid-run, rich would call `Task._reset` and clear elapsed/ETA — not a
  concern here since `total = total_epochs * batches` is stable across epochs.
- **`global_step` double-count (#308).** Out of scope; the epoch-derived model
  sidesteps it. Do not wire `global_step`/`start_step` into bar position.
- **`advance_outer` vs `reset_inner` epoch source.** Both update `_epoch`;
  because the trainer now passes the authoritative `epoch` to `reset_inner`, a
  drift between `advance_outer` increments and the loop epoch cannot desync the
  bar position (only the label could momentarily differ, and `reset_inner`
  reasserts it each epoch).
