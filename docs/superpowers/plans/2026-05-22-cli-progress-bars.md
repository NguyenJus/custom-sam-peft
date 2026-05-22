# CLI Progress Bars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-22-cli-progress-bars-design.md`](../specs/2026-05-22-cli-progress-bars-design.md)
**Issue:** [#76](https://github.com/NguyenJus/custom-sam-peft/issues/76) — *feat(cli): progress bars for train / eval / predict — bottom-pinned, on by default, scramble-free*
**Branch:** `feat-cli-progress-bars` (worktree at `/home/justin/projects/custom-sam-peft/.worktrees/feat-cli-progress-bars`)

**Goal:** Add a `rich.Progress`-backed process-global progress handle (`P`) with a no-op default, wire it into all long-running CLI commands (`train`, `eval`, `run`, `export`), add the `T20` ruff lint rule banning bare `print()`, and ship a 10-test CPU-only test suite.

**Architecture:** A single new module `src/custom_sam_peft/cli/_progress.py` (~250 lines) owns all progress state. It exports a `progress_session(...)` context manager and a module-level `progress as P` singleton. When no session is active, `P.*` is a no-op — library callers and unit tests are unaffected. CLI commands open a session around their runner call; `train/loop.py`, `train/trainer.py`, and `eval/evaluator.py` call `P.*` directly without importing any CLI machinery. `rich.Progress` owns both the log-scroll region and the pinned bar region through a shared console, using `RichHandler(console=progress.console)` to route log output above the pinned bar. Third-party progress (HF, datasets) is suppressed via a `_silence_third_party_progress()` function called once at app entry and defensively on session entry.

**Tech Stack:** Python 3.12, `rich>=13` (already a hard dep), `typer>=0.12`, `pytest`, `ruff` with T20 rule, `mypy` strict.

---

## Planner decisions on spec ambiguities

### §4 PLAIN-mode `step` field semantics

The spec §4 snapshot shows `step=1240/45300`, where `1240` is the local step within the current epoch and `45300` is the total across all epochs (`10 × 4530`). **Decision: emit the *global* step (cumulative across epochs), not the local epoch step.** So for epoch 2 (0-indexed), local step 1240, with 4530 batches/epoch and 10 epochs, the emitted line is `step=10300/45300` (= `2 × 4530 + 1240`). Rationale: global step is more useful for CI log grepping and matches what `tracker.log_scalars` uses as its step key. The spec snapshot was illustrative; the frozen test in Test F uses the global step value. Test F uses a `startswith` assertion for the ETA portion because `time.monotonic()` advances between `_PlainHandle.__init__` and `_emit`, making the exact ETA value non-deterministic in CI.

### §5.2 `total_epochs` default for eval sessions

The spec says `total_epochs` defaults to "1 or None". **Decision: `total_epochs: int | None = None`.** When `total_epochs is None`, `progress_session` omits the outer epoch task entirely, so the eval display shows only the inner per-image bar. This is more idiomatic Python than defaulting to 1 (which would show an always-complete outer bar) and easier to test ("outer task absent" vs "outer task at 1/1"). Implementers: when `kind=EVAL`, do not pass `total_epochs` — omit it entirely.

### §10 Predict wiring (`predict_cmd.py`)

The spec leaves it to the orchestrator to decide whether predict wiring lands as a fixup commit on the #76 branch or in #74's PR. **Decision: fixup commit on the #76 branch after #74 has merged.** PR responsibilities stay disjoint: #74 ships the `predict_cmd.py` file; #76 ships progress infrastructure + predict wiring. The #76 PR is kept ready-but-held until #74 merges. Once #74 merges and `predict_cmd.py` exists, add a fixup commit wiring `--progress` into `predict_cmd.py` using `kind=PREDICT` per spec §5.4, then open the PR. The orchestrator must halt and notify the user if #74 has not yet merged when it is time to open the PR.

---

## File Map

**New files:**

```
src/custom_sam_peft/cli/_progress.py                  NEW  (~250 lines)
tests/unit/test_progress_resolve.py                   NEW  (test A)
tests/unit/test_progress_module.py                    NEW  (tests B–H)
tests/unit/test_progress_integration.py               NEW  (tests I–J)
```

**Modified files:**

```
src/custom_sam_peft/cli/_logging.py                   TOUCHED  (+console kwarg, spec §7.3)
src/custom_sam_peft/cli/main.py                       TOUCHED  (+_silence_third_party_progress call, spec §7.1)
src/custom_sam_peft/cli/train_cmd.py                  TOUCHED  (--progress option + progress_session, spec §5.1)
src/custom_sam_peft/cli/eval_cmd.py                   TOUCHED  (--progress option + progress_session, spec §5.2)
src/custom_sam_peft/cli/run_cmd.py                    TOUCHED  (three sequential sessions, spec §5.3)
src/custom_sam_peft/cli/export_cmd.py                 TOUCHED  (progress_session, spec §5)
src/custom_sam_peft/cli/doctor_cmd.py                 TOUCHED  (# noqa: T201, spec §8)
src/custom_sam_peft/train/loop.py                     TOUCHED  (P.advance_inner, P.update_postfix, spec §5.1)
src/custom_sam_peft/train/trainer.py                  TOUCHED  (P.advance_outer, P.reset_inner, spec §5.1)
src/custom_sam_peft/eval/evaluator.py                 TOUCHED  (P.advance_inner, P.update_postfix, spec §5.2)
pyproject.toml                                        TOUCHED  ("T20" added to ruff lint select, spec §8)
```

No deletions. No new top-level directories. `predict_cmd.py` not touched by the #76 PR (see §10 decision above).

---

## Parallelization opportunities (for orchestrator dispatch)

**Phase 0** (pre-flight) blocks everything — it is a guard.

**Phase 1** (noqa + T20 lint rule) must run before Phase 2 because the `T20` lint rule is activated in Phase 1 and will cause lint failures in Phase 2 and beyond if the `# noqa: T201` is not in place first. Serialize: Phase 1 → Phase 2 → everything else.

**Phase 2** (`_progress.py` module shell + `_logging.py` + `main.py`) is the prerequisite for all consumer phases because they import from `_progress.py`.

**Phases 3, 4, and 5** are file-disjoint once Phase 2 is committed and can be fanned out in parallel via `superpowers:dispatching-parallel-agents`:

- **Phase 3** touches only `tests/unit/test_progress_resolve.py` — the pure `resolve_mode` test that has no dependency on `_progress.py` internals (only the exported `resolve_mode` pure function).
- **Phase 4** touches only `tests/unit/test_progress_module.py` and `tests/unit/test_progress_integration.py`.
- **Phase 5** touches only `train/loop.py`, `train/trainer.py`, `eval/evaluator.py`, `cli/train_cmd.py`, `cli/eval_cmd.py`, `cli/run_cmd.py`, `cli/export_cmd.py`.

**Phase 6** (verification gate) serializes after Phases 3, 4, and 5.

**Phase 7** (PR) serializes after Phase 6, plus the #74 merge dependency.

Dependency graph:

```
Phase 0 (pre-flight)
  → Phase 1 (noqa + T20)
    → Phase 2 (_progress.py shell + _logging.py + main.py)
      ├─→ Phase 3 (test_progress_resolve.py)      ┐
      ├─→ Phase 4 (test_progress_module.py + _integration.py) ├─→ Phase 6 (gate) → Phase 7 (PR, after #74)
      └─→ Phase 5 (CLI wiring + train/eval callsites)          ┘
```

**Reviewer model floor:** sonnet/high for every phase. Never haiku.

---

## Assumptions for the cold reader

1. **Working directory.** Every shell command below runs with `cwd = /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-progress-bars`. Use absolute paths when invoking external tools; use repo-relative paths inside plan text.
2. **Tooling.** `uv` is on PATH. Use `uv run …` for all Python entry points. Do NOT shell out to `python` directly.
3. **`rich` is already a hard dep.** `rich>=13` is in `pyproject.toml` dependencies. No lockfile changes needed for the progress feature.
4. **`total_epochs: int | None = None` decision.** When `total_epochs is None`, no outer epoch task is created. `kind=EVAL` always omits `total_epochs`. `kind=TRAIN` always passes `total_epochs=cfg.train.epochs`.
5. **`ProgressMode` enum has three values: `ON | OFF | PLAIN`.** `AUTO` is not a `ProgressMode` — it is the string `"auto"` on the CLI side, resolved by `resolve_mode` before the session opens.
6. **`T20` ordering.** Add `# noqa: T201` to `doctor_cmd.py` (Phase 1) *before* adding `"T20"` to `pyproject.toml` (also Phase 1, same commit). The noqa comment must be present when the rule activates or ruff will fail.
7. **Tests are CPU-only.** No `@pytest.mark.gpu` markers needed for any of the 10 tests.
8. **`PLAIN` mode is not a no-op.** `ProgressMode.PLAIN` uses `_PlainHandle`, which emits one `progress:` log line per `log_every` window via stdlib `logging`. Only `ProgressMode.OFF` suppresses all progress output.
9. **Current version is `0.8.0`.** Next semver for this MINOR feature addition (new CLI flag + new module) would be `v0.9.0` under pre-1.0 conventions. The orchestrator confirms this at PR-open time per CLAUDE.md.
10. **Spec and plan are already committed.** The brainstormer session committed `docs/superpowers/specs/2026-05-22-cli-progress-bars-design.md` and `docs/superpowers/plans/2026-05-22-cli-progress-bars.md` in commit `18229d8` and pushed the branch. Phase 0 step P0-4 is therefore a no-op idempotency check — `git status` should show those paths as already tracked/committed; if they appear untracked or modified, the orchestrator is in the wrong worktree.

---

## Phase 0: Pre-flight checks

**Model/effort:** sonnet / medium (one subagent, ~5 minutes).
**Parallel:** No. **Blocks:** all later phases.
**Spec:** none (guard).

- [ ] **Step P0-1: Confirm working tree state**

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-progress-bars status
```

Expected: branch `feat-cli-progress-bars`. Spec and plan file untracked or committed. No unexpected modifications. If the working tree is dirty, halt — do not start on a broken baseline.

- [ ] **Step P0-2: Confirm baseline unit tests pass**

```bash
uv run pytest tests/unit/ -x -q
```

Expected: all green. If anything is red, halt and report — Phase 6 cannot validate progress without a clean baseline.

- [ ] **Step P0-3: Confirm no `print()` calls exist in source yet (for baseline)**

```bash
uv run ruff check src --select T201 2>/dev/null || echo "T201 not in select yet — OK"
```

Expected: either "T201 not in select yet — OK" (rule not activated) or zero lint errors. This establishes the baseline so Phase 1's `# noqa: T201` addition is the only exception we need to track.

- [ ] **Step P0-4: Confirm spec and plan are committed** (idempotency check — already done in commit `18229d8`)

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-progress-bars \
  log --oneline -1 -- docs/superpowers/specs/2026-05-22-cli-progress-bars-design.md \
                      docs/superpowers/plans/2026-05-22-cli-progress-bars.md
```

Expected: one line showing commit `18229d8 docs(superpowers): spec + plan for #76 CLI progress bars` (or a later amend / merge commit). If the output is empty, the spec and plan have not been committed — halt and surface the discrepancy. Do not attempt to re-commit; the brainstormer session has already done so. Any orchestrator-side fixes to the plan (see assumption 10) should be committed as a separate `docs(superpowers): clarify ...` commit.

---

## Phase 1: Add `# noqa: T201` and activate the `T20` lint rule

**Model/effort:** sonnet / medium.
**Parallel:** No. **Blocks:** all subsequent phases.
**Spec:** §8.

**Files:**
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py` (line 83)
- Modify: `pyproject.toml` (line 67)

**Goal:** Add the `# noqa: T201` comment to the legitimate `print()` in `doctor_cmd.py`, then activate `"T20"` in the ruff select list — in a single commit so the rule is never active without its single exception in place. No other files are touched.

### Task 1a: Add `# noqa: T201` to `doctor_cmd.py`

- [ ] **Step P1-1: Edit `doctor_cmd.py` line 83**

Current line 83:

```python
        print(json.dumps(dataclasses.asdict(report), default=str, indent=2))
```

Replace with:

```python
        print(json.dumps(dataclasses.asdict(report), default=str, indent=2))  # noqa: T201
```

No other change to this file.

### Task 1b: Add `"T20"` to `pyproject.toml`

- [ ] **Step P1-2: Edit `pyproject.toml` line 67**

Current:

```toml
select = ["E", "F", "I", "B", "UP", "SIM", "RUF", "S"]
```

Replace with:

```toml
select = ["E", "F", "I", "B", "UP", "SIM", "RUF", "S", "T20"]
```

### Task 1c: Verify no new lint failures

- [ ] **Step P1-3: Run ruff with the new rule**

```bash
uv run ruff check src tests
```

Expected: exits 0. If `T201` fires on `doctor_cmd.py` line 83, the `# noqa: T201` is missing or misplaced — fix the comment. If `T201` fires on any other file, that file has a bare `print()` that was not identified at plan-write time — fix it.

- [ ] **Step P1-4: Commit both changes together**

```bash
git add src/custom_sam_peft/cli/doctor_cmd.py pyproject.toml
git commit -m "chore(lint): add T20 (flake8-print) rule; noqa doctor_cmd.py print (#76)"
```

---

## Phase 2: Create `_progress.py`, update `_logging.py`, update `main.py`

**Model/effort:** sonnet / high.
**Parallel:** No (must complete before Phases 3–5). **Depends on:** Phase 1.
**Spec:** §3, §4, §6, §7.

**Files:**
- Create: `src/custom_sam_peft/cli/_progress.py`
- Modify: `src/custom_sam_peft/cli/_logging.py`
- Modify: `src/custom_sam_peft/cli/main.py`

**Goal:** Implement the full `_progress.py` module with `_NoOpHandle`, `_ProgressHandle`, `progress_session`, `resolve_mode`, `ProgressKind`, `ProgressMode`, and the module-level `progress as P` singleton. Add the `console` kwarg to `configure_logging`. Add the `_silence_third_party_progress()` call to `main.py`.

### Task 2a: Write `src/custom_sam_peft/cli/_progress.py`

The spec (§3, §4, §6, §7) defines the full API. The module is approximately 250 lines. Key design decisions, in order:

**Imports:**

```python
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
```

**`ProgressMode` enum** (spec §3.1, §4):

```python
class ProgressMode(str, Enum):
    ON = "on"
    OFF = "off"
    PLAIN = "plain"
```

**`ProgressKind` enum** (spec §3.1):

```python
class ProgressKind(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    PREDICT = "predict"
    EXPORT_MERGE = "export-merge"
```

**`resolve_mode` pure function** (spec §4):

```python
def resolve_mode(
    cli_flag: str | None,
    env: Mapping[str, str],
    stdout_isatty: bool,
    is_jupyter: bool,
) -> ProgressMode:
    """Resolve the effective ProgressMode. Pure function; no side effects.

    Precedence: explicit --progress flag > CSP_NO_PROGRESS env var > auto fallback.
    """
    if cli_flag is not None and cli_flag != "auto":
        return ProgressMode(cli_flag)
    if env.get("CSP_NO_PROGRESS") == "1":
        return ProgressMode.OFF
    if is_jupyter:
        return ProgressMode.PLAIN
    if not stdout_isatty:
        return ProgressMode.PLAIN
    return ProgressMode.ON
```

**`_NoOpHandle` class** — process-global handle when no session is active. Every method is a no-op. `console` returns a plain `Console()`.

```python
class _NoOpHandle:
    """No-op progress handle used when no session is active (default)."""

    @property
    def console(self) -> Console:
        return Console()

    def advance_outer(self, n: int = 1) -> None:
        pass

    def advance_inner(self, n: int = 1) -> None:
        pass

    def reset_inner(self) -> None:
        pass

    def update_postfix(self, **kwargs: Any) -> None:
        pass

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        yield
```

**`_ProgressHandle` class** — live handle backed by `rich.Progress`. Used when a session is active.

```python
class _ProgressHandle:
    """Live progress handle backed by rich.Progress."""

    def __init__(
        self,
        rich_progress: Progress,
        outer_task_id: TaskID | None,
        inner_task_id: TaskID,
        kind: ProgressKind,
        total_batches_per_epoch: int,
        log_every: int = 50,
    ) -> None:
        self._progress = rich_progress
        self._outer = outer_task_id
        self._inner = inner_task_id
        self._kind = kind
        self._total_batches = total_batches_per_epoch
        self._log_every = log_every
        self._step = 0
        self._epoch = 0
        self._plain_postfix: dict[str, Any] = {}

    @property
    def console(self) -> Console:
        return self._progress.console

    def advance_outer(self, n: int = 1) -> None:
        if self._outer is not None:
            self._progress.advance(self._outer, n)
        self._epoch += n

    def reset_inner(self) -> None:
        self._progress.reset(self._inner)
        self._step = 0

    def advance_inner(self, n: int = 1) -> None:
        self._progress.advance(self._inner, n)
        self._step += n

    def update_postfix(self, **kwargs: Any) -> None:
        self._plain_postfix.update(kwargs)
        # Build description string for rich display.
        desc = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._progress.update(self._inner, description=desc)

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        task_id = self._progress.add_task(label, total=total)
        try:
            yield
        finally:
            self._progress.remove_task(task_id)
```

**`_silence_third_party_progress` function** (spec §7.1):

```python
def _silence_third_party_progress() -> None:
    """Suppress HF / datasets progress output. Idempotent — safe to call twice."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import datasets as _ds  # type: ignore[import-untyped]
        _ds.disable_progress_bar()
    except ImportError:
        pass
```

**Module-level singleton** — starts as `_NoOpHandle`:

```python
progress: _NoOpHandle | _ProgressHandle | _PlainHandle = _NoOpHandle()
```

**`progress_session` context manager** (spec §3.1, §6, §7.2):

This is the most complex part. Key responsibilities:
1. Raise `RuntimeError` if a session is already active (no nesting).
2. Build the `rich.Progress` instance (for `ON` mode) or skip (for `OFF`/`PLAIN`).
3. Attach `RichHandler(console=progress.console)` to the root logger on entry; restore on exit.
4. Call `_silence_third_party_progress()` defensively.
5. Install SIGINT handler that calls `progress.stop()` before `KeyboardInterrupt`; restore on exit.
6. Swap the module-level `progress` singleton from `_NoOpHandle` to `_ProgressHandle` on entry; restore on exit.

```python
_SESSION_ACTIVE = False


@contextmanager
def progress_session(
    kind: ProgressKind,
    total_batches_per_epoch: int,
    mode: ProgressMode,
    total_epochs: int | None = None,
    log_every: int = 50,
) -> Generator[None, None, None]:
    """Context manager that activates the process-global progress handle.

    Opens a rich.Progress live display for mode=ON; emits plain progress lines
    for mode=PLAIN; suppresses progress output for mode=OFF.

    Raises RuntimeError if called while a session is already active (nesting
    is not supported).
    """
    global _SESSION_ACTIVE, progress

    if _SESSION_ACTIVE:
        raise RuntimeError(
            "nested session: a progress_session is already active in this process"
        )
    _SESSION_ACTIVE = True
    _silence_third_party_progress()

    root_logger = logging.getLogger()
    prior_handlers = list(root_logger.handlers)
    prior_capture = logging.captureWarnings.__doc__  # just a sentinel; see below

    rich_prog: Progress | None = None
    handle: _ProgressHandle | _PlainHandle | _NoOpHandle

    if mode == ProgressMode.ON:
        rich_prog = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("it/s={task.speed:.1f}" if False else ""),  # overridden below
            TimeRemainingColumn(),
            console=Console(stderr=False),
            transient=False,
        )
        rich_prog.start()

        # Attach RichHandler so logs scroll above the pinned bar.
        root_logger.handlers.clear()
        root_logger.addHandler(
            RichHandler(
                console=rich_prog.console,
                rich_tracebacks=True,
                show_path=False,
            )
        )
        logging.captureWarnings(True)

        outer_id: TaskID | None = None
        if total_epochs is not None:
            outer_id = rich_prog.add_task(
                f"{kind.value} epoch",
                total=total_epochs,
            )
        inner_id = rich_prog.add_task(
            f"{kind.value} step",
            total=total_batches_per_epoch,
        )
        handle = _ProgressHandle(
            rich_progress=rich_prog,
            outer_task_id=outer_id,
            inner_task_id=inner_id,
            kind=kind,
            total_batches_per_epoch=total_batches_per_epoch,
            log_every=log_every,
        )
    elif mode == ProgressMode.PLAIN:
        handle = _PlainHandle(
            kind=kind,
            total_batches_per_epoch=total_batches_per_epoch,
            total_epochs=total_epochs,
            log_every=log_every,
        )
    else:  # ProgressMode.OFF
        handle = _NoOpHandle()

    # Install SIGINT handler that tears down Live before raising KeyboardInterrupt.
    prior_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum: int, frame: Any) -> None:
        if rich_prog is not None:
            rich_prog.stop()
        if callable(prior_sigint):
            prior_sigint(signum, frame)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    # Swap module-level singleton.
    progress = handle
    try:
        yield
    finally:
        progress = _NoOpHandle()
        _SESSION_ACTIVE = False
        signal.signal(signal.SIGINT, prior_sigint)
        if rich_prog is not None:
            rich_prog.stop()
        root_logger.handlers.clear()
        for h in prior_handlers:
            root_logger.addHandler(h)
        logging.captureWarnings(False)
```

- [ ] **Step P2-1: Write `src/custom_sam_peft/cli/_progress.py`** with all the above components assembled into a single coherent module. Order of definitions: imports → `ProgressMode` → `ProgressKind` → `resolve_mode` → `_NoOpHandle` → `_ProgressHandle` → `format_eta` → `_PlainHandle` → `_silence_third_party_progress` → `_SESSION_ACTIVE` sentinel → `progress` singleton → `progress_session`.

  - [ ] **Step P2-1.5: Add `format_eta` helper and `_PlainHandle` class** — insert between `_ProgressHandle` and `_silence_third_party_progress`.

**`format_eta` helper:**

```python
def format_eta(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"
```

**`_PlainHandle` class** — PLAIN mode handle that emits one log line per `log_every` window through stdlib `logging`. No `rich.Progress`; no ANSI; pure append-only output. Implements the same interface as `_ProgressHandle` so all `P.*` call sites work unchanged:

```python
class _PlainHandle:
    """Plain-mode progress handle — emits one log line per log_every window.

    Used when mode == ProgressMode.PLAIN. No rich.Progress; no ANSI; pure
    stdlib logging output. The line format is the spec §4 contract:
        progress: <kind> epoch=E/TE step=S/TS loss=L it/s=R eta=ETA
    """

    def __init__(
        self,
        kind: ProgressKind,
        total_batches_per_epoch: int,
        total_epochs: int | None,
        log_every: int = 50,
    ) -> None:
        self._kind = kind
        self._total_batches = total_batches_per_epoch
        self._total_epochs = total_epochs
        self._log_every = log_every
        self._step = 0
        self._epoch = 0
        self._postfix: dict[str, Any] = {}
        self._logger = logging.getLogger("custom_sam_peft.progress")
        self._start_time = time.monotonic()

    @property
    def console(self) -> Console:
        return Console()  # plain console; logs route through stdlib logger

    def advance_outer(self, n: int = 1) -> None:
        self._epoch += n

    def reset_inner(self, total: int | None = None) -> None:
        if total is not None:
            self._total_batches = total
        self._step = 0

    def advance_inner(self, n: int = 1) -> None:
        self._step += n
        if self._step % self._log_every == 0 or self._step == self._total_batches:
            self._emit()

    def update_postfix(self, **kwargs: Any) -> None:
        self._postfix.update(kwargs)

    @contextmanager
    def push_subtask(self, label: str, total: int) -> Generator[None, None, None]:
        # Plain mode: subtask boundaries are logged as plain lines.
        self._logger.info("progress: %s subtask=%s start total=%d", self._kind.value, label, total)
        try:
            yield
        finally:
            self._logger.info("progress: %s subtask=%s end", self._kind.value, label)

    def _emit(self) -> None:
        """Emit one progress line in the spec §4 format (global step)."""
        # Compute global step across epochs and total steps.
        if self._total_epochs is not None and self._total_batches > 0:
            global_step = self._epoch * self._total_batches + self._step
            global_total = self._total_epochs * self._total_batches
            epoch_str = f"epoch={self._epoch + 1}/{self._total_epochs}"
            step_str = f"step={global_step}/{global_total}"
        else:
            epoch_str = ""
            step_str = f"step={self._step}/{self._total_batches}"

        # Format postfix metrics.
        postfix_parts = []
        if "loss" in self._postfix:
            postfix_parts.append(f"loss={self._postfix['loss']:.3f}")
        if "it_s" in self._postfix:
            postfix_parts.append(f"it/s={self._postfix['it_s']:.1f}")
        # Compute ETA from elapsed and progress.
        elapsed = time.monotonic() - self._start_time
        if self._step > 0:
            total_for_eta = (
                self._total_epochs * self._total_batches
                if self._total_epochs is not None and self._total_batches > 0
                else self._total_batches
            )
            current = (
                self._epoch * self._total_batches + self._step
                if self._total_epochs is not None and self._total_batches > 0
                else self._step
            )
            eta_seconds = elapsed * (total_for_eta - current) / max(current, 1)
            eta = format_eta(eta_seconds)
            postfix_parts.append(f"eta={eta}")

        parts = [f"progress: {self._kind.value}"]
        if epoch_str:
            parts.append(epoch_str)
        parts.append(step_str)
        parts.extend(postfix_parts)
        self._logger.info(" ".join(parts))
```

- [ ] **Step P2-2: Verify the module imports cleanly**

```bash
uv run python -c "
from custom_sam_peft.cli._progress import (
    progress, progress_session, resolve_mode, ProgressKind, ProgressMode,
    _silence_third_party_progress,
)
print('imports OK')
print('P type:', type(progress).__name__)
"
```

Expected: prints `imports OK` and `P type: _NoOpHandle`. Exits 0.

### Task 2b: Update `_logging.py` — add `console` kwarg

- [ ] **Step P2-3: Edit `src/custom_sam_peft/cli/_logging.py`**

Current full file content:

```python
"""Shared CLI logging setup. Idempotent — safe to call from every command."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool) -> None:
    """Configure root logging for a custom-sam-peft CLI invocation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,  # Override pytest/dev-tool prior config.
    )
```

Replace with:

```python
"""Shared CLI logging setup. Idempotent — safe to call from every command."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(verbose: bool, console: Console | None = None) -> None:
    """Configure root logging for a custom-sam-peft CLI invocation.

    When ``console`` is provided, attaches a ``RichHandler`` backed by that
    console so log output flows through an existing rich Live display. The
    default (``console=None``) uses a plain ``basicConfig`` format — unchanged
    from before.

    Note: ``progress_session`` uses its own handler-attachment path (spec §7.2)
    and does not call this function with ``console``. This kwarg exists for
    library callers or custom CLI wrappers that inject a console without a full
    session.
    """
    level = logging.DEBUG if verbose else logging.INFO
    if console is not None:
        handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,  # Override pytest/dev-tool prior config.
        )
```

- [ ] **Step P2-4: Verify `configure_logging` still works with no `console` arg**

```bash
uv run python -c "
from custom_sam_peft.cli._logging import configure_logging
configure_logging(False)
import logging
logging.getLogger('test').info('configure_logging ok')
print('_logging OK')
"
```

Expected: prints a log line and `_logging OK`. Exits 0.

### Task 2c: Update `main.py` — call `_silence_third_party_progress()`

- [ ] **Step P2-5: Edit `src/custom_sam_peft/cli/main.py`**

Current file imports `typer` and the six CLI modules. The `_silence_third_party_progress()` call should be made once at module load, after the imports and before the `app = typer.Typer(...)` line. This ensures it fires before any subcommand dispatch regardless of mode.

Replace the current file with:

```python
"""`custom-sam-peft` CLI entry point — wires subcommands into a Typer app."""

from __future__ import annotations

import typer

import custom_sam_peft._bootstrap  # noqa: F401  # populate plugin registry before subcommand imports
from custom_sam_peft.cli import (
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    run_cmd,
    train_cmd,
)
from custom_sam_peft.cli._progress import _silence_third_party_progress

# Suppress HF / datasets progress bars once at app entry, unconditionally.
# progress_session also calls this defensively on entry — the double-call is safe.
_silence_third_party_progress()

app = typer.Typer(
    name="custom-sam-peft",
    help="Closed-vocab finetuning of SAM-family models with LoRA / QLoRA.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("train", help="Run a finetune.")(train_cmd.train)
app.command("eval", help="Evaluate a checkpoint.")(eval_cmd.evaluate)
app.command("export", help="Export adapter or merged model.")(export_cmd.export)
app.command("init", help="Write a starter config.")(init_cmd.init)
app.command("doctor", help="Report environment + dependency status.")(doctor_cmd.doctor)
app.command("run", help="Train + eval + (optional) export + bundle, in one shot.")(run_cmd.run)


if __name__ == "__main__":  # pragma: no cover
    app()
```

- [ ] **Step P2-6: Verify the app still imports and --help works**

```bash
uv run custom-sam-peft --help
```

Expected: prints the top-level help text and exits 0. If it fails with an import error, check that `_silence_third_party_progress` is exported from `_progress.py` without triggering the `datasets` import at module level (the `try/except ImportError` handles it).

- [ ] **Step P2-7: Commit Phase 2 changes**

```bash
git add \
  src/custom_sam_peft/cli/_progress.py \
  src/custom_sam_peft/cli/_logging.py \
  src/custom_sam_peft/cli/main.py
git commit -m "feat(cli): add _progress.py module with progress_session, resolve_mode, P singleton (#76)"
```

---

## Phase 3: Write `tests/unit/test_progress_resolve.py` (test A)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 4 and 5. **Depends on:** Phase 2 committed.
**Spec:** §9, test A.

**Files:**
- Create: `tests/unit/test_progress_resolve.py`

**Goal:** Parametrized test over the `resolve_mode` pure function. This is the simplest of the test files and has zero dependency on `_progress.py` internals — it only needs `resolve_mode`, `ProgressMode` exported.

### Task 3a: Write the test file

- [ ] **Step P3-1: Write `tests/unit/test_progress_resolve.py`**

```python
"""Test A: resolve_mode matrix (spec §9 / §4).

Pure-function test — no session, no side effects, no file I/O.
"""
from __future__ import annotations

import pytest

from custom_sam_peft.cli._progress import ProgressMode, resolve_mode


@pytest.mark.parametrize(
    ("cli_flag", "env", "isatty", "is_jupyter", "expected"),
    [
        # Explicit flag wins over everything.
        ("on",    {},                           False, True,  ProgressMode.ON),
        ("on",    {"CSP_NO_PROGRESS": "1"},     False, True,  ProgressMode.ON),
        ("off",   {},                           True,  False, ProgressMode.OFF),
        ("plain", {},                           True,  False, ProgressMode.PLAIN),
        # auto/None with CSP_NO_PROGRESS=1 → OFF.
        ("auto",  {"CSP_NO_PROGRESS": "1"},     True,  False, ProgressMode.OFF),
        (None,    {"CSP_NO_PROGRESS": "1"},     True,  False, ProgressMode.OFF),
        # Jupyter auto-fallback → PLAIN.
        ("auto",  {},                           True,  True,  ProgressMode.PLAIN),
        (None,    {},                           True,  True,  ProgressMode.PLAIN),
        # Non-TTY auto-fallback → PLAIN (Jupyter already handled above).
        ("auto",  {},                           False, False, ProgressMode.PLAIN),
        (None,    {},                           False, False, ProgressMode.PLAIN),
        # TTY, no env, no Jupyter → ON.
        ("auto",  {},                           True,  False, ProgressMode.ON),
        (None,    {},                           True,  False, ProgressMode.ON),
    ],
)
def test_resolve_mode_matrix(
    cli_flag: str | None,
    env: dict[str, str],
    isatty: bool,
    is_jupyter: bool,
    expected: ProgressMode,
) -> None:
    """Test A: resolve_mode covers flag > env > auto fallback precedence."""
    result = resolve_mode(cli_flag, env, isatty, is_jupyter)
    assert result == expected, (
        f"resolve_mode({cli_flag!r}, {env}, isatty={isatty}, is_jupyter={is_jupyter}) "
        f"returned {result!r}, expected {expected!r}"
    )
```

- [ ] **Step P3-2: Run test A to verify it passes**

```bash
uv run pytest tests/unit/test_progress_resolve.py -v
```

Expected: 12 parametrized cases, all green. Exit 0.

- [ ] **Step P3-3: Commit**

```bash
git add tests/unit/test_progress_resolve.py
git commit -m "test(cli): test A — resolve_mode parametrized matrix (#76)"
```

---

## Phase 4: Write `tests/unit/test_progress_module.py` and `tests/unit/test_progress_integration.py` (tests B–J)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 3 and 5. **Depends on:** Phase 2 committed.
**Spec:** §9, tests B–J.

**Files:**
- Create: `tests/unit/test_progress_module.py`
- Create: `tests/unit/test_progress_integration.py`

**Goal:** Tests B–H (module lifecycle, log routing, push_subtask, plain mode snapshot, third-party silencing, SIGINT) and tests I–J (fake trainer smoke test, ruff T201 lint rule guard).

### Task 4a: Write `tests/unit/test_progress_module.py` (tests B–H)

- [ ] **Step P4-1: Write `tests/unit/test_progress_module.py`**

```python
"""Tests B–H: _progress.py module lifecycle, routing, and env handling (spec §9).

All CPU-only. No GPU markers.
"""
from __future__ import annotations

import io
import logging
import os
import signal
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from custom_sam_peft.cli._progress import (
    ProgressKind,
    ProgressMode,
    _NoOpHandle,
    _silence_third_party_progress,
    progress,
    progress_session,
)


# ---------------------------------------------------------------------------
# Test B: no-op default
# ---------------------------------------------------------------------------

def test_no_op_default() -> None:
    """Test B: P.* calls outside any session are no-ops — no exception, no terminal writes."""
    # Calling all handle methods must not raise.
    from custom_sam_peft.cli._progress import progress as P

    assert isinstance(P, _NoOpHandle), "expected _NoOpHandle when no session is active"

    P.advance_outer()
    P.advance_inner()
    P.advance_inner(n=5)
    P.update_postfix(loss=0.5, lr=1e-4)

    # console property returns a plain Console without writing anything.
    con = P.console
    buf = io.StringIO()
    con.file = buf
    con.log("hello")
    # No assertion on content — just no exception.

    # push_subtask is a no-op context manager.
    with P.push_subtask("test", total=10):
        pass


# ---------------------------------------------------------------------------
# Test C: session lifecycle + nesting raises RuntimeError
# ---------------------------------------------------------------------------

def test_session_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    """Test C: entering a session attaches RichHandler; exiting restores prior handlers.
    Opening a second session before the first exits raises RuntimeError.
    """
    root = logging.getLogger()
    # Record prior handler ids.
    prior_handler_ids = [id(h) for h in root.handlers]

    from rich.console import Console

    console = Console(file=io.StringIO(), no_color=True)

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=1,
        total_batches_per_epoch=10,
        mode=ProgressMode.OFF,  # OFF: no rich.Progress — just session bookkeeping
    ):
        # Session is active: _SESSION_ACTIVE is True. Attempting to open another should fail.
        with pytest.raises(RuntimeError, match="nested session"):
            with progress_session(
                kind=ProgressKind.EVAL,
                total_batches_per_epoch=5,
                mode=ProgressMode.OFF,
            ):
                pass  # unreachable

    # After exit: prior handlers restored.
    restored_ids = [id(h) for h in root.handlers]
    assert restored_ids == prior_handler_ids, (
        f"handlers after session: {restored_ids} != prior {prior_handler_ids}"
    )


# ---------------------------------------------------------------------------
# Test D: log routing through Live (ON mode)
# ---------------------------------------------------------------------------

def test_log_through_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test D: logger.info() inside an ON session writes exactly one line to the progress console.

    Verifies that RichHandler is properly attached and routes log output above
    the pinned bar without duplication.
    """
    import io
    from rich.console import Console as _RichConsole

    captured = io.StringIO()

    # Monkeypatch Console() inside _progress.py so the session's rich.Progress
    # uses our StringIO-backed console.
    def _fake_console(*args: Any, **kwargs: Any) -> _RichConsole:
        kwargs.pop("stderr", None)
        return _RichConsole(file=captured, force_terminal=True, no_color=True, width=120)

    monkeypatch.setattr("custom_sam_peft.cli._progress.Console", _fake_console)

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=1,
        total_batches_per_epoch=10,
        mode=ProgressMode.ON,
    ):
        logging.getLogger("test.d").info("test-D unique-marker-12345")

    output = captured.getvalue()
    assert output.count("test-D unique-marker-12345") == 1, (
        f"Expected log line to appear exactly once in progress console output, "
        f"got count={output.count('test-D unique-marker-12345')}.\nFull output:\n{output}"
    )


# ---------------------------------------------------------------------------
# Test E: push_subtask lifecycle
# ---------------------------------------------------------------------------

def test_push_subtask_lifecycle() -> None:
    """Test E: push_subtask adds a task inside the block; the task is removed on exit."""
    from custom_sam_peft.cli._progress import progress as P

    # No session: push_subtask is a no-op — just confirm no exception.
    with P.push_subtask("lite-eval", total=10):
        pass

    # With a session in ON mode, verify the subtask is added then removed.
    import custom_sam_peft.cli._progress as _pmod

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=2,
        total_batches_per_epoch=3,
        mode=ProgressMode.ON,
    ):
        live_handle = _pmod.progress
        # _ProgressHandle should be active.
        assert not isinstance(live_handle, _NoOpHandle), "expected _ProgressHandle inside session"
        task_count_before = len(
            [t for t in live_handle._progress.tasks if not t.finished]  # type: ignore[union-attr]
        )
        with live_handle.push_subtask("lite-eval", total=10):
            task_count_during = len(live_handle._progress.tasks)  # type: ignore[union-attr]
            assert task_count_during > task_count_before, "subtask should be added during block"
        # After block: task should be removed.
        task_count_after = len(live_handle._progress.tasks)  # type: ignore[union-attr]
        assert task_count_after == task_count_before, "subtask should be removed on exit"


# ---------------------------------------------------------------------------
# Test F: plain mode line snapshot
# ---------------------------------------------------------------------------

def test_plain_line_snapshot(caplog: pytest.LogCaptureFixture) -> None:
    """Test F: frozen snapshot of the plain-mode progress line format.

    Calls into _PlainHandle._emit directly with fixed inputs and asserts the
    captured log line exactly matches the spec §4 contract.

    Uses a startswith assertion for the ETA portion because time.monotonic()
    advances between _PlainHandle.__init__ and _emit(), making the exact
    sub-second component of ETA non-deterministic in CI. The prefix up to
    the seconds field is deterministic given the pinned elapsed time.

    Step field uses the global step (planner decision §4 PLAIN-mode step semantics):
    epoch 2 (0-indexed) × 4530 batches/epoch + 1240 local step = 10300 global step.
    """
    import logging
    from custom_sam_peft.cli._progress import _PlainHandle

    handle = _PlainHandle(
        kind=ProgressKind.TRAIN,
        total_batches_per_epoch=4530,
        total_epochs=10,
        log_every=50,
    )
    # Manually set internal state to the snapshot's inputs:
    handle._epoch = 2  # epoch 3 of 10 (0-indexed internally)
    handle._step = 1240
    handle._postfix = {"loss": 0.812, "it_s": 2.3}
    # Pin elapsed so ETA is deterministic.
    # For step=1240, total=45300, elapsed=t such that eta≈2530s = 0:42:10
    # eta = elapsed * (total - current) / current
    # 2530 = elapsed * (45300 - 10300) / 10300 = elapsed * 35000 / 10300
    # → elapsed = 2530 * 10300 / 35000 ≈ 744.43 seconds
    import time as _time
    handle._start_time = _time.monotonic() - 744.43

    with caplog.at_level(logging.INFO, logger="custom_sam_peft.progress"):
        handle._emit()

    # Expect exactly one log line matching the snapshot.
    matching = [r for r in caplog.records if "progress: train" in r.getMessage()]
    assert len(matching) == 1, f"expected exactly one progress line, got {len(matching)}"
    msg = matching[0].getMessage()

    # Snapshot — ETA is approximate due to monotonic() timing; assert prefix only.
    expected_prefix = "progress: train epoch=3/10 step=10300/45300 loss=0.812 it/s=2.3 eta=0:42:"
    assert msg.startswith(expected_prefix), (
        f"plain format snapshot mismatch:\n  got:      {msg!r}\n  expected prefix: {expected_prefix!r}"
    )


# ---------------------------------------------------------------------------
# Test G: _silence_third_party_progress
# ---------------------------------------------------------------------------

def test_silence_third_party_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test G: _silence_third_party_progress sets env vars and calls datasets.disable_progress_bar."""
    # Clear any existing values so setdefault takes effect.
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    mock_datasets = MagicMock()
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        _silence_third_party_progress()

    assert os.environ["TRANSFORMERS_VERBOSITY"] == "warning"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    mock_datasets.disable_progress_bar.assert_called_once()

    # Second call is idempotent — setdefault won't overwrite, mock called again once.
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        _silence_third_party_progress()

    assert mock_datasets.disable_progress_bar.call_count == 2


# ---------------------------------------------------------------------------
# Test H: SIGINT handler
# ---------------------------------------------------------------------------

def test_sigint_handler() -> None:
    """Test H (part 2 — clean-exit path): after progress_session exits normally,
    signal.getsignal(SIGINT) equals the handler registered before the session opened.
    """
    prior_handler = signal.getsignal(signal.SIGINT)

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=1,
        total_batches_per_epoch=5,
        mode=ProgressMode.OFF,
    ):
        pass  # normal exit

    restored = signal.getsignal(signal.SIGINT)
    assert restored == prior_handler, (
        f"SIGINT handler not restored after session: {restored!r} != {prior_handler!r}"
    )
```

- [ ] **Step P4-2: Run tests B–H to verify they pass**

```bash
uv run pytest tests/unit/test_progress_module.py -v
```

Expected: 7 tests (B through H), all green. Exit 0.

Common failure modes:
- Test C "nested session" error not raised: the `_SESSION_ACTIVE` global is not being checked at session entry. Verify `_SESSION_ACTIVE` is set to `True` before `yield` in `progress_session`.
- Test G "mock called once" failing: `datasets` may already be in `sys.modules` from a prior import. The `patch.dict("sys.modules", ...)` approach correctly overrides for the duration of the call.
- Test E "push_subtask" failing with AttributeError on `_progress.tasks`: the `_ProgressHandle._progress` attribute holds the `rich.Progress` instance — `progress.tasks` is a property on `rich.Progress`.

### Task 4b: Write `tests/unit/test_progress_integration.py` (tests I–J)

- [ ] **Step P4-3: Write `tests/unit/test_progress_integration.py`**

```python
"""Tests I–J: integration smoke test and ruff T201 lint rule guard (spec §9).

All CPU-only. No GPU markers.
"""
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from custom_sam_peft.cli._progress import (
    ProgressKind,
    ProgressMode,
    progress,
    progress_session,
)
import custom_sam_peft.cli._progress as _pmod


# ---------------------------------------------------------------------------
# Test I: fake trainer smoke test
# ---------------------------------------------------------------------------

def test_fake_trainer_smoke() -> None:
    """Test I: dummy train loop with progress_session(kind=TRAIN, mode=ON).

    Verifies the real data flow contract: advance_outer twice, advance_inner
    six times total (3 batches × 2 epochs), update_postfix at epoch end.
    No GPU required.
    """
    total_epochs = 2
    batches_per_epoch = 3

    outer_advances = 0
    inner_advances = 0

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_epochs=total_epochs,
        total_batches_per_epoch=batches_per_epoch,
        mode=ProgressMode.ON,
    ):
        handle = _pmod.progress
        assert not isinstance(handle, _pmod._NoOpHandle), "expected live handle inside session"

        for epoch in range(total_epochs):
            handle.advance_outer()
            outer_advances += 1
            handle.reset_inner()

            for batch in range(batches_per_epoch):
                handle.advance_inner()
                inner_advances += 1

            handle.update_postfix(loss=0.5 - epoch * 0.1, it_s=2.3)

    # After session exits: singleton reverts to _NoOpHandle.
    assert isinstance(_pmod.progress, _pmod._NoOpHandle), (
        "expected _NoOpHandle after session exits"
    )
    assert outer_advances == total_epochs, f"expected {total_epochs} outer advances"
    assert inner_advances == total_epochs * batches_per_epoch, (
        f"expected {total_epochs * batches_per_epoch} inner advances"
    )


# ---------------------------------------------------------------------------
# Test J: ruff T201 lint rule guard
# ---------------------------------------------------------------------------

def test_ruff_t201_lint_rule() -> None:
    """Test J: a file with bare print() fails T201; one with # noqa: T201 passes.

    Guards the lint-config change against accidental reversion.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # File that should FAIL lint (bare print).
        failing = tmp_path / "bad.py"
        failing.write_text("print('x')\n")

        result_fail = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "T201", str(failing)],
            capture_output=True,
            text=True,
        )
        assert result_fail.returncode != 0, (
            f"Expected ruff T201 to fail on bare print(), got returncode=0.\n"
            f"stdout: {result_fail.stdout}\nstderr: {result_fail.stderr}"
        )

        # File that should PASS lint (noqa: T201).
        import json

        passing = tmp_path / "good.py"
        passing.write_text(
            textwrap.dedent("""\
                import json
                x = {"a": 1}
                print(json.dumps(x))  # noqa: T201
            """)
        )

        result_pass = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "T201", str(passing)],
            capture_output=True,
            text=True,
        )
        assert result_pass.returncode == 0, (
            f"Expected ruff T201 to pass with # noqa: T201, got returncode={result_pass.returncode}.\n"
            f"stdout: {result_pass.stdout}\nstderr: {result_pass.stderr}"
        )
```

- [ ] **Step P4-4: Run tests I–J to verify they pass**

```bash
uv run pytest tests/unit/test_progress_integration.py -v
```

Expected: 2 tests (I and J), all green. Exit 0.

Common failure modes:
- Test I "expected live handle inside session": `progress_session` is not swapping the module-level `progress` singleton to a `_ProgressHandle` when `mode=ON`. Check that the `progress = handle` assignment inside `progress_session` modifies the module global (not a local variable).
- Test J ruff subprocess: `sys.executable -m ruff` must resolve to the project's ruff. If it fails with "No module named ruff", use `uv run ruff` instead — but using `sys.executable -m ruff` is correct since ruff is a dev dependency installed in the same venv.

- [ ] **Step P4-5: Commit Phase 4 test files**

```bash
git add tests/unit/test_progress_module.py tests/unit/test_progress_integration.py
git commit -m "test(cli): tests B–J — module lifecycle, routing, smoke, lint rule (#76)"
```

---

## Phase 5: CLI command wiring and train/eval call sites

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 3 and 4. **Depends on:** Phase 2 committed.
**Spec:** §5.1, §5.2, §5.3, §5.5.

**Files:**
- Modify: `src/custom_sam_peft/cli/train_cmd.py`
- Modify: `src/custom_sam_peft/cli/eval_cmd.py`
- Modify: `src/custom_sam_peft/cli/run_cmd.py`
- Modify: `src/custom_sam_peft/cli/export_cmd.py`
- Modify: `src/custom_sam_peft/train/loop.py`
- Modify: `src/custom_sam_peft/train/trainer.py`
- Modify: `src/custom_sam_peft/eval/evaluator.py`

**Goal:** Wire `--progress` option and `progress_session` into all four CLI commands, and add `P.*` call sites to `run_epoch`, `Trainer.fit`, and `Evaluator.evaluate`.

### Task 5a: Wire `train_cmd.py`

- [ ] **Step P5-1: Replace `src/custom_sam_peft/cli/train_cmd.py`**

```python
"""`custom-sam-peft train` — thin CLI shell over custom_sam_peft.train.runner.run_training."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._progress import ProgressKind, ProgressMode, progress_session, resolve_mode
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training


def train(
    config: Path = typer.Option(..., "--config", help="Path to training config YAML."),
    override: list[str] = typer.Option(
        [], "--override", help="Override config keys: dotted.key=value."
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Path to resume checkpoint."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
) -> None:
    """Run a finetune."""
    configure_logging(verbose)
    cfg = load_config(config, overrides=override)
    if cfg.data.prompt_mode == "bbox":
        raise typer.BadParameter(
            "prompt_mode='bbox' is not supported for training in v0.",
            param_hint="--config",
        )

    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    # Determine total_batches_per_epoch lazily — runner builds the loader.
    # Use a placeholder of 0 so the bar shows indeterminate total until the
    # runner calls back. The runner is responsible for calling P.advance_inner.
    # A better approach: pass after loading config; len(loader) needs the loader.
    # For now, pass cfg.train.epochs and let the trainer drive the bar via P.*.
    try:
        with progress_session(
            kind=ProgressKind.TRAIN,
            total_epochs=cfg.train.epochs,
            total_batches_per_epoch=0,  # Trainer updates dynamically via reset_inner
            mode=mode,
        ):
            result = run_training(cfg, resume_from=resume)
    except (ValueError, NotImplementedError) as e:
        rprint(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e

    rprint(f"[green]done[/green] run_dir={result.run_dir} adapter={result.adapter_path}")
    if result.final_metrics is not None:
        rprint(f"  mAP={result.final_metrics.overall.get('mAP', float('nan')):.4f}")
```

**Implementation note:** The `total_batches_per_epoch=0` placeholder is intentional. The `Trainer.fit` loop uses `max(len(train_loader), 1)` to determine the actual batch count, and `P.reset_inner()` is called at the start of each epoch (Phase 5c below) — the `rich.Progress` task is updated with the real total at that point by passing `total=` on `reset_inner`. This requires a small extension to `_ProgressHandle.reset_inner`:

```python
def reset_inner(self, total: int | None = None) -> None:
    update_kwargs: dict[str, Any] = {"completed": 0}
    if total is not None:
        update_kwargs["total"] = total
        self._total_batches = total
    self._progress.update(self._inner, **update_kwargs)
    self._step = 0
```

Add this signature to `_ProgressHandle.reset_inner` in `_progress.py` and update the `_NoOpHandle.reset_inner` signature to match:

```python
def reset_inner(self, total: int | None = None) -> None:
    pass
```

- [ ] **Step P5-2: Verify `train_cmd.py` imports cleanly**

```bash
uv run python -c "from custom_sam_peft.cli.train_cmd import train; print('train_cmd OK')"
```

Expected: prints `train_cmd OK`. Exits 0.

### Task 5b: Wire `eval_cmd.py`

- [ ] **Step P5-3: Replace `src/custom_sam_peft/cli/eval_cmd.py`**

```python
"""`custom-sam-peft eval` — thin CLI shell over custom_sam_peft.eval.runner.run_eval."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, cast

import typer
from rich import print as rprint
from rich.console import Console

from custom_sam_peft.cli._logging import configure_logging
from custom_sam_peft.cli._progress import ProgressKind, ProgressMode, progress_session, resolve_mode
from custom_sam_peft.config.loader import load_config
from custom_sam_peft.eval.runner import run_eval


def evaluate(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    split: str = typer.Option("val", "--split", help="Dataset split: val | test."),
    output: Path | None = typer.Option(
        None, "--output", help="Output dir; defaults to checkpoint.parent."
    ),
    save_predictions: bool | None = typer.Option(
        None,
        "--save-predictions/--no-save-predictions",
        help="Override cfg.eval.save_predictions.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
) -> None:
    """Evaluate a checkpoint on the val or test split."""
    configure_logging(verbose)
    if split not in ("val", "test"):
        raise typer.BadParameter(f"--split must be val|test; got {split!r}", param_hint="--split")
    cfg = load_config(config)
    split_lit = cast(Literal["val", "test"], split)

    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )

    try:
        with progress_session(
            kind=ProgressKind.EVAL,
            total_batches_per_epoch=0,  # Evaluator updates via P.advance_inner
            mode=mode,
            # total_epochs intentionally omitted — no outer epoch bar for eval (planner decision)
        ):
            report = run_eval(
                cfg,
                checkpoint=checkpoint,
                split=split_lit,
                output_dir=output,
                save_predictions=save_predictions,
            )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--checkpoint") from e

    rprint(f"[green]eval complete[/green] — {report.overall}")
```

- [ ] **Step P5-4: Verify `eval_cmd.py` imports cleanly**

```bash
uv run python -c "from custom_sam_peft.cli.eval_cmd import evaluate; print('eval_cmd OK')"
```

Expected: prints `eval_cmd OK`. Exits 0.

### Task 5c: Wire `run_cmd.py` (three sequential sessions)

- [ ] **Step P5-5: Edit `src/custom_sam_peft/cli/run_cmd.py`**

The `run_cmd.py` orchestrates train → eval → optional export-merge. Each phase gets its own session. The `--progress` option is added to the `run()` function. The three sessions are sequential (spec §5.3).

Add the following import near the top of the file:

```python
import os
import sys

from rich.console import Console

from custom_sam_peft.cli._progress import ProgressKind, ProgressMode, progress_session, resolve_mode
```

Add `progress_flag` parameter to `run()`:

```python
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
```

Add mode resolution near the top of `run()` body (after `configure_logging`):

```python
    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )
```

Wrap the `_orchestrate(cfg, resume)` call inside a `progress_session` for the TRAIN phase. Because `_orchestrate` handles the train phase internally and calls `run_training` directly, the cleanest approach is to pass `mode` into `_orchestrate` and have it open the three sessions around the three phases. Update `_orchestrate` signature:

```python
def _orchestrate(cfg: TrainConfig, resume: Path | None, mode: ProgressMode) -> int:
```

Inside `_orchestrate`, wrap each phase:

```python
    # Phase: train.
    try:
        with progress_session(
            kind=ProgressKind.TRAIN,
            total_epochs=cfg.train.epochs,
            total_batches_per_epoch=0,
            mode=mode,
        ):
            train_result = run_training(cfg, resume_from=resume)
    except Exception as exc:
        rprint(f"[red]train failed[/red] {exc}")
        raise typer.Exit(code=1) from exc
```

```python
    # Phase: eval.
    try:
        with progress_session(
            kind=ProgressKind.EVAL,
            total_batches_per_epoch=0,
            mode=mode,
        ):
            report, per_example_iou = cast(
                tuple[Any, list[float]],
                run_eval(
                    cfg,
                    checkpoint=adapter_path,
                    output_dir=run_dir,
                    val_dataset=val_dataset,
                    model=wrapper,
                    return_per_example_iou=True,
                ),
            )
    except Exception as exc:
        rprint(f"[red]eval failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc
```

```python
    # Phase: export-merge (conditional, soft-fail).
    merged_dir: Path | None = None
    merged_export_error: str | None = None
    if cfg.export.merge:
        target = run_dir / "merged"
        try:
            with progress_session(
                kind=ProgressKind.EXPORT_MERGE,
                total_batches_per_epoch=0,
                mode=mode,
            ):
                save_merged(wrapper, target)
            merged_dir = target
        except Exception as exc:
            _LOG.warning("export-merge failed: %s", exc)
            merged_export_error = str(exc)
```

Update the `_orchestrate` call in `run()`:

```python
    _orchestrate(cfg, resume, mode)
```

- [ ] **Step P5-6: Verify `run_cmd.py` imports cleanly**

```bash
uv run python -c "from custom_sam_peft.cli.run_cmd import run; print('run_cmd OK')"
```

Expected: prints `run_cmd OK`. Exits 0.

### Task 5d: Wire `export_cmd.py`

- [ ] **Step P5-7: Edit `src/custom_sam_peft/cli/export_cmd.py`**

Add the following imports:

```python
import os
import sys

from rich.console import Console

from custom_sam_peft.cli._progress import ProgressKind, ProgressMode, progress_session, resolve_mode
```

Add `progress_flag` parameter to `export()`:

```python
    progress_flag: str = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto|on|off|plain.",
        metavar="MODE",
    ),
```

Add mode resolution near the top of `export()` body (after `configure_logging`):

```python
    mode = resolve_mode(
        progress_flag if progress_flag != "auto" else None,
        os.environ,
        sys.stdout.isatty(),
        Console().is_jupyter,
    )
```

Wrap the `save_merged` / `save_adapter` block inside a `progress_session`:

```python
    wrapper = load_sam31(cfg.model)
    load_adapter(wrapper, checkpoint)
    with progress_session(
        kind=ProgressKind.EXPORT_MERGE,
        total_batches_per_epoch=0,
        mode=mode,
    ):
        if merge:
            save_merged(wrapper, out)
        else:
            save_adapter(wrapper, out)

    if merge:
        rprint(f"[green]merged[/green] {out}")
    else:
        rprint(f"[green]adapter[/green] {out}")
```

Note: `rprint` calls are moved *outside* the `with` block so `Live` tears down first (spec §6.3).

- [ ] **Step P5-8: Verify `export_cmd.py` imports cleanly**

```bash
uv run python -c "from custom_sam_peft.cli.export_cmd import export; print('export_cmd OK')"
```

Expected: prints `export_cmd OK`. Exits 0.

### Task 5e: Add `P.*` call sites to `train/loop.py`

- [ ] **Step P5-9: Edit `src/custom_sam_peft/train/loop.py`**

Add the import at the top of the file (after the existing imports):

```python
from custom_sam_peft.cli._progress import progress as P
```

Inside `run_epoch`, after `window.update(result, lr=...)` and before the `log_every` branch, add:

```python
        P.advance_inner()
```

And inside the `log_every` branch (after `tracker.log_scalars`), add:

```python
        if global_step % cfg.train.log_every == 0:
            scalars = window.flush()
            tracker.log_scalars(global_step, scalars)
            P.update_postfix(
                loss=scalars.get("loss/total", 0.0),
                lr=scalars.get("lr", 0.0),
                it_s=scalars.get("throughput/img_s", 0.0),
            )
```

The full `run_epoch` batch loop body after the changes:

```python
    for batch in loader:
        result = train_step(
            model,
            batch,
            optimizer,
            scheduler,
            cfg,
            class_names=class_names,
            global_step=global_step,
            nan_streak=nan_streak,
        )
        nan_streak = result.nan_streak
        global_step += 1
        window.update(result, lr=float(scheduler.get_last_lr()[0]))
        P.advance_inner()
        if global_step % cfg.train.log_every == 0:
            scalars = window.flush()
            tracker.log_scalars(global_step, scalars)
            P.update_postfix(
                loss=scalars.get("loss/total", 0.0),
                lr=scalars.get("lr", 0.0),
                it_s=scalars.get("throughput/img_s", 0.0),
            )
        if global_step % cfg.train.save_every == 0:
            on_checkpoint(global_step, epoch, result.p_t, nan_streak)
        if global_step > 0 and global_step % cfg.train.eval_every == 0:
            on_eval(global_step)
    return global_step, nan_streak
```

### Task 5f: Add `P.*` call sites to `train/trainer.py`

- [ ] **Step P5-10: Edit `src/custom_sam_peft/train/trainer.py`**

Add the import at the top of the file (after the existing imports):

```python
from custom_sam_peft.cli._progress import progress as P
```

Inside `Trainer.fit`, at the top of the epoch loop (before `run_epoch` is called), add:

```python
            for epoch in range(start_epoch, cfg.train.epochs):
                total_batches = max(len(train_loader), 1)
                P.reset_inner(total=total_batches)
                global_step, nan_streak = run_epoch(
                    ...
                )
                P.advance_outer()
```

The `P.reset_inner(total=total_batches)` call resets the inner bar to 0 and updates the total for this epoch. `P.advance_outer()` ticks the epoch counter at the end of each epoch.

Full epoch loop after the changes:

```python
            for epoch in range(start_epoch, cfg.train.epochs):
                total_batches = max(len(train_loader), 1)
                P.reset_inner(total=total_batches)
                global_step, nan_streak = run_epoch(
                    self.model,
                    train_loader,
                    optimizer,
                    scheduler,
                    self.tracker,
                    cfg,
                    run_dir,
                    epoch,
                    global_step,
                    nan_streak,
                    class_names,
                    self.val_ds,
                    on_checkpoint,
                    on_eval,
                )
                P.advance_outer()
```

### Task 5g: Add `P.*` call sites to `eval/evaluator.py`

- [ ] **Step P5-11: Edit `src/custom_sam_peft/eval/evaluator.py`**

Add the import at the top of the file (after the existing imports):

```python
from custom_sam_peft.cli._progress import progress as P
```

Inside `Evaluator.evaluate`, after building `examples` (the pre-fetched list) and before the forward loop, add a reset call to set the total:

```python
        examples = [dataset[i] for i in indices]
        P.reset_inner(total=len(examples))
```

Inside the forward loop, after each image's classes are processed (after the inner `for cat_idx, class_name in enumerate(dataset.class_names)` loop completes for a given `ex`), add:

```python
        with torch.no_grad():
            for ex in examples:
                original_hw = (int(ex.image.shape[-2]), int(ex.image.shape[-1]))
                int_id = _int_image_id(ex.image_id)
                for cat_idx, class_name in enumerate(dataset.class_names):
                    ...
                    predictions.extend(entries)
                P.advance_inner()  # per image, not per (image, class)
```

Every `max(1, N // 50)` images, emit postfix (where N = len(examples)):

```python
        log_every_n = max(1, len(examples) // 50)
        with torch.no_grad():
            for img_idx, ex in enumerate(examples):
                ...
                for cat_idx, class_name in enumerate(dataset.class_names):
                    ...
                    predictions.extend(entries)
                P.advance_inner()
                if (img_idx + 1) % log_every_n == 0:
                    # running_map is not available mid-loop; use placeholder
                    P.update_postfix(it_s=float(img_idx + 1))
```

- [ ] **Step P5-12: Verify all modified files still import cleanly**

```bash
uv run python -c "
from custom_sam_peft.train.loop import run_epoch
from custom_sam_peft.train.trainer import Trainer
from custom_sam_peft.eval.evaluator import Evaluator
print('train/eval callsite imports OK')
"
```

Expected: prints `train/eval callsite imports OK`. Exits 0.

- [ ] **Step P5-13: Commit Phase 5**

```bash
git add \
  src/custom_sam_peft/cli/train_cmd.py \
  src/custom_sam_peft/cli/eval_cmd.py \
  src/custom_sam_peft/cli/run_cmd.py \
  src/custom_sam_peft/cli/export_cmd.py \
  src/custom_sam_peft/train/loop.py \
  src/custom_sam_peft/train/trainer.py \
  src/custom_sam_peft/eval/evaluator.py
git commit -m "feat(cli): wire progress_session into train/eval/run/export and call P.* in loop/trainer/evaluator (#76)"
```

---

## Phase 6: Verification gate

**Model/effort:** sonnet / high.
**Parallel:** No. **Depends on:** Phases 1–5 all committed.
**Spec:** §9 (test plan), §8 (lint).

**Goal:** Run the full lint, type, test, and coverage suite. Fix any issues inline before marking this phase complete.

- [ ] **Step P6-1: Run ruff lint**

```bash
uv run ruff check src tests
```

Expected: exits 0. If `T201` fires on any file other than a file with `# noqa: T201`, there is a stray `print()` — fix it. If any other rule fires on newly added code, fix inline.

- [ ] **Step P6-2: Run ruff format check**

```bash
uv run ruff format --check src tests
```

Expected: exits 0 ("N files left unchanged"). If any file fails, run `uv run ruff format src tests` to auto-fix and commit the format changes separately.

- [ ] **Step P6-3: Run mypy strict**

```bash
uv run mypy src/custom_sam_peft
```

Expected: exits 0 with no errors. Common failure modes for the new module:
- Missing return type on `progress_session` generator: ensure the return type annotation is `Generator[None, None, None]`.
- `progress` module-level variable type: `_NoOpHandle | _ProgressHandle` should satisfy mypy if both classes are defined in the same module. If mypy complains about attribute access on the union type, add `# type: ignore[union-attr]` only as a last resort — prefer narrowing the type with `isinstance` checks in the handle methods.
- `signal.signal` type: `signal.signal(signal.SIGINT, _sigint_handler)` — the second argument type is `signal.Handlers | Callable[[int, FrameType | None], Any] | None`. The lambda/closure is compatible.

- [ ] **Step P6-4: Run all unit tests with coverage**

```bash
uv run pytest tests/unit/ -v
```

Expected: all 10 new tests (A through J) plus all pre-existing tests pass. Coverage gate (`--cov-fail-under=80`) passes. Exit 0.

If coverage fails, the new `_progress.py` module is likely not covered by tests B–I. Run:

```bash
uv run pytest tests/unit/ --cov=custom_sam_peft --cov-report=term-missing 2>&1 | grep "_progress"
```

to identify uncovered lines and add targeted assertions in the existing tests.

- [ ] **Step P6-5: Run pre-commit**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass. Fix any trailing whitespace or end-of-file issues inline and commit.

- [ ] **Step P6-6: Commit any fix-up changes from the gate**

If any files were changed during P6-1 through P6-5 (format fixes, mypy fixes):

```bash
git add -u
git commit -m "fix(cli): lint/format/type fixups from verification gate (#76)"
```

---

## Phase 7: PR — hold until #74 merges, then add predict fixup commit

**Model/effort:** sonnet / medium.
**Parallel:** No. **Depends on:** Phase 6 complete AND #74 merged.
**Spec:** §10 (rollout and sequencing).

**Goal:** Wait for #74 to merge (which adds `predict_cmd.py`), add a predict wiring fixup commit, then open the PR.

### Task 7a: Check #74 status

- [ ] **Step P7-1: Check whether #74 has merged**

```bash
gh pr list --search "is:merged" | grep "#74"
```

Or check by PR number:

```bash
gh pr view 74 --json state -q .state
```

Expected: `"MERGED"`. If #74 has not merged, **halt and notify the user** — the predict fixup commit cannot be applied until `predict_cmd.py` exists. The `feat-cli-progress-bars` branch is fully ready; just the PR needs to wait.

### Task 7b: Add predict wiring fixup commit (after #74 merges)

- [ ] **Step P7-2: Merge main into the working branch to pick up #74**

```bash
git fetch origin
git merge origin/main --no-edit
```

Expected: clean merge. `predict_cmd.py` now exists in the worktree.

- [ ] **Step P7-3: Wire `--progress` into `predict_cmd.py`**

Read the new `predict_cmd.py` that #74 added. Add the following, matching the pattern from `eval_cmd.py`:

1. Import `os`, `sys`, `Console`, `ProgressKind`, `ProgressMode`, `progress_session`, `resolve_mode` at the top.
2. Add `progress_flag: str = typer.Option("auto", "--progress", ...)` to the predict function signature.
3. Add `mode = resolve_mode(...)` call after `configure_logging`.
4. Wrap the predict runner call inside `progress_session(kind=ProgressKind.PREDICT, total_batches_per_epoch=len(image_paths), mode=mode)`.
5. Move any final `rprint(...)` success line outside the `with` block.

The predict runner itself should call `P.advance_inner()` per processed image (spec §5.4). If #74's predict runner does not yet have this, add `P.advance_inner()` in the same pattern as `evaluator.py`.

- [ ] **Step P7-4: Run the verification gate on the predict additions**

```bash
uv run ruff check src tests && uv run mypy src/custom_sam_peft && uv run pytest tests/unit/ -q
```

Expected: all green. Exit 0.

- [ ] **Step P7-5: Commit the predict fixup**

```bash
git add src/custom_sam_peft/cli/predict_cmd.py
# (and any runner file that received P.advance_inner calls)
git commit -m "feat(cli): wire --progress into predict_cmd.py after #74 (#76)"
```

### Task 7c: Open the PR

- [ ] **Step P7-6: Push the branch**

```bash
git push -u origin feat-cli-progress-bars
```

- [ ] **Step P7-7: Determine the next semver version**

```bash
git fetch --tags
git describe --tags --abbrev=0 2>/dev/null
```

Current version in `pyproject.toml` is `0.8.0`. This feature adds a user-facing CLI flag (`--progress`) and a new module — MINOR under pre-1.0 conventions. Next version: **`v0.9.0`**. Stamp it:

```bash
# Update pyproject.toml version field.
# Then:
git add pyproject.toml
git commit -m "chore: bump version to 0.9.0 for progress bars release (#76)"
git push
```

- [ ] **Step P7-8: Open the PR**

```bash
gh pr create \
  --assignee @me \
  --title "feat(cli): bottom-pinned progress bars for train/eval/predict/export (#76)" \
  --label "enhancement" \
  --body "$(cat <<'EOF'
## Summary

- Adds `src/custom_sam_peft/cli/_progress.py`: process-global `rich.Progress` handle with `progress_session` context manager, `resolve_mode` pure function, `ProgressKind`/`ProgressMode` enums, and a no-op default singleton `P`.
- Wires `--progress {auto|on|off|plain}` option into `train`, `eval`, `run`, `export`, and `predict` commands.
- Routes log output through `RichHandler(console=progress.console)` so logs scroll above the pinned bar.
- Suppresses HF/datasets third-party progress output via `_silence_third_party_progress()` called at app entry and defensively on session entry.
- Adds `P.advance_inner`, `P.update_postfix` to `train/loop.py`; `P.advance_outer`, `P.reset_inner` to `train/trainer.py`; `P.advance_inner`, `P.update_postfix` to `eval/evaluator.py`.
- Adds `"T20"` to `[tool.ruff.lint] select` (bans bare `print()`); adds `# noqa: T201` to the single legitimate `print()` in `doctor_cmd.py`.
- Ships 10 CPU-only unit tests (A through J) covering mode resolution, session lifecycle, log routing, subtask lifecycle, plain-mode snapshot, third-party silencing, SIGINT restore, fake trainer smoke, and ruff T201 lint rule guard.

**Spec:** `docs/superpowers/specs/2026-05-22-cli-progress-bars-design.md`
**Plan:** `docs/superpowers/plans/2026-05-22-cli-progress-bars.md`
**Closes:** #76

## Test plan

- [ ] `uv run pytest tests/unit/ -v` — all 10 new tests (A–J) pass, all pre-existing tests pass.
- [ ] `uv run ruff check src tests` — exits 0 (T20 rule active, only `doctor_cmd.py` line has `# noqa: T201`).
- [ ] `uv run mypy src/custom_sam_peft` — exits 0 (strict mode).
- [ ] `uv run pre-commit run --all-files` — all hooks pass.
- [ ] Coverage gate (`--cov-fail-under=80`) passes.
- [ ] `custom-sam-peft --help` exits 0 on the branch.
EOF
  )"
```

- [ ] **Step P7-9: Watch CI**

Monitor CI without polling-sleeps. When CI is green, notify the user that the PR is ready to merge.

---

## Definition of done

All items below must be checked before the PR can be marked ready for review:

- [ ] `src/custom_sam_peft/cli/_progress.py` exists with `ProgressMode`, `ProgressKind`, `resolve_mode`, `_NoOpHandle`, `_ProgressHandle`, `_silence_third_party_progress`, `progress` (module-level singleton), `progress_session`.
- [ ] `src/custom_sam_peft/cli/_logging.py:configure_logging` accepts `console: Console | None = None`.
- [ ] `src/custom_sam_peft/cli/main.py` calls `_silence_third_party_progress()` at module load.
- [ ] `train_cmd.py`, `eval_cmd.py`, `run_cmd.py`, `export_cmd.py`, `predict_cmd.py` (post-#74) each have `--progress` option wired to `progress_session`.
- [ ] `doctor_cmd.py` line 83 has `# noqa: T201`.
- [ ] `pyproject.toml` `[tool.ruff.lint] select` includes `"T20"`.
- [ ] `train/loop.py` calls `P.advance_inner()` per batch and `P.update_postfix(...)` every `log_every` steps.
- [ ] `train/trainer.py` calls `P.reset_inner(total=...)` at epoch start and `P.advance_outer()` at epoch end.
- [ ] `eval/evaluator.py` calls `P.advance_inner()` per image and `P.update_postfix(...)` at cadence.
- [ ] All 10 tests (A–J) in `tests/unit/test_progress_*.py` pass.
- [ ] `uv run ruff check src tests` exits 0.
- [ ] `uv run mypy src/custom_sam_peft` exits 0.
- [ ] `uv run pytest tests/unit/` exits 0 (coverage gate met).
- [ ] `uv run pre-commit run --all-files` exits 0.
- [ ] PR body links issue #76, spec path, and plan path.
- [ ] Version bumped to `0.9.0` in `pyproject.toml`.

---

## Self-review

**1. Spec coverage:**

| Spec section | Covered by |
|---|---|
| §3.1 `progress_session` API | Phase 2, Task 2a |
| §3.2 `progress as P` singleton (no-op default) | Phase 2, Task 2a |
| §4 `resolve_mode` pure function + precedence rules | Phase 2, Task 2a; Phase 3, test A |
| §4 `--progress` CLI flag on each command | Phase 5 (all four commands) |
| §5.1 `train` call sites | Phase 5, Tasks 5a, 5e, 5f |
| §5.2 `eval` call sites; `total_epochs=None` decision | Phase 5, Tasks 5b, 5g; planner decisions §5.2 above |
| §5.3 `run` three sequential sessions | Phase 5, Task 5c |
| §5.4 `predict` contract | Phase 7, Task 7b |
| §5.5 `push_subtask` mid-train lite eval | Phase 2, Task 2a (_ProgressHandle.push_subtask); Phase 4, test E |
| §6.1 SIGINT handler | Phase 2, Task 2a; Phase 4, test H |
| §6.2 runner exception → stop + re-raise | Phase 2, Task 2a (finally block) |
| §6.3 success rprint outside with block | Phase 5, Tasks 5a/5b/5d |
| §7.1 `_silence_third_party_progress` | Phase 2, Task 2c; Phase 4, test G |
| §7.2 `RichHandler` attachment | Phase 2, Task 2a |
| §7.3 `configure_logging` `console` kwarg | Phase 2, Task 2b |
| §8 T20 rule + `# noqa: T201` | Phase 1; Phase 4, test J |
| §9 tests A–J | Phases 3–4 |
| §10 predict wiring decision | Planner decisions above; Phase 7 |
| §11 module layout | File Map above |

No spec section is missing a corresponding plan task.

**2. Placeholder scan:** No "TBD", "TODO", "implement later" language. The two `total_batches_per_epoch=0` calls in `train_cmd.py` and `eval_cmd.py` are intentional design choices (dynamic update via `reset_inner`) with explicit notes in the plan steps.

**3. Type consistency:**
- `_NoOpHandle.reset_inner(total: int | None = None)` — updated in Task 5a step P5-1 to match `_ProgressHandle.reset_inner(total: int | None = None)`.
- `ProgressMode.ON/OFF/PLAIN` — consistent across all phases (never `ProgressMode.AUTO`; `"auto"` is always a CLI-side string).
- `progress_session(mode: ProgressMode)` — receives `ProgressMode` (not `str`), resolved by `resolve_mode` before the call in all five command bodies.
- `P.update_postfix(**kwargs: Any)` — called with keyword arguments only; consistent in `loop.py` and `evaluator.py`.

**4. Ordering correctness:**
- Phase 1 (noqa + T20) before Phase 2 (new source). New source `_progress.py` has no `print()` calls, so T20 will not fire on it. But the rule is activated in Phase 1, so all subsequent phases are lint-clean from the start.
- Phase 2 (`_progress.py`) before Phases 3–5 (consumers). All consumers import `from custom_sam_peft.cli._progress import ...` — the module must exist before tests or CLI wiring can import it.
- Phase 7 (PR) after #74 merges and after Phase 6 (verification gate).
