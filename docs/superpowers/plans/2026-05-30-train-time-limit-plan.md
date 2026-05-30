# Wall-clock Time Limit with Resumable Graceful Stop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-30-train-time-limit-design.md`](../specs/2026-05-30-train-time-limit-design.md)
**Issue:** [#198](https://github.com/NguyenJus/custom-sam-peft/issues/198) — *Add a wall-clock time limit for train/run with resumable graceful stop*

**Goal:** Add an opt-in `train.time_limit` config knob + `--time-limit` CLI flag (on `train` and `run`) that cap an invocation by wall-clock time; on expiry the current micro-step finishes, a resumable full-state checkpoint is flushed, all end-of-run finalization is skipped, a resume message prints, and the process exits 0.

**Architecture:** A monotonic `deadline` is computed in `Trainer.fit()` and threaded down through `_train_epoch` → `run_epoch`, which checks it at the existing per-micro-step hook point (the same boundary as `nan_abort_after`). On expiry, `run_epoch` flushes a `save_full_state` checkpoint directly and raises an internal `_TimeLimitReached`; `fit()` catches it around the epoch loop, skips the entire post-loop finalize block while still closing the tracker, and returns an `EvalArtifacts` carrying a new optional `TimeLimitStop` field. The CLI inspects that field, prints a shared formatter's message, and exits 0 (`run` short-circuits before eval/export/bundle).

**Tech Stack:** Python 3.12, Pydantic v2 (`config/schema.py` strict `_Strict` models), frozen dataclasses (`eval/_artifacts.py` seam), PyTorch + SAM 3.1 (`train/loop.py`, `train/trainer.py`), Typer CLI (`cli/train_cmd.py`, `cli/run_cmd.py`), pytest + pytest-cov (TDD, `--cov-fail-under=80` gate), ruff + `mypy --strict` + markdownlint-cli2 (CI gates).

---

## Phase structure & boundary contracts (read first)

```text
Phase 1  ──►  Phase 2  ──►  Phase 3
Foundation    Enforcement   CLI + UX + docs (opens PR)
(pure utils)  core          (consumes Phase 2's TimeLimitStop)
```

Three sequential phases, each an independently-reviewable feature block. Every boundary states an **interface contract** so a later phase, executed in a fresh session with no memory of earlier phases' code, can build on it by reading only the contract.

| Phase | Scope (one line) | Interface contract OUT |
|---|---|---|
| 1 | Duration parser module + `TrainHyperparams.time_limit` field & validator | `config/_duration.py` exports `parse_duration_to_seconds(value: str \| int) -> int` and `format_seconds(seconds: int) -> str`; `TrainHyperparams.time_limit: str \| int \| None = None` is stored verbatim and validated strictly-positive. |
| 2 | `deadline` threading + post-step check + flush + `_TimeLimitReached` + `fit()` catch/skip-finalize + `TimeLimitStop`/`EvalArtifacts.time_limit_stop` + `_time_limited_artifacts` | `EvalArtifacts.time_limit_stop: TimeLimitStop \| None` (default `None`); `TimeLimitStop` is a frozen dataclass with the 7 fields below; `Trainer.fit(...)` returns it populated with exit-0 semantics on a time-limited stop, `None` otherwise. |
| 3 | `--time-limit` on both commands + shared message formatter + `run` short-circuit + `train` skip `--eval`/`--export` + docs | (terminal phase — opens the PR) |

### Phase 1 → Phase 2 contract (detail)

```python
# src/custom_sam_peft/config/_duration.py
def parse_duration_to_seconds(value: str | int) -> int: ...
def format_seconds(seconds: int) -> str: ...
```

- `parse_duration_to_seconds`: bare int / all-digit string → seconds; `<n>h<n>m<n>s` combos (each unit ≤ once, `h→m→s` order) → summed seconds; result must be `> 0`. Raises `ValueError` on `0`, negatives, `""`, whitespace-only, malformed strings (`"abc"`, `"10x"`, `"2h30"`), with a message naming the bad value.
- `format_seconds`: positive seconds → canonical largest-units string (`9000 → "2h30m"`, `5400 → "1h30m"`, `3600 → "1h"`, `90 → "1m30s"`, `45 → "45s"`); drops zero components.
- `TrainHyperparams.time_limit`: `str | int | None`, default `None`, **stored verbatim** (not normalized to seconds); a `@field_validator` runs `parse_duration_to_seconds` purely to validate non-`None` values and re-raises as a Pydantic `ValueError`/`ValidationError` naming the bad value.

### Phase 2 → Phase 3 contract (detail)

```python
# src/custom_sam_peft/eval/_artifacts.py
@dataclass(frozen=True)
class TimeLimitStop:
    stop_step: int
    stop_epoch: int          # zero-based epoch index at the stop
    total_epochs: int        # cfg.train.epochs
    checkpoint_dir: Path     # run_dir/checkpoints/step_<N>/
    duration_label: str      # format_seconds(budget_seconds), e.g. "2h30m"
    best_dir: Path | None    # run_dir/best/ if it exists, else None
    best_map: float | None   # best.json "value" if best/ exists, else None

@dataclass(frozen=True)
class EvalArtifacts:
    ...
    time_limit_stop: TimeLimitStop | None = field(default=None)   # NEW, defaults None
```

- On a time-limited stop, `Trainer.fit(...)` returns an `EvalArtifacts` with `time_limit_stop` set and `checkpoint_path == run_dir/checkpoints/step_<N>/adapter` (NOT `run_dir/adapter`, which is not written on a stop), `final_metrics is None`, and `oom_events` as usual. On the normal path `time_limit_stop is None` and all existing behavior is byte-for-byte unchanged.
- Phase 3 detects the stop solely via `result.time_limit_stop is not None`. It never reads trainer internals.

---

## File structure

### New files

- `src/custom_sam_peft/config/_duration.py` (Phase 1) — `parse_duration_to_seconds`, `format_seconds`. Pure, no I/O, no logging.
- `src/custom_sam_peft/cli/_time_limit.py` (Phase 3) — `format_time_limit_message(stop, *, subcommand, config_path) -> str`. Pure string builder, Typer-free.
- `tests/config/test_duration.py` (Phase 1)
- `tests/config/test_schema_time_limit.py` (Phase 1)
- `tests/train/test_time_limit_stop.py` (Phase 2)
- `tests/train/test_time_limit_resume.py` (Phase 2)
- `tests/train/test_time_limit_noop.py` (Phase 2)
- `tests/cli/test_time_limit_message.py` (Phase 3)
- `tests/cli/test_time_limit_cli.py` (Phase 3)

### Modified files

- `src/custom_sam_peft/config/schema.py` (Phase 1) — `TrainHyperparams.time_limit` field + `@field_validator`.
- `src/custom_sam_peft/eval/_artifacts.py` (Phase 2) — `TimeLimitStop` dataclass + `time_limit_stop` field.
- `src/custom_sam_peft/train/loop.py` (Phase 2) — `_TimeLimitReached`; `deadline` param on `run_epoch`; post-step check + direct `save_full_state` flush + raise.
- `src/custom_sam_peft/train/trainer.py` (Phase 2) — `deadline`/`budget_seconds` in `fit()`; `deadline` param on `_train_epoch`; restructured try/except/finally; `_time_limited_artifacts` helper.
- `src/custom_sam_peft/cli/train_cmd.py` (Phase 3) — `--time-limit` option + early-validate + override + stop detection (skip `--eval`/`--export`).
- `src/custom_sam_peft/cli/run_cmd.py` (Phase 3) — `--time-limit` option + early-validate + override; `_orchestrate` short-circuit.
- `tests/integration/test_trainer_evaluator_seam.py` (Phase 2) — one new assertion (`time_limit_stop is None` on the normal path).
- `docs/config-schema.md` (Phase 3) — `train.time_limit` advanced-field row + `--time-limit` CLI-flag note.
- `docs/defaults-provenance.md` (Phase 3) — optional note that `time_limit` is intentionally opt-in with no default.

---

## Code-aware notes & verified hazards (read before implementing)

Symbols are authoritative; line numbers may drift. All verified against the worktree at planning time.

1. **`paths.checkpoint_path` does NOT return the checkpoint directory.** `paths.checkpoint_path(run_dir, step=N)` returns `run_dir/checkpoints/step_{N:08d}.pt` — a **zero-padded `.pt` filename**, not the `step_<N>/` directory `save_full_state`/`find_latest_checkpoint` use. The existing `Trainer._maybe_checkpoint` (`trainer.py`) builds the state dir as `paths.checkpoint_path(run_dir, step=step).parent / f"step_{step}"` (i.e. take the `checkpoints/` parent, then join the **non-padded** `step_<N>`). The flush in `run_epoch` (Phase 2) **must replicate this exact pattern**; do not pass `checkpoint_path(...)`'s return as a directory.

2. **`on_checkpoint`'s real signature is 3-arg.** In the current code the closure is `def on_checkpoint(step: int, epoch: int, streak: int)` (`trainer.py`) and `run_epoch`'s param type is `Callable[[int, int, int], None]` (`loop.py`). The spec text mentions a 4-arg `on_checkpoint(step, epoch, p_t, streak)` form — that is from a pre-`box_hint`-removal codebase and is **not** the current state. The Phase 2 flush **prefers a direct `save_full_state(...)` call** (not `on_checkpoint`) so the panel render is skipped and the signature question is moot. The spec's mention of `result.p_t` for `box_hint_p` is **obsolete**: `StepResult` no longer carries `p_t` and `save_full_state` no longer takes `box_hint_p` (both removed in #88). Do not add either.

3. **The `try/finally` in `fit()` wraps BOTH the epoch loop AND the finalize block.** Currently `fit()` has one `try:` (opening just before `for epoch in ...`) whose `finally: self.tracker.close()` closes the tracker, and the body spans the epoch loop **plus** the post-loop finalize (`save_adapter`, optional merged export, end-of-run full eval, `metrics.json` write). The Phase 2 restructure must catch `_TimeLimitReached` around the epoch loop, and on a stop **skip the finalize block entirely** while still running `tracker.close()`. Verify the finalize block does not depend on the tracker being open (it does not: `save_adapter`/`save_merged`/`Evaluator(...).evaluate`/`metrics.json` write touch neither `self.tracker` nor the closed handle — the only tracker uses are `log_scalars`/`log_images` inside the epoch loop, which the stop has already exited).

4. **`checkpoint_path` on a time-limited stop points at the step checkpoint's adapter** (`run_dir/checkpoints/step_<N>/adapter`), not `run_dir/adapter` — because `save_adapter(run_dir/adapter)` is skipped on a stop. The only consumers that dereference `EvalArtifacts.checkpoint_path` are the eval/bundle phases, which the CLI short-circuits past on a stop (Phase 3), so this is consistent.

5. **Idempotent re-flush.** If the deadline trips exactly on a `save_every` boundary, the periodic checkpoint may have just written the same `step_<N>/` dir; `save_full_state` writes by step-keyed directory, so the re-flush overwrites the same dir harmlessly. No special-casing.

6. **`best.json` shape** (written by `Trainer._maybe_save_best`): `{"metric": "mAP", "value": <float>, "global_step": <int>}`. `_time_limited_artifacts` reads `value` for `best_map`. If `best/` is absent or `best.json` is unreadable, `best_dir`/`best_map` stay `None`.

7. **`time.monotonic` is already imported in both modules** (`loop.py` imports `time`; `trainer.py` does not yet — add `import time`). Use `time.monotonic()`, never `time.time()`.

---

## Verification gates (every phase)

Run from the repo root. The repo type-checks **strict** and CI runs:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/custom_sam_peft      # strict = true (pyproject [tool.mypy])
uv run pytest                        # global addopts carries --cov-fail-under=80
```

During fast iteration on a **subset**, the global `--cov-fail-under=80` in `addopts` will fail an under-80 partial run. Bypass it with `-o "addopts="` (clears all addopts incl. the cov gate; `--no-cov` does NOT work here — MEMORY: pytest subset coverage gate):

```bash
uv run pytest -o "addopts=" tests/config/test_duration.py -q   # example subset run
```

The **end-of-phase** verification runs the full gate (no `-o "addopts="`). All tests are **CPU-only**.

### Markdown lint gate (Phase 3 docs + this plan/spec)

Before committing any tracked `.md`, run CI's exact linter (`markdownlint-cli2@0.14.0`, config `.config/markdownlint-cli2.jsonc`, which disables only MD013/MD018/MD029). This box has no system node — use the Python-bundled Node path (MEMORY: markdown-lint gate):

```bash
uv run --no-project --with nodejs-bin python -c "
from nodejs import node, npx
import os, sys
os.environ['PATH'] = os.path.dirname(node.path) + os.pathsep + os.environ['PATH']
sys.exit(npx.run(['--yes','markdownlint-cli2@0.14.0','--config','.config/markdownlint-cli2.jsonc', *sys.argv[1:]]).returncode)
" docs/config-schema.md docs/defaults-provenance.md docs/superpowers/plans/2026-05-30-train-time-limit-plan.md docs/superpowers/specs/2026-05-30-train-time-limit-design.md
```

Expected: clean exit (0).

---

# PHASE 1 — Foundation: duration parser + config knob

**One coherent unit:** the pure parser/formatter module and the schema field that uses it for validation. Nothing here imports trainer or CLI code. Ends green at the full gate, then commits.

**Phase boundary — interface contract OUT:** restated at the top ("Phase 1 → Phase 2 contract"). Later phases import `from custom_sam_peft.config._duration import parse_duration_to_seconds, format_seconds` and read `cfg.train.time_limit`.

## Task 1: Duration parser/formatter (`config/_duration.py`)

**Files:**

- Create: `src/custom_sam_peft/config/_duration.py`
- Test: `tests/config/test_duration.py`

- [ ] **Step 1: Write the failing tests** (spec §11.1)

Create `tests/config/test_duration.py`:

```python
"""Unit tests for the duration parser/formatter (spec §11.1)."""

from __future__ import annotations

import pytest

from custom_sam_peft.config._duration import format_seconds, parse_duration_to_seconds


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2h30m", 9000),
        ("90m", 5400),
        ("3600s", 3600),
        (3600, 3600),
        ("3600", 3600),
        ("1h", 3600),
        ("45m", 2700),
        ("30s", 30),
        ("1h5m30s", 3930),
        ("  2h30m  ", 9000),  # surrounding whitespace tolerated
    ],
)
def test_parse_accepts(value: str | int, expected: int) -> None:
    assert parse_duration_to_seconds(value) == expected


@pytest.mark.parametrize(
    "value",
    [0, -1, "", "   ", "abc", "10x", "2h30", "-2h", "-5", "0s", "h", "1m2h"],
)
def test_parse_rejects(value: str | int) -> None:
    with pytest.raises(ValueError):
        parse_duration_to_seconds(value)


def test_parse_error_names_the_bad_value() -> None:
    with pytest.raises(ValueError, match="10x"):
        parse_duration_to_seconds("10x")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (9000, "2h30m"),
        (3600, "1h"),
        (5400, "1h30m"),  # canonical: 90m collapses to 1h30m
        (90, "1m30s"),
        (45, "45s"),
        (3930, "1h5m30s"),
    ],
)
def test_format(seconds: int, expected: str) -> None:
    assert format_seconds(seconds) == expected


@pytest.mark.parametrize("n", [1, 30, 45, 90, 3600, 3930, 5400, 9000])
def test_round_trip(n: int) -> None:
    assert parse_duration_to_seconds(format_seconds(n)) == n
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/config/test_duration.py -q`
Expected: FAIL — `custom_sam_peft.config._duration` does not exist (ModuleNotFoundError on collection).

- [ ] **Step 3: Implement `config/_duration.py`**

Create `src/custom_sam_peft/config/_duration.py`:

```python
"""Pure duration parsing/formatting utilities.

No I/O, no logging. Used by the config schema (validation) and the CLI
(early-validate + exit-message rendering). Spec §4.2.
"""

from __future__ import annotations

import re

# One <number><unit> group per unit, each optional, but enforced order h -> m -> s.
# An all-digits string is handled separately (bare seconds) before this matches.
_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")
_EXAMPLES = 'use e.g. "2h30m", "90m", "3600s", or bare seconds'


def parse_duration_to_seconds(value: str | int) -> int:
    """Parse a duration to a strictly-positive integer number of seconds.

    Accepts a bare int (3600), a bare all-digits string ("3600"), or an
    h/m/s combo ("1h", "45m", "30s", "2h30m", "1h5m30s"). Surrounding
    whitespace is tolerated. Units are lowercase, each at most once, in
    h -> m -> s order. Raises ValueError on non-positive results, empty /
    whitespace-only strings, or any string outside the grammar (e.g. "abc",
    "10x", "2h30" -- a trailing number with no unit).
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly.
        raise ValueError(f"time_limit: {value!r} is not a valid duration ({_EXAMPLES})")
    if isinstance(value, int):
        seconds = value
    else:
        text = value.strip()
        if not text:
            raise ValueError(f"time_limit: {value!r} is not a valid duration ({_EXAMPLES})")
        if text.isdigit():
            seconds = int(text)
        else:
            m = _DURATION_RE.fullmatch(text)
            if m is None or not any(m.groups()):
                raise ValueError(
                    f"time_limit: {value!r} is not a valid duration ({_EXAMPLES})"
                )
            h, mm, s = (int(g) if g else 0 for g in m.groups())
            seconds = h * 3600 + mm * 60 + s
    if seconds <= 0:
        raise ValueError(
            f"time_limit: {value!r} must be a strictly-positive duration ({_EXAMPLES})"
        )
    return seconds


def format_seconds(seconds: int) -> str:
    """Render a positive second count as a canonical human string.

    Collapses to the largest applicable units, dropping zero components:
    9000 -> "2h30m", 3600 -> "1h", 5400 -> "1h30m", 90 -> "1m30s", 45 -> "45s".
    A zero/negative total never reaches here (rejected upstream).
    """
    if seconds <= 0:
        raise ValueError(f"format_seconds: expected a positive second count, got {seconds!r}")
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    return "".join(parts)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/config/test_duration.py -q`
Expected: PASS (all parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/_duration.py tests/config/test_duration.py
git commit -m "feat(config): duration parser/formatter utility (#198)"
```

## Task 2: `TrainHyperparams.time_limit` field + validator

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` — add `time_limit` to `TrainHyperparams` (advanced section, after `eval_every`) + `@field_validator`.
- Test: `tests/config/test_schema_time_limit.py`

- [ ] **Step 1: Write the failing tests** (spec §11.2)

Create `tests/config/test_schema_time_limit.py`:

```python
"""Schema tests for TrainHyperparams.time_limit (spec §11.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import TrainHyperparams


def test_time_limit_defaults_none() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.time_limit is None


@pytest.mark.parametrize(("value", "expected"), [("2h30m", "2h30m"), (3600, 3600), ("3600", "3600")])
def test_time_limit_stored_verbatim(value: str | int, expected: str | int) -> None:
    """The field is validated but NOT normalized to seconds; it echoes what was passed."""
    hp = TrainHyperparams(epochs=1, time_limit=value)
    assert hp.time_limit == expected
    assert type(hp.time_limit) is type(expected)


@pytest.mark.parametrize("value", [0, -5, "abc", "10x", ""])
def test_time_limit_rejected(value: str | int) -> None:
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, time_limit=value)


def test_time_limit_error_names_bad_value() -> None:
    with pytest.raises(ValidationError, match="10x"):
        TrainHyperparams(epochs=1, time_limit="10x")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/config/test_schema_time_limit.py -q`
Expected: FAIL — `time_limit` is currently an unknown key; `_Strict`'s `extra="forbid"` raises `ValidationError` even for valid values, so `test_time_limit_stored_verbatim` and `test_time_limit_defaults_none` fail (no `time_limit` attribute).

- [ ] **Step 3: Add the field + validator to `TrainHyperparams`**

In `src/custom_sam_peft/config/schema.py`, add the import near the top (the module already imports `field_validator` from pydantic):

```python
from custom_sam_peft.config._duration import parse_duration_to_seconds
```

In `class TrainHyperparams(_Strict)`, in the `# --- advanced ---` section, **after** the `eval_every` field, add:

```python
    time_limit: str | int | None = Field(
        default=None,
        description=(
            "Wall-clock budget for this invocation. Accepts a human duration "
            '("2h30m", "90m", "3600s") or bare seconds (3600). None (default) '
            "means unlimited. The budget is per-run: --resume restarts the clock."
        ),
    )

    @field_validator("time_limit")
    @classmethod
    def _validate_time_limit(cls, v: str | int | None) -> str | int | None:
        """Validate (don't rewrite) the duration. Stored verbatim; parsed in fit()."""
        if v is None:
            return v
        parse_duration_to_seconds(v)  # raises ValueError on bad input; Pydantic wraps it
        return v
```

(`parse_duration_to_seconds` raising `ValueError` inside a `@field_validator` surfaces as a Pydantic `ValidationError` carrying the message — which names the bad value.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/config/test_schema_time_limit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/config/test_schema_time_limit.py
git commit -m "feat(config): TrainHyperparams.time_limit field + validator (#198)"
```

## Task 3: Phase 1 verification gate

**Files:** none (verification).

- [ ] **Step 1: Lint + type + format**

Run:

```bash
uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft
```

Expected: clean. (No new mypy issues: `_duration.py` is fully typed; the validator returns the declared union.)

- [ ] **Step 2: Full gated suite**

Run: `uv run pytest`
Expected: PASS including the `--cov-fail-under=80` gate. The two new test files cover the new module/field; existing tests are unaffected (no production call site yet reads `time_limit`).

---

# PHASE 2 — Enforcement core: clock, flush, graceful stop, artifacts

**Phase boundary — interface contract IN:** Phase 1's `parse_duration_to_seconds` / `format_seconds` and the verbatim `cfg.train.time_limit` field.

**One coherent unit:** all the trainer/loop wiring that makes a budget actually stop a run, flush a resumable checkpoint, skip finalization, and return the typed stop metadata. No CLI changes here — the CLI consumes the contract in Phase 3.

**Phase boundary — interface contract OUT:** restated at the top ("Phase 2 → Phase 3 contract"): `EvalArtifacts.time_limit_stop: TimeLimitStop | None` and the populated-on-stop / exit-0 semantics.

> **Implementation order note.** This phase is interdependent: the `run_epoch` flush+raise, the `fit()` catch, and the artifacts shape reference each other. Implement Task 4 (artifacts shape) first so the type exists, then Task 5 (loop flush+raise), then Task 6 (`fit()` restructure). Run the focused subset with `-o "addopts="` during iteration; the full gate runs at Task 8.

## Task 4: `TimeLimitStop` dataclass + `EvalArtifacts.time_limit_stop` field

**Files:**

- Modify: `src/custom_sam_peft/eval/_artifacts.py`
- Test: `tests/integration/test_trainer_evaluator_seam.py` (one new assertion; spec §11.8)

- [ ] **Step 1: Add the new assertion to the seam test first** (spec §11.8)

In `tests/integration/test_trainer_evaluator_seam.py`, inside `test_trainer_fit_returns_eval_artifacts`, after the existing `assert result.peft_method in {"lora", "qlora"}` line, add:

```python
    # Normal (non-time-limited) path: the new optional field defaults to None.
    assert result.time_limit_stop is None
```

- [ ] **Step 2: Run the seam test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/integration/test_trainer_evaluator_seam.py::test_trainer_fit_returns_eval_artifacts -q`
Expected: FAIL — `EvalArtifacts` has no `time_limit_stop` attribute yet (`AttributeError`).

- [ ] **Step 3: Add `TimeLimitStop` + the field**

In `src/custom_sam_peft/eval/_artifacts.py`, add `TimeLimitStop` above `EvalArtifacts` and the new field on `EvalArtifacts`. The file already imports `dataclass`, `field`, and `Path`:

```python
@dataclass(frozen=True)
class TimeLimitStop:
    """Set when Trainer.fit stopped early on a wall-clock budget. None otherwise.

    Carried on EvalArtifacts as an optional field the evaluator never reads;
    the CLI uses it to print the resume message and exit 0. Spec §4.7.
    """

    stop_step: int
    stop_epoch: int  # zero-based epoch index at the stop
    total_epochs: int  # cfg.train.epochs
    checkpoint_dir: Path  # run_dir/checkpoints/step_<N>/
    duration_label: str  # format_seconds(budget_seconds), e.g. "2h30m"
    best_dir: Path | None  # run_dir/best/ if it exists, else None
    best_map: float | None  # best.json "value" if best/ exists, else None
```

Then add to `EvalArtifacts` (after `oom_events`):

```python
    # Set when training stopped early on a wall-clock budget; None on the
    # normal path. The evaluator never reads it (seam-safe optional field).
    time_limit_stop: TimeLimitStop | None = field(default=None)
```

- [ ] **Step 4: Run the seam test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/integration/test_trainer_evaluator_seam.py -q`
Expected: PASS — all existing seam tests plus the new `time_limit_stop is None` assertion.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/_artifacts.py tests/integration/test_trainer_evaluator_seam.py
git commit -m "feat(eval): TimeLimitStop + optional EvalArtifacts.time_limit_stop field (#198)"
```

## Task 5: `_TimeLimitReached` + `deadline` check + direct flush in `run_epoch`

**Files:**

- Modify: `src/custom_sam_peft/train/loop.py` — `_TimeLimitReached` class; `deadline` param on `run_epoch`; post-step check + `save_full_state` flush + raise.
- Test: covered by `tests/train/test_time_limit_stop.py` (written in Task 7, exercised via `fit()`). A focused unit assert for the raise is included below.

> **Why direct `save_full_state`, not `on_checkpoint`:** the spec prefers a minimal flush that skips the image-panel render. `run_epoch` already receives `model`, `optimizer`, `scheduler`, `cfg`, `run_dir`, `epoch`, `global_step`, `nan_streak` — everything `save_full_state` needs. Build the state dir with the **exact** `_maybe_checkpoint` pattern (hazard note #1): `paths.checkpoint_path(run_dir, step=global_step).parent / f"step_{global_step}"`.

- [ ] **Step 1: Add `_TimeLimitReached` and the `deadline` param + check to `loop.py`**

In `src/custom_sam_peft/train/loop.py`, add imports near the existing ones (the module already imports `time`, `Path`, `Callable`):

```python
from custom_sam_peft import paths
from custom_sam_peft.train.checkpoint import save_full_state
```

> Note: `loop.py` does not currently import from `checkpoint.py`. Confirm no import cycle: `checkpoint.py` imports from `config.schema`, `errors`, `models.sam3`, `paths`, `peft_adapters` — none import `train.loop`, so `loop -> checkpoint` is acyclic.

Add the exception class near the top of the module (after `_MicrobatchExhausted`, or anywhere at module scope):

```python
class _TimeLimitReached(Exception):
    """Internal signal: the wall-clock budget expired. Carries the stop point.

    Graceful (exit 0), in contrast to the nan_abort_after RuntimeError which is
    a user-facing failure (exit 1). Never propagates past Trainer.fit(). Spec §4.5.
    """

    def __init__(self, step: int, epoch: int) -> None:
        super().__init__(f"time limit reached at step {step} (epoch {epoch})")
        self.step = step
        self.epoch = epoch
```

Add `deadline` to `run_epoch`'s signature (after `oom_state`, keyword with default):

```python
    oom_state: OomState | None = None,
    deadline: float | None = None,
) -> tuple[int, int]:
```

Inside the `for batch in loader:` loop, **after** the existing `save_every`/`eval_every` boundary checks (i.e. as the last action in the loop body, after the `on_eval` block), add the post-step deadline check:

```python
        if deadline is not None and time.monotonic() >= deadline:
            state_dir = paths.checkpoint_path(run_dir, step=global_step).parent / f"step_{global_step}"
            save_full_state(
                state_dir=state_dir,
                wrapper=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch,
                nan_streak=nan_streak,
                cfg=cfg,
            )
            raise _TimeLimitReached(global_step, epoch)
```

Update `run_epoch`'s docstring to note the optional `deadline` and that on expiry it flushes a full-state checkpoint (no panel) and raises `_TimeLimitReached`.

- [ ] **Step 2: Add a focused unit test for the flush + raise**

This is a thin direct-`run_epoch` test (the full `fit()`-level behavior is Task 7). Add to a new file `tests/train/test_time_limit_stop.py` (Task 7 extends it; create it here with this first test):

```python
"""Time-limit stop trigger + checkpoint flush (spec §11.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.loop import _TimeLimitReached, run_epoch
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def _loader(ds: _TinyDataset) -> list[dict[str, object]]:
    from custom_sam_peft.data.collate import collate_batch

    return [collate_batch([ds[i]]) for i in range(len(ds))]


def test_run_epoch_flushes_and_raises_on_past_deadline(tmp_path: Path) -> None:
    """A deadline already in the past flushes step_<N>/ and raises _TimeLimitReached."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    with pytest.raises(_TimeLimitReached) as exc:
        run_epoch(
            wrapper, _loader(ds), optimizer, scheduler, NoopTracker(), cfg, run_dir,
            epoch=0, global_step=0, nan_streak=0, class_names=ds.class_names,
            on_checkpoint=lambda *a: None, on_eval=lambda *a: None,
            deadline=0.0,  # monotonic 0 is always in the past -> fires after step 1
        )
    assert exc.value.step >= 1  # at least one micro-step completed
    ckpt = run_dir / "checkpoints" / f"step_{exc.value.step}"
    assert (ckpt / "adapter").exists()
    assert (ckpt / "training_state.pt").exists()
```

(`deadline=0.0` is a valid past monotonic instant: `time.monotonic()` is always `> 0` after process start, so the check fires on the first post-step evaluation — the "stop within ~one step" guarantee.)

- [ ] **Step 3: Run the focused test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/train/test_time_limit_stop.py::test_run_epoch_flushes_and_raises_on_past_deadline -q`
Expected: PASS — one step runs, the flush writes `step_1/adapter` + `training_state.pt`, and `_TimeLimitReached` is raised. (If it fails on the loader/collate import, confirm `collate_batch` is importable as used here; the seam test uses the same fixtures.)

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/train/loop.py tests/train/test_time_limit_stop.py
git commit -m "feat(train): deadline check + resumable flush + _TimeLimitReached in run_epoch (#198)"
```

## Task 6: Thread `deadline` through `_train_epoch`; catch in `fit()`; skip finalize; `_time_limited_artifacts`

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` — `import time`; compute `deadline`/`budget_seconds` in `fit()`; pass `deadline` through `_train_epoch`; restructure the try/except/finally; add `_time_limited_artifacts`.

- [ ] **Step 1: Add `import time` and thread `deadline` through `_train_epoch`**

At the top of `trainer.py`, add `import time` to the stdlib imports (alongside `json`, `logging`, `random`).

Add the Phase-1 import:

```python
from custom_sam_peft.config._duration import format_seconds, parse_duration_to_seconds
```

Import `TimeLimitStop` and `_TimeLimitReached`:

```python
from custom_sam_peft.eval._artifacts import EvalArtifacts, TimeLimitStop
from custom_sam_peft.train.loop import OomState, _TimeLimitReached, run_epoch
```

(The existing imports already pull `EvalArtifacts` from `_artifacts` and `OomState, run_epoch` from `loop`; extend those lines rather than duplicating.)

Add `deadline` to `_train_epoch`'s signature (keyword, default `None`) and pass it to `run_epoch`:

```python
    def _train_epoch(
        self,
        ...
        oom_state: OomState | None = None,
        deadline: float | None = None,
    ) -> tuple[int, int]:
        """Run one training epoch; returns (global_step, nan_streak)."""
        return run_epoch(
            ...
            oom_state=oom_state,
            deadline=deadline,
        )
```

- [ ] **Step 2: Compute the deadline in `fit()` before the epoch loop**

In `fit()`, immediately before the `merged_path`/`full_report` initialization and the `try:` (i.e. after `on_eval` is defined, before the existing `try:`), add:

```python
        deadline: float | None = None
        budget_seconds: int | None = None
        if cfg.train.time_limit is not None:
            budget_seconds = parse_duration_to_seconds(cfg.train.time_limit)
            deadline = time.monotonic() + budget_seconds
            _LOG.info("time limit: %s (%ds) — stops at the first micro-step past the deadline",
                      format_seconds(budget_seconds), budget_seconds)
```

- [ ] **Step 3: Restructure the try/except/finally to catch the stop and skip finalize**

Replace the existing `try: ... finally: self.tracker.close()` block. The catch wraps the epoch loop; on a stop, skip the entire finalize block but still close the tracker. The `_train_epoch` call gains `deadline=deadline`:

```python
        stop: _TimeLimitReached | None = None
        try:
            try:
                for epoch in range(start_epoch, cfg.train.epochs):
                    total_batches = max(len(train_loader), 1)
                    P.reset_inner(total=total_batches)
                    global_step, nan_streak = self._train_epoch(
                        epoch,
                        train_loader,
                        optimizer,
                        scheduler,
                        run_dir,
                        global_step,
                        nan_streak,
                        class_names,
                        on_checkpoint,
                        on_eval,
                        oom_state=oom_state,
                        deadline=deadline,
                    )
                    P.advance_outer()
            except _TimeLimitReached as e:
                stop = e
                global_step = e.step  # the flushed checkpoint's step

            if stop is None:
                adapter_path = run_dir / "adapter"
                save_adapter(self.model, adapter_path)
                if cfg.export.merge:
                    merged_path = run_dir / "merged"
                    save_merged(self.model, merged_path)

                if self.val_ds is not None:
                    full_eval_cfg = cfg.eval
                    if full_eval_cfg.batch_size == "auto":
                        from custom_sam_peft.presets import decide_eval_batch_size

                        bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
                        bs = self._cap_eval_batch_size(bs, oom_state.micro_batch_size)
                        full_eval_cfg = full_eval_cfg.model_copy(update={"batch_size": bs})
                    full_report = Evaluator(full_eval_cfg).evaluate(self.model, self.val_ds)
                if full_report is not None:
                    (run_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "overall": full_report.overall,
                                "per_class": full_report.per_class,
                                "n_images": full_report.n_images,
                                "n_predictions": full_report.n_predictions,
                                "global_step": global_step,
                                "epoch": cfg.train.epochs - 1,
                            },
                            indent=2,
                        )
                    )
                else:
                    (run_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "note": "no validation set provided",
                                "global_step": global_step,
                                "epoch": cfg.train.epochs - 1,
                            },
                            indent=2,
                        )
                    )
        finally:
            self.tracker.close()

        if stop is not None:
            return self._time_limited_artifacts(run_dir, stop, budget_seconds, oom_state)

        return EvalArtifacts(
            checkpoint_path=run_dir / "adapter",
            peft_method=self.cfg.peft.method,
            run_dir=run_dir,
            final_metrics=full_report,
            oom_events=tuple(oom_state.pending_oom_events),
        )
```

> Notes for the implementer:
>
> - The original had a single `try:`; this introduces a **nested** inner `try/except _TimeLimitReached` around just the epoch loop, with the outer `try/finally` retaining `tracker.close()`. The finalize block is moved inside the outer `try` and guarded by `if stop is None:`.
> - Keep the original **plain `else:`** of `if full_report is not None:` (do NOT narrow it to `elif self.val_ds is None:`). Verified against the source: the `else` fires whenever `full_report is None`, which is the no-val case **and** the case where a val set exists but the end-of-run eval raised (leaving `full_report = None`). Narrowing the condition would change behavior in that eval-raised case. The only structural change is that this whole `if/else` now sits inside the `if stop is None:` guard so it is skipped on a time-limited stop.
> - `budget_seconds` is non-`None` on every path where `stop` is non-`None` (a `_TimeLimitReached` only fires when `deadline` is non-`None`, which requires `time_limit` set, which sets `budget_seconds`). `_time_limited_artifacts` may therefore assert/`cast` it non-`None`.

- [ ] **Step 4: Add the `_time_limited_artifacts` helper**

Add to `Trainer` (a private method, near `_maybe_checkpoint`):

```python
    def _time_limited_artifacts(
        self,
        run_dir: Path,
        stop: Any,  # _TimeLimitReached
        budget_seconds: int | None,
        oom_state: OomState | None,
    ) -> EvalArtifacts:
        """Build the EvalArtifacts for a time-limited stop (spec §4.7).

        checkpoint_path points at the flushed step checkpoint's adapter
        (run_dir/adapter is intentionally NOT written on a stop).
        """
        assert budget_seconds is not None  # noqa: S101 — invariant: set whenever a stop fires
        checkpoint_dir = (
            paths.checkpoint_path(run_dir, step=stop.step).parent / f"step_{stop.step}"
        )
        best_dir: Path | None = None
        best_map: float | None = None
        best_candidate = run_dir / "best"
        best_json = best_candidate / "best.json"
        if best_candidate.is_dir() and best_json.is_file():
            try:
                data = json.loads(best_json.read_text())
                best_dir = best_candidate
                best_map = float(data["value"])
            except Exception:
                best_dir = None
                best_map = None
        time_limit_stop = TimeLimitStop(
            stop_step=stop.step,
            stop_epoch=stop.epoch,
            total_epochs=self.cfg.train.epochs,
            checkpoint_dir=checkpoint_dir,
            duration_label=format_seconds(budget_seconds),
            best_dir=best_dir,
            best_map=best_map,
        )
        return EvalArtifacts(
            checkpoint_path=checkpoint_dir / "adapter",
            peft_method=self.cfg.peft.method,
            run_dir=run_dir,
            final_metrics=None,
            oom_events=tuple(oom_state.pending_oom_events) if oom_state is not None else (),
            time_limit_stop=time_limit_stop,
        )
```

- [ ] **Step 5: Run the focused stop test (still green from Task 5)**

Run: `uv run pytest -o "addopts=" tests/train/test_time_limit_stop.py -q`
Expected: PASS (the direct-`run_epoch` test is unaffected by the `fit()` restructure). Full `fit()`-level coverage lands in Task 7.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py
git commit -m "feat(train): fit() computes deadline, catches stop, skips finalize, returns TimeLimitStop (#198)"
```

## Task 7: `fit()`-level stop, no-op, and resume tests

**Files:**

- Test (extend): `tests/train/test_time_limit_stop.py` (spec §11.3)
- Test (create): `tests/train/test_time_limit_noop.py` (spec §11.5)
- Test (create): `tests/train/test_time_limit_resume.py` (spec §11.4)

- [ ] **Step 1: Add the `fit()`-level stop test** (spec §11.3)

Append to `tests/train/test_time_limit_stop.py`:

```python
def test_fit_stops_flushes_and_skips_finalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() with a near-immediate budget stops, flushes step_<N>/, skips finalize."""
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    # save_every large so no periodic checkpoint fires; epochs high so the budget,
    # not the epoch count, ends the run.
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"time_limit": "2h30m", "epochs": 50})}
    )
    apply_lora(wrapper, cfg.peft)

    # Force the deadline into the past on the first monotonic read inside fit().
    monotonic_calls = {"n": 0}
    real = __import__("time").monotonic

    def fake_monotonic() -> float:
        monotonic_calls["n"] += 1
        # First call (deadline base) returns a huge value; subsequent checks read
        # the real clock, so deadline = huge + budget is far future... invert:
        # instead make the deadline base tiny and checks large.
        return real() if monotonic_calls["n"] > 1 else 0.0

    import custom_sam_peft.train.trainer as trainer_mod
    import custom_sam_peft.train.loop as loop_mod

    monkeypatch.setattr(trainer_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(loop_mod.time, "monotonic", real)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "stop-run")

    assert isinstance(result, EvalArtifacts)
    assert result.time_limit_stop is not None
    stop = result.time_limit_stop
    assert stop.stop_step >= 1
    assert stop.total_epochs == 50
    assert stop.duration_label == "2h30m"
    # Flushed step checkpoint exists; run_dir/adapter and metrics.json do NOT.
    ckpt = tmp_path / "stop-run" / "checkpoints" / f"step_{stop.stop_step}"
    assert (ckpt / "adapter").exists()
    assert result.checkpoint_path == ckpt / "adapter"
    assert not (tmp_path / "stop-run" / "adapter").exists()
    assert not (tmp_path / "stop-run" / "metrics.json").exists()
    assert result.final_metrics is None
```

> The deadline base is captured by the **first** `time.monotonic()` call inside `fit()` (in the deadline computation). Setting that first call to `0.0` makes `deadline = 0.0 + budget`. Real monotonic in `run_epoch` (which we leave un-patched via `loop_mod`) is always `> deadline` only if `budget` is small — but `"2h30m"` is large. **Simpler, robust approach:** instead of monkeypatching, pass a tiny effective budget by setting `time_limit` to a value that always trips. Because the check is `time.monotonic() >= deadline` and one step always runs first, the cleanest injection is to monkeypatch **`loop_mod.time.monotonic` to a function that returns a value past the deadline** after the first call. Implementers: prefer patching `loop.time.monotonic` to return `float("inf")` so the very first post-step check trips regardless of budget:
>
> ```python
> monkeypatch.setattr(loop_mod.time, "monotonic", lambda: float("inf"))
> ```
>
> Leave `trainer.time.monotonic` real (so `deadline` is a finite real instant), and set `time_limit` to any valid value (e.g. `"2h30m"` for the `duration_label` assertion). Replace the fragile `fake_monotonic` above with this one-liner.

Rewrite the test body to use the `float("inf")` patch (drop the `monotonic_calls`/`fake_monotonic` scaffold):

```python
    import custom_sam_peft.train.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "monotonic", lambda: float("inf"))
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "stop-run")
```

- [ ] **Step 2: Add the no-op test** (spec §11.5)

Create `tests/train/test_time_limit_noop.py`:

```python
"""Unset time_limit runs the full loop + finalize exactly as today (spec §11.5)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_fit_without_time_limit_finalizes_as_today(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)  # no time_limit set (default None)
    apply_lora(wrapper, cfg.peft)

    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "noop-run")

    assert isinstance(result, EvalArtifacts)
    assert result.time_limit_stop is None
    assert (tmp_path / "noop-run" / "adapter").exists()
    assert (tmp_path / "noop-run" / "metrics.json").exists()
    assert result.checkpoint_path == tmp_path / "noop-run" / "adapter"
```

- [ ] **Step 3: Add the resume test** (spec §11.4)

Create `tests/train/test_time_limit_resume.py`:

```python
"""Resume cleanly continues from a time-limited stop's flushed checkpoint (spec §11.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.checkpoint import find_latest_checkpoint
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_resume_after_time_limited_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_sam_peft.train.loop as loop_mod

    ds = _TinyDataset()
    cfg = _make_cfg(tmp_path)
    # output_dir must match cfg.run.output_dir so find_latest_checkpoint discovers it.
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"time_limit": "2h30m", "epochs": 50})}
    )

    # First run: stop on the budget.
    w1 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w1, cfg.peft)
    monkeypatch.setattr(loop_mod.time, "monotonic", lambda: float("inf"))
    run_dir_1 = Path(cfg.run.output_dir) / "tlr-run-1"
    (run_dir_1.parent).mkdir(parents=True, exist_ok=True)
    r1 = Trainer(w1, ds, ds, NoopTracker(), cfg).fit(run_dir=run_dir_1)
    assert r1.time_limit_stop is not None

    latest = find_latest_checkpoint(cfg)
    assert latest.name.startswith("step_")

    # Second run: no budget, resume from the flushed checkpoint, run to completion.
    monkeypatch.setattr(loop_mod.time, "monotonic", __import__("time").monotonic)
    cfg2 = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"time_limit": None, "epochs": 1})}
    )
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg2.peft)
    run_dir_2 = Path(cfg2.run.output_dir) / "tlr-run-2"
    r2 = Trainer(w2, ds, ds, NoopTracker(), cfg2).fit(run_dir=run_dir_2, resume_from=latest)
    assert r2.time_limit_stop is None
    assert (run_dir_2 / "adapter").exists()
    assert (run_dir_2 / "metrics.json").exists()
```

> `find_latest_checkpoint(cfg)` searches `cfg.run.output_dir` for `<run.name>-*/checkpoints/step_*`. The first run's `run_dir` must live under `cfg.run.output_dir` and start with `<run.name>-` for discovery. `_make_cfg` sets `run.name="seam-test"` and `output_dir=str(tmp_path)`; name the first run dir `tmp_path / "seam-test-1"` (prefix match) so `find_latest_checkpoint` finds it. **Adjust `run_dir_1` to `Path(cfg.run.output_dir) / f"{cfg.run.name}-1"`** (and `run_dir_2` similarly) so the prefix matches; the literal `"tlr-run-1"` above will NOT be discovered. Implementers: use the run-name prefix.

- [ ] **Step 4: Run all Phase-2 train tests**

Run: `uv run pytest -o "addopts=" tests/train/test_time_limit_stop.py tests/train/test_time_limit_noop.py tests/train/test_time_limit_resume.py -q`
Expected: PASS — stop flushes + skips finalize; no-op finalizes; resume continues to a full finalize.

- [ ] **Step 5: Commit**

```bash
git add tests/train/test_time_limit_stop.py tests/train/test_time_limit_noop.py tests/train/test_time_limit_resume.py
git commit -m "test(train): fit()-level time-limit stop, no-op, and resume coverage (#198)"
```

## Task 8: Phase 2 verification gate

**Files:** none (verification).

- [ ] **Step 1: Lint + type + format**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft`
Expected: clean. Watch for: the `loop -> checkpoint` import (acyclic, see Task 5); `_train_epoch`/`run_epoch` `deadline` param typed `float | None`; `_time_limited_artifacts`'s `assert budget_seconds is not None` narrowing for the `format_seconds(budget_seconds)` call.

- [ ] **Step 2: Full gated suite**

Run: `uv run pytest`
Expected: PASS including `--cov-fail-under=80`. The new branch is fully exercised by the three train test files + the seam assertion; the unlimited path is exercised by every existing trainer/seam test (all of which leave `time_limit` unset).

---

# PHASE 3 — CLI flags, exit UX, run short-circuit, docs

**Phase boundary — interface contract IN:** Phase 2's `EvalArtifacts.time_limit_stop: TimeLimitStop | None` and the populated-on-stop / exit-0 semantics; Phase 1's `parse_duration_to_seconds`.

**One coherent unit:** everything user-facing — the `--time-limit` option on both commands (override + early-validate, exit 1 on bad input), the shared message formatter, the `run` `_orchestrate` short-circuit, `train` skipping `--eval`/`--export` on a stop, and the docs. Terminal phase: opens the PR.

## Task 9: Shared message formatter (`cli/_time_limit.py`)

**Files:**

- Create: `src/custom_sam_peft/cli/_time_limit.py`
- Test: `tests/cli/test_time_limit_message.py` (spec §11.6)

- [ ] **Step 1: Write the failing tests** (spec §11.6)

Create `tests/cli/test_time_limit_message.py`:

```python
"""Exit-message formatter tests (spec §11.6)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.cli._time_limit import format_time_limit_message
from custom_sam_peft.eval._artifacts import TimeLimitStop


def _stop(*, best: bool, label: str = "2h30m") -> TimeLimitStop:
    return TimeLimitStop(
        stop_step=4120,
        stop_epoch=3,
        total_epochs=10,
        checkpoint_dir=Path("runs/x/checkpoints/step_4120"),
        duration_label=label,
        best_dir=Path("runs/x/best") if best else None,
        best_map=0.612 if best else None,
    )


def test_message_has_resume_command_train() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "custom-sam-peft train --config configs/run.yaml --resume __latest__" in msg


def test_message_with_best_includes_best_lines() -> None:
    msg = format_time_limit_message(
        _stop(best=True), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "Best so far" in msg
    assert "best" in msg  # the best/ path
    assert "0.612" in msg
    assert "Use best as-is" in msg


def test_message_without_best_omits_best_lines() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "Best so far" not in msg
    assert "Use best as-is" not in msg
    assert "--resume __latest__" in msg  # resume still present


def test_message_subcommand_run() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="run", config_path=Path("configs/run.yaml")
    )
    assert "custom-sam-peft run --config configs/run.yaml --resume __latest__" in msg


def test_message_duration_from_format_seconds() -> None:
    # A 9000s stop ("2h30m") and a literal "2h30m" stop render identical text.
    a = format_time_limit_message(
        _stop(best=False, label="2h30m"), subcommand="train", config_path=Path("c.yaml")
    )
    assert "(2h30m)" in a


def test_message_epoch_rendered_one_based() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="train", config_path=Path("c.yaml")
    )
    assert "(epoch 4/10)" in msg  # stop_epoch 3 (zero-based) -> "4/10"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/cli/test_time_limit_message.py -q`
Expected: FAIL — `custom_sam_peft.cli._time_limit` does not exist.

- [ ] **Step 3: Implement the formatter**

Create `src/custom_sam_peft/cli/_time_limit.py`:

```python
"""Shared exit-message formatter for a time-limited stop (spec §4.8).

Pure string builder — no Typer, no I/O — so it is unit-testable directly.
Both `train` and `run` call it and print via rprint, then exit 0.
"""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.eval._artifacts import TimeLimitStop


def _rel(path: Path) -> str:
    """Render relative to cwd when under it, else absolute (matches `done run_dir=` style)."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def format_time_limit_message(
    stop: TimeLimitStop, *, subcommand: str, config_path: Path
) -> str:
    """Build the resume message for a time-limited stop.

    subcommand is "train" or "run"; config_path is the actual --config the user
    passed. The duration label comes from stop.duration_label (format_seconds),
    so a 9000s and a "2h30m" stop render identically. Best lines appear only
    when stop.best_dir is set.
    """
    lines = [
        f"⏱  Time limit ({stop.duration_label}) reached at step {stop.stop_step} "
        f"(epoch {stop.stop_epoch + 1}/{stop.total_epochs}).",
        f"   Checkpoint saved: {_rel(stop.checkpoint_dir)}/",
    ]
    if stop.best_dir is not None and stop.best_map is not None:
        lines.append(f"   Best so far:      {_rel(stop.best_dir)}/ (mAP {stop.best_map:.3f})")
    lines.append("")
    resume = f"custom-sam-peft {subcommand} --config {config_path} --resume __latest__"
    lines.append(f"   • Resume:            {resume}")
    if stop.best_dir is not None:
        lines.append(f"   • Use best as-is:    {_rel(stop.best_dir)}/adapter/")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/cli/test_time_limit_message.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/_time_limit.py tests/cli/test_time_limit_message.py
git commit -m "feat(cli): shared time-limit exit-message formatter (#198)"
```

## Task 10: `--time-limit` on `train` + stop detection (skip `--eval`/`--export`)

**Files:**

- Modify: `src/custom_sam_peft/cli/train_cmd.py`
- Test: `tests/cli/test_time_limit_cli.py` (spec §11.7 — `train` portions)

- [ ] **Step 1: Write the failing `train` CLI tests** (spec §11.7)

Create `tests/cli/test_time_limit_cli.py` (the `run` portions are added in Task 11):

```python
"""CLI integration for --time-limit (spec §11.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.eval._artifacts import EvalArtifacts, TimeLimitStop

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    """A minimal valid config the loader accepts (no real data load triggered:
    run_train is patched in these tests, so dataset paths are never opened)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: tl\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def _stop_artifacts(run_dir: Path) -> EvalArtifacts:
    return EvalArtifacts(
        checkpoint_path=run_dir / "checkpoints" / "step_5" / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=None,
        time_limit_stop=TimeLimitStop(
            stop_step=5,
            stop_epoch=0,
            total_epochs=1,
            checkpoint_dir=run_dir / "checkpoints" / "step_5",
            duration_label="2h",
            best_dir=None,
            best_map=None,
        ),
    )


def test_train_bad_time_limit_exits_1_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import train_cmd

    called = {"run": False}
    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **k: called.__setitem__("run", True))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["train", "--config", str(cfg), "--time-limit", "10x"])
    assert result.exit_code == 1
    assert "invalid --time-limit" in result.output
    assert called["run"] is False


def test_train_time_limit_overrides_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import train_cmd

    seen: dict[str, Any] = {}

    def fake_run_train(cfg: Any, **k: Any) -> EvalArtifacts:
        seen["time_limit"] = cfg.train.time_limit
        return _stop_artifacts(tmp_path / "run")

    monkeypatch.setattr(train_cmd, "run_train", fake_run_train)
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["train", "--config", str(cfg), "--time-limit", "2h"])
    assert result.exit_code == 0
    assert seen["time_limit"] == "2h"


def test_train_stop_prints_message_and_skips_eval_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import train_cmd

    run_dir = tmp_path / "run"
    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **k: _stop_artifacts(run_dir))
    eval_called = {"n": 0}
    export_called = {"n": 0}
    monkeypatch.setattr(train_cmd, "run_eval", lambda *a, **k: eval_called.__setitem__("n", 1))
    monkeypatch.setattr(train_cmd, "run_export", lambda *a, **k: export_called.__setitem__("n", 1))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(
        app, ["train", "--config", str(cfg), "--time-limit", "2h", "--eval", "--export"]
    )
    assert result.exit_code == 0
    assert "Time limit (2h) reached" in result.output
    assert eval_called["n"] == 0
    assert export_called["n"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/cli/test_time_limit_cli.py -q`
Expected: FAIL — `train` has no `--time-limit` option (Typer errors on the unknown flag → non-zero, but not the asserted message/behavior).

- [ ] **Step 3: Edit `train_cmd.py`**

Add imports:

```python
from custom_sam_peft.cli._time_limit import format_time_limit_message
from custom_sam_peft.config._duration import parse_duration_to_seconds
```

Add the `--time-limit` option to `train(...)` (after `resume`, before `do_eval`):

```python
    time_limit: str | None = typer.Option(
        None,
        "--time-limit",
        help=(
            'Wall-clock budget for this run (e.g. "2h30m", "90m", "3600s", or bare '
            "seconds). Overrides train.time_limit. The budget is per-run; --resume "
            "restarts the clock."
        ),
        metavar="DURATION",
    ),
```

After `cfg = load_config(config, overrides=override)`, add the early-validate + override:

```python
    if time_limit is not None:
        try:
            parse_duration_to_seconds(time_limit)
        except ValueError as e:
            rprint(f"[red]error[/red] invalid --time-limit: {e}")
            raise typer.Exit(code=1) from e
        cfg = cfg.model_copy(
            update={"train": cfg.train.model_copy(update={"time_limit": time_limit})}
        )
```

After `result = run_train(cfg, resume_from=resume_path)` and **before** the `rprint("[green]done[/green] ...")` line, add the stop detection that prints and returns early (skipping the done message, `--eval`, and `--export`):

```python
    if result.time_limit_stop is not None:
        rprint(
            format_time_limit_message(
                result.time_limit_stop, subcommand="train", config_path=config
            )
        )
        return
```

(Returning from the Typer command exits 0; `--eval`/`--export` blocks are after this point and are skipped.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/cli/test_time_limit_cli.py -q`
Expected: PASS — bad input exits 1 before training; valid input overrides cfg; a stop prints the message and skips eval/export with exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/train_cmd.py tests/cli/test_time_limit_cli.py
git commit -m "feat(cli): --time-limit on train; print resume message + skip eval/export on stop (#198)"
```

## Task 11: `--time-limit` on `run` + `_orchestrate` short-circuit

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py`
- Test (extend): `tests/cli/test_time_limit_cli.py` (spec §11.7 — `run` portions)

- [ ] **Step 1: Add the failing `run` CLI tests** (spec §11.7)

Append to `tests/cli/test_time_limit_cli.py`:

```python
def test_run_bad_time_limit_exits_1_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import run_cmd

    called = {"run": False}
    monkeypatch.setattr(run_cmd, "run_training", lambda *a, **k: called.__setitem__("run", True))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["run", "--config", str(cfg), "--time-limit", "10x"])
    assert result.exit_code == 1
    assert "invalid --time-limit" in result.output
    assert called["run"] is False


def test_run_stop_short_circuits_before_eval_export_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import run_cmd

    run_dir = tmp_path / "run"
    (run_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_cmd, "run_training", lambda *a, **k: _stop_artifacts(run_dir))
    phase_calls = {"val": 0, "load": 0, "eval": 0, "merged": 0, "bundle": 0}
    monkeypatch.setattr(
        "custom_sam_peft.data.val_source.load_val_source",
        lambda *a, **k: phase_calls.__setitem__("val", 1),
    )
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: phase_calls.__setitem__("load", 1))
    monkeypatch.setattr(run_cmd, "run_eval", lambda *a, **k: phase_calls.__setitem__("eval", 1))
    monkeypatch.setattr(run_cmd, "save_merged", lambda *a, **k: phase_calls.__setitem__("merged", 1))
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: phase_calls.__setitem__("bundle", 1))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["run", "--config", str(cfg), "--time-limit", "2h"])
    assert result.exit_code == 0
    assert "Time limit (2h) reached" in result.output
    # No phase after train ran:
    assert phase_calls["load"] == 0
    assert phase_calls["eval"] == 0
    assert phase_calls["merged"] == 0
    assert phase_calls["bundle"] == 0
```

> The short-circuit must land **before** `load_val_source` (so even `vs`-resolution is skipped). The test patches `load_val_source` at its definition module (`custom_sam_peft.data.val_source`) because `_orchestrate` imports it locally inside the function (`from custom_sam_peft.data.val_source import load_val_source`). If the short-circuit is correctly placed, `load_val_source` is never reached and `phase_calls["val"]` stays 0 — but the binding assertions are the post-train phases (`load`/`eval`/`merged`/`bundle`), which are patched on `run_cmd`'s namespace where they are imported at module top.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/cli/test_time_limit_cli.py -k run -q`
Expected: FAIL — `run` has no `--time-limit` option and `_orchestrate` does not short-circuit.

- [ ] **Step 3: Edit `run_cmd.py`**

Add imports:

```python
from custom_sam_peft.cli._time_limit import format_time_limit_message
from custom_sam_peft.config._duration import parse_duration_to_seconds
```

Add the `--time-limit` option to `run(...)` (after `resume`, before `verbose`):

```python
    time_limit: str | None = typer.Option(
        None,
        "--time-limit",
        help=(
            'Wall-clock budget for this run (e.g. "2h30m", "90m", "3600s", or bare '
            "seconds). Overrides train.time_limit. The budget is per-run; --resume "
            "restarts the clock."
        ),
        metavar="DURATION",
    ),
```

After `cfg = load_config(config)`, add the early-validate + override:

```python
    if time_limit is not None:
        try:
            parse_duration_to_seconds(time_limit)
        except ValueError as e:
            rprint(f"[red]error[/red] invalid --time-limit: {e}")
            raise typer.Exit(code=1) from e
        cfg = cfg.model_copy(
            update={"train": cfg.train.model_copy(update={"time_limit": time_limit})}
        )
```

`_orchestrate` needs the `config` path to format the resume command. Add a `config_path` keyword parameter to `_orchestrate`:

```python
def _orchestrate(
    cfg: TrainConfig, resume: Path | None, mode: ProgressMode, *, visualize: bool, config_path: Path
) -> int:
```

In `_orchestrate`, immediately after the train phase (`train_result = run_training(...)`), before `run_dir = train_result.run_dir`, add the short-circuit:

```python
    if train_result.time_limit_stop is not None:
        rprint(
            format_time_limit_message(
                train_result.time_limit_stop, subcommand="run", config_path=config_path
            )
        )
        return 0
```

Update the `run(...)` call site to pass `config_path=config`:

```python
    _orchestrate(cfg, resume_path, mode, visualize=visualize, config_path=config)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/cli/test_time_limit_cli.py -q`
Expected: PASS — all `train` and `run` CLI tests.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/cli/test_time_limit_cli.py
git commit -m "feat(cli): --time-limit on run; _orchestrate short-circuits before eval/export/bundle (#198)"
```

## Task 12: Docs — `config-schema.md` + `defaults-provenance.md`

**Files:**

- Modify: `docs/config-schema.md`
- Modify: `docs/defaults-provenance.md`

- [ ] **Step 1: Add the `train.time_limit` advanced-field row to `config-schema.md`**

In `docs/config-schema.md`, in the `### Advanced fields` table (under "Training hyperparameters"), add a row (place it after the `train.num_workers` row, or grouped with the opt-in scheduling fields — match surrounding row order):

```markdown
| `train.time_limit` | `str \| int \| None` | `None` (unlimited) | advanced | Wall-clock budget for one invocation. Accepts a human duration (`"2h30m"`, `"90m"`, `"3600s"`) or bare seconds (`3600`); `None` (default) = unlimited. On expiry the current micro-step finishes, a resumable checkpoint is flushed under `checkpoints/step_<N>/`, finalization is skipped, and the process exits 0. The budget is per-run: `--resume` restarts the clock. Overridable via `--time-limit` on `train`/`run`. | Opt-in; unset = unlimited, so there is no default value to justify. |
```

- [ ] **Step 2: Document the `--time-limit` CLI flag in `config-schema.md`**

Locate the section of `docs/config-schema.md` that documents CLI flags (the file that commit `837f80f` updated to document `--no-visualize`; search for `--no-visualize` / `--resume`). Add a note describing `--time-limit DURATION` on both `train` and `run`: it overrides `train.time_limit`, accepts the same duration grammar, the budget is per-run (`--resume` restarts the clock), and a bad value exits 1 before training starts. If no dedicated CLI-flags section exists, the advanced-field row above plus its "Overridable via `--time-limit`" clause suffices — note that in the PR description.

- [ ] **Step 3: Add the opt-in note to `defaults-provenance.md`**

`defaults-provenance.md` requires no new default-value row (the default is `None`, not a chosen number). In the `## config/schema.py` section, optionally add an index-only row recording the intentional opt-in:

```markdown
| `config/schema.py:TrainHyperparams.time_limit` | `None` | `index-only` | — | — | Opt-in wall-clock budget; `None` = unlimited. No default value to cite (the feature is off unless set). |
```

- [ ] **Step 4: Markdown-lint the changed docs**

Run the markdown-lint gate command (from the "Verification gates" section) on `docs/config-schema.md docs/defaults-provenance.md`. Expected: clean exit (0). Fix findings.

- [ ] **Step 5: Commit**

```bash
git add docs/config-schema.md docs/defaults-provenance.md
git commit -m "docs: document train.time_limit + --time-limit flag (#198)"
```

## Task 13: Phase 3 verification gate + PR

**Files:** none (verification + `gh`).

- [ ] **Step 1: Lint + type + format**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft`
Expected: clean.

- [ ] **Step 2: Full gated suite**

Run: `uv run pytest`
Expected: PASS including `--cov-fail-under=80`.

- [ ] **Step 3: Markdown-lint spec + plan + docs (CI lints all `**/*.md`)**

Run the markdown-lint gate command on `docs/config-schema.md docs/defaults-provenance.md docs/superpowers/plans/2026-05-30-train-time-limit-plan.md docs/superpowers/specs/2026-05-30-train-time-limit-design.md`.
Expected: clean exit (0). Fix findings.

- [ ] **Step 4: Push + open the PR closing #198, linking spec + plan**

```bash
git push -u origin HEAD
gh label list   # pick an existing label (enhancement/feature) or create one inline
gh pr create --assignee @me --label <label> \
  --title "Wall-clock time limit with resumable graceful stop (#198)" \
  --body "Closes #198.

Adds an opt-in \`train.time_limit\` config knob and a \`--time-limit\` flag on \`train\`/\`run\`. On expiry the current micro-step finishes, a resumable full-state checkpoint is flushed under checkpoints/step_<N>/, all finalization (full eval / metrics.json / adapter rewrite / merged export / bundle) is skipped, a resume message prints, and the process exits 0. Unset = unlimited (unchanged). The budget is per-run; --resume restarts the clock.

Spec: docs/superpowers/specs/2026-05-30-train-time-limit-design.md
Plan: docs/superpowers/plans/2026-05-30-train-time-limit-plan.md"
```

---

## Self-review (planner — completed)

**Spec acceptance-criteria coverage (§8):**

1. Config + CLI (`time_limit` field; `--time-limit` on both, overrides) → T2 (field), T10/T11 (CLI override).
2. Duration formats accept/reject → T1 (`test_duration.py`), T2 (`test_schema_time_limit.py`).
3. Unset = unlimited → T7 (`test_time_limit_noop.py`); every existing trainer/seam test stays green (T8 full gate).
4. Responsive stop, current step finishes → T5 (post-step check), T5/T7 (`stop_step >= 1`).
5. Resumable flush regardless of `save_every` → T5 (direct `save_full_state`; `save_every=1000` in fixture), T7.
6. Resume continues, fresh per-run budget → T7 (`test_time_limit_resume.py`).
7. Best usable when present / absent → T6 (`_time_limited_artifacts` best read), T9 (message best lines present/omitted).
8. Graceful exit 0 + resume message naming subcommand + config → T9 (formatter), T10/T11 (exit 0 + skip).
9. `run` short-circuits (no bundle/eval/export) → T11.
10. No finalize on stop (no eval/metrics.json/adapter rewrite/merged) → T6 (`if stop is None` guard), T7 (asserts none written).
11. Seam intact with the optional field → T4 (`time_limit_stop is None` assertion; existing seam tests untouched-green).
12. Tests + ruff + mypy + pytest clean → per-phase gates T3/T8/T13.

**Spec §11 testing-matrix coverage:** §11.1→T1; §11.2→T2; §11.3→T5+T7; §11.4→T7; §11.5→T7; §11.6→T9; §11.7→T10+T11; §11.8→T4. Every named test file is created/extended by a task.

**Spec §5 module table coverage:** `config/_duration.py`→T1; `schema.py time_limit`→T2; `--time-limit` on `train`/`run`→T10/T11; trainer `deadline`/catch/skip/`_time_limited_artifacts`→T6; `loop.py` `deadline`/check/flush/raise→T5; `_TimeLimitReached`→T5; `TimeLimitStop` + `time_limit_stop` field→T4; `cli/_time_limit.py`→T9; `run` short-circuit→T11; doc rows (`config-schema.md`, `defaults-provenance.md`, CLI-flags)→T12.

**Phase-boundary contracts:** all three boundaries have written OUT/IN contracts (top table + the two detail blocks). Phase 1 exposes the two parser signatures + verbatim field; Phase 2 exposes `EvalArtifacts.time_limit_stop`/`TimeLimitStop` + exit-0 semantics consumed solely via `result.time_limit_stop is not None`; Phase 3 is terminal.

**Hazards surfaced as explicit notes/sub-tasks:** (#1) `paths.checkpoint_path` returns a padded `.pt` filename, not a dir — flush uses `.parent / f"step_{N}"`; (#2) `on_checkpoint` is 3-arg and the spec's `result.p_t`/`box_hint_p` is obsolete (post-#88) — flush uses direct `save_full_state`; (#3) the single `try/finally` wraps loop+finalize, restructure skips finalize but keeps `tracker.close()`, finalize verified tracker-independent; (#4) `checkpoint_path` points at the step adapter on a stop; (#5) idempotent re-flush; (#6) `best.json` shape; (#7) `time.monotonic` import. All appear in "Code-aware notes & verified hazards" and are referenced from the relevant tasks.

**Placeholder scan:** no TBD/TODO; every code step shows the actual edit; every verify step has a concrete command + expected result. Two tests (T7 stop, T7 resume) carry explicit implementer guidance that **replaces** a fragile first draft with a robust `float("inf")` monkeypatch / run-name-prefix run dir — the final intended form is stated unambiguously.

**Type consistency:** `parse_duration_to_seconds(value: str | int) -> int` / `format_seconds(seconds: int) -> str` consistent across T1/T2/T6/T9/T10/T11; `deadline: float | None` consistent across `_train_epoch` (T6) and `run_epoch` (T5); `TimeLimitStop` 7-field shape identical in T4 (def), T6 (construct), T9 (tests); `EvalArtifacts.time_limit_stop` default `None` consistent T4/T6/T7; `_time_limited_artifacts(run_dir, stop, budget_seconds, oom_state)` signature matches its T6 call site; `format_time_limit_message(stop, *, subcommand, config_path)` signature matches T9 def + T10/T11 calls.

**Out-of-scope respected (spec §12):** no full eval / export / merged export / bundle / best-finalization on a stop (→ #197); no cumulative cross-resume time; no signal handling; no per-epoch/per-eval budgets. No task touches those.
