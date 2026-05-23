# spec/cli-progress-bars — Bottom-pinned progress UI for train / eval / predict (issue #76)

**Status:** Draft (2026-05-22)
**Tracking:** [#76](https://github.com/NguyenJus/custom-sam-peft/issues/76) — *feat(cli): progress bars for train / eval / predict — bottom-pinned, on by default, scramble-free*
**Scope:** Add a single new module `src/custom_sam_peft/cli/_progress.py` that wraps any long-running CLI command with a `rich.Progress`-backed live display; wires into `train`, `eval`, `run`, and `export`; defines the `predict` contract for when #74 lands. Defines mode resolution (`--progress` flag + `CSP_NO_PROGRESS` env var), third-party output suppression, a `T20` lint rule banning bare `print()`, and a 10-test CPU-only test suite. No source code outside `cli/`, `train/loop.py`, `train/trainer.py`, and `eval/evaluator.py` is touched.

**Builds on / relates to:**
[`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) (Typer app structure, `configure_logging`, shared option patterns);
[`2026-05-21-yaml-config-defaults-audit-design.md`](2026-05-21-yaml-config-defaults-audit-design.md) (the normalization-fallback WARN line in §3 is exactly the class of message this spec prevents from scrambling a pinned bar);
[#74](https://github.com/NguyenJus/custom-sam-peft/issues/74) (introduces `csp predict` — #76 defines the predict progress contract; the `predict_cmd.py` wiring lands when #74 merges);
[#70](https://github.com/NguyenJus/custom-sam-peft/issues/70) (v1.0 criteria — "looks stuck" UX gap identified there is closed by this spec).

---

## 1. Context & Motivation

No progress UI exists anywhere in the codebase today — zero `tqdm`, zero `rich.progress`, zero `Progress` references (verified). `cli/_logging.py` is a 15-line `logging.basicConfig` wrapper. Long-running commands (`train` over multiple epochs, `eval` over a full validation set) produce only periodic log lines every `log_every` steps (default 50 batches), which leaves users wondering whether the process is stuck (#70, #76).

This is the single biggest "is it stuck?" UX gap blocking a credible v1.0 (`rich==15.0.0` is already a hard dep — the capability cost is zero).

**Five failure modes that any solution must address** (#76):

1. **tqdm + logging interleave** — `\r` vs newline races produce duplicate / overwritten bars under WSL, Colab, and `script(1)`.
2. **Multiple bars** — outer epoch + inner step cursor fights without region pinning.
3. **Third-party output ignores your bar** — `transformers` bars, HF `datasets` progress, `bitsandbytes` stderr, `warnings.warn()` all bypass a naively owned console and scramble the display.
4. **Non-TTY** — CI logs, redirected files, and `nohup` produce CR-spam or empty output without explicit mode detection.
5. **Notebook / Colab** — terminal ANSI is fragile in Colab cells; requires explicit detection and fallback.

---

## 2. Design Overview

One process-global display handle (`P`) backed by `rich.progress.Progress` when a session is active, no-op otherwise. The CLI entry opens a `progress_session(...)` context manager around its runner call; the runner calls `P.advance_inner()` / `P.update_postfix()` / etc. without importing any CLI machinery. When no session is active — library callers, unit tests, programmatic use — every `P.*` is a no-op.

```
┌─────────────────────────────────────────────────────┐
│ 12:03:01 INFO  train: epoch 3 batch 1240            │  ← log stream (scrolls up)
│ 12:03:02 INFO  train: loss=0.812 lr=8e-5            │
│ 12:03:03 WARN  AutoImageProcessor cache miss …      │
│ …                                                   │
├─────────────────────────────────────────────────────┤
│ epoch  3/10  ████████░░░░░░░  1240/4530  it=2.3/s   │  ← pinned progress (always at bottom)
│ batch        ██████████████░  loss=0.812             │
└─────────────────────────────────────────────────────┘
```

`rich.Progress` owns both regions through a single console: logs are routed through `progress.console.log(...)` (via a `RichHandler(console=progress.console)` attached to the root logger), and `Progress` internally pins its bar region below the scrolling log stream. No custom `rich.live.Live` + `Group` wrapper is needed — `Progress` is already backed by a `Live`, and `progress.console.log(...)` interleaves cleanly above the pinned bars without a separate `Group`. This is the deliberate simplification of the architecture diagram in #76's issue body: same UX, less surface area.

---

## 3. Public API

### 3.1 CLI side — one call per command body

```python
from custom_sam_peft.cli._progress import progress_session, ProgressKind, ProgressMode

with progress_session(
    kind=ProgressKind.TRAIN,
    total_epochs=cfg.train.epochs,
    total_batches_per_epoch=len(train_loader),
    mode=resolved_mode,
):
    run_training(cfg, ...)
# success rprint(...)  ← always OUTSIDE the with block so Live tears down first
```

`progress_session` is a context manager (class or `@contextmanager`) that:
- Builds the `rich.Progress` instance and starts it (unless `mode` is `OFF` or `PLAIN`).
- Attaches a `RichHandler(console=progress.console)` to the root logger on entry; restores the prior handlers on exit.
- Calls `_silence_third_party_progress()` defensively on entry.
- Installs a SIGINT handler on entry that calls `progress.stop()` *before* `KeyboardInterrupt` unwinds Python frames; restores the prior SIGINT handler on exit.
- Raises `RuntimeError` if a second `progress_session` is opened before the first exits (nested sessions are unsupported).

`ProgressKind` enum: `TRAIN | EVAL | PREDICT | EXPORT_MERGE`.

`ProgressMode` enum: `ON | OFF | PLAIN` (note: `AUTO` is resolved *before* a session is opened — see §4).

`resolved_mode` comes from `resolve_mode(cli_flag, os.environ, sys.stdout.isatty(), Console().is_jupyter)` called in the command body before entering the context.

### 3.2 Runtime side — process-global no-op handle

```python
from custom_sam_peft.cli._progress import progress as P

P.advance_outer()                        # epoch tick (TRAIN)
P.advance_inner(n=1)                     # batch / image tick
P.update_postfix(loss=..., lr=..., it_s=...)
P.console.log(...)                       # routes through Live when active
with P.push_subtask("lite-eval", total=N):
    ...                                  # transient third task, auto-removed on exit
```

When no session is active: every method is a no-op, `P.console` returns a vanilla `Console()`. Callers in `train/loop.py`, `train/trainer.py`, `eval/evaluator.py`, and (when #74 lands) `predict`-related code import `P` directly — they have no dependency on the CLI layer. Library callers and unit tests are unaffected by default.

`P` is the module-level singleton exported from `_progress.py`. It holds either a live `_ProgressHandle` (when a session is active) or a `_NoOpHandle` (otherwise). The swap happens inside `progress_session.__enter__` / `__exit__`.

---

## 4. Mode Resolution

`resolve_mode` is a pure function with no side effects.

```python
def resolve_mode(
    cli_flag: str | None,
    env: Mapping[str, str],
    stdout_isatty: bool,
    is_jupyter: bool,
) -> ProgressMode:
    # 1. Explicit flag wins (overrides everything, including CSP_NO_PROGRESS).
    if cli_flag is not None and cli_flag != "auto":
        return ProgressMode(cli_flag)        # "on" | "off" | "plain"
    # 2. Env var applies only when flag is absent or "auto".
    if env.get("CSP_NO_PROGRESS") == "1":
        return ProgressMode.OFF
    # 3. Auto fallback chain.
    if is_jupyter:
        return ProgressMode.PLAIN
    if not stdout_isatty:
        return ProgressMode.PLAIN
    return ProgressMode.ON
```

**Precedence:** explicit `--progress` flag > `CSP_NO_PROGRESS=1` env var. An explicit `--progress on` overrides `CSP_NO_PROGRESS=1`; this is intentional (debugging a script that sets the env var).

**CLI flag:** `--progress {auto|on|off|plain}` (Typer `Enum` option, default `"auto"`). Add as a shared option to `train`, `eval`, `run`, `export`, and — when #74 lands — `predict`. `init` and `doctor` do not need it (both complete in milliseconds; `doctor` is instant by design, `init` is a single file write).

### Behaviors per mode

| mode    | `Progress` started? | `RichHandler` attached? | HF/transformers silenced? | progress output |
|---------|---------------------|-------------------------|---------------------------|-----------------|
| `on`    | yes                 | yes                     | yes                       | Rich live bar, pinned |
| `plain` | no                  | no (basicConfig)        | yes                       | one `progress:` line per `log_every` window, appended; no ANSI redraw |
| `off`   | no                  | no (basicConfig)        | yes                       | log lines only; no progress lines |
| `auto`  | resolves to `on` or `plain` before session opens | — | — | — |

`auto` never reaches the session constructor. It is resolved to `on` or `plain` by `resolve_mode` in the command body before `progress_session(mode=resolved_mode)` is called.

**`plain` line format** (fixed; snapshot-tested — see §9.F):

```
progress: train epoch=3/10 step=1240/45300 loss=0.812 it/s=2.3 eta=0:42:10
```

Emit at most once per `log_every` window (the same cadence as `tracker.log_scalars`). No CR redraw; append-only; grep-friendly.

---

## 5. Data Flow per Command

### 5.1 `train`

**Call sites:**

- `train/loop.py:run_epoch` — `P.advance_inner()` on every batch; `P.update_postfix(loss=..., lr=..., it_s=...)` every `cfg.train.log_every` batches (same window that drives `tracker.log_scalars`).
- `train/trainer.py:Trainer.fit` — `P.advance_outer()` then `P.reset_inner()` at the top of each epoch loop iteration.

Total batches per epoch = `max(len(train_loader), 1)` (same expression already used in `Trainer.fit`).

The `train_cmd.py` body resolves mode, opens `progress_session(kind=TRAIN, total_epochs=..., total_batches_per_epoch=..., mode=...)`, calls the existing trainer, then prints the success summary *outside* the `with` block.

### 5.2 `eval`

**Call site:**

- `eval/evaluator.py:evaluate` — `P.advance_inner()` after the inner per-class forward loop completes for each image (advancement is **per image, not per (image, class)**). `P.update_postfix(running_mAP=..., it_s=...)` every `max(1, N // 50)` images where `N = len(examples)`.

`eval_cmd.py` opens `progress_session(kind=EVAL, total_batches_per_epoch=len(examples), mode=...)`. No outer epoch bar for eval — `total_epochs` is omitted (defaults to `1` or `None`; the session suppresses the outer task when `total_epochs` is not provided).

### 5.3 `run`

`run_cmd.py` orchestrates train → eval → optional export-merge as three sequential phases, each in its own session:

1. `progress_session(kind=TRAIN, ...)` wraps the train phase. Exits cleanly.
2. `progress_session(kind=EVAL, ...)` wraps the eval phase. Exits cleanly.
3. When export-merge is triggered: `progress_session(kind=EXPORT_MERGE, total_batches_per_epoch=..., mode=...)`. Exits cleanly.

Sequential sessions are fully supported (only simultaneous/nested sessions are forbidden). Each teardown restores handlers before the next session's entry, so there is no handler accumulation.

### 5.4 `predict` (contract — implemented when #74 lands)

`predict_cmd.py` opens `progress_session(kind=PREDICT, total_batches_per_epoch=len(image_paths), mode=...)`. No outer bar. Inner bar over `len(image_paths)`. Postfix: `done=N/M it/s=…`.

The predict runner calls `P.advance_inner()` per processed image and `P.update_postfix(done=f"{n}/{m}", it_s=...)` at a suitable cadence. This spec defines the contract; the implementation PR for #76 does **not** add `predict_cmd.py` — that is #74's responsibility.

### 5.5 Mid-train lite eval (`push_subtask`)

When a lite mid-epoch eval is invoked from within `run_epoch` (e.g. called as a hook or inline branch), it runs *inside* the active TRAIN session. Use `P.push_subtask("lite-eval", total=N)` as a context manager: it adds a transient third progress task that auto-removes on exit. The outer epoch bar and inner batch bar remain visible above it. The subtask's total is the number of validation examples for the lite eval.

---

## 6. Error Handling

### 6.1 SIGINT (Ctrl-C)

`progress_session.__enter__` installs a SIGINT handler that:
1. Calls `progress.stop()` (tears down the `Live` region, restores the terminal cursor).
2. Calls the prior SIGINT handler (or raises `KeyboardInterrupt` if none).

`progress_session.__exit__` restores the prior handler. **Critical ordering**: `progress.stop()` must run before `KeyboardInterrupt` unwinds Python frames, otherwise the final terminal state is corrupted and the cursor-restore escape is never written. The subprocess-based SIGINT test (§9.H) verifies no dangling ANSI cursor escapes at end-of-stream.

### 6.2 Runner exception

`progress_session.__exit__` handles any exception type symmetrically: calls `progress.stop()`, restores handlers, re-raises. Existing `rprint(f"[red]error[/red] {e}")` lines in `train_cmd.py` etc. run *after* `progress.stop()` has torn down the `Live` region — they print as ordinary lines with no ANSI contamination.

### 6.3 Shutdown order on success

Command bodies put the final success `rprint(...)` **outside** (after) the `with progress_session(...)` block. `Live` tears down before the summary prints. No exceptions needed to the existing `rprint` usage patterns in `train_cmd.py` / `eval_cmd.py` — just move them outside the new `with` block.

---

## 7. Third-Party Output Routing

### 7.1 `_silence_third_party_progress()` — always active

A new top-level function in `main.py`, called once before any subcommand dispatch, regardless of `--progress` mode:

```python
def _silence_third_party_progress() -> None:
    """Suppress HF / datasets progress output. Idempotent — safe to call twice."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import datasets as _ds
        _ds.disable_progress_bar()
    except ImportError:
        pass
```

`progress_session.__enter__` also calls this function defensively, covering programmatic use where `main.py`'s top-level call was skipped.

### 7.2 `RichHandler` attachment (`on` mode only)

When `mode == ON`, `progress_session.__enter__`:
1. Captures the root logger's current handlers list.
2. Clears the root logger's handlers.
3. Adds `RichHandler(console=progress.console, rich_tracebacks=True, show_path=False)`.
4. `logging.captureWarnings(True)` so `warnings.warn(...)` flows through the logger and through the `RichHandler` to the `Progress` console.

`progress_session.__exit__` restores the root logger's handlers to the captured list exactly, and calls `logging.captureWarnings(False)`.

### 7.3 `cli/_logging.py` change

`configure_logging` gains an optional `console: rich.console.Console | None = None` kwarg:

```python
def configure_logging(verbose: bool, console: Console | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if console is not None:
        handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
```

Default behavior (no `console` arg) is unchanged. The `progress_session` uses its own handler attachment path described in §7.2, not `configure_logging` — the kwarg is provided so library callers or custom CLI wrappers can inject a console without going through a full session.

---

## 8. Lint Rule: T20 (flake8-print)

Add `"T20"` to `[tool.ruff.lint] select` in `pyproject.toml`:

```toml
select = ["E", "F", "I", "B", "UP", "SIM", "RUF", "S", "T20"]
```

`T20` bans bare `print()` calls (`T201`) and `pprint()` calls (`T203`) across the entire `src/custom_sam_peft/` tree. No new per-file-ignore entry is added for `T20` — the rule applies to all source files.

**Single legitimate exception:** `src/custom_sam_peft/cli/doctor_cmd.py` line ~83 uses `print(json.dumps(...))` for the `--json` output path. This is intentional (machine-readable stdout, not a debug print). It receives a `# noqa: T201` inline comment:

```python
print(json.dumps(dataclasses.asdict(report), default=str, indent=2))  # noqa: T201
```

No other `print()` calls exist in `src/custom_sam_peft/` today (verified). The lint rule guards against future regressions.

---

## 9. Test Plan

All tests are CPU-only; no `@pytest.mark.gpu` markers. No ANSI byte-exactness tests, no Colab rendering, no DDP coverage. The project's 80% `--cov-fail-under` gate is met comfortably by `_progress.py` (~250 lines) plus 10 tests.

### `tests/unit/test_progress_resolve.py`

**A. `test_resolve_mode_matrix`** — parametrized table over `(cli_flag, env, isatty, is_jupyter) → expected_mode`. Covers: flag `"on"` / `"off"` / `"plain"` override all; `CSP_NO_PROGRESS=1` yields `OFF` when flag is unset/`auto`; Jupyter yields `PLAIN`; non-tty yields `PLAIN`; tty yields `ON`.

### `tests/unit/test_progress_module.py`

**B. `test_no_op_default`** — `P.advance_outer()`, `P.advance_inner()`, `P.update_postfix(loss=0.5)`, and `P.console.log("x")` all outside any session: no exception, no terminal writes (assert `Console().file` is not written via `StringIO` monkeypatch).

**C. `test_session_lifecycle`** — entering a `progress_session` attaches a `RichHandler` to the root logger; exiting restores the prior handlers list exactly (same `id`s, same count). Opening a second `progress_session` before the first exits raises `RuntimeError("nested session")`.

**D. `test_log_through_live`** — `logger.info("hello")` inside a `progress_session(mode=ON)` with a `StringIO`-backed console produces exactly one line in the captured output (no duplication from double-handler attachment).

**E. `test_push_subtask_lifecycle`** — `P.push_subtask("lite-eval", total=10)` as a context manager: subtask is present in `progress.tasks` inside the block; removed (or marked hidden) on exit; outer and inner tasks unaffected.

**F. `test_plain_line_snapshot`** — frozen-string snapshot of the `plain` mode progress line:
```
progress: train epoch=3/10 step=1240/45300 loss=0.812 it/s=2.3 eta=0:42:10
```
Verifies the exact format does not silently drift (CI-stability contract).

**G. `test_silence_third_party_progress`** — after calling `_silence_third_party_progress()`: `os.environ["TRANSFORMERS_VERBOSITY"] == "warning"`; `os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"`; a mocked `datasets.disable_progress_bar` was called exactly once. A second call is idempotent (no exception, mock called again only once more).

**H. `test_sigint_handler`** — two assertions in one test. (1) *Interrupted path*: launches a minimal subprocess that enters a `progress_session(mode=ON)` and sleeps, then receives `SIGINT`; asserts process exits non-zero and captured stdout/stderr has no dangling ANSI cursor-hide escapes (`\x1b[?25l` not at end-of-stream without a matching show `\x1b[?25h`). (2) *Clean-exit path*: enters a `progress_session` and exits normally; asserts `signal.getsignal(signal.SIGINT)` equals the handler that was registered before the session opened (prior handler restored).

### `tests/unit/test_progress_integration.py`

**I. `test_fake_trainer_smoke`** — dummy trainer loop using a `StringIO`-backed console and `progress_session(kind=TRAIN, total_epochs=2, total_batches_per_epoch=3, mode=ON)`. The loop calls `P.advance_outer()` twice (once per epoch) and `P.advance_inner()` six times total (3 batches × 2 epochs), plus `P.update_postfix(loss=..., it_s=...)` at the end of each epoch. Captured output shows correct outer/inner tick counts and the final postfix payload. Verifies the real data flow contract without a GPU.

**J. `test_ruff_t201_lint_rule`** — uses `subprocess.run(["ruff", "check", "--select", "T201", ...])` to verify: a temp file containing `print("x")` fails lint; a temp file containing `print(json.dumps(x))  # noqa: T201` passes lint. This test guards the lint-config change against accidental reversion.

---

## 10. Rollout & Sequencing

### PR for #76

The `feat-cli-progress-bars` branch implements everything in this spec except `predict_cmd.py` wiring. The PR:
- Adds `src/custom_sam_peft/cli/_progress.py`.
- Wires `train_cmd.py`, `eval_cmd.py`, `run_cmd.py`, `export_cmd.py`.
- Touches `train/loop.py`, `train/trainer.py`, `eval/evaluator.py` to add `P.*` call sites.
- Adds the `T20` ruff rule and the `# noqa: T201` on `doctor_cmd.py`.
- Adds the `console` kwarg to `cli/_logging.py:configure_logging`.
- Adds `_silence_third_party_progress()` in `main.py`.
- Ships the 10 tests.

**The PR for #76 is held ready and merges *after* #74 has merged.** This avoids a cross-branch conflict on `cli/main.py` (both PRs touch the app wiring), and ensures `predict_cmd.py` exists before #76 wires `--progress` into it.

### Predict wiring

When #74's PR merges and `predict_cmd.py` exists, the #76 PR adds `--progress` to `predict_cmd.py` using `kind=PREDICT` per the contract in §5.4. This addition is small enough to be done as a final fixup commit on the #76 branch immediately before merging, or included in #74's PR directly — the orchestrator decides at implementation time based on which branch is furthest ahead. Either way, the contract is fully specified here.

---

## 11. Module Layout

```
src/custom_sam_peft/cli/_progress.py          NEW  (~250 lines)
src/custom_sam_peft/cli/_logging.py           TOUCHED  (+console kwarg)
src/custom_sam_peft/cli/main.py               TOUCHED  (+_silence_third_party_progress)
src/custom_sam_peft/cli/train_cmd.py          TOUCHED  (progress_session wiring)
src/custom_sam_peft/cli/eval_cmd.py           TOUCHED  (progress_session wiring)
src/custom_sam_peft/cli/run_cmd.py            TOUCHED  (three sequential sessions)
src/custom_sam_peft/cli/export_cmd.py         TOUCHED  (progress_session wiring)
src/custom_sam_peft/cli/doctor_cmd.py         TOUCHED  (# noqa: T201 on print(...))
src/custom_sam_peft/train/loop.py             TOUCHED  (P.advance_inner, P.update_postfix)
src/custom_sam_peft/train/trainer.py          TOUCHED  (P.advance_outer, P.reset_inner)
src/custom_sam_peft/eval/evaluator.py         TOUCHED  (P.advance_inner, P.update_postfix)
pyproject.toml                                TOUCHED  (T20 added to ruff select)
tests/unit/test_progress_resolve.py           NEW
tests/unit/test_progress_module.py            NEW
tests/unit/test_progress_integration.py       NEW
```

No deletions. No new top-level directories. `predict_cmd.py` not touched by the #76 PR (see §10).

---

## 12. Out of Scope

- **DDP / multi-rank progress.** Only rank-0 draws in a future multi-GPU context; no multi-process display protocol is designed here.
- **ipywidget UI for notebooks.** `PLAIN` mode is the notebook story. The Colab notebook (`notebooks/custom_sam_peft_train.ipynb`) should document `--progress plain`; updating the notebook is a follow-up (file an issue if needed after #76 merges).
- **Web dashboard.** Orthogonal to the terminal progress UI; not addressed here.
- **Smarter ETA.** `rich.Progress`'s built-in moving-average it/s is sufficient. No custom ETA model.
- **Replacing stdlib `logging` wholesale.** `logging` is kept; only the handler changes while a session is active.
- **`--progress-refresh-ms` knob.** Rich's internal ~4 Hz refresh cadence is correct; no user-facing knob.
- **ANSI byte-exactness tests.** Terminal output is tested at the line-content level only (snapshot for `plain` format; presence checks for `on` mode). Exact escape sequences vary by Rich version.
- **Colab rendering validation.** `Console().is_jupyter` auto-detects and forces `PLAIN`; Colab-specific widget rendering is not validated or shipped.
