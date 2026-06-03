# CLI Flag-Surface Audit + Consistency Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the eight-command CLI flag surface and implement the consistency fixes — one shared option vocabulary, validated enums, closed coverage gaps, leaner `predict`, a positional config form, and a `export --merge` rework — in three reviewable phases.

**Architecture:** A new `cli/_options.py` becomes the single source of truth for shared `Annotated[T, typer.Option(...)]` aliases, the `Progress`/`Split` enums, the `merge_cli_overrides` conflict-checking helper, and a lifted `discover_config` tree-walk. All eight commands consume that module. Phase 1 is a pure zero-behavior refactor; Phase 2 adds non-breaking surface; Phase 3 makes the two breaking changes (positional config, always-required export `--output`) and propagates emitted-command/README updates.

**Tech Stack:** Python 3.12, Typer (≥ 0.12, `Annotated` form), Click `Choice`, Pydantic config schema, pytest + `typer.testing.CliRunner`.

**Source of truth:** `docs/superpowers/specs/2026-06-02-cli-flag-audit-design.md`. The spec's §7 fixes phase boundaries + interface contracts; §6 lists the six verify-during-implementation items, baked in below.

---

## Conventions for every task

These apply to all tasks; do not restate them per step.

- **Worktree:** all paths are relative to the worktree root
  `/home/justin/projects/custom-sam-peft/.claude/worktrees/cli-flag-audit-115`.
- **Eager-import hazard:** `custom_sam_peft/__init__.py` eagerly imports the train
  chain, so a symbol-removal / refactor can un-import the whole package mid-edit.
  After every refactor step, gate on **all three**:
  `ruff check src tests`, `ruff format --check src tests`,
  `python -m py_compile <edited files>` (or `python -c "import custom_sam_peft.cli.main"`).
- **CLI test inner loop (CPU-only, no GPU fixtures):** run the CLI subset with the
  coverage gate bypassed:
  `uv run pytest tests/cli/ -o "addopts=" -q`
  (the global `--cov-fail-under=80` lives in `addopts`; `-o "addopts="` clears it).
  **Never** invoke `scripts/run_gpu_tests.sh` for these — the CLI tests must not
  pull GPU fixtures.
- **Full-suite regression** (run once at the end of Phase 2 and Phase 3, because a
  required-field / positional-arg change ripples beyond the named files):
  `uv run pytest -q` for CPU dirs; the GPU suite via `scripts/run_gpu_tests.sh`
  only if a touched path is GPU-covered. The CLI-flag work is CPU-only, so the
  CPU suite is the gate.
- **Commit message style:** Conventional Commits (`feat(cli): …`, `refactor(cli): …`,
  `test(cli): …`).
- **No new uncited numeric defaults:** this work removes/relocates flags; it adds
  no numeric hyperparameter default. The only internal constants introduced are
  `predict`'s fixed seed (`0`, matching today's CLI default) and dtype/device
  (`auto`) — these are behavior-preserving relocations of existing CLI defaults,
  not new tunables, so no provenance citation is required. Note this in the commit
  body.

---

## File Structure

**Phase 1 creates:**

- `src/custom_sam_peft/cli/_options.py` — shared vocabulary: `Annotated` aliases,
  `Progress` + `Split` enums, `merge_cli_overrides`, `discover_config`.
- `tests/cli/test_flag_consistency.py` — parser-introspection consistency guard.
- `tests/cli/test_options_unit.py` — unit tests for `merge_cli_overrides` +
  `discover_config`.

**Phase 1 modifies (refactor to consume `_options.py`, zero behavior change):**

- All eight command modules: `train_cmd.py`, `run_cmd.py`, `eval_cmd.py`,
  `export_cmd.py`, `init_cmd.py`, `doctor_cmd.py`, `predict_cmd.py`,
  `calibrate_cmd.py`.

**Phase 2 modifies:** the same command modules (additive flags) + extends both test
files; adds `tests/cli/test_predict_surface.py`.

**Phase 3 modifies:** `train_cmd.py`, `run_cmd.py`, `export_cmd.py`,
`runs/bundle.py` (`run_export`), `cli/_interactive.py`, `cli/setup_wizard.py`,
`README.md`; extends test files; adds `tests/cli/test_export_surface.py`.

---

## Phase 1 — Foundation (pure refactor, ZERO behavior change)

**Phase goal:** Create `_options.py` (aliases + enums + `merge_cli_overrides` +
`discover_config`) and refactor all eight commands to consume it, changing **no**
observable behavior. Add a consistency test that asserts only the *currently true*
vocabulary.

### Interface contract exposed by Phase 1 (consumed by Phases 2 & 3)

`src/custom_sam_peft/cli/_options.py` exports:

- **Enums:** `Progress(str, Enum)` with members `auto="auto"`, `on="on"`,
  `off="off"`, `plain="plain"`; `Split(str, Enum)` with members `val="val"`,
  `test="test"`.
- **Annotated aliases** (each `Annotated[T, typer.Option(...)]`, help text lives on
  the alias): `VerboseOpt` (`bool`, `-v`/`--verbose`), `OverrideOpt` (`list[str]`,
  `--override`), `ProgressOpt` (`Progress`, `--progress`, `metavar="MODE"`),
  `DryRunOpt` (`bool`, `--dry-run`), `NameOpt` (`str | None`, `--name`),
  `OutputDirOpt` (`Path | None`, `--output-dir`), `ConfigOpt` (`Path | None`,
  `--config`), `ConfigArg` (`Path | None`, positional `typer.Argument`, optional).
- **Helpers:** `merge_cli_overrides(explicit_overrides: list[str], *, name: str | None, output_dir: Path | None) -> list[str]`
  (error-on-conflict); `discover_config(checkpoint: Path) -> Path`.

Later phases import **exclusively** from this module for shared vocabulary.

> **Phase 1 behavior-preservation rule:** Because `ProgressOpt` types `--progress`
> as the `Progress` enum but every command body today reads a bare `str` and passes
> `progress_flag if progress_flag != "auto" else None` into `resolve_mode(...)`, the
> refactored bodies must compute the same value from the enum:
> `None if progress is Progress.auto else progress.value`. The parser still accepts
> exactly the strings `auto|on|off|plain` (enum values are those strings), so no CLI
> input changes. Verify by running the existing `tests/cli/` suite unchanged after
> each command refactor.

---

### Task 1.1: Create `_options.py` enums + `discover_config`

**Files:**

- Create: `src/custom_sam_peft/cli/_options.py`
- Test: `tests/cli/test_options_unit.py`

- [ ] **Step 1: Write the failing test for `discover_config`**

Add to `tests/cli/test_options_unit.py`:

```python
"""Unit tests for cli/_options.py helpers (CPU-only, no model load)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from custom_sam_peft.cli._options import Progress, Split, discover_config


def test_progress_enum_values() -> None:
    assert [m.value for m in Progress] == ["auto", "on", "off", "plain"]


def test_split_enum_values() -> None:
    assert [m.value for m in Split] == ["val", "test"]


def test_discover_config_finds_sibling(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ckpt = run_dir / "checkpoints" / "step_5" / "adapter"
    ckpt.mkdir(parents=True)
    cfg = run_dir / "config.yaml"
    cfg.write_text("run:\n  name: x\n")
    assert discover_config(ckpt).resolve() == cfg.resolve()


def test_discover_config_raises_when_absent(tmp_path: Path) -> None:
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()
    with pytest.raises(typer.BadParameter):
        discover_config(ckpt)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/cli/test_options_unit.py -o "addopts=" -q`
Expected: FAIL with `ModuleNotFoundError: custom_sam_peft.cli._options` (module not yet created).

- [ ] **Step 3: Create `_options.py` with the enums + `discover_config`**

Create `src/custom_sam_peft/cli/_options.py`. Lift the tree-walk **verbatim** from
`export_cmd._discover_config` (same `BadParameter` message + `param_hint`):

```python
"""Single source of truth for the shared CLI flag vocabulary.

Exposes Annotated[T, typer.Option(...)] aliases, the Progress/Split enums, the
merge_cli_overrides conflict-checking helper, and the shared discover_config
tree-walk. Every command imports its shared parameters from here so the surface
cannot drift the way `predict` once drifted from `train`.

Spec: docs/superpowers/specs/2026-06-02-cli-flag-audit-design.md §4.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer


class Progress(str, Enum):
    """Progress display mode (--progress). Values match the legacy bare-str flag."""

    auto = "auto"
    on = "on"
    off = "off"
    plain = "plain"


class Split(str, Enum):
    """Dataset split for `eval --split`. Only val/test are supported by eval.runner."""

    val = "val"
    test = "test"


def discover_config(checkpoint: Path) -> Path:
    """Walk up from *checkpoint* to the nearest sibling/ancestor config.yaml.

    Verbatim lift of the former export_cmd._discover_config tree-walk. Issue #249
    will later upgrade this single helper for self-describing checkpoints.
    """
    current = checkpoint.resolve()
    for parent in (current, *current.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise typer.BadParameter(
        f"could not auto-discover config.yaml above {checkpoint}; pass --config",
        param_hint="--config",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/cli/test_options_unit.py -o "addopts=" -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + import smoke**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -c "import custom_sam_peft.cli._options"`
Expected: all pass, no output errors.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/_options.py tests/cli/test_options_unit.py
git commit -m "feat(cli): add _options.py with Progress/Split enums and discover_config"
```

---

### Task 1.2: Add `merge_cli_overrides` (error-on-conflict)

**Files:**

- Modify: `src/custom_sam_peft/cli/_options.py`
- Test: `tests/cli/test_options_unit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli/test_options_unit.py`:

```python
from custom_sam_peft.cli._options import merge_cli_overrides


def test_merge_appends_name_and_output_dir() -> None:
    out = merge_cli_overrides(["train.epochs=10"], name="my-run", output_dir=Path("runs/exp1"))
    assert out == ["train.epochs=10", "run.name=my-run", "run.output_dir=runs/exp1"]


def test_merge_noop_when_no_convenience_flags() -> None:
    assert merge_cli_overrides(["a.b=c"], name=None, output_dir=None) == ["a.b=c"]


def test_merge_conflict_on_name_raises() -> None:
    with pytest.raises(typer.BadParameter):
        merge_cli_overrides(["run.name=bar"], name="foo", output_dir=None)


def test_merge_conflict_on_output_dir_raises() -> None:
    with pytest.raises(typer.BadParameter):
        merge_cli_overrides(["run.output_dir=x"], name=None, output_dir=Path("y"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_options_unit.py -o "addopts=" -q`
Expected: FAIL with `ImportError: cannot import name 'merge_cli_overrides'`.

- [ ] **Step 3: Implement `merge_cli_overrides`**

Add to `_options.py`:

```python
def merge_cli_overrides(
    explicit_overrides: list[str],
    *,
    name: str | None,
    output_dir: Path | None,
) -> list[str]:
    """Append synthesized run.name / run.output_dir overrides for convenience flags.

    The merged list is fed unchanged into load_config(config, overrides=...) ->
    apply_overrides. Error-on-conflict: if a convenience flag and an explicit
    --override target the same dotted key, raise typer.BadParameter rather than
    silently choosing a precedence.
    """
    explicit_keys = {ov.partition("=")[0] for ov in explicit_overrides if "=" in ov}
    merged = list(explicit_overrides)
    if name is not None:
        if "run.name" in explicit_keys:
            raise typer.BadParameter(
                "conflict: --name and --override run.name= both set run.name; pass only one",
                param_hint="--name",
            )
        merged.append(f"run.name={name}")
    if output_dir is not None:
        if "run.output_dir" in explicit_keys:
            raise typer.BadParameter(
                "conflict: --output-dir and --override run.output_dir= both set "
                "run.output_dir; pass only one",
                param_hint="--output-dir",
            )
        merged.append(f"run.output_dir={output_dir}")
    return merged
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_options_unit.py -o "addopts=" -q`
Expected: PASS (8 tests total).

- [ ] **Step 5: Lint**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/_options.py tests/cli/test_options_unit.py
git commit -m "feat(cli): add merge_cli_overrides with error-on-conflict"
```

---

### Task 1.3: Add the `Annotated` option aliases

**Files:**

- Modify: `src/custom_sam_peft/cli/_options.py`

- [ ] **Step 1: Add the aliases (no test yet — they are consumed/asserted by 1.4–1.11)**

Append to `_options.py`. Help text is copied verbatim from the current per-command
declarations so behavior is unchanged:

```python
from typing import Annotated

VerboseOpt = Annotated[
    bool, typer.Option("-v", "--verbose", help="Enable DEBUG logging.")
]
OverrideOpt = Annotated[
    list[str],
    typer.Option("--override", help="Override config keys: dotted.key=value."),
]
ProgressOpt = Annotated[
    Progress,
    typer.Option("--progress", help="Progress display mode: auto|on|off|plain.", metavar="MODE"),
]
DryRunOpt = Annotated[
    bool,
    typer.Option("--dry-run", help="Preview resolved inputs/config; do not run."),
]
NameOpt = Annotated[
    "str | None",
    typer.Option("--name", help="Convenience for run.name (synthesizes an --override)."),
]
OutputDirOpt = Annotated[
    "Path | None",
    typer.Option(
        "--output-dir",
        help="Convenience for run.output_dir, a run directory (synthesizes an --override).",
    ),
]
ConfigOpt = Annotated[
    "Path | None",
    typer.Option("--config", help="Path to config YAML."),
]
ConfigArg = Annotated[
    "Path | None",
    typer.Argument(help="Path to config YAML (the launch input)."),
]
```

> Note: the forward-ref strings (`"str | None"`, `"Path | None"`) keep
> `from __future__ import annotations` happy inside `Annotated`. Typer resolves
> them at command-build time. If a forward ref does not resolve in your Typer
> version, import `Path` at module top (already needed) and write the unquoted
> union directly — verify with the import smoke in Step 2.

- [ ] **Step 2: Lint + import smoke**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -c "import custom_sam_peft.cli._options as o; print(o.VerboseOpt, o.ConfigArg)"`
Expected: prints the alias reprs, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/custom_sam_peft/cli/_options.py
git commit -m "feat(cli): add shared Annotated option aliases to _options.py"
```

---

### Task 1.4: Refactor `export_cmd.py` to consume `discover_config` + aliases

**Files:**

- Modify: `src/custom_sam_peft/cli/export_cmd.py`

Do `export` first because it owns the original `_discover_config` and proves the lift
is byte-for-byte equivalent.

- [ ] **Step 1: Replace the local `_discover_config` with the shared helper**

In `export_cmd.py`: delete the local `def _discover_config(...)` body and import the
shared one. Keep a thin re-export name `_discover_config = discover_config` only if a
test references it; otherwise call `discover_config` directly. (Grep first:
`grep -rn "_discover_config" tests src` — if no external reference, remove it.)

```python
from custom_sam_peft.cli._options import ProgressOpt, VerboseOpt, discover_config
```

Replace `config_path = config if config is not None else _discover_config(checkpoint)`
with `config_path = config if config is not None else discover_config(checkpoint)`.

- [ ] **Step 2: Swap `-v` and `--progress` to the aliases**

Change the `verbose` param to use `VerboseOpt` and `progress_flag` to use
`ProgressOpt`. The body's `resolve_mode` call becomes:

```python
mode = resolve_mode(
    None if progress is Progress.auto else progress.value,
    os.environ,
    sys.stdout.isatty(),
    Console().is_jupyter,
)
```

Rename the parameter `progress_flag` → `progress` and import `Progress` from
`_options`. Leave `--checkpoint`, `--merge`, `--output`, `--config` exactly as they
are (still inline `typer.Option`; `--config` uses `ConfigOpt`? — for Phase 1 keep
`--config` inline to avoid help-text churn; aliasing `--config` is Phase 2/3 scope).

- [ ] **Step 3: Verify behavior unchanged**

Run: `uv run pytest tests/cli/ -o "addopts=" -q`
Expected: PASS (existing suite green; no export tests regress).

- [ ] **Step 4: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/export_cmd.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/export_cmd.py
git commit -m "refactor(cli): export consumes shared discover_config + option aliases"
```

---

### Task 1.5: Refactor `train_cmd.py` to consume aliases

**Files:**

- Modify: `src/custom_sam_peft/cli/train_cmd.py`

- [ ] **Step 1: Swap `--override`, `-v`, `--progress` to aliases**

Import `from custom_sam_peft.cli._options import OverrideOpt, Progress, ProgressOpt, VerboseOpt`.
Change `override` → `OverrideOpt`, `verbose` → `VerboseOpt`, `progress_flag` →
`ProgressOpt` (rename to `progress`). Update the `resolve_mode` call to
`None if progress is Progress.auto else progress.value`. Leave `--config` (required
`typer.Option`), `--resume`, `--time-limit`, `--eval`, `--export` inline and
unchanged.

- [ ] **Step 2: Verify behavior unchanged**

Run: `uv run pytest tests/cli/ -o "addopts=" -q`
Expected: PASS (`test_time_limit_cli.py`, `test_host_ram_cli.py` for `train` green).

- [ ] **Step 3: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/train_cmd.py`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/cli/train_cmd.py
git commit -m "refactor(cli): train consumes shared option aliases"
```

---

### Task 1.6: Refactor `run_cmd.py` to consume aliases

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py`

- [ ] **Step 1: Swap `-v`, `--progress` to aliases**

Import `Progress, ProgressOpt, VerboseOpt` from `_options`. Change `verbose` →
`VerboseOpt`, `progress_flag` → `ProgressOpt` (rename `progress`); update the
`resolve_mode` call as in 1.5. **Do NOT** add `--override` yet (Phase 2). Leave
`--config` required inline, `--resume`/`--time-limit`/`--finalize`/`--visualize`
unchanged.

- [ ] **Step 2: Verify behavior unchanged**

Run: `uv run pytest tests/cli/ -o "addopts=" -q`
Expected: PASS (`run`-targeted tests green).

- [ ] **Step 3: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/run_cmd.py`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py
git commit -m "refactor(cli): run consumes shared option aliases"
```

---

### Task 1.7: Refactor `eval_cmd.py` to consume aliases

**Files:**

- Modify: `src/custom_sam_peft/cli/eval_cmd.py`

- [ ] **Step 1: Swap `-v`, `--progress` to aliases (keep `--split` as bare str for now)**

Import `Progress, ProgressOpt, VerboseOpt`. Change `verbose` → `VerboseOpt`,
`progress_flag` → `ProgressOpt`; update `resolve_mode` as in 1.5. **Keep** `--split`
as the current bare `str` with the body's `if split not in ("val", "test")` raise
(typing it as `Split` is Phase 2). Keep `--config` optional inline and the
`--config is required` raise unchanged.

- [ ] **Step 2: Verify behavior unchanged**

Run: `uv run pytest tests/cli/ -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 3: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/eval_cmd.py`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/cli/eval_cmd.py
git commit -m "refactor(cli): eval consumes shared option aliases"
```

---

### Task 1.8: Refactor `predict_cmd.py` to consume `-v` + `--progress` aliases

**Files:**

- Modify: `src/custom_sam_peft/cli/predict_cmd.py`

- [ ] **Step 1: Swap `-v`, `--progress` to aliases only**

Import `Progress, ProgressOpt, VerboseOpt`. Change `verbose` → `VerboseOpt`,
`progress_flag` → `ProgressOpt`; update `resolve_mode` as in 1.5. **Keep**
`--device`/`--dtype`/`--seed`/`--merge-adapter`/`--config` exactly as they are
(their removal is Phase 2). Keep all callbacks + `click.Choice` flags unchanged.

- [ ] **Step 2: Verify behavior unchanged**

Run: `uv run pytest tests/cli/ tests/predict/ -o "addopts=" -q`
Expected: PASS (predict tests build `PredictOptions` directly; unaffected).

- [ ] **Step 3: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/predict_cmd.py`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/custom_sam_peft/cli/predict_cmd.py
git commit -m "refactor(cli): predict consumes shared -v/--progress aliases"
```

---

### Task 1.9: Refactor `doctor`, `init`, `calibrate` (no shared `-v` yet — they lack it)

**Files:**

- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`, `src/custom_sam_peft/cli/init_cmd.py`, `src/custom_sam_peft/cli/calibrate_cmd.py`

These three have **no** `--progress` and **no** `-v` today. Phase 1 is zero-behavior,
so there is nothing to swap to an alias yet (adding `-v` is Phase 2). This task is a
**no-op placeholder**: confirm there is nothing to refactor here in Phase 1, and skip
straight to Phase 2 for these three. No commit.

- [ ] **Step 1: Confirm no Phase-1 change needed**

Run: `grep -n "progress\|verbose\|'-v'\|\"-v\"" src/custom_sam_peft/cli/doctor_cmd.py src/custom_sam_peft/cli/init_cmd.py src/custom_sam_peft/cli/calibrate_cmd.py`
Expected: no `--progress`/`-v` declarations. Nothing to do in Phase 1.

---

### Task 1.10: Add the consistency test asserting *currently true* vocabulary

**Files:**

- Create: `tests/cli/test_flag_consistency.py`

- [ ] **Step 1: Write the consistency test**

It introspects each command's compiled Click params via
`typer.main.get_command(app)`. Assert only what is true **after Phase 1**: every
command that has `--progress` uses the `Progress` enum (post-1.4–1.8 that is
`train`/`run`/`eval`/`export`/`predict`); `train`/`run`/`eval`/`export`/`predict`
expose `-v`/`--verbose`; `train` exposes `--override`.

```python
"""Cross-command flag-consistency guard (parser introspection, CPU-only).

Tightened phase by phase. Phase 1 asserts only the currently-true vocabulary.
Spec §4.7.
"""

from __future__ import annotations

import click
import pytest
import typer.main

from custom_sam_peft.cli._options import Progress
from custom_sam_peft.cli.main import app

_GROUP = typer.main.get_command(app)


def _command(name: str) -> click.Command:
    cmd = _GROUP.get_command(None, name)  # type: ignore[attr-defined]
    assert cmd is not None, f"no such command: {name}"
    return cmd


def _opt(cmd: click.Command, name: str) -> click.Option | None:
    for p in cmd.params:
        if isinstance(p, click.Option) and p.name == name:
            return p
    return None


# Commands that carry --progress after Phase 1.
_PROGRESS_CMDS = ["train", "run", "eval", "export", "predict"]
# Commands that carry -v/--verbose after Phase 1.
_VERBOSE_CMDS = ["train", "run", "eval", "export", "predict"]


@pytest.mark.parametrize("name", _PROGRESS_CMDS)
def test_progress_is_progress_enum(name: str) -> None:
    opt = _opt(_command(name), "progress")
    assert opt is not None, f"{name} missing --progress"
    # Typer renders an Enum-typed option with a click.Choice of the enum values.
    assert isinstance(opt.type, click.Choice)
    assert set(opt.type.choices) == {m.value for m in Progress}


@pytest.mark.parametrize("name", _VERBOSE_CMDS)
def test_verbose_present(name: str) -> None:
    opt = _opt(_command(name), "verbose")
    assert opt is not None, f"{name} missing -v/--verbose"
    assert "-v" in opt.opts or "-v" in opt.secondary_opts


def test_train_has_override() -> None:
    assert _opt(_command("train"), "override") is not None
```

> If `opt.name` for `--progress` is not `"progress"` after the rename (it tracks the
> Python parameter name), adjust `_opt(..., "progress")`. Confirm by printing
> `[p.name for p in _command("train").params]` in a scratch run.

- [ ] **Step 2: Run the consistency test**

Run: `uv run pytest tests/cli/test_flag_consistency.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cli/test_flag_consistency.py
git commit -m "test(cli): add cross-command flag-consistency guard (Phase 1 vocabulary)"
```

---

### Task 1.11: Phase 1 behavior-preservation gate

**Files:** none (verification only).

- [ ] **Step 1: Full CLI suite + import smoke**

Run: `uv run pytest tests/cli/ -o "addopts=" -q && uv run python -c "import custom_sam_peft.cli.main"`
Expected: all green; clean import (proves the eager-import chain is intact).

- [ ] **Step 2: Lint gate (both ruff commands)**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: pass. End of Phase 1.

---

## Phase 2 — Additive standardization

**Phase goal:** Add all non-breaking surface: `--override` on `run`;
`--name`/`--output-dir` on `train`/`run`; `-v` on `doctor`/`init`/`calibrate`; `-y`
on `init`; `--dry-run` on `train`/`run`/`eval`; the `Split` enum + discover-then-
fallback on `eval`; the `init` tier-flag enums; **strip** `predict`'s
`--device`/`--seed`/`--dtype`/`--merge-adapter` (auto / derived / override replace
them); route predict `auto` dtype through `coerce_dtype_for_capability`. Tighten the
consistency test.

### Interface contract exposed by Phase 2 (consumed by Phase 3)

All non-breaking additions are present. `train`/`run` still take `--config` (no
positional yet). `predict`'s lean surface is final: it accepts no
`--device`/`--seed`/`--dtype`/`--merge-adapter`; merge is derived from
`detect_adapter_kind` (`lora` → merge, `qlora` → no merge); dtype `auto` is coerced
for compute capability. `eval` discovers a sibling `config.yaml` from `--checkpoint`
when `--config` is omitted; baseline eval (no checkpoint) still requires `--config`.
Phase 3 only changes config *form* (positional) and `export --merge`.

> **Phase 2 prerequisites baked in:** verify-item §6.3 (LoRA merge is
> result-neutral) is settled in Task 2.7; §6.4 (dtype coercion integration) in
> Task 2.8.

---

### Task 2.1: Add `--override` to `run`

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py`
- Test: `tests/cli/test_overrides_cli.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_overrides_cli.py`. Patterns follow `test_time_limit_cli.py`
(`CliRunner`, patch `run_training`/`run_init`, minimal config). `run` calls
`load_config(config)` with NO overrides today; assert that after the change a
`--override` reaches the loaded cfg.

```python
"""CLI integration for --override / --name / --output-dir (Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
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


def test_run_override_reaches_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import run_cmd

    seen: dict[str, Any] = {}

    def fake_orchestrate(cfg: Any, *a: Any, **k: Any) -> int:
        seen["epochs"] = cfg.train.epochs
        return 0

    monkeypatch.setattr(run_cmd, "_orchestrate", fake_orchestrate)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg), "--override", "train.epochs=7"])
    assert result.exit_code == 0, result.output
    assert seen["epochs"] == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_overrides_cli.py -o "addopts=" -q`
Expected: FAIL — `run` has no `--override` (unknown option) OR override not applied.

- [ ] **Step 3: Add `--override` to `run` and thread it into `load_config`**

In `run_cmd.run`, add an `override: OverrideOpt` parameter (import `OverrideOpt` from
`_options`). Change `cfg = load_config(config)` to
`cfg = load_config(config, overrides=override)`. Keep the auto-init branch ordering:
the `run_init(...)` call happens before `load_config` when the file is missing — that
is unchanged; overrides apply at load.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_overrides_cli.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/run_cmd.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/cli/test_overrides_cli.py
git commit -m "feat(cli): add --override to run (close the headline gap)"
```

---

### Task 2.2: Add `--name`/`--output-dir` to `train` and `run`

**Files:**

- Modify: `src/custom_sam_peft/cli/train_cmd.py`, `src/custom_sam_peft/cli/run_cmd.py`
- Test: `tests/cli/test_overrides_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli/test_overrides_cli.py`:

```python
def test_train_name_synthesizes_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.eval._artifacts import EvalArtifacts

    seen: dict[str, Any] = {}

    def fake_run_train(cfg: Any, **k: Any) -> EvalArtifacts:
        seen["name"] = cfg.run.name
        return EvalArtifacts(
            checkpoint_path=tmp_path / "adapter",
            peft_method="lora",
            run_dir=tmp_path,
            final_metrics=None,
        )

    monkeypatch.setattr(train_cmd, "run_train", fake_run_train)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg), "--name", "my-run"])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "my-run"


def test_train_name_conflict_raises(tmp_path: Path) -> None:
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(
        app, ["train", "--config", str(cfg), "--name", "foo", "--override", "run.name=bar"]
    )
    assert result.exit_code != 0
    assert "conflict" in result.output.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_overrides_cli.py -o "addopts=" -q`
Expected: FAIL — `train` has no `--name`.

- [ ] **Step 3: Add the flags + route through `merge_cli_overrides`**

In **both** `train_cmd.train` and `run_cmd.run`, add `name: NameOpt = None` and
`output_dir: OutputDirOpt = None` parameters (import `NameOpt`, `OutputDirOpt`,
`merge_cli_overrides`). Replace the `load_config(config, overrides=override)` call
with:

```python
merged = merge_cli_overrides(override, name=name, output_dir=output_dir)
cfg = load_config(config, overrides=merged)
```

For `run` (which had no `override` before 2.1, now has it), use the same pattern.
`merge_cli_overrides` raises `typer.BadParameter` on conflict — `CliRunner` surfaces
that as a non-zero exit with the message in `result.output`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_overrides_cli.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py tests/cli/test_overrides_cli.py
git commit -m "feat(cli): add --name/--output-dir convenience flags to train and run"
```

---

### Task 2.3: Add `--dry-run` to `train`/`run`/`eval`

**Files:**

- Modify: `src/custom_sam_peft/cli/train_cmd.py`, `src/custom_sam_peft/cli/run_cmd.py`, `src/custom_sam_peft/cli/eval_cmd.py`
- Test: `tests/cli/test_dry_run_cli.py` (create)

`--dry-run` here means: load + validate the config (applying overrides), print a
short resolved-config preview, and exit 0 **without** loading the model or training/
evaluating. Mirror predict's existing `--dry-run` intent (preview, skip model load).

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_dry_run_cli.py`:

```python
"""CLI --dry-run preview for train/run/eval (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: dr\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def test_train_dry_run_skips_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import train_cmd

    called = {"run": False}
    monkeypatch.setattr(
        train_cmd, "run_train", lambda *a, **k: called.__setitem__("run", True)
    )
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["run"] is False


def test_run_dry_run_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import run_cmd

    called = {"orch": False}
    monkeypatch.setattr(
        run_cmd, "_orchestrate", lambda *a, **k: called.__setitem__("orch", True) or 0
    )
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["orch"] is False


def test_eval_dry_run_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import eval_cmd

    called = {"eval": False}
    monkeypatch.setattr(
        eval_cmd, "run_eval", lambda *a, **k: called.__setitem__("eval", True)
    )
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["eval"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_dry_run_cli.py -o "addopts=" -q`
Expected: FAIL — `--dry-run` unknown on these commands.

- [ ] **Step 3: Add `--dry-run` to each command**

Add `dry_run: DryRunOpt = False` (import `DryRunOpt`). In each body, **after**
`load_config(...)` (so config errors still surface) and **before** the
`progress_session` / runner call, insert:

```python
if dry_run:
    rprint(f"[cyan]dry-run[/cyan] config={config} run.name={cfg.run.name} "
           f"output_dir={cfg.run.output_dir}")
    return
```

For `train`, place it after the `--time-limit` parse-and-apply so the previewed cfg
reflects the override; for `eval`, place it after `load_config` and after the split
check. For `run`, place it after the `finalize`/`time_limit` validation and
`load_config`, before computing `mode`. (The exact preview string content is not
asserted by the tests — only that the runner is not called and exit is 0.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_dry_run_cli.py -o "addopts=" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py src/custom_sam_peft/cli/eval_cmd.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py src/custom_sam_peft/cli/eval_cmd.py tests/cli/test_dry_run_cli.py
git commit -m "feat(cli): add --dry-run preview to train/run/eval"
```

---

### Task 2.4: Type `eval --split` as the `Split` enum + discover-then-fallback

**Files:**

- Modify: `src/custom_sam_peft/cli/eval_cmd.py`
- Test: `tests/cli/test_eval_surface.py` (create)

This bundles two §6.1-grounded changes. **Verify-item §6.1 is settled:** `eval.runner`
only branches on `split == "val"` / `split == "test"` (no `train` branch — confirmed
in `src/custom_sam_peft/eval/runner.py:110–124`), so `Split = {val, test}` stays.

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_eval_surface.py`:

```python
"""CLI surface for eval: Split enum validation + config discovery (Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_config(path: Path, tmp_path: Path) -> None:
    path.write_text(
        "run:\n  name: ev\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )


def test_eval_bad_split_rejected_by_parser(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    _write_config(cfg, tmp_path)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output or "split" in result.output.lower()


def test_eval_discovers_sibling_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import eval_cmd

    run_dir = tmp_path / "run"
    ckpt = run_dir / "checkpoints" / "step_5" / "adapter"
    ckpt.mkdir(parents=True)
    _write_config(run_dir / "config.yaml", tmp_path)

    seen: dict[str, Any] = {}

    def fake_run_eval(cfg: Any, **k: Any) -> Any:
        seen["name"] = cfg.run.name

        class _R:
            overall = {"mAP": 0.0}

        return _R()

    monkeypatch.setattr(eval_cmd, "run_eval", fake_run_eval)
    result = runner.invoke(app, ["eval", "--checkpoint", str(ckpt)])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "ev"


def test_eval_baseline_without_config_still_raises(tmp_path: Path) -> None:
    # No --checkpoint, no --config: baseline eval still requires --config.
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0
    assert "config" in result.output.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_eval_surface.py -o "addopts=" -q`
Expected: FAIL — bad split currently passes the parser; discovery not wired.

- [ ] **Step 3: Implement the Split enum + discover-then-fallback**

In `eval_cmd.evaluate`:

- Import `Split`, `discover_config` from `_options`.
- Change `split: str = typer.Option("val", "--split", ...)` to `split: Split = Split.val`
  (use a shared alias if you choose to add `SplitOpt`; the spec lists `Split` enum
  directly on `eval`). Update the body: remove the manual
  `if split not in ("val", "test"): raise ...` block, and replace `split_lit` with
  `split.value` passed to `run_eval(..., split=split.value, ...)`. (Keep the
  `Literal["val","test"]` typing satisfied by `cast` if mypy needs it:
  `cast(Literal["val", "test"], split.value)`.)
- Replace the config requirement logic. Today:

  ```python
  if config is None:
      raise typer.BadParameter("--config is required", param_hint="--config")
  ```

  Change to discover-then-fallback **only when a checkpoint is present**:

  ```python
  if config is None:
      if checkpoint is not None:
          config = discover_config(checkpoint)
      else:
          raise typer.BadParameter(
              "--config is required for baseline (zero-shot) eval",
              param_hint="--config",
          )
  ```

  Keep the `--interactive` early-return above this unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_eval_surface.py -o "addopts=" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/eval_cmd.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/eval_cmd.py tests/cli/test_eval_surface.py
git commit -m "feat(cli): eval --split is a Split enum; discover config from --checkpoint"
```

---

### Task 2.5: Add `-v` to `doctor`/`init`/`calibrate` and `-y` to `init`

**Files:**

- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`, `src/custom_sam_peft/cli/init_cmd.py`, `src/custom_sam_peft/cli/calibrate_cmd.py`
- Test: `tests/cli/test_verbose_help.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_verbose_help.py` (uses `--help` so no model/CUDA needed):

```python
"""-v on doctor/init/calibrate and -y on init are surfaced (Phase 2)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("cmd", ["doctor", "init", "calibrate"])
def test_verbose_in_help(cmd: str) -> None:
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "-v" in result.output


def test_init_short_yes_in_help() -> None:
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "-y" in result.output
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_verbose_help.py -o "addopts=" -q`
Expected: FAIL — `-v` absent on these commands; `-y` absent on `init`.

- [ ] **Step 3: Add the flags + wire `-v`**

- `doctor`: import `VerboseOpt`, `configure_logging`. Add `verbose: VerboseOpt = False`.
  At the top of the body call `configure_logging(verbose)` (import from
  `custom_sam_peft.cli._logging`, matching the other commands). Behavior otherwise
  unchanged.
- `init`: add `verbose: VerboseOpt = False`; call `configure_logging(verbose)` at the
  top of `init` (before the interactive branch). Add the `-y` short form to the
  existing `yes` option: change `typer.Option(False, "--yes", help=...)` to
  `typer.Option(False, "--yes", "-y", help=...)`.
- `calibrate`: add `verbose: VerboseOpt = False`; call `configure_logging(verbose)`
  at the top of `calibrate` (before the CUDA check). Behavior otherwise unchanged.

> `configure_logging` is the existing `cli/_logging.configure_logging(verbose: bool)`
> the other five commands already call.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_verbose_help.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/doctor_cmd.py src/custom_sam_peft/cli/init_cmd.py src/custom_sam_peft/cli/calibrate_cmd.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/doctor_cmd.py src/custom_sam_peft/cli/init_cmd.py src/custom_sam_peft/cli/calibrate_cmd.py tests/cli/test_verbose_help.py
git commit -m "feat(cli): add -v to doctor/init/calibrate and -y to init"
```

---

### Task 2.6: Type `init`'s tier flags as validated enums

**Files:**

- Modify: `src/custom_sam_peft/cli/init_cmd.py`, `src/custom_sam_peft/cli/_options.py`
- Test: `tests/cli/test_init_surface.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_init_surface.py`:

```python
"""init tier flags reject bad values at the parser (Phase 2)."""

from __future__ import annotations

from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_init_bad_preset_rejected() -> None:
    result = runner.invoke(app, ["init", "--preset", "bogus", "--output", "x.yaml"])
    assert result.exit_code != 0
    assert "bogus" in result.output or "preset" in result.output.lower()


def test_init_bad_intensity_rejected() -> None:
    result = runner.invoke(app, ["init", "--intensity", "nope", "--output", "x.yaml"])
    assert result.exit_code != 0


def test_init_bad_class_imbalance_rejected() -> None:
    result = runner.invoke(app, ["init", "--class-imbalance", "nope", "--output", "x.yaml"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_init_surface.py -o "addopts=" -q`
Expected: FAIL — bad values currently reach `run_init` (post-parse), and the test
may even start a CUDA-less init that errors differently; the point is the parser must
reject *before* `run_init`.

- [ ] **Step 3: Add tier enums + type the flags**

Add three enums to `_options.py`, mirroring the schema `Literal`s in
`config/schema.py` (`Preset`, `Intensity`, `ClassImbalance`):

```python
class PresetChoice(str, Enum):
    natural = "natural"
    medical = "medical"
    satellite = "satellite"
    microscopy = "microscopy"
    none = "none"
    custom = "custom"


class IntensityChoice(str, Enum):
    safe = "safe"
    medium = "medium"
    aggressive = "aggressive"


class ClassImbalanceChoice(str, Enum):
    balanced = "balanced"
    moderate = "moderate"
    severe = "severe"
```

In `init_cmd.init`, change the three `typer.Option(... case_sensitive=False ...)`
params:

- `preset: PresetChoice = typer.Option(PresetChoice.natural, "--preset", case_sensitive=False, help=...)`
- `intensity: IntensityChoice = typer.Option(IntensityChoice.medium, "--intensity", case_sensitive=False, help=...)`
- `class_imbalance: ClassImbalanceChoice = typer.Option(ClassImbalanceChoice.balanced, "--class-imbalance", case_sensitive=False, help=...)`

In the body, pass `.value` (lowercased values already match) into `run_init`:
`run_init(template, output, preset=preset.value, intensity=intensity.value, class_imbalance=class_imbalance.value, force=force)`. Keep `run_init`'s
post-parse `get_args(...)` validation as the belt-and-suspenders check (it is also
called from `run`'s auto-init path and tests) — do not remove it.

> The enum member names must be valid Python identifiers; `none` is fine (not a
> keyword). Values match the schema `Literal` strings exactly.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_init_surface.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/init_cmd.py src/custom_sam_peft/cli/_options.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/init_cmd.py src/custom_sam_peft/cli/_options.py tests/cli/test_init_surface.py
git commit -m "feat(cli): validate init tier flags via enums at the parser"
```

---

### Task 2.7: Settle verify-item §6.3 (LoRA merge is result-neutral)

**Files:** none (investigation + a docstring note).

Before dropping `predict --merge-adapter`, confirm LoRA merge changes only
speed/memory and never inference outputs; QLoRA-unmerged is the safe default.

- [ ] **Step 1: Confirm the merge semantics from the codebase**

Read `src/custom_sam_peft/predict/adapter_load.py` (`detect_adapter_kind`,
`maybe_merge_adapter`, `_lora.merge_lora`) and `src/custom_sam_peft/train/checkpoint.py`
(`save_merged` docstring: "For QLoRA wrappers, merge_lora dequantizes the 4-bit base
to compute_dtype during folding"). Confirm:
(a) LoRA `merge_lora` folds deltas into base weights — mathematically equivalent
forward, a speed/memory change only;
(b) QLoRA merge dequantizes 4-bit → compute_dtype (a memory blowup), so unmerged is
the safe default.

Record the finding inline in the eventual derivation comment in Task 2.9. No code
change in this task; it gates 2.9.

- [ ] **Step 2: Record the decision**

Confirm the derivation is `merge = (detect_adapter_kind(checkpoint) == "lora")`.
Proceed to 2.8/2.9. No commit (no file change).

---

### Task 2.8: Route predict's `auto` dtype through `coerce_dtype_for_capability` (§6.4)

**Files:**

- Modify: `src/custom_sam_peft/predict/runner.py`
- Test: `tests/predict/test_dtype_coercion.py` (create)

This must land **before** dropping `--dtype` (Task 2.9), so the always-`auto` surface
is correct on sub-CC-8.0 GPUs. Integration point: `_resolve_config`, the
`--- dtype resolution ---` block (`runner.py:193–199`).

- [ ] **Step 1: Write the failing test (CPU-only, capability injected)**

`coerce_dtype_for_capability` accepts an explicit `capability` tuple (CPU-testable).
But `_resolve_config` reads device capability from CUDA. Add a small seam: have
`_resolve_config` call `coerce_dtype_for_capability(dtype_torch, device=torch.device(device_str))`
when `device_str == "cuda"`. Test the *helper wiring* indirectly by asserting the
function is invoked for the cuda branch via monkeypatch.

Create `tests/predict/test_dtype_coercion.py`:

```python
"""predict auto-dtype is routed through coerce_dtype_for_capability (§6.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from custom_sam_peft.predict.runner import PredictOptions, _resolve_config


def _opts(**over: Any) -> PredictOptions:
    base = dict(
        images=Path("img"),
        prompts="a",
        output=Path("out"),
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.3,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="auto",
        dtype="auto",
        seed=0,
        dry_run=True,
        verbose=False,
    )
    base.update(over)
    return PredictOptions(**base)  # type: ignore[arg-type]


def test_auto_dtype_routed_through_coercion(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    def fake_coerce(dtype: torch.dtype, **k: Any) -> torch.dtype:
        calls["dtype"] = dtype
        calls["kwargs"] = k
        return torch.float16  # pretend sub-CC-8.0 coercion

    monkeypatch.setattr(
        "custom_sam_peft.predict.runner.coerce_dtype_for_capability", fake_coerce, raising=False
    )
    # Force the cuda branch deterministically without a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rcfg = _resolve_config(_opts(device="cuda", dtype="auto"))
    assert calls["dtype"] is torch.bfloat16
    assert rcfg.dtype is torch.float16
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/predict/test_dtype_coercion.py -o "addopts=" -q`
Expected: FAIL — `coerce_dtype_for_capability` is not imported/called in
`_resolve_config`.

- [ ] **Step 3: Wire the coercion into `_resolve_config`**

In `runner.py`, import at module top:
`from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability`.
In the dtype block (after `dtype_torch = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32`),
add:

```python
if device_str == "cuda":
    dtype_torch = coerce_dtype_for_capability(dtype_torch, device=torch.device("cuda"))
    dtype_str = "float16" if dtype_torch is torch.float16 else dtype_str
```

This keeps `auto` correct on sub-CC-8.0 cards (bf16 → fp16) and is a no-op on CPU /
CC ≥ 8.0. The coerced `dtype_torch` flows into `model.to(rcfg.device, dtype=rcfg.dtype)`
unchanged (`runner.py:310`).

> Add `"float16"` to the `_ResolvedConfig.dtype_str` set of expected values in any
> nearby docstring; `dtype_str` is used only for logging (`runner.py:282`, the
> dry-run print), so a `"float16"` string there is correct and harmless.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/predict/test_dtype_coercion.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/predict/runner.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/predict/runner.py tests/predict/test_dtype_coercion.py
git commit -m "fix(predict): route auto dtype through coerce_dtype_for_capability (sub-CC-8.0 safe)"
```

---

### Task 2.9: Strip predict `--device`/`--seed`/`--dtype`/`--merge-adapter` (hard removal)

**Files:**

- Modify: `src/custom_sam_peft/cli/predict_cmd.py`
- Test: `tests/cli/test_predict_surface.py` (create)

**Blast radius (verified):** `PredictOptions(` is constructed in
`src/custom_sam_peft/cli/predict_cmd.py` and 8 test files
(`tests/gpu/test_predict_nchannel_gpu.py`, `tests/unit/test_predict_image_size_contract.py`,
`tests/predict/test_gpu_predict.py`, `tests/predict/test_runner_smoke.py`,
`tests/predict/test_dry_run.py`, `tests/predict/test_config_layering.py`,
`tests/predict/test_predict_oom_ladder.py`, `tests/predict/test_predict_fits_8gb.py`).

**Decision (minimize blast radius):** Keep the `PredictOptions` dataclass fields
`device`, `dtype`, `seed`, `merge_adapter` **internal** (so the 8 test constructors
stay valid), but **remove them from the CLI signature**. The CLI shell sets fixed
internal defaults: `device="auto"`, `dtype="auto"`, `seed=0`, and derives
`merge_adapter` from `detect_adapter_kind`. This is the spec's "the field may stay
internal" path (§4.6).

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_predict_surface.py`:

```python
"""predict CLI surface: removed flags error; merge derived from adapter kind."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize(
    "flag",
    [["--device", "cuda"], ["--dtype", "float32"], ["--seed", "1"], ["--merge-adapter"]],
)
def test_removed_predict_flags_error(flag: list[str], tmp_path: Path) -> None:
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    result = runner.invoke(
        app,
        ["predict", "--images", str(imgs), "--prompts", "a", "--output", str(tmp_path / "o"), *flag],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def _make_lora_ckpt(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    return d


def test_merge_derived_lora_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import predict_cmd

    seen: dict[str, Any] = {}

    def fake_run_predict(opts: Any) -> Any:
        seen["merge"] = opts.merge_adapter

        class _R:
            n_images = 0
            n_predictions = 0
            elapsed_sec = 0.0

        return _R()

    monkeypatch.setattr(predict_cmd, "run_predict", fake_run_predict)
    monkeypatch.setattr(predict_cmd, "detect_adapter_kind", lambda p: "lora")
    ckpt = _make_lora_ckpt(tmp_path / "ckpt")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    result = runner.invoke(
        app,
        ["predict", "--images", str(imgs), "--prompts", "a",
         "--output", str(tmp_path / "o"), "--checkpoint", str(ckpt)],
    )
    assert result.exit_code == 0, result.output
    assert seen["merge"] is True


def test_merge_derived_qlora_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import predict_cmd

    seen: dict[str, Any] = {}

    def fake_run_predict(opts: Any) -> Any:
        seen["merge"] = opts.merge_adapter

        class _R:
            n_images = 0
            n_predictions = 0
            elapsed_sec = 0.0

        return _R()

    monkeypatch.setattr(predict_cmd, "run_predict", fake_run_predict)
    monkeypatch.setattr(predict_cmd, "detect_adapter_kind", lambda p: "qlora")
    ckpt = _make_lora_ckpt(tmp_path / "ckpt")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    result = runner.invoke(
        app,
        ["predict", "--images", str(imgs), "--prompts", "a",
         "--output", str(tmp_path / "o"), "--checkpoint", str(ckpt)],
    )
    assert result.exit_code == 0, result.output
    assert seen["merge"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_predict_surface.py -o "addopts=" -q`
Expected: FAIL — the removed flags are still accepted; merge not derived.

- [ ] **Step 3: Remove the four flags + derive merge**

In `predict_cmd.predict`:

- Delete the `merge_adapter`, `device`, `dtype`, `seed` **parameters** from the
  signature (and their `click.Choice` / help).
- Import `from custom_sam_peft.predict.adapter_load import detect_adapter_kind`.
- In the body, after the `--interactive` branch and before constructing
  `PredictOptions`, derive merge:

  ```python
  merge_adapter = (
      detect_adapter_kind(checkpoint) == "lora" if checkpoint is not None else False
  )
  ```

- Construct `PredictOptions(...)` with fixed internal values for the dropped CLI
  fields: `merge_adapter=merge_adapter, device="auto", dtype="auto", seed=0`. Remove
  the now-unused `cast(...)` for `device`/`dtype`. Keep `config`, `score_threshold`,
  `top_k`, `save_masks`, `visualize`, `batch_size`, `dry_run`, `verbose` as-is.

> Importing `detect_adapter_kind` at module top is fine — `adapter_load.py` does not
> import torch at module top for `detect_adapter_kind`/`read_adapter_base_model_name`
> (verify with the import smoke). If it does pull a heavy import, import it lazily
> inside the body instead, but keep it referenceable as
> `predict_cmd.detect_adapter_kind` so the test's monkeypatch resolves — i.e. assign
> `detect_adapter_kind` to a module-level name.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_predict_surface.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Regression — full predict-construction blast radius**

Run: `uv run pytest tests/predict/ tests/unit/test_predict_image_size_contract.py -o "addopts=" -q`
Expected: PASS — the 8 `PredictOptions(...)` constructors still pass `device`/`dtype`/
`seed`/`merge_adapter` and those fields remain on the dataclass, so they are
unaffected. (GPU-marked predict tests are skipped on CPU; that is expected — do not
run the GPU runner here.)

- [ ] **Step 6: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/predict_cmd.py && uv run python -c "import custom_sam_peft.cli.main"`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/predict_cmd.py tests/cli/test_predict_surface.py
git commit -m "feat(cli): drop predict --device/--dtype/--seed/--merge-adapter; derive merge from adapter kind"
```

---

### Task 2.10: Tighten the consistency test to the Phase-2 vocabulary

**Files:**

- Modify: `tests/cli/test_flag_consistency.py`

- [ ] **Step 1: Extend the test**

Now `-v` exists on **all eight** commands; `--dry-run` is on `train`/`run`/`eval`/
`predict`; `eval --split` is a `Split` enum; `run` has `--override`.

```python
# After Phase 2, -v is on all eight commands.
_ALL_CMDS = ["train", "run", "eval", "export", "init", "doctor", "predict", "calibrate"]
# --dry-run on these four.
_DRY_RUN_CMDS = ["train", "run", "eval", "predict"]
```

Add:

```python
@pytest.mark.parametrize("name", _ALL_CMDS)
def test_verbose_present_all(name: str) -> None:
    opt = _opt(_command(name), "verbose")
    assert opt is not None, f"{name} missing -v/--verbose"
    assert "-v" in opt.opts


@pytest.mark.parametrize("name", _DRY_RUN_CMDS)
def test_dry_run_present(name: str) -> None:
    assert _opt(_command(name), "dry_run") is not None, f"{name} missing --dry-run"


def test_run_has_override_after_phase2() -> None:
    assert _opt(_command("run"), "override") is not None


def test_eval_split_is_split_enum() -> None:
    from custom_sam_peft.cli._options import Split

    opt = _opt(_command("eval"), "split")
    assert opt is not None
    assert isinstance(opt.type, click.Choice)
    assert set(opt.type.choices) == {m.value for m in Split}
```

Replace the Phase-1 `_VERBOSE_CMDS` test usage with `_ALL_CMDS` (keep the
`_PROGRESS_CMDS` test as-is — `doctor`/`init`/`calibrate` still have no `--progress`,
which is intentional per §5.3).

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/cli/test_flag_consistency.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cli/test_flag_consistency.py
git commit -m "test(cli): tighten consistency guard to Phase-2 vocabulary"
```

---

### Task 2.11: Phase 2 regression gate

**Files:** none (verification only).

- [ ] **Step 1: Full CLI + predict subset**

Run: `uv run pytest tests/cli/ tests/predict/ tests/unit/test_predict_image_size_contract.py -o "addopts=" -q`
Expected: all green (GPU-marked tests skipped on CPU).

- [ ] **Step 2: Full CPU suite (required-field ripple check)**

Run: `uv run pytest -q`
Expected: green (or only pre-existing GPU skips). Investigate any new failure before
proceeding — a removed flag or new param can ripple beyond the named files.

- [ ] **Step 3: Lint gate**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: pass. End of Phase 2.

---

## Phase 3 — Breaking cleanup + propagation

**Phase goal:** Promote `--config` to an optional positional argument on `train`/
`run` with a hidden `--config` alias; rework `export --merge` so `--output` is always
required (drop the conditional `ValueError` in `run_export`); update
`setup_wizard`/`_interactive` emitted commands + README quickstart/command table to
the positional form.

### Interface contract exposed by Phase 3

Canonical user-facing form is positional (`csp run config.yaml`,
`csp train config.yaml`); the hidden `--config` alias preserves back-compat for #244's
examples and every emitted command. `export` requires `--output` in both `--merge`
and non-merge modes; `run_export` no longer raises the conditional ValueError. No
later phase.

---

### Task 3.1: `train`/`run` — optional positional `config` + hidden `--config` alias

**Files:**

- Modify: `src/custom_sam_peft/cli/train_cmd.py`, `src/custom_sam_peft/cli/run_cmd.py`
- Test: `tests/cli/test_positional_config.py` (create)

- [ ] **Step 1: Write the failing tests**

Both the positional form and the `--config` alias must parse and produce the same
loaded config; passing neither must raise the existing required-config error.

```python
"""train/run accept positional config + hidden --config alias (Phase 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: pc\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def _patch_train(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], tmp_path: Path) -> None:
    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.eval._artifacts import EvalArtifacts

    def fake_run_train(cfg: Any, **k: Any) -> EvalArtifacts:
        seen["name"] = cfg.run.name
        return EvalArtifacts(
            checkpoint_path=tmp_path / "adapter", peft_method="lora",
            run_dir=tmp_path, final_metrics=None,
        )

    monkeypatch.setattr(train_cmd, "run_train", fake_run_train)


def test_train_positional_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    _patch_train(monkeypatch, seen, tmp_path)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", str(cfg)])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "pc"


def test_train_config_alias_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    _patch_train(monkeypatch, seen, tmp_path)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "pc"


def test_train_no_config_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
```

(Mirror with `run` using the `_orchestrate` patch from Task 2.1's test module.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_positional_config.py -o "addopts=" -q`
Expected: FAIL — positional `config` not accepted (it is `--config`-only today).

- [ ] **Step 3: Add the positional + hidden alias**

In `train_cmd.train` and `run_cmd.run`, replace the required
`config: Path = typer.Option(..., "--config", help=...)` with **two** parameters:

```python
config_arg: ConfigArg = None,        # optional positional
config_opt: ConfigOpt = None,        # hidden --config alias
```

Use a dedicated hidden alias declaration rather than the shared `ConfigOpt` so we can
set `hidden=True`. Define a local option (or add a `HiddenConfigOpt` alias to
`_options.py`):

```python
# in _options.py
HiddenConfigOpt = Annotated[
    "Path | None",
    typer.Option("--config", hidden=True, help="Alias for the positional config (back-compat)."),
]
```

In each body, resolve the effective config and enforce required-in-body:

```python
config = config_arg if config_arg is not None else config_opt
if config is None:
    rprint("[red]error[/red] a config path is required (positional or --config).")
    raise typer.Exit(code=1)
```

Then proceed with the existing `merge_cli_overrides` + `load_config(config, ...)`
flow. Keep every downstream reference to `config` (the `format_time_limit_message`
calls etc.) working by binding the local `config` name as above.

> Typer maps the positional `typer.Argument` parameter and the `--config` option
> param to distinct Python names; pick `config_arg` / `config_opt` so they do not
> collide. The `_ResumeAwareGroup` patch keys on `p.name == "resume"`, unaffected.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_positional_config.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Regression — existing train/run tests still green**

Run: `uv run pytest tests/cli/ -o "addopts=" -q`
Expected: PASS — `test_time_limit_cli.py`/`test_host_ram_cli.py` use
`["train", "--config", str(cfg), ...]` which still works via the alias.

- [ ] **Step 6: Lint + py_compile + import smoke**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py src/custom_sam_peft/cli/_options.py && uv run python -c "import custom_sam_peft.cli.main"`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py src/custom_sam_peft/cli/_options.py tests/cli/test_positional_config.py
git commit -m "feat(cli): train/run accept positional config with hidden --config alias"
```

---

### Task 3.2: `export --merge` rework — `--output` always required (§6.2)

**Files:**

- Modify: `src/custom_sam_peft/cli/export_cmd.py`, `src/custom_sam_peft/runs/bundle.py`
- Test: `tests/cli/test_export_surface.py` (create)

**Verify-item §6.2 (settled from code):** `run_export(merge=True)` calls
`save_merged(wrapper, out)` and `save_merged` writes `pytorch_model.bin` +
`channel_adapter.pt` **directly into the path it is given** (`checkpoint.py:130–147`).
So with `--output` always required, merged weights land in `--output` itself (not a
nested `--output/merged`). The CLI success message must print that path.

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_export_surface.py`:

```python
"""export --output is always required; merge lands at --output (Phase 3, §6.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _ckpt_with_config(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    ckpt = run_dir / "adapter"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (run_dir / "config.yaml").write_text(
        "run:\n  name: ex\ndata:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\ntrain:\n  epochs: 1\n"
    )
    return ckpt


def test_export_requires_output_non_merge(tmp_path: Path) -> None:
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(app, ["export", "--checkpoint", str(ckpt)])
    assert result.exit_code != 0
    assert "output" in result.output.lower()


def test_export_requires_output_with_merge(tmp_path: Path) -> None:
    ckpt = _ckpt_with_config(tmp_path)
    result = runner.invoke(app, ["export", "--checkpoint", str(ckpt), "--merge"])
    assert result.exit_code != 0
    assert "output" in result.output.lower()


def test_run_export_merge_lands_at_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Library-level: merged weights land at the given output path itself.
    from custom_sam_peft.runs import bundle

    out = tmp_path / "merged-out"
    captured: dict[str, Any] = {}

    def fake_load_sam31(*a: Any, **k: Any) -> Any:
        return object()

    monkeypatch.setattr("custom_sam_peft.models.sam3.load_sam31", fake_load_sam31)
    monkeypatch.setattr("custom_sam_peft.train.checkpoint.load_adapter", lambda *a, **k: None)
    monkeypatch.setattr(
        "custom_sam_peft.train.checkpoint.save_merged",
        lambda wrapper, path: captured.__setitem__("path", path),
    )

    class _Cfg:
        class model: ...
        class data:
            channels = 3
            channel_semantics = "rgb"

    result = bundle.run_export(_Cfg(), tmp_path / "adapter", merge=True, output=out)
    assert captured["path"] == out
    assert result == out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_export_surface.py -o "addopts=" -q`
Expected: FAIL — merge currently allows omitting `--output` (defaults to
`run_dir/merged`); the CLI accepts no `--output` with `--merge`.

- [ ] **Step 3: Make `--output` required in `export_cmd` + drop the conditional in `run_export`**

In `export_cmd.export`:

- Change `output: Path | None = typer.Option(None, "--output", ...)` to
  `output: Path = typer.Option(..., "--output", help="Output directory (created if missing).")`
  (required; help states it is a dir per §5.4).
- The success message already prints `out` (the path `run_export` returns), which for
  merge is now `--output` itself — keep `rprint(f"[green]merged[/green] {out}")` /
  `[green]adapter[/green] {out}`; both now print the user-supplied `--output`.

In `runs/bundle.run_export`:

- Change the signature `output: Path | None = None` → keep the param but make the
  body unconditional. Remove the `if output is None: raise ValueError(...)` branch and
  the `output if output is not None else (run_dir / "merged")` default. Since
  `export_cmd` now always passes `--output`, write:

  ```python
  if merge:
      save_merged(wrapper, output)
  else:
      save_adapter(wrapper, output)
  return output
  ```

- Update the docstring: drop "default: `<run_dir>/merged`" and "required when not
  merging"; state output is always required and merged weights land at `output`.

> **Caller audit:** `run_export` is also called from `train_cmd` (`run_export(cfg, result.checkpoint_path)`),
> `eval_cmd` (`run_export(cfg, checkpoint)`), and `run`'s pipeline. These call with
> NO `output` (relying on the merge-default). Grep:
> `grep -rn "run_export(" src tests`. For each non-export-CLI caller, the call is the
> `--export`-toggle path that exports an **adapter** bundle (merge=False) and TODAY
> relies on... — VERIFY: these callers pass `merge` defaulting to False and `output`
> defaulting to None, which TODAY raises `ValueError("output is required when not
> merging")`. Confirm current behavior by reading each call site:
> if they currently pass no `output` with `merge=False`, they are *already* hitting
> the raise — meaning they must pass an output. Re-read `train_cmd:137`,
> `eval_cmd:111`. If they rely on the default, you must preserve a default for the
> *library* path while making the *CLI* require `--output`. **Resolution:** keep a
> safe default in `run_export` for the non-CLI callers — default merged →
> `run_dir/merged`, default adapter → `run_dir/exported` — but the CLI `export`
> command always supplies `--output`. Only remove the *conditional-required raise*;
> keep sensible library defaults so train/eval `--export` still work. Implement
> whichever matches the verified current behavior; the test
> `test_run_export_merge_lands_at_output` passes an explicit `output`, so it pins the
> explicit-output contract regardless.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_export_surface.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Regression — train/eval `--export` paths**

Run: `uv run pytest tests/cli/test_host_ram_cli.py tests/cli/test_run_single_eval.py -o "addopts=" -q`
Expected: PASS — the `train --export` / `run` pipeline still exports without a CLI
`--output` because the library default is preserved for those callers.

- [ ] **Step 6: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/export_cmd.py src/custom_sam_peft/runs/bundle.py`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/export_cmd.py src/custom_sam_peft/runs/bundle.py tests/cli/test_export_surface.py
git commit -m "feat(cli): export --output always required; drop conditional ValueError in run_export"
```

---

### Task 3.3: Propagate the positional form into `_interactive` + `setup_wizard` (§6.5)

**Files:**

- Modify: `src/custom_sam_peft/cli/_interactive.py`
- Test: `tests/cli/test_emitted_commands.py` (create)

**Verified emit sites in `_interactive.py`:** `_launch_command` (line ~249–250, uses
`_LAUNCH_VERB` → `custom-sam-peft {train|run|eval} --config {output}`); eval reuse
(line ~343–344) and baseline (line ~362); predict assembly (lines ~497–517,
including the now-removed `--merge-adapter` at ~505 and the `--device/--dtype/--seed`
note at ~515).

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_emitted_commands.py`:

```python
"""Emitted copy-paste commands use the positional config form (§6.5)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.cli._interactive import _launch_command


def test_launch_command_positional_train() -> None:
    assert _launch_command(Path("config.yaml"), "train") == "custom-sam-peft train config.yaml"


def test_launch_command_positional_run() -> None:
    assert _launch_command(Path("config.yaml"), "run") == "custom-sam-peft run config.yaml"
```

(For eval/predict assembly which build strings inline, assert via a small refactor or
leave them — see Step 3. The `_launch_command` test is the load-bearing one.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cli/test_emitted_commands.py -o "addopts=" -q`
Expected: FAIL — emits `--config config.yaml`, not the positional.

- [ ] **Step 3: Update the emit sites to the positional form**

- `_launch_command` (line ~250): change
  `f"custom-sam-peft {_LAUNCH_VERB[run_mode]} --config {output}"` →
  `f"custom-sam-peft {_LAUNCH_VERB[run_mode]} {output}"`. Note `_LAUNCH_VERB` maps
  `train`/`run`/`eval`; **eval is Tier B (not positional)** — eval keeps `--config`.
  So branch: for `train`/`run` use the positional; for `eval` keep `--config`:

  ```python
  def _launch_command(output: Path, run_mode: RunMode) -> str:
      verb = _LAUNCH_VERB[run_mode]
      if run_mode == "eval":
          return f"custom-sam-peft {verb} --config {output}"
      return f"custom-sam-peft {verb} {output}"
  ```

- eval reuse (line ~344) and baseline (line ~362): **keep `--config`** — eval is
  Tier B, not positional. No change. (Confirm this is consistent with §4.2 Tier B and
  §6.5.)
- predict assembly: remove the `--merge-adapter`/`--no-merge-adapter` line (~505) —
  predict no longer accepts it (Phase 2). Update the trailing note (~515) from
  `"--top-k, --device, --dtype, --batch-size, --seed stay at defaults"` to
  `"--top-k, --batch-size stay at defaults"` (device/dtype/seed are no longer flags).
  Also stop collecting `merge_adapter` interactively (line ~443–444 `ask_confirm`) —
  remove that prompt since the flag is gone; merge is derived from the adapter kind at
  predict time.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cli/test_emitted_commands.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Lint + py_compile**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run python -m py_compile src/custom_sam_peft/cli/_interactive.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/_interactive.py tests/cli/test_emitted_commands.py
git commit -m "feat(cli): emit positional config for train/run; drop predict merge-adapter from interactive"
```

---

### Task 3.4: Update README quickstart + command table (§6.6)

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Update the quickstart (lines ~58–67)**

Change the run/train examples to the positional form:

- `uv run csp run --config config.yaml` → `uv run csp run config.yaml`
- `uv run csp train --config config.yaml` → `uv run csp train config.yaml`
- `uv run csp train --config config.yaml --eval` → `uv run csp train config.yaml --eval`
- `uv run csp train --config config.yaml --eval --export` → `uv run csp train config.yaml --eval --export`
- `run --config cfg.yaml` is shorthand for `train --config cfg.yaml --eval --export.`
  → `run config.yaml` is shorthand for `train config.yaml --eval --export.`

- [ ] **Step 2: Update the command table (lines ~88–100)**

- `csp run --config CONFIG [...]` → `csp run CONFIG [--resume PATH] [-v]`
- `csp train --config CONFIG [...]` → `csp train CONFIG [--eval] [--export] [--override key=val]... [--name NAME] [--output-dir DIR] [--resume PATH] [--dry-run] [-v]`
- `csp eval --config CONFIG --checkpoint PATH [...]` → keep `--config` for baseline
  but show discovery: `csp eval --checkpoint PATH [--config CONFIG] [--split val\|test] [--export] [--output PATH] [--dry-run] [--interactive]`
- `csp export --checkpoint PATH [--merge] [--output PATH] [--config PATH]` →
  `csp export --checkpoint PATH --output DIR [--merge] [--config PATH]` (output now
  always required).
- `csp init [...]` → add `[-y]`.
- Bottom line: `run --config CONFIG is equivalent to train --config CONFIG --eval --export`
  → `run CONFIG is equivalent to train CONFIG --eval --export`.

Keep the `csp predict` table row updated to drop nothing visible (it already omits
the removed flags); leave it as-is.

> Do NOT add a pinned version or "what's new" block (project convention: README
> carries no hardcoded version). This task touches only the quickstart + table.

- [ ] **Step 3: Markdown-lint the README**

Run the project's markdownlint via the Python-bundled node (no system node on this
box):

```bash
cp .config/markdownlint-cli2.jsonc /tmp/x.markdownlint-cli2.jsonc
uv run --no-project --with nodejs-bin python -c "
from nodejs import node, npx
import os, sys
os.environ['PATH'] = os.path.dirname(node.path) + os.pathsep + os.environ['PATH']
sys.exit(npx.run(['--yes','markdownlint-cli2@0.14.0','--config','/tmp/x.markdownlint-cli2.jsonc', *sys.argv[1:]]).returncode)
" README.md
```

Expected: no findings (fix any MD violations introduced).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(cli): restore positional config form in README quickstart + command table"
```

---

### Task 3.5: Phase 3 regression + final gate

**Files:** none (verification only).

- [ ] **Step 1: Full CLI suite**

Run: `uv run pytest tests/cli/ -o "addopts=" -q`
Expected: green.

- [ ] **Step 2: Full CPU suite**

Run: `uv run pytest -q`
Expected: green (only pre-existing GPU skips). Investigate any new failure.

- [ ] **Step 3: Lint gate (both)**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: pass.

- [ ] **Step 4: Import smoke**

Run: `uv run python -c "import custom_sam_peft.cli.main"`
Expected: clean import. End of Phase 3.

---

## Self-Review (spec coverage)

- §3.1 goals 1–7: source-of-truth `_options.py` (1.1–1.3); validated enums for
  `--progress`/`eval --split`/`init` tiers (1.4–1.8, 2.4, 2.6); coverage gaps —
  `run --override` (2.1), `-v` on doctor/init/calibrate (2.5), `-y` (2.5), `--dry-run`
  (2.3), `--name`/`--output-dir` (2.2); `--config` tiering — positional (3.1),
  discovery (1.1/2.4), doctor unchanged; predict level-down (2.8, 2.9); consistency
  test (1.10, 2.10); audit deliverables = the spec's §5 tables (durable spec content,
  no code task needed).
- §4.2 Tier A/B/C: 3.1 (positional + hidden alias), 2.4 (eval discover-then-fallback).
  **predict is exempt from `discover_config` (decided — spec §4.2):** predict tolerates
  a missing config (resolves the base model from the adapter), and routing it through
  the raising `discover_config` would regress the bare-adapter-predict case. Phase 2
  keeps predict's existing `_resolve_config` precedence unchanged; the shared helper is
  consumed only by `eval`/`export`. No predict-discovery task — by design.
- §4.6 four removals + dtype coercion + merge derivation: 2.7, 2.8, 2.9.
- §5.4 `--output` help clarity: folded into 3.2 (export) and 2.5 help text; eval/init
  `--output` help-only tweaks are low-risk — add to the relevant tasks' help strings
  if time permits (non-blocking).
- §5.5 export rework: 3.2.
- §6.1–§6.6 verify items: 2.4 (§6.1 settled = val/test), 3.2 (§6.2), 2.7 (§6.3),
  2.8 (§6.4), 3.3 (§6.5), 3.4 (§6.6).

### Spec/code mismatches + gaps found (carry into review)

1. **Predict is exempt from `discover_config` (RESOLVED — spec §4.2/§4.3 updated).**
   The earlier spec draft grouped predict with eval/export under shared discovery.
   That is now corrected: predict tolerates a missing config (it resolves the base
   model from the adapter's `base_model_name_or_path` and never *requires* a full
   `TrainConfig`), and routing it through the *raising* `discover_config` would
   **regress** the bare-adapter-predict case (a checkpoint copied out of its run dir
   has no ancestor `config.yaml`). Decision: predict keeps `--config` as a direct
   optional override with no tree-walk; the shared helper is consumed only by
   `eval`/`export`. This plan correctly adds **no** predict-discovery task.
2. **`run_export` non-CLI callers (train/eval `--export`).** §5.5 says "drop the
   conditional ValueError." But `train_cmd`/`eval_cmd` call `run_export(cfg, ckpt)`
   with no `output`/`merge`, which today hits that very ValueError unless they pass
   output. Task 3.2 Step 3 resolves this by keeping safe **library** defaults while
   making only the **CLI** require `--output` — the spec's intent (CLI clarity) is
   met without breaking the pipeline callers. The implementer must verify the current
   train/eval `--export` behavior before choosing the default.
3. **`eval`/`init`/`calibrate` `--output` help-clarity (§5.4)** is folded into the
   tasks that already touch those files; no dedicated task. If the reviewer wants the
   exact help wording enforced, add one-line help edits to 2.4 (eval) and 2.6 (init).
