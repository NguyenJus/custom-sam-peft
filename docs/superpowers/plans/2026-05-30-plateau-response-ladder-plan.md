# Plateau-Response Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-30-plateau-response-ladder-design.md`](../specs/2026-05-30-plateau-response-ladder-design.md)
**Issue:** [#197](https://github.com/NguyenJus/custom-sam-peft/issues/197) — *Plateau-response ladder: LR decay before early stopping; early stop on by default*
**Research basis:** [`docs/research/2026-05-30-issue-197-plateau-lr-decay-early-stopping-lit-review.md`](../../research/2026-05-30-issue-197-plateau-lr-decay-early-stopping-lit-review.md)

**Goal:** Add a two-rung plateau-response ladder to training (reduce-on-plateau LR decay then metric early-stop, both driven by `mAP` at each `eval_every` boundary), introduce a new default `lr_schedule: plateau`, generalize end-of-run close-out to restore the best weights as the final adapter on every termination path, persist ladder state across `--resume` (fixing a best-clobber bug), and add a `run --finalize` entry that productionizes a paused run with no training.

**Architecture:** A trainer-owned `LadderState` (in a new `train/ladder.py`) holds two counters fed the same `mAP` at each successful eval. Rung 1 reuses `torch.optim.lr_scheduler.ReduceLROnPlateau`'s internal bad-eval counter (stepped per-eval in `plateau` mode; warmup stays a per-step ramp); rung 2 is an independent early-stop counter that resets only on genuine improvement. The ladder hooks into `_eval_epoch` right after `_maybe_save_best`. On stop, `run_epoch` raises a new `_EarlyStop` exception (mirroring `_TimeLimitReached`), caught in `fit()`, which then funnels both early stop and normal completion into a single reusable `close_out(...)` that restores `best/`, runs one full eval, and writes the adapter (+ optional merged + `metrics.json`) on the best weights. Ladder state persists in `training_state` and re-seeds on resume. A `run --finalize` flag rebuilds a model from a paused run's checkpoint and calls `close_out` with no training.

**Tech Stack:** Python 3.12, Pydantic v2 (`config/schema.py` strict `_Strict` models), frozen dataclasses (`train/ladder.py`, `eval/_artifacts.py`), PyTorch + SAM 3.1 (`train/loop.py`, `train/trainer.py`, `train/close_out.py`, `train/checkpoint.py`), Typer CLI (`cli/run_cmd.py`), pytest + pytest-cov (TDD, `--cov-fail-under=80` gate), ruff + `mypy --strict` + markdownlint-cli2 (CI gates).

---

## Phase structure & boundary contracts (read first)

```text
Phase 1  ──►  Phase 2  ──►  Phase 3
Ladder in     Resumable     Finalize-a-paused-run
the trainer   state +       entry (opens PR)
              best-as-final
              close-out
```

Three sequential phases, each an independently-reviewable feature block (spec §12). Every boundary states an **interface contract** so a later phase, executed in a fresh session with no memory of earlier phases' code, can build on it by reading only the contract + spec.

| Phase | Scope (one line) | Interface contract OUT |
|---|---|---|
| 1 | `lr_schedule: plateau` default + two config blocks; scheduler split (warmup → `ReduceLROnPlateau`); `LadderState`/`StopDecision`/`LadderEvents`/`_EarlyStop`; two counters; val-fallback; eval-tick wiring; LR-cut/stop telemetry | `_EarlyStop(step, epoch, reason)` raised from `run_epoch`; `LadderState` with `observe`/`state_dict`/`load_state_dict`/`best`; `LadderEvents` accumulator; a stop signal `fit()` honors, distinct from the time-limit pause |
| 2 | Persist/restore `ladder` + `best_metric_value` + `scheduler_kind`; clobber-bug fix; extract `close_out`; wire into early-stop + normal completion; best-as-final in `EvalArtifacts`/`metrics.json`/bundle/`summary.md`; single eval surfaced as `per_example_iou`; orchestrator drops its eval + export-merge phases | `close_out(run_dir, model, cfg, *, evaluator_val_ds, oom_state, final_step, final_epoch, ladder_events=None) -> EvalArtifacts` whose single eval populates `EvalArtifacts.per_example_iou` and whose returned `EvalArtifacts` also carries `ladder_events`; `checkpoint_path == run_dir/adapter` holds the best weights |
| 3 | `run --finalize` flag + `_finalize` helper: rebuild model from a run's checkpoint, `close_out`, no training, validation | (terminal phase — opens the PR) |

### Phase 1 → Phase 2 contract (detail)

```python
# src/custom_sam_peft/train/ladder.py
@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str          # "early_stop: N evals without mAP improvement (>= stop_patience)"
    triggering_step: int
    triggering_map: float

@dataclass(frozen=True)
class LrCut:
    step: int
    old_lr: float
    new_lr: float
    triggering_map: float

@dataclass(frozen=True)
class LadderEvents:
    cuts: tuple[LrCut, ...] = ()
    stop_reason: str | None = None

class LadderState:
    best: float                      # best mAP seen by the ladder
    evals_without_improvement: int   # rung-2 counter
    def observe(self, mAP: float | None, step: int,
                scheduler: Any, cfg: TrainConfig) -> StopDecision: ...
    def state_dict(self) -> dict[str, Any]: ...        # {best, evals_without_improvement}
    def load_state_dict(self, d: dict[str, Any]) -> None: ...

# src/custom_sam_peft/train/loop.py
class _EarlyStop(Exception):
    def __init__(self, step: int, epoch: int, reason: str) -> None: ...
    # attributes: .step, .epoch, .reason
```

- `_EarlyStop` is raised from `run_epoch` (via a new `should_stop_early` callback param) immediately after an `on_eval(step)` call when the trainer's predicate reports a stop. `fit()` catches it around the epoch loop alongside `_TimeLimitReached`.
- `LadderState.observe` performs the improvement test (`mAP > best + cfg.train.early_stop.min_delta`, strict `>`), updates rung-2, steps the plateau scheduler (rung 1, plateau mode only), records any LR cut, and returns the `StopDecision`. It no-ops (returns `should_stop=False`) when `mAP is None`.
- `_maybe_save_best` (`trainer.py`) is **unchanged** (still strict `>`, no `min_delta`); the ladder reads the same eval but owns its own `best` baseline.
- Phase 2 reads the `ReduceLROnPlateau`'s own `state_dict()` (carrying `best`/`num_bad_epochs`/`cooldown_counter` — the rung-1 counter) via the existing `scheduler` slot in `training_state`; only `LadderState.state_dict()` (rung-2 + `best`) is added explicitly.

### Phase 2 → Phase 3 contract (detail)

```python
# src/custom_sam_peft/train/close_out.py
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
    Falls back to the current (last-step) in-memory weights when no best/ exists."""
```

- `close_out` writes `run_dir/adapter` (best weights, or last-step in the fallback), optional `run_dir/merged` (when `cfg.export.merge`), and `run_dir/metrics.json` (carrying a `"final_weights": "best" | "last_step"` field and the ladder events). It runs the full eval **exactly once** with `return_per_example_iou=True`.
- It returns an `EvalArtifacts` whose `checkpoint_path == run_dir/adapter`, `final_metrics` is the **best** eval (or `None` no-val), a new optional field `per_example_iou: list[float] | None` carries the bundle's IoU data (or `None` no-val), and a new optional field `ladder_events: LadderEvents | None` carries the accumulated ladder telemetry (mirroring `oom_events`). Phase 3's `_finalize` consumes this verbatim to build the bundle with no second eval; callers read `artifacts.ladder_events` directly.
- `close_out` does **not** assemble or write the bundle — that stays the caller's job.

---

## File structure

### New files

- `src/custom_sam_peft/train/ladder.py` (Phase 1) — `StopDecision`, `LrCut`, `LadderEvents`, `LadderState`. Frozen dataclasses + one mutable counter class. No I/O.
- `src/custom_sam_peft/train/_scheduler.py` (Phase 1) — `PlateauOrLambda` type alias + `step_per_train_step(scheduler, *, global_step, base_lr, warmup_steps, mode)` mode-aware per-step helper.
- `src/custom_sam_peft/train/close_out.py` (Phase 2) — `close_out(...)`.
- `tests/train/test_ladder.py` (Phase 1)
- `tests/train/test_plateau_val_fallback.py` (Phase 1)
- `tests/train/test_plateau_scheduler.py` (Phase 1)
- `tests/train/test_ladder_resume.py` (Phase 2)
- `tests/train/test_best_clobber_regression.py` (Phase 2)
- `tests/train/test_close_out.py` (Phase 2)
- `tests/train/test_early_stop_integration.py` (Phase 2)
- `tests/cli/test_finalize.py` (Phase 3)

### Modified files

- `src/custom_sam_peft/config/schema.py` (Phase 1) — `LRSchedule += "plateau"`; `lr_schedule` default → `plateau`; `LrDecayOnPlateauConfig` + `EarlyStopConfig` models; mount both on `TrainHyperparams`.
- `src/custom_sam_peft/train/loop.py` (Phase 1) — `_EarlyStop`; `should_stop_early` callback param on `run_epoch`; `param_groups[0]["lr"]` LR read for logging; call the per-step scheduler helper from `train_step`.
- `src/custom_sam_peft/train/trainer.py` (Phase 1 + Phase 2) — `_build_scheduler` branch (plateau → `ReduceLROnPlateau`); val-fallback; ladder construction; eval-tick in `_eval_epoch`; catch `_EarlyStop`; clobber-bug re-seed (Phase 2); call `close_out` (Phase 2).
- `src/custom_sam_peft/train/checkpoint.py` (Phase 2) — `ladder` + `best_metric_value` + `scheduler_kind` in payload; `ResumeState` fields; new `save_full_state` args.
- `src/custom_sam_peft/eval/_artifacts.py` (Phase 2) — `per_example_iou` + `final_weights` + `ladder_events` optional fields on `EvalArtifacts`.
- `src/custom_sam_peft/runs/bundle.py` (Phase 2) — `BundleContext.ladder_events`; best-adapter + ladder-event summary lines.
- `src/custom_sam_peft/cli/run_cmd.py` (Phase 2 + Phase 3) — orchestrator drops eval + export-merge phases (Phase 2); `--finalize` flag + `_finalize` helper (Phase 3).
- `tests/integration/test_trainer_evaluator_seam.py` (Phase 2) — one new assertion (`per_example_iou is None` on the no-`return` path stays seam-safe; `final_metrics` semantics).
- `docs/config-schema.md` (Phase 2/3) — `lr_schedule` row, `lr_decay_on_plateau` + `early_stop` sub-blocks, `--finalize` CLI note.
- `docs/defaults-provenance.md` (Phase 2/3) — `lr_schedule` row + six new knob rows + `config_full` cross-link.

---

## Code-aware notes & verified hazards (read before implementing)

Symbols are authoritative; line numbers verified against the worktree at planning time and may drift.

1. **`LRSchedule` lives at `config/schema.py:96`**: `LRSchedule = Literal["constant", "cosine", "linear"]`. The field is `TrainHyperparams.lr_schedule` at `schema.py:526` (default `"cosine"`). `field_validator`, `PositiveInt`, `PositiveFloat` are already imported (`schema.py:21-31`). The two new config blocks mount in the `# --- advanced ---` section, after `time_limit` (`schema.py:547-567`) and before `loss` (`schema.py:569`).

2. **The demoted `early_stop_p_threshold` seam is already gone (A1).** A tree grep finds it only in `CHANGELOG.md` and historical specs/plans — never as a settable attribute on any model. `tests/unit/test_box_hint_schedule.py` already regression-asserts `not hasattr(s, "early_stop_p_threshold")`. Phase 1's "confirm absence" task is a verification no-op (re-run that grep + that test); add **no** new field.

3. **`_build_scheduler` (`trainer.py:69-86`) returns a `LambdaLR`** built from a per-step `lr_lambda`. The plateau branch must return a `ReduceLROnPlateau`, which is **not** an `LRScheduler` subclass and has **no** `get_last_lr()`. Widen the return type to a `PlateauOrLambda = LRScheduler | ReduceLROnPlateau` alias (in `train/_scheduler.py`). The signature gains an explicit `effective_schedule: str` parameter so the val-fallback (§6.5) governs construction rather than re-reading `cfg.train.lr_schedule`.

4. **`scheduler.step()` is called unconditionally in `train_step` at `loop.py:389`** (inside the grad-accum gate). This becomes mode-aware via `step_per_train_step(...)`: non-plateau → `scheduler.step()` (unchanged); plateau → warmup writes `param_groups` LR for `global_step < warmup_steps`, else no-op (the plateau scheduler is stepped only at evals). Note `train_step` does **not** currently receive `warmup_steps`/`base_lr`/`mode` as separate args — they all live on `cfg.train` (`cfg.train.warmup_steps`, `cfg.train.learning_rate`, `cfg.train.lr_schedule`), so the helper reads them from `cfg` passed through, OR `train_step` passes them explicitly. **Prefer reading from `cfg`** to avoid widening `train_step`'s signature. The plateau scheduler step argument `global_step` here is the step **after** increment (`loop.py:506` increments before the window update); pass `global_step` consistently.

5. **LR logging reads `scheduler.get_last_lr()[0]` at `loop.py:507`.** `ReduceLROnPlateau` has no `get_last_lr`. Change this **one** call site to `float(optimizer.param_groups[0]["lr"])` for **all** modes — a `LambdaLR` keeps `param_groups` LR in sync after `step()`, so the read is value-preserving for non-plateau modes (asserted by a regression test in §14.3). `run_epoch` already has `optimizer` in scope.

6. **The eval-tick must sit inside `_eval_epoch`'s `try`, after `_maybe_save_best` (`trainer.py:316`).** `_eval_epoch` swallows eval failures (`trainer.py:317-328`) and returns **before** `_maybe_save_best` on a failed/OOM eval. Placing `ladder.observe(...)` after `_maybe_save_best` inside the same `try` guarantees a skipped eval ticks **neither** counter (spec §10). `ladder.observe` must also no-op when `report.overall.get("mAP") is None` (mirrors `_maybe_save_best`'s early-return).

7. **`fit()` already catches `_TimeLimitReached` (`trainer.py:577-579`) in a nested `try` around the epoch loop.** Add an `except _EarlyStop as e: early = e` arm to the **same** inner `try`. The post-loop branch becomes: `stop is not None` → `_time_limited_artifacts` (unchanged); else (early stop **or** normal completion) → `close_out`. `close_out` **replaces** the inline finalize block at `trainer.py:581-621`.

8. **`save_full_state` (`checkpoint.py:147-175`) already round-trips `scheduler.state_dict()`** generically (`checkpoint.py:164`), so a `ReduceLROnPlateau`'s rung-1 state (`best`/`num_bad_epochs`/`cooldown_counter`) persists **for free** once the scheduler is a `ReduceLROnPlateau`. Phase 2 adds three **additive optional** keys to the payload (`ladder`, `best_metric_value`, `scheduler_kind`); `_FORMAT_VERSION` stays `1`. `load_full_state` reads them with `.get(...)` so pre-#197 checkpoints load (missing keys → `None`).

9. **`resume_run_dir = resume_from.parent.parent` is the run dir** (`runner.py:91`); `checkpoints/step_N/` → run_dir. The clobber-fix reads `resume_run_dir / "best" / "best.json"`, whose shape is `{"metric": "mAP", "value": <float>, "global_step": <int>}` (`trainer.py:344-353`).

10. **The `run` orchestrator (`run_cmd.py:72-186`) runs train → eval → export-merge → bundle as separate phases.** With Phase 2's single-eval `close_out`, the train phase already does the export-merge and the one full eval; the orchestrator **drops** its own eval phase (`run_cmd.py:114-138`) and export-merge phase (`run_cmd.py:142-157`) for the normal path, building the bundle directly from `train_result.final_metrics` + `train_result.per_example_iou`, with `checkpoint_path = run_dir/adapter`. The time-limit short-circuit (`run_cmd.py:91-97`) is unchanged.

11. **`load_sam31` signature (`models/sam3.py:624`)** takes `cfg.model, channels=..., channel_semantics=...` (see `run_cmd.py:106-108`, `runner.py:114-116`). `_finalize` reuses this verbatim. `load_adapter`/`save_merged` come from `train/checkpoint.py`. `load_val_source(run_dir)` (`data/val_source.py:133`) returns a `ValSource | None` with a `.mode`; `_build_val_dataset(cfg, vs)` (`run_cmd.py:59`) builds the eval dataset.

12. **`run_eval(..., return_per_example_iou=True)` returns `(MetricsReport, list[float])`** (`eval/runner.py:60-71`); the underlying `Evaluator.evaluate(model, ds, return_per_example_iou=True)` (`evaluator.py:306`) is what `close_out` calls directly (it already has a live model). The auto-batch-cap logic to mirror is `trainer.py:588-596`.

---

## Verification gates (every phase)

Run from the repo root. The repo type-checks **strict** (`pyproject.toml:106 strict = true`) and CI runs **both** ruff lint and ruff format-check (separate gates — format-check is a common miss):

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/custom_sam_peft
uv run pytest                        # global addopts carries --cov-fail-under=80
```

During fast iteration on a **subset**, the global `--cov-fail-under=80` in `addopts` fails an under-80 partial run. Bypass it with `-o "addopts="` (clears all addopts incl. the cov gate; `--no-cov` does **not** work here — MEMORY: pytest subset coverage gate):

```bash
uv run pytest -o "addopts=" tests/train/test_ladder.py -q   # example subset run
```

The **end-of-phase** verification runs the full gate (no `-o "addopts="`). All tests are **CPU-only**.

### Eager-import caveat

`custom_sam_peft/__init__.py` eagerly imports the train chain, so removing or renaming a symbol can un-import the whole package mid-phase. After any symbol-removal/rename task (none are planned, but the close-out extraction in Phase 2 moves the inline finalize logic), verify with `uv run ruff check` / `uv run python -m py_compile src/custom_sam_peft/train/trainer.py` and defer behavioral gating to the phase-end full suite.

### Markdown lint gate (Phase 2/3 docs + this plan)

Before committing any tracked `.md`, run CI's exact linter (`markdownlint-cli2@0.14.0`, config `.config/markdownlint-cli2.jsonc`, which disables only MD013/MD018/MD029). This box has no system node — use the Python-bundled Node path (MEMORY: markdown-lint gate):

```bash
uv run --no-project --with nodejs-bin python -c "
from nodejs import node, npx
import os, sys
os.environ['PATH'] = os.path.dirname(node.path) + os.pathsep + os.environ['PATH']
sys.exit(npx.run(['--yes','markdownlint-cli2@0.14.0','--config','.config/markdownlint-cli2.jsonc', *sys.argv[1:]]).returncode)
" docs/config-schema.md docs/defaults-provenance.md docs/superpowers/plans/2026-05-30-plateau-response-ladder-plan.md docs/superpowers/specs/2026-05-30-plateau-response-ladder-design.md
```

Expected: clean exit (0).

---

# PHASE 1 — Ladder in the trainer

**One coherent unit (spec §12):** the config blocks + `lr_schedule: plateau` default, the scheduler split (warmup → `ReduceLROnPlateau`, per-eval vs per-step stepping, `param_groups` LR read for logging), the `LadderState`/`StopDecision`/`LadderEvents`/`_EarlyStop` machinery, the two counters fed one shared improvement test, the val-fallback to cosine, the eval-tick wiring in `_eval_epoch`, and LR-cut/stop telemetry. No persistence, no close-out — those are Phase 2. Ends green at the full gate, then commits.

**Phase boundary — interface contract OUT:** restated at the top ("Phase 1 → Phase 2 contract"). Phase 2 imports `from custom_sam_peft.train.ladder import LadderState, LadderEvents, LrCut, StopDecision` and `from custom_sam_peft.train.loop import _EarlyStop`, and reads `cfg.train.lr_decay_on_plateau` / `cfg.train.early_stop`.

## Task 1: Config blocks + `lr_schedule: plateau` default (`config/schema.py`)

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` — `LRSchedule` union (line 96); `LrDecayOnPlateauConfig` + `EarlyStopConfig` models; mount on `TrainHyperparams` (advanced section, after `time_limit`); flip `lr_schedule` default (line 526).
- Test: `tests/config/test_plateau_config.py`

- [ ] **Step 1: Write the failing tests** (spec §5.1–§5.3, §5.6)

Create `tests/config/test_plateau_config.py`:

```python
"""Schema tests for the plateau ladder config blocks (spec §5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import (
    EarlyStopConfig,
    LrDecayOnPlateauConfig,
    TrainHyperparams,
)


def test_lr_schedule_default_is_plateau() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.lr_schedule == "plateau"


def test_lr_schedule_accepts_legacy_modes() -> None:
    for mode in ("constant", "cosine", "linear", "plateau"):
        assert TrainHyperparams(epochs=1, lr_schedule=mode).lr_schedule == mode


def test_lr_schedule_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, lr_schedule="poly")


def test_lr_decay_on_plateau_defaults() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.lr_decay_on_plateau.patience == 5
    assert hp.lr_decay_on_plateau.factor == 0.1
    assert hp.lr_decay_on_plateau.min_lr == 1.0e-6


def test_lr_decay_factor_must_shrink() -> None:
    with pytest.raises(ValidationError):
        LrDecayOnPlateauConfig(factor=1.0)
    with pytest.raises(ValidationError):
        LrDecayOnPlateauConfig(factor=1.5)


def test_early_stop_defaults() -> None:
    hp = TrainHyperparams(epochs=1)
    assert hp.early_stop.enabled is True
    assert hp.early_stop.monitor == "mAP"
    assert hp.early_stop.min_delta == 0.001
    assert hp.early_stop.stop_patience == 10


def test_early_stop_monitor_is_single_valued() -> None:
    with pytest.raises(ValidationError):
        EarlyStopConfig(monitor="DSC")


def test_blocks_reject_extra_keys() -> None:
    with pytest.raises(ValidationError):
        LrDecayOnPlateauConfig(bogus=1)
    with pytest.raises(ValidationError):
        EarlyStopConfig(bogus=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/config/test_plateau_config.py -q`
Expected: FAIL — `LrDecayOnPlateauConfig`/`EarlyStopConfig` not importable; `lr_schedule` default is still `"cosine"`.

- [ ] **Step 3: Extend `LRSchedule` and add the two config models**

In `src/custom_sam_peft/config/schema.py`, change line 96:

```python
LRSchedule = Literal["constant", "cosine", "linear", "plateau"]
```

Add the two models just **above** `class TrainHyperparams(_Strict)` (line 520):

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

    @field_validator("factor")
    @classmethod
    def _factor_must_shrink(cls, v: float) -> float:
        if v >= 1.0:
            raise ValueError(f"lr_decay_on_plateau.factor must be < 1.0 (got {v})")
        return v


class EarlyStopConfig(_Strict):
    """Rung-2 early-stop knobs. monitor/min_delta are the SHARED improvement
    definition consumed by rung 1 too: when early_stop.enabled is false while
    lr_schedule is plateau, these two fields still configure the rung-1 LR-decay
    threshold (they feed ReduceLROnPlateau's threshold and the monitored metric).
    Documented wart, spec §5.4 / docs/config-schema.md."""

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

- [ ] **Step 4: Flip the default and mount the blocks on `TrainHyperparams`**

Change line 526:

```python
    lr_schedule: LRSchedule = "plateau"
    # cite: ReduceLROnPlateau (PyTorch/Keras) + the canonical early-stop pairing
    #       (research §2–§4); # tbd: #197 — the cosine→plateau default flip.
```

In the `# --- advanced ---` section, **after** the `time_limit` field/validator (after `schema.py:567`) and **before** `loss` (line 569), add:

```python
    lr_decay_on_plateau: LrDecayOnPlateauConfig = Field(default_factory=LrDecayOnPlateauConfig)
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/config/test_plateau_config.py -q`
Expected: PASS.

- [ ] **Step 6: Run the existing schema suite (guard the default flip)**

Run: `uv run pytest -o "addopts=" tests/config -q`
Expected: PASS. If any existing test asserts `lr_schedule == "cosine"` as the default, update it to `"plateau"` (the flip is intentional, spec §5.1). Note: `extra="forbid"` means existing configs that did not set these blocks still validate (the new fields have defaults).

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/config/test_plateau_config.py
git commit -m "feat(config): plateau lr_schedule default + lr_decay_on_plateau/early_stop blocks (#197)"
```

## Task 2: Confirm the demoted `early_stop_p_threshold` seam is absent (A1 — no-op)

**Files:** none (verification only; spec §5.5, §17 A1).

- [ ] **Step 1: Grep the source tree**

Run: `grep -rn "early_stop_p_threshold" src/ tests/`
Expected: **no** matches under `src/` as a settable attribute. The only hits are the existing regression assertion in `tests/unit/test_box_hint_schedule.py` (`not hasattr(s, "early_stop_p_threshold")`). Confirm there is nothing to remove.

- [ ] **Step 2: Run the box-hint regression test**

Run: `uv run pytest -o "addopts=" tests/unit/test_box_hint_schedule.py -q`
Expected: PASS (already asserts absence). No code change; no commit.

## Task 3: `LadderState` + `StopDecision` + `LadderEvents` + `LrCut` (`train/ladder.py`)

**Files:**

- Create: `src/custom_sam_peft/train/ladder.py`
- Test: `tests/train/test_ladder.py`

- [ ] **Step 1: Write the failing counter/staircase tests** (spec §6.3, §6.4, §14.1)

Create `tests/train/test_ladder.py`:

```python
"""Ladder counter + staircase tests (spec §6.3, §14.1). CPU-only, no model."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import (
    EarlyStopConfig,
    LrDecayOnPlateauConfig,
    TrainHyperparams,
)
from custom_sam_peft.train.ladder import LadderState


def _cfg(**train_kw: object):
    """A minimal object exposing cfg.train.early_stop / cfg.train.lr_decay_on_plateau."""

    class _Cfg:
        train = TrainHyperparams(epochs=1, **train_kw)  # type: ignore[arg-type]

    return _Cfg()


def _plateau_scheduler(lr: float = 1e-4, *, patience: int = 5, factor: float = 0.1,
                       min_lr: float = 1e-6, min_delta: float = 0.001):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=factor, patience=patience,
        threshold=min_delta, threshold_mode="abs", min_lr=min_lr,
    ), opt


def test_improvement_resets_both_counters() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler()
    ladder = LadderState()
    for i, m in enumerate([0.5, 0.6, 0.7], start=1):
        d = ladder.observe(m, step=i, scheduler=sched, cfg=cfg)
        assert not d.should_stop
    assert ladder.evals_without_improvement == 0
    assert opt.param_groups[0]["lr"] == 1e-4  # no cut


def test_rung1_staircase_one_cut_at_patience() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler(patience=5)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)  # establishes best
    for i in range(2, 7):  # five more non-improving evals (steps 2..6)
        ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
    assert opt.param_groups[0]["lr"] == 1e-5  # one ×0.1 cut


def test_rung2_independent_of_cut_stops_at_stop_patience() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10))
    sched, opt = _plateau_scheduler(patience=5)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)  # best
    stop = None
    for i in range(2, 12):  # ten non-improving evals → stop at the 10th
        d = ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
        if d.should_stop:
            stop = d
            break
    assert stop is not None
    assert stop.triggering_step == 11  # 10 non-improving evals after the first


def test_one_cut_before_stop_with_shipped_defaults() -> None:
    cfg = _cfg()  # patience=5, stop_patience=10, min_delta=0.001
    sched, opt = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    cut_lr = None
    stopped_at = None
    for i in range(2, 12):
        d = ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
        if opt.param_groups[0]["lr"] == 1e-5 and cut_lr is None:
            cut_lr = i
        if d.should_stop:
            stopped_at = i
            break
    assert cut_lr == 6        # cut after 5 non-improving evals
    assert stopped_at == 11   # stop after 10
    assert opt.param_groups[0]["lr"] == 1e-5  # exactly one cut


def test_min_lr_floor() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler(patience=1, min_lr=1e-6)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    for i in range(2, 30):
        ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
    assert opt.param_groups[0]["lr"] >= 1e-6


def test_min_delta_boundary_is_strict() -> None:
    cfg = _cfg()  # min_delta=0.001
    sched, _ = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.500, step=1, scheduler=sched, cfg=cfg)
    # Exactly +min_delta is NOT an improvement (strict >).
    d = ladder.observe(0.501, step=2, scheduler=sched, cfg=cfg)
    assert ladder.evals_without_improvement == 1
    # Just above is an improvement.
    ladder.observe(0.5021, step=3, scheduler=sched, cfg=cfg)
    assert ladder.evals_without_improvement == 0
    assert not d.should_stop


def test_shared_improvement_when_early_stop_disabled() -> None:
    """early_stop.enabled=False but plateau mode → rung-1 still cuts on min_delta/mAP (wart §5.4)."""
    cfg = _cfg(early_stop=EarlyStopConfig(enabled=False))
    sched, opt = _plateau_scheduler(patience=5)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    stopped = False
    for i in range(2, 30):
        d = ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
        stopped = stopped or d.should_stop
    assert opt.param_groups[0]["lr"] == 1e-6 or opt.param_groups[0]["lr"] < 1e-4  # cut(s) fired
    assert not stopped  # no early stop when disabled


def test_observe_none_map_noops() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    before = ladder.evals_without_improvement
    d = ladder.observe(None, step=2, scheduler=sched, cfg=cfg)
    assert not d.should_stop
    assert ladder.evals_without_improvement == before
    assert opt.param_groups[0]["lr"] == 1e-4  # no cut on a None tick


def test_state_dict_round_trip() -> None:
    cfg = _cfg()
    sched, _ = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    ladder.observe(0.5, step=2, scheduler=sched, cfg=cfg)
    d = ladder.state_dict()
    restored = LadderState()
    restored.load_state_dict(d)
    assert restored.best == ladder.best
    assert restored.evals_without_improvement == ladder.evals_without_improvement
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder.py -q`
Expected: FAIL — `custom_sam_peft.train.ladder` does not exist.

- [ ] **Step 3: Implement `train/ladder.py`**

Create `src/custom_sam_peft/train/ladder.py`:

```python
"""Plateau-response ladder state (spec §6.3, §6.4).

Two counters fed the same mAP at each successful eval, sharing one improvement
test (mAP > best + min_delta, strict). Rung 1 reuses ReduceLROnPlateau's internal
bad-eval counter (stepped here in plateau mode); rung 2 is an independent
early-stop counter that resets ONLY on genuine improvement, never on an LR cut.

Note (spec §6.3 A2): _maybe_save_best (trainer.py) saves on strict `>` (no
min_delta); this ladder counts improvement on `> best + min_delta`. A tiny
improvement can save a new best yet still count as non-improvement for patience.
Intentional: always save a strictly-better checkpoint; only reset patience on a
meaningfully-better one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from custom_sam_peft.config.schema import TrainConfig


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str
    triggering_step: int
    triggering_map: float


@dataclass(frozen=True)
class LrCut:
    step: int
    old_lr: float
    new_lr: float
    triggering_map: float


@dataclass(frozen=True)
class LadderEvents:
    """Accumulated ladder telemetry, threaded into close_out (Phase 2)."""

    cuts: tuple[LrCut, ...] = ()
    stop_reason: str | None = None


@dataclass
class LadderState:
    best: float = float("-inf")  # best mAP seen by the ladder
    evals_without_improvement: int = 0  # rung-2 counter
    # rung-1 counter lives inside the ReduceLROnPlateau (not duplicated here)
    last_cut: LrCut | None = field(default=None, compare=False)

    def observe(
        self,
        mAP: float | None,
        step: int,
        scheduler: Any,
        cfg: TrainConfig,
    ) -> StopDecision:
        """Tick both rungs on one successful eval. A None mAP is a no-op tick."""
        self.last_cut = None
        if mAP is None:
            return StopDecision(False, "", step, float("nan"))

        min_delta = float(cfg.train.early_stop.min_delta)
        improved = mAP > self.best + min_delta
        if improved:
            self.best = mAP
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1

        # Rung 1 (plateau mode only): step ReduceLROnPlateau, detect a cut by
        # comparing pre/post param_groups[0]["lr"].
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            opt = scheduler.optimizer
            old_lr = float(opt.param_groups[0]["lr"])
            scheduler.step(mAP)
            new_lr = float(opt.param_groups[0]["lr"])
            if new_lr < old_lr:
                self.last_cut = LrCut(step, old_lr, new_lr, mAP)

        # Rung 2: stop only when enabled and the counter reaches stop_patience.
        if (
            cfg.train.early_stop.enabled
            and self.evals_without_improvement >= cfg.train.early_stop.stop_patience
        ):
            reason = (
                f"early_stop: {self.evals_without_improvement} evals without mAP "
                f"improvement (>= {cfg.train.early_stop.stop_patience})"
            )
            return StopDecision(True, reason, step, mAP)
        return StopDecision(False, "", step, mAP)

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "evals_without_improvement": self.evals_without_improvement}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.best = float(d["best"])
        self.evals_without_improvement = int(d["evals_without_improvement"])
```

> Note: the `triggering_step` assertions in the tests expect the step passed to the **stop-triggering** `observe` call. Because `LadderState` is constructed once and `observe` is called per eval, the first `observe(0.5, step=1, ...)` sets `best` and counts as the baseline; the 10th non-improving eval is at `step=11`, where the counter reaches `stop_patience=10`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder.py -q`
Expected: PASS (all counter/staircase cases).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/ladder.py tests/train/test_ladder.py
git commit -m "feat(train): LadderState two-counter staircase + LadderEvents (#197)"
```

## Task 4: Per-step scheduler helper + plateau build branch + LR-logging change

**Files:**

- Create: `src/custom_sam_peft/train/_scheduler.py` — `PlateauOrLambda` alias + `step_per_train_step(...)`.
- Modify: `src/custom_sam_peft/train/trainer.py` — `_build_scheduler` gains an `effective_schedule` arg + plateau branch.
- Modify: `src/custom_sam_peft/train/loop.py` — `train_step` calls `step_per_train_step`; LR read via `param_groups[0]["lr"]`.
- Test: `tests/train/test_plateau_scheduler.py`

- [ ] **Step 1: Write the failing scheduler-mechanics tests** (spec §6.1, §6.2, §14.3)

Create `tests/train/test_plateau_scheduler.py`:

```python
"""Scheduler split: per-step warmup vs per-eval plateau cut (spec §6, §14.3)."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import TrainHyperparams
from custom_sam_peft.train._scheduler import step_per_train_step


def _opt(lr: float = 1e-4):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_warmup_ramp_then_hold_in_plateau_mode() -> None:
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    base_lr, warmup = 1e-4, 100
    # Mid-warmup step 49 → lr = base * 50/100.
    step_per_train_step(sched, global_step=49, base_lr=base_lr,
                        warmup_steps=warmup, mode="plateau")
    assert abs(opt.param_groups[0]["lr"] - base_lr * 50 / 100) < 1e-12
    # After warmup, the per-step helper holds the LR (no write, no plateau step).
    opt.param_groups[0]["lr"] = base_lr
    step_per_train_step(sched, global_step=150, base_lr=base_lr,
                        warmup_steps=warmup, mode="plateau")
    assert opt.param_groups[0]["lr"] == base_lr


def test_plateau_scheduler_not_stepped_per_train_step() -> None:
    """In plateau mode the per-step helper never calls ReduceLROnPlateau.step()."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    calls = {"n": 0}
    orig = sched.step

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    sched.step = spy  # type: ignore[method-assign]
    for s in range(0, 300):
        step_per_train_step(sched, global_step=s, base_lr=1e-4,
                            warmup_steps=100, mode="plateau")
    assert calls["n"] == 0  # plateau scheduler only ticks at evals


def test_lambda_lr_stepped_per_train_step_in_non_plateau_mode() -> None:
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 0.5)
    step_per_train_step(sched, global_step=0, base_lr=1e-4,
                        warmup_steps=0, mode="cosine")
    # LambdaLR.step() applied the 0.5 multiplier.
    assert abs(opt.param_groups[0]["lr"] - 0.5e-4) < 1e-12


def test_param_groups_lr_matches_get_last_lr_for_lambda() -> None:
    """Regression: the param_groups read equals get_last_lr() for LambdaLR (§14.3)."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0 - s * 0.01)
    for s in range(5):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - sched.get_last_lr()[0]) < 1e-12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/train/test_plateau_scheduler.py -q`
Expected: FAIL — `custom_sam_peft.train._scheduler` does not exist.

- [ ] **Step 3: Implement `train/_scheduler.py`**

Create `src/custom_sam_peft/train/_scheduler.py`:

```python
"""Mode-aware scheduler stepping (spec §6.2).

In plateau mode the warmup is a per-step linear ramp written directly to
param_groups; the ReduceLROnPlateau is stepped only at evals (via
LadderState.observe). In non-plateau modes the per-step LambdaLR is stepped
exactly as before. This keeps the branch in one place so run_epoch/train_step
stay otherwise unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

PlateauOrLambda = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau


def step_per_train_step(
    scheduler: Any,
    *,
    global_step: int,
    base_lr: float,
    warmup_steps: int,
    mode: str,
) -> None:
    """Advance the scheduler for ONE training step.

    - non-plateau modes: scheduler.step() (per-step LambdaLR), unchanged.
    - plateau mode: during warmup (global_step < warmup_steps) write
      param_groups LR = base_lr * (global_step + 1) / max(warmup_steps, 1);
      after warmup, no-op (the plateau scheduler is stepped only at evals).
    """
    if mode != "plateau":
        scheduler.step()
        return
    if global_step < warmup_steps:
        factor = (global_step + 1) / max(warmup_steps, 1)
        lr = base_lr * factor
        for group in scheduler.optimizer.param_groups:
            group["lr"] = lr
    # else: hold — ReduceLROnPlateau owns the LR from the first eval onward.
```

- [ ] **Step 4: Branch `_build_scheduler` on `effective_schedule` (plateau → `ReduceLROnPlateau`)**

In `src/custom_sam_peft/train/trainer.py`, add the import:

```python
from custom_sam_peft.train._scheduler import PlateauOrLambda
```

Change `_build_scheduler` (lines 69-86) to take an explicit `effective_schedule` and branch:

```python
def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_steps: int,
    effective_schedule: str,
) -> PlateauOrLambda:
    if effective_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.train.lr_decay_on_plateau.factor,
            patience=cfg.train.lr_decay_on_plateau.patience,
            threshold=cfg.train.early_stop.min_delta,
            threshold_mode="abs",  # absolute mAP units — matches the early-stop test
            min_lr=cfg.train.lr_decay_on_plateau.min_lr,
        )

    warmup = cfg.train.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        if effective_schedule == "constant":
            return 1.0
        if effective_schedule == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

> The `fit()` call site at `trainer.py:514` is updated in Task 7 (val-fallback) to pass `effective_schedule`. For now, to keep the tree importable mid-task, update the call site temporarily to `_build_scheduler(optimizer, cfg, total_steps, cfg.train.lr_schedule)`. Task 7 replaces `cfg.train.lr_schedule` with the fallback-resolved `effective_schedule`.

- [ ] **Step 5: Wire `train_step` to the per-step helper and change the LR read in `run_epoch`**

In `src/custom_sam_peft/train/loop.py`, add the import:

```python
from custom_sam_peft.train._scheduler import step_per_train_step
```

Replace the bare `scheduler.step()` at `loop.py:389` (inside the grad-accum gate) with:

```python
        step_per_train_step(
            scheduler,
            global_step=global_step,
            base_lr=cfg.train.learning_rate,
            warmup_steps=cfg.train.warmup_steps,
            mode=cfg.train.lr_schedule,
        )
```

> Note `global_step` here is the value `train_step` received as its parameter (pre-increment within `train_step`). This matches today's behavior: the `LambdaLR` is stepped once per optimizer step. The plateau warmup ramp uses the same `global_step`, giving an identical linear ramp over the first `warmup_steps` steps.

Change the LR read at `loop.py:507` from `float(scheduler.get_last_lr()[0])` to:

```python
        window.update(result, lr=float(optimizer.param_groups[0]["lr"]))
```

- [ ] **Step 6: Run the scheduler tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/train/test_plateau_scheduler.py -q`
Expected: PASS.

- [ ] **Step 7: Verify the package still imports (eager-import caveat)**

Run: `uv run python -m py_compile src/custom_sam_peft/train/trainer.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/_scheduler.py && uv run ruff check src/custom_sam_peft/train`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/custom_sam_peft/train/_scheduler.py src/custom_sam_peft/train/trainer.py src/custom_sam_peft/train/loop.py tests/train/test_plateau_scheduler.py
git commit -m "feat(train): plateau scheduler build + per-step helper + param_groups LR read (#197)"
```

## Task 5: `_EarlyStop` + `should_stop_early` callback in `run_epoch`

**Files:**

- Modify: `src/custom_sam_peft/train/loop.py` — `_EarlyStop` class; `should_stop_early` param on `run_epoch`; raise after `on_eval`.
- Modify: `src/custom_sam_peft/train/trainer.py` — pass `should_stop_early` through `_train_epoch` → `run_epoch`.
- Test: `tests/train/test_early_stop_signal.py`

- [ ] **Step 1: Write the failing signal test** (spec §4.2)

Create `tests/train/test_early_stop_signal.py`:

```python
"""run_epoch raises _EarlyStop after an eval when the predicate fires (spec §4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from custom_sam_peft.data.collate import collate_batch
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.loop import _EarlyStop, run_epoch
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def _loader(ds: _TinyDataset) -> list[dict[str, object]]:
    return [collate_batch([ds[i]]) for i in range(len(ds))]


def test_run_epoch_raises_early_stop_after_eval(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"eval_every": 1})})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)

    # Predicate returns a stop after the first eval.
    fired = {"n": 0}

    def should_stop_early() -> _EarlyStop | None:
        fired["n"] += 1
        return _EarlyStop(step=fired["n"], epoch=0, reason="test stop")

    with pytest.raises(_EarlyStop) as exc:
        run_epoch(
            wrapper, _loader(ds), opt, sched, NoopTracker(), cfg, run_dir,
            epoch=0, global_step=0, nan_streak=0, class_names=ds.class_names,
            on_checkpoint=lambda *a: None, on_eval=lambda *a: None,
            should_stop_early=should_stop_early,
        )
    assert exc.value.reason == "test stop"


def test_run_epoch_no_stop_when_predicate_returns_none(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"eval_every": 1})})
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run2"
    (run_dir / "checkpoints").mkdir(parents=True)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    # No predicate (None) → runs to the end of the epoch without raising.
    gs, _ = run_epoch(
        wrapper, _loader(ds), opt, sched, NoopTracker(), cfg, run_dir,
        epoch=0, global_step=0, nan_streak=0, class_names=ds.class_names,
        on_checkpoint=lambda *a: None, on_eval=lambda *a: None,
        should_stop_early=None,
    )
    assert gs == len(ds)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/train/test_early_stop_signal.py -q`
Expected: FAIL — `_EarlyStop` not importable; `run_epoch` has no `should_stop_early` param.

- [ ] **Step 3: Add `_EarlyStop` + the callback to `loop.py`**

In `src/custom_sam_peft/train/loop.py`, add the exception class near `_TimeLimitReached` (after `loop.py:60`):

```python
class _EarlyStop(Exception):
    """Internal signal: the plateau ladder requested an early stop. Graceful —
    proceeds to close-out (NOT a pause). Carries the stop point + reason. Never
    propagates past Trainer.fit(). Spec §4.2."""

    def __init__(self, step: int, epoch: int, reason: str) -> None:
        super().__init__(f"early stop at step {step} (epoch {epoch}): {reason}")
        self.step = step
        self.epoch = epoch
        self.reason = reason
```

Add `should_stop_early` to `run_epoch`'s signature (after `deadline`, line 475):

```python
    deadline: float | None = None,
    should_stop_early: Callable[[], "_EarlyStop | None"] | None = None,
) -> tuple[int, int]:
```

Immediately **after** the `on_eval(global_step)` call (after `loop.py:524`, before the deadline check), add:

```python
            if should_stop_early is not None:
                stop = should_stop_early()
                if stop is not None:
                    raise stop
```

Update `run_epoch`'s docstring to note the optional `should_stop_early` predicate, evaluated right after each `on_eval`, that raises `_EarlyStop` to unwind the epoch loop.

- [ ] **Step 4: Thread `should_stop_early` through `_train_epoch`**

In `src/custom_sam_peft/train/trainer.py`, add `should_stop_early` to `_train_epoch`'s signature (after `deadline`, line 260) and pass it to `run_epoch`:

```python
        oom_state: OomState | None = None,
        deadline: float | None = None,
        should_stop_early: Any = None,
    ) -> tuple[int, int]:
        """Run one training epoch; returns (global_step, nan_streak)."""
        return run_epoch(
            ...
            oom_state=oom_state,
            deadline=deadline,
            should_stop_early=should_stop_early,
        )
```

Also import `_EarlyStop` (extend the existing `from custom_sam_peft.train.loop import ...` line at `trainer.py:37`):

```python
from custom_sam_peft.train.loop import OomState, _EarlyStop, _TimeLimitReached, run_epoch
```

> The `fit()` wiring (constructing the ladder, defining the predicate, catching `_EarlyStop`) lands in Task 7. This task only plumbs the parameter through and exports the exception.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/train/test_early_stop_signal.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/train/test_early_stop_signal.py
git commit -m "feat(train): _EarlyStop signal + should_stop_early callback in run_epoch (#197)"
```

## Task 6: Val-fallback to cosine (`plateau` + no val → cosine + warning)

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` — compute `effective_schedule` in `fit()` before `_build_scheduler`.
- Test: `tests/train/test_plateau_val_fallback.py`

- [ ] **Step 1: Write the failing val-fallback test** (spec §6.5, §14.2)

Create `tests/train/test_plateau_val_fallback.py`:

```python
"""plateau + no val falls back to cosine with a warning (spec §6.5, §14.2)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_plateau_no_val_falls_back_to_cosine(tmp_path: Path, caplog) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr_schedule": "plateau"})})
    apply_lora(wrapper, cfg.peft)

    # val_ds=None → no plateau signal.
    trainer = Trainer(wrapper, ds, None, NoopTracker(), cfg)
    with caplog.at_level(logging.WARNING):
        result = trainer.fit(run_dir=tmp_path / "fallback-run")

    # Fell back to a per-step LambdaLR (cosine), not ReduceLROnPlateau.
    assert any("falling back to lr_schedule=cosine" in r.message for r in caplog.records)
    # The run completed normally (no early stop, no crash).
    assert result.run_dir.is_dir()
    # config.yaml still echoes the requested plateau.
    import yaml

    saved = yaml.safe_load((result.run_dir / "config.yaml").read_text())
    assert saved["train"]["lr_schedule"] == "plateau"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/train/test_plateau_val_fallback.py -q`
Expected: FAIL — `fit()` does not yet build the fallback; with `val_ds=None` the trainer currently builds a plateau scheduler that never ticks (or errors), and no warning is logged.

- [ ] **Step 3: Compute `effective_schedule` in `fit()` and pass it to `_build_scheduler`**

In `src/custom_sam_peft/train/trainer.py`, in `fit()`, **before** the `_build_scheduler` call (line 514), add:

```python
        effective_schedule = cfg.train.lr_schedule
        if cfg.train.lr_schedule == "plateau" and self.val_ds is None:
            _LOG.warning(
                "lr_schedule=plateau requires a validation set for the plateau signal; "
                "no val set provided — falling back to lr_schedule=cosine. "
                "Early stop is a no-op."
            )
            effective_schedule = "cosine"
```

Change the `_build_scheduler` call (line 514) to pass it:

```python
        scheduler = _build_scheduler(optimizer, cfg, total_steps, effective_schedule)
```

> Keep `effective_schedule` in a local — Phase 2 persists it as `scheduler_kind` (§8.1). The written `config.yaml` already reflects the requested `plateau` (it is dumped from `cfg`, unchanged).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/train/test_plateau_val_fallback.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py tests/train/test_plateau_val_fallback.py
git commit -m "feat(train): plateau→cosine val-fallback with warning (#197)"
```

## Task 7: Wire the ladder into `fit()` + `_eval_epoch` (eval-tick + catch `_EarlyStop`)

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` — construct `LadderState`/`LadderEvents`; eval-tick in `_eval_epoch` after `_maybe_save_best`; `should_stop_early` predicate; catch `_EarlyStop`; LR-cut/stop telemetry.
- Test: covered end-to-end by `tests/train/test_early_stop_integration.py` in Phase 2; a focused tick test is added here.

> **Implementation note.** This wires Phase-1 machinery into `fit()` **without** calling `close_out` (that is Phase 2). On `_EarlyStop` in Phase 1, `fit()` falls through to the existing inline finalize block (`trainer.py:581-621`) — which still exports last-step weights. Phase 2 replaces that block with `close_out`. The Phase-1 deliverable is: the ladder ticks, telemetry accrues, and the epoch loop stops; best-as-final is Phase 2. State the boundary clearly so the reviewer does not expect best-as-final yet.

- [ ] **Step 1: Write the focused eval-tick test** (spec §4.1, §10)

Create `tests/train/test_ladder_tick.py`:

```python
"""The ladder ticks only on a successful eval, after _maybe_save_best (spec §4.1, §10)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_failed_eval_does_not_tick(tmp_path: Path, monkeypatch) -> None:
    """An eval that raises advances NEITHER counter (the tick is inside the try, after save_best)."""
    import custom_sam_peft.eval.evaluator as ev

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(
        update={"lr_schedule": "plateau", "eval_every": 1, "epochs": 1})})
    apply_lora(wrapper, cfg.peft)

    def boom(self, model, dataset, **k):
        raise RuntimeError("eval OOM at batch_size=1")

    monkeypatch.setattr(ev.Evaluator, "evaluate", boom)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    trainer.fit(run_dir=tmp_path / "tick-run")
    # The ladder exists and was never advanced (all evals failed).
    assert trainer._ladder.evals_without_improvement == 0  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder_tick.py -q`
Expected: FAIL — `Trainer` has no `_ladder` attribute yet.

- [ ] **Step 3: Construct the ladder + events in `fit()`**

In `src/custom_sam_peft/train/trainer.py`, add the import (extend an existing block):

```python
from custom_sam_peft.train.ladder import LadderEvents, LadderState, LrCut, StopDecision
```

In `__init__`, after `self._best_metric_key` (line 182), add:

```python
        self._ladder: LadderState = LadderState()
        self._ladder_cuts: list[LrCut] = []
        self._early_stop: StopDecision | None = None
        self._scheduler: Any = None  # set in fit() after _build_scheduler
```

In `fit()`, after `scheduler = _build_scheduler(...)` (line 514), store it and reset ladder telemetry:

```python
        self._scheduler = scheduler
        self._ladder = LadderState()
        self._ladder_cuts = []
        self._early_stop = None
```

- [ ] **Step 4: Add the eval-tick to `_eval_epoch` (inside the `try`, after `_maybe_save_best`)**

In `_eval_epoch`, immediately after `self._maybe_save_best(report, step, run_dir)` (line 316) and **still inside** the `try`, add:

```python
            mAP = report.overall.get(self._best_metric_key)
            decision = self._ladder.observe(mAP, step, self._scheduler, self.cfg)
            if self._ladder.last_cut is not None:
                cut = self._ladder.last_cut
                self._ladder_cuts.append(cut)
                _LOG.info(
                    "LR cut ×%.3g → %.3g at eval step %d (mAP %.4f)",
                    cut.new_lr / cut.old_lr if cut.old_lr else 0.0,
                    cut.new_lr,
                    cut.step,
                    cut.triggering_map,
                )
            if decision.should_stop:
                self._early_stop = decision
                _LOG.info("early stop signalled: %s (mAP %.4f at step %d)",
                          decision.reason, decision.triggering_map, decision.triggering_step)
```

- [ ] **Step 5: Define the predicate + pass it through, and catch `_EarlyStop` in the epoch loop**

In `fit()`, define the predicate after the `on_eval` closure (after line 541):

```python
        def should_stop_early() -> _EarlyStop | None:
            d = self._early_stop
            if d is not None and d.should_stop:
                return _EarlyStop(d.triggering_step, _current_epoch[0], d.reason)
            return None
```

`_current_epoch` is a tiny mutable holder the epoch loop updates so the predicate knows the epoch. Just before the `try` (line 557), add `_current_epoch = [start_epoch]`, and inside the loop body set `_current_epoch[0] = epoch` as the first statement after `for epoch in range(...)`.

Pass `should_stop_early=should_stop_early` to the `self._train_epoch(...)` call (line 562) alongside `deadline=deadline`.

Add the `_EarlyStop` catch to the **inner** try (the one catching `_TimeLimitReached` at line 577):

```python
            except _TimeLimitReached as e:
                stop = e
                global_step = e.step
            except _EarlyStop as e:
                early = e
                global_step = e.step
```

Declare `early: _EarlyStop | None = None` next to `stop: _TimeLimitReached | None = None` (line 556).

> Phase 1 keeps the existing `if stop is None:` inline finalize block. Because `early` is not `stop`, an early-stopped run still falls into that block and exports last-step weights for now — **Phase 2 replaces this block with `close_out`** to make it best-as-final. Do not add `close_out` here.

- [ ] **Step 6: Run the focused tick test + the ladder/scheduler/fallback suite**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder_tick.py tests/train/test_ladder.py tests/train/test_plateau_scheduler.py tests/train/test_plateau_val_fallback.py tests/train/test_early_stop_signal.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py tests/train/test_ladder_tick.py
git commit -m "feat(train): wire ladder into _eval_epoch + catch _EarlyStop in fit() (#197)"
```

## Task 8: Phase 1 verification gate

**Files:** none (verification).

- [ ] **Step 1: Lint + format-check + type**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft`
Expected: clean. Watch for: the widened `_build_scheduler` return type (`PlateauOrLambda`); `should_stop_early`'s `Callable[[], _EarlyStop | None] | None` typing; `self._scheduler: Any` is acceptable but prefer `PlateauOrLambda` if mypy is satisfied. `ladder.observe`'s `scheduler: Any` is intentional (avoids a circular import).

- [ ] **Step 2: Full gated suite**

Run: `uv run pytest`
Expected: PASS including `--cov-fail-under=80`. The new ladder/scheduler/fallback tests cover the new code; existing trainer/seam tests are unaffected (they leave `lr_schedule` at the new default `plateau` but pass a val set, so the plateau scheduler builds and ticks once per epoch without stopping on a 1-epoch run).

> **Phase 1 → Phase 2 contract (restated for the next session):** `train/ladder.py` exports `StopDecision`, `LrCut`, `LadderEvents`, `LadderState` (with `observe(mAP, step, scheduler, cfg) -> StopDecision`, `state_dict()`, `load_state_dict(d)`, and a `best` field). `train/loop.py` exports `_EarlyStop(step, epoch, reason)` (attributes `.step`/`.epoch`/`.reason`), raised from `run_epoch` via the `should_stop_early` callback and caught in `fit()`. The trainer accumulates `self._ladder_cuts: list[LrCut]` and `self._early_stop: StopDecision | None`. `_maybe_save_best` is unchanged (strict `>`). `_build_scheduler(optimizer, cfg, total_steps, effective_schedule)` returns a `ReduceLROnPlateau` in plateau mode (else a `LambdaLR`); the effective schedule (post val-fallback) is the local `effective_schedule` in `fit()`. An early-stopped run currently falls through to the inline last-step finalize — Phase 2 replaces it with `close_out`.

---

# PHASE 2 — Resumable state + best-as-final close-out

**Phase boundary — interface contract IN (from Phase 1):** `_EarlyStop`, `LadderState`, `LadderEvents`, `LrCut`, `StopDecision`; the trainer's `self._ladder`, `self._ladder_cuts`, `self._early_stop`, `self._scheduler`, and the `effective_schedule` local in `fit()`.

**One coherent unit (spec §12):** persist/restore `ladder` + `best_metric_value` + `scheduler_kind` in `training_state`; the clobber-bug fix; extract `close_out`; wire it into early stop + normal completion (replacing the inline finalize block); reflect best-as-final in `EvalArtifacts` (+ `per_example_iou`, `final_weights`), `metrics.json` (`final_weights`, ladder events), and the bundle/`summary.md`; orchestrator drops its eval + export-merge phases. No CLI finalize entry — that is Phase 3.

**Phase boundary — interface contract OUT:** restated at the top ("Phase 2 → Phase 3 contract").

> **Implementation order note.** This phase is interdependent. Implement Task 9 (EvalArtifacts shape) first so the types exist, then Task 10 (checkpoint persistence), then Task 11 (clobber fix), then Task 12 (`close_out`), then Task 13 (wire into `fit()`), then Task 14 (bundle/summary), then Task 15 (orchestrator). Run focused subsets with `-o "addopts="`; the full gate runs at Task 16.

## Task 9: `EvalArtifacts.per_example_iou` + `final_weights` fields

**Files:**

- Modify: `src/custom_sam_peft/eval/_artifacts.py`
- Test: `tests/integration/test_trainer_evaluator_seam.py` (one new assertion)

- [ ] **Step 1: Add the new assertion to the seam test first**

In `tests/integration/test_trainer_evaluator_seam.py`, inside `test_trainer_fit_returns_eval_artifacts`, after the existing `assert result.time_limit_stop is None` line, add:

```python
    # Best-as-final close-out: new optional fields exist (default-safe).
    assert hasattr(result, "per_example_iou")
    assert result.final_weights in {"best", "last_step", None}
```

- [ ] **Step 2: Run the seam test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/integration/test_trainer_evaluator_seam.py::test_trainer_fit_returns_eval_artifacts -q`
Expected: FAIL — `EvalArtifacts` has no `per_example_iou`/`final_weights` attributes.

- [ ] **Step 3: Add the fields**

In `src/custom_sam_peft/eval/_artifacts.py`, add to `EvalArtifacts` (after `time_limit_stop`, line 50):

```python
    # close_out's single eval's per-example IoU (return_per_example_iou=True),
    # so the run/finalize bundle reuses it with no second eval. None no-val.
    per_example_iou: list[float] | None = field(default=None)
    # "best" | "last_step" | None — which weights the final adapter holds.
    # None on the normal pre-close_out / time-limit paths.
    final_weights: str | None = field(default=None)
    # Ladder telemetry threaded out of close_out (mirroring oom_events).
    # Uses a string annotation + TYPE_CHECKING guard to avoid an import cycle
    # (_artifacts.py importing from train/ladder.py). Exactly the pattern used
    # for BundleContext.ladder_events. Add to _artifacts.py:
    #   if TYPE_CHECKING: from custom_sam_peft.train.ladder import LadderEvents
    ladder_events: "LadderEvents | None" = field(default=None)
```

Import-safety: `eval/_artifacts.py` must not import from `train/ladder.py` at module level (circular). Add a `TYPE_CHECKING`-guarded import to `_artifacts.py` (if not already present):

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.train.ladder import LadderEvents
```

Use the **string** annotation `"LadderEvents | None"` (shown above) so the reference is not evaluated at runtime — exactly the pattern already used for `BundleContext.ladder_events` (line 2133: `ladder_events: "LadderEvents | None" = None`).

- [ ] **Step 3b: Add assertions for the new `ladder_events` field to the seam test**

In `tests/integration/test_trainer_evaluator_seam.py`, inside `test_trainer_fit_returns_eval_artifacts`, extend the assertions added in Step 1 to also cover `ladder_events`:

```python
    # Best-as-final close-out: new optional fields exist (default-safe).
    assert hasattr(result, "per_example_iou")
    assert result.final_weights in {"best", "last_step", None}
    # ladder_events field defaults to None (no early stop in this 1-epoch run).
    assert result.ladder_events is None
```

Also add a standalone unit test for `EvalArtifacts` field defaults and round-trip in `tests/integration/test_trainer_evaluator_seam.py` (or a dedicated `tests/eval/test_artifacts.py`):

```python
def test_eval_artifacts_ladder_events_field() -> None:
    """EvalArtifacts.ladder_events defaults to None and round-trips a passed value."""
    from custom_sam_peft.eval._artifacts import EvalArtifacts
    from custom_sam_peft.train.ladder import LadderEvents, LrCut

    art = EvalArtifacts(
        checkpoint_path=Path("/tmp/adapter"),
        peft_method="lora",
        run_dir=Path("/tmp"),
        final_metrics=None,
    )
    assert art.ladder_events is None  # default

    events = LadderEvents(cuts=(LrCut(3, 1e-4, 1e-5, 0.5),), stop_reason="early_stop: 10 evals")
    art2 = EvalArtifacts(
        checkpoint_path=Path("/tmp/adapter"),
        peft_method="lora",
        run_dir=Path("/tmp"),
        final_metrics=None,
        ladder_events=events,
    )
    assert art2.ladder_events == events
```

- [ ] **Step 4: Run the seam test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/integration/test_trainer_evaluator_seam.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/_artifacts.py tests/integration/test_trainer_evaluator_seam.py
git commit -m "feat(eval): EvalArtifacts.per_example_iou + final_weights optional fields (#197)"
```

## Task 10: Persist `ladder` + `best_metric_value` + `scheduler_kind` in `training_state`

**Files:**

- Modify: `src/custom_sam_peft/train/checkpoint.py` — `save_full_state` args + payload keys; `ResumeState` fields; `load_full_state` reads.
- Test: `tests/train/test_ladder_resume.py` (round-trip + old-checkpoint compat).

- [ ] **Step 1: Write the failing persistence test** (spec §8.1, §14.4)

Create `tests/train/test_ladder_resume.py`:

```python
"""Ladder state persists in training_state and restores on load (spec §8, §14.4)."""

from __future__ import annotations

from pathlib import Path

import torch

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import (
    ResumeState,
    load_full_state,
    save_full_state,
)
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg


def test_ladder_round_trips_through_full_state(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=1)
    # Advance the plateau scheduler so num_bad_epochs is non-trivial.
    sched.step(0.5)
    sched.step(0.5)

    state_dir = tmp_path / "checkpoints" / "step_3"
    save_full_state(
        state_dir=state_dir, wrapper=wrapper, optimizer=opt, scheduler=sched,
        global_step=3, epoch=0, nan_streak=0, cfg=cfg,
        ladder={"best": 0.5, "evals_without_improvement": 2},
        best_metric_value=0.5, scheduler_kind="plateau",
    )

    # Fresh objects to load into.
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg.peft)
    o2 = torch.optim.AdamW([p for p in w2.parameters() if p.requires_grad], lr=1e-4)
    s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(o2, mode="max", patience=1)
    rs = load_full_state(state_dir, w2, o2, s2, cfg)

    assert isinstance(rs, ResumeState)
    assert rs.ladder == {"best": 0.5, "evals_without_improvement": 2}
    assert rs.best_metric_value == 0.5
    assert rs.scheduler_kind == "plateau"
    # The ReduceLROnPlateau's own state restored (num_bad_epochs continued).
    assert s2.num_bad_epochs == sched.num_bad_epochs


def test_old_checkpoint_without_ladder_loads(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    state_dir = tmp_path / "checkpoints" / "step_1"
    # Save WITHOUT the new args (defaults) — simulates a pre-#197 payload shape.
    save_full_state(
        state_dir=state_dir, wrapper=wrapper, optimizer=opt, scheduler=sched,
        global_step=1, epoch=0, nan_streak=0, cfg=cfg,
    )
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg.peft)
    o2 = torch.optim.AdamW([p for p in w2.parameters() if p.requires_grad], lr=1e-4)
    s2 = torch.optim.lr_scheduler.LambdaLR(o2, lr_lambda=lambda s: 1.0)
    rs = load_full_state(state_dir, w2, o2, s2, cfg)
    assert rs.ladder is None
    assert rs.best_metric_value is None
    assert rs.scheduler_kind is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder_resume.py -q`
Expected: FAIL — `save_full_state` has no `ladder`/`best_metric_value`/`scheduler_kind` kwargs; `ResumeState` has no new fields.

- [ ] **Step 3: Extend `ResumeState` and `save_full_state`/`load_full_state`**

In `src/custom_sam_peft/train/checkpoint.py`, extend `ResumeState` (lines 77-81):

```python
@dataclass(frozen=True)
class ResumeState:
    start_step: int
    start_epoch: int
    nan_streak: int
    ladder: dict[str, Any] | None = None  # NEW — None for pre-#197 checkpoints
    best_metric_value: float | None = None  # NEW
    scheduler_kind: str | None = None  # NEW — effective LR schedule; governs resume rebuild
```

Extend `save_full_state`'s signature (lines 147-156) with three keyword args (defaults so old call sites work):

```python
def save_full_state(
    state_dir: Path,
    wrapper: Sam3Wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    nan_streak: int,
    cfg: TrainConfig,
    ladder: dict[str, Any] | None = None,
    best_metric_value: float | None = None,
    scheduler_kind: str | None = None,
) -> None:
```

> Widen the `scheduler` param annotation from `torch.optim.lr_scheduler.LRScheduler` to `Any` (a `ReduceLROnPlateau` is not an `LRScheduler` subclass but has `.state_dict()`). The existing `scheduler.state_dict()` call (line 164) is unchanged.

Add to the payload (after `cfg_hash`, line 173):

```python
        "ladder": ladder,
        "best_metric_value": best_metric_value,
        "scheduler_kind": scheduler_kind,
```

In `load_full_state`, widen its `scheduler` param to `Any` as well, and change the `return ResumeState(...)` (lines 237-241) to read the new keys with `.get(...)`:

```python
    return ResumeState(
        start_step=int(state["global_step"]),
        start_epoch=int(state["epoch"]),
        nan_streak=int(state["nan_streak"]),
        ladder=state.get("ladder"),
        best_metric_value=state.get("best_metric_value"),
        scheduler_kind=state.get("scheduler_kind"),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder_resume.py -q`
Expected: PASS.

- [ ] **Step 5: Pass ladder state from the two `save_full_state` call sites**

Two call sites flush full state: `_maybe_checkpoint` (`trainer.py:428`) and the time-limit flush in `run_epoch` (`loop.py:529`). Both must now pass the ladder state so a periodic or time-limited flush persists it (spec §9.1). For `_maybe_checkpoint`, pass the trainer's ladder:

```python
        save_full_state(
            state_dir=state_dir,
            wrapper=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=step,
            epoch=epoch,
            nan_streak=nan_streak,
            cfg=self.cfg,
            ladder=self._ladder.state_dict(),
            best_metric_value=self._best_metric_value,
            scheduler_kind=self._scheduler_kind,
        )
```

`self._scheduler_kind` is set in `fit()` from `effective_schedule` — add `self._scheduler_kind: str = effective_schedule` right after the val-fallback block (Task 6's code). For the `run_epoch` time-limit flush (`loop.py:529`), `run_epoch` does not own the ladder; thread the ladder dict + values in as optional `run_epoch` params (`ladder_state_dict`, `best_metric_value`, `scheduler_kind`) defaulting to `None`, passed from `_train_epoch` → `run_epoch`, and forward them to `save_full_state`. The trainer computes them from `self._ladder.state_dict()` / `self._best_metric_value` / `self._scheduler_kind` and passes them to `_train_epoch`.

> Keep this minimal: the time-limit flush is the only `run_epoch`-owned `save_full_state` call. If threading three params is noisy, an acceptable alternative is a single `flush_extra: dict | None` param carrying all three. Pick one; the test in §14.9 only asserts the keys are present in the flushed payload.

- [ ] **Step 6: Run the resume + time-limit non-regression subset**

Run: `uv run pytest -o "addopts=" tests/train/test_ladder_resume.py tests/train/test_time_limit_resume.py tests/train/test_time_limit_stop.py -q`
Expected: PASS — ladder round-trips; time-limit flush still works and now also carries ladder keys.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/train/checkpoint.py src/custom_sam_peft/train/trainer.py src/custom_sam_peft/train/loop.py tests/train/test_ladder_resume.py
git commit -m "feat(train): persist ladder + best_metric_value + scheduler_kind in training_state (#197)"
```

## Task 11: Clobber-bug fix + ladder re-seed on resume; resume from `scheduler_kind`

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` — re-seed `_best_metric_value` + `ladder.best` after `load_full_state`; build the scheduler from the persisted `scheduler_kind` on resume.
- Test: `tests/train/test_best_clobber_regression.py`

- [ ] **Step 1: Write the failing clobber regression** (spec §8.2, §14.5)

Create `tests/train/test_best_clobber_regression.py`:

```python
"""A post-resume eval must not clobber a better best/ (spec §8.2, §14.5)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import save_full_state
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_resume_reseeds_best_from_best_json(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)

    # Build a run_dir with a best/best.json claiming mAP=0.7 and a step checkpoint.
    run_dir = tmp_path / "clobber-run"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.7, "global_step": 5})
    )
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    state_dir = run_dir / "checkpoints" / "step_5"
    save_full_state(
        state_dir=state_dir, wrapper=wrapper, optimizer=opt, scheduler=sched,
        global_step=5, epoch=0, nan_streak=0, cfg=cfg,
        ladder={"best": 0.7, "evals_without_improvement": 0},
        best_metric_value=0.7, scheduler_kind="plateau",
    )

    trainer = Trainer(wrapper, ds, ds, NoopTracker := __import__(
        "custom_sam_peft.tracking.noop", fromlist=["NoopTracker"]).NoopTracker(), cfg)
    # Resume — fit() re-seeds _best_metric_value from best.json BEFORE any eval.
    trainer.fit(run_dir=run_dir, resume_from=state_dir)
    assert trainer._best_metric_value >= 0.7  # type: ignore[attr-defined]
    # best/best.json still claims 0.7 (a worse post-resume eval did not overwrite it).
    saved = json.loads((run_dir / "best" / "best.json").read_text())
    assert saved["value"] >= 0.7
```

> The pre-fix behavior (`_best_metric_value` reset to `-inf` each `fit()`) would let the first post-resume eval (a finite mAP) overwrite `best/`; this test must FAIL before the fix and PASS after.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/train/test_best_clobber_regression.py -q`
Expected: FAIL — `_best_metric_value` is `-inf` post-resume; `best/` gets clobbered.

- [ ] **Step 3: Re-seed + resume-from-scheduler_kind in `fit()`**

In `src/custom_sam_peft/train/trainer.py`, the resume block is `if resume_from is not None: rs = load_full_state(...)` (lines 521-522). Two changes:

First, build the scheduler from the **persisted** `scheduler_kind` on resume (spec §8.3). Restructure so the resume's `scheduler_kind` (if present) overrides `effective_schedule` **before** `_build_scheduler`:

```python
        # On resume, the persisted scheduler_kind governs construction so the
        # rebuilt scheduler type matches the persisted state_dict (spec §8.3).
        resume_kind: str | None = None
        if resume_from is not None:
            import torch as _torch  # already imported at module scope; alias for clarity

            peek = _torch.load(resume_from / "training_state.pt", weights_only=False)
            resume_kind = peek.get("scheduler_kind")
        if resume_kind is not None and resume_kind != effective_schedule:
            _LOG.warning(
                "resume: persisted scheduler_kind=%s overrides cfg lr_schedule=%s "
                "(persisted kind wins).",
                resume_kind, effective_schedule,
            )
        if resume_kind is not None:
            effective_schedule = resume_kind
        self._scheduler_kind = effective_schedule
```

> Place this block right after the val-fallback computes `effective_schedule` (Task 6) and **before** `_build_scheduler` (line 514). Then `_build_scheduler(optimizer, cfg, total_steps, effective_schedule)` and `load_full_state(...)` restore the matching `state_dict`. (A lightweight `torch.load` peek avoids re-ordering the whole resume flow; `load_full_state` still does the authoritative restore.)

Second, after `rs = load_full_state(...)` (line 522), re-seed (spec §8.2):

```python
            if rs.best_metric_value is not None:
                self._best_metric_value = rs.best_metric_value
            resume_run_dir = resume_from.parent.parent
            best_json = resume_run_dir / "best" / "best.json"
            if best_json.is_file():
                try:
                    disk_best = float(json.loads(best_json.read_text())["value"])
                    self._best_metric_value = max(self._best_metric_value, disk_best)
                except Exception:
                    _LOG.warning("resume: could not read %s; keeping in-memory best.", best_json)
            self._ladder.best = self._best_metric_value
            if rs.ladder is not None:
                self._ladder.load_state_dict(rs.ladder)
```

> `self._ladder` is constructed in `__init__` and reset in `fit()` after `_build_scheduler` (Phase 1 Task 7). Ensure the re-seed runs **after** that reset — i.e. the reset (`self._ladder = LadderState()`) must happen before the `if resume_from is not None:` block, or fold the reset into the non-resume path only. Cleanest: construct/reset `self._ladder = LadderState()` immediately after storing `self._scheduler`, then the resume block re-seeds it.

- [ ] **Step 4: Run the regression to verify it passes**

Run: `uv run pytest -o "addopts=" tests/train/test_best_clobber_regression.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py tests/train/test_best_clobber_regression.py
git commit -m "fix(train): re-seed best_metric_value+ladder on resume; rebuild scheduler from persisted kind (#197)"
```

## Task 12: `close_out(...)` — best-restoration + single eval + write (`train/close_out.py`)

**Files:**

- Create: `src/custom_sam_peft/train/close_out.py`
- Test: `tests/train/test_close_out.py`

- [ ] **Step 1: Write the failing close_out tests** (spec §7.2, §14.6)

Create `tests/train/test_close_out.py`:

```python
"""close_out best-restoration + single eval + write (spec §7.2, §14.6)."""

from __future__ import annotations

import json
from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import save_adapter
from custom_sam_peft.train.close_out import close_out
from custom_sam_peft.train.ladder import LadderEvents, LrCut
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_close_out_restores_best_and_writes_adapter(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run"
    (run_dir / "best").mkdir(parents=True)
    # Save a distinguishable best/ adapter + best.json.
    save_adapter(wrapper, run_dir / "best" / "adapter")
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.8, "global_step": 7})
    )

    events = LadderEvents(cuts=(LrCut(6, 1e-4, 1e-5, 0.5),), stop_reason="early_stop: 10 ...")
    art = close_out(
        run_dir, wrapper, cfg, evaluator_val_ds=ds, oom_state=None,
        final_step=7, final_epoch=0, ladder_events=events,
    )
    assert isinstance(art, EvalArtifacts)
    assert (run_dir / "adapter").is_dir()  # adapter written
    assert art.checkpoint_path == run_dir / "adapter"
    assert art.final_weights == "best"
    assert art.per_example_iou is not None  # single eval returned per-example IoU
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["final_weights"] == "best"
    assert "ladder_events" in metrics


def test_close_out_falls_back_to_last_step_when_no_best(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-nobest"
    run_dir.mkdir(parents=True)
    art = close_out(
        run_dir, wrapper, cfg, evaluator_val_ds=ds, oom_state=None,
        final_step=3, final_epoch=0, ladder_events=None,
    )
    assert art.final_weights == "last_step"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["final_weights"] == "last_step"


def test_close_out_no_val_returns_none_metrics(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / "run-noval"
    run_dir.mkdir(parents=True)
    art = close_out(
        run_dir, wrapper, cfg, evaluator_val_ds=None, oom_state=None,
        final_step=3, final_epoch=0, ladder_events=None,
    )
    assert art.final_metrics is None
    assert art.per_example_iou is None
    assert (run_dir / "adapter").is_dir()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/train/test_close_out.py -q`
Expected: FAIL — `custom_sam_peft.train.close_out` does not exist.

- [ ] **Step 3: Implement `train/close_out.py`**

Create `src/custom_sam_peft/train/close_out.py`:

```python
"""Best-as-final close-out (spec §7).

Restore run_dir/best/ into the model, run ONE full eval
(return_per_example_iou=True), and write run_dir/adapter (+ optional
run_dir/merged + metrics.json) — all on the BEST weights. Falls back to the
current in-memory (last-step) weights when no best/ exists, or when restoring
best/ raises (swallow-and-continue, mirroring _maybe_save_best).

Called on early stop, normal completion, and the finalize entry. Never called
for a _TimeLimitReached pause (spec §9.1).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
from custom_sam_peft.train.checkpoint import load_adapter, save_adapter, save_merged

if TYPE_CHECKING:
    from custom_sam_peft.config.schema import TrainConfig
    from custom_sam_peft.data.base import Dataset
    from custom_sam_peft.models.sam3 import Sam3Wrapper
    from custom_sam_peft.train.ladder import LadderEvents
    from custom_sam_peft.train.loop import OomState

_LOG = logging.getLogger(__name__)


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
    # 1. Restore best/ (or keep last-step weights on absence/failure).
    best_adapter = run_dir / "best" / "adapter"
    final_weights = "last_step"
    if best_adapter.is_dir():
        try:
            load_adapter(model, best_adapter)
            final_weights = "best"
        except Exception:
            _LOG.warning(
                "close_out: failed to restore best/ — finalizing on last-step weights.",
                exc_info=True,
            )

    # 2. Write the (best, or last-step) adapter.
    save_adapter(model, run_dir / "adapter")

    # 3. Optional merged.
    if cfg.export.merge:
        save_merged(model, run_dir / "merged")

    # 4. Single full eval on the restored weights (return_per_example_iou=True).
    report: Any = None
    per_example_iou: list[float] | None = None
    if evaluator_val_ds is not None:
        full_eval_cfg = cfg.eval
        if full_eval_cfg.batch_size == "auto":
            from custom_sam_peft.presets import decide_eval_batch_size

            bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
            if oom_state is not None and bs > oom_state.micro_batch_size:
                bs = oom_state.micro_batch_size
            full_eval_cfg = full_eval_cfg.model_copy(update={"batch_size": bs})
        report, per_example_iou = Evaluator(full_eval_cfg).evaluate(
            model, evaluator_val_ds, return_per_example_iou=True
        )

    # 5. metrics.json (best mAP), final_weights, ladder events.
    metrics: dict[str, Any] = {
        "final_weights": final_weights,
        "global_step": final_step,
        "epoch": final_epoch,
    }
    if report is not None:
        metrics.update(
            {
                "overall": report.overall,
                "per_class": report.per_class,
                "n_images": report.n_images,
                "n_predictions": report.n_predictions,
            }
        )
    else:
        metrics["note"] = "no validation set provided"
    if ladder_events is not None:
        metrics["ladder_events"] = {
            "cuts": [asdict(c) for c in ladder_events.cuts],
            "stop_reason": ladder_events.stop_reason,
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return EvalArtifacts(
        checkpoint_path=run_dir / "adapter",
        peft_method=cfg.peft.method,
        run_dir=run_dir,
        final_metrics=report,
        oom_events=tuple(oom_state.pending_oom_events) if oom_state is not None else (),
        per_example_iou=per_example_iou,
        final_weights=final_weights,
        ladder_events=ladder_events,  # ride the returned artifacts (callers read artifacts.ladder_events)
    )
```

> `ladder_events` is already the parameter received by `close_out` — pass it through verbatim. Writing to `metrics.json` (the persisted record) AND setting it on the returned `EvalArtifacts` are both done; they serve different consumers (disk persistence vs in-memory caller).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/train/test_close_out.py -q`
Expected: PASS. Also verify the existing `test_close_out_restores_best_and_writes_adapter` assertion `"ladder_events" in metrics` still passes, and add an assertion:

```python
    assert art.ladder_events == events  # also rides the returned EvalArtifacts
```

Update `test_close_out_falls_back_to_last_step_when_no_best` and `test_close_out_no_val_returns_none_metrics` to assert `art.ladder_events is None` (since `ladder_events=None` is passed in those cases).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/close_out.py tests/train/test_close_out.py
git commit -m "feat(train): close_out — best-as-final restore + single eval + write (#197)"
```

## Task 13: Wire `close_out` into `fit()` (early stop + normal completion)

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` — replace the inline finalize block (`trainer.py:581-621`) with a `close_out` call; build `LadderEvents`; return its `EvalArtifacts`.
- Test: `tests/train/test_early_stop_integration.py`

- [ ] **Step 1: Write the failing early-stop integration test** (spec §14.7)

Create `tests/train/test_early_stop_integration.py`:

```python
"""Early stop funnels into close_out; best-as-final on stop + normal completion (§14.7)."""

from __future__ import annotations

import json
from pathlib import Path

from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking.noop import NoopTracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _TinyDataset, _make_cfg


def test_normal_completion_closes_out_on_best(tmp_path: Path) -> None:
    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(
        update={"lr_schedule": "plateau", "eval_every": 1, "epochs": 1})})
    apply_lora(wrapper, cfg.peft)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "normal-run")
    assert isinstance(result, EvalArtifacts)
    assert result.checkpoint_path == tmp_path / "normal-run" / "adapter"
    assert result.final_weights in {"best", "last_step"}
    metrics = json.loads((tmp_path / "normal-run" / "metrics.json").read_text())
    assert "final_weights" in metrics


def test_early_stop_stops_before_epochs_and_closes_out(tmp_path: Path, monkeypatch) -> None:
    """Injected plateau mAPs trigger _EarlyStop; fit returns best-as-final artifacts."""
    import custom_sam_peft.eval.evaluator as ev

    ds = _TinyDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={
        "lr_schedule": "plateau", "eval_every": 1, "epochs": 50,
        "early_stop": cfg.train.early_stop.model_copy(update={"stop_patience": 2}),
    })})
    apply_lora(wrapper, cfg.peft)

    # Force every lite eval to report a flat mAP so the ladder never improves.
    from custom_sam_peft.eval.metrics import MetricsReport

    flat = MetricsReport(overall={"mAP": 0.1}, per_class={}, n_images=1, n_predictions=0)

    def fake_eval(self, model, dataset, **k):
        if k.get("return_per_example_iou"):
            return flat, [0.1]
        return flat

    monkeypatch.setattr(ev.Evaluator, "evaluate", fake_eval)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit(run_dir=tmp_path / "stop-run")
    assert isinstance(result, EvalArtifacts)
    metrics = json.loads((tmp_path / "stop-run" / "metrics.json").read_text())
    assert "final_weights" in metrics
    # Stopped well before 50 epochs (stop_patience=2 → stops within a few evals).
    assert metrics["global_step"] < 50 * len(ds)
```

> Construct `MetricsReport` with whatever fields the real class requires — check `eval/metrics.py` and match the constructor. If it has required fields beyond those shown, add them with trivial values.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/train/test_early_stop_integration.py -q`
Expected: FAIL — `fit()` still uses the inline last-step finalize; `metrics.json` lacks `final_weights`.

- [ ] **Step 3: Replace the inline finalize block with `close_out`**

In `src/custom_sam_peft/train/trainer.py`, add the import:

```python
from custom_sam_peft.train.close_out import close_out
```

Replace the entire `if stop is None: ... ` inline finalize block (lines 581-621) with:

```python
            if stop is None:
                final_epoch = (
                    early.epoch if early is not None else cfg.train.epochs - 1
                )
                ladder_events = LadderEvents(
                    cuts=tuple(self._ladder_cuts),
                    stop_reason=(early.reason if early is not None else None),
                )
                close_out_result = close_out(
                    run_dir,
                    self.model,
                    cfg,
                    evaluator_val_ds=self.val_ds,
                    oom_state=oom_state,
                    final_step=global_step,
                    final_epoch=final_epoch,
                    ladder_events=ladder_events,
                )
```

> `close_out_result` is assigned inside the outer `try` but used after the `finally`. Declare `close_out_result: EvalArtifacts | None = None` next to `stop`/`early` (line 556).

Change the post-`finally` return logic (lines 625-634) to:

```python
        if stop is not None:
            return self._time_limited_artifacts(run_dir, stop, budget_seconds, oom_state)

        assert close_out_result is not None  # set whenever stop is None
        return close_out_result
```

> This removes the now-dead `merged_path`/`full_report` locals and the old `EvalArtifacts(...)` return (lines 628-634). Delete the `merged_path`/`full_report` initialization (lines 554-555) if no longer referenced, and remove the `save_adapter`/`save_merged`/`Evaluator` imports only if they are unused elsewhere in the file (they are likely still used by `_maybe_save_best`/`_log_image_panel` — verify with `grep` before deleting any import).

- [ ] **Step 4: Run the integration test + the eager-import check**

Run: `uv run python -m py_compile src/custom_sam_peft/train/trainer.py && uv run pytest -o "addopts=" tests/train/test_early_stop_integration.py tests/integration/test_trainer_evaluator_seam.py -q`
Expected: PASS — early stop + normal completion both close out on best; the seam tests stay green (their 1-epoch run now closes out via `close_out`, writing `run_dir/adapter` + `metrics.json` as before, plus `final_weights`).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py tests/train/test_early_stop_integration.py
git commit -m "feat(train): fit() funnels early-stop + normal completion into close_out (#197)"
```

## Task 14: Bundle/summary reflect best-as-final + ladder events

**Files:**

- Modify: `src/custom_sam_peft/runs/bundle.py` — `BundleContext.ladder_events`; best-adapter summary line; ladder-event lines.
- Test: `tests/runs/test_bundle_ladder.py`

- [ ] **Step 1: Write the failing bundle test** (spec §7.6)

Create `tests/runs/test_bundle_ladder.py`:

```python
"""Bundle surfaces best-as-final adapter + ladder events (spec §7.6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from custom_sam_peft.presets import decide_preset
from custom_sam_peft.runs.bundle import BundleContext, write_bundle
from custom_sam_peft.train.ladder import LadderEvents, LrCut


def _ctx(run_dir: Path, **kw) -> BundleContext:
    return BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=datetime.now(UTC),
        end_ts=datetime.now(UTC),
        preset=decide_preset(),
        per_example_iou=[],
        merged_dir=None,
        merged_export_error=None,
        oom_events=(),
        **kw,
    )


def test_bundle_context_accepts_ladder_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best").mkdir()
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.8, "global_step": 6})
    )
    events = LadderEvents(cuts=(LrCut(6, 1e-4, 1e-5, 0.5),), stop_reason="early stop: 10 evals")
    ctx = _ctx(run_dir, ladder_events=events)
    # No-val path → summary.md only; must not raise and must mention the cut/stop.
    write_bundle(ctx, None, val_dataset=None, model_wrapper=None)
    body = (run_dir / "summary.md").read_text()
    assert "best checkpoint" in body or "best/" in body
    assert "LR cut" in body
    assert "early stop" in body


def test_bundle_default_ladder_events_renders_unchanged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    ctx = _ctx(run_dir)  # ladder_events defaults to None
    write_bundle(ctx, None, val_dataset=None, model_wrapper=None)
    body = (run_dir / "summary.md").read_text()
    assert "LR cut" not in body  # nothing rendered for a no-ladder run
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/runs/test_bundle_ladder.py -q`
Expected: FAIL — `BundleContext` has no `ladder_events` field.

- [ ] **Step 3: Add `ladder_events` + render lines to `bundle.py`**

In `src/custom_sam_peft/runs/bundle.py`, add to `BundleContext` (after `oom_events`, line 103):

```python
    ladder_events: "LadderEvents | None" = None
```

Add the import (TYPE_CHECKING-guarded to avoid a cycle, or a direct import — `bundle.py` already imports from `train.types`):

```python
from custom_sam_peft.train.ladder import LadderEvents, LrCut
```

Add a helper near `_oom_edge_note` (line 333):

```python
def _best_adapter_line(run_dir: Path) -> str:
    """Outputs-section adapter line: best checkpoint if best.json exists, else last-step."""
    best_json = run_dir / "best" / "best.json"
    if best_json.is_file():
        try:
            data = json.loads(best_json.read_text())
            return (
                f"- Adapter: adapter/ (best checkpoint, mAP {float(data['value']):.4f} "
                f"at step {int(data['global_step'])})"
            )
        except Exception:
            pass
    return "- Adapter: adapter/ (last-step weights — no best/ produced)"


def _ladder_event_lines(events: LadderEvents | None) -> list[str]:
    if events is None:
        return []
    lines: list[str] = []
    for c in events.cuts:
        ratio = c.new_lr / c.old_lr if c.old_lr else 0.0
        lines.append(
            f"- LR cut ×{ratio:.3g} → {c.new_lr:.3g} at eval step {c.step} (mAP {c.triggering_map:.4f})"
        )
    if events.stop_reason:
        lines.append(f"- early stop: {events.stop_reason}")
    return lines
```

In **both** `_write_summary_no_val` (line 386) and `_collect_artifacts` (line 530), replace the static `f"- Adapter: {adapter_rel}\n"` line with `f"{_best_adapter_line(ctx.run_dir)}\n"`, and append a `## Training` section when ladder events exist. For example, after the `## Outputs` block, add:

```python
    training_lines = _ladder_event_lines(ctx.ladder_events)
    if training_lines:
        body += "\n## Training\n" + "\n".join(training_lines) + "\n"
```

> Apply the `## Training` addition to both summary writers so a no-val early-stopped run still surfaces its cuts/stop. Keep the existing `## Edge cases` block intact.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/runs/test_bundle_ladder.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/runs/bundle.py tests/runs/test_bundle_ladder.py
git commit -m "feat(runs): bundle surfaces best-as-final adapter + ladder events (#197)"
```

## Task 15: `run` orchestrator drops eval + export-merge phases (single eval)

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py` — `_orchestrate` builds the bundle from `train_result.final_metrics` + `train_result.per_example_iou`; drops its eval + export-merge phases for the normal path.
- Test: `tests/cli/test_run_single_eval.py`

- [ ] **Step 1: Write the failing single-eval test** (spec §7.4, §14 A4)

Create `tests/cli/test_run_single_eval.py`:

```python
"""run orchestrator runs exactly one eval (in close_out) — none of its own (§7.4)."""

from __future__ import annotations

from pathlib import Path

import custom_sam_peft.cli.run_cmd as run_cmd
from custom_sam_peft.eval._artifacts import EvalArtifacts


def test_orchestrate_does_not_call_run_eval(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "config.yaml").write_text("train: {}\n")

    # Stub a completed train_result carrying final_metrics + per_example_iou.
    from custom_sam_peft.eval.metrics import MetricsReport

    report = MetricsReport(overall={"mAP": 0.8}, per_class={}, n_images=1, n_predictions=0)
    artifacts = EvalArtifacts(
        checkpoint_path=run_dir / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=report,
        per_example_iou=[0.8],
        final_weights="best",
    )

    monkeypatch.setattr(run_cmd, "run_training", lambda cfg, resume_from=None: artifacts)

    called = {"eval": 0, "merge": 0}
    monkeypatch.setattr(run_cmd, "run_eval", lambda *a, **k: called.__setitem__("eval", 1))
    monkeypatch.setattr(run_cmd, "save_merged", lambda *a, **k: called.__setitem__("merge", 1))
    # val_source so the bundle path knows the mode.
    import custom_sam_peft.data.val_source as vs_mod

    monkeypatch.setattr(
        run_cmd, "load_sam31", lambda *a, **k: object()
    )
    monkeypatch.setattr(run_cmd, "load_adapter", lambda *a, **k: None)
    # The orchestrator builds the bundle from artifacts; stub write_bundle.
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: None)
    # Provide a val_source.json (mode none keeps it simple — no val dataset build).
    (run_dir / "val_source.json").write_text('{"mode": "none"}')

    from custom_sam_peft.cli._progress import ProgressMode

    rc = run_cmd._orchestrate(
        cfg=_FakeCfg(), resume=None, mode=ProgressMode.OFF,
        visualize=False, config_path=run_dir / "config.yaml",
    )
    assert rc == 0
    assert called["eval"] == 0  # the orchestrator ran NO eval of its own


class _FakeCfg:
    class export:  # noqa: N801
        merge = False

    class train:  # noqa: N801
        epochs = 1
```

> This test sketches the contract (no `run_eval` call on the normal path). Adapt the stubs to the actual `_orchestrate` signature/branches; the load-bearing assertion is `called["eval"] == 0`. If `_FakeCfg` is too thin for the bundle path, reuse `_make_cfg` and a real no-val `val_source.json` instead. Keep it CPU-only and fully mocked.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -o "addopts=" tests/cli/test_run_single_eval.py -q`
Expected: FAIL — `_orchestrate` still calls `run_eval` (`run_cmd.py:124`).

- [ ] **Step 3: Rewrite `_orchestrate`'s normal path to reuse close_out's eval**

In `src/custom_sam_peft/cli/run_cmd.py`, after the time-limit short-circuit (line 97), replace the eval phase (lines 111-138) and the export-merge phase (lines 142-157) so the bundle is built from `train_result`:

```python
    run_dir = train_result.run_dir
    adapter_path = train_result.checkpoint_path  # run_dir/adapter (best weights)

    vs = load_val_source(run_dir)
    if vs is None:
        raise RuntimeError(f"runner did not save val_source.json in {run_dir}")

    # close_out (inside fit()) already ran the single eval + export-merge on the
    # best weights; reuse its results — no second eval, no second merge.
    report = train_result.final_metrics
    per_example_iou = train_result.per_example_iou or []

    val_dataset: Dataset | None = None
    wrapper: Any = None
    if vs.mode != "none":
        # Rebuild the model + val dataset only for bundle re-inference (sample panels).
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        load_adapter(wrapper, adapter_path)
        val_dataset = _build_val_dataset(cfg, vs)

    end_ts = datetime.now(UTC)

    # merged/ was written by close_out when cfg.export.merge; reflect it.
    merged_dir = (run_dir / "merged") if (cfg.export.merge and (run_dir / "merged").is_dir()) else None
    merged_export_error: str | None = None
```

Then build the `BundleContext` with `ladder_events=...`. The ladder events ride on `train_result` (the `EvalArtifacts` returned by `close_out`) — read them directly from `train_result.ladder_events`:

```python
    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset=_load_preset_or_fallback(cfg),
        per_example_iou=per_example_iou,
        merged_dir=merged_dir,
        merged_export_error=merged_export_error,
        oom_events=train_result.oom_events,
        ladder_events=train_result.ladder_events,  # from EvalArtifacts, not metrics.json
    )
```

Do **not** add a `_read_ladder_events` helper — it is not needed. `ladder_events` are already available on the `EvalArtifacts` object returned by `close_out` (via `fit()`). The `metrics.json` write in `close_out` is the persisted record; the in-memory `EvalArtifacts.ladder_events` field is what callers read. Remove the now-unused `run_eval` import and `save_merged` import if no longer referenced (grep first).

> Net: exactly one eval on the `run` path (in `close_out`), down from two (spec §7.4, A4). The bundle still re-infers sample panels via `model_wrapper`, which is the same re-inference it did before — that is not a metrics eval.

- [ ] **Step 4: Run the test + the existing run-cmd tests**

Run: `uv run pytest -o "addopts=" tests/cli/test_run_single_eval.py tests/cli -q`
Expected: PASS. Adjust any existing `run` CLI test that asserted two-eval behavior or the dropped phases (update expectations to the single-eval path).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/cli/test_run_single_eval.py
git commit -m "feat(cli): run orchestrator reuses close_out's single eval; drops its eval+merge phases (#197)"
```

## Task 16: Phase 2 docs (config-schema.md + defaults-provenance.md)

**Files:**

- Modify: `docs/config-schema.md` — `lr_schedule` row + `lr_decay_on_plateau`/`early_stop` sub-blocks; the §5.4 wart; val-fallback note.
- Modify: `docs/defaults-provenance.md` — `lr_schedule` row (line 89); six new knob rows; `config_full` cross-link (line 206).

- [ ] **Step 1: Update `docs/config-schema.md`**

Change the `train.lr_schedule` row (line 122) so the type union includes `"plateau"` and the default is `"plateau"`:

```text
| `train.lr_schedule` | `"constant"` \| `"cosine"` \| `"linear"` \| `"plateau"` | `"plateau"` | common | Learning-rate decay schedule. `plateau` (default) reduces LR on a validation-mAP plateau (rung 1) and is paired with early stop (rung 2); `cosine`/`linear`/`constant` keep the per-step schedule. `plateau` requires a val set and falls back to `cosine` (with a warning) when none is present. | #197: plateau pairs reduce-on-plateau with early stopping (research §2–§4). |
```

Under **Advanced fields** (after line 127), add rows for the two new sub-blocks:

```text
| `train.lr_decay_on_plateau.patience` | int (>0) | `5` | advanced | Non-improving evals before one LR cut (rung 1; plateau mode only). | Keras example 5 (research §2,§7). |
| `train.lr_decay_on_plateau.factor` | float (<1) | `0.1` | advanced | LR multiplier on a cut. | PyTorch default 0.1 (research §2,§7). |
| `train.lr_decay_on_plateau.min_lr` | float (>0) | `1e-6` | advanced | LR floor; cuts never go below it. | # tbd: learning_rate/100 (research §7). |
| `train.early_stop.enabled` | bool | `true` | advanced | Stop after `stop_patience` non-improving evals (rung 2). | Issue #197: on by default. |
| `train.early_stop.monitor` | `"mAP"` | `"mAP"` | advanced | Monitored metric (seam; only mAP wired). | Existing best-metric key. |
| `train.early_stop.min_delta` | float (>0) | `0.001` | advanced | Shared improvement threshold for BOTH rungs (see wart). | Keras/practitioner 0.001–0.01 (research §5,§7). |
| `train.early_stop.stop_patience` | int (>0) | `10` | advanced | Non-improving evals before early stop. | PyTorch default 10 / Prechelt (research §5,§7). |
```

Add a prose note after the table documenting the §5.4 wart: `monitor`/`min_delta` live in `early_stop` but configure **both** rungs' "what counts as improvement" — even when `early_stop.enabled=false`, those two fields still drive the rung-1 LR-decay threshold in `plateau` mode.

- [ ] **Step 2: Update `docs/defaults-provenance.md`**

Change the `lr_schedule` row (line 89): default `"cosine"` → `"plateau"`; basis += the ReduceLROnPlateau/early-stop pairing + the research §3–§4 horizon-mismatch argument + `# tbd: #197` for the flip; keep the SGDR cite for the still-available cosine shape. Add six rows (mirroring the §5.6 table) for `lr_decay_on_plateau.{patience,factor,min_lr}` and `early_stop.{enabled,monitor,min_delta,stop_patience}`, each cross-linking the research notes (`docs/research/2026-05-30-issue-197-plateau-lr-decay-early-stopping-lit-review.md` §2/§5/§7). Update the `config_full.yaml:train.lr_schedule` cross-link row (line 206) default `cosine` → `plateau`.

- [ ] **Step 3: Markdown-lint the docs**

Run the markdownlint gate (from "Verification gates") on `docs/config-schema.md docs/defaults-provenance.md`.
Expected: clean exit (0). Fix any MD findings (blank lines around tables, no bare URLs) before committing.

- [ ] **Step 4: Commit**

```bash
git add docs/config-schema.md docs/defaults-provenance.md
git commit -m "docs: plateau ladder config rows + provenance (#197)"
```

## Task 17: Phase 2 verification gate

**Files:** none (verification).

- [ ] **Step 1: Lint + format-check + type**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft`
Expected: clean. Watch for: the widened `scheduler: Any` params in `checkpoint.py`; `close_out`'s `Any` report; `EvalArtifacts.per_example_iou`/`final_weights` typing; unused-import removals in `trainer.py`/`run_cmd.py` (ruff F401).

- [ ] **Step 2: #198 + OOM non-regression (spec §14.9)**

Run: `uv run pytest -o "addopts=" tests/train/test_time_limit_stop.py tests/train/test_time_limit_resume.py tests/train/test_time_limit_noop.py -q`
Expected: PASS — a `_TimeLimitReached` still routes to `_time_limited_artifacts` (no `close_out`, no eval); the flushed `training_state` now also carries `ladder`/`best_metric_value`/`scheduler_kind` (present but unread on the pause path). Also run any existing OOM-ladder tests: `uv run pytest -o "addopts=" tests/unit/test_eval_batch_size_cap.py -q` (and any `tests/train/test_oom*`).

- [ ] **Step 3: Full gated suite**

Run: `uv run pytest`
Expected: PASS including `--cov-fail-under=80`.

> **Phase 2 → Phase 3 contract (restated for the next session):** `train/close_out.py` exports `close_out(run_dir, model, cfg, *, evaluator_val_ds, oom_state, final_step, final_epoch, ladder_events=None) -> EvalArtifacts`. It restores `run_dir/best/adapter` into `model` (falling back to last-step on absence/failure), writes `run_dir/adapter` (best weights), optional `run_dir/merged` (when `cfg.export.merge`), and `run_dir/metrics.json` (with `final_weights` + `ladder_events`), and runs the full eval **exactly once** (`return_per_example_iou=True`). The returned `EvalArtifacts` has `checkpoint_path == run_dir/adapter`, `final_metrics` = best eval (or `None`), `per_example_iou` = the bundle's IoU data (or `None`), `final_weights` ∈ {`"best"`, `"last_step"`}, and `ladder_events: LadderEvents | None` carrying the accumulated telemetry (mirroring `oom_events`). `BundleContext` gained `ladder_events: LadderEvents | None = None`; the orchestrator reads ladder events from `train_result.ladder_events` (the `EvalArtifacts` field) directly — no `_read_ladder_events` helper. Phase 3's `_finalize` calls `close_out` directly and reads `artifacts.ladder_events` for the bundle.

---

# PHASE 3 — Finalize-a-paused-run entry

**Phase boundary — interface contract IN (from Phase 2):** `close_out(...)` and the `EvalArtifacts` semantics above (including the `ladder_events` field); `run_cmd._build_val_dataset`, `_load_preset_or_fallback`, `BundleContext`, `write_bundle`; `load_sam31`/`load_adapter` from Phase 2's imports. There is no `_read_ladder_events` helper — callers read `artifacts.ladder_events` directly.

**One coherent unit (spec §11):** the `--finalize` flag on `run` + a `_finalize` helper that rebuilds the model from a paused run's checkpoint (its saved `config.yaml`, base model, `load_adapter` of best/latest), rebuilds the val dataset from `val_source.json`, calls `close_out`, builds the bundle, prints a done message — and runs **no** training. Validation: `--finalize` requires `--resume`; rejects `--time-limit`. Terminal phase: opens the PR.

## Task 18: `--finalize` flag + validation + `_finalize` helper (`run_cmd.py`)

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py` — `finalize: bool` option; validation; route to `_finalize`; the helper.
- Test: `tests/cli/test_finalize.py`

- [ ] **Step 1: Write the failing finalize tests** (spec §11, §14.8)

Create `tests/cli/test_finalize.py`:

```python
"""run --finalize: rebuild + close_out, no training (spec §11, §14.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import custom_sam_peft.cli.run_cmd as run_cmd
from custom_sam_peft.eval._artifacts import EvalArtifacts


def _make_paused_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "paused-run"
    (run_dir / "checkpoints" / "step_5").mkdir(parents=True)
    (run_dir / "best" / "adapter").mkdir(parents=True)
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.8, "global_step": 5})
    )
    (run_dir / "config.yaml").write_text("run:\n  name: paused\n")
    (run_dir / "val_source.json").write_text('{"mode": "none"}')
    return run_dir / "checkpoints" / "step_5"


def test_finalize_calls_close_out_no_training(tmp_path: Path, monkeypatch) -> None:
    resume = _make_paused_run(tmp_path)
    run_dir = resume.parent.parent

    called = {"close_out": 0, "train": 0, "fit": 0}
    artifacts = EvalArtifacts(
        checkpoint_path=run_dir / "adapter", peft_method="lora", run_dir=run_dir,
        final_metrics=None, per_example_iou=None, final_weights="best",
    )
    monkeypatch.setattr(run_cmd, "close_out",
                        lambda *a, **k: (called.__setitem__("close_out", 1), artifacts)[1])
    monkeypatch.setattr(run_cmd, "run_training",
                        lambda *a, **k: called.__setitem__("train", 1))
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: object())
    monkeypatch.setattr(run_cmd, "load_adapter", lambda *a, **k: None)
    monkeypatch.setattr(run_cmd, "load_config", lambda p: _SavedCfg())
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: None)

    rc = run_cmd._finalize(
        cfg=_SavedCfg(), resume=resume, mode=run_cmd.ProgressMode.OFF,
        visualize=False, config_path=run_dir / "config.yaml",
    )
    assert rc == 0
    assert called["close_out"] == 1
    assert called["train"] == 0  # NO training


def test_finalize_requires_resume(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("run:\n  name: x\n")
    with pytest.raises(typer.Exit):
        run_cmd.run(config=cfg_path, resume=None, time_limit=None, finalize=True,
                    verbose=False, progress_flag="off", visualize=False)


def test_finalize_rejects_time_limit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("run:\n  name: x\n")
    with pytest.raises(typer.Exit):
        run_cmd.run(config=cfg_path, resume="__latest__", time_limit="1h", finalize=True,
                    verbose=False, progress_flag="off", visualize=False)


class _SavedCfg:
    class export:  # noqa: N801
        merge = False

    class model:  # noqa: N801
        pass

    class data:  # noqa: N801
        channels = 3
        channel_semantics = "rgb"

    class run:  # noqa: N801
        name = "paused"
```

> Adapt the stub `_SavedCfg` to whatever attributes `_finalize` reads (`cfg.model`, `cfg.data.channels`, `cfg.data.channel_semantics`, `cfg.export.merge`). If `run --finalize` validation happens before `load_config`, the two validation tests may need `config.is_file()` to pass — point them at a real (minimal) YAML or stub `load_config`. Keep them CPU-only and fully mocked.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -o "addopts=" tests/cli/test_finalize.py -q`
Expected: FAIL — `run` has no `finalize` param; `_finalize` does not exist.

- [ ] **Step 3: Add the `--finalize` flag + validation to `run(...)`**

In `src/custom_sam_peft/cli/run_cmd.py`, add the option to `run(...)` (after `resume`, line 198):

```python
    finalize: bool = typer.Option(
        False,
        "--finalize",
        help=(
            "Finalize a paused (time-limited) run: rebuild the model from --resume's "
            "checkpoint, restore the best weights, run eval, and write adapter/merged/"
            "metrics/bundle. Runs NO training. Requires --resume; rejects --time-limit."
        ),
    ),
```

After resolving `resume_path` (line 260) and before `_orchestrate` (line 262), add:

```python
    if finalize:
        if resume_path is None:
            rprint("[red]error[/red] --finalize requires --resume (a checkpoint or __latest__).")
            raise typer.Exit(code=1)
        if time_limit is not None:
            rprint("[red]error[/red] --finalize cannot be combined with --time-limit (no training).")
            raise typer.Exit(code=1)
        raise typer.Exit(code=_finalize(cfg, resume_path, mode, visualize=visualize, config_path=config))

    _orchestrate(cfg, resume_path, mode, visualize=visualize, config_path=config)
```

> Place the `--time-limit` rejection so it fires even though `time_limit` was already applied to `cfg` above — the check reads the raw `time_limit` option, so it is correct. (Alternatively, move the `--finalize` validation above the `time_limit`-application block; either ordering works as long as the rejection reads the flag.)

- [ ] **Step 4: Implement `_finalize`**

Add to `src/custom_sam_peft/cli/run_cmd.py`, parallel to `_orchestrate`:

```python
def _finalize(
    cfg: TrainConfig,
    resume: Path,
    mode: ProgressMode,
    *,
    visualize: bool,
    config_path: Path,
) -> int:
    """Productionize a paused run: rebuild + close_out, NO training (spec §11)."""
    from custom_sam_peft.data.val_source import load_val_source
    from custom_sam_peft.train.close_out import close_out

    start_ts = datetime.now(UTC)
    run_dir = resume.parent.parent  # checkpoints/step_N → run_dir

    # The run's OWN config governs model/eval/export shape (spec §11.2, A6).
    saved_cfg_path = run_dir / "config.yaml"
    saved_cfg = load_config(saved_cfg_path) if saved_cfg_path.is_file() else cfg
    if saved_cfg_path.is_file():
        _LOG.info("finalize: using the run's saved config.yaml (not --config) for fidelity.")

    # Rebuild base model + adapter (prefer best/, else the resumed checkpoint's adapter).
    wrapper: Any = load_sam31(
        saved_cfg.model,
        channels=saved_cfg.data.channels,
        channel_semantics=saved_cfg.data.channel_semantics,
    )
    best_adapter = run_dir / "best" / "adapter"
    adapter = best_adapter if best_adapter.is_dir() else resume / "adapter"
    load_adapter(wrapper, adapter)

    # Rebuild val dataset from the saved record.
    vs = load_val_source(run_dir)
    val_ds: Dataset | None = (
        _build_val_dataset(saved_cfg, vs) if (vs is not None and vs.mode != "none") else None
    )

    # final_step/final_epoch from best.json (when finalizing on best) or the checkpoint.
    final_step, final_epoch = _read_final_step_epoch(run_dir, resume)

    artifacts = close_out(
        run_dir, wrapper, saved_cfg,
        evaluator_val_ds=val_ds, oom_state=None,
        final_step=final_step, final_epoch=final_epoch, ladder_events=None,
    )

    end_ts = datetime.now(UTC)
    merged_dir = (run_dir / "merged") if (saved_cfg.export.merge and (run_dir / "merged").is_dir()) else None
    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset=_load_preset_or_fallback(saved_cfg),
        per_example_iou=artifacts.per_example_iou or [],
        merged_dir=merged_dir,
        merged_export_error=None,
        oom_events=artifacts.oom_events,
        ladder_events=artifacts.ladder_events,  # from EvalArtifacts; None since _finalize passes ladder_events=None
    )
    write_bundle(ctx, artifacts.final_metrics, val_dataset=val_ds, model_wrapper=wrapper)

    mAP_str = (
        f"{artifacts.final_metrics.overall.get('mAP', float('nan')):.4f}"
        if artifacts.final_metrics is not None else "n/a (no val)"
    )
    rprint(
        f"[green]finalized[/green] run_dir={run_dir} adapter={run_dir / 'adapter'} "
        f"summary={run_dir / 'summary.md'} mAP={mAP_str}"
    )
    return 0


def _read_final_step_epoch(run_dir: Path, resume: Path) -> tuple[int, int]:
    """Read (global_step, epoch) from best.json or the checkpoint's training_state."""
    best_json = run_dir / "best" / "best.json"
    if best_json.is_file():
        try:
            data = json.loads(best_json.read_text())
            return int(data["global_step"]), 0
        except Exception:
            pass
    state_file = resume / "training_state.pt"
    if state_file.is_file():
        import torch

        state = torch.load(state_file, weights_only=False)
        return int(state.get("global_step", 0)), int(state.get("epoch", 0))
    return 0, 0
```

> Add `import json` at the top of `run_cmd.py` if not already imported. `ProgressMode` is already imported (`run_cmd.py:24`). There is no `_read_ladder_events` helper — `_finalize` reads `artifacts.ladder_events` from the `EvalArtifacts` returned by `close_out` (which passes `ladder_events=None` for the finalize path, so `artifacts.ladder_events` is `None`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -o "addopts=" tests/cli/test_finalize.py -q`
Expected: PASS — finalize calls `close_out`, never `run_training`/`fit`; the two validation cases exit non-zero.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/cli/test_finalize.py
git commit -m "feat(cli): run --finalize — productionize a paused run via close_out, no training (#197)"
```

## Task 19: `--finalize __latest__` resolution test + CLI docs

**Files:**

- Test (extend): `tests/cli/test_finalize.py` — `__latest__` resolution.
- Modify: `docs/config-schema.md` (CLI section) — document `run --finalize`.

- [ ] **Step 1: Add the `__latest__` resolution test** (spec §14.8)

Append to `tests/cli/test_finalize.py`:

```python
def test_finalize_resolves_latest(tmp_path: Path, monkeypatch) -> None:
    resume = _make_paused_run(tmp_path)
    cfg_path = resume.parent.parent / "config.yaml"

    monkeypatch.setattr(run_cmd, "find_latest_checkpoint", lambda cfg: resume)
    captured = {"resume": None}

    def fake_finalize(cfg, resume_path, mode, *, visualize, config_path):
        captured["resume"] = resume_path
        return 0

    monkeypatch.setattr(run_cmd, "_finalize", fake_finalize)
    monkeypatch.setattr(run_cmd, "load_config", lambda p: _SavedCfg())

    with pytest.raises(typer.Exit) as exc:
        run_cmd.run(config=cfg_path, resume="__latest__", time_limit=None, finalize=True,
                    verbose=False, progress_flag="off", visualize=False)
    assert exc.value.exit_code == 0
    assert captured["resume"] == resume  # resolved via find_latest_checkpoint
```

> `cfg_path` must be a real file so `config.is_file()` passes (it is — `_make_paused_run` wrote `config.yaml`). If `run` auto-inits a missing config before `--finalize` validation, ensure the file exists so no init fires.

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest -o "addopts=" tests/cli/test_finalize.py -q`
Expected: PASS.

- [ ] **Step 3: Document `run --finalize` in `docs/config-schema.md`**

In the CLI section (find where `run`/`train` flags are documented; search for `--time-limit` or `--resume`), add a `run --finalize` entry: purpose (productionize a paused, time-limited run), requires `--resume` (a checkpoint or `__latest__`), rejects `--time-limit`, runs **no** training, writes best-as-final artifacts (adapter, merged, metrics, bundle). Also add a one-line note that the normal `run`/`train` paths now close out on the **best** checkpoint (not last-step).

- [ ] **Step 4: Markdown-lint the doc**

Run the markdownlint gate on `docs/config-schema.md`.
Expected: clean exit (0). Fix findings before committing.

- [ ] **Step 5: Commit**

```bash
git add tests/cli/test_finalize.py docs/config-schema.md
git commit -m "feat(cli): finalize __latest__ resolution + run --finalize docs (#197)"
```

## Task 20: Phase 3 verification gate + open the PR

**Files:** none (verification + PR).

- [ ] **Step 1: Lint + format-check + type**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft`
Expected: clean.

- [ ] **Step 2: Full gated suite**

Run: `uv run pytest`
Expected: PASS including `--cov-fail-under=80`.

- [ ] **Step 3: Markdown-lint all touched docs + this plan + the spec**

Run the markdownlint gate on `docs/config-schema.md docs/defaults-provenance.md docs/superpowers/plans/2026-05-30-plateau-response-ladder-plan.md docs/superpowers/specs/2026-05-30-plateau-response-ladder-design.md`.
Expected: clean exit (0).

- [ ] **Step 4: Open the PR**

```bash
gh pr create --assignee @me --label enhancement \
  --title "Plateau-response ladder: LR decay before early stop; best-as-final close-out (#197)" \
  --body "Implements docs/superpowers/specs/2026-05-30-plateau-response-ladder-design.md (plan: docs/superpowers/plans/2026-05-30-plateau-response-ladder-plan.md). Closes #197."
```

---

## Acceptance criteria → task mapping (spec §16)

| # | Criterion | Satisfied by |
|---|---|---|
| 1 | Early stop on by default; best restored as final adapter on stop | Task 1 (`early_stop.enabled=true`), Task 7 (stop signal), Task 13 (close_out on stop) |
| 2 | LR-decay rung via `ReduceLROnPlateau` in the new `plateau` mode | Task 1 (config), Task 4 (build + step), Task 3/7 (rung-1 cut) |
| 3 | Two-counter semantics (rung 1 reset-on-cut; rung 2 reset-only-on-improvement) | Task 3 (`LadderState`), test_ladder.py |
| 4 | `plateau` is the default; cosine/linear/constant remain; no-val → cosine fallback | Task 1 (default flip), Task 6 (val-fallback) |
| 5 | Best-as-final `close_out` (restore best/, eval, write adapter+merged+metrics) on stop/completion/finalize; last-step fallback | Task 12 (`close_out`), Task 13 (wire), Task 18 (finalize) |
| 6 | `run --finalize --resume <ckpt>` productionizes a paused run, no training | Task 18, Task 19 |
| 7 | Resume persists ladder state + re-seeds `_best_metric_value` from `best.json` (clobber fix) | Task 10 (persist), Task 11 (re-seed + scheduler_kind) |
| 8 | All new defaults cited or `# tbd:`-tagged, each with a doc row | Task 1 (`# cite:`/`# tbd:` comments), Task 16 (provenance rows) |
| 9 | Resume, OOM ladder, eval seams stay green; time-limit stays a pure pause | Task 17 (#198 + OOM non-regression), eval-tick inside `try` (Task 7) |
| 10 | Docs updated for new knobs + `--finalize` CLI | Task 16 (config/provenance), Task 19 (CLI docs) |

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batched with checkpoints.

Phase boundaries are explicit handoff points: after Phase 1's gate, after Phase 2's gate, and Phase 3 opens the PR.
