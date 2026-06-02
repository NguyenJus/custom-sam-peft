# Local Disk Tracker + Resume-Dir Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stdlib-only `local` disk tracker, make it the default backend, fix the resume run-dir bug, and move `tensorboard` back to an opt-in extra.

**Architecture:** Three sequential, independently reviewable phases. Phase 1 fixes `run_training` so resume reuses the old run dir and the trainer preserves the original `config.yaml`. Phase 2 adds `LocalTracker` (metrics-only, writes `metrics.jsonl`), a `wants_images` capability flag on the `Tracker` protocol and all backends, and a trainer panel-render gate keyed on that flag. Phase 3 flips schema/template defaults to `local`, moves `tensorboard` to a `[tensorboard]` extra with a friendly `ImportError` guard.

**Tech Stack:** Python 3.12, pydantic schema, pytest, ruff, mypy. Tracker subsystem under `src/custom_sam_peft/tracking/`; training entry under `src/custom_sam_peft/train/`.

**Spec:** `docs/superpowers/specs/2026-06-01-local-disk-tracker-design.md`

---

## Conventions used in every phase

These commands are referenced by name in each task's verification steps. Copy-paste exactly.

- **Lint (two separate gates — both must pass; CI runs both):**

  ```bash
  uv run ruff check
  uv run ruff format --check
  ```

- **Type check (the exact CI invocation):**

  ```bash
  uv run mypy src/custom_sam_peft
  ```

- **CPU tests (bypass the global `--cov-fail-under=80`; `--no-cov` does NOT work on this box, `-o "addopts="` does):**

  ```bash
  uv run pytest -o "addopts=" <test-path> -v
  ```

- **Do NOT** run `pytest --cov` locally — it segfaults torch's C-extension on this box; trust CI for coverage.
- **Do NOT** run the full GPU suite ad hoc. All new tests in this plan are CPU-only. GPU-marked tests (none added here) go through `scripts/run_gpu_tests.sh`.
- The `backend` default change (`tensorboard` -> `local`) is a behavior default, not a hyperparameter, so no citation tag is required.

---

## File Structure

Files created or modified across the three phases:

- `src/custom_sam_peft/train/runner.py` — resume reuses old run dir (Phase 1).
- `src/custom_sam_peft/train/trainer.py` — skip `config.yaml` overwrite on resume; gate panel render on `wants_images` (Phases 1, 2).
- `src/custom_sam_peft/tracking/base.py` — add `wants_images: bool` protocol member (Phase 2).
- `src/custom_sam_peft/tracking/local.py` — **new** `LocalTracker` + `build_local` factory (Phase 2).
- `src/custom_sam_peft/tracking/noop.py` — add `wants_images = False` (Phase 2).
- `src/custom_sam_peft/tracking/tensorboard.py` — add `wants_images = True`; add `ImportError` guard (Phases 2, 3).
- `src/custom_sam_peft/tracking/wandb.py` — add `wants_images = True` (Phase 2).
- `src/custom_sam_peft/tracking/__init__.py` — add `local` dispatch branch + comment (Phase 2).
- `src/custom_sam_peft/config/schema.py` — add `"local"` to `TrackerBackend` (Phase 2); change default to `"local"` (Phase 3).
- `src/custom_sam_peft/cli/templates/config_full.yaml` — `backend: tensorboard` -> `backend: local` (Phase 3).
- `pyproject.toml` — move `tensorboard>=2.18` to a `[tensorboard]` extra (Phase 3).
- Tests: `tests/unit/test_train_runner.py`, `tests/unit/test_tracking_local.py` (**new**), `tests/unit/test_tracking_build.py`, `tests/unit/test_tracking_protocol.py`, `tests/integration/test_tracker_swap.py`, plus a trainer panel-gating test.

---

## Phase 1 — Resume-dir bug fix

Scope: `runner.py` run-dir reuse on resume + `trainer.py` `config.yaml` skip-if-exists, plus tests.

**Interface exposed at this phase boundary (Phase 2/3 may rely on it without re-reading the code):**

- `run_training(cfg, resume_from=<old_run>/checkpoints/step_N)` reuses `resume_from.parent.parent` as `run_dir`; the returned `EvalArtifacts.run_dir` equals that resume run dir. Fresh runs (`resume_from is None`) still mint a new timestamped dir via `make_run_dir(cfg)`.
- `Trainer.fit` no longer overwrites an existing `run_dir/config.yaml`; on a fresh run it still writes it.

### Task 1.1: Resume reuses the old run dir in `run_training`

**Files:**

- Modify: `src/custom_sam_peft/train/runner.py` (the `run_training` function body, around line 87).
- Test: `tests/unit/test_train_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_train_runner.py` (it already imports `run_training`, `make_run_dir`, `MagicMock`, `pytest`, `Path`, and has the tiny-COCO + stub fixtures). This test asserts resume reuses the old run dir and does not mint a new stamped dir.

```python
def test_run_training_resume_reuses_run_dir(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume must continue in the old run dir, not mint a new stamped dir.

    Spec Change 1: run_training reuses resume_from.parent.parent on resume.
    Note: tests/integration/test_train_resume.py drives trainer.fit with an
    explicit run_dir, bypassing run_training, so it is unaffected by this fix.
    """
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrackingConfig,
        TrainConfig,
        TrainHyperparams,
        ValSplitConfig,
    )
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    def _cfg() -> TrainConfig:
        return TrainConfig(
            run=RunConfig(name="resumedir", output_dir=str(tmp_path), seed=0),
            data=DataConfig(
                format="coco",
                train=DataSplit(
                    annotations=str(tiny_coco_dir / "annotations.json"),
                    images=str(tiny_coco_dir / "images"),
                ),
                val=None,
                val_split=ValSplitConfig(fraction=0.5, seed=None),
            ),
            peft=PEFTConfig(
                method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
            ),
            train=TrainHyperparams(
                epochs=1,
                batch_size=1,
                grad_accum_steps=1,
                save_every=1,
                log_every=1,
                warmup_steps=0,
                num_workers=0,
            ),
            tracking=TrackingConfig(backend="none"),
        )

    monkeypatch.setattr(
        "custom_sam_peft.train.runner.load_sam31",
        lambda _m, **_kw: make_stub_wrapper(dim=8, working=True),
    )

    r1 = run_training(_cfg())
    ckpts = sorted((r1.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "first run produced no checkpoint"

    dirs_before = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    r2 = run_training(_cfg(), resume_from=ckpts[0])
    dirs_after = {p.name for p in tmp_path.iterdir() if p.is_dir()}

    assert r2.run_dir == r1.run_dir, "resume must reuse the original run dir"
    assert dirs_after == dirs_before, "resume must not mint a new stamped run dir"
```

- [ ] **Step 2: Run the test to verify it fails**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_train_runner.py::test_run_training_resume_reuses_run_dir -v
  ```

  Expected: FAIL — `r2.run_dir` is a new stamped dir (assertion `r2.run_dir == r1.run_dir` fails) and a new dir appears in `dirs_after`.

- [ ] **Step 3: Implement the fix in `run_training`**

In `src/custom_sam_peft/train/runner.py`, replace the unconditional `run_dir = make_run_dir(cfg)` (currently line 87) so resume reuses the checkpoint-owning dir. The existing `resume_run_dir = resume_from.parent.parent if resume_from is not None else None` (currently line 91) stays. Change:

```python
    run_dir = make_run_dir(cfg)

    # On resume, look for val_source.json in the run dir that owns the
    # checkpoint (checkpoints live at <run_dir>/checkpoints/step_N/).
    resume_run_dir = resume_from.parent.parent if resume_from is not None else None
```

to:

```python
    # On resume, continue in the run dir that owns the checkpoint
    # (checkpoints live at <run_dir>/checkpoints/step_N/), so resumed
    # artifacts (config.yaml, best/, metrics, val_source.json) stay in the
    # original folder. Fresh runs mint a new timestamped dir.
    resume_run_dir = resume_from.parent.parent if resume_from is not None else None
    run_dir = resume_run_dir if resume_run_dir is not None else make_run_dir(cfg)
```

This leaves `resolve_val_source(cfg, run_dir=resume_run_dir)` on the following line unchanged. `make_run_dir` itself is untouched.

- [ ] **Step 4: Run the test to verify it passes**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_train_runner.py::test_run_training_resume_reuses_run_dir -v
  ```

  Expected: PASS.

- [ ] **Step 5: Run the existing runner tests to confirm no regression**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_train_runner.py -v
  ```

  Expected: PASS, including `test_make_run_dir_creates_timestamped_subdir` (fresh-run path unchanged) and `test_run_training_resume_reuses_saved_val_source`.

- [ ] **Step 6: Commit**

  ```bash
  git add src/custom_sam_peft/train/runner.py tests/unit/test_train_runner.py
  git commit -m "fix(#206): run_training reuses old run dir on resume"
  ```

### Task 1.2: Preserve `config.yaml` on resume in the trainer

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` (the `config.yaml` write, currently line 668).
- Test: `tests/integration/test_tracker_swap.py` (reuses its existing `_run_fit` helper to drive `Trainer.fit` twice into the same `run_dir`).

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_tracker_swap.py` (which already has `_run_fit`, `NoopTracker`, `Path`, `pytest`, and `pytestmark = pytest.mark.integration`). It writes a sentinel `config.yaml`, runs `fit` into that dir, and asserts the sentinel survives.

```python
def test_fit_preserves_existing_config_yaml(tmp_path: Path) -> None:
    """Trainer.fit must NOT overwrite an existing run_dir/config.yaml (resume).

    Spec Change 1 (config.yaml preservation): on resume into an existing dir
    the original config.yaml is preserved.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    sentinel = "SENTINEL_ORIGINAL_CONFIG\n"
    (run_dir / "config.yaml").write_text(sentinel)

    _run_fit(NoopTracker(), run_dir=run_dir)

    assert (run_dir / "config.yaml").read_text() == sentinel, (
        "fit() overwrote an existing config.yaml; resume must preserve it"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

  ```bash
  uv run pytest -o "addopts=" tests/integration/test_tracker_swap.py::test_fit_preserves_existing_config_yaml -v
  ```

  Expected: FAIL — the sentinel is overwritten by `yaml.safe_dump(cfg_dict)`.

- [ ] **Step 3: Implement the skip-if-exists guard**

In `src/custom_sam_peft/train/trainer.py`, guard the write at line 668. Change:

```python
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
        self.tracker.start_run(run_dir, cfg_dict, resume_from)
```

to:

```python
        # Skip when config.yaml already exists (resume into an existing run
        # dir per Change 1) so the original run's config is preserved.
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(cfg_dict))
        self.tracker.start_run(run_dir, cfg_dict, resume_from)
```

The `self.tracker.start_run(...)` call is unchanged and still runs in both cases.

- [ ] **Step 4: Run the test to verify it passes**

  ```bash
  uv run pytest -o "addopts=" tests/integration/test_tracker_swap.py::test_fit_preserves_existing_config_yaml -v
  ```

  Expected: PASS.

- [ ] **Step 5: Confirm fresh-run config.yaml still written**

  ```bash
  uv run pytest -o "addopts=" tests/integration/test_tracker_swap.py -v
  ```

  Expected: PASS — the existing swap tests run `fit` into a fresh dir and rely on normal behavior; none should regress.

- [ ] **Step 6: Commit**

  ```bash
  git add src/custom_sam_peft/train/trainer.py tests/integration/test_tracker_swap.py
  git commit -m "fix(#206): preserve config.yaml on resume into existing run dir"
  ```

### Task 1.3: Phase 1 verification gate

- [ ] **Step 1: Lint**

  ```bash
  uv run ruff check
  uv run ruff format --check
  ```

  Expected: both pass with no findings. If `ruff format --check` reports a file, run `uv run ruff format <file>` and re-stage.

- [ ] **Step 2: Type check**

  ```bash
  uv run mypy src/custom_sam_peft
  ```

  Expected: PASS (no new errors). The Phase 1 source edits are control-flow only and add no new types.

- [ ] **Step 3: Run the touched test files**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_train_runner.py tests/integration/test_tracker_swap.py -v
  ```

  Expected: all PASS.

---

## Phase 2 — `local` tracker + capability gate

Scope: new `LocalTracker` module, the `TrackerBackend` literal accepting `"local"` plus `build_tracker` `local` dispatch, `wants_images` on the protocol and all four backends, trainer panel gate, plus unit/parametrized/integration tests. (The literal gains `"local"` here so the backend is selectable and testable this phase; the *default* flips in Phase 3.)

**Depends on Phase 1** (resume dedup assumes `metrics.jsonl` is already in the reused `run_dir`).

**Interface exposed at this phase boundary (Phase 3 may rely on it without re-reading the code):**

- Backend `local` is registered via `@register("tracker", "local")` and resolvable through `build_tracker` when `cfg.tracking.backend == "local"`. Factory: `build_local(cfg: TrainConfig) -> LocalTracker`.
- `Tracker.wants_images: bool` exists as a protocol member; every backend defines it as a class attribute (`local`/`none` -> `False`; `tensorboard`/`wandb` -> `True`).
- `LocalTracker` writes `run_dir/metrics.jsonl`: one JSON object per `log_scalars` call as a single line `{"step": step, "wall_time": time.time(), **finite_values}` + `"\n"`, flushed after each write. `log_images` is a no-op. `close()` is idempotent.

> **Important ordering note:** Adding `wants_images` to the `@runtime_checkable` protocol (Task 2.4) makes `isinstance(obj, Tracker)` return `False` for any object lacking `wants_images`. The in-test `_RecordingTracker` in `tests/integration/test_tracker_swap.py` has a **module-import-time** `assert isinstance(_RecordingTracker(), Tracker)` (line 85) that will break the entire test module's import unless the fake gains `wants_images`. Tasks 2.4 (protocol member), 2.5 (all backends), and 2.6 (test fakes) must land together in the same commit chain before running the full tracking suite. Follow the task order below.

### Task 2.1: `LocalTracker` — fresh-run write path (TDD)

**Files:**

- Create: `src/custom_sam_peft/tracking/local.py`
- Test: `tests/unit/test_tracking_local.py` (**new**)

- [ ] **Step 1: Write the failing tests (fresh-run subset)**

Create `tests/unit/test_tracking_local.py` with the fresh-run behaviors first. Use a `MagicMock` cfg like `tests/unit/test_train_runner.py` does (the tracker only stores `cfg`; it reads nothing from it).

```python
"""Unit tests for LocalTracker — stdlib-only metrics-to-disk tracker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from custom_sam_peft.tracking.local import LocalTracker


def _read_rows(run_dir: Path) -> list[dict]:
    text = (run_dir / "metrics.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _tracker() -> LocalTracker:
    return LocalTracker(MagicMock())


def test_start_run_fresh_creates_metrics_jsonl(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {"some": "config"})
    assert (tmp_path / "metrics.jsonl").is_file()
    assert _read_rows(tmp_path) == []
    t.close()


def test_log_scalars_appends_one_json_line_per_call(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.log_scalars(0, {"loss": 1.5})
    t.log_scalars(1, {"loss": 1.0, "lr": 0.001})
    t.close()
    rows = _read_rows(tmp_path)
    assert len(rows) == 2
    assert rows[0]["step"] == 0
    assert "wall_time" in rows[0]
    assert rows[0]["loss"] == 1.5
    assert rows[1]["step"] == 1
    assert rows[1]["loss"] == 1.0
    assert rows[1]["lr"] == 0.001


def test_log_scalars_filters_non_finite(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.log_scalars(0, {"loss": float("inf"), "bad": float("nan"), "good": 2.0})
    t.close()
    rows = _read_rows(tmp_path)
    assert rows == [{"step": 0, "wall_time": rows[0]["wall_time"], "good": 2.0}] or (
        rows[0]["good"] == 2.0 and "loss" not in rows[0] and "bad" not in rows[0]
    )


def test_log_scalars_before_start_run_raises(tmp_path: Path) -> None:
    t = _tracker()
    with pytest.raises(RuntimeError, match=r"start_run\(\) must be called before log_scalars\(\)"):
        t.log_scalars(0, {"loss": 1.0})


def test_close_is_idempotent(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.close()
    t.close()  # must not raise


def test_log_images_is_noop(tmp_path: Path) -> None:
    t = _tracker()
    t.start_run(tmp_path, {})
    t.log_images(0, {"panel": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()
    # metrics-only: no scalar rows, no extra files written
    assert _read_rows(tmp_path) == []
    assert not (tmp_path / "panels").exists()


def test_wants_images_is_false() -> None:
    assert LocalTracker.wants_images is False
```

- [ ] **Step 2: Run the tests to verify they fail**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_local.py -v
  ```

  Expected: FAIL at import — `ModuleNotFoundError: No module named 'custom_sam_peft.tracking.local'`.

- [ ] **Step 3: Implement `LocalTracker` (fresh + resume + factory)**

Create `src/custom_sam_peft/tracking/local.py`. This implements the full behavior (fresh write, resume dedup, idempotent close, no-op images, factory) so the resume tests in Task 2.2 reuse the same module without re-editing.

```python
"""LocalTracker — stdlib-only metrics-to-disk tracker. Backend "local".

Persists the per-step scalar time-series to ``run_dir/metrics.jsonl`` (one
JSON object per line) using only the standard library. Metrics-only by owner
decision: ``log_images`` is a no-op and ``wants_images`` is False, so the
trainer skips panel-render compute for this backend (see Change 3).
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TrainConfig

if TYPE_CHECKING:
    import numpy as np

_LOG = logging.getLogger(__name__)

_METRICS_FILENAME = "metrics.jsonl"


class LocalTracker:
    """Tracker backend writing scalar rows to run_dir/metrics.jsonl."""

    wants_images = False

    def __init__(self, cfg: TrainConfig) -> None:
        self._cfg = cfg
        self._run_dir: Path | None = None
        self._fh: TextIO | None = None
        self._closed = False

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        self._run_dir = run_dir
        metrics_path = run_dir / _METRICS_FILENAME
        if resume_from is None:
            # Fresh run: create/truncate, then open for append.
            self._fh = metrics_path.open("w")
            return
        # Resume: run_dir is the old run dir (Change 1), so metrics.jsonl
        # already exists. Drop rows the interrupted run logged AFTER its last
        # checkpoint (step >= resume_step) so resume does not duplicate them.
        resume_step = self._parse_resume_step(resume_from)
        if resume_step is not None and metrics_path.is_file():
            kept: list[str] = []
            for line in metrics_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if int(row["step"]) < resume_step:
                        kept.append(line)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    # Preserve unparseable lines defensively.
                    kept.append(line)
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        self._fh = metrics_path.open("a")

    @staticmethod
    def _parse_resume_step(resume_from: Path) -> int | None:
        """Parse N from a checkpoint dir name of the form ``step_<N>``.

        Returns None (warn + plain append, no dedup) when the name does not
        match — defensive; never crashes the run.
        """
        name = resume_from.name
        prefix = "step_"
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
        _LOG.warning(
            "LocalTracker: resume_from name %r does not match 'step_<N>'; "
            "appending to metrics.jsonl without dedup.",
            name,
        )
        return None

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        if self._fh is None:
            raise RuntimeError("start_run() must be called before log_scalars()")
        finite = {k: v for k, v in values.items() if math.isfinite(v)}
        row = {"step": step, "wall_time": time.time(), **finite}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        # Metrics-only: no-op. Never called for "local" because of the
        # wants_images gate in the trainer (Change 3).
        return None

    def close(self) -> None:
        if self._closed:
            return
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._closed = True


@register("tracker", "local")
def build_local(cfg: TrainConfig) -> LocalTracker:
    """Factory called by build_tracker for backend='local'."""
    return LocalTracker(cfg)
```

- [ ] **Step 4: Run the fresh-run tests to verify they pass**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_local.py -v
  ```

  Expected: PASS for all tests authored in Step 1 (resume tests are added next).

- [ ] **Step 5: Commit**

  ```bash
  git add src/custom_sam_peft/tracking/local.py tests/unit/test_tracking_local.py
  git commit -m "feat(#206): add LocalTracker metrics-to-disk backend"
  ```

### Task 2.2: `LocalTracker` resume dedup + fallback (TDD)

**Files:**

- Test: `tests/unit/test_tracking_local.py` (extend)
- (Implementation already present from Task 2.1, Step 3.)

- [ ] **Step 1: Write the failing resume tests**

Append to `tests/unit/test_tracking_local.py`:

```python
def _write_rows(run_dir: Path, steps: list[int]) -> None:
    lines = [json.dumps({"step": s, "wall_time": 1.0, "loss": float(s)}) for s in steps]
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def test_resume_keeps_only_rows_before_resume_step(tmp_path: Path) -> None:
    # Interrupted run logged steps 0..5; last checkpoint was step_4.
    _write_rows(tmp_path, [0, 1, 2, 3, 4, 5])
    ckpt = tmp_path / "checkpoints" / "step_4"
    ckpt.mkdir(parents=True)

    t = _tracker()
    t.start_run(tmp_path, {}, resume_from=ckpt)
    # Re-walk re-logs step 4 onward; appends must not duplicate prior steps.
    t.log_scalars(4, {"loss": 4.0})
    t.log_scalars(5, {"loss": 5.0})
    t.close()

    steps = [r["step"] for r in _read_rows(tmp_path)]
    assert steps == [0, 1, 2, 3, 4, 5], f"expected dedup to keep <4 then re-append; got {steps}"


def test_resume_fallback_when_name_not_step_n(tmp_path: Path) -> None:
    _write_rows(tmp_path, [0, 1, 2])
    ckpt = tmp_path / "checkpoints" / "latest"  # does NOT match step_<N>
    ckpt.mkdir(parents=True)

    t = _tracker()
    t.start_run(tmp_path, {}, resume_from=ckpt)  # warns, appends without dedup
    t.log_scalars(3, {"loss": 3.0})
    t.close()

    steps = [r["step"] for r in _read_rows(tmp_path)]
    assert steps == [0, 1, 2, 3], f"fallback must retain all existing rows; got {steps}"
```

- [ ] **Step 2: Run to verify they pass (implementation already exists)**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_local.py -v
  ```

  Expected: PASS for both new resume tests and all earlier `LocalTracker` tests. If `test_resume_keeps_only_rows_before_resume_step` fails, the dedup predicate is wrong — it must keep `step < resume_step` and drop `step >= resume_step`.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/unit/test_tracking_local.py
  git commit -m "test(#206): LocalTracker resume dedup + fallback"
  ```

### Task 2.3: Make `local` selectable — `TrackerBackend` literal + `build_tracker` dispatch (TDD)

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (line 97 `TrackerBackend` literal — value only, default unchanged).
- Modify: `src/custom_sam_peft/tracking/__init__.py`
- Test: `tests/unit/test_tracking_build.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tracking_build.py` (mirrors `test_build_tracker_returns_noop`):

```python
def test_build_tracker_returns_local(tmp_path: Path) -> None:
    from custom_sam_peft.tracking import build_tracker

    t = build_tracker(_cfg(tmp_path, "local"))
    assert type(t).__name__ == "LocalTracker"
```

- [ ] **Step 2: Run the test to verify it fails**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py::test_build_tracker_returns_local -v
  ```

  Expected: FAIL — `_cfg(tmp_path, "local")` raises a pydantic `ValidationError` because `TrackerBackend` does not yet include `"local"`. (The `# type: ignore[arg-type]` in `_cfg` only silences mypy; pydantic still validates the `Literal` at runtime.) Both the schema literal and the dispatch branch are added below.

- [ ] **Step 3: Add `"local"` to the `TrackerBackend` literal (default unchanged)**

In `src/custom_sam_peft/config/schema.py` line 97, change:

```python
TrackerBackend = Literal["tensorboard", "wandb", "none"]
```

to:

```python
TrackerBackend = Literal["local", "tensorboard", "wandb", "none"]
```

Leave `TrackingConfig.backend`'s default as `"tensorboard"` for now — the default flips to `"local"` in Phase 3 (Task 3.1). This step only makes `"local"` a *valid, selectable* value so `build_tracker` can resolve it and the Phase 2 tests can construct `backend="local"` configs.

- [ ] **Step 4: Add the `local` dispatch branch**

In `src/custom_sam_peft/tracking/__init__.py`, update the comment on the `backend = ...` line and add the `local` branch before the `tensorboard` branch. Change:

```python
    backend = cfg.tracking.backend  # Literal["tensorboard", "wandb", "none"]
    if backend == "tensorboard":
        from custom_sam_peft.tracking import tensorboard as _tb  # noqa: F401
    elif backend == "wandb":
```

to:

```python
    backend = cfg.tracking.backend  # Literal["local", "tensorboard", "wandb", "none"]
    if backend == "local":
        from custom_sam_peft.tracking import local as _local  # noqa: F401
    elif backend == "tensorboard":
        from custom_sam_peft.tracking import tensorboard as _tb  # noqa: F401
    elif backend == "wandb":
```

- [ ] **Step 5: Run the test to verify it passes**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py -v
  ```

  Expected: PASS for `test_build_tracker_returns_local`; existing build tests still pass.

- [ ] **Step 6: Commit**

  ```bash
  git add src/custom_sam_peft/config/schema.py src/custom_sam_peft/tracking/__init__.py tests/unit/test_tracking_build.py
  git commit -m "feat(#206): make local a selectable tracking backend"
  ```

### Task 2.4: Add `wants_images` to the `Tracker` protocol

> Blast-radius task. Adding a member to the `@runtime_checkable` protocol changes `isinstance` semantics for every conforming object. Steps 3–4 below (all backends + test fakes) must accompany this change so the test suite imports cleanly.

**Files:**

- Modify: `src/custom_sam_peft/tracking/base.py`

- [ ] **Step 1: Add the protocol member**

In `src/custom_sam_peft/tracking/base.py`, add a `wants_images: bool` member inside the `Tracker` protocol body, after the docstring and before `start_run`:

```python
@runtime_checkable
class Tracker(Protocol):
    """Minimal logging contract that every backend must implement.

    Lifecycle: ``__init__`` → ``start_run`` → ``log_*`` … → ``close``.
    """

    #: Whether the backend consumes image panels. The trainer skips the
    #: per-eval panel-render forward pass entirely when this is False.
    wants_images: bool

    def start_run(
```

- [ ] **Step 2: Type check the protocol change alone**

  ```bash
  uv run mypy src/custom_sam_peft
  ```

  Expected: this may surface no error yet (protocol members are structural), but DO NOT commit until backends define the attribute (Task 2.5) — otherwise `isinstance` conformance tests and the swap module import break. Proceed directly to Task 2.5 before running the tracking test suite.

### Task 2.5: Define `wants_images` on all four backends

**Files:**

- Modify: `src/custom_sam_peft/tracking/noop.py`
- Modify: `src/custom_sam_peft/tracking/tensorboard.py`
- Modify: `src/custom_sam_peft/tracking/wandb.py`
- (`local.py` already sets `wants_images = False` from Task 2.1.)

- [ ] **Step 1: `NoopTracker` -> False**

In `src/custom_sam_peft/tracking/noop.py`, add the class attribute at the top of the class body:

```python
class NoopTracker:
    """Tracker that drops all calls on the floor."""

    wants_images = False

    def start_run(
```

- [ ] **Step 2: `TensorBoardTracker` -> True**

In `src/custom_sam_peft/tracking/tensorboard.py`, add the class attribute at the top of the class body (above `__init__`):

```python
class TensorBoardTracker:
    """Tracker backend writing to TensorBoard event files under run_dir."""

    wants_images = True

    def __init__(self, cfg: TrainConfig) -> None:
```

- [ ] **Step 3: `WandBTracker` -> True**

In `src/custom_sam_peft/tracking/wandb.py`, add the class attribute at the top of the class body (above `__init__`):

```python
class WandBTracker:
    """Tracker backend writing to Weights & Biases."""

    wants_images = True

    def __init__(self, cfg: TrainConfig) -> None:
```

- [ ] **Step 4: Do NOT run the tracking suite yet** — `tests/integration/test_tracker_swap.py` and `tests/unit/test_tracking_protocol.py` still reference fakes without `wants_images`. Proceed to Task 2.6, then verify together.

### Task 2.6: Update test fakes + conformance tests for `wants_images` (blast radius)

> The spec flags this explicitly: `tests/integration/test_tracker_swap.py` line 85 has a module-import-time `assert isinstance(_RecordingTracker(), Tracker)` that breaks unless the fake gains `wants_images`. `tests/unit/test_tracking_protocol.py` also constructs in-test classes whose conformance must now account for `wants_images`.

**Files:**

- Modify: `tests/integration/test_tracker_swap.py`
- Modify: `tests/unit/test_tracking_protocol.py`

- [ ] **Step 1: Give `_RecordingTracker` a `wants_images` attribute**

In `tests/integration/test_tracker_swap.py`, add the class attribute at the top of `_RecordingTracker`'s body:

```python
class _RecordingTracker:
    """Tracker that records every protocol call for post-run assertion.

    Satisfies the ``Tracker`` runtime-checkable Protocol so ``isinstance``
    checks pass.  All four methods are implemented; ``log_scalars`` and
    ``log_images`` append their arguments to public lists for inspection.
    """

    wants_images = False

    def __init__(self) -> None:
```

The existing module-level `assert isinstance(_RecordingTracker(), Tracker)` (line 85) now holds again.

- [ ] **Step 2: Add a `wants_images` conformance test + fix the existing incomplete fakes**

In `tests/unit/test_tracking_protocol.py`:

First, update `test_noop_is_a_tracker` is fine as-is (NoopTracker now has `wants_images`). Add a positive `wants_images` assertion and a negative conformance test. Append:

```python
def test_noop_wants_images_is_false() -> None:
    assert NoopTracker.wants_images is False


def test_local_is_a_tracker_and_wants_images_false() -> None:
    from unittest.mock import MagicMock

    from custom_sam_peft.tracking.local import LocalTracker

    t = LocalTracker(MagicMock())
    assert isinstance(t, Tracker)
    assert t.wants_images is False


def test_missing_wants_images_is_not_a_tracker() -> None:
    class Incomplete:
        def start_run(
            self,
            run_dir: Path,
            config: dict[str, Any],
            resume_from: Path | None = None,
        ) -> None:
            pass

        def log_scalars(self, step: int, values: dict[str, float]) -> None:
            pass

        def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
            pass

        def close(self) -> None:
            pass

    assert not isinstance(Incomplete(), Tracker)
```

> Note on the existing `test_missing_start_run_is_not_a_tracker` and `test_missing_close_is_not_a_tracker`: those classes are missing a *method* and were already non-conforming; they remain non-conforming (now also missing `wants_images`), so they still pass. No edit needed there.

- [ ] **Step 3: Run the full tracking test suite now that protocol + backends + fakes are aligned**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_local.py tests/unit/test_tracking_protocol.py tests/unit/test_tracking_build.py tests/integration/test_tracker_swap.py -v
  ```

  Expected: all PASS, including the module-import of `test_tracker_swap.py`.

- [ ] **Step 4: Type check**

  ```bash
  uv run mypy src/custom_sam_peft
  ```

  Expected: PASS.

- [ ] **Step 5: Commit (protocol + backends + fakes together)**

  ```bash
  git add src/custom_sam_peft/tracking/base.py src/custom_sam_peft/tracking/noop.py \
    src/custom_sam_peft/tracking/tensorboard.py src/custom_sam_peft/tracking/wandb.py \
    tests/integration/test_tracker_swap.py tests/unit/test_tracking_protocol.py
  git commit -m "feat(#206): add wants_images capability flag to Tracker protocol + backends"
  ```

### Task 2.7: Add `wants_images` per-backend assertions to the build tests

**Files:**

- Modify: `tests/unit/test_tracking_build.py`

- [ ] **Step 1: Extend the build tests with capability assertions**

In `tests/unit/test_tracking_build.py`, add `wants_images` assertions. Extend the existing per-backend tests rather than duplicating construction.

For `test_build_tracker_returns_local` (added in Task 2.3), add:

```python
    assert t.wants_images is False
```

For `test_build_tracker_returns_noop`, add after the type assertion:

```python
    assert t.wants_images is False
```

For `test_build_tracker_returns_tensorboard`, add after the type assertion:

```python
    assert t.wants_images is True
```

For `test_build_tracker_returns_wandb`, add after the type assertion:

```python
    assert t.wants_images is True
```

- [ ] **Step 2: Run the build tests**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py -v
  ```

  Expected: all PASS. (`test_build_tracker_returns_tensorboard` skips if the `tensorboard` package is absent via the existing `pytest.importorskip("tensorboard")`.)

- [ ] **Step 3: Commit**

  ```bash
  git add tests/unit/test_tracking_build.py
  git commit -m "test(#206): assert wants_images per tracker backend"
  ```

### Task 2.8: Gate `_log_image_panel` on `wants_images` in the trainer (TDD)

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` (the unconditional call at line 547).
- Test: `tests/integration/test_tracker_swap.py` (reuses `_run_fit`, `_RecordingTracker`).

- [ ] **Step 1: Write the failing gating tests**

Append to `tests/integration/test_tracker_swap.py`. These monkeypatch `Trainer._log_image_panel` with a spy and assert it is invoked iff `tracker.wants_images`. `_RecordingTracker` has `wants_images = False`.

```python
def test_panel_render_skipped_when_wants_images_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wants_images=False (e.g. local/none/recording): no panel forward pass."""
    calls: list[int] = []

    def _spy(self: Any, val_examples: Any, class_names: Any, global_step: int) -> None:
        calls.append(global_step)

    monkeypatch.setattr(Trainer, "_log_image_panel", _spy)
    _run_fit(_RecordingTracker(), run_dir=tmp_path / "run")  # wants_images = False
    assert calls == [], "panel render must be skipped for a wants_images=False tracker"


def test_panel_render_invoked_when_wants_images_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wants_images=True: the panel forward pass runs."""

    class _ImageWantingTracker(_RecordingTracker):
        wants_images = True

    calls: list[int] = []

    def _spy(self: Any, val_examples: Any, class_names: Any, global_step: int) -> None:
        calls.append(global_step)

    monkeypatch.setattr(Trainer, "_log_image_panel", _spy)
    _run_fit(_ImageWantingTracker(), run_dir=tmp_path / "run")
    assert calls, "panel render must be invoked for a wants_images=True tracker"
```

- [ ] **Step 2: Run to verify they fail**

  ```bash
  uv run pytest -o "addopts=" tests/integration/test_tracker_swap.py::test_panel_render_skipped_when_wants_images_false tests/integration/test_tracker_swap.py::test_panel_render_invoked_when_wants_images_true -v
  ```

  Expected: `test_panel_render_skipped_when_wants_images_false` FAILS (panel currently called unconditionally, so `calls` is non-empty); `test_panel_render_invoked_when_wants_images_true` passes already (call is unconditional today).

- [ ] **Step 3: Add the gate in the trainer**

In `src/custom_sam_peft/train/trainer.py`, change the unconditional call at line 547:

```python
        self._log_image_panel(val_examples, class_names, step)
```

to:

```python
        if self.tracker.wants_images:
            self._log_image_panel(val_examples, class_names, step)
```

- [ ] **Step 4: Run the gating tests to verify both pass**

  ```bash
  uv run pytest -o "addopts=" tests/integration/test_tracker_swap.py::test_panel_render_skipped_when_wants_images_false tests/integration/test_tracker_swap.py::test_panel_render_invoked_when_wants_images_true -v
  ```

  Expected: both PASS.

- [ ] **Step 5: Run the whole swap module to confirm no regression**

  ```bash
  uv run pytest -o "addopts=" tests/integration/test_tracker_swap.py -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/custom_sam_peft/train/trainer.py tests/integration/test_tracker_swap.py
  git commit -m "feat(#206): gate panel render on tracker.wants_images"
  ```

### Task 2.9: End-to-end `local` integration test (tiny-COCO)

**Files:**

- Test: `tests/unit/test_train_runner.py` (extend — it already has tiny-COCO + stub fixtures and runs `run_training` end-to-end on CPU).

- [ ] **Step 1: Write the end-to-end test**

Append to `tests/unit/test_train_runner.py`:

```python
def test_run_training_local_backend_writes_metrics_jsonl(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 1: backend=local runs end-to-end, writes metrics.jsonl, no panels."""
    import json

    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrackingConfig,
        TrainConfig,
        TrainHyperparams,
        ValSplitConfig,
    )
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    cfg = TrainConfig(
        run=RunConfig(name="localrun", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            val_split=ValSplitConfig(fraction=0.5, seed=None),
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            save_every=1,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
        tracking=TrackingConfig(backend="local"),
    )
    monkeypatch.setattr(
        "custom_sam_peft.train.runner.load_sam31",
        lambda _m, **_kw: make_stub_wrapper(dim=8, working=True),
    )

    result = run_training(cfg)
    metrics_path = result.run_dir / "metrics.jsonl"
    assert metrics_path.is_file()
    rows = [json.loads(ln) for ln in metrics_path.read_text().splitlines() if ln.strip()]
    assert rows, "expected at least one logged scalar row"
    assert all("step" in r and "wall_time" in r for r in rows)
    assert not (result.run_dir / "panels").exists(), "metrics-only: no panels dir"
```

> This test requires `"local"` to be accepted by `TrackerBackend`, which Task 2.3 adds in this same phase. It goes green as soon as Task 2.3 has landed — run it at the Phase 2 gate (Task 2.10).

- [ ] **Step 2: Commit**

  ```bash
  git add tests/unit/test_train_runner.py
  git commit -m "test(#206): end-to-end local backend writes metrics.jsonl"
  ```

### Task 2.10: Phase 2 verification gate + blast-radius grep

- [ ] **Step 1: Grep for every Tracker construction / conformance site**

Repo lesson: a protocol/required-member change breaks consumers beyond the named files. Confirm no other object constructs a `Tracker`-conforming class or asserts `isinstance(..., Tracker)` without `wants_images`.

  ```bash
  uv run grep -rn "isinstance(.*Tracker)" src tests
  uv run grep -rn "class .*Tracker" src tests
  uv run grep -rn "build_tracker\|register(\"tracker\"" src tests
  ```

Expected sites and their status:

- `src/custom_sam_peft/tracking/{base,noop,tensorboard,wandb,local}.py` — all backends now define `wants_images` (done).
- `tests/integration/test_tracker_swap.py::_RecordingTracker` and its `_ImageWantingTracker` subclass — `wants_images` present (done).
- `tests/unit/test_tracking_protocol.py` `Incomplete` classes — intentionally non-conforming (no change needed).
- No production code elsewhere constructs a tracker outside `build_tracker`.

If the grep reveals any *other* class implementing the four tracker methods or any other `isinstance(..., Tracker)` assertion, add `wants_images` to that class in the same phase.

- [ ] **Step 2: Lint**

  ```bash
  uv run ruff check
  uv run ruff format --check
  ```

  Expected: both pass.

- [ ] **Step 3: Type check**

  ```bash
  uv run mypy src/custom_sam_peft
  ```

  Expected: PASS.

- [ ] **Step 4: Run the full set of touched test dirs (CPU)**

  ```bash
  uv run pytest -o "addopts=" \
    tests/unit/test_tracking_local.py \
    tests/unit/test_tracking_protocol.py \
    tests/unit/test_tracking_build.py \
    tests/unit/test_train_runner.py \
    tests/integration/test_tracker_swap.py -v
  ```

  Expected: all PASS, including `test_run_training_local_backend_writes_metrics_jsonl` (Task 2.9) — now green because Task 2.3 added the `"local"` literal this phase.

---

## Phase 3 — Defaults + dependency move

Scope: schema default `local`, template default, `pyproject.toml` `tensorboard` -> `[tensorboard]` extra, `TensorBoardTracker` `ImportError` guard, recorded decision, plus schema/template assertions.

**Depends on Phase 2** (the default must resolve to a registered, working backend).

**Interface exposed at this phase boundary:**

- `TrackingConfig().backend == "local"`; `TrackerBackend` accepts `"local" | "tensorboard" | "wandb" | "none"`.
- Selecting `tensorboard` without the extra raises a friendly `ImportError` naming the `[tensorboard]` extra at `build_tracker` time.

### Task 3.1: Make `local` the schema default (TDD)

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (line 645 default; the `TrackerBackend` literal already gained `"local"` in Task 2.3).
- Test: `tests/unit/test_tracking_build.py` (schema default assertion).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tracking_build.py`:

```python
def test_tracking_config_default_is_local() -> None:
    from custom_sam_peft.config.schema import TrackingConfig

    assert TrackingConfig().backend == "local"
```

- [ ] **Step 2: Run to verify it fails**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py::test_tracking_config_default_is_local -v
  ```

  Expected: FAIL — default is currently `"tensorboard"`.

- [ ] **Step 3: Update the schema default**

In `src/custom_sam_peft/config/schema.py`, change `TrackingConfig.backend`'s default (line 645). The `TrackerBackend` literal already includes `"local"` from Task 2.3, so only the default changes here. Change:

```python
    backend: TrackerBackend = "tensorboard"
```

to:

```python
    backend: TrackerBackend = "local"
```

- [ ] **Step 4: Run to verify it passes**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py -v
  ```

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/custom_sam_peft/config/schema.py tests/unit/test_tracking_build.py
  git commit -m "feat(#206): default tracking backend to local"
  ```

### Task 3.2: Make `local` the template default (TDD)

**Files:**

- Modify: `src/custom_sam_peft/cli/templates/config_full.yaml` (line 77).
- Test: locate the existing template-render test and add an assertion. First find it.

- [ ] **Step 1: Find the existing template/render test**

  ```bash
  uv run grep -rln "config_full\|render.*template\|backend:" tests
  ```

Expected: a CLI/init test that renders `config_full.yaml`. If a render test exists, extend it; otherwise add a focused unit test that reads the rendered template and asserts `backend: local`.

- [ ] **Step 2: Write/extend the failing test**

If a render test is found (e.g. under `tests/unit/` or `tests/integration/` covering `csp init`), add an assertion that the rendered output contains `backend: local`. If none exists, create `tests/unit/test_config_template_default.py`:

```python
"""The config_full.yaml template defaults tracking.backend to local."""

from __future__ import annotations

from importlib import resources


def test_config_full_template_defaults_backend_local() -> None:
    text = (
        resources.files("custom_sam_peft.cli.templates")
        .joinpath("config_full.yaml")
        .read_text()
    )
    assert "backend: local" in text
    assert "backend: tensorboard" not in text
```

- [ ] **Step 3: Run to verify it fails**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_config_template_default.py -v
  ```

  Expected: FAIL — template still says `backend: tensorboard`.

- [ ] **Step 4: Edit the template**

In `src/custom_sam_peft/cli/templates/config_full.yaml`, change line 77:

```yaml
tracking:
  backend: tensorboard
```

to:

```yaml
tracking:
  backend: local
```

- [ ] **Step 5: Run to verify it passes**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_config_template_default.py -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/custom_sam_peft/cli/templates/config_full.yaml tests/unit/test_config_template_default.py
  git commit -m "feat(#206): config_full template defaults to backend local"
  ```

### Task 3.3: Move `tensorboard` to a `[tensorboard]` extra

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Remove `tensorboard` from base dependencies**

In `pyproject.toml`, delete the base-dependency line (currently line 28):

```python
  "tensorboard>=2.18",
```

from the `[project] dependencies` list. Leave the `sam3 @ git+...` line directly below it intact.

- [ ] **Step 2: Add the `[tensorboard]` extra**

In `[project.optional-dependencies]`, add a `tensorboard` extra alongside the existing `wandb` extra. Change:

```python
[project.optional-dependencies]
wandb = ["wandb>=0.18"]
qlora = ["bitsandbytes>=0.43"]
```

to:

```python
[project.optional-dependencies]
wandb = ["wandb>=0.18"]
tensorboard = ["tensorboard>=2.18"]
qlora = ["bitsandbytes>=0.43"]
```

- [ ] **Step 3: Re-sync the dev environment with the dev extra**

> Repo lesson: a bare `uv sync` prunes pytest. Sync with the dev extra so the test tools survive. The `tensorboard` package may stay installed from a prior sync; that is fine — tests that need it use `pytest.importorskip`.

  ```bash
  uv sync --extra dev
  ```

  Expected: completes; pytest/ruff/mypy remain available.

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml
  git commit -m "build(#206): move tensorboard to an opt-in [tensorboard] extra"
  ```

### Task 3.4: Friendly `ImportError` guard in `TensorBoardTracker.__init__` (TDD)

**Files:**

- Modify: `src/custom_sam_peft/tracking/tensorboard.py` (`__init__`, line 23).
- Test: `tests/unit/test_tracking_build.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tracking_build.py` (mirrors the existing `test_build_tracker_raises_when_wandb_extra_missing`):

```python
def test_build_tracker_raises_when_tensorboard_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the SummaryWriter import to fail at construction time.
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)

    from custom_sam_peft.tracking import build_tracker

    with pytest.raises(ImportError, match=r"\[tensorboard\]"):
        build_tracker(_cfg(tmp_path, "tensorboard"))
```

- [ ] **Step 2: Run to verify it fails**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py::test_build_tracker_raises_when_tensorboard_extra_missing -v
  ```

  Expected: FAIL — today `__init__` does not import `SummaryWriter`, so no `ImportError` is raised at construction (the lazy import only fires in `start_run`). The test sees no `ImportError`.

- [ ] **Step 3: Add the construction-time guard**

In `src/custom_sam_peft/tracking/tensorboard.py`, replace `__init__` (currently lines 23–26). The class attribute `wants_images = True` (added in Task 2.5) stays above it. Change:

```python
    def __init__(self, cfg: TrainConfig) -> None:
        self._cfg = cfg
        self._writer: SummaryWriter | None = None
        self._closed = False
```

to:

```python
    def __init__(self, cfg: TrainConfig) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tracking.backend='tensorboard' requires the [tensorboard] extra. "
                "Install with: pip install 'custom-sam-peft[tensorboard]'"
            ) from e
        self._cfg = cfg
        self._writer: SummaryWriter | None = None
        self._closed = False
```

The existing lazy `from torch.utils.tensorboard import SummaryWriter` inside `start_run` (line 34) stays — the construction guard just front-loads the failure.

- [ ] **Step 4: Run to verify it passes**

  ```bash
  uv run pytest -o "addopts=" tests/unit/test_tracking_build.py -v
  ```

  Expected: PASS — including `test_build_tracker_returns_tensorboard` when the `tensorboard` package is present (skipped via `importorskip` otherwise).

- [ ] **Step 5: Commit**

  ```bash
  git add src/custom_sam_peft/tracking/tensorboard.py tests/unit/test_tracking_build.py
  git commit -m "feat(#206): friendly ImportError guard for missing tensorboard extra"
  ```

### Task 3.5: Phase 3 verification gate

- [ ] **Step 1: Lint**

  ```bash
  uv run ruff check
  uv run ruff format --check
  ```

  Expected: both pass.

- [ ] **Step 2: Type check**

  ```bash
  uv run mypy src/custom_sam_peft
  ```

  Expected: PASS.

- [ ] **Step 3: Run the full set of touched CPU tests**

  ```bash
  uv run pytest -o "addopts=" \
    tests/unit/test_tracking_local.py \
    tests/unit/test_tracking_protocol.py \
    tests/unit/test_tracking_build.py \
    tests/unit/test_train_runner.py \
    tests/unit/test_config_template_default.py \
    tests/integration/test_tracker_swap.py -v
  ```

  Expected: all PASS. (`test_run_training_local_backend_writes_metrics_jsonl` was already green from Phase 2; re-running here confirms the default flip did not regress it.)

- [ ] **Step 4: Sanity-check no other consumer hardcoded the old default**

  ```bash
  uv run grep -rn "backend.*tensorboard\|tensorboard.*default" src tests docs
  ```

  Expected: only intentional references (the `tensorboard` extra, `TensorBoardTracker`, the build test). If a test or doc still asserts the default is `tensorboard`, update it within this phase.

---

## Self-review: spec coverage map

Every spec section maps to a task:

- Change 1 (resume run-dir reuse) -> Task 1.1.
- Change 1 (config.yaml preservation) -> Task 1.2.
- Change 2 (`LocalTracker` + factory + registration) -> Tasks 2.1, 2.2.
- Change 2 (resume dedup + fallback) -> Task 2.2.
- Change 3 (`wants_images` on protocol) -> Task 2.4.
- Change 3 (`wants_images` on all backends) -> Tasks 2.1 (local), 2.5 (noop/tb/wandb).
- Change 3 (trainer panel gate) -> Task 2.8.
- Change 4 (`TrackerBackend` literal accepts `local` + `build_tracker` dispatch + comment) -> Task 2.3.
- Change 4 (schema default `local`) -> Task 3.1.
- Change 4 (template default) -> Task 3.2.
- Change 4 (pyproject `[tensorboard]` extra) -> Task 3.3.
- Change 4 (TensorBoard `ImportError` guard) -> Task 3.4.
- Test plan (new `test_tracking_local.py`) -> Tasks 2.1, 2.2.
- Test plan (backend-parametrized: build / protocol / swap + `wants_images` asserts) -> Tasks 2.3, 2.6, 2.7.
- Test plan (trainer panel-gating) -> Task 2.8.
- Test plan (resume-dir-reuse) -> Task 1.1.
- Test plan (schema/template default assertions) -> Tasks 3.1, 3.2.
- Test plan (end-to-end tiny-COCO `local`) -> Task 2.9 (green in Phase 2).

## Acceptance criteria checklist (spec section "Acceptance criteria")

- [ ] **AC1** — `backend: local` runs end-to-end on tiny-COCO with no heavy deps, produces `metrics.jsonl`, no panel PNGs. -> Task 2.9 (green in Phase 2; re-confirmed at Task 3.5).
- [ ] **AC2** — Resume appends without duplicating steps. -> Task 2.2 (`test_resume_keeps_only_rows_before_resume_step`), resting on Task 1.1.
- [ ] **AC3** — `local` is the schema + template default; `tensorboard`/`wandb`/`none` remain selectable. -> Tasks 3.1, 3.2 (default assertions) + Task 2.3 / existing build tests (others still resolve).
- [ ] **AC4** — Parametrized tracker tests cover `local` alongside existing backends, including `wants_images` assertions. -> Tasks 2.3, 2.6, 2.7.
- [ ] **AC5** — `tensorboard` returns to an opt-in `[tensorboard]` extra with a friendly `ImportError` guard. -> Tasks 3.3, 3.4.
- [ ] **AC6** — Resume-dir bug fixed: resumed run continues in the old folder; `config.yaml` preserved. -> Tasks 1.1, 1.2.

## Notes carried from the spec (do not re-derive)

- `tests/integration/test_train_resume.py` calls `trainer.fit` with an explicit `run_dir`, bypassing `run_training`, so Phase 1 does NOT affect it. Do not "fix" that test.
- `make_run_dir` is unchanged and still unit-tested by `test_make_run_dir_creates_timestamped_subdir`.
- `WandBTracker` has its own resume continuity via `wandb_run_id.txt`; unaffected.
- The `backend` default change is a behavior default, not a hyperparameter — no citation tag required.
- Coverage is unmeasurable locally (torch C-extension segfaults under `--cov`); trust CI for the `--cov-fail-under=80` gate.
