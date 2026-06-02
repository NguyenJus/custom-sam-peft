# Local disk tracker as the default + resume-dir bug fix

Issue: [#206](https://github.com/NguyenJus/custom-sam-peft/issues/206) — "Tiny in-house
disk tracker to become the default (replace none + tensorboard for the common case)".

## Motivation

Today the two default-tracking options are both unsatisfying:

- `none` (`NoopTracker`) drops every `log_scalars`/`log_images` call. You keep
  `metrics.json` (final), `best/best.json`, and the console tail, but you lose the
  persisted time-series: per-step loss/LR/throughput and per-eval mAP-per-IoU.
- `tensorboard` gives you curves but pulls a heavy dependency tree (protobuf,
  grpcio). PR #205 moved `tensorboard` into **base** dependencies purely to make the
  config wizard work out of the box, inflating the base install footprint.

This repo targets ~160-epoch convergence runs, so losing the curves is a real
diagnostic loss. A tiny in-house tracker that persists the time-series to disk using
only the standard library plus already-present base deps gives out-of-the-box
persisted metrics with zero heavy deps. It becomes the new default and lets us drop
`tensorboard` from base deps again (keeping `tensorboard`/`wandb` as opt-in extras).

This spec couples two changes that must land together because the tracker's resume
semantics depend on the dir-reuse fix:

1. A resume-dir bug fix in `run_training`.
2. The `local` tracker feature, made default, with `tensorboard` moved to an extra,
   plus backend-capability-gated image-panel computation.

## Scope

In scope:

- Fix `run_training` to reuse the existing run dir on resume; preserve the original
  `config.yaml`.
- Add a `LocalTracker` (metrics-only, stdlib-only) registered as backend `local`.
- Add a `wants_images: bool` capability flag to the `Tracker` protocol and all four
  backends; gate the image-panel forward pass on it in the trainer.
- Make `local` the schema + template default.
- Move `tensorboard` from base deps to an opt-in `[tensorboard]` extra, with a
  friendly `ImportError` guard.

Out of scope: see [Non-goals](#non-goals).

### Owner decision (divergence from the issue text)

The issue's original wording said the tracker would persist time-series "and panel
PNGs". The repo owner has decided `local` is **metrics-only** — no image panels. This
is recorded as an intentional divergence: `LocalTracker.log_images` is a no-op and the
panel-render compute is skipped entirely for `local` (and `none`) via the capability
gate. Per-epoch qualitative panels are judged not worth the per-eval forward-pass cost
by default; users who want panels select `tensorboard` or `wandb`.

## Detailed design

### Change 1 — Resume-dir bug fix

`run_training` (`src/custom_sam_peft/train/runner.py`) calls `make_run_dir(cfg)`
unconditionally at line 87. On resume this mints a **new** timestamped
`runs/{name}-{UTC-stamp}/` dir and scatters `config.yaml`, `best/`, fresh
`checkpoints/`, and the metric curves there — while the checkpoint is read from the
**old** dir. A resumed run should instead continue in the old folder.

Confirming evidence: `src/custom_sam_peft/cli/run_cmd.py:203` already computes
`run_dir = resume.parent.parent  # checkpoints/step_N -> run_dir`, so downstream code
already assumes resumed artifacts live in the old dir, contradicting the training half.

**Fix (localized to `run_training`):** when `resume_from is not None`, set
`run_dir = resume_from.parent.parent` (the run dir that owns the checkpoint) instead of
calling `make_run_dir(cfg)`. Fresh runs (`resume_from is None`) keep calling
`make_run_dir(cfg)`, unchanged.

The trainer is already run-dir-agnostic: `Trainer.fit(run_dir=..., resume_from=...)`
writes to whatever `run_dir` it is given (the integration test
`tests/integration/test_train_resume.py` already drives `fit` with an explicit
`run_dir`), so no trainer signature change is needed. `train_result.run_dir`
(`EvalArtifacts.run_dir`) flows downstream and is consumed at `run_cmd.py:110`, so the
reuse propagates cleanly.

`make_run_dir` itself is **unchanged** — still used for fresh runs and still unit-tested
at `tests/unit/test_train_runner.py:32`
(`test_make_run_dir_creates_timestamped_subdir`).

Note: the existing `resume_run_dir = resume_from.parent.parent if resume_from is not
None else None` at `runner.py:91` (used for `resolve_val_source`) already derives the
same path; the fix makes `run_dir` consistent with it on resume.

#### config.yaml preservation

`src/custom_sam_peft/train/trainer.py:668` writes
`(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))` on every `fit()`. With
the dir-reuse fix, this would overwrite the original run's `config.yaml` on resume.

**Fix:** guard this write to **skip when `run_dir / "config.yaml"` already exists**
(i.e., on resume into an existing dir), preserving the original `config.yaml`. The
immediately-following `self.tracker.start_run(run_dir, cfg_dict, resume_from)` at
`trainer.py:669` is unchanged and still runs.

### Change 2 — The `local` tracker

New module `src/custom_sam_peft/tracking/local.py`. Implements the `Tracker` protocol
(`src/custom_sam_peft/tracking/base.py`), registers via
`@register("tracker", "local")` on a factory
`build_local(cfg: TrainConfig) -> LocalTracker`. Uses only the standard library
(`json`, `math`, `time`, `pathlib`) — **no heavy deps**. (`numpy`/`pillow` are already
base deps but a metrics-only tracker needs neither.)

The `Tracker` protocol (`base.py`) methods:

- `start_run(run_dir: Path, config: dict[str, Any], resume_from: Path | None = None) -> None`
- `log_scalars(step: int, values: dict[str, float]) -> None`
- `log_images(step: int, images: dict[str, np.ndarray]) -> None`
- `close() -> None`

`base.py` also has `_validate_image(tag, arr)` enforcing uint8 `(H, W, 3)`; the
metrics-only `LocalTracker` does not use it.

#### LocalTracker behavior

`__init__(self, cfg)`:

- Store `cfg`. Initialize state: `_run_dir = None`, file handle `= None`,
  `_closed = False`.

`start_run(run_dir, config, resume_from=None)`:

- Target file is `run_dir / "metrics.jsonl"`.
- **Fresh** (`resume_from is None`): create/truncate `metrics.jsonl`, open it for
  appending.
- **Resume** (`resume_from is not None`): because of Change 1, `run_dir` **is** the old
  run dir, so `metrics.jsonl` already exists there.
  1. Parse `resume_step` as the integer `N` from the checkpoint dir name
     `resume_from.name`, which has the form `step_<N>` (checkpoints live at
     `<run_dir>/checkpoints/step_N/`).
  2. Read the existing `metrics.jsonl`, **keep only rows with `step < resume_step`**,
     rewrite the file with the kept rows, then open it for appending.
  3. This drops rows the interrupted run logged **after** its last checkpoint (steps
     that will be re-walked on resume) so they are not duplicated.
  - **Defensive fallback:** if `resume_from.name` does not match `step_<N>`, log a
    warning and fall back to plain append (no dedup).
- Do **not** touch `config.yaml` — the trainer owns it (`trainer.py:668`). Do not
  re-persist `config` either.

`log_scalars(step, values)`:

- Raise `RuntimeError` if `start_run` was not called first. Match the message used by
  `TensorBoardTracker`/`WandBTracker`:
  `"start_run() must be called before log_scalars()"`.
- Filter out non-finite values with `math.isfinite`, matching the other backends.
- Append one JSON object per call as a single line followed by `"\n"`:
  `{"step": step, "wall_time": time.time(), **finite_values}`.
- **Flush after each write** for crash-safety.
- `wall_time` is absolute unix seconds (`time.time()`), matching TensorBoard's
  `wall_time` semantics.

`log_images(step, images)`:

- **No-op.** `LocalTracker` is metrics-only. Implemented to satisfy the protocol but
  never called for `local` because of the `wants_images` gate (Change 3).

`close()`:

- **Idempotent** (guard via `_closed`): flush and close the file handle, then set
  `_closed = True`.

Capability flag (see Change 3): `LocalTracker.wants_images = False`.

### Change 3 — Image panels gated on backend capability

`src/custom_sam_peft/train/trainer.py:547` calls
`self._log_image_panel(val_examples, class_names, step)` **unconditionally** inside the
eval/checkpoint path. `_log_image_panel` (`trainer.py:857`) runs a model forward pass
**per validation example** to render panels, then calls
`self.tracker.log_images(...)` at `trainer.py:912-913`. So today even `none` pays full
panel-render compute every eval and discards it.

**Design:** add a capability flag `wants_images: bool` to the `Tracker` protocol
(`base.py`). Implementers set it as a class attribute:

| Backend              | `wants_images` |
| -------------------- | -------------- |
| `LocalTracker`       | `False`        |
| `NoopTracker`        | `False`        |
| `TensorBoardTracker` | `True`         |
| `WandBTracker`       | `True`         |

In `trainer.py:547`, gate the call: only invoke `self._log_image_panel(...)` when
`self.tracker.wants_images` is `True`. The default (`local`) and `none` then skip the
panel compute entirely; only image-capable backends (`tensorboard`/`wandb`) trigger it.

No new user-facing config knob is added: behavior follows the backend choice, which
already signals intent (consistent with the repo's preference against speed-only config
knobs).

The `Tracker` protocol is `@runtime_checkable` (`base.py` uses `@runtime_checkable`).
Adding `wants_images` means every backend must define it and the protocol-conformance
test must be updated (see [Test plan](#test-plan)). Declare `wants_images: bool` as a
protocol member so conformance is checked.

### Change 4 — Make `local` the default; move tensorboard to an extra

- `src/custom_sam_peft/config/schema.py:97`:
  `TrackerBackend = Literal["tensorboard", "wandb", "none"]` →
  `Literal["local", "tensorboard", "wandb", "none"]` (add `"local"`).
- `src/custom_sam_peft/config/schema.py:644-645`:
  `class TrackingConfig(_Strict): backend: TrackerBackend = "tensorboard"` → default
  `"local"`.
- `src/custom_sam_peft/tracking/__init__.py` `build_tracker`: add a `local` dispatch
  branch mirroring the existing lazy-import branches:

  ```python
  elif backend == "local":
      from custom_sam_peft.tracking import local as _local  # noqa: F401
  ```

  `lookup("tracker", backend)` then resolves the registered factory. Also update the
  inline comment listing the `Literal` members to include `"local"`.
- `src/custom_sam_peft/cli/templates/config_full.yaml` lines 76-77:
  `tracking:` / `backend: tensorboard` → `backend: local`.
- `pyproject.toml`: move `"tensorboard>=2.18"` out of base `dependencies` (currently
  line 28) into `[project.optional-dependencies]` as
  `tensorboard = ["tensorboard>=2.18"]` (alongside the existing
  `wandb = ["wandb>=0.18"]`). This reverts PR #205's base-footprint cost.
- `src/custom_sam_peft/tracking/tensorboard.py`: add a friendly `ImportError` guard so
  selecting `tensorboard` without the extra fails clearly. Mirror `WandBTracker`'s
  pattern (`wandb.py:26-33`, which guards in `__init__`). The import that fails when the
  extra is absent is `from torch.utils.tensorboard import SummaryWriter` (`torch` is a
  base dep, but the `tensorboard` **package** is what is missing). Guard at construction
  (`TensorBoardTracker.__init__` already exists at `tensorboard.py:23`) for fail-fast at
  `build_tracker` time:

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

  The existing lazy `from torch.utils.tensorboard import SummaryWriter` inside
  `start_run` (`tensorboard.py:34`) stays; the construction guard just front-loads the
  failure.
- Keep `none` (`NoopTracker`) as an explicit no-op.

## File-by-file changes

| File | Change |
| ---- | ------ |
| `src/custom_sam_peft/train/runner.py` | On resume, set `run_dir = resume_from.parent.parent` instead of `make_run_dir(cfg)`; fresh runs unchanged. |
| `src/custom_sam_peft/train/trainer.py` | Skip the `config.yaml` write when it already exists (line 668); gate `_log_image_panel` call on `self.tracker.wants_images` (line 547). |
| `src/custom_sam_peft/tracking/base.py` | Add `wants_images: bool` member to the `Tracker` protocol. |
| `src/custom_sam_peft/tracking/local.py` | **New.** `LocalTracker` + `build_local` factory, `@register("tracker", "local")`, `wants_images = False`. |
| `src/custom_sam_peft/tracking/noop.py` | Add `wants_images = False`. |
| `src/custom_sam_peft/tracking/tensorboard.py` | Add `wants_images = True`; add `ImportError` guard in `__init__`. |
| `src/custom_sam_peft/tracking/wandb.py` | Add `wants_images = True`. |
| `src/custom_sam_peft/tracking/__init__.py` | Add `local` dispatch branch in `build_tracker`; update comment. |
| `src/custom_sam_peft/config/schema.py` | Add `"local"` to `TrackerBackend` (line 97); change `TrackingConfig.backend` default to `"local"` (line 645). |
| `src/custom_sam_peft/cli/templates/config_full.yaml` | `backend: tensorboard` → `backend: local` (line 77). |
| `pyproject.toml` | Move `tensorboard>=2.18` from base deps to a `[tensorboard]` extra. |

Tests are listed in the [Test plan](#test-plan).

## Resume semantics

The dedup contract for `local` rests on Change 1:

1. On resume, `run_training` reuses the old run dir (`resume_from.parent.parent`), so
   `metrics.jsonl` from the interrupted run is already present.
2. The interrupted run may have logged scalar rows **after** its last saved checkpoint
   (`step_N`). On resume the trainer re-walks from `step_N`, so those post-`N` steps
   would be re-logged.
3. `LocalTracker.start_run` parses `N` from `resume_from.name == "step_<N>"`, keeps only
   rows with `step < N`, rewrites the file, and opens for append. Subsequent appends do
   not duplicate steps.
4. If `resume_from.name` does not match `step_<N>`, it logs a warning and appends
   without dedup (defensive; never crashes the run).
5. The original `config.yaml` is preserved by the trainer's skip-if-exists guard.

`WandBTracker` already has its own resume continuity via `wandb_run_id.txt`
(`wandb.py:58-78`) and is unaffected. `tests/integration/test_train_resume.py` drives
`trainer.fit` with an explicit `run_dir`, **bypassing `run_training`**, so it is
unaffected by Change 1.

## Test plan

### New `tests/unit/test_tracking_local.py`

- `start_run` (fresh) creates `metrics.jsonl`.
- `log_scalars` appends exactly one JSON line per call, each containing keys `step`,
  `wall_time`, plus the scalar values.
- Non-finite values (`inf`, `nan`) are filtered out of the written line.
- `log_scalars` before `start_run` raises `RuntimeError` (match
  `"start_run() must be called before log_scalars()"`).
- `close` is idempotent (two calls, no error).
- **Resume:** given an existing `metrics.jsonl` with rows up to `step > resume_step`
  and a `resume_from` path `.../checkpoints/step_<N>`, after `start_run` the file keeps
  only rows with `step < N`, and subsequent appends do not duplicate steps.
- **Resume fallback:** a `resume_from` whose name does not match `step_<N>` logs a
  warning and appends without dedup (existing rows retained).
- `log_images` is a no-op (writes nothing to disk).

### Backend-parametrized tests

Add `"local"` to the parametrized backend coverage, matching each file's existing
parametrization style:

- `tests/unit/test_tracking_build.py`: assert `build_tracker(_cfg(tmp_path, "local"))`
  returns a `LocalTracker` (no extra needed). Mirror the existing per-backend tests
  (`test_build_tracker_returns_noop`, etc.).
- `tests/unit/test_tracking_protocol.py`: assert `LocalTracker()` satisfies the
  `Tracker` protocol; update the conformance tests for the new `wants_images` member
  (an implementer missing `wants_images` should fail `isinstance(..., Tracker)`).
- `tests/integration/test_tracker_swap.py`: add `local` to the swap coverage; ensure the
  in-test `_RecordingTracker` gains a `wants_images` attribute so it still satisfies the
  protocol.

Add `wants_images` assertions per backend: `local` and `none` → `False`;
`tensorboard` and `wandb` → `True`.

### Trainer panel-gating test

- With a `wants_images=False` tracker, `_log_image_panel` is **not** invoked (spy /
  monkeypatch on `_log_image_panel`).
- With a `wants_images=True` tracker, `_log_image_panel` **is** invoked.

### Resume-dir-reuse test

Extend `tests/unit/test_train_runner.py` (or add a focused test):

- `run_training(cfg, resume_from=<old_run>/checkpoints/step_N)` returns `EvalArtifacts`
  whose `run_dir == <old_run>` and does **not** create a new stamped dir.
- Note in the test/comment that `tests/integration/test_train_resume.py` calls
  `trainer.fit` with an explicit `run_dir`, bypassing `run_training`, so it is
  unaffected.

### Schema / template default assertions

- `TrackingConfig().backend == "local"`.
- The rendered `config_full.yaml` template contains `backend: local`.

### End-to-end (integration, tiny-COCO fixture)

- With `backend: local`, a full run on the tiny-COCO fixture produces
  `run_dir/metrics.jsonl` containing the logged scalars.
- No `panels/` directory is produced (metrics-only).
- No heavy deps are required (stdlib + base deps only).

### Test gating notes

Per repo conventions: run CPU-only unit tests with `-o "addopts="` to bypass the global
`--cov-fail-under=80`; do not run the full GPU suite ad hoc — use
`scripts/run_gpu_tests.sh` for GPU-marked tests. New unit tests here are CPU-only.

## Acceptance criteria

1. `backend: local` runs end-to-end on tiny-COCO with no heavy deps and produces
   `metrics.jsonl` with the logged scalars; **no panel PNGs** (metrics-only per owner
   decision — recorded divergence from the issue's "and panel PNGs" wording).
   — [Change 2](#change-2--the-local-tracker), end-to-end test.
2. Resume appends without duplicating steps.
   — [Resume semantics](#resume-semantics), resume test.
3. `local` is the schema + template default; `tensorboard`/`wandb`/`none` remain
   selectable. — [Change 4](#change-4--make-local-the-default-move-tensorboard-to-an-extra).
4. Parametrized tracker tests cover `local` alongside the existing backends, including
   `wants_images` assertions. — [Backend-parametrized tests](#backend-parametrized-tests).
5. Recorded decision: `tensorboard` returns to an opt-in `[tensorboard]` extra
   (reverting PR #205's base footprint), with a friendly `ImportError` guard.
   — [Change 4](#change-4--make-local-the-default-move-tensorboard-to-an-extra).
6. Resume-dir bug fixed: a resumed run continues in the old run folder; `config.yaml`
   is preserved. — [Change 1](#change-1--resume-dir-bug-fix).

## Non-goals

Per the issue:

- No live web UI/server.
- No `csp plot <run_dir>` static renderer (possible follow-up, out of scope).
- No run-comparison / experiment-tracking database.

## Phasing

Each phase is an independently reviewable feature block with explicit interface
contracts at its boundary, so a later phase's fresh session can build on earlier ones
without re-reading their code.

### Phase 1 — Resume-dir bug fix

Scope: `runner.py` run-dir reuse on resume + `trainer.py` `config.yaml`
skip-if-exists, plus their tests.

Interface exposed / consumed:

- `run_training(cfg, resume_from=<old_run>/checkpoints/step_N)` reuses
  `resume_from.parent.parent` as `run_dir`; `EvalArtifacts.run_dir` equals the resume
  run dir.
- The trainer no longer overwrites an existing `run_dir/config.yaml`.

### Phase 2 — `local` tracker + capability gate

Scope: `LocalTracker` module + `build_tracker` `local` dispatch + `wants_images` on the
protocol and all four backends + trainer panel gate + the unit / parametrized /
integration tests for the tracker and the gate.

Depends on Phase 1 (resume dedup assumes `metrics.jsonl` is already in the reused
`run_dir`).

Interface exposed / consumed:

- Backend `local` is registered (`@register("tracker", "local")`) and resolvable via
  `build_tracker` for `cfg.tracking.backend == "local"`.
- `Tracker.wants_images: bool` exists on the protocol; every backend defines it
  (`local`/`none` → `False`; `tensorboard`/`wandb` → `True`).
- `LocalTracker` writes `run_dir/metrics.jsonl` (one JSON line per `log_scalars`,
  keys `step` + `wall_time` + scalars) and is metrics-only (`log_images` is a no-op).

### Phase 3 — Defaults + dependency move

Scope: schema default `local`, `config_full.yaml` template default, `pyproject.toml`
`tensorboard` → `[tensorboard]` extra, `TensorBoardTracker` `ImportError` guard, and the
recorded decision; plus schema/template default assertions.

Depends on Phase 2 (the default must resolve to a registered, working backend).

Interface exposed / consumed:

- `TrackingConfig().backend == "local"`; `TrackerBackend` accepts
  `"local" | "tensorboard" | "wandb" | "none"`.
- Selecting `tensorboard` without the extra raises a friendly `ImportError` naming the
  `[tensorboard]` extra.
