# Tracking Subsystem Design (architecture step 7 — `spec/tracking`)

**Status:** Draft (2026-05-18)
**Scope:** `src/esam3/tracking/` — real `TensorBoardTracker` and `WandBTracker` implementations, a `build_tracker` factory, one Protocol-lifecycle addition (`start_run`), and the helper that flattens `MetricsReport` into scalar keys consumed by `log_scalars`. Trainer touches limited to a single `start_run` call site.

**Builds on:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md) §3, §4, §5; [`2026-05-17-training-loop-design.md`](2026-05-17-training-loop-design.md); [`2026-05-17-eval-design.md`](2026-05-17-eval-design.md) (for `MetricsReport` shape).

---

## 1. Goals & v0 Scope

Replace the `NotImplementedError` stubs in `tracking/tensorboard.py` and `tracking/wandb.py` with working backends, give the Trainer a single, well-defined call site to hand the tracker its `run_dir` + config, and continue W&B runs across checkpoint resume.

**In scope:**

| Item | Where |
| --- | --- |
| `TensorBoardTracker` wrapping `torch.utils.tensorboard.SummaryWriter` | `tracking/tensorboard.py` |
| `WandBTracker` wrapping `wandb.init/log/finish` | `tracking/wandb.py` |
| `start_run(run_dir, config, resume_from=None)` on the `Tracker` Protocol | `tracking/base.py` + all three backends |
| `build_tracker(cfg: TrainConfig) -> Tracker` factory | `tracking/__init__.py` |
| `flatten_metrics_report(report, prefix="eval") -> dict[str, float]` helper | `tracking/__init__.py` |
| W&B run continuation via `wandb_run_id.txt` written into `run_dir` | `tracking/wandb.py` |
| One-line trainer edit: `tracker.start_run(...)` at the top of `fit()` | `train/trainer.py` |
| Eager `ImportError` with a `pip install` hint when the chosen backend's extra is missing | `tensorboard.py`, `wandb.py` constructors |
| Image contract: `np.ndarray[uint8]` shape `(H, W, 3)`; `ValueError` on violation | both backends |
| Tests: Protocol conformance, factory, TB round-trip, mocked W&B (incl. resume), metric flatten | `tests/unit/test_tracking_*.py` |

**Out of scope (explicitly deferred):**

- `log_text`, `log_histogram`, `log_artifact` — no caller in v0.
- W&B `Tables`, artifact uploads of checkpoints, sweep integration.
- Profiler / system-metric integration.
- Multi-process / DDP-aware tracking (single-device only per architecture §6).
- Restricting / sanitizing W&B environment variables — `WANDB_PROJECT`/`WANDB_ENTITY` are honored by the SDK natively.
- Per-step / mid-run config updates.

---

## 2. Architectural Approach

The architecture's §5 `Tracker` Protocol is constructed *before* `Trainer.fit()` creates the run directory, so the tracker has no place to receive `run_dir` or the full config. This spec adds one lifecycle method — `start_run` — that the Trainer calls once at the top of `fit()`. `NoopTracker` makes it a no-op; `TensorBoardTracker` instantiates its `SummaryWriter` and logs the config as a markdown blob; `WandBTracker` calls `wandb.init(...)` and writes `wandb_run_id.txt` for later resume.

The two backends do *not* take a stale snapshot of `cfg.tracking` at construction. Construction stores `cfg`; `start_run` is the single point where the run becomes real on the backend. This keeps construction cheap and side-effect-free, which matters for CLI startup latency and for tests that want to instantiate a tracker without spinning up TB/W&B.

Lazy `import` of `tensorboard` / `wandb` happens inside the constructors (so the *eager* `ImportError` only fires when the user actually selects that backend). The `build_tracker` factory similarly imports the chosen backend's module lazily so that `from esam3.tracking import build_tracker` does not require either extra to be installed.

---

## 3. Public Surfaces

### 3.1 `tracking/base.py` — Protocol

```python
from __future__ import annotations
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Tracker(Protocol):
    """Stable seam between Trainer and logging backends.

    Lifecycle: __init__ → start_run → log_*... → close.
    """

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None: ...

    def log_scalars(self, step: int, values: dict[str, float]) -> None: ...

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None: ...

    def close(self) -> None: ...
```

**Contract:**

- `start_run` is called exactly once, after the Trainer has created `run_dir` and written `config.yaml`, and before any `log_*` call.
- `config` is the JSON-mode dump of `TrainConfig`, i.e. `cfg.model_dump(mode="json")`. Backends MAY choose how to render it (TB: markdown text; W&B: native `config=`); they MUST NOT mutate it.
- `resume_from` is the value passed to `Trainer.fit(resume_from=...)` — a path to a checkpoint directory (e.g. `runs/<old>/checkpoints/step_N`), or `None` for a fresh run. Backends that don't model resume (TB, Noop) ignore it.
- `log_scalars` / `log_images` MAY be called any number of times after `start_run`. Calling them before `start_run` is a programmer error; backends MAY assert.
- `close()` is idempotent and called once in the Trainer's `finally`. After `close()`, no further calls are valid.
- Backends MUST silently drop non-finite scalar values (NaN, ±Inf). They MUST raise `ValueError` on images that violate the `uint8 (H, W, 3)` contract.

### 3.2 `tracking/__init__.py` — factory + helper

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

from esam3._registry import lookup
from esam3.config.schema import TrainConfig
from esam3.tracking.base import Tracker

if TYPE_CHECKING:
    # Type-only import; avoids a runtime tracking → eval dependency so
    # tracking remains independent of subsystems 1–6 per architecture §11.
    from esam3.eval.metrics import MetricsReport

__all__ = ["Tracker", "build_tracker", "flatten_metrics_report"]


def build_tracker(cfg: TrainConfig) -> Tracker:
    """Resolve cfg.tracking.backend to a concrete Tracker.

    Imports the chosen backend module lazily so missing optional extras
    only surface when that backend is actually requested.
    """
    backend = cfg.tracking.backend  # Literal["tensorboard", "wandb", "none"]
    if backend == "tensorboard":
        from esam3.tracking import tensorboard as _tb  # noqa: F401
    elif backend == "wandb":
        from esam3.tracking import wandb as _wb  # noqa: F401
    elif backend == "none":
        from esam3.tracking import noop as _noop  # noqa: F401
    else:
        raise ValueError(f"unknown tracking.backend: {backend!r}")
    factory = lookup("tracker", backend)
    return cast(Tracker, factory(cfg))


def flatten_metrics_report(
    report: "MetricsReport",
    prefix: str = "eval",
) -> dict[str, float]:
    """Render a MetricsReport as a flat scalar dict suitable for log_scalars.

    Duck-typed at runtime: accepts any object with ``overall: dict[str, float]``
    and ``per_class: dict[str, dict[str, float]]`` — no runtime import of
    ``MetricsReport``. Keys are namespaced under ``prefix``:

        eval/mAP, eval/mAP_50, eval/mAP_75
        eval/per_class/<class_name>/AP, eval/per_class/<class_name>/AP_50, ...

    ``/`` characters in class names are replaced with ``_`` to avoid colliding
    with the namespace separator.
    """
    out: dict[str, float] = {f"{prefix}/{k}": float(v) for k, v in report.overall.items()}
    for cls, metrics in report.per_class.items():
        safe = cls.replace("/", "_")
        for k, v in metrics.items():
            out[f"{prefix}/per_class/{safe}/{k}"] = float(v)
    return out
```

### 3.3 `tracking/noop.py` — no-op

```python
class NoopTracker:
    def start_run(self, run_dir, config, resume_from=None) -> None:
        return None
    def log_scalars(self, step, values) -> None:
        return None
    def log_images(self, step, images) -> None:
        return None
    def close(self) -> None:
        return None


@register("tracker", "none")
def build_noop(_cfg: TrainConfig) -> NoopTracker:
    return NoopTracker()
```

(Existing `build_noop` signature widens from `dict[str, Any]` to `TrainConfig`. Compatible with all current callers — none yet exist.)

### 3.4 `tracking/tensorboard.py` — real implementation

```python
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from esam3._registry import register
from esam3.config.schema import TrainConfig


class TensorBoardTracker:
    def __init__(self, cfg: TrainConfig) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tracking.backend='tensorboard' requires the [tensorboard] extra. "
                "Install with: pip install 'efficient-sam3-finetuning[tensorboard]'"
            ) from e
        self._cfg = cfg
        self._writer: "SummaryWriter | None" = None
        self._closed = False

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=str(run_dir))
        # Markdown code fence so TB's text tab renders YAML monospaced.
        self._writer.add_text("config", "```yaml\n" + yaml.safe_dump(config) + "\n```", 0)

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        assert self._writer is not None, "start_run() must be called before log_scalars()"
        for tag, value in values.items():
            if not math.isfinite(value):
                continue
            self._writer.add_scalar(tag, value, step)

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        assert self._writer is not None, "start_run() must be called before log_images()"
        for tag, arr in images.items():
            _validate_image(tag, arr)
            self._writer.add_image(tag, arr, step, dataformats="HWC")

    def close(self) -> None:
        if self._closed or self._writer is None:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True


@register("tracker", "tensorboard")
def build_tensorboard(cfg: TrainConfig) -> TensorBoardTracker:
    return TensorBoardTracker(cfg)
```

Event files land directly in `run_dir`, so `tensorboard --logdir <output_dir>` (the parent that holds many `run_dir`s) shows every run side-by-side.

### 3.5 `tracking/wandb.py` — real implementation

```python
import math
from pathlib import Path
from typing import Any

import numpy as np

from esam3._registry import register
from esam3.config.schema import TrainConfig

_WANDB_ID_FILENAME = "wandb_run_id.txt"


class WandBTracker:
    def __init__(self, cfg: TrainConfig) -> None:
        try:
            import wandb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tracking.backend='wandb' requires the [wandb] extra. "
                "Install with: pip install 'efficient-sam3-finetuning[wandb]'"
            ) from e
        self._cfg = cfg
        self._run: Any | None = None  # wandb.sdk.wandb_run.Run
        self._closed = False

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        import wandb
        run_id, resume_mode = self._maybe_resume_id(resume_from)
        self._run = wandb.init(
            project=self._cfg.tracking.wandb.project,
            entity=self._cfg.tracking.wandb.entity,
            name=run_dir.name,
            dir=str(run_dir),
            config=config,
            id=run_id,
            resume=resume_mode,
        )
        (run_dir / _WANDB_ID_FILENAME).write_text(self._run.id)

    @staticmethod
    def _maybe_resume_id(resume_from: Path | None) -> tuple[str | None, str | None]:
        """Walk up from a checkpoint path looking for `wandb_run_id.txt`.

        Returns (run_id, resume_mode) — both None when we should start fresh.
        Checks `resume_from` itself plus up to 3 ancestors, so a path like
            runs/<old_run>/checkpoints/step_100
        finds runs/<old_run>/wandb_run_id.txt (2 levels up).
        """
        if resume_from is None:
            return None, None
        candidate_dir = Path(resume_from)
        for _ in range(4):
            candidate = candidate_dir / _WANDB_ID_FILENAME
            if candidate.is_file():
                return candidate.read_text().strip(), "allow"
            if candidate_dir.parent == candidate_dir:
                break
            candidate_dir = candidate_dir.parent
        return None, None

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        assert self._run is not None, "start_run() must be called before log_scalars()"
        finite = {k: v for k, v in values.items() if math.isfinite(v)}
        if finite:
            self._run.log(finite, step=step)

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        import wandb
        assert self._run is not None, "start_run() must be called before log_images()"
        payload: dict[str, Any] = {}
        for tag, arr in images.items():
            _validate_image(tag, arr)
            payload[tag] = wandb.Image(arr)
        if payload:
            self._run.log(payload, step=step)

    def close(self) -> None:
        if self._closed or self._run is None:
            return
        self._run.finish()
        self._closed = True


@register("tracker", "wandb")
def build_wandb(cfg: TrainConfig) -> WandBTracker:
    return WandBTracker(cfg)
```

`WANDB_PROJECT` / `WANDB_ENTITY` env vars are honored by the SDK natively; this spec doesn't add a layer on top.

### 3.6 Shared validator (in `tracking/base.py`)

```python
def _validate_image(tag: str, arr: np.ndarray[Any, Any]) -> None:
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image '{tag}' must be uint8 (H, W, 3); got dtype={arr.dtype} shape={tuple(arr.shape)}"
        )
```

Module-private (`_`-prefixed); reused by both real backends so the contract lives in one place.

---

## 4. Trainer-Side Changes

Only one edit to `src/esam3/train/trainer.py`, in `Trainer.fit()`, immediately after `run_dir.mkdir(...)` and the `config.yaml` write:

```python
self.tracker.start_run(run_dir, cfg.model_dump(mode="json"), resume_from)
```

All other tracker usage in `trainer.py` and `loop.py` (`log_scalars`, `log_images`, `close`) remains unchanged. No new imports.

**Eval-during-training:** When the eval subsystem wires `Evaluator` into `Trainer.fit()`, it will call

```python
self.tracker.log_scalars(global_step, flatten_metrics_report(report))
```

That wiring lives in the eval spec / `train/trainer.py`, not in `tracking/`. The flatten helper is the only API this spec exposes to it.

---

## 5. CLI Changes

`src/esam3/cli/train_cmd.py` and `src/esam3/cli/eval_cmd.py` will each get one new line (when those CLI commands actually wire training/eval):

```python
tracker = build_tracker(cfg)
```

This spec does not implement those CLI commands — it only ensures `build_tracker` exists and is suitable for them. The CLI spec (architecture step 8) consumes it.

---

## 6. Optional-Dependency Behavior

| Backend selected | Extra missing | Behavior |
| --- | --- | --- |
| `tensorboard` | `[tensorboard]` not installed | `build_tracker(cfg)` raises `ImportError` with `pip install 'efficient-sam3-finetuning[tensorboard]'` |
| `wandb` | `[wandb]` not installed | `build_tracker(cfg)` raises `ImportError` with `pip install 'efficient-sam3-finetuning[wandb]'` |
| `none` | n/a | always works |

The error fires at construction (during `build_tracker`), not at step 50 — so the CLI fails fast before any training cost is paid. `from esam3.tracking import build_tracker` itself never imports either optional package.

---

## 7. Resume Semantics

| State | Behavior |
| --- | --- |
| Fresh `fit()` (no `resume_from`) | TB: new event file in new `run_dir`. W&B: new run, new `id`, `wandb_run_id.txt` written. |
| `fit(resume_from=Path)` and `wandb_run_id.txt` found by walking ≤3 parents from `resume_from` | W&B: `wandb.init(id=<saved>, resume="allow")` — same W&B run continues, charts stay contiguous across restarts. New `run_dir` still gets its own `wandb_run_id.txt` (same id). |
| `fit(resume_from=Path)` and no `wandb_run_id.txt` found | W&B: brand-new run started. No error — the checkpoint may have been moved between machines. |
| TB on resume | Always starts a new event file in the new `run_dir`. Continuous timelines across resumes require launching `tensorboard --logdir <output_dir>` on the parent dir; TB UI merges them by step. |

The trainer's existing checkpoint logic already passes `resume_from` to `fit()`. No changes to checkpoint state.

---

## 8. Image Contract

The Trainer's existing `_log_image_panel` produces:

- `render_mask_panel(...)` → `np.ndarray[uint8]` shape `(H, W, 3)` per validation example.
- `np.concatenate(panels, axis=0)` → single `(H_total, W, 3)` `uint8` array passed as `images={"val_panels": panel}`.

The spec locks this as the only legal image format. Anything else raises `ValueError` from `_validate_image`. The shape `(H, W, 3)` (HWC) is preserved end-to-end:

- TB: `add_image(tag, arr, step, dataformats="HWC")`.
- W&B: `wandb.Image(arr)` accepts HWC uint8 natively.

Greyscale, float, and CHW arrays are out of scope. If a future caller needs them, that's a separate small spec.

---

## 9. Scalar-Key Conventions

The Trainer already emits these via `_ScalarWindow.flush()`:

```text
loss/total, loss/mask, loss/box, loss/obj, loss/presence,
lr, box_hint/p, box_hint/applied, grad_norm,
throughput/img_s, skipped_steps
```

The eval-flatten helper extends the namespace:

```text
eval/mAP, eval/mAP_50, eval/mAP_75
eval/per_class/<class>/AP, eval/per_class/<class>/AP_50, eval/per_class/<class>/AP_75
```

`/` is the namespace separator. Class names containing `/` are sanitized to `_`.

This spec does not invent new scalar keys; it only defines the namespacing rule and the flatten transform.

---

## 10. Testing Strategy

All tests live under `tests/unit/` unless otherwise marked. All are CPU-only and run on every commit.

### 10.1 `test_tracking_protocol.py` (new)

- `isinstance(NoopTracker(), Tracker)` is `True`.
- `isinstance(TensorBoardTracker(cfg), Tracker)` is `True` — skipped via `pytest.importorskip("tensorboard")`.
- `isinstance(WandBTracker(cfg), Tracker)` is `True` — uses a monkeypatched `wandb` module fixture so the test runs even without the extra installed.
- `runtime_checkable` Protocol catches signature drift: a subclass missing `start_run` is `not isinstance(..., Tracker)`.

### 10.2 `test_tracking_noop.py` (existing — extend)

- Add a `start_run(tmp_path, {"x": 1})` call returns `None` and does not raise.
- Add a smoke test that `Tracker` Protocol matches `NoopTracker` after the `start_run` addition (regression for the Protocol change).

### 10.3 `test_tracking_build.py` (new)

- `build_tracker(cfg_with_backend("none"))` returns a `NoopTracker`.
- `build_tracker(cfg_with_backend("tensorboard"))` returns `TensorBoardTracker` — skipped if `tensorboard` not importable.
- `build_tracker(cfg_with_backend("wandb"))` returns `WandBTracker` when `wandb` is mocked into `sys.modules`.
- Missing-extra path: monkeypatch `sys.modules` so `import tensorboard` raises `ImportError`; assert `build_tracker(cfg_with_backend("tensorboard"))` raises `ImportError` whose message includes `pip install 'efficient-sam3-finetuning[tensorboard]'`. Same shape for the wandb case.
- Unknown backend value: skipped because `TrackerBackend` is a `Literal` and pydantic rejects it at config-load time. (Defensive `ValueError` in `build_tracker` is kept; not tested.)

### 10.4 `test_tracking_tensorboard.py` (new, `pytest.importorskip("tensorboard")`)

- `start_run(tmp_path, {"a": 1}) → log_scalars(0, {"loss": 0.5}) → log_images(0, {"panel": uint8_hwc}) → close()`.
- Read back via `tensorboard.backend.event_processing.event_accumulator.EventAccumulator(str(tmp_path))`:
  - `Tags()["scalars"]` contains `"loss"`; `Scalars("loss")[0].value == 0.5`.
  - `Tags()["images"]` contains `"panel"`.
  - `Tags()["tensors"]` contains `"config"` (TB stores `add_text` as a text tensor).
- Non-finite scalar values (NaN, +Inf, -Inf) are dropped — read-back scalar list lacks them.
- Wrong-image-dtype (`float32`) and wrong-shape (`(H, W)`) calls raise `ValueError` mentioning the tag.
- `close()` is idempotent (calling twice does not raise).
- Calling `log_scalars` before `start_run` raises `AssertionError` (programmer-error sentinel).

### 10.5 `test_tracking_wandb.py` (new — no real network)

- Fixture `mock_wandb`: monkeypatches `sys.modules["wandb"]` with a fake module exposing `init`, `Image`, and a fake `Run` recording `log` / `finish` and exposing a deterministic `.id`.
- `start_run(tmp_path, {"x": 1})` calls `wandb.init` with the right kwargs:
  - `project=cfg.tracking.wandb.project`, `entity=cfg.tracking.wandb.entity`, `name=tmp_path.name`, `dir=str(tmp_path)`, `config={"x": 1}`, `id=None`, `resume=None`.
- `tmp_path / "wandb_run_id.txt"` is created with the fake run's id.
- `log_scalars(5, {"loss": 0.1, "nan": float("nan")})` calls `fake_run.log({"loss": 0.1}, step=5)` (NaN dropped).
- `log_images(5, {"panel": uint8_hwc})` calls `fake_run.log({"panel": wandb.Image(...)}, step=5)`.
- Wrong-image-dtype raises `ValueError` before any `log` call.
- Resume path:
  - Pre-write `wandb_run_id.txt` into a fake prior `run_dir`.
  - `start_run(new_run_dir, cfg_dump, resume_from=prior_run_dir/"checkpoints"/"step_100")`.
  - Assert `wandb.init` called with `id=<saved>`, `resume="allow"`.
  - Assert `new_run_dir / "wandb_run_id.txt"` matches the saved id.
- Resume path with missing id file: `wandb.init` called with `id=None`, `resume=None`. No error.
- `close()` is idempotent.

### 10.6 `test_tracking_flatten.py` (new)

- `flatten_metrics_report(report_with_two_classes)` produces exactly the expected key set:

  ```text
  eval/mAP, eval/mAP_50, eval/mAP_75,
  eval/per_class/cat/AP, eval/per_class/cat/AP_50,
  eval/per_class/dog/AP, eval/per_class/dog/AP_50
  ```

- Class name `"animals/cat"` becomes `"eval/per_class/animals_cat/AP"`.
- All values are `float`.
- Custom `prefix="val"` swaps `eval/` for `val/`.

### 10.7 Integration (`@pytest.mark.integration`)

- Parametrize the existing `tests/integration/test_train_end_to_end.py` over `("none", "tensorboard")` (skipping `"tensorboard"` if the extra is absent). Verify end-to-end fit on the stub model and the `tiny_coco` fixture produces a TB event file and no exceptions.
- W&B is *not* added to integration — the mocked unit coverage is sufficient and real W&B requires network/credentials.

### 10.8 Coverage gate

`pytest-cov` gate remains 80% on `src/esam3`. New files in `tracking/` are expected to land above 90% based on the test plan above.

### 10.9 Explicitly NOT tested

- The internals of `torch.utils.tensorboard.SummaryWriter` or the `wandb` SDK — we test our wrapper, not theirs.
- W&B network paths (`api.wandb.ai` round-trips).
- Multi-process / DDP tracker behavior — single-device only in v0.

---

## 11. Out of Scope (deferred to v1+ or other specs)

- `log_text`, `log_histogram`, `log_artifact`.
- Checkpoint upload as a W&B artifact.
- W&B sweep / agent integration.
- Profiler / `torch.profiler` integration.
- TB `add_hparams` (we use `add_text("config", ...)` instead; cleaner, no projector clutter).
- Non-RGB / non-uint8 image formats.
- Sanitizing or surfacing `WANDB_*` env vars beyond what the SDK already does.
- Distributed-aware tracker (rank-0-only logging gates) — relevant when Ray Train lands.

---

## 12. File Layout

```text
src/esam3/tracking/
  __init__.py        # build_tracker(cfg), flatten_metrics_report(report, prefix)
  base.py            # Tracker Protocol (adds start_run); _validate_image helper
  noop.py            # NoopTracker (adds start_run no-op); build_noop factory
  tensorboard.py     # TensorBoardTracker (real impl); build_tensorboard factory
  wandb.py           # WandBTracker (real impl) + _WANDB_ID_FILENAME; build_wandb factory

tests/unit/
  test_tracking_noop.py          # existing — extend with start_run
  test_tracking_protocol.py      # new
  test_tracking_build.py         # new
  test_tracking_tensorboard.py   # new (importorskip tensorboard)
  test_tracking_wandb.py         # new (mocked SDK)
  test_tracking_flatten.py       # new

src/esam3/train/trainer.py       # +1 call site: self.tracker.start_run(...)
tests/integration/test_train_end_to_end.py  # parametrize backend over ("none", "tensorboard")
```

No deletions. No moves. ~250 LOC new implementation, ~350 LOC new tests.
