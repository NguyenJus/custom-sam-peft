# spec/smoke-test — GPU Overfit Smoke Tests

**Status:** Draft (2026-05-18)
**Architecture step:** 9 of 9 (see `2026-05-15-esam3-architecture-design.md` §11)
**Scope:** Harden the existing real-GPU LoRA overfit test and add a sibling QLoRA test, both driven from committed example YAMLs through `run_training(cfg)`. Pure test + example-config diff; no `src/esam3/` changes.

---

## 1. Current State

| Surface | State today | This spec |
|---|---|---|
| `tests/gpu/test_real_train_overfits.py` | Builds a `TrainConfig` from in-code literals, constructs `Trainer` directly, asserts ≥30% loss drop on a `_RecordingTracker`. LoRA + text + COCO. | **Rewritten** to load `configs/examples/gpu_smoke_lora.yaml` and call `run_training(cfg)`. Adds VRAM-ceiling and finite-value assertions. |
| Real-GPU QLoRA training | Not covered. `tests/integration/test_peft_qlora_real.py` covers `apply_qlora` / save / load / merge but never runs a training step. | **New** `tests/gpu/test_real_train_qlora.py` overfits 50 steps via QLoRA + text + COCO. |
| `_RecordingTracker` | Defined inline inside the LoRA test. | Lifted into `tests/gpu/conftest.py` and reused by both tests. |
| `_bnb_available()` | Lives in `tests/integration/test_peft_qlora_real.py`. | Lifted into `tests/gpu/conftest.py`. |
| `requires_bnb` pytest marker | Not registered. | Registered in `tests/conftest.py`. |
| Example configs validating against `TrainConfig` | Two exist (`coco_text_lora.yaml`, `coco_text_qlora.yaml`); no test covers them. | **New** `tests/unit/test_config_examples.py` `load_config()`s every YAML under `configs/examples/` (drive-by win — covers the new smoke YAMLs and the two pre-existing ones). |
| `logs/TODO.md` | Tracks deferred work. | Appended with four new gaps (see §7). |

The existing test's `Trainer(...).fit()` path is replaced by `run_training(cfg)`, which is the same library entrypoint the CLI's `esam3 train` command uses (`src/esam3/train/runner.py`). A user running `esam3 train --config configs/examples/gpu_smoke_lora.yaml` then executes exactly what the test executes — modulo monkeypatched tracker.

---

## 2. Goals & Non-Goals

**Goals.**

- Prove on real hardware that LoRA + SAM3.1 trains end-to-end (hardens the existing test).
- Prove on real hardware that QLoRA + SAM3.1 trains end-to-end (currently only stub-validated).
- Make both paths reproducible from one committed YAML each — no in-test `TrainConfig` literals.
- Assert defensively: loss-drop ratio, peak VRAM ceiling, finite loss at every step.
- Exercise the same `run_training(cfg)` seam the CLI calls, so the test and the user's `esam3 train` command share one code path.

**Non-goals.**

- bbox prompt mode — rejected by `Trainer.__init__` (`src/esam3/train/trainer.py:115`); out of v0 training entirely.
- HF dataset adapter on real GPU — stub-only coverage remains; gap logged to TODO.
- Real-GPU resume, real-GPU eval, `esam3 doctor` real-GPU smoke — CPU integration tests are the path; gaps logged to TODO.
- Nightly CI scheduling, runners, notifications — separate future spec.
- Human-gate "must run on real GPU before merge". This spec governs code correctness only; the first real-GPU run happens post-merge.
- New scalar keys, new `Trainer` outputs, or any `src/esam3/` change.

---

## 3. Files Touched / Module Layout

```
configs/examples/
  gpu_smoke_lora.yaml          # NEW — source of truth for the LoRA smoke
  gpu_smoke_qlora.yaml         # NEW — source of truth for the QLoRA smoke

tests/gpu/
  test_real_train_overfits.py  # REWRITTEN — drops in-code TrainConfig, loads
                               # gpu_smoke_lora.yaml, calls run_training(cfg),
                               # adds VRAM + finite-value assertions.
  test_real_train_qlora.py     # NEW — same skeleton, gpu_smoke_qlora.yaml,
                               # looser thresholds, requires_bnb skip.
  conftest.py                  # NEW — shared _RecordingTracker + _bnb_available().

tests/conftest.py              # CHANGED — register `requires_bnb` marker.

tests/unit/
  test_config_examples.py      # NEW — CPU-only; load_config()s every YAML under
                               # configs/examples/. Covers the new smoke YAMLs
                               # AND the existing coco_text_lora / coco_text_qlora.

logs/TODO.md                   # APPENDED — four deferred items (see §7).
```

No file under `src/esam3/` is modified. Pure test + example-config diff.

---

## 4. Test Design

### 4.1 Shared helpers (`tests/gpu/conftest.py`)

The conftest is new. It contains exactly two helpers, both lifted from existing tests:

```python
"""GPU-tier conftest: shared helpers for real-SAM3.1 smoke tests."""

from __future__ import annotations

from esam3.tracking.noop import NoopTracker


class _RecordingTracker(NoopTracker):
    """NoopTracker subclass that captures every (step, scalars) log call.

    Lifted from the inline definition in the previous version of
    test_real_train_overfits.py. Both GPU smoke tests share this instance shape;
    assertions on tracker.scalars are the test surface.
    """

    def __init__(self) -> None:
        self.scalars: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        self.scalars.append((step, values))

    def log_images(self, step: int, images: dict[str, object]) -> None:
        pass

    def close(self) -> None:
        pass


def _bnb_available() -> bool:
    """Return True iff bitsandbytes is importable. Lifted from
    tests/integration/test_peft_qlora_real.py."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True
```

These are module-level helpers (not pytest fixtures) because both tests need them at decoration time (`@pytest.mark.skipif(not _bnb_available(), ...)`) and at runtime (`tracker = _RecordingTracker()`).

### 4.2 Tracker injection seam

`run_training(cfg)` constructs its own tracker via `esam3.tracking.build_tracker(cfg)` (confirmed: this symbol lives at `src/esam3/tracking/__init__.py:20`). The smoke tests need to capture every `log_scalars` call without depending on TensorBoard or W&B side effects. Therefore each test does:

```python
monkeypatch.setattr("esam3.tracking.build_tracker", lambda *_a, **_kw: tracker)
```

This replaces the symbol at its public import path. `src/esam3/train/runner.py` imports `build_tracker` from `esam3.tracking` (line 13: `from esam3.tracking import build_tracker`), so the monkeypatch must target either:

- `esam3.tracking.build_tracker` (the original definition), **and** `esam3.train.runner.build_tracker` (the re-bound name in the runner module), or
- just `esam3.train.runner.build_tracker` if a single patch is enough.

**Decision:** monkeypatch `esam3.train.runner.build_tracker`. The runner does `from esam3.tracking import build_tracker`, which binds the name in `runner.__dict__`; patching the source symbol after that bind is ineffective. Tests must patch the consumer's namespace, not the producer's.

**Rejected alternative:** adding a `tracker_override` kwarg to `run_training(cfg, *, tracker_override=...)`. This would pollute the public library signature with a test hook. The monkeypatch-the-consumer pattern is standard pytest and adds no library API surface. Revisit if a non-test caller ever needs to inject a tracker.

### 4.3 Test skeleton

Both tests share this skeleton. Per-test parameters are in §4.4.

```python
import math
from pathlib import Path

import pytest
import torch

from esam3.config.loader import load_config
from esam3.train.runner import run_training
from tests.gpu.conftest import _RecordingTracker, _bnb_available

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "<YAML>"
LOSS_RATIO_CEIL = <ratio>
VRAM_CEIL_GB = <gb>


def test_<name>(tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
        ],
    )
    tracker = _RecordingTracker()
    monkeypatch.setattr("esam3.train.runner.build_tracker", lambda *_a, **_kw: tracker)

    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    losses = [s["loss/total"] for _, s in tracker.scalars if s["loss/total"] > 0]
    assert losses, "expected at least one logged loss scalar"
    assert all(
        math.isfinite(v)
        for _, s in tracker.scalars
        for v in s.values()
    ), "non-finite scalar logged during training"
    assert losses[-1] <= LOSS_RATIO_CEIL * losses[0], (
        f"loss did not drop enough: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
    assert peak_vram_gb <= VRAM_CEIL_GB, (
        f"peak VRAM {peak_vram_gb:.2f}GB exceeded ceiling {VRAM_CEIL_GB}GB"
    )
```

`tiny_coco_dir` is the existing fixture in `tests/conftest.py` (line 52). `load_config(..., overrides=...)` already supports string overrides parsed YAML-style (`src/esam3/config/loader.py:62`).

### 4.4 Per-test parameters

| Test file | `CONFIG_PATH` (basename) | `LOSS_RATIO_CEIL` | `VRAM_CEIL_GB` | Extra marker |
|---|---|---|---|---|
| `test_real_train_overfits.py` | `gpu_smoke_lora.yaml` | `0.70` | `14` | — |
| `test_real_train_qlora.py` | `gpu_smoke_qlora.yaml` | `0.75` | `10` | `requires_bnb` |

**Rationale for thresholds.**
- LoRA loss ratio `0.70` matches the existing test's "≥30% drop" assertion (`tests/gpu/test_real_train_overfits.py:112`), preserving behavior.
- QLoRA loss ratio `0.75` is looser by 5 percentage points because 4-bit base + bf16 LoRA on 50 steps converges slightly slower than bf16 LoRA. A 25% drop on 2-image overfit is still trivially achievable; tighter and the test flakes on hardware variance.
- LoRA VRAM ceiling `14` GB is the architecture's stated 12–16 GB consumer-GPU target's upper edge with headroom for the bf16 base.
- QLoRA VRAM ceiling `10` GB reflects the "12 GB recipe" promise (architecture §6) with a 2 GB safety margin.

Both tests inherit `pytestmark = [gpu, requires_compatible_gpu, requires_checkpoint]` from the current LoRA test's convention. The QLoRA test adds a `@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")` per-test decorator (not in `pytestmark`) so the skip reason is specific to that test.

### 4.5 `requires_bnb` marker

Registered in `tests/conftest.py` alongside the existing markers:

```python
config.addinivalue_line(
    "markers",
    "requires_bnb: skip unless bitsandbytes is importable",
)
```

This is purely an inivalue registration so pytest doesn't warn on unknown markers. The actual skip is done by the per-test `@pytest.mark.skipif(not _bnb_available(), ...)` decorator (lifted pattern from the integration test). Registering the marker keeps the symbol visible if a future test wants to filter with `-m requires_bnb`.

### 4.6 Finite-value assertion: scope today

The Trainer currently emits these scalar keys via `_ScalarWindow.flush()` (per `2026-05-18-tracking-design.md` §9): `loss/total`, `loss/mask`, `loss/box`, `loss/obj`, `loss/presence`, `lr`, `box_hint/p`, `box_hint/applied`, `grad_norm`, `throughput/img_s`, `skipped_steps`. The `all(math.isfinite(v) ...)` assertion in the skeleton covers every value in every logged scalar dict, which today is all of the above.

**Deferred — grad-norm finiteness coverage gap.** If the Trainer ever stops emitting `grad_norm` (or renames it to `grad_norm/total`), the finite-value assertion's coverage shifts automatically — it asserts on whatever the Trainer logs. Appended to `logs/TODO.md`: "ensure Trainer continues to emit `grad_norm` so GPU smoke catches silent grad explosions; consider standardizing on `grad_norm/total` for namespace consistency with `loss/total`."

### 4.7 Why monkeypatch and not `tracking.backend: "none"`

The YAML sets `tracking.backend: "none"` so a user running `esam3 train --config gpu_smoke_lora.yaml` does not need any extra installed. The test monkeypatches anyway because `NoopTracker` discards `log_scalars` calls — we need the data to assert on. Net effect: the YAML is standalone-runnable; the test substitutes a recording sink in-process. Both behaviors are correct for their consumer.

---

## 5. YAML Config Design

Both YAMLs live under `configs/examples/` so users can copy them or pass them directly to `esam3 train`. Data paths are placeholders the test overrides via `load_config(..., overrides=[...])`. Train hyperparameters are tuned for 50 grad updates on 2 images.

### 5.1 `configs/examples/gpu_smoke_lora.yaml`

Mirrors the in-code `TrainConfig` from the current `tests/gpu/test_real_train_overfits.py`:

```yaml
run:
  name: gpu-smoke-lora
  output_dir: ./runs
  seed: 0

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  gradient_checkpointing: true
  dtype: bfloat16

data:
  format: coco
  train:
    annotations: data/placeholder/annotations.json
    images: data/placeholder/images
  val:
    annotations: data/placeholder/annotations.json
    images: data/placeholder/images
  prompt_mode: text
  image_size: 1008
  augmentations:
    hflip: true
    color_jitter: 0.0

peft:
  method: lora
  scope: vision_decoder

train:
  epochs: 25
  batch_size: 1
  grad_accum_steps: 1
  optimizer: adamw
  lr: 5.0e-4
  lr_schedule: constant
  warmup_steps: 0
  save_every: 50
  log_every: 10
  num_workers: 0
  box_hint:
    p_start: 1.0
    p_end: 0.0
    decay_steps: 25

tracking:
  backend: none
```

**Rationale block.**
- `run.seed: 0` matches the existing test (deterministic on overfit setup).
- `data.augmentations.color_jitter: 0.0` matches the existing test (avoids color-augmentation variance on a 2-image set).
- `data.image_size: 1008` is SAM3.1's native input — confirmed by the model-loading spec and the existing test.
- `peft.method: lora`, `peft.scope: vision_decoder` — defaults for the rest (`r: 16`, `alpha: 32`, `dropout: 0.05`) come from `PEFTConfig` schema defaults; the YAML stays minimal.
- `train.epochs: 25` × 2 images × `grad_accum_steps: 1` = 50 grad updates. Matches the architecture §9 "~50 steps" target.
- `train.lr: 5.0e-4` matches the existing test (higher than the example configs' `1e-4` because we want aggressive overfit).
- `train.lr_schedule: constant`, `warmup_steps: 0` — no LR warmup or decay on a 50-step overfit.
- `train.save_every: 50` — exactly one checkpoint at the end; minimizes I/O.
- `train.log_every: 10` — yields ~5 logged windows, enough for the `losses[-1] / losses[0]` ratio test.
- `train.num_workers: 0` — DataLoader runs in-process; deterministic and CUDA-safe for a 2-image set.
- `train.box_hint.{p_start, p_end, decay_steps}` — preserves the box-hint curriculum from the existing test exactly (1.0 → 0.0 over the first 25 steps, then text-only for the remaining 25).
- `tracking.backend: none` — keeps the YAML standalone-runnable. The test monkeypatches `build_tracker` before `run_training` runs, so the backend value is overridden in practice (see §4.7).

### 5.2 `configs/examples/gpu_smoke_qlora.yaml`

Diverges from the LoRA YAML only where it must:

```yaml
run:
  name: gpu-smoke-qlora
  # ... identical to gpu_smoke_lora.yaml from here ...

peft:
  method: qlora
  scope: vision_decoder
  qlora:
    quant_type: nf4
    compute_dtype: bfloat16

train:
  # ... identical to gpu_smoke_lora.yaml ...
  optimizer: adamw8bit
  # ... remaining fields identical ...
```

Concrete differences from the LoRA YAML:
- `run.name: gpu-smoke-qlora`
- `peft.method: qlora`
- `peft.qlora: { quant_type: nf4, compute_dtype: bfloat16 }`
- `train.optimizer: adamw8bit`

Everything else identical. QLoRA pairs with `adamw8bit` because the 8-bit optimizer is the memory-pairing the architecture (§6) prescribes for the 12 GB recipe; users running this YAML are already installing the `[qlora]` extra (which brings bitsandbytes), so `adamw8bit` is available.

### 5.3 Drive-by: `tests/unit/test_config_examples.py`

A tiny CPU test that iterates `configs/examples/*.yaml` and asserts each loads:

```python
from pathlib import Path
import pytest
from esam3.config.loader import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "examples"

@pytest.mark.parametrize("yaml_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_example_config_validates(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    assert cfg.run.name  # smoke: schema parsed and produced a populated TrainConfig
```

This locks in that every shipped example is valid against `TrainConfig`. Today that means `coco_text_lora.yaml`, `coco_text_qlora.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml`. The test is parametrized so a new YAML automatically gets coverage.

`load_config` resolves paths relative to the config file's directory (`src/esam3/config/loader.py:99`); the placeholder strings in the smoke YAMLs become absolute paths under `configs/examples/data/placeholder/...`, which don't exist on disk. `TrainConfig` validation does **not** stat paths (verified: `DataSplit` only enforces `min_length=1`), so the test passes without the placeholder files existing.

---

## 6. Exit Criteria

**Code.**

- [ ] `configs/examples/gpu_smoke_lora.yaml` exists and validates against `TrainConfig`.
- [ ] `configs/examples/gpu_smoke_qlora.yaml` exists and validates against `TrainConfig`.
- [ ] `tests/gpu/test_real_train_overfits.py` rewritten: loads `gpu_smoke_lora.yaml`, calls `run_training`, asserts loss ratio ≤ `0.70`, peak VRAM ≤ `14` GB, all logged scalars finite.
- [ ] `tests/gpu/test_real_train_qlora.py` added: loads `gpu_smoke_qlora.yaml`, calls `run_training`, asserts loss ratio ≤ `0.75`, peak VRAM ≤ `10` GB, all logged scalars finite, skips when `bitsandbytes` is unimportable.
- [ ] `tests/gpu/conftest.py` exists with `_RecordingTracker` and `_bnb_available()`.
- [ ] `tests/conftest.py` registers the `requires_bnb` marker.
- [ ] `tests/unit/test_config_examples.py` added; passes for all four current YAMLs.
- [ ] `logs/TODO.md` appended with the four deferred items in §7.

**Tests (CPU-only — what runs in CI).**

- [ ] `ruff check && mypy && pytest` green.
- [ ] `tests/unit/test_config_examples.py` passes.
- [ ] `tests/gpu/test_real_train_overfits.py` and `tests/gpu/test_real_train_qlora.py` *collect* cleanly on a CPU box and skip via the `requires_compatible_gpu` / `requires_checkpoint` markers (no `ImportError`, no collection failure, no marker-registration warning).

**Explicitly NOT a gate.** Real-GPU runs of either test are a post-merge follow-up. The merge bar is "code is correct, CPU CI is green, GPU tests collect cleanly." A reviewer or maintainer schedules the real-GPU runs after the PR lands.

---

## 7. Deferred (Appended to `logs/TODO.md`)

Four entries are appended verbatim:

- `real-GPU resume smoke` — `Trainer.fit(resume_from=...)` exercised end-to-end on real SAM3.1. CPU integration coverage exists (`tests/integration/test_train_*`); the real-GPU variant requires a two-phase smoke (initial fit → checkpoint → resume) that doesn't fit the 50-step overfit shape.
- `real-GPU eval smoke` — `esam3 eval --checkpoint runs/.../adapter` against real SAM3.1. The current `tests/gpu/` tier only covers training. CPU eval coverage exists via the stub-model integration tests.
- `HF dataset adapter on real GPU` — `data.format: hf` runs end-to-end through `run_training` today (registered via `@register`), but no real-GPU test exercises the HF path against SAM3.1.
- `Trainer grad-norm key naming` — Trainer emits `grad_norm`; consider standardizing on `grad_norm/total` for namespace consistency with `loss/total`. The finite-value assertion in the GPU smoke tests already covers `grad_norm` and would continue to cover a renamed `grad_norm/total` automatically.
