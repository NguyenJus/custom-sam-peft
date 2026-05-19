# spec/smoke-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land two real-GPU overfit smoke tests (LoRA + QLoRA, both text-prompt COCO) driven from committed example YAMLs through `run_training(cfg)`, plus a CPU-only drive-by unit test that validates every YAML under `configs/examples/`. See `docs/superpowers/specs/2026-05-18-smoke-test-design.md`.

**Architecture:** Pure test + example-config diff — no `src/esam3/` source change. Both GPU tests share helpers in a new `tests/gpu/conftest.py` (`_RecordingTracker` + `_bnb_available()`), inherit the existing GPU-tier marker triple, monkeypatch `esam3.train.runner.build_tracker` (the consumer's namespace, see spec §4.2) to capture scalars, then assert on loss-ratio, peak VRAM, and finite values. Two new YAMLs under `configs/examples/` are the source of truth — users running `esam3 train --config configs/examples/gpu_smoke_<method>.yaml` execute the same path as the test. A new `tests/unit/test_config_examples.py` parametrizes over `configs/examples/*.yaml` so every shipped example is validated against `TrainConfig`.

**Tech Stack:** Python 3.12, pytest, pydantic v2 config (`load_config`), PyTorch (`torch.cuda.max_memory_allocated`), `bitsandbytes` (optional — gated by `requires_bnb`). No new library code.

---

## File Map

**New files:**

```
configs/examples/
  gpu_smoke_lora.yaml
  gpu_smoke_qlora.yaml

tests/gpu/
  conftest.py
  test_real_train_qlora.py

tests/unit/
  test_config_examples.py

logs/TODO.md                     # created (does not yet exist on this branch)
```

**Modified files:**

```
tests/conftest.py                # register `requires_bnb` marker
tests/gpu/test_real_train_overfits.py  # rewritten to load YAML + call run_training
```

No file under `src/esam3/` is modified.

---

## Pre-flight check

- [ ] **Step 0a: Confirm working tree clean**

```bash
git status
```
Expected: only this plan file (and the approved spec) staged or untracked. No other modifications.

- [ ] **Step 0b: Confirm baseline unit test suite passes before changes**

```bash
uv run pytest tests/unit -x -q
```
Expected: all unit tests pass.

- [ ] **Step 0c: Confirm the existing GPU test collects cleanly on CPU**

```bash
uv run pytest tests/gpu/test_real_train_overfits.py --collect-only -q
```
Expected: 1 test collected (it will be skipped at run time on a CPU box via `requires_compatible_gpu` / `requires_checkpoint`, but collection must succeed without error).

---

## Task 1: Register the `requires_bnb` marker

**Files:**
- Modify: `tests/conftest.py`

The marker must be registered before any test references it, otherwise pytest emits a `PytestUnknownMarkWarning` (or fails under `-W error`). This is the foundation Task 4's QLoRA test depends on.

- [ ] **Step 1a: Edit `tests/conftest.py` to register the marker**

Open `tests/conftest.py` and find the `pytest_configure` function (lines 18–28). Append one more `addinivalue_line` call so the function reads:

```python
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_checkpoint: skip unless models/sam3.1/sam3.1_multiplex.pt exists",
    )
    config.addinivalue_line(
        "markers",
        "requires_compatible_gpu: skip unless a CUDA device with compute capability "
        ">= 7.5 is available",
    )
    config.addinivalue_line(
        "markers",
        "requires_bnb: skip unless bitsandbytes is importable",
    )
```

No other change to this file.

- [ ] **Step 1b: Confirm the marker is registered**

```bash
uv run pytest --markers | grep requires_bnb
```
Expected: a single line `@pytest.mark.requires_bnb: skip unless bitsandbytes is importable`.

- [ ] **Step 1c: Confirm no existing tests regressed**

```bash
uv run pytest tests/unit -x -q
```
Expected: all unit tests still pass.

- [ ] **Step 1d: Commit**

```bash
git add tests/conftest.py
git commit -m "test(markers): register requires_bnb pytest marker"
```

---

## Task 2: Create `tests/gpu/conftest.py` shared helpers

**Files:**
- Create: `tests/gpu/conftest.py`

`_RecordingTracker` and `_bnb_available()` are lifted from existing tests so both GPU smoke tests share one source of truth. These are module-level helpers (not pytest fixtures) because they're needed at decoration time (`@pytest.mark.skipif(not _bnb_available(), ...)`) and at runtime (`tracker = _RecordingTracker()`). See spec §4.1.

- [ ] **Step 2a: Create the conftest**

Create `tests/gpu/conftest.py`:

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

- [ ] **Step 2b: Confirm the conftest module imports cleanly**

```bash
uv run python -c "from tests.gpu.conftest import _RecordingTracker, _bnb_available; t = _RecordingTracker(); t.log_scalars(0, {'a': 1.0}); assert t.scalars == [(0, {'a': 1.0})]; print('ok')"
```
Expected: `ok`.

- [ ] **Step 2c: ruff + mypy on the new file**

```bash
uv run ruff check tests/gpu/conftest.py && uv run mypy tests/gpu/conftest.py
```
Expected: both clean.

- [ ] **Step 2d: Confirm existing GPU test still collects**

```bash
uv run pytest tests/gpu --collect-only -q
```
Expected: 1 test collected (still `test_overfits_in_50_steps` — the rewrite happens in Task 5).

- [ ] **Step 2e: Commit**

```bash
git add tests/gpu/conftest.py
git commit -m "test(gpu): add shared _RecordingTracker + _bnb_available helpers"
```

---

## Task 3: Add `configs/examples/gpu_smoke_lora.yaml`

**Files:**
- Create: `configs/examples/gpu_smoke_lora.yaml`

The YAML is the source of truth for the LoRA smoke — mirrors the in-code `TrainConfig` from the current `tests/gpu/test_real_train_overfits.py`. See spec §5.1.

- [ ] **Step 3a: Create the YAML**

Create `configs/examples/gpu_smoke_lora.yaml`:

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

- [ ] **Step 3b: Validate the YAML loads against `TrainConfig`**

```bash
uv run python -c "from esam3.config.loader import load_config; cfg = load_config('configs/examples/gpu_smoke_lora.yaml'); print('ok', cfg.run.name, cfg.peft.method)"
```
Expected: `ok gpu-smoke-lora lora`.

- [ ] **Step 3c: Commit**

```bash
git add configs/examples/gpu_smoke_lora.yaml
git commit -m "examples: add gpu_smoke_lora.yaml for the LoRA GPU smoke test"
```

---

## Task 4: Add `configs/examples/gpu_smoke_qlora.yaml`

**Files:**
- Create: `configs/examples/gpu_smoke_qlora.yaml`

Diverges from the LoRA YAML only where the spec mandates (`peft.method`, `peft.qlora`, `train.optimizer`). See spec §5.2.

- [ ] **Step 4a: Create the YAML**

Create `configs/examples/gpu_smoke_qlora.yaml`:

```yaml
run:
  name: gpu-smoke-qlora
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
  method: qlora
  scope: vision_decoder
  qlora:
    quant_type: nf4
    compute_dtype: bfloat16

train:
  epochs: 25
  batch_size: 1
  grad_accum_steps: 1
  optimizer: adamw8bit
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

- [ ] **Step 4b: Validate the YAML loads against `TrainConfig`**

```bash
uv run python -c "from esam3.config.loader import load_config; cfg = load_config('configs/examples/gpu_smoke_qlora.yaml'); print('ok', cfg.run.name, cfg.peft.method, cfg.train.optimizer)"
```
Expected: `ok gpu-smoke-qlora qlora adamw8bit`.

- [ ] **Step 4c: Commit**

```bash
git add configs/examples/gpu_smoke_qlora.yaml
git commit -m "examples: add gpu_smoke_qlora.yaml for the QLoRA GPU smoke test"
```

---

## Task 5: Add the CPU-only `test_config_examples.py` unit test

**Files:**
- Create: `tests/unit/test_config_examples.py`

A parametrized test that `load_config()`s every YAML under `configs/examples/`. Today that means four YAMLs: `coco_text_lora.yaml`, `coco_text_qlora.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml`. New YAMLs automatically gain coverage. See spec §5.3.

- [ ] **Step 5a: Write the test**

Create `tests/unit/test_config_examples.py`:

```python
"""Every YAML under configs/examples/ must validate against TrainConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.config.loader import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "examples"


@pytest.mark.parametrize(
    "yaml_path",
    sorted(CONFIG_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_example_config_validates(yaml_path: Path) -> None:
    cfg = load_config(yaml_path)
    assert cfg.run.name  # smoke: schema parsed and produced a populated TrainConfig
```

- [ ] **Step 5b: Run the test**

```bash
uv run pytest tests/unit/test_config_examples.py -v
```
Expected: 4 tests pass — one each for `coco_text_lora.yaml`, `coco_text_qlora.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml`. (If Tasks 3 or 4 were skipped, the count drops to 2.)

- [ ] **Step 5c: ruff + mypy**

```bash
uv run ruff check tests/unit/test_config_examples.py && uv run mypy tests/unit/test_config_examples.py
```
Expected: both clean.

- [ ] **Step 5d: Commit**

```bash
git add tests/unit/test_config_examples.py
git commit -m "test(config): validate every configs/examples/*.yaml against TrainConfig"
```

---

## Task 6: Rewrite `tests/gpu/test_real_train_overfits.py`

**Files:**
- Rewrite: `tests/gpu/test_real_train_overfits.py`

Drops the in-code `TrainConfig` literal, loads `gpu_smoke_lora.yaml`, calls `run_training(cfg)`, monkeypatches `esam3.train.runner.build_tracker` (the consumer's namespace — see spec §4.2), and adds VRAM-ceiling and finite-value assertions. See spec §4.3 / §4.4.

- [ ] **Step 6a: Rewrite the test file**

Replace the entire contents of `tests/gpu/test_real_train_overfits.py` with:

```python
"""50-step LoRA overfit on tiny_coco via run_training(gpu_smoke_lora.yaml).

Gated by `@pytest.mark.gpu`, `@requires_compatible_gpu`, and
`@requires_checkpoint`. Not in CI by default. Run with:
    pytest -m gpu tests/gpu/test_real_train_overfits.py -v

This test exercises the same `run_training(cfg)` seam that `esam3 train` uses,
so the YAML at configs/examples/gpu_smoke_lora.yaml is both the user-facing
example and the test's source of truth (modulo the monkeypatched tracker).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from esam3.config.loader import load_config
from esam3.train.runner import run_training
from tests.gpu.conftest import _RecordingTracker

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_lora.yaml"
LOSS_RATIO_CEIL = 0.70
VRAM_CEIL_GB = 14.0


def test_overfits_in_50_steps(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    # Patch the consumer's namespace (esam3.train.runner) rather than the producer
    # (esam3.tracking) — runner.py does `from esam3.tracking import build_tracker`
    # at import time, so the bound name lives in runner.__dict__. See spec §4.2.
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

- [ ] **Step 6b: Confirm the test collects cleanly on CPU**

```bash
uv run pytest tests/gpu/test_real_train_overfits.py --collect-only -q
```
Expected: `1 test collected`. No `ImportError`, no marker warnings.

- [ ] **Step 6c: Confirm the test correctly skips on a CPU box**

```bash
uv run pytest tests/gpu/test_real_train_overfits.py -v
```
Expected: `1 skipped` with reason `real SAM 3.1 forward requires a CUDA GPU with CC >= 7.5` or `real SAM 3.1 checkpoint not present locally` (whichever skip-marker matches first on this machine). The key requirement: no collection failure, no `ImportError`.

- [ ] **Step 6d: ruff + mypy on the rewritten file**

```bash
uv run ruff check tests/gpu/test_real_train_overfits.py && uv run mypy tests/gpu/test_real_train_overfits.py
```
Expected: both clean.

- [ ] **Step 6e: Commit**

```bash
git add tests/gpu/test_real_train_overfits.py
git commit -m "test(gpu): rewrite LoRA smoke to load YAML + call run_training(cfg)"
```

---

## Task 7: Add `tests/gpu/test_real_train_qlora.py`

**Files:**
- Create: `tests/gpu/test_real_train_qlora.py`

Sibling of the LoRA smoke with the QLoRA YAML, looser thresholds, and `requires_bnb` skip. See spec §4.3 / §4.4.

- [ ] **Step 7a: Write the test**

Create `tests/gpu/test_real_train_qlora.py`:

```python
"""50-step QLoRA overfit on tiny_coco via run_training(gpu_smoke_qlora.yaml).

Gated by `@pytest.mark.gpu`, `@requires_compatible_gpu`, `@requires_checkpoint`,
plus a per-test `skipif(not _bnb_available())`. Not in CI by default. Run with:
    pytest -m gpu tests/gpu/test_real_train_qlora.py -v

This test exercises the same `run_training(cfg)` seam that `esam3 train` uses,
proving 4-bit base + bf16 LoRA + 8-bit optimizer trains end-to-end on real
SAM 3.1. Loss-ratio and VRAM ceilings are looser than the LoRA smoke because
4-bit base converges slightly slower and pairs with adamw8bit on the 12 GB
recipe (architecture §6).
"""

from __future__ import annotations

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

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_qlora.yaml"
LOSS_RATIO_CEIL = 0.75
VRAM_CEIL_GB = 10.0


@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_qlora_overfits_in_50_steps(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    # Patch the consumer's namespace (esam3.train.runner) rather than the producer
    # (esam3.tracking). See spec §4.2.
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

- [ ] **Step 7b: Confirm the test collects cleanly on CPU**

```bash
uv run pytest tests/gpu/test_real_train_qlora.py --collect-only -q
```
Expected: `1 test collected`. No `ImportError`, no marker warnings.

- [ ] **Step 7c: Confirm the test correctly skips on a CPU/no-bnb box**

```bash
uv run pytest tests/gpu/test_real_train_qlora.py -v
```
Expected: `1 skipped`. Skip reason will be one of: `bitsandbytes not installed`, `real SAM 3.1 forward requires a CUDA GPU with CC >= 7.5`, or `real SAM 3.1 checkpoint not present locally` — whichever fires first. No collection failure.

- [ ] **Step 7d: ruff + mypy on the new file**

```bash
uv run ruff check tests/gpu/test_real_train_qlora.py && uv run mypy tests/gpu/test_real_train_qlora.py
```
Expected: both clean.

- [ ] **Step 7e: Commit**

```bash
git add tests/gpu/test_real_train_qlora.py
git commit -m "test(gpu): add QLoRA overfit smoke via run_training(gpu_smoke_qlora.yaml)"
```

---

## Task 8: Append four deferred items to `logs/TODO.md`

**Files:**
- Create-or-append: `logs/TODO.md`

Four items are appended verbatim from spec §7. The file does not exist on this branch today; create it if missing, append if present. See spec §7.

- [ ] **Step 8a: Ensure `logs/TODO.md` exists**

```bash
mkdir -p logs && touch logs/TODO.md
```
Expected: no output, exit 0.

- [ ] **Step 8b: Append the four entries**

```bash
cat >> logs/TODO.md <<'EOF'

## spec/smoke-test deferrals (2026-05-18)

- **real-GPU resume smoke** — `Trainer.fit(resume_from=...)` exercised end-to-end on real SAM3.1. CPU integration coverage exists (`tests/integration/test_train_*`); the real-GPU variant requires a two-phase smoke (initial fit → checkpoint → resume) that doesn't fit the 50-step overfit shape.
- **real-GPU eval smoke** — `esam3 eval --checkpoint runs/.../adapter` against real SAM3.1. The current `tests/gpu/` tier only covers training. CPU eval coverage exists via the stub-model integration tests.
- **HF dataset adapter on real GPU** — `data.format: hf` runs end-to-end through `run_training` today (registered via `@register`), but no real-GPU test exercises the HF path against SAM3.1.
- **Trainer grad-norm key naming** — Trainer emits `grad_norm`; consider standardizing on `grad_norm/total` for namespace consistency with `loss/total`. The finite-value assertion in the GPU smoke tests already covers `grad_norm` and would continue to cover a renamed `grad_norm/total` automatically.
EOF
```

- [ ] **Step 8c: Verify the file content**

```bash
tail -n 7 logs/TODO.md
```
Expected: the four bullet lines plus the `## spec/smoke-test deferrals (2026-05-18)` header are visible, in order.

- [ ] **Step 8d: Commit**

```bash
git add logs/TODO.md
git commit -m "docs(todo): defer real-GPU resume/eval/HF + grad-norm naming"
```

---

## Task 9: Final verification sweep (CPU-only — this is the merge gate)

The exit gate is CPU-only. Real-GPU runs of the smoke tests are post-merge follow-up (spec §6). What must be green here: `ruff`, `mypy`, full `pytest`, and clean collection on both GPU tests.

- [ ] **Step 9a: Full ruff**

```bash
uv run ruff check . && uv run ruff format --check .
```
Expected: clean.

- [ ] **Step 9b: Full mypy --strict over the project**

```bash
uv run mypy
```
Expected: `Success: no issues found`.

- [ ] **Step 9c: Full pytest (unit + integration)**

```bash
uv run pytest tests/unit tests/integration -q
```
Expected: full pass.

- [ ] **Step 9d: GPU tests collect cleanly on CPU (no ImportError, no marker warning)**

```bash
uv run pytest tests/gpu --collect-only -q -W error::pytest.PytestUnknownMarkWarning
```
Expected: `2 tests collected`. No warnings escalated to errors.

- [ ] **Step 9e: GPU tests skip cleanly on CPU**

```bash
uv run pytest tests/gpu -v
```
Expected: both tests skipped (skip reasons from `requires_compatible_gpu`, `requires_checkpoint`, or `requires_bnb`). No failures, no collection errors.

- [ ] **Step 9f: Confirm the four shipped example YAMLs all validate**

```bash
uv run pytest tests/unit/test_config_examples.py -v
```
Expected: 4 passed — `coco_text_lora.yaml`, `coco_text_qlora.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml`.

---

## Spec coverage map

| Spec section | Tasks |
|---|---|
| §1 Current State — `test_real_train_overfits.py` rewrite | 6 |
| §1 Current State — new `test_real_train_qlora.py` | 7 |
| §1 Current State — `_RecordingTracker` lifted to `tests/gpu/conftest.py` | 2 |
| §1 Current State — `_bnb_available()` lifted to `tests/gpu/conftest.py` | 2 |
| §1 Current State — `requires_bnb` marker registered | 1 |
| §1 Current State — `tests/unit/test_config_examples.py` drive-by | 5 |
| §1 Current State — `logs/TODO.md` appended | 8 |
| §3 File Map — `configs/examples/gpu_smoke_lora.yaml` | 3 |
| §3 File Map — `configs/examples/gpu_smoke_qlora.yaml` | 4 |
| §4.1 Shared helpers (`_RecordingTracker`, `_bnb_available`) | 2 |
| §4.2 Tracker injection seam (`monkeypatch esam3.train.runner.build_tracker`) | 6, 7 |
| §4.3 Test skeleton (load_config + overrides + run_training + assertions) | 6, 7 |
| §4.4 Per-test parameters (LoRA 0.70/14GB, QLoRA 0.75/10GB) | 6, 7 |
| §4.5 `requires_bnb` marker registration | 1 |
| §4.6 Finite-value assertion covers all 11 current scalar keys | 6, 7 |
| §4.7 YAML uses `tracking.backend: none`; test monkeypatches anyway | 3, 4, 6, 7 |
| §5.1 `gpu_smoke_lora.yaml` content | 3 |
| §5.2 `gpu_smoke_qlora.yaml` content | 4 |
| §5.3 `tests/unit/test_config_examples.py` | 5 |
| §6 Exit Criteria — Code checkboxes | 1–8 |
| §6 Exit Criteria — CPU tests green | 9a, 9b, 9c, 9f |
| §6 Exit Criteria — GPU tests collect/skip cleanly on CPU | 9d, 9e |
| §7 Deferred items appended verbatim | 8 |
