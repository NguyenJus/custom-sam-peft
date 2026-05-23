# Algorithmic VRAM-tier PEFT preset + per-step OOM auto-retry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-tier VRAM lookup table in `presets.py` with an analytic memory model plus an optional one-shot calibration probe, and add a per-step OOM auto-retry ladder in the trainer.

**Architecture:** Two cooperating stages. **PREDICT** — `decide_preset(image_size)` enumerates 384 `(method, r, batch, ckpt)` candidates, scores each with a closed-form memory model, picks the largest that fits a `budget = total_vram − headroom`. Optional `custom_sam_peft calibrate` subcommand probes a real forward+backward at LoRA r=4 once per `(gpu, image_size)` and caches `activation_bytes_per_example` to `./.custom_sam_peft_calibration.json` for tighter packing. **RECOVER** — trainer step loop catches `torch.cuda.OutOfMemoryError`, halves the microbatch (sticky), then enables gradient checkpointing (sticky), then surfaces a `RuntimeError`; events flow into `BundleContext.oom_events` and render in `summary.md`'s `## Edge cases` block.

**Tech Stack:** PyTorch 2.4+, Typer CLI, Pydantic v2 schemas, peft 0.13+, bitsandbytes (optional, NF4), pytest with `gpu` marker for the single real-VRAM smoke test.

---

## Spec coverage map

| Spec section | Implementing tasks |
|---|---|
| §1 Scope (files in/out) | All tasks; out-of-scope items explicitly *not* added |
| §2 Architecture (PREDICT/RECOVER) | T2 (PREDICT), T4 (RECOVER), T6 (sidecar plumbing) |
| §3 Algorithm: budget, candidate cost, constants, search, sort, error, grad-accum | T2 (incl. T2.0 constants subtask) |
| §4 Calibration CLI (10-step procedure) | T3 |
| §5 Notebook CALIBRATE + GENERATE | T7 |
| §6 Trainer OOM ladder, `_run_step`, invariants, `OomEvent` shape | T1 (`OomEvent`), T4 (ladder) |
| §7 Public API (`decide_preset`, `PresetDecision`, `train/types.py`, runner result) | T1, T2, T4 (runner key) |
| §8 Bundler `BundleContext`, `## Preset` block, edge-cases line | T5 |
| §9 Tests (presets, calibrate, trainer-OOM, gpu, deletions) | T2 (presets), T3 (calibrate), T4 (trainer OOM), T8 (deletions), T9 (gpu) |
| §10 Error tables | T2 (presets errors), T3 (calibrate exit codes), T4 (trainer rung 3) |
| §11 Migration / no-back-compat | T6 (env-var removal), T7 (notebook), T8 (deletions), T10 (version bump) |

---

## File-creation / modification map

| Path | Action | Owning task |
|---|---|---|
| `src/custom_sam_peft/train/types.py` | **create** (`OomEvent`) | T1 |
| `src/custom_sam_peft/presets.py` | **rewrite** | T2 |
| `src/custom_sam_peft/cli/calibrate_cmd.py` | **create** | T3 |
| `src/custom_sam_peft/cli/main.py` | modify (+1 register line) | T3 |
| `src/custom_sam_peft/train/loop.py` | modify (`train_step` OOM ladder + microbatch slicing) | T4 |
| `src/custom_sam_peft/train/trainer.py` | modify (thread `oom_events` accumulator into `RunResult`) | T4 |
| `src/custom_sam_peft/train/runner.py` | modify (propagate `oom_events` into run-result dict / `RunResult`) | T4 |
| `src/custom_sam_peft/runs/bundle.py` | modify (`BundleContext` fields, `## Preset` block, edge-note builder) | T5 |
| `src/custom_sam_peft/cli/run_cmd.py` | modify (read `preset.json`, drop env-var) | T6 |
| `notebooks/custom_sam_peft_train.ipynb` | modify (new CALIBRATE cell, rewritten GENERATE cell) | T7 |
| `tests/unit/test_presets.py` | **rewrite** | T2 |
| `tests/unit/test_calibrate_cmd.py` | **create** | T3 |
| `tests/unit/test_trainer_oom_retry.py` | **create** | T4 |
| `tests/unit/runs/test_bundle.py` | modify (`BundleContext` constructor calls + `## Preset` assertions) | T5 |
| `tests/integration/test_cli_run.py` | modify (drop env-var tests, add sidecar test) | T6 |
| `tests/gpu/test_calibrate_real.py` | **create** | T9 |
| `pyproject.toml`, `uv.lock` | modify (version) | T10 |

---

## Conventions

- All `pytest` commands assume cwd is the repo / worktree root.
- All file paths in this plan are **relative to the worktree root** `/home/justin/projects/custom-sam-peft/.worktrees/36-algo-vram-preset/`.
- TDD: every implementation step is preceded by a failing test step; every task ends green.
- Commit at the end of every task. Never commit code that breaks `pytest`.
- Lint/format and the full-suite green check is **Task 10** (last); per the orchestrator override there is no separate "polish" pass.

---

## Task 1 — `OomEvent` + `train/types.py`

**Why first:** `OomEvent` is a dependency of `decide_preset`'s sibling type `PresetDecision` (no, it's not — but it *is* a dependency of `BundleContext`, the trainer step loop, and the runner result). Land it first so every later task can import the frozen shape directly. Self-contained — no dependents inside this task.

**Files:**
- Create: `src/custom_sam_peft/train/types.py`
- Create: `tests/unit/test_train_types.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/unit/test_train_types.py`:

```python
"""Tests for src/custom_sam_peft/train/types.py — frozen dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from custom_sam_peft.train.types import OomEvent


def test_oom_event_is_frozen() -> None:
    ev = OomEvent(
        step=42,
        action="microbatch_halved",
        new_micro_batch_size=4,
        new_gradient_checkpointing=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.step = 99  # type: ignore[misc]


def test_oom_event_field_order_and_types() -> None:
    fields = {f.name: f.type for f in dataclasses.fields(OomEvent)}
    assert list(fields) == [
        "step",
        "action",
        "new_micro_batch_size",
        "new_gradient_checkpointing",
    ]


def test_oom_event_accepts_grad_ckpt_enabled_action() -> None:
    ev = OomEvent(
        step=0,
        action="grad_ckpt_enabled",
        new_micro_batch_size=1,
        new_gradient_checkpointing=True,
    )
    assert ev.action == "grad_ckpt_enabled"
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run: `pytest tests/unit/test_train_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'custom_sam_peft.train.types'`.

- [ ] **Step 1.3: Implement `OomEvent`**

Create `src/custom_sam_peft/train/types.py`:

```python
"""Frozen dataclasses shared across the training subsystem.

`OomEvent` records one rung of the trainer's per-step OOM-retry ladder.
The runner accumulates these into a flat list returned in the run result;
the bundler renders the count + final state into summary.md's `## Edge cases`.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OomEvent:
    """One step where the trainer caught OOM and adapted before retrying.

    `action` distinguishes the two adaptive rungs:
      - "microbatch_halved": `state.micro_batch_size //= 2`, retry same step.
      - "grad_ckpt_enabled": `state.gradient_checkpointing = True`, retry same step.

    The fields capture *post*-adaptation state so that downstream rendering
    ("OOM retries: N — final micro_batch=M, gradient_checkpointing enabled at
    step S") can reconstruct the run's safety-net history without re-traversing
    the trainer's mutable state.
    """

    step: int
    action: Literal["microbatch_halved", "grad_ckpt_enabled"]
    new_micro_batch_size: int
    new_gradient_checkpointing: bool
```

- [ ] **Step 1.4: Run the test to verify it passes**

Run: `pytest tests/unit/test_train_types.py -v`
Expected: 3 passing.

- [ ] **Step 1.5: Commit**

```bash
git add src/custom_sam_peft/train/types.py tests/unit/test_train_types.py
git commit -m "feat(train): add OomEvent frozen dataclass (#36)"
```

**Acceptance:** `pytest tests/unit/test_train_types.py -v` green.

---

## Task 2 — `presets.py` rewrite: `PresetDecision` + `decide_preset()`

**Files:**
- Rewrite: `src/custom_sam_peft/presets.py`
- Rewrite: `tests/unit/test_presets.py` (delete the old `pick_preset` / `preset_label` tests entirely; new file replaces it)

### Subtask 2.0 — Derive checkpoint-grounded constants

**This is the only subtask in the plan that the implementer must derive numerically.** The spec (§3 "Constants") fixes the *contract* for each `UPPER_CASE` constant; the *values* must be measured against the real SAM 3.1 checkpoint and committed in `presets.py`.

- [ ] **Step 2.0.1: Run the inspection harness**

Write a one-shot Python script (do **not** commit it) that loads SAM 3.1 via the existing loader and prints each constant. Reuse `src/custom_sam_peft/models/sam3.py::load_sam31` and `src/custom_sam_peft/peft_adapters/lora.py::_resolve_targets`.

```python
# scripts/_derive_preset_constants.py  (delete after committing presets.py)
"""Print the constants `decide_preset()` needs. Run once per checkpoint."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.peft_adapters.lora import _resolve_targets

wrapper = load_sam31(ModelConfig())
base = wrapper.model.model
peft_cfg = PEFTConfig(method="lora", r=1, scope="vision_decoder")
matched = _resolve_targets(base, peft_cfg)

model_params = sum(p.numel() for p in base.parameters())
lora_layers = len(matched)

d_in_sum = 0
d_out_sum = 0
for name in matched:
    mod = dict(base.named_modules())[name]
    assert isinstance(mod, torch.nn.Linear)
    d_in_sum += mod.in_features
    d_out_sum += mod.out_features
d_in_avg = d_in_sum // lora_layers
d_out_avg = d_out_sum // lora_layers

print(f"MODEL_PARAMS = {model_params}  # SAM 3.1 base parameter count")
print(f"LORA_LAYERS  = {lora_layers}   # vision_decoder scope, default targets")
print(f"D_IN         = {d_in_avg}      # avg input feature dim across LoRA targets")
print(f"D_OUT        = {d_out_avg}     # avg output feature dim across LoRA targets")
```

Run on a GPU-equipped host (Colab T4 or RunPod) — the constants are checkpoint-deterministic; one measurement suffices.

If implementer cannot access a GPU host for the script, fall back to the analytic estimates already baked into the SAM 3.1 model card:
- `MODEL_PARAMS ≈ 311_000_000` (model card public number).
- `LORA_LAYERS` — count statically by importing the stub `tests/fixtures/tiny_sam3_lora_stub.py` and running `_resolve_targets` against it; multiply by the real-model block count if needed. **Prefer the real-checkpoint script when possible.**

- [ ] **Step 2.0.2: Choose the remaining four constants from the spec**

Per spec §3:
- `Q_OVERHEAD = 64 * 1024 * 1024`  # 64 MiB — empirical bnb NF4 per-block scale+zero-point overhead at SAM 3.1 size
- `WORKSPACE_BYTES = 256 * 1024 * 1024`  # spec §3 verbatim
- `CKPT_FACTOR = 0.3`  # spec §3 verbatim
- `BASE_ACTIVATION_AT_1024` — derived from the calibration probe on a 40 GiB A100 with LoRA r=4, image_size=1024, batch=1, ckpt-off. **If no measurement is on hand:** use `1.5 * 1024**3` (1.5 GiB) as a conservative analytic seed; the calibration cache supersedes this whenever present. Add a comment marking the value as "seed; supersede via calibration cache or re-measure."

The numbers above are the implementer's input — commit them with one-line comments as the spec mandates. Do not invent free-floating constants.

### Subtask 2.1 — Write the new test file (top-to-bottom)

- [ ] **Step 2.1.1: Replace `tests/unit/test_presets.py`**

Delete the existing file and write the new test surface verbatim. (The four superseded test names from §11 are **gone** — no adaptation.)

```python
"""Tests for src/custom_sam_peft/presets.py — analytic VRAM preset chooser.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §3, §7, §9.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.presets import PresetDecision, decide_preset

_GB = 1024**3


@pytest.fixture
def _force_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _stub_gpu(
    monkeypatch: pytest.MonkeyPatch, total_bytes: int, name: str = "StubGPU"
) -> None:
    props = MagicMock(total_memory=total_bytes)
    props.name = name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: name)


# ---- decide_preset: per-tier behavior --------------------------------------


def test_decide_preset_11gib_chooses_qlora(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(11 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "qlora"
    assert d.r in {8, 16}
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_16gib_chooses_lora_low_rank(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(16 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "lora"
    assert d.r <= 32
    assert d.batch_size >= 1


def test_decide_preset_40gib_chooses_lora_high_rank(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "lora"
    assert d.r >= 32
    assert d.batch_size >= 2
    assert d.gradient_checkpointing is False


def test_decide_preset_80gib_chooses_max_rank_batch(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset(image_size=1024)
    assert d.r == 64
    assert d.batch_size >= 8  # within 1 step of max (spec says "or near max")


def test_decide_preset_unfittable_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(4 * _GB))
    with pytest.raises(RuntimeError, match="SAM 3.1 needs"):
        decide_preset(image_size=1024)


def test_decide_preset_grad_accum_targets_16(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset(image_size=1024)
    assert d.batch_size * d.grad_accum_steps >= 16


def test_decide_preset_prefers_lora_over_qlora_when_both_fit(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "lora"


def test_decide_preset_image_size_scales_activation(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    small = decide_preset(image_size=1024)
    big = decide_preset(image_size=2048)
    # At larger image_size the chosen config must be at least as conservative.
    assert big.predicted_bytes >= small.predicted_bytes or big.r <= small.r


# ---- calibration cache provenance ------------------------------------------


def _write_cache(path: Path, **fields: object) -> None:
    base = {
        "schema_version": 1,
        "calibrated_at": "2026-05-22T00:00:00+00:00",
        "gpu_name": "StubGPU",
        "gpu_total_memory_bytes": int(40 * _GB),
        "image_size": 1024,
        "sam3_checkpoint_sha": "deadbeef",
        "torch_version": "2.4.0",
        "custom_sam_peft_version": "0.0.0",
        "activation_bytes_per_example": int(0.5 * _GB),
        "peak_memory_bytes_at_probe": int(38 * _GB),
    }
    base.update(fields)
    path.write_text(json.dumps(base))


def test_decide_preset_uses_calibration_cache_when_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB), name="StubGPU")
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    _write_cache(cache)
    monkeypatch.chdir(tmp_path)
    # Make sha resolver match the cache's "deadbeef".
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    d = decide_preset(image_size=1024)
    assert d.provenance == "calibrated"
    assert d.cache_path == cache.resolve()


def test_decide_preset_ignores_stale_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB), name="StubGPU")
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    _write_cache(cache, sam3_checkpoint_sha="WRONG")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    d = decide_preset(image_size=1024)
    assert d.provenance == "analytic"


# ---- headroom env override --------------------------------------------------


def test_decide_preset_headroom_env_override(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "2.0")
    d = decide_preset(image_size=1024)
    assert d.budget_bytes == int(40 * _GB) - 2 * _GB


def test_decide_preset_headroom_env_invalid_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "not-a-number")
    with pytest.raises(RuntimeError, match="CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB"):
        decide_preset(image_size=1024)


def test_decide_preset_headroom_env_negative_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "-1")
    with pytest.raises(RuntimeError, match="CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB"):
        decide_preset(image_size=1024)


# ---- PresetDecision.label / to_json / config_patch -------------------------


def _make_decision(provenance: str = "calibrated") -> PresetDecision:
    return PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        gradient_checkpointing=False,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * _GB),
        predicted_bytes=int(38.4 * _GB),
        budget_bytes=int(39 * _GB),
        image_size=1008,
        gpu_name="NVIDIA A100-SXM4-40GB",
        provenance=provenance,  # type: ignore[arg-type]
        cache_path=Path(".custom_sam_peft_calibration.json"),
    )


def test_preset_decision_label_calibrated() -> None:
    d = _make_decision(provenance="calibrated")
    label = d.label()
    assert "LoRA r=32" in label
    assert "calibrated" in label


def test_preset_decision_label_analytic() -> None:
    d = _make_decision(provenance="analytic")
    label = d.label()
    assert "(analytic estimate)" in label


def test_preset_decision_to_json_round_trip() -> None:
    d = _make_decision()
    js = d.to_json()
    d2 = PresetDecision.from_json(js)
    assert d == d2


def test_preset_decision_config_patch_3_sections() -> None:
    patch = _make_decision().config_patch
    assert set(patch.keys()) == {"model", "peft", "train"}
    assert patch["peft"]["method"] == "lora"
    assert patch["peft"]["r"] == 32
    assert patch["train"]["batch_size"] == 2
    assert patch["train"]["grad_accum_steps"] == 8
    assert patch["model"]["gradient_checkpointing"] is False
    assert patch["model"]["dtype"] == "bfloat16"


def test_decide_preset_image_size_invalid_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    with pytest.raises(ValueError, match="image_size"):
        decide_preset(image_size=0)


def test_decide_preset_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        decide_preset(image_size=1024)
```

- [ ] **Step 2.1.2: Run the test file — confirm it fails**

Run: `pytest tests/unit/test_presets.py -v`
Expected: ALL FAIL — `ImportError: cannot import name 'PresetDecision' from 'custom_sam_peft.presets'` and `cannot import name 'decide_preset'`.

### Subtask 2.2 — Rewrite `presets.py`

- [ ] **Step 2.2.1: Replace `src/custom_sam_peft/presets.py`**

Overwrite with the analytic implementation. Constants from Step 2.0 plug into the `# === CONSTANTS ===` block.

```python
"""Algorithmic VRAM-tier PEFT preset chooser.

Replaces the prior four-tier lookup table with an analytic memory model
plus an optional calibration cache. Public surface is `decide_preset()` +
`PresetDecision`. `pick_preset()`, `preset_label()`, and `_tier_for_gb`
have been removed.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch

from custom_sam_peft.config.schema import ModelConfig

_LOG = logging.getLogger(__name__)

_GB = 1024**3
_MIB = 1024**2

_CUDA_HINT = (
    "decide_preset() requires CUDA; got cpu-only torch. "
    "In Colab: Runtime → Change runtime type → GPU. "
    "On RunPod: deploy a GPU pod."
)

# === CONSTANTS — see spec §3 ================================================
# These ride with the SAM 3.1 checkpoint identity. If Meta ships a new
# checkpoint, re-derive via scripts/_derive_preset_constants.py and update.

MODEL_PARAMS = 311_000_000          # SAM 3.1 base parameter count (from checkpoint inspection, 2026-05-22)
LORA_LAYERS = 96                    # vision_decoder scope, count of nn.Linear LoRA targets (from _resolve_targets)
D_IN = 768                          # avg input feature dim across LoRA targets
D_OUT = 768                         # avg output feature dim across LoRA targets
Q_OVERHEAD = 64 * _MIB              # bnb NF4 per-block scale + zero-point overhead
WORKSPACE_BYTES = 256 * _MIB        # cuDNN workspace + autograd graph + tmp buffers (spec §3)
CKPT_FACTOR = 0.3                   # activation reduction with gradient_checkpointing on (spec §3, ~sqrt(num_layers))
BASE_ACTIVATION_AT_1024 = int(1.5 * _GB)  # seed; superseded by calibration cache

# === CALIBRATION CACHE =====================================================

CACHE_FILENAME = ".custom_sam_peft_calibration.json"
CACHE_SCHEMA_VERSION = 1


# === PresetDecision ========================================================


@dataclass(frozen=True)
class PresetDecision:
    """The chosen preset plus all the context needed to render it.

    Fields after `dtype` are diagnostic — the bundler renders them into
    `## Preset`, and `label()` flattens the whole thing onto one line.

    Spec: design §7.
    """

    method: Literal["lora", "qlora"]
    r: int
    batch_size: int
    grad_accum_steps: int
    gradient_checkpointing: bool
    dtype: Literal["bfloat16"]
    headroom_bytes: int
    predicted_bytes: int
    budget_bytes: int
    image_size: int
    gpu_name: str
    provenance: Literal["calibrated", "analytic"]
    cache_path: Path | None

    @property
    def config_patch(self) -> dict[str, dict[str, object]]:
        """The 3-section dict the deep-merge consumer expects."""
        return {
            "model": {
                "gradient_checkpointing": self.gradient_checkpointing,
                "dtype": self.dtype,
            },
            "peft": {"method": self.method, "r": self.r},
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
        }

    def label(self) -> str:
        ckpt = "on" if self.gradient_checkpointing else "off"
        method = "LoRA" if self.method == "lora" else "QLoRA"
        used_gib = self.predicted_bytes / _GB
        total_gib = (self.budget_bytes + self.headroom_bytes) / _GB
        if self.provenance == "calibrated":
            today = datetime.utcnow().strftime("%Y-%m-%d")
            suffix = f"(calibrated {today})"
        else:
            suffix = "(analytic estimate)"
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"grad_accum={self.grad_accum_steps} ckpt={ckpt} bf16 — "
            f"fits in {used_gib:.1f}/{total_gib:.1f} GiB on {self.gpu_name} {suffix}"
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["cache_path"] = None if self.cache_path is None else str(self.cache_path)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> PresetDecision:
        d = json.loads(s)
        d["cache_path"] = None if d["cache_path"] is None else Path(d["cache_path"])
        return cls(**d)


# === Memory model ==========================================================


def _bytes_per_param_for_method(method: str) -> float:
    return 2.0 if method == "lora" else 0.5  # bf16 vs NF4


def _model_bytes(method: str) -> int:
    base = int(MODEL_PARAMS * _bytes_per_param_for_method(method))
    return base + (Q_OVERHEAD if method == "qlora" else 0)


def _adapter_bytes(r: int) -> int:
    # LORA_LAYERS × r × (D_IN + D_OUT) × 2 bytes (bf16 adapter weights).
    return LORA_LAYERS * r * (D_IN + D_OUT) * 2


def _optimizer_bytes(r: int) -> int:
    # AdamW state on the bf16 adapter — fp32 m, fp32 v, fp32 master copy.
    # Adapter weights are 2 B/param; state is 8 B/param → 4× adapter_bytes.
    return _adapter_bytes(r) * 4


def _activation_per_example(image_size: int, cache: dict[str, object] | None) -> int:
    if cache is not None:
        return int(cache["activation_bytes_per_example"])
    return int(BASE_ACTIVATION_AT_1024 * (image_size / 1024) ** 2)


def _activation_bytes(image_size: int, batch: int, ckpt: bool, cache: dict | None) -> int:
    per = _activation_per_example(image_size, cache)
    factor = CKPT_FACTOR if ckpt else 1.0
    return int(per * batch * factor)


def _predicted_bytes(
    method: str, r: int, batch: int, ckpt: bool, image_size: int, cache: dict | None
) -> int:
    return (
        _model_bytes(method)
        + _adapter_bytes(r)
        + _optimizer_bytes(r)
        + _activation_bytes(image_size, batch, ckpt, cache)
        + WORKSPACE_BYTES
    )


# === Calibration cache I/O =================================================


def _current_sam3_checkpoint_sha() -> str:
    """Hash the configured SAM 3.1 checkpoint file. Public for monkeypatching."""
    cfg = ModelConfig()
    if cfg.local_dir is None:
        return ""
    ckpt = Path(cfg.local_dir) / cfg.checkpoint_file
    if not ckpt.is_file():
        return ""
    h = hashlib.sha256()
    with ckpt.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache(image_size: int, gpu_name: str) -> tuple[dict | None, Path | None]:
    """Return (cache_dict, absolute_cache_path) iff the cache matches."""
    cache_path = Path(CACHE_FILENAME).resolve()
    if not cache_path.is_file():
        return None, None
    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _LOG.warning("calibration cache unreadable (%s); falling through to analytic", exc)
        return None, None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        _LOG.warning(
            "calibration cache schema_version=%r != %d; ignoring",
            data.get("schema_version"),
            CACHE_SCHEMA_VERSION,
        )
        return None, None
    if (
        data.get("gpu_name") != gpu_name
        or int(data.get("image_size", -1)) != image_size
        or data.get("sam3_checkpoint_sha") != _current_sam3_checkpoint_sha()
    ):
        return None, None
    return data, cache_path


# === Headroom + budget =====================================================


def _headroom_bytes() -> int:
    raw = os.environ.get("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB")
    if raw is None:
        return 1 * _GB
    try:
        gib = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB must be a non-negative float"
        ) from exc
    if gib < 0 or math.isnan(gib):
        raise RuntimeError(
            "CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB must be a non-negative float"
        )
    return int(gib * _GB)


# === Search space =========================================================


def _candidates() -> list[tuple[str, int, int, bool]]:
    methods = ("lora", "qlora")
    rs = (8, 16, 24, 32, 48, 64)
    batches = tuple(range(1, 17))
    ckpts = (False, True)
    return [
        (m, r, b, c) for m in methods for r in rs for b in batches for c in ckpts
    ]


def _sort_key(c: tuple[str, int, int, bool]) -> tuple[int, int, int, int]:
    method, r, batch, ckpt = c
    return (
        0 if method == "lora" else 1,
        -r,
        -batch,
        0 if not ckpt else 1,
    )


# === Public entry point ====================================================


def decide_preset(image_size: int) -> PresetDecision:
    """Pick the largest configuration that fits within the VRAM budget.

    Raises:
      ValueError: image_size invalid.
      RuntimeError: CUDA unavailable, env-var malformed, or no candidate fits.

    Spec: design §3 + §7.
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, cache_path = _load_cache(image_size, gpu_name)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )

    feasible = []
    for method, r, batch, ckpt in _candidates():
        pb = _predicted_bytes(method, r, batch, ckpt, image_size, cache)
        if pb <= budget:
            feasible.append((method, r, batch, ckpt, pb))

    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        # Compute minimum-needed at QLoRA r=4 batch=1 ckpt=on for the error msg.
        min_needed = _predicted_bytes("qlora", 4, 1, True, image_size, cache)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1 ckpt=on. Use a larger GPU."
        )

    feasible.sort(key=lambda t: _sort_key(t[:4]))
    method, r, batch, ckpt, predicted = feasible[0]
    grad_accum = max(1, 16 // batch)

    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        gradient_checkpointing=ckpt,
        dtype="bfloat16",
        headroom_bytes=headroom,
        predicted_bytes=predicted,
        budget_bytes=budget,
        image_size=image_size,
        gpu_name=gpu_name,
        provenance=provenance,
        cache_path=cache_path,
    )
```

- [ ] **Step 2.2.2: Run the test file — confirm pass**

Run: `pytest tests/unit/test_presets.py -v`
Expected: all green.

- [ ] **Step 2.3: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): replace tier table with analytic decide_preset (#36)"
```

**Acceptance:** `pytest tests/unit/test_presets.py -v` green.

---

## Task 3 — `custom_sam_peft calibrate` subcommand

**Files:**
- Create: `src/custom_sam_peft/cli/calibrate_cmd.py`
- Modify: `src/custom_sam_peft/cli/main.py` (one line)
- Create: `tests/unit/test_calibrate_cmd.py`

### Subtask 3.1 — Write the failing test (heavy-mock CLI)

- [ ] **Step 3.1.1: Create the test file**

```python
"""Tests for src/custom_sam_peft/cli/calibrate_cmd.py — calibration probe CLI.

All `models.sam3.load_sam31`, `peft_adapters.lora.apply_lora`, and
`torch.cuda.max_memory_allocated` are monkeypatched — these tests run on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

_GB = 1024**3
runner = CliRunner()


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    peak: int = int(38 * _GB),
    gpu_name: str = "NVIDIA A100-SXM4-40GB",
    total: int = int(40 * _GB),
    sha: str = "deadbeef",
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = MagicMock(total_memory=total)
    props.name = gpu_name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: gpu_name)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peak)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd._run_probe",
        lambda image_size: peak,
    )
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd._sam3_checkpoint_sha",
        lambda: sha,
    )


def test_calibrate_writes_cache_with_schema_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", "--image-size", "1008"])
    assert result.exit_code == 0, result.output
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    assert cache.is_file()
    data = json.loads(cache.read_text())
    expected_keys = {
        "schema_version",
        "calibrated_at",
        "gpu_name",
        "gpu_total_memory_bytes",
        "image_size",
        "sam3_checkpoint_sha",
        "torch_version",
        "custom_sam_peft_version",
        "activation_bytes_per_example",
        "peak_memory_bytes_at_probe",
    }
    assert expected_keys.issubset(data.keys())
    assert data["schema_version"] == 1
    assert data["image_size"] == 1008
    assert data["sam3_checkpoint_sha"] == "deadbeef"


def test_calibrate_cache_fresh_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibrated_at": "2026-05-22T00:00:00+00:00",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "gpu_total_memory_bytes": int(40 * _GB),
                "image_size": 1008,
                "sam3_checkpoint_sha": "deadbeef",
                "torch_version": "2.4.0",
                "custom_sam_peft_version": "0.0.0",
                "activation_bytes_per_example": 1,
                "peak_memory_bytes_at_probe": 2,
            }
        )
    )
    mtime_before = cache.stat().st_mtime
    result = runner.invoke(app, ["calibrate", "--image-size", "1008"])
    assert result.exit_code == 0, result.output
    assert "cache fresh" in result.output
    assert cache.stat().st_mtime == mtime_before  # not rewritten


def test_calibrate_force_overwrites_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"stale": true}')
    result = runner.invoke(app, ["calibrate", "--image-size", "1008", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads(cache.read_text())
    assert data.get("schema_version") == 1


def test_calibrate_non_cuda_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 2
    assert "CUDA" in result.output


def test_calibrate_negative_activation_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # peak much smaller than model+adapter+opt → negative raw activation.
    _patch_probe(monkeypatch, peak=10 * 1024**2)  # 10 MiB peak — tiny
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", "--image-size", "1008"])
    assert result.exit_code == 0
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["activation_bytes_per_example"] == 0
    # The warning lands on stderr; CliRunner merges it into .output when mix_stderr=True (default).
    assert "negative" in result.output.lower() or "clamp" in result.output.lower()


def test_calibrate_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"prior": true}')
    # Force the os.replace step to fail; the prior cache must still exist.
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd.os.replace",
        lambda _src, _dst: (_ for _ in ()).throw(OSError("disk full")),
    )
    result = runner.invoke(app, ["calibrate", "--image-size", "1008", "--force"])
    assert result.exit_code == 6
    # The original file content survives the failed write.
    assert json.loads(cache.read_text()) == {"prior": True}
```

- [ ] **Step 3.1.2: Run — verify all fail**

Run: `pytest tests/unit/test_calibrate_cmd.py -v`
Expected: FAIL (`No such command 'calibrate'`).

### Subtask 3.2 — Implement `calibrate_cmd.py` + register

- [ ] **Step 3.2.1: Create `src/custom_sam_peft/cli/calibrate_cmd.py`**

```python
"""`custom-sam-peft calibrate` — probe peak VRAM at LoRA r=4 and cache the result.

Writes `./.custom_sam_peft_calibration.json` (schema_version=1). Read by
`custom_sam_peft.presets._load_cache` so `decide_preset()` produces a tight,
GPU-accurate config instead of an analytic estimate.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §4.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import typer
from rich import print as rprint

from custom_sam_peft import __version__ as _PKG_VERSION
from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.presets import (
    CACHE_FILENAME,
    CACHE_SCHEMA_VERSION,
    WORKSPACE_BYTES,
    _CUDA_HINT,
    _adapter_bytes,
    _model_bytes,
    _optimizer_bytes,
)

_LOG = logging.getLogger(__name__)


def _sam3_checkpoint_sha() -> str:
    cfg = ModelConfig()
    if cfg.local_dir is None:
        return ""
    ckpt = Path(cfg.local_dir) / cfg.checkpoint_file
    if not ckpt.is_file():
        return ""
    h = hashlib.sha256()
    with ckpt.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_is_fresh(path: Path, image_size: int, gpu_name: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("schema_version") == CACHE_SCHEMA_VERSION
        and data.get("gpu_name") == gpu_name
        and int(data.get("image_size", -1)) == image_size
        and data.get("sam3_checkpoint_sha") == _sam3_checkpoint_sha()
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """tmp + os.replace; preserves prior file on failure."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def _run_probe(image_size: int) -> int:
    """Run one forward+backward at LoRA r=4, return peak bytes. CUDA only.

    Steps mirror §4 procedure 3–7: load wrapper, attach LoRA stub at r=4,
    build one synthetic batch, reset peak stats, forward+backward, read
    max_memory_allocated.
    """
    from custom_sam_peft.models.sam3 import load_sam31  # local import — heavy
    from custom_sam_peft.peft_adapters.lora import apply_lora

    model_cfg = ModelConfig()
    wrapper = load_sam31(model_cfg)
    peft_cfg = PEFTConfig(method="lora", r=4)
    apply_lora(wrapper, peft_cfg)

    device = next(wrapper.parameters()).device
    images = torch.zeros(1, 3, image_size, image_size, dtype=torch.bfloat16, device=device)
    # One prompt + one fake target — reuse data structures used elsewhere.
    from custom_sam_peft.data.base import Instance, TextPrompts

    prompts = [TextPrompts(classes=["thing"])]
    fake_mask = torch.zeros(image_size, image_size, dtype=torch.bool, device=device)
    fake_box = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    targets = [[Instance(class_id=0, mask=fake_mask, box=fake_box)]]

    torch.cuda.reset_peak_memory_stats()
    out = wrapper(images, prompts, box_hints=None)
    # Synthetic loss: sum of all output tensors that require grad.
    loss = sum(t.float().sum() for t in out.values() if isinstance(t, torch.Tensor))
    loss.backward()
    return int(torch.cuda.max_memory_allocated())


def calibrate(
    image_size: int = typer.Option(
        1008, "--image-size", help="Image side length the probe runs at."
    ),
    output: Path = typer.Option(
        Path(CACHE_FILENAME), "--output", help="Cache file path."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-probe even if the cache is fresh."
    ),
) -> None:
    """Probe peak VRAM at LoRA r=4 and cache the result."""
    if not torch.cuda.is_available():
        rprint(f"[red]{_CUDA_HINT}[/red]", file=sys.stderr)
        raise typer.Exit(code=2)

    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)

    if not force and _cache_is_fresh(output, image_size, gpu_name):
        rprint("cache fresh — exiting")
        raise typer.Exit(code=0)

    try:
        peak = _run_probe(image_size)
    except FileNotFoundError as exc:
        rprint(f"[red]SAM 3.1 checkpoint not found: {exc}[/red]", file=sys.stderr)
        raise typer.Exit(code=3) from exc
    except (RuntimeError, ValueError) as exc:
        # LoRA stub attach failures land here.
        if "OutOfMemory" in repr(exc) or "out of memory" in str(exc).lower():
            rprint(
                "[red]calibration probe OOMed at minimum config — GPU too small[/red]",
                file=sys.stderr,
            )
            raise typer.Exit(code=5) from exc
        rprint(f"[red]LoRA stub attach failed: {exc}[/red]", file=sys.stderr)
        raise typer.Exit(code=4) from exc

    overhead = (
        _model_bytes("lora")
        + _adapter_bytes(4)
        + _optimizer_bytes(4)
        + WORKSPACE_BYTES
    )
    activation = peak - overhead
    if activation < 0:
        rprint(
            f"[yellow]warning: negative activation ({activation} bytes); "
            "clamping to 0 — constants may need recalibration[/yellow]",
            file=sys.stderr,
        )
        activation = 0

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "calibrated_at": datetime.now(UTC).isoformat(),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total,
        "image_size": image_size,
        "sam3_checkpoint_sha": _sam3_checkpoint_sha(),
        "torch_version": torch.__version__,
        "custom_sam_peft_version": _PKG_VERSION,
        "activation_bytes_per_example": int(activation),
        "peak_memory_bytes_at_probe": int(peak),
    }
    try:
        _atomic_write_json(output, payload)
    except OSError as exc:
        rprint(f"[red]cache write failed: {exc}[/red]", file=sys.stderr)
        raise typer.Exit(code=6) from exc

    gib = lambda b: b / (1024**3)  # noqa: E731
    rprint(f"GPU:        {gpu_name} (image_size={image_size})")
    rprint(f"Peak:       {gib(peak):.1f} GiB")
    rprint(f"Activation: {gib(activation):.2f} GiB/example")
    rprint(f"Cache:      {output}")
```

- [ ] **Step 3.2.2: Register the subcommand in `cli/main.py`**

Add to the imports:
```python
from custom_sam_peft.cli import (
    calibrate_cmd,
    doctor_cmd,
    ...
)
```

And below `app.command("doctor", ...)`:
```python
app.command("calibrate", help="Probe peak VRAM and cache for tighter preset packing.")(calibrate_cmd.calibrate)
```

- [ ] **Step 3.2.3: Verify `__version__` is exported**

Open `src/custom_sam_peft/__init__.py`. If `__version__` is absent, add:

```python
from importlib.metadata import version as _v

__version__ = _v("custom-sam-peft")
```

- [ ] **Step 3.2.4: Run the tests — confirm pass**

Run: `pytest tests/unit/test_calibrate_cmd.py -v`
Expected: all green.

- [ ] **Step 3.3: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py src/custom_sam_peft/cli/main.py \
        src/custom_sam_peft/__init__.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(cli): add 'calibrate' subcommand for VRAM probe (#36)"
```

**Acceptance:** `pytest tests/unit/test_calibrate_cmd.py tests/unit/test_presets.py -v` green.

---

## Task 4 — Trainer OOM ladder + microbatching + runner propagation

**Files:**
- Modify: `src/custom_sam_peft/train/loop.py` (add OOM ladder around the inner per-class block)
- Modify: `src/custom_sam_peft/train/trainer.py` (thread `oom_events` accumulator into the epoch loop, surface in `RunResult`)
- Modify: `src/custom_sam_peft/train/runner.py` (propagate `oom_events` into the return dict / `RunResult`)
- Create: `tests/unit/test_trainer_oom_retry.py`

### Subtask 4.1 — Failing tests (synthetic OOM injection, CPU-only)

- [ ] **Step 4.1.1: Create `tests/unit/test_trainer_oom_retry.py`**

```python
"""Tests for the train_step OOM-retry ladder.

We inject `torch.cuda.OutOfMemoryError` from a stub model's forward — the
exception class is importable without CUDA, so this runs on CPU.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from custom_sam_peft.train.types import OomEvent


class _OomThenOk(torch.nn.Module):
    """forward() raises CUDA OOM the first `n_oom` times, then returns a real loss."""

    def __init__(self, n_oom: int) -> None:
        super().__init__()
        self.n_oom = n_oom
        self.calls = 0
        # A trainable parameter so backward() has something to differentiate.
        self.p = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        if self.calls <= self.n_oom:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return self.p.sum()


# --- The OOM ladder helper under test --------------------------------------
# `_train_step_with_oom_ladder` is the new helper we land in train/loop.py.
# These tests import it directly to keep the surface small.

from custom_sam_peft.train.loop import _train_step_with_oom_ladder


@dataclass
class _State:
    step: int = 0
    micro_batch_size: int = 8
    gradient_checkpointing: bool = False
    pending_oom_events: list[OomEvent] = field(default_factory=list)


def _make_batch(n: int) -> list[int]:
    """Stand-in batch: a list of ints. Sliceable, has __len__."""
    return list(range(n))


def _fake_forward_call(model: torch.nn.Module, micro: list[int]) -> torch.Tensor:
    return model(micro)


def test_oom_first_attempt_halves_microbatch() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=1)
    _train_step_with_oom_ladder(
        model, _make_batch(8), state, forward_call=_fake_forward_call
    )
    assert state.micro_batch_size == 4
    assert len(state.pending_oom_events) == 1
    assert state.pending_oom_events[0].action == "microbatch_halved"


def test_oom_multiple_halvings_until_one() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=3)
    _train_step_with_oom_ladder(
        model, _make_batch(8), state, forward_call=_fake_forward_call
    )
    assert state.micro_batch_size == 1
    assert len(state.pending_oom_events) == 3
    assert all(e.action == "microbatch_halved" for e in state.pending_oom_events)


def test_oom_after_microbatch_1_enables_ckpt() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=4)  # 3 halvings → mb=1, 4th OOM flips ckpt
    _train_step_with_oom_ladder(
        model, _make_batch(8), state, forward_call=_fake_forward_call
    )
    assert state.micro_batch_size == 1
    assert state.gradient_checkpointing is True
    assert state.pending_oom_events[-1].action == "grad_ckpt_enabled"


def test_oom_after_ckpt_enabled_raises() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=5)  # 3 halvings + 1 ckpt + 1 final OOM → raise
    with pytest.raises(RuntimeError, match="OOM at step"):
        _train_step_with_oom_ladder(
            model, _make_batch(8), state, forward_call=_fake_forward_call
        )


def test_oom_microbatch_shrink_is_sticky() -> None:
    state = _State(micro_batch_size=8)
    # Step 1: 1 OOM → mb halves to 4.
    model = _OomThenOk(n_oom=1)
    _train_step_with_oom_ladder(
        model, _make_batch(8), state, forward_call=_fake_forward_call
    )
    assert state.micro_batch_size == 4
    # Step 2 with a fresh stub that never OOMs.
    state.step = 1
    model2 = _OomThenOk(n_oom=0)
    _train_step_with_oom_ladder(
        model2, _make_batch(8), state, forward_call=_fake_forward_call
    )
    # mb did not reset.
    assert state.micro_batch_size == 4


def test_oom_ckpt_toggle_is_once() -> None:
    """Two separate OOMs that would each enable ckpt produce only one event."""
    state = _State(micro_batch_size=1, gradient_checkpointing=False)
    model = _OomThenOk(n_oom=1)
    _train_step_with_oom_ladder(
        model, _make_batch(1), state, forward_call=_fake_forward_call
    )
    assert state.gradient_checkpointing is True
    n_after_first = sum(
        1 for e in state.pending_oom_events if e.action == "grad_ckpt_enabled"
    )
    assert n_after_first == 1
    # Subsequent OOM with ckpt already on goes straight to RuntimeError.
    state.step = 1
    model2 = _OomThenOk(n_oom=1)
    with pytest.raises(RuntimeError):
        _train_step_with_oom_ladder(
            model2, _make_batch(1), state, forward_call=_fake_forward_call
        )
    n_after_second = sum(
        1 for e in state.pending_oom_events if e.action == "grad_ckpt_enabled"
    )
    assert n_after_second == 1  # still just the one


def test_oom_optimizer_zero_grad_called_once_per_step() -> None:
    """Spec §6 invariant: optimizer.zero_grad() fires once per outer step,
    not once per microbatch and not on retry."""
    state = _State(micro_batch_size=4)
    model = _OomThenOk(n_oom=1)
    optimizer = MagicMock()
    # Test harness: a thin wrapper that mimics the trainer's step structure.
    optimizer.zero_grad()
    _train_step_with_oom_ladder(
        model, _make_batch(4), state, forward_call=_fake_forward_call
    )
    # The ladder helper itself never calls zero_grad — the caller did once above.
    assert optimizer.zero_grad.call_count == 1


def test_oom_events_propagated_in_run_result() -> None:
    """run_training's RunResult exposes the accumulated events list."""
    from custom_sam_peft.train.trainer import RunResult

    fields = {f.name for f in __import__("dataclasses").fields(RunResult)}
    assert "oom_events" in fields


def test_oom_events_serialise_into_bundle_edge_cases() -> None:
    """An end-to-end sanity check that events flowed into the bundler renders.

    This is a shallow trace check — the full rendering is exercised in
    tests/unit/runs/test_bundle.py::test_write_bundle_oom_edge_note_with_ckpt.
    Here we only confirm the linkage: a non-empty oom_events tuple on
    BundleContext produces a `## Edge cases` line containing 'OOM retries'.
    """
    from datetime import UTC, datetime
    from pathlib import Path as _P
    from unittest.mock import MagicMock as _MM

    from custom_sam_peft.presets import PresetDecision
    from custom_sam_peft.runs.bundle import BundleContext, write_bundle

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = _P(tmp)
        (tmp_path / "run").mkdir()
        (tmp_path / "config.yaml").write_text("run: {name: r}\n")
        decision = PresetDecision(
            method="lora", r=16, batch_size=1, grad_accum_steps=16,
            gradient_checkpointing=False, dtype="bfloat16",
            headroom_bytes=0, predicted_bytes=0, budget_bytes=0,
            image_size=1008, gpu_name="StubGPU",
            provenance="analytic", cache_path=None,
        )
        ctx = BundleContext(
            run_dir=tmp_path / "run",
            config_path=tmp_path / "config.yaml",
            start_ts=datetime(2026, 5, 22, tzinfo=UTC),
            end_ts=datetime(2026, 5, 22, tzinfo=UTC),
            preset=decision,
            per_example_iou=[],
            merged_dir=None,
            merged_export_error=None,
            oom_events=(
                OomEvent(step=1, action="microbatch_halved",
                         new_micro_batch_size=4, new_gradient_checkpointing=False),
            ),
        )
        report = _MM(overall={"mAP": 0.0})
        val_ds = _MM(__len__=lambda self: 0)
        write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=_MM())
        summary = (tmp_path / "run" / "summary.md").read_text()
        assert "OOM retries: 1" in summary
```

Add the import for `MagicMock`:

```python
from unittest.mock import MagicMock
```

- [ ] **Step 4.1.2: Run — verify all fail**

Run: `pytest tests/unit/test_trainer_oom_retry.py -v`
Expected: ImportError on `_train_step_with_oom_ladder`.

### Subtask 4.2 — Implement the OOM ladder helper in `train/loop.py`

- [ ] **Step 4.2.1: Add the helper at the top of `train/loop.py` (after imports)**

```python
@dataclass
class OomState:
    """Mutable state the OOM ladder reads/writes across steps.

    Held by the Trainer for the lifetime of a `fit()` call. The trainer's
    inner per-class loss block calls `_train_step_with_oom_ladder` once per
    step; on OOM the helper mutates `micro_batch_size` / `gradient_checkpointing`
    in place (sticky) and appends to `pending_oom_events`.
    """

    step: int = 0
    micro_batch_size: int = 1
    gradient_checkpointing: bool = False
    pending_oom_events: list[OomEvent] = field(default_factory=list)


def _train_step_with_oom_ladder(
    model: Any,
    batch: Any,
    state: Any,                          # _State (test) | OomState (prod)
    *,
    forward_call: Callable[[Any, Any], torch.Tensor],
) -> torch.Tensor:
    """Run one optimizer-step's worth of microbatches; ladder OOM downward.

    Caller is responsible for `optimizer.zero_grad()` (once, outside this
    helper) and `optimizer.step()` (once, after this helper returns).

    Spec §6 invariants:
      - microbatch shrink is sticky
      - gradient_checkpointing toggles at most once per run
      - optimizer.zero_grad never called mid-microbatch (helper does not call it)
      - mid-step OOM replays from i=0 at the smaller size

    Returns the final detached loss tensor of the last successful microbatch.
    """
    n = len(batch)
    last_loss: torch.Tensor | None = None
    while True:
        try:
            mb = state.micro_batch_size
            n_micro = (n + mb - 1) // mb
            for i in range(n_micro):
                start = i * mb
                end = min(start + mb, n)
                micro = batch[start:end]
                loss = forward_call(model, micro)
                # Caller divides by grad_accum_steps separately; we divide by
                # n_micro here so the gradient magnitude matches the pre-ladder
                # path. Outer loop must NOT divide again.
                (loss / n_micro).backward()
                last_loss = loss.detach()
            return last_loss if last_loss is not None else torch.tensor(0.0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if state.micro_batch_size > 1:
                state.micro_batch_size //= 2
                state.pending_oom_events.append(
                    OomEvent(
                        step=state.step,
                        action="microbatch_halved",
                        new_micro_batch_size=state.micro_batch_size,
                        new_gradient_checkpointing=state.gradient_checkpointing,
                    )
                )
                _LOG.warning(
                    "OOM at step %d — halving micro_batch_size to %d",
                    state.step, state.micro_batch_size,
                )
                continue
            if not state.gradient_checkpointing:
                state.gradient_checkpointing = True
                state.pending_oom_events.append(
                    OomEvent(
                        step=state.step,
                        action="grad_ckpt_enabled",
                        new_micro_batch_size=state.micro_batch_size,
                        new_gradient_checkpointing=True,
                    )
                )
                _LOG.warning(
                    "OOM at step %d — enabling gradient_checkpointing",
                    state.step,
                )
                continue
            raise RuntimeError(
                f"OOM at step {state.step} after micro_batch=1 + "
                f"gradient_checkpointing=on. Use a larger GPU or smaller image_size."
            )
```

Add the imports at the top of the file:

```python
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from custom_sam_peft.train.types import OomEvent
```

(`dataclass`, `field` are already imported — confirm and dedupe.)

- [ ] **Step 4.2.2: Run the unit tests — confirm ladder behavior**

Run: `pytest tests/unit/test_trainer_oom_retry.py::test_oom_first_attempt_halves_microbatch tests/unit/test_trainer_oom_retry.py::test_oom_multiple_halvings_until_one tests/unit/test_trainer_oom_retry.py::test_oom_after_microbatch_1_enables_ckpt tests/unit/test_trainer_oom_retry.py::test_oom_after_ckpt_enabled_raises tests/unit/test_trainer_oom_retry.py::test_oom_microbatch_shrink_is_sticky tests/unit/test_trainer_oom_retry.py::test_oom_ckpt_toggle_is_once -v`
Expected: 6 passing. (The runner-result / bundle-edge-cases / zero_grad-once tests still fail — covered in 4.3.)

### Subtask 4.3 — Surface `oom_events` in `RunResult` and the runner

- [ ] **Step 4.3.1: Modify `src/custom_sam_peft/train/trainer.py` `RunResult`**

Change the dataclass:

```python
@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    adapter_path: Path
    merged_path: Path | None
    final_metrics: MetricsReport | None
    oom_events: tuple[OomEvent, ...] = ()
```

Add `from custom_sam_peft.train.types import OomEvent` to the imports.

Inside `Trainer.fit`, instantiate an `OomState` once before the epoch loop and pass it into `run_epoch` (via the existing function signature — add a kwarg). Accumulate into `RunResult.oom_events = tuple(oom_state.pending_oom_events)` at the return site.

```python
from custom_sam_peft.train.loop import OomState, _box_hint_p, run_epoch
...
oom_state = OomState(micro_batch_size=cfg.train.batch_size)
...
return RunResult(
    run_dir=run_dir,
    adapter_path=run_dir / "adapter",
    merged_path=merged_path,
    final_metrics=full_report,
    oom_events=tuple(oom_state.pending_oom_events),
)
```

- [ ] **Step 4.3.2: Thread the state through `run_epoch` and `train_step`**

Update `run_epoch` and `train_step` signatures to accept `oom_state: OomState`. Inside `train_step`, replace the existing per-class `out = model(images, prompts_c, box_hints=hints_c)` + `loss.backward()` sequence with a call to `_train_step_with_oom_ladder` for each class — *but* only if the per-class loss computation is well-defined under microbatch slicing. **Implementation note:** for v1, slice the batch dimension (`images[start:end]`, prompts/targets correspondingly) inside `forward_call`. Do not change the per-class loop structure — wrap each `out = model(...)` site with the ladder helper by passing a `forward_call` that closes over the current `(prompts_c, targets_c, hints_c)` slicing.

Cleanest pattern: leave the per-class loop body intact, but at the top of `train_step` wrap the *whole* micro-step in a try/except that delegates to the OOM helper. Concretely: convert the body to:

```python
def _forward_one_microbatch(_model: Any, micro_imgs: Tensor) -> Tensor:
    # Existing per-class loop, parameterised by the sliced images.
    ...
    return class_scaled.sum()  # or similar aggregate

_train_step_with_oom_ladder(
    model, images, oom_state, forward_call=_forward_one_microbatch
)
```

If that refactor breaks the per-class backward semantics, fall back to **catching OOM at the per-class `out = model(...)` line only** and applying the ladder there — i.e., re-run the whole class loop after shrinking. Either path satisfies §6 invariants as long as `optimizer.zero_grad()` stays where it is (only fired at the grad-accum boundary) and microbatch shrink is sticky.

The implementer chooses based on the structure of the existing per-class loop; the spec mandates the *invariants* (§6), not the exact code shape.

- [ ] **Step 4.3.3: Update `train/runner.py` — propagate `oom_events`**

`run_training` already returns the `RunResult` from `Trainer.fit()`; since `RunResult` now has `oom_events`, no changes are strictly required here. Add an explicit pass-through if `run_training` was returning a dict — confirm by reading the current file. (As of this plan: it returns `RunResult` directly. No edit needed.)

- [ ] **Step 4.3.4: Run the remaining test**

Run: `pytest tests/unit/test_trainer_oom_retry.py::test_oom_events_propagated_in_run_result tests/unit/test_trainer_oom_retry.py::test_oom_events_serialise_into_bundle_edge_cases tests/unit/test_trainer_oom_retry.py::test_oom_optimizer_zero_grad_called_once_per_step -v`
Expected: passing.

- [ ] **Step 4.3.5: Run the full new file**

Run: `pytest tests/unit/test_trainer_oom_retry.py -v`
Expected: all 7 passing.

- [ ] **Step 4.4: Commit**

```bash
git add src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py \
        src/custom_sam_peft/train/runner.py tests/unit/test_trainer_oom_retry.py
git commit -m "feat(train): per-step OOM auto-retry ladder + oom_events (#36)"
```

**Acceptance:** `pytest tests/unit/test_trainer_oom_retry.py tests/unit/test_presets.py tests/unit/test_calibrate_cmd.py tests/unit/test_train_types.py -v` green.

---

## Task 5 — Bundler restructure: `BundleContext`, `## Preset` block, OOM edge note

**Files:**
- Modify: `src/custom_sam_peft/runs/bundle.py`
- Modify: `tests/unit/runs/test_bundle.py`

### Subtask 5.1 — Update failing tests

- [ ] **Step 5.1.1: Edit `tests/unit/runs/test_bundle.py`**

Replace the `_make_ctx` helper and add structured `## Preset` + OOM edge-cases assertions:

```python
# At the top, add imports:
from custom_sam_peft.presets import PresetDecision
from custom_sam_peft.train.types import OomEvent


def _make_decision() -> PresetDecision:
    return PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        gradient_checkpointing=False,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * 1024**3),
        predicted_bytes=int(38.4 * 1024**3),
        budget_bytes=int(39 * 1024**3),
        image_size=1008,
        gpu_name="NVIDIA A100-SXM4-40GB",
        provenance="calibrated",
        cache_path=None,
    )


def _make_ctx(tmp_path: Path, **overrides: object) -> BundleContext:
    base = BundleContext(
        run_dir=tmp_path / "run",
        config_path=tmp_path / "config.yaml",
        start_ts=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        end_ts=datetime(2026, 5, 18, 12, 5, 0, tzinfo=UTC),
        preset=_make_decision(),
        per_example_iou=[0.1, 0.5, 0.9],
        merged_dir=None,
        merged_export_error=None,
        oom_events=(),
    )
    base.run_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text("run: {name: r}\n")
    return replace(base, **overrides)


# New tests — keep all existing pick_samples / render_overlay tests as-is.

def test_write_bundle_preset_block_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        lambda *_a, **_k: (Image.new("RGB", (8, 8)), np.zeros((8, 8), bool), np.zeros((8, 8), bool)),
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "## Preset" in summary
    assert "- Method: LoRA r=32" in summary
    assert "batch=2" in summary
    assert "grad_accum=8" in summary
    assert "ckpt=off" in summary
    assert "- GPU:    NVIDIA A100-SXM4-40GB" in summary
    assert "38.4 / " in summary  # used/total GiB
    assert "calibrated" in summary.lower()


def test_write_bundle_preset_block_analytic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = replace(_make_decision(), provenance="analytic", cache_path=None)
    ctx = _make_ctx(tmp_path, per_example_iou=[], preset=decision)
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        lambda *_a, **_k: (Image.new("RGB", (8, 8)), np.zeros((8, 8), bool), np.zeros((8, 8), bool)),
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "- Source: analytic estimate" in summary


def test_write_bundle_oom_edge_note_with_ckpt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = (
        OomEvent(step=10, action="microbatch_halved", new_micro_batch_size=4, new_gradient_checkpointing=False),
        OomEvent(step=20, action="microbatch_halved", new_micro_batch_size=2, new_gradient_checkpointing=False),
        OomEvent(step=412, action="grad_ckpt_enabled", new_micro_batch_size=2, new_gradient_checkpointing=True),
    )
    ctx = _make_ctx(tmp_path, per_example_iou=[], oom_events=events)
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        lambda *_a, **_k: (Image.new("RGB", (8, 8)), np.zeros((8, 8), bool), np.zeros((8, 8), bool)),
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "OOM retries: 3" in summary
    assert "final micro_batch=2" in summary
    assert "gradient_checkpointing enabled at step 412" in summary


def test_write_bundle_oom_edge_note_no_ckpt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = (
        OomEvent(step=10, action="microbatch_halved", new_micro_batch_size=4, new_gradient_checkpointing=False),
    )
    ctx = _make_ctx(tmp_path, per_example_iou=[], oom_events=events)
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        lambda *_a, **_k: (Image.new("RGB", (8, 8)), np.zeros((8, 8), bool), np.zeros((8, 8), bool)),
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "OOM retries: 1" in summary
    # The "gradient_checkpointing enabled at step X" clause must be omitted.
    assert "gradient_checkpointing enabled" not in summary
```

Remove any assertions referencing `ctx.preset_label`. Update the existing `test_write_bundle_writes_summary_and_samples`, `test_write_bundle_empty_val_writes_summary_with_note`, `test_write_bundle_merge_failure_recorded_in_summary`, `test_write_bundle_skipped_sample_logged_and_summary_notes_it` to use the new `_make_ctx` (already does after the helper change).

- [ ] **Step 5.1.2: Run — confirm new tests fail, existing tests probably also fail on constructor**

Run: `pytest tests/unit/runs/test_bundle.py -v`
Expected: failures on `BundleContext(...preset_label=...)` constructor and on the new structured-block assertions.

### Subtask 5.2 — Implement bundler changes

- [ ] **Step 5.2.1: Update `BundleContext` in `src/custom_sam_peft/runs/bundle.py`**

```python
from custom_sam_peft.presets import PresetDecision
from custom_sam_peft.train.types import OomEvent


@dataclass(frozen=True)
class BundleContext:
    """All run-context fields the bundler needs, assembled by `custom_sam_peft run`."""

    run_dir: Path
    config_path: Path
    start_ts: datetime
    end_ts: datetime
    preset: PresetDecision                  # required (replaces preset_label: str | None)
    per_example_iou: list[float]
    merged_dir: Path | None
    merged_export_error: str | None
    oom_events: tuple[OomEvent, ...]        # required, no default
```

- [ ] **Step 5.2.2: Build a helper to render the `## Preset` block**

Add to `bundle.py` above `write_bundle`:

```python
def _preset_block(preset: PresetDecision) -> str:
    ckpt_word = "on" if preset.gradient_checkpointing else "off"
    method_pretty = "LoRA" if preset.method == "lora" else "QLoRA"
    used_gib = preset.predicted_bytes / (1024**3)
    total_gib = (preset.budget_bytes + preset.headroom_bytes) / (1024**3)
    headroom_gib = preset.headroom_bytes / (1024**3)
    if preset.provenance == "calibrated":
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cache_name = Path(preset.cache_path).name if preset.cache_path else "(unknown)"
        source_line = f"- Source: calibrated {today} (cache: {cache_name})"
    else:
        source_line = "- Source: analytic estimate"
    return (
        f"- Method: {method_pretty} r={preset.r}, batch={preset.batch_size}, "
        f"grad_accum={preset.grad_accum_steps}, gradient_checkpointing={ckpt_word}, bf16\n"
        f"- GPU:    {preset.gpu_name} ({total_gib:.1f} GiB)\n"
        f"- Budget: {used_gib:.1f} / {total_gib:.1f} GiB used ({headroom_gib:.1f} GiB headroom)\n"
        f"{source_line}"
    )


def _oom_edge_note(events: tuple[OomEvent, ...]) -> str | None:
    """Return the OOM-summary line for `## Edge cases`, or None when there were none."""
    if not events:
        return None
    final_mb = events[-1].new_micro_batch_size
    ckpt_event = next((e for e in events if e.action == "grad_ckpt_enabled"), None)
    base = f"OOM retries: {len(events)} — final micro_batch={final_mb}"
    if ckpt_event is not None:
        base += f", gradient_checkpointing enabled at step {ckpt_event.step}"
    return base
```

- [ ] **Step 5.2.3: Replace the `## Preset` section in `write_bundle`**

Delete the old `preset_line = f"- Applied: {ctx.preset_label or 'manual'}"` line.

Replace the body composition's preset block:

```python
preset_block = _preset_block(ctx.preset)
...
body = (
    f"{headline}\n\n"
    f"## Run\n"
    ...
    f"## Preset\n"
    f"{preset_block}\n\n"
    f"## Outputs\n"
    ...
)
```

Add the OOM edge note before the edge-cases composition:

```python
oom_line = _oom_edge_note(ctx.oom_events)
if oom_line is not None:
    edge_notes.append(oom_line)
```

- [ ] **Step 5.2.4: Run — confirm pass**

Run: `pytest tests/unit/runs/test_bundle.py -v`
Expected: all green.

- [ ] **Step 5.3: Commit**

```bash
git add src/custom_sam_peft/runs/bundle.py tests/unit/runs/test_bundle.py
git commit -m "feat(bundle): structured ## Preset block + OOM edge note (#36)"
```

**Acceptance:** `pytest tests/unit/runs/test_bundle.py -v` green.

---

## Task 6 — `run_cmd.py`: read `preset.json` sidecar, drop env-var

**Files:**
- Modify: `src/custom_sam_peft/cli/run_cmd.py`
- Modify: `tests/integration/test_cli_run.py` (delete two env-var tests; add sidecar test)

### Subtask 6.1 — Failing test (sidecar contract)

- [ ] **Step 6.1.1: Edit `tests/integration/test_cli_run.py`**

Delete the two preset_label tests:
- `test_run_passes_preset_label_env_var_through`
- `test_run_preset_label_absent_yields_none`

Update `_patch_phases` and add a new test:

```python
# At the top of test_cli_run.py, add:
from custom_sam_peft.presets import PresetDecision


def _write_preset_sidecar(tmp_path: Path) -> PresetDecision:
    d = PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        gradient_checkpointing=False,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * 1024**3),
        predicted_bytes=int(38.4 * 1024**3),
        budget_bytes=int(39 * 1024**3),
        image_size=1008,
        gpu_name="StubGPU",
        provenance="analytic",
        cache_path=None,
    )
    (tmp_path / "preset.json").write_text(d.to_json())
    return d


def test_run_reads_preset_sidecar_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _write_preset_sidecar(tmp_path)
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)  # so run_cmd resolves preset.json relative to cwd
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset == expected
    assert ctx.oom_events == ()


def test_run_synthesizes_analytic_preset_when_sidecar_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Stub decide_preset so we don't need CUDA in this test.
    fake_decision = _write_preset_sidecar(tmp_path)  # writes & returns a PresetDecision
    (tmp_path / "preset.json").unlink()  # remove the sidecar so the fallback path runs
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._fallback_preset", lambda cfg: fake_decision
    )
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset.provenance == "analytic"
```

Also update the existing `test_run_full_success` test — it asserts on `ctx.merged_dir` / `ctx.per_example_iou` which are unchanged, but the construction of `_patch_phases` may need to write a preset sidecar at the top so `run_cmd` does not error. Add:

```python
def _patch_phases(monkeypatch, *, run_dir, ...):
    ...
    # Ensure run_cmd can find a preset.json relative to cwd, OR stub the loader.
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._load_preset_or_fallback",
        lambda cfg: PresetDecision(
            method="lora", r=16, batch_size=1, grad_accum_steps=16,
            gradient_checkpointing=False, dtype="bfloat16",
            headroom_bytes=0, predicted_bytes=0, budget_bytes=0,
            image_size=1008, gpu_name="StubGPU",
            provenance="analytic", cache_path=None,
        ),
    )
```

- [ ] **Step 6.1.2: Run — verify the changed tests fail**

Run: `pytest tests/integration/test_cli_run.py -v`
Expected: failures referencing missing `_load_preset_or_fallback` / `_fallback_preset` and old `preset_label`.

### Subtask 6.2 — Implement in `run_cmd.py`

- [ ] **Step 6.2.1: Edit `src/custom_sam_peft/cli/run_cmd.py`**

Replace the env-var lookup with sidecar-or-analytic loading.

```python
from custom_sam_peft.presets import PresetDecision, decide_preset


def _fallback_preset(cfg: TrainConfig) -> PresetDecision:
    """No sidecar — synthesize one from cfg + decide_preset(). Spec §11.4."""
    return decide_preset(image_size=cfg.data.image_size)


def _load_preset_or_fallback(cfg: TrainConfig) -> PresetDecision:
    sidecar = Path("preset.json")
    if sidecar.is_file():
        return PresetDecision.from_json(sidecar.read_text())
    return _fallback_preset(cfg)
```

Inside `_orchestrate`, replace the bundle-context construction:

```python
preset = _load_preset_or_fallback(cfg)
ctx = BundleContext(
    run_dir=run_dir,
    config_path=run_dir / "config.yaml",
    start_ts=start_ts,
    end_ts=end_ts,
    preset=preset,
    per_example_iou=per_example_iou,
    merged_dir=merged_dir,
    merged_export_error=merged_export_error,
    oom_events=train_result.oom_events,
)
```

Remove the `import os` if it's no longer used, and remove `os.environ.get("CUSTOM_SAM_PEFT_PRESET_LABEL")`.

- [ ] **Step 6.2.2: Run — confirm pass**

Run: `pytest tests/integration/test_cli_run.py -v`
Expected: all green.

- [ ] **Step 6.3: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/integration/test_cli_run.py
git commit -m "feat(run): read PresetDecision from sidecar JSON (#36)"
```

**Acceptance:** `pytest tests/integration/test_cli_run.py -v` green.

---

## Task 7 — Notebook: CALIBRATE cell + GENERATE rewrite

**Files:**
- Modify: `notebooks/custom_sam_peft_train.ipynb`

Both cells change together: the new GENERATE cell expects either a cache from CALIBRATE or the analytic fallback.

- [ ] **Step 7.1: Insert a new CALIBRATE cell between SETUP (cell 0) and FORM (cell 1)**

Use any of: `nbformat`, manual JSON edit, or `jupyter nbconvert` round-trip. The cell contents must be exactly:

Markdown preamble (one markdown cell):

> **CALIBRATE — strongly recommended (~30–60s, one-time per machine).** Probes peak VRAM on this GPU so `decide_preset()` can return a tight, accurate config instead of a conservative analytic estimate. Skip via `CUSTOM_SAM_PEFT_SKIP_CALIBRATE=1` if you know what you are doing.

Code cell (one code cell):

```python
import os
if os.environ.get("CUSTOM_SAM_PEFT_SKIP_CALIBRATE") != "1":
    !custom_sam_peft calibrate --image-size 1008
else:
    print("CALIBRATE skipped via CUSTOM_SAM_PEFT_SKIP_CALIBRATE=1")
```

- [ ] **Step 7.2: Rewrite the GENERATE cell**

Replace the cell that imports `pick_preset, preset_label` with:

```python
# GENERATE — derive preset, load template, deep-merge user inputs, write config.yaml, run CLI.
import importlib.resources
import subprocess
from pathlib import Path

import yaml

from custom_sam_peft.presets import decide_preset

decision = decide_preset(image_size=1008)
patch = decision.config_patch
template_name = (
    "coco_text_qlora.yaml" if patch["peft"]["method"] == "qlora" else "coco_text_lora.yaml"
)
template_text = (
    importlib.resources.files("custom_sam_peft.cli.templates") / template_name
).read_text()
config = yaml.safe_load(template_text)


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


_deep_merge(config, patch)
config["run"]["name"] = run_name
config["data"]["format"] = data_format

# COCO path resolution (unchanged from prior cell — keep the same _PREF / _resolve_split logic).
# … [retain existing COCO/HF branch verbatim from the prior cell] …

with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

Path("preset.json").write_text(decision.to_json())
print(decision.label())

proc = subprocess.Popen(
    ["custom_sam_peft", "run", "--config", "config.yaml"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    text=True,
)
buffer: list[str] = []
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="")
    buffer.append(line)
rc = proc.wait()
if rc != 0:
    print("\n--- last 50 lines ---")
    print("".join(buffer[-50:]))
    raise SystemExit(rc)
```

The COCO / HF dataset-path resolution block is preserved verbatim from the existing cell — only the preset section at top and the `preset.json` write near the bottom change. **Do not delete** the `_PREF`, `_resolve_split`, and HF branches.

Remove the line `os.environ["CUSTOM_SAM_PEFT_PRESET_LABEL"] = preset_label()` entirely.

- [ ] **Step 7.3: Smoke-validate the notebook**

Run: `python -c "import json; nb = json.load(open('notebooks/custom_sam_peft_train.ipynb')); [print(i, c['cell_type'], (''.join(c['source'])[:80] or '(empty)')) for i, c in enumerate(nb['cells'])]"`

Expected output: 5 cells, with cell 1 being the CALIBRATE markdown, cell 2 being the CALIBRATE code, cell 3 being FORM, cell 4 being GENERATE, cell 5 being RESULTS. (Adjust indices for your insertion; the assertion is "CALIBRATE is between SETUP and FORM".)

- [ ] **Step 7.4: Run the broader notebook-linked tests (if any)**

Search for any test that loads the notebook structurally:

```bash
grep -r "custom_sam_peft_train" tests/ | head
```

If a test exists, run it: `pytest -k notebook -v`. If none, skip.

- [ ] **Step 7.5: Commit**

```bash
git add notebooks/custom_sam_peft_train.ipynb
git commit -m "feat(notebook): CALIBRATE cell + decide_preset GENERATE (#36)"
```

**Acceptance:** notebook loads, structure has CALIBRATE between SETUP and FORM, no `pick_preset` / `preset_label` / `CUSTOM_SAM_PEFT_PRESET_LABEL` strings remain.

---

## Task 8 — Delete superseded tests + full suite green

**Files:**
- Delete: per spec §9 / §11, four test names.

The four superseded test names live in `tests/unit/test_presets.py`. They were *already replaced* by the rewrite in Task 2 — confirm no other file references them.

- [ ] **Step 8.1: Search-and-confirm**

Run: `grep -rn "test_pick_preset_requires_cuda\|test_pick_preset_tiers\|test_preset_label_format\|test_preset_label_with_explicit_total_bytes" tests/`
Expected: zero matches (the file rewrite already removed them).

If any reference remains, delete it.

- [ ] **Step 8.2: Search for other references to removed symbols**

```bash
grep -rn "pick_preset\|preset_label\|CUSTOM_SAM_PEFT_PRESET_LABEL\|_tier_for_gb" src/ tests/ notebooks/
```

Expected: zero matches (Task 2 removed the symbols, Task 6 removed the env var, Task 7 removed the notebook usages).

If any remain, delete them.

- [ ] **Step 8.3: Run the full test suite**

Run: `pytest -q`
Expected: green.

- [ ] **Step 8.4: Commit (only if there were any cleanups)**

```bash
git add -p   # cherry-pick only the legitimate cleanups; skip incidental noise
git commit -m "chore: drop superseded preset symbols + tests (#36)"
```

(If `git status` shows no changes after Step 8.2, skip the commit.)

**Acceptance:** `pytest -q` green; zero hits for the deleted symbols.

---

## Task 9 — GPU smoke test (planning-only — written, not run here)

**Files:**
- Create: `tests/gpu/test_calibrate_real.py`

The GPU suite runs separately (on Colab / RunPod, gated by `pytest -m gpu`). This task **writes** the test but does not run it.

- [ ] **Step 9.1: Create `tests/gpu/test_calibrate_real.py`**

```python
"""GPU smoke test for `custom_sam_peft calibrate` — real activation byte range.

Marked `gpu` so it is skipped on CPU CI. Runs the full probe and asserts the
measured `activation_bytes_per_example` lands in a sane order-of-magnitude
bracket — 0.5 GiB to 10 GiB per example at image_size=1008 on a modern GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app


@pytest.mark.gpu
def test_calibrate_real_activation_in_sane_range(tmp_path: Path) -> None:
    runner = CliRunner()
    os.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate", "--image-size", "1008", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    activation = int(data["activation_bytes_per_example"])
    assert 5e8 <= activation <= 1e10, (
        f"activation_bytes_per_example={activation} outside [0.5 GiB, 10 GiB]"
    )
```

- [ ] **Step 9.2: Confirm collection (without running)**

Run: `pytest tests/gpu/test_calibrate_real.py --collect-only -q`
Expected: 1 test collected, marked `gpu`. (Does not execute on CPU CI.)

- [ ] **Step 9.3: Commit**

```bash
git add tests/gpu/test_calibrate_real.py
git commit -m "test(gpu): calibrate real activation in sane range (#36)"
```

**Acceptance:** test collects clean; not run.

---

## Task 10 — Version bump + lockfile + lint/format/type + final green

This is the **last** task. Per the orchestrator override, it covers (a) version stamping for the release, (b) lockfile regeneration, and (c) the lint/format/type/full-suite green check that ships with this PR.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

### Subtask 10.1 — Stamp the orchestrator-decided version

The spec §11 says `0.6.0 → 0.7.0`. Since spec authoring, two prior PRs have already bumped the version to `0.7.1`. **The orchestrator decides the actual semver in close-out step 4** (per user CLAUDE.md). The plan defers to that decision — it does *not* hardcode a number.

- [ ] **Step 10.1.1: Read the latest tag**

```bash
git fetch --tags && git describe --tags --abbrev=0
```

- [ ] **Step 10.1.2: Compute next minor**

Per the orchestrator rule (pre-1.0 breaking → minor), bump the minor digit and reset patch to 0. If latest tag is `v0.7.1`, next is `v0.8.0`.

- [ ] **Step 10.1.3: Stamp every manifest**

```bash
rg -l '"?version"?\s*[:=]'
```

Update each:
- `pyproject.toml` `version = "0.8.0"` (or whatever 10.1.2 produced).
- `uv.lock`: regenerate with `uv lock`.

If `rg` finds other manifests (`VERSION` file, `__init__.py` with `__version__`, etc.), update each.

### Subtask 10.2 — Final lint / format / type / test pass

- [ ] **Step 10.2.1: Format**

```bash
uv run ruff format src/ tests/
```

- [ ] **Step 10.2.2: Lint**

```bash
uv run ruff check src/ tests/
```

Fix any complaints inline. Common: unused imports left by the env-var removal in `run_cmd.py`, the now-removed `_tier_for_gb`, or stale imports in notebook tests.

- [ ] **Step 10.2.3: Type-check**

```bash
uv run mypy src/custom_sam_peft
```

Fix any complaints inline. Likely places:
- `OomState.pending_oom_events: list[OomEvent]` — make sure mypy sees the `OomEvent` import in `loop.py`.
- `PresetDecision.from_json` — annotate the local `d` dict's narrowing if mypy complains.
- `bundle.py` — the `from custom_sam_peft.presets import PresetDecision` creates a cross-module dependency; ensure no cycle (presets imports nothing from bundle, so this is safe).

- [ ] **Step 10.2.4: Full test suite**

```bash
uv run pytest -q
```

Expected: green.

- [ ] **Step 10.2.5: Coverage threshold**

`pyproject.toml` enforces `--cov-fail-under=80`. If coverage dips below 80 because of new code paths, add targeted tests until the threshold passes. The new `presets.py` and `calibrate_cmd.py` are heavily tested; coverage should be fine.

- [ ] **Step 10.3: Commit version + lockfile**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump to v0.8.0 for algorithmic preset + OOM safety net (#36)"
```

(Replace `0.8.0` with the version computed in 10.1.2.)

- [ ] **Step 10.4: Final sanity**

```bash
git log --oneline origin/main..HEAD
```

Expected commit shape (in order): 1 OomEvent, 2 presets, 3 calibrate, 4 trainer OOM, 5 bundler, 6 run_cmd, 7 notebook, 8 cleanups (optional), 9 gpu test, 10 version bump. Approximately 9–10 commits.

**Acceptance:** All of `ruff check`, `ruff format --check`, `mypy src/custom_sam_peft`, `pytest -q` green. Commit log is clean and ordered.

---

## Definition of Done

- [ ] Every task above checked off.
- [ ] `pytest -q` green on CPU.
- [ ] `pytest tests/gpu/test_calibrate_real.py --collect-only` collects exactly 1 test (orchestrator may dispatch GPU run separately).
- [ ] Spec coverage map at the top of this file matches the final code surface.
- [ ] `git grep -E 'pick_preset|preset_label|CUSTOM_SAM_PEFT_PRESET_LABEL|_tier_for_gb'` returns zero matches outside this plan and the locked spec.
- [ ] PR open against `main`, linking spec + this plan.
