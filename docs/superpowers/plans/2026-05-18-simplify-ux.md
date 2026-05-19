# spec/simplify-ux Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the *non-technical-user front door*: a new `esam3 run` orchestrator, a VRAM-tier preset helper, three notebook helpers, a results bundler (`runs/bundle.py` + `BundleContext`), a new `notebooks/esam3_train.ipynb`, and a README restructure (Beginner on top, Advanced wrapping existing content) plus a `cloud/runpod/README.md` walkthrough. See `docs/superpowers/specs/2026-05-18-simplify-ux-design.md`.

**Architecture:** Two front doors, one spine. The notebook is thin glue — env detect / preset patch / config write / `subprocess.Popen("esam3 run …")`. `esam3 run` composes the existing `run_training` → `run_eval` → (optional) `save_merged` → new `write_bundle`. The bundler accepts a frozen `BundleContext` dataclass that the orchestrator assembles in one place; per-example IoUs flow from an additive `Evaluator.evaluate(..., return_per_example_iou=True)` kwarg (and a matching `run_eval` extension) so the bundler doesn't pay a second pass over val. No existing subcommand changes behavior. The notebook never imports the bundler.

**Tech Stack:** Python 3.12, Typer, pydantic v2, rich, PyTorch, Pillow, NumPy, pytest, `typer.testing.CliRunner`, `IPython.display` (notebook only).

**Spec:** `docs/superpowers/specs/2026-05-18-simplify-ux-design.md`

---

## File Map

**New source files:**

```
src/esam3/
  presets.py                       # pick_preset(), preset_label()
  notebook_helpers.py              # detect_env, check_local_checkpoint, resolve_hf_token
  runs/
    __init__.py                    # package marker (empty)
    bundle.py                      # BundleContext, pick_samples, render_overlay, write_bundle
  cli/
    run_cmd.py                     # esam3 run (≤30 LOC body)
```

**New test files:**

```
tests/unit/
  test_presets.py
  test_notebook_helpers.py
  runs/
    __init__.py                    # empty (pytest package marker)
    test_bundle.py
tests/integration/
  test_cli_run.py
tests/gpu/
  test_run_end_to_end_gpu.py       # @pytest.mark.gpu, requires_compatible_gpu, requires_checkpoint
```

**New non-source artefacts:**

```
notebooks/
  esam3_train.ipynb                # user-facing (new; colab_gpu_tests.ipynb untouched)

cloud/
  runpod/
    README.md                      # layperson walkthrough
```

**Modified files:**

```
src/esam3/cli/main.py              # +1 registration for run_cmd.run
src/esam3/eval/evaluator.py        # additive Evaluator.evaluate(return_per_example_iou=False)
src/esam3/eval/runner.py           # additive run_eval kwargs: val_dataset, model, return_per_example_iou
README.md                          # restructure in place: Beginner section on top, Advanced wraps existing content
```

No existing source file is moved or deleted. Existing tests are not edited (the eval extensions are additive and the default-False kwarg keeps the existing return type intact).

---

## Phase Ordering & Dependency Chain

Phases are ordered for safe incremental review. Each leaves `main` working:

1. **Foundational helpers** — `presets.py` + `notebook_helpers.py`. Pure, no imports from `cli/` or `runs/`. (Tasks 1–2)
2. **Eval extensions** — additive `Evaluator.evaluate(return_per_example_iou=…)` and `run_eval(..., val_dataset=…, model=…, return_per_example_iou=…)` kwargs. Backward compatible; `esam3 eval` CLI behavior unchanged. (Tasks 3–4)
3. **Bundler module** — `runs/bundle.py` with `BundleContext`, `pick_samples`, `render_overlay`, `write_bundle`. Depends on the eval extension's per-example IoU output via `BundleContext.per_example_iou`. (Task 5)
4. **CLI orchestrator** — `cli/run_cmd.py` + registration in `cli/main.py`. Composes all of the above. (Task 6)
5. **User notebook** — `notebooks/esam3_train.ipynb`. Depends on `notebook_helpers`, `presets`, and `esam3 run` being available end-to-end. (Task 7)
6. **Documentation** — README restructure + `cloud/runpod/README.md`. Depends on the user-facing artefacts being final. (Task 8)
7. **GPU smoke + manual dry-runs** — `tests/gpu/test_run_end_to_end_gpu.py` + the two mandatory manual dry-runs (Colab + RunPod) per spec §8.4. (Task 9)

---

## Pre-flight check

- [ ] **Step 0a: Confirm working tree clean**

```bash
git status
```

Expected: only this plan file (and the approved spec) present. No other modifications.

- [ ] **Step 0b: Confirm baseline test suite passes before changes**

```bash
uv run pytest tests/unit tests/integration -x -q
```

Expected: all unit + integration tests pass on the current branch.

- [ ] **Step 0c: Confirm the five follow-up issues are open**

```bash
gh issue view 33 --json state -q .state
gh issue view 34 --json state -q .state
gh issue view 35 --json state -q .state
gh issue view 36 --json state -q .state
gh issue view 37 --json state -q .state
```

Expected: each prints `OPEN`. If any is `CLOSED`, halt and surface to the user — the spec assumes they're tracked.

---

## Task 1: `src/esam3/presets.py` — VRAM-tier patch generator

**Why first:** Pure helper, no internal dependencies. The notebook GENERATE cell and the bundler env-var plumbing both consume `preset_label()`; the orchestrator never imports `presets` directly.

**Spec ref:** §4.

**Files:**
- Create: `src/esam3/presets.py`
- Create: `tests/unit/test_presets.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_presets.py`:

```python
"""Tests for src/esam3/presets.py — VRAM-tier patch generator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from esam3.presets import pick_preset, preset_label

_GB = 1024**3


@pytest.fixture
def _force_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _stub_props(monkeypatch: pytest.MonkeyPatch, total_bytes: int) -> None:
    props = MagicMock(total_memory=total_bytes, name="StubGPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)


def test_pick_preset_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        pick_preset()


@pytest.mark.parametrize(
    "total_bytes, expected_method, expected_r, expected_bs, expected_ga, expected_ckpt",
    [
        (int(11.9 * _GB), "qlora", 8, 1, 16, True),     # <12 tier (upper edge)
        (int(12.0 * _GB), "qlora", 16, 1, 8, True),     # 12-24 tier (lower edge, inclusive)
        (int(23.9 * _GB), "qlora", 16, 1, 8, True),     # 12-24 tier (upper edge)
        (int(24.0 * _GB), "lora", 16, 2, 4, False),     # 24-48 tier (lower edge, inclusive)
        (int(47.9 * _GB), "lora", 16, 2, 4, False),     # 24-48 tier (upper edge)
        (int(48.0 * _GB), "lora", 32, 4, 2, False),     # ≥48 tier
    ],
)
def test_pick_preset_tiers(
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    total_bytes: int,
    expected_method: str,
    expected_r: int,
    expected_bs: int,
    expected_ga: int,
    expected_ckpt: bool,
) -> None:
    _stub_props(monkeypatch, total_bytes)
    patch = pick_preset()
    assert patch["peft"]["method"] == expected_method
    assert patch["peft"]["r"] == expected_r
    assert patch["train"]["batch_size"] == expected_bs
    assert patch["train"]["grad_accum_steps"] == expected_ga
    assert patch["model"]["gradient_checkpointing"] is expected_ckpt
    assert patch["model"]["dtype"] == "bfloat16"


@pytest.mark.parametrize(
    "total_bytes, must_contain",
    [
        (int(11.0 * _GB), "<12GB"),
        (int(16.0 * _GB), "12-24GB"),
        (int(40.0 * _GB), "24-48GB"),
        (int(80.0 * _GB), "≥48GB"),
    ],
)
def test_preset_label_format(
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    total_bytes: int,
    must_contain: str,
) -> None:
    _stub_props(monkeypatch, total_bytes)
    label = preset_label()
    assert "auto:" in label
    assert must_contain in label


def test_preset_label_with_explicit_total_bytes() -> None:
    # Does not need CUDA when explicit bytes provided.
    assert "12-24GB" in preset_label(total_bytes=int(16 * _GB))
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
uv run pytest tests/unit/test_presets.py -v
```

Expected: `ModuleNotFoundError: No module named 'esam3.presets'`.

- [ ] **Step 3: Implement `src/esam3/presets.py`**

```python
"""VRAM-tier preset patch generator.

The notebook GENERATE cell calls `pick_preset()` to derive PEFT method,
LoRA rank, batch size, grad-accum steps, gradient checkpointing, and
dtype from the current GPU's VRAM. `preset_label()` produces the matching
short string the orchestrator forwards to the bundler via env var.

Replacement plan: see logs/TODO.md / issue #36 — algorithmic derivation
will replace this table-driven helper in a future spec.
"""

from __future__ import annotations

import torch

_GB = 1024**3

_CUDA_HINT = (
    "pick_preset() requires CUDA; got cpu-only torch. "
    "In Colab: Runtime → Change runtime type → GPU. "
    "On RunPod: deploy a GPU pod."
)


def _device_total_bytes() -> int:
    return int(torch.cuda.get_device_properties(0).total_memory)


def _tier_for_gb(total_gb: float) -> str:
    if total_gb < 12.0:
        return "<12GB"
    if total_gb < 24.0:
        return "12-24GB"
    if total_gb < 48.0:
        return "24-48GB"
    return "≥48GB"


def pick_preset() -> dict[str, dict[str, object]]:
    """Return a config-patch dict keyed by the current GPU's VRAM tier.

    Raises:
        RuntimeError: torch.cuda.is_available() is False.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)

    total_gb = _device_total_bytes() / _GB
    tier = _tier_for_gb(total_gb)

    if tier == "<12GB":
        return {
            "peft": {"method": "qlora", "r": 8},
            "train": {"batch_size": 1, "grad_accum_steps": 16},
            "model": {"gradient_checkpointing": True, "dtype": "bfloat16"},
        }
    if tier == "12-24GB":
        return {
            "peft": {"method": "qlora", "r": 16},
            "train": {"batch_size": 1, "grad_accum_steps": 8},
            "model": {"gradient_checkpointing": True, "dtype": "bfloat16"},
        }
    if tier == "24-48GB":
        return {
            "peft": {"method": "lora", "r": 16},
            "train": {"batch_size": 2, "grad_accum_steps": 4},
            "model": {"gradient_checkpointing": False, "dtype": "bfloat16"},
        }
    # ≥48GB
    return {
        "peft": {"method": "lora", "r": 32},
        "train": {"batch_size": 4, "grad_accum_steps": 2},
        "model": {"gradient_checkpointing": False, "dtype": "bfloat16"},
    }


def preset_label(total_bytes: int | None = None) -> str:
    """Return a short tier label like 'auto: 12-24GB tier'.

    If `total_bytes` is None, reads device 0's total memory (requires CUDA).
    """
    if total_bytes is None:
        if not torch.cuda.is_available():
            raise RuntimeError(_CUDA_HINT)
        total_bytes = _device_total_bytes()
    return f"auto: {_tier_for_gb(total_bytes / _GB)} tier"
```

- [ ] **Step 4: Run the tests and confirm pass**

```bash
uv run pytest tests/unit/test_presets.py -v
```

Expected: all green.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/esam3/presets.py tests/unit/test_presets.py
uv run mypy src/esam3/presets.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/presets.py tests/unit/test_presets.py
git commit -m "feat(presets): VRAM-tier preset patch + label helper"
```

---

## Task 2: `src/esam3/notebook_helpers.py` — env detect, local checkpoint, HF token

**Spec ref:** §5.

**Files:**
- Create: `src/esam3/notebook_helpers.py`
- Create: `tests/unit/test_notebook_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_notebook_helpers.py`:

```python
"""Tests for src/esam3/notebook_helpers.py."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from esam3.notebook_helpers import (
    check_local_checkpoint,
    detect_env,
    resolve_hf_token,
)


# ---- detect_env ----------------------------------------------------------


def test_detect_env_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_GPU", "1")
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert detect_env() == "colab"


def test_detect_env_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123")
    assert detect_env() == "runpod"


def test_detect_env_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert detect_env() == "unknown"


def test_detect_env_colab_wins_over_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_GPU", "1")
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123")
    assert detect_env() == "colab"


# ---- check_local_checkpoint ----------------------------------------------


def test_check_local_checkpoint_present(tmp_path: Path) -> None:
    (tmp_path / "sam3.1_multiplex.pt").write_bytes(b"x")
    assert check_local_checkpoint(tmp_path, "sam3.1_multiplex.pt") is True


def test_check_local_checkpoint_absent(tmp_path: Path) -> None:
    assert check_local_checkpoint(tmp_path, "sam3.1_multiplex.pt") is False


def test_check_local_checkpoint_dir_not_file(tmp_path: Path) -> None:
    (tmp_path / "sam3.1_multiplex.pt").mkdir()
    assert check_local_checkpoint(tmp_path, "sam3.1_multiplex.pt") is False


# ---- resolve_hf_token ----------------------------------------------------


@pytest.mark.parametrize("env", ["colab", "runpod", "unknown"])
def test_resolve_hf_token_local_short_circuits(
    env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # The function MUST NOT read env or import google.colab when local_present.
    assert resolve_hf_token(env, local_present=True) is None


def test_resolve_hf_token_missing_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        resolve_hf_token("colab", local_present=False)


def test_resolve_hf_token_missing_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Environment Variables"):
        resolve_hf_token("runpod", local_present=False)


def test_resolve_hf_token_missing_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="shell environment"):
        resolve_hf_token("unknown", local_present=False)


def test_resolve_hf_token_runpod_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_runpod_xyz")
    assert resolve_hf_token("runpod", local_present=False) == "hf_runpod_xyz"


def test_resolve_hf_token_unknown_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_shell_xyz")
    assert resolve_hf_token("unknown", local_present=False) == "hf_shell_xyz"


def test_resolve_hf_token_colab_userdata_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub google.colab.userdata so the colab arm returns the token."""
    fake_userdata = SimpleNamespace(get=lambda _key: "hf_colab_abc")
    fake_colab = SimpleNamespace(userdata=fake_userdata)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(colab=fake_colab))
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    assert resolve_hf_token("colab", local_present=False) == "hf_colab_abc"


def test_resolve_hf_token_colab_userdata_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub userdata.get → None → still surfaces the 'Colab Secrets' message."""

    def _get(_key: str) -> Any:
        return None

    fake_colab = SimpleNamespace(userdata=SimpleNamespace(get=_get))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(colab=fake_colab))
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        resolve_hf_token("colab", local_present=False)
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
uv run pytest tests/unit/test_notebook_helpers.py -v
```

Expected: `ModuleNotFoundError: No module named 'esam3.notebook_helpers'`.

- [ ] **Step 3: Implement `src/esam3/notebook_helpers.py`**

```python
"""Helpers used by `notebooks/esam3_train.ipynb` for env detection,
local-checkpoint short-circuit, and HF-token resolution.

CLI never imports this module. Tests and the notebook do.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

_LOG = logging.getLogger(__name__)

Env = Literal["colab", "runpod", "unknown"]

_COLAB_ERR = "Set HF_TOKEN in Colab Secrets (left sidebar → 🔑)."
_RUNPOD_ERR = (
    "Set HF_TOKEN in your pod's Environment Variables, "
    "or mount a network volume containing models/sam3.1/sam3.1_multiplex.pt."
)
_UNKNOWN_ERR = "Set HF_TOKEN in your shell environment (export HF_TOKEN=…)."


def detect_env() -> Env:
    """Best-effort environment detection from env vars.

    - 'colab' if os.environ.get('COLAB_GPU') is set (any value).
    - 'runpod' elif os.environ.get('RUNPOD_POD_ID') is set.
    - 'unknown' otherwise.
    """
    if os.environ.get("COLAB_GPU") is not None:
        return "colab"
    if os.environ.get("RUNPOD_POD_ID") is not None:
        return "runpod"
    return "unknown"


def check_local_checkpoint(local_dir: Path, checkpoint_file: str) -> bool:
    """Return True iff `(local_dir / checkpoint_file).is_file()`."""
    return (Path(local_dir) / checkpoint_file).is_file()


def _resolve_colab_token() -> str | None:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    return userdata.get("HF_TOKEN")


def resolve_hf_token(env: Env, local_present: bool) -> str | None:
    """Resolve the HF token according to environment and local-checkpoint state.

    - If `local_present` is True: log 'local checkpoint detected — skipping HF
      auth' and return None.
    - Else, fetch the token from the env-appropriate source. Missing token
      raises RuntimeError with an env-specific friendly message.
    """
    if local_present:
        _LOG.info("local checkpoint detected — skipping HF auth")
        return None

    if env == "colab":
        token = _resolve_colab_token()
        if not token:
            raise RuntimeError(_COLAB_ERR)
        return token

    if env == "runpod":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(_RUNPOD_ERR)
        return token

    # unknown
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(_UNKNOWN_ERR)
    return token
```

- [ ] **Step 4: Run the tests and confirm pass**

```bash
uv run pytest tests/unit/test_notebook_helpers.py -v
```

Expected: all green.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/esam3/notebook_helpers.py tests/unit/test_notebook_helpers.py
uv run mypy src/esam3/notebook_helpers.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/notebook_helpers.py tests/unit/test_notebook_helpers.py
git commit -m "feat(notebook-helpers): detect_env + check_local_checkpoint + resolve_hf_token"
```

---

## Task 3: Extend `Evaluator.evaluate(return_per_example_iou=False)`

**Why now:** The bundler depends on per-example mean IoU. Eval is the cheapest place to compute it — we get them for free during the IoU thresholding loop. This extension is **additive and backward compatible** (`Evaluator.evaluate(model, dataset)` keeps returning `MetricsReport`).

**Spec ref:** §6.3.

**Files:**
- Modify: `src/esam3/eval/evaluator.py`
- Modify: `tests/unit/test_evaluator.py` (add coverage; existing tests must not change)

- [ ] **Step 1: Inspect the existing `Evaluator.evaluate` body**

```bash
uv run python -c "import inspect, esam3.eval.evaluator as e; print(inspect.getsource(e.Evaluator.evaluate))"
```

Confirm: today's body computes COCO mAP via pycocotools over the whole prediction set. Per-example IoU is **not** currently computed by `compute_coco_map` — it returns per-class AP only. The new path will re-use the same `predictions` list and the GT `COCO` object to compute per-example mean IoU across `cfg.iou_thresholds`.

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_evaluator.py`:

```python
def test_evaluate_returns_per_example_iou_when_requested(
    tmp_path: pytest.Path,  # type: ignore[attr-defined]
) -> None:
    """When return_per_example_iou=True, return (MetricsReport, list[float]) aligned with dataset indices."""
    from esam3.config.schema import EvalConfig
    from esam3.eval.evaluator import Evaluator
    from tests.unit.test_evaluator import _build_synthetic_dataset, _build_perfect_model

    # _build_synthetic_dataset / _build_perfect_model — re-use existing helpers
    # at the top of this test file. If they don't exist with these exact names,
    # use whatever the file already exposes for `test_evaluate_perfect_predictions`.
    dataset = _build_synthetic_dataset(n=4)
    model = _build_perfect_model(dataset)

    cfg = EvalConfig(mode="full")
    evaluator = Evaluator(cfg)
    out = evaluator.evaluate(model, dataset, return_per_example_iou=True)

    assert isinstance(out, tuple)
    report, ious = out
    assert hasattr(report, "overall")
    assert isinstance(ious, list)
    assert len(ious) == len(dataset)
    # Perfect predictions → all mean-IoUs at or near 1.0.
    assert all(0.99 <= v <= 1.0 for v in ious)


def test_evaluate_default_unchanged_returns_report_only() -> None:
    """Backward-compat: omitting the flag returns MetricsReport, not a tuple."""
    from esam3.config.schema import EvalConfig
    from esam3.eval.evaluator import Evaluator
    from tests.unit.test_evaluator import _build_synthetic_dataset, _build_perfect_model

    dataset = _build_synthetic_dataset(n=2)
    model = _build_perfect_model(dataset)

    out = Evaluator(EvalConfig(mode="full")).evaluate(model, dataset)
    # Must NOT be a tuple — backward-compat contract.
    assert not isinstance(out, tuple)
    assert hasattr(out, "overall")
```

If `_build_synthetic_dataset` / `_build_perfect_model` are not the existing helper names in `tests/unit/test_evaluator.py`, open that file and reuse the dataset/model construction helper(s) it already exposes for a passing perfect-prediction test — the only constraint is "small, deterministic, in-memory dataset of N text-prompt examples where the model returns the GT mask".

- [ ] **Step 3: Run the new tests and confirm they fail**

```bash
uv run pytest tests/unit/test_evaluator.py -k "per_example_iou or default_unchanged" -v
```

Expected: `TypeError: evaluate() got an unexpected keyword argument 'return_per_example_iou'` for the first; the second already passes (since omitting an unknown kwarg can't trigger the new branch).

- [ ] **Step 4: Implement the extension in `Evaluator.evaluate`**

Open `src/esam3/eval/evaluator.py`. Change the signature and add the per-example IoU computation. The change pattern:

```python
def evaluate(
    self,
    model: Any,
    dataset: Dataset,
    *,
    return_per_example_iou: bool = False,
) -> MetricsReport | tuple[MetricsReport, list[float]]:
    """Run the model over the dataset and return a MetricsReport.

    When ``return_per_example_iou=True``, also returns a list of per-example
    MEAN IoU values across ``cfg.iou_thresholds`` aligned with dataset indices.
    The default ``False`` preserves the previous return type for backward
    compatibility (e.g. `esam3 eval` CLI, mid-training eval).
    """
    # … existing body up to and including compute_coco_map …

    self._last_predictions = predictions

    if not return_per_example_iou:
        return report

    per_example_iou = self._compute_per_example_iou(examples, predictions, gt)
    return report, per_example_iou
```

Add the helper method on `Evaluator`:

```python
def _compute_per_example_iou(
    self,
    examples: Sequence[Example],
    predictions: list[dict[str, object]],
    gt: COCO,
) -> list[float]:
    """Compute mean IoU per example across self.cfg.iou_thresholds.

    The 'IoU' here is segmentation IoU between the best-matched predicted
    mask and any GT mask in the same image (greedy match, max IoU). For an
    example with no GT instances, IoU is 0.0 if it has predictions, else 1.0
    (vacuous match — consistent with COCO's empty-image handling). Examples
    skipped during model inference are marked NaN; pick_samples treats NaN
    as -inf for ranking and they are eligible only as 'worst' picks.
    """
    import numpy as np
    import pycocotools.mask as mask_utils

    out: list[float] = []
    # Group predictions by image_id for cheap lookup.
    preds_by_image: dict[int, list[dict[str, object]]] = {}
    for entry in predictions:
        preds_by_image.setdefault(int(entry["image_id"]), []).append(entry)

    for ex in examples:
        int_id = _int_image_id(ex.image_id)
        gt_anns = gt.imgToAnns.get(int_id, [])
        ex_preds = preds_by_image.get(int_id, [])

        if not gt_anns and not ex_preds:
            out.append(1.0)  # vacuous match
            continue
        if not gt_anns or not ex_preds:
            out.append(0.0)
            continue

        # Build (n_pred, n_gt) IoU matrix for this example.
        pred_rles = [p["segmentation"] for p in ex_preds]
        gt_rles = [a["segmentation"] for a in gt_anns]
        iscrowd = [0] * len(gt_rles)
        iou_mat = mask_utils.iou(pred_rles, gt_rles, iscrowd)
        # max-IoU greedy: for each GT, the best predicted IoU; mean over thresholds.
        # Spec §6.1: "the MEAN IoU across the eval's IoU thresholds [0.5, …, 0.95]".
        # We compute the per-GT best-pred IoU once, then average across thresholds:
        # at threshold t, the per-GT-IoU is the best-pred IoU if >= t else 0, so the
        # threshold-mean reduces to mean_t(best_iou >= t) which is the cdf at the
        # discrete thresholds. Use that as the example score.
        if iou_mat.size == 0:
            out.append(0.0)
            continue
        best_per_gt = np.asarray(iou_mat).max(axis=0)  # (n_gt,)
        thresholds = np.asarray(self.cfg.iou_thresholds)
        # Mean over (gt, thresholds) of (best_per_gt[g] >= thresholds[t]).
        hit = best_per_gt[:, None] >= thresholds[None, :]
        out.append(float(hit.mean()))

    return out
```

- [ ] **Step 5: Run the new tests + existing test suite**

```bash
uv run pytest tests/unit/test_evaluator.py -v
uv run pytest tests/unit -x -q
```

Expected: both clean. Existing callers (none provide `return_per_example_iou`) continue to receive `MetricsReport`.

- [ ] **Step 6: Lint + type-check**

```bash
uv run ruff check src/esam3/eval/evaluator.py tests/unit/test_evaluator.py
uv run mypy src/esam3/eval/evaluator.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/esam3/eval/evaluator.py tests/unit/test_evaluator.py
git commit -m "feat(eval): Evaluator.evaluate gains return_per_example_iou kwarg"
```

---

## Task 4: Extend `run_eval` with `val_dataset`, `model`, `return_per_example_iou` kwargs

**Why now:** Task 3 is the producer; Task 4 is the runner-level passthrough that lets `esam3 run` reuse a single val dataset + model wrapper across the eval and bundle phases (spec §3.4).

**Spec ref:** §3.4, §6.3.

**Files:**
- Modify: `src/esam3/eval/runner.py`
- Modify: `tests/unit/test_eval_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_runner.py`:

```python
def test_run_eval_accepts_prebuilt_val_dataset_and_model(
    tmp_path: pytest.Path,  # type: ignore[attr-defined]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If val_dataset/model are provided, runner MUST NOT call lookup('dataset', …)
    or load_sam31."""
    from unittest.mock import MagicMock

    cfg = _make_cfg(format_="coco", peft_method="lora")
    forbidden: list[str] = []

    def _forbidden_lookup(kind: str, name: str) -> object:
        forbidden.append(f"{kind}:{name}")
        return lambda *a, **kw: None

    def _forbidden_load(_m: object) -> object:
        forbidden.append("load_sam31")
        return None

    monkeypatch.setattr("esam3.eval.runner.lookup", _forbidden_lookup)
    monkeypatch.setattr("esam3.eval.runner.load_sam31", _forbidden_load)
    monkeypatch.setattr("esam3.eval.runner.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.5})
    captured: dict[str, object] = {}

    def _fake_evaluator_init(_cfg: object) -> object:
        ev = MagicMock()

        def _evaluate(model: object, dataset: object, *, return_per_example_iou: bool = False) -> object:
            captured["model"] = model
            captured["dataset"] = dataset
            captured["return_per_example_iou"] = return_per_example_iou
            if return_per_example_iou:
                return fake_report, [0.1, 0.5, 0.9]
            return fake_report

        ev.evaluate = _evaluate
        ev.evaluate_and_save = MagicMock(return_value=fake_report)
        return ev

    monkeypatch.setattr("esam3.eval.runner.Evaluator", _fake_evaluator_init)

    fake_ds = MagicMock(__len__=lambda self: 3, class_names=["a"])
    fake_model = MagicMock()
    report, per_ex = run_eval(
        cfg,
        checkpoint=tmp_path,
        split="val",
        output_dir=tmp_path,
        val_dataset=fake_ds,
        model=fake_model,
        return_per_example_iou=True,
    )
    assert report is fake_report
    assert per_ex == [0.1, 0.5, 0.9]
    assert captured["dataset"] is fake_ds
    assert captured["model"] is fake_model
    assert captured["return_per_example_iou"] is True
    assert forbidden == []  # neither lookup nor load_sam31 should have been called


def test_run_eval_return_per_example_iou_default_false_unchanged(
    tmp_path: pytest.Path,  # type: ignore[attr-defined]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default kwarg path returns MetricsReport (not tuple) — existing CLI contract."""
    from unittest.mock import MagicMock

    cfg = _make_cfg(format_="coco", peft_method="lora")
    monkeypatch.setattr("esam3.eval.runner.lookup", lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]))
    monkeypatch.setattr("esam3.eval.runner.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("esam3.eval.runner.load_lora", lambda *_a, **_kw: None)

    fake_report = MagicMock(overall={"mAP": 0.0})
    monkeypatch.setattr(
        "esam3.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    out = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert out is fake_report
    assert not isinstance(out, tuple)
```

- [ ] **Step 2: Run the new tests and confirm they fail**

```bash
uv run pytest tests/unit/test_eval_runner.py -k "prebuilt or return_per_example_iou_default" -v
```

Expected: `TypeError: run_eval() got an unexpected keyword argument 'val_dataset'`.

- [ ] **Step 3: Modify `src/esam3/eval/runner.py`**

Replace the body with the extended signature:

```python
"""End-to-end eval pipeline.

The CLI (`esam3 eval`) is a thin wrapper over `run_eval`. `esam3 run`
calls it with `val_dataset` / `model` / `return_per_example_iou=True` so
it can re-use a single dataset+wrapper across the eval and bundle phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from esam3._registry import lookup
from esam3.config.schema import TrainConfig
from esam3.data.base import Dataset
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import MetricsReport
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import load_lora


def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: bool = False,
) -> MetricsReport | tuple[MetricsReport, list[float]]:
    """Load model + adapter, build dataset, run Evaluator.

    Optional additive kwargs (used by `esam3 run`):
      - ``val_dataset``: pre-built dataset; skips registry lookup + transform setup.
      - ``model``: pre-loaded + adapted wrapper; skips ``load_sam31`` + ``load_lora``.
      - ``return_per_example_iou``: when True, returns ``(MetricsReport, list[float])``.

    Backward-compat: defaults preserve the previous behavior (rebuild
    dataset, load model + LoRA, return ``MetricsReport``).

    Raises:
        ValueError: cfg.peft.method != 'lora' AND model is None (QLoRA load
            from disk is not yet supported; pre-loaded wrappers bypass this).
        ValueError: split == 'test' and cfg.data.test is None.
    """
    if model is None and cfg.peft.method != "lora":
        raise ValueError(
            f"checkpoint loading currently supports only LoRA adapters; "
            f"got peft.method={cfg.peft.method!r}"
        )
    if split == "test" and cfg.data.test is None:
        raise ValueError("--split test requires data.test in config; got None for data.test")

    if val_dataset is None:
        cfg_dict = cfg.data.model_dump()
        if split == "test":
            cfg_dict["val"] = cfg_dict["test"]
        builder = lookup("dataset", cfg.data.format)
        dataset = cast(Dataset, builder(cfg_dict, model_name=cfg.model.name, pipeline="eval"))
    else:
        dataset = val_dataset

    if model is None:
        wrapper = load_sam31(cfg.model)
        load_lora(wrapper, checkpoint)
    else:
        wrapper = model

    eval_cfg = cfg.eval
    if save_predictions is not None:
        eval_cfg = eval_cfg.model_copy(update={"save_predictions": save_predictions})

    evaluator = Evaluator(eval_cfg)
    out = output_dir if output_dir is not None else checkpoint.parent

    if return_per_example_iou:
        # We need both the metrics report (and metrics.json on disk) AND the
        # per-example IoUs. `evaluate_and_save` only persists; call `evaluate`
        # for the data we need and then mirror the persistence the CLI path does.
        out.mkdir(parents=True, exist_ok=True)
        result = evaluator.evaluate(wrapper, dataset, return_per_example_iou=True)
        report, per_example_iou = cast(tuple[MetricsReport, list[float]], result)
        import json
        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "overall": report.overall,
                    "per_class": report.per_class,
                    "n_images": report.n_images,
                    "n_predictions": report.n_predictions,
                },
                indent=2,
            )
        )
        if eval_cfg.save_predictions and eval_cfg.mode == "full":
            (out / "predictions.json").write_text(json.dumps(evaluator._last_predictions))
        return report, per_example_iou

    return evaluator.evaluate_and_save(wrapper, dataset, out)
```

- [ ] **Step 4: Run the new tests and confirm pass**

```bash
uv run pytest tests/unit/test_eval_runner.py -v
```

Expected: every test green (existing + 2 new).

- [ ] **Step 5: Run the eval CLI integration tests to confirm no regression**

```bash
uv run pytest tests/unit/test_cli.py -k "eval" -v
```

Expected: PASS — `esam3 eval` still passes the default `return_per_example_iou=False` (it doesn't set it) and continues to receive `MetricsReport`.

- [ ] **Step 6: Lint + type-check**

```bash
uv run ruff check src/esam3/eval/runner.py tests/unit/test_eval_runner.py
uv run mypy src/esam3/eval/runner.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/esam3/eval/runner.py tests/unit/test_eval_runner.py
git commit -m "feat(eval-runner): additive val_dataset/model/return_per_example_iou kwargs"
```

---

## Task 5: `src/esam3/runs/bundle.py` — BundleContext, pick_samples, render_overlay, write_bundle

**Why now:** All upstream dependencies (eval per-example IoU; runner-level passthrough) are in place. The bundler is the largest single deliverable. Implement the three public functions in dependency order: `pick_samples` (pure), `render_overlay` (pure + Pillow), `write_bundle` (composes them + writes disk).

**Spec ref:** §6.

**Files:**
- Create: `src/esam3/runs/__init__.py` (empty)
- Create: `src/esam3/runs/bundle.py`
- Create: `tests/unit/runs/__init__.py` (empty)
- Create: `tests/unit/runs/test_bundle.py`

- [ ] **Step 1: Create the package markers**

```bash
mkdir -p src/esam3/runs tests/unit/runs
touch src/esam3/runs/__init__.py tests/unit/runs/__init__.py
```

- [ ] **Step 2: Write the failing tests for `pick_samples`**

Create `tests/unit/runs/test_bundle.py`:

```python
"""Tests for src/esam3/runs/bundle.py."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from esam3.runs.bundle import (
    BundleContext,
    pick_samples,
    render_overlay,
    write_bundle,
)


# ---- pick_samples --------------------------------------------------------


@pytest.mark.parametrize(
    "mAP, ious, expected_composition",
    [
        # mAP >= 0.7 → 4 best + 1 median + 1 worst (n_val=6)
        (0.80, [0.1, 0.4, 0.5, 0.6, 0.85, 0.95], (4, 1, 1)),
        # 0.4 <= mAP < 0.7 → 2 best + 2 median + 2 worst (n_val=6)
        (0.50, [0.1, 0.2, 0.4, 0.5, 0.85, 0.9], (2, 2, 2)),
        # mAP < 0.4 → 1 best + 1 median + 4 worst (n_val=6)
        (0.10, [0.0, 0.1, 0.2, 0.3, 0.5, 0.95], (1, 1, 4)),
        # NaN mAP → 'poor' bracket
        (float("nan"), [0.0, 0.1, 0.2, 0.3, 0.5, 0.95], (1, 1, 4)),
    ],
)
def test_pick_samples_brackets_at_n6(
    mAP: float, ious: list[float], expected_composition: tuple[int, int, int]
) -> None:
    picks = pick_samples(ious, mAP, n_val=6)
    assert len(picks) == 6
    # Slot-bucketing tracks the construction order — best…, median…, worst….
    b, m, w = expected_composition
    assert b + m + w == 6
    # Verify the highest-IoU index is in the 'best' slot region.
    best_region = picks[:b]
    worst_region = picks[b + m :]
    assert max(ious) in [ious[i] for i in best_region]
    assert min(ious) in [ious[i] for i in worst_region]


def test_pick_samples_empty_returns_empty() -> None:
    assert pick_samples([], 0.5, n_val=0) == []


def test_pick_samples_n_lt_6_caps_and_topups_with_worst() -> None:
    # n_val=2, mAP=0.5 (mid bracket 2/2/2) → cap 2; floor 0/0/0 → topup worst → (0, 0, 2)
    picks = pick_samples([0.7, 0.1], 0.5, n_val=2)
    assert len(picks) == 2
    # Both indices land in 'worst' (idx asc tiebreak).
    assert picks == [1, 0]  # 0.1 < 0.7 so idx 1 first


def test_pick_samples_n_val_1_landed_in_worst_for_poor_bracket() -> None:
    picks = pick_samples([0.42], 0.1, n_val=1)
    assert picks == [0]


def test_pick_samples_identical_ious_tiebreak_by_index_asc() -> None:
    picks = pick_samples([0.5] * 6, 0.5, n_val=6)
    # All identical → best/worst sort stable by index asc; median fills the remainder.
    assert sorted(picks) == [0, 1, 2, 3, 4, 5]


def test_pick_samples_all_zero_ious() -> None:
    picks = pick_samples([0.0] * 6, 0.0, n_val=6)  # 'poor' bracket
    assert sorted(picks) == [0, 1, 2, 3, 4, 5]


def test_pick_samples_nan_iou_treated_as_minus_inf() -> None:
    # idx 2 is NaN; should NOT appear in 'best' but must be eligible for 'worst'.
    ious = [0.9, 0.5, float("nan"), 0.1, 0.2, 0.6]
    picks = pick_samples(ious, 0.10, n_val=6)
    # Poor bracket = 1 best + 1 median + 4 worst → best is idx 0 (0.9).
    assert picks[0] == 0
    # NaN-IoU index must be present in worst region (treated as -inf, sorts as worst).
    assert 2 in picks[2:]


def test_pick_samples_ordering_best_then_median_then_worst() -> None:
    ious = [0.9, 0.4, 0.5, 0.3, 0.1, 0.85]
    picks = pick_samples(ious, 0.50, n_val=6)  # 2/2/2 bracket
    # First 2 = best (sorted desc by IoU, ties by idx asc): 0.9 → idx 0, 0.85 → idx 5.
    assert picks[:2] == [0, 5]
    # Last 2 = worst (sorted asc): 0.1 → 4, 0.3 → 3.
    assert picks[4:] == [4, 3]


# ---- render_overlay ------------------------------------------------------


def test_render_overlay_returns_rgb_image_of_input_size() -> None:
    img = Image.new("RGB", (32, 24), (10, 10, 10))
    pred = np.zeros((24, 32), dtype=bool)
    pred[:, :16] = True
    gt = np.zeros((24, 32), dtype=bool)
    gt[:, 16:] = True
    out = render_overlay(img, pred, gt, caption="best @ IoU=0.83")
    assert out.mode == "RGB"
    assert out.size == img.size


def test_render_overlay_recolours_prediction_and_gt() -> None:
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    pred = np.zeros((16, 16), dtype=bool)
    pred[:, :8] = True       # left half = prediction (magenta)
    gt = np.zeros((16, 16), dtype=bool)
    gt[:, 8:] = True         # right half = GT (cyan)
    out = render_overlay(img, pred, gt, caption="x")
    arr = np.asarray(out)
    # Left half should pick up magenta (high R, low G, high B).
    left = arr[8, 4]
    # Right half should pick up cyan (low R, high G, high B).
    right = arr[8, 12]
    assert left[0] > 30 and left[2] > 30          # magenta has R + B
    assert right[1] > 30 and right[2] > 30        # cyan has G + B
    assert left[1] < left[0]                       # less green than red in magenta
    assert right[0] < right[1]                     # less red than green in cyan


def test_render_overlay_raises_on_shape_mismatch() -> None:
    img = Image.new("RGB", (16, 16))
    pred = np.zeros((16, 16), dtype=bool)
    gt = np.zeros((15, 16), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        render_overlay(img, pred, gt, caption="x")


# ---- write_bundle --------------------------------------------------------


def _make_metrics(mAP: float) -> MagicMock:
    r = MagicMock()
    r.overall = {"mAP": mAP, "mAP_50": mAP, "mAP_75": mAP}
    r.per_class = {}
    r.n_images = 3
    r.n_predictions = 3
    return r


def _make_ctx(tmp_path: Path, **overrides: object) -> BundleContext:
    base = BundleContext(
        run_dir=tmp_path / "run",
        config_path=tmp_path / "config.yaml",
        start_ts=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        end_ts=datetime(2026, 5, 18, 12, 5, 0, tzinfo=UTC),
        preset_label="auto: 24-48GB tier",
        per_example_iou=[0.1, 0.5, 0.9],
        merged_dir=None,
        merged_export_error=None,
    )
    base.run_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text("run: {name: r}\n")
    return replace(base, **overrides)


def _make_dataset(n: int) -> MagicMock:
    ds = MagicMock()
    ds.__len__ = lambda self: n
    return ds


def test_write_bundle_writes_summary_and_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(tmp_path)
    val_ds = _make_dataset(3)
    model = MagicMock()
    report = _make_metrics(mAP=0.42)

    # Monkeypatch the per-example inference helper to deterministic 1-pixel masks.
    def _fake_render(image: Image.Image, pred: object, gt: object, *, caption: str) -> Image.Image:
        return Image.new("RGB", (8, 8), (1, 2, 3))

    monkeypatch.setattr("esam3.runs.bundle.render_overlay", _fake_render)

    # Monkeypatch the per-sample re-inference to a no-op that yields blank masks.
    def _fake_run_one(_model: object, _ds: object, _idx: int) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        return Image.new("RGB", (8, 8)), np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=bool)

    monkeypatch.setattr("esam3.runs.bundle._reinfer_one_example", _fake_run_one)

    write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=model)
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "0.4200" in summary
    assert "## Run" in summary
    assert "## Hardware" in summary
    assert "## Preset" in summary
    assert "## Outputs" in summary
    assert "## Samples" in summary
    pngs = sorted((ctx.run_dir / "samples").glob("*.png"))
    assert len(pngs) >= 1
    # Names embed bracket label and ordinal.
    names = [p.name for p in pngs]
    assert any("worst" in n for n in names)


def test_write_bundle_empty_val_writes_summary_with_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    val_ds = _make_dataset(0)
    report = _make_metrics(mAP=0.0)
    write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "empty val" in summary.lower()
    assert (ctx.run_dir / "samples").is_dir()
    assert list((ctx.run_dir / "samples").glob("*.png")) == []


def test_write_bundle_merge_failure_recorded_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(
        tmp_path,
        merged_dir=None,
        merged_export_error="ValueError: rank mismatch",
    )
    val_ds = _make_dataset(0)
    report = _make_metrics(mAP=0.0)
    write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "FAILED" in summary
    assert "rank mismatch" in summary


def test_write_bundle_skipped_sample_logged_and_summary_notes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _make_ctx(tmp_path)
    val_ds = _make_dataset(3)
    report = _make_metrics(mAP=0.42)

    def _explode(_model: object, _ds: object, _idx: int) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        raise RuntimeError("forward kaboom")

    monkeypatch.setattr("esam3.runs.bundle._reinfer_one_example", _explode)

    with caplog.at_level("WARNING"):
        write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=MagicMock())

    summary = (ctx.run_dir / "summary.md").read_text()
    assert "skipped samples" in summary.lower()
    assert "forward kaboom" in caplog.text or any("forward kaboom" in r.message for r in caplog.records)
```

- [ ] **Step 3: Run the tests and confirm they fail**

```bash
uv run pytest tests/unit/runs/test_bundle.py -v
```

Expected: `ModuleNotFoundError: No module named 'esam3.runs'`.

- [ ] **Step 4: Implement `src/esam3/runs/bundle.py`**

```python
"""Results bundler — writes ``runs/<id>/summary.md`` + ``samples/*.png``.

Three public functions in dependency order:

1. ``pick_samples`` — pure: index selection from per-example IoU + overall mAP.
2. ``render_overlay`` — pure: image + pred/gt masks → PIL image with caption.
3. ``write_bundle`` — composes the above, runs per-sample re-inference, writes disk.

The orchestrator (``esam3 run``) assembles a frozen ``BundleContext`` and
calls ``write_bundle(ctx, …)``. The notebook does not import this module.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §6.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from esam3.data.base import Dataset, Example, TextPrompts
from esam3.eval.metrics import MetricsReport

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BundleContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleContext:
    """All run-context fields the bundler needs, assembled by `esam3 run`."""

    run_dir: Path
    config_path: Path
    start_ts: datetime
    end_ts: datetime
    preset_label: str | None
    per_example_iou: list[float]
    merged_dir: Path | None
    merged_export_error: str | None


# ---------------------------------------------------------------------------
# pick_samples
# ---------------------------------------------------------------------------


def _bracket(mAP: float) -> tuple[int, int, int]:
    """Return (best, median, worst) triple for the spec brackets at n_val=6."""
    if math.isnan(mAP) or mAP < 0.4:
        return (1, 1, 4)
    if mAP < 0.7:
        return (2, 2, 2)
    return (4, 1, 1)


def _score(per_example_iou: list[float]) -> list[float]:
    """Replace NaN with -inf for ranking (NaN sorts as worst)."""
    return [(-math.inf if math.isnan(x) else x) for x in per_example_iou]


def pick_samples(
    per_example_iou: list[float],
    overall_mAP: float,
    n_val: int,
) -> list[int]:
    """Pick up to 6 val indices to render, partitioned by bracket.

    Returns the concatenation (best…, median…, worst…). Tie-break by index asc.
    See spec §6.1 for the bracket table.
    """
    if n_val == 0:
        return []
    if len(per_example_iou) != n_val:
        raise ValueError(
            f"len(per_example_iou)={len(per_example_iou)} != n_val={n_val}"
        )

    cap = min(6, n_val)
    b, m, w = _bracket(overall_mAP)

    if cap < 6:
        ratios = [b / 6.0, m / 6.0, w / 6.0]
        picks = [int(r * cap) for r in ratios]
        while sum(picks) < cap:
            picks[2] += 1  # top up with 'worst'
        b, m, w = picks

    scores = _score(per_example_iou)
    nan_count = sum(1 for x in per_example_iou if math.isnan(x))
    if nan_count:
        _LOG.warning("bundle: %d val examples had NaN IoU; treated as worst", nan_count)

    indexed = list(enumerate(scores))
    by_desc = sorted(indexed, key=lambda kv: (-kv[1], kv[0]))   # best
    by_asc = sorted(indexed, key=lambda kv: (kv[1], kv[0]))     # worst

    best_idx = [i for i, _ in by_desc[:b]]
    worst_idx = [i for i, _ in by_asc[:w]]
    used = set(best_idx) | set(worst_idx)

    if scores:
        finite = [s for s in scores if math.isfinite(s)]
        if finite:
            median = float(np.median(finite))
        else:
            median = 0.0
    else:
        median = 0.0

    # median: closest-to-median by |score - median|, excluding used + NaN-only.
    eligible = [
        (i, s) for i, s in indexed if i not in used and math.isfinite(s)
    ]
    eligible.sort(key=lambda kv: (abs(kv[1] - median), kv[0]))
    median_idx = [i for i, _ in eligible[:m]]

    # If median is short (e.g. all-finite eligibility exhausted), fall back to
    # any remaining unused indices in ascending order.
    if len(median_idx) < m:
        remaining = [i for i, _ in indexed if i not in used and i not in median_idx]
        median_idx.extend(remaining[: m - len(median_idx)])

    return best_idx + median_idx + worst_idx


# ---------------------------------------------------------------------------
# render_overlay
# ---------------------------------------------------------------------------


_PRED_RGBA = (255, 0, 255, 96)
_GT_RGBA = (0, 255, 255, 96)


def render_overlay(
    image: Image.Image,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    *,
    caption: str,
) -> Image.Image:
    """Return a single PNG-able RGB image with prediction + GT overlaid.

    Visual contract:
      - Prediction in semi-transparent magenta (255, 0, 255, 96).
      - GT in semi-transparent cyan (0, 255, 255, 96).
      - Caption text at the bottom-left, white on a black 50%-opacity strip.
    """
    expected_hw = (image.size[1], image.size[0])  # PIL is (W, H); numpy is (H, W)
    if predicted_mask.shape != expected_hw or ground_truth_mask.shape != expected_hw:
        raise ValueError(
            f"mask shape mismatch: image={expected_hw}, "
            f"pred={predicted_mask.shape}, gt={ground_truth_mask.shape}"
        )

    base = image.convert("RGBA")
    pred_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gt_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    pred_pixels = np.zeros((expected_hw[0], expected_hw[1], 4), dtype=np.uint8)
    pred_pixels[predicted_mask.astype(bool)] = _PRED_RGBA
    pred_layer = Image.fromarray(pred_pixels, mode="RGBA")

    gt_pixels = np.zeros((expected_hw[0], expected_hw[1], 4), dtype=np.uint8)
    gt_pixels[ground_truth_mask.astype(bool)] = _GT_RGBA
    gt_layer = Image.fromarray(gt_pixels, mode="RGBA")

    composed = Image.alpha_composite(Image.alpha_composite(base, gt_layer), pred_layer)

    draw = ImageDraw.Draw(composed, mode="RGBA")
    text_w = max(60, min(base.size[0], 8 * len(caption)))
    strip_h = 16
    y0 = base.size[1] - strip_h
    draw.rectangle([(0, y0), (text_w, y0 + strip_h)], fill=(0, 0, 0, 128))
    draw.text((4, y0 + 2), caption, fill=(255, 255, 255, 255))

    return composed.convert("RGB")


# ---------------------------------------------------------------------------
# write_bundle
# ---------------------------------------------------------------------------


def _reinfer_one_example(
    model_wrapper: Any,
    val_dataset: Dataset,
    idx: int,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Re-run inference for a single example and return (image, pred, gt).

    `image`: source image as RGB PIL (already-resized, the same view fed into
    the model). `pred`: H×W bool (model's binarized mask, union over GT classes
    for that example). `gt`: H×W bool (union of all GT instance masks for the
    example). Raises if the model forward errors — caught one level up.
    """
    ex: Example = val_dataset[idx]
    # The collator/wrapper expects a batched image and one TextPrompts per image.
    classes = list(getattr(val_dataset, "class_names", []))
    if not classes:
        raise RuntimeError(f"val_dataset has no class_names; cannot prompt example {idx}")

    image_chw = ex.image  # (3, H, W) normalized — already on the model's device path
    h, w = int(image_chw.shape[-2]), int(image_chw.shape[-1])
    with torch.no_grad():
        outputs = model_wrapper(
            image_chw.unsqueeze(0),
            [TextPrompts(classes=classes)],
            box_hints=None,
        )
    # Outputs include `pred_masks` of shape (1, Q, H, W) — take union over queries
    # thresholded at 0.0 (same as Evaluator's default).
    pred_masks_logits = outputs["pred_masks"][0]
    pred_union = (pred_masks_logits > 0.0).any(dim=0).cpu().numpy().astype(bool)

    gt_union = np.zeros((h, w), dtype=bool)
    for inst in ex.instances:
        m = inst.mask.cpu().numpy().astype(bool)
        # Pad/truncate to the expected (h, w) if necessary; trust the dataset.
        gt_union |= m

    # Source image — denormalize to display-friendly RGB.
    arr = image_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(arr, mode="RGB")
    return image_pil, pred_union, gt_union


def _bracket_label(picks: list[int], composition: tuple[int, int, int]) -> list[str]:
    """Return per-index bracket label aligned with `picks`."""
    b, m, _w = composition
    labels: list[str] = []
    for i in range(len(picks)):
        if i < b:
            labels.append("best")
        elif i < b + m:
            labels.append("median")
        else:
            labels.append("worst")
    return labels


def _format_duration(start: datetime, end: datetime) -> str:
    delta = end - start
    secs = int(delta.total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _hardware_lines() -> tuple[str, float | None]:
    if not torch.cuda.is_available():
        return "(no CUDA)", None
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / (1024**3)


def write_bundle(
    ctx: BundleContext,
    metrics_report: MetricsReport,
    val_dataset: Dataset,
    model_wrapper: Any,
) -> None:
    """Write `ctx.run_dir/summary.md` and `ctx.run_dir/samples/*.png`.

    Idempotent: re-runs overwrite. Failure modes:
      - Per-sample inference raises → that PNG is skipped; WARNING logged;
        "skipped samples" note in summary.md. Bundle does not abort.
      - All other errors propagate.
    """
    samples_dir = ctx.run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale samples from prior runs.
    for stale in samples_dir.glob("*.png"):
        stale.unlink()

    mAP = float(metrics_report.overall.get("mAP", float("nan")))
    n_val = len(val_dataset)
    indices = pick_samples(ctx.per_example_iou, mAP, n_val)
    composition = _bracket(mAP)
    if n_val < 6 and n_val > 0:
        # Re-derive the prorated composition for caption purposes.
        ratios = [composition[0] / 6.0, composition[1] / 6.0, composition[2] / 6.0]
        picks = [int(r * n_val) for r in ratios]
        while sum(picks) < n_val:
            picks[2] += 1
        composition = (picks[0], picks[1], picks[2])

    edge_notes: list[str] = []
    if n_val == 0:
        edge_notes.append("empty val: no samples rendered (n_val == 0)")
    elif n_val < 6:
        edge_notes.append(
            f"capped: n_val={n_val} < 6 → rendered {len(indices)} samples per prorated composition"
        )
    if math.isnan(mAP):
        edge_notes.append("NaN mAP: classified as 'poor' bracket")
    if ctx.merged_export_error is not None:
        edge_notes.append(
            f"export-merge failed: {ctx.merged_export_error} — bundle continued"
        )

    skipped: list[tuple[int, str]] = []
    labels = _bracket_label(indices, composition)
    per_bracket_ordinal: dict[str, int] = {"best": 0, "median": 0, "worst": 0}
    sample_filenames: list[str] = []

    for picked_idx, bracket in zip(indices, labels):
        ordinal = per_bracket_ordinal[bracket]
        per_bracket_ordinal[bracket] += 1
        iou = ctx.per_example_iou[picked_idx]
        caption = f"{bracket} @ IoU={iou:.2f}"
        try:
            image, pred, gt = _reinfer_one_example(model_wrapper, val_dataset, picked_idx)
            png = render_overlay(image, pred, gt, caption=caption)
            fname = f"{ordinal}_{bracket}.png"
            png.save(samples_dir / fname)
            sample_filenames.append(fname)
        except Exception as exc:  # noqa: BLE001 — per-sample isolation
            _LOG.warning(
                "bundle: skipped sample idx=%d (%s): %s", picked_idx, bracket, exc
            )
            skipped.append((picked_idx, type(exc).__name__))

    if skipped:
        details = ", ".join(f"{i} raised {cls}" for i, cls in skipped)
        edge_notes.append(f"skipped samples: {details} — see log")

    # ---- summary.md -----------------------------------------------------
    headline = f"# {ctx.config_path.parent.name} — {mAP:.4f}"
    gpu_name, vram_gb = _hardware_lines()
    vram_line = f"- VRAM: {vram_gb:.1f} GB" if vram_gb is not None else "- VRAM: (n/a)"
    preset_line = f"- Applied: {ctx.preset_label or 'manual'}"

    adapter_path = (ctx.run_dir / "adapter").resolve()
    try:
        adapter_rel = adapter_path.relative_to(ctx.run_dir.resolve())
    except ValueError:
        adapter_rel = adapter_path

    if ctx.merged_export_error is not None:
        merged_line = f"- Merged:  FAILED — {ctx.merged_export_error} — see logs"
    elif ctx.merged_dir is None:
        merged_line = "- Merged:  skipped (cfg.export.merge=false)"
    else:
        try:
            merged_rel = ctx.merged_dir.resolve().relative_to(ctx.run_dir.resolve())
            merged_line = f"- Merged:  {merged_rel}"
        except ValueError:
            merged_line = f"- Merged:  {ctx.merged_dir}"

    samples_md = "\n".join(f"![{fn}](samples/{fn})" for fn in sample_filenames)
    edges_md = "\n".join(f"- {line}" for line in edge_notes) if edge_notes else ""

    config_rel = ctx.config_path.name

    body = (
        f"{headline}\n\n"
        f"## Run\n"
        f"- Start:  {ctx.start_ts.isoformat()}\n"
        f"- End:    {ctx.end_ts.isoformat()}\n"
        f"- Duration: {_format_duration(ctx.start_ts, ctx.end_ts)}\n\n"
        f"## Hardware\n"
        f"- GPU:  {gpu_name}\n"
        f"{vram_line}\n\n"
        f"## Preset\n"
        f"{preset_line}\n\n"
        f"## Outputs\n"
        f"- Adapter: {adapter_rel}\n"
        f"{merged_line}\n"
        f"- Config:  {config_rel}\n\n"
        f"## Samples\n"
        f"{samples_md}\n"
    )
    if edges_md:
        body += f"\n## Edge cases\n{edges_md}\n"

    (ctx.run_dir / "summary.md").write_text(body)
```

- [ ] **Step 5: Run the bundle tests and confirm pass**

```bash
uv run pytest tests/unit/runs/test_bundle.py -v
```

Expected: all green.

- [ ] **Step 6: Lint + type-check**

```bash
uv run ruff check src/esam3/runs tests/unit/runs
uv run mypy src/esam3/runs/bundle.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/esam3/runs/__init__.py src/esam3/runs/bundle.py \
        tests/unit/runs/__init__.py tests/unit/runs/test_bundle.py
git commit -m "feat(bundle): BundleContext + pick_samples + render_overlay + write_bundle"
```

---

## Task 6: `src/esam3/cli/run_cmd.py` — `esam3 run` orchestrator

**Why now:** All four pieces it composes (`run_training`, `run_eval`, `save_merged`, `write_bundle`) are in place. The CLI body is ≤ 30 lines per the cli-design boundary rule; the heavy lifting lives in a helper inside the same module.

**Spec ref:** §3.

**Files:**
- Create: `src/esam3/cli/run_cmd.py`
- Modify: `src/esam3/cli/main.py` (+1 registration line + 1 import)
- Create: `tests/integration/test_cli_run.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_cli_run.py`:

```python
"""End-to-end CLI integration tests for `esam3 run`."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from esam3.cli.main import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def _make_cfg_yaml(tmp_path: Path, *, merge: bool = False, bbox: bool = False) -> Path:
    cfg = tmp_path / "config.yaml"
    prompt = "bbox" if bbox else "text"
    cfg.write_text(
        f"""
run: {{name: t, output_dir: {tmp_path / "runs"}, seed: 0}}
data:
  format: coco
  train: {{annotations: t.json, images: t/}}
  val: {{annotations: v.json, images: v/}}
  prompt_mode: {prompt}
peft: {{method: lora}}
train: {{epochs: 1}}
export: {{merge: {str(merge).lower()}}}
"""
    )
    return cfg


def _patch_phases(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_dir: Path,
    train_raises: Exception | None = None,
    eval_raises: Exception | None = None,
    merge_raises: Exception | None = None,
    bundle_raises: Exception | None = None,
) -> dict[str, object]:
    """Patch every phase entry point. Return a dict that records calls."""
    captured: dict[str, object] = {"order": []}

    fake_result = MagicMock(
        run_dir=run_dir,
        adapter_path=run_dir / "adapter",
        merged_path=None,
        final_metrics=None,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter").mkdir(exist_ok=True)

    fake_report = MagicMock(overall={"mAP": 0.42})
    fake_per_ex = [0.1, 0.5, 0.9]

    def _train(cfg: object, *, resume_from: object = None) -> object:
        captured["order"].append("train")  # type: ignore[union-attr]
        captured["resume_from"] = resume_from
        if train_raises is not None:
            raise train_raises
        return fake_result

    def _eval(
        cfg: object,
        *,
        checkpoint: object,
        output_dir: object,
        val_dataset: object,
        model: object,
        return_per_example_iou: bool,
        **_kw: object,
    ) -> object:
        captured["order"].append("eval")  # type: ignore[union-attr]
        captured["return_per_example_iou"] = return_per_example_iou
        if eval_raises is not None:
            raise eval_raises
        return fake_report, fake_per_ex

    def _save_merged(_wrapper: object, _path: object) -> None:
        captured["order"].append("merge")  # type: ignore[union-attr]
        if merge_raises is not None:
            raise merge_raises

    def _write_bundle(ctx: object, report: object, *, val_dataset: object, model_wrapper: object) -> None:
        captured["order"].append("bundle")  # type: ignore[union-attr]
        captured["bundle_ctx"] = ctx
        if bundle_raises is not None:
            raise bundle_raises

    monkeypatch.setattr("esam3.cli.run_cmd.run_training", _train)
    monkeypatch.setattr("esam3.cli.run_cmd.run_eval", _eval)
    monkeypatch.setattr("esam3.cli.run_cmd.save_merged", _save_merged)
    monkeypatch.setattr("esam3.cli.run_cmd.write_bundle", _write_bundle)
    monkeypatch.setattr("esam3.cli.run_cmd.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("esam3.cli.run_cmd.load_adapter", lambda *_a, **_kw: None)
    # Build a stub val_dataset.
    fake_ds = MagicMock(__len__=lambda self: 3, class_names=["a"])

    def _build_val(_cfg: object) -> object:
        captured["order"].append("build_val")  # type: ignore[union-attr]
        return fake_ds

    monkeypatch.setattr("esam3.cli.run_cmd._build_val_dataset", _build_val)
    return captured


def test_run_help_exits_zero() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Train + eval" in _plain(result.output)


def test_run_full_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    # Every phase called in order.
    order = captured["order"]
    assert order[0] == "train"
    # build_val may run before or after train depending on impl, but bundle is last.
    assert order[-1] == "bundle"
    assert "eval" in order
    assert captured["return_per_example_iou"] is True
    ctx = captured["bundle_ctx"]
    assert ctx.merged_dir is None
    assert ctx.merged_export_error is None
    assert ctx.per_example_iou == [0.1, 0.5, 0.9]


def test_run_train_failure_skips_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch, run_dir=tmp_path / "runs" / "r",
        train_raises=RuntimeError("kaboom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "kaboom" in _plain(result.output) or "kaboom" in (result.stderr or "")
    order = captured["order"]
    assert "eval" not in order
    assert "merge" not in order
    assert "bundle" not in order


def test_run_eval_failure_skips_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch, run_dir=tmp_path / "runs" / "r",
        eval_raises=RuntimeError("eval-boom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "merge" not in captured["order"]
    assert "bundle" not in captured["order"]


def test_run_merge_failure_still_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch, run_dir=tmp_path / "runs" / "r",
        merge_raises=ValueError("rank mismatch"),
    )
    cfg = _make_cfg_yaml(tmp_path, merge=True)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "merge" in captured["order"]
    assert "bundle" in captured["order"]
    ctx = captured["bundle_ctx"]
    assert ctx.merged_dir is None
    assert "rank mismatch" in (ctx.merged_export_error or "")


def test_run_bundle_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "runs" / "r"
    captured = _patch_phases(
        monkeypatch, run_dir=run_dir,
        bundle_raises=RuntimeError("bundle-boom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    # run_dir and adapter remain on disk.
    assert run_dir.exists()
    assert (run_dir / "adapter").exists()


def test_run_rejects_bbox_prompt_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path, bbox=True)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 2
    assert "train" not in captured["order"]
    assert "eval" not in captured["order"]
    assert "bbox" in _plain(result.output).lower()


def test_run_passes_preset_label_env_var_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESAM3_PRESET_LABEL", "auto: 12-24GB tier")
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset_label == "auto: 12-24GB tier"


def test_run_preset_label_absent_yields_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ESAM3_PRESET_LABEL", raising=False)
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset_label is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
uv run pytest tests/integration/test_cli_run.py -v
```

Expected: every test fails at the import step (`No module named 'esam3.cli.run_cmd'`) or `'run' is not a command on this Typer app`.

- [ ] **Step 3: Create `src/esam3/cli/run_cmd.py`**

```python
"""`esam3 run` — train + eval + (optional) export + bundle in one shot.

Body is ≤ 30 lines per the cli-design boundary rule. Phase composition and
context assembly live in `_orchestrate` so the Typer command stays a thin shell.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §3.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import typer
from rich import print as rprint

from esam3._registry import lookup
from esam3.cli._logging import configure_logging
from esam3.config.loader import load_config
from esam3.config.schema import TrainConfig
from esam3.data.base import Dataset
from esam3.eval.runner import run_eval
from esam3.models.sam3 import load_sam31
from esam3.runs.bundle import BundleContext, write_bundle
from esam3.train.checkpoint import load_adapter, save_merged
from esam3.train.runner import run_training

_LOG = logging.getLogger(__name__)


def _build_val_dataset(cfg: TrainConfig) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(cfg.data.model_dump(), model_name=cfg.model.name, pipeline="eval"))


def _orchestrate(cfg: TrainConfig, resume: Path | None) -> int:
    start_ts = datetime.now(UTC)

    # Phase: train.
    try:
        train_result = run_training(cfg, resume_from=resume)
    except Exception as exc:  # noqa: BLE001 — orchestrator boundary
        rprint(f"[red]train failed[/red] {exc}")
        raise typer.Exit(code=1) from exc
    run_dir = train_result.run_dir
    adapter_path = train_result.adapter_path

    # Build val + load wrapper exactly once for the rest of the run.
    val_dataset = _build_val_dataset(cfg)
    wrapper: Any = load_sam31(cfg.model)
    load_adapter(wrapper, adapter_path)

    # Phase: eval.
    try:
        report, per_example_iou = cast(
            tuple[Any, list[float]],
            run_eval(
                cfg,
                checkpoint=adapter_path,
                output_dir=run_dir,
                val_dataset=val_dataset,
                model=wrapper,
                return_per_example_iou=True,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        rprint(f"[red]eval failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc

    end_ts = datetime.now(UTC)

    # Phase: export-merge (conditional, soft-fail).
    merged_dir: Path | None = None
    merged_export_error: str | None = None
    if cfg.export.merge:
        target = run_dir / "merged"
        try:
            save_merged(wrapper, target)
            merged_dir = target
        except Exception as exc:  # noqa: BLE001 — soft-fail
            _LOG.warning("export-merge failed: %s", exc)
            merged_export_error = str(exc)

    # Phase: bundle.
    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset_label=os.environ.get("ESAM3_PRESET_LABEL"),
        per_example_iou=per_example_iou,
        merged_dir=merged_dir,
        merged_export_error=merged_export_error,
    )
    try:
        write_bundle(ctx, report, val_dataset=val_dataset, model_wrapper=wrapper)
    except Exception as exc:  # noqa: BLE001
        rprint(f"[red]bundle failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc

    rprint(
        f"[green]done[/green] run_dir={run_dir} adapter={adapter_path} "
        f"merged={(merged_dir or merged_export_error or 'skipped')} "
        f"summary={run_dir / 'summary.md'} mAP={report.overall.get('mAP', float('nan')):.4f}"
    )
    return 0


def run(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    resume: Path | None = typer.Option(None, "--resume", help="Path to resume checkpoint."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging."),
) -> None:
    """Train + eval + (optional) export + bundle, in one shot."""
    configure_logging(verbose)
    cfg = load_config(config)
    if cfg.data.prompt_mode == "bbox":
        raise typer.BadParameter(
            "prompt_mode='bbox' is not supported for training in v0; "
            "see docs/superpowers/specs/2026-05-15-esam3-architecture-design.md §1.",
            param_hint="--config",
        )
    _orchestrate(cfg, resume)
```

- [ ] **Step 4: Register `run` in `cli/main.py`**

Open `src/esam3/cli/main.py` and add the import + registration:

```python
from esam3.cli import (
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    run_cmd,                # NEW
    train_cmd,
)

# … after the existing app.command(…) lines:

app.command("run", help="Train + eval + (optional) export + bundle in one shot.")(run_cmd.run)
```

- [ ] **Step 5: Run the integration tests and confirm pass**

```bash
uv run pytest tests/integration/test_cli_run.py -v
```

Expected: all green.

- [ ] **Step 6: Run the entire suite to confirm no regression**

```bash
uv run pytest -x -q
uv run ruff check src tests
uv run mypy src/esam3
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/esam3/cli/run_cmd.py src/esam3/cli/main.py tests/integration/test_cli_run.py
git commit -m "feat(cli): esam3 run — train+eval+merge+bundle orchestrator"
```

---

## Task 7: `notebooks/esam3_train.ipynb` — user notebook

**Why now:** `esam3 run` + helpers are available end-to-end, so the notebook is the thin glue the spec promises. The notebook is **separate from** `colab_gpu_tests.ipynb`; do not edit that file.

**Spec ref:** §7.

**Files:**
- Create: `notebooks/esam3_train.ipynb`

The notebook has four cells. There is no automated execution test (spec §8.4); helpers are unit-tested. Manual dry-runs happen in Task 9.

- [ ] **Step 1: Author the notebook**

Create `notebooks/esam3_train.ipynb` as a single JSON file with four code cells in order. The cell sources MUST be exactly the following — the dry-runs will run them as-is.

**Cell 1 (SETUP):**

```python
# SETUP — install esam3, detect environment, resolve HF token (or skip if local).
import os, sys
import subprocess
from pathlib import Path

import torch

# Install the repo from GitHub with the qlora + tensorboard extras.
# (Docker image is deferred — see GitHub issue #34.)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/NguyenJus/Efficient-SAM3-Finetuning.git#egg=efficient-sam3-finetuning[qlora,tensorboard]",
])

from esam3.notebook_helpers import detect_env, check_local_checkpoint, resolve_hf_token

env = detect_env()
assert torch.cuda.is_available(), (
    "No CUDA detected. In Colab: Runtime → Change runtime type → GPU. "
    "On RunPod: deploy a GPU pod."
)
local_present = check_local_checkpoint(Path("models/sam3.1"), "sam3.1_multiplex.pt")
token = resolve_hf_token(env, local_present)
if token is not None:
    os.environ["HF_TOKEN"] = token
print(f"mode: env={env}, local_checkpoint={local_present}, "
      f"hf_auth={'skipped' if token is None else 'enabled'}")
```

**Cell 2 (FORM):**

```python
# FORM — three values: dataset path, format, run name. v0 is text-only.
dataset_path: str = ""        #@param {type:"string"}
data_format: str = "coco"     #@param ["coco", "hf"]
run_name: str = "my-run"      #@param {type:"string"}

assert dataset_path, "dataset_path is required. For coco: path to a folder with train/ and val/. For hf: an HF dataset id."
```

**Cell 3 (GENERATE):**

```python
# GENERATE — derive preset, load template, deep-merge user inputs, write config.yaml, run esam3.
import importlib.resources
import subprocess
import yaml

from esam3.presets import pick_preset, preset_label

patch = pick_preset()
template_name = "coco_text_qlora.yaml" if patch["peft"]["method"] == "qlora" else "coco_text_lora.yaml"
template_text = (importlib.resources.files("esam3.cli.templates") / template_name).read_text()
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

# COCO: dataset_path is a directory containing train/ and val/ subdirectories.
# HF:   dataset_path is the HF dataset id, passed through as-is.
if data_format == "coco":
    _PREF = ("_annotations.coco.json", "instances.json", "annotations.json")

    def _resolve_split(split_dir: Path) -> str:
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"COCO split directory not found: {split_dir}. Expected layout: "
                f"<dataset_path>/train/ and <dataset_path>/val/."
            )
        for pref in _PREF:
            cand = split_dir / pref
            if cand.is_file():
                return str(cand)
        json_candidates = sorted(p for p in split_dir.glob("*.json"))
        if json_candidates:
            return str(json_candidates[0])
        raise FileNotFoundError(
            f"No COCO annotation JSON found under {split_dir}. "
            f"Tried (in order): {_PREF + ('*.json (lex sort)',)}."
        )

    dataset_root = Path(dataset_path)
    for split in ("train", "val"):
        split_dir = dataset_root / split
        config["data"][split]["annotations"] = _resolve_split(split_dir)
        config["data"][split]["images"] = str(split_dir)
else:  # hf
    # HF adapter receives the dataset id directly. The template's data.train/val
    # paths are placeholders the adapter ignores under format='hf'.
    config.setdefault("data", {})["hf"] = {"id": dataset_path}

with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

os.environ["ESAM3_PRESET_LABEL"] = preset_label()

proc = subprocess.Popen(
    ["esam3", "run", "--config", "config.yaml"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    bufsize=1, text=True,
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

**Cell 4 (RESULTS):**

```python
# RESULTS — render summary.md inline and display sample overlays.
import shutil
from pathlib import Path

from IPython.display import Image, Markdown, display

latest = max(Path("runs").iterdir(), key=lambda p: p.stat().st_mtime)
display(Markdown((latest / "summary.md").read_text()))
for png in sorted((latest / "samples").glob("*.png")):
    display(Image(filename=str(png)))

# Download:
if detect_env() == "colab":
    archive = Path(shutil.make_archive(str(latest), "zip", root_dir=latest.parent, base_dir=latest.name))
    print(f"To download: from google.colab import files; files.download({str(archive)!r})")
else:
    print("RunPod: copy the run with")
    print(f"  scp -P <pod_port> root@<pod_host>:/workspace/{latest}.zip ./")
    print(f"(or zip first: shutil.make_archive('{latest}', 'zip'))")
```

- [ ] **Step 2: Verify the notebook is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/esam3_train.ipynb'))"
```

Expected: no error.

- [ ] **Step 3: Verify the notebook does NOT clobber colab_gpu_tests.ipynb**

```bash
git status notebooks/
```

Expected: only `notebooks/esam3_train.ipynb` is new; `notebooks/colab_gpu_tests.ipynb` is untouched.

- [ ] **Step 4: Commit**

```bash
git add notebooks/esam3_train.ipynb
git commit -m "feat(notebook): user-facing esam3_train.ipynb"
```

---

## Task 8: README restructure + `cloud/runpod/README.md`

**Why now:** the user-facing surface (CLI + notebook) is final; docs cite final paths.

**Spec ref:** §8.

**Files:**
- Modify: `README.md` (restructure in place; no content removed)
- Create: `cloud/runpod/README.md`

- [ ] **Step 1: Restructure `README.md`**

Open `README.md`. Today (after the cli plan landed) it looks like:

```
# efficient-sam3-finetuning
<blurb + status callout>

## Quickstart
## CLI
## What's supported in v0
## v0 Training scope
## Repo layout
## Development
### GPU test automation
## License
```

Transform it into:

```
# efficient-sam3-finetuning

<blurb + status callout>            ← unchanged

## Beginner — train in 3 clicks

(Plain-English, no jargon. Bullet list of prerequisites.)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenJus/Efficient-SAM3-Finetuning/blob/main/notebooks/esam3_train.ipynb)

1. Open the notebook in Colab via the badge above.
2. In Colab Secrets, set `HF_TOKEN` (Hugging Face token with read access
   to gated `facebook/sam3.1`). If you've already downloaded the
   checkpoint to `models/sam3.1/sam3.1_multiplex.pt` (e.g. on a RunPod
   network volume), skip this step.
3. Either upload a dataset (a folder with `train/` and `val/` COCO
   subdirectories) or paste a HF dataset id, then click Runtime → Run All.

When the run finishes, scroll to the bottom of the notebook for a
summary, sample mask overlays, and a one-line download command.

For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).

## Advanced

### Quickstart
   <unchanged content>

### CLI

| Command | Status |
|---|---|
| `esam3 run --config CONFIG [--resume PATH] [-v]` | Functional |
| `esam3 train --config CONFIG [--override key=val]... [--resume PATH] [-v]` | Functional |
| `esam3 eval --config CONFIG --checkpoint PATH [--split val\|test] [--output PATH] [--save-predictions]` | Functional (LoRA adapters only) |
| `esam3 export --checkpoint PATH [--merge] [--output PATH] [--config PATH]` | Functional |
| `esam3 init [--template coco-text-lora\|coco-text-qlora] [--output PATH] [--force]` | Functional |
| `esam3 doctor [--weights-path PATH] [--json]` | Functional |

(`esam3 run` is "train + eval + (optional) export + bundle in one shot"; the others are unchanged.)

   <"deferred templates" line unchanged>

### What's supported in v0
   <unchanged>

### v0 Training scope
   <unchanged>

### Repo layout
   <unchanged>

### Development
   <unchanged>

### GPU test automation
   <unchanged — badge still points at colab_gpu_tests.ipynb>

## License
   <unchanged>
```

**Constraints enforced by these edits:**

- The Beginner badge points at `esam3_train.ipynb` — the new user notebook.
- The Advanced section's `### GPU test automation` badge keeps pointing at `colab_gpu_tests.ipynb` — the dev smoke notebook.
- No content is removed; existing section bodies move from H2 to H3 under `## Advanced`. GitHub renders the anchors (`#quickstart`, `#cli`, etc.) identically regardless of nesting.
- The CLI table gains one new row (`esam3 run`) and the others stay verbatim.

- [ ] **Step 2: Create `cloud/runpod/README.md`**

```bash
mkdir -p cloud/runpod
```

Create `cloud/runpod/README.md` with these sections in this order (use this exact outline; expand each into 2–4 sentences of plain English):

```markdown
# Running esam3 on RunPod

A step-by-step guide for non-technical users. If you're on Colab, use the
notebook badge in the [main README](../../README.md) — this file is the
RunPod equivalent.

## 1. Sign up
- Go to [runpod.io](https://runpod.io). Create an account.
- Pay-as-you-go is fine; spot pricing is cheaper but can be interrupted.

## 2. Pick a GPU
- **A40 is the recommended entry tier** — 48 GB VRAM lands in the LoRA preset,
  good $/VRAM ratio.
- L4, RTX 4090, and A100 also work. Anything with ≥ 12 GB VRAM can run the
  QLoRA preset.

## 3. Deploy a stock RunPod PyTorch template
- We deliberately do **not** publish or maintain a custom RunPod image —
  see [issue #34](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/34).
- Use the stock **"RunPod PyTorch 2.x"** template from the Templates page.
- Templates → Deploy → pick the GPU from step 2 → Deploy.

## 4. Set `HF_TOKEN`
- Pod → Edit → Environment Variables → add `HF_TOKEN` (your Hugging Face
  read-access token for gated `facebook/sam3.1`).
- **Or** skip this if you've mounted a network volume that contains
  `models/sam3.1/sam3.1_multiplex.pt` — `esam3` will detect the local file
  and skip HF auth.

## 5. Open Jupyter Lab
- Click the pod's **Connect** button → Jupyter Lab.

## 6. Upload `notebooks/esam3_train.ipynb`
- Two options:
  - Drag and drop the notebook file into the Jupyter file browser.
  - In Jupyter, File → Open from URL → paste the raw GitHub URL for
    `notebooks/esam3_train.ipynb`.

## 7. Click Run All
- Same beginner flow as Colab — fill in dataset path, format, and run name
  in the FORM cell, then Runtime → Run All.

## Data upload

- **Small dataset (≤ 1 GB):** drag-and-drop into the Jupyter file browser.
- **Large dataset:** RunPod network volume — one-time upload, persists across
  pods.
- **HF dataset:** easiest — paste the dataset id into the FORM cell; no
  upload needed.

## What you get back

Every run writes a `runs/<id>/` directory with:
- `summary.md` — headline metric, run timing, hardware, sample overlays.
- `samples/*.png` — up to 6 best / median / worst predictions.
- `adapter/` — the LoRA / QLoRA adapter weights.
- `metrics.json` — raw eval numbers.

Download with `scp` or zip + download from Jupyter.
```

- [ ] **Step 3: Sanity-check the docs render**

```bash
ls cloud/runpod/README.md
grep -c "## " cloud/runpod/README.md
grep -F "esam3 run" README.md
grep -F "esam3_train.ipynb" README.md
grep -F "colab_gpu_tests.ipynb" README.md
```

Expected: `cloud/runpod/README.md` exists with ≥ 8 H2-headed sections; the main README references both notebooks (Beginner → user notebook; Advanced GPU test automation → dev smoke notebook) and lists `esam3 run` in the CLI table.

- [ ] **Step 4: Commit**

```bash
git add README.md cloud/runpod/README.md
git commit -m "docs: README Beginner/Advanced split + RunPod walkthrough"
```

---

## Task 9: GPU smoke test + manual dry-run checklists

**Why now:** all source + docs are final. The GPU test gates the run end-to-end against the existing smoke fixture; the manual dry-runs are the spec's mandatory gate before PR-ready.

**Spec ref:** §8.3, §8.4.

**Files:**
- Create: `tests/gpu/test_run_end_to_end_gpu.py`

- [ ] **Step 1: Write the GPU smoke test**

Create `tests/gpu/test_run_end_to_end_gpu.py`:

```python
"""End-to-end `esam3 run` GPU smoke test.

Drives `esam3 run` (Typer entry) against the same `configs/examples/gpu_smoke_lora.yaml`
fixture used by the other GPU smoke tests. Asserts on the artefacts on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from esam3.cli.main import app

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_lora.yaml"


def test_run_end_to_end_writes_bundle(tmp_path: Path, tiny_coco_dir: Path) -> None:
    from esam3.config.loader import load_config

    # Materialize a copy of the smoke config pointing at tmp_path output, tiny_coco data.
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
    cfg_path = tmp_path / "smoke.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    result = CliRunner().invoke(app, ["run", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    runs = sorted(tmp_path.glob("gpu-smoke-lora-*"))
    assert runs, f"no run dir under {tmp_path}"
    run_dir = runs[-1]

    # Adapter present and non-empty.
    adapter_files = list((run_dir / "adapter").iterdir())
    assert adapter_files, f"adapter dir empty: {run_dir / 'adapter'}"

    # metrics.json parses; has overall.mAP numeric.
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "overall" in metrics
    assert isinstance(metrics["overall"].get("mAP"), (int, float))

    # summary.md exists and mentions mAP.
    summary = (run_dir / "summary.md").read_text()
    assert "mAP" in summary or "0." in summary  # headline embeds the float

    # samples/ has ≤ 6 PNGs.
    pngs = sorted((run_dir / "samples").glob("*.png"))
    assert 0 <= len(pngs) <= 6

    # cfg.export.merge=False in the smoke YAML → no merged/ dir.
    assert not (run_dir / "merged").exists()
```

- [ ] **Step 2: Collect-only verify (CPU box)**

```bash
uv run pytest tests/gpu/test_run_end_to_end_gpu.py --collect-only -q
```

Expected: 1 test collected. (It will skip at runtime on a CPU-only box via the `requires_compatible_gpu` / `requires_checkpoint` markers.)

- [ ] **Step 3: (Optional) Append the new GPU test to `colab_gpu_tests.ipynb` Run All**

Spec §8.4 calls this "optional addition" — open `notebooks/colab_gpu_tests.ipynb` and, in the cell that already invokes `pytest -m gpu …`, add `tests/gpu/test_run_end_to_end_gpu.py` to the explicit file list (or leave the `-m gpu` selector — the new test is already gated by the marker, so it picks up automatically).

If the existing cell uses `-m gpu` without a file list, **no edit is needed** — the new test will run as part of the next Run All by virtue of the marker. Skip the rest of this step.

If the existing cell uses an explicit file list, append the new path. Keep all other content of `colab_gpu_tests.ipynb` untouched.

- [ ] **Step 4: Commit**

```bash
git add tests/gpu/test_run_end_to_end_gpu.py
# also stage colab_gpu_tests.ipynb only if step 3 modified it
git commit -m "test(gpu): end-to-end esam3 run smoke against gpu_smoke_lora.yaml"
```

- [ ] **Step 5: Manual Colab dry-run (GATE — required before PR-ready)**

This is a manual gate; do not mark the PR ready without it.

Checklist:

- [ ] In a fresh Colab session (Runtime → Change runtime type → **T4 GPU**), open the new notebook via the README badge.
- [ ] In Colab Secrets, set `HF_TOKEN`.
- [ ] In the FORM cell, supply a small COCO-format dataset (a folder with `train/` and `val/` subdirectories, each containing one of the recognized annotation JSON names). A 10-image fixture is sufficient.
- [ ] Runtime → Run All.
- [ ] **Verify:**
  - SETUP cell prints `mode: env=colab, local_checkpoint=False, hf_auth=enabled`.
  - FORM widgets render (string for `dataset_path`, dropdown for `data_format`).
  - GENERATE writes `config.yaml`; the file contains `peft.method: qlora` (T4 is 15 GB → 12-24GB tier).
  - Subprocess output streams live into the cell (not buffered until end).
  - RESULTS renders `summary.md` inline (markdown + headline mAP) and at least one PNG.
  - The "To download: from google.colab import files;…" line prints.

- [ ] **Step 6: Manual RunPod dry-run (GATE — required before PR-ready)**

Checklist:

- [ ] Deploy a fresh A40 pod with the stock **"RunPod PyTorch 2.x"** template, following `cloud/runpod/README.md`.
- [ ] Set `HF_TOKEN` in the pod's Environment Variables.
- [ ] Open Jupyter Lab; upload `notebooks/esam3_train.ipynb` (drag-and-drop or Open from URL).
- [ ] In the FORM cell, supply the same fixture dataset (or an HF dataset id).
- [ ] Runtime → Run All.
- [ ] **Verify:**
  - SETUP prints `mode: env=runpod, …`.
  - GENERATE picks the `24-48GB` tier (A40 → 48 GB falls into the LoRA tier per spec §4.3).
  - The bundle's `samples/` directory has non-trivial overlays (not blank).
  - `scp -P … root@<pod_host>:/workspace/runs/<id>.zip ./` (after `shutil.make_archive`) works.

Once both checklists are complete, mark the PR ready.

---

## Self-Review Notes

- **Spec §3 (esam3 run):** Task 6 (`run_cmd.py` body ≤ 30 LOC; `_orchestrate` helper carries the phase composition). Pre-flight bbox check is in the Typer entry. Partial-output preservation is enforced by never deleting `run_dir`.
- **Spec §4 (presets):** Task 1. Tier table bucket boundaries (`<12`, `12-24`, `24-48`, `≥48`) are parametrized in the unit test, including the inclusive-low / exclusive-high boundary cases (`12.0`, `24.0`, `48.0`).
- **Spec §5 (notebook_helpers):** Task 2. The three error-string arms (`Colab Secrets`, `Environment Variables`, `shell environment`) are pinned by substring match.
- **Spec §6 (bundle):** Task 5. `BundleContext` is the sole assembly point. `pick_samples` is pure (n=0, n<6, NaN-mAP, NaN-IoU, all-zero, identical-IoU all covered). `render_overlay` is shape-checked. `write_bundle` per-sample failure isolation matches spec §9.
- **Spec §6.3 / §3.4 (eval extensions):** Tasks 3 + 4. Both are additive; default kwargs preserve the previous return types for every existing caller.
- **Spec §7 (notebook):** Task 7. Four cells in order, exact text. No `prompt_mode` widget.
- **Spec §8 (docs):** Task 8. Beginner section on top with the user-notebook badge; Advanced wraps every existing section unchanged; one new CLI row.
- **Spec §8.3 / §8.4 (testing):** Task 9. GPU smoke + two manual dry-runs as explicit gate items.
- **Phase ordering:** Tasks 1–2 (pure helpers) → 3–4 (eval extensions) → 5 (bundle, depends on 3) → 6 (CLI, depends on 5 + 4 + train.runner) → 7 (notebook, depends on 6) → 8 (docs) → 9 (smoke + dry-runs). At each task boundary the repo is in a working state with passing tests.

---

## Out of Scope (follow-ups, NOT in this plan)

These are tracked as GitHub issues and explicitly excluded from this plan's scope. They do not gate the PR.

| Issue | Title | Why deferred |
|---|---|---|
| [**#33**](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/33) | Folder-format dataset adapter | Lowest-friction format for laypeople, but additive to the dataset registry; slots in without touching this spec's scope. |
| [**#34**](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/34) | Publish Docker image to GHCR | Would replace the notebook SETUP cell's `pip install git+…` with `docker pull`. Doable but adds release surface. |
| [**#35**](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/35) | Investigate AWS SageMaker + Lambda Labs cloud targets | Stronger PII / security posture. Future work once the Colab/RunPod flow is battle-tested. |
| [**#36**](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/36) | Algorithmically derive preset from VRAM | Replaces the hand-tuned table in `presets.py` §4.3. Needs its own brainstorm. |
| [**#37**](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/37) | Audit and minimize the `pytest -m gpu` surface | This plan adds one more GPU-marker test (Task 9 Step 1). #37 will triage the overall set later. |

Also explicitly out of scope (no issue needed):

- New tracker backends or hosted dashboards.
- Sweep / hyperparameter-search integration.
- Any change to `notebooks/colab_gpu_tests.ipynb` other than the optional Run-All addition in Task 9 Step 3.
- Any change to `esam3 train` / `eval` / `export` / `init` / `doctor` behavior. The eval extensions in Tasks 3–4 are strictly additive: every existing caller continues to use default kwargs and receives the same return type as before.
