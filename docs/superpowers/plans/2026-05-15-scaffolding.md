# esam3 Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a repo where `esam3` imports cleanly, the CLI runs `--help`, CPU tests pass, CI is green, and every public surface from the design spec exists as a typed stub or signature — with no training/eval/data-loading logic.

**Architecture:** `src/`-layout Python 3.13 package managed by `uv`. Implemented now: pydantic config schema + loader, the plugin registry, all Protocol/dataclass type contracts, the Typer CLI shell, the `noop` tracker. Stubbed (NotImplementedError): every other module. Tooling: ruff (lint + format), mypy --strict, pytest, pre-commit, GitHub Actions CI.

**Tech Stack:** Python 3.13, uv, pydantic v2, Typer, ruff, mypy, pytest, GitHub Actions. Declared (not yet imported in tests): torch, transformers, peft, datasets, pycocotools, numpy, rich. Optional extras: `[wandb]`, `[qlora]` (bitsandbytes), `[tensorboard]`, `[dev]`.

**Exit criteria (Task 19):** `uv sync --all-extras --dev && uv run ruff check && uv run ruff format --check && uv run mypy src/esam3 && uv run pytest && uv run esam3 --help` — all pass cleanly.

---

## File Structure (created or modified by this plan)

```
.
├── .github/workflows/ci.yml                    [create]
├── .gitignore                                  [modify]
├── .pre-commit-config.yaml                     [create]
├── ARCHITECTURE.md                             [create]
├── LICENSE                                     [replace if exists]
├── README.md                                   [rewrite]
├── configs/examples/
│   ├── coco_bbox_qlora.yaml                    [create]
│   └── coco_text_lora.yaml                     [create]
├── logs/
│   ├── log.md                                  [create]
│   └── TODO.md                                 [create]
├── main.py                                     [delete]
├── pyproject.toml                              [rewrite]
├── src/esam3/
│   ├── __init__.py                             [create]
│   ├── _registry.py                            [create — implemented]
│   ├── cli/
│   │   ├── __init__.py                         [create]
│   │   ├── main.py                             [create — Typer app]
│   │   ├── train_cmd.py                        [create — stub command]
│   │   ├── eval_cmd.py                         [create — stub command]
│   │   ├── export_cmd.py                       [create — stub command]
│   │   ├── init_cmd.py                         [create — stub command]
│   │   └── doctor_cmd.py                       [create — stub command]
│   ├── config/
│   │   ├── __init__.py                         [create]
│   │   ├── schema.py                           [create — implemented pydantic]
│   │   └── loader.py                           [create — implemented]
│   ├── data/
│   │   ├── __init__.py                         [create]
│   │   ├── base.py                             [create — implemented protocols]
│   │   ├── coco.py                             [create — stub]
│   │   ├── hf.py                               [create — stub]
│   │   ├── transforms.py                       [create — stub]
│   │   └── collate.py                          [create — stub]
│   ├── eval/
│   │   ├── __init__.py                         [create]
│   │   ├── evaluator.py                        [create — stub]
│   │   └── metrics.py                          [create — MetricsReport implemented, fns stubbed]
│   ├── models/
│   │   ├── __init__.py                         [create]
│   │   ├── losses.py                           [create — stub]
│   │   └── sam3.py                             [create — stub]
│   ├── peft_adapters/
│   │   ├── __init__.py                         [create]
│   │   ├── lora.py                             [create — stub]
│   │   └── qlora.py                            [create — stub]
│   ├── tracking/
│   │   ├── __init__.py                         [create]
│   │   ├── base.py                             [create — Protocol implemented]
│   │   ├── noop.py                             [create — implemented]
│   │   ├── tensorboard.py                      [create — stub]
│   │   └── wandb.py                            [create — stub]
│   └── train/
│       ├── __init__.py                         [create]
│       ├── checkpoint.py                       [create — stub]
│       ├── loop.py                             [create — stub]
│       └── trainer.py                          [create — RunResult implemented, Trainer stub]
└── tests/
    ├── __init__.py                             [create]
    ├── conftest.py                             [create]
    ├── fixtures/
    │   ├── __init__.py                         [create]
    │   ├── make_tiny_coco.py                   [create — one-shot fixture generator]
    │   ├── tiny_coco/
    │   │   ├── annotations.json                [create]
    │   │   └── images/
    │   │       ├── img_000001.png              [create — generated, committed]
    │   │       └── img_000002.png              [create — generated, committed]
    │   └── tiny_sam3_stub.py                   [create]
    └── unit/
        ├── __init__.py                         [create]
        ├── test_cli.py                         [create]
        ├── test_config_loader.py               [create]
        ├── test_config_schema.py               [create]
        ├── test_fixtures.py                    [create]
        ├── test_imports.py                     [create]
        ├── test_registry.py                    [create]
        ├── test_stubs_raise.py                 [create]
        └── test_tracking_noop.py               [create]
```

---

## Task 1: Bootstrap project metadata & tooling config

**Files:**
- Modify: `pyproject.toml`
- Delete: `main.py`
- Modify: `.gitignore`

- [ ] **Step 1: Replace `pyproject.toml`**

Write the file at `/home/justin/projects/Efficient-SAM3-Finetuning/pyproject.toml`:

```toml
[project]
name = "efficient-sam3-finetuning"
version = "0.0.1"
description = "Parameter-efficient finetuning of SAM3.1 on niche image datasets."
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.13"
authors = [{ name = "Justin Nguyen" }]
dependencies = [
  "torch>=2.4",
  "transformers>=4.50",
  "peft>=0.13",
  "datasets>=3.0",
  "pydantic>=2.7",
  "typer>=0.12",
  "pyyaml>=6.0",
  "pycocotools>=2.0",
  "numpy>=1.26",
  "rich>=13",
]

[project.optional-dependencies]
wandb = ["wandb>=0.18"]
qlora = ["bitsandbytes>=0.43"]
tensorboard = ["tensorboard>=2.18"]

[dependency-groups]
dev = [
  "ruff>=0.7",
  "mypy>=1.13",
  "pytest>=8",
  "pytest-cov>=5",
  "pre-commit>=4",
  "pillow>=10",
  "types-PyYAML>=6",
]

[project.scripts]
esam3 = "esam3.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/esam3"]

[tool.ruff]
line-length = 100
target-version = "py313"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = []

[tool.mypy]
python_version = "3.13"
strict = true
files = ["src/esam3"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["pycocotools.*", "bitsandbytes.*", "wandb.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "integration: end-to-end tests using the stub model (CPU)",
  "gpu: tests requiring a CUDA device with real SAM3.1 weights",
]
addopts = "-ra --strict-markers"

[tool.coverage.run]
source = ["src/esam3"]
branch = true

[tool.coverage.report]
exclude_lines = [
  "pragma: no cover",
  "raise NotImplementedError",
  "if TYPE_CHECKING:",
]
```

- [ ] **Step 2: Delete `main.py`**

Run: `rm /home/justin/projects/Efficient-SAM3-Finetuning/main.py`

- [ ] **Step 3: Replace `.gitignore` contents**

Write the file at `/home/justin/projects/Efficient-SAM3-Finetuning/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.eggs/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/

# Virtualenv
.venv/
venv/

# Editor
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Runtime artifacts
runs/
checkpoints/
wandb/
*.ckpt
*.safetensors

# Local logs not tracked
*.log
```

- [ ] **Step 4: Sync deps and verify**

Run: `uv sync --all-extras --group dev`
Expected: completes successfully, creates/updates `.venv` and `uv.lock`.

Run: `uv run python -c "import sys; print(sys.version)"`
Expected: a Python 3.13.x version string.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore
git rm main.py
git commit -m "chore: bootstrap project metadata, tooling, deps"
```

---

## Task 2: Empty package skeleton

**Files:**
- Create: `src/esam3/__init__.py`
- Create: `src/esam3/{config,data,models,peft_adapters,train,eval,tracking,cli}/__init__.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`, `tests/fixtures/__init__.py`

- [ ] **Step 1: Write a failing import test**

Create `tests/unit/test_imports.py`:

```python
"""Smoke test that every public esam3 submodule imports without raising."""

from __future__ import annotations

import importlib

MODULES = [
    "esam3",
    "esam3._registry",
    "esam3.config",
    "esam3.config.schema",
    "esam3.config.loader",
    "esam3.data",
    "esam3.data.base",
    "esam3.data.coco",
    "esam3.data.hf",
    "esam3.data.transforms",
    "esam3.data.collate",
    "esam3.models",
    "esam3.models.sam3",
    "esam3.models.losses",
    "esam3.peft_adapters",
    "esam3.peft_adapters.lora",
    "esam3.peft_adapters.qlora",
    "esam3.train",
    "esam3.train.trainer",
    "esam3.train.loop",
    "esam3.train.checkpoint",
    "esam3.eval",
    "esam3.eval.metrics",
    "esam3.eval.evaluator",
    "esam3.tracking",
    "esam3.tracking.base",
    "esam3.tracking.tensorboard",
    "esam3.tracking.wandb",
    "esam3.tracking.noop",
    "esam3.cli",
    "esam3.cli.main",
    "esam3.cli.train_cmd",
    "esam3.cli.eval_cmd",
    "esam3.cli.export_cmd",
    "esam3.cli.init_cmd",
    "esam3.cli.doctor_cmd",
]


def test_all_modules_import() -> None:
    for name in MODULES:
        importlib.import_module(name)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'esam3'`).

- [ ] **Step 3: Create empty `__init__.py` files**

Create each of the following files with the exact content `"""esam3 package."""\n` (for `src/esam3/__init__.py`) or an empty body for everything else:

- `src/esam3/__init__.py`:
  ```python
  """esam3 — parameter-efficient finetuning of SAM3.1."""

  __version__ = "0.0.1"
  ```
- `src/esam3/config/__init__.py` — empty (just `""""""` docstring optional, leave empty file).
- `src/esam3/data/__init__.py` — empty.
- `src/esam3/models/__init__.py` — empty.
- `src/esam3/peft_adapters/__init__.py` — empty.
- `src/esam3/train/__init__.py` — empty.
- `src/esam3/eval/__init__.py` — empty.
- `src/esam3/tracking/__init__.py` — empty.
- `src/esam3/cli/__init__.py` — empty.
- `tests/__init__.py` — empty.
- `tests/unit/__init__.py` — empty.
- `tests/fixtures/__init__.py` — empty.

For every module in `test_imports.py::MODULES` except those whose `__init__.py` you just created, create the leaf file as an empty stub now so the import test can pass. Use this exact content for each leaf file (just enough to import):

```python
"""Stub — implemented in a later task."""
```

The leaf files to create are:
- `src/esam3/_registry.py`
- `src/esam3/config/schema.py`
- `src/esam3/config/loader.py`
- `src/esam3/data/base.py`
- `src/esam3/data/coco.py`
- `src/esam3/data/hf.py`
- `src/esam3/data/transforms.py`
- `src/esam3/data/collate.py`
- `src/esam3/models/sam3.py`
- `src/esam3/models/losses.py`
- `src/esam3/peft_adapters/lora.py`
- `src/esam3/peft_adapters/qlora.py`
- `src/esam3/train/trainer.py`
- `src/esam3/train/loop.py`
- `src/esam3/train/checkpoint.py`
- `src/esam3/eval/metrics.py`
- `src/esam3/eval/evaluator.py`
- `src/esam3/tracking/base.py`
- `src/esam3/tracking/tensorboard.py`
- `src/esam3/tracking/wandb.py`
- `src/esam3/tracking/noop.py`
- `src/esam3/cli/main.py`
- `src/esam3/cli/train_cmd.py`
- `src/esam3/cli/eval_cmd.py`
- `src/esam3/cli/export_cmd.py`
- `src/esam3/cli/init_cmd.py`
- `src/esam3/cli/doctor_cmd.py`

- [ ] **Step 4: Run the import test to verify it passes**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ tests/
git commit -m "feat: scaffold empty esam3 package skeleton + import smoke test"
```

---

## Task 3: Plugin registry (`_registry.py`)

**Files:**
- Modify: `src/esam3/_registry.py`
- Create: `tests/unit/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_registry.py`:

```python
"""Tests for the plugin registry."""

from __future__ import annotations

import pytest

from esam3._registry import (
    RegistryError,
    list_registered,
    lookup,
    register,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_registry()


def test_register_and_lookup_roundtrip() -> None:
    @register("dataset", "fake")
    def factory() -> str:
        return "ok"

    assert lookup("dataset", "fake") is factory
    assert factory() == "ok"


def test_lookup_unknown_raises() -> None:
    with pytest.raises(RegistryError, match="unknown 'dataset' entry 'missing'"):
        lookup("dataset", "missing")


def test_duplicate_name_raises() -> None:
    @register("tracker", "dup")
    def first() -> None:
        return None

    with pytest.raises(RegistryError, match="'tracker' entry 'dup' already registered"):

        @register("tracker", "dup")
        def second() -> None:
            return None


def test_separate_kinds_do_not_collide() -> None:
    @register("dataset", "shared")
    def a() -> str:
        return "dataset"

    @register("tracker", "shared")
    def b() -> str:
        return "tracker"

    assert lookup("dataset", "shared") is a
    assert lookup("tracker", "shared") is b


def test_list_registered_returns_names_for_kind() -> None:
    @register("peft", "lora")
    def _lora() -> None:
        return None

    @register("peft", "qlora")
    def _qlora() -> None:
        return None

    assert set(list_registered("peft")) == {"lora", "qlora"}


def test_list_registered_unknown_kind_returns_empty() -> None:
    assert list_registered("nonexistent") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_registry.py -v`
Expected: FAIL (`ImportError` — names don't exist yet).

- [ ] **Step 3: Implement `_registry.py`**

Replace `src/esam3/_registry.py`:

```python
"""Plugin registry for dataset adapters, PEFT methods, and trackers.

Pluggable surfaces declare themselves via the @register decorator at import
time. The CLI and library look them up by (kind, name). Adding a new
implementation = one file + one @register + one test; no edits to dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable[..., object])


class RegistryError(KeyError):
    """Raised on duplicate registration or unknown lookup."""


_REGISTRY: dict[str, dict[str, Callable[..., object]]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    """Decorator: register `fn` under (kind, name)."""

    def decorator(fn: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise RegistryError(f"'{kind}' entry '{name}' already registered")
        bucket[name] = fn
        return fn

    return decorator


def lookup(kind: str, name: str) -> Callable[..., object]:
    """Return the callable registered under (kind, name)."""
    bucket = _REGISTRY.get(kind, {})
    if name not in bucket:
        raise RegistryError(f"unknown '{kind}' entry '{name}'")
    return bucket[name]


def list_registered(kind: str) -> list[str]:
    """Return the sorted names registered under `kind`. Empty list if no kind."""
    return sorted(_REGISTRY.get(kind, {}).keys())


def reset_registry() -> None:
    """Clear the registry — test-only helper."""
    _REGISTRY.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_registry.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/_registry.py tests/unit/test_registry.py
git commit -m "feat(registry): plugin registry with TDD"
```

---

## Task 4: Config schema (pydantic models)

**Files:**
- Modify: `src/esam3/config/schema.py`
- Create: `tests/unit/test_config_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_config_schema.py`:

```python
"""Tests for the pydantic config schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import (
    AugmentationsConfig,
    DataConfig,
    DataSplit,
    EvalConfig,
    ExportConfig,
    ModelConfig,
    PEFTConfig,
    QLoRAConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
    WandbConfig,
)


def _minimal_dict() -> dict[str, object]:
    return {
        "run": {"name": "test-run", "output_dir": "./runs", "seed": 42},
        "model": {"name": "facebook/sam3.1"},
        "data": {
            "format": "coco",
            "train": {"annotations": "data/train.json", "images": "data/train/"},
            "val": {"annotations": "data/val.json", "images": "data/val/"},
            "prompt_mode": "bbox",
            "image_size": 1024,
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 10},
        "eval": {},
        "tracking": {"backend": "tensorboard"},
        "export": {"merge": False},
    }


def test_full_config_validates() -> None:
    cfg = TrainConfig.model_validate(_minimal_dict())
    assert cfg.run.name == "test-run"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.peft.method == "lora"
    assert cfg.train.batch_size == 1
    assert cfg.train.grad_accum_steps == 8
    assert cfg.train.optimizer == "adamw"
    assert cfg.tracking.backend == "tensorboard"


def test_invalid_dtype_rejected() -> None:
    d = _minimal_dict()
    d["model"]["dtype"] = "float32"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_prompt_mode_rejected() -> None:
    d = _minimal_dict()
    d["data"]["prompt_mode"] = "points"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_peft_method_rejected() -> None:
    d = _minimal_dict()
    d["peft"]["method"] = "ia3"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_data_format_rejected() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "yolo"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_optimizer_rejected() -> None:
    d = _minimal_dict()
    d["train"]["optimizer"] = "sgd"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_tracker_backend_rejected() -> None:
    d = _minimal_dict()
    d["tracking"]["backend"] = "mlflow"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_negative_lr_rejected() -> None:
    d = _minimal_dict()
    d["train"]["lr"] = -1.0  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_zero_epochs_rejected() -> None:
    d = _minimal_dict()
    d["train"]["epochs"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_unknown_top_level_key_rejected() -> None:
    d = _minimal_dict()
    d["extra_section"] = {}
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_qlora_subconfig_defaults() -> None:
    d = _minimal_dict()
    d["peft"]["method"] = "qlora"  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.peft.qlora.quant_type == "nf4"
    assert cfg.peft.qlora.compute_dtype == "bfloat16"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_schema.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `config/schema.py`**

Replace `src/esam3/config/schema.py`:

```python
"""Pydantic v2 schema for esam3 training configurations.

This module is the source of truth for every default and constraint. The
loader merges YAML + CLI overrides into a plain dict, then validates once
against TrainConfig.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

Dtype = Literal["bfloat16", "float16"]
DataFormat = Literal["coco", "hf"]
PromptMode = Literal["text", "bbox"]
PEFTMethod = Literal["lora", "qlora"]
QuantType = Literal["nf4", "fp4"]
Optimizer = Literal["adamw", "adamw8bit"]
LRSchedule = Literal["constant", "cosine", "linear"]
TrackerBackend = Literal["tensorboard", "wandb", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Strict):
    name: str
    output_dir: str = "./runs"
    seed: int = 42


class ModelConfig(_Strict):
    name: str = "facebook/sam3.1"
    revision: str | None = None
    gradient_checkpointing: bool = True
    dtype: Dtype = "bfloat16"


class DataSplit(_Strict):
    annotations: str
    images: str


class AugmentationsConfig(_Strict):
    hflip: bool = True
    color_jitter: float = Field(default=0.1, ge=0.0, le=1.0)


class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit
    prompt_mode: PromptMode
    image_size: PositiveInt = 1024
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)


class QLoRAConfig(_Strict):
    quant_type: QuantType = "nf4"
    compute_dtype: Dtype = "bfloat16"


class PEFTConfig(_Strict):
    method: PEFTMethod
    r: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)


class TrainHyperparams(_Strict):
    epochs: PositiveInt
    batch_size: PositiveInt = 1
    grad_accum_steps: PositiveInt = 8
    optimizer: Optimizer = "adamw"
    lr: PositiveFloat = 1.0e-4
    lr_schedule: LRSchedule = "cosine"
    warmup_steps: int = Field(default=100, ge=0)
    max_grad_norm: PositiveFloat = 1.0
    eval_every: PositiveInt = 500
    save_every: PositiveInt = 1000


class EvalConfig(_Strict):
    metrics: list[str] = Field(
        default_factory=lambda: ["mAP", "mAP_50", "mAP_75", "per_class_AP"]
    )
    iou_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )


class WandbConfig(_Strict):
    project: str = "esam3"
    entity: str | None = None


class TrackingConfig(_Strict):
    backend: TrackerBackend = "tensorboard"
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class ExportConfig(_Strict):
    merge: bool = False


class TrainConfig(_Strict):
    """Top-level config produced by the loader."""

    run: RunConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    peft: PEFTConfig
    train: TrainHyperparams
    eval: EvalConfig = Field(default_factory=EvalConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config_schema.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(config): pydantic schema with strict validation"
```

---

## Task 5: Config loader (YAML + overrides)

**Files:**
- Modify: `src/esam3/config/loader.py`
- Create: `tests/unit/test_config_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_config_loader.py`:

```python
"""Tests for the YAML config loader and --override merging."""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.config.loader import ConfigError, apply_overrides, load_config


def _write_minimal_yaml(p: Path) -> Path:
    p.write_text(
        """
run:
  name: t
model:
  name: facebook/sam3.1
data:
  format: coco
  train: { annotations: train.json, images: train/ }
  val: { annotations: val.json, images: val/ }
  prompt_mode: bbox
peft:
  method: lora
train:
  epochs: 3
""".lstrip()
    )
    return p


def test_load_config_returns_validated_train_config(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    cfg = load_config(cfg_file)
    assert cfg.run.name == "t"
    assert cfg.train.epochs == 3
    assert cfg.peft.method == "lora"


def test_paths_resolved_relative_to_config_file(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    cfg = load_config(cfg_file)
    # data.train.annotations resolves relative to cfg_file's directory
    assert Path(cfg.data.train.annotations).is_absolute()
    assert Path(cfg.data.train.annotations) == (tmp_path / "train.json").resolve()
    assert Path(cfg.data.val.images) == (tmp_path / "val").resolve()


def test_apply_overrides_modifies_nested_key() -> None:
    base = {"a": {"b": {"c": 1}}}
    apply_overrides(base, ["a.b.c=42"])
    assert base == {"a": {"b": {"c": 42}}}


def test_apply_overrides_parses_int_float_bool_null() -> None:
    base: dict[str, object] = {"x": {}}
    apply_overrides(base, ["x.i=7", "x.f=1.5", "x.t=true", "x.f2=false", "x.n=null"])
    assert base["x"] == {"i": 7, "f": 1.5, "t": True, "f2": False, "n": None}


def test_apply_overrides_creates_missing_intermediate_keys() -> None:
    base: dict[str, object] = {}
    apply_overrides(base, ["deeply.nested.key=value"])
    assert base == {"deeply": {"nested": {"key": "value"}}}


def test_load_config_with_override(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    cfg = load_config(cfg_file, overrides=["train.epochs=99", "peft.r=8"])
    assert cfg.train.epochs == 99
    assert cfg.peft.r == 8


def test_invalid_config_raises_config_error(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("run: { name: t }\n")  # missing required sections
    with pytest.raises(ConfigError):
        load_config(cfg_file)


def test_malformed_override_raises(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    with pytest.raises(ConfigError, match="malformed override"):
        load_config(cfg_file, overrides=["not_an_assignment"])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_loader.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `config/loader.py`**

Replace `src/esam3/config/loader.py`:

```python
"""Load + validate YAML configs into a TrainConfig.

Responsibilities:
  - Load YAML.
  - Apply `--override key.subkey=value` flags onto the dict.
  - Resolve every path in DataConfig relative to the config file's directory.
  - Validate via pydantic; surface errors as ConfigError.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from esam3.config.schema import TrainConfig

_PATH_KEYS: tuple[tuple[str, ...], ...] = (
    ("data", "train", "annotations"),
    ("data", "train", "images"),
    ("data", "val", "annotations"),
    ("data", "val", "images"),
    ("run", "output_dir"),
)


class ConfigError(ValueError):
    """Raised when a config cannot be loaded, parsed, or validated."""


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
) -> TrainConfig:
    """Load YAML at `path`, apply overrides, resolve paths, return TrainConfig."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    if overrides:
        apply_overrides(raw, overrides)

    _resolve_paths(raw, base_dir=p.parent.resolve())

    try:
        return TrainConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e


def apply_overrides(target: dict[str, Any], overrides: Sequence[str]) -> None:
    """Mutate `target` in place: each override is `dotted.key=scalar_value`."""
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"malformed override (expected key=value): {ov!r}")
        key, _, raw_value = ov.partition("=")
        keys = key.split(".")
        node = target
        for k in keys[:-1]:
            existing = node.get(k)
            if not isinstance(existing, dict):
                existing = {}
                node[k] = existing
            node = existing
        node[keys[-1]] = _parse_scalar(raw_value)


def _parse_scalar(s: str) -> Any:
    """YAML-style scalar parsing for override values."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> None:
    for key_path in _PATH_KEYS:
        node: Any = raw
        for k in key_path[:-1]:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if not isinstance(node, dict):
            continue
        leaf = key_path[-1]
        val = node.get(leaf)
        if isinstance(val, str):
            candidate = Path(val)
            if not candidate.is_absolute():
                node[leaf] = str((base_dir / candidate).resolve())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config_loader.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/config/loader.py tests/unit/test_config_loader.py
git commit -m "feat(config): YAML loader with override merging and path resolution"
```

---

## Task 6: Data protocols and dataclasses (`data/base.py`)

**Files:**
- Modify: `src/esam3/data/base.py`
- Create: `tests/unit/test_data_base.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_data_base.py`:

```python
"""Tests for data/base.py protocols and dataclasses."""

from __future__ import annotations

import torch

from esam3.data.base import (
    BoxPrompts,
    Dataset,
    Example,
    Instance,
    TextPrompts,
    is_dataset,
)


def test_text_prompts_and_box_prompts_are_distinct_types() -> None:
    t = TextPrompts(classes=["cat", "dog"])
    b = BoxPrompts(
        boxes=torch.zeros((2, 4)),
        class_ids=torch.tensor([0, 1]),
    )
    assert isinstance(t, TextPrompts)
    assert isinstance(b, BoxPrompts)


def test_example_holds_image_prompts_and_instances() -> None:
    inst = Instance(
        mask=torch.zeros((4, 4), dtype=torch.bool),
        class_id=0,
        box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    ex = Example(
        image=torch.zeros((3, 4, 4)),
        image_id="img-1",
        prompts=TextPrompts(classes=["cat"]),
        instances=[inst],
    )
    assert ex.image_id == "img-1"
    assert ex.instances[0].class_id == 0


class _FakeDataset:
    def __init__(self) -> None:
        self._items = [
            Example(
                image=torch.zeros((3, 2, 2)),
                image_id=f"i-{i}",
                prompts=TextPrompts(classes=["a"]),
                instances=[],
            )
            for i in range(3)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> Example:
        return self._items[i]

    @property
    def class_names(self) -> list[str]:
        return ["a"]


def test_dataset_protocol_recognizes_conforming_class() -> None:
    ds: Dataset = _FakeDataset()
    assert len(ds) == 3
    assert ds[0].image_id == "i-0"
    assert ds.class_names == ["a"]
    assert is_dataset(ds) is True


def test_dataset_protocol_rejects_nonconforming() -> None:
    assert is_dataset(object()) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_base.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `data/base.py`**

Replace `src/esam3/data/base.py`:

```python
"""Data protocols and dataclasses — the stable seam between data and trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class TextPrompts:
    """Open-vocabulary class names used as prompts for one image."""

    classes: list[str]


@dataclass(frozen=True)
class BoxPrompts:
    """Per-image box prompts and their target class ids."""

    boxes: torch.Tensor  # (N, 4) xyxy, pixel coords
    class_ids: torch.Tensor  # (N,) int64


Prompts = TextPrompts | BoxPrompts


@dataclass(frozen=True)
class Instance:
    """Ground-truth instance for one mask in one image."""

    mask: torch.Tensor  # (H, W) bool
    class_id: int
    box: torch.Tensor  # (4,) xyxy


@dataclass(frozen=True)
class Example:
    """One training/eval example."""

    image: torch.Tensor  # (3, H, W) normalized
    image_id: str
    prompts: Prompts
    instances: list[Instance]


@runtime_checkable
class Dataset(Protocol):
    """Read-only mapping from index to Example, plus a class vocabulary."""

    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Example: ...
    @property
    def class_names(self) -> list[str]: ...


def is_dataset(obj: object) -> bool:
    """Structural check used by tests and CLI doctor."""
    return isinstance(obj, Dataset)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_base.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/base.py tests/unit/test_data_base.py
git commit -m "feat(data): Example/Prompts/Dataset protocol contracts"
```

---

## Task 7: Tracking — Tracker protocol + noop implementation

**Files:**
- Modify: `src/esam3/tracking/base.py`
- Modify: `src/esam3/tracking/noop.py`
- Create: `tests/unit/test_tracking_noop.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tracking_noop.py`:

```python
"""Tests for the Tracker protocol and the noop implementation."""

from __future__ import annotations

import numpy as np

from esam3._registry import lookup, reset_registry
from esam3.tracking.base import Tracker
from esam3.tracking.noop import NoopTracker, build_noop  # noqa: F401


def test_noop_tracker_conforms_to_protocol() -> None:
    t: Tracker = NoopTracker()
    t.log_scalars(0, {"loss": 1.0})
    t.log_images(0, {"sample": np.zeros((4, 4, 3), dtype=np.uint8)})
    t.close()


def test_noop_registered_under_tracker_kind() -> None:
    reset_registry()
    # Re-import to re-execute the @register decorator
    import importlib

    import esam3.tracking.noop as mod

    importlib.reload(mod)
    factory = lookup("tracker", "none")
    instance = factory({})
    assert isinstance(instance, NoopTracker)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_noop.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `tracking/base.py`**

Replace `src/esam3/tracking/base.py`:

```python
"""Tracker protocol — the stable seam between trainer and logging backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tracker(Protocol):
    """Minimal logging contract that every backend must implement."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None: ...
    def log_images(self, step: int, images: dict[str, np.ndarray]) -> None: ...
    def close(self) -> None: ...
```

- [ ] **Step 4: Implement `tracking/noop.py`**

Replace `src/esam3/tracking/noop.py`:

```python
"""No-op tracker. Selected via tracking.backend = "none"."""

from __future__ import annotations

from typing import Any

import numpy as np

from esam3._registry import register


class NoopTracker:
    """Tracker that drops all calls on the floor."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        return None

    def log_images(self, step: int, images: dict[str, np.ndarray]) -> None:
        return None

    def close(self) -> None:
        return None


@register("tracker", "none")
def build_noop(_cfg: dict[str, Any]) -> NoopTracker:
    """Factory called by trainer's tracker-building dispatch."""
    return NoopTracker()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_noop.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/tracking/base.py src/esam3/tracking/noop.py tests/unit/test_tracking_noop.py
git commit -m "feat(tracking): Tracker protocol + NoopTracker"
```

---

## Task 8: Eval — MetricsReport dataclass + Evaluator stub

**Files:**
- Modify: `src/esam3/eval/metrics.py`
- Modify: `src/esam3/eval/evaluator.py`

- [ ] **Step 1: Implement `eval/metrics.py`**

Replace `src/esam3/eval/metrics.py`:

```python
"""Evaluation metrics — MetricsReport contract + stub computation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricsReport:
    """Result of an Evaluator.evaluate() call."""

    overall: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    n_images: int = 0
    n_predictions: int = 0


def compute_coco_map(
    predictions: object,
    ground_truth: object,
    iou_thresholds: list[float],
) -> MetricsReport:
    """Compute COCO-style mAP + per-class AP. Stub — see spec/eval."""
    raise NotImplementedError("filled in by spec: spec/eval")
```

- [ ] **Step 2: Implement `eval/evaluator.py`**

Replace `src/esam3/eval/evaluator.py`:

```python
"""Evaluator — runs a model over a dataset and returns a MetricsReport."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import EvalConfig
from esam3.data.base import Dataset
from esam3.eval.metrics import MetricsReport


class Evaluator:
    """Compute COCO metrics for a model on a dataset.

    Implementation deferred to spec/eval.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    def evaluate(self, model: Any, dataset: Dataset) -> MetricsReport:
        raise NotImplementedError("filled in by spec: spec/eval")
```

- [ ] **Step 3: Run the import test to verify nothing broke**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/esam3/eval/metrics.py src/esam3/eval/evaluator.py
git commit -m "feat(eval): MetricsReport dataclass + Evaluator/compute_coco_map stubs"
```

---

## Task 9: Train — RunResult dataclass + Trainer stub + loop/checkpoint stubs

**Files:**
- Modify: `src/esam3/train/trainer.py`
- Modify: `src/esam3/train/loop.py`
- Modify: `src/esam3/train/checkpoint.py`

- [ ] **Step 1: Implement `train/trainer.py`**

Replace `src/esam3/train/trainer.py`:

```python
"""Trainer — public training entrypoint. Loop body lives in train/loop.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esam3.config.schema import TrainConfig
from esam3.data.base import Dataset
from esam3.eval.metrics import MetricsReport
from esam3.tracking.base import Tracker


@dataclass(frozen=True)
class RunResult:
    """Returned from Trainer.fit() — what a run produced on disk."""

    run_dir: Path
    adapter_path: Path
    merged_path: Path | None
    final_metrics: MetricsReport | None


class Trainer:
    """Drive a finetuning run end-to-end.

    Implementation deferred to spec/training-loop.
    """

    def __init__(
        self,
        model: Any,
        train_ds: Dataset,
        val_ds: Dataset,
        tracker: Tracker,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tracker = tracker
        self.cfg = cfg

    def fit(self) -> RunResult:
        raise NotImplementedError("filled in by spec: spec/training-loop")
```

- [ ] **Step 2: Implement `train/loop.py`**

Replace `src/esam3/train/loop.py`:

```python
"""Inner training step / epoch loop. Implementation deferred to spec/training-loop."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import TrainHyperparams


def run_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    cfg: TrainHyperparams,
    step: int,
) -> int:
    """Run one epoch. Returns the updated global step counter."""
    raise NotImplementedError("filled in by spec: spec/training-loop")
```

- [ ] **Step 3: Implement `train/checkpoint.py`**

Replace `src/esam3/train/checkpoint.py`:

```python
"""Checkpoint save/load. Implementation deferred to spec/training-loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_adapter(model: Any, path: Path) -> None:
    raise NotImplementedError("filled in by spec: spec/training-loop")


def save_merged(model: Any, path: Path) -> None:
    raise NotImplementedError("filled in by spec: spec/training-loop")


def load_adapter(model: Any, path: Path) -> Any:
    raise NotImplementedError("filled in by spec: spec/training-loop")
```

- [ ] **Step 4: Run import test**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/train/
git commit -m "feat(train): RunResult dataclass + Trainer/loop/checkpoint stubs"
```

---

## Task 10: Data adapter stubs (coco, hf, transforms, collate)

**Files:**
- Modify: `src/esam3/data/coco.py`
- Modify: `src/esam3/data/hf.py`
- Modify: `src/esam3/data/transforms.py`
- Modify: `src/esam3/data/collate.py`

- [ ] **Step 1: Implement `data/coco.py`**

Replace `src/esam3/data/coco.py`:

```python
"""COCO instance-JSON dataset adapter. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.data.base import Dataset, Example


class COCODataset:
    """Read a COCO instance-segmentation JSON + image folder as a Dataset."""

    def __init__(self, annotations: str, images: str, prompt_mode: str) -> None:
        self.annotations = annotations
        self.images = images
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by spec: spec/data-loading")


@register("dataset", "coco")
def build_coco(cfg: dict[str, Any]) -> Dataset:
    return COCODataset(
        annotations=cfg["annotations"],
        images=cfg["images"],
        prompt_mode=cfg["prompt_mode"],
    )
```

- [ ] **Step 2: Implement `data/hf.py`**

Replace `src/esam3/data/hf.py`:

```python
"""HuggingFace `datasets` adapter. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.data.base import Dataset, Example


class HFDataset:
    def __init__(self, name: str, split: str, prompt_mode: str) -> None:
        self.name = name
        self.split = split
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by spec: spec/data-loading")


@register("dataset", "hf")
def build_hf(cfg: dict[str, Any]) -> Dataset:
    return HFDataset(
        name=cfg["name"],
        split=cfg["split"],
        prompt_mode=cfg["prompt_mode"],
    )
```

- [ ] **Step 3: Implement `data/transforms.py`**

Replace `src/esam3/data/transforms.py`:

```python
"""Image + prompt augmentations. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import AugmentationsConfig


def build_train_transforms(cfg: AugmentationsConfig, image_size: int) -> Any:
    raise NotImplementedError("filled in by spec: spec/data-loading")


def build_eval_transforms(image_size: int) -> Any:
    raise NotImplementedError("filled in by spec: spec/data-loading")
```

- [ ] **Step 4: Implement `data/collate.py`**

Replace `src/esam3/data/collate.py`:

```python
"""Variable-shape batch collator. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3.data.base import Example


def collate_batch(examples: list[Example]) -> dict[str, Any]:
    raise NotImplementedError("filled in by spec: spec/data-loading")
```

- [ ] **Step 5: Run import test**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/data/
git commit -m "feat(data): adapter + transforms + collate stubs"
```

---

## Task 11: Model stubs (sam3, losses)

**Files:**
- Modify: `src/esam3/models/sam3.py`
- Modify: `src/esam3/models/losses.py`

- [ ] **Step 1: Implement `models/sam3.py`**

Replace `src/esam3/models/sam3.py`:

```python
"""SAM3.1 loader + forward wrapper. Implementation deferred to spec/model-loading."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import ModelConfig


def load_sam31(cfg: ModelConfig) -> Any:
    """Load SAM3.1 from HuggingFace, applying dtype + grad-checkpointing flags."""
    raise NotImplementedError("filled in by spec: spec/model-loading")
```

- [ ] **Step 2: Implement `models/losses.py`**

Replace `src/esam3/models/losses.py`:

```python
"""SAM3.1 training losses. Implementation deferred to spec/model-loading."""

from __future__ import annotations

from typing import Any

import torch


def mask_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def box_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def objectness_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def total_loss(outputs: dict[str, Any], targets: dict[str, Any]) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")
```

- [ ] **Step 3: Run import test**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/esam3/models/
git commit -m "feat(models): SAM3.1 loader + loss stubs"
```

---

## Task 12: PEFT adapter stubs (lora, qlora)

**Files:**
- Modify: `src/esam3/peft_adapters/lora.py`
- Modify: `src/esam3/peft_adapters/qlora.py`

- [ ] **Step 1: Implement `peft_adapters/lora.py`**

Replace `src/esam3/peft_adapters/lora.py`:

```python
"""LoRA adapter via huggingface/peft. Implementation deferred to spec/peft-lora."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.config.schema import PEFTConfig


@register("peft", "lora")
def apply_lora(model: Any, cfg: PEFTConfig) -> Any:
    """Wrap `model` with a LoRA PeftModel, returning the wrapped module."""
    raise NotImplementedError("filled in by spec: spec/peft-lora")
```

- [ ] **Step 2: Implement `peft_adapters/qlora.py`**

Replace `src/esam3/peft_adapters/qlora.py`:

```python
"""QLoRA adapter (4-bit base + LoRA). Implementation deferred to spec/peft-qlora.

Requires the [qlora] optional extra (bitsandbytes).
"""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.config.schema import PEFTConfig


@register("peft", "qlora")
def apply_qlora(model: Any, cfg: PEFTConfig) -> Any:
    """Quantize the base model to nf4 and wrap with LoRA, returning the wrapped module."""
    raise NotImplementedError("filled in by spec: spec/peft-qlora")
```

- [ ] **Step 3: Run import test**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/esam3/peft_adapters/
git commit -m "feat(peft): LoRA/QLoRA factory stubs registered"
```

---

## Task 13: Tracking backend stubs (tensorboard, wandb)

**Files:**
- Modify: `src/esam3/tracking/tensorboard.py`
- Modify: `src/esam3/tracking/wandb.py`

- [ ] **Step 1: Implement `tracking/tensorboard.py`**

Replace `src/esam3/tracking/tensorboard.py`:

```python
"""TensorBoard tracker. Implementation deferred to spec/tracking.

Requires the [tensorboard] optional extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from esam3._registry import register


class TensorBoardTracker:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def log_images(self, step: int, images: dict[str, np.ndarray]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def close(self) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")


@register("tracker", "tensorboard")
def build_tensorboard(cfg: dict[str, Any]) -> TensorBoardTracker:
    return TensorBoardTracker(log_dir=cfg.get("log_dir", "./runs"))
```

- [ ] **Step 2: Implement `tracking/wandb.py`**

Replace `src/esam3/tracking/wandb.py`:

```python
"""Weights & Biases tracker. Implementation deferred to spec/tracking.

Requires the [wandb] optional extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from esam3._registry import register


class WandBTracker:
    def __init__(self, project: str, entity: str | None) -> None:
        self.project = project
        self.entity = entity

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def log_images(self, step: int, images: dict[str, np.ndarray]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def close(self) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")


@register("tracker", "wandb")
def build_wandb(cfg: dict[str, Any]) -> WandBTracker:
    return WandBTracker(project=cfg.get("project", "esam3"), entity=cfg.get("entity"))
```

- [ ] **Step 3: Run import test**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/esam3/tracking/
git commit -m "feat(tracking): TensorBoard + W&B backend stubs registered"
```

---

## Task 14: CLI Typer skeleton

**Files:**
- Modify: `src/esam3/cli/main.py`
- Modify: `src/esam3/cli/train_cmd.py`
- Modify: `src/esam3/cli/eval_cmd.py`
- Modify: `src/esam3/cli/export_cmd.py`
- Modify: `src/esam3/cli/init_cmd.py`
- Modify: `src/esam3/cli/doctor_cmd.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cli.py`:

```python
"""Tests for the Typer CLI skeleton."""

from __future__ import annotations

from typer.testing import CliRunner

from esam3.cli.main import app

runner = CliRunner()


def test_root_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "eval" in result.stdout
    assert "export" in result.stdout
    assert "init" in result.stdout
    assert "doctor" in result.stdout


def test_train_help_exits_zero() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout


def test_eval_help_exits_zero() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout


def test_export_help_exits_zero() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint" in result.stdout


def test_init_help_exits_zero() -> None:
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


def test_doctor_runs_and_prints_not_implemented() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.stdout.lower()


def test_train_with_valid_config_prints_not_implemented(tmp_path: object) -> None:
    # Use the committed example config
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "configs" / "examples" / "coco_text_lora.yaml"
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "not yet implemented" in result.stdout.lower()
```

> Note: the final test depends on Task 16 (example configs). Skip running it until then; the earlier tests should pass after this task.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: FAIL (ImportError or attribute errors).

- [ ] **Step 3: Implement `cli/main.py`**

Replace `src/esam3/cli/main.py`:

```python
"""`esam3` CLI entry point — wires subcommands into a Typer app."""

from __future__ import annotations

import typer

from esam3.cli import (
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    train_cmd,
)

app = typer.Typer(
    name="esam3",
    help="Parameter-efficient finetuning of SAM3.1.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("train", help="Run a finetune.")(train_cmd.train)
app.command("eval", help="Evaluate a checkpoint.")(eval_cmd.evaluate)
app.command("export", help="Export adapter or merged model.")(export_cmd.export)
app.command("init", help="Write a starter config.")(init_cmd.init)
app.command("doctor", help="Report environment + dependency status.")(doctor_cmd.doctor)


if __name__ == "__main__":  # pragma: no cover
    app()
```

- [ ] **Step 4: Implement `cli/train_cmd.py`**

Replace `src/esam3/cli/train_cmd.py`:

```python
"""`esam3 train` — parses config, hands off to library. Body deferred to spec/cli."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def train(
    config: Path = typer.Option(..., "--config", help="Path to training config YAML."),
    override: list[str] = typer.Option(
        [], "--override", help="Override config keys: dotted.key=value."
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Path to resume checkpoint."),
) -> None:
    """Run a finetune."""
    from esam3.config.loader import load_config

    cfg = load_config(config, overrides=override)
    rprint(f"[yellow]not yet implemented[/yellow] — would train run '{cfg.run.name}'")
    if resume is not None:
        rprint(f"  resume: {resume}")
```

- [ ] **Step 5: Implement `cli/eval_cmd.py`**

Replace `src/esam3/cli/eval_cmd.py`:

```python
"""`esam3 eval` — Body deferred to spec/cli."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def evaluate(
    config: Path = typer.Option(..., "--config", help="Path to config YAML."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    split: str = typer.Option("val", "--split", help="Dataset split: val | test."),
) -> None:
    """Evaluate a checkpoint."""
    rprint(
        f"[yellow]not yet implemented[/yellow] — would eval {checkpoint} "
        f"on {split} split of {config}"
    )
```

- [ ] **Step 6: Implement `cli/export_cmd.py`**

Replace `src/esam3/cli/export_cmd.py`:

```python
"""`esam3 export` — Body deferred to spec/cli."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint


def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to adapter checkpoint."),
    merge: bool = typer.Option(False, "--merge", help="Also export merged full-model weights."),
    output: Path | None = typer.Option(None, "--output", help="Output directory."),
) -> None:
    """Export adapter or merged model."""
    rprint(
        f"[yellow]not yet implemented[/yellow] — would export {checkpoint} "
        f"(merge={merge}) to {output}"
    )
```

- [ ] **Step 7: Implement `cli/init_cmd.py`**

Replace `src/esam3/cli/init_cmd.py`:

```python
"""`esam3 init` — Body deferred to spec/cli."""

from __future__ import annotations

import typer
from rich import print as rprint

VALID_TEMPLATES = ("coco-text", "coco-bbox", "hf-text")


def init(
    template: str = typer.Option(
        "coco-bbox",
        "--template",
        help=f"Starter config template. One of: {', '.join(VALID_TEMPLATES)}.",
    ),
) -> None:
    """Write a starter config to ./config.yaml."""
    if template not in VALID_TEMPLATES:
        raise typer.BadParameter(f"unknown template '{template}'")
    rprint(f"[yellow]not yet implemented[/yellow] — would write {template} starter config")
```

- [ ] **Step 8: Implement `cli/doctor_cmd.py`**

Replace `src/esam3/cli/doctor_cmd.py`:

```python
"""`esam3 doctor` — Body deferred to spec/cli."""

from __future__ import annotations

from rich import print as rprint


def doctor() -> None:
    """Report environment + dependency status."""
    rprint("[yellow]not yet implemented[/yellow] — would report CUDA, deps, VRAM, weight cache")
```

- [ ] **Step 9: Run tests to verify (skipping config-dependent test)**

Run: `uv run pytest tests/unit/test_cli.py -v -k "not test_train_with_valid_config"`
Expected: 6 passed.

- [ ] **Step 10: Verify the installed entry point works**

Run: `uv run esam3 --help`
Expected: prints help text listing `train`, `eval`, `export`, `init`, `doctor`. Exit code 0.

- [ ] **Step 11: Commit**

```bash
git add src/esam3/cli/ tests/unit/test_cli.py
git commit -m "feat(cli): Typer skeleton with five subcommands (bodies stubbed)"
```

---

## Task 15: Test fixtures (tiny COCO + stub model)

**Files:**
- Create: `tests/fixtures/make_tiny_coco.py`
- Create: `tests/fixtures/tiny_coco/annotations.json`
- Create: `tests/fixtures/tiny_coco/images/img_000001.png`
- Create: `tests/fixtures/tiny_coco/images/img_000002.png`
- Create: `tests/fixtures/tiny_sam3_stub.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_fixtures.py`

- [ ] **Step 1: Create the fixture generator**

Create `tests/fixtures/make_tiny_coco.py`:

```python
"""One-shot generator for the tiny_coco test fixture.

Run from the repo root once: `uv run python tests/fixtures/make_tiny_coco.py`.
The generated PNGs + JSON are committed; this script exists so the fixture
is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "tiny_coco"
IMG_DIR = OUT / "images"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Two 32x32 RGB images, each a solid color.
    Image.new("RGB", (32, 32), color=(200, 50, 50)).save(IMG_DIR / "img_000001.png")
    Image.new("RGB", (32, 32), color=(50, 200, 50)).save(IMG_DIR / "img_000002.png")

    annotations = {
        "info": {"description": "tiny_coco — esam3 test fixture", "version": "1.0"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "img_000001.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "img_000002.png", "width": 32, "height": 32},
        ],
        "categories": [
            {"id": 1, "name": "thing_a", "supercategory": "thing"},
            {"id": 2, "name": "thing_b", "supercategory": "thing"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 12, 12],
                "area": 144,
                "iscrowd": 0,
                "segmentation": [[4, 4, 16, 4, 16, 16, 4, 16]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [18, 18, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "segmentation": [[18, 18, 28, 18, 28, 28, 18, 28]],
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            },
        ],
    }
    (OUT / "annotations.json").write_text(json.dumps(annotations, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `uv run python tests/fixtures/make_tiny_coco.py`
Expected: creates `tests/fixtures/tiny_coco/annotations.json` and the two PNG files.

- [ ] **Step 3: Verify the fixture files exist**

Run: `ls tests/fixtures/tiny_coco/ tests/fixtures/tiny_coco/images/`
Expected: lists `annotations.json`, `images/`, `img_000001.png`, `img_000002.png`.

- [ ] **Step 4: Create the stub SAM3.1 model**

Create `tests/fixtures/tiny_sam3_stub.py`:

```python
"""A tiny `nn.Module` matching the (planned) SAM3.1 forward contract.

Used to unit-test trainer/eval/peft adapter logic without loading real weights.
The forward signature is intentionally loose — the contract gets pinned in
spec/model-loading.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinySam3Stub(nn.Module):
    """Returns deterministically-shaped random outputs given image + prompts."""

    def __init__(self, num_classes: int = 2, mask_size: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        # A single trainable param so optimizers have something to update.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, prompts: Any) -> dict[str, torch.Tensor]:
        del prompts  # ignored by the stub
        batch = image.shape[0] if image.ndim == 4 else 1
        return {
            "masks": torch.zeros(batch, 1, self.mask_size, self.mask_size) + self.dummy,
            "boxes": torch.zeros(batch, 1, 4) + self.dummy,
            "objectness": torch.zeros(batch, 1) + self.dummy,
            "class_logits": torch.zeros(batch, 1, self.num_classes) + self.dummy,
        }
```

- [ ] **Step 5: Create `tests/conftest.py`**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.tracking.noop import NoopTracker
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tiny_coco_dir() -> Path:
    return FIXTURES / "tiny_coco"


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def stub_model() -> TinySam3Stub:
    return TinySam3Stub()


@pytest.fixture
def noop_tracker() -> NoopTracker:
    return NoopTracker()
```

- [ ] **Step 6: Write tests that exercise the fixtures**

Create `tests/unit/test_fixtures.py`:

```python
"""Verify the fixtures are well-formed."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_tiny_coco_annotations_load(tiny_coco_dir: Path) -> None:
    data = json.loads((tiny_coco_dir / "annotations.json").read_text())
    assert len(data["images"]) == 2
    assert len(data["categories"]) == 2
    assert len(data["annotations"]) == 3


def test_tiny_coco_images_exist(tiny_coco_dir: Path) -> None:
    for name in ("img_000001.png", "img_000002.png"):
        assert (tiny_coco_dir / "images" / name).is_file()


def test_stub_model_forward_returns_expected_keys(stub_model: TinySam3Stub) -> None:
    image = torch.zeros((2, 3, 32, 32))
    out = stub_model(image, prompts=None)
    assert set(out.keys()) == {"masks", "boxes", "objectness", "class_logits"}
    assert out["masks"].shape == (2, 1, 32, 32)
    assert out["boxes"].shape == (2, 1, 4)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_fixtures.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/ tests/conftest.py tests/unit/test_fixtures.py
git commit -m "test: tiny_coco + stub model fixtures with smoke tests"
```

---

## Task 16: Example configs

**Files:**
- Create: `configs/examples/coco_text_lora.yaml`
- Create: `configs/examples/coco_bbox_qlora.yaml`

- [ ] **Step 1: Write `coco_text_lora.yaml`**

Create `configs/examples/coco_text_lora.yaml`:

```yaml
run:
  name: coco-text-lora
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  gradient_checkpointing: true
  dtype: bfloat16

data:
  format: coco
  train:
    annotations: data/coco/instances_train2017.json
    images: data/coco/train2017
  val:
    annotations: data/coco/instances_val2017.json
    images: data/coco/val2017
  prompt_mode: text
  image_size: 1024
  augmentations:
    hflip: true
    color_jitter: 0.1

peft:
  method: lora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj]

train:
  epochs: 10
  batch_size: 1
  grad_accum_steps: 8
  optimizer: adamw
  lr: 1.0e-4
  lr_schedule: cosine
  warmup_steps: 100
  max_grad_norm: 1.0
  eval_every: 500
  save_every: 1000

eval:
  metrics: [mAP, mAP_50, mAP_75, per_class_AP]
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

tracking:
  backend: tensorboard

export:
  merge: false
```

- [ ] **Step 2: Write `coco_bbox_qlora.yaml`**

Create `configs/examples/coco_bbox_qlora.yaml`:

```yaml
run:
  name: coco-bbox-qlora
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  gradient_checkpointing: true
  dtype: bfloat16

data:
  format: coco
  train:
    annotations: data/coco/instances_train2017.json
    images: data/coco/train2017
  val:
    annotations: data/coco/instances_val2017.json
    images: data/coco/val2017
  prompt_mode: bbox
  image_size: 1024
  augmentations:
    hflip: true
    color_jitter: 0.1

peft:
  method: qlora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj]
  qlora:
    quant_type: nf4
    compute_dtype: bfloat16

train:
  epochs: 10
  batch_size: 1
  grad_accum_steps: 8
  optimizer: adamw8bit
  lr: 1.0e-4
  lr_schedule: cosine
  warmup_steps: 100
  max_grad_norm: 1.0
  eval_every: 500
  save_every: 1000

eval:
  metrics: [mAP, mAP_50, mAP_75, per_class_AP]
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

tracking:
  backend: tensorboard

export:
  merge: false
```

- [ ] **Step 3: Verify both configs load and validate**

Run:
```bash
uv run python -c "
from esam3.config.loader import load_config
for p in ['configs/examples/coco_text_lora.yaml', 'configs/examples/coco_bbox_qlora.yaml']:
    cfg = load_config(p)
    print(p, '->', cfg.run.name, cfg.peft.method)
"
```
Expected: prints both names and methods, no exception.

- [ ] **Step 4: Re-run the previously skipped CLI test**

Run: `uv run pytest tests/unit/test_cli.py::test_train_with_valid_config_prints_not_implemented -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/
git commit -m "feat(configs): coco_text_lora + coco_bbox_qlora example configs"
```

---

## Task 17: Stub-call test (every stub raises NotImplementedError with the right message)

**Files:**
- Create: `tests/unit/test_stubs_raise.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_stubs_raise.py`:

```python
"""Verify every stub raises NotImplementedError with a spec: reference."""

from __future__ import annotations

import pytest
import torch

from esam3.config.schema import (
    AugmentationsConfig,
    EvalConfig,
    ModelConfig,
    PEFTConfig,
)
from esam3.data.coco import COCODataset
from esam3.data.collate import collate_batch
from esam3.data.hf import HFDataset
from esam3.data.transforms import build_eval_transforms, build_train_transforms
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import compute_coco_map
from esam3.models.losses import box_loss, mask_loss, objectness_loss, total_loss
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import apply_lora
from esam3.peft_adapters.qlora import apply_qlora
from esam3.train.checkpoint import load_adapter, save_adapter, save_merged
from esam3.train.loop import run_epoch


def _assert_stub(call: object) -> None:
    with pytest.raises(NotImplementedError, match="filled in by spec:"):
        call()  # type: ignore[operator]


def test_data_stubs() -> None:
    _assert_stub(lambda: COCODataset("a", "b", "bbox").__len__())
    _assert_stub(lambda: HFDataset("a", "train", "text").__len__())
    _assert_stub(lambda: build_train_transforms(AugmentationsConfig(), 1024))
    _assert_stub(lambda: build_eval_transforms(1024))
    _assert_stub(lambda: collate_batch([]))


def test_model_stubs() -> None:
    _assert_stub(lambda: load_sam31(ModelConfig()))
    t = torch.zeros((1,))
    _assert_stub(lambda: mask_loss(t, t))
    _assert_stub(lambda: box_loss(t, t))
    _assert_stub(lambda: objectness_loss(t, t))
    _assert_stub(lambda: total_loss({}, {}))


def test_peft_stubs() -> None:
    cfg = PEFTConfig(method="lora")
    _assert_stub(lambda: apply_lora(object(), cfg))
    qcfg = PEFTConfig(method="qlora")
    _assert_stub(lambda: apply_qlora(object(), qcfg))


def test_eval_stubs() -> None:
    _assert_stub(lambda: compute_coco_map(object(), object(), [0.5]))
    ev = Evaluator(EvalConfig())
    _assert_stub(lambda: ev.evaluate(object(), object()))  # type: ignore[arg-type]


def test_train_stubs(tmp_path: object) -> None:
    from pathlib import Path

    p = Path(str(tmp_path)) / "x"  # type: ignore[arg-type]
    _assert_stub(lambda: save_adapter(object(), p))
    _assert_stub(lambda: save_merged(object(), p))
    _assert_stub(lambda: load_adapter(object(), p))
    from esam3.config.schema import TrainHyperparams

    _assert_stub(
        lambda: run_epoch(object(), object(), object(), TrainHyperparams(epochs=1), 0)
    )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_stubs_raise.py -v`
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_stubs_raise.py
git commit -m "test: assert every stub raises NotImplementedError with spec reference"
```

---

## Task 18: Logs, ARCHITECTURE, README, LICENSE

**Files:**
- Create: `logs/log.md`
- Create: `logs/TODO.md`
- Create: `ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `LICENSE` (replace with Apache-2.0)

- [ ] **Step 1: Create `logs/log.md`**

Create `logs/log.md`:

```markdown
<!-- Append-only activity log per ~/.claude/CLAUDE.md.
     Format: [TIMESTAMP] [ROLE] action | [DEFERRED] issue
     Never read during task execution. -->

[2026-05-15] [planner] scaffolding plan written and committed
```

- [ ] **Step 2: Create `logs/TODO.md`**

Create `logs/TODO.md`:

```markdown
<!-- Append-only deferred-work log per ~/.claude/CLAUDE.md.
     Format: [TIMESTAMP] [ROLE] action | [DEFERRED] issue
     Never read during task execution. -->
```

- [ ] **Step 3: Create `ARCHITECTURE.md`**

Create `ARCHITECTURE.md`:

```markdown
# esam3 Architecture

This is the one-page reference for how `esam3` is wired together. The
full design rationale lives in `docs/superpowers/specs/`.

## Module map

```
src/esam3/
  _registry.py         plugin registry: register(kind, name) + lookup
  config/
    schema.py          pydantic v2 — defaults + validation contract
    loader.py          load YAML + apply --override + resolve paths
  data/
    base.py            Example, Prompts (TextPrompts | BoxPrompts), Dataset protocol
    coco.py / hf.py    @register("dataset", ...) adapters
    transforms.py      image + prompt augmentation
    collate.py         batch collator (variable-shape per image)
  models/
    sam3.py            HF SAM3.1 loader + forward wrapper
    losses.py          mask + box + objectness losses
  peft_adapters/
    lora.py / qlora.py @register("peft", ...) methods
  train/
    trainer.py         Trainer.fit() -> RunResult
    loop.py            inner step / epoch loop
    checkpoint.py      adapter + merged save/load
  eval/
    metrics.py         MetricsReport + COCO mAP
    evaluator.py       Evaluator.evaluate(model, dataset)
  tracking/
    base.py            Tracker protocol
    noop.py / tensorboard.py / wandb.py   @register("tracker", ...)
  cli/
    main.py            Typer entry point
    {train,eval,export,init,doctor}_cmd.py
```

## Data flow (one training run)

```
my.yaml ─► config.loader.load() ─► TrainConfig
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
       build_dataset()           load_sam31()              build_tracker()
       (coco | hf)               (model + dtype)           (tb | wandb | none)
            │                          │
            │                   apply_lora / apply_qlora
            │                          │
            └──────────────┬───────────┘
                           ▼
         train.Trainer(model, train_ds, val_ds, tracker, cfg).fit()
                           │
                           ▼
              runs/{run_id}/adapter/        ← always
              runs/{run_id}/merged/         ← if cfg.export.merge
              runs/{run_id}/metrics.json    ← final eval
```

## Stable seams

These are the only interfaces a forking developer should expect to remain
stable across patch releases. Everything else is internal.

- `esam3.data.base.Dataset` — `__len__`, `__getitem__(i) -> Example`, `class_names`.
- `esam3.tracking.base.Tracker` — `log_scalars`, `log_images`, `close`.
- `esam3.train.trainer.Trainer.fit() -> RunResult`.
- `esam3.eval.evaluator.Evaluator.evaluate(model, dataset) -> MetricsReport`.

## Adding a new pluggable surface

The registry pattern is used for three kinds: `dataset`, `peft`, `tracker`.
A new implementation is one file plus a decorator:

```python
# src/esam3/data/my_format.py
from esam3._registry import register

@register("dataset", "my_format")
def build_my_format(cfg: dict) -> Dataset:
    return MyDataset(**cfg)
```

Plus one test that imports the module and calls `lookup("dataset", "my_format")`.
No edits to dispatch code.
```

- [ ] **Step 4: Rewrite `README.md`**

Replace `README.md`:

```markdown
# efficient-sam3-finetuning

Parameter-efficient finetuning of [SAM3.1](https://huggingface.co/facebook/sam3.1)
on niche image instance-segmentation datasets — runnable on a single
consumer GPU.

> **Status:** v0 scaffolding only. The CLI and library surfaces exist;
> training/eval/data-loading bodies land in subsequent specs. See
> `docs/superpowers/specs/` for design and `docs/superpowers/plans/`
> for the build sequence.

## Quickstart

```bash
# Install
uv sync --all-extras --group dev

# Sanity check the CLI
uv run esam3 --help
uv run esam3 doctor

# Run the (currently stubbed) train command against an example config
uv run esam3 train --config configs/examples/coco_bbox_qlora.yaml
```

## What's supported in v0

| | v0 | Deferred |
|---|---|---|
| Model | SAM3.1 | SAM3 |
| Prompts | text, bounding boxes | points, masks |
| Data | static images, COCO + HF datasets | video |
| Output | instance segmentation | semantic segmentation |
| Distribution | single GPU | Ray Train, Argo workflows |
| PEFT | LoRA, QLoRA | other PEFT methods |
| Tracking | TensorBoard, W&B, none | — |

## Repo layout

See `ARCHITECTURE.md` for the module map and data flow.

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/esam3
uv run pytest
```

GPU smoke test (requires CUDA + SAM3.1 weights):

```bash
uv run pytest -m gpu
```

## License

Apache-2.0. See `LICENSE`.
```

- [ ] **Step 5: Replace `LICENSE` with the full Apache-2.0 text**

Run:
```bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

If the curl command fails (offline), write the canonical Apache-2.0 text manually to `LICENSE`. (The canonical text is 11,357 bytes; do not paraphrase it.)

Verify:
```bash
head -1 LICENSE
```
Expected: `                                 Apache License`

- [ ] **Step 6: Commit**

```bash
git add logs/ ARCHITECTURE.md README.md LICENSE
git commit -m "docs: README, ARCHITECTURE, logs scaffolding, Apache-2.0 LICENSE"
```

---

## Task 19: CI workflow + pre-commit + final verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Set Python version
        run: uv python install 3.13

      - name: Install deps
        run: uv sync --all-extras --group dev

      - name: Lint
        run: uv run ruff check

      - name: Format check
        run: uv run ruff format --check

      - name: Type check
        run: uv run mypy src/esam3

      - name: Test
        run: uv run pytest
```

- [ ] **Step 2: Create the pre-commit config**

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 3: Run the full exit-criteria suite locally**

Run each command and confirm it exits 0:

```bash
uv sync --all-extras --group dev
uv run ruff check
uv run ruff format --check
uv run mypy src/esam3
uv run pytest
uv run esam3 --help
```

Expected: all commands exit 0. `pytest` should report > 40 passing tests, 0 failing.

If `ruff format --check` fails, run `uv run ruff format` and re-stage the changes.
If `mypy` fails, fix the type errors before committing.

- [ ] **Step 4: Commit**

```bash
git add .github/ .pre-commit-config.yaml
git commit -m "ci: GitHub Actions workflow (ruff + mypy + pytest) and pre-commit config"
```

- [ ] **Step 5: Append to the activity log**

Append one line to `logs/log.md`:

```
[2026-05-15] [implementer] scaffolding complete; exit criteria pass
```

Run:
```bash
git add logs/log.md
git commit -m "chore(logs): mark scaffolding complete"
```

---

## Self-Review

After Task 19, confirm:

1. **Spec Section 10 coverage:**
   - `pyproject.toml` with deps + groups → Task 1 ✓
   - Full `src/esam3/` tree → Tasks 2, 7, 8, 9, 10, 11, 12, 13, 14 ✓
   - `config/schema.py` fully implemented → Task 4 ✓
   - `config/loader.py` implemented (YAML + override + path resolve) → Task 5 ✓
   - `data/base.py`, `tracking/base.py` Protocols + dataclasses → Tasks 6, 7 ✓
   - `_registry.py` implemented → Task 3 ✓
   - All other modules: class skeletons with type signatures, docstrings, NotImplementedError with "filled in by spec: <name>" → Tasks 8–13, asserted by Task 17 ✓
   - CLI Typer wired; each command parses config and prints "not yet implemented" → Task 14 ✓
   - tests/ with conftest, `tiny_coco/`, `tiny_sam3_stub.py` → Task 15 ✓
   - Tests covering: config loading + validation + override merging; registry register/lookup; `esam3 --help` exits 0; every public module imports → Tasks 2, 3, 4, 5, 14 ✓
   - `logs/log.md`, `logs/TODO.md` with header comments → Task 18 ✓
   - `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `.gitignore` → Tasks 1, 19 ✓
   - `README.md`, `ARCHITECTURE.md`, `LICENSE` → Task 18 ✓
   - `configs/examples/coco_text_lora.yaml`, `coco_bbox_qlora.yaml` valid → Task 16 ✓
   - Exit criteria pass → Task 19 ✓

2. **No placeholders.** Every step contains the file contents or the exact command to run. No "TBD", no "add appropriate handling", no "similar to Task N".

3. **Type consistency.** Cross-checked:
   - `register(kind, name)` / `lookup(kind, name)` / `list_registered(kind)` signatures match across `_registry.py`, the dataset/peft/tracker @register decorators, and the registry tests.
   - `Example`, `TextPrompts`, `BoxPrompts`, `Instance` field names match across `data/base.py` and the stub model fixture / tests.
   - `MetricsReport` field names (`overall`, `per_class`, `n_images`, `n_predictions`) match between `eval/metrics.py`, `eval/evaluator.py` return type, and `train/trainer.py` `RunResult.final_metrics`.
   - `RunResult` fields (`run_dir`, `adapter_path`, `merged_path`, `final_metrics`) are consistent with the spec Section 5.
   - `Tracker` protocol signature (`log_scalars(step, values)`, `log_images(step, images)`, `close()`) consistent across `tracking/base.py`, `NoopTracker`, `TensorBoardTracker`, `WandBTracker`.
   - Optimizer literal: `adamw | adamw8bit` — consistent between `config/schema.py` and the example QLoRA config which uses `adamw8bit`.
