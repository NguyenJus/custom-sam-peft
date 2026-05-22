# Data Subset Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-22-data-subset-limit-design.md`](../specs/2026-05-22-data-subset-limit-design.md)
**Issue:** [#72](https://github.com/NguyenJus/custom-sam-peft/issues/72) — *feat(data): support fast training runs on subsets of the full dataset*
**Branch:** `feat/data-subset-limit-72`

**Goal:** Add `data.limit` config so users can train on a deterministic small slice of a real COCO or HF dataset without editing JSON files or maintaining a separate data directory.

**Architecture:** `LimitConfig` mounts on `DataConfig`. A new `data/subset.py` module provides `resolve_subset_indices` (pure function) and `SubsetDataset` (transparent wrapper). `_build_dataset` in `runner.py` is the single integration seam — it wraps the inner dataset after construction when `limit_val` is not `None`. `image_class_labels` properties on `COCODataset` (eager) and `HFDataset` (lazy+cached) supply per-image class sets for stratified sampling. Doctor gains `--config` to report resolved dataset sizes without crashing on errors.

**Tech Stack:** Python 3.12+, pydantic v2, pytest (CPU-only, no `pytest.mark.gpu`), existing project stack.

---

## File Map

### New files

```
src/custom_sam_peft/data/subset.py           # resolve_subset_indices + SubsetDataset
tests/unit/test_data_subset.py               # pure-function + wrapper unit tests (schema + sampling + delegation)
tests/unit/test_data_coco_limit.py           # COCODataset.image_class_labels + limit integration
tests/unit/test_data_hf_limit.py             # HFDataset.image_class_labels lazy cache tests
tests/unit/test_train_runner_limit.py        # _build_dataset wrapping + subset.json write
tests/unit/test_cli_doctor_config.py         # csp doctor --config happy+sad paths
configs/examples/coco_text_lora_subset.yaml  # example YAML with data.limit block
```

### Modified files

```
src/custom_sam_peft/config/schema.py         # LimitConfig + DataConfig.limit field
src/custom_sam_peft/data/coco.py             # COCODataset.image_class_labels (eager)
src/custom_sam_peft/data/hf.py               # HFDataset.image_class_labels (lazy + cached)
src/custom_sam_peft/train/runner.py          # _build_dataset wrapping + subset.json write
src/custom_sam_peft/diagnostics.py           # DatasetResolution dataclass + DoctorReport.dataset field + run_doctor config_path param
src/custom_sam_peft/cli/doctor_cmd.py        # --config Typer option + DatasetResolution table rendering
```

---

## Pre-flight check

- [ ] **Step 0a: Confirm working tree clean**

```bash
git status
```

Expected: only spec + plan files (and any already-staged files). No other modifications.

- [ ] **Step 0b: Confirm baseline unit tests pass**

```bash
uv run pytest tests/unit -x -q
```

Expected: all pass. If anything is red, halt and report.

---

## Task 1: `LimitConfig` + `DataConfig.limit` schema

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py`
- Test: `tests/unit/test_data_subset.py` (created — schema cases only this task)

This task introduces the config model and its validator. It is file-disjoint from Tasks 2–5 only in the sense that later tasks *import* these types — so Task 1 must land first.

**Critical:** Pydantic v2 coerces `bool` to `int` silently. The validator must check `isinstance(v, bool)` **before** the numeric range check; otherwise `True` (which equals `int(1) >= 1`) slips through as valid.

- [ ] **Step 1a: Write the failing schema tests**

Create `tests/unit/test_data_subset.py`:

```python
"""Tests for data/subset.py — schema validation, resolve_subset_indices, SubsetDataset."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import LimitConfig


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("train", None),
        ("val", None),
        ("train", 1),
        ("train", 64),
        ("val", 100),
        ("train", 0.5),
        ("train", 1.0),
        ("val", 0.01),
    ],
)
def test_limit_config_valid(field: str, value: object) -> None:
    cfg = LimitConfig(**{field: value})
    assert getattr(cfg, field) == value


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("train", True, "bool"),
        ("train", False, "bool"),
        ("val", True, "bool"),
        ("train", 0, "int"),
        ("train", -1, "int"),
        ("val", 0, "int"),
        ("train", 0.0, "float"),
        ("train", -0.1, "float"),
        ("train", 1.1, "float"),
        ("val", 1.5, "float"),
    ],
)
def test_limit_config_invalid(field: str, value: object, match: str) -> None:
    with pytest.raises(ValidationError):
        LimitConfig(**{field: value})


def test_limit_config_defaults() -> None:
    cfg = LimitConfig()
    assert cfg.train is None
    assert cfg.val is None
    assert cfg.seed == 42
    assert cfg.strategy == "random"


def test_limit_config_strategy_valid() -> None:
    for s in ("random", "stratified", "first_n"):
        cfg = LimitConfig(strategy=s)  # type: ignore[arg-type]
        assert cfg.strategy == s
```

- [ ] **Step 1b: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/test_data_subset.py -v
```

Expected: `ImportError: cannot import name 'LimitConfig' from 'custom_sam_peft.config.schema'`.

- [ ] **Step 1c: Add `LimitConfig` and `DataConfig.limit` to `schema.py`**

In `src/custom_sam_peft/config/schema.py`, add `LimitConfig` below `HFDatasetConfig` (before `DataConfig`):

```python
class LimitConfig(_Strict):
    """Optional dataset size limit for fast iteration / smoke runs.

    train/val: int >= 1 (cap), float in (0, 1] (fraction), None = no-op.
    seed: salts the subset RNG independently from run.seed.
    strategy: random | stratified | first_n.

    bool is explicitly rejected: True would pass the int >= 1 check silently.
    """

    train: int | float | None = None
    val: int | float | None = None
    seed: int = 42
    strategy: Literal["random", "stratified", "first_n"] = "random"

    @model_validator(mode="after")
    def _check_limits(self) -> LimitConfig:
        for name in ("train", "val"):
            v = getattr(self, name)
            if v is None:
                continue
            if isinstance(v, bool):
                raise ValueError(
                    f"limit.{name} must not be a bool; got {v!r}"
                )
            if isinstance(v, int):
                if v < 1:
                    raise ValueError(
                        f"limit.{name} int must be >= 1; got {v}"
                    )
            elif isinstance(v, float):
                if not (0.0 < v <= 1.0):
                    raise ValueError(
                        f"limit.{name} float must be in (0, 1]; got {v}"
                    )
        return self
```

Then extend `DataConfig` with the `limit` field. Find the `DataConfig` class in `schema.py` and add one line after `normalize`:

```python
    limit: LimitConfig = Field(default_factory=LimitConfig)
```

The full `DataConfig` becomes:

```python
class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit
    test: DataSplit | None = None
    hf: HFDatasetConfig | None = None
    prompt_mode: PromptMode
    image_size: PositiveInt = 1024
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    text_prompt: TextPromptConfig = Field(default_factory=TextPromptConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    limit: LimitConfig = Field(default_factory=LimitConfig)

    @model_validator(mode="after")
    def _check_format_specific(self) -> DataConfig:
        if self.format == "hf" and self.hf is None:
            raise ValueError("data.hf is required when data.format == 'hf'")
        return self
```

- [ ] **Step 1d: Run schema tests, confirm they pass**

```bash
uv run pytest tests/unit/test_data_subset.py -v
```

Expected: all schema tests pass.

- [ ] **Step 1e: Confirm existing schema + config tests still pass (regression gate)**

```bash
uv run pytest tests/unit/test_config_schema.py tests/unit/test_data_schema_extensions.py tests/unit/test_config_loader.py tests/unit/test_config_examples.py -v
```

Expected: all pass. The default `LimitConfig()` keeps both `train` and `val` as `None`; no existing test is affected.

- [ ] **Step 1f: ruff + mypy**

```bash
uv run ruff check src/custom_sam_peft/config/schema.py tests/unit/test_data_subset.py
uv run mypy src/custom_sam_peft/config/schema.py
```

Expected: both clean.

- [ ] **Step 1g: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_data_subset.py
git commit -m "feat(config): add LimitConfig + DataConfig.limit with bool-rejection validator"
```

---

## Task 2: `resolve_subset_indices` + `SubsetDataset` (new module)

**Files:**
- Create: `src/custom_sam_peft/data/subset.py`
- Modify: `tests/unit/test_data_subset.py` (append sampling + delegation tests)

Depends on Task 1 (`LimitConfig` types needed for imports; the type annotation in `resolve_subset_indices` uses `Literal`).

The stratified implementation (~50 lines) tracks per-class remaining quota proportional to source class frequencies, selects images greedy by rarest-still-needed-class deficit, tie-breaks by smallest current quota then image index, and random-fills any shortfall from the unselected pool. No new runtime dependencies (pure Python/list arithmetic).

- [ ] **Step 2a: Append sampling and wrapper tests to `tests/unit/test_data_subset.py`**

Append to the existing file:

```python

# ---------------------------------------------------------------------------
# resolve_subset_indices
# ---------------------------------------------------------------------------

import logging

from custom_sam_peft.data.subset import SubsetDataset, resolve_subset_indices


def test_first_n_ascending_range() -> None:
    idx = resolve_subset_indices(10, 4, seed=0, strategy="first_n", image_class_labels=None)
    assert idx == [0, 1, 2, 3]


def test_first_n_clips_to_n_total() -> None:
    idx = resolve_subset_indices(5, 10, seed=0, strategy="first_n", image_class_labels=None)
    assert idx == [0, 1, 2, 3, 4]


def test_first_n_ignores_seed_and_labels() -> None:
    a = resolve_subset_indices(10, 3, seed=0, strategy="first_n", image_class_labels=[[0], [1]])
    b = resolve_subset_indices(10, 3, seed=99, strategy="first_n", image_class_labels=None)
    assert a == b == [0, 1, 2]


def test_random_correct_count_sorted_unique() -> None:
    idx = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    assert len(idx) == 20
    assert idx == sorted(idx)
    assert len(set(idx)) == 20
    assert all(0 <= i < 100 for i in idx)


def test_random_deterministic_same_seed_n_total() -> None:
    a = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    b = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    assert a == b


def test_random_different_n_total_gives_different_subset() -> None:
    a = resolve_subset_indices(100, 20, seed=42, strategy="random", image_class_labels=None)
    b = resolve_subset_indices(101, 20, seed=42, strategy="random", image_class_labels=None)
    assert a != b


def test_random_float_limit_rounds() -> None:
    # 0.25 * 20 = 5
    idx = resolve_subset_indices(20, 0.25, seed=0, strategy="random", image_class_labels=None)
    assert len(idx) == 5


def test_random_float_1_0_returns_all() -> None:
    idx = resolve_subset_indices(10, 1.0, seed=0, strategy="random", image_class_labels=None)
    assert idx == list(range(10))


def test_cap_exceeds_n_total_warns_and_returns_all(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.data.subset"):
        idx = resolve_subset_indices(5, 100, seed=0, strategy="random", image_class_labels=None)
    assert idx == [0, 1, 2, 3, 4]
    assert any("exceeds" in r.message for r in caplog.records)


def test_stratified_correct_count() -> None:
    # 20 images, 4 classes: indices 0-4 class {0}, 5-9 class {1}, 10-14 class {2}, 15-19 class {3}
    labels = [frozenset([i // 5]) for i in range(20)]
    idx = resolve_subset_indices(20, 8, seed=0, strategy="stratified", image_class_labels=labels)
    assert len(idx) == 8
    assert idx == sorted(idx)
    assert len(set(idx)) == 8


def test_stratified_preserves_all_classes() -> None:
    labels = [frozenset([i // 5]) for i in range(20)]
    idx = resolve_subset_indices(20, 8, seed=0, strategy="stratified", image_class_labels=labels)
    classes_present = set()
    for i in idx:
        classes_present.update(labels[i])
    assert classes_present == {0, 1, 2, 3}


def test_stratified_none_labels_falls_back_to_random(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.data.subset"):
        idx = resolve_subset_indices(10, 4, seed=0, strategy="stratified", image_class_labels=None)
    assert len(idx) == 4
    assert idx == sorted(idx)
    assert any("stratified" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# SubsetDataset
# ---------------------------------------------------------------------------


class _StubDataset:
    class_names = ["a", "b"]

    def __init__(self, size: int = 10) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, i: int) -> int:  # returns int for simplicity
        return i * 10


def test_subset_dataset_len() -> None:
    inner = _StubDataset(10)
    ds = SubsetDataset(inner, [0, 2, 4])  # type: ignore[arg-type]
    assert len(ds) == 3


def test_subset_dataset_getitem_delegates() -> None:
    inner = _StubDataset(10)
    ds = SubsetDataset(inner, [2, 5, 7])  # type: ignore[arg-type]
    assert ds[0] == 20   # inner[2]
    assert ds[1] == 50   # inner[5]
    assert ds[2] == 70   # inner[7]


def test_subset_dataset_class_names_delegates() -> None:
    inner = _StubDataset()
    ds = SubsetDataset(inner, [0, 1])  # type: ignore[arg-type]
    assert ds.class_names == ["a", "b"]


def test_subset_dataset_satisfies_protocol() -> None:
    from custom_sam_peft.data.base import is_dataset
    from custom_sam_peft.data.base import Example, Instance, TextPrompts
    import torch

    class _ExDataset:
        class_names = ["x"]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, i: int) -> Example:
            return Example(
                image=torch.zeros(3, 8, 8),
                image_id=str(i),
                prompts=TextPrompts(classes=["x"]),
                instances=[],
            )

    ds = SubsetDataset(_ExDataset(), [0])  # type: ignore[arg-type]
    assert is_dataset(ds)
```

- [ ] **Step 2b: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/test_data_subset.py -v -k "resolve_subset or SubsetDataset or subset_dataset"
```

Expected: `ImportError: cannot import name 'SubsetDataset' from 'custom_sam_peft.data.subset'` (module doesn't exist yet).

- [ ] **Step 2c: Create `src/custom_sam_peft/data/subset.py`**

```python
"""Dataset subsetting — pure sampling function + transparent wrapper.

Public API:
  resolve_subset_indices(n_total, limit, *, seed, strategy, image_class_labels)
  SubsetDataset(inner, indices)
"""

from __future__ import annotations

import logging
import random as _random
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from custom_sam_peft.data.base import Dataset

_LOG = logging.getLogger(__name__)


def resolve_subset_indices(
    n_total: int,
    limit: int | float,
    *,
    seed: int,
    strategy: str,
    image_class_labels: Sequence[Sequence[int]] | None,
) -> list[int]:
    """Return sorted-ascending unique indices in [0, n_total).

    Cap resolution:
      int   → min(limit, n_total); warns if limit > n_total.
      float → max(1, round(limit * n_total)); 1.0 yields full range.
    """
    if isinstance(limit, bool):
        raise TypeError("limit must not be a bool")

    # Resolve cap
    if isinstance(limit, int):
        cap = min(limit, n_total)
        if limit > n_total:
            _LOG.warning(
                "limit=%d exceeds dataset size %d; using full dataset",
                limit,
                n_total,
            )
    else:
        cap = max(1, round(limit * n_total))

    if cap >= n_total:
        return list(range(n_total))

    if strategy == "first_n":
        return list(range(cap))

    if strategy == "random":
        return _random_indices(n_total, cap, seed=seed)

    if strategy == "stratified":
        if image_class_labels is None:
            _LOG.warning(
                "stratified subset requested but image_class_labels is None; "
                "falling back to random"
            )
            return _random_indices(n_total, cap, seed=seed)
        return _stratified_indices(n_total, cap, seed=seed, labels=image_class_labels)

    raise ValueError(f"unknown strategy: {strategy!r}")


def _random_indices(n_total: int, cap: int, *, seed: int) -> list[int]:
    rng = _random.Random(f"{seed}:{n_total}:random")
    pool = list(range(n_total))
    rng.shuffle(pool)
    return sorted(pool[:cap])


def _stratified_indices(
    n_total: int,
    cap: int,
    *,
    seed: int,
    labels: Sequence[Sequence[int]],
) -> list[int]:
    """Multi-label proportional sampling (Sechidis et al. 2011 iterative re-weighting).

    Algorithm:
      1. Collect all unique class ids.
      2. Compute desired per-class count: quota[c] = round(cap * freq[c] / n_total).
      3. Greedy: at each step, find the image (not yet selected) whose rarest
         still-needed class has the highest remaining deficit (quota[c] - selected[c]).
         Tie-break by class with smallest current quota, then by image index.
      4. After the greedy pass, if len(selected) < cap, fill from the remaining
         pool using a seeded random draw.
    """
    # Build class → image index mapping
    all_classes: set[int] = set()
    for row in labels:
        all_classes.update(row)
    if not all_classes:
        return _random_indices(n_total, cap, seed=seed)

    class_list = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    n_classes = len(class_list)

    # Per-class frequency in the full dataset
    freq = [0] * n_classes
    for row in labels:
        for c in row:
            freq[class_to_idx[c]] += 1

    # Desired quota per class
    quota = [max(1, round(cap * freq[i] / n_total)) for i in range(n_classes)]

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_per_class = [0] * n_classes

    remaining = list(range(n_total))

    for _ in range(min(cap, n_total)):
        if not remaining:
            break

        best_img = -1
        best_key: tuple[int, int, int] = (0, 0, 0)  # (deficit, -quota, -img_idx) — max by deficit

        for img_idx in remaining:
            img_classes = [class_to_idx[c] for c in labels[img_idx] if c in class_to_idx]
            if not img_classes:
                img_classes = []

            # Find the rarest still-needed class for this image
            deficits = [
                (quota[c] - selected_per_class[c], -quota[c], -img_idx)
                for c in img_classes
                if quota[c] - selected_per_class[c] > 0
            ]
            if not deficits:
                # No class needs more from this image; use a neutral key
                key = (0, 0, -img_idx)
            else:
                key = max(deficits)

            if best_img == -1 or key > best_key:
                best_key = key
                best_img = img_idx

        if best_img == -1:
            break

        selected.append(best_img)
        selected_set.add(best_img)
        for c in labels[best_img]:
            if c in class_to_idx:
                selected_per_class[class_to_idx[c]] += 1
        remaining.remove(best_img)

    # Fill shortfall with random draw from remaining
    if len(selected) < cap and remaining:
        rng = _random.Random(f"{seed}:{n_total}:stratified_fill")
        fill_pool = list(remaining)
        rng.shuffle(fill_pool)
        needed = cap - len(selected)
        selected.extend(fill_pool[:needed])

    return sorted(selected)


class SubsetDataset:
    """Transparent index-mapping wrapper that satisfies the Dataset Protocol.

    The inner dataset never sees the subset — all indexing is at this layer.
    """

    def __init__(self, inner: Dataset, indices: list[int]) -> None:
        self._inner = inner
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Any:
        return self._inner[self._indices[i]]

    @property
    def class_names(self) -> list[str]:
        return self._inner.class_names
```

- [ ] **Step 2d: Run the full test_data_subset.py, confirm all pass**

```bash
uv run pytest tests/unit/test_data_subset.py -v
```

Expected: all tests pass (schema + sampling + delegation).

- [ ] **Step 2e: ruff + mypy**

```bash
uv run ruff check src/custom_sam_peft/data/subset.py tests/unit/test_data_subset.py
uv run mypy src/custom_sam_peft/data/subset.py
```

Expected: both clean.

- [ ] **Step 2f: Commit**

```bash
git add src/custom_sam_peft/data/subset.py tests/unit/test_data_subset.py
git commit -m "feat(data): add resolve_subset_indices (random/stratified/first_n) + SubsetDataset"
```

---

## Task 3: `COCODataset.image_class_labels` (eager property)

**Files:**
- Modify: `src/custom_sam_peft/data/coco.py`
- Create: `tests/unit/test_data_coco_limit.py`

Depends on Task 1. File-disjoint from Tasks 4, 5, 6 — can run in parallel with them after Task 1 lands.

`image_class_labels` is built eagerly at the end of `__init__` once `_ann_index` and `_cat_id_to_dense` are populated. The data is already in memory — it's a dict lookup over `_ann_index`. Returns `list[frozenset[int]]` of dense class IDs, aligned with `_image_ids` order.

- [ ] **Step 3a: Write the failing tests**

Create `tests/unit/test_data_coco_limit.py`:

```python
"""COCODataset.image_class_labels + limit wrapping via _build_dataset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from custom_sam_peft.config.schema import (
    DataSplit,
    LimitConfig,
    NormalizeConfig,
    TextPromptConfig,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_eval_transforms


@pytest.fixture
def coco_ds(tiny_coco_dir: Path) -> COCODataset:
    transforms = build_eval_transforms(32, model_name="facebook/sam3.1", normalize=NormalizeConfig())
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def test_image_class_labels_populated_at_init(coco_ds: COCODataset) -> None:
    labels = coco_ds.image_class_labels
    assert isinstance(labels, list)
    assert len(labels) == len(coco_ds)


def test_image_class_labels_are_frozensets(coco_ds: COCODataset) -> None:
    for entry in coco_ds.image_class_labels:
        assert isinstance(entry, frozenset)
        # All class ids must be valid dense ids (0..C-1)
        for c in entry:
            assert 0 <= c < len(coco_ds.class_names)


def test_image_class_labels_length_matches_image_ids(coco_ds: COCODataset) -> None:
    assert len(coco_ds.image_class_labels) == len(coco_ds)


def test_int_limit_via_build_dataset(tiny_coco_dir: Path) -> None:
    """_build_dataset with an int limit returns a SubsetDataset of the right size."""
    from unittest.mock import MagicMock

    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {
        "format": "coco",
        "train": {"annotations": str(tiny_coco_dir / "annotations.json"),
                  "images": str(tiny_coco_dir / "images")},
        "val": {"annotations": str(tiny_coco_dir / "annotations.json"),
                "images": str(tiny_coco_dir / "images")},
        "prompt_mode": "bbox",
        "image_size": 32,
        "augmentations": {"hflip": False, "color_jitter": 0.0},
        "text_prompt": {"mode": "present", "negatives_per_image": 0, "k": 16},
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "limit": {"train": 1, "val": None, "seed": 42, "strategy": "random"},
    }
    cfg.model.name = "facebook/sam3.1"
    cfg.data.limit.train = 1
    cfg.data.limit.val = None
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = "random"

    ds = _build_dataset(cfg, "train")
    assert isinstance(ds, SubsetDataset)
    assert len(ds) == 1


def test_fraction_limit_rounds_correctly(tiny_coco_dir: Path) -> None:
    """float limit rounds to max(1, round(fraction * n_total))."""
    from unittest.mock import MagicMock

    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {
        "format": "coco",
        "train": {"annotations": str(tiny_coco_dir / "annotations.json"),
                  "images": str(tiny_coco_dir / "images")},
        "val": {"annotations": str(tiny_coco_dir / "annotations.json"),
                "images": str(tiny_coco_dir / "images")},
        "prompt_mode": "bbox",
        "image_size": 32,
        "augmentations": {"hflip": False, "color_jitter": 0.0},
        "text_prompt": {"mode": "present", "negatives_per_image": 0, "k": 16},
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "limit": {"train": 0.5, "val": None, "seed": 0, "strategy": "first_n"},
    }
    cfg.model.name = "facebook/sam3.1"
    cfg.data.limit.train = 0.5
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "first_n"

    # tiny_coco has 2 images; 0.5 * 2 = 1
    ds = _build_dataset(cfg, "train")
    assert isinstance(ds, SubsetDataset)
    assert len(ds) == 1


def test_stratified_limit_preserves_all_classes(tiny_coco_dir: Path) -> None:
    transforms = build_eval_transforms(32, model_name="facebook/sam3.1", normalize=NormalizeConfig())
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )
    from custom_sam_peft.data.subset import resolve_subset_indices

    labels = ds.image_class_labels
    # Use all images (tiny_coco has 2) — just verify the call works
    idx = resolve_subset_indices(
        len(ds), len(ds), seed=0, strategy="stratified", image_class_labels=labels
    )
    assert len(idx) == len(ds)
```

- [ ] **Step 3b: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/test_data_coco_limit.py -v -k "image_class_labels"
```

Expected: `AttributeError: 'COCODataset' object has no attribute 'image_class_labels'`.

- [ ] **Step 3c: Add `image_class_labels` to `COCODataset.__init__` in `coco.py`**

In `src/custom_sam_peft/data/coco.py`, at the end of `COCODataset.__init__` (after the INFO log line at ~line 163), add:

```python
        # Eager per-image class label sets for stratified subset sampling.
        # Built once here because _ann_index is already in memory.
        self.image_class_labels: list[frozenset[int]] = [
            frozenset(
                self._cat_id_to_dense[int(ann["category_id"])]
                for ann in self._ann_index.get(img_id, [])
            )
            for img_id in self._image_ids
        ]
```

- [ ] **Step 3d: Run all coco_limit tests**

```bash
uv run pytest tests/unit/test_data_coco_limit.py -v
```

Expected: `test_image_class_labels_*` tests pass. The `_build_dataset` tests will fail until Task 5 lands — that is expected.

- [ ] **Step 3e: Confirm existing COCO tests still pass**

```bash
uv run pytest tests/unit/test_data_coco.py -v
```

Expected: all pass.

- [ ] **Step 3f: ruff + mypy**

```bash
uv run ruff check src/custom_sam_peft/data/coco.py
uv run mypy src/custom_sam_peft/data/coco.py
```

Expected: clean.

- [ ] **Step 3g: Commit**

```bash
git add src/custom_sam_peft/data/coco.py tests/unit/test_data_coco_limit.py
git commit -m "feat(data): COCODataset.image_class_labels eager property for stratified sampling"
```

---

## Task 4: `HFDataset.image_class_labels` (lazy + cached property)

**Files:**
- Modify: `src/custom_sam_peft/data/hf.py`
- Create: `tests/unit/test_data_hf_limit.py`

Depends on Task 1. File-disjoint from Tasks 3, 5, 6 — can run in parallel with them after Task 1 lands.

The property uses a `_image_class_labels: list[frozenset[int]] | None = None` cache sentinel. On first access it logs exactly one INFO line `"stratified subset: scanning N rows for class labels…"`, then scans every row's `objects.category` field via `_resolve_field`, builds and caches the result. Subsequent calls return the cached list.

- [ ] **Step 4a: Write the failing tests**

Create `tests/unit/test_data_hf_limit.py`:

```python
"""HFDataset.image_class_labels lazy cache tests."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


def _make_hf_dataset(n: int = 4, n_classes: int = 3):
    """Build a minimal HFDataset backed by an in-memory datasets.Dataset."""
    import datasets as hf_datasets

    from custom_sam_peft.config.schema import HFFieldMap, TextPromptConfig
    from custom_sam_peft.data.hf import HFDataset

    rows = {
        "image": [None] * n,
        "objects": [{"bbox": [[0, 0, 1, 1]], "category": [i % n_classes], "segmentation": [None]}
                    for i in range(n)],
    }
    # categories as a Sequence feature (name→value pairs)
    fake_ds = MagicMock()
    fake_ds.__len__ = lambda self: n
    fake_ds.__getitem__ = lambda self, i: {
        "image": None,
        "objects": {"bbox": [[0, 0, 1, 1]], "category": [i % n_classes], "segmentation": [None]},
    }
    fake_ds.features = {"objects": MagicMock()}

    ds = HFDataset.__new__(HFDataset)
    ds._name = "fake"
    ds._split = "train"
    ds._prompt_mode = "bbox"
    ds._transforms = MagicMock()
    ds._text_prompt_cfg = TextPromptConfig()
    ds._field_map = HFFieldMap()
    ds._seed = 0
    ds._multiplex_cap = 16
    ds._warned_truncation = False
    ds._warned_masks_from_boxes = False
    ds._ds = fake_ds
    ds._class_names = [f"cls{i}" for i in range(n_classes)]
    ds._image_class_labels = None  # cache sentinel
    return ds


def test_image_class_labels_not_computed_before_access() -> None:
    ds = _make_hf_dataset()
    assert ds._image_class_labels is None


def test_image_class_labels_computed_on_first_access(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ds = _make_hf_dataset(n=4, n_classes=3)

    def _resolve_field_stub(row, path):
        parts = path.split(".")
        v = row
        for p in parts:
            v = v[p]
        return v

    with patch("custom_sam_peft.data.hf._resolve_field", side_effect=_resolve_field_stub):
        with caplog.at_level(logging.INFO, logger="custom_sam_peft.data.hf"):
            labels = ds.image_class_labels

    assert labels is not None
    assert len(labels) == 4
    assert all(isinstance(s, frozenset) for s in labels)
    assert any("scanning 4 rows" in r.message for r in caplog.records)


def test_image_class_labels_cached_no_second_scan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ds = _make_hf_dataset(n=4, n_classes=3)

    def _resolve_field_stub(row, path):
        parts = path.split(".")
        v = row
        for p in parts:
            v = v[p]
        return v

    with patch("custom_sam_peft.data.hf._resolve_field", side_effect=_resolve_field_stub):
        with caplog.at_level(logging.INFO, logger="custom_sam_peft.data.hf"):
            _ = ds.image_class_labels
            caplog.clear()
            _ = ds.image_class_labels

    # Second access must NOT produce a second scan log
    scan_msgs = [r for r in caplog.records if "scanning" in r.message]
    assert scan_msgs == []


def test_image_class_labels_not_accessed_for_random_strategy() -> None:
    """_build_dataset with random strategy never reads image_class_labels."""
    from unittest.mock import MagicMock

    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=10)
    inner.class_names = ["x"]
    # image_class_labels should NOT be accessed; track via spec
    accessed = []

    type(inner).image_class_labels = property(
        lambda self: accessed.append(True) or [frozenset()]
    )

    cfg = MagicMock()
    cfg.data.format = "hf"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        ds = _build_dataset(cfg, "train")

    assert isinstance(ds, SubsetDataset)
    assert accessed == []  # image_class_labels was never accessed
```

- [ ] **Step 4b: Run tests, confirm failure**

```bash
uv run pytest tests/unit/test_data_hf_limit.py -v -k "not_accessed_for_random"
```

Expected: `AttributeError: 'HFDataset' object has no attribute 'image_class_labels'` (or similar).

- [ ] **Step 4c: Add `image_class_labels` lazy property to `HFDataset` in `hf.py`**

In `src/custom_sam_peft/data/hf.py`, add a cache sentinel field in `__init__` (after `self._warned_masks_from_boxes = False`):

```python
        self._image_class_labels: list[frozenset[int]] | None = None
```

Then add the property method to the `HFDataset` class (after `__len__`):

```python
    @property
    def image_class_labels(self) -> list[frozenset[int]]:
        """Per-image dense class id sets for stratified subset sampling.

        Computed lazily on first access; subsequent accesses return the cache.
        Emits exactly one INFO log per dataset instance when computed.
        """
        if self._image_class_labels is None:
            _LOG.info(
                "stratified subset: scanning %d rows for class labels…",
                len(self._ds),
            )
            cat_field = self._field_map.category
            result: list[frozenset[int]] = []
            for i in range(len(self._ds)):
                row = self._ds[i]
                raw = _resolve_field(row, cat_field)
                if isinstance(raw, list):
                    cats = [int(c) for c in raw]
                else:
                    cats = [int(raw)]
                # Map to dense ids (category values are already dense ints in HF datasets)
                result.append(frozenset(cats))
            self._image_class_labels = result
        return self._image_class_labels
```

- [ ] **Step 4d: Run all hf_limit tests**

```bash
uv run pytest tests/unit/test_data_hf_limit.py -v
```

Expected: `test_image_class_labels_*` tests pass. The `not_accessed_for_random` test will pass once Task 5 lands; mark it as expected to fail for now (it verifies `_build_dataset` behavior).

- [ ] **Step 4e: Confirm existing HF tests still pass**

```bash
uv run pytest tests/unit/test_data_hf.py -v
```

Expected: all pass.

- [ ] **Step 4f: ruff + mypy**

```bash
uv run ruff check src/custom_sam_peft/data/hf.py
uv run mypy src/custom_sam_peft/data/hf.py
```

Expected: clean.

- [ ] **Step 4g: Commit**

```bash
git add src/custom_sam_peft/data/hf.py tests/unit/test_data_hf_limit.py
git commit -m "feat(data): HFDataset.image_class_labels lazy cached property for stratified sampling"
```

---

## Task 5: `_build_dataset` wrapping + INFO log + `subset.json` manifest

**Files:**
- Modify: `src/custom_sam_peft/train/runner.py`
- Create: `tests/unit/test_train_runner_limit.py`

Depends on Tasks 1 and 2. File-disjoint from Tasks 3, 4, 6.

`_build_dataset` checks `cfg.data.limit` and wraps the inner dataset when `limit_val` is not `None`. It accesses `getattr(inner, "image_class_labels", None)` only when `strategy == "stratified"` — this ensures the HF lazy property is never triggered for other strategies. `run_training` writes `subset.json` after both datasets are built, before constructing `Trainer`.

- [ ] **Step 5a: Write the failing tests**

Create `tests/unit/test_train_runner_limit.py`:

```python
"""_build_dataset limit wrapping + subset.json write."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from custom_sam_peft.data.subset import SubsetDataset
from custom_sam_peft.train.runner import _build_dataset, run_training


def _make_stub_inner(size: int = 10) -> MagicMock:
    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=size)
    inner.class_names = ["a", "b"]
    return inner


def _make_cfg(tmp_path: Path, train_limit=None, val_limit=None, strategy="random") -> MagicMock:
    cfg = MagicMock()
    cfg.run.output_dir = str(tmp_path)
    cfg.run.name = "smoke"
    cfg.run.seed = 0
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {"format": "coco"}
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.tracking.backend = "none"
    cfg.tracking.wandb.project = "custom_sam_peft"
    cfg.tracking.wandb.entity = None
    cfg.data.limit.train = train_limit
    cfg.data.limit.val = val_limit
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = strategy
    return cfg


def test_no_limit_returns_inner_dataset_directly() -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = None
    cfg.data.limit.val = None

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        ds = _build_dataset(cfg, "train")

    assert ds is inner  # no wrapping


def test_train_limit_returns_subset_dataset() -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        ds = _build_dataset(cfg, "train")

    assert isinstance(ds, SubsetDataset)
    assert len(ds) == 3


def test_val_limit_only_wraps_val() -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = None
    cfg.data.limit.val = 4
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "first_n"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        train_ds = _build_dataset(cfg, "train")
        val_ds = _build_dataset(cfg, "eval")

    assert train_ds is inner
    assert isinstance(val_ds, SubsetDataset)
    assert len(val_ds) == 4


def test_info_log_emitted_when_limit_applied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inner = _make_stub_inner(10)

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 42
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        with caplog.at_level(logging.INFO, logger="custom_sam_peft.train.runner"):
            _build_dataset(cfg, "train")

    assert any("data.limit applied" in r.message for r in caplog.records)


def test_stratified_strategy_accesses_image_class_labels() -> None:
    inner = _make_stub_inner(10)
    accessed = []
    type(inner).image_class_labels = property(
        lambda self: accessed.append(True) or [frozenset([0])] * 10
    )

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "stratified"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        _build_dataset(cfg, "train")

    assert accessed  # image_class_labels was accessed


def test_random_strategy_does_not_access_image_class_labels() -> None:
    inner = _make_stub_inner(10)
    accessed = []
    type(inner).image_class_labels = property(
        lambda self: accessed.append(True) or [frozenset([0])] * 10
    )

    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.model_dump.return_value = {}
    cfg.model.name = "n/a"
    cfg.data.limit.train = 3
    cfg.data.limit.val = None
    cfg.data.limit.seed = 0
    cfg.data.limit.strategy = "random"

    with patch("custom_sam_peft.train.runner.lookup", return_value=lambda *a, **kw: inner):
        _build_dataset(cfg, "train")

    assert accessed == []


def test_subset_json_written_when_at_least_one_limit_set(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, train_limit=3, val_limit=2)

    def fake_lookup(kind: str, name: str):
        if kind == "peft":
            return lambda wrapper, _peft_cfg: wrapper
        return lambda *a, **kw: _make_stub_inner(10)

    with patch("custom_sam_peft.train.runner.lookup", side_effect=fake_lookup), \
         patch("custom_sam_peft.train.runner.load_sam31", return_value=MagicMock()), \
         patch("custom_sam_peft.train.runner.build_tracker",
               return_value=MagicMock(close=MagicMock(), start_run=MagicMock())), \
         patch("custom_sam_peft.train.runner.Trainer.fit",
               return_value=MagicMock()):
        run_training(cfg)

    # Find the run_dir that was created
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    subset_json = run_dirs[0] / "subset.json"
    assert subset_json.exists()
    data = json.loads(subset_json.read_text())
    assert "limit" in data
    assert "train" in data
    assert "val" in data
    assert data["train"]["n_kept"] == 3
    assert data["val"]["n_kept"] == 2


def test_subset_json_not_written_when_both_limits_none(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, train_limit=None, val_limit=None)

    def fake_lookup(kind: str, name: str):
        if kind == "peft":
            return lambda wrapper, _peft_cfg: wrapper
        return lambda *a, **kw: _make_stub_inner(10)

    with patch("custom_sam_peft.train.runner.lookup", side_effect=fake_lookup), \
         patch("custom_sam_peft.train.runner.load_sam31", return_value=MagicMock()), \
         patch("custom_sam_peft.train.runner.build_tracker",
               return_value=MagicMock(close=MagicMock(), start_run=MagicMock())), \
         patch("custom_sam_peft.train.runner.Trainer.fit",
               return_value=MagicMock()):
        run_training(cfg)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "subset.json").exists()
```

- [ ] **Step 5b: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/test_train_runner_limit.py -v
```

Expected: `test_no_limit_returns_inner_dataset_directly` may pass; the rest fail because `_build_dataset` doesn't import or use `SubsetDataset` yet.

- [ ] **Step 5c: Rewrite `src/custom_sam_peft/train/runner.py`**

Replace the entire file:

```python
"""End-to-end train pipeline. CLI is a thin wrapper over `run_training`."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.subset import SubsetDataset, resolve_subset_indices
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import RunResult, Trainer

_LOG = logging.getLogger(__name__)


def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Returns the created path."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataset(cfg: TrainConfig, pipeline: str) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    inner = cast(Dataset, builder(cfg.data.model_dump(), model_name=cfg.model.name, pipeline=pipeline))

    lim_cfg = cfg.data.limit
    limit_val = lim_cfg.train if pipeline == "train" else lim_cfg.val
    if limit_val is None:
        return inner

    labels = None
    if lim_cfg.strategy == "stratified":
        labels = getattr(inner, "image_class_labels", None)

    indices = resolve_subset_indices(
        len(inner),
        limit_val,
        seed=lim_cfg.seed,
        strategy=lim_cfg.strategy,
        image_class_labels=labels,
    )
    _LOG.info(
        "data.limit applied: %s=%d/%d (strategy=%s, seed=%d)",
        pipeline,
        len(indices),
        len(inner),
        lim_cfg.strategy,
        lim_cfg.seed,
    )
    return SubsetDataset(inner, indices)


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    """Build datasets, load model + PEFT, build tracker, run Trainer.fit."""
    run_dir = make_run_dir(cfg)
    train_ds = _build_dataset(cfg, "train")
    val_ds = _build_dataset(cfg, "eval")

    # Write subset.json when at least one side has a limit applied
    lim_cfg = cfg.data.limit
    if lim_cfg.train is not None or lim_cfg.val is not None:
        _write_subset_manifest(run_dir, train_ds, val_ds, cfg)

    wrapper: Any = load_sam31(cfg.model)
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)


def _write_subset_manifest(
    run_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
) -> None:
    """Write <run_dir>/subset.json recording resolved indices per side."""
    lim_cfg = cfg.data.limit
    manifest: dict[str, Any] = {
        "limit": {
            "train": lim_cfg.train,
            "val": lim_cfg.val,
            "seed": lim_cfg.seed,
            "strategy": lim_cfg.strategy,
        }
    }
    if lim_cfg.train is not None and isinstance(train_ds, SubsetDataset):
        inner_len = len(train_ds._inner)
        manifest["train"] = {
            "n_total": inner_len,
            "n_kept": len(train_ds),
            "indices": train_ds._indices,
        }
    if lim_cfg.val is not None and isinstance(val_ds, SubsetDataset):
        inner_len = len(val_ds._inner)
        manifest["val"] = {
            "n_total": inner_len,
            "n_kept": len(val_ds),
            "indices": val_ds._indices,
        }
    (run_dir / "subset.json").write_text(json.dumps(manifest, indent=2))
```

- [ ] **Step 5d: Run runner limit tests**

```bash
uv run pytest tests/unit/test_train_runner_limit.py -v
```

Expected: all pass.

- [ ] **Step 5e: Confirm existing runner tests still pass**

```bash
uv run pytest tests/unit/test_train_runner.py -v
```

Expected: all pass.

- [ ] **Step 5f: Run coco_limit and hf_limit integration tests that depend on _build_dataset**

```bash
uv run pytest tests/unit/test_data_coco_limit.py tests/unit/test_data_hf_limit.py -v
```

Expected: all pass (Tasks 3 and 4 must already be landed).

- [ ] **Step 5g: ruff + mypy**

```bash
uv run ruff check src/custom_sam_peft/train/runner.py tests/unit/test_train_runner_limit.py
uv run mypy src/custom_sam_peft/train/runner.py
```

Expected: clean.

- [ ] **Step 5h: Commit**

```bash
git add src/custom_sam_peft/train/runner.py tests/unit/test_train_runner_limit.py
git commit -m "feat(train): _build_dataset wraps with SubsetDataset + writes subset.json manifest"
```

---

## Task 6: `DatasetResolution` + `DoctorReport.dataset` + `run_doctor --config`

**Files:**
- Modify: `src/custom_sam_peft/diagnostics.py`
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`
- Create: `tests/unit/test_cli_doctor_config.py`

Depends on Task 5 (calls `_build_dataset`). File-disjoint from Tasks 3, 4 — can run in parallel with them but requires Task 5.

Doctor must never crash. All three failure modes (bad path, schema error, dataset-build error) result in `exit code 0` with the error string in `report.issues` and `dataset=None`.

- [ ] **Step 6a: Write the failing tests**

Create `tests/unit/test_cli_doctor_config.py`:

```python
"""csp doctor --config happy + sad paths."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.doctor_cmd import doctor
from custom_sam_peft.diagnostics import DatasetResolution, DoctorReport, run_doctor

import typer

app = typer.Typer()
app.command()(doctor)

runner = CliRunner()


@pytest.fixture
def valid_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid YAML and patch _build_dataset to return stubs."""
    p = tmp_path / "config.yaml"
    p.write_text(
        """
run:
  name: test
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt

data:
  format: coco
  train:
    annotations: tests/fixtures/tiny_coco/annotations.json
    images: tests/fixtures/tiny_coco/images
  val:
    annotations: tests/fixtures/tiny_coco/annotations.json
    images: tests/fixtures/tiny_coco/images
  prompt_mode: bbox
  limit:
    train: 1
    val: 1
    seed: 0
    strategy: random

peft:
  method: lora
  r: 4
  alpha: 8

train:
  epochs: 1
"""
    )
    return p


def test_doctor_config_happy_path_prints_dataset_section(
    valid_config_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--config: Dataset table appears in rich output."""
    stub_ds = MagicMock()
    stub_ds.__len__ = MagicMock(return_value=2)
    stub_ds.class_names = ["a", "b"]

    with patch("custom_sam_peft.diagnostics._build_dataset", return_value=stub_ds):
        result = runner.invoke(app, ["--config", str(valid_config_yaml)])

    assert result.exit_code == 0, result.output
    assert "Dataset" in result.output


def test_doctor_config_no_config_flag_dataset_none() -> None:
    """Without --config, report.dataset is None."""
    r = run_doctor()
    assert r.dataset is None


def test_doctor_config_bad_path_exit_0_issue_in_report(tmp_path: Path) -> None:
    """Non-existent config path: exit 0, 'config' or 'load' in issues."""
    absent = tmp_path / "no_such.yaml"
    result = runner.invoke(app, ["--config", str(absent)])
    assert result.exit_code == 0
    # Issues should mention the path
    assert str(absent) in result.output or "config" in result.output.lower()


def test_doctor_config_schema_error_exit_0_issue_in_report(tmp_path: Path) -> None:
    """Malformed YAML (schema error): exit 0, error text in output."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("run:\n  name: x\n  seed: not-a-number-for-int: oops\n")
    result = runner.invoke(app, ["--config", str(bad_yaml)])
    assert result.exit_code == 0


def test_doctor_config_build_error_exit_0_couldnt_build(
    valid_config_yaml: Path,
) -> None:
    """_build_dataset raises: exit 0, 'couldn't build' in issues."""
    with patch(
        "custom_sam_peft.diagnostics._build_dataset",
        side_effect=RuntimeError("injected build error"),
    ):
        result = runner.invoke(app, ["--config", str(valid_config_yaml)])

    assert result.exit_code == 0
    assert "couldn't build" in result.output.lower()


def test_doctor_config_json_output_has_dataset_field(
    valid_config_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json: blob["dataset"] has all fields when --config succeeds."""
    stub_ds = MagicMock()
    stub_ds.__len__ = MagicMock(return_value=2)
    stub_ds.class_names = ["a", "b"]

    with patch("custom_sam_peft.diagnostics._build_dataset", return_value=stub_ds):
        result = runner.invoke(app, ["--config", str(valid_config_yaml), "--json"])

    assert result.exit_code == 0, result.output
    blob = json.loads(result.output)
    assert blob["dataset"] is not None
    assert "train_kept" in blob["dataset"]
    assert "val_kept" in blob["dataset"]


def test_doctor_json_output_dataset_null_without_config() -> None:
    """--json without --config: blob['dataset'] is null."""
    result = runner.invoke(app, ["--json"])
    assert result.exit_code == 0
    blob = json.loads(result.output)
    assert blob["dataset"] is None
```

- [ ] **Step 6b: Run tests, confirm they fail**

```bash
uv run pytest tests/unit/test_cli_doctor_config.py -v
```

Expected: `ImportError: cannot import name 'DatasetResolution' from 'custom_sam_peft.diagnostics'`.

- [ ] **Step 6c: Extend `diagnostics.py`**

In `src/custom_sam_peft/diagnostics.py`:

1. Add `DatasetResolution` dataclass before `DoctorReport`:

```python
@dataclass(frozen=True)
class DatasetResolution:
    format: str
    train_total: int
    train_kept: int
    val_total: int
    val_kept: int
    limit_strategy: str
    limit_seed: int
    limit_train: int | float | None
    limit_val: int | float | None
```

2. Add `dataset: DatasetResolution | None = None` field to `DoctorReport` immediately before `issues`:

```python
@dataclass(frozen=True)
class DoctorReport:
    python_version: str
    platform: str
    torch_version: str
    cuda_build: str | None
    cuda_available: bool
    gpus: list[GpuInfo]
    optional_deps: dict[str, str | None]
    core_versions: dict[str, str]
    sam3_weights: WeightsInfo
    hf_auth: HuggingFaceAuthInfo
    dataset: DatasetResolution | None = None
    issues: list[str] = field(default_factory=list)
```

3. Add `_build_dataset_for_doctor` helper and update `run_doctor` signature:

Add this import at the top of `diagnostics.py` (inside the function to avoid circular imports at module load time — use a lazy import pattern):

Then add the helper (placed before `run_doctor`):

```python
def _build_dataset_for_doctor(
    config_path: Path, issues: list[str]
) -> DatasetResolution | None:
    """Load config + build train/val datasets. Returns None and appends to issues on any error.

    Failure modes (all result in return None, exit code 0):
      - Config file not found or bad YAML  → appends "couldn't load config: <msg>"
      - Schema validation error             → appends "couldn't load config: <msg>"
      - Dataset build error                 → appends "couldn't build train/val dataset: <msg>"
    """
    from custom_sam_peft.config.loader import ConfigError, load_config
    from custom_sam_peft.data.subset import SubsetDataset
    from custom_sam_peft.train.runner import _build_dataset

    try:
        cfg = load_config(config_path)
    except (ConfigError, Exception) as e:
        issues.append(f"couldn't load config {config_path}: {e}")
        return None

    try:
        train_ds = _build_dataset(cfg, "train")
        val_ds = _build_dataset(cfg, "eval")
    except Exception as e:
        issues.append(f"couldn't build train/val dataset: {e}")
        return None

    train_total = (
        len(train_ds._inner) if isinstance(train_ds, SubsetDataset) else len(train_ds)
    )
    val_total = (
        len(val_ds._inner) if isinstance(val_ds, SubsetDataset) else len(val_ds)
    )
    lim = cfg.data.limit
    return DatasetResolution(
        format=cfg.data.format,
        train_total=train_total,
        train_kept=len(train_ds),
        val_total=val_total,
        val_kept=len(val_ds),
        limit_strategy=lim.strategy,
        limit_seed=lim.seed,
        limit_train=lim.train,
        limit_val=lim.val,
    )
```

4. Update `run_doctor` signature to accept `config_path`:

```python
def run_doctor(
    *,
    weights_path: Path | None = None,
    config_path: Path | None = None,
) -> DoctorReport:
    """Cheap-to-run environment audit.

    config_path is optional and heavy: loads the YAML, validates the config,
    builds train and val datasets (may trigger pycocotools or datasets.load_dataset).
    The existing no-config path remains cheap and network-free.
    """
    import torch

    issues: list[str] = []

    if sys.version_info < (3, 12):  # noqa: UP036
        issues.append(f"python {sys.version_info.major}.{sys.version_info.minor} < 3.12")

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        issues.append("CUDA not available; training will run on CPU")

    optional = {name: _version_or_none(name) for name in _OPTIONAL}
    core = {name: _required_version(name) for name in _CORE}

    wp = weights_path or _default_weights_path()
    weights = WeightsInfo(
        path=wp,
        exists=wp.is_file(),
        size_bytes=(wp.stat().st_size if wp.is_file() else None),
    )
    if not weights.exists:
        issues.append(f"SAM 3.1 weights not found at {wp}")

    hf_auth = _hf_auth_info()
    if hf_auth.token_source == "none":  # noqa: S105
        issues.append(
            "no HuggingFace token found; gated repos like facebook/sam3.1 "
            "will not download (set HF_TOKEN or run `huggingface-cli login`)"
        )

    dataset_resolution: DatasetResolution | None = None
    if config_path is not None:
        dataset_resolution = _build_dataset_for_doctor(config_path, issues)

    return DoctorReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_build=torch.version.cuda,
        cuda_available=cuda_available,
        gpus=_gpus(),
        optional_deps=optional,
        core_versions=core,
        sam3_weights=weights,
        hf_auth=hf_auth,
        dataset=dataset_resolution,
        issues=issues,
    )
```

- [ ] **Step 6d: Update `cli/doctor_cmd.py`**

Add the `--config` Typer option and the Dataset table renderer. Replace the entire file:

```python
"""`custom-sam-peft doctor` — environment diagnostics formatter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from custom_sam_peft.diagnostics import DoctorReport, run_doctor


def _render_table(report: DoctorReport) -> None:
    console = Console()

    runtime = Table(title="Runtime", show_header=False, box=None)
    runtime.add_row("python", report.python_version)
    runtime.add_row("platform", report.platform)
    runtime.add_row("torch", report.torch_version)
    runtime.add_row("cuda build", report.cuda_build or "(none)")
    runtime.add_row("cuda available", str(report.cuda_available))
    console.print(runtime)

    if report.gpus:
        gpu = Table(title="GPU")
        gpu.add_column("idx")
        gpu.add_column("name")
        gpu.add_column("cap")
        gpu.add_column("free MiB", justify="right")
        gpu.add_column("total MiB", justify="right")
        for g in report.gpus:
            gpu.add_row(
                str(g.index),
                g.name,
                f"{g.capability[0]}.{g.capability[1]}",
                str(g.free_mib),
                str(g.total_mib),
            )
        console.print(gpu)

    opt = Table(title="Optional deps", show_header=False, box=None)
    for name, ver in report.optional_deps.items():
        opt.add_row(name, ver or "(missing)")
    console.print(opt)

    core = Table(title="Core versions", show_header=False, box=None)
    for name, ver in report.core_versions.items():
        core.add_row(name, ver)
    console.print(core)

    w = report.sam3_weights
    weights = Table(title="SAM 3.1 weights", show_header=False, box=None)
    weights.add_row("path", str(w.path))
    weights.add_row("exists", str(w.exists))
    weights.add_row("size", f"{w.size_bytes:,}" if w.size_bytes is not None else "(n/a)")
    console.print(weights)

    hf = report.hf_auth
    auth = Table(title="HuggingFace auth", show_header=False, box=None)
    auth.add_row("token source", hf.token_source)
    auth.add_row("has token", str(hf.has_token))
    console.print(auth)

    if report.dataset is not None:
        ds = report.dataset
        tbl = Table(title="Dataset", show_header=False, box=None)
        tbl.add_row("format", ds.format)
        tbl.add_row("train", f"{ds.train_kept}/{ds.train_total}")
        tbl.add_row("val", f"{ds.val_kept}/{ds.val_total}")
        tbl.add_row("limit.strategy", ds.limit_strategy)
        tbl.add_row("limit.seed", str(ds.limit_seed))
        tbl.add_row("limit.train", str(ds.limit_train))
        tbl.add_row("limit.val", str(ds.limit_val))
        console.print(tbl)

    if report.issues:
        issues = Table(title="Issues", show_header=False, box=None)
        for msg in report.issues:
            issues.add_row("•", msg)
        console.print(issues)


def doctor(
    weights_path: Path | None = typer.Option(
        None, "--weights-path", help="Override SAM 3.1 weights file path."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Load + validate a config YAML and report resolved dataset sizes. "
            "Heavy: may import pycocotools or trigger datasets.load_dataset."
        ),
    ),
) -> None:
    """Report environment + dependency status."""
    report = run_doctor(weights_path=weights_path, config_path=config_path)
    if json_output:
        print(json.dumps(dataclasses.asdict(report), default=str, indent=2))
    else:
        _render_table(report)
```

- [ ] **Step 6e: Run doctor config tests**

```bash
uv run pytest tests/unit/test_cli_doctor_config.py -v
```

Expected: all pass.

- [ ] **Step 6f: Confirm existing diagnostics + doctor CLI tests still pass**

```bash
uv run pytest tests/unit/test_diagnostics.py tests/unit/test_cli_doctor.py -v
```

Expected: all pass. The `dataset=None` default keeps existing DoctorReport tests green.

- [ ] **Step 6g: ruff + mypy**

```bash
uv run ruff check src/custom_sam_peft/diagnostics.py src/custom_sam_peft/cli/doctor_cmd.py tests/unit/test_cli_doctor_config.py
uv run mypy src/custom_sam_peft/diagnostics.py src/custom_sam_peft/cli/doctor_cmd.py
```

Expected: clean.

- [ ] **Step 6h: Commit**

```bash
git add src/custom_sam_peft/diagnostics.py src/custom_sam_peft/cli/doctor_cmd.py tests/unit/test_cli_doctor_config.py
git commit -m "feat(cli): csp doctor --config reports resolved dataset sizes via DatasetResolution"
```

---

## Task 7: Example YAML

**Files:**
- Create: `configs/examples/coco_text_lora_subset.yaml`

File-disjoint from all other tasks. Can run any time after Task 1.

- [ ] **Step 7a: Write the failing test (validate it loads)**

Append to `tests/unit/test_config_examples.py` (or create a new test if that file only has round-trip tests):

```python
def test_coco_text_lora_subset_yaml_validates() -> None:
    """coco_text_lora_subset.yaml must parse cleanly with a non-None limit."""
    import yaml
    from pathlib import Path
    from custom_sam_peft.config.schema import TrainConfig

    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "configs" / "examples" / "coco_text_lora_subset.yaml"
    raw = yaml.safe_load(p.read_text())
    cfg = TrainConfig.model_validate(raw)
    assert cfg.data.limit.train == 64
    assert cfg.data.limit.val == 16
    assert cfg.data.limit.seed == 42
    assert cfg.data.limit.strategy == "random"
```

- [ ] **Step 7b: Run test, confirm it fails (file doesn't exist)**

```bash
uv run pytest tests/unit/test_config_examples.py::test_coco_text_lora_subset_yaml_validates -v
```

Expected: `FileNotFoundError` or similar — the YAML doesn't exist yet.

- [ ] **Step 7c: Create `configs/examples/coco_text_lora_subset.yaml`**

```yaml
# Demonstrates data.limit for fast sanity-check runs on a real COCO dataset.
# Remove or comment out the limit block for full training.
run:
  name: coco-text-lora-subset
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  gradient_checkpointing: false
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
  image_size: 1008
  augmentations:
    hflip: true
    color_jitter: 0.1
  text_prompt:
    mode: present_plus_negatives
    negatives_per_image: 4
  normalize:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
  limit:
    train: 64
    val: 16
    seed: 42
    strategy: random

peft:
  method: lora
  r: 16
  alpha: 32
  dropout: 0.05

train:
  epochs: 10
  batch_size: 1
  grad_accum_steps: 8
  optimizer: auto
  lr: 1.0e-4
  lr_schedule: cosine
  warmup_steps: 100
  max_grad_norm: 1.0
  eval_every: 500
  save_every: 1000
  log_every: 50
  nan_abort_after: 20
  box_hint:
    p_start: 1.0
    p_end: 0.0
    decay_steps: 5000
    early_stop_p_threshold: 0.05
  loss:
    w_mask: 1.0
    w_obj: 1.0
    w_presence: 1.0
    matcher_weights:
      lambda_mask: 5.0

eval:
  metrics: [mAP, mAP_50, mAP_75, per_class_AP]
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

tracking:
  backend: tensorboard

export:
  merge: false
```

- [ ] **Step 7d: Run the test, confirm it passes**

```bash
uv run pytest tests/unit/test_config_examples.py::test_coco_text_lora_subset_yaml_validates -v
```

Expected: PASS.

- [ ] **Step 7e: Commit**

```bash
git add configs/examples/coco_text_lora_subset.yaml tests/unit/test_config_examples.py
git commit -m "feat(config): add coco_text_lora_subset.yaml example with data.limit block"
```

---

## Task 8: Full suite sweep + lint

**Files:**
- No new files.

Run after all tasks are merged to the working branch. This is the final quality gate before requesting a code review.

- [ ] **Step 8a: Run the full unit test suite**

```bash
uv run pytest tests/unit -x -q
```

Expected: all pass. Zero failures.

- [ ] **Step 8b: ruff format + check on all touched files**

```bash
uv run ruff format src/custom_sam_peft/config/schema.py \
    src/custom_sam_peft/data/subset.py \
    src/custom_sam_peft/data/coco.py \
    src/custom_sam_peft/data/hf.py \
    src/custom_sam_peft/train/runner.py \
    src/custom_sam_peft/diagnostics.py \
    src/custom_sam_peft/cli/doctor_cmd.py
uv run ruff check src/custom_sam_peft/ tests/unit/test_data_subset.py \
    tests/unit/test_data_coco_limit.py \
    tests/unit/test_data_hf_limit.py \
    tests/unit/test_train_runner_limit.py \
    tests/unit/test_cli_doctor_config.py
```

Expected: no errors; formatter is a no-op (code was already clean per task steps).

- [ ] **Step 8c: mypy on all touched source files**

```bash
uv run mypy src/custom_sam_peft/config/schema.py \
    src/custom_sam_peft/data/subset.py \
    src/custom_sam_peft/data/coco.py \
    src/custom_sam_peft/data/hf.py \
    src/custom_sam_peft/train/runner.py \
    src/custom_sam_peft/diagnostics.py \
    src/custom_sam_peft/cli/doctor_cmd.py
```

Expected: `Success: no issues found`.

- [ ] **Step 8d: Commit if any formatting fixes were applied**

If ruff format made changes:

```bash
git add -u
git commit -m "style: ruff format sweep after data-subset-limit implementation"
```

---

## Review Checkpoint — after Task 8

Pause for orchestrator code review before opening the PR. Reviewer checks:

- `isinstance(v, bool)` guard precedes numeric check in `LimitConfig._check_limits`.
- `resolve_subset_indices` output is always sorted ascending and within `[0, n_total)`.
- Stratified fill path is seeded and deterministic.
- `_build_dataset` accesses `image_class_labels` only when `strategy == "stratified"`.
- `subset.json` is skipped when both `lim_cfg.train` and `lim_cfg.val` are `None`.
- `run_doctor` with `config_path` never raises; all three failure modes append to `issues` and return `dataset=None`.
- No GPU markers on any new test.
- All existing tests remain green.
