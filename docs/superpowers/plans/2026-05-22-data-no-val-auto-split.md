# No-val mode & train→train/val auto-split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md`](../specs/2026-05-22-data-no-val-auto-split-design.md)
**Issue:** [#71](https://github.com/NguyenJus/custom-sam-peft/issues/71) — *feat(data): support missing validation set — no-val mode + train→train/val auto-split*
**Branch:** `feat/data-no-val-auto-split`

**Goal:** Make `data.val` optional, add `data.val_split` for deterministic Sechidis-style auto-stratification of `data.train` into train/val, persist the resolved split in `<run_dir>/val_source.json`, and degrade the trainer/eval/bundle/CLI/doctor paths gracefully when no validation set exists.

**Architecture:** Two new modules (`data/splitter.py` — pure splitter; `data/val_source.py` — resolver + persistence) feed a private `_resolved_image_ids` dict key into the existing COCO/HF dataset builders. The runner orchestrates resolve → save → inject → build; the trainer accepts `val_ds: Dataset | None` and short-circuits eval, image panels, end-of-run eval, and bundle samples when None. `<run_dir>/val_source.json` is authoritative on resume — saved partitions are never re-stratified. CLI `doctor --config <yaml>` and `csp run` both read the resolved mode from cfg or saved record without re-running the splitter.

**Tech Stack:** Python 3.12, pydantic v2 (strict models with `model_validator`), Sechidis 2011 iterative multi-label stratification (in-tree, ~80 LOC), pycocotools, HuggingFace `datasets`, Typer + rich for CLI, pytest. CPU-only tests — no new GPU tests per `feedback_gpu_vs_cpu_testing`.

---

## File Map

**New files:**

```
src/custom_sam_peft/data/splitter.py                NEW   (~80 LOC)
src/custom_sam_peft/data/val_source.py              NEW   (~150 LOC)
tests/unit/test_splitter.py                         NEW
tests/unit/test_val_source.py                       NEW
tests/unit/test_trainer_no_val.py                   NEW
configs/examples/coco_text_no_val.yaml              NEW
configs/examples/coco_text_auto_split.yaml          NEW
```

**Modified files:**

```
src/custom_sam_peft/config/schema.py                MODIFIED   (lines 129-145)
src/custom_sam_peft/data/coco.py                    MODIFIED   (lines 113-163, 282-317)
src/custom_sam_peft/data/hf.py                      MODIFIED   (lines 125-157, 300-332)
src/custom_sam_peft/train/runner.py                 MODIFIED   (lines 25-46)
src/custom_sam_peft/train/trainer.py                MODIFIED   (lines 127-280)
src/custom_sam_peft/train/loop.py                   MODIFIED   (lines 227-242)
src/custom_sam_peft/eval/runner.py                  MODIFIED   (lines 75-91)
src/custom_sam_peft/runs/bundle.py                  MODIFIED   (lines 262-279)
src/custom_sam_peft/cli/run_cmd.py                  MODIFIED   (lines 34-109)
src/custom_sam_peft/cli/doctor_cmd.py               MODIFIED   (lines 74-86)
src/custom_sam_peft/diagnostics.py                  MODIFIED   (lines 43-55, 118)
src/custom_sam_peft/cli/templates/coco_text_lora.yaml   MODIFIED   (data: block)
src/custom_sam_peft/cli/templates/coco_text_qlora.yaml  MODIFIED   (data: block)
tests/unit/test_config_schema.py                    MODIFIED   (additive)
tests/unit/test_data_coco.py                        MODIFIED   (additive)
tests/unit/test_data_hf.py                          MODIFIED   (additive)
tests/unit/test_train_runner.py                     MODIFIED   (additive)
tests/unit/test_eval_runner.py                      MODIFIED   (additive)
tests/unit/test_cli_doctor.py                       MODIFIED   (additive)
tests/unit/runs/test_bundle.py                      MODIFIED   (additive)
tests/integration/test_train_end_to_end.py          MODIFIED   (additive)
```

No deletions, no moves. The runner's old `_build_dataset` (`runner.py:25-28`) is renamed/replaced by `_build_dataset_from_dict` (signature change is internal to the file). The `cli/run_cmd.py`'s old `_build_val_dataset(cfg)` (line 34) is replaced by `_build_val_dataset(cfg, vs)`.

---

## Dependencies & Parallelization

Steps 1–3 form the pure-module foundation. Step 4 is the adapter layer. Steps 5–10 are wiring/orchestration that all depend on Steps 1–4. Steps 11–12 are config/integration. Step 13 is final verification.

```
Step 1 (splitter) ────┐
                       ├── Step 3 (val_source) ─── Step 4 (adapters) ─┐
Step 2 (schema)   ────┘                                                │
                                                                       ├── Step 5 (runner)
                                                                       ├── Step 6 (trainer/loop)
                                                                       ├── Step 7 (eval)
                                                                       ├── Step 8 (bundle)
                                                                       │
Step 5 (runner) ─── Step 9 (cli/run_cmd)                              │
Step 2 (schema) ─── Step 10 (doctor)                                  │
                                                                       │
                Step 11 (example YAMLs) ◀─── (no code deps)            │
                                                                       │
                                                Step 12 (integration) ◀┘
                                                                       │
                                                Step 13 (final verify) ◀┘
```

**Parallelization opportunities (per `superpowers:dispatching-parallel-agents`):**

- **After Step 1+2+3+4 are merged:** Steps 5, 7, 8 (no shared files) can be dispatched in parallel. Step 6 conflicts with Step 5 on `train/runner.py` interaction *only via test mocks*, not on source files — safe in parallel. Step 9 depends on Step 5's saved-`val_source.json` contract. Step 10 depends on Step 2 only.
- **Step 11** (YAMLs + templates) is file-disjoint with everything except the templates themselves — safe in parallel with any code step.
- **Step 12** (integration) must run after Steps 5–9.
- **Step 13** must be last.

In practice, the orchestrator should serialize **Steps 1 → 2 → 3 → 4** (each builds on prior types), then fan **Steps 5, 6, 7, 8, 10, 11** in parallel, then **Step 9**, then **Step 12**, then **Step 13**.

---

## Pre-flight check

- [ ] **Step 0a: Confirm the worktree state**

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split status --short
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split rev-parse --abbrev-ref HEAD
```

Expected: branch is `feat/data-no-val-auto-split`; the only untracked/modified files are the spec and plan under `docs/superpowers/`.

- [ ] **Step 0b: Confirm baseline unit + integration tests are green**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit -x -q
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/integration -x -q -m "not gpu"
```

Expected: all CPU tests pass. If anything is red, halt and report.

- [ ] **Step 0c: Commit the spec + plan (orchestrator inline, no subagent)**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md \
        docs/superpowers/plans/2026-05-22-data-no-val-auto-split.md
git commit -m "$(cat <<'EOF'
docs(superpowers): spec + plan for no-val mode & train/val auto-split (#71)

Spec proposes (a) making data.val optional, (b) a new data.val_split block
that triggers deterministic Sechidis-style multi-label stratification of
data.train, and (c) <run_dir>/val_source.json as the single source of truth,
authoritative on resume. The trainer, evaluator, bundle, csp run, and
doctor all gain a no-val branch. Plan stages the work across 13 ordered
steps grouped by layer (splitter → schema → resolver → adapters → wiring
→ integration), with explicit dependencies for parallel dispatch.

Refs #71.
EOF
)"
```

---

## Step 1: Splitter (`data/splitter.py`) — pure module + unit tests

**Files:**
- Create: `src/custom_sam_peft/data/splitter.py`
- Create: `tests/unit/test_splitter.py`

**Dispatch:** implementer subagent, sonnet/high. Pure module, no IO, no torch. No dependencies on any other step.

**Spec:** §4 (algorithm), §9.1 (tests).

**Rationale for ordering first:** The splitter is consumed by `data/val_source.py` (Step 3). Implementing it first lets the resolver's tests in Step 3 use the real splitter instead of a mock.

### Task 1a: Write the failing splitter tests

- [ ] **Step 1.1: Create `tests/unit/test_splitter.py`**

Create the file with the seven tests from spec §9.1. Tests are CPU-only, no fixtures required beyond `pytest`. Use this exact content (matches spec §4.1 dataclass shapes):

```python
"""Unit tests for the Sechidis 2011 iterative multi-label stratifier.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §4, §9.1.
"""

from __future__ import annotations

import random

import pytest

from custom_sam_peft.data.splitter import SplitResult, SplittableItem, stratified_split


def _items(spec: list[tuple[str, frozenset[int]]]) -> list[SplittableItem]:
    return [SplittableItem(image_id=iid, class_ids=cls) for iid, cls in spec]


def test_determinism_identical_calls_produce_identical_results() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, fraction=0.2, seed=42)
    b = stratified_split(items, fraction=0.2, seed=42)
    assert a == b


def test_order_independence_shuffle_then_split() -> None:
    items = _items([(str(i), frozenset({i % 5})) for i in range(50)])
    a = stratified_split(items, fraction=0.2, seed=42)
    shuffled = list(items)
    random.Random(123).shuffle(shuffled)
    b = stratified_split(shuffled, fraction=0.2, seed=42)
    assert a == b


def test_realized_fraction_close_to_requested_for_single_class() -> None:
    items = _items([(str(i), frozenset({0})) for i in range(100)])
    res = stratified_split(items, fraction=0.1, seed=42)
    assert abs(res.realized_fraction - 0.1) <= 0.01


def test_multiclass_coverage_rare_class_appears_in_both_sides() -> None:
    # 50 items: classes 0..9. The 47 "bulk" items cover classes 0..8 (i % 9),
    # and 3 "rare" items each carry class 9 alone so n_c=3 for class 9. With
    # fraction=0.2 the quota is `quota_val[9] = round(3 * 0.2) = 1`, so the
    # spec's greedy placement must land at least one rare item in val and the
    # rest in train — i.e. class 9 appears on both sides.
    #
    # NOTE: the rare items are intentionally single-label. The spec's score
    # uses `max(quota[c] for c in item.class_ids)`, so pairing class 9 with an
    # abundant class would let the abundant class dominate scoring and pull
    # every rare item into train. Single-label rare items isolate class 9 in
    # the score, which is what the test is meant to exercise.
    item_specs: list[tuple[str, frozenset[int]]] = []
    for i in range(47):
        item_specs.append((str(i), frozenset({i % 9})))  # classes 0..8
    item_specs.append(("47", frozenset({9})))
    item_specs.append(("48", frozenset({9})))
    item_specs.append(("49", frozenset({9})))
    items = _items(item_specs)
    res = stratified_split(items, fraction=0.2, seed=42)
    train_set = set(res.train_ids)
    val_set = set(res.val_ids)
    rare = {"47", "48", "49"}
    rare_train = bool(rare & train_set)
    rare_val = bool(rare & val_set)
    assert rare_train and rare_val, (
        f"rare class 9 must land in both sides: "
        f"train={rare & train_set}, val={rare & val_set}"
    )


def test_empty_class_set_does_not_crash_and_skips_missing_in_val() -> None:
    items = _items([(str(i), frozenset()) for i in range(5)])
    res = stratified_split(items, fraction=0.2, seed=42)
    assert len(res.train_ids) + len(res.val_ids) == 5
    # Empty-class items must not appear in per_class_counts or missing_in_val.
    assert res.per_class_counts == {}
    assert res.missing_in_val == ()


def test_edge_size_zero() -> None:
    res = stratified_split([], fraction=0.1, seed=42)
    assert res == SplitResult(
        train_ids=(),
        val_ids=(),
        realized_fraction=0.0,
        per_class_counts={},
        missing_in_val=(),
    )


def test_edge_size_one_all_to_train() -> None:
    items = _items([("0", frozenset({0}))])
    res = stratified_split(items, fraction=0.1, seed=42)
    assert res.train_ids == ("0",)
    assert res.val_ids == ()
    assert res.realized_fraction == 0.0


def test_edge_size_two_fraction_half_one_each() -> None:
    items = _items([("0", frozenset({0})), ("1", frozenset({0}))])
    res = stratified_split(items, fraction=0.5, seed=42)
    assert len(res.train_ids) == 1
    assert len(res.val_ids) == 1
    assert set(res.train_ids) | set(res.val_ids) == {"0", "1"}


def test_quota_deviation_records_missing_in_val() -> None:
    # 4 items: class 0 appears 3 times, class 1 appears 1 time (only).
    # With fraction=0.25 and 4 items, V=1; class 1's quota v_c=round(1*0.25)=0,
    # so class 1 has total >= 2 should NOT trigger missing_in_val. We want a
    # case where missing_in_val activates: class 2 has 2 items, fraction=0.1
    # gives v_c=0 → if both placed in train it's still missing_in_val.
    items = _items(
        [
            ("0", frozenset({0})),
            ("1", frozenset({0})),
            ("2", frozenset({0})),
            ("3", frozenset({0})),
            ("4", frozenset({0})),
            ("5", frozenset({2})),
            ("6", frozenset({2})),
        ]
    )
    res = stratified_split(items, fraction=0.1, seed=42)
    # Class 2 has 2 items; with fraction=0.1 the round-down quota is 0 in val.
    # If both class-2 items land in train, class 2 appears in missing_in_val.
    if all(iid in res.train_ids for iid in ("5", "6")):
        assert 2 in res.missing_in_val
    # Realized fraction must be a valid probability.
    assert 0.0 <= res.realized_fraction < 1.0


def test_train_and_val_ids_are_sorted() -> None:
    items = _items([(str(i), frozenset({i % 3})) for i in range(20)])
    res = stratified_split(items, fraction=0.3, seed=42)
    assert list(res.train_ids) == sorted(res.train_ids)
    assert list(res.val_ids) == sorted(res.val_ids)


def test_train_and_val_ids_disjoint() -> None:
    items = _items([(str(i), frozenset({i % 3})) for i in range(20)])
    res = stratified_split(items, fraction=0.3, seed=42)
    assert set(res.train_ids).isdisjoint(set(res.val_ids))
```

- [ ] **Step 1.2: Run the tests to verify they fail with ImportError**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_splitter.py -x -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'custom_sam_peft.data.splitter'`. This proves the test file is collected and is in the right place; the module is what's missing.

### Task 1b: Implement the splitter

- [ ] **Step 1.3: Create `src/custom_sam_peft/data/splitter.py`**

Create the file with the exact dataclasses from spec §4.1 and the Sechidis 2011 algorithm from spec §4.2. Pure module — no IO, no torch.

```python
"""Sechidis 2011 iterative multi-label stratification.

Pure, no IO, no torch. Used by data.val_source to carve a train+val
partition from a list of `SplittableItem`s representing dataset rows.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §4.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SplittableItem:
    """An item (image, HF row, etc.) eligible for stratification.

    `image_id` is an opaque string id (COCO int_id is str(int_id); HF row index
    is str(row_index)). `class_ids` is the dense (post-remap) class ids present
    in this item.
    """

    image_id: str
    class_ids: frozenset[int]


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    realized_fraction: float
    per_class_counts: dict[int, tuple[int, int]]
    missing_in_val: tuple[int, ...]


def stratified_split(
    items: Sequence[SplittableItem],
    fraction: float,
    seed: int,
) -> SplitResult:
    """Carve `items` into a train+val partition via Sechidis 2011 iterative
    multi-label stratification.

    Deterministic given `(items, fraction, seed)`: items are sorted by
    `image_id` before processing so caller ordering does not matter.

    See spec §4.2 for the algorithm and §4.3 for edge-case behavior.
    """
    # 1. Sort input.
    items_sorted = sorted(items, key=lambda it: it.image_id)
    n = len(items_sorted)
    if n == 0:
        return SplitResult(
            train_ids=(),
            val_ids=(),
            realized_fraction=0.0,
            per_class_counts={},
            missing_in_val=(),
        )

    # 2. Quotas.
    v_total = round(n * fraction)
    t_total = n - v_total
    class_totals: dict[int, int] = defaultdict(int)
    for it in items_sorted:
        for c in it.class_ids:
            class_totals[c] += 1
    quota_train: dict[int, int] = {c: nc - round(nc * fraction) for c, nc in class_totals.items()}
    quota_val: dict[int, int] = {c: round(nc * fraction) for c, nc in class_totals.items()}
    remaining = {"train": t_total, "val": v_total}

    # 3. Initial ordering: rarest-class items first, RNG tiebreak.
    rng = random.Random(seed)

    def _min_class_count(it: SplittableItem) -> float:
        if not it.class_ids:
            return math.inf
        return float(min(class_totals[c] for c in it.class_ids))

    decorated = [(_min_class_count(it), rng.random(), it) for it in items_sorted]
    decorated.sort(key=lambda t: (t[0], t[1]))

    # 4. Greedy placement.
    train_ids: list[str] = []
    val_ids: list[str] = []

    def _score(side: str, it: SplittableItem) -> int:
        quota = quota_train if side == "train" else quota_val
        if not it.class_ids:
            return remaining[side]
        return max(quota[c] for c in it.class_ids)

    for _min_c, _tiebreak, it in decorated:
        if remaining["train"] == 0:
            chosen = "val"
        elif remaining["val"] == 0:
            chosen = "train"
        else:
            s_t = _score("train", it)
            s_v = _score("val", it)
            if s_t > s_v:
                chosen = "train"
            elif s_v > s_t:
                chosen = "val"
            else:
                # Tie on score: prefer side with larger remaining capacity.
                if remaining["train"] > remaining["val"]:
                    chosen = "train"
                elif remaining["val"] > remaining["train"]:
                    chosen = "val"
                else:
                    # Still tied: seeded coin flip.
                    chosen = "train" if rng.random() < 0.5 else "val"
        (train_ids if chosen == "train" else val_ids).append(it.image_id)
        remaining[chosen] -= 1
        quota = quota_train if chosen == "train" else quota_val
        for c in it.class_ids:
            if quota[c] > 0:
                quota[c] -= 1

    # 5. Post-checks.
    realized_fraction = len(val_ids) / max(n, 1)
    per_class_counts: dict[int, tuple[int, int]] = {}
    train_set = set(train_ids)
    val_set = set(val_ids)
    for c, total in class_totals.items():
        t_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in train_set)
        v_count = sum(1 for it in items_sorted if c in it.class_ids and it.image_id in val_set)
        per_class_counts[c] = (t_count, v_count)
    missing_in_val = tuple(
        sorted(c for c, (t, v) in per_class_counts.items() if class_totals[c] >= 2 and v == 0)
    )

    return SplitResult(
        train_ids=tuple(sorted(train_ids)),
        val_ids=tuple(sorted(val_ids)),
        realized_fraction=realized_fraction,
        per_class_counts=per_class_counts,
        missing_in_val=missing_in_val,
    )
```

- [ ] **Step 1.4: Run the splitter tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_splitter.py -x -v
```

Expected: 11 tests pass (determinism, order-independence, single-class realized fraction, multiclass coverage, empty class set, three edge sizes, quota deviation, sorted, disjoint).

- [ ] **Step 1.5: Commit Step 1**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/data/splitter.py tests/unit/test_splitter.py
git commit -m "feat(data): add Sechidis multi-label stratifier for auto-split (#71)"
```

---

## Step 2: Schema (`config/schema.py`) — `ValSplitConfig`, optional `val`, two validators

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (lines 129-145 region)
- Modify: `tests/unit/test_config_schema.py` (append)

**Dispatch:** implementer subagent, sonnet/high. Independent of Step 1. May run in parallel with Step 1.

**Spec:** §3 (schema), §9.3 (tests).

### Task 2a: Write failing schema tests

- [ ] **Step 2.1: Append the 7 schema tests to `tests/unit/test_config_schema.py`**

After the existing tests (the file ends at line 227), append:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): optional val + val_split + validators
# ---------------------------------------------------------------------------


def test_val_null_validates() -> None:
    """data.val: null resolves to no-val mode; must not raise."""
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val is None
    assert cfg.data.val_split is None


def test_val_omitted_validates() -> None:
    """Omitting data.val entirely also resolves to no-val mode."""
    d = _minimal_dict()
    del d["data"]["val"]  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val is None
    assert cfg.data.val_split is None


def test_val_and_val_split_mutually_exclusive() -> None:
    d = _minimal_dict()
    d["data"]["val_split"] = {"fraction": 0.1}  # type: ignore[index]
    # val is still present from _minimal_dict.
    with pytest.raises(ValidationError, match="mutually exclusive"):
        TrainConfig.model_validate(d)


def test_val_split_fraction_above_half_rejected() -> None:
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.6}  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_val_split_fraction_zero_or_negative_rejected() -> None:
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.0}  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)
    d["data"]["val_split"] = {"fraction": -0.1}  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_hf_split_val_custom_with_val_split_rejected() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {  # type: ignore[index]
        "name": "tiny/dataset",
        "split_train": "train",
        "split_val": "custom_val",
    }
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.1}  # type: ignore[index]
    with pytest.raises(ValidationError, match="split_val cannot be customized"):
        TrainConfig.model_validate(d)


def test_hf_split_val_default_with_val_split_validates() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {"name": "tiny/dataset"}  # default split_val="validation"
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.1, "seed": 7}  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val_split is not None
    assert cfg.data.val_split.fraction == 0.1
    assert cfg.data.val_split.seed == 7


def test_neither_val_nor_val_split_validates() -> None:
    """Spec §3.3: neither set → resolves to no-val mode (WARN at resolve, not validation)."""
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    # val_split is not present in _minimal_dict.
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val is None
    assert cfg.data.val_split is None
```

- [ ] **Step 2.2: Run the appended tests to verify they fail**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_config_schema.py -x -v -k "val_null or val_omitted or val_and_val_split or val_split_fraction or hf_split_val or neither_val"
```

Expected: most tests fail because the schema doesn't have `val_split` yet and rejects `val: null`. Some may pass for the wrong reason — that's fine, they will pass for the right reason after Step 2.3.

### Task 2b: Implement the schema changes

- [ ] **Step 2.3: Modify `src/custom_sam_peft/config/schema.py`**

Insert a new `ValSplitConfig` class immediately before the `DataConfig` class (currently at line 129). After the existing `HFDatasetConfig` (line 120-127), add:

```python
class ValSplitConfig(_Strict):
    """Auto-split parameters. Used when DataConfig.val_split is set.

    Carves data.train into train+val deterministically. In v0:
      - stratification is always-on Sechidis multi-label iterative;
        not configurable.
      - split unit is always 'image'; not configurable. Splitting by
        annotation can leak the same image into both sides.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §3.1.
    """

    fraction: float = Field(default=0.1, gt=0.0, le=0.5)
    seed: int | None = None  # None → inherit run.seed at resolve time
```

Then modify the existing `DataConfig` (lines 129-145). Replace the whole class body with:

```python
class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit | None = None
    val_split: ValSplitConfig | None = None
    test: DataSplit | None = None
    hf: HFDatasetConfig | None = None
    prompt_mode: PromptMode
    image_size: PositiveInt = 1024
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    text_prompt: TextPromptConfig = Field(default_factory=TextPromptConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)

    @model_validator(mode="after")
    def _check_format_specific(self) -> DataConfig:
        if self.format == "hf" and self.hf is None:
            raise ValueError("data.hf is required when data.format == 'hf'")
        return self

    @model_validator(mode="after")
    def _check_val_modes(self) -> DataConfig:
        if self.val is not None and self.val_split is not None:
            raise ValueError(
                "data.val and data.val_split are mutually exclusive. "
                "Set one to provide a validation set, neither for no-val mode."
            )
        return self

    @model_validator(mode="after")
    def _check_hf_split_val_compat(self) -> DataConfig:
        if (
            self.format == "hf"
            and self.val_split is not None
            and self.hf is not None
            and self.hf.split_val != "validation"
        ):
            raise ValueError(
                "data.hf.split_val cannot be customized when data.val_split is set; "
                "auto-split carves the val set from data.hf.split_train. "
                "Remove split_val or remove val_split."
            )
        return self
```

The two key changes: `val: DataSplit` → `val: DataSplit | None = None`; new field `val_split: ValSplitConfig | None = None`; two new `model_validator(mode="after")` methods.

- [ ] **Step 2.4: Run the schema tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_config_schema.py -x -v
```

Expected: all original tests + 8 new tests pass.

- [ ] **Step 2.5: Spot-check the `_minimal_dict` baseline still has `val`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && grep -A3 "_minimal_dict" tests/unit/test_config_schema.py | head -20
```

Expected: `_minimal_dict()` still defines `data.val`. The new tests modify `d` per-test.

- [ ] **Step 2.6: Commit Step 2**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(config): make data.val optional + add ValSplitConfig (#71)"
```

---

## Step 3: Resolver (`data/val_source.py`) — types, enumeration helpers, `resolve_val_source`, persistence

**Files:**
- Create: `src/custom_sam_peft/data/val_source.py`
- Create: `tests/unit/test_val_source.py`

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Step 1 (splitter), Step 2 (schema).

**Spec:** §5 (resolver), §9.2 (tests).

### Task 3a: Write failing resolver tests

- [ ] **Step 3.1: Create `tests/unit/test_val_source.py`**

```python
"""Unit tests for data/val_source.py — resolver + persistence.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §5, §9.2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    HFDatasetConfig,
    HFFieldMap,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
    ValSplitConfig,
)
from custom_sam_peft.data.val_source import (
    ValSource,
    _enumerate_coco_items,
    load_val_source,
    resolve_val_source,
    save_val_source,
)


def _base_cfg(tiny_coco_dir: Path, *, val: bool, val_split: bool) -> TrainConfig:
    """Build a TrainConfig pinned at tiny_coco; flags pick the resolved mode."""
    return TrainConfig(
        run=RunConfig(name="r", output_dir="./runs", seed=7),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=(
                DataSplit(
                    annotations=str(tiny_coco_dir / "annotations.json"),
                    images=str(tiny_coco_dir / "images"),
                )
                if val
                else None
            ),
            val_split=(ValSplitConfig(fraction=0.5, seed=None) if val_split else None),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )


def test_resolve_mode_explicit(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=True, val_split=False)
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.mode == "explicit"
    assert vs.train_ids is None
    assert vs.val_ids is None


def test_resolve_mode_auto_split(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.mode == "auto_split"
    assert vs.train_ids is not None
    assert vs.val_ids is not None
    assert vs.seed_used == 7  # inherited from run.seed
    assert vs.fraction_requested == 0.5
    # Tiny COCO has 2 keep-after-crowd-filter images; fraction=0.5 yields 1+1.
    assert len(vs.train_ids) + len(vs.val_ids) == 2


def test_resolve_mode_none_warns(
    tiny_coco_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=False)
    with caplog.at_level(logging.WARNING):
        vs = resolve_val_source(cfg, run_dir=None)
    assert vs.mode == "none"
    assert vs.train_ids is None
    assert vs.val_ids is None
    # No WARN emitted by the resolver itself; the trainer-side _log_val_source
    # is where the WARN happens. The resolver may emit INFO only.
    # (Spec §5.2 case 4: resolver returns ValSource(mode='none'); §5.3 logs WARN
    # at training start via _log_val_source.) Adjust if implementation logs here.


def test_log_val_source_warns_for_none(
    tiny_coco_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Spec §5.3: _log_val_source emits the no-val WARN."""
    from custom_sam_peft.data.val_source import _log_val_source

    vs = ValSource(
        mode="none",
        train_ids=None,
        val_ids=None,
        realized_fraction=None,
        per_class_counts=None,
        missing_in_val=None,
        fraction_requested=None,
        seed_used=None,
    )
    with caplog.at_level(logging.WARNING):
        _log_val_source(vs)
    assert any("no-op" in r.message or "no validation" in r.message.lower() for r in caplog.records)


def test_save_and_load_round_trip_auto_split(
    tiny_coco_dir: Path, tmp_path: Path
) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    assert (tmp_path / "val_source.json").is_file()
    loaded = load_val_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == vs.mode
    assert loaded.train_ids == vs.train_ids
    assert loaded.val_ids == vs.val_ids
    assert loaded.realized_fraction == vs.realized_fraction
    assert loaded.fraction_requested == vs.fraction_requested
    assert loaded.seed_used == vs.seed_used
    # per_class_counts JSON-serializes int keys as strings; loader must re-cast.
    if loaded.per_class_counts is not None:
        for k in loaded.per_class_counts:
            assert isinstance(k, int)


def test_save_and_load_round_trip_explicit(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=True, val_split=False)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    loaded = load_val_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == "explicit"
    assert loaded.train_ids is None
    assert loaded.val_ids is None


def test_save_and_load_round_trip_none(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=False)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    loaded = load_val_source(tmp_path)
    assert loaded is not None
    assert loaded.mode == "none"


def test_load_val_source_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_val_source(tmp_path) is None


def test_resume_preference_loads_saved_record(
    tiny_coco_dir: Path, tmp_path: Path
) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs_first = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs_first, tmp_path)

    # Now change fraction in cfg; resolver MUST return the saved record.
    cfg2 = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    assert cfg2.data.val_split is not None
    # Mutate frozen via model_copy
    cfg2 = cfg2.model_copy(
        update={"data": cfg2.data.model_copy(update={"val_split": ValSplitConfig(fraction=0.1)})}
    )
    vs_loaded = resolve_val_source(cfg2, run_dir=tmp_path)
    assert vs_loaded.fraction_requested == 0.5  # the SAVED fraction, not the new one.


def test_coco_enumeration_excludes_crowd_only(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=False)
    items = _enumerate_coco_items(cfg.data)
    # tiny_coco's surviving images all have class ids; no empty class_ids.
    assert len(items) >= 1
    for it in items:
        assert isinstance(it.image_id, str)
        assert isinstance(it.class_ids, frozenset)


def test_seed_inheritance_from_run_seed(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    # val_split.seed is None in _base_cfg → must inherit run.seed (7).
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.seed_used == 7


def test_seed_override_explicit_seed(tiny_coco_dir: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"val_split": ValSplitConfig(fraction=0.5, seed=99)})
        }
    )
    vs = resolve_val_source(cfg, run_dir=None)
    assert vs.seed_used == 99


def test_atomic_save_does_not_leave_tmp_file(tiny_coco_dir: Path, tmp_path: Path) -> None:
    cfg = _base_cfg(tiny_coco_dir, val=False, val_split=True)
    vs = resolve_val_source(cfg, run_dir=None)
    save_val_source(vs, tmp_path)
    # No leftover .tmp file from the os.replace flow.
    assert not (tmp_path / "val_source.json.tmp").exists()
    assert (tmp_path / "val_source.json").is_file()
```

- [ ] **Step 3.2: Run the resolver tests; verify they fail with ImportError**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_val_source.py -x -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'custom_sam_peft.data.val_source'`.

### Task 3b: Implement the resolver

- [ ] **Step 3.3: Create `src/custom_sam_peft/data/val_source.py`**

```python
"""Resolve the validation source for a run: explicit, auto_split, or none.

The resolver is the single seam between schema and the splitter. It also
owns persistence (`save_val_source` / `load_val_source`) of the resolved
record to `<run_dir>/val_source.json`. Trainer hparams logging and tracker
hparams injection both read the saved record, so the resolver writes the
authoritative copy once per run (before Trainer.fit begins).

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §5.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from custom_sam_peft.config.schema import DataConfig, TrainConfig
from custom_sam_peft.data.splitter import SplittableItem, stratified_split

_LOG = logging.getLogger(__name__)

ValMode = Literal["explicit", "auto_split", "none"]


@dataclass(frozen=True)
class ValSource:
    mode: ValMode
    train_ids: tuple[str, ...] | None
    val_ids: tuple[str, ...] | None
    realized_fraction: float | None
    per_class_counts: dict[int, tuple[int, int]] | None
    missing_in_val: tuple[int, ...] | None
    fraction_requested: float | None
    seed_used: int | None


def resolve_val_source(cfg: TrainConfig, *, run_dir: Path | None = None) -> ValSource:
    """Resolve which validation source to use for this run.

    Dispatch (spec §5.2):
      1. run_dir/val_source.json exists → load_val_source(run_dir) (resume).
      2. cfg.data.val_split is not None → enumerate + stratify.
      3. cfg.data.val is not None → mode='explicit'.
      4. else → mode='none'.
    """
    if run_dir is not None:
        saved = load_val_source(run_dir)
        if saved is not None:
            _LOG.info("val_source: resumed from %s (mode=%s)", run_dir, saved.mode)
            return saved

    if cfg.data.val_split is not None:
        seed_used = (
            cfg.data.val_split.seed
            if cfg.data.val_split.seed is not None
            else cfg.run.seed
        )
        items = _enumerate_items(cfg.data)
        result = stratified_split(items, cfg.data.val_split.fraction, seed_used)
        return ValSource(
            mode="auto_split",
            train_ids=result.train_ids,
            val_ids=result.val_ids,
            realized_fraction=result.realized_fraction,
            per_class_counts=result.per_class_counts,
            missing_in_val=result.missing_in_val,
            fraction_requested=cfg.data.val_split.fraction,
            seed_used=seed_used,
        )

    if cfg.data.val is not None:
        return ValSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            fraction_requested=None,
            seed_used=None,
        )

    return ValSource(
        mode="none",
        train_ids=None,
        val_ids=None,
        realized_fraction=None,
        per_class_counts=None,
        missing_in_val=None,
        fraction_requested=None,
        seed_used=None,
    )


def save_val_source(vs: ValSource, run_dir: Path) -> None:
    """Write `<run_dir>/val_source.json`. Atomic via tmp + os.replace."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "mode": vs.mode,
        "fraction_requested": vs.fraction_requested,
        "seed_used": vs.seed_used,
        "realized_fraction": vs.realized_fraction,
        "n_train": (len(vs.train_ids) if vs.train_ids is not None else None),
        "n_val": (len(vs.val_ids) if vs.val_ids is not None else None),
        "per_class_counts": (
            {str(k): list(v) for k, v in vs.per_class_counts.items()}
            if vs.per_class_counts is not None
            else None
        ),
        "missing_in_val": (list(vs.missing_in_val) if vs.missing_in_val is not None else None),
        "train_ids": (list(vs.train_ids) if vs.train_ids is not None else None),
        "val_ids": (list(vs.val_ids) if vs.val_ids is not None else None),
    }
    tmp = run_dir / "val_source.json.tmp"
    final = run_dir / "val_source.json"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, final)


def load_val_source(run_dir: Path) -> ValSource | None:
    """Read `<run_dir>/val_source.json`. Returns None if missing."""
    p = run_dir / "val_source.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    per_class_raw = raw.get("per_class_counts")
    per_class: dict[int, tuple[int, int]] | None = (
        {int(k): (int(v[0]), int(v[1])) for k, v in per_class_raw.items()}
        if per_class_raw is not None
        else None
    )
    train_ids_raw = raw.get("train_ids")
    val_ids_raw = raw.get("val_ids")
    missing_raw = raw.get("missing_in_val")
    return ValSource(
        mode=raw["mode"],
        train_ids=(tuple(train_ids_raw) if train_ids_raw is not None else None),
        val_ids=(tuple(val_ids_raw) if val_ids_raw is not None else None),
        realized_fraction=raw.get("realized_fraction"),
        per_class_counts=per_class,
        missing_in_val=(tuple(missing_raw) if missing_raw is not None else None),
        fraction_requested=raw.get("fraction_requested"),
        seed_used=raw.get("seed_used"),
    )


def _log_val_source(vs: ValSource) -> None:
    """Emit the INFO/WARN log lines documented in spec §4.5 / §5.3."""
    if vs.mode == "explicit":
        _LOG.info("val source: explicit (cfg.data.val)")
        return
    if vs.mode == "auto_split":
        assert vs.train_ids is not None and vs.val_ids is not None
        assert vs.realized_fraction is not None and vs.fraction_requested is not None
        assert vs.per_class_counts is not None
        n_train, n_val = len(vs.train_ids), len(vs.val_ids)
        pct = 100.0 * vs.realized_fraction
        covered = sum(1 for (_t, v) in vs.per_class_counts.values() if v > 0)
        total_classes = len(vs.per_class_counts)
        _LOG.info(
            "val source: auto-split fraction=%.2f, realized=train=%d/val=%d (%.2f%%); "
            "coverage=%d/%d classes in val",
            vs.fraction_requested,
            n_train,
            n_val,
            pct,
            covered,
            total_classes,
        )
        if vs.missing_in_val:
            _LOG.warning(
                "auto-split: %d classes missing from val: %s",
                len(vs.missing_in_val),
                list(vs.missing_in_val),
            )
        if (
            abs(vs.realized_fraction - vs.fraction_requested) / vs.fraction_requested > 0.2
            or n_val < 8
        ):
            _LOG.warning(
                "auto-split: realized fraction deviates from requested or val is small"
            )
        return
    # mode == "none"
    _LOG.warning(
        "training without validation set; eval_every is a no-op, end-of-run "
        "eval and bundle samples are skipped. Use data.val to provide one or "
        "data.val_split to auto-split."
    )


def _enumerate_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Dispatch to the right per-format enumerator."""
    if data_cfg.format == "coco":
        return _enumerate_coco_items(data_cfg)
    if data_cfg.format == "hf":
        return _enumerate_hf_items(data_cfg)
    raise ValueError(f"unknown data.format: {data_cfg.format!r}")


def _enumerate_coco_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Walk data.train COCO annotations and return SplittableItems.

    Reuses _load_coco_index / _build_category_remap / _drop_crowd_only_images
    from data/coco.py. Each SplittableItem.image_id is str(int_image_id);
    class_ids is the frozenset of dense ids present after crowd filtering.
    """
    from custom_sam_peft.data.coco import (
        _build_category_remap,
        _drop_crowd_only_images,
        _load_coco_index,
    )

    coco = _load_coco_index(data_cfg.train.annotations)
    _sparse, sparse_to_dense, _names = _build_category_remap(coco)
    kept, ann_index, _dropped = _drop_crowd_only_images(coco)
    items: list[SplittableItem] = []
    for image_id in kept:
        anns = ann_index[image_id]
        class_ids = frozenset(sparse_to_dense[int(a["category_id"])] for a in anns)
        items.append(SplittableItem(image_id=str(image_id), class_ids=class_ids))
    return items


def _enumerate_hf_items(data_cfg: DataConfig) -> list[SplittableItem]:
    """Walk data.hf.split_train and return SplittableItems.

    image_id = str(row_index). class_ids is the frozenset of int category ids
    in the row's data.hf.field_map.category field.
    """
    from custom_sam_peft.data.hf import _resolve_field, hf_load_dataset

    assert data_cfg.hf is not None
    ds = hf_load_dataset(data_cfg.hf.name, split=data_cfg.hf.split_train)
    items: list[SplittableItem] = []
    for i in range(len(ds)):
        row = ds[i]
        try:
            classes_raw = _resolve_field(row, data_cfg.hf.field_map.category)
        except KeyError:
            classes_raw = []
        class_ids = frozenset(int(c) for c in classes_raw)
        items.append(SplittableItem(image_id=str(i), class_ids=class_ids))
    return items
```

- [ ] **Step 3.4: Run the resolver tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_val_source.py -x -v
```

Expected: all tests pass. If `test_resolve_mode_none_warns` fails because the resolver does (or does not) emit a WARN, adjust the test to match the spec — `_log_val_source` is where the no-val WARN happens, not the resolver itself. Both behaviors are tested separately.

- [ ] **Step 3.5: Commit Step 3**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/data/val_source.py tests/unit/test_val_source.py
git commit -m "feat(data): add val_source resolver + per-run persistence (#71)"
```

---

## Step 4: Adapter ctor extensions — `COCODataset.image_ids` / `HFDataset.row_indices`

**Files:**
- Modify: `src/custom_sam_peft/data/coco.py` (lines 113-163 ctor; lines 282-317 builder)
- Modify: `src/custom_sam_peft/data/hf.py` (lines 125-157 ctor; lines 300-332 builder)
- Modify: `tests/unit/test_data_coco.py` (append)
- Modify: `tests/unit/test_data_hf.py` (append)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Step 1 (splitter — for the leak-invariant test).

**Spec:** §6.1 (COCO), §6.2 (HF), §6.3 (builder injection), §9.4 (tests).

### Task 4a: Failing adapter integration tests

- [ ] **Step 4.1: Append the 5 COCO adapter tests to `tests/unit/test_data_coco.py`**

Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): image_ids subset parameter
# ---------------------------------------------------------------------------


def test_cocodataset_image_ids_filters_to_subset(tiny_coco_dir: Path) -> None:
    """Spec §6.1: image_ids restricts the dataset to the requested subset."""
    with _patch_imagenet_ctx():
        full = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
    all_ids = list(full._image_ids)  # noqa: SLF001 — internal use, test only
    assert len(all_ids) >= 2
    subset = all_ids[:1]
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=subset,
        )
    assert len(ds) == 1
    ex = ds[0]
    assert int(ex.image_id) == subset[0]


def test_cocodataset_image_ids_missing_raises_value_error(tiny_coco_dir: Path) -> None:
    """Spec §6.1: requesting an image_id not present (or crowd-only) raises ValueError."""
    with _patch_imagenet_ctx(), pytest.raises(ValueError, match="not present"):
        COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=[999999],
        )


def test_cocodataset_image_ids_none_preserves_existing_behavior(tiny_coco_dir: Path) -> None:
    """When image_ids is None, the dataset behaves exactly as before."""
    with _patch_imagenet_ctx():
        ds_a = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
        ds_b = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=None,
        )
    assert len(ds_a) == len(ds_b)


def test_cocodataset_image_ids_sorted_order_preserved(tiny_coco_dir: Path) -> None:
    """Internal _image_ids list must be in ascending order regardless of caller-supplied order."""
    with _patch_imagenet_ctx():
        full = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
        )
    ids_sorted_desc = sorted(full._image_ids, reverse=True)  # noqa: SLF001
    with _patch_imagenet_ctx():
        ds = COCODataset(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            image_ids=ids_sorted_desc,
        )
    assert ds._image_ids == sorted(ds._image_ids)  # noqa: SLF001


def test_image_level_leak_invariant_on_tiny_coco(tiny_coco_dir: Path) -> None:
    """Spec §9.4.5: stratified_split on tiny_coco items yields disjoint train/val ids."""
    from custom_sam_peft.config.schema import DataConfig, DataSplit
    from custom_sam_peft.data.splitter import stratified_split
    from custom_sam_peft.data.val_source import _enumerate_coco_items

    data_cfg = DataConfig(
        format="coco",
        train=DataSplit(
            annotations=str(tiny_coco_dir / "annotations.json"),
            images=str(tiny_coco_dir / "images"),
        ),
        prompt_mode="text",
        image_size=32,
    )
    items = _enumerate_coco_items(data_cfg)
    if len(items) < 2:
        pytest.skip("tiny_coco has < 2 keep-after-crowd-filter images; cannot test split")
    res = stratified_split(items, fraction=0.5, seed=0)
    assert set(res.train_ids).isdisjoint(set(res.val_ids))
```

- [ ] **Step 4.2: Append the 3 HF adapter tests to `tests/unit/test_data_hf.py`**

Use the file's existing `_patch_load_dataset(monkeypatch, ds)` helper (already at line 132). Append at the end:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): row_indices subset parameter
# ---------------------------------------------------------------------------


def test_hfdataset_row_indices_filters_to_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §6.2: row_indices restricts the dataset to the requested rows."""
    ds_underlying = _make_min_ds_with_class_label(n=5)
    _patch_load_dataset(monkeypatch, ds_underlying)
    ds = HFDataset(
        name="x",
        split="train",
        prompt_mode="text",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(),
        row_indices=[0, 2],
    )
    assert len(ds) == 2


def test_hfdataset_row_indices_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §6.2: out-of-range row_indices raise ValueError."""
    ds_underlying = _make_min_ds_with_class_label(n=3)
    _patch_load_dataset(monkeypatch, ds_underlying)
    with pytest.raises(ValueError, match="out of range"):
        HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(),
            row_indices=[-1],
        )
    with pytest.raises(ValueError, match="out of range"):
        HFDataset(
            name="x",
            split="train",
            prompt_mode="text",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(),
            row_indices=[100],
        )


def test_hfdataset_image_id_uses_underlying_row_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §6.2 last paragraph: image_id in the returned Example uses the
    underlying dataset row index (not the post-subset position)."""
    ds_underlying = _make_min_ds_with_class_label(n=5)
    _patch_load_dataset(monkeypatch, ds_underlying)
    ds = HFDataset(
        name="x",
        split="train",
        prompt_mode="text",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(),
        row_indices=[2, 4],
    )
    ex0 = ds[0]
    # Subset position 0 → underlying row 2 → image_id == "2".
    assert ex0.image_id == "2"
    ex1 = ds[1]
    assert ex1.image_id == "4"
```

Notes for the HF subagent:
- The test names `_make_min_ds_with_class_label` and `_patch_load_dataset` and `_build_eval` are existing helpers in `tests/unit/test_data_hf.py`. If they have different names, look them up at the top of that file (around lines 60-140) and substitute.
- If a `_make_min_ds_with_class_label(n=...)` helper does not exist, add one or extend an existing fixture builder to take an `n` count parameter. Be additive.

- [ ] **Step 4.3: Run the failing tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_data_coco.py tests/unit/test_data_hf.py -x -v -k "image_ids or row_indices or image_level_leak"
```

Expected: failures because `COCODataset.__init__` does not accept `image_ids`, `HFDataset.__init__` does not accept `row_indices`, and the leak-invariant test fails until `_enumerate_coco_items` exists (which it does — Step 3).

### Task 4b: Implement COCO adapter changes

- [ ] **Step 4.4: Modify `src/custom_sam_peft/data/coco.py` — ctor**

Change the imports at line 14 to add `Iterable`:

```python
from collections.abc import Iterable
from typing import Any, Literal
```

Modify `COCODataset.__init__` (lines 122-163). Add `image_ids: Iterable[int] | None = None` as a keyword-only parameter after `seed: int = 0`, and apply the subset filter after the existing `_drop_crowd_only_images` call (line 149). Replace lines 122-163 with:

```python
    def __init__(
        self,
        annotations: str,
        images: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        seed: int = 0,
        image_ids: Iterable[int] | None = None,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._image_root = Path(images)
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False

        self._coco = _load_coco_index(annotations)
        sparse_ids, mapping, class_names = _build_category_remap(self._coco)
        self._coco_category_ids = sparse_ids
        self.coco_category_ids = sparse_ids
        self._cat_id_to_dense = mapping
        self._class_names = class_names

        kept, ann_index, dropped = _drop_crowd_only_images(self._coco)
        if image_ids is not None:
            requested = {int(x) for x in image_ids}
            kept_set = set(kept)
            missing = requested - kept_set
            if missing:
                first_few = sorted(missing)[:10]
                raise ValueError(
                    f"COCODataset: {len(missing)} image_ids requested but not present "
                    f"(or dropped as iscrowd-only): {first_few}"
                    f"{'…' if len(missing) > 10 else ''}"
                )
            self._image_ids = [i for i in kept if i in requested]
        else:
            self._image_ids = kept
        self._ann_index = ann_index
        if dropped:
            _LOG.info(
                "custom_sam_peft.data.coco: dropped %d images (iscrowd-only) from %s",
                dropped,
                annotations,
            )
        _LOG.info(
            "custom_sam_peft.data.coco: loaded %d images, %d dense classes from %s",
            len(self._image_ids),
            len(self._class_names),
            annotations,
        )
```

- [ ] **Step 4.5: Modify `src/custom_sam_peft/data/coco.py` — `build_coco`**

Replace the `build_coco` body at line 311-317 (the `return COCODataset(...)` block) with:

```python
    resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)
    return COCODataset(
        annotations=split["annotations"],
        images=split["images"],
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
        image_ids=[int(s) for s in resolved] if resolved is not None else None,
    )
```

### Task 4c: Implement HF adapter changes

- [ ] **Step 4.6: Modify `src/custom_sam_peft/data/hf.py` — ctor**

Change the imports at line 11 to add `Iterable`:

```python
from collections.abc import Iterable
from typing import Any, Literal
```

Modify `HFDataset.__init__` (lines 128-154). Add `row_indices: Iterable[int] | None = None` after `seed`. Initialize `self._index_map: list[int] | None` based on it. Replace lines 128-154 with:

```python
    def __init__(
        self,
        name: str,
        split: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        field_map: HFFieldMap,
        seed: int = 0,
        row_indices: Iterable[int] | None = None,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._name = name
        self._split = split
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._field_map = field_map
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False
        self._warned_masks_from_boxes = False

        self._ds = hf_load_dataset(name, split=split)
        _validate_required_fields(self._ds, field_map)
        self._class_names = _resolve_class_names(self._ds, field_map)
        if row_indices is not None:
            self._index_map: list[int] | None = [int(i) for i in row_indices]
            invalid = [i for i in self._index_map if i < 0 or i >= len(self._ds)]
            if invalid:
                raise ValueError(
                    f"HFDataset: {len(invalid)} row_indices out of range "
                    f"[0, {len(self._ds)}): first few = {invalid[:10]}"
                )
        else:
            self._index_map = None
```

Now modify `__len__` (currently line 156) and `__getitem__` (currently line 159) to honor `_index_map`. Replace `__len__`:

```python
    def __len__(self) -> int:
        return len(self._index_map) if self._index_map is not None else len(self._ds)
```

In `__getitem__`, the first line (currently `row = self._ds[i]` at line 168) becomes:

```python
        row_i = self._index_map[i] if self._index_map is not None else i
        row = self._ds[row_i]
```

And the `image_id = str(i)` line at line 232 must use `row_i`:

```python
        image_id = str(row_i)
```

And the existing `image_id=i` inside `_build_text_prompts(... image_id=i, ...)` at line 241 becomes:

```python
            image_id=row_i,
```

That keeps the seeded RNG and the returned `Example.image_id` in the underlying-row namespace.

- [ ] **Step 4.7: Modify `src/custom_sam_peft/data/hf.py` — `build_hf`**

Replace the `build_hf` `return HFDataset(...)` block at the end (line 325-332) with:

```python
    resolved = (cfg.get("_resolved_image_ids") or {}).get(pipeline)
    return HFDataset(
        name=hf_cfg["name"],
        split=split,
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
        field_map=field_map,
        row_indices=[int(s) for s in resolved] if resolved is not None else None,
    )
```

- [ ] **Step 4.8: Run the adapter tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_data_coco.py tests/unit/test_data_hf.py -x -v
```

Expected: all tests pass (existing + new). The existing COCO/HF tests must still pass — the new param defaults to `None` and falls through to the existing fast path.

- [ ] **Step 4.9: Commit Step 4**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py \
        tests/unit/test_data_coco.py tests/unit/test_data_hf.py
git commit -m "feat(data): add image_ids/row_indices subset params to adapters (#71)"
```

---

## Step 5: Runner orchestration (`train/runner.py`)

**Files:**
- Modify: `src/custom_sam_peft/train/runner.py` (lines 25-46)
- Modify: `tests/unit/test_train_runner.py` (append)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Steps 1, 2, 3, 4.

**Spec:** §6.4 (runner), §9.7 (tests).

### Task 5a: Failing runner tests

- [ ] **Step 5.1: Append the 2 runner tests to `tests/unit/test_train_runner.py`**

Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): val_source orchestration
# ---------------------------------------------------------------------------


def test_run_training_writes_val_source_json_on_auto_split(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §6.4 + §9.7.1: end-to-end auto-split writes <run_dir>/val_source.json.

    Uses tiny_coco + LoRA stub to keep this CPU-bound.
    """
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrainConfig,
        TrainHyperparams,
        ValSplitConfig,
    )
    from custom_sam_peft.data.val_source import load_val_source
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    cfg = TrainConfig(
        run=RunConfig(name="autosplit", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            val_split=ValSplitConfig(fraction=0.5, seed=None),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1, batch_size=1, grad_accum_steps=1, save_every=2, log_every=1,
            warmup_steps=0, num_workers=0,
        ),
    )

    monkeypatch.setattr(
        "custom_sam_peft.train.runner.load_sam31", lambda _m: make_stub_wrapper(dim=8, working=True)
    )
    from custom_sam_peft import train as _train_pkg  # noqa: F401
    # peft_factory must accept (wrapper, cfg.peft) and apply lora; reuse real.

    from custom_sam_peft.train.runner import run_training

    result = run_training(cfg)
    assert (result.run_dir / "val_source.json").is_file()
    vs = load_val_source(result.run_dir)
    assert vs is not None
    assert vs.mode == "auto_split"
    assert vs.train_ids is not None and vs.val_ids is not None


def test_run_training_resume_reuses_saved_val_source(
    tmp_path: Path, tiny_coco_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §8.2 + §9.7.2: resume reuses the saved partition; splitter not re-called."""
    from custom_sam_peft.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrainConfig,
        TrainHyperparams,
        ValSplitConfig,
    )
    from custom_sam_peft.data.val_source import load_val_source
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    def _cfg() -> TrainConfig:
        return TrainConfig(
            run=RunConfig(name="resume", output_dir=str(tmp_path), seed=0),
            data=DataConfig(
                format="coco",
                train=DataSplit(
                    annotations=str(tiny_coco_dir / "annotations.json"),
                    images=str(tiny_coco_dir / "images"),
                ),
                val=None,
                val_split=ValSplitConfig(fraction=0.5, seed=None),
                prompt_mode="text",
                image_size=32,
            ),
            peft=PEFTConfig(
                method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
            ),
            train=TrainHyperparams(
                epochs=1, batch_size=1, grad_accum_steps=1, save_every=2, log_every=1,
                warmup_steps=0, num_workers=0,
            ),
        )

    monkeypatch.setattr(
        "custom_sam_peft.train.runner.load_sam31",
        lambda _m: make_stub_wrapper(dim=8, working=True),
    )
    from custom_sam_peft.train.runner import run_training

    # First run.
    r1 = run_training(_cfg())
    vs1 = load_val_source(r1.run_dir)
    assert vs1 is not None
    saved_train = vs1.train_ids
    saved_val = vs1.val_ids
    ckpts = sorted((r1.run_dir / "checkpoints").glob("step_*"))
    assert ckpts, "first run produced no checkpoint"

    # Second run with resume_from set; if splitter is invoked the test fails.
    def _splitter_must_not_run(*a: object, **kw: object) -> object:
        raise AssertionError("splitter must not be re-called on resume")

    monkeypatch.setattr(
        "custom_sam_peft.data.val_source.stratified_split", _splitter_must_not_run
    )
    r2 = run_training(_cfg(), resume_from=ckpts[0])
    vs2 = load_val_source(r2.run_dir)
    assert vs2 is not None
    assert vs2.train_ids == saved_train
    assert vs2.val_ids == saved_val
```

- [ ] **Step 5.2: Run the failing tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_train_runner.py -x -v -k "val_source or resume_reuses"
```

Expected: failures because `run_training` does not yet write `val_source.json` and does not yet handle `val=None`.

### Task 5b: Implement runner changes

- [ ] **Step 5.3: Modify `src/custom_sam_peft/train/runner.py` end-to-end**

Replace the whole file (it's 46 lines) with:

```python
"""End-to-end train pipeline. CLI is a thin wrapper over `run_training`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.val_source import (
    _log_val_source,
    resolve_val_source,
    save_val_source,
)
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import RunResult, Trainer


def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Returns the created path."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataset_from_dict(
    data_cfg_dict: dict[str, Any], cfg: TrainConfig, pipeline: str
) -> Dataset:
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline=pipeline))


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    """Build datasets, load model + PEFT, build tracker, run Trainer.fit.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §6.4.
    """
    run_dir = make_run_dir(cfg)

    # On resume, look for val_source.json in the run dir that owns the
    # checkpoint (checkpoints live at <run_dir>/checkpoints/step_N/).
    resume_run_dir = resume_from.parent.parent if resume_from is not None else None
    vs = resolve_val_source(cfg, run_dir=resume_run_dir)
    save_val_source(vs, run_dir)
    _log_val_source(vs)

    data_cfg_dict = cfg.data.model_dump()
    if vs.mode == "auto_split":
        assert vs.train_ids is not None and vs.val_ids is not None
        data_cfg_dict["_resolved_image_ids"] = {
            "train": list(vs.train_ids),
            "eval": list(vs.val_ids),
        }

    train_ds = _build_dataset_from_dict(data_cfg_dict, cfg, "train")
    val_ds: Dataset | None = (
        None if vs.mode == "none" else _build_dataset_from_dict(data_cfg_dict, cfg, "eval")
    )

    wrapper: Any = load_sam31(cfg.model)
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)
    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)
```

The old `_build_dataset` helper is gone — replaced by `_build_dataset_from_dict` which is now keyed on the dict (so the runner can inject `_resolved_image_ids` once).

- [ ] **Step 5.4: Run runner tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_train_runner.py -x -v
```

Expected: all existing tests + 2 new tests pass.

NOTE: `Trainer.__init__` may not yet accept `val_ds: Dataset | None` (that change lands in Step 6). If `test_run_training_writes_val_source_json_on_auto_split` fails because Trainer rejects `val_ds=None`, that test depends on Step 6 — run only the existing + the auto-split test (which doesn't pass None). Mark the resume test for re-verification at Step 6 end.

In practice: the auto-split test passes a real `val_ds` (not None), so it should succeed after Step 5 alone. The two added tests don't trigger the no-val path.

- [ ] **Step 5.5: Commit Step 5**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/train/runner.py tests/unit/test_train_runner.py
git commit -m "feat(train): wire val_source resolver/save/inject into runner (#71)"
```

---

## Step 6: Trainer no-val path (`train/trainer.py`) + `run_epoch` dead-arg cleanup (`train/loop.py`)

**Files:**
- Modify: `src/custom_sam_peft/train/trainer.py` (lines 127-280)
- Modify: `src/custom_sam_peft/train/loop.py` (lines 227-242)
- Create: `tests/unit/test_trainer_no_val.py`

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Steps 1-5 (saved `val_source.json` from runner).

**Spec:** §7.1 (trainer), §7.2 (loop), §7.3 (panel), §9.5 (tests).

### Task 6a: Failing trainer no-val tests

- [ ] **Step 6.1: Create `tests/unit/test_trainer_no_val.py`**

```python
"""Trainer.no_val mode tests — val_ds=None short-circuits eval/panel/end-of-run eval.

Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.1, §9.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrackingConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.data.coco import COCODataset
from custom_sam_peft.data.transforms import build_train_transforms
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _ds_train(tiny_coco_dir: Path) -> COCODataset:
    from custom_sam_peft.config.schema import (
        AugmentationsConfig,
        NormalizeConfig,
        TextPromptConfig,
    )

    transforms = build_train_transforms(
        AugmentationsConfig(hflip=False, color_jitter=0.0),
        32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def _cfg(tmp_path: Path, tiny_coco_dir: Path) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="no-val", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            val_split=None,
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1, batch_size=1, grad_accum_steps=1, save_every=2,
            eval_every=1, log_every=1, warmup_steps=0, num_workers=0,
        ),
        tracking=TrackingConfig(backend="none"),
    )


def test_fit_with_val_ds_none_completes_and_writes_no_val_metrics(
    tmp_path: Path, tiny_coco_dir: Path
) -> None:
    """Trainer(val_ds=None).fit() completes; metrics.json carries the no-val note."""
    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    # Pre-save a val_source.json so the trainer's tracker hparams reader sees it.
    run_dir = tmp_path / f"{cfg.run.name}-test"
    run_dir.mkdir(parents=True)
    (run_dir / "val_source.json").write_text(
        json.dumps(
            {
                "mode": "none",
                "fraction_requested": None,
                "seed_used": None,
                "realized_fraction": None,
                "n_train": None,
                "n_val": None,
                "per_class_counts": None,
                "missing_in_val": None,
                "train_ids": None,
                "val_ids": None,
            }
        )
    )
    trainer = Trainer(wrapper, ds_train, None, build_tracker(cfg), cfg)
    result = trainer.fit(run_dir=run_dir)
    assert result.final_metrics is None
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload.get("note") == "no validation set provided"
    assert "global_step" in payload


def test_fit_with_val_ds_none_does_not_invoke_evaluator(
    tmp_path: Path, tiny_coco_dir: Path
) -> None:
    """Evaluator must not be constructed/called when val_ds is None."""
    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test2"
    run_dir.mkdir(parents=True)
    (run_dir / "val_source.json").write_text(json.dumps({"mode": "none"}))

    mock_evaluator = MagicMock()
    with patch("custom_sam_peft.train.trainer.Evaluator", return_value=mock_evaluator):
        trainer = Trainer(wrapper, ds_train, None, build_tracker(cfg), cfg)
        trainer.fit(run_dir=run_dir)
    mock_evaluator.evaluate.assert_not_called()


def test_fit_with_val_ds_none_does_not_log_image_panel(
    tmp_path: Path, tiny_coco_dir: Path
) -> None:
    """The image-panel writer never fires when val_ds is None."""
    cfg = _cfg(tmp_path, tiny_coco_dir)
    ds_train = _ds_train(tiny_coco_dir)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)
    run_dir = tmp_path / f"{cfg.run.name}-test3"
    run_dir.mkdir(parents=True)
    (run_dir / "val_source.json").write_text(json.dumps({"mode": "none"}))

    tracker = build_tracker(cfg)
    tracker.log_images = MagicMock()  # type: ignore[method-assign]

    trainer = Trainer(wrapper, ds_train, None, tracker, cfg)
    trainer.fit(run_dir=run_dir)
    tracker.log_images.assert_not_called()
```

- [ ] **Step 6.2: Run the failing tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_trainer_no_val.py -x -v
```

Expected: failures because `Trainer.__init__` does not accept `val_ds=None`, and `fit()` unconditionally evaluates.

### Task 6b: Implement trainer changes

- [ ] **Step 6.3: Modify `src/custom_sam_peft/train/trainer.py` — `Trainer.__init__`**

Replace line 132 (`val_ds: Dataset,`) with `val_ds: Dataset | None,`. Add a no-val INFO log at the end of `__init__` (after line 153):

After:

```python
        if cfg.train.optimizer == "auto":
            _LOG.info(
                "optimizer=auto resolved to %s (peft.method=%s)",
                self._optimizer_name,
                cfg.peft.method,
            )
```

Add:

```python
        if val_ds is None:
            _LOG.info(
                "training without validation set; eval_every is a no-op, "
                "end-of-run eval and bundle samples are skipped."
            )
```

- [ ] **Step 6.4: Modify `src/custom_sam_peft/train/trainer.py` — `fit()` guards**

In `fit()`:

a. Replace line 186 (`val_examples = [self.val_ds[i] for i in range(min(4, len(self.val_ds)))]`) with:

```python
        val_examples: list[Any] = (
            [] if self.val_ds is None else [self.val_ds[i] for i in range(min(4, len(self.val_ds)))]
        )
```

b. Add a guard at the top of `on_eval` (currently line 222). After `def on_eval(step: int) -> None:` and before `try:`:

```python
        def on_eval(step: int) -> None:
            if self.val_ds is None:
                return
            try:
```

c. Replace line 257 (end-of-run eval) with:

```python
            if self.val_ds is not None:
                full_report = Evaluator(cfg.eval).evaluate(self.model, self.val_ds)
```

d. Replace lines 258-271 (the `metrics.json` write). Change to branch on whether `full_report` is populated:

```python
            if full_report is not None:
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "overall": full_report.overall,
                            "per_class": full_report.per_class,
                            "n_images": full_report.n_images,
                            "n_predictions": full_report.n_predictions,
                            "global_step": global_step,
                            "epoch": cfg.train.epochs - 1,
                            "box_hint_p_final": _box_hint_p(global_step, cfg.train.box_hint),
                        },
                        indent=2,
                    )
                )
            else:
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "note": "no validation set provided",
                            "global_step": global_step,
                            "epoch": cfg.train.epochs - 1,
                            "box_hint_p_final": _box_hint_p(global_step, cfg.train.box_hint),
                        },
                        indent=2,
                    )
                )
```

- [ ] **Step 6.5: Modify `src/custom_sam_peft/train/trainer.py` — tracker hparams injection**

Right before `self.tracker.start_run(...)` at line 167, replace that single line with:

```python
        cfg_dict = cfg.model_dump(mode="json")
        vs_path = run_dir / "val_source.json"
        if vs_path.exists():
            saved = json.loads(vs_path.read_text())
            cfg_dict["val_source"] = {
                "mode": saved["mode"],
                "fraction_requested": saved.get("fraction_requested"),
                "realized_fraction": saved.get("realized_fraction"),
                "n_train": saved.get("n_train"),
                "n_val": saved.get("n_val"),
            }
        self.tracker.start_run(run_dir, cfg_dict, resume_from)
```

### Task 6c: Drop `val_ds` dead arg from `run_epoch`

- [ ] **Step 6.6: Modify `src/custom_sam_peft/train/loop.py` — `run_epoch` signature**

Remove the `val_ds: Any,` parameter at line 239. The new signature (lines 227-242) becomes:

```python
def run_epoch(
    model: Sam3Wrapper,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tracker: Tracker,
    cfg: TrainConfig,
    run_dir: Path,
    epoch: int,
    global_step: int,
    nan_streak: int,
    class_names: list[str],
    on_checkpoint: Callable[[int, int, float, int], None],
    on_eval: Callable[[int], None],
) -> tuple[int, int]:
```

The body is unchanged — `val_ds` was never used (eval happens via the `on_eval` closure that captures `self.val_ds` in the Trainer).

- [ ] **Step 6.7: Modify `src/custom_sam_peft/train/trainer.py` — `run_epoch` call site**

Remove `self.val_ds,` from the `run_epoch(...)` argument list at line 246. The call (lines 234-249) becomes:

```python
                global_step, nan_streak = run_epoch(
                    self.model,
                    train_loader,
                    optimizer,
                    scheduler,
                    self.tracker,
                    cfg,
                    run_dir,
                    epoch,
                    global_step,
                    nan_streak,
                    class_names,
                    on_checkpoint,
                    on_eval,
                )
```

- [ ] **Step 6.8: Run trainer no-val + existing trainer-adjacent tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_trainer_no_val.py tests/integration/test_train_end_to_end.py -x -v -m "not gpu"
```

Expected: all pass. Importantly, the existing end-to-end test (with explicit val) still passes — the no-val branch is purely additive.

- [ ] **Step 6.9: Run runner tests too — should now pass fully**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_train_runner.py -x -v
```

Expected: all pass.

- [ ] **Step 6.10: Commit Step 6**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/train/trainer.py src/custom_sam_peft/train/loop.py \
        tests/unit/test_trainer_no_val.py
git commit -m "feat(train): trainer val_ds Optional + drop run_epoch dead val_ds arg (#71)"
```

---

## Step 7: Eval CLI guards (`eval/runner.py`)

**Files:**
- Modify: `src/custom_sam_peft/eval/runner.py` (lines 75-91)
- Modify: `tests/unit/test_eval_runner.py` (append)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Steps 1-4.

**Spec:** §7.4 (eval runner), §9.9 (tests).

### Task 7a: Failing eval tests

- [ ] **Step 7.1: Append the 2 eval tests to `tests/unit/test_eval_runner.py`**

Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): --split val guard + auto-split in eval
# ---------------------------------------------------------------------------


def test_run_eval_rejects_val_split_when_data_val_and_val_split_none(
    tmp_path: Path,
) -> None:
    """Spec §7.4 A: --split val requires data.val or data.val_split."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    cfg.data.val_split = None  # type: ignore[attr-defined]
    cfg.data.test = None
    with pytest.raises(ValueError, match=r"--split val requires data\.val or data\.val_split"):
        run_eval(cfg, checkpoint=tmp_path, split="val")


def test_run_eval_auto_split_threads_resolved_image_ids_to_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §7.4 B: when val_dataset is None and val_split is set, builder receives
    _resolved_image_ids."""
    cfg = _make_cfg(format_="coco", peft_method="lora")
    cfg.data.val = None  # type: ignore[attr-defined]
    # Build a real ValSplitConfig so the guard passes.
    from custom_sam_peft.config.schema import ValSplitConfig

    cfg.data.val_split = ValSplitConfig(fraction=0.5, seed=1)  # type: ignore[attr-defined]

    # Mock resolve_val_source to return a known partition.
    from custom_sam_peft.data.val_source import ValSource

    fake_vs = ValSource(
        mode="auto_split",
        train_ids=("1", "2"),
        val_ids=("3", "4"),
        realized_fraction=0.5,
        per_class_counts={0: (2, 2)},
        missing_in_val=(),
        fraction_requested=0.5,
        seed_used=1,
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.resolve_val_source", lambda *_a, **_kw: fake_vs
    )

    captured: dict[str, object] = {}
    fake_ds = MagicMock(__len__=lambda self: 2, class_names=["c"])

    def fake_builder(cfg_dict: object, **kwargs: object) -> object:
        captured["cfg_dict"] = cfg_dict
        return fake_ds

    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda kind, name: fake_builder,
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_lora", lambda *_a, **_kw: None)
    fake_report = MagicMock(overall={"mAP": 0.5}, per_class={}, n_images=2, n_predictions=2)
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)
    assert "cfg_dict" in captured
    cfg_dict = captured["cfg_dict"]
    assert isinstance(cfg_dict, dict)
    assert "_resolved_image_ids" in cfg_dict
    assert cfg_dict["_resolved_image_ids"] == {"eval": ["3", "4"]}
```

- [ ] **Step 7.2: Run the failing tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_eval_runner.py -x -v -k "val_split"
```

Expected: failures.

### Task 7b: Implement eval changes

- [ ] **Step 7.3: Modify `src/custom_sam_peft/eval/runner.py`**

Add the new guard between the existing peft check (line 76-80) and the test guard (line 81-82). After the peft-check block, before `if split == "test"`:

```python
    if split == "val" and cfg.data.val is None and cfg.data.val_split is None:
        raise ValueError(
            "--split val requires data.val or data.val_split in config; got neither."
        )
```

Then modify the `if val_dataset is None:` block (line 84-89) to support auto-split:

```python
    if val_dataset is None:
        cfg_dict = cfg.data.model_dump()
        if split == "test":
            cfg_dict["val"] = cfg_dict["test"]
        elif split == "val" and cfg.data.val_split is not None:
            from custom_sam_peft.data.val_source import resolve_val_source

            vs = resolve_val_source(cfg, run_dir=None)
            assert vs.val_ids is not None
            cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
        builder = lookup("dataset", cfg.data.format)
        dataset = cast(Dataset, builder(cfg_dict, model_name=cfg.model.name, pipeline="eval"))
    else:
        dataset = val_dataset
```

- [ ] **Step 7.4: Run the eval tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_eval_runner.py -x -v
```

Expected: all original + 2 new tests pass.

- [ ] **Step 7.5: Commit Step 7**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/eval/runner.py tests/unit/test_eval_runner.py
git commit -m "feat(eval): --split val guard + standalone auto-split path (#71)"
```

---

## Step 8: Bundle no-val path (`runs/bundle.py`)

**Files:**
- Modify: `src/custom_sam_peft/runs/bundle.py` (lines 262-279)
- Modify: `tests/unit/runs/test_bundle.py` (append)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** none beyond pre-flight (independent file).

**Spec:** §7.5 (bundle), §9.6 (tests).

### Task 8a: Failing bundle tests

- [ ] **Step 8.1: Append the 3 bundle tests to `tests/unit/runs/test_bundle.py`**

Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): no-val bundle path
# ---------------------------------------------------------------------------


def test_write_bundle_no_val_writes_summary_only(tmp_path: Path) -> None:
    """Spec §7.5: write_bundle(val_dataset=None, metrics_report=None) writes summary.md only."""
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=MagicMock())
    summary_path = ctx.run_dir / "summary.md"
    assert summary_path.is_file()
    summary = summary_path.read_text()
    assert "no-val" in summary.lower() or "no validation" in summary.lower()
    # No samples directory should be created in no-val mode.
    samples_dir = ctx.run_dir / "samples"
    assert not samples_dir.exists() or not any(samples_dir.glob("*.png"))


def test_write_bundle_no_val_headline_says_no_val(tmp_path: Path) -> None:
    """Spec §7.5: headline reads '... — no-val' (no mAP number)."""
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    first_line = summary.splitlines()[0]
    assert first_line.startswith("# ")
    assert "no-val" in first_line.lower()


def test_write_bundle_no_val_contains_no_validation_set_line(tmp_path: Path) -> None:
    """Spec §7.5: summary body contains 'No validation set'."""
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "No validation set" in summary
```

- [ ] **Step 8.2: Run the failing tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/runs/test_bundle.py -x -v -k "no_val"
```

Expected: failures because the signature is `metrics_report: MetricsReport`, not `Optional`.

### Task 8b: Implement bundle changes

- [ ] **Step 8.3: Modify `src/custom_sam_peft/runs/bundle.py` — widen signature**

Replace the signature of `write_bundle` (line 262-267) and add a no-val branch at the top of the body:

```python
def write_bundle(
    ctx: BundleContext,
    metrics_report: MetricsReport | None,
    val_dataset: Dataset | None,
    model_wrapper: Any,
) -> None:
    """Write `ctx.run_dir/summary.md` and `ctx.run_dir/samples/*.png`.

    No-val mode: when val_dataset is None, writes summary.md only with the
    "no-val" headline and skips the samples/ directory.

    Idempotent: re-runs overwrite. Failure modes:
      - Per-sample inference raises → that PNG is skipped; WARNING logged;
        "skipped samples" note in summary.md. Bundle does not abort.
      - All other errors propagate.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.5.
    """
    if val_dataset is None:
        _write_summary_no_val(ctx)
        return
    samples_dir = ctx.run_dir / "samples"
```

(The existing body continues unchanged starting at `samples_dir = ctx.run_dir / "samples"`.)

- [ ] **Step 8.4: Add `_write_summary_no_val` helper to `src/custom_sam_peft/runs/bundle.py`**

Add immediately above `def write_bundle(...)`:

```python
def _write_summary_no_val(ctx: BundleContext) -> None:
    """Spec §7.5: write summary.md only; no samples directory.

    Headline is `# <run-name> — no-val` instead of `# <run-name> — <mAP>`.
    """
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

    config_rel = ctx.config_path.name

    headline = f"# {ctx.config_path.parent.name} — no-val"
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
        f"## Validation\n"
        f"No validation set; this run did not produce mAP or per-example IoU.\n"
        f"Tracker scalars and training-loss curve are at the configured TB run dir.\n"
    )
    if ctx.merged_export_error is not None:
        body += f"\n## Edge cases\n- export-merge failed: {ctx.merged_export_error}\n"

    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    (ctx.run_dir / "summary.md").write_text(body)
```

- [ ] **Step 8.5: Run the bundle tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/runs/test_bundle.py -x -v
```

Expected: all original + 3 new tests pass.

- [ ] **Step 8.6: Commit Step 8**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/runs/bundle.py tests/unit/runs/test_bundle.py
git commit -m "feat(runs): widen write_bundle for no-val mode + summary writer (#71)"
```

---

## Step 9: `csp run` orchestration (`cli/run_cmd.py`)

**Files:**
- Modify: `src/custom_sam_peft/cli/run_cmd.py` (lines 34-109)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Steps 5, 8.

**Spec:** §7.6 (run_cmd orchestration).

Note: no new unit test added here — `tests/integration/test_cli_run.py` (existing) is extended in Step 12 to drive both the no-val and auto-split paths end-to-end.

### Task 9a: Implement run_cmd changes

- [ ] **Step 9.1: Modify `src/custom_sam_peft/cli/run_cmd.py` — replace `_build_val_dataset`**

Replace the existing `_build_val_dataset` (line 34-36) with a new signature accepting the `ValSource`:

```python
def _build_val_dataset(cfg: TrainConfig, vs: "ValSource") -> Dataset:
    """Build the val dataset using the same image ids the trainer used.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.6.
    """
    data_cfg_dict = cfg.data.model_dump()
    if vs.mode == "auto_split":
        assert vs.val_ids is not None
        data_cfg_dict["_resolved_image_ids"] = {"eval": list(vs.val_ids)}
    builder = lookup("dataset", cfg.data.format)
    return cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline="eval"))
```

Add the `ValSource` import in a `TYPE_CHECKING` guard at the top of the file (after the existing imports):

```python
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from custom_sam_peft.data.val_source import ValSource
```

(If `Any, cast` are already imported via `from typing import Any, cast`, add `TYPE_CHECKING` to that import line.)

- [ ] **Step 9.2: Modify `src/custom_sam_peft/cli/run_cmd.py` — `_orchestrate`**

Replace the body of `_orchestrate` (lines 39-109) with:

```python
def _orchestrate(cfg: TrainConfig, resume: Path | None) -> int:
    from custom_sam_peft.data.val_source import load_val_source

    start_ts = datetime.now(UTC)

    # Phase: train.
    try:
        train_result = run_training(cfg, resume_from=resume)
    except Exception as exc:
        rprint(f"[red]train failed[/red] {exc}")
        raise typer.Exit(code=1) from exc
    run_dir = train_result.run_dir
    adapter_path = train_result.adapter_path

    # Decide val mode from the saved record — same source of truth the trainer used.
    vs = load_val_source(run_dir)
    assert vs is not None, "runner must have saved val_source.json"

    wrapper: Any = load_sam31(cfg.model)
    load_adapter(wrapper, adapter_path)

    val_dataset: Dataset | None = None
    report: Any = None
    per_example_iou: list[float] = []
    if vs.mode != "none":
        val_dataset = _build_val_dataset(cfg, vs)

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
        except Exception as exc:
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
        except Exception as exc:
            _LOG.warning("export-merge failed: %s", exc)
            merged_export_error = str(exc)

    # Phase: bundle.
    ctx = BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=start_ts,
        end_ts=end_ts,
        preset_label=os.environ.get("CUSTOM_SAM_PEFT_PRESET_LABEL"),
        per_example_iou=per_example_iou,
        merged_dir=merged_dir,
        merged_export_error=merged_export_error,
    )
    try:
        write_bundle(ctx, report, val_dataset=val_dataset, model_wrapper=wrapper)
    except Exception as exc:
        rprint(f"[red]bundle failed[/red] run_dir={run_dir} — {exc}")
        raise typer.Exit(code=1) from exc

    mAP_str = (
        f"{report.overall.get('mAP', float('nan')):.4f}"
        if report is not None
        else "n/a (no val)"
    )
    rprint(
        f"[green]done[/green] run_dir={run_dir} adapter={adapter_path} "
        f"merged={(merged_dir or merged_export_error or 'skipped')} "
        f"summary={run_dir / 'summary.md'} mAP={mAP_str}"
    )
    return 0
```

- [ ] **Step 9.3: Smoke-run `csp run` against an inline no-val config to confirm it doesn't crash**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run python -c "from custom_sam_peft.cli.run_cmd import _orchestrate, _build_val_dataset; print('imports ok')"
```

Expected: prints `imports ok` (no `ImportError`). Full end-to-end is covered by Step 12's integration test.

- [ ] **Step 9.4: Commit Step 9**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/cli/run_cmd.py
git commit -m "feat(cli): csp run reads val_source.json and degrades on no-val (#71)"
```

---

## Step 10: Doctor `--config` (`cli/doctor_cmd.py` + `diagnostics.py`)

**Files:**
- Modify: `src/custom_sam_peft/diagnostics.py`
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`
- Modify: `tests/unit/test_cli_doctor.py` (append)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Step 2 (schema additions).

**Spec:** §7.7 (doctor), §9.8 (tests).

### Task 10a: Failing doctor tests

- [ ] **Step 10.1: Append the 4 doctor tests to `tests/unit/test_cli_doctor.py`**

Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): doctor --config / Data table
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, *, val: bool, val_split: bool) -> Path:
    """Write a minimal valid TrainConfig YAML to disk; return its path."""
    import yaml

    data_block: dict[str, object] = {
        "format": "coco",
        "train": {"annotations": str(tmp_path / "t.json"), "images": str(tmp_path / "imgs")},
        "prompt_mode": "text",
        "image_size": 32,
    }
    # Create the referenced files so the loader doesn't fail at path resolve.
    (tmp_path / "t.json").write_text("{}")
    (tmp_path / "imgs").mkdir(exist_ok=True)
    if val:
        data_block["val"] = {
            "annotations": str(tmp_path / "v.json"),
            "images": str(tmp_path / "vimgs"),
        }
        (tmp_path / "v.json").write_text("{}")
        (tmp_path / "vimgs").mkdir(exist_ok=True)
    if val_split:
        data_block["val_split"] = {"fraction": 0.2, "seed": 5}
    cfg = {
        "run": {"name": "doc", "output_dir": str(tmp_path / "runs"), "seed": 11},
        "data": data_block,
        "peft": {"method": "lora"},
        "train": {"epochs": 1},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_doctor_config_auto_split_renders_data_table(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, val=False, val_split=True)
    result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Data" in text
    assert "auto_split" in text
    assert "0.200" in text  # fraction formatted as 3 decimals
    assert "5" in text  # seed


def test_doctor_config_explicit_renders_data_table(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, val=True, val_split=False)
    result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Data" in text
    assert "explicit" in text


def test_doctor_config_none_renders_data_table(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, val=False, val_split=False)
    result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Data" in text
    assert "none" in text


def test_doctor_without_config_does_not_call_enumerate_or_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §7.7: doctor must never invoke the splitter or enumerate items."""

    def _must_not_run(*_a: object, **_kw: object) -> object:
        raise AssertionError("doctor must not call this")

    monkeypatch.setattr("custom_sam_peft.data.val_source._enumerate_coco_items", _must_not_run)
    monkeypatch.setattr("custom_sam_peft.data.splitter.stratified_split", _must_not_run)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
```

- [ ] **Step 10.2: Run the failing tests**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_cli_doctor.py -x -v -k "config or auto_split or explicit or none or enumerate"
```

Expected: failures because `--config` flag doesn't exist and `DataReport` doesn't exist.

### Task 10b: Implement diagnostics changes

- [ ] **Step 10.3: Modify `src/custom_sam_peft/diagnostics.py` — add `DataReport`**

Add a new dataclass after `HuggingFaceAuthInfo` (line 41), before `DoctorReport`:

```python
@dataclass(frozen=True)
class DataReport:
    """Validation source plan for the given config (no dataset materialization).

    Populated only when `run_doctor(config_path=...)` is called.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.7.
    """

    val_mode: Literal["explicit", "auto_split", "none"]
    val_path: str | None
    val_split_fraction: float | None
    val_split_seed: int | None
```

Extend `DoctorReport` (line 43-55) with the new optional field:

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
    issues: list[str] = field(default_factory=list)
    data: DataReport | None = None
```

Extend `run_doctor` (line 118) to accept `config_path`. Replace the signature line and add the `data` computation at the end:

```python
def run_doctor(
    *,
    weights_path: Path | None = None,
    config_path: Path | None = None,
) -> DoctorReport:
```

After all existing body lines but **before** `return DoctorReport(...)` (line 150-162), add:

```python
    data: DataReport | None = None
    if config_path is not None:
        from custom_sam_peft.config.loader import load_config

        cfg = load_config(config_path)
        if cfg.data.val_split is not None:
            seed = (
                cfg.data.val_split.seed
                if cfg.data.val_split.seed is not None
                else cfg.run.seed
            )
            data = DataReport(
                val_mode="auto_split",
                val_path=None,
                val_split_fraction=cfg.data.val_split.fraction,
                val_split_seed=seed,
            )
        elif cfg.data.val is not None:
            data = DataReport(
                val_mode="explicit",
                val_path=cfg.data.val.annotations,
                val_split_fraction=None,
                val_split_seed=None,
            )
        else:
            data = DataReport(
                val_mode="none",
                val_path=None,
                val_split_fraction=None,
                val_split_seed=None,
            )
```

Then pass `data=data` into the final `DoctorReport(...)` constructor.

### Task 10c: Implement doctor_cmd changes

- [ ] **Step 10.4: Modify `src/custom_sam_peft/cli/doctor_cmd.py` — `--config` flag**

Replace the `doctor()` Typer function (lines 74-86) with:

```python
def doctor(
    weights_path: Path | None = typer.Option(
        None, "--weights-path", help="Override SAM 3.1 weights file path."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Optional config YAML; enables the Data table."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Report environment + dependency status."""
    report = run_doctor(weights_path=weights_path, config_path=config_path)
    if json_output:
        print(json.dumps(dataclasses.asdict(report), default=str, indent=2))
    else:
        _render_table(report)
```

- [ ] **Step 10.5: Modify `src/custom_sam_peft/cli/doctor_cmd.py` — render Data table**

Append to `_render_table` (after the existing `if report.issues:` block, line 67-71), add:

```python
    if report.data is not None:
        d = Table(title="Data", show_header=False, box=None)
        d.add_row("val mode", report.data.val_mode)
        if report.data.val_path is not None:
            d.add_row("val path", report.data.val_path)
        if report.data.val_split_fraction is not None:
            d.add_row("val_split.fraction", f"{report.data.val_split_fraction:.3f}")
            d.add_row("val_split.seed", str(report.data.val_split_seed))
        console.print(d)
```

- [ ] **Step 10.6: Run doctor tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_cli_doctor.py tests/unit/test_diagnostics.py -x -v
```

Expected: all existing + 4 new tests pass.

- [ ] **Step 10.7: Commit Step 10**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add src/custom_sam_peft/diagnostics.py src/custom_sam_peft/cli/doctor_cmd.py \
        tests/unit/test_cli_doctor.py
git commit -m "feat(cli): doctor --config <yaml> prints Data table (#71)"
```

---

## Step 11: Example YAMLs + CLI templates

**Files:**
- Create: `configs/examples/coco_text_no_val.yaml`
- Create: `configs/examples/coco_text_auto_split.yaml`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`

**Dispatch:** implementer subagent, haiku/medium (non-code; copy-derive). **Depends on:** Step 2 (schema must support these).

**Spec:** §2 (file table), §3.3 (val mode resolution table).

### Task 11a: Create `configs/examples/coco_text_no_val.yaml`

- [ ] **Step 11.1: Create the no-val example**

Use this content (derived from existing `configs/examples/coco_text_lora.yaml` with the `val:` block omitted and a comment block explaining the resolution):

```yaml
# Demonstrates no-val mode: data.val is omitted, data.val_split is omitted.
# Training proceeds without validation; eval_every is a no-op; no image panels;
# no end-of-run eval; bundle writes summary.md only (no samples/).
#
# To switch back to explicit val, add a data.val block.
# To auto-split a fraction of train, add a data.val_split block.
# See: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §3.3.

run:
  name: coco-text-no-val
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
  # No `val:` block → no-val mode (one WARN at training start).
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
  eval_every: 500  # No-op in no-val mode.
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

### Task 11b: Create `configs/examples/coco_text_auto_split.yaml`

- [ ] **Step 11.2: Create the auto-split example**

```yaml
# Demonstrates auto-split: data.train carved deterministically into train+val.
# val_split.seed: null → inherit run.seed (42 here).
# The resolved split is recorded once in <run_dir>/val_source.json and is
# authoritative on resume.
# See: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §4-§5.

run:
  name: coco-text-auto-split
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
  # No `val:` block; instead, auto-split data.train.
  val_split:
    fraction: 0.1
    # seed: null → inherit run.seed at resolve time
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

### Task 11c: Modify CLI templates

- [ ] **Step 11.3: Modify `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`**

After the existing `val:` block (lines 21-23), add a commented `val_split:` reference block. Replace lines 16-25 (the `data:` block down to `prompt_mode`) with:

```yaml
data:
  format: coco
  train:
    annotations: data/train.json
    images: data/train/
  val:
    annotations: data/val.json
    images: data/val/
  # Alternatives — uncomment ONE to swap modes:
  #
  # No-val mode (remove the `val:` block entirely):
  # (just delete the val block above)
  #
  # Auto-split mode (carve data.train into train+val):
  # val_split:
  #   fraction: 0.1
  #   seed: null   # null → inherit run.seed at resolve time
  prompt_mode: text
```

- [ ] **Step 11.4: Modify `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`**

Apply the same patch as Step 11.3 to the QLoRA template. The `data:` block layout is identical (same `val:` block, same `prompt_mode:` follow-up).

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && diff src/custom_sam_peft/cli/templates/coco_text_lora.yaml src/custom_sam_peft/cli/templates/coco_text_qlora.yaml | head -30
```

Use the diff to confirm both templates differ only on PEFT fields, not the `data:` block.

- [ ] **Step 11.5: Verify all new YAMLs parse as valid `TrainConfig`s**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run python -c "
from custom_sam_peft.config.loader import load_config
for p in [
    'configs/examples/coco_text_no_val.yaml',
    'configs/examples/coco_text_auto_split.yaml',
    'configs/examples/coco_text_lora.yaml',
    'src/custom_sam_peft/cli/templates/coco_text_lora.yaml',
    'src/custom_sam_peft/cli/templates/coco_text_qlora.yaml',
]:
    cfg = load_config(p)
    mode = ('explicit' if cfg.data.val else ('auto_split' if cfg.data.val_split else 'none'))
    print(f'{p}: mode={mode}')
"
```

Expected: each YAML loads; modes match the file names. The two example YAMLs print `none` and `auto_split` respectively; the existing ones still print `explicit`.

- [ ] **Step 11.6: Re-run the config_examples test if one exists**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit/test_config_examples.py -x -v
```

Expected: pass. If the test enumerates example YAMLs and validates them, this catches typos in the two new files.

- [ ] **Step 11.7: Commit Step 11**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add configs/examples/coco_text_no_val.yaml \
        configs/examples/coco_text_auto_split.yaml \
        src/custom_sam_peft/cli/templates/coco_text_lora.yaml \
        src/custom_sam_peft/cli/templates/coco_text_qlora.yaml
git commit -m "docs(configs): add no-val + auto-split examples; templates show val_split (#71)"
```

---

## Step 12: Integration tests (`tests/integration/test_train_end_to_end.py`, `test_cli_run.py`)

**Files:**
- Modify: `tests/integration/test_train_end_to_end.py` (append)

**Dispatch:** implementer subagent, sonnet/high. **Depends on:** Steps 5, 6, 8, 9.

**Spec:** §9.10 (integration tests).

### Task 12a: Add auto-split and no-val integration tests

- [ ] **Step 12.1: Append the 2 integration tests to `tests/integration/test_train_end_to_end.py`**

Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): auto-split + no-val end-to-end
# ---------------------------------------------------------------------------


def test_e2e_auto_split_on_tiny_coco(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """Spec §9.10.2: end-to-end run with val_split=0.5 creates val_source.json
    and metrics.json (with overall mAP from the carved val set)."""
    from custom_sam_peft.config.schema import ValSplitConfig
    from custom_sam_peft.data.val_source import load_val_source
    from custom_sam_peft.train.runner import run_training

    cfg = TrainConfig(
        run=RunConfig(name="e2e-auto", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            val_split=ValSplitConfig(fraction=0.5, seed=None),
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1, batch_size=1, grad_accum_steps=1, save_every=2, log_every=1,
            warmup_steps=0, num_workers=0,
        ),
    )

    # Stub the model so this runs on CPU.
    import custom_sam_peft.train.runner as runner_mod

    orig_load = runner_mod.load_sam31
    runner_mod.load_sam31 = lambda _m: make_stub_wrapper(dim=8, working=True)  # type: ignore[assignment]
    try:
        result = run_training(cfg)
    finally:
        runner_mod.load_sam31 = orig_load  # type: ignore[assignment]

    vs = load_val_source(result.run_dir)
    assert vs is not None
    assert vs.mode == "auto_split"
    assert (result.run_dir / "metrics.json").is_file()
    # In auto-split mode, val_ds is non-empty so metrics.json carries overall, not the no-val note.
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert "overall" in payload or "note" in payload  # tolerate either depending on tiny size


def test_e2e_no_val_on_tiny_coco(tmp_path: Path, tiny_coco_dir: Path) -> None:
    """Spec §9.10.3: end-to-end no-val run creates val_source.json with mode=none
    and metrics.json with the no-val note."""
    from custom_sam_peft.data.val_source import load_val_source
    from custom_sam_peft.train.runner import run_training

    cfg = TrainConfig(
        run=RunConfig(name="e2e-noval", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=None,
            val_split=None,
            prompt_mode="text",
            image_size=32,
        ),
        peft=PEFTConfig(
            method="lora", scope="vision", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]
        ),
        train=TrainHyperparams(
            epochs=1, batch_size=1, grad_accum_steps=1, save_every=2, log_every=1,
            warmup_steps=0, num_workers=0,
        ),
    )

    import custom_sam_peft.train.runner as runner_mod

    orig_load = runner_mod.load_sam31
    runner_mod.load_sam31 = lambda _m: make_stub_wrapper(dim=8, working=True)  # type: ignore[assignment]
    try:
        result = run_training(cfg)
    finally:
        runner_mod.load_sam31 = orig_load  # type: ignore[assignment]

    vs = load_val_source(result.run_dir)
    assert vs is not None
    assert vs.mode == "none"
    payload = json.loads((result.run_dir / "metrics.json").read_text())
    assert payload.get("note") == "no validation set provided"
```

- [ ] **Step 12.2: Run the integration tests; verify all pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/integration/test_train_end_to_end.py -x -v -m "not gpu"
```

Expected: all existing + 2 new tests pass.

- [ ] **Step 12.3: Optionally — extend `tests/integration/test_cli_run.py` to verify the no-val summary line**

Read the existing test in `tests/integration/test_cli_run.py` (cf. spec §9.10.3 referencing it). If the test already exercises the full `csp run` path, append a no-val variant that asserts `summary.md` contains `"No validation set"`. If extending is not straightforward (it depends on the existing test's structure), the assertion is already covered by Step 8's unit tests on `_write_summary_no_val`.

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && head -30 tests/integration/test_cli_run.py
```

If the existing test is structured to take a YAML path, point it at `configs/examples/coco_text_no_val.yaml` via a tmp-path copy with rewritten data paths. Otherwise skip this extension — the unit-level coverage suffices.

- [ ] **Step 12.4: Commit Step 12**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git add tests/integration/test_train_end_to_end.py
# If test_cli_run.py was extended:
# git add tests/integration/test_cli_run.py
git commit -m "test(integration): e2e auto-split + no-val on tiny_coco (#71)"
```

---

## Step 13: Final coverage + lint check + readiness

**Dispatch:** orchestrator (inline). **Depends on:** all prior steps.

### Task 13a: Run the full test suite

- [ ] **Step 13.1: Run all unit + integration tests excluding GPU**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit tests/integration -x -q -m "not gpu"
```

Expected: all tests pass. Halt and diagnose any failure before continuing.

### Task 13b: Coverage gate (spec §9.12)

- [ ] **Step 13.2: Confirm coverage on `src/custom_sam_peft/data/` is ≥80%**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run pytest tests/unit -q --cov=src/custom_sam_peft/data --cov-report=term-missing
```

Expected: combined coverage on `src/custom_sam_peft/data/` (including the two new modules `splitter.py` and `val_source.py`) is ≥80%. If below threshold, add tests for uncovered branches in `data/splitter.py` and `data/val_source.py` until the gate is met.

### Task 13c: Lint + format

- [ ] **Step 13.3: Run ruff format + check**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run ruff format src tests configs
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run ruff check src tests configs --fix
```

Expected: format applies in-place; check is clean (or fixes are auto-applied). If `--fix` cannot auto-fix something (e.g. unused imports in a way that requires context), fix manually.

- [ ] **Step 13.4: Run mypy on changed files**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run mypy src/custom_sam_peft/data/splitter.py src/custom_sam_peft/data/val_source.py src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py src/custom_sam_peft/train/runner.py src/custom_sam_peft/train/trainer.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/eval/runner.py src/custom_sam_peft/runs/bundle.py src/custom_sam_peft/cli/run_cmd.py src/custom_sam_peft/cli/doctor_cmd.py src/custom_sam_peft/diagnostics.py src/custom_sam_peft/config/schema.py
```

Expected: clean. Fix any type errors.

### Task 13d: Commit any lint/format fixes

- [ ] **Step 13.5: Commit fixes if any**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split
git status --short
# If anything is staged or modified after lint/format:
git add -u
git commit -m "style: ruff format + mypy fixes for #71"
```

If nothing changed (everything was already clean), skip this step.

### Task 13e: Final readiness check

- [ ] **Step 13.6: Verify the full set of spec deliverables landed**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && ls src/custom_sam_peft/data/splitter.py src/custom_sam_peft/data/val_source.py configs/examples/coco_text_no_val.yaml configs/examples/coco_text_auto_split.yaml tests/unit/test_splitter.py tests/unit/test_val_source.py tests/unit/test_trainer_no_val.py
```

Expected: all seven new files exist.

- [ ] **Step 13.7: Smoke-run `csp doctor --config` against both new example YAMLs**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run custom-sam-peft doctor --config configs/examples/coco_text_no_val.yaml 2>&1 | tail -20
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && uv run custom-sam-peft doctor --config configs/examples/coco_text_auto_split.yaml 2>&1 | tail -20
```

Expected: both runs exit 0; the first shows `val mode none`; the second shows `val mode auto_split` plus fraction and seed.

- [ ] **Step 13.8: Push the branch (orchestrator close-out)**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-data-no-val-auto-split && git push -u origin feat/data-no-val-auto-split
```

The PR open + CI watch happens via the orchestrator close-out step (per `~/.claude/CLAUDE.md` Implementation-Orchestrator pipeline §4) — not part of this plan.

---

## Definition of done

All items below must be checked before marking the PR ready:

- [ ] **Schema:** `data.val` is optional; `val_split` is added; both mutex/HF-compat validators wired (spec §3.1-§3.3).
- [ ] **Splitter:** `data/splitter.py` exists with `SplittableItem`, `SplitResult`, `stratified_split` per spec §4.1-§4.5; 11 unit tests pass.
- [ ] **Resolver:** `data/val_source.py` exists with `ValSource`, `resolve_val_source`, `save_val_source`, `load_val_source`, `_enumerate_coco_items`, `_enumerate_hf_items`, `_log_val_source`; ~13 unit tests pass; atomic save via tmp + os.replace.
- [ ] **Adapters:** `COCODataset(image_ids=…)` and `HFDataset(row_indices=…)` accept the subset param; builders pick up `cfg["_resolved_image_ids"][pipeline]`; image-level leak invariant test passes.
- [ ] **Runner:** `train/runner.py` resolves → saves → injects → builds; `_build_dataset_from_dict` replaces `_build_dataset`; `val_ds: Dataset | None` flows to Trainer.
- [ ] **Trainer:** `val_ds: Dataset | None` accepted; `fit()` short-circuits eval/panel/end-of-run eval and writes the no-val metrics.json note; tracker hparams injection reads `val_source.json`.
- [ ] **Loop:** `run_epoch` no longer takes `val_ds`; single call site in Trainer updated.
- [ ] **Eval:** `--split val` guard added between peft and test checks; standalone `--split val` with `val_split` recomputes and injects `_resolved_image_ids`.
- [ ] **Bundle:** `write_bundle` widened to `metrics_report: MetricsReport | None` + `val_dataset: Dataset | None`; `_write_summary_no_val` produces the "— no-val" headline + "No validation set" line; no `samples/` dir written in no-val mode.
- [ ] **Run CLI:** `cli/run_cmd.py` reads saved `val_source.json`; skips eval phase entirely when `mode == "none"`; degraded "done" line shows `mAP=n/a (no val)`.
- [ ] **Doctor:** `--config` flag added; `DataReport` populated only when set; "Data" table renders mode + (path | fraction+seed); no `--config` → splitter/enumerator not invoked.
- [ ] **Examples:** `configs/examples/coco_text_no_val.yaml` + `configs/examples/coco_text_auto_split.yaml` exist and validate; both CLI templates show a commented `val_split:` block.
- [ ] **Integration:** e2e auto-split + e2e no-val tests pass on `tiny_coco` + LoRA stub.
- [ ] **Coverage:** `src/custom_sam_peft/data/` ≥80% coverage after spec lands.
- [ ] **Lint/format:** ruff + mypy clean.

---

## Self-review

**1. Spec coverage** — Walking spec §2's file table and §11's deliverables, mapped to plan steps:

| Spec deliverable | Plan step(s) |
| --- | --- |
| `data/splitter.py` (§2, §4, §11 Step 2) | Step 1 |
| `data/val_source.py` (§2, §5, §11 Step 3) | Step 3 |
| `config/schema.py` mods (§2, §3, §11 Step 1) | Step 2 |
| `data/coco.py` mods (§2, §6.1, §6.3, §11 Step 4) | Step 4 |
| `data/hf.py` mods (§2, §6.2, §6.3, §11 Step 4) | Step 4 |
| `train/runner.py` mods (§2, §6.4, §11 Step 6) | Step 5 |
| `train/trainer.py` mods (§2, §7.1, §11 Step 5) | Step 6 |
| `train/loop.py` mods (§2, §7.2, §11 Step 5) | Step 6 |
| `eval/runner.py` mods (§2, §7.4, §11 Step 7) | Step 7 |
| `runs/bundle.py` mods (§2, §7.5, §11 Step 8) | Step 8 |
| `cli/run_cmd.py` mods (§2, §7.6, §11 Step 9) | Step 9 |
| `cli/doctor_cmd.py` mods (§2, §7.7, §11 Step 10) | Step 10 |
| `diagnostics.py` mods (§2, §7.7, §11 Step 10) | Step 10 |
| `configs/examples/coco_text_no_val.yaml` (§2, §11 Step 11) | Step 11 |
| `configs/examples/coco_text_auto_split.yaml` (§2, §11 Step 11) | Step 11 |
| `cli/templates/coco_text_lora.yaml` + `coco_text_qlora.yaml` (§2, §11 Step 11) | Step 11 |
| `<run_dir>/val_source.json` 3 schemas + atomic write (§8.1) | Step 3 |
| Resume flow + `vs_saved` vs current cfg (§8.2-§8.4) | Step 3 (resolver loads), Step 5 (runner orchestrates) |
| Tracker hparams (§8.5) | Step 6 |
| Splitter tests (§9.1) | Step 1 |
| Resolver tests (§9.2) | Step 3 |
| Schema tests (§9.3) | Step 2 |
| Adapter tests + leak invariant (§9.4) | Step 4 |
| Trainer no-val tests (§9.5) | Step 6 |
| Bundle no-val tests (§9.6) | Step 8 |
| Runner tests (§9.7) | Step 5 |
| Doctor tests (§9.8) | Step 10 |
| Eval tests (§9.9) | Step 7 |
| Integration tests (§9.10) | Step 12 |
| GPU tests unchanged (§9.11) | Implicit — no GPU step in this plan |
| Coverage gate ≥80% on `data/` (§9.12) | Step 13.2 |

Every spec deliverable maps to at least one task. The `vs_saved` config-drift WARN behavior described in §8.4 is built into Step 3's resolver (via `resolve_val_source` returning the saved record on resume); the WARN messages are owned by `_log_val_source` and exercised in Step 3.4 tests.

**2. Placeholder scan** — No "TBD", "TODO", "implement later", or "fill in details" phrases. Step 12.3 includes a conditional ("If extending is not straightforward, skip — unit coverage suffices") — this is **not** a placeholder; it's an explicit decision pinned to the unit-test coverage that already validates the contract. Step 4.2's note about helper-name lookup is also explicit (subagent inspects `tests/unit/test_data_hf.py:60-140`) — not a placeholder.

**3. Type consistency** — Cross-step type checks:

- `ValSource.train_ids: tuple[str, ...] | None` — consistent across Step 3 (definition), Step 5 (runner reads `list(vs.train_ids)`), Step 7 (eval reads `list(vs.val_ids)`), Step 9 (`_build_val_dataset(cfg, vs)` reads `list(vs.val_ids)`).
- `ValSplitConfig.fraction: float` with `gt=0.0, le=0.5` — Step 2 (schema), Step 3 (resolver passes `cfg.data.val_split.fraction` into `stratified_split`).
- `Dataset | None` — Step 5 returns it from `run_training`; Step 6 accepts it in `Trainer.__init__`; Step 8 accepts it in `write_bundle`; Step 9 binds it in `_orchestrate`.
- `_resolved_image_ids` dict key — Step 4 reads `cfg.get("_resolved_image_ids") or {}).get(pipeline)`; Step 5 writes `data_cfg_dict["_resolved_image_ids"] = {"train": list(vs.train_ids), "eval": list(vs.val_ids)}`; Step 7 writes `{"eval": list(vs.val_ids)}`; Step 9 writes `{"eval": list(vs.val_ids)}`. All three writers and the reader use the same key name and same nested-dict shape — verified.
- `DataReport.val_mode: Literal["explicit", "auto_split", "none"]` — Step 10 (definition); same string set as `ValSource.mode`. No drift.

No type mismatches found.

**4. Dependency graph** — explicit in the "Dependencies & Parallelization" section. Each step has a "Depends on:" line in its header. Parallelizable sets are spelled out for the orchestrator.

**5. Test-first ordering** — Each implementation step starts with a "Task Na: Failing tests" subsection that writes the test first, runs it to confirm it fails, then implements. Steps 1, 2, 3, 4, 5, 6, 7, 8, 10 all follow this pattern. Steps 9, 11, 12 don't write tests first because (9) is purely glue covered by Step 12's integration tests, (11) is non-code, (12) is the integration tests themselves.

**6. Final verification gate** — Step 13 enforces (a) all CPU tests pass, (b) coverage gate per spec §9.12, (c) lint+format clean. The orchestrator close-out (PR + CI watch) is out of plan scope per `~/.claude/CLAUDE.md`.
