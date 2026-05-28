# Text-only prompts refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-28-text-only-prompts-design.md`](../specs/2026-05-28-text-only-prompts-design.md)
**Issue:** [#126](https://github.com/NguyenJus/custom-sam-peft/issues/126) — `refactor: remove non-text primary prompt pathways`
**Branch / worktree:** `refactor+text-only-prompts-126`

**Goal:** Delete the half-built `prompt_mode='bbox'` / `BoxPrompts` primary-prompt surface and replace the wrapper's `box_hints=...` kwarg with a new `support=SupportPrompts(boxes=...)` auxiliary-prompt container, in a single PR with no back-compat shims.

**Architecture:** Six-phase ordering, anchored on three pivots: (1) introduce `SupportPrompts` so the wrapper can adopt it before `BoxPrompts` leaves the tree; (2) swap `Sam3Wrapper.forward(box_hints=...)` to `support=SupportPrompts(...)` together with its two in-repo callers (`train/loop.py`, `train/trainer.py`); (3) delete `BoxPrompts`/`PromptMode`/`DataConfig.prompt_mode` and the three hand-rolled `prompt_mode=='bbox'` guards once nothing references them. The `box_hint` curriculum (sampler, schedule, metric) is preserved verbatim — only the kwarg name at the wrapper boundary changes.

**Tech Stack:** Python 3.12, Pydantic v2 (schema/validation), frozen dataclasses (data seam), pytest + pytest-cov (TDD, 80% gate), ruff + mypy + markdownlint-cli2 + yamllint (CI gates).

---

## Sequencing rationale (read before starting)

1. **`SupportPrompts` lands before any consumer.** Task 1 adds the dataclass and its test alongside the still-present `BoxPrompts`. Without this, Task 2 cannot import the new symbol.
2. **Wrapper signature + its two callers move as one batch.** `Sam3Wrapper.forward` switches from `box_hints=` to `support=SupportPrompts(...)` in Task 2; `train/loop.py:294` and `train/loop.py:313` (callers) plus `train/trainer.py:485` (eval panel) must change in the same task to keep `import` + `pytest --collect-only` green. The wrapper unit tests are rewritten in the same task.
3. **Data layer collapses to text-only in one batch.** Task 3 removes `prompt_mode` from `data/coco.py` and `data/hf.py` ctors, validators, branches, and `build_coco` / `build_hf`. Every test that constructs a `COCODataset` / `HFDataset` with `prompt_mode=...` is updated in the same task (broader than spec §9 — see §"Footprint" below).
4. **`BoxPrompts` deletion is sequenced last among the data-layer edits.** Task 4 deletes `BoxPrompts` and collapses `Prompts = TextPrompts | BoxPrompts` to `Prompts = TextPrompts`. It runs after Task 2 (wrapper no longer references `BoxPrompts`) AND Task 3 (data adapters no longer emit it). Test files that imported `BoxPrompts` for the now-deleted mixed-batch / bbox-mode tests are scrubbed in the same task.
5. **Schema field removal lands together with all configs and fixtures.** Task 5 deletes `PromptMode` + `DataConfig.prompt_mode`. Because `_Strict` sets `extra="forbid"`, *every* YAML carrying a `prompt_mode:` line — the 8 example configs, `config_full.yaml`, the wizard render-test YAML — must lose that line in the same commit, else `load_config` will fail for any of them.
6. **Guards are removed together but only after the schema gate is in place.** Task 6 deletes the three hand-rolled `prompt_mode=='bbox'` runtime guards (`train/trainer.py`, `cli/train_cmd.py`, `cli/run_cmd.py`) and the tests that exercised them. It runs after Task 5 so the schema's `extra_forbidden` error is the only remaining gate. Note: with Task 5 having removed `DataConfig.prompt_mode`, the three `cfg.data.prompt_mode == "bbox"` checks become attribute errors at *import-resolve time* anyway — Task 6 deletes them before that surfaces. (In practice Tasks 5 and 6 can be a single batched commit.)
7. **Docs + CHANGELOG + final grep are the close-out.** Task 7 updates `docs/ARCHITECTURE.md`, `docs/config-schema.md`, `CHANGELOG.md`, and adds the anchoring comment at `models/sam3.py:616`, then runs the final `grep -rn 'prompt_mode\|BoxPrompts' configs/ src/ docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md` (expected: 0 matches) and the full `pytest` / `mypy` / `ruff` suite.

### Footprint (planner-verified — broader than spec §9)

The spec's §9 test table names ~12 test files. A `grep -rn` across the worktree at branch HEAD finds `prompt_mode=...` or `BoxPrompts` references in **47 test files** (147 lines, 89 of which are `prompt_mode="text"` / `prompt_mode="bbox"` kwargs in `COCODataset` / `HFDataset` constructors). Every one of those kwargs must be deleted in the same task that removes the `prompt_mode` ctor param from those classes (Task 3), else collection fails. The full list of touched test files is given in Task 3 and Task 4 below. Don't be surprised by the cross-cutting test edit — it is mechanical.

### Breaking-change note (NOT a migration step — spec §10.3)

This PR is a clean breaking change with **no shim and no migration**:

- Any config carrying `data.prompt_mode:` (any value) now **fails to load** with a Pydantic `extra_forbidden` `ValidationError`. The 8 shipped `configs/examples/*.yaml` and the unified `config_full.yaml` template are stripped in Task 5; downstream users must delete the line from their own YAML.
- `Sam3Wrapper.forward(box_hints=...)` is replaced by `Sam3Wrapper.forward(support=SupportPrompts(boxes=...))` in one shot. The only in-repo callers are migrated in Task 2; external callers migrate per CHANGELOG §10.3.
- `BoxPrompts` and `PromptMode` are removed from `data.base` / `config.schema`. Any external import of either name breaks at import time.

These are consequences to document, not tasks to mitigate.

---

## File structure

**Source — modify:**

- `src/custom_sam_peft/data/base.py` — add `SupportPrompts`; later delete `BoxPrompts`; collapse `Prompts = TextPrompts | BoxPrompts` → `Prompts = TextPrompts`.
- `src/custom_sam_peft/models/sam3.py` — drop `BoxPrompts` import; add `SupportPrompts` import; change `Sam3Wrapper.forward` signature; simplify `_validate_inputs`; update docstring; add anchoring comment at line 616.
- `src/custom_sam_peft/data/coco.py` — drop `prompt_mode` ctor param + validator + field + bbox branch; collapse `_pack_example` to always emit `TextPrompts`; drop `prompt_mode=cfg["prompt_mode"]` in `build_coco`.
- `src/custom_sam_peft/data/hf.py` — same pattern as `coco.py`.
- `src/custom_sam_peft/config/schema.py` — delete `PromptMode` Literal, the `"PromptMode"` entry in `__all__`, `DataConfig.prompt_mode` field, and the `TextPromptConfig` docstring reference to `prompt_mode='text'`.
- `src/custom_sam_peft/train/trainer.py` — delete the bbox guard at lines 135–140; replace `box_hints=None` on line 485 with `support=None`.
- `src/custom_sam_peft/train/loop.py` — replace `box_hints=micro_hints` / `box_hints=hints_g` with `support=SupportPrompts(boxes=...)` at lines 294 / 313; add `SupportPrompts` to the existing `from custom_sam_peft.data.base import ...` line.
- `src/custom_sam_peft/cli/train_cmd.py` — delete the bbox guard at lines 48–52.
- `src/custom_sam_peft/cli/run_cmd.py` — delete the bbox guard at lines 191–195.

**Configs / templates / fixtures — modify:**

- `configs/examples/coco_text_lora.yaml`, `coco_text_qlora.yaml`, `coco_text_lora_subset.yaml`, `coco_text_no_val.yaml`, `coco_text_auto_split.yaml`, `min_gpu_qlora.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml` — strip the `prompt_mode: text` line (8 files).
- `src/custom_sam_peft/cli/templates/config_full.yaml` — strip the `prompt_mode: text` line.
- `tests/fixtures/tiny_sam3_stub.py` — drop the `BoxPrompts` mention in the class docstring.
- `tests/conftest.py` — drop the `prompt_mode="bbox"` line from the `tiny_coco_dataset` fixture (line 163).

**Tests — rewrite / scrub:**

The cross-cutting test scrub touches every file the planner found via `grep -rn 'prompt_mode\|BoxPrompts' tests/`. Concretely:

- **`tests/unit/test_data_base.py`** — drop `BoxPrompts` from imports + delete the `test_text_prompts_and_box_prompts_are_distinct_types` test; add `test_support_prompts_dataclass` (Task 1, then Task 4 for the deletion).
- **`tests/unit/test_sam3_wrapper.py`** — drop the `BoxPrompts` import + mixed-batch test (Task 2).
- **`tests/unit/test_sam3_wrapper_box_hints.py`** → renamed to **`tests/unit/test_sam3_wrapper_support.py`** — rewrite to use `support=SupportPrompts(boxes=...)`; drop the `BoxPrompts`-with-`box_hints` rejection test (Task 2).
- **`tests/unit/test_data_collate.py`** — drop the `BoxPrompts` import + construction + `isinstance` assertion (Task 4).
- **`tests/unit/test_data_coco.py`** — drop `BoxPrompts` import; delete every `prompt_mode="bbox"` test (incl. `test_class_names_dense_and_ordered`, `test_len_drops_empty_after_iscrowd`, `test_multiplex_truncation_box`, `test_getitem_bbox_mode_returns_BoxPrompts`, `test_polygon_segmentation_decoded`, `test_rle_segmentation_decoded`, `test_iscrowd_skipped`, `test_dropped_empty_image_logged_once`, `test_image_resize_geometry`, `test_sparse_to_dense_remap`) OR convert them to text-mode if the underlying assertion still applies; strip `prompt_mode=...` from every remaining call site (Task 3).
- **`tests/unit/test_data_hf.py`** — drop `BoxPrompts` import; delete `test_getitem_bbox_mode`, `test_bbox_format_xywh_conversion`, and other `prompt_mode="bbox"` tests; strip the kwarg from text-mode constructions (Task 3).
- **`tests/unit/test_data_hf_limit.py`** — drop the `ds._prompt_mode = "bbox"` line at 28; the attribute no longer exists (Task 3).
- **`tests/unit/test_data_apply_transforms_bbox_drop.py`** — strip the `prompt_mode="bbox"` kwarg at lines 148 / 272 / 310 / 334; tests still test bbox-drop in `_apply_transforms` (unchanged behavior — the bbox augmentation pipeline is orthogonal to `prompt_mode`) (Task 3).
- **`tests/unit/test_data_coco_limit.py`** — strip the `prompt_mode="bbox"` kwargs at lines 25, 131 and the `"prompt_mode": "bbox"` dict entries at lines 68, 105 (Task 3).
- **`tests/unit/test_config_schema.py`** — change `_minimal_dict()`'s `"prompt_mode": "bbox"` to drop the key entirely; delete `test_invalid_prompt_mode_rejected`; drop `"prompt_mode": "text"` from `minimal_data_config_dict` (line 201); add `test_prompt_mode_rejected_by_schema` (Task 5).
- **`tests/unit/test_config_loader.py`** — strip `prompt_mode: bbox` from line 23 and `prompt_mode: text` from line 150 (the YAML strings embedded in this test) (Task 5).
- **`tests/unit/test_data_schema_extensions.py`** — strip `"prompt_mode": "bbox"` at line 104 and `prompt_mode="text"` at line 156 (Task 5).
- **`tests/unit/test_config_examples.py`** — drop the `prompt_mode=...` arg from the test's embedded dict at line 46 (Task 5).
- **`tests/unit/test_cli.py`** — delete `test_train_rejects_bbox_prompt_mode` (lines 95–115); strip the `prompt_mode: text` lines from the YAML strings at 130, 175, 209 (Task 6 — the deletion of the bbox-rejection test is coupled to the guard deletion, but the YAML scrubs are coupled to Task 5; do both in Task 6 once the schema field is gone).
- **`tests/unit/test_cli_init.py`** — strip the `assert "prompt_mode: text" in body` at line 62 (the template no longer carries it) (Task 5).
- **`tests/unit/test_cli_doctor.py`** — drop the `"prompt_mode": "text"` entry at line 131 (Task 5).
- **`tests/unit/test_cli_doctor_config.py`** — strip `prompt_mode: bbox` from the YAML string at line 46 (Task 5). Also re-check what this test asserts — it may have been validating the bbox guard; if so delete the test.
- **`tests/unit/test_cli_export.py`** — strip `prompt_mode: text` from the YAML string at line 26 (Task 5).
- **`tests/unit/test_eval_runner_gate.py`** — drop `"prompt_mode": "text"` entry at line 22 (Task 5).
- **`tests/unit/test_eval_runner.py`** — drop `"prompt_mode": "text"` entry at line 23 (Task 5).
- **`tests/unit/test_load_sam31_callsites.py`** — drop `cfg.data.prompt_mode = "text"` at line 22 (Task 5).
- **`tests/unit/test_stubs_raise.py`** — delete the bbox-rejection test (lines 22, 40–45) (Task 6).
- **`tests/unit/test_trainer_guards.py`** — drop `prompt_mode` from `_cfg`'s sig/body (lines 20, 28); delete `test_trainer_rejects_bbox_prompt_mode` (lines 50–55) (Task 6).
- **`tests/unit/test_trainer_nan_behavior.py`** — drop `prompt_mode="text"` at lines 55, 80 (Task 3).
- **`tests/unit/test_trainer_no_val.py`** — drop `prompt_mode="text"` at lines 45, 62 (Task 3).
- **`tests/unit/test_trainer_run_dir.py`** — drop every `prompt_mode="text"` (lines 38, 129, 197, 204, 221, 352, 359, 376, 452, 459, 476) (Task 3).
- **`tests/unit/test_train_runner.py`** — drop the `cfg.data.prompt_mode = "text"` at line 20 and every `prompt_mode="text"` (lines 128, 191) (Task 3 for the kwargs, Task 5 for the `cfg.data.prompt_mode = "text"` line).
- **`tests/unit/test_val_source.py`** — drop `prompt_mode="text"` at line 50 and the `"prompt_mode": "text"` dict entry at line 225 (Task 3 / Task 5).
- **`tests/unit/test_train_step.py`** — drop `prompt_mode="text"` at line 37 (Task 3).
- **`tests/unit/test_train_loop_legacy_k1.py`** — drop `prompt_mode="text"` at line 44 (Task 3).
- **`tests/unit/test_train_loop_multiplex.py`** — drop `prompt_mode="text"` at line 42 (Task 3).
- **`tests/unit/test_train_checkpoint.py`** — drop `prompt_mode="text"` at line 40 (Task 3).
- **`tests/unit/test_tracking_noop.py`** — drop `prompt_mode="text"` at line 30 (Task 3).
- **`tests/unit/test_tracking_tensorboard.py`** — drop `prompt_mode="text"` at line 32 (Task 3).
- **`tests/unit/test_tracking_wandb.py`** — drop `prompt_mode="text"` at line 34 (Task 3).
- **`tests/unit/test_checkpoint_roundtrip.py`** — drop `prompt_mode="text"` at line 47 (Task 3).
- **`tests/unit/cli/test_setup_wizard.py`** — drop the `assert "prompt_mode: text" in rendered` at line 176 (Task 5).
- **`tests/integration/test_train_resume.py`** — drop `prompt_mode="text"` at lines 43, 62 (Task 3).
- **`tests/integration/test_trainer_evaluator_seam.py`** — drop `prompt_mode="text"` at line 81 (Task 3).
- **`tests/integration/test_train_end_to_end.py`** — drop every `prompt_mode="text"` (lines 51, 77, 141, 195, 235, 276, 341, 388, 439) (Task 3).
- **`tests/integration/test_cli_run.py`** — strip `prompt_mode: {prompt}` placeholder line at 34; delete `test_run_rejects_bbox_prompt_mode` (line 257 onward — the entire test) (Task 6).
- **`tests/integration/test_train_then_eval.py`** — drop `prompt_mode="text"` at line 57 (Task 3).
- **`tests/integration/test_peft_extensibility.py`** — drop `prompt_mode="text"` at line 58 (Task 3).
- **`tests/integration/test_tracker_swap.py`** — drop `prompt_mode="text"` at line 103 (Task 3).

**Docs — modify:**

- `docs/ARCHITECTURE.md` — replace line 15's `Prompts (TextPrompts | BoxPrompts)` with `Prompts (= TextPrompts), SupportPrompts`; add the text-primary invariant sentence near the top (Task 7).
- `docs/config-schema.md` — delete the `data.prompt_mode` row at line 52 (Task 7).
- `CHANGELOG.md` — add the spec §10.3 entry under `## [Unreleased]` (Task 7).

---

## Routing summary (for orchestrator)

| Task | Suggested impl model | Reviewer | Notes |
| --- | --- | --- | --- |
| 0 | n/a | n/a | Pre-flight; orchestrator runs commands directly. |
| 1 | sonnet/high | sonnet/high | Add dataclass + one test; foundation. |
| 2 | sonnet/high | sonnet/high | Wrapper signature change + 2 call-site rewires + wrapper unit tests rewrite. |
| 3 | sonnet/high | sonnet/high | Largest task — data-layer collapse + cross-cutting test scrub. |
| 4 | sonnet/high | sonnet/high | Delete `BoxPrompts` once nothing references it. |
| 5 | sonnet/high | sonnet/high | Schema field deletion + configs + template + remaining test scrub. |
| 6 | sonnet/high | sonnet/high | Delete 3 guards + their tests. |
| 7 | sonnet/high | sonnet/high | Docs + CHANGELOG + anchoring comment + final green-gate. |

**Parallelization:** Tasks 5 and 6 are file-disjoint after Task 4 lands and could in principle be dispatched in parallel; the orchestrator may serialize them in one batched commit for review simplicity. All other task pairs share files or chain logically (T1→T2, T1→T3, T2+T3→T4, T4→T5/T6, T5+T6→T7).

---

## Task 0: Verify clean baseline

**Files:** none (commands only)

Pre-flight gate. Orchestrator runs these directly.

- [ ] **Step 0a: Confirm working tree is clean and on the right branch**

```bash
git status
git branch --show-current
```

Expected: branch `refactor+text-only-prompts-126`, working tree clean (only the spec + this plan if not yet committed). If dirty, halt.

- [ ] **Step 0b: Confirm venv has dev extras**

```bash
uv sync --extra dev
```

Expected: resolves without errors. Required once per fresh worktree; subsequent runs are fast.

- [ ] **Step 0c: Confirm baseline unit tests pass**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green. If anything is red before changes, halt and surface — later tasks cannot validate against a broken baseline.

- [ ] **Step 0d: Confirm ruff is clean and mypy is clean**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

Expected: all clean.

---

## Task 1: Add `SupportPrompts` (foundation)

**Files:**

- Modify: `src/custom_sam_peft/data/base.py` (add `SupportPrompts` between `TextPrompts` and the `Prompts` alias)
- Modify: `tests/unit/test_data_base.py` (add `test_support_prompts_dataclass`)

**Objective:** Introduce the `SupportPrompts` frozen dataclass next to `TextPrompts`. Do NOT delete `BoxPrompts` yet — Task 2 and Task 3 still reference it. Add one test exercising the new dataclass.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_data_base.py` (place after the existing tests, before the `_FakeDataset` helper — order doesn't matter but the import at the top must already include `SupportPrompts`; do not touch the existing imports until Step 3):

```python
def test_support_prompts_dataclass() -> None:
    """SupportPrompts is a frozen dataclass with one optional `boxes` field."""
    import dataclasses

    import pytest

    from custom_sam_peft.data.base import SupportPrompts

    # Default ctor: boxes is None.
    s_default = SupportPrompts()
    assert s_default.boxes is None
    assert dataclasses.is_dataclass(s_default)

    # With per-image boxes (some None, some (M_i, 4) tensors).
    s = SupportPrompts(boxes=[torch.zeros(2, 4), None])
    assert s.boxes is not None
    assert s.boxes[0].shape == (2, 4)
    assert s.boxes[1] is None

    # Frozen: direct field assignment raises FrozenInstanceError.
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.boxes = None  # type: ignore[misc]

    # dataclasses.replace works (frozen instances support copy-with-changes).
    s2 = dataclasses.replace(s, boxes=None)
    assert s2.boxes is None
```

Update the top-of-file import block from:

```python
from custom_sam_peft.data.base import (
    BoxPrompts,
    Dataset,
    Example,
    Instance,
    TextPrompts,
    is_dataset,
)
```

to:

```python
from custom_sam_peft.data.base import (
    BoxPrompts,
    Dataset,
    Example,
    Instance,
    SupportPrompts,
    TextPrompts,
    is_dataset,
)
```

(`BoxPrompts` stays — Task 4 removes it.)

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/unit/test_data_base.py -v
```

Expected: `test_support_prompts_dataclass` FAILS with `ImportError: cannot import name 'SupportPrompts' from 'custom_sam_peft.data.base'`. Other tests in the file remain green.

- [ ] **Step 3: Add `SupportPrompts` to `data/base.py`**

In `src/custom_sam_peft/data/base.py`, insert this dataclass between `BoxPrompts` (currently ends at line 27) and the `Prompts = TextPrompts | BoxPrompts` alias (currently line 30). Place it after `TextPrompts` and `BoxPrompts`, before `Prompts`:

```python
@dataclass(frozen=True)
class SupportPrompts:
    """Auxiliary localization prompts that ride alongside TextPrompts.

    Never replaces text; never used at inference. Today carries only optional
    per-image GT box hints (the ``box_hint`` curriculum from #14). Future fields
    (masks, positive points, negative points) will be added when their
    plumbing is built — see #126 §12.

    Length convention for ``boxes`` (identical to the legacy ``box_hints`` kwarg):

    - Length is ``B*K`` (image-major, class-minor), where ``K`` is the number
      of class prompts per multiplex forward call.
    - Each element is either ``None`` (no hint for that image/class slot) or a
      ``(M_i, 4)`` float tensor of absolute pixel xyxy boxes.
    - For the common ``K=1`` case, length is ``B`` and the ordering is
      trivially image-major.
    """

    boxes: list[torch.Tensor | None] | None = None
```

Resulting region of the file (lines ~11–35):

```python
@dataclass(frozen=True)
class TextPrompts:
    """Open-vocabulary class names used as prompts for one image."""

    classes: list[str]


@dataclass(frozen=True)
class BoxPrompts:
    """Per-image box prompts and their target class ids.

    `boxes` is `(N, 4)` xyxy in pixel coords; converted to normalized cxcywh
    at the collator boundary before reaching the matcher/losses.
    """

    boxes: torch.Tensor  # (N, 4) xyxy, pixel coords
    class_ids: torch.Tensor  # (N,) int64


@dataclass(frozen=True)
class SupportPrompts:
    """..."""

    boxes: list[torch.Tensor | None] | None = None


Prompts = TextPrompts | BoxPrompts
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/test_data_base.py -v
```

Expected: PASS (all tests in the file, including the new `test_support_prompts_dataclass`).

- [ ] **Step 5: Lint + mypy + commit**

```bash
uv run ruff check src/custom_sam_peft/data/base.py tests/unit/test_data_base.py
uv run ruff format --check src/custom_sam_peft/data/base.py tests/unit/test_data_base.py
uv run mypy src/custom_sam_peft/data/base.py
git add src/custom_sam_peft/data/base.py tests/unit/test_data_base.py
git commit -m "feat(data): introduce SupportPrompts auxiliary-prompt container (#126)"
```

---

## Task 2: Swap `Sam3Wrapper.forward(box_hints=...)` → `support=SupportPrompts(...)`

**Files:**

- Modify: `src/custom_sam_peft/models/sam3.py` (imports line 30; `Sam3Wrapper` docstring lines 178–204; `forward` signature lines 223–231; `_validate_inputs` lines 233–302)
- Modify: `src/custom_sam_peft/train/loop.py` (imports line 26; call sites lines 294 and 313)
- Modify: `src/custom_sam_peft/train/trainer.py` (eval-panel forward at line 485)
- Modify: `tests/unit/test_sam3_wrapper.py` (drop `BoxPrompts` import + mixed-batch test)
- Rename: `tests/unit/test_sam3_wrapper_box_hints.py` → `tests/unit/test_sam3_wrapper_support.py` (rewrite contents)

**Objective:** Change the public wrapper API from `box_hints=...` to `support=SupportPrompts(boxes=...)`. Update the two in-repo callers (`train/loop.py` and `train/trainer.py:485`) in the same batch so `pytest --collect-only` stays green. Rewrite the wrapper unit tests. Do NOT touch `BoxPrompts`-from-data tests yet (Task 4 deletes those).

- [ ] **Step 1: Rewrite `tests/unit/test_sam3_wrapper.py` — drop the mixed-batch test**

Apply two edits:

1. Change line 8's import from:

```python
from custom_sam_peft.data.base import BoxPrompts, TextPrompts
```

to:

```python
from custom_sam_peft.data.base import TextPrompts
```

2. Delete the entire `test_wrapper_rejects_mixed_prompt_variants` function (current lines 36–45). After the deletion the file goes straight from `test_wrapper_rejects_multi_class_text_prompts` to `test_wrapper_rejects_batch_size_mismatch`.

- [ ] **Step 2: Replace `tests/unit/test_sam3_wrapper_box_hints.py` → `tests/unit/test_sam3_wrapper_support.py`**

Delete `tests/unit/test_sam3_wrapper_box_hints.py` and create `tests/unit/test_sam3_wrapper_support.py` with this body:

```python
"""Sam3Wrapper.forward accepts a SupportPrompts container with strict validation."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.data.base import SupportPrompts, TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _wrapper() -> Sam3Wrapper:
    return Sam3Wrapper(TinySam3Stub(), image_size=8, mask_size=8)


def test_forward_accepts_none_support() -> None:
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = w(images, prompts, support=None)
    assert set(out) >= {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_forward_default_support_is_none() -> None:
    """Omitting `support` is equivalent to passing `support=None`."""
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = w(images, prompts)
    assert set(out) >= {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_forward_accepts_per_image_support_boxes() -> None:
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    # K=1, so SupportPrompts.boxes length == B*K == 2 (image-major / class-minor).
    support = SupportPrompts(boxes=[torch.tensor([[1.0, 2.0, 3.0, 4.0]]), None])
    out = w(images, prompts, support=support)
    assert "pred_masks" in out


def test_forward_accepts_support_with_none_boxes() -> None:
    """SupportPrompts(boxes=None) is equivalent to support=None."""
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = w(images, prompts, support=SupportPrompts(boxes=None))
    assert "pred_masks" in out


def test_forward_rejects_mismatched_support_boxes_length() -> None:
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    # K=1, so expected length is B*K=2; passing length 1 should fail.
    with pytest.raises(ValueError, match=r"len.*boxes"):
        w(images, prompts, support=SupportPrompts(boxes=[None]))


def test_forward_rejects_wrong_support_box_shape() -> None:
    w = _wrapper()
    images = torch.zeros(1, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"])]
    bad = torch.zeros(2, 5)
    with pytest.raises(ValueError, match=r"\(M_i, 4\)|shape"):
        w(images, prompts, support=SupportPrompts(boxes=[bad]))
```

Note the new error-message regex: the new `_validate_inputs` error mentions `boxes` (from `support.boxes`), not `box_hints`. The implementation in Step 4 raises `f"len(support.boxes)={...} must equal ..."`; the regex `len.*boxes` matches both that and any future reword that keeps the word `boxes`.

- [ ] **Step 3: Run the wrapper tests to verify they fail**

```bash
uv run pytest tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py -v
```

Expected: FAIL — `test_sam3_wrapper_support.py` tests call `w(..., support=...)` but the wrapper still expects `box_hints=`; `BoxPrompts` import in `test_sam3_wrapper.py` succeeds but the mixed-batch test deletion is OK so that file passes (the file is in a transient state — drop the test but keep `BoxPrompts`-free imports).

- [ ] **Step 4: Update `models/sam3.py` — imports, docstring, `forward`, `_validate_inputs`**

In `src/custom_sam_peft/models/sam3.py`:

(a) Change line 30 from:

```python
from custom_sam_peft.data.base import BoxPrompts, Prompts, TextPrompts
```

to:

```python
from custom_sam_peft.data.base import Prompts, SupportPrompts, TextPrompts
```

(`BoxPrompts` is dropped from this import; it's still referenced elsewhere in the file, but not after the rest of this step.)

(b) Rewrite the `Sam3Wrapper` class docstring (lines 178–204) to drop the `BoxPrompts` mentions and reflect the new API. Replace the whole `"""..."""` block with:

```python
    """Thin wrapper around Meta's SAM 3.1 model.

    Contract:
      - ``forward(images, prompts, support=None)`` accepts a batch of B images
        and a list of B ``Prompts`` objects (always ``TextPrompts`` after #126).
      - ``support``: optional ``SupportPrompts`` carrying auxiliary localization
        prompts. Today the only field is ``boxes``: a flat list of length
        ``B·K``, ordered image-major / class-minor (all K class slots for image
        0, then all K class slots for image 1, …). Each element is either
        ``None`` (no geometric hint for that slot) or a ``(M_i, 4)`` float tensor
        of absolute pixel xyxy boxes. For the common K=1 case the list length
        equals B and the ordering is trivially image-major.
      - Each ``TextPrompts`` may contain 1..MULTIPLEX_CAP class names; all
        prompts in a batch must share the same class list in the same order
        (multiplex forward assumes a shared K-prompt vocabulary).
      - Returns Meta's native output dict unchanged.
      - ``forward`` supports both training (``model.train()``) and inference
        (``model.eval()``) modes.  The internal ``_Sam3ImageAdapter``
        hardcodes ``find_target=None`` when calling sam3's
        ``forward_grounding``; sam3's training-mode side-effect that would
        otherwise call ``back_convert(None)`` is neutralized by
        ``_patch_forward_grounding_skip_matching_on_none_target`` (installed
        by ``load_sam31``).  The trainer runs its own ``HungarianMatcher`` in
        ``custom_sam_peft.models.losses.total_loss``; ``out["indices"]`` written by
        sam3's matching call is never read by us.
    """
```

(c) Replace the `forward` method (lines 223–231) with:

```python
    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        support: SupportPrompts | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(images, prompts, support)
        box_hints = support.boxes if support is not None else None
        out: dict[str, Any] = self.model(images, prompts, box_hints=box_hints)
        return out
```

(`_Sam3ImageAdapter` still accepts the inner `box_hints=` kwarg unchanged; only the public `Sam3Wrapper.forward` boundary changes.)

(d) Rewrite `_validate_inputs` (lines 233–302) to drop the now-impossible branches and accept a `SupportPrompts`:

```python
    def _validate_inputs(
        self,
        images: Tensor,
        prompts: list[Prompts],
        support: SupportPrompts | None,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must be (B, {self.channels}, H, W); got shape {tuple(images.shape)}"
            )
        if images.shape[1] != self.channels:
            raise ValueError(
                f"images must be (B, {self.channels}, H, W); got "
                f"{images.shape[1]} channels in shape {tuple(images.shape)}"
            )
        b = images.shape[0]
        if len(prompts) != b:
            raise ValueError(f"len(prompts)={len(prompts)} must equal batch size {b}")
        if not prompts:
            return

        # After #126, Prompts == TextPrompts; the mixed-batch / BoxPrompts checks
        # are gone. Per-prompt: validate K ∈ [1, MULTIPLEX_CAP].
        for p in prompts:
            if not isinstance(p, TextPrompts) or not (1 <= len(p.classes) <= MULTIPLEX_CAP):
                raise ValueError(
                    f"TextPrompts must contain 1..MULTIPLEX_CAP (={MULTIPLEX_CAP}) classes per "
                    f"call (got {len(p.classes) if isinstance(p, TextPrompts) else 0}). Configure "
                    f"train.multiplex.classes_per_forward to bound K."
                )

        # Shared-class-list check (multiplex forward assumes shared K-prompt vocab).
        ref = tuple(cast(TextPrompts, prompts[0]).classes)
        for p in prompts[1:]:
            if tuple(cast(TextPrompts, p).classes) != ref:
                raise ValueError(
                    "All TextPrompts in a batch must carry the same class "
                    "list in the same order (multiplex forward assumes a "
                    "shared K-prompt vocabulary)."
                )

        boxes = support.boxes if support is not None else None
        if boxes is not None:
            # boxes length must be B*K (image-major / class-minor).
            k = len(cast(TextPrompts, prompts[0]).classes)
            expected_len = b * k
            if len(boxes) != expected_len:
                raise ValueError(
                    f"len(support.boxes)={len(boxes)} must equal batch size x classes "
                    f"({b}x{k}={expected_len})"
                )
            for i, h in enumerate(boxes):
                if h is None:
                    continue
                if h.ndim != 2 or h.shape[-1] != 4:
                    raise ValueError(
                        f"support.boxes[{i}] must have shape (M_i, 4); got {tuple(h.shape)}"
                    )
```

(`MULTIPLEX_CAP` is already imported via the module; `cast` is already imported on line 21 of `sam3.py`.)

- [ ] **Step 5: Update `train/loop.py` — add `SupportPrompts` import and rewire two call sites**

In `src/custom_sam_peft/train/loop.py`:

(a) Change line 26 from:

```python
from custom_sam_peft.data.base import Instance, TextPrompts
```

to:

```python
from custom_sam_peft.data.base import Instance, SupportPrompts, TextPrompts
```

(b) Change line 294 from:

```python
                        micro_out = _model(micro_imgs, micro_prompts, box_hints=micro_hints)
```

to:

```python
                        micro_out = _model(
                            micro_imgs, micro_prompts, support=SupportPrompts(boxes=micro_hints)
                        )
```

(c) Change line 313 from:

```python
                    out = model(images, prompts_g, box_hints=hints_g)
```

to:

```python
                    out = model(images, prompts_g, support=SupportPrompts(boxes=hints_g))
```

Do NOT touch `_box_hint_p`, `hints_g` construction (lines 235–247), the `box_hint/applied` metric (line 397), or the `box_hint/p` flush key (line 416) — those continue to read `n_hint_applied` and `p_t` from `StepResult`, unaffected by the kwarg rename.

- [ ] **Step 6: Update `train/trainer.py:485`**

In `src/custom_sam_peft/train/trainer.py`, change line 485 from:

```python
                        box_hints=None,
                    )
```

to:

```python
                        support=None,
                    )
```

No import needed (`None` doesn't require `SupportPrompts`). Lines 135–140 (the bbox guard) stay for now — Task 6 deletes them.

- [ ] **Step 7: Run the wrapper tests + train-step tests to verify they pass**

```bash
uv run pytest tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py tests/unit/test_train_step.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py -v
```

Expected: PASS — all wrapper tests, including the new `support=...` API; the train-step / loop tests still work because the call-site rewires preserve the kwarg semantics (`SupportPrompts(boxes=None)` ≡ `box_hints=None` ≡ `None`, and `SupportPrompts(boxes=hints_g)` carries the same list payload as the old `box_hints=hints_g`).

If `test_train_step.py` / `test_train_loop_*` fails because they still construct a `COCODataset(..., prompt_mode="text", ...)`, that is expected at this point — those tests load the data adapter which still requires `prompt_mode`. Task 3 will remove the kwarg in the same task that removes the param. For Task 2, only the wrapper tests are required to pass; loop tests that depend on data-adapter construction may be deferred.

If you need a sharper gate for Task 2, run the subset that doesn't exercise data adapters:

```bash
uv run pytest tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py -v
```

- [ ] **Step 8: Lint + mypy + commit**

```bash
uv run ruff check src/custom_sam_peft/models/sam3.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py
uv run ruff format --check src/custom_sam_peft/models/sam3.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py
uv run mypy src/custom_sam_peft/models/sam3.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py
git add src/custom_sam_peft/models/sam3.py src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py
git rm tests/unit/test_sam3_wrapper_box_hints.py
git commit -m "feat(models)!: swap Sam3Wrapper.forward box_hints= → support=SupportPrompts (#126)"
```

---

## Task 3: Collapse the data layer to text-only (`data/coco.py`, `data/hf.py`, all dataset-constructing tests)

**Files:**

- Modify: `src/custom_sam_peft/data/coco.py` (ctor lines 124–144, `_pack_example` lines 263–348, `build_coco` line 412)
- Modify: `src/custom_sam_peft/data/hf.py` (ctor lines 130–154, `_pack_example` lines 294–381, `build_hf` line 445)
- Modify: `tests/conftest.py` (line 163 — drop `prompt_mode="bbox"` from `tiny_coco_dataset`)
- Modify (drop `prompt_mode=...` kwargs everywhere, delete bbox-only tests):
  `tests/unit/test_data_coco.py`, `tests/unit/test_data_hf.py`, `tests/unit/test_data_hf_limit.py`,
  `tests/unit/test_data_apply_transforms_bbox_drop.py`, `tests/unit/test_data_coco_limit.py`,
  `tests/unit/test_trainer_nan_behavior.py`, `tests/unit/test_trainer_no_val.py`,
  `tests/unit/test_trainer_run_dir.py`, `tests/unit/test_train_runner.py`,
  `tests/unit/test_val_source.py`, `tests/unit/test_train_step.py`,
  `tests/unit/test_train_loop_legacy_k1.py`, `tests/unit/test_train_loop_multiplex.py`,
  `tests/unit/test_train_checkpoint.py`, `tests/unit/test_tracking_noop.py`,
  `tests/unit/test_tracking_tensorboard.py`, `tests/unit/test_tracking_wandb.py`,
  `tests/unit/test_checkpoint_roundtrip.py`,
  `tests/integration/test_train_resume.py`, `tests/integration/test_trainer_evaluator_seam.py`,
  `tests/integration/test_train_end_to_end.py`, `tests/integration/test_train_then_eval.py`,
  `tests/integration/test_peft_extensibility.py`, `tests/integration/test_tracker_swap.py`

**Objective:** Delete `prompt_mode` from `COCODataset` / `HFDataset` and their builders; always emit `TextPrompts`. In the same batch, strip the `prompt_mode=...` kwarg from every test that constructs one of these classes — else collection fails. Also delete the bbox-only tests in `test_data_coco.py` and `test_data_hf.py` (they exercise a path that no longer exists). Do NOT touch `BoxPrompts` yet (Task 4); the `test_data_coco.py` / `test_data_hf.py` `from custom_sam_peft.data.base import BoxPrompts, ...` lines stay until Task 4 removes them.

- [ ] **Step 1: Edit `src/custom_sam_peft/data/coco.py` — ctor + `_pack_example` + `build_coco`**

(a) `COCODataset.__init__` (current lines 124–144). Remove the `prompt_mode` param (line 128), the validator (lines 136–137), and the `self._prompt_mode` field (line 140):

Before:

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
        channels: int = 3,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._image_root = Path(images)
        self._channels = channels
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        ...
```

After:

```python
    def __init__(
        self,
        annotations: str,
        images: str,
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        seed: int = 0,
        image_ids: Iterable[int] | None = None,
        channels: int = 3,
    ) -> None:
        self._image_root = Path(images)
        self._channels = channels
        self._transforms = transforms
        ...
```

(b) `_pack_example` (current lines 263–348). Drop the `BoxPrompts` from the local import, drop the `if self._prompt_mode == "text":` gate, delete the `# bbox mode` branch entirely. The function becomes:

```python
    def _pack_example(
        self,
        raw: tuple[int, dict[str, Any], list[dict[str, Any]]],
        image_tensor: Any,
        out_bboxes: list[Any],
        out_masks: list[Any],
        out_classes: list[int],
    ) -> Example:
        """Assemble `Instance` objects and return the final `Example`."""
        import torch

        from custom_sam_peft.data.base import Instance, TextPrompts

        image_id, _rec, _anns = raw

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

        present = sorted(set(out_classes))
        rng = random.Random(f"{self._seed}:{int(image_id)}")  # noqa: S311 — deterministic seeded RNG for prompt sampling, not security
        prompts_list = _build_text_prompts(
            present_dense_ids=present,
            class_names=self._class_names,
            cfg=self._text_prompt_cfg,
            rng=rng,
            image_id=int(image_id),
        )
        if len(prompts_list) > self._multiplex_cap:
            if not self._warned_truncation:
                _LOG.warning(
                    "custom_sam_peft.data.coco: image_id=%s requested %d text prompts; "
                    "truncating to %d. Suppressing further warnings for this dataset.",
                    image_id,
                    len(prompts_list),
                    self._multiplex_cap,
                )
                self._warned_truncation = True
            prompts_list = prompts_list[: self._multiplex_cap]
        return Example(
            image=image_tensor,
            image_id=str(image_id),
            prompts=TextPrompts(classes=prompts_list),
            instances=instances,
        )
```

(c) `build_coco` (current line 412 — the `prompt_mode=cfg["prompt_mode"]` kwarg). Remove the line so the `COCODataset(...)` call becomes:

```python
    return COCODataset(
        annotations=split["annotations"],
        images=split["images"],
        transforms=transforms,
        text_prompt=text_prompt,
        image_ids=[int(s) for s in resolved] if resolved is not None else None,
        channels=int(cfg.get("channels", 3)),
    )
```

Also delete the now-unused `Literal["text", "bbox"]` import in `coco.py` if it's no longer referenced (check via `grep -n 'Literal\["text"' src/custom_sam_peft/data/coco.py` — it appears in `Literal["text", "bbox"]` only on the deleted param; the `Literal["train", "eval"]` on `build_coco` stays).

- [ ] **Step 2: Edit `src/custom_sam_peft/data/hf.py` — symmetric to coco.py**

(a) `HFDataset.__init__` (current lines 130–154). Drop the `prompt_mode` param (line 134), the validator (lines 143–144), and `self._prompt_mode` (line 148):

After:

```python
    def __init__(
        self,
        name: str,
        split: str,
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        field_map: HFFieldMap,
        seed: int = 0,
        row_indices: Iterable[int] | None = None,
        channels: int = 3,
    ) -> None:
        self._name = name
        self._split = split
        self._channels = channels
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._field_map = field_map
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False
        ...  # rest unchanged
```

(b) `_pack_example` (current lines 294–381). Drop `BoxPrompts` from the local import; remove the `if self._prompt_mode == "text":` gate; delete the bbox branch (lines 350–381):

```python
    def _pack_example(
        self,
        i: int,
        image_tensor: Any,
        out_bboxes: list[Any],
        out_masks: list[Any],
        out_classes: list[int],
    ) -> Example:
        """Assemble `Instance` objects and return the final `Example`."""
        import random as _random

        import numpy as _np
        import torch

        from custom_sam_peft.data.base import Instance, TextPrompts
        from custom_sam_peft.data.coco import _build_text_prompts

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(_np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

        image_id = str(i)
        present = sorted(set(out_classes))
        rng = _random.Random(f"{self._seed}:{i}")  # noqa: S311 — deterministic seeded RNG for prompt sampling, not security
        prompts_list = _build_text_prompts(
            present_dense_ids=present,
            class_names=self._class_names,
            cfg=self._text_prompt_cfg,
            rng=rng,
            image_id=i,
        )
        if len(prompts_list) > self._multiplex_cap:
            if not self._warned_truncation:
                _LOG.warning(
                    "custom_sam_peft.data.hf: image_id=%s requested %d text prompts; "
                    "truncating to %d. Suppressing further warnings.",
                    image_id,
                    len(prompts_list),
                    self._multiplex_cap,
                )
                self._warned_truncation = True
            prompts_list = prompts_list[: self._multiplex_cap]
        return Example(
            image=image_tensor,
            image_id=image_id,
            prompts=TextPrompts(classes=prompts_list),
            instances=instances,
        )
```

(c) `build_hf` (current line 445 — the `prompt_mode=cfg["prompt_mode"]` kwarg). Remove the line so the `HFDataset(...)` call becomes:

```python
    return HFDataset(
        name=hf_cfg["name"],
        split=split,
        transforms=transforms,
        text_prompt=text_prompt,
        field_map=field_map,
        row_indices=[int(s) for s in resolved] if resolved is not None else None,
        channels=int(cfg.get("channels", 3)),
    )
```

- [ ] **Step 3: Strip `prompt_mode=...` kwargs from `tests/conftest.py`**

In `tests/conftest.py`, change the `tiny_coco_dataset` fixture (lines 160–166):

Before:

```python
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )
```

After:

```python
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )
```

Also update the fixture docstring on line 153 from `"""A COCODataset pointing at the tiny_coco fixture (bbox prompt mode)."""` to `"""A COCODataset pointing at the tiny_coco fixture."""`.

- [ ] **Step 4: Strip `prompt_mode=...` kwargs from every dataset-constructing test**

For each file listed in the §"Files" section above (under "tests/unit" and "tests/integration"), open it and apply this **mechanical** rule:

- Any `COCODataset(...)`, `HFDataset(...)`, or `builder(cfg, ...)` call that includes `prompt_mode=...`: delete the `prompt_mode=...` line entirely. The kwarg appears on its own line in every observed occurrence; no other code needs to change.
- For dict-style configs (e.g. `cfg: dict[str, Any] = {"format": "coco", ..., "prompt_mode": "...", ...}`): delete the `"prompt_mode": "..."` key-value entry.
- For `ds._prompt_mode = "..."` direct attribute assignments (e.g. in `test_data_hf_limit.py:28`): delete the line.
- For `cfg.data.prompt_mode = "..."` (e.g. in `test_train_runner.py:20`, `test_load_sam31_callsites.py:22`): delete the line.

The full, planner-verified call-site list (drop the kwarg / entry / assignment at each):

| File | Lines (drop these) |
| --- | --- |
| `tests/unit/test_data_coco.py` | 203, 255, 268, 282, 326, 368, 417, 434, 450, 467, 512, 559, 611, 624, 677, 705, 731, 750, 774, 785, 801, 814, 821, 835, 844, 864 |
| `tests/unit/test_data_hf.py` | 150, 186, 225, 257, 274, 309, 330, 409, 430, 490, 507, 517, 534 |
| `tests/unit/test_data_hf_limit.py` | 28 |
| `tests/unit/test_data_apply_transforms_bbox_drop.py` | 148, 272, 310, 334 |
| `tests/unit/test_data_coco_limit.py` | 25, 68, 105, 131 |
| `tests/unit/test_trainer_nan_behavior.py` | 55, 80 |
| `tests/unit/test_trainer_no_val.py` | 45, 62 |
| `tests/unit/test_trainer_run_dir.py` | 38, 129, 197, 204, 221, 352, 359, 376, 452, 459, 476 |
| `tests/unit/test_train_runner.py` | 20, 128, 191 |
| `tests/unit/test_val_source.py` | 50, 225 |
| `tests/unit/test_train_step.py` | 37 |
| `tests/unit/test_train_loop_legacy_k1.py` | 44 |
| `tests/unit/test_train_loop_multiplex.py` | 42 |
| `tests/unit/test_train_checkpoint.py` | 40 |
| `tests/unit/test_tracking_noop.py` | 30 |
| `tests/unit/test_tracking_tensorboard.py` | 32 |
| `tests/unit/test_tracking_wandb.py` | 34 |
| `tests/unit/test_checkpoint_roundtrip.py` | 47 |
| `tests/integration/test_train_resume.py` | 43, 62 |
| `tests/integration/test_trainer_evaluator_seam.py` | 81 |
| `tests/integration/test_train_end_to_end.py` | 51, 77, 141, 195, 235, 276, 341, 388, 439 |
| `tests/integration/test_train_then_eval.py` | 57 |
| `tests/integration/test_peft_extensibility.py` | 58 |
| `tests/integration/test_tracker_swap.py` | 103 |

Method recommendation: run

```bash
git grep -nE 'prompt_mode\s*[:=]\s*"(text|bbox)"' tests/
```

to list every match before editing; after edits, the same command must return 0 matches under `tests/unit/test_data_*` + the loop / trainer / tracker / integration tests. (`tests/unit/test_config_schema.py`, `test_config_loader.py`, `test_cli.py`, `test_cli_init.py`, `test_cli_doctor*.py`, `test_cli_export.py`, `test_eval_runner*.py`, `test_data_schema_extensions.py`, `test_load_sam31_callsites.py`, `test_data_base.py`, `test_data_collate.py`, `test_stubs_raise.py`, `test_trainer_guards.py`, and the wizard test still match — those are handled in Tasks 4–6.)

- [ ] **Step 5: Delete the bbox-only tests in `tests/unit/test_data_coco.py`**

The following functions reference a code path that no longer exists. Delete each one in its entirety (function header through last line of the function body):

- `test_multiplex_truncation_box` (currently lines 427–442)
- `test_getitem_bbox_mode_returns_BoxPrompts` (currently lines 445–459)

The other tests at lines 203, 255, 268, 282, 326, 368, 417, 467, 512, 559, 611, 624, 677, 705, 731 KEEP their bodies — only their `prompt_mode=...` kwarg / `"prompt_mode": "..."` entry is dropped (Step 4 above). The tests' assertions about class names, masks, segmentation decoding, etc. are orthogonal to `prompt_mode` and remain valid for the always-text path. After Step 4 + Step 5, every remaining test in `test_data_coco.py` constructs the dataset without the kwarg.

- [ ] **Step 6: Delete the bbox-only tests in `tests/unit/test_data_hf.py`**

Delete in entirety:

- `test_getitem_bbox_mode` (currently lines 267–282)
- `test_bbox_format_xywh_conversion` (currently lines 285–340 — entire function)

The remaining tests keep their bodies; only `prompt_mode=...` is dropped (Step 4 above).

- [ ] **Step 7: Run the data-layer + loop / trainer tests to verify they pass**

```bash
uv run pytest tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_data_hf_limit.py tests/unit/test_data_apply_transforms_bbox_drop.py tests/unit/test_data_coco_limit.py tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_no_val.py tests/unit/test_trainer_run_dir.py tests/unit/test_train_runner.py tests/unit/test_val_source.py tests/unit/test_train_step.py tests/unit/test_train_loop_legacy_k1.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_checkpoint.py tests/unit/test_tracking_noop.py tests/unit/test_tracking_tensorboard.py tests/unit/test_tracking_wandb.py tests/unit/test_checkpoint_roundtrip.py -v
```

Expected: PASS. Any failure here indicates either a missed `prompt_mode=...` kwarg deletion, or a test body that still imports `BoxPrompts` (left for Task 4 — those imports stay for now; the test bodies that *referenced* `BoxPrompts` were deleted in Steps 5 / 6).

If `test_data_coco.py::test_register_coco_lookup` (or similar `lookup("dataset", "coco")` builder test) fails, double-check that the `cfg` dict in that test no longer carries `"prompt_mode": "..."` — Step 4's mechanical scrub must catch those.

- [ ] **Step 8: Lint + mypy + commit**

```bash
uv run ruff check src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py tests/
uv run ruff format --check src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py tests/
uv run mypy src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py
git add src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py tests/conftest.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_data_hf_limit.py tests/unit/test_data_apply_transforms_bbox_drop.py tests/unit/test_data_coco_limit.py tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_no_val.py tests/unit/test_trainer_run_dir.py tests/unit/test_train_runner.py tests/unit/test_val_source.py tests/unit/test_train_step.py tests/unit/test_train_loop_legacy_k1.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_checkpoint.py tests/unit/test_tracking_noop.py tests/unit/test_tracking_tensorboard.py tests/unit/test_tracking_wandb.py tests/unit/test_checkpoint_roundtrip.py tests/integration/test_train_resume.py tests/integration/test_trainer_evaluator_seam.py tests/integration/test_train_end_to_end.py tests/integration/test_train_then_eval.py tests/integration/test_peft_extensibility.py tests/integration/test_tracker_swap.py
git commit -m "feat(data)!: collapse COCO/HF adapters to always-TextPrompts (#126)"
```

---

## Task 4: Delete `BoxPrompts`; collapse `Prompts = TextPrompts | BoxPrompts` → `Prompts = TextPrompts`

**Files:**

- Modify: `src/custom_sam_peft/data/base.py` (delete `BoxPrompts` class lines 18–27; collapse `Prompts` alias line 30)
- Modify: `tests/unit/test_data_base.py` (drop `BoxPrompts` from imports; delete `test_text_prompts_and_box_prompts_are_distinct_types`)
- Modify: `tests/unit/test_data_collate.py` (drop `BoxPrompts` import + construction + `isinstance` assertion)
- Modify: `tests/unit/test_data_coco.py` (drop `BoxPrompts` from the top-of-file import — every BoxPrompts-using test was deleted in Task 3)
- Modify: `tests/unit/test_data_hf.py` (drop `BoxPrompts` from the top-of-file import)
- Modify: `tests/fixtures/tiny_sam3_stub.py` (drop the `BoxPrompts` mention in the class docstring)

**Objective:** Now that no code path constructs or references `BoxPrompts`, delete the class and shrink the `Prompts` alias. Keep `Prompts` as an alias (= `TextPrompts`) so external call sites referring to `Prompts` still resolve.

- [ ] **Step 1: Delete `BoxPrompts` from `src/custom_sam_peft/data/base.py`**

Delete lines 18–27 (the entire `@dataclass(frozen=True) class BoxPrompts: ...` block) and change line 30 from:

```python
Prompts = TextPrompts | BoxPrompts
```

to:

```python
# After #126, `Prompts` is an alias for `TextPrompts`. The alias is preserved
# so call sites referring to `Prompts` continue to resolve.
Prompts = TextPrompts
```

Resulting region of the file (lines ~11–28):

```python
@dataclass(frozen=True)
class TextPrompts:
    """Open-vocabulary class names used as prompts for one image."""

    classes: list[str]


@dataclass(frozen=True)
class SupportPrompts:
    """..."""  # (body unchanged from Task 1)

    boxes: list[torch.Tensor | None] | None = None


# After #126, `Prompts` is an alias for `TextPrompts`. The alias is preserved
# so call sites referring to `Prompts` continue to resolve.
Prompts = TextPrompts
```

- [ ] **Step 2: Scrub `BoxPrompts` from `tests/unit/test_data_base.py`**

Change the top-of-file import block from:

```python
from custom_sam_peft.data.base import (
    BoxPrompts,
    Dataset,
    Example,
    Instance,
    SupportPrompts,
    TextPrompts,
    is_dataset,
)
```

to:

```python
from custom_sam_peft.data.base import (
    Dataset,
    Example,
    Instance,
    SupportPrompts,
    TextPrompts,
    is_dataset,
)
```

Delete the entire `test_text_prompts_and_box_prompts_are_distinct_types` function (currently lines 17–24):

```python
def test_text_prompts_and_box_prompts_are_distinct_types() -> None:
    t = TextPrompts(classes=["cat", "dog"])
    b = BoxPrompts(
        boxes=torch.zeros((2, 4)),
        class_ids=torch.tensor([0, 1]),
    )
    assert isinstance(t, TextPrompts)
    assert isinstance(b, BoxPrompts)
```

Leave the rest of the file (Example test, Dataset protocol tests, and the new `test_support_prompts_dataclass` from Task 1) unchanged.

- [ ] **Step 3: Scrub `BoxPrompts` from `tests/unit/test_data_collate.py`**

Change line 8 from:

```python
from custom_sam_peft.data.base import BoxPrompts, Example, Instance, TextPrompts
```

to:

```python
from custom_sam_peft.data.base import Example, Instance, TextPrompts
```

Rewrite `test_collate_keeps_prompts_as_list` (currently lines 33–49) so the middle example uses `TextPrompts` instead of `BoxPrompts`:

```python
def test_collate_keeps_prompts_as_list() -> None:
    a = _ex("a")
    b = Example(
        image=torch.zeros((3, 64, 64)),
        image_id="b",
        prompts=TextPrompts(classes=["b-class"]),
        instances=[],
    )
    c = _ex("c")
    batch = collate_batch([a, b, c])
    assert isinstance(batch["prompts"], list)
    assert len(batch["prompts"]) == 3
    assert isinstance(batch["prompts"][0], TextPrompts)
    assert isinstance(batch["prompts"][1], TextPrompts)
    assert isinstance(batch["prompts"][2], TextPrompts)
```

(The collator never branched on prompt variant; the test's intent was to confirm the collator passes prompts through as a list, which holds for any prompt type. We collapse the assertion to `TextPrompts` everywhere.)

- [ ] **Step 4: Scrub `BoxPrompts` from `tests/unit/test_data_coco.py` and `tests/unit/test_data_hf.py`**

In `tests/unit/test_data_coco.py`, change line 178 from:

```python
from custom_sam_peft.data.base import BoxPrompts, TextPrompts
```

to:

```python
from custom_sam_peft.data.base import TextPrompts
```

In `tests/unit/test_data_hf.py`, change line 120 from:

```python
from custom_sam_peft.data.base import BoxPrompts, TextPrompts
```

to:

```python
from custom_sam_peft.data.base import TextPrompts
```

(Every `BoxPrompts(...)` construction and every `isinstance(ex.prompts, BoxPrompts)` assertion in those files was deleted in Task 3 alongside the bbox-only tests. Verify by running `git grep -n 'BoxPrompts' tests/unit/test_data_coco.py tests/unit/test_data_hf.py` → 0 matches.)

- [ ] **Step 5: Scrub `BoxPrompts` from `tests/fixtures/tiny_sam3_stub.py` docstring**

In `tests/fixtures/tiny_sam3_stub.py`, change the class docstring (currently lines 19–28) from:

```python
    """Returns Meta-shaped output dict given image + prompts.

    Q = number of decoder queries (default 4 for fast tests).

    In multiplex mode (K > 1 classes per TextPrompts), the real SAM 3.1 model
    returns (B*K, Q, ...) shaped outputs (one row per image-class pair).  This
    stub replicates that contract: when prompts is a list of TextPrompts with
    K classes each, the output batch dimension is B*K.  For K=1 (legacy) or
    BoxPrompts / ignored-prompt modes the output batch dimension is B.
    """
```

to:

```python
    """Returns Meta-shaped output dict given image + prompts.

    Q = number of decoder queries (default 4 for fast tests).

    In multiplex mode (K > 1 classes per TextPrompts), the real SAM 3.1 model
    returns (B*K, Q, ...) shaped outputs (one row per image-class pair).  This
    stub replicates that contract: when prompts is a list of TextPrompts with
    K classes each, the output batch dimension is B*K.  For K=1 (legacy) the
    output batch dimension is B.
    """
```

- [ ] **Step 6: Run the affected tests to verify they pass**

```bash
uv run pytest tests/unit/test_data_base.py tests/unit/test_data_collate.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_wrapper_support.py -v
```

Expected: PASS — `BoxPrompts` is gone from imports and constructions; everything else still resolves through the `Prompts = TextPrompts` alias.

- [ ] **Step 7: Confirm no `BoxPrompts` lingers anywhere in src/ or tests/**

```bash
! git grep -n 'BoxPrompts' src/ tests/
```

Expected: exit 0 (the `!` inverts an empty grep). If any matches remain, the prior steps missed a reference — fix before committing.

- [ ] **Step 8: Lint + mypy + commit**

```bash
uv run ruff check src/custom_sam_peft/data/base.py tests/
uv run ruff format --check src/custom_sam_peft/data/base.py tests/
uv run mypy src/custom_sam_peft/data/base.py
git add src/custom_sam_peft/data/base.py tests/unit/test_data_base.py tests/unit/test_data_collate.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/fixtures/tiny_sam3_stub.py
git commit -m "feat(data)!: delete BoxPrompts; collapse Prompts alias to TextPrompts (#126)"
```

---

## Task 5: Delete `PromptMode` + `DataConfig.prompt_mode`; strip `prompt_mode:` from configs / templates / docstring; remaining test scrubs

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (drop `"PromptMode"` from `__all__` line 77; delete `PromptMode = Literal[...]` line 92; delete `DataConfig.prompt_mode` field line 386; rewrite the `prompt_mode='text'` mention in `TextPromptConfig` docstring line 219)
- Modify: `configs/examples/coco_text_lora.yaml`, `coco_text_qlora.yaml`, `coco_text_lora_subset.yaml`, `coco_text_no_val.yaml`, `coco_text_auto_split.yaml`, `min_gpu_qlora.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml` (drop the `prompt_mode: text` line in each)
- Modify: `src/custom_sam_peft/cli/templates/config_full.yaml` (drop the `prompt_mode: text` line at 23)
- Modify: tests with embedded YAML strings or schema-dict literals (drop `prompt_mode:` / `"prompt_mode"`):
  `tests/unit/test_config_schema.py`, `tests/unit/test_config_loader.py`,
  `tests/unit/test_data_schema_extensions.py`, `tests/unit/test_config_examples.py`,
  `tests/unit/test_cli.py` (YAML strings — Task 6 deletes the bbox-reject test),
  `tests/unit/test_cli_init.py`, `tests/unit/test_cli_doctor.py`,
  `tests/unit/test_cli_doctor_config.py`, `tests/unit/test_cli_export.py`,
  `tests/unit/test_eval_runner_gate.py`, `tests/unit/test_eval_runner.py`,
  `tests/unit/test_load_sam31_callsites.py`, `tests/unit/cli/test_setup_wizard.py`,
  `tests/integration/test_cli_run.py` (YAML format placeholder)

**Objective:** Delete the schema field; rely on `_Strict`'s `extra="forbid"` to gate `prompt_mode:` everywhere. Strip the now-illegal line from every shipped YAML and embedded YAML / dict in tests. Add a schema-rejection test.

- [ ] **Step 1: Write the failing schema-rejection test**

In `tests/unit/test_config_schema.py`, add this test (place it after `test_invalid_prompt_mode_rejected`, which will be deleted in Step 2):

```python
def test_prompt_mode_rejected_by_schema() -> None:
    """Spec #126 §6: any prompt_mode key (regardless of value) fails at load.

    The schema is the sole gate; `_Strict`'s extra="forbid" rejects the key
    with a Pydantic ValidationError of type "extra_forbidden".
    """
    d = _minimal_dict()
    # _minimal_dict() now constructs a payload without prompt_mode (Step 2).
    # Add it back as an extra key and assert it is rejected.
    for value in ("text", "bbox", "something_else"):
        d["data"]["prompt_mode"] = value  # type: ignore[index]
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig.model_validate(d)
        errors = exc_info.value.errors()
        assert any(
            e["type"] == "extra_forbidden" and e["loc"][-1] == "prompt_mode"
            for e in errors
        ), f"expected extra_forbidden on data.prompt_mode for value={value!r}; got {errors}"
```

- [ ] **Step 2: Update `tests/unit/test_config_schema.py` fixtures**

(a) Change `_minimal_dict()` (lines 11–27) — drop the `"prompt_mode": "bbox"` entry at line 19. Resulting `data` block:

```python
        "data": {
            "format": "coco",
            "train": {"annotations": "data/train.json", "images": "data/train/"},
            "val": {"annotations": "data/val.json", "images": "data/val/"},
            "image_size": 1024,
        },
```

(b) Delete the now-obsolete `test_invalid_prompt_mode_rejected` (lines 48–52):

```python
def test_invalid_prompt_mode_rejected() -> None:
    d = _minimal_dict()
    d["data"]["prompt_mode"] = "points"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)
```

(That test asserted the bbox/text Literal rejected `"points"`. After the field is removed, the new `test_prompt_mode_rejected_by_schema` is the canonical proof.)

(c) Drop the `"prompt_mode": "text"` entry from `minimal_data_config_dict` (the fixture starting line ~195, line 201):

```python
@pytest.fixture
def minimal_data_config_dict() -> dict:
    return {
        "format": "coco",
        "train": {"annotations": "t.json", "images": "t/"},
        "val": {"annotations": "v.json", "images": "v/"},
    }
```

- [ ] **Step 3: Run the new test to verify it fails**

```bash
uv run pytest tests/unit/test_config_schema.py::test_prompt_mode_rejected_by_schema -v
```

Expected: FAIL — `DataConfig.prompt_mode` is still a required field, so `_minimal_dict()` with `prompt_mode` removed errors with `field_required` (or a `data.prompt_mode` `extra_forbidden` once the field is removed, but the field is still there at this point).

Note: after Step 2's removal of `"prompt_mode": "bbox"` from `_minimal_dict()`, `test_full_config_validates` and similar tests fail with `field_required` on `data.prompt_mode`. That is expected and resolves in Step 4.

- [ ] **Step 4: Delete `PromptMode` and `DataConfig.prompt_mode` from `src/custom_sam_peft/config/schema.py`**

(a) Remove `"PromptMode"` from the `__all__` list (line 77). The line currently reads:

```python
    "PromptMode",
```

Delete that entire line.

(b) Remove the `PromptMode` Literal definition (line 92). The line currently reads:

```python
PromptMode = Literal["text", "bbox"]
```

Delete that entire line.

(c) Remove the `DataConfig.prompt_mode` field (line 386). The line currently reads:

```python
    prompt_mode: PromptMode
```

Delete that entire line. The surrounding `DataConfig` class fields (`format`, `train`, `val`, `val_split` above; `image_size`, `channels` below) keep their positions.

(d) Update the `TextPromptConfig` class docstring (line 219). Change:

```python
class TextPromptConfig(_Strict):
    """How TextPrompts.classes is populated for each image when prompt_mode='text'.

    - present:                Use exactly the categories present in the image's
```

to:

```python
class TextPromptConfig(_Strict):
    """How TextPrompts.classes is populated for each image.

    - present:                Use exactly the categories present in the image's
```

(After #126 the wording "when prompt_mode='text'" is misleading — text is the only path.)

- [ ] **Step 5: Run the schema tests to verify they pass**

```bash
uv run pytest tests/unit/test_config_schema.py -v
```

Expected: PASS, including the new `test_prompt_mode_rejected_by_schema`. Pydantic now rejects any `prompt_mode:` key with `extra_forbidden`.

- [ ] **Step 6: Strip `prompt_mode: text` from the 8 example configs**

For each of the following files, delete the single line `prompt_mode: text` (the line containing exactly that text after stripping leading whitespace; YAML indentation may vary):

- `configs/examples/coco_text_lora.yaml` (line 28)
- `configs/examples/coco_text_qlora.yaml` (line 28)
- `configs/examples/coco_text_lora_subset.yaml` (line 22)
- `configs/examples/coco_text_no_val.yaml` (line 26)
- `configs/examples/coco_text_auto_split.yaml` (line 27)
- `configs/examples/min_gpu_qlora.yaml` (line 27)
- `configs/examples/gpu_smoke_lora.yaml` (line 20)
- `configs/examples/gpu_smoke_qlora.yaml` (line 20)

After each edit, verify the file still loads — the `data:` block keys `format`, `train`, `val` (or `val_split`), `image_size`, etc. flow without other changes.

Run yamllint to confirm no regression:

```bash
uv run --with yamllint yamllint -c .config/yamllint.yml configs/examples/
```

- [ ] **Step 7: Strip `prompt_mode: text` from the unified template**

In `src/custom_sam_peft/cli/templates/config_full.yaml`, delete the line `  prompt_mode: text` (currently line 23). The surrounding `data:` block keys keep their layout.

- [ ] **Step 8: Strip `prompt_mode:` lines from tests with embedded YAML / dicts**

For each file, the planner-verified location list:

| File | Action |
| --- | --- |
| `tests/unit/test_config_loader.py` | Line 23: strip `prompt_mode: bbox` from the embedded YAML. Line 150: strip `prompt_mode: text` from the embedded YAML. |
| `tests/unit/test_data_schema_extensions.py` | Line 104: drop `"prompt_mode": "bbox"` entry. Line 156: drop `prompt_mode="text"` (Task 3 may have caught this; verify). |
| `tests/unit/test_config_examples.py` | Line 46: drop `prompt_mode="..."` from the test's embedded dict. |
| `tests/unit/test_cli.py` | Lines 130, 175, 209: strip `prompt_mode: text` from the embedded YAML strings. (Line 95–115's `test_train_rejects_bbox_prompt_mode` is deleted in Task 6.) |
| `tests/unit/test_cli_init.py` | Line 62: drop the `assert "prompt_mode: text" in body` assertion. |
| `tests/unit/test_cli_doctor.py` | Line 131: drop `"prompt_mode": "text"` dict entry. |
| `tests/unit/test_cli_doctor_config.py` | Line 46: strip `prompt_mode: bbox` from embedded YAML. If the surrounding test asserted the bbox guard, also delete the test — read the test body to decide; the test name will signal intent. |
| `tests/unit/test_cli_export.py` | Line 26: strip `prompt_mode: text` from embedded YAML. |
| `tests/unit/test_eval_runner_gate.py` | Line 22: drop `"prompt_mode": "text"` dict entry. |
| `tests/unit/test_eval_runner.py` | Line 23: drop `"prompt_mode": "text"` dict entry. |
| `tests/unit/test_load_sam31_callsites.py` | Line 22: drop `cfg.data.prompt_mode = "text"` assignment. |
| `tests/unit/cli/test_setup_wizard.py` | Line 176: drop the `assert "prompt_mode: text" in rendered` assertion. |
| `tests/integration/test_cli_run.py` | Line 34: strip `prompt_mode: {prompt}` placeholder from the YAML format string (and the corresponding `prompt=...` arg downstream — read the file to find the call site). |

Method: run

```bash
git grep -nE 'prompt_mode' tests/
```

After all edits, the only remaining matches under `tests/` should be in `tests/unit/test_cli.py:95-115` (the bbox-rejection test deleted in Task 6), `tests/unit/test_trainer_guards.py:20/28/50` (also Task 6), `tests/unit/test_stubs_raise.py:22/40-45` (Task 6), and `tests/integration/test_cli_run.py:257-...` (`test_run_rejects_bbox_prompt_mode`, deleted in Task 6).

- [ ] **Step 9: Run a broad test pass to verify everything still loads**

```bash
uv run pytest tests/unit -q --no-cov
```

Expected: PASS, except for the bbox-guard-related tests (Task 6) which still reference deleted code paths. Specifically the following may still fail (Task 6 fixes them):

- `tests/unit/test_trainer_guards.py::test_trainer_rejects_bbox_prompt_mode`
- `tests/unit/test_cli.py::test_train_rejects_bbox_prompt_mode`
- `tests/unit/test_stubs_raise.py` bbox-rejection test
- `tests/integration/test_cli_run.py::test_run_rejects_bbox_prompt_mode`

If any other test fails, debug — that signals a missed `prompt_mode:` reference.

- [ ] **Step 10: Lint + mypy + markdownlint configs + commit**

```bash
uv run ruff check src/custom_sam_peft/config/schema.py tests/
uv run ruff format --check src/custom_sam_peft/config/schema.py tests/
uv run mypy src/custom_sam_peft/config/schema.py
uv run --with yamllint yamllint -c .config/yamllint.yml configs/examples/ src/custom_sam_peft/cli/templates/
git add src/custom_sam_peft/config/schema.py configs/examples/ src/custom_sam_peft/cli/templates/config_full.yaml tests/unit/test_config_schema.py tests/unit/test_config_loader.py tests/unit/test_data_schema_extensions.py tests/unit/test_config_examples.py tests/unit/test_cli.py tests/unit/test_cli_init.py tests/unit/test_cli_doctor.py tests/unit/test_cli_doctor_config.py tests/unit/test_cli_export.py tests/unit/test_eval_runner_gate.py tests/unit/test_eval_runner.py tests/unit/test_load_sam31_callsites.py tests/unit/cli/test_setup_wizard.py tests/integration/test_cli_run.py
git commit -m "feat(schema)!: remove PromptMode + DataConfig.prompt_mode; strip configs (#126)"
```

---

## Task 6: Delete the three hand-rolled `prompt_mode == "bbox"` guards + their tests

**Files:**

- Modify: `src/custom_sam_peft/train/trainer.py` (delete the bbox guard at lines 135–140)
- Modify: `src/custom_sam_peft/cli/train_cmd.py` (delete the bbox guard at lines 48–52)
- Modify: `src/custom_sam_peft/cli/run_cmd.py` (delete the bbox guard at lines 191–195)
- Modify: `tests/unit/test_trainer_guards.py` (drop `prompt_mode` from `_cfg`'s sig/body; delete `test_trainer_rejects_bbox_prompt_mode`)
- Modify: `tests/unit/test_cli.py` (delete `test_train_rejects_bbox_prompt_mode`)
- Modify: `tests/unit/test_stubs_raise.py` (delete the bbox-rejection test)
- Modify: `tests/integration/test_cli_run.py` (delete `test_run_rejects_bbox_prompt_mode`)

**Objective:** Delete the three runtime guards (the schema's `extra_forbidden` is the sole gate) and the four tests that exercised them.

- [ ] **Step 1: Delete the trainer guard**

In `src/custom_sam_peft/train/trainer.py`, delete lines 135–140:

```python
        if cfg.data.prompt_mode == "bbox":
            raise ValueError(
                "prompt_mode='bbox' is not supported for training in v0; v0 trains "
                "text-only with optional GT-box hints sampled per-image. See "
                "logs/TODO.md for the deferred spec."
            )
```

The next statement (`self.model = model`) immediately follows the function header's `runtime` kwarg and `...) -> None:` line.

- [ ] **Step 2: Delete the CLI `train` guard**

In `src/custom_sam_peft/cli/train_cmd.py`, delete lines 48–52:

```python
    if cfg.data.prompt_mode == "bbox":
        raise typer.BadParameter(
            "prompt_mode='bbox' is not supported for training in v0.",
            param_hint="--config",
        )
```

The next statement (`mode = resolve_mode(...)`) immediately follows the `cfg = load_config(...)` line.

- [ ] **Step 3: Delete the CLI `run` guard**

In `src/custom_sam_peft/cli/run_cmd.py`, delete lines 191–195:

```python
    if cfg.data.prompt_mode == "bbox":
        raise typer.BadParameter(
            "prompt_mode='bbox' is not supported for training in v0.",
            param_hint="--config",
        )
```

If `typer` is no longer used after this deletion, run `git grep -n 'typer' src/custom_sam_peft/cli/run_cmd.py` to confirm — likely it is still imported for other commands. Do NOT touch other imports.

- [ ] **Step 4: Scrub `tests/unit/test_trainer_guards.py`**

(a) Edit the `_cfg` helper (currently lines 19–32). Remove `prompt_mode: str = "text"` from the signature and the `prompt_mode=prompt_mode` from the body.

Before:

```python
def _cfg(
    prompt_mode: str = "text", peft_method: str = "lora", optimizer: str = "auto"
) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "run": {"name": "x"},
            "model": {"name": "facebook/sam3.1"},
            "data": {
                "format": "coco",
                "train": {"annotations": "t.json", "images": "t/"},
                "val": {"annotations": "v.json", "images": "v/"},
                "prompt_mode": prompt_mode,
            },
            "peft": {"method": peft_method},
            "train": {"epochs": 1, "optimizer": optimizer},
        }
    )
```

After:

```python
def _cfg(peft_method: str = "lora", optimizer: str = "auto") -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "run": {"name": "x"},
            "model": {"name": "facebook/sam3.1"},
            "data": {
                "format": "coco",
                "train": {"annotations": "t.json", "images": "t/"},
                "val": {"annotations": "v.json", "images": "v/"},
            },
            "peft": {"method": peft_method},
            "train": {"epochs": 1, "optimizer": optimizer},
        }
    )
```

(b) Delete `test_trainer_rejects_bbox_prompt_mode` (currently lines 50–55):

```python
def test_trainer_rejects_bbox_prompt_mode(
    stub_model: Sam3Wrapper, noop_tracker: NoopTracker
) -> None:
    cfg = _cfg(prompt_mode="bbox")
    with pytest.raises(ValueError, match="prompt_mode='bbox'"):
        Trainer(stub_model, MagicMock(), None, noop_tracker, cfg)
```

Keep the other tests in the file (qlora-optimizer-coercion etc.).

- [ ] **Step 5: Delete `test_train_rejects_bbox_prompt_mode` in `tests/unit/test_cli.py`**

Delete the entire function `test_train_rejects_bbox_prompt_mode` (currently lines 95–115). The function body includes a YAML string with `prompt_mode: bbox` — both go away.

- [ ] **Step 6: Delete the bbox-rejection test in `tests/unit/test_stubs_raise.py`**

Open `tests/unit/test_stubs_raise.py`. Around line 22 there is a comment about the bbox guard; lines 40–45 contain the test. Delete the test function in full, and remove the orienting comment at line 22 (it now describes deleted code).

- [ ] **Step 7: Delete `test_run_rejects_bbox_prompt_mode` in `tests/integration/test_cli_run.py`**

Open `tests/integration/test_cli_run.py`, find `def test_run_rejects_bbox_prompt_mode` (starts at line 257), and delete the entire function (header through the last line of the body).

If the file imports anything used only by that test (e.g. a specific `pytest.raises` match string), the linter will flag the unused import in Step 8 — clean those up.

- [ ] **Step 8: Run the affected tests + a broad smoke pass**

```bash
uv run pytest tests/unit/test_trainer_guards.py tests/unit/test_cli.py tests/unit/test_stubs_raise.py tests/integration/test_cli_run.py -v
uv run pytest tests/unit -q --no-cov
```

Expected: PASS in both. Any failure here points to a missed `prompt_mode:` reference; check the failure message for the file + line.

- [ ] **Step 9: Lint + mypy + commit**

```bash
uv run ruff check src/custom_sam_peft/train/trainer.py src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py tests/
uv run ruff format --check src/custom_sam_peft/train/trainer.py src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py tests/
uv run mypy src/custom_sam_peft/train/trainer.py src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py
git add src/custom_sam_peft/train/trainer.py src/custom_sam_peft/cli/train_cmd.py src/custom_sam_peft/cli/run_cmd.py tests/unit/test_trainer_guards.py tests/unit/test_cli.py tests/unit/test_stubs_raise.py tests/integration/test_cli_run.py
git commit -m "refactor!: delete hand-rolled prompt_mode='bbox' guards (#126)"
```

---

## Task 7: Docs, CHANGELOG, anchoring comment, final green-gate

**Files:**

- Modify: `docs/ARCHITECTURE.md` (line 15 + add text-primary invariant near top)
- Modify: `docs/config-schema.md` (delete line 52 — the `data.prompt_mode` row)
- Modify: `CHANGELOG.md` (add a `## [Unreleased]` entry per spec §10.3)
- Modify: `src/custom_sam_peft/models/sam3.py` (add the anchoring comment at line 616)

**Objective:** Final close-out — docs, changelog, anchoring comment, and the spec §11 acceptance grep + full green-gate.

- [ ] **Step 1: Update `docs/ARCHITECTURE.md`**

(a) Add a short sentence near the top stating the text-primary invariant. After the module-overview opening paragraph (top of the file, before the `## Module map` heading), insert this paragraph:

```markdown
**Prompt invariant:** Text is the only primary prompt — the model takes one or more text (class) prompts and segments all matching instances. Auxiliary localization hints (currently just GT box hints, the `box_hint` curriculum from #14) ride alongside via `SupportPrompts`. They never replace text and are never used at inference. See [#126](https://github.com/NguyenJus/custom-sam-peft/issues/126).
```

(b) Replace line 15 (the `base.py` description in the module map). Before:

```text
    base.py            Example, Prompts (TextPrompts | BoxPrompts), Dataset protocol
```

After:

```text
    base.py            Example, Prompts (= TextPrompts), SupportPrompts, Dataset protocol
```

- [ ] **Step 2: Update `docs/config-schema.md`**

Delete the `data.prompt_mode` row at line 52. The line currently reads:

```markdown
| `data.prompt_mode` | `"text"` \| `"bbox"` | (required) | common | Whether to ...
```

Delete the entire line. The surrounding table rows (lines 47–58) reflow without other changes; verify the markdown table separators are still well-formed (`|` columns line up; markdownlint will flag MD056/MD058 if not).

- [ ] **Step 3: Add the `CHANGELOG.md` entry (exact spec §10.3 phrasing)**

In `CHANGELOG.md`, insert a new `## [Unreleased]` heading immediately after the top intro (after line 8's `---` separator, before the current top-most version heading `## [0.12.0] — 2026-05-23`):

```markdown
## [Unreleased]

### Breaking — text-primary prompt invariant (#126)

- **schema**: removed the `data.prompt_mode` field. Any config that carries
  `prompt_mode:` (any value) now fails at load with a Pydantic
  `extra_forbidden` error. Migration: delete the line from your YAML.
- **api**: replaced `Sam3Wrapper.forward(..., box_hints=...)` with
  `Sam3Wrapper.forward(..., support=SupportPrompts(boxes=...))`. Downstream
  callers that pass per-image GT boxes as a training hint must wrap them in
  a `SupportPrompts(boxes=...)` and pass via `support=`. Passing
  `support=None` (the default) is equivalent to today's `box_hints=None`.
- **types**: removed `BoxPrompts` and `PromptMode`. `Prompts` is now an alias
  for `TextPrompts`.
- **trainer/CLI**: removed three hand-rolled `prompt_mode == "bbox"` guards
  (`train/trainer.py`, `cli/train_cmd.py`, `cli/run_cmd.py`) — the schema is
  the sole gate.

The `box_hint` training curriculum (`train.box_hint.*`, `BoxHintSchedule`) is
unchanged — it continues to sample per-image GT boxes alongside text prompts
as an auxiliary localization hint, now flowing through `SupportPrompts`.

---
```

(Match the existing CHANGELOG style — three-dash horizontal rules between versions.)

- [ ] **Step 4: Add the anchoring comment at `models/sam3.py:616`**

In `src/custom_sam_peft/models/sam3.py`, immediately above line 616 (`enable_inst_interactivity=False,`), insert the two-line comment from the spec §4. The surrounding context is `_construct_raw_model` calling `sam3.build_sam3_image_model(...)`. Before:

```python
        raw_model = sam3.build_sam3_image_model(
            device=device,
            eval_mode=False,  # training mode — gradients flow.
            checkpoint_path=str(ckpt_path),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
```

After:

```python
        raw_model = sam3.build_sam3_image_model(
            device=device,
            eval_mode=False,  # training mode — gradients flow.
            checkpoint_path=str(ckpt_path),
            load_from_HF=False,
            enable_segmentation=True,
            # Disabled by design: this is SAM3's vendor point/box-primary
            # interactive pipe. Our prompt invariant is text-primary (see
            # #126); no code path routes to it.
            enable_inst_interactivity=False,
            compile=False,
        )
```

- [ ] **Step 5: Final acceptance grep (spec §11)**

```bash
! grep -rn 'prompt_mode\|BoxPrompts' configs/ src/ docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md
```

Expected: exit 0 (the `!` inverts an empty grep). If any matches remain, fix them before proceeding — the spec's §11 acceptance criterion requires zero matches in these paths. Dated specs under `docs/superpowers/specs/` may still mention `prompt_mode`; that is intentional (point-in-time records).

- [ ] **Step 6: Markdownlint the touched docs + CHANGELOG**

```bash
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md docs/superpowers/specs/2026-05-28-text-only-prompts-design.md docs/superpowers/plans/2026-05-28-text-only-prompts-plan.md
```

Expected: no findings. Fix any before committing.

- [ ] **Step 7: Run the full test + type + lint suite**

```bash
uv run pytest -q  # full suite (including integration) — 80% coverage gate must hold
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run --with yamllint yamllint -c .config/yamllint.yml configs/examples/ src/custom_sam_peft/cli/templates/
```

Expected: all green. The full `pytest` is the coverage-gate (`tests/unit` alone does not hit 80%).

- [ ] **Step 8: Commit**

```bash
git add docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md src/custom_sam_peft/models/sam3.py
git commit -m "docs(#126): document text-primary invariant; anchor enable_inst_interactivity"
```

---

## REVIEW CHECKPOINT — implementation complete

Before opening the PR, verify the acceptance criteria from spec §11:

- [ ] `BoxPrompts`, `PromptMode`, `DataConfig.prompt_mode`, and `box_hints=` on `Sam3Wrapper.forward` no longer exist anywhere under `src/custom_sam_peft/`.

  ```bash
  ! git grep -n 'BoxPrompts\|PromptMode\|DataConfig\.prompt_mode\|box_hints=' src/custom_sam_peft/
  ```

  (Note: `box_hints=` in `_Sam3ImageAdapter.forward` is the vendor-facing kwarg into Meta's `sam3` package and is unchanged; the grep above will match those usages, which is correct. Verify by reading the matches — they should all be in `models/sam3.py` inside `_Sam3ImageAdapter` or `_build_geometric_prompt`, never on `Sam3Wrapper`. The wrapper-level `box_hints=` should be 0 matches.)

- [ ] `Prompts` resolves to `TextPrompts`.

  ```bash
  python -c "from custom_sam_peft.data.base import Prompts, TextPrompts; assert Prompts is TextPrompts"
  ```

- [ ] `SupportPrompts` exists with a single optional field `boxes: list[Tensor | None] | None = None`, frozen, dataclass.

  ```bash
  python -c "from dataclasses import is_dataclass, fields; from custom_sam_peft.data.base import SupportPrompts; s = SupportPrompts(); assert is_dataclass(s); assert [f.name for f in fields(s)] == ['boxes']; assert s.boxes is None"
  ```

- [ ] `Sam3Wrapper.forward(..., support=SupportPrompts(boxes=...))` is the auxiliary-prompt API.

  ```bash
  python -c "import inspect; from custom_sam_peft.models.sam3 import Sam3Wrapper; sig = inspect.signature(Sam3Wrapper.forward); assert 'support' in sig.parameters; assert 'box_hints' not in sig.parameters"
  ```

- [ ] Loading a config with `prompt_mode:` (any value) raises `extra_forbidden` (verified by `test_prompt_mode_rejected_by_schema`).

  ```bash
  uv run pytest tests/unit/test_config_schema.py::test_prompt_mode_rejected_by_schema -v
  ```

- [ ] `enable_inst_interactivity=False` is retained at `models/sam3.py:616` with the new anchoring comment citing #126.

  ```bash
  grep -B2 'enable_inst_interactivity=False' src/custom_sam_peft/models/sam3.py | grep '#126'
  ```

- [ ] Final acceptance grep returns 0.

  ```bash
  ! grep -rn 'prompt_mode\|BoxPrompts' configs/ src/ docs/ARCHITECTURE.md docs/config-schema.md CHANGELOG.md
  ```

- [ ] `pytest`, `mypy`, `ruff` are green; coverage gate holds. (Verified in Task 7 Step 7.)

- [ ] Dispatch a final code-review subagent (min sonnet/high) over the cumulative diff (`git diff origin/main...HEAD`). Confirm: (a) no stray `BoxPrompts` / `prompt_mode` references, (b) the `box_hint` curriculum still flows end-to-end (sampler + schedule + metric unchanged), (c) the anchoring comment cites #126 verbatim.

After the checkpoint passes, open the PR per the orchestrator pipeline (CLAUDE.md §"Implementation-Orchestrator Pipeline" step 3).
