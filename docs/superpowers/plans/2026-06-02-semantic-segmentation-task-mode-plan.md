# Semantic Segmentation Task Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level `task: semantic` mode to SAM3 — fixed-class multi-class semantic segmentation produced head-free by mask-classification marginalization over the existing grounding forward, with a domain-aware semantic-loss preset system, an mIoU evaluator, and full CLI lifecycle branching — while keeping the instance path byte-for-byte unchanged.

**Architecture:** A new `task` axis on `TrainConfig` gates a parallel path: a `mask_png`/semantic-HF data layer producing `SemanticTarget` dense label maps; a `models/semantic.py` `marginalize_group` that turns the SAM3 grounding output dict into a `(B, K+1, H, W)` per-concept foreground logit volume (background-prepended); a `SemanticLossConfig` + `SEMANTIC_PRESET_TABLE` mirroring the #112 instance loss machinery; a `SemanticEvaluator` computing mIoU/pixel-acc/per-class IoU from a streaming confusion matrix; and `cfg.task` dispatch across `train`/`eval`/`predict`/`export`/`doctor`. Zero new model heads — marginalization is pure inference-time math over keys the existing forward already validates.

**Tech Stack:** Python 3.12, PyTorch, pydantic v2, typer, pytest, albumentations (label-map augmentation), numpy (confusion matrix), Pillow/opencv (label PNG I/O). SAM 3.1 (`sam3`) for the real-forward GPU tests.

**Spec (source of truth):** `docs/superpowers/specs/2026-06-02-semantic-segmentation-task-mode-design.md`

**Builds on:** #112 domain-aware loss presets (`models/losses/` resolver + table + sidecar + doctor table); #75 aug presets; #111 channel work (carries over unchanged).

---

## Conventions every task obeys

Bake these into every implementer task — they are non-negotiable project gates.

- **Lint/format gate before each commit landing on a ready PR** (implementer commits during a draft PR are exempt, but run them anyway to avoid a pile-up):
  - `uv run ruff check <touched files>`
  - `uv run ruff format --check <touched files>` (this is **separate** from `ruff check`; CI runs format-check too — a task is not done if format-check fails).
  - `uv run mypy --strict <touched files>` (CI scopes mypy to `src/custom_sam_peft`).
- **No `assert isinstance(...)` (or any bare `assert`) in `src/`** — ruff S101 / bandit forbids `assert` in `src/`. Narrow structurally (`if isinstance(x, T) and ...:` / `if x is None: raise ...`), never with a bare `assert`. Tests under `tests/` may assert freely.
- **CPU test runs bypass the coverage gate:** run CPU subsets with `uv run pytest -o "addopts=" <path>` to bypass the global `--cov-fail-under=80` (the `--no-cov` flag does **not** work on this box). Do **NOT** run `pytest --cov` locally (it segfaults torch on this WSL2/sm_120 box) — trust CI for coverage; keep coverage ≥ 80%.
- **NEVER run a bare `pytest tests/`** — it triggers the full real-model GPU suite in one process and can freeze a 16 GB box. Always scope to a directory/file under `tests/unit`, `tests/config`, `tests/data`, etc., and always pass `-o "addopts="`.
- **GPU tests run ONLY via `scripts/run_gpu_tests.sh`** and are gated by the `requires_checkpoint` + `requires_compatible_gpu` markers. The implementer writes/edits GPU tests but verifies them **structurally** (ruff/mypy/`py_compile`) when no GPU is present, and relies on CI / the GPU runner to execute them. This plan names exactly which GPU smokes to add (spec §11) — keep them minimal (one process, real-model freeze risk).
- **cite / `# tbd:` discipline:** every new or changed default hyperparameter carries a `# cite:` or `# tbd:` tag. The spec §7.3, §4, §6 already resolve every tag in this plan — copy them **verbatim**; do not invent new ones and do not drop required ones. A silent untagged default is a task failure.
- **Eager-import caveat:** `src/custom_sam_peft/__init__.py` eagerly imports the train chain, so an import error anywhere in the package breaks the whole package. After any symbol-add/rename, verify with `uv run python -c "import custom_sam_peft"` and `uv run python -m py_compile <touched files>`.
- **Blast-radius gate (spec §12):** the tasks that touch broadly-consumed types — `Example` (Task A2), `collate_batch` (Task A6), `StepResult` / `_ScalarWindow` (Task C4) — must, before being declared done, `grep` every call site of the changed symbol **and run the full CPU suite** (not just the new test file): `uv run pytest -o "addopts=" tests/unit tests/config tests/data tests/cli tests/eval tests/train tests/predict -q`. These tasks are flagged **[BLAST-RADIUS]** below.
- **Instance-path invariance (hard, spec §1/§12):** any config omitting `task` (or `task: instance`) must validate and run exactly as today. `StepResult.empty()` keys, instance `metrics.json` keys, instance logging, and `collate_batch`'s existing keys must be byte-identical. Regression assertions are baked into Tasks A2, A6, C4, D3.
- **Pure-Python loss module:** `models/losses/semantic_presets.py` must NOT import torch (mirrors `presets.py`), so `csp doctor` and schema tests import it without dragging torch in. Verify with the import-boundary test in Task B2.

---

## File structure (what each new/modified file owns)

**New files:**

| File | Responsibility | Phase |
| --- | --- | --- |
| `src/custom_sam_peft/data/mask_png.py` | `MaskPngDataset` + `build_mask_png` builder (paired image/label-PNG dirs → `SemanticTarget`) | A |
| `src/custom_sam_peft/data/semantic_hf.py` | `SemanticHFDataset` (HF label-map feature → `SemanticTarget`) | A |
| `src/custom_sam_peft/models/semantic.py` | `marginalize_group` + `build_semantic_logits` (the `(B,K+1,H,W)` builder) + `semantic_argmax` | C |
| `src/custom_sam_peft/models/losses/semantic_presets.py` | `SEMANTIC_PRESET_TABLE`, `resolve`, `LOCKED_OFF`, `dump_semantic_loss_bundle`, `_SEM_TERM_CLASS_NAMES` (torch-free) | B |
| `src/custom_sam_peft/models/losses/semantic_compose.py` | `SemanticLoss` nn.Module + `build_semantic_loss` | B |
| `src/custom_sam_peft/eval/semantic_evaluator.py` | `SemanticEvaluator` (mIoU streaming-confusion forward loop) | D |

**Modified files:**

| File | Change | Phase |
| --- | --- | --- |
| `src/custom_sam_peft/config/schema.py` | `Task` alias, `task` field + `_check_task_data_compat` validator, `mask_png` `DataFormat`, `SemanticDataConfig`, `DataConfig.semantic`, `HFFieldMap.label_map`, `SemanticLossConfig`/`SemanticLossOverrides`/`SemMaskFamily`, `TrainHyperparams.semantic_loss`, `__all__` additions | A, B |
| `src/custom_sam_peft/data/base.py` | `SemanticTarget` dataclass; `Example.instances` defaulted + `Example.semantic` added | A |
| `src/custom_sam_peft/data/collate.py` | `collate_batch` adds `"semantic"` key | A |
| `src/custom_sam_peft/data/hf.py` | `build_hf` task branch → dispatch to `SemanticHFDataset` | A |
| `src/custom_sam_peft/models/losses/__init__.py` | export `resolve`/`SemanticLossConfig` helpers / `build_semantic_loss` / `dump_semantic_loss_bundle` | B |
| `src/custom_sam_peft/train/loop.py` | `train_step` semantic branch; `StepResult`/`_ScalarWindow` parametrized key set | C |
| `src/custom_sam_peft/train/trainer.py` | semantic loss-bundle sidecar; pass task loss-key set down | C |
| `src/custom_sam_peft/train/runner.py` | thread `task` into the `hf` builder | A/C |
| `src/custom_sam_peft/eval/metrics.py` | `SemanticMetrics` + `compute_semantic_metrics`; task-tagged `metrics.json` | D |
| `src/custom_sam_peft/eval/runner.py` | `run_eval` dispatch to `SemanticEvaluator` on `cfg.task` | D, E |
| `src/custom_sam_peft/predict/runner.py` | semantic predict path (marginalize → label map) | E |
| `src/custom_sam_peft/predict/writers.py` | `write_semantic_label_map` | E |
| `src/custom_sam_peft/predict/visualize.py` | semantic overlay viz | E |
| `src/custom_sam_peft/cli/predict_cmd.py` | prompt-defaulting + instance-only-flag INFO under semantic | E |
| `src/custom_sam_peft/cli/doctor_cmd.py` | task row + "Resolved semantic losses" table + `--json` `task`/`semantic_loss` | E |

---

## Phase dependency map

| Phase | Title | Depends on | Notes |
| --- | --- | --- | --- |
| A | config + data | — | CPU-only. Heart of the new schema + data layer. |
| B | semantic loss | A (`SemanticTarget`) | CPU-only. File-disjoint from A's *data* files but shares `schema.py` — see ordering note. |
| C | forward/marginalization + train branch | A (encoding), B (loss) | CPU-stub + 1 GPU smoke. |
| D | eval | A, C | CPU + 1 GPU smoke. |
| E | CLI / predict / export / doctor | A–D | Wiring phase. Mostly CPU (stubbed). |

**Session boundaries (per the user's one-phase-per-session orchestration):** A natural grouping is **A+B** (data + loss, both pure/CPU), then **C**, then **D+E**. A and B both edit `config/schema.py`; if run in the same session, **serialize the schema edits** (A's `Task`/`SemanticDataConfig`/validator land first, then B's `SemanticLossConfig` block) — they touch different regions but the same file, so a parallel-agent commit race would orphan a commit (per memory). Within a phase, tasks are grouped below by file-disjointness; **[PARALLEL-OK]** marks groups the orchestrator MAY dispatch in parallel, **[SERIALIZE]** marks chained/shared-file tasks.

---

## Phase A — config + data (§4, §5)

**Feature block:** the `task`/`data.semantic` schema with cross-validation, the `SemanticTarget`/`Example.semantic` types, the `mask_png` + semantic-HF adapters, the §5.2 GT encoding, label-map augmentation alignment, and the collate `"semantic"` key. CPU-only and fully testable with synthetic label PNGs + a tiny in-memory HF dataset.

**Consumes:** nothing downstream.

**Interface contract this phase PRODUCES (spec §5.7) — restate to the next session verbatim-faithful:**

> Exposes `SemanticTarget(labels: (H,W) int64 in {0..K} ∪ {ignore_index}, ignore_index: int)`; `Example.semantic: SemanticTarget | None` (and `Example.instances` now defaulted); the `(K+1)`-channel ↔ GT-label convention (**§5.2**: `dataset.class_names` = concept names in ascending class_map-pixel-value order with any explicit background class removed, `len == K`; concept `dense_id = i` → prompted as `class_names[i]` → logit channel `i+1`; GT label = `0` background / `i+1` concept `i` / `ignore_index` void); the `mask_png` and semantic-`hf` builders returning a `Dataset` whose `class_names` has length K; the collate `"semantic"` key (`[None]*B` under instance); and the `task`/`data.semantic`/`DataFormat: mask_png`/`HFFieldMap.label_map` schema. Consumes nothing downstream. Fully CPU-testable.

### Task A1: `Task` axis, `mask_png` DataFormat, `SemanticDataConfig`, task↔data validator

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (`DataFormat` at `:90`; `Task` alias near it; `DataConfig` at `:383`; `DataSplit` docstring at `:123`; `TrainConfig` model validator; `__all__` at `:39`)
- Test: `tests/config/test_config_schema.py` (or `tests/unit/test_config_schema.py` — place next to the existing schema tests; locate the file with `grep -rln "class TrainConfig\|TrainConfig(" tests/`)

This task is **[SERIALIZE]** — it is the schema foundation every later task imports.

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_semantic_schema.py  (new file alongside the existing schema tests)
"""Schema coverage for the #113 task axis + semantic data config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import TrainConfig


def _base_cfg(**over):
    """Minimal valid instance config dict; callers override task/data/etc."""
    cfg = {
        "run": {"name": "t", "output_dir": "runs/t"},
        "data": {
            "format": "coco",
            "splits": {"train": {"images": "img", "annotations": "ann.json"}},
        },
        "peft": {"method": "lora"},
        "train": {},
    }
    cfg.update(over)
    return cfg


def test_task_defaults_to_instance():
    cfg = TrainConfig.model_validate(_base_cfg())
    assert cfg.task == "instance"


def test_semantic_rejects_coco_format():
    with pytest.raises(ValidationError, match="does not support data.format: coco"):
        TrainConfig.model_validate(
            _base_cfg(
                task="semantic",
                data={
                    "format": "coco",
                    "splits": {"train": {"images": "img", "annotations": "ann.json"}},
                },
            )
        )


def test_semantic_requires_data_semantic():
    with pytest.raises(ValidationError, match="requires data.semantic"):
        TrainConfig.model_validate(
            _base_cfg(
                task="semantic",
                data={
                    "format": "mask_png",
                    "splits": {"train": {"images": "img", "annotations": "labels"}},
                },
            )
        )


def test_instance_rejects_data_semantic():
    with pytest.raises(ValidationError, match="data.semantic is only valid"):
        TrainConfig.model_validate(
            _base_cfg(
                data={
                    "format": "hf",
                    "splits": {"train": {"images": "x", "annotations": "y"}},
                    "semantic": {"class_map": "cm.json"},
                },
            )
        )


def test_instance_rejects_mask_png_format():
    with pytest.raises(ValidationError, match="mask_png requires task: semantic"):
        TrainConfig.model_validate(
            _base_cfg(
                data={
                    "format": "mask_png",
                    "splits": {"train": {"images": "img", "annotations": "labels"}},
                },
            )
        )


def test_semantic_mask_png_valid():
    cfg = TrainConfig.model_validate(
        _base_cfg(
            task="semantic",
            data={
                "format": "mask_png",
                "splits": {"train": {"images": "img", "annotations": "labels"}},
                "semantic": {"class_map": "cm.json"},
            },
        )
    )
    assert cfg.task == "semantic"
    assert cfg.data.semantic is not None
    assert cfg.data.semantic.ignore_index == 255  # default void
    assert cfg.data.semantic.label_suffix == "_labelIds.png"


def test_semantic_rejects_nondefault_eval_iou_thresholds():
    with pytest.raises(ValidationError, match="iou_thresholds"):
        TrainConfig.model_validate(
            _base_cfg(
                task="semantic",
                data={
                    "format": "mask_png",
                    "splits": {"train": {"images": "img", "annotations": "labels"}},
                    "semantic": {"class_map": "cm.json"},
                },
                eval={"iou_thresholds": [0.5, 0.75]},
            )
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -o "addopts=" tests/config/test_semantic_schema.py -v`
Expected: FAIL — `task` field / `mask_png` format / `SemanticDataConfig` do not exist yet (ValidationError on `task`/`mask_png`, or AttributeError).

- [ ] **Step 3: Implement the schema changes**

In `src/custom_sam_peft/config/schema.py`:

1. Add the `Task` alias next to `DataFormat` (`~:90`) and extend `DataFormat`:

```python
DataFormat = Literal["coco", "hf", "mask_png"]  # mask_png is semantic-only (§4.2)
Task = Literal["instance", "semantic"]  # cite: #113 — kept trivially extensible (panoptic out of scope)
```

2. Add `SemanticDataConfig` (place near `DataConfig`, before it). Copy the field bodies and tags **verbatim** from spec §4.2:

```python
class SemanticDataConfig(_Strict):
    """Semantic-segmentation data parameters. Required when task == 'semantic'.

    Lives under DataConfig.semantic. None for instance datasets.
    """

    class_map: str = Field(
        min_length=1,
        description=(
            "Path to a JSON file mapping integer pixel value -> class name, e.g. "
            '{"0": "background", "1": "road", "2": "building"}. The set of NAMES '
            "(excluding any explicit background, see §4.5) is the prompted concept "
            "vocabulary AND the dataset class_names, in ascending-pixel-value order."
        ),
    )
    ignore_index: int = Field(
        default=255,  # cite: PASCAL VOC / Cityscapes void convention (255)
        description=(
            "Pixel value in the label map treated as void/unlabeled. Excluded from "
            "both loss and metrics. Not a class. Default 255 is the de-facto standard."
        ),
    )
    label_suffix: str = Field(
        default="_labelIds.png",  # tbd: #113 — Cityscapes-style; override per dataset
        description=(
            "Filename suffix that maps an image file to its label-map PNG (mask_png "
            "format only). image 'aachen_000000.png' -> label "
            "'aachen_000000{label_suffix}'. Set to '.png' for same-stem pairing."
        ),
    )
```

3. Add to `DataConfig` (`~:383`): `semantic: SemanticDataConfig | None = None  # required when task == 'semantic'`.

4. Extend the `DataSplit` docstring (`~:123`) to note: for `mask_png`, `annotations` is reinterpreted as the **label-map PNG directory** and `images` as the image directory (no JSON file). Do not add a new split type.

5. Add `task: Task = "instance"  # cite: #113 — default preserves the instance path exactly` as the **last** field of `TrainConfig` (so existing positional/dict construction is unaffected).

6. Add the cross-field validator on `TrainConfig` (it sees `self.task` and `self.data`). Per spec §4.3 + the §4.4 **Decision** ("validator errors ONLY on data.format/data.semantic mismatches and on explicit non-default `eval.iou_thresholds`/`eval.mask_threshold` under semantic"):

```python
@model_validator(mode="after")
def _check_task_data_compat(self) -> TrainConfig:
    if self.task == "semantic":
        if self.data.format == "coco":
            raise ValueError(
                "task: semantic does not support data.format: coco (instance JSON). "
                "Use data.format: mask_png or hf with a semantic field map."
            )
        if self.data.semantic is None:
            raise ValueError(
                "task: semantic requires data.semantic (class_map, ignore_index)."
            )
        # eval knobs that are inert under semantic (§4.4): reject only if set non-default.
        if self.eval.iou_thresholds != EvalConfig().iou_thresholds:
            raise ValueError(
                "eval.iou_thresholds is inert under task: semantic (mIoU has no "
                "threshold sweep). Remove it."
            )
        if self.eval.mask_threshold != EvalConfig().mask_threshold:
            raise ValueError(
                "eval.mask_threshold is inert under task: semantic (argmax, not "
                "per-mask binarize). Remove it."
            )
    else:  # instance
        if self.data.semantic is not None:
            raise ValueError("data.semantic is only valid when task: semantic.")
        if self.data.format == "mask_png":
            raise ValueError("data.format: mask_png requires task: semantic.")
    return self
```

> Implementation note: `EvalConfig().iou_thresholds`/`.mask_threshold` give the default values to compare against — construct a fresh `EvalConfig()` inside the validator (cheap; `EvalConfig` has a default factory). Keep the validator pure (no I/O).

7. Add `"Task"` and `"SemanticDataConfig"` to `__all__` (`:39`).

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest -o "addopts=" tests/config/test_semantic_schema.py -v`
Expected: PASS (all 7).

- [ ] **Step 5: Lint/type + import smoke**

```bash
uv run ruff check src/custom_sam_peft/config/schema.py
uv run ruff format --check src/custom_sam_peft/config/schema.py
uv run mypy --strict src/custom_sam_peft/config/schema.py
uv run python -c "import custom_sam_peft"
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/config/test_semantic_schema.py
git commit -m "feat(#113): add task axis, mask_png format, SemanticDataConfig + task<->data validator"
```

### Task A2: `SemanticTarget` + `Example.semantic` (`data/base.py`) **[BLAST-RADIUS]**

**Files:**
- Modify: `src/custom_sam_peft/data/base.py` (`Example` at `:47`)
- Test: `tests/unit/test_data_base.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_data_base.py  (append)
import torch

from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts


def test_semantic_target_holds_labels_and_ignore_index():
    labels = torch.zeros(4, 4, dtype=torch.int64)
    tgt = SemanticTarget(labels=labels, ignore_index=255)
    assert tgt.labels.dtype == torch.int64
    assert tgt.ignore_index == 255


def test_example_instances_now_defaulted_and_semantic_none():
    # Instance construction unchanged: positional instances still valid.
    ex = Example(
        image=torch.zeros(3, 8, 8),
        image_id="a",
        prompts=TextPrompts(classes=["cat"]),
        instances=[],
    )
    assert ex.semantic is None
    assert ex.instances == []


def test_example_semantic_path_leaves_instances_empty():
    tgt = SemanticTarget(labels=torch.zeros(8, 8, dtype=torch.int64), ignore_index=255)
    ex = Example(
        image=torch.zeros(3, 8, 8),
        image_id="b",
        prompts=TextPrompts(classes=["road", "building"]),
        semantic=tgt,
    )
    assert ex.instances == []
    assert ex.semantic is tgt
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_base.py -v`
Expected: FAIL — `SemanticTarget` does not exist; `Example(... semantic=...)` raises TypeError.

- [ ] **Step 3: Implement**

In `src/custom_sam_peft/data/base.py`, add `SemanticTarget` (above `Example`) and modify `Example`:

```python
@dataclass(frozen=True)
class SemanticTarget:
    """Dense per-pixel class labels for one image (semantic task).

    `labels` holds class ids in {0..K} where 0 == background and 1..K == concept
    dense_id + 1 (the +1 makes room for the background channel). Pixels equal to
    `ignore_index` are void: excluded from loss and metrics.
    """

    labels: torch.Tensor  # (H, W) int64, values in {0..K} ∪ {ignore_index}
    ignore_index: int  # carried so collate/loss/eval need no extra plumbing


@dataclass(frozen=True)
class Example:
    """One training/eval example. Carries instances XOR a semantic target."""

    image: torch.Tensor  # (C, H, W) normalized  (C from data.channels)
    image_id: str
    prompts: Prompts
    instances: list[Instance] = field(default_factory=list)  # populated iff task == instance
    semantic: SemanticTarget | None = None  # populated iff task == semantic
```

Add `from dataclasses import dataclass, field` (the module currently imports only `dataclass`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_base.py -v`
Expected: PASS.

- [ ] **Step 5 [BLAST-RADIUS]: grep all `Example(` call sites + full CPU suite**

```bash
grep -rn "Example(" src/ tests/ | grep -v "def \|SemanticTarget\|# "
uv run pytest -o "addopts=" tests/unit tests/config tests/data tests/cli tests/eval tests/train tests/predict -q
```
Expected: every existing `Example(...)` call passes `instances` (positionally or by keyword) — defaulting it keeps them valid; the full CPU suite is GREEN. If any site relied on `instances` being positionally required in a way the default breaks, fix it here.

- [ ] **Step 6: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/data/base.py
uv run ruff format --check src/custom_sam_peft/data/base.py
uv run mypy --strict src/custom_sam_peft/data/base.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/data/base.py tests/unit/test_data_base.py
git commit -m "feat(#113): SemanticTarget dataclass + Example.semantic (instances now defaulted)"
```

### Task A3: Shared §5.2 GT-encoding helper (`data/_semantic_encode.py`, new) **[PARALLEL-OK with A2 after A1]**

The `mask_png` and HF adapters both need the identical class_map → `(class_names, value_to_label)` derivation (§5.2/§4.5). Factor it into one helper so both adapters share a single source of truth (DRY) and the encoding is tested once.

**Files:**
- Create: `src/custom_sam_peft/data/_semantic_encode.py`
- Test: `tests/unit/test_semantic_encode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_semantic_encode.py
"""§5.2/§4.5 GT-encoding helper: class_map -> (class_names, value->label)."""

from __future__ import annotations

import json

from custom_sam_peft.data._semantic_encode import build_value_to_label


def test_ascending_pixel_value_order_drops_background(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "2": "building", "1": "road"}))
    names, value_to_label, ignore = build_value_to_label(
        str(cm), ignore_index=255, background_class_name=None
    )
    # class_names = concept names in ascending pixel-value order, bg removed.
    assert names == ["road", "building"]  # value 1 -> road (dense 0), value 2 -> building (dense 1)
    # GT label = dense_id + 1; background -> 0; ignore stays ignore.
    assert value_to_label[0] == 0  # explicit background class
    assert value_to_label[1] == 1  # road -> channel 1
    assert value_to_label[2] == 2  # building -> channel 2
    assert ignore == 255


def test_recognized_background_names_case_insensitive(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "Unlabeled", "1": "road"}))
    names, value_to_label, _ = build_value_to_label(
        str(cm), ignore_index=255, background_class_name=None
    )
    assert names == ["road"]
    assert value_to_label[0] == 0  # "Unlabeled" recognized as background


def test_custom_background_class_name(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"7": "void_region", "1": "road", "2": "tree"}))
    names, value_to_label, _ = build_value_to_label(
        str(cm), ignore_index=255, background_class_name="void_region"
    )
    assert names == ["road", "tree"]
    assert value_to_label[7] == 0  # custom bg -> channel 0


def test_no_background_class_all_concepts(tmp_path):
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"1": "road", "2": "tree"}))
    names, value_to_label, _ = build_value_to_label(
        str(cm), ignore_index=255, background_class_name=None
    )
    assert names == ["road", "tree"]
    assert value_to_label == {1: 1, 2: 2}  # nothing maps to channel 0 from data
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_encode.py -v`
Expected: FAIL — `build_value_to_label` does not exist.

- [ ] **Step 3: Implement**

```python
# src/custom_sam_peft/data/_semantic_encode.py
"""§5.2/§4.5 GT-encoding: class_map JSON -> concept names + pixel-value->GT-label map.

Single source of truth shared by the mask_png and semantic-HF adapters. The
prompted concept order DEFINES the dense ids and the (K+1)-channel <-> GT-label
correspondence (spec §5.2). Pure-Python (no torch).
"""

from __future__ import annotations

import json

# Recognized explicit-background class names (case-insensitive), §4.5.
_BACKGROUND_NAMES = frozenset({"background", "bg", "none", "unlabeled"})


def build_value_to_label(
    class_map_path: str,
    *,
    ignore_index: int,
    background_class_name: str | None,
) -> tuple[list[str], dict[int, int], int]:
    """Return (class_names, value_to_label, ignore_index).

    - class_names: concept names in ASCENDING class_map pixel-value order, with any
      explicit background class removed. len == K.
    - value_to_label: pixel value -> GT label, where 0 == background, i+1 == concept
      with dense_id i. The configured ignore_index value is NOT placed in this map;
      callers remap it separately (it always wins, §4.5).
    """
    with open(class_map_path, encoding="utf-8") as fh:
        raw: dict[str, str] = json.load(fh)
    # Sort by integer pixel value ascending.
    pairs = sorted(((int(v), name) for v, name in raw.items()), key=lambda kv: kv[0])

    bg_lower = background_class_name.lower() if background_class_name is not None else None

    def _is_background(name: str) -> bool:
        low = name.lower()
        if bg_lower is not None:
            return low == bg_lower
        return low in _BACKGROUND_NAMES

    class_names: list[str] = []
    value_to_label: dict[int, int] = {}
    for value, name in pairs:
        if _is_background(name):
            value_to_label[value] = 0  # background channel
            continue
        dense_id = len(class_names)
        class_names.append(name)
        value_to_label[value] = dense_id + 1  # +1 for the prepended background channel
    return class_names, value_to_label, ignore_index
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_encode.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check src/custom_sam_peft/data/_semantic_encode.py
uv run ruff format --check src/custom_sam_peft/data/_semantic_encode.py
uv run mypy --strict src/custom_sam_peft/data/_semantic_encode.py
git add src/custom_sam_peft/data/_semantic_encode.py tests/unit/test_semantic_encode.py
git commit -m "feat(#113): shared §5.2 class_map -> value->label encoding helper"
```

### Task A4: `mask_png` adapter (`data/mask_png.py`) **[SERIALIZE — depends on A1, A2, A3]**

**Files:**
- Create: `src/custom_sam_peft/data/mask_png.py`
- Test: `tests/unit/test_data_mask_png.py`

Mirror `build_coco` (`data/coco.py:325`) and `CocoDataset` (`:124`): split-subdict selection, `build_train_transforms`/`build_eval_transforms`, a `Dataset` with `class_names` + `image_class_labels`. Reference `data/io.read_image` for image loading and `data/coco.py:181` for the `image_class_labels` pattern.

- [ ] **Step 1: Write the failing tests** (synthetic temp tree: 2 images + 2 label PNGs + class_map.json)

```python
# tests/unit/test_data_mask_png.py
"""MaskPngDataset against a synthetic temp tree (§5.3)."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from custom_sam_peft.data._registry import lookup
from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts


def _make_tree(tmp_path):
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for stem in ("a", "b"):
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(img_dir / f"{stem}.png")
        lbl = np.zeros((16, 16), dtype=np.uint8)
        lbl[:8, :8] = 1  # road
        lbl[8:, 8:] = 2  # building
        lbl[0, 0] = 255  # void
        Image.fromarray(lbl, mode="L").save(lbl_dir / f"{stem}.png")
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "1": "road", "2": "building"}))
    return img_dir, lbl_dir, cm


def _build(tmp_path):
    img_dir, lbl_dir, cm = _make_tree(tmp_path)
    cfg = {
        "splits": {"train": {"images": str(img_dir), "annotations": str(lbl_dir)}},
        "semantic": {"class_map": str(cm), "ignore_index": 255, "label_suffix": ".png"},
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }
    builder = lookup("dataset", "mask_png")
    return builder(cfg, model_name="sam3.1", pipeline="eval")


def test_class_names_ascending_drop_background(tmp_path):
    ds = _build(tmp_path)
    assert ds.class_names == ["road", "building"]  # K == 2


def test_getitem_returns_semantic_example(tmp_path):
    ds = _build(tmp_path)
    ex = ds[0]
    assert isinstance(ex, Example)
    assert ex.instances == []
    assert isinstance(ex.semantic, SemanticTarget)
    assert ex.semantic.labels.dtype == torch.int64
    assert ex.semantic.ignore_index == 255
    # §5.2 encoding: road -> 1, building -> 2, void -> 255, bg -> 0.
    vals = set(ex.semantic.labels.unique().tolist())
    assert vals <= {0, 1, 2, 255}
    assert 255 in vals  # the void pixel survived (nearest interp; not normalized)


def test_prompts_are_full_vocabulary_mode_all(tmp_path):
    ds = _build(tmp_path)
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["road", "building"]


def test_missing_label_pair_raises(tmp_path):
    img_dir, lbl_dir, cm = _make_tree(tmp_path)
    (lbl_dir / "a.png").unlink()
    cfg = {
        "splits": {"train": {"images": str(img_dir), "annotations": str(lbl_dir)}},
        "semantic": {"class_map": str(cm), "ignore_index": 255, "label_suffix": ".png"},
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }
    builder = lookup("dataset", "mask_png")
    import pytest

    with pytest.raises(FileNotFoundError, match="a"):
        builder(cfg, model_name="sam3.1", pipeline="eval")
```

> Note: confirm the exact builder cfg-dict shape by reading `build_coco`'s body (`data/coco.py:326`) — it reads `cfg["splits"]`, `cfg.get("channels")`, `cfg.get("text_prompt")`, etc. Match its key access exactly so the trainer's `_build_dataset_from_dict` (which passes the `data` model-dump) works unchanged. Adjust the test's cfg dict to whatever `build_coco` actually consumes.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_mask_png.py -v`
Expected: FAIL — no `mask_png` registry entry (`lookup("dataset", "mask_png")` raises KeyError).

- [ ] **Step 3: Implement `MaskPngDataset` + `build_mask_png`**

Per spec §5.3. Key points (read `data/coco.py:124-322` for the mirror):

- `@register("dataset", "mask_png")` decorating `build_mask_png(cfg, *, model_name, pipeline)`.
- `build_mask_png` selects the split sub-dict, builds transforms via `build_train_transforms`/`build_eval_transforms` (pipeline-keyed), reads `cfg["semantic"]` for `class_map`/`ignore_index`/`label_suffix`, and returns a `MaskPngDataset`.
- `MaskPngDataset.__init__(images_dir, labels_dir, *, class_map_path, ignore_index, label_suffix, transforms, text_prompt, channels)`:
  - call `build_value_to_label(class_map_path, ignore_index=ignore_index, background_class_name=None)` (A3) → `class_names`, `value_to_label`, `ignore_index`.
  - enumerate `images_dir`; pair each image stem to `labels_dir / (stem + label_suffix)`. Collect missing pairs; if any, `raise FileNotFoundError(f"mask_png: {len(missing)} images have no label; first few: {missing[:5]}")`.
- `__getitem__`:
  - `image = read_image(img_path, channels)` — `data/io.read_image(path, channels)` returns an `(H, W, C)` numpy array (NOT a torch tensor). The transforms convert to a normalized `(C, H, W)` tensor; mirror exactly how `CocoDataset.__getitem__` (`coco.py`) feeds `read_image`'s output through `transforms` so the image-tensor shape/normalization matches the instance path.
  - read label PNG as a single-channel array **without normalization** (raw class indices): `np.array(Image.open(label_path))` — keep uint8/uint16 dtype; do NOT scale.
  - remap pixel values → GT labels: build an `(H, W)` int64 array where each pixel `v` maps to `value_to_label.get(v, ...)`; the configured `ignore_index` value remaps to `ignore_index` (it always wins, §4.5). Any value not in `class_map` and not `ignore_index` → treat as background `0` (document this; it is the safe default).
  - run transforms with the image + the label map as **one aligned mask target** (§5.5 — Task A5 verifies nearest interp; for this task, just pass the label through the transforms' mask channel).
  - `prompts = TextPrompts(classes=list(class_names))` (full vocabulary, mode forced `all`, §5.6).
  - `semantic = SemanticTarget(labels=torch.from_numpy(remapped).to(torch.int64), ignore_index=ignore_index)`.
  - return `Example(image, image_id=stem, prompts=prompts, semantic=semantic)` (instances left default `[]`).
- `class_names` property → the K concept names.
- `image_class_labels` property (mirror `coco.py:181`): per image, the `frozenset` of present GT class ids **excluding background (0) and ignore_index**. (Used by stratified subset.)

> §5.6 INFO: if `text_prompt.mode` is set to a non-`all` value, emit a one-time `_LOG.info("task: semantic forces text-prompt mode 'all'; ignoring mode=%s", mode)`. Force `all` regardless.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_mask_png.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/data/mask_png.py
uv run ruff format --check src/custom_sam_peft/data/mask_png.py
uv run mypy --strict src/custom_sam_peft/data/mask_png.py
uv run python -c "import custom_sam_peft; from custom_sam_peft.data import mask_png"
git add src/custom_sam_peft/data/mask_png.py tests/unit/test_data_mask_png.py
git commit -m "feat(#113): mask_png semantic data adapter"
```

### Task A5: Label-map augmentation alignment — nearest-interp (§5.5) **[SERIALIZE — depends on A4]**

**Files:**
- Modify: `src/custom_sam_peft/data/transforms.py` (verify/force nearest interp on mask targets) and/or `src/custom_sam_peft/data/mask_png.py` (how the label is passed to transforms)
- Test: `tests/unit/test_semantic_aug_nearest.py`

Spec §5.5/§14 (Risk): a bilinear resize of the label map would silently invent fractional class ids. This task adds the explicit regression test the spec mandates.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_semantic_aug_nearest.py
"""Label-map augmentation must use nearest interp (no fractional class ids), §5.5."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from custom_sam_peft.data._registry import lookup


def test_label_map_resize_preserves_integer_class_ids(tmp_path):
    img_dir = tmp_path / "img"
    lbl_dir = tmp_path / "lbl"
    img_dir.mkdir()
    lbl_dir.mkdir()
    # An odd-sized label map with sharp class boundaries -> resize must NOT blend.
    Image.fromarray(np.zeros((37, 37, 3), dtype=np.uint8)).save(img_dir / "a.png")
    lbl = np.zeros((37, 37), dtype=np.uint8)
    lbl[:18] = 1
    lbl[18:] = 2
    Image.fromarray(lbl, mode="L").save(lbl_dir / "a.png")
    cm = tmp_path / "cm.json"
    cm.write_text(json.dumps({"0": "background", "1": "road", "2": "building"}))
    cfg = {
        "splits": {"train": {"images": str(img_dir), "annotations": str(lbl_dir)}},
        "semantic": {"class_map": str(cm), "ignore_index": 255, "label_suffix": ".png"},
        "channels": 3,
        "text_prompt": {"mode": "all"},
    }
    ds = lookup("dataset", "mask_png")(cfg, model_name="sam3.1", pipeline="train")
    labels = ds[0].semantic.labels
    # After any geometric resize, every value is still an exact integer class id.
    present = set(labels.unique().tolist())
    assert present <= {0, 1, 2, 255}, f"fractional/blended ids leaked: {present}"
    assert labels.dtype == torch.int64
```

- [ ] **Step 2: Run to verify it fails (or passes-by-accident — confirm the mechanism)**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_aug_nearest.py -v`
Expected: This may FAIL if the resize step uses bilinear on the mask. Read `data/transforms.py`'s resize construction; albumentations applies nearest to mask targets by default, but the **resize** op must be confirmed to use `interpolation=cv2.INTER_NEAREST` for masks (or be a `Resize` whose `mask_interpolation`/default is nearest). If the test passes immediately, still add an explicit assertion/comment in `transforms.py` documenting the nearest requirement so a future edit can't silently regress it.

- [ ] **Step 3: Ensure nearest interp on the label-map mask target**

In `data/transforms.py`, locate the resize/geometric ops in `build_train_transforms`/`build_eval_transforms`. Ensure the mask target is resized with `cv2.INTER_NEAREST`:
- If using `A.Resize(...)`, set `mask_interpolation=cv2.INTER_NEAREST` (albumentations ≥ 1.4 supports it) OR confirm masks are routed through the nearest path. Add a `# cite: standard semantic-seg practice (nearest interp; bilinear invents fractional class ids)` comment.
- The image keeps its existing interpolation (bilinear). No photometric/color aug touches the label map (it is a mask target, not an image).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_aug_nearest.py -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check src/custom_sam_peft/data/transforms.py
uv run ruff format --check src/custom_sam_peft/data/transforms.py
uv run mypy --strict src/custom_sam_peft/data/transforms.py
git add src/custom_sam_peft/data/transforms.py tests/unit/test_semantic_aug_nearest.py
git commit -m "feat(#113): nearest-interp label-map augmentation alignment + regression test"
```

### Task A6: Collate `"semantic"` key (`data/collate.py`) **[BLAST-RADIUS]**

**Files:**
- Modify: `src/custom_sam_peft/data/collate.py:29-34`
- Test: `tests/unit/test_data_collate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_data_collate.py  (append)
import torch

from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts
from custom_sam_peft.data.collate import collate_batch


def test_collate_adds_semantic_key_none_under_instance():
    exs = [
        Example(torch.zeros(3, 8, 8), "a", TextPrompts(["cat"]), instances=[]),
        Example(torch.zeros(3, 8, 8), "b", TextPrompts(["cat"]), instances=[]),
    ]
    out = collate_batch(exs)
    assert out["semantic"] == [None, None]
    assert out["instances"] == [[], []]


def test_collate_carries_semantic_targets():
    tgt = SemanticTarget(torch.zeros(8, 8, dtype=torch.int64), ignore_index=255)
    exs = [
        Example(torch.zeros(3, 8, 8), "a", TextPrompts(["road"]), semantic=tgt),
        Example(torch.zeros(3, 8, 8), "b", TextPrompts(["road"]), semantic=tgt),
    ]
    out = collate_batch(exs)
    assert out["semantic"] == [tgt, tgt]
    assert out["instances"] == [[], []]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_collate.py -v`
Expected: FAIL — `"semantic"` key absent (KeyError).

- [ ] **Step 3: Implement**

In `collate_batch`'s return dict (`:29`), add the `"semantic"` key (spec §5.7):

```python
    return {
        "images": images,
        "image_ids": [ex.image_id for ex in examples],
        "prompts": [ex.prompts for ex in examples],
        "instances": [list(ex.instances) for ex in examples],  # [] under semantic
        "semantic": [ex.semantic for ex in examples],  # [None]*B under instance
    }
```

Update the docstring to list the new key. Image-shape consistency check is unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_collate.py -v`
Expected: PASS.

- [ ] **Step 5 [BLAST-RADIUS]: grep `collate_batch` consumers + full CPU suite**

```bash
grep -rn "collate_batch\|batch\[\"semantic\"\]\|batch\[\"instances\"\]" src/ tests/
uv run pytest -o "addopts=" tests/unit tests/config tests/data tests/cli tests/eval tests/train tests/predict -q
```
Expected: no existing consumer breaks (the new key is additive); full CPU suite GREEN.

- [ ] **Step 6: Lint/type + commit**

```bash
uv run ruff check src/custom_sam_peft/data/collate.py
uv run ruff format --check src/custom_sam_peft/data/collate.py
uv run mypy --strict src/custom_sam_peft/data/collate.py
git add src/custom_sam_peft/data/collate.py tests/unit/test_data_collate.py
git commit -m "feat(#113): collate_batch semantic key ([None]*B under instance)"
```

### Task A7: Semantic HF adapter (`data/semantic_hf.py` + `build_hf` task branch + `HFFieldMap.label_map`) **[SERIALIZE — depends on A1, A2, A3]**

**Files:**
- Create: `src/custom_sam_peft/data/semantic_hf.py`
- Modify: `src/custom_sam_peft/data/hf.py` (`build_hf` at `:363`), `src/custom_sam_peft/config/schema.py` (`HFFieldMap` at `:303`), `src/custom_sam_peft/train/runner.py` (`_build_dataset_from_dict` at `:65` — thread `task`)
- Test: `tests/unit/test_data_semantic_hf.py`

- [ ] **Step 1: Add `HFFieldMap.label_map` (sub-step, schema)**

In `schema.py` `HFFieldMap` (`:303`), add:

```python
    label_map: str | None = None  # cite: #113 — HF feature holding the (H,W) label image
```

When `task == "semantic"`, `label_map` is **required** — validate that in the semantic builder (not the schema), to keep `HFFieldMap` task-agnostic (spec §5.4).

- [ ] **Step 2: Write the failing tests** (tiny in-memory HF `Dataset` with a label feature)

```python
# tests/unit/test_data_semantic_hf.py
"""SemanticHFDataset against a tiny in-memory HF dataset (§5.4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

datasets = pytest.importorskip("datasets")

from custom_sam_peft.data.base import Example, SemanticTarget
from custom_sam_peft.data.semantic_hf import SemanticHFDataset


def _tiny_ds():
    imgs = [Image.fromarray(np.zeros((16, 16, 3), np.uint8)) for _ in range(2)]
    lbls = []
    for _ in range(2):
        a = np.zeros((16, 16), np.uint8)
        a[:8] = 1
        a[8:] = 2
        lbls.append(Image.fromarray(a, mode="L"))
    return datasets.Dataset.from_dict({"image": imgs, "annotation": lbls})


def test_semantic_hf_requires_label_map_field():
    ds = _tiny_ds()
    with pytest.raises(ValueError, match="label_map"):
        SemanticHFDataset(
            ds,
            image_field="image",
            label_map_field=None,  # missing -> error
            class_names=["road", "building"],
            ignore_index=255,
            transforms=None,
            channels=3,
        )


def test_semantic_hf_getitem(tmp_path):
    ds = _tiny_ds()
    sds = SemanticHFDataset(
        ds,
        image_field="image",
        label_map_field="annotation",
        class_names=["road", "building"],
        ignore_index=255,
        transforms=None,
        channels=3,
    )
    ex = sds[0]
    assert isinstance(ex, Example)
    assert isinstance(ex.semantic, SemanticTarget)
    assert ex.semantic.labels.dtype == torch.int64
    assert sds.class_names == ["road", "building"]
```

> The exact `SemanticHFDataset.__init__` signature is the implementer's to finalize against `HFDataset` (`hf.py:130`); match its transform/channels plumbing. The test above pins the contract: requires a label-map field, materializes a `SemanticTarget`, exposes `class_names`.

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_semantic_hf.py -v`
Expected: FAIL — `SemanticHFDataset` does not exist.

- [ ] **Step 4: Implement `SemanticHFDataset` + the `build_hf` task branch**

Per spec §5.4:
- `SemanticHFDataset` consumes an HF dataset exposing a per-pixel label-map feature (`label`/`annotation` image, e.g. `scene_parse_150`). It materializes the same `SemanticTarget` + §5.2 encoding as `mask_png`. If `data.semantic.class_map` is provided, use it via `build_value_to_label` (A3); else derive `class_names` from the HF label feature's `ClassLabel.names` (with explicit background removed per §4.5 — reuse the `_BACKGROUND_NAMES` set from `_semantic_encode`). Raise `ValueError` if `label_map_field is None`.
- In `hf.py`'s `build_hf` (`:363`): thread `task` (from the cfg dict — see step 5). If `task == "semantic"`, construct and return `SemanticHFDataset` (import locally to avoid a torch/datasets import cycle); else the existing `HFDataset`. **One registry key `hf`** — do NOT add a second `DataFormat` (spec §5.4 Decision).
- Force text-prompt mode `all` (§5.6) as in `mask_png`.

- [ ] **Step 5: Thread `task` into the `hf` builder (`train/runner.py`)**

In `train/runner.py:_build_dataset_from_dict` (`:65`), the builder is resolved via `lookup("dataset", cfg.data.format)`. Add `cfg.task` to the cfg dict passed to the builder (e.g. `data_dict["task"] = cfg.task`) so `build_hf` can branch. `mask_png` is unambiguous (semantic-only) and ignores `task`. Confirm `eval/runner.py:126` (which also calls `lookup("dataset", cfg.data.format)`) threads `task` the same way — do it in both call sites, or factor a tiny helper. (This is the same wiring Phase E's eval dispatch builds on; doing it here keeps the data layer self-contained.)

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_data_semantic_hf.py -v`
Expected: PASS.

- [ ] **Step 7: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/data/semantic_hf.py src/custom_sam_peft/data/hf.py src/custom_sam_peft/config/schema.py src/custom_sam_peft/train/runner.py
uv run ruff format --check src/custom_sam_peft/data/semantic_hf.py src/custom_sam_peft/data/hf.py src/custom_sam_peft/config/schema.py src/custom_sam_peft/train/runner.py
uv run mypy --strict src/custom_sam_peft/data/semantic_hf.py src/custom_sam_peft/data/hf.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/data/semantic_hf.py src/custom_sam_peft/data/hf.py src/custom_sam_peft/config/schema.py src/custom_sam_peft/train/runner.py tests/unit/test_data_semantic_hf.py
git commit -m "feat(#113): semantic HF adapter + build_hf task branch + HFFieldMap.label_map"
```

### Task A8: Phase A verification gate **[SERIALIZE — last in A]**

- [ ] **Step 1: Full CPU suite + lint/format/type across all Phase-A touched files**

```bash
uv run ruff check src/custom_sam_peft/config/schema.py src/custom_sam_peft/data/ 
uv run ruff format --check src/custom_sam_peft/config/schema.py src/custom_sam_peft/data/
uv run mypy --strict src/custom_sam_peft
uv run python -c "import custom_sam_peft"
uv run pytest -o "addopts=" tests/unit tests/config tests/data -q
```
Expected: all GREEN. Instance-path tests unchanged (the §12 invariance holds — `task` defaults to instance, `Example.instances` defaulting + `collate_batch`'s new key are additive).

- [ ] **Step 2: Confirm the Phase A interface contract is satisfied** (no commit — review checkpoint)

Verify: `SemanticTarget`, `Example.semantic`, `mask_png`/`hf`-semantic builders return `Dataset` with `class_names` of length K in §5.2 order, collate `"semantic"` key present. Restate the §5.7 contract (above) in the phase handoff.

---

## Phase B — semantic loss (§7)

**Feature block:** the `SemanticLossConfig` schema subtree, the **fully-enumerated** `SEMANTIC_PRESET_TABLE` (all 12 cells + microscopy=medical alias, every cell tagged), the `resolve`/`LOCKED_OFF`/`dump_semantic_loss_bundle` resolver mirroring #112, and the `SemanticLoss` nn.Module + `build_semantic_loss`. CPU-only.

**Consumes from Phase A:** `SemanticTarget` + the §5.2 encoding.

**Interface contract this phase PRODUCES (spec §7.4) — restate verbatim-faithful:**

> Exposes `SemanticLossConfig` schema (`preset`, `class_imbalance`, `overrides`, `background_logit`, `background_class_name`, `query_reduce`, `source`); `resolve(cfg) -> ResolvedSemanticLoss`, `SEMANTIC_PRESET_TABLE`, `LOCKED_OFF`, `dump_semantic_loss_bundle(cfg) -> dict` (torch-free); and `SemanticLoss.forward((B, K+1, H, W) logits, list[SemanticTarget]) -> {"ce": .., "region": .., "total": ..}` plus `build_semantic_loss(resolved) -> SemanticLoss`. Consumes Phase A's `SemanticTarget` + §5.2 encoding. Produces the loss-key set `{"ce", "region", "total"}` that Phase C's train branch logs. Fully CPU-testable.

**Schema-ordering note:** this phase edits `config/schema.py` (the `SemanticLossConfig` block + `TrainHyperparams.semantic_loss`). If Phase A is in the same session, land A's schema edits first, then B's — same file, serialize commits.

### Task B1: `SemanticLossConfig` / `SemanticLossOverrides` / `SemMaskFamily` schema **[SERIALIZE]**

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (`SemMaskFamily` near the loss literals `~:128`; the config classes; `TrainHyperparams.semantic_loss` at `~:621`; `__all__`)
- Test: `tests/config/test_semantic_loss_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_semantic_loss_config.py
"""SemanticLossConfig schema coverage (§7.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import SemanticLossConfig


def test_defaults():
    c = SemanticLossConfig()
    assert c.preset == "natural"
    assert c.class_imbalance == "balanced"
    assert c.background_logit == 0.0
    assert c.background_class_name is None
    assert c.query_reduce == "max"
    assert c.source == "marginalize"


def test_strict_extra_rejected():
    with pytest.raises(ValidationError):
        SemanticLossConfig(unknown_knob=1)


def test_query_reduce_literal():
    assert SemanticLossConfig(query_reduce="sum").query_reduce == "sum"
    with pytest.raises(ValidationError):
        SemanticLossConfig(query_reduce="mean")


def test_source_literal():
    assert SemanticLossConfig(source="semantic_seg").source == "semantic_seg"
    with pytest.raises(ValidationError):
        SemanticLossConfig(source="head")


def test_overrides_sem_family_literal():
    c = SemanticLossConfig(overrides={"sem_family": "focal_tversky"})
    assert c.overrides.sem_family == "focal_tversky"
    with pytest.raises(ValidationError):
        SemanticLossConfig(overrides={"sem_family": "bce"})  # not a SemMaskFamily
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/config/test_semantic_loss_config.py -v`
Expected: FAIL — `SemanticLossConfig` does not exist.

- [ ] **Step 3: Implement** (copy spec §7.2 verbatim, including tags)

```python
SemMaskFamily = Literal["ce_dice", "focal_dice", "focal_tversky", "boundary", "ce", "dice"]


class SemanticLossOverrides(_Strict):
    """Per-knob overrides; None -> inherit from (preset, class_imbalance)."""

    sem_family: SemMaskFamily | None = None
    w_ce: PositiveFloat | None = None
    w_region: PositiveFloat | None = None  # weight on the Dice/Tversky/Boundary term
    focal_gamma: PositiveFloat | None = None
    focal_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_gamma: PositiveFloat | None = None
    boundary_weight: float | None = Field(default=None, ge=0.0, le=1.0)


class SemanticLossConfig(_Strict):
    preset: Preset = "natural"  # reuse #112 Preset verbatim
    class_imbalance: ClassImbalance = "balanced"  # reuse #112 axis verbatim
    overrides: SemanticLossOverrides = Field(default_factory=SemanticLossOverrides)
    # --- argmax / background / reduction knobs (§4.5, §6.2) ---
    background_logit: float = 0.0  # cite: degenerate logit boundary (sigmoid(0)=0.5)
    background_class_name: str | None = None  # tbd: #113 — custom explicit-bg name
    query_reduce: Literal["max", "sum"] = "max"  # tbd: #113 — see §6.2
    source: Literal["marginalize", "semantic_seg"] = "marginalize"  # cite: §3.3 / OQ-1
```

Add `semantic_loss: SemanticLossConfig = Field(default_factory=SemanticLossConfig)` to `TrainHyperparams` (`~:621`, next to `loss`). Add `"SemanticLossConfig"`, `"SemanticLossOverrides"`, `"SemMaskFamily"` to `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/config/test_semantic_loss_config.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/config/schema.py
uv run ruff format --check src/custom_sam_peft/config/schema.py
uv run mypy --strict src/custom_sam_peft/config/schema.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/config/schema.py tests/config/test_semantic_loss_config.py
git commit -m "feat(#113): SemanticLossConfig schema subtree + TrainHyperparams.semantic_loss"
```

### Task B2: `SEMANTIC_PRESET_TABLE` + `resolve` + `LOCKED_OFF` + sidecar (torch-free) **[SERIALIZE — depends on B1]**

**Files:**
- Create: `src/custom_sam_peft/models/losses/semantic_presets.py`
- Test: `tests/unit/test_semantic_presets.py`, `tests/unit/test_data_import_boundary.py` (extend the existing import-boundary test to assert `semantic_presets` is torch-free)

This task **FULLY ENUMERATES** the table — all 12 `(preset, class_imbalance)` cells for natural/medical/satellite + the microscopy=medical alias. "Table complete and every cell tagged" is an explicit acceptance criterion (spec requirement #5).

- [ ] **Step 1: Write the failing tests** (including a completeness assertion)

```python
# tests/unit/test_semantic_presets.py
"""SEMANTIC_PRESET_TABLE completeness + resolve + override-WARN + sidecar (§7.3)."""

from __future__ import annotations

import logging

import pytest

from custom_sam_peft.config.schema import SemanticLossConfig
from custom_sam_peft.models.losses.semantic_presets import (
    SEMANTIC_PRESET_TABLE,
    dump_semantic_loss_bundle,
    resolve,
)

_REAL_PRESETS = ("natural", "medical", "satellite", "microscopy")
_IMBALANCE = ("balanced", "moderate", "severe")


def test_table_is_complete_all_16_cells():
    # 4 real presets (incl. microscopy alias) x 3 imbalance = 12 + microscopy(3) = 16 keys.
    for p in _REAL_PRESETS:
        for ci in _IMBALANCE:
            assert (p, ci) in SEMANTIC_PRESET_TABLE, f"missing cell ({p}, {ci})"
    assert len(SEMANTIC_PRESET_TABLE) == 12  # microscopy added by alias below 'natural/medical/satellite'? see note


def test_microscopy_is_alias_of_medical():
    for ci in _IMBALANCE:
        assert SEMANTIC_PRESET_TABLE[("microscopy", ci)] == SEMANTIC_PRESET_TABLE[("medical", ci)]


def test_every_cell_has_sem_family_and_weights():
    for key, cell in SEMANTIC_PRESET_TABLE.items():
        assert "sem_family" in cell, key
        assert "w_ce" in cell and "w_region" in cell, key


def test_resolve_natural_balanced_is_ce_dice_samed_weights():
    r = resolve(SemanticLossConfig(preset="natural", class_imbalance="balanced"))
    assert r.sem_family == "ce_dice"
    assert r.w_ce == 0.2 and r.w_region == 0.8  # SAMed (S)


def test_resolve_override_wins():
    r = resolve(
        SemanticLossConfig(
            preset="natural", class_imbalance="balanced",
            overrides={"sem_family": "boundary", "boundary_weight": 0.3},
        )
    )
    assert r.sem_family == "boundary"
    assert r.boundary_weight == 0.3


def test_locked_off_warns(caplog):
    # natural preset overriding to focal_tversky/boundary WARNs (§7.3).
    with caplog.at_level(logging.WARNING):
        resolve(
            SemanticLossConfig(
                preset="natural", class_imbalance="balanced",
                overrides={"sem_family": "focal_tversky"},
            )
        )
    assert any("override" in r.message.lower() for r in caplog.records)


def test_dump_sidecar_shape():
    bundle = dump_semantic_loss_bundle(SemanticLossConfig(preset="medical", class_imbalance="moderate"))
    assert bundle["preset"] == "medical"
    assert bundle["class_imbalance"] == "moderate"
    assert "resolved" in bundle and "sem_family" in bundle["resolved"]
    assert "term_classes" in bundle
    assert "library_version" in bundle
```

> **Cell-count note for the implementer:** mirror `presets.py` exactly — store the **12** cells for `natural`/`medical`/`satellite` × 3 imbalance literally, then add the 3 `microscopy` keys via `dict(SEMANTIC_PRESET_TABLE[("medical", ci)])` (alias). So `len(SEMANTIC_PRESET_TABLE) == 15` after the alias insert. **Fix `test_table_is_complete_all_16_cells`'s final assert to `== 15`** when you see the real count (the test's `_REAL_PRESETS` loop already covers all 4 presets × 3 = 12 required keys plus microscopy; the `len` check just pins no stray keys). Decide the exact number against `presets.py` (`PRESET_TABLE` has 12 stored + 3 microscopy = 15) and make the test match.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_presets.py -v`
Expected: FAIL — module/table do not exist.

- [ ] **Step 3: Implement the full table + resolver** (mirror `presets.py` structure exactly)

Create `src/custom_sam_peft/models/losses/semantic_presets.py`, torch-free. Use the spec §7.3 legend and representative cells, FULLY enumerating all 12 cells. Every cell carries a tag. The `0.2/0.8` CE/region split is `# cite: (S)` (SAMed) for every cell. Concrete table (resolve the spec's representative cells into all 12):

```python
"""Domain-aware SEMANTIC loss presets — torch-free resolver + run-metadata helpers.

Mirrors models/losses/presets.py. Pure-Python (no torch) so `csp doctor` and
schema tests import it without dragging torch in.

Citation legend (spec §7.3):
  (S) SAMed (Zhang & Liu 2023, arXiv:2304.13785) §3.3 — CE/region = 0.2/0.8.
  (C) Lin et al. 2017 (focal) — γ=2.0, α=0.25.
  (D) Abraham & Khan 2019 (Focal-Tversky) — γ=0.75 best on ISIC.
  (E) Salehi et al. 2017 (Tversky) — β/α=0.7 (FN weight).
  (H) Kervadec et al. 2019 (boundary) — blend ~0.2.
  (F) degenerate identity (α=0.5 → Dice; γ=1.0 → Tversky).
  (G) alias-of-medical (microscopy copies medical).
γ escalations beyond a cited source -> `# tbd: #191`; unsourced tversky_alpha -> `# tbd: #191`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from custom_sam_peft._version import __version__ as _LIB_VERSION
except ImportError:
    _LIB_VERSION = "unknown"
from custom_sam_peft.config.schema import (
    ClassImbalance,
    Preset,
    SemanticLossConfig,
)

_LOG = logging.getLogger(__name__)

# Region-term class names per sem_family (avoids importing semantic_compose here).
# Kept in lockstep with semantic_compose via a sync-check test (B3).
_SEM_TERM_CLASS_NAMES: dict[str, str] = {
    "ce_dice": "SemCEDiceLoss",
    "focal_dice": "SemFocalDiceLoss",
    "focal_tversky": "SemFocalTverskyLoss",
    "boundary": "SemBoundaryLoss",
    "ce": "SemCELoss",
    "dice": "SemDiceLoss",
}

SEMANTIC_PRESET_TABLE: dict[tuple[Preset, ClassImbalance], dict[str, Any]] = {
    # ----- natural -----
    ("natural", "balanced"): {
        "sem_family": "ce_dice",  # cite: (S)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("natural", "moderate"): {
        "sem_family": "focal_dice",  # cite: (S,C)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("natural", "severe"): {
        "sem_family": "focal_dice",  # cite: (S)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 3.0,  # tbd: #191
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    # ----- medical -----
    ("medical", "balanced"): {
        "sem_family": "focal_dice",  # cite: (S,C)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("medical", "moderate"): {
        "sem_family": "focal_tversky",  # cite: (S,E,D)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.7,  # cite: (E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("medical", "severe"): {
        "sem_family": "boundary",  # cite: (S,H)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.7,  # cite: (E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.2,  # cite: (H)
    },
    # ----- satellite -----
    ("satellite", "balanced"): {
        "sem_family": "ce_dice",  # cite: (S)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.0,  # cite: (F)
    },
    ("satellite", "moderate"): {
        "sem_family": "boundary",  # cite: (S,H)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.5,  # cite: (F)
        "tversky_gamma": 1.0,  # cite: (F)
        "boundary_weight": 0.2,  # cite: (H)
    },
    ("satellite", "severe"): {
        "sem_family": "focal_tversky",  # cite: (S,E,D)
        "w_ce": 0.2,  # cite: (S)
        "w_region": 0.8,  # cite: (S)
        "focal_gamma": 2.0,  # cite: (C)
        "focal_alpha": 0.25,  # cite: (C)
        "tversky_alpha": 0.7,  # cite: (E)
        "tversky_gamma": 0.75,  # cite: (D)
        "boundary_weight": 0.0,  # cite: (F)
    },
}

# Microscopy = strict alias of medical (§7.3 (G)).
SEMANTIC_PRESET_TABLE[("microscopy", "balanced")] = dict(SEMANTIC_PRESET_TABLE[("medical", "balanced")])  # cite: (G)
SEMANTIC_PRESET_TABLE[("microscopy", "moderate")] = dict(SEMANTIC_PRESET_TABLE[("medical", "moderate")])  # cite: (G)
SEMANTIC_PRESET_TABLE[("microscopy", "severe")] = dict(SEMANTIC_PRESET_TABLE[("medical", "severe")])  # cite: (G)


LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "sem_family": (
            "the medical preset chose focal/tversky/boundary to handle rare positives; "
            "overriding to ce or dice may underweight them"
        ),
    },
    "natural": {
        "sem_family": (
            "the natural preset chose ce_dice/focal_dice; overriding to focal_tversky or "
            "boundary is unusual for balanced natural-image data"
        ),
    },
}


@dataclass(frozen=True)
class ResolvedSemanticLoss:
    sem_family: str
    w_ce: float
    w_region: float
    focal_gamma: float
    focal_alpha: float
    tversky_alpha: float
    tversky_gamma: float
    boundary_weight: float


def _override_triggers_warn(field_name, value, preset, class_imbalance) -> bool:
    if preset not in LOCKED_OFF or field_name not in LOCKED_OFF[preset] or value is None:
        return False
    seed = SEMANTIC_PRESET_TABLE[(preset, class_imbalance)][field_name]
    return bool(value != seed)


def resolve(cfg: SemanticLossConfig) -> ResolvedSemanticLoss:
    base = dict(SEMANTIC_PRESET_TABLE[(cfg.preset, cfg.class_imbalance)])
    ov = cfg.overrides.model_dump(exclude_unset=False)
    for fname, override in ov.items():
        if override is None:
            continue
        if _override_triggers_warn(fname, override, cfg.preset, cfg.class_imbalance):
            _LOG.warning(
                "You overrode %s=%s under preset=%s; %s. The override will be applied as-is.",
                fname, override, cfg.preset, LOCKED_OFF[cfg.preset][fname],
            )
        base[fname] = override
    return ResolvedSemanticLoss(**base)


def dump_semantic_loss_bundle(cfg: SemanticLossConfig) -> dict[str, Any]:
    resolved = resolve(cfg)
    return {
        "preset": cfg.preset,
        "class_imbalance": cfg.class_imbalance,
        "resolved": {
            "sem_family": resolved.sem_family,
            "w_ce": resolved.w_ce,
            "w_region": resolved.w_region,
            "focal_gamma": resolved.focal_gamma,
            "focal_alpha": resolved.focal_alpha,
            "tversky_alpha": resolved.tversky_alpha,
            "tversky_gamma": resolved.tversky_gamma,
            "boundary_weight": resolved.boundary_weight,
        },
        "term_classes": {"region": _SEM_TERM_CLASS_NAMES[resolved.sem_family]},
        "source": cfg.source,
        "query_reduce": cfg.query_reduce,
        "background_logit": cfg.background_logit,
        "library_version": _LIB_VERSION or "unknown",
    }
```

> **Preset/Preset-literal caveat:** the #112 `Preset` Literal includes `"none"` and `"custom"`; semantic does NOT use those branches (no legacy default). `resolve` indexes `SEMANTIC_PRESET_TABLE[(preset, ci)]` directly — `none`/`custom` are not stored. If you want to be defensive, raise a clear `ValueError` for `preset in ("none", "custom")` (semantic has no legacy default), or document that the schema default keeps them out. Pin this with a test if you add the guard.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_presets.py -v`
Expected: PASS (fix the `len` assert to the real count first if it trips).

- [ ] **Step 5: torch-free import-boundary assertion**

Extend `tests/unit/test_data_import_boundary.py` (or the nearest import-boundary test) to assert `semantic_presets` imports without torch — mirror however the existing test checks `presets.py`. Example:

```python
def test_semantic_presets_is_torch_free():
    import sys
    sys.modules.pop("torch", None)  # or use the project's existing torch-absence harness
    import importlib
    importlib.import_module("custom_sam_peft.models.losses.semantic_presets")
    # If the project has a helper asserting "torch not imported", use it instead.
```

> Use the project's existing pattern for this check (read `test_data_import_boundary.py` / `test_loss_presets.py` to see how `presets.py`'s torch-freeness is asserted, and copy it).

- [ ] **Step 6: Lint/type + commit**

```bash
uv run ruff check src/custom_sam_peft/models/losses/semantic_presets.py
uv run ruff format --check src/custom_sam_peft/models/losses/semantic_presets.py
uv run mypy --strict src/custom_sam_peft/models/losses/semantic_presets.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/models/losses/semantic_presets.py tests/unit/test_semantic_presets.py tests/unit/test_data_import_boundary.py
git commit -m "feat(#113): SEMANTIC_PRESET_TABLE (all 12 cells + microscopy alias) + resolver + sidecar"
```

### Task B3: `SemanticLoss` term + `build_semantic_loss` (`semantic_compose.py`) **[SERIALIZE — depends on B2]**

**Files:**
- Create: `src/custom_sam_peft/models/losses/semantic_compose.py`
- Modify: `src/custom_sam_peft/models/losses/__init__.py` (export the public symbols)
- Test: `tests/unit/test_semantic_loss.py`

Reuse the per-pixel math in `terms/mask.py` (`_dice` `:35`, `_tversky_index` `:124`, `_focal_bce_per_pixel` `:52`, `_kervadec_boundary` `:172`) by calling them per class over `(B, H, W)` slices (the helpers already mean-reduce). See spec §7.4.

- [ ] **Step 1: Write the failing tests** (shapes, ignore_index exclusion, degenerate identities, sync test)

```python
# tests/unit/test_semantic_loss.py
"""SemanticLoss.forward shapes + ignore_index + degenerate identities (§7.4)."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import SemanticLossConfig
from custom_sam_peft.data.base import SemanticTarget
from custom_sam_peft.models.losses.semantic_compose import build_semantic_loss
from custom_sam_peft.models.losses.semantic_presets import (
    _SEM_TERM_CLASS_NAMES,
    resolve,
)


def _loss(preset="natural", ci="balanced", **ov):
    cfg = SemanticLossConfig(preset=preset, class_imbalance=ci, overrides=ov or {})
    return build_semantic_loss(resolve(cfg))


def test_forward_returns_ce_region_total():
    loss = _loss()
    B, K, H, W = 2, 3, 16, 16
    logits = torch.randn(B, K + 1, H, W, requires_grad=True)
    tgts = [
        SemanticTarget(torch.randint(0, K + 1, (H, W), dtype=torch.int64), ignore_index=255)
        for _ in range(B)
    ]
    out = loss(logits, tgts)
    assert set(out.keys()) == {"ce", "region", "total"}
    out["total"].backward()  # gradients flow
    assert logits.grad is not None


def test_fully_ignored_image_finite_loss():
    loss = _loss()
    B, K, H, W = 1, 2, 8, 8
    logits = torch.randn(B, K + 1, H, W, requires_grad=True)
    labels = torch.full((H, W), 255, dtype=torch.int64)  # all void
    out = loss(logits, [SemanticTarget(labels, ignore_index=255)])
    assert torch.isfinite(out["total"])


def test_gt_downsampled_to_logit_res_nearest():
    # GT at full res (32) vs logits at 16 -> loss downsamples GT, no crash, finite.
    loss = _loss()
    logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    labels = torch.randint(0, 3, (32, 32), dtype=torch.int64)
    out = loss(logits, [SemanticTarget(labels, ignore_index=255)])
    assert torch.isfinite(out["total"])


def test_sem_term_class_names_match_compose_registry():
    # §7.3 sync test: every sem_family resolves to a known compose term.
    from custom_sam_peft.models.losses.semantic_compose import SEM_FAMILY_BUILDERS

    assert set(_SEM_TERM_CLASS_NAMES.keys()) == set(SEM_FAMILY_BUILDERS.keys())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_loss.py -v`
Expected: FAIL — `semantic_compose` does not exist.

- [ ] **Step 3: Implement `SemanticLoss` + `build_semantic_loss`** (spec §7.4)

Key contract points (copy verbatim from §7.4):
- `SemanticLoss(torch.nn.Module).forward(sem_logits: Tensor, targets: list[SemanticTarget]) -> dict[str, Tensor]` returning `{"ce", "region", "total"}` with `total = w_ce·ce + w_region·region`.
- `sem_logits`: `(B, K+1, H_l, W_l)`. `targets[b].labels`: `(H_g, W_g)` int64. **Downsample GT** to `(H_l, W_l)` with **nearest** (`F.interpolate(labels[None,None].float(), size=(H_l,W_l), mode="nearest").long()`) — never bilinear (`# cite: standard seg-loss practice`).
- `ignore_index` plumbed through EVERY reduction: `F.cross_entropy(logits, labels, ignore_index=ii)`; the region terms build `valid = labels != ii`, and zero both pred and one-hot at void pixels before per-class Dice/Tversky/Boundary.
- Region terms operate per class `c ∈ {0..K}`: `pred_c = softmax(sem_logits, dim=1)[:, c]`, `tgt_c = (labels == c) & valid`, call the `terms/mask.py` helpers on `(B, H, W)` per class, mean over classes. **Background channel (0) is INCLUDED in both CE and region** (§7.4 decision: it is a real argmax class).
- `SEM_FAMILY_BUILDERS: dict[str, callable]` maps each `sem_family` to a region-term assembler (this is what the B3 sync test checks against `_SEM_TERM_CLASS_NAMES`). Families per spec §7.3 table:
  - `ce_dice` → `w_ce·CE + w_region·Dice`
  - `focal_dice` → `w_ce·FocalCE + w_region·Dice`
  - `focal_tversky` → `w_ce·FocalCE + w_region·FocalTversky`
  - `boundary` → `w_ce·CE + w_region·(boundary_weight·Kervadec + (1-bw)·Dice)`
  - `ce` → `CE` only (region = 0)
  - `dice` → `Dice` only (ce = 0)
- `CE` = `F.cross_entropy(logits, labels, ignore_index=ii)`. `FocalCE` = multi-class focal CE (γ, α) — generalize `_focal_bce_per_pixel` to multi-class, or compute `(1-p_t)^γ · CE` over the softmax. Reuse `_dice`/`_tversky_index`/`_kervadec_boundary` per class.
- `build_semantic_loss(resolved: ResolvedSemanticLoss) -> SemanticLoss` mirrors `build_loss_bundle` (`compose.py:173`) — read the resolved knobs, build the term once, store on the module.
- **No `assert` in src** — narrow shapes with `if logits.ndim != 4: raise ValueError(...)`.

- [ ] **Step 4: Export public symbols**

In `src/custom_sam_peft/models/losses/__init__.py`, export `SemanticLoss`, `build_semantic_loss`, `resolve` (as `resolve_semantic` to avoid clashing with the instance `resolve` — check the existing `__init__` exports and namespace carefully; the instance `resolve` is already exported, so alias the semantic one), `dump_semantic_loss_bundle`, `SEMANTIC_PRESET_TABLE`. Read the current `__init__.py` to match its export style and avoid a name collision with the instance `resolve`/`dump_loss_bundle`.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_loss.py -v`
Expected: PASS (all 4).

- [ ] **Step 6: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/models/losses/semantic_compose.py src/custom_sam_peft/models/losses/__init__.py
uv run ruff format --check src/custom_sam_peft/models/losses/semantic_compose.py src/custom_sam_peft/models/losses/__init__.py
uv run mypy --strict src/custom_sam_peft/models/losses/semantic_compose.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/models/losses/semantic_compose.py src/custom_sam_peft/models/losses/__init__.py tests/unit/test_semantic_loss.py
git commit -m "feat(#113): SemanticLoss multi-class term + build_semantic_loss + compose registry"
```

### Task B4: Phase B verification gate **[SERIALIZE — last in B]**

- [ ] **Step 1: Full CPU loss/config suite + lint/format/type**

```bash
uv run ruff check src/custom_sam_peft/models/losses/ src/custom_sam_peft/config/schema.py
uv run ruff format --check src/custom_sam_peft/models/losses/ src/custom_sam_peft/config/schema.py
uv run mypy --strict src/custom_sam_peft
uv run python -c "import custom_sam_peft"
uv run pytest -o "addopts=" tests/unit tests/config -q
```
Expected: GREEN. Restate the §7.4 contract in the phase handoff. Confirm the table-completeness + every-cell-tagged acceptance criterion holds (grep the new file for any line lacking a `# cite:`/`# tbd:` on a default value).

---

## Phase C — forward / marginalization + train branch (§6, §10.1)

**Feature block:** `models/semantic.py` (`marginalize_group` + the `(B, K+1, H, W)` builder + argmax), the `train_step` semantic branch (assemble-then-loss topology), and the parametrized `StepResult`/`_ScalarWindow` loss-key set. CPU-stub tested exhaustively + one GPU smoke.

**Consumes from Phase A:** `class_names` order (§5.2), `SemanticTarget`, the SAM3 output dict (§3.2). **From Phase B:** `SemanticLoss` + `build_semantic_loss` + `resolve`.

**Interface contract this phase PRODUCES (spec §6.5) — restate verbatim-faithful:**

> Exposes `marginalize_group(outputs: dict[str, Tensor], b: int, k: int, *, query_reduce: str, source: str) -> Tensor` returning `(b, k, H, W)` per-concept foreground **logits** for one group; `build_semantic_logits(group_logit_slices, *, background_logit) -> (B, K+1, H, W)` (background-prepend); and `semantic_argmax(sem_logits) -> (B, H, W) int64`. Consumes Phase A's `class_names` order (§5.2) and the SAM3 output dict (§3.2). Produces the `(B, K+1, H, W)` tensor the Phase B loss consumes. The `train_step` semantic branch assembles graph-connected per-group slices and computes ONE semantic loss. CPU-testable against `tiny_sam3_stub` outputs.

### Task C1: `marginalize_group` (`models/semantic.py`) **[SERIALIZE — heart of Phase C]**

**Files:**
- Create: `src/custom_sam_peft/models/semantic.py`
- Test: `tests/unit/test_marginalize.py`

Spec §6.2/§6.3/§6.4. `marginalize_group` consumes only `pred_logits` `(N,Q,1)`, `pred_masks` `(N,Q,H,W)`, `presence_logit_dec` `(N,1)` (and `semantic_seg` `(N,1,H,W)` for the `source="semantic_seg"` path). `N = b·k`, column `n` = (image `n//k`, concept `n%k`).

- [ ] **Step 1: Write the failing tests** (both `query_reduce` modes, both `source` modes, background-prepend → argmax)

```python
# tests/unit/test_marginalize.py
"""marginalize_group + build_semantic_logits + semantic_argmax (§6)."""

from __future__ import annotations

import torch

from custom_sam_peft.models.semantic import (
    build_semantic_logits,
    marginalize_group,
    semantic_argmax,
)


def _stub_outputs(b, k, q=4, h=8, w=8):
    n = b * k
    return {
        "pred_logits": torch.randn(n, q, 1),
        "pred_masks": torch.randn(n, q, h, w),
        "presence_logit_dec": torch.randn(n, 1),
        "semantic_seg": torch.randn(n, 1, h, w),
    }


def test_marginalize_max_shape():
    out = _stub_outputs(2, 3)
    fg = marginalize_group(out, 2, 3, query_reduce="max", source="marginalize")
    assert fg.shape == (2, 3, 8, 8)  # (b, k, H, W) per-concept LOGITS


def test_marginalize_sum_shape():
    out = _stub_outputs(2, 3)
    fg = marginalize_group(out, 2, 3, query_reduce="sum", source="marginalize")
    assert fg.shape == (2, 3, 8, 8)


def test_semantic_seg_source_shape():
    out = _stub_outputs(2, 3)
    fg = marginalize_group(out, 2, 3, query_reduce="max", source="semantic_seg")
    assert fg.shape == (2, 3, 8, 8)  # surfaced directly from out["semantic_seg"]


def test_build_semantic_logits_prepends_background():
    # one group covering all K=3 concepts; b=2.
    fg = torch.randn(2, 3, 8, 8)
    sem = build_semantic_logits([fg], background_logit=0.0)
    assert sem.shape == (2, 4, 8, 8)  # K+1 channels
    assert torch.allclose(sem[:, 0], torch.zeros(2, 8, 8))  # bg channel == background_logit


def test_argmax_background_wins_when_all_fg_negative():
    # All concept logits very negative -> argmax picks bg channel 0.
    fg = torch.full((1, 2, 4, 4), -10.0)
    sem = build_semantic_logits([fg], background_logit=0.0)
    labels = semantic_argmax(sem)
    assert labels.shape == (1, 4, 4)
    assert torch.all(labels == 0)


def test_argmax_concept_wins_when_fg_high():
    fg = torch.full((1, 2, 4, 4), -10.0)
    fg[:, 1] = 10.0  # concept 1 (channel 2) dominates
    sem = build_semantic_logits([fg], background_logit=0.0)
    labels = semantic_argmax(sem)
    assert torch.all(labels == 2)  # channel index 2 == concept dense_id 1 + 1


def test_multigroup_concat_preserves_concept_order():
    # two groups of 2 concepts each -> K=4 total, concat along concept axis.
    g0 = torch.randn(1, 2, 4, 4)
    g1 = torch.randn(1, 2, 4, 4)
    sem = build_semantic_logits([g0, g1], background_logit=0.0)
    assert sem.shape == (1, 5, 4, 4)  # 4 concepts + bg
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_marginalize.py -v`
Expected: FAIL — `models/semantic.py` does not exist.

- [ ] **Step 3: Implement** (spec §6.2/§6.3/§6.4)

```python
# src/custom_sam_peft/models/semantic.py
"""Head-free semantic marginalization over the SAM3 grounding output dict (§6).

Pure functions; the model itself is unchanged. Consumes only keys the existing
forward already produces and validates: pred_logits, pred_masks,
presence_logit_dec (+ semantic_seg for the opt-in source path).
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-6


def marginalize_group(
    outputs: dict[str, Tensor],
    b: int,
    k: int,
    *,
    query_reduce: str,
    source: str,
) -> Tensor:
    """(N=b*k columns) -> (b, k, H, W) per-concept foreground LOGITS for this group.

    column n corresponds to image n//k and concept n%k (image-major / class-minor).
    """
    if source == "semantic_seg":
        # surfaced single-channel foreground map (§6.4); (N,1,H,W) -> (b,k,H,W).
        seg = outputs["semantic_seg"]  # (N, 1, H, W)
        n, _, h, w = seg.shape
        return seg.reshape(b, k, h, w)

    pred_logits = outputs["pred_logits"]  # (N, Q, 1)
    pred_masks = outputs["pred_masks"]  # (N, Q, H, W)
    presence = outputs["presence_logit_dec"]  # (N, 1)
    n, q, h, w = pred_masks.shape

    obj_q = torch.sigmoid(pred_logits[..., 0])  # (N, Q)
    mask_q = torch.sigmoid(pred_masks)  # (N, Q, H, W)
    pres = torch.sigmoid(presence[:, 0])  # (N,)

    weighted = obj_q[:, :, None, None] * mask_q  # (N, Q, H, W)
    if query_reduce == "max":
        fg = weighted.amax(dim=1)  # (N, H, W) in [0,1]
    elif query_reduce == "sum":
        fg = weighted.sum(dim=1)  # (N, H, W) in [0, +)
    else:
        raise ValueError(f"unknown query_reduce: {query_reduce!r}")
    fg = pres[:, None, None] * fg  # gate by presence
    fg = fg.clamp(_EPS, 1.0 - _EPS)
    fg_logits = torch.log(fg) - torch.log1p(-fg)  # logit(fg)
    return fg_logits.reshape(b, k, h, w)


def build_semantic_logits(
    group_logit_slices: list[Tensor],
    *,
    background_logit: float,
) -> Tensor:
    """Concat per-group (b, k_g, H, W) slices along concept axis, prepend bg -> (B, K+1, H, W)."""
    concept = torch.cat(group_logit_slices, dim=1)  # (B, K, H, W)
    b, _, h, w = concept.shape
    bg = torch.full((b, 1, h, w), float(background_logit), device=concept.device, dtype=concept.dtype)
    return torch.cat([bg, concept], dim=1)  # (B, K+1, H, W), channel 0 = background


def semantic_argmax(sem_logits: Tensor) -> Tensor:
    """(B, K+1, H, W) -> (B, H, W) int64 in {0..K}; channel 0 == background."""
    return sem_logits.argmax(dim=1)
```

> §6.3 note for eval/predict (Task D2/E2): upsample `sem_logits` bilinearly to GT/original resolution BEFORE argmax (matches `eval/postprocess.py:33` `_upsample_mask_logits`). `semantic_argmax` itself is resolution-agnostic.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_marginalize.py -v`
Expected: PASS (all 8).

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check src/custom_sam_peft/models/semantic.py
uv run ruff format --check src/custom_sam_peft/models/semantic.py
uv run mypy --strict src/custom_sam_peft/models/semantic.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/models/semantic.py tests/unit/test_marginalize.py
git commit -m "feat(#113): marginalize_group + (B,K+1,H,W) builder + semantic_argmax (head-free)"
```

### Task C2: Parametrized `StepResult` / `_ScalarWindow` loss-key set **[BLAST-RADIUS — depends on nothing in C; do before C3]**

**Files:**
- Modify: `src/custom_sam_peft/train/loop.py` (`StepResult.empty` at `:188`, `_ScalarWindow` at `:457`)
- Test: `tests/unit/test_train_loop_keys.py` (or wherever `StepResult`/`_ScalarWindow` are tested — `grep -rln "StepResult.empty\|_ScalarWindow" tests/`)

Spec §10.1: generalize the hardcoded `{"mask","box","obj","presence","total"}` to the **task's loss-key set**, defaulting to the instance keys (instance logging byte-identical).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_train_loop_keys.py
"""StepResult/_ScalarWindow parametrized loss-key set (§10.1)."""

from __future__ import annotations

from custom_sam_peft.train.loop import StepResult, _ScalarWindow

INSTANCE_KEYS = ("mask", "box", "obj", "presence", "total")
SEMANTIC_KEYS = ("ce", "region", "total")


def test_step_result_empty_defaults_to_instance_keys():
    r = StepResult.empty()
    assert set(r.losses.keys()) == set(INSTANCE_KEYS)  # byte-identical instance default


def test_step_result_empty_accepts_semantic_keys():
    r = StepResult.empty(loss_keys=SEMANTIC_KEYS)
    assert set(r.losses.keys()) == set(SEMANTIC_KEYS)


def test_scalar_window_instance_keys_unchanged():
    w = _ScalarWindow()  # default
    out = w.flush()
    # instance loss/* keys present exactly as today
    assert "loss/mask" in out and "loss/total" in out


def test_scalar_window_semantic_keys():
    w = _ScalarWindow(loss_keys=SEMANTIC_KEYS)
    r = StepResult(
        losses={"ce": 1.0, "region": 2.0, "total": 3.0},
        n_classes=3, grad_norm=0.5, skipped=False, nan_streak=0, images_processed=2,
    )
    w.update(r, lr=1e-4)
    out = w.flush()
    assert "loss/ce" in out and "loss/region" in out and "loss/total" in out
    assert "loss/mask" not in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_loop_keys.py -v`
Expected: FAIL — `StepResult.empty(loss_keys=...)` / `_ScalarWindow(loss_keys=...)` not supported.

- [ ] **Step 3: Implement the parametrization**

In `train/loop.py`:
- `StepResult.empty(cls, nan_streak=0, loss_keys: tuple[str, ...] | None = None)`: default `loss_keys` to the instance tuple `("mask","box","obj","presence","total")`; build `losses={k: 0.0 for k in loss_keys}`. Instance callers (no `loss_keys`) get the byte-identical default.
- `_ScalarWindow`: add a `loss_keys: tuple[str, ...]` field (default the instance tuple). Build `sums` generically: `{f"loss/{k}": 0.0 for k in loss_keys} | {"throughput/img_s": 0.0, "grad_norm": 0.0}`. `update` iterates `self.loss_keys` (`self.sums[f"loss/{k}"] += r.losses[k]`). `flush` emits `loss/<k>` for each key generically. Preserve `self.__init__()` reset behavior — it must re-seed with the same `loss_keys` (store `loss_keys` and pass it through the reset, or reset `sums` explicitly rather than via `__init__`).

> **Care:** `_ScalarWindow.flush` currently calls `self.__init__()` to reset (`:507`). With a `loss_keys` field this must re-seed the same keys. Either store `loss_keys` and call `self.__init__(loss_keys=self.loss_keys)`, or refactor the reset to not go through `__init__`. Pin the reset behavior with a test (flush twice, second flush still has the right keys).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_loop_keys.py -v`
Expected: PASS.

- [ ] **Step 5 [BLAST-RADIUS]: grep + full CPU suite**

```bash
grep -rn "StepResult.empty\|_ScalarWindow\|loss/mask\|losses\[\"mask\"\]\|losses\[\"total\"\]" src/ tests/
uv run pytest -o "addopts=" tests/unit tests/config tests/data tests/cli tests/eval tests/train tests/predict -q
```
Expected: instance tests asserting exact `StepResult.empty()` keys still pass (default unchanged); any test asserting the hardcoded list is updated to the parametrized form; full CPU suite GREEN.

- [ ] **Step 6: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/train/loop.py
uv run ruff format --check src/custom_sam_peft/train/loop.py
uv run mypy --strict src/custom_sam_peft/train/loop.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/train/loop.py tests/unit/test_train_loop_keys.py
git commit -m "feat(#113): parametrize StepResult/_ScalarWindow loss-key set (instance default unchanged)"
```

### Task C3: `train_step` semantic branch (`loop.py`) **[SERIALIZE — depends on C1, C2, B3, A]**

**Files:**
- Modify: `src/custom_sam_peft/train/loop.py` (`train_step` at `:212`)
- Modify: `src/custom_sam_peft/train/trainer.py` (build/pass the semantic loss + key set; semantic sidecar)
- Test: `tests/unit/test_train_step_semantic.py`

Spec §10.1 — the one place loop topology genuinely differs: instance sums **per-group-independent** losses; semantic must **assemble graph-connected per-group concept-logit slices and compute ONE loss** over the `(B, K+1, H, W)` stack.

- [ ] **Step 1: Write the failing test** (CPU stub model returning fixed output dicts)

```python
# tests/unit/test_train_step_semantic.py
"""train_step semantic branch with a CPU stub model (§10.1)."""

from __future__ import annotations

import torch

from custom_sam_peft.data.base import SemanticTarget, TextPrompts
from custom_sam_peft.train.loop import train_step
# Reuse the project's tiny stub + a minimal semantic TrainConfig builder. Read
# tests/fixtures/tiny_sam3_stub.py and an existing instance train_step test to
# mirror the model/optimizer/scheduler/cfg wiring exactly.


def test_semantic_train_step_produces_ce_region_total(make_semantic_cfg, stub_model):
    # make_semantic_cfg / stub_model are fixtures the implementer adds (see note).
    B, K, H, W = 2, 3, 16, 16
    batch = {
        "images": torch.zeros(B, 3, H, W),
        "image_ids": ["a", "b"],
        "prompts": [TextPrompts(["road", "tree", "car"]) for _ in range(B)],
        "instances": [[], []],
        "semantic": [
            SemanticTarget(torch.randint(0, K + 1, (H, W), dtype=torch.int64), 255)
            for _ in range(B)
        ],
    }
    cfg = make_semantic_cfg(class_names=["road", "tree", "car"])
    model, opt, sched = stub_model(cfg)
    r = train_step(model, batch, opt, sched, cfg, cfg_class_names := ["road", "tree", "car"],
                   global_step=0, nan_streak=0)
    assert set(r.losses.keys()) == {"ce", "region", "total"}
    assert not r.skipped
```

> **Fixture note:** the implementer must add `make_semantic_cfg` + `stub_model` fixtures (or inline construction) by reading an existing instance `train_step` test (`grep -rln "train_step(" tests/`) to copy the exact `TrainConfig` + `Sam3Wrapper`-wrapped `TinySam3Stub` + optimizer/scheduler wiring, then flip `task="semantic"`, add `data.semantic`, and set `train.semantic_loss`. The stub's output dict (`tiny_sam3_stub.py`) already matches §3.2.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_step_semantic.py -v`
Expected: FAIL — `train_step` has no semantic branch (it reads `batch["instances"]` and runs the instance loss).

- [ ] **Step 3: Implement the semantic branch**

In `train_step` (`:212`), branch on `cfg.task` (spec §10.1):
- **Shared (unchanged):** device moves, `classes_in_batch` collection, the `while True` K-replay loop, `_chunked` grouping, per-group `_autocast_ctx`, the OOM ladder, NaN-skip, grad-accum, `clip_grad_norm_`, scheduler step.
- **Branched (semantic):**
  - For each group, build `prompts_g = [TextPrompts(classes=list(group)) for _ in range(B)]` (full vocab, NO per-class target gather — no Hungarian matching).
  - Run the forward → `outputs`; call `marginalize_group(outputs, B, len(group), query_reduce=cfg.train.semantic_loss.query_reduce, source=cfg.train.semantic_loss.source)` → `(B, k_g, H, W)`. **Hold the graph-connected slice in a list** (do NOT detach — gradients are needed).
  - After the last group: `sem_logits = build_semantic_logits(slices, background_logit=cfg.train.semantic_loss.background_logit)` → `(B, K+1, H, W)`; compute ONE `semantic_loss(sem_logits, batch["semantic"])` → `{"ce","region","total"}`; backward once.
  - **OOM ladder interaction (§10.1):** on a K-rung shrink, the assembled stack is rebuilt from the recomputed groups (the slices are re-collected on replay; the existing `while True` replay handles this). Memory: K+1 channels at 288²×B is small (≪ per-query mask tensors), so holding graph-connected slices across groups is acceptable.
  - Return `StepResult(losses={"ce": ..., "region": ..., "total": ...}, n_classes=K, grad_norm=..., skipped=False, nan_streak=..., images_processed=B)`. Note: `StepResult` itself has **no** `loss_keys` field — its `losses` dict simply carries the semantic keys. `loss_keys` is only a parameter of `StepResult.empty(...)` (the skip path) and a field of `_ScalarWindow` (Task C2). On a skipped semantic step return `StepResult.empty(nan_streak=..., loss_keys=("ce","region","total"))`.
- **Instance branch:** the existing code path, untouched.

In `train/trainer.py`:
- Resolve the task's loss-key set (`("ce","region","total")` for semantic, the instance tuple otherwise) and thread it to `StepResult.empty(loss_keys=...)` / `_ScalarWindow(loss_keys=...)` (the trainer knows `cfg.task`).
- Build the semantic loss once via `build_semantic_loss(resolve(cfg.train.semantic_loss))` when `cfg.task == "semantic"` and pass it into the step (mirror how the instance `LossBundle` is built + passed).
- **Sidecar:** under semantic, write `run_dir/semantic_loss_bundle.json` via `dump_semantic_loss_bundle(cfg.train.semantic_loss)` (mirror the instance `loss_bundle.json` write). `task` already round-trips into `run_dir/config.yaml` (the schema field is persisted).

> Read `train/trainer.py` to find exactly where the instance `LossBundle` is built and passed to `train_step`, and where `loss_bundle.json` is written, then mirror for semantic. Keep instance behavior byte-identical.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_train_step_semantic.py -v`
Expected: PASS.

- [ ] **Step 5: Instance regression — confirm the instance `train_step` is unchanged**

Run the existing instance train_step tests:
```bash
uv run pytest -o "addopts=" tests/unit tests/train -k "train_step or loop" -q
```
Expected: GREEN (instance branch byte-identical).

- [ ] **Step 6: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py
uv run ruff format --check src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py
uv run mypy --strict src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/unit/test_train_step_semantic.py
git commit -m "feat(#113): train_step semantic branch (assemble-then-loss) + semantic sidecar"
```

### Task C4: GPU smoke — semantic `train_step` on the real wrapper **[SERIALIZE — last in C]**

**Files:**
- Create: `tests/gpu/test_semantic_train_step_gpu.py`
- Test: itself (GPU, gated)

Spec §11 GPU + §14 Risk (K=16 headroom). One end-to-end semantic `train_step` on the real SAM3 wrapper over a 2-image, 3-class synthetic mask_png fixture → finite loss + a backward + one optimizer step. Add a K=16 variant (or parametrize) to confirm the assembled-stack memory headroom on the 5070 Ti.

- [ ] **Step 1: Write the GPU test** (gated by `requires_checkpoint` + `requires_compatible_gpu`)

Mirror `tests/gpu/test_real_train_overfits.py` for the model/wrapper/optimizer wiring (read it first). Build a tiny mask_png fixture in a `tmp_path` (2 images + 2 label PNGs + class_map.json with 3 classes), construct a semantic `TrainConfig`, run one `train_step`, assert `torch.isfinite(loss)`, a non-None `grad_norm`, and `not skipped`. Add a `@pytest.mark.parametrize` K=3 and K=16 (force `classes_per_forward`/concept count) to exercise the assembled-stack peak (§14 risk).

- [ ] **Step 2: Structural verification (no GPU present)**

```bash
uv run ruff check tests/gpu/test_semantic_train_step_gpu.py
uv run ruff format --check tests/gpu/test_semantic_train_step_gpu.py
uv run python -m py_compile tests/gpu/test_semantic_train_step_gpu.py
```
Expected: pass. (The test EXECUTES only on the GPU runner / CI via `scripts/run_gpu_tests.sh`.)

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_semantic_train_step_gpu.py
git commit -m "test(#113): GPU smoke — semantic train_step on real SAM3 (K=3, K=16 headroom)"
```

### Task C5: Phase C verification gate **[SERIALIZE — last in C]**

- [ ] **Step 1: CPU suite + lint/format/type + import smoke**

```bash
uv run ruff check src/custom_sam_peft/models/semantic.py src/custom_sam_peft/train/
uv run ruff format --check src/custom_sam_peft/models/semantic.py src/custom_sam_peft/train/
uv run mypy --strict src/custom_sam_peft
uv run python -c "import custom_sam_peft"
uv run pytest -o "addopts=" tests/unit tests/train -q
```
Expected: GREEN. Restate the §6.5 contract in the phase handoff.

---

## Phase D — eval (§8)

**Feature block:** `SemanticEvaluator` (mIoU forward loop over a streaming confusion matrix), `compute_semantic_metrics` + `SemanticMetrics`, and the task-tagged `MetricsReport`/`metrics.json`. CPU (confusion math) + one GPU smoke.

**Consumes from Phase A:** §5.2 encoding, the semantic `Dataset`. **From Phase C:** `marginalize_group` + `build_semantic_logits` + `semantic_argmax`.

**Interface contract this phase PRODUCES (spec §8.2) — restate verbatim-faithful:**

> Exposes `SemanticEvaluator` with the SAME public surface as `Evaluator` (`evaluate(model, dataset, *, return_per_example_iou=False) -> MetricsReport | tuple[MetricsReport, list[float]]`, `evaluate_and_save(model, dataset, output_dir) -> MetricsReport`), `compute_semantic_metrics(confusion: (K+1,K+1) np.ndarray, class_names: list[str]) -> SemanticMetrics`, and the task-tagged `MetricsReport`/`metrics.json` (semantic `overall = {"mIoU", "pixel_acc"}`, `per_class = {name: {"IoU": ..}}`, `n_predictions` repurposed as pixels-scored, plus a `"task"` field in the JSON). Consumes Phase C's `marginalize_group` + §5.2. CPU-testable: streaming-confusion math against synthetic label maps; one GPU test asserts the real forward + mIoU on a tiny mask_png fixture.

### Task D1: `SemanticMetrics` + `compute_semantic_metrics` (`eval/metrics.py`) **[PARALLEL-OK — file-disjoint from D2]**

**Files:**
- Modify: `src/custom_sam_peft/eval/metrics.py` (add `SemanticMetrics` + `compute_semantic_metrics`; keep `MetricsReport` shared)
- Test: `tests/unit/test_semantic_metrics.py`

Spec §8.1. Pure numpy confusion-matrix math — no torch, no model.

- [ ] **Step 1: Write the failing tests** (hand-built confusion matrices)

```python
# tests/unit/test_semantic_metrics.py
"""compute_semantic_metrics on hand-built confusion matrices (§8.1)."""

from __future__ import annotations

import numpy as np

from custom_sam_peft.eval.metrics import compute_semantic_metrics


def test_perfect_prediction_miou_one():
    # 3 classes (bg + 2 concepts), diagonal confusion -> mIoU == 1.
    conf = np.diag([10, 20, 30]).astype(np.int64)
    m = compute_semantic_metrics(conf, class_names=["road", "tree"])
    assert m.mean_iou == 1.0
    assert m.pixel_accuracy == 1.0
    assert set(m.per_class.keys()) == {"background", "road", "tree"}
    assert m.per_class["road"] == 1.0


def test_iou_formula_tp_fp_fn():
    # class 1: TP=8, FP=2 (col1 row0), FN=2 (row1 col0). IoU = 8/(8+2+2)=0.666...
    conf = np.array([[10, 2, 0], [2, 8, 0], [0, 0, 5]], dtype=np.int64)
    m = compute_semantic_metrics(conf, class_names=["a", "b"])
    assert abs(m.per_class["a"] - 8 / 12) < 1e-9


def test_no_gt_class_skipped_from_miou():
    # class 2 has zero GT pixels (row all-zero) -> omitted from mIoU.
    conf = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=np.int64)
    m = compute_semantic_metrics(conf, class_names=["a", "b"])
    # only bg + a have GT; mIoU = mean over those two.
    assert "b" in m.per_class  # reported but...
    # mIoU computed over classes-with-GT only.
    assert abs(m.mean_iou - 1.0) < 1e-9


def test_pixel_accuracy_is_trace_over_total():
    conf = np.array([[8, 2], [0, 10]], dtype=np.int64)
    m = compute_semantic_metrics(conf, class_names=["x"])  # bg + 1 concept
    assert abs(m.pixel_accuracy - 18 / 20) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_metrics.py -v`
Expected: FAIL — `compute_semantic_metrics`/`SemanticMetrics` do not exist.

- [ ] **Step 3: Implement** (spec §8.1; `ignore_index` is excluded upstream — never added to the matrix)

```python
@dataclass(frozen=True)
class SemanticMetrics:
    mean_iou: float
    pixel_accuracy: float
    per_class_iou: dict[str, float]  # class_name (incl "background") -> IoU


def compute_semantic_metrics(
    confusion: np.ndarray,  # (K+1, K+1) int64, rows=GT, cols=pred
    class_names: list[str],  # len K; index 0 reported as "background"
) -> SemanticMetrics:
    names = ["background", *class_names]
    conf = confusion.astype(np.float64)
    tp = np.diag(conf)
    gt = conf.sum(axis=1)  # row sums = GT per class
    pred = conf.sum(axis=0)  # col sums = pred per class
    denom = gt + pred - tp
    per_class: dict[str, float] = {}
    ious_with_gt: list[float] = []
    for c, name in enumerate(names):
        if denom[c] > 0:
            iou = float(tp[c] / denom[c])
        else:
            iou = 0.0
        per_class[name] = iou
        if gt[c] > 0:  # only classes with GT support count toward mIoU
            ious_with_gt.append(iou)
    mean_iou = float(np.mean(ious_with_gt)) if ious_with_gt else 0.0
    total = float(conf.sum())
    pixel_accuracy = float(tp.sum() / total) if total > 0 else 0.0
    return SemanticMetrics(mean_iou=mean_iou, pixel_accuracy=pixel_accuracy, per_class_iou=per_class)
```

> Mirror the COCO "skip no-GT class" behavior (`evaluator.py:255` / `metrics.py:93`). `per_class` reports every class for transparency, but mIoU averages only classes with GT support (matches the test). Adjust `test_no_gt_class_skipped_from_miou`/`per_class` membership to whatever the implementer settles on — pin it consistently.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/metrics.py
uv run ruff format --check src/custom_sam_peft/eval/metrics.py
uv run mypy --strict src/custom_sam_peft/eval/metrics.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/eval/metrics.py tests/unit/test_semantic_metrics.py
git commit -m "feat(#113): SemanticMetrics + compute_semantic_metrics (streaming confusion mIoU)"
```

### Task D2: `SemanticEvaluator` + task-tagged `metrics.json` (`eval/semantic_evaluator.py`) **[SERIALIZE — depends on D1, C1]**

**Files:**
- Create: `src/custom_sam_peft/eval/semantic_evaluator.py`
- Modify: `src/custom_sam_peft/eval/evaluator.py` (only if a shared helper must be extracted — prefer NOT touching it; reuse via import) and the `metrics.json` write to add a `"task"` field
- Test: `tests/unit/test_semantic_evaluator.py`

Spec §8.1/§8.2. Same public surface as `Evaluator` (so `run_eval` + the trainer's mid-run eval dispatch on task with no caller rewrite). Reuses the `_iter_predictions`-style multiplex forward loop + OOM ladder (`evaluator.py:121`), but accumulates a `(K+1)`-class confusion matrix via `marginalize_group` instead of `queries_to_coco_results`, upsampling logits to GT res before argmax (§6.3), skipping `ignore_index` pixels.

- [ ] **Step 1: Write the failing tests** (stub model returning fixed output dicts + synthetic semantic dataset)

```python
# tests/unit/test_semantic_evaluator.py
"""SemanticEvaluator with a CPU stub model + synthetic semantic dataset (§8)."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.eval.semantic_evaluator import SemanticEvaluator
# Build a tiny in-memory semantic Dataset (class_names of len K, __getitem__ ->
# Example with SemanticTarget) + a stub model whose forward returns §3.2 dicts.
# Read tests for the instance Evaluator (grep -rln "Evaluator(" tests/) to mirror
# the stub-model + dataset wiring.


def test_evaluate_returns_metrics_report_with_miou(stub_semantic_model, tiny_semantic_dataset):
    ev = SemanticEvaluator(EvalConfig())
    report = ev.evaluate(stub_semantic_model, tiny_semantic_dataset)
    assert isinstance(report, MetricsReport)
    assert "mIoU" in report.overall and "pixel_acc" in report.overall
    assert 0.0 <= report.overall["mIoU"] <= 1.0
    assert report.per_class  # populated, keyed by class name


def test_evaluate_and_save_writes_task_tagged_json(stub_semantic_model, tiny_semantic_dataset, tmp_path):
    ev = SemanticEvaluator(EvalConfig())
    ev.evaluate_and_save(stub_semantic_model, tiny_semantic_dataset, tmp_path)
    import json
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["task"] == "semantic"
    assert "mIoU" in data["overall"]


def test_per_example_iou_returned(stub_semantic_model, tiny_semantic_dataset):
    ev = SemanticEvaluator(EvalConfig())
    report, per_ex = ev.evaluate(stub_semantic_model, tiny_semantic_dataset, return_per_example_iou=True)
    assert isinstance(per_ex, list)
    assert len(per_ex) == len(tiny_semantic_dataset)
```

> **Fixture note:** add `stub_semantic_model` + `tiny_semantic_dataset` fixtures by mirroring the instance `Evaluator` tests. The stub model is the `Sam3Wrapper`-wrapped `TinySam3Stub`; the dataset is a tiny in-memory `Dataset` returning semantic `Example`s. The `metrics.json` filename + the `task` field must match whatever `evaluate_and_save` writes — pin both.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_evaluator.py -v`
Expected: FAIL — `SemanticEvaluator` does not exist.

- [ ] **Step 3: Implement** (spec §8.1/§8.2)

- `SemanticEvaluator.__init__(self, cfg: EvalConfig)` — same as `Evaluator`.
- `evaluate(model, dataset, *, return_per_example_iou=False)`:
  - Run the multiplex forward loop (mirror `evaluator.py:_iter_predictions` structure + OOM ladder) one K-group at a time; collect graph-free per-group slices (eval → `torch.no_grad`), `build_semantic_logits` → `(B, K+1, H, W)`, upsample bilinearly to GT res (§6.3, like `_upsample_mask_logits` `eval/postprocess.py:33`), `semantic_argmax`.
  - For each image, build a `(K+1, K+1)` confusion via `np.bincount((K+1)*gt[valid] + pred[valid], minlength=(K+1)**2)` over `valid = gt != ignore_index` pixels; sum into a running matrix. `ignore_index` pixels never enter the matrix.
  - `compute_semantic_metrics(confusion, dataset.class_names)` → `SemanticMetrics`.
  - Pack into the **shared** `MetricsReport`: `overall = {"mIoU": m.mean_iou, "pixel_acc": m.pixel_accuracy}`, `per_class = {name: {"IoU": iou} for name, iou in m.per_class_iou.items()}`, `n_images = len(dataset)`, `n_predictions = <pixels scored>` (repurposed; document).
  - `per_example_iou`: per image, its mean IoU over present classes (for the worst-image viz picker, `runner.py:210`).
- `evaluate_and_save(model, dataset, output_dir)`: call `evaluate`, write `metrics.json` with a `"task": "semantic"` field plus `overall`/`per_class`/`n_images`/`n_predictions`. **Add the `"task"` field to the instance `metrics.json` write too** (additive, `"task": "instance"`) so a reader always knows which keys to expect — do this in the instance write site (`evaluator.py:414`) carefully, asserting the instance JSON otherwise unchanged.

> **Caution (instance metrics.json):** adding `"task": "instance"` to the instance JSON is the only instance-path change in Phase D. It is purely additive. Confirm with the instance eval tests that no test asserts an exact key set that would break — if one does, update it and note the additive field. If risk is unacceptable, gate the `"task"` field to semantic only and document that instance JSON omits it. Prefer the additive field for reader symmetry; pin the choice with a test.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -o "addopts=" tests/unit/test_semantic_evaluator.py -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + import smoke + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/semantic_evaluator.py src/custom_sam_peft/eval/evaluator.py
uv run ruff format --check src/custom_sam_peft/eval/semantic_evaluator.py src/custom_sam_peft/eval/evaluator.py
uv run mypy --strict src/custom_sam_peft/eval/semantic_evaluator.py src/custom_sam_peft/eval/evaluator.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/eval/semantic_evaluator.py src/custom_sam_peft/eval/evaluator.py tests/unit/test_semantic_evaluator.py
git commit -m "feat(#113): SemanticEvaluator (streaming mIoU) + task-tagged metrics.json"
```

### Task D3: GPU smoke — `SemanticEvaluator.evaluate` on the real forward **[SERIALIZE — last in D]**

**Files:**
- Create: `tests/gpu/test_semantic_eval_gpu.py`

Spec §11. One `SemanticEvaluator.evaluate` on a tiny mask_png fixture (2-image, 3-class) → finite mIoU in `[0,1]` + populated `per_class`. Mirror `tests/gpu/test_run_end_to_end_gpu.py` / the GPU eval test for wiring.

- [ ] **Step 1: Write the GPU test** (gated by `requires_checkpoint` + `requires_compatible_gpu`)

Build the tiny mask_png fixture in `tmp_path`, load the real wrapper, run `SemanticEvaluator(EvalConfig()).evaluate(model, dataset)`, assert `0.0 <= report.overall["mIoU"] <= 1.0` and `report.per_class` non-empty.

- [ ] **Step 2: Structural verification + commit**

```bash
uv run ruff check tests/gpu/test_semantic_eval_gpu.py
uv run ruff format --check tests/gpu/test_semantic_eval_gpu.py
uv run python -m py_compile tests/gpu/test_semantic_eval_gpu.py
git add tests/gpu/test_semantic_eval_gpu.py
git commit -m "test(#113): GPU smoke — SemanticEvaluator mIoU on real SAM3 (tiny mask_png)"
```

### Task D4: Phase D verification gate **[SERIALIZE — last in D]**

- [ ] **Step 1: CPU suite + lint/format/type + import smoke**

```bash
uv run ruff check src/custom_sam_peft/eval/
uv run ruff format --check src/custom_sam_peft/eval/
uv run mypy --strict src/custom_sam_peft
uv run python -c "import custom_sam_peft"
uv run pytest -o "addopts=" tests/unit tests/eval -q
```
Expected: GREEN — including the instance eval tests (the `"task"` field is additive). Restate the §8.2 contract.

---

## Phase E — CLI / predict / export / doctor (§10)

**Feature block:** the `cfg.task` dispatch across `eval`/`predict`/`export`/`doctor` — predict's semantic label-map writers + viz, the doctor task-aware tables, and confirmation that export round-trips `task`. Mostly CPU (stubbed). This is the wiring phase.

**Consumes:** Phases A–D (datasets, marginalization, loss bundle, evaluator).

**Interface contract (spec §10.5):**

> Each command reads `cfg.task` and dispatches. No new cross-module exports; this is the wiring phase. `eval` → `SemanticEvaluator`; `predict` → marginalize → label map + colorized/index PNG writers + viz; `doctor --config` → task row + "Resolved semantic losses" table + `--json` `task`/`semantic_loss`; `export` round-trips `task` (no code change beyond the config field already round-tripping).

### Task E1: `eval` dispatch to `SemanticEvaluator` (`eval/runner.py`) **[PARALLEL-OK — file-disjoint from E2/E3/E4]**

**Files:**
- Modify: `src/custom_sam_peft/eval/runner.py` (`run_eval` at `:60`; `Evaluator(eval_cfg)` at `:173`)
- Test: `tests/unit/test_eval_runner_semantic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_runner_semantic.py
"""run_eval dispatches to SemanticEvaluator under task: semantic (§10.2)."""

from __future__ import annotations

from custom_sam_peft.eval.semantic_evaluator import SemanticEvaluator
# Build a semantic cfg + stub model/dataset; assert run_eval constructs a
# SemanticEvaluator (monkeypatch/spy) and returns a report with mIoU.


def test_run_eval_uses_semantic_evaluator_under_semantic_task(monkeypatch, semantic_cfg, stub_model, tiny_semantic_dataset):
    seen = {}
    orig = SemanticEvaluator.evaluate_and_save

    def spy(self, *a, **k):
        seen["used"] = True
        return orig(self, *a, **k)

    monkeypatch.setattr(SemanticEvaluator, "evaluate_and_save", spy)
    # ... invoke run_eval with semantic_cfg + the stub model/dataset path ...
    assert seen.get("used")
```

> The exact `run_eval` invocation shape (it loads a checkpoint/adapter + builds a dataset) means the test likely monkeypatches the model/dataset construction. Read `tests/unit/test_eval_runner.py` to mirror how the instance `run_eval` is tested with stubs, then flip to semantic.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_eval_runner_semantic.py -v`
Expected: FAIL — `run_eval` always builds `Evaluator`.

- [ ] **Step 3: Implement**

In `eval/runner.py:run_eval`, where it constructs `Evaluator(eval_cfg)` (`:173`), branch: `evaluator = SemanticEvaluator(eval_cfg) if cfg.task == "semantic" else Evaluator(eval_cfg)`. The `evaluate`/`evaluate_and_save` signatures match (§8.1), so the surrounding metrics.json/viz wiring is reused. `eval.batch_size == "auto"` resolution (`:156`) and the OOM cap carry over. `--split` handling unchanged. (The dataset builder already threads `task` from Task A7's `eval/runner.py:126` change — confirm it does.)

- [ ] **Step 4: Run to verify it passes + commit**

```bash
uv run pytest -o "addopts=" tests/unit/test_eval_runner_semantic.py -v
uv run ruff check src/custom_sam_peft/eval/runner.py
uv run ruff format --check src/custom_sam_peft/eval/runner.py
uv run mypy --strict src/custom_sam_peft/eval/runner.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/eval/runner.py tests/unit/test_eval_runner_semantic.py
git commit -m "feat(#113): run_eval dispatches to SemanticEvaluator under task: semantic"
```

### Task E2: Semantic predict — label-map writer + viz + runner branch **[SERIALIZE within E2; file-disjoint from E1/E4]**

**Files:**
- Modify: `src/custom_sam_peft/predict/writers.py` (add `write_semantic_label_map`), `src/custom_sam_peft/predict/visualize.py` (semantic overlay), `src/custom_sam_peft/predict/runner.py` (semantic branch at `:446`), `src/custom_sam_peft/cli/predict_cmd.py` (prompt-defaulting + instance-only-flag INFO)
- Test: `tests/unit/test_predict_semantic.py`

Spec §10.3. Sub-task ordering: writer first (pure I/O, testable alone), then runner branch, then CLI prompt-defaulting.

- [ ] **Step 1: Write the failing tests** (writer + runner output schema)

```python
# tests/unit/test_predict_semantic.py
"""Semantic predict: label-map writer + colorized PNG + predictions.json schema (§10.3)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from custom_sam_peft.predict.writers import write_semantic_label_map


def test_write_semantic_label_map_emits_index_and_colorized(tmp_path):
    label_map = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
    paths = write_semantic_label_map(
        label_map, image_id="a", out_dir=tmp_path, class_names=["road", "tree"]
    )
    # raw single-channel index PNG + colorized PNG.
    idx = np.array(Image.open(paths["index_path"]))
    assert idx.dtype == np.uint8 or idx.dtype == np.uint16
    assert set(np.unique(idx).tolist()) <= {0, 1, 2}
    col = Image.open(paths["colorized_path"])
    assert col.mode in ("RGB", "RGBA")
    # background (0) -> black in the colorized map.
    col_arr = np.array(col)
    assert tuple(col_arr[0, 0][:3]) == (0, 0, 0)  # label 0 == background == black
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_predict_semantic.py -v`
Expected: FAIL — `write_semantic_label_map` does not exist.

- [ ] **Step 3: Implement** (spec §10.3)

- `write_semantic_label_map(label_map, *, image_id, out_dir, class_names) -> dict[str, Path]`: write a raw single-channel index PNG (uint8/uint16 raw class indices) AND a colorized label-map PNG using a **deterministic palette keyed by concept index** (background = black). Return `{"index_path": ..., "colorized_path": ...}`.
- `predict/visualize.py`: a semantic overlay (blend the colorized label map over the image at some alpha). Mirror the existing instance viz API so the runner can call it uniformly.
- `predict/runner.py` (`:446`): under `task: semantic`, replace the per-query COCO-style result entries with `marginalize_group` → `(N_imgs, K+1, H, W)` → upsample → `semantic_argmax` → `(H, W)` label map per image; call `write_semantic_label_map`; per-image `predictions.json` entry becomes `{"image_id", "label_map_path", "concepts": class_names}`.
- `cli/predict_cmd.py`: under `--config` with `task: semantic`, default prompts to the dataset's `class_names` when `--prompts` omitted (else use the user's concepts — open-vocab is *available*, caveat-documented §2). `--score-threshold`/`--top-k`/`--save-masks` are instance-only → emit a one-time INFO and ignore under semantic.

- [ ] **Step 4: Run to verify it passes + commit**

```bash
uv run pytest -o "addopts=" tests/unit/test_predict_semantic.py -v
uv run ruff check src/custom_sam_peft/predict/ src/custom_sam_peft/cli/predict_cmd.py
uv run ruff format --check src/custom_sam_peft/predict/ src/custom_sam_peft/cli/predict_cmd.py
uv run mypy --strict src/custom_sam_peft/predict src/custom_sam_peft/cli/predict_cmd.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/predict/ src/custom_sam_peft/cli/predict_cmd.py tests/unit/test_predict_semantic.py
git commit -m "feat(#113): semantic predict (label-map writer + viz + runner branch + prompt defaulting)"
```

### Task E3: `csp doctor` task-aware tables (`cli/doctor_cmd.py`) **[PARALLEL-OK — file-disjoint from E1/E2]**

**Files:**
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py` (the "Resolved losses" table at `:134`; the "Dataset" table at `:78`; the `--json` block)
- Test: `tests/unit/test_cli_doctor_config.py` (extend) or a new `tests/unit/test_cli_doctor_semantic.py`

Spec §10.5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_cli_doctor_semantic.py
"""doctor --config renders task + semantic-loss tables under task: semantic (§10.5)."""

from __future__ import annotations

# Build a semantic TrainConfig; render the doctor report; assert:
#  - a "Task" row showing "semantic"
#  - a "Resolved semantic losses" table with sem_family/w_ce/w_region
#  - the instance "Resolved losses" table is suppressed
#  - --json carries "task": "semantic" and a "semantic_loss" sub-key

def test_doctor_semantic_renders_task_and_semantic_loss_table(semantic_cfg, capsys):
    ...  # mirror tests/unit/test_cli_doctor_config.py wiring


def test_doctor_json_carries_task_and_semantic_loss(semantic_cfg):
    ...  # assert the --json dict has "task" and "semantic_loss"
```

> Read `tests/unit/test_cli_doctor_config.py` for how the instance "Resolved losses" table + `--json` are tested, and mirror.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_cli_doctor_semantic.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (spec §10.5)

- Always show a `task` row (one-line "Task" in the runtime/config summary, and a `task` row in the "Dataset" table at `:78`).
- When `task: semantic`: render a "Resolved semantic losses" table (mirror the "Resolved losses" table at `:134`) from `resolve(cfg.train.semantic_loss)` + `dump_semantic_loss_bundle`, plus a one-line "Head: marginalization (head-free)" / "Head: semantic_seg (surfaced)" per `cfg.train.semantic_loss.source`. **Suppress** the instance "Resolved losses" table (it is inert).
- When `task: instance`: unchanged (instance table shown, semantic suppressed).
- `--json`: add `"task"`, and under semantic a `"semantic_loss"` sub-key from `dump_semantic_loss_bundle`, replacing the inert `"loss"` block. Import `dump_semantic_loss_bundle` from `models.losses` (torch-free).

- [ ] **Step 4: Run to verify it passes + commit**

```bash
uv run pytest -o "addopts=" tests/unit/test_cli_doctor_semantic.py tests/unit/test_cli_doctor_config.py -v
uv run ruff check src/custom_sam_peft/cli/doctor_cmd.py
uv run ruff format --check src/custom_sam_peft/cli/doctor_cmd.py
uv run mypy --strict src/custom_sam_peft/cli/doctor_cmd.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/cli/doctor_cmd.py tests/unit/test_cli_doctor_semantic.py
git commit -m "feat(#113): doctor task row + Resolved semantic losses table + --json task/semantic_loss"
```

### Task E4: `export` round-trips `task` (regression) **[PARALLEL-OK — file-disjoint]**

**Files:**
- Verify/Modify: `src/custom_sam_peft/cli/export_cmd.py` (likely NO code change — confirm the config round-trips `task`)
- Test: `tests/unit/test_cli_export_semantic.py`

Spec §10.4 Decision: export is task-agnostic (ships the LoRA adapter + `config.yaml`); the only change is the config now carries a `task` field which round-trips. No head to export (marginalization is inference-time math).

- [ ] **Step 1: Write the regression test**

```python
# tests/unit/test_cli_export_semantic.py
"""Export round-trips task: semantic in the bundled config.yaml (§10.4)."""

from __future__ import annotations

# Build a semantic TrainConfig, run export to tmp_path, load the bundled config,
# assert task == "semantic" and semantic_loss is present. Mirror
# tests/unit/test_cli_export.py.

def test_export_bundles_semantic_config(semantic_cfg, tmp_path):
    ...  # assert exported config.yaml round-trips task == "semantic"
```

- [ ] **Step 2: Run; implement only if it fails**

Run: `uv run pytest -o "addopts=" tests/unit/test_cli_export_semantic.py -v`
Expected: likely PASS with no code change (the config field round-trips). If export strips unknown/new fields or rebuilds the config in a way that drops `task`, fix `export_cmd.py` so the full `cfg` (including `task` + `semantic_loss`) is serialized. Add a `# note: future seam` comment that if `source: semantic_seg` with an unfrozen head were ever supported (post-v1), the head weights would join the export bundle — no v1 code.

- [ ] **Step 3: Commit**

```bash
uv run ruff check src/custom_sam_peft/cli/export_cmd.py
uv run ruff format --check src/custom_sam_peft/cli/export_cmd.py
git add src/custom_sam_peft/cli/export_cmd.py tests/unit/test_cli_export_semantic.py
git commit -m "test(#113): export round-trips task: semantic in bundled config"
```

### Task E5: Phase E verification gate + final blast-radius sweep **[SERIALIZE — last task]**

- [ ] **Step 1: Full CPU suite + lint/format/type + import smoke**

```bash
uv run ruff check src/custom_sam_peft tests/
uv run ruff format --check src/custom_sam_peft tests/
uv run mypy --strict src/custom_sam_peft
uv run python -c "import custom_sam_peft"
uv run pytest -o "addopts=" tests/unit tests/config tests/data tests/cli tests/eval tests/train tests/predict -q
```
Expected: ALL GREEN.

- [ ] **Step 2: §12 blast-radius + instance-invariance sweep**

```bash
grep -rn "Example(" src/ tests/ | grep -v "SemanticTarget"   # all carry instances or default
grep -rn "StepResult.empty(\|collate_batch(\|loss/mask" src/ tests/
```
Confirm: every `Example(...)` valid under the defaulted `instances`; instance `StepResult.empty()` keys + instance `metrics.json` keys + instance logging byte-identical; `collate_batch`'s instance consumers unaffected; a config omitting `task` validates + runs as today.

- [ ] **Step 3: GPU smoke note (CI/runner)**

The two GPU smokes (C4, D3) execute via `scripts/run_gpu_tests.sh` on the 5070 Ti runner / CI — do NOT run them locally in-process. Confirm they are collected by the runner (markers present) and structurally valid (`py_compile` passed in C4/D3).

- [ ] **Step 4: Markdown-lint the spec + plan (PR gate)**

Before the PR, run the project's markdown linter (CI's exact `markdownlint-cli2` invocation — discover it from the workflow) on this plan + the spec and fix findings. (Per the Markdown lint gate: spec/plan are linted by CI even though they predate the PR.)

- [ ] **Step 5: Final commit (if Step 4 changed anything)**

```bash
git add docs/superpowers/plans/2026-06-02-semantic-segmentation-task-mode-plan.md docs/superpowers/specs/2026-06-02-semantic-segmentation-task-mode-design.md
git commit -m "docs(#113): markdown-lint fixes on semantic-segmentation spec + plan"
```

---

## Self-review — spec coverage map

| Spec section | Covered by |
| --- | --- |
| §1 Goals (task axis, fixed-class, head-free, text-prompt vocab, domain-aware loss, mIoU, adapters, CPU-testable) | A1, A4, A7, B1–B3, C1, C3, D1–D2 |
| §2 Non-goals | (no tasks — explicitly out of scope; open-vocab caveat documented in E2) |
| §3 SAM3 facts | grounding for C1 (consumes pred_logits/pred_masks/presence_logit_dec/semantic_seg) |
| §4.1 task field | A1 |
| §4.2 DataFormat + SemanticDataConfig | A1 |
| §4.3 cross-field validation | A1 (`_check_task_data_compat`) |
| §4.4 inert knobs | A1 (eval.iou_thresholds/mask_threshold rejection; rest documented-as-ignored) |
| §4.5 none/background semantics | A3 (`build_value_to_label` bg-name set + custom bg), B1 (`background_logit`/`background_class_name`), C1 (bg-prepend) |
| §5.1 SemanticTarget/Example | A2 |
| §5.2 encoding (single source of truth) | A3 |
| §5.3 mask_png adapter | A4 |
| §5.4 semantic HF adapter | A7 |
| §5.5 transforms nearest-interp | A5 |
| §5.6 text-prompt mode all | A4, A7 |
| §5.7 collate | A6 |
| §6.2 marginalization + query_reduce | C1 |
| §6.3 argmax | C1 (`semantic_argmax`; upsample in D2/E2) |
| §6.4 semantic_seg source | C1 (`source` branch) |
| §6.5 marginalize_group | C1 |
| §7.1–7.2 SemanticLossConfig | B1 |
| §7.3 preset table (all 12 + alias) | B2 |
| §7.4 SemanticLoss + compose | B3 |
| §8.1 SemanticEvaluator + metrics | D1, D2 |
| §8.2 MetricsReport shared + task-tagged json | D2 |
| §9 PEFT (no new scope; semantic_seg head frozen) | (no code — documented; C1's semantic_seg path reads pre-trained head) |
| §10.1 train/run branch + StepResult parametrization | C2, C3 |
| §10.2 eval dispatch | E1 |
| §10.3 predict | E2 |
| §10.4 export | E4 |
| §10.5 doctor | E3 |
| §11 testing strategy | every task's TDD steps; GPU smokes C4, D3 |
| §12 backward compat / blast radius | A2, A6, C2 (BLAST-RADIUS), E5 sweep |
| §13 phasing | the A–E phase structure + interface contracts |
| §14 open questions / risks | OQ-1 (C1 source branch), OQ-2 (C1 query_reduce both modes), OQ-3 (B2 tags), train-loop topology (C3 + C4 K=16 GPU smoke), nearest-interp (A5) |

**Open decisions carried as implemented-with-default (spec requirement #6):** `query_reduce` default `max` with `sum` available — both implemented (C1) + both tested (`test_marginalize_max_shape`/`test_marginalize_sum_shape`). `source` default `marginalize` with `semantic_seg` available — both implemented (C1) + both tested (`test_marginalize_max_shape`/`test_semantic_seg_source_shape`).
