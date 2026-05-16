# Model-Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `load_sam31` (Meta `sam3` package builder + `.pt` remap), `Sam3Wrapper` (pass-through forward), `models/matching.py` (canonical adapter + Hungarian matcher), and `models/losses.py` (DETR-standard mix), so the trainer has a real model + losses for COCO image finetuning.

**Architecture:** The loader builds the full multiplex-capable SAM 3.1 via Meta's `sam3.build_sam3` helper, loads weights from the local `.pt` checkpoint with key remapping, applies bf16 + gradient checkpointing, and returns a thin `Sam3Wrapper` whose `forward` returns Meta's native output dict unchanged. A separate `matching.py` module owns the Meta→canonical adapter and the Hungarian matcher; `losses.py` computes per-component losses on the matched subset and totals them with weights from `LossConfig`.

**Tech Stack:** Python 3.13, PyTorch, Meta `sam3` (git), HuggingFace `transformers` ≥5.0 (CLIPTokenizer), `pydantic` v2, `scipy.optimize.linear_sum_assignment`, `pytest`, `ruff`.

**Reference spec:** `docs/superpowers/specs/2026-05-16-model-loading-design.md`

---

## Pre-flight checks

Before starting any task, verify the environment:

```bash
# Confirm the project venv has torch and transformers
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

# Confirm the SAM 3.1 checkpoint is present
ls -lh models/sam3.1/sam3.1_multiplex.pt
```

If the checkpoint is missing, follow the README in `models/sam3.1/` or run:
```bash
uv run huggingface-cli download facebook/sam3.1 --local-dir models/sam3.1
```
(You must have accepted Meta's gated license on Hugging Face first.)

---

## File map (what gets touched)

| File | Action | Owning task |
| --- | --- | --- |
| `pyproject.toml` | Modify | 1 |
| `src/esam3/config/schema.py` | Modify | 2 |
| `tests/unit/test_model_config.py` | Create | 2 |
| `tests/unit/test_loss_config.py` | Create | 2 |
| `src/esam3/models/matching.py` | Create | 3, 4, 5 |
| `tests/unit/test_matching.py` | Create | 4 |
| `tests/unit/test_meta_to_canonical.py` | Create | 5 |
| `tests/fixtures/tiny_sam3_stub.py` | Rewrite | 5 |
| `src/esam3/models/losses.py` | Rewrite | 6, 7, 8, 9, 10 |
| `tests/unit/test_losses.py` | Create | 6, 7, 8, 9, 10 |
| `tests/unit/test_stubs_raise.py` | Modify | 11 |
| `src/esam3/models/sam3.py` | Rewrite | 12, 13 |
| `tests/unit/test_sam3_wrapper.py` | Create | 12 |
| `tests/integration/test_load_sam31_real.py` | Create | 14 |
| `tests/conftest.py` | Modify (add marker) | 14 |
| `configs/examples/coco_text_lora.yaml` | Modify | 15 |
| `configs/examples/coco_bbox_qlora.yaml` | Modify | 15 |

---

## Task 1: Add dependencies (sam3 git, transformers bump, scipy)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Pin the Meta sam3 commit SHA**

Open `github.com/facebookresearch/sam3` in a browser, copy the latest commit SHA on the default branch (e.g. `main`). Record it; you'll paste it into the dependency line below. If a release tag exists (e.g. `v0.1.0`), prefer the tag.

- [ ] **Step 2: Edit `pyproject.toml` dependencies block**

Replace the `[project] dependencies` list in `pyproject.toml` with the version below. Key changes: bump `transformers>=4.50` → `>=5.0`, add `scipy>=1.10`, add `sam3 @ git+...`.

```toml
dependencies = [
  "torch>=2.4",
  "transformers>=5.0",
  "peft>=0.13",
  "datasets>=3.0",
  "pydantic>=2.7",
  "typer>=0.12",
  "pyyaml>=6.0",
  "pycocotools>=2.0",
  "numpy>=1.26",
  "rich>=13",
  "pillow>=10",
  "albumentations>=1.4",
  "opencv-python-headless>=4.10",
  "scipy>=1.10",
  "sam3 @ git+https://github.com/facebookresearch/sam3@<PASTE-SHA-HERE>",
]
```

- [ ] **Step 3: Sync deps**

Run: `uv sync`
Expected: completes without error. The sam3 git checkout may take 30–60 s on first install.

- [ ] **Step 4: Smoke import**

Run: `uv run python -c "import sam3, transformers, scipy.optimize; print(sam3.__file__); print(transformers.__version__)"`
Expected: prints path to sam3 package, transformers version ≥ 5.0, no errors.

If `import sam3` fails with "no module named sam3", inspect the package — Meta may use a different top-level name (e.g. `sam_3` or `segment_anything_3`). Adjust the dependency line and try again.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): add sam3 git dep, bump transformers, add scipy"
```

---

## Task 2: Extend ModelConfig + add LossConfig and MatcherWeights

**Files:**
- Modify: `src/esam3/config/schema.py:31-35` (ModelConfig)
- Modify: `src/esam3/config/schema.py` (add new classes near PEFTConfig)
- Modify: `src/esam3/config/schema.py:TrainConfig` (add `loss` field)
- Create: `tests/unit/test_model_config.py`
- Create: `tests/unit/test_loss_config.py`

- [ ] **Step 1: Write failing tests for the new ModelConfig fields**

Create `tests/unit/test_model_config.py`:

```python
"""Unit tests for ModelConfig schema additions in spec/model-loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import ModelConfig


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.name == "facebook/sam3.1"
    assert cfg.local_dir == "models/sam3.1"
    assert cfg.checkpoint_file == "sam3.1_multiplex.pt"
    assert cfg.revision is None
    assert cfg.gradient_checkpointing is True
    assert cfg.dtype == "bfloat16"
    assert cfg.device is None


def test_model_config_overrides() -> None:
    cfg = ModelConfig(local_dir=None, device="cpu", gradient_checkpointing=False)
    assert cfg.local_dir is None
    assert cfg.device == "cpu"
    assert cfg.gradient_checkpointing is False


def test_model_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(unknown_field="x")  # type: ignore[call-arg]
```

- [ ] **Step 2: Write failing tests for LossConfig + MatcherWeights**

Create `tests/unit/test_loss_config.py`:

```python
"""Unit tests for LossConfig + MatcherWeights schemas in spec/model-loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import LossConfig, MatcherWeights, TrainConfig


def test_matcher_weights_defaults() -> None:
    w = MatcherWeights()
    assert w.lambda_cls == 2.0
    assert w.lambda_l1 == 5.0
    assert w.lambda_giou == 2.0
    assert w.lambda_mask == 5.0


def test_loss_config_defaults() -> None:
    cfg = LossConfig()
    assert cfg.w_mask == 1.0
    assert cfg.w_box == 5.0
    assert cfg.w_obj == 1.0
    assert cfg.w_cls == 2.0
    assert cfg.focal_gamma == 2.0
    assert cfg.focal_alpha == 0.25
    assert isinstance(cfg.matcher_weights, MatcherWeights)


def test_loss_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LossConfig(unknown=1.0)  # type: ignore[call-arg]


def test_train_config_includes_loss() -> None:
    """TrainConfig must expose `loss: LossConfig` with defaults."""
    from esam3.config.schema import (
        DataConfig,
        DataSplit,
        PEFTConfig,
        RunConfig,
        TrainHyperparams,
    )

    tc = TrainConfig(
        run=RunConfig(name="x"),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a", images="b"),
            val=DataSplit(annotations="a", images="b"),
            prompt_mode="bbox",
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )
    assert isinstance(tc.train.loss if hasattr(tc.train, "loss") else tc.loss, LossConfig)
```

- [ ] **Step 3: Run tests — verify they fail**

Run: `uv run pytest tests/unit/test_model_config.py tests/unit/test_loss_config.py -v`
Expected: FAIL — `local_dir` / `checkpoint_file` / `device` not in ModelConfig; `LossConfig` / `MatcherWeights` not importable.

- [ ] **Step 4: Implement ModelConfig extensions**

In `src/esam3/config/schema.py`, replace the existing `ModelConfig` class with:

```python
class ModelConfig(_Strict):
    name: str = "facebook/sam3.1"
    local_dir: str | None = "models/sam3.1"
    checkpoint_file: str = "sam3.1_multiplex.pt"
    revision: str | None = None
    gradient_checkpointing: bool = True
    dtype: Dtype = "bfloat16"
    device: str | None = None
```

- [ ] **Step 5: Add MatcherWeights and LossConfig classes**

In `src/esam3/config/schema.py`, immediately after `class PEFTConfig` (before `class TrainHyperparams`), insert:

```python
class MatcherWeights(_Strict):
    """Per-term cost weights for the Hungarian matcher."""

    lambda_cls: PositiveFloat = 2.0
    lambda_l1: PositiveFloat = 5.0
    lambda_giou: PositiveFloat = 2.0
    lambda_mask: PositiveFloat = 5.0


class LossConfig(_Strict):
    """Loss-mix weights and focal CE params for SAM3.1 training."""

    w_mask: PositiveFloat = 1.0
    w_box: PositiveFloat = 5.0
    w_obj: PositiveFloat = 1.0
    w_cls: PositiveFloat = 2.0
    matcher_weights: MatcherWeights = Field(default_factory=MatcherWeights)
    focal_gamma: PositiveFloat = 2.0
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
```

- [ ] **Step 6: Wire `loss` into TrainHyperparams**

In `src/esam3/config/schema.py`, modify `class TrainHyperparams` to add a `loss` field at the bottom:

```python
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
    loss: LossConfig = Field(default_factory=LossConfig)
```

- [ ] **Step 7: Run tests — verify they pass**

Run: `uv run pytest tests/unit/test_model_config.py tests/unit/test_loss_config.py -v`
Expected: PASS — all tests green.

- [ ] **Step 8: Run the full unit suite to confirm no regression**

Run: `uv run pytest tests/unit -v`
Expected: all tests still pass (the stub tests in `test_stubs_raise.py` are still valid until Task 11).

- [ ] **Step 9: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_model_config.py tests/unit/test_loss_config.py
git commit -m "feat(config): add LossConfig/MatcherWeights, extend ModelConfig"
```

---

## Task 3: Scaffold `models/matching.py` with `CanonicalOutputs`

**Files:**
- Create: `src/esam3/models/matching.py`

- [ ] **Step 1: Create the module skeleton**

Create `src/esam3/models/matching.py`:

```python
"""Adapter + Hungarian matcher for SAM 3.1 training.

`meta_to_canonical` is the SINGLE point in the codebase that knows Meta's
native output dict key names. If Meta renames a field, only this function
breaks. Filled in by Task 5 once the actual key names are inspected against
a real `Sam3Wrapper` forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CanonicalOutputs:
    """Output of `meta_to_canonical`. Used by the matcher and losses.

    Shapes:
      class_logits: (B, Q, C+1)   # last index = "no-object"
      pred_boxes:   (B, Q, 4)     # normalized cx,cy,w,h in [0, 1]
      pred_masks:   (B, Q, 288, 288)
      presence:     (B, Q)        # objectness logit
    """

    class_logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    presence: Tensor


def meta_to_canonical(outputs: dict) -> CanonicalOutputs:
    """Convert Meta sam3's native output dict to CanonicalOutputs.

    Implementation deferred to Task 5 (requires inspection of real Meta output).
    """
    raise NotImplementedError("filled in by Task 5 of spec/model-loading")
```

- [ ] **Step 2: Smoke-import**

Run: `uv run python -c "from esam3.models.matching import CanonicalOutputs, meta_to_canonical; print(CanonicalOutputs.__annotations__)"`
Expected: prints `{'class_logits': Tensor, 'pred_boxes': Tensor, ...}`.

- [ ] **Step 3: Commit**

```bash
git add src/esam3/models/matching.py
git commit -m "feat(models): scaffold matching.py with CanonicalOutputs"
```

---

## Task 4: Implement `HungarianMatcher`

**Files:**
- Modify: `src/esam3/models/matching.py`
- Create: `tests/unit/test_matching.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_matching.py`:

```python
"""Unit tests for HungarianMatcher in models/matching.py."""

from __future__ import annotations

import torch

from esam3.data.base import Instance
from esam3.models.matching import CanonicalOutputs, HungarianMatcher


def _make_outputs(q: int = 4, c: int = 2, mask_size: int = 16) -> CanonicalOutputs:
    """Synthetic CanonicalOutputs with B=1, Q queries, C classes."""
    return CanonicalOutputs(
        class_logits=torch.zeros(1, q, c + 1),
        pred_boxes=torch.zeros(1, q, 4),
        pred_masks=torch.zeros(1, q, mask_size, mask_size),
        presence=torch.zeros(1, q),
    )


def _instance(class_id: int, box: list[float], mask_size: int = 16) -> Instance:
    return Instance(
        mask=torch.zeros(mask_size, mask_size),
        class_id=class_id,
        box=torch.tensor(box, dtype=torch.float32),
    )


def test_matcher_empty_targets_returns_empty_pairs() -> None:
    matcher = HungarianMatcher(
        lambda_cls=2.0, lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0
    )
    outputs = _make_outputs(q=4)
    indices = matcher(outputs, [[]])
    assert len(indices) == 1
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 0
    assert tgt_idx.numel() == 0


def test_matcher_returns_one_match_per_target() -> None:
    matcher = HungarianMatcher(
        lambda_cls=2.0, lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0
    )
    outputs = _make_outputs(q=4)
    targets = [[_instance(0, [0.5, 0.5, 0.1, 0.1]), _instance(1, [0.2, 0.2, 0.1, 0.1])]]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2
    # Each target index appears exactly once
    assert sorted(tgt_idx.tolist()) == [0, 1]
    # Each pred index is unique
    assert len(set(pred_idx.tolist())) == 2


def test_matcher_handles_more_targets_than_queries() -> None:
    matcher = HungarianMatcher(
        lambda_cls=2.0, lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0
    )
    outputs = _make_outputs(q=2)
    targets = [[
        _instance(0, [0.1, 0.1, 0.1, 0.1]),
        _instance(0, [0.3, 0.3, 0.1, 0.1]),
        _instance(0, [0.5, 0.5, 0.1, 0.1]),
    ]]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    # Only Q=2 matches possible
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2


def test_matcher_batched() -> None:
    matcher = HungarianMatcher(
        lambda_cls=2.0, lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0
    )
    outputs = CanonicalOutputs(
        class_logits=torch.zeros(2, 3, 3),
        pred_boxes=torch.zeros(2, 3, 4),
        pred_masks=torch.zeros(2, 3, 16, 16),
        presence=torch.zeros(2, 3),
    )
    targets = [
        [_instance(0, [0.5, 0.5, 0.1, 0.1])],
        [_instance(1, [0.2, 0.2, 0.1, 0.1]), _instance(0, [0.7, 0.7, 0.1, 0.1])],
    ]
    indices = matcher(outputs, targets)
    assert len(indices) == 2
    assert indices[0][0].numel() == 1
    assert indices[1][0].numel() == 2
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/test_matching.py -v`
Expected: FAIL — `HungarianMatcher` not importable.

- [ ] **Step 3: Implement `HungarianMatcher`**

Append to `src/esam3/models/matching.py`:

```python
from scipy.optimize import linear_sum_assignment  # noqa: E402
from torch.nn.functional import interpolate  # noqa: E402

from esam3.data.base import Instance  # noqa: E402


def _box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU between every pair in boxes1 (N,4) and boxes2 (M,4), xyxy."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def _dice_cost(pred_masks: Tensor, tgt_masks: Tensor) -> Tensor:
    """Dice cost between every pred (Q, H, W) and target (N, H, W) mask. Returns (Q, N)."""
    p = pred_masks.sigmoid().flatten(1)  # (Q, H*W)
    t = tgt_masks.flatten(1).float()      # (N, H*W)
    num = 2 * p @ t.t()
    den = p.sum(-1)[:, None] + t.sum(-1)[None, :]
    return 1.0 - (num + 1.0) / (den + 1.0)


class HungarianMatcher:
    """DETR-style bipartite matcher. Non-differentiable; called under no_grad."""

    def __init__(
        self,
        lambda_cls: float,
        lambda_l1: float,
        lambda_giou: float,
        lambda_mask: float,
    ) -> None:
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_mask = lambda_mask

    @torch.no_grad()
    def __call__(
        self,
        outputs: CanonicalOutputs,
        targets: list[list[Instance]],
    ) -> list[tuple[Tensor, Tensor]]:
        b, q, _ = outputs.class_logits.shape
        mask_h, mask_w = outputs.pred_masks.shape[-2:]
        results: list[tuple[Tensor, Tensor]] = []
        for i in range(b):
            tgts = targets[i]
            if len(tgts) == 0:
                results.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue
            # Class cost: -prob[target_class] using softmax over class_logits.
            probs = outputs.class_logits[i].softmax(-1)  # (Q, C+1)
            tgt_class = torch.tensor(
                [t.class_id for t in tgts], dtype=torch.long, device=probs.device
            )
            cost_cls = -probs[:, tgt_class]  # (Q, N)

            # Box L1 and GIoU in normalized cxcywh.
            tgt_boxes = torch.stack([t.box for t in tgts]).to(outputs.pred_boxes.device)
            cost_l1 = torch.cdist(outputs.pred_boxes[i], tgt_boxes, p=1)  # (Q, N)
            cost_giou = -_giou(
                _box_cxcywh_to_xyxy(outputs.pred_boxes[i]),
                _box_cxcywh_to_xyxy(tgt_boxes),
            )

            # Mask cost at 288x288 (downsample targets under no_grad).
            tgt_masks = torch.stack([t.mask for t in tgts]).to(outputs.pred_masks.device)
            tgt_masks_low = interpolate(
                tgt_masks[None].float(),
                size=(mask_h, mask_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            cost_mask = _dice_cost(outputs.pred_masks[i], tgt_masks_low)

            cost = (
                self.lambda_cls * cost_cls
                + self.lambda_l1 * cost_l1
                + self.lambda_giou * cost_giou
                + self.lambda_mask * cost_mask
            )
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            results.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))
        return results
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/test_matching.py -v`
Expected: PASS — 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/matching.py tests/unit/test_matching.py
git commit -m "feat(models): implement HungarianMatcher in matching.py"
```

---

## Task 5: Inspect Meta output dict → implement `meta_to_canonical` + rewrite stub

**Files:**
- Modify: `src/esam3/models/matching.py`
- Rewrite: `tests/fixtures/tiny_sam3_stub.py`
- Create: `tests/unit/test_meta_to_canonical.py`

This task has a **real-model inspection sub-step** because the spec deliberately defers the
exact Meta output key names to implementation time (§12).

- [ ] **Step 1: Inspect Meta's build helper signature**

Run: `uv run python -c "import sam3; help(sam3)"` (or `dir(sam3)`).
Identify the build/load function. Typical names: `build_sam3`, `build_sam`, `load_model`.
Record the exact name; call it `BUILD_FN` below.

- [ ] **Step 2: Inspect Meta's forward output keys**

Create a one-shot inspection script (do NOT commit it):

```python
# tmp_inspect.py — DELETE after recording output
import torch
import sam3

model = sam3.BUILD_FN(checkpoint="models/sam3.1/sam3.1_multiplex.pt", device="cpu")  # adjust kwargs as needed
model.eval()

# Try a tiny forward — adjust to Meta's expected signature.
image = torch.zeros(1, 3, 1008, 1008)
with torch.no_grad():
    outputs = model.image_forward(image, text_prompts=["cat"])  # signature placeholder
print(type(outputs))
if isinstance(outputs, dict):
    for k, v in outputs.items():
        print(k, getattr(v, "shape", type(v)))
else:
    for attr in dir(outputs):
        if not attr.startswith("_"):
            v = getattr(outputs, attr)
            print(attr, getattr(v, "shape", type(v)))
```

Run: `uv run python tmp_inspect.py`
Record the printed key names and shapes. You will use them to fill in `meta_to_canonical`.
Delete `tmp_inspect.py` afterward.

If Meta's API differs from the signature guess above, consult `sam3`'s README for the actual
inference entrypoint. The output should contain (under some names):
- per-query class logits (B, Q, C+1)
- per-query boxes (B, Q, 4)
- per-query low-res mask logits (B, Q, 288, 288)
- per-query presence/objectness logits (B, Q)

- [ ] **Step 3: Rewrite `TinySam3Stub` to mimic Meta's output keys**

Rewrite `tests/fixtures/tiny_sam3_stub.py`. Replace the existing flat-dict output with a
dict using **the exact keys you recorded in Step 2**. Below is a template — replace
`<META_KEY_*>` placeholders with the real key names. Keep the stub small so tests stay fast.

```python
"""A tiny `nn.Module` that mimics Meta sam3's image-forward output dict.

Used to unit-test the Sam3Wrapper, meta_to_canonical adapter, and the loss
pipeline without loading the real ~3.5 GB checkpoint. Output key names match
Meta's contract as observed in spec/model-loading Task 5.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinySam3Stub(nn.Module):
    """Returns Meta-shaped output dict given image + prompts.

    Q = number of queries (default 2 for fast tests).
    C = number of classes (default 2).
    """

    def __init__(self, num_queries: int = 2, num_classes: int = 2, mask_size: int = 16) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.mask_size = mask_size
        # A single trainable param so optimizers have something to update.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, prompts: Any) -> dict[str, torch.Tensor]:
        del prompts  # ignored by the stub
        b = image.shape[0] if image.ndim == 4 else 1
        q, c, m = self.num_queries, self.num_classes, self.mask_size
        # Replace the keys below with the exact Meta keys recorded in Task 5 Step 2.
        return {
            "<META_KEY_CLASS_LOGITS>": torch.zeros(b, q, c + 1) + self.dummy,
            "<META_KEY_PRED_BOXES>": torch.zeros(b, q, 4) + self.dummy,
            "<META_KEY_PRED_MASKS>": torch.zeros(b, q, m, m) + self.dummy,
            "<META_KEY_PRESENCE>": torch.zeros(b, q) + self.dummy,
        }
```

- [ ] **Step 4: Implement `meta_to_canonical`**

In `src/esam3/models/matching.py`, replace the placeholder `meta_to_canonical` with the
real implementation. Substitute `<META_KEY_*>` with the actual key names from Step 2:

```python
def meta_to_canonical(outputs: dict) -> CanonicalOutputs:
    """Convert Meta sam3's native output dict to CanonicalOutputs.

    SINGLE point of contact for Meta key names. Update only here if Meta
    renames an output field.
    """
    return CanonicalOutputs(
        class_logits=outputs["<META_KEY_CLASS_LOGITS>"],
        pred_boxes=outputs["<META_KEY_PRED_BOXES>"],
        pred_masks=outputs["<META_KEY_PRED_MASKS>"],
        presence=outputs["<META_KEY_PRESENCE>"],
    )
```

- [ ] **Step 5: Write tests for `meta_to_canonical`**

Create `tests/unit/test_meta_to_canonical.py`:

```python
"""Unit tests for meta_to_canonical adapter."""

from __future__ import annotations

import torch

from esam3.models.matching import CanonicalOutputs, meta_to_canonical
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_adapter_round_trips_stub_output() -> None:
    stub = TinySam3Stub(num_queries=3, num_classes=2, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    raw = stub(image, prompts=None)
    canonical = meta_to_canonical(raw)
    assert isinstance(canonical, CanonicalOutputs)
    assert canonical.class_logits.shape == (2, 3, 3)  # B, Q, C+1
    assert canonical.pred_boxes.shape == (2, 3, 4)
    assert canonical.pred_masks.shape == (2, 3, 16, 16)
    assert canonical.presence.shape == (2, 3)
```

- [ ] **Step 6: Run tests — verify they pass**

Run: `uv run pytest tests/unit/test_meta_to_canonical.py tests/unit/test_matching.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/esam3/models/matching.py tests/fixtures/tiny_sam3_stub.py tests/unit/test_meta_to_canonical.py
git commit -m "feat(models): implement meta_to_canonical, rewrite TinySam3Stub for Meta keys"
```

---

## Task 6: Implement `mask_loss`

**Files:**
- Rewrite: `src/esam3/models/losses.py`
- Create: `tests/unit/test_losses.py`

- [ ] **Step 1: Write failing test for `mask_loss`**

Create `tests/unit/test_losses.py`:

```python
"""Unit tests for per-component losses + total_loss in models/losses.py."""

from __future__ import annotations

import torch

from esam3.models.losses import (
    box_loss,
    class_loss,
    mask_loss,
    objectness_loss,
)


def test_mask_loss_zero_on_perfect_match() -> None:
    # Predictions are large positive logits where the target is 1, large negative elsewhere.
    pred = torch.full((2, 32, 32), -10.0)
    pred[:, :16, :] = 10.0
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_mask_loss_positive_when_wrong() -> None:
    pred = torch.zeros(2, 32, 32)  # neutral predictions
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.item() > 0.0


def test_mask_loss_upsamples_pred_to_target_resolution() -> None:
    # Pred at 16x16, target at 32x32 — function must upsample.
    pred = torch.zeros(2, 16, 16)
    target = torch.zeros(2, 32, 32)
    loss = mask_loss(pred, target)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py::test_mask_loss_zero_on_perfect_match -v`
Expected: FAIL — `mask_loss` is still the NotImplementedError stub.

- [ ] **Step 3: Implement `mask_loss`**

Replace the contents of `src/esam3/models/losses.py` with:

```python
"""SAM 3.1 training losses (DETR-standard mix)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate

from esam3.config.schema import LossConfig
from esam3.data.base import Instance
from esam3.models.matching import CanonicalOutputs, HungarianMatcher, meta_to_canonical


def _dice_loss(pred_logits: Tensor, target: Tensor) -> Tensor:
    p = pred_logits.sigmoid().flatten(1)
    t = target.flatten(1).float()
    num = 2 * (p * t).sum(-1) + 1.0
    den = p.sum(-1) + t.sum(-1) + 1.0
    return (1.0 - num / den).mean()


def mask_loss(pred: Tensor, target: Tensor) -> Tensor:
    """0.5 * dice + 0.5 * BCE on matched mask pairs.

    `pred` and `target` are (N, H_p, W_p) and (N, H_t, W_t). If shapes differ,
    `pred` is bilinear-upsampled to target resolution.
    """
    if pred.shape[-2:] != target.shape[-2:]:
        pred = interpolate(
            pred[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    bce = binary_cross_entropy_with_logits(pred, target.float())
    dice = _dice_loss(pred, target)
    return 0.5 * dice + 0.5 * bce
```

- [ ] **Step 4: Run mask_loss tests — verify they pass**

Run: `uv run pytest tests/unit/test_losses.py -k mask_loss -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement mask_loss (dice + BCE) with auto-upsample"
```

---

## Task 7: Implement `box_loss`

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_losses.py`:

```python
def test_box_loss_zero_on_perfect_match() -> None:
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss = box_loss(pred, target)
    assert loss.item() < 1e-4


def test_box_loss_positive_when_offset() -> None:
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    loss = box_loss(pred, target)
    assert loss.item() > 0.5
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k box_loss -v`
Expected: FAIL — `box_loss` still raises NotImplementedError.

- [ ] **Step 3: Implement `box_loss`**

Append to `src/esam3/models/losses.py`:

```python
def _box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _giou_pairwise(b1: Tensor, b2: Tensor) -> Tensor:
    """Element-wise GIoU between two (N, 4) tensors in xyxy."""
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(b1[:, :2], b2[:, :2])
    rb_c = torch.max(b1[:, 2:], b2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def box_loss(pred: Tensor, target: Tensor) -> Tensor:
    """smoothL1 + (1 - GIoU) on matched box pairs. Boxes in normalized cxcywh."""
    smooth_l1 = torch.nn.functional.smooth_l1_loss(pred, target, reduction="mean")
    giou = _giou_pairwise(_box_cxcywh_to_xyxy(pred), _box_cxcywh_to_xyxy(target))
    return smooth_l1 + (1.0 - giou).mean()
```

- [ ] **Step 4: Run box_loss tests — verify they pass**

Run: `uv run pytest tests/unit/test_losses.py -k box_loss -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement box_loss (smoothL1 + 1-GIoU)"
```

---

## Task 8: Implement `objectness_loss` (focal CE on presence)

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_losses.py`:

```python
def test_objectness_loss_zero_when_predictions_agree() -> None:
    # presence logit large positive where matched_mask is 1, large negative elsewhere
    presence = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(presence, matched)
    assert loss.item() < 0.05


def test_objectness_loss_high_when_predictions_invert() -> None:
    presence = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(presence, matched)
    assert loss.item() > 1.0
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k objectness_loss -v`
Expected: FAIL.

- [ ] **Step 3: Implement `objectness_loss`**

Append to `src/esam3/models/losses.py`:

```python
def _focal_bce(
    logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> Tensor:
    """Sigmoid focal BCE, mean-reduced. logits and targets are (...,)."""
    p = logits.sigmoid()
    ce = binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def objectness_loss(
    presence: Tensor,
    matched_mask: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Focal CE on the presence logit for every query (B, Q).

    matched_mask: bool (B, Q) — True for queries assigned to a target.
    """
    return _focal_bce(presence, matched_mask.float(), gamma=gamma, alpha=alpha)
```

- [ ] **Step 4: Run objectness_loss tests — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -k objectness_loss -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement objectness_loss (focal BCE on presence)"
```

---

## Task 9: Implement `class_loss` (focal CE)

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_losses.py`:

```python
def test_class_loss_low_when_correct() -> None:
    # 2 classes + no-object; logits strongly favor class 0 for both queries.
    logits = torch.tensor([[[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]]])
    targets = torch.tensor([[0, 0]], dtype=torch.long)
    loss = class_loss(logits, targets)
    assert loss.item() < 0.05


def test_class_loss_higher_when_wrong() -> None:
    logits = torch.tensor([[[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]]])
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    loss = class_loss(logits, targets)
    assert loss.item() > 1.0
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k class_loss -v`
Expected: FAIL — `class_loss` not yet defined.

- [ ] **Step 3: Implement `class_loss`**

Append to `src/esam3/models/losses.py`:

```python
def class_loss(
    class_logits: Tensor,
    target_class_ids: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Multi-class focal CE over (B, Q, C+1) against (B, Q) integer targets.

    Target index C means "no-object" (added by the gather helper for unmatched queries).
    """
    b, q, cp1 = class_logits.shape
    log_probs = torch.nn.functional.log_softmax(class_logits, dim=-1)
    targets_one_hot = torch.nn.functional.one_hot(target_class_ids, num_classes=cp1).float()
    probs = log_probs.exp()
    focal = (1.0 - (probs * targets_one_hot).sum(-1)).pow(gamma)
    ce = -(targets_one_hot * log_probs).sum(-1)
    alpha_t = torch.where(
        target_class_ids == cp1 - 1,
        torch.full_like(focal, 1.0 - alpha),
        torch.full_like(focal, alpha),
    )
    return (alpha_t * focal * ce).mean()
```

- [ ] **Step 4: Run class_loss tests — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -k class_loss -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement class_loss (multi-class focal CE)"
```

---

## Task 10: Implement `total_loss` + private gather helpers

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_losses.py`:

```python
def test_total_loss_returns_all_components() -> None:
    from esam3.config.schema import LossConfig
    from esam3.data.base import Instance
    from tests.fixtures.tiny_sam3_stub import TinySam3Stub

    stub = TinySam3Stub(num_queries=4, num_classes=2, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    raw = stub(image, prompts=None)

    targets = [[Instance(
        mask=torch.zeros(32, 32),
        class_id=0,
        box=torch.tensor([0.5, 0.5, 0.2, 0.2]),
    )]]
    losses = total_loss(raw, targets, LossConfig())
    assert set(losses.keys()) == {"total", "mask", "box", "obj", "cls"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_total_loss_total_equals_weighted_sum() -> None:
    from esam3.config.schema import LossConfig
    from esam3.data.base import Instance
    from tests.fixtures.tiny_sam3_stub import TinySam3Stub

    stub = TinySam3Stub(num_queries=4, num_classes=2, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    raw = stub(image, prompts=None)
    targets = [[Instance(
        mask=torch.zeros(32, 32),
        class_id=0,
        box=torch.tensor([0.5, 0.5, 0.2, 0.2]),
    )]]
    cfg = LossConfig()
    losses = total_loss(raw, targets, cfg)
    expected = (
        cfg.w_mask * losses["mask"]
        + cfg.w_box * losses["box"]
        + cfg.w_obj * losses["obj"]
        + cfg.w_cls * losses["cls"]
    )
    assert torch.allclose(losses["total"], expected, atol=1e-6)


def test_total_loss_handles_empty_targets() -> None:
    from esam3.config.schema import LossConfig
    from tests.fixtures.tiny_sam3_stub import TinySam3Stub

    stub = TinySam3Stub(num_queries=4, num_classes=2, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    raw = stub(image, prompts=None)
    losses = total_loss(raw, [[]], LossConfig())
    # mask + box should be zero when no matches; obj + cls still defined.
    assert losses["mask"].item() == 0.0
    assert losses["box"].item() == 0.0
    assert torch.isfinite(losses["obj"])
    assert torch.isfinite(losses["cls"])
```

- [ ] **Step 2: Update the test_losses.py import block**

At the top of `tests/unit/test_losses.py`, ensure the imports include `total_loss`:

```python
from esam3.models.losses import (
    box_loss,
    class_loss,
    mask_loss,
    objectness_loss,
    total_loss,
)
```

- [ ] **Step 3: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k total_loss -v`
Expected: FAIL — `total_loss` still raises NotImplementedError.

- [ ] **Step 4: Implement `total_loss` and gather helpers**

Append to `src/esam3/models/losses.py`:

```python
def _gather_matched_boxes_masks(
    canonical: CanonicalOutputs,
    targets: list[list[Instance]],
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Concatenate matched (pred_box, tgt_box, pred_mask, tgt_mask) across the batch."""
    pred_boxes, tgt_boxes, pred_masks, tgt_masks = [], [], [], []
    for i, (pred_idx, tgt_idx) in enumerate(indices):
        if pred_idx.numel() == 0:
            continue
        pred_boxes.append(canonical.pred_boxes[i, pred_idx])
        tgt_boxes.append(torch.stack([targets[i][j].box for j in tgt_idx.tolist()]).to(
            canonical.pred_boxes.device
        ))
        pred_masks.append(canonical.pred_masks[i, pred_idx])
        tgt_masks.append(torch.stack([targets[i][j].mask for j in tgt_idx.tolist()]).to(
            canonical.pred_masks.device
        ))
    if not pred_boxes:
        empty = canonical.pred_boxes.new_zeros((0, 4))
        empty_mask = canonical.pred_masks.new_zeros((0, 1, 1))
        return empty, empty, empty_mask, empty_mask
    return (
        torch.cat(pred_boxes),
        torch.cat(tgt_boxes),
        torch.cat(pred_masks),
        torch.cat(tgt_masks),
    )


def _matched_query_mask(
    canonical: CanonicalOutputs,
    indices: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Bool tensor (B, Q): True where a query is matched to some target."""
    b, q, _ = canonical.class_logits.shape
    mask = torch.zeros((b, q), dtype=torch.bool, device=canonical.class_logits.device)
    for i, (pred_idx, _) in enumerate(indices):
        if pred_idx.numel() > 0:
            mask[i, pred_idx] = True
    return mask


def _full_class_targets(
    canonical: CanonicalOutputs,
    targets: list[list[Instance]],
    indices: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """(B, Q) long tensor: matched class id for matched queries, C (no-object) elsewhere."""
    b, q, cp1 = canonical.class_logits.shape
    no_object = cp1 - 1
    out = torch.full((b, q), no_object, dtype=torch.long, device=canonical.class_logits.device)
    for i, (pred_idx, tgt_idx) in enumerate(indices):
        for p, t in zip(pred_idx.tolist(), tgt_idx.tolist(), strict=True):
            out[i, p] = targets[i][t].class_id
    return out


def total_loss(
    outputs: dict,
    targets: list[list[Instance]],
    cfg: LossConfig,
) -> dict[str, Tensor]:
    """Run matching, compute per-component losses, return dict with 'total' summed."""
    canonical = meta_to_canonical(outputs)
    matcher = HungarianMatcher(
        lambda_cls=cfg.matcher_weights.lambda_cls,
        lambda_l1=cfg.matcher_weights.lambda_l1,
        lambda_giou=cfg.matcher_weights.lambda_giou,
        lambda_mask=cfg.matcher_weights.lambda_mask,
    )
    indices = matcher(canonical, targets)

    pred_boxes_m, tgt_boxes_m, pred_masks_m, tgt_masks_m = _gather_matched_boxes_masks(
        canonical, targets, indices
    )
    matched_mask = _matched_query_mask(canonical, indices)
    full_targets = _full_class_targets(canonical, targets, indices)

    zero = canonical.class_logits.new_zeros(())
    losses: dict[str, Tensor] = {
        "mask": mask_loss(pred_masks_m, tgt_masks_m) if pred_masks_m.numel() > 0 else zero,
        "box": box_loss(pred_boxes_m, tgt_boxes_m) if pred_boxes_m.numel() > 0 else zero,
        "obj": objectness_loss(
            canonical.presence, matched_mask, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha
        ),
        "cls": class_loss(
            canonical.class_logits, full_targets, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha
        ),
    }
    losses["total"] = (
        cfg.w_mask * losses["mask"]
        + cfg.w_box * losses["box"]
        + cfg.w_obj * losses["obj"]
        + cfg.w_cls * losses["cls"]
    )
    return losses
```

- [ ] **Step 5: Run all loss tests — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement total_loss + matched-pair gather helpers"
```

---

## Task 11: Prune `test_stubs_raise.py` (model + loss stubs now implemented)

**Files:**
- Modify: `tests/unit/test_stubs_raise.py`

- [ ] **Step 1: Remove the model-stub and loss-stub assertions**

In `tests/unit/test_stubs_raise.py`, replace the entire `def test_model_stubs()` function with this version that **only** asserts PEFT/eval/train stubs (not load_sam31/losses, which are now real):

```python
# DELETE the existing test_model_stubs function entirely.
# The module-level imports of load_sam31, mask_loss, box_loss, objectness_loss,
# total_loss should also be removed since they're no longer used here.
```

Concretely, edit the file to:

1. Remove the imports:
   - `from esam3.models.losses import box_loss, mask_loss, objectness_loss, total_loss`
   - `from esam3.models.sam3 import load_sam31`
2. Delete the entire `def test_model_stubs() -> None: ...` function.
3. Keep `test_peft_stubs`, `test_eval_stubs`, `test_train_stubs`, `test_trainer_fit_stub` unchanged.

- [ ] **Step 2: Run the suite — verify no regression**

Run: `uv run pytest tests/unit -v`
Expected: every test that previously passed still passes; the deleted `test_model_stubs` is gone from the report.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_stubs_raise.py
git commit -m "test: drop NotImplementedError asserts for load_sam31 + losses"
```

---

## Task 12: Implement `Sam3Wrapper`

**Files:**
- Rewrite: `src/esam3/models/sam3.py` (Wrapper only; loader still stubbed)
- Create: `tests/unit/test_sam3_wrapper.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_sam3_wrapper.py`:

```python
"""Unit tests for Sam3Wrapper that exercise routing without a real model."""

from __future__ import annotations

import pytest
import torch

from esam3.data.base import BoxPrompts, TextPrompts
from esam3.models.sam3 import Sam3Wrapper
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_wrapper_passes_through_text_prompts() -> None:
    stub = TinySam3Stub(num_queries=2, num_classes=2)
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat", "dog"])]
    out = wrapper(image, prompts)
    assert isinstance(out, dict)
    # Output dict should be Meta-shaped (passed through from the stub unchanged).
    assert len(out) == 4


def test_wrapper_passes_through_box_prompts() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    prompts = [BoxPrompts(
        boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        class_ids=torch.tensor([0]),
    )]
    out = wrapper(image, prompts)
    assert isinstance(out, dict)


def test_wrapper_rejects_mixed_prompt_variants() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [
        TextPrompts(classes=["cat"]),
        BoxPrompts(boxes=torch.zeros(1, 4), class_ids=torch.zeros(1, dtype=torch.long)),
    ]
    with pytest.raises(ValueError, match="same prompt variant"):
        wrapper(image, prompts)


def test_wrapper_rejects_more_than_16_prompts() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    prompts = [TextPrompts(classes=[f"c{i}" for i in range(17)])]
    with pytest.raises(ValueError, match="multiplex cap"):
        wrapper(image, prompts)
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_sam3_wrapper.py -v`
Expected: FAIL — `Sam3Wrapper` not importable.

- [ ] **Step 3: Implement `Sam3Wrapper`**

Replace `src/esam3/models/sam3.py` with the version below (loader still stubbed — Task 13 fills it in):

```python
"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import BoxPrompts, Prompts, TextPrompts

_MULTIPLEX_CAP = 16


class Sam3Wrapper(nn.Module):
    """Thin wrapper around Meta's sam3 model. Returns Meta's native output dict unchanged."""

    def __init__(self, model: nn.Module, image_size: int = 1008, mask_size: int = 288) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size

    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Any]:
        self._validate_prompts(prompts)
        # Routing is delegated to the underlying Meta model. Whether prompts go through
        # the text encoder or the geometry encoder is determined by their variant.
        return self.model(images, prompts)

    @staticmethod
    def _validate_prompts(prompts: list[Prompts]) -> None:
        if not prompts:
            return
        first = type(prompts[0])
        for p in prompts:
            if type(p) is not first:
                raise ValueError(
                    "All prompts in a batch must be the same prompt variant "
                    "(TextPrompts or BoxPrompts), not mixed."
                )
            count = len(p.classes) if isinstance(p, TextPrompts) else p.boxes.shape[0]
            if count > _MULTIPLEX_CAP:
                raise ValueError(
                    f"Prompt count {count} exceeds multiplex cap of {_MULTIPLEX_CAP}."
                )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's sam3 package. Implementation lands in Task 13."""
    raise NotImplementedError("filled in by Task 13 of spec/model-loading")
```

- [ ] **Step 4: Run wrapper tests — verify pass**

Run: `uv run pytest tests/unit/test_sam3_wrapper.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/sam3.py tests/unit/test_sam3_wrapper.py
git commit -m "feat(models): implement Sam3Wrapper with prompt validation"
```

---

## Task 13: Implement `load_sam31`

**Files:**
- Modify: `src/esam3/models/sam3.py`

- [ ] **Step 1: Inspect Meta's build helper signature**

Refer back to Task 5 Step 1's recording. The build helper is called `BUILD_FN`; its exact
kwargs (e.g. `checkpoint=`, `device=`, `dtype=`) are what you'll use below.

- [ ] **Step 2: Inspect the state-dict prefix mapping**

Open the model and the checkpoint, then diff the key namespaces:

```bash
uv run python <<'PY'
import torch, sam3
model = sam3.BUILD_FN(device="cpu")
ckpt = torch.load("models/sam3.1/sam3.1_multiplex.pt", map_location="cpu", weights_only=True)

model_keys = set(model.state_dict().keys())
ckpt_flat = {}
for root, sub in ckpt.items():
    for k, v in sub.items():
        ckpt_flat[f"{root}.{k}"] = v
ckpt_keys = set(ckpt_flat.keys())

print("In ckpt but renamed in model (sample):")
for k in sorted(ckpt_keys - model_keys)[:10]:
    print(" ", k)
print("In model but missing in ckpt (sample):")
for k in sorted(model_keys - ckpt_keys)[:10]:
    print(" ", k)
PY
```

Record the prefix transformation needed (likely identity once `detector.`/`tracker.` are
kept as-is, or a strip if the model uses unprefixed names). Use this transformation in the
`_load_state_dict` helper below.

- [ ] **Step 3: Replace the `load_sam31` stub with the real implementation**

In `src/esam3/models/sam3.py`, replace `load_sam31` with the implementation below.
Substitute `sam3.BUILD_FN` and any kwargs with the real ones inspected in Step 1.

```python
import logging
from pathlib import Path

import sam3  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_BACKBONE_PREFIX = "detector.backbone"  # update post-Step 2 if the post-flatten prefix differs


def _resolve_checkpoint_path(cfg: ModelConfig) -> Path:
    if cfg.local_dir is None:
        raise FileNotFoundError(
            f"ModelConfig.local_dir is None and Hub fetch is not implemented; "
            f"set local_dir to a directory containing {cfg.checkpoint_file}. "
            f"To download: `huggingface-cli download {cfg.name} --local-dir models/sam3.1`."
        )
    path = Path(cfg.local_dir) / cfg.checkpoint_file
    if not path.exists():
        raise FileNotFoundError(
            f"SAM 3.1 checkpoint not found at {path}. "
            f"Run: huggingface-cli download {cfg.name} --local-dir {cfg.local_dir}"
        )
    return path


def _flatten_state_dict(ckpt: dict) -> dict[str, Tensor]:
    """Convert {'detector': {...}, 'tracker': {...}} into a single namespace.

    The exact prefix transformation is pinned at Task 13 Step 2.
    """
    flat: dict[str, Tensor] = {}
    for root, sub in ckpt.items():
        for k, v in sub.items():
            flat[f"{root}.{k}"] = v
    return flat


def _apply_grad_checkpointing(model: nn.Module) -> None:
    """Hook gradient checkpointing on the ViT encoder."""
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        return
    # Fallback: torch.utils.checkpoint.checkpoint_wrapper on encoder blocks.
    from torch.utils.checkpoint import checkpoint as _checkpoint  # noqa: F401
    logger.warning(
        "Meta sam3 model does not expose set_grad_checkpointing; gradient checkpointing "
        "fallback not yet implemented for this revision."
    )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's sam3 package + .pt remap.

    Returns a Sam3Wrapper that exposes Meta's native forward output dict.
    """
    ckpt_path = _resolve_checkpoint_path(cfg)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

    try:
        # SUBSTITUTE the actual build entrypoint name and kwargs:
        model = sam3.BUILD_FN(device=device, dtype=dtype)
    except AttributeError as e:  # pragma: no cover - depends on package state
        raise ImportError(
            "Meta sam3 package does not expose the expected build entrypoint. "
            "Verify the pinned commit in pyproject.toml matches the spec."
        ) from e

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = _flatten_state_dict(raw_ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    backbone_missing = [k for k in missing if k.startswith(_BACKBONE_PREFIX)]
    if backbone_missing:
        raise RuntimeError(
            f"Catastrophic: {len(backbone_missing)} backbone keys missing from checkpoint: "
            f"{backbone_missing[:5]}{'...' if len(backbone_missing) > 5 else ''}"
        )
    if missing:
        logger.warning("Non-backbone missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    if cfg.gradient_checkpointing:
        _apply_grad_checkpointing(model)

    return Sam3Wrapper(model, image_size=1008, mask_size=288)
```

- [ ] **Step 4: Smoke-test with a missing checkpoint**

Run:
```bash
uv run python -c "
from esam3.config.schema import ModelConfig
from esam3.models.sam3 import load_sam31
try:
    load_sam31(ModelConfig(local_dir='/nonexistent'))
except FileNotFoundError as e:
    print('OK:', e)
"
```
Expected: prints `OK: SAM 3.1 checkpoint not found at /nonexistent/sam3.1_multiplex.pt. Run: ...`.

- [ ] **Step 5: Run unit suite — verify no regression**

Run: `uv run pytest tests/unit -v`
Expected: all PASS. (The `load_sam31` unit smoke ran above; integration tests come in Task 14.)

- [ ] **Step 6: Commit**

```bash
git add src/esam3/models/sam3.py
git commit -m "feat(models): implement load_sam31 via Meta sam3 + .pt remap"
```

---

## Task 14: Add gated integration test for real checkpoint loading

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/integration/test_load_sam31_real.py`

- [ ] **Step 1: Register the `requires_checkpoint` marker**

Open `tests/conftest.py`. If it does not already configure markers, add:

```python
import pathlib

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_checkpoint: skip unless models/sam3.1/sam3.1_multiplex.pt exists",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    ckpt = pathlib.Path("models/sam3.1/sam3.1_multiplex.pt")
    skip = pytest.mark.skip(reason="real SAM 3.1 checkpoint not present locally")
    for item in items:
        if "requires_checkpoint" in item.keywords and not ckpt.exists():
            item.add_marker(skip)
```

If `tests/conftest.py` already has `pytest_configure` or `pytest_collection_modifyitems`,
**merge** the new logic in rather than overwriting.

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_load_sam31_real.py`:

```python
"""Integration test: load real SAM 3.1 checkpoint and run a forward pass.

Skipped automatically unless `models/sam3.1/sam3.1_multiplex.pt` is present.
"""

from __future__ import annotations

import pytest
import torch

from esam3.config.schema import ModelConfig
from esam3.data.base import TextPrompts
from esam3.models.matching import meta_to_canonical
from esam3.models.sam3 import Sam3Wrapper, load_sam31


@pytest.mark.requires_checkpoint
def test_load_sam31_returns_wrapper() -> None:
    cfg = ModelConfig(device="cpu", gradient_checkpointing=False, dtype="float16")
    wrapper = load_sam31(cfg)
    assert isinstance(wrapper, Sam3Wrapper)


@pytest.mark.requires_checkpoint
def test_load_sam31_forward_to_canonical() -> None:
    cfg = ModelConfig(device="cpu", gradient_checkpointing=False, dtype="float16")
    wrapper = load_sam31(cfg)
    image = torch.zeros(1, 3, 1008, 1008, dtype=torch.float16)
    with torch.no_grad():
        outputs = wrapper(image, [TextPrompts(classes=["cat"])])
    canonical = meta_to_canonical(outputs)
    assert canonical.class_logits.dim() == 3
    assert canonical.pred_boxes.shape[-1] == 4
    assert canonical.pred_masks.shape[-1] == 288
    assert canonical.presence.dim() == 2
```

- [ ] **Step 3: Run the integration test (will skip on CI, run if you have the checkpoint)**

Run: `uv run pytest tests/integration/test_load_sam31_real.py -v`
Expected: PASS if the checkpoint is present locally; otherwise SKIPPED with the
"real SAM 3.1 checkpoint not present locally" reason.

If it fails on a real run, the most likely cause is a Meta API drift; revisit Task 13
Step 1 to update the build entrypoint name and kwargs, or Task 5 Step 2 to update output
key names.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/integration/test_load_sam31_real.py
git commit -m "test(integration): add gated load_sam31 + forward smoke test"
```

---

## Task 15: Fix example configs (image_size, normalization, model paths, loss block)

**Files:**
- Modify: `configs/examples/coco_text_lora.yaml`
- Modify: `configs/examples/coco_bbox_qlora.yaml`

- [ ] **Step 1: Inspect the current example configs**

Run: `cat configs/examples/coco_text_lora.yaml configs/examples/coco_bbox_qlora.yaml`
Note the current values of `data.image_size`, `data.normalize`, and the absence of
`model.local_dir` / `model.checkpoint_file` / `train.loss`.

- [ ] **Step 2: Patch `coco_text_lora.yaml`**

In `configs/examples/coco_text_lora.yaml`:

- Change `data.image_size: 1024` → `data.image_size: 1008`.
- Change `data.normalize.mean: [0.485, 0.456, 0.406]` → `data.normalize.mean: [0.5, 0.5, 0.5]`.
- Change `data.normalize.std: [0.229, 0.224, 0.225]` → `data.normalize.std: [0.5, 0.5, 0.5]`.
- Under `model:`, add (preserving existing fields):
  ```yaml
  model:
    name: facebook/sam3.1
    local_dir: models/sam3.1
    checkpoint_file: sam3.1_multiplex.pt
  ```
- Under `train:`, add a commented loss block at the bottom of the section:
  ```yaml
    # Loss-mix weights (defaults shown — uncomment to override):
    # loss:
    #   w_mask: 1.0
    #   w_box: 5.0
    #   w_obj: 1.0
    #   w_cls: 2.0
    #   focal_gamma: 2.0
    #   focal_alpha: 0.25
    #   matcher_weights:
    #     lambda_cls: 2.0
    #     lambda_l1: 5.0
    #     lambda_giou: 2.0
    #     lambda_mask: 5.0
  ```

- [ ] **Step 3: Patch `coco_bbox_qlora.yaml`**

Apply the same four changes (image_size, normalize.mean, normalize.std, model block, train.loss block) to `configs/examples/coco_bbox_qlora.yaml`.

- [ ] **Step 4: Verify both configs parse**

Run:
```bash
uv run python -c "
from esam3.config.loader import load_train_config
for p in ['configs/examples/coco_text_lora.yaml', 'configs/examples/coco_bbox_qlora.yaml']:
    cfg = load_train_config(p)
    assert cfg.data.image_size == 1008
    assert cfg.data.normalize.mean == [0.5, 0.5, 0.5]
    assert cfg.model.local_dir == 'models/sam3.1'
    print('OK:', p)
"
```
Expected: prints `OK:` for both paths.

If `esam3.config.loader` does not exist under that exact name, find the entrypoint via
`grep -r 'def load.*config' src/esam3/config/` and adjust the import.

- [ ] **Step 5: Commit**

```bash
git add configs/examples/coco_text_lora.yaml configs/examples/coco_bbox_qlora.yaml
git commit -m "fix(configs): align example yamls with SAM3.1 native size + normalization"
```

---

## Task 16: Final lint, format, and full test run

**Files:** all files touched in this plan

- [ ] **Step 1: Run ruff format**

Run: `uv run ruff format src/esam3/models src/esam3/config tests`
Expected: completes; any reformatting is staged automatically.

- [ ] **Step 2: Run ruff lint**

Run: `uv run ruff check src/esam3/models src/esam3/config tests`
Expected: zero issues. Fix any reported issues inline (most likely: unused imports from
TDD scaffolding; minor unused variables in test helpers).

- [ ] **Step 3: Run mypy**

Run: `uv run mypy src/esam3/models src/esam3/config`
Expected: zero errors. If `sam3` lacks type stubs, the `import sam3  # type: ignore[import-untyped]`
in `sam3.py` should already suppress complaints; otherwise add a `[[tool.mypy.overrides]]` block
to `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = "sam3.*"
ignore_missing_imports = true
```

- [ ] **Step 4: Full unit suite**

Run: `uv run pytest tests/unit -v`
Expected: all PASS, no skips other than any pre-existing ones.

- [ ] **Step 5: Integration suite (will skip if no checkpoint)**

Run: `uv run pytest tests/integration -v`
Expected: PASS or SKIPPED (with reason "real SAM 3.1 checkpoint not present locally").

- [ ] **Step 6: Commit any lint/format/mypy fixes**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: ruff format + mypy fixes for spec/model-loading"
```

(If `git diff --cached --stat` shows no changes, skip the commit.)

- [ ] **Step 7: Push and open a PR**

```bash
git push -u origin model-loading
gh pr create --title "feat(models): SAM 3.1 loader, forward wrapper, matcher, losses" --body "$(cat <<'EOF'
## Summary
- Implements `load_sam31` (Meta `sam3` builder + `.pt` remap) and `Sam3Wrapper` (pass-through forward, prompt-variant validation, 16-prompt multiplex cap).
- Adds `models/matching.py` (`CanonicalOutputs`, `meta_to_canonical`, `HungarianMatcher` with class/L1/GIoU/dice cost).
- Implements DETR-standard losses in `models/losses.py`: dice+BCE mask, smoothL1+(1-GIoU) box, focal CE objectness, focal CE class, weighted `total_loss`.
- Adds `LossConfig` + `MatcherWeights` to the schema; extends `ModelConfig` with `local_dir` / `checkpoint_file` / `device`.
- Fixes example YAML mismatches (image_size 1024→1008, ImageNet→[0.5,0.5,0.5] normalization, model.local_dir).
- Bumps `transformers>=5.0`, adds `scipy`, adds Meta `sam3` git dep.

Reference: `docs/superpowers/specs/2026-05-16-model-loading-design.md`.

## Test plan
- [ ] `uv run pytest tests/unit -v` passes
- [ ] `uv run pytest tests/integration -v` passes locally with checkpoint present
- [ ] `uv run ruff check` and `uv run ruff format --check` clean
- [ ] `uv run mypy src/esam3/models src/esam3/config` clean
EOF
)"
```

---

## Self-review checklist (for the implementer)

After completing every task, verify:

1. **Spec coverage:** Every numbered section of the spec is implemented:
   - §3 `load_sam31` → Task 13
   - §4 `Sam3Wrapper` → Task 12
   - §5 `matching.py` → Tasks 3, 4, 5
   - §6 `losses.py` → Tasks 6–10
   - §7 Config schema → Task 2
   - §8 Example config fixes → Task 15
   - §9 Tests → embedded in each task; integration → Task 14
2. **No NotImplementedError remains** for `load_sam31`, `mask_loss`, `box_loss`,
   `objectness_loss`, `class_loss`, `total_loss`, `meta_to_canonical`, `HungarianMatcher`.
3. **All commits are small and labeled**; no "wip" or "fix" commits without context.
4. **Ruff + mypy clean** on every touched file.
5. **Spec §11 deferred items remain untouched** (PEFT, training loop, video, distributed,
   evaluation, profiling).
