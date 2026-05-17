# Model-Loading Implementation Plan — Revised

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `load_sam31`, `Sam3Wrapper`, `models/matching.py`, and `models/losses.py` so the trainer has a real model + losses for COCO image finetuning **against the actual Meta SAM 3.1 open-vocabulary head** (per-query binary score + global image presence), with the user's fixed class vocabulary supplied as text prompts.

**Why this revision:** The original plan (`docs/superpowers/plans/2026-05-16-model-loading.md`) was written before the Meta `sam3` package was inspected. Live inspection during the first execution attempt revealed a design mismatch — Meta's head is per-query binary (text-image similarity), not DETR-style multi-class, and presence is global (one token per image), not per-query. The plan's `class_loss` and per-query presence design cannot work against the real model.

**Architecture:** SAM 3.1 is an open-vocabulary detector — text prompts ARE the classes. The user has a fixed class vocabulary `[c_0, …, c_{C-1}]` known at config-load time. Training loops over classes (each class becomes a separate text prompt), runs one forward per class, and supervises:
- Matched-pair losses (mask, box) on the queries matched to per-class GT instances.
- Per-query objectness loss on `pred_logits` (binary "is this query a real detection of this class").
- Image-level presence loss on `presence_logit_dec` (binary "does this image contain any instance of this class").

No multi-class `class_loss` — discrimination across classes comes from per-prompt forward passes.

**Tech Stack:** Python 3.13, PyTorch, Meta `sam3` (git, pinned), HuggingFace `transformers ≥5.0`, `pydantic v2`, `scipy.optimize.linear_sum_assignment`, `pytest`, `ruff`.

**Reference spec:** `docs/superpowers/specs/2026-05-16-model-loading-design.md` (the schema/Sam3Wrapper API contract still applies; the loss design has been replaced by this plan).

---

## Pipeline state at the start of this revised plan

The original plan's Tasks 1–4 were completed and remain committed (`7cd25ad` is HEAD):

- ✅ Task 1: dependencies (`598651d`) — `sam3 @ 2814fa619...`, `scipy`, `transformers>=5.0`, `einops`, `[tool.hatch.metadata] allow-direct-references`.
- ✅ Task 2: schema additions (`ff86e14`, `4f6e7b8`) — `ModelConfig` extended; `LossConfig`/`MatcherWeights` added with `class`-related fields that **Task 5 of this revised plan removes**.
- ✅ Task 3: `matching.py` scaffold with `CanonicalOutputs` (`f71e3e7`) — **Task 6 of this revised plan rewrites the dataclass.**
- ✅ Task 4: `HungarianMatcher` (`b20f24c`, `7cd25ad`) — **Task 9 of this revised plan drops the class cost term.**

Tasks 5–15 of the original plan are SUPERSEDED by Tasks 5–20 below.

---

## Pre-flight checks

```bash
# Confirm Task 1 is committed and sam3 imports
git log --oneline -5
uv run python -c "import sam3; print(sam3.__file__)"

# Confirm checkpoint is present locally (3.5 GB)
ls -lh models/sam3.1/sam3.1_multiplex.pt
ls -lh models/sam3.1/merges.txt
```

If `sam3` is missing or the checkpoint is missing, stop and resolve before proceeding.

**Project pytest enforces `--cov-fail-under=80` via `addopts`.** Append `--no-cov` to every targeted `pytest` invocation during TDD. The full-suite run in Task 20 lets coverage gate be enforced.

---

## File map (what gets touched)

| File | Action | Owning task |
| --- | --- | --- |
| `src/esam3/config/schema.py` | Modify | 5 |
| `tests/unit/test_loss_config.py` | Modify | 5 |
| `src/esam3/models/matching.py` | Rewrite parts | 6, 7, 9 |
| `tests/unit/test_matching.py` | Modify | 9 |
| `tests/unit/test_meta_to_canonical.py` | Create | 7 |
| `tests/fixtures/tiny_sam3_stub.py` | Rewrite | 8 |
| `tests/unit/test_fixtures.py` | Modify | 8 |
| `src/esam3/models/losses.py` | Rewrite | 10–14 |
| `tests/unit/test_losses.py` | Create | 10–14 |
| `tests/unit/test_stubs_raise.py` | Modify | 15 |
| `src/esam3/models/sam3.py` | Rewrite | 16, 17 |
| `tests/unit/test_sam3_wrapper.py` | Create | 16 |
| `tests/integration/test_load_sam31_real.py` | Create | 18 |
| `tests/conftest.py` | Modify | 18 |
| `configs/examples/coco_text_lora.yaml` | Modify | 19 |
| `configs/examples/coco_bbox_qlora.yaml` | Modify | 19 |
| `pyproject.toml` (mypy override) | Possibly modify | 20 |

---

## Task 5: Revise LossConfig + MatcherWeights schema

**Why:** No multi-class head means no `w_cls` / `lambda_cls`. Add `w_presence` for the new image-level presence loss.

**Files:**
- Modify: `src/esam3/config/schema.py` (`MatcherWeights`, `LossConfig`)
- Modify: `tests/unit/test_loss_config.py` (drop `w_cls` asserts; add `w_presence` asserts)

- [ ] **Step 1: Update tests first**

Replace the body of `tests/unit/test_loss_config.py` with:

```python
"""Unit tests for LossConfig + MatcherWeights schemas (revised plan)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import LossConfig, MatcherWeights, TrainConfig


def test_matcher_weights_defaults() -> None:
    w = MatcherWeights()
    assert w.lambda_l1 == 5.0
    assert w.lambda_giou == 2.0
    assert w.lambda_mask == 5.0
    # No lambda_cls — open-vocab head has no per-class classification.
    assert not hasattr(w, "lambda_cls")


def test_matcher_weights_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MatcherWeights(lambda_cls=2.0)  # type: ignore[call-arg]


def test_loss_config_defaults() -> None:
    cfg = LossConfig()
    assert cfg.w_mask == 1.0
    assert cfg.w_box == 5.0
    assert cfg.w_obj == 1.0
    assert cfg.w_presence == 1.0
    assert cfg.focal_gamma == 2.0
    assert cfg.focal_alpha == 0.25
    assert isinstance(cfg.matcher_weights, MatcherWeights)
    # No w_cls — open-vocab head has no per-class classification.
    assert not hasattr(cfg, "w_cls")


def test_loss_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LossConfig(w_cls=2.0)  # type: ignore[call-arg]


def test_train_config_includes_loss() -> None:
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
    assert isinstance(tc.train.loss, LossConfig)
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run pytest tests/unit/test_loss_config.py -v --no-cov`
Expected: FAIL — `MatcherWeights.lambda_cls` still defined; `LossConfig.w_cls` still defined; no `w_presence`.

- [ ] **Step 3: Edit `src/esam3/config/schema.py`**

Replace `class MatcherWeights(_Strict):` with:

```python
class MatcherWeights(_Strict):
    """Per-term cost weights for the Hungarian matcher.

    No `lambda_cls` term: SAM 3.1's open-vocab head has no multi-class
    classification logits; class identity comes from the text prompt itself,
    so matching uses only geometric (L1/GIoU) and mask (Dice) costs.
    """

    lambda_l1: PositiveFloat = 5.0
    lambda_giou: PositiveFloat = 2.0
    lambda_mask: PositiveFloat = 5.0
```

Replace `class LossConfig(_Strict):` with:

```python
class LossConfig(_Strict):
    """Loss-mix weights and focal CE params for SAM 3.1 training.

    No `w_cls`: discrimination across classes comes from running one forward
    pass per class prompt. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision applied to
    `presence_logit_dec`.
    """

    w_mask: PositiveFloat = 1.0
    w_box: PositiveFloat = 5.0
    w_obj: PositiveFloat = 1.0
    w_presence: PositiveFloat = 1.0
    matcher_weights: MatcherWeights = Field(default_factory=MatcherWeights)
    focal_gamma: PositiveFloat = 2.0
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
```

(`TrainHyperparams.loss` field is unchanged.)

- [ ] **Step 4: Run tests — verify pass**

Run: `uv run pytest tests/unit/test_loss_config.py -v --no-cov`
Expected: 5 PASS.

- [ ] **Step 5: Run full unit suite — no regression**

Run: `uv run pytest tests/unit -v --no-cov`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_loss_config.py
git commit -m "feat(config): align LossConfig/MatcherWeights with SAM3.1 open-vocab head"
```

---

## Task 6: Revise `CanonicalOutputs` dataclass

**Why:** Replace `class_logits: (B, Q, C+1)` and `presence: (B, Q)` with `obj_logits: (B, Q)` and `img_presence: (B,)` — matching Meta's real output shape after squeezing.

**Files:**
- Modify: `src/esam3/models/matching.py` (top of file: the dataclass)

- [ ] **Step 1: Replace the `CanonicalOutputs` dataclass**

In `src/esam3/models/matching.py`, replace the existing `@dataclass class CanonicalOutputs: ...` block with:

```python
@dataclass
class CanonicalOutputs:
    """Adapter output of `meta_to_canonical`. Per-class (one prompt per call).

    Shapes (B = batch size, Q = number of decoder queries):
      obj_logits:   (B, Q)         per-query binary score (text-image similarity).
                                   Positive = "this query detects an instance of the
                                   current prompt class." From Meta's `pred_logits`
                                   after squeezing the trailing size-1 dim.
      pred_boxes:   (B, Q, 4)      normalized cx,cy,w,h in [0, 1].
      pred_masks:   (B, Q, H, W)   instance mask logits; H=W=288 at 1008-px input.
      img_presence: (B,)           image-level binary score "does this image contain
                                   any instance of the current prompt class." From
                                   Meta's `presence_logit_dec` after squeezing.
    """

    obj_logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    img_presence: Tensor
```

(Leave the `meta_to_canonical` placeholder in place; Task 7 implements it.)

- [ ] **Step 2: Smoke import**

Run: `uv run python -c "from esam3.models.matching import CanonicalOutputs; print(CanonicalOutputs.__annotations__)"`
Expected: prints `{'obj_logits': Tensor, 'pred_boxes': Tensor, 'pred_masks': Tensor, 'img_presence': Tensor}`.

- [ ] **Step 3: Commit**

```bash
git add src/esam3/models/matching.py
git commit -m "feat(models): revise CanonicalOutputs for SAM3.1 per-query+image-presence head"
```

---

## Task 7: Implement `meta_to_canonical` with live-inspection verification

**Files:**
- Modify: `src/esam3/models/matching.py` (replace the placeholder `meta_to_canonical`)
- Create: `tests/unit/test_meta_to_canonical.py`

The Meta output keys were already inspected via static analysis: `pred_logits`, `pred_boxes`, `pred_masks`, `presence_logit_dec`. This task encodes them and adds a small unit test using the stub fixture (which Task 8 rewrites — order matters; do Task 7 first against the **current** stub keys, then Task 8 rewrites the stub and Task 7's test will be re-verified at the end of Task 8).

- [ ] **Step 1: Replace the placeholder `meta_to_canonical`**

In `src/esam3/models/matching.py`, find:

```python
def meta_to_canonical(outputs: dict) -> CanonicalOutputs:
    """Convert Meta sam3's native output dict to CanonicalOutputs.

    Implementation deferred to Task 5 (requires inspection of real Meta output).
    """
    raise NotImplementedError("filled in by Task 5 of spec/model-loading")
```

Replace it with:

```python
def meta_to_canonical(outputs: dict) -> CanonicalOutputs:
    """Convert Meta SAM 3.1's native output dict to CanonicalOutputs.

    SINGLE point of contact for Meta key names. Update only this function if
    Meta renames a field.

    Meta keys (from `sam3.model.sam3_image.Sam3Image.forward_grounding`):
      "pred_logits":        (B, Q, 1)  per-query text-image similarity logit.
      "pred_boxes":         (B, Q, 4)  normalized cx,cy,w,h.
      "pred_masks":         (B, Q, H, W)  instance mask logits (288×288 at 1008px).
      "presence_logit_dec": (B, 1)     single global presence logit per image.

    The trailing size-1 dims of pred_logits and presence_logit_dec are squeezed.
    """
    pred_logits: Tensor = outputs["pred_logits"]
    presence: Tensor = outputs["presence_logit_dec"]
    return CanonicalOutputs(
        obj_logits=pred_logits.squeeze(-1),
        pred_boxes=outputs["pred_boxes"],
        pred_masks=outputs["pred_masks"],
        img_presence=presence.squeeze(-1),
    )
```

- [ ] **Step 2: Create `tests/unit/test_meta_to_canonical.py`**

```python
"""Unit tests for meta_to_canonical adapter (revised plan)."""

from __future__ import annotations

import pytest
import torch

from esam3.models.matching import CanonicalOutputs, meta_to_canonical


def _raw_outputs(b: int = 2, q: int = 3, h: int = 16) -> dict:
    """Hand-crafted dict that mimics Meta's forward_grounding output shape."""
    return {
        "pred_logits": torch.randn(b, q, 1),
        "pred_boxes": torch.rand(b, q, 4),
        "pred_masks": torch.randn(b, q, h, h),
        "presence_logit_dec": torch.randn(b, 1),
    }


def test_adapter_squeezes_trailing_dims() -> None:
    raw = _raw_outputs(b=2, q=3, h=16)
    canonical = meta_to_canonical(raw)
    assert isinstance(canonical, CanonicalOutputs)
    assert canonical.obj_logits.shape == (2, 3)
    assert canonical.pred_boxes.shape == (2, 3, 4)
    assert canonical.pred_masks.shape == (2, 3, 16, 16)
    assert canonical.img_presence.shape == (2,)


def test_adapter_preserves_values() -> None:
    raw = _raw_outputs(b=1, q=2, h=8)
    canonical = meta_to_canonical(raw)
    assert torch.equal(canonical.obj_logits, raw["pred_logits"].squeeze(-1))
    assert canonical.pred_boxes is raw["pred_boxes"]
    assert canonical.pred_masks is raw["pred_masks"]
    assert torch.equal(canonical.img_presence, raw["presence_logit_dec"].squeeze(-1))


def test_adapter_raises_on_missing_key() -> None:
    raw = _raw_outputs()
    del raw["pred_masks"]
    with pytest.raises(KeyError):
        meta_to_canonical(raw)
```

- [ ] **Step 3: Run tests — verify pass**

Run: `uv run pytest tests/unit/test_meta_to_canonical.py -v --no-cov`
Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/esam3/models/matching.py tests/unit/test_meta_to_canonical.py
git commit -m "feat(models): implement meta_to_canonical for SAM3.1 open-vocab head"
```

---

## Task 8: Rewrite `TinySam3Stub` to mimic Meta's output dict

**Files:**
- Rewrite: `tests/fixtures/tiny_sam3_stub.py`
- Modify: `tests/unit/test_fixtures.py`

- [ ] **Step 1: Replace `tests/fixtures/tiny_sam3_stub.py`**

```python
"""A tiny `nn.Module` that mimics Meta SAM 3.1's image-forward output dict.

Used to unit-test the Sam3Wrapper, meta_to_canonical adapter, and the loss
pipeline without loading the real ~3.5 GB checkpoint. Output keys match
Meta's `Sam3Image.forward_grounding` contract.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinySam3Stub(nn.Module):
    """Returns Meta-shaped output dict given image + prompts.

    Q = number of decoder queries (default 4 for fast tests).
    The stub is per-class (one prompt at a time), matching Sam3Wrapper's
    single-prompt forward contract.
    """

    def __init__(self, num_queries: int = 4, mask_size: int = 16) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.mask_size = mask_size
        # One trainable param so optimizers have something to update.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, prompts: Any) -> dict[str, torch.Tensor]:
        del prompts  # ignored by the stub
        b = image.shape[0] if image.ndim == 4 else 1
        q, m = self.num_queries, self.mask_size
        return {
            "pred_logits": torch.zeros(b, q, 1) + self.dummy,
            "pred_boxes": torch.zeros(b, q, 4) + self.dummy,
            "pred_masks": torch.zeros(b, q, m, m) + self.dummy,
            "presence_logit_dec": torch.zeros(b, 1) + self.dummy,
        }
```

- [ ] **Step 2: Update `tests/unit/test_fixtures.py`**

Find `def test_stub_model_forward_returns_expected_keys(...)` and replace its body with:

```python
def test_stub_model_forward_returns_expected_keys(stub_model: TinySam3Stub) -> None:
    image = torch.zeros((2, 3, 32, 32))
    out = stub_model(image, prompts=None)
    assert set(out.keys()) == {
        "pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec",
    }
    assert out["pred_logits"].shape == (2, stub_model.num_queries, 1)
    assert out["pred_boxes"].shape == (2, stub_model.num_queries, 4)
    assert out["pred_masks"].shape == (
        2, stub_model.num_queries, stub_model.mask_size, stub_model.mask_size,
    )
    assert out["presence_logit_dec"].shape == (2, 1)
```

If the `stub_model` fixture in `tests/conftest.py` (or wherever it lives) passes `num_classes=...` to `TinySam3Stub`, **remove that kwarg** — the new stub doesn't take it. Search via:

Run: `grep -rn "TinySam3Stub(" tests/`
Update every call site to drop `num_classes=...` (it never affected the matcher/loss tests anyway).

- [ ] **Step 3: Run targeted tests — verify pass**

Run: `uv run pytest tests/unit/test_fixtures.py tests/unit/test_meta_to_canonical.py -v --no-cov`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/tiny_sam3_stub.py tests/unit/test_fixtures.py tests/conftest.py
git commit -m "test(fixtures): rewrite TinySam3Stub for Meta SAM3.1 output keys"
```

(`tests/conftest.py` only included if a `stub_model` fixture lives there and needed editing.)

---

## Task 9: Revise `HungarianMatcher` — drop class cost, keep L1+GIoU+mask

**Why:** No `class_logits` field on `CanonicalOutputs` anymore. Matching uses geometric + mask costs only. `obj_logits` could in principle be used as a soft prior for matching, but DETR's original design works fine without it and the plan keeps the matcher simple.

**Files:**
- Modify: `src/esam3/models/matching.py` (`HungarianMatcher.__init__`, `HungarianMatcher.__call__`)
- Modify: `tests/unit/test_matching.py`

- [ ] **Step 1: Update tests first**

Replace the body of `tests/unit/test_matching.py` with:

```python
"""Unit tests for HungarianMatcher (revised plan — no class cost)."""

from __future__ import annotations

import torch

from esam3.data.base import Instance
from esam3.models.matching import CanonicalOutputs, HungarianMatcher


def _make_outputs(q: int = 4, mask_size: int = 16) -> CanonicalOutputs:
    return CanonicalOutputs(
        obj_logits=torch.zeros(1, q),
        pred_boxes=torch.zeros(1, q, 4),
        pred_masks=torch.zeros(1, q, mask_size, mask_size),
        img_presence=torch.zeros(1),
    )


def _instance(box: list[float], mask_size: int = 16) -> Instance:
    return Instance(
        mask=torch.zeros(mask_size, mask_size),
        class_id=0,  # class_id is irrelevant — matcher does not use it.
        box=torch.tensor(box, dtype=torch.float32),
    )


def test_matcher_empty_targets_returns_empty_pairs() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = _make_outputs(q=4)
    indices = matcher(outputs, [[]])
    assert len(indices) == 1
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 0
    assert tgt_idx.numel() == 0


def test_matcher_returns_one_match_per_target() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = _make_outputs(q=4)
    targets = [[_instance([0.5, 0.5, 0.1, 0.1]), _instance([0.2, 0.2, 0.1, 0.1])]]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2
    assert sorted(tgt_idx.tolist()) == [0, 1]
    assert len(set(pred_idx.tolist())) == 2


def test_matcher_handles_more_targets_than_queries() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = _make_outputs(q=2)
    targets = [[
        _instance([0.1, 0.1, 0.1, 0.1]),
        _instance([0.3, 0.3, 0.1, 0.1]),
        _instance([0.5, 0.5, 0.1, 0.1]),
    ]]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2


def test_matcher_batched() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = CanonicalOutputs(
        obj_logits=torch.zeros(2, 3),
        pred_boxes=torch.zeros(2, 3, 4),
        pred_masks=torch.zeros(2, 3, 16, 16),
        img_presence=torch.zeros(2),
    )
    targets = [
        [_instance([0.5, 0.5, 0.1, 0.1])],
        [_instance([0.2, 0.2, 0.1, 0.1]), _instance([0.7, 0.7, 0.1, 0.1])],
    ]
    indices = matcher(outputs, targets)
    assert len(indices) == 2
    assert indices[0][0].numel() == 1
    assert indices[1][0].numel() == 2
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run pytest tests/unit/test_matching.py -v --no-cov`
Expected: FAIL — `HungarianMatcher.__init__` still requires `lambda_cls`; outputs still expects `class_logits`.

- [ ] **Step 3: Rewrite `HungarianMatcher` in `src/esam3/models/matching.py`**

Replace the entire `class HungarianMatcher: ...` block with:

```python
class HungarianMatcher:
    """DETR-style bipartite matcher for per-class SAM 3.1 outputs.

    No class-cost term: prompts encode class identity, so the only meaningful
    pairwise affinities are geometric (L1, GIoU on cxcywh boxes) and mask (Dice).
    Non-differentiable; called under `@torch.no_grad()`.
    """

    def __init__(
        self,
        lambda_l1: float,
        lambda_giou: float,
        lambda_mask: float,
    ) -> None:
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_mask = lambda_mask

    @torch.no_grad()
    def __call__(
        self,
        outputs: CanonicalOutputs,
        targets: list[list[Instance]],
    ) -> list[tuple[Tensor, Tensor]]:
        b = outputs.obj_logits.shape[0]
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
            tgt_boxes = torch.stack([t.box for t in tgts]).to(outputs.pred_boxes.device)
            cost_l1 = torch.cdist(outputs.pred_boxes[i], tgt_boxes, p=1)
            cost_giou = -_giou(
                _box_cxcywh_to_xyxy(outputs.pred_boxes[i]),
                _box_cxcywh_to_xyxy(tgt_boxes),
            )

            tgt_masks = torch.stack([t.mask for t in tgts]).to(outputs.pred_masks.device)
            tgt_masks_low = interpolate(
                tgt_masks[None].float(),
                size=(mask_h, mask_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            cost_mask = _dice_cost(outputs.pred_masks[i], tgt_masks_low)

            cost = (
                self.lambda_l1 * cost_l1
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

The helpers `_box_cxcywh_to_xyxy`, `_giou`, `_dice_cost`, and the imports (`scipy.optimize.linear_sum_assignment`, `torch.nn.functional.interpolate`, `esam3.data.base.Instance`) below the `meta_to_canonical` definition are unchanged.

- [ ] **Step 4: Run targeted tests — verify pass**

Run: `uv run pytest tests/unit/test_matching.py -v --no-cov`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/matching.py tests/unit/test_matching.py
git commit -m "feat(models): drop class cost from HungarianMatcher (open-vocab head)"
```

---

## Task 10: Implement `mask_loss` (dice + BCE)

Identical to the original plan's Task 6. Restated here for self-containment.

**Files:**
- Rewrite: `src/esam3/models/losses.py`
- Create: `tests/unit/test_losses.py`

- [ ] **Step 1: Create `tests/unit/test_losses.py`**

```python
"""Unit tests for per-component losses + total_loss in models/losses.py."""

from __future__ import annotations

import torch

from esam3.models.losses import (
    box_loss,
    mask_loss,
    objectness_loss,
    presence_loss,
)


def test_mask_loss_zero_on_perfect_match() -> None:
    pred = torch.full((2, 32, 32), -10.0)
    pred[:, :16, :] = 10.0
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_mask_loss_positive_when_wrong() -> None:
    pred = torch.zeros(2, 32, 32)
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.item() > 0.0


def test_mask_loss_upsamples_pred_to_target_resolution() -> None:
    pred = torch.zeros(2, 16, 16)
    target = torch.zeros(2, 32, 32)
    loss = mask_loss(pred, target)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k mask_loss -v --no-cov`
Expected: FAIL — `mask_loss` is still a NotImplementedError stub.

- [ ] **Step 3: Rewrite `src/esam3/models/losses.py`**

Replace the file's entire contents with:

```python
"""SAM 3.1 training losses (per-class, open-vocab head)."""

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
    """0.5 · Dice + 0.5 · BCE on matched mask pairs.

    `pred` and `target` are (N, H_p, W_p) and (N, H_t, W_t). If the spatial
    shapes differ, `pred` is bilinear-upsampled to the target resolution.
    """
    if pred.shape[-2:] != target.shape[-2:]:
        pred = interpolate(
            pred[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    bce = binary_cross_entropy_with_logits(pred, target.float())
    dice = _dice_loss(pred, target)
    return 0.5 * dice + 0.5 * bce
```

- [ ] **Step 4: Run mask_loss tests — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -k mask_loss -v --no-cov`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement mask_loss (dice + BCE) with auto-upsample"
```

---

## Task 11: Implement `box_loss` (smoothL1 + 1-GIoU)

Identical to the original plan's Task 7.

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Append failing tests**

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

Run: `uv run pytest tests/unit/test_losses.py -k box_loss -v --no-cov`
Expected: FAIL — `box_loss` not yet importable.

- [ ] **Step 3: Append to `src/esam3/models/losses.py`**

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
    """smoothL1 + (1 - GIoU) on matched box pairs. Boxes are normalized cxcywh."""
    smooth_l1 = torch.nn.functional.smooth_l1_loss(pred, target, reduction="mean")
    giou = _giou_pairwise(_box_cxcywh_to_xyxy(pred), _box_cxcywh_to_xyxy(target))
    return smooth_l1 + (1.0 - giou).mean()
```

- [ ] **Step 4: Run box_loss tests — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -k box_loss -v --no-cov`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement box_loss (smoothL1 + 1-GIoU)"
```

---

## Task 12: Implement `objectness_loss` — binary focal BCE on per-query `obj_logits`

**Why:** `obj_logits` (B, Q) is per-query "is this query a real detection of the current prompt class?". Supervise as focal binary BCE against `matched_mask: (B, Q)` (True where the matcher paired query `i,q` to some GT instance).

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_losses.py`:

```python
def test_objectness_loss_zero_when_predictions_agree() -> None:
    obj_logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(obj_logits, matched)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_objectness_loss_high_when_predictions_invert() -> None:
    obj_logits = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(obj_logits, matched)
    assert loss.item() > 1.0
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k objectness_loss -v --no-cov`
Expected: FAIL.

- [ ] **Step 3: Append to `src/esam3/models/losses.py`**

```python
def _focal_bce(
    logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> Tensor:
    """Sigmoid focal BCE, mean-reduced. logits and targets broadcastable to the same shape."""
    p = logits.sigmoid()
    ce = binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def objectness_loss(
    obj_logits: Tensor,
    matched_mask: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Per-query binary focal BCE.

    obj_logits:    (B, Q) — Meta's `pred_logits` squeezed.
    matched_mask:  (B, Q) bool — True for queries assigned to some target by the matcher.
    """
    return _focal_bce(obj_logits, matched_mask.float(), gamma=gamma, alpha=alpha)
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -k objectness_loss -v --no-cov`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement objectness_loss (focal BCE on per-query obj_logits)"
```

---

## Task 13: Implement `presence_loss` — image-level BCE on `img_presence`

**Why:** Meta exposes a global per-image `presence_logit_dec: (B, 1)` that — under the user's fixed-class regime — answers "does this image contain any instance of the current prompt class?". Supervise as a plain binary BCE against `image_has_target: (B,)`.

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_losses.py`:

```python
def test_presence_loss_zero_when_agree() -> None:
    img_presence = torch.tensor([10.0, -10.0, 10.0])
    image_has_target = torch.tensor([True, False, True])
    loss = presence_loss(img_presence, image_has_target)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_presence_loss_high_when_inverted() -> None:
    img_presence = torch.tensor([-10.0, 10.0, -10.0])
    image_has_target = torch.tensor([True, False, True])
    loss = presence_loss(img_presence, image_has_target)
    assert loss.item() > 1.0
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k presence_loss -v --no-cov`
Expected: FAIL.

- [ ] **Step 3: Append to `src/esam3/models/losses.py`**

```python
def presence_loss(
    img_presence: Tensor,
    image_has_target: Tensor,
) -> Tensor:
    """Image-level binary BCE on the global presence logit.

    img_presence:     (B,) — Meta's `presence_logit_dec` squeezed.
    image_has_target: (B,) bool — True if the image contains any instance of the
                      current prompt class.
    """
    return binary_cross_entropy_with_logits(img_presence, image_has_target.float())
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -k presence_loss -v --no-cov`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement presence_loss (image-level BCE on presence_logit_dec)"
```

---

## Task 14: Implement `total_loss` + matched-pair gather helpers

**Why:** Tie matching + per-component losses together for ONE forward (one image batch, one prompt class). Caller invokes `total_loss` once per class and accumulates.

**Files:**
- Modify: `src/esam3/models/losses.py`
- Modify: `tests/unit/test_losses.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_losses.py`:

```python
def _stub_outputs(b: int = 1, q: int = 4, h: int = 16) -> dict:
    return {
        "pred_logits": torch.zeros(b, q, 1),
        "pred_boxes": torch.zeros(b, q, 4),
        "pred_masks": torch.zeros(b, q, h, h),
        "presence_logit_dec": torch.zeros(b, 1),
    }


def test_total_loss_returns_all_components() -> None:
    from esam3.config.schema import LossConfig
    from esam3.data.base import Instance
    from esam3.models.losses import total_loss

    raw = _stub_outputs()
    targets = [[Instance(
        mask=torch.zeros(32, 32),
        class_id=0,
        box=torch.tensor([0.5, 0.5, 0.2, 0.2]),
    )]]
    losses = total_loss(raw, targets, LossConfig())
    assert set(losses.keys()) == {"total", "mask", "box", "obj", "presence"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_total_loss_total_equals_weighted_sum() -> None:
    from esam3.config.schema import LossConfig
    from esam3.data.base import Instance
    from esam3.models.losses import total_loss

    raw = _stub_outputs()
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
        + cfg.w_presence * losses["presence"]
    )
    assert torch.allclose(losses["total"], expected, atol=1e-6)


def test_total_loss_handles_empty_targets() -> None:
    from esam3.config.schema import LossConfig
    from esam3.models.losses import total_loss

    raw = _stub_outputs()
    losses = total_loss(raw, [[]], LossConfig())
    # No matches → mask + box are zero; obj + presence are still finite (no-object supervision).
    assert losses["mask"].item() == 0.0
    assert losses["box"].item() == 0.0
    assert torch.isfinite(losses["obj"])
    assert torch.isfinite(losses["presence"])
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_losses.py -k total_loss -v --no-cov`
Expected: FAIL.

- [ ] **Step 3: Append to `src/esam3/models/losses.py`**

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
        tgt_boxes.append(
            torch.stack([targets[i][j].box for j in tgt_idx.tolist()]).to(
                canonical.pred_boxes.device
            )
        )
        pred_masks.append(canonical.pred_masks[i, pred_idx])
        tgt_masks.append(
            torch.stack([targets[i][j].mask for j in tgt_idx.tolist()]).to(
                canonical.pred_masks.device
            )
        )
    if not pred_boxes:
        empty_b = canonical.pred_boxes.new_zeros((0, 4))
        empty_m = canonical.pred_masks.new_zeros((0, 1, 1))
        return empty_b, empty_b, empty_m, empty_m
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
    """Bool (B, Q): True where a query is matched to some target."""
    b, q = canonical.obj_logits.shape
    mask = torch.zeros((b, q), dtype=torch.bool, device=canonical.obj_logits.device)
    for i, (pred_idx, _) in enumerate(indices):
        if pred_idx.numel() > 0:
            mask[i, pred_idx] = True
    return mask


def _image_has_target(targets: list[list[Instance]], device: torch.device) -> Tensor:
    """Bool (B,): True if image has any target instance of the current prompt class."""
    return torch.tensor([len(t) > 0 for t in targets], dtype=torch.bool, device=device)


def total_loss(
    outputs: dict,
    targets: list[list[Instance]],
    cfg: LossConfig,
) -> dict[str, Tensor]:
    """Run matching, compute per-component losses, return dict with 'total' summed.

    `outputs` is Meta's raw per-class forward dict. `targets[i]` is the list of
    GT instances of the prompt's class for image i (may be empty).
    """
    canonical = meta_to_canonical(outputs)
    matcher = HungarianMatcher(
        lambda_l1=cfg.matcher_weights.lambda_l1,
        lambda_giou=cfg.matcher_weights.lambda_giou,
        lambda_mask=cfg.matcher_weights.lambda_mask,
    )
    indices = matcher(canonical, targets)

    pred_boxes_m, tgt_boxes_m, pred_masks_m, tgt_masks_m = _gather_matched_boxes_masks(
        canonical, targets, indices
    )
    matched_mask = _matched_query_mask(canonical, indices)
    has_target = _image_has_target(targets, canonical.img_presence.device)

    zero = canonical.obj_logits.new_zeros(())
    losses: dict[str, Tensor] = {
        "mask": mask_loss(pred_masks_m, tgt_masks_m) if pred_masks_m.numel() > 0 else zero,
        "box": box_loss(pred_boxes_m, tgt_boxes_m) if pred_boxes_m.numel() > 0 else zero,
        "obj": objectness_loss(
            canonical.obj_logits,
            matched_mask,
            gamma=cfg.focal_gamma,
            alpha=cfg.focal_alpha,
        ),
        "presence": presence_loss(canonical.img_presence, has_target),
    }
    losses["total"] = (
        cfg.w_mask * losses["mask"]
        + cfg.w_box * losses["box"]
        + cfg.w_obj * losses["obj"]
        + cfg.w_presence * losses["presence"]
    )
    return losses
```

- [ ] **Step 4: Run all loss tests — verify pass**

Run: `uv run pytest tests/unit/test_losses.py -v --no-cov`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/losses.py tests/unit/test_losses.py
git commit -m "feat(losses): implement total_loss (mask+box+obj+presence, per-class)"
```

---

## Task 15: Prune `test_stubs_raise.py`

Identical intent to the original Task 11. Updated for the new loss public surface.

**Files:**
- Modify: `tests/unit/test_stubs_raise.py`

- [ ] **Step 1: Read the current file**

Run: `cat tests/unit/test_stubs_raise.py | head -60`

It currently imports `load_sam31` and the loss functions and asserts they raise `NotImplementedError`. The losses are now real (Task 14) but `load_sam31` is still a stub until Task 17.

- [ ] **Step 2: Delete the loss-stub imports + the `test_model_stubs` function**

Edit `tests/unit/test_stubs_raise.py` to:
1. Remove the import line:
   ```python
   from esam3.models.losses import box_loss, mask_loss, objectness_loss, total_loss
   ```
   (also delete any `class_loss` import if present)
2. Remove the `from esam3.models.sam3 import load_sam31` import.
3. Delete the entire `def test_model_stubs() -> None: ...` function.
4. Keep `test_peft_stubs`, `test_eval_stubs`, `test_train_stubs`, `test_trainer_fit_stub` (or whatever non-model stubs exist there) unchanged.

- [ ] **Step 3: Run the full unit suite — no regression**

Run: `uv run pytest tests/unit -v --no-cov`
Expected: all currently-passing tests still pass; the removed `test_model_stubs` is gone from the report.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_stubs_raise.py
git commit -m "test: drop NotImplementedError asserts for load_sam31 + losses (now implemented)"
```

---

## Task 16: Implement `Sam3Wrapper`

**Files:**
- Rewrite: `src/esam3/models/sam3.py` (wrapper only; `load_sam31` still stubbed)
- Create: `tests/unit/test_sam3_wrapper.py`

**Important contract decision:** The wrapper accepts a SINGLE prompt class per forward call (one prompt per batch image, one class per prompt). Multi-class images are handled by the trainer looping over the fixed class vocabulary and calling the wrapper once per class. Multiplex (16-prompts-at-once) is deferred to a future plan.

For Sam3Image's actual training-mode API (`forward_grounding(backbone_out, find_input, find_target, geometric_prompt)`), the wrapper bottoms out in `self.model(images, prompts)` — i.e., it relies on the underlying model's `__call__` to do whatever orchestration is needed to convert `(images, prompts)` into the right `forward_grounding` arguments. This works for `TinySam3Stub` (which mimics that interface). Task 17 will pin down whether real `Sam3Image` exposes a similar high-level entry point or whether a small adapter is needed inside `load_sam31`; if so, that adapter is wrapped INSIDE `load_sam31`'s returned model, not surfaced in `Sam3Wrapper`.

- [ ] **Step 1: Create `tests/unit/test_sam3_wrapper.py`**

```python
"""Unit tests for Sam3Wrapper using TinySam3Stub (no real model)."""

from __future__ import annotations

import pytest
import torch

from esam3.data.base import BoxPrompts, TextPrompts
from esam3.models.sam3 import Sam3Wrapper
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_wrapper_passes_through_single_class_text_prompts() -> None:
    stub = TinySam3Stub(num_queries=2, mask_size=16)
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = wrapper(image, prompts)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_wrapper_rejects_multi_class_text_prompts() -> None:
    """One forward = one class. Multi-class prompts must be split by the caller."""
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat", "dog"])]
    with pytest.raises(ValueError, match="exactly one class"):
        wrapper(image, prompts)


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


def test_wrapper_rejects_batch_size_mismatch() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"])]  # B=2 images but 1 prompt
    with pytest.raises(ValueError, match="len\\(prompts\\)"):
        wrapper(image, prompts)
```

- [ ] **Step 2: Run — verify failure**

Run: `uv run pytest tests/unit/test_sam3_wrapper.py -v --no-cov`
Expected: FAIL — `Sam3Wrapper` not importable (still stubbed by the original Task 4-era code).

- [ ] **Step 3: Replace `src/esam3/models/sam3.py`**

```python
"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md.

Revised by docs/superpowers/plans/2026-05-16-model-loading-revised.md to match
Meta's open-vocab head: one prompt class per forward call. Trainer loops over
the fixed class vocabulary externally.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import BoxPrompts, Prompts, TextPrompts


class Sam3Wrapper(nn.Module):
    """Thin wrapper around Meta's SAM 3.1 model.

    Contract:
      - `forward(images, prompts)` accepts a batch of B images and a list of
        B `Prompts` objects, one per image.
      - All prompts in a batch MUST be the same variant (TextPrompts XOR
        BoxPrompts); the wrapper raises on mixed batches.
      - For TextPrompts, each image's prompt MUST contain exactly one class
        name; the trainer is responsible for looping over the fixed class
        vocabulary and accumulating losses across classes.
      - Returns Meta's native output dict unchanged.
    """

    def __init__(self, model: nn.Module, image_size: int = 1008, mask_size: int = 288) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size

    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Any]:
        self._validate_prompts(images, prompts)
        return self.model(images, prompts)

    @staticmethod
    def _validate_prompts(images: Tensor, prompts: list[Prompts]) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be (B, 3, H, W); got shape {tuple(images.shape)}")
        b = images.shape[0]
        if len(prompts) != b:
            raise ValueError(
                f"len(prompts)={len(prompts)} must equal batch size {b}"
            )
        if not prompts:
            return
        first = type(prompts[0])
        for p in prompts:
            if type(p) is not first:
                raise ValueError(
                    "All prompts in a batch must be the same prompt variant "
                    "(TextPrompts or BoxPrompts), not mixed."
                )
            if isinstance(p, TextPrompts):
                if len(p.classes) != 1:
                    raise ValueError(
                        f"TextPrompts must contain exactly one class per forward "
                        f"call (got {len(p.classes)}). Loop over the class vocabulary "
                        f"externally."
                    )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's sam3 package. Implementation lands in Task 17."""
    raise NotImplementedError("filled in by Task 17 of spec/model-loading-revised")
```

- [ ] **Step 4: Run wrapper tests — verify pass**

Run: `uv run pytest tests/unit/test_sam3_wrapper.py -v --no-cov`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/sam3.py tests/unit/test_sam3_wrapper.py
git commit -m "feat(models): implement Sam3Wrapper with single-class per-forward contract"
```

---

## Task 17: Implement `load_sam31`

**Files:**
- Modify: `src/esam3/models/sam3.py` (replace the `load_sam31` stub)

The Meta API was inspected via static analysis:
- Build entrypoint: `sam3.build_sam3_image_model(bpe_path=None, device='cuda', eval_mode=True, checkpoint_path=None, load_from_HF=True, enable_segmentation=True, enable_inst_interactivity=False, compile=False)`.
- Passing `checkpoint_path=<path-to-.pt>` and `load_from_HF=False` makes the builder load weights internally. **No manual `.pt` flatten/remap is needed.**
- BPE merges file lives at `models/sam3.1/merges.txt` (downloaded from HF alongside the checkpoint).
- The returned model is a `Sam3Image` with a `forward_grounding` method (not a plain `__call__` that takes `(images, prompts)`). The Sam3Wrapper assumes `self.model(images, prompts)` works, which means `load_sam31` needs to wrap the raw `Sam3Image` in a small adapter `nn.Module` that re-exposes that interface. The adapter's job is the focus of Step 2 below.

- [ ] **Step 1: Inspect Meta's high-level forward entrypoint**

Run this one-shot inspection (do NOT commit):

```bash
uv run python <<'PY'
import inspect, sam3
from sam3.model.sam3_image import Sam3Image
print("=== Sam3Image methods relevant for forward ===")
for name in dir(Sam3Image):
    if name.startswith("_") and name not in ("__call__",):
        continue
    if any(kw in name.lower() for kw in ("forward", "predict", "image", "encode")):
        obj = getattr(Sam3Image, name)
        if callable(obj):
            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):
                sig = "<no signature>"
            print(f"  {name}{sig}")
PY
```

Look for a high-level method that takes raw images + text prompts. Likely candidates: `predict_inst`, `predict_inst_batch`. Record the signature(s). If none are obvious, fall back to building inputs manually for `forward_grounding`.

- [ ] **Step 2: Replace `load_sam31` in `src/esam3/models/sam3.py`**

Add these imports near the top:

```python
import logging
from pathlib import Path

import sam3  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
```

Below `Sam3Wrapper` (replacing the `load_sam31` stub), add:

```python
def _resolve_checkpoint_path(cfg: ModelConfig) -> Path:
    if cfg.local_dir is None:
        raise FileNotFoundError(
            "ModelConfig.local_dir is None and Hub fetch is not implemented. "
            f"Set local_dir to a directory containing {cfg.checkpoint_file}. "
            f"To download: `huggingface-cli download {cfg.name} --local-dir models/sam3.1`."
        )
    path = Path(cfg.local_dir) / cfg.checkpoint_file
    if not path.exists():
        raise FileNotFoundError(
            f"SAM 3.1 checkpoint not found at {path}. "
            f"Run: huggingface-cli download {cfg.name} --local-dir {cfg.local_dir}"
        )
    return path


def _resolve_bpe_path(cfg: ModelConfig) -> Path:
    """The BPE merges file is shipped alongside the checkpoint in the HF repo."""
    if cfg.local_dir is None:
        raise FileNotFoundError("ModelConfig.local_dir is None; cannot resolve BPE path.")
    path = Path(cfg.local_dir) / "merges.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"SAM 3.1 BPE merges file not found at {path}. Re-download the checkpoint "
            f"directory from {cfg.name}."
        )
    return path


class _Sam3ImageAdapter(nn.Module):
    """Adapt raw Sam3Image to the (images, prompts) calling convention used by Sam3Wrapper.

    Sam3Image's training-mode forward (`forward_grounding`) expects
    `(backbone_out, find_input, find_target, geometric_prompt)`, none of which are
    raw image tensors or our `Prompts` dataclasses. This adapter holds the inner
    `Sam3Image` and orchestrates the conversion based on what Meta's high-level
    methods (inspected in Step 1) expose. If Meta exposes `predict_inst` or
    similar that takes raw images, prefer that path here.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Tensor]:
        # IMPLEMENTOR: based on Step 1's inspection, fill this in.
        # The simplest case: Sam3Image exposes a method that takes
        # `(images, list_of_class_names)` and returns the per-image dict.
        #
        # For TextPrompts (which is the only supported case per Sam3Wrapper):
        #   class_names = [p.classes[0] for p in prompts]
        #   return self.model.<entrypoint>(images, class_names)
        #
        # If no such method exists, build the lower-level Sam3 inputs here
        # (backbone_out via self.model.image_encoder(images), find_input from
        # tokenized class names, etc.). Keep the function body small — if it
        # exceeds ~30 lines, factor out helpers in this same file.
        raise NotImplementedError(
            "Sam3Image high-level forward entrypoint not yet pinned; complete this "
            "function after running Step 1's inspection in your local environment."
        )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's `sam3` package and wrap it for our trainer.

    Returns a `Sam3Wrapper` whose `forward(images, prompts)` returns Meta's
    native per-class output dict (`pred_logits`, `pred_boxes`, `pred_masks`,
    `presence_logit_dec`).
    """
    ckpt_path = _resolve_checkpoint_path(cfg)
    bpe_path = _resolve_bpe_path(cfg)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw_model = sam3.build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=False,  # training mode — gradients flow.
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    if cfg.gradient_checkpointing:
        if hasattr(raw_model, "set_grad_checkpointing"):
            raw_model.set_grad_checkpointing(True)
        else:
            logger.warning(
                "Meta sam3 model has no `set_grad_checkpointing`; "
                "gradient_checkpointing=True is a no-op on this revision."
            )

    if cfg.dtype == "bfloat16":
        raw_model = raw_model.to(dtype=torch.bfloat16)
    elif cfg.dtype == "float16":
        raw_model = raw_model.to(dtype=torch.float16)

    adapter = _Sam3ImageAdapter(raw_model)
    return Sam3Wrapper(adapter, image_size=1008, mask_size=288)
```

- [ ] **Step 3: Smoke-test the missing-checkpoint error path**

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

- [ ] **Step 4: Run the full unit suite — verify no regression**

Run: `uv run pytest tests/unit -v --no-cov`
Expected: all PASS. (The real `load_sam31` is not unit-tested here; Task 18's integration test covers it.)

- [ ] **Step 5: Commit**

```bash
git add src/esam3/models/sam3.py
git commit -m "feat(models): implement load_sam31 via sam3.build_sam3_image_model"
```

**Known carry-over**: `_Sam3ImageAdapter.forward` is intentionally left raising `NotImplementedError` if Step 1's inspection does not yield a clean high-level entrypoint. The unit suite does NOT call this code path. The integration test in Task 18 WILL call it and will reveal whatever remaining gap exists; resolve it there (or escalate if Meta's API genuinely requires building `find_input` / `find_target` from scratch — that's plausibly a separate plan).

---

## Task 18: Add gated integration test for real checkpoint loading

**Files:**
- Modify: `tests/conftest.py` (add `requires_checkpoint` and `requires_compatible_gpu` markers)
- Create: `tests/integration/test_load_sam31_real.py`

The user's local hardware (GTX 1080, sm_61) is below the installed PyTorch's minimum supported CC (sm_75); furthermore, Meta's `sam3/perflib/fused.py` forces bf16 on the `addmm_act` fast path, which fails on CPU. Both make the integration test conditional.

- [ ] **Step 1: Register the markers in `tests/conftest.py`**

Open the file and read its current contents:

Run: `cat tests/conftest.py`

If it already has `pytest_configure` / `pytest_collection_modifyitems`, merge in the additions below; otherwise replace/extend the file with:

```python
"""Project-level pytest hooks (markers, autoskips)."""

from __future__ import annotations

import pathlib

import pytest
import torch


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


def _has_compatible_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    return (major, minor) >= (7, 5)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    ckpt = pathlib.Path("models/sam3.1/sam3.1_multiplex.pt")
    skip_no_ckpt = pytest.mark.skip(reason="real SAM 3.1 checkpoint not present locally")
    skip_no_gpu = pytest.mark.skip(
        reason="real SAM 3.1 forward requires a CUDA GPU with CC >= 7.5"
    )
    have_gpu = _has_compatible_gpu()
    for item in items:
        if "requires_checkpoint" in item.keywords and not ckpt.exists():
            item.add_marker(skip_no_ckpt)
        if "requires_compatible_gpu" in item.keywords and not have_gpu:
            item.add_marker(skip_no_gpu)
```

Add the markers to `pyproject.toml`'s `[tool.pytest.ini_options].markers` list:

```toml
markers = [
  "integration: end-to-end tests using the stub model (CPU)",
  "gpu: tests requiring a CUDA device with real SAM3.1 weights",
  "requires_checkpoint: skip unless real checkpoint is present",
  "requires_compatible_gpu: skip unless CUDA CC >= 7.5",
]
```

- [ ] **Step 2: Create `tests/integration/test_load_sam31_real.py`**

```python
"""Integration test: load real SAM 3.1 checkpoint and run a forward pass.

Skipped automatically unless the .pt checkpoint is present AND a CUDA GPU
with compute capability >= 7.5 is available.
"""

from __future__ import annotations

import pytest
import torch

from esam3.config.schema import ModelConfig
from esam3.data.base import TextPrompts
from esam3.models.matching import meta_to_canonical
from esam3.models.sam3 import Sam3Wrapper, load_sam31


@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_returns_wrapper() -> None:
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    assert isinstance(wrapper, Sam3Wrapper)


@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_forward_to_canonical() -> None:
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    image = torch.zeros(1, 3, 1008, 1008, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        outputs = wrapper(image, [TextPrompts(classes=["cat"])])
    canonical = meta_to_canonical(outputs)
    assert canonical.obj_logits.dim() == 2          # (B, Q)
    assert canonical.pred_boxes.shape[-1] == 4
    assert canonical.pred_masks.shape[-1] == 288
    assert canonical.img_presence.dim() == 1         # (B,)
```

- [ ] **Step 3: Create the `tests/integration/` directory if missing**

Run: `mkdir -p tests/integration && touch tests/integration/__init__.py`

- [ ] **Step 4: Run the integration test (should SKIP without compatible GPU)**

Run: `uv run pytest tests/integration/test_load_sam31_real.py -v --no-cov`
Expected on this user's machine: SKIPPED with reason "real SAM 3.1 forward requires a CUDA GPU with CC >= 7.5". On a compatible GPU it should PASS or surface the remaining `_Sam3ImageAdapter.forward` NotImplementedError from Task 17.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/integration/test_load_sam31_real.py pyproject.toml
git commit -m "test(integration): add gated load_sam31 smoke test"
```

---

## Task 19: Fix example configs (image_size, normalization, model paths, loss block)

**Files:**
- Modify: `configs/examples/coco_text_lora.yaml`
- Modify: `configs/examples/coco_bbox_qlora.yaml`

- [ ] **Step 1: Read current configs**

Run: `cat configs/examples/coco_text_lora.yaml configs/examples/coco_bbox_qlora.yaml`

Note the current values of `data.image_size`, `data.normalize.mean/std`, the `model` block, and whether a `train.loss` block exists.

- [ ] **Step 2: Patch `configs/examples/coco_text_lora.yaml`**

Apply these edits:

- `data.image_size: 1024` → `data.image_size: 1008`
- `data.normalize.mean: [0.485, 0.456, 0.406]` → `data.normalize.mean: [0.5, 0.5, 0.5]`
- `data.normalize.std: [0.229, 0.224, 0.225]` → `data.normalize.std: [0.5, 0.5, 0.5]`
- In the `model:` block, ensure these fields are present (add or update):
  ```yaml
  model:
    name: facebook/sam3.1
    local_dir: models/sam3.1
    checkpoint_file: sam3.1_multiplex.pt
  ```
- Inside the `train:` block, append (commented — defaults shown):
  ```yaml
    # Loss-mix weights (defaults shown — uncomment to override):
    # loss:
    #   w_mask: 1.0
    #   w_box: 5.0
    #   w_obj: 1.0
    #   w_presence: 1.0
    #   focal_gamma: 2.0
    #   focal_alpha: 0.25
    #   matcher_weights:
    #     lambda_l1: 5.0
    #     lambda_giou: 2.0
    #     lambda_mask: 5.0
  ```

- [ ] **Step 3: Apply the same patches to `configs/examples/coco_bbox_qlora.yaml`**

Same 4 edits.

- [ ] **Step 4: Verify both configs still parse**

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
Expected: `OK:` printed for both. If `esam3.config.loader` does not exist under that exact name, find the entrypoint via `grep -rn "def load.*config" src/esam3/config/` and adapt the import.

- [ ] **Step 5: Commit**

```bash
git add configs/examples/coco_text_lora.yaml configs/examples/coco_bbox_qlora.yaml
git commit -m "fix(configs): align example yamls with SAM3.1 native size + open-vocab losses"
```

---

## Task 20: Final lint, format, full test run, PR

**Files:** all files touched in this plan; possibly `pyproject.toml` for mypy overrides.

- [ ] **Step 1: Ruff format**

Run: `uv run ruff format src/esam3/models src/esam3/config tests`
Expected: completes; reformatting (if any) is applied in-place.

- [ ] **Step 2: Ruff check**

Run: `uv run ruff check src/esam3/models src/esam3/config tests`
Expected: zero issues. Fix any inline.

- [ ] **Step 3: Mypy**

Run: `uv run mypy src/esam3/models src/esam3/config`
Expected: zero errors. If `sam3` triggers `import-untyped`, the `# type: ignore[import-untyped]` on its import should suppress it; otherwise add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = "sam3.*"
ignore_missing_imports = true
```

- [ ] **Step 4: Full unit suite (with coverage gate)**

Run: `uv run pytest tests/unit -v`
Expected: all PASS, coverage ≥ 80%.

If coverage fails, do NOT lower the threshold; instead add targeted tests for any uncovered branch in the new modules (`matching.py`, `losses.py`, `sam3.py`).

- [ ] **Step 5: Integration suite (skip on incompatible hardware)**

Run: `uv run pytest tests/integration -v --no-cov`
Expected: SKIPPED on this machine (no compatible GPU). PASS on a compatible GPU.

- [ ] **Step 6: Commit any lint/format/mypy fixes**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: ruff format + mypy fixes for spec/model-loading"
```

(Skip if `git diff --cached --stat` shows nothing.)

- [ ] **Step 7: Push and open PR**

```bash
git push -u origin model-loading
gh pr create --title "feat(models): SAM 3.1 loader, wrapper, matcher, open-vocab losses" --body "$(cat <<'EOF'
## Summary
- Implements `load_sam31` via `sam3.build_sam3_image_model` (checkpoint + BPE loaded by the builder; no manual `.pt` remap needed).
- Implements `Sam3Wrapper` (one prompt class per forward; trainer loops over the fixed class vocabulary externally).
- Implements `models/matching.py`: `CanonicalOutputs` (per-query `obj_logits`, image-level `img_presence`), `meta_to_canonical`, `HungarianMatcher` with L1+GIoU+Dice costs (no class cost — open-vocab head).
- Implements `models/losses.py`: dice+BCE mask, smoothL1+(1-GIoU) box, focal BCE per-query objectness, image-level presence BCE, weighted `total_loss`. No multi-class `class_loss` — discrimination across classes comes from per-prompt forward passes.
- Schema: drops `w_cls` / `lambda_cls`, adds `w_presence`. Extends `ModelConfig` with `local_dir` / `checkpoint_file` / `device`.
- Fixes example YAMLs (image_size 1024→1008, ImageNet→[0.5]³ normalization, model.local_dir, loss block).
- Bumps `transformers>=5.0`, adds `scipy`, `einops`, and the Meta `sam3` git dep.

Reference: `docs/superpowers/specs/2026-05-16-model-loading-design.md` (schema/wrapper contract) + `docs/superpowers/plans/2026-05-16-model-loading-revised.md` (loss design — supersedes Tasks 5–15 of the original plan).

## Test plan
- [ ] `uv run pytest tests/unit -v` PASS with coverage ≥ 80%
- [ ] `uv run pytest tests/integration -v --no-cov` PASS on a compatible GPU; SKIPPED otherwise
- [ ] `uv run ruff check` clean
- [ ] `uv run mypy src/esam3/models src/esam3/config` clean

## Carry-overs
- `_Sam3ImageAdapter.forward` may still raise `NotImplementedError` if Meta's high-level forward entrypoint was not pinned in Task 17 Step 1; the integration test surfaces this. If it does, file a follow-up plan to build the low-level `forward_grounding` input plumbing.
- Multiplex (>1 prompt per forward call) is deferred to a future plan.
- `Instance.box` xyxy-vs-cxcywh format mismatch is logged in `logs/TODO.md`; the matcher/losses assume cxcywh normalized, so the data layer must convert before reaching the model. Track separately.
EOF
)"
```

---

## Self-review checklist (for the implementer)

After completing every task, verify:

1. **Spec coverage:** Every numbered section of the spec is implemented:
   - §3 `load_sam31` → Task 17
   - §4 `Sam3Wrapper` → Task 16
   - §5 `matching.py` → Tasks 6, 7, 9
   - §6 `losses.py` → Tasks 10–14
   - §7 Config schema → Tasks 5 (revised) + (Task 2 of original, already committed)
   - §8 Example config fixes → Task 19
   - §9 Tests → embedded in each task; integration → Task 18
2. **No `NotImplementedError` remains** for `mask_loss`, `box_loss`, `objectness_loss`, `presence_loss`, `total_loss`, `meta_to_canonical`, `HungarianMatcher`. `_Sam3ImageAdapter.forward` may still raise (see Task 17 carry-over) but `load_sam31` itself is real.
3. **No `class_loss` references remain** anywhere in `src/` or `tests/`.
4. **Ruff + mypy clean** on every touched file.
5. **Spec §11 deferred items remain untouched** (PEFT internals, training loop, video, distributed, evaluation, profiling).
6. **Per-class training contract honored**: `Sam3Wrapper.forward` accepts exactly one class per prompt; `total_loss` operates on one class at a time; the trainer (out of scope here) is responsible for looping over the fixed class vocabulary.
