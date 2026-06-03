"""Profiling-bucket tests for the eval surfaces (issue #273 / §3b).

Covers:
  - eval.gt_rle_encode  (evaluator._mask_to_rle)
  - eval.pair_iou       (Evaluator._compute_per_example_iou)

CPU-only — no model, no GPU required.

Run with:
    uv run pytest tests/unit/test_eval_profiling_buckets.py -o "addopts=" -p no:cacheprovider -q
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import custom_sam_peft.profiling as prof
from custom_sam_peft.eval.evaluator import Evaluator, _mask_to_rle

# ---------------------------------------------------------------------------
# Fixture: ensure each test starts with a clean, disabled profiler state.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_profiler():  # type: ignore[return]
    """Reset + disable before and after every test."""
    prof.reset()
    prof.disable()
    yield
    prof.reset()
    prof.disable()


# ---------------------------------------------------------------------------
# eval.gt_rle_encode
# ---------------------------------------------------------------------------


class TestGtRleEncodeBucket:
    def test_bucket_present_after_mask_to_rle(self) -> None:
        """_mask_to_rle must record eval.gt_rle_encode in the profiler snapshot."""
        prof.enable()
        prof.reset()

        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[:4, :4] = True
        _mask_to_rle(mask)

        buckets, _meta = prof.snapshot()
        assert "eval.gt_rle_encode" in buckets, (
            f"eval.gt_rle_encode not found; buckets = {list(buckets)}"
        )

    def test_bucket_absent_when_profiler_disabled(self) -> None:
        """_mask_to_rle must be a strict no-op when profiler is disabled."""
        # profiler is disabled by fixture
        mask = torch.zeros(8, 8, dtype=torch.bool)
        _mask_to_rle(mask)

        buckets, _meta = prof.snapshot()
        assert "eval.gt_rle_encode" not in buckets


# ---------------------------------------------------------------------------
# eval.pair_iou
# ---------------------------------------------------------------------------


def _make_tiny_coco_gt(examples, class_names):
    """Build an in-memory COCO object from a list of Example objects."""
    from pycocotools.coco import COCO

    from custom_sam_peft.eval.evaluator import _int_image_id

    categories = [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(class_names)]
    images = []
    annotations = []
    ann_id = 1
    for ex in examples:
        img_id = _int_image_id(ex.image_id)
        h = int(ex.image.shape[-2])
        w = int(ex.image.shape[-1])
        images.append({"id": img_id, "height": h, "width": w})
        for inst in ex.instances or []:
            rle = _mask_to_rle(inst.mask.bool())
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": inst.class_id + 1,
                    "segmentation": rle,
                    "area": float(inst.mask.sum()),
                    "iscrowd": 0,
                    "bbox": [0.0, 0.0, float(w), float(h)],
                }
            )
            ann_id += 1

    dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


class TestPairIouBucket:
    def test_bucket_present_after_compute_per_example_iou(self) -> None:
        """_compute_per_example_iou must record eval.pair_iou when called."""
        from custom_sam_peft.data.base import Example, Instance, TextPrompts

        class_names = ["cat"]
        h = w = 8

        def make_ex(image_id: str) -> Example:
            mask = torch.zeros(h, w, dtype=torch.bool)
            mask[:4, :4] = True
            return Example(
                image=torch.zeros(3, h, w),
                image_id=image_id,
                prompts=TextPrompts(classes=class_names),
                instances=[Instance(mask=mask, class_id=0, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
            )

        examples = [make_ex("img_0"), make_ex("img_1")]

        # Build COCO GT
        prof.disable()
        prof.reset()
        gt = _make_tiny_coco_gt(examples, class_names)

        # Build a minimal predictions list: one prediction per image that mimics
        # what queries_to_coco_results would produce (segmentation = RLE dict).
        from custom_sam_peft.eval.evaluator import _int_image_id

        predictions: list[dict[str, Any]] = []
        for ex in examples:
            img_id = _int_image_id(ex.image_id)
            mask = torch.zeros(h, w, dtype=torch.bool)
            mask[:4, :4] = True
            rle = _mask_to_rle(mask)
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": 0.9,
                }
            )

        # Now enable profiler and call the method under test.
        prof.enable()
        prof.reset()

        cfg = __import__("custom_sam_peft.config.schema", fromlist=["EvalConfig"]).EvalConfig(
            mode="full", iou_thresholds=[0.5], batch_size=1
        )
        ev = Evaluator(cfg)
        ev._compute_per_example_iou(examples, predictions, gt)

        buckets, _meta = prof.snapshot()
        assert "eval.pair_iou" in buckets, f"eval.pair_iou not found; buckets = {list(buckets)}"

    def test_bucket_absent_when_profiler_disabled(self) -> None:
        """_compute_per_example_iou must not write any bucket when profiler is off."""
        from custom_sam_peft.data.base import Example, Instance, TextPrompts

        class_names = ["cat"]
        h = w = 8

        def make_ex(image_id: str) -> Example:
            mask = torch.zeros(h, w, dtype=torch.bool)
            mask[:4, :4] = True
            return Example(
                image=torch.zeros(3, h, w),
                image_id=image_id,
                prompts=TextPrompts(classes=class_names),
                instances=[Instance(mask=mask, class_id=0, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
            )

        examples = [make_ex("img_a")]
        gt = _make_tiny_coco_gt(examples, class_names)

        from custom_sam_peft.eval.evaluator import _int_image_id

        predictions = []
        for ex in examples:
            img_id = _int_image_id(ex.image_id)
            mask = torch.zeros(h, w, dtype=torch.bool)
            mask[:4, :4] = True
            rle = _mask_to_rle(mask)
            predictions.append(
                {"image_id": img_id, "category_id": 1, "segmentation": rle, "score": 0.9}
            )

        # profiler is disabled (via autouse fixture)
        cfg = __import__("custom_sam_peft.config.schema", fromlist=["EvalConfig"]).EvalConfig(
            mode="full", iou_thresholds=[0.5], batch_size=1
        )
        ev = Evaluator(cfg)
        ev._compute_per_example_iou(examples, predictions, gt)

        buckets, _ = prof.snapshot()
        assert "eval.pair_iou" not in buckets
