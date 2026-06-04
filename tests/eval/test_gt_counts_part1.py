"""Part 1 tests: exact GT counts from the evaluator (spec §Design — Part 1).

Covers:
- gt_counts returned by Evaluator.evaluate(..., return_per_example_iou=True) equals
  [len(ex.instances) for ex in examples] on a fixture dataset.
- pick_samples(..., gt_counts=...) returns indices identical to the decode-fallback
  path (gt_counts=None) on the same fixture.
- gt_counts=None exercises the fallback and matches current behavior.
- A spy/counter on dataset.__getitem__ asserts it is NOT called over the full range
  during viz selection when gt_counts is provided — only the count selected images
  get decoded (via render_eval_pair).
- SemanticEvaluator.evaluate(..., return_per_example_iou=True) returns a 3-tuple
  with None as the third element.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.visualize import pick_samples

# TinySam3Stub and make_stub_wrapper are imported inside tests to keep
# top-level imports minimal (they pull in the full stub fixture tree).


# ---------------------------------------------------------------------------
# Minimal in-memory fixtures
# ---------------------------------------------------------------------------


def _make_instance() -> Instance:
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:4, :4] = True
    return Instance(
        mask=mask,
        class_id=0,
        box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
    )


class _SimpleDataset:
    """Three-image in-memory dataset. image_0 has 1 instance, image_1 has 0, image_2 has 2."""

    class_names: ClassVar[list[str]] = ["cat"]

    def __init__(self) -> None:
        self._examples = [
            Example(
                image=torch.zeros(3, 8, 8),
                image_id="img_0",
                prompts=TextPrompts(classes=["cat"]),
                instances=[_make_instance()],  # 1 GT instance
            ),
            Example(
                image=torch.zeros(3, 8, 8),
                image_id="img_1",
                prompts=TextPrompts(classes=["cat"]),
                instances=[],  # 0 GT instances
            ),
            Example(
                image=torch.zeros(3, 8, 8),
                image_id="img_2",
                prompts=TextPrompts(classes=["cat"]),
                instances=[_make_instance(), _make_instance()],  # 2 GT instances
            ),
        ]
        self._getitem_calls: list[int] = []

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        self._getitem_calls.append(i)
        return self._examples[i]


# ---------------------------------------------------------------------------
# Test: gt_counts equality
# ---------------------------------------------------------------------------


def test_evaluator_gt_counts_equals_instance_lengths() -> None:
    """gt_counts[i] == len(dataset[i].instances) for all i."""
    from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper

    ds = _SimpleDataset()
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    evaluator = Evaluator(cfg)
    wrapper = make_stub_wrapper(dim=8, working=True)

    result = evaluator.evaluate(wrapper, ds, return_per_example_iou=True)
    assert isinstance(result, tuple)
    assert len(result) == 3
    _report, _per_iou, gt_counts = result

    assert gt_counts is not None
    assert len(gt_counts) == len(ds)
    for i, count in enumerate(gt_counts):
        assert count == len(ds._examples[i].instances), (
            f"gt_counts[{i}]={count} != len(instances)={len(ds._examples[i].instances)}"
        )


# ---------------------------------------------------------------------------
# Test: pick_samples with gt_counts is identical to fallback
# ---------------------------------------------------------------------------


def test_pick_samples_gt_counts_identical_to_fallback() -> None:
    """pick_samples with gt_counts= returns the same indices as gt_counts=None."""
    ds = _SimpleDataset()
    # Manually reset call tracking since __getitem__ was called in setUp
    ds._getitem_calls.clear()

    per_iou = [0.5, 0.0, 0.8]  # index-aligned to 3 examples
    gt_counts = [1, 0, 2]

    # With gt_counts provided (no decode)
    indices_with_counts = pick_samples(per_iou, ds, count=3, gt_counts=gt_counts)

    # With gt_counts=None (decode-based fallback) — reset call tracking
    ds._getitem_calls.clear()
    indices_fallback = pick_samples(per_iou, ds, count=3, gt_counts=None)

    assert sorted(indices_with_counts) == sorted(indices_fallback), (
        f"gt_counts path {indices_with_counts} != fallback {indices_fallback}"
    )


def test_pick_samples_fallback_none_matches_current_behavior() -> None:
    """gt_counts=None falls back to dataset.__getitem__ and includes only GT-bearing images."""
    ds = _SimpleDataset()
    ds._getitem_calls.clear()

    per_iou = [0.5, 0.0, 0.8]
    # image_1 has 0 instances — should be excluded
    indices = pick_samples(per_iou, ds, count=3, gt_counts=None)

    # The fallback path calls __getitem__ on all candidates
    assert 1 not in indices, "image_1 has 0 GT instances and must be excluded"
    assert 0 in indices
    assert 2 in indices


# ---------------------------------------------------------------------------
# Test: __getitem__ NOT called over full range when gt_counts provided
# ---------------------------------------------------------------------------


def test_pick_samples_no_full_decode_when_gt_counts_provided() -> None:
    """When gt_counts is provided, __getitem__ is NOT called for viz selection.

    Only the 'count' selected images decode later in render_eval_pair — but
    pick_samples itself issues zero __getitem__ calls.
    """
    ds = _SimpleDataset()
    ds._getitem_calls.clear()

    per_iou = [0.5, 0.0, 0.8]
    gt_counts = [1, 0, 2]

    _selected = pick_samples(per_iou, ds, count=3, gt_counts=gt_counts)

    # pick_samples itself must not call __getitem__ at all when gt_counts provided
    assert ds._getitem_calls == [], (
        f"pick_samples called __getitem__ on indices {ds._getitem_calls} "
        "even though gt_counts was provided"
    )


def test_pick_samples_full_decode_without_gt_counts() -> None:
    """When gt_counts=None, __getitem__ IS called over the full candidate range."""
    ds = _SimpleDataset()
    ds._getitem_calls.clear()

    per_iou = [0.5, 0.0, 0.8]
    _selected = pick_samples(per_iou, ds, count=3, gt_counts=None)

    # The fallback path must have called __getitem__ for all 3 indices to check instances
    called_indices = sorted(set(ds._getitem_calls))
    assert called_indices == [0, 1, 2], (
        f"Expected __getitem__ called on [0, 1, 2] but got {called_indices}"
    )


# ---------------------------------------------------------------------------
# Test: SemanticEvaluator returns None 3-tuple
# ---------------------------------------------------------------------------


def test_semantic_evaluator_returns_none_gt_counts() -> None:
    """SemanticEvaluator.evaluate(return_per_example_iou=True) returns (report, ious, None)."""
    from custom_sam_peft.data.base import SemanticTarget
    from custom_sam_peft.eval.semantic_evaluator import SemanticEvaluator
    from tests.fixtures.tiny_sam3_stub import TinySam3Stub

    class _SemanticDataset:
        class_names: ClassVar[list[str]] = ["road", "building"]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, i: int) -> Example:
            labels = torch.zeros(8, 8, dtype=torch.int64)
            labels[:4, :4] = 1  # road
            return Example(
                image=torch.zeros(3, 8, 8),
                image_id=f"img_{i}",
                prompts=TextPrompts(classes=self.class_names),
                semantic=SemanticTarget(labels=labels, ignore_index=255),
            )

    ds = _SemanticDataset()
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)

    stub = TinySam3Stub()
    evaluator = SemanticEvaluator(cfg)
    result = evaluator.evaluate(stub, ds, return_per_example_iou=True)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    _report, per_ious, gt_counts = result
    assert gt_counts is None, f"SemanticEvaluator must return None gt_counts, got {gt_counts!r}"
    assert isinstance(per_ious, list)


def test_semantic_evaluator_none_fallback_in_pick_samples() -> None:
    """None gt_counts from SemanticEvaluator routes pick_samples to the fallback without error."""
    ds = _SimpleDataset()
    ds._getitem_calls.clear()

    per_iou = [0.5, 0.0]
    # Simulate what happens when the semantic evaluator returns None gt_counts
    gt_counts = None
    # Should not raise; falls back to decode path
    indices = pick_samples(per_iou, ds, count=2, gt_counts=gt_counts)
    # image_1 (index 1) has 0 instances → excluded
    assert 1 not in indices
