"""SemanticEvaluator with a CPU stub model + synthetic semantic dataset (§8)."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
import torch

import custom_sam_peft.profiling as prof
from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.eval.semantic_evaluator import SemanticEvaluator
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

# ---------------------------------------------------------------------------
# Fixture: clean profiler state around each test in this module.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_profiler():  # type: ignore[return]
    """Reset + disable before and after every test."""
    prof.reset()
    prof.disable()
    yield
    prof.reset()
    prof.disable()


@pytest.fixture
def stub_semantic_model() -> TinySam3Stub:
    return TinySam3Stub()


@pytest.fixture
def tiny_semantic_dataset():
    class_names = ["road", "building"]  # K = 2

    def make_ex(image_id: str) -> Example:
        h = w = 16
        image = torch.zeros(3, h, w)
        labels = torch.zeros(h, w, dtype=torch.int64)
        labels[:8, :8] = 1  # road
        labels[8:, 8:] = 2  # building
        labels[0, 0] = 255  # void / ignore
        return Example(
            image=image,
            image_id=image_id,
            prompts=TextPrompts(classes=class_names),
            semantic=SemanticTarget(labels=labels, ignore_index=255),
        )

    examples = [make_ex("img_0"), make_ex("img_1")]

    class _InMemDataset:
        class_names: ClassVar[list[str]] = ["road", "building"]

        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, i: int) -> Example:
            return examples[i]

    return _InMemDataset()


def test_evaluate_returns_metrics_report_with_miou(stub_semantic_model, tiny_semantic_dataset):
    ev = SemanticEvaluator(EvalConfig(batch_size=1))
    report = ev.evaluate(stub_semantic_model, tiny_semantic_dataset)
    assert isinstance(report, MetricsReport)
    assert "mIoU" in report.overall and "pixel_acc" in report.overall
    assert 0.0 <= report.overall["mIoU"] <= 1.0
    assert report.per_class  # populated, keyed by class name


def test_evaluate_and_save_writes_task_tagged_json(
    stub_semantic_model, tiny_semantic_dataset, tmp_path
):
    ev = SemanticEvaluator(EvalConfig(batch_size=1))
    ev.evaluate_and_save(stub_semantic_model, tiny_semantic_dataset, tmp_path)
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["task"] == "semantic"
    assert "mIoU" in data["overall"]


def test_per_example_iou_returned(stub_semantic_model, tiny_semantic_dataset):
    ev = SemanticEvaluator(EvalConfig(batch_size=1))
    _report, per_ex, gt_counts = ev.evaluate(
        stub_semantic_model, tiny_semantic_dataset, return_per_example_iou=True
    )
    assert isinstance(per_ex, list)
    assert len(per_ex) == len(tiny_semantic_dataset)
    assert gt_counts is None  # SemanticEvaluator always returns None for gt_counts


# ---------------------------------------------------------------------------
# Profiling buckets — semantic_eval.* (issue #273 §3b)
# ---------------------------------------------------------------------------

_SEMANTIC_BUCKETS = [
    "semantic_eval.total",
    "semantic_eval.forward",
    "semantic_eval.upsample",
    "semantic_eval.transfer",
    "semantic_eval.confusion",
]


class TestSemanticEvalProfileBuckets:
    def test_all_buckets_present_after_evaluate(
        self, stub_semantic_model, tiny_semantic_dataset
    ) -> None:
        """SemanticEvaluator.evaluate must record all five semantic_eval.* buckets."""
        prof.enable()
        prof.reset()

        ev = SemanticEvaluator(EvalConfig(batch_size=1))
        ev.evaluate(stub_semantic_model, tiny_semantic_dataset)

        buckets, _meta = prof.snapshot()
        missing = [b for b in _SEMANTIC_BUCKETS if b not in buckets]
        assert not missing, f"Missing semantic_eval buckets: {missing}; found: {list(buckets)}"

    def test_meta_keys_present(self, stub_semantic_model, tiny_semantic_dataset) -> None:
        """profiling.note() must record n_images, K, and sem_forward_dtype in meta."""
        prof.enable()
        prof.reset()

        ev = SemanticEvaluator(EvalConfig(batch_size=1))
        ev.evaluate(stub_semantic_model, tiny_semantic_dataset)

        _buckets, meta = prof.snapshot()
        assert "n_images" in meta, f"n_images missing from meta: {meta}"
        assert "K" in meta, f"K missing from meta: {meta}"
        assert "sem_forward_dtype" in meta, f"sem_forward_dtype missing from meta: {meta}"

    def test_forwards_counter_incremented(self, stub_semantic_model, tiny_semantic_dataset) -> None:
        """profiling.incr('semantic_eval.forwards') must be called for each forward.

        incr() stores in _META (not _BUCKETS), so the counter is read from meta.
        """
        prof.enable()
        prof.reset()

        ev = SemanticEvaluator(EvalConfig(batch_size=1))
        ev.evaluate(stub_semantic_model, tiny_semantic_dataset)

        _buckets, meta = prof.snapshot()
        assert "semantic_eval.forwards" in meta, (
            f"semantic_eval.forwards counter missing from meta; meta = {list(meta)}"
        )
        assert meta["semantic_eval.forwards"] > 0, (
            "semantic_eval.forwards == 0 in meta; expected > 0"
        )

    def test_buckets_absent_when_profiler_disabled(
        self, stub_semantic_model, tiny_semantic_dataset
    ) -> None:
        """No bucket must appear when the profiler is disabled (strict no-op)."""
        # profiler is disabled by autouse fixture
        ev = SemanticEvaluator(EvalConfig(batch_size=1))
        ev.evaluate(stub_semantic_model, tiny_semantic_dataset)

        buckets, _ = prof.snapshot()
        present = [b for b in _SEMANTIC_BUCKETS if b in buckets]
        assert not present, f"Unexpected semantic_eval buckets when disabled: {present}"
