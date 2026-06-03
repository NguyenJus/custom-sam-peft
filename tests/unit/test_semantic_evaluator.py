"""SemanticEvaluator with a CPU stub model + synthetic semantic dataset (§8)."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
import torch

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.data.base import Example, SemanticTarget, TextPrompts
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.eval.semantic_evaluator import SemanticEvaluator
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


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
    _report, per_ex = ev.evaluate(
        stub_semantic_model, tiny_semantic_dataset, return_per_example_iou=True
    )
    assert isinstance(per_ex, list)
    assert len(per_ex) == len(tiny_semantic_dataset)
