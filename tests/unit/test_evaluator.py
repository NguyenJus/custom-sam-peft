"""Orchestration tests for eval/evaluator.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from esam3.config.schema import EvalConfig
from esam3.eval.evaluator import Evaluator
from esam3.eval.metrics import MetricsReport


def test_evaluate_full_returns_metrics_report(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5])
    report = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert isinstance(report, MetricsReport)
    assert report.n_images == 2
    assert isinstance(report.per_class, dict)
    assert "mAP" in report.overall


def test_evaluate_lite_caps_images_and_skips_per_class(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5])
    report = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert report.n_images == 1
    assert report.per_class == {}


def test_evaluate_does_not_mutate_training_state(stub_model, tiny_text_dataset):
    stub_model.train()
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5])
    Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert stub_model.training is True


def test_evaluate_and_save_full_writes_predictions(stub_model, tiny_text_dataset, tmp_path: Path):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], save_predictions=True)
    out = tmp_path / "out"
    Evaluator(cfg).evaluate_and_save(stub_model, tiny_text_dataset, out)
    assert (out / "metrics.json").exists()
    assert (out / "predictions.json").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert "overall" in metrics


def test_evaluate_and_save_lite_never_writes_predictions(
    stub_model, tiny_text_dataset, tmp_path: Path
):
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], save_predictions=True)
    out = tmp_path / "out"
    Evaluator(cfg).evaluate_and_save(stub_model, tiny_text_dataset, out)
    assert (out / "metrics.json").exists()
    assert not (out / "predictions.json").exists()


def test_image_id_collision_detected(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5])
    # Force every image_id to hash to the same int.
    with (
        patch("esam3.eval.evaluator._int_image_id", return_value=42),
        pytest.raises(RuntimeError, match="image_id hash collision"),
    ):
        Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
