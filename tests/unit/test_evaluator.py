"""Orchestration tests for eval/evaluator.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport


def test_evaluate_full_returns_metrics_report(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5])
    report = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert isinstance(report, MetricsReport)
    assert report.n_images == 2
    assert report.per_class, "full mode must populate per_class"
    assert "cat" in report.per_class
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
        patch("custom_sam_peft.eval.evaluator._int_image_id", return_value=42),
        pytest.raises(RuntimeError, match="image_id hash collision"),
    ):
        Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)


def test_evaluate_disables_grad(tiny_text_dataset):
    """grad must be disabled inside model.forward during evaluate()."""
    grad_enabled_during_forward: list[bool] = []

    class GradSpyModel:
        """Minimal model that records torch.is_grad_enabled() on each forward."""

        training = False

        def __call__(self, image: Any, prompts: Any, box_hints: Any) -> dict:
            grad_enabled_during_forward.append(torch.is_grad_enabled())
            b = image.shape[0]
            q = 1
            h, w = image.shape[-2], image.shape[-1]
            return {
                "pred_logits": torch.zeros(b, q, 1),
                "pred_boxes": torch.zeros(b, q, 4),
                "pred_masks": torch.zeros(b, q, h, w),
                "presence_logit_dec": torch.zeros(b, 1),
            }

    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5])
    Evaluator(cfg).evaluate(GradSpyModel(), tiny_text_dataset)

    assert grad_enabled_during_forward, "model was never called"
    assert all(not enabled for enabled in grad_enabled_during_forward), (
        "grad was enabled during at least one forward pass"
    )
    # Grad should be restored after evaluate() returns.
    assert torch.is_grad_enabled(), "grad not restored after evaluate()"


def test_evaluate_single_dataset_traversal(stub_model):
    """Each dataset index must be fetched exactly once during evaluate()."""
    from custom_sam_peft.data.base import Example, Instance, TextPrompts

    access_counts: dict[int, int] = {}

    class CountingDataset:
        class_names: ClassVar[list[str]] = ["cat"]

        def __len__(self) -> int:
            return 3

        def __getitem__(self, i: int) -> Example:
            access_counts[i] = access_counts.get(i, 0) + 1
            h = w = 8
            image = torch.zeros(3, h, w)
            mask = torch.zeros(h, w, dtype=torch.bool)
            mask[:4, :4] = True
            return Example(
                image=image,
                image_id=f"img_{i}",
                prompts=TextPrompts(classes=["cat"]),
                instances=[
                    Instance(
                        mask=mask,
                        class_id=0,
                        box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
                    ),
                ],
            )

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5])
    Evaluator(cfg).evaluate(stub_model, CountingDataset())

    assert set(access_counts.keys()) == {0, 1, 2}, "not all indices were accessed"
    for idx, count in access_counts.items():
        assert count == 1, f"index {idx} was accessed {count} times (expected exactly 1)"


def test_evaluate_returns_per_example_iou_when_requested(stub_model, tiny_text_dataset):
    """When return_per_example_iou=True, return (MetricsReport, list[float])
    aligned with dataset indices."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5])
    out = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset, return_per_example_iou=True)
    assert isinstance(out, tuple)
    report, ious = out
    assert isinstance(report, MetricsReport)
    assert isinstance(ious, list)
    assert len(ious) == len(tiny_text_dataset)
    assert all(0.0 <= v <= 1.0 or v != v for v in ious)  # 0..1 or NaN


def test_evaluate_default_unchanged_returns_report_only(stub_model, tiny_text_dataset):
    """Backward-compat: omitting the flag returns MetricsReport, not a tuple."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5])
    out = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert not isinstance(out, tuple)
    assert isinstance(out, MetricsReport)


# ---------------------------------------------------------------------------
# Device-placement contract: Evaluator must move dataset images to the model's
# device before forward. Regression for the manual GPU pass on issue #44:
# tests/gpu/test_real_train_overfits crashed with
# `Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type) should be
# the same` because Evaluator passed CPU dataset tensors straight to a CUDA
# model. The stub here pins its parameter on a non-CPU sentinel device
# ("meta") so the test fails if `.to(device)` is dropped.
# ---------------------------------------------------------------------------


class _DeviceRecordingStub(torch.nn.Module):
    """Records image.device per forward; param lives on a sentinel device."""

    def __init__(self, param_device: str = "meta") -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1, device=param_device))
        self.received_image_devices: list[torch.device] = []

    def forward(
        self, image: torch.Tensor, prompts: Any, box_hints: Any = None
    ) -> dict[str, torch.Tensor]:
        del prompts, box_hints
        self.received_image_devices.append(image.device)
        b = image.shape[0]
        # Outputs are CPU and independent of the meta param so downstream
        # postprocess.queries_to_coco_results works without GPU.
        return {
            "pred_logits": torch.zeros(b, 4, 1),
            "pred_boxes": torch.zeros(b, 4, 4),
            "pred_masks": torch.zeros(b, 4, 16, 16),
            "presence_logit_dec": torch.zeros(b, 1),
        }


def test_evaluate_moves_image_to_model_device(tiny_text_dataset) -> None:
    """Evaluator must call `.to(device)` on dataset images before forward."""
    stub = _DeviceRecordingStub(param_device="meta")
    cfg = EvalConfig(mode="lite", lite_max_images=2, iou_thresholds=[0.5])
    Evaluator(cfg).evaluate(stub, tiny_text_dataset)
    assert stub.received_image_devices, "model.forward was never called"
    assert all(d.type == "meta" for d in stub.received_image_devices), (
        f"expected every forward to receive a meta-device image; got "
        f"{[str(d) for d in stub.received_image_devices]}"
    )


def test_evaluate_falls_back_to_cpu_for_parameterless_model(tiny_text_dataset) -> None:
    """A model with no parameters defaults to CPU device (no StopIteration)."""

    class _Parameterless(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[torch.device] = []

        def forward(self, image: torch.Tensor, prompts: Any, box_hints: Any = None):
            del prompts, box_hints
            self.seen.append(image.device)
            b = image.shape[0]
            return {
                "pred_logits": torch.zeros(b, 4, 1),
                "pred_boxes": torch.zeros(b, 4, 4),
                "pred_masks": torch.zeros(b, 4, 16, 16),
                "presence_logit_dec": torch.zeros(b, 1),
            }

    stub = _Parameterless()
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5])
    Evaluator(cfg).evaluate(stub, tiny_text_dataset)
    assert stub.seen and all(d.type == "cpu" for d in stub.seen)
