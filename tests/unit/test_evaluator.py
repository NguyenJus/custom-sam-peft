"""Orchestration tests for eval/evaluator.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.paths import predictions_path


def test_evaluate_full_returns_metrics_report(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    report = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert isinstance(report, MetricsReport)
    assert report.n_images == 2
    assert report.per_class, "full mode must populate per_class"
    assert "cat" in report.per_class
    assert "mAP" in report.overall


def test_evaluate_lite_caps_images_and_skips_per_class(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], batch_size=1)
    report = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert report.n_images == 1
    assert report.per_class == {}


def test_evaluate_does_not_mutate_training_state(stub_model, tiny_text_dataset):
    stub_model.train()
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], batch_size=1)
    Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    assert stub_model.training is True


def test_evaluate_and_save_full_writes_predictions(stub_model, tiny_text_dataset, tmp_path: Path):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], save_predictions=True, batch_size=1)
    out = tmp_path / "out"
    Evaluator(cfg).evaluate_and_save(stub_model, tiny_text_dataset, out)
    assert (out / "metrics.json").exists()
    # Predictions are written via paths.predictions_path → artifacts/predictions_val.jsonl
    assert (out / "artifacts" / "predictions_val.jsonl").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert "overall" in metrics


def test_evaluate_and_save_lite_never_writes_predictions(
    stub_model, tiny_text_dataset, tmp_path: Path
):
    cfg = EvalConfig(
        mode="lite", lite_max_images=1, iou_thresholds=[0.5], save_predictions=True, batch_size=1
    )
    out = tmp_path / "out"
    Evaluator(cfg).evaluate_and_save(stub_model, tiny_text_dataset, out)
    assert (out / "metrics.json").exists()
    assert not (out / "predictions.json").exists()


def test_image_id_collision_detected(stub_model, tiny_text_dataset):
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
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
        """Minimal model that records torch.is_grad_enabled() on each forward.

        Returns multiplex-shaped output (B*K_g, Q, ...) as required by the
        flat (image_chunk, class_group) loop in _iter_predictions.
        """

        training = False

        def __call__(self, image: Any, prompts: Any, support: Any = None) -> dict:
            grad_enabled_during_forward.append(torch.is_grad_enabled())
            b = image.shape[0]
            k_g = len(prompts[0].classes) if prompts else 1
            rows = b * k_g
            q = 1
            h, w = image.shape[-2], image.shape[-1]
            return {
                "pred_logits": torch.zeros(rows, q, 1),
                "pred_boxes": torch.zeros(rows, q, 4),
                "pred_masks": torch.zeros(rows, q, h, w),
                "presence_logit_dec": torch.zeros(rows, 1),
            }

    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], batch_size=1)
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

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    Evaluator(cfg).evaluate(stub_model, CountingDataset())

    assert set(access_counts.keys()) == {0, 1, 2}, "not all indices were accessed"
    for idx, count in access_counts.items():
        assert count == 1, f"index {idx} was accessed {count} times (expected exactly 1)"


def test_evaluate_returns_per_example_iou_when_requested(stub_model, tiny_text_dataset):
    """When return_per_example_iou=True, return (MetricsReport, list[float])
    aligned with dataset indices."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    out = Evaluator(cfg).evaluate(stub_model, tiny_text_dataset, return_per_example_iou=True)
    assert isinstance(out, tuple)
    report, ious = out
    assert isinstance(report, MetricsReport)
    assert isinstance(ious, list)
    assert len(ious) == len(tiny_text_dataset)
    assert all(0.0 <= v <= 1.0 or math.isnan(v) for v in ious)  # 0..1 or NaN


def test_evaluate_default_unchanged_returns_report_only(stub_model, tiny_text_dataset):
    """Backward-compat: omitting the flag returns MetricsReport, not a tuple."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
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
    """Records image.device per forward; param lives on a sentinel device.

    Returns multiplex-shaped output (B*K_g, Q, ...) as required by the
    flat (image_chunk, class_group) loop in _iter_predictions.
    """

    def __init__(self, param_device: str = "meta") -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1, device=param_device))
        self.received_image_devices: list[torch.device] = []

    def forward(
        self, image: torch.Tensor, prompts: Any, support: Any = None
    ) -> dict[str, torch.Tensor]:
        self.received_image_devices.append(image.device)
        b = image.shape[0]
        k_g = len(prompts[0].classes) if prompts else 1
        rows = b * k_g
        # Outputs are CPU and independent of the meta param so downstream
        # postprocess.queries_to_coco_results works without GPU.
        return {
            "pred_logits": torch.zeros(rows, 4, 1),
            "pred_boxes": torch.zeros(rows, 4, 4),
            "pred_masks": torch.zeros(rows, 4, 16, 16),
            "presence_logit_dec": torch.zeros(rows, 1),
        }


def test_evaluate_moves_image_to_model_device(tiny_text_dataset) -> None:
    """Evaluator must call `.to(device)` on dataset images before forward."""
    stub = _DeviceRecordingStub(param_device="meta")
    cfg = EvalConfig(mode="lite", lite_max_images=2, iou_thresholds=[0.5], batch_size=1)
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

        def forward(self, image: torch.Tensor, prompts: Any, support: Any = None):
            self.seen.append(image.device)
            b = image.shape[0]
            k_g = len(prompts[0].classes) if prompts else 1
            rows = b * k_g
            return {
                "pred_logits": torch.zeros(rows, 4, 1),
                "pred_boxes": torch.zeros(rows, 4, 4),
                "pred_masks": torch.zeros(rows, 4, 16, 16),
                "presence_logit_dec": torch.zeros(rows, 1),
            }

    stub = _Parameterless()
    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], batch_size=1)
    Evaluator(cfg).evaluate(stub, tiny_text_dataset)
    assert stub.seen and all(d.type == "cpu" for d in stub.seen)


# ---------------------------------------------------------------------------
# Tests for decomposed private helpers
# ---------------------------------------------------------------------------


def test_iter_predictions_returns_list(stub_model, tiny_text_dataset):
    """_iter_predictions returns a list of COCO-format prediction dicts."""
    from custom_sam_peft.eval.evaluator import _build_coco_gt_from_examples

    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [tiny_text_dataset[0]]
    _gt, _ = _build_coco_gt_from_examples(examples, tiny_text_dataset)
    preds = ev._iter_predictions(stub_model, examples, tiny_text_dataset)
    assert isinstance(preds, list)
    for p in preds:
        assert "image_id" in p
        assert "category_id" in p
        assert "score" in p


def test_aggregate_metrics_returns_metrics_report(stub_model, tiny_text_dataset):
    """_aggregate_metrics wraps compute_coco_map and returns a MetricsReport."""
    from custom_sam_peft.eval.evaluator import _build_coco_gt_from_examples

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    n = len(tiny_text_dataset)
    examples = [tiny_text_dataset[i] for i in range(n)]
    gt, _ = _build_coco_gt_from_examples(examples, tiny_text_dataset)
    preds = ev._iter_predictions(stub_model, examples, tiny_text_dataset)
    report = ev._aggregate_metrics(preds, gt, tiny_text_dataset)
    assert isinstance(report, MetricsReport)
    assert "mAP" in report.overall


def test_maybe_save_predictions_noop_when_run_dir_none(stub_model, tiny_text_dataset):
    """_maybe_save_predictions is a no-op when run_dir is None."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], save_predictions=True, batch_size=1)
    ev = Evaluator(cfg)
    # Should not raise; no file should be written.
    ev._maybe_save_predictions([{"image_id": 1}], run_dir=None)


def test_maybe_save_predictions_uses_canonical_path(tmp_path: Path):
    """_maybe_save_predictions writes to paths.predictions_path, not a bare filename."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], save_predictions=True, batch_size=1)
    ev = Evaluator(cfg)
    preds = [{"image_id": 1, "category_id": 1, "score": 0.9, "segmentation": {}}]
    ev._maybe_save_predictions(preds, run_dir=tmp_path, split="val")
    expected = predictions_path(tmp_path, split="val")
    assert expected.exists(), f"expected predictions at {expected}"


def test_maybe_save_predictions_noop_in_lite_mode(tmp_path: Path):
    """_maybe_save_predictions skips disk I/O in lite mode."""
    cfg = EvalConfig(
        mode="lite", lite_max_images=1, iou_thresholds=[0.5], save_predictions=True, batch_size=1
    )
    ev = Evaluator(cfg)
    ev._maybe_save_predictions([{"image_id": 1}], run_dir=tmp_path, split="val")
    assert not predictions_path(tmp_path, split="val").exists()


# ---------------------------------------------------------------------------
# Flat (image_chunk x class_group) iteration - T9
# ---------------------------------------------------------------------------


def test_iter_predictions_iterates_image_chunks_x_groups(monkeypatch) -> None:
    """Evaluator iterates (image_chunk, class_group) flat; one model call per pair.

    batch_size=2 (resolved), 4 images, 3 classes, MULTIPLEX_CAP=16 -> 1 group per chunk.
    Expect ceil(4/2) * 1 = 2 model calls.
    """
    from unittest.mock import MagicMock

    from custom_sam_peft.data.base import Example, Instance, TextPrompts

    # Patch MULTIPLEX_CAP to 16 (the default); 3 classes < 16 so 1 group per chunk.
    monkeypatch.setattr("custom_sam_peft.eval.evaluator.MULTIPLEX_CAP", 16, raising=False)

    # Build a 4-image, 3-class in-memory dataset.
    class_names = ["cat", "dog", "bird"]

    def _make_ex(idx: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        return Example(
            image=image,
            image_id=f"img_{idx}",
            prompts=TextPrompts(classes=class_names),
            instances=[
                Instance(
                    mask=mask,
                    class_id=0,
                    box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
                )
            ],
        )

    class _DS4:
        class_names: ClassVar[list[str]] = ["cat", "dog", "bird"]

        def __len__(self) -> int:
            return 4

        def __getitem__(self, i: int) -> Example:
            return _make_ex(i)

    dataset = _DS4()
    examples = [dataset[i] for i in range(4)]

    # Mock model: returns (B*K_g, Q, ...) shaped outputs.
    K_g = 3  # 3 classes, 1 group
    Q = 2

    def _mock_forward(images, prompts, support=None):
        B = images.shape[0]
        h, w = images.shape[-2], images.shape[-1]
        rows = B * K_g
        return {
            "pred_logits": torch.zeros(rows, Q, 1),
            "pred_boxes": torch.zeros(rows, Q, 4),
            "pred_masks": torch.zeros(rows, Q, h, w),
            "presence_logit_dec": torch.zeros(rows, 1),
        }

    model = MagicMock(side_effect=_mock_forward)
    model.training = False
    # No parameters — will default to CPU
    del model.parameters

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=2)
    ev = Evaluator(cfg)
    ev._iter_predictions(model, examples, dataset)

    # 4 images / batch_size=2 = 2 chunks, 1 group each -> 2 calls.
    assert model.call_count == 2

    # Each call should carry 3 class names in every prompt.
    for args, _ in model.call_args_list:
        prompts_arg = args[1]
        assert all(len(p.classes) == 3 for p in prompts_arg)


def test_row_outputs_returns_single_row_dict() -> None:
    """_row_outputs(outputs, r) returns a per-row dict shaped (1, ...) for postprocess."""
    from custom_sam_peft.eval.evaluator import _row_outputs

    outputs = {
        "pred_logits": torch.zeros(6, 2, 1),
        "pred_boxes": torch.zeros(6, 2, 4),
        "pred_masks": torch.zeros(6, 2, 4, 4),
        "presence_logit_dec": torch.zeros(6, 1),
    }
    row = _row_outputs(outputs, r=3)
    assert row["pred_logits"].shape == (1, 2, 1)
    assert row["pred_boxes"].shape == (1, 2, 4)
    assert row["pred_masks"].shape == (1, 2, 4, 4)
    assert row["presence_logit_dec"].shape == (1, 1)


def test_row_outputs_skips_non_tensor_values() -> None:
    """_row_outputs drops non-tensor entries (e.g. sam3's prev_encoder_out dict).

    Bug: the real model's forward_grounding returns non-tensor values such as
    ``prev_encoder_out`` (a nested dict) alongside the prediction tensors.
    ``dict[slice]`` raises KeyError; _row_outputs must skip non-tensor entries.
    """
    from custom_sam_peft.eval.evaluator import _row_outputs

    outputs = {
        "pred_logits": torch.randn(4, 3, 1),
        "pred_boxes": torch.randn(4, 3, 4),
        # non-tensor entries that the real model returns:
        "prev_encoder_out": {"x": 1, "y": [2, 3]},
        "encoder_hidden_states": None,
    }
    # Must not raise; non-tensor entries are dropped
    row = _row_outputs(outputs, r=0)

    # Only tensor keys are present
    assert set(row.keys()) == {"pred_logits", "pred_boxes"}

    # Tensor values are correctly sliced to batch dim 1
    assert row["pred_logits"].shape == (1, 3, 1)
    assert row["pred_boxes"].shape == (1, 3, 4)

    # Non-tensor entries are absent
    assert "prev_encoder_out" not in row
    assert "encoder_hidden_states" not in row


# ---------------------------------------------------------------------------
# Regression: eval inside a training progress session must not mutate the
# shared inner-task total (bug #153).
# ---------------------------------------------------------------------------


def test_eval_inside_progress_session_does_not_clobber_inner_total(
    stub_model, tiny_text_dataset
) -> None:
    """Evaluator.evaluate() inside a training progress_session leaves the inner
    task total unchanged (regression for #153: evaluator was calling
    P.reset_inner(total=len(examples)) which overwrote the shared train bar).
    """
    from custom_sam_peft.cli._progress import (
        ProgressKind,
        ProgressMode,
        _state,
        progress_session,
    )

    TRAIN_TOTAL = 100  # deliberately different from dataset size (2)

    with progress_session(
        kind=ProgressKind.TRAIN,
        total_batches_per_epoch=TRAIN_TOTAL,
        mode=ProgressMode.PLAIN,
    ):
        cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
        Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)

        # The plain handle's _total_batches must still equal what the session opened with.
        handle = _state.handle
        assert handle._total_batches == TRAIN_TOTAL, (
            f"Evaluator clobbered the shared inner total: "
            f"expected {TRAIN_TOTAL}, got {handle._total_batches}"
        )
