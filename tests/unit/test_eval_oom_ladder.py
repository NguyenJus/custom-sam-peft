"""_eval_forward_with_oom_ladder: sticky B halving on OOM; <= 1 warn per call.

Also contains a regression test for the mid-chunk OOM duplicate-predictions bug:
when _eval_forward_with_oom_ladder raises on the 2nd group of a chunk, the evaluator
must NOT emit the already-processed group's predictions twice (once for the first
pass, once when the outer while-loop retries the same image_chunk at smaller bs).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pytest
import torch

from custom_sam_peft.eval.evaluator import _eval_forward_with_oom_ladder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oom_error() -> torch.cuda.OutOfMemoryError:
    """Construct a torch.cuda.OutOfMemoryError without actually running out of VRAM."""
    return torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_oom_halves_batch_size_sticky_and_warns_once(caplog) -> None:
    """Ladder halves batch_size on OOM and emits exactly one warning per call."""
    caplog.set_level(logging.WARNING)

    call_count = 0

    def _flaky_model(images, prompts, box_hints=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_oom_error()
        # Second call succeeds — return a minimal valid dict.
        b = images.shape[0]
        return {
            "pred_logits": torch.zeros(b, 1, 1),
            "pred_boxes": torch.zeros(b, 1, 4),
            "pred_masks": torch.zeros(b, 1, 4, 4),
            "presence_logit_dec": torch.zeros(b, 1),
        }

    state: dict = {"batch_size": 4, "warned": False}
    images = torch.zeros(4, 3, 8, 8)
    prompts = [None] * 4

    # First call raises OOM (ladder halves and re-raises for outer loop).
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _eval_forward_with_oom_ladder(_flaky_model, images, prompts, state=state)

    # batch_size should be halved.
    assert state["batch_size"] == 2
    assert state["warned"] is True

    # Exactly one warning about "eval OOM" should have been emitted.
    warns = [r for r in caplog.records if "eval OOM" in r.message]
    assert len(warns) == 1

    # Second call with smaller images succeeds; no additional warning.
    caplog.clear()
    images2 = torch.zeros(2, 3, 8, 8)
    prompts2 = [None] * 2
    result = _eval_forward_with_oom_ladder(_flaky_model, images2, prompts2, state=state)
    assert result is not None

    # No second warning emitted (warned=True suppresses it).
    extra_warns = [r for r in caplog.records if "eval OOM" in r.message]
    assert len(extra_warns) == 0


def test_mid_chunk_oom_does_not_produce_duplicate_predictions(monkeypatch) -> None:
    """Regression: mid-chunk OOM on group 2 must NOT emit group 1's predictions twice.

    Setup:
      - 1 image, 2 class groups (MULTIPLEX_CAP=1 so each class is its own group).
      - First group call succeeds; second group call raises OOM (B>1 -> halved).
      - The outer while-loop re-runs the same image_chunk at the halved batch size.
      - Both calls succeed on the retry.
    Expected: the final predictions list has NO duplicate (image_id, category_id) pairs.
    """
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.data.base import Example, Instance, TextPrompts
    from custom_sam_peft.eval.evaluator import Evaluator

    # Force MULTIPLEX_CAP=1 so each class becomes its own group.
    monkeypatch.setattr("custom_sam_peft.eval.evaluator.MULTIPLEX_CAP", 1, raising=False)

    # Build a 2-image, 2-class in-memory dataset so batch_size=2 triggers the OOM path.
    class_names = ["cat", "dog"]

    def _make_ex(idx: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[:4, :4] = True
        return Example(
            image=image,
            image_id=f"img_{idx}",
            prompts=TextPrompts(classes=class_names),
            instances=[
                Instance(
                    mask=mask,
                    class_id=idx % len(class_names),
                    box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
                )
            ],
        )

    class _TwoImageDataset:
        class_names: ClassVar[list[str]] = ["cat", "dog"]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, i: int) -> Example:
            return _make_ex(i)

    dataset = _TwoImageDataset()

    # Track forward call count.
    call_count: list[int] = [0]

    def _model(images: torch.Tensor, prompts: list[Any], box_hints: Any = None):
        call_count[0] += 1
        # Call 2 is the 2nd group of the first chunk (B=2).
        # Raise OOM to simulate mid-chunk failure.
        if call_count[0] == 2:
            raise _make_oom_error()
        b = images.shape[0]
        k_g = len(prompts[0].classes)
        rows = b * k_g
        h, w = images.shape[-2], images.shape[-1]
        return {
            "pred_logits": torch.zeros(rows, 1, 1),
            "pred_boxes": torch.zeros(rows, 1, 4),
            "pred_masks": torch.zeros(rows, 1, h, w),
            "presence_logit_dec": torch.zeros(rows, 1),
        }

    # Use batch_size=2 so the first pass processes both images together and can hit OOM.
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=2)
    ev = Evaluator(cfg)
    examples = [dataset[i] for i in range(2)]
    preds = ev._iter_predictions(_model, examples, dataset)

    # Build a set of (image_id, category_id) pairs to detect duplicates.
    seen: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int]] = []
    for p in preds:
        key = (int(p["image_id"]), int(p["category_id"]))
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    assert not duplicates, (
        f"Duplicate (image_id, category_id) entries found in predictions: {duplicates}. "
        "This indicates mid-chunk OOM partial results were emitted twice."
    )


def test_oom_raises_at_B1_floor() -> None:
    """Persistent OOM at B=1 raises a RuntimeError mentioning OOM."""

    def _always_oom(images, prompts, box_hints=None):
        raise _make_oom_error()

    state: dict = {"batch_size": 1, "warned": False}
    images = torch.zeros(1, 3, 8, 8)
    prompts = [None]

    with pytest.raises(RuntimeError, match="OOM"):
        _eval_forward_with_oom_ladder(_always_oom, images, prompts, state=state)
