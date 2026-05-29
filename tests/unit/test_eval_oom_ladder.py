"""Eval OOM ladder integration tests.

Tests cover the end-to-end _iter_predictions behavior:
- Mid-chunk B-OOM: chunk_buf is discarded and the chunk is restarted at the
  smaller batch size (no dup/drop).
- K-rung: at B==1, an OOM halves effective_K and resumes from the current class
  index; completed K-groups' rows are retained (no dup/drop).

The B-halving/floor/retry behaviors that were previously tested via
_eval_forward_with_oom_ladder directly are now covered by tests/unit/test_oom_ladder.py
(test_decision_sequence_b_then_k_then_floor_then_terminal, test_pending_oom_events_emission,
test_empty_cache_guarded_called_when_available).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oom_error() -> torch.cuda.OutOfMemoryError:
    """Construct a torch.cuda.OutOfMemoryError without actually running out of VRAM."""
    return torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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

    def _model(images: torch.Tensor, prompts: list[Any], support: Any = None):
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


def test_eval_k_rung_resumes_mid_chunk_no_dup_no_drop(monkeypatch) -> None:
    """At B==1 with K>1, an OOM on a multi-class group halves effective_K and
    resumes from the current class index; completed K-groups' rows are retained;
    no (image_id, category_id) is duplicated or dropped. Spec §5.2 / §7.3."""
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.data.base import Example, Instance, TextPrompts
    from custom_sam_peft.eval.evaluator import Evaluator

    # 4 classes, start K=4 (MULTIPLEX_CAP high enough). batch_size=1 so B is at
    # the floor immediately and the FIRST OOM goes straight to the K-rung.
    class_names = ["a", "b", "c", "d"]
    monkeypatch.setattr("custom_sam_peft.eval.evaluator.MULTIPLEX_CAP", 4, raising=False)

    def _make_ex(idx: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[:4, :4] = True
        return Example(
            image=image,
            image_id=f"img_{idx}",
            prompts=TextPrompts(classes=class_names),
            instances=[Instance(mask=mask, class_id=0, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
        )

    class _DS:
        class_names: ClassVar[list[str]] = ["a", "b", "c", "d"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> Example:
            return _make_ex(i)

    dataset = _DS()
    calls: list[int] = [0]

    def _model(images, prompts, support=None):
        calls[0] += 1
        k_g = len(prompts[0].classes)
        # First forward sees K_g=4 (the full group) -> OOM. After K halves to 2,
        # forwards with K_g<=2 succeed.
        if k_g > 2:
            raise _make_oom_error()
        b = images.shape[0]
        rows = b * k_g
        h, w = images.shape[-2], images.shape[-1]
        return {
            "pred_logits": torch.zeros(rows, 1, 1),
            "pred_boxes": torch.zeros(rows, 1, 4),
            "pred_masks": torch.zeros(rows, 1, h, w),
            "presence_logit_dec": torch.zeros(rows, 1),
        }

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [dataset[0]]
    preds = ev._iter_predictions(_model, examples, dataset)

    seen: set[tuple[int, int]] = set()
    dups: list[tuple[int, int]] = []
    for p in preds:
        key = (int(p["image_id"]), int(p["category_id"]))
        if key in seen:
            dups.append(key)
        seen.add(key)
    assert not dups, f"duplicate (image_id, category_id): {dups}"
    # All 4 classes (category_id 1..4) must appear exactly once for the 1 image.
    assert {cid for _, cid in seen} == {1, 2, 3, 4}, f"missing/extra classes: {seen}"


def test_eval_k_rung_retains_nonempty_buffer_across_halving(monkeypatch) -> None:
    """chunk_buf rows buffered BEFORE a RETRY_K OOM must survive the K halving.

    Setup:
      - MULTIPLEX_CAP=2, 4 classes ["a","b","c","d"], batch_size=1.
        → effective_K starts at 2; first group is [a,b] (j=0), second is [c,d] (j=2).
      - Forward 1: group [a,b] succeeds → a,b rows enter chunk_buf.
      - Forward 2: group [c,d] OOMs ONCE → RETRY_K decision; K halves to 1.
      - Resume j=2 at K_g=1: [c] succeeds, [d] succeeds.
      - All four a/b/c/d rows must appear in the final prediction list.

    A buffer-retention bug (dropping chunk_buf on RETRY_K) would drop category_ids
    1 and 2 (the a,b rows buffered before the OOM). Spec invariant (f) §7.3.
    """
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.data.base import Example, Instance, TextPrompts
    from custom_sam_peft.eval.evaluator import Evaluator

    class_names = ["a", "b", "c", "d"]
    monkeypatch.setattr("custom_sam_peft.eval.evaluator.MULTIPLEX_CAP", 2, raising=False)

    def _make_ex(idx: int) -> Example:
        h = w = 8
        image = torch.zeros(3, h, w)
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[:4, :4] = True
        return Example(
            image=image,
            image_id=f"img_{idx}",
            prompts=TextPrompts(classes=class_names),
            instances=[Instance(mask=mask, class_id=0, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
        )

    class _DS:
        class_names: ClassVar[list[str]] = ["a", "b", "c", "d"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> Example:
            return _make_ex(i)

    dataset = _DS()

    # Track call count to fire OOM exactly once on the [c,d] group (K_g==2, j==2).
    # After K halves to 1, [c] and [d] are each forwarded alone and succeed.
    cd_oom_fired: list[bool] = [False]

    def _model(images, prompts, support=None):
        classes = list(prompts[0].classes)
        k_g = len(classes)
        # OOM once when we see a 2-class group containing "c" (i.e. the [c,d] group).
        if k_g == 2 and "c" in classes and not cd_oom_fired[0]:
            cd_oom_fired[0] = True
            raise _make_oom_error()
        b = images.shape[0]
        rows = b * k_g
        h, w = images.shape[-2], images.shape[-1]
        return {
            "pred_logits": torch.zeros(rows, 1, 1),
            "pred_boxes": torch.zeros(rows, 1, 4),
            "pred_masks": torch.zeros(rows, 1, h, w),
            "presence_logit_dec": torch.zeros(rows, 1),
        }

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [dataset[0]]
    preds = ev._iter_predictions(_model, examples, dataset)

    # Verify no duplicates.
    seen: set[tuple[int, int]] = set()
    dups: list[tuple[int, int]] = []
    for p in preds:
        key = (int(p["image_id"]), int(p["category_id"]))
        if key in seen:
            dups.append(key)
        seen.add(key)
    assert not dups, f"duplicate (image_id, category_id): {dups}"

    # All 4 classes must appear exactly once.
    found_cids = {cid for _, cid in seen}
    assert found_cids == {1, 2, 3, 4}, f"missing/extra category_ids: {found_cids}"

    # Specifically confirm that category_ids 1 and 2 (the a,b rows buffered BEFORE
    # the halving) are present — proving non-empty chunk_buf was retained across RETRY_K.
    assert 1 in found_cids, "category_id 1 (class 'a', buffered before OOM) was dropped"
    assert 2 in found_cids, "category_id 2 (class 'b', buffered before OOM) was dropped"
