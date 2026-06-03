"""Pure-function tests for eval/postprocess.py."""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch

from custom_sam_peft.eval.metrics import coco_max_dets_cap
from custom_sam_peft.eval.postprocess import (
    queries_to_coco_results,
    score_and_topk_filter,
)


def _outputs(
    n: int = 3,
    h: int = 8,
    w: int = 8,
    *,
    logits: float = 0.0,
    presence: float = 0.0,
    boxes: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "pred_logits": torch.full((1, n, 1), logits),
        "pred_boxes": boxes if boxes is not None else torch.full((1, n, 4), 0.5),
        "pred_masks": masks if masks is not None else torch.full((1, n, h, w), -10.0),
        "presence_logit_dec": torch.full((1, 1), presence),
    }


def test_coco_max_dets_cap_is_pycocotools_default_100():
    # pycocotools segm Params default maxDets == [1, 10, 100]; the scorer reads the
    # LAST (max) slice, so the cap the postprocess filter must match is 100.
    assert coco_max_dets_cap() == 100


def test_shapes_and_keys():
    entries = queries_to_coco_results(
        _outputs(n=3), image_id=1, category_id=2, original_hw=(16, 16)
    )
    assert len(entries) == 3
    for e in entries:
        assert set(e) == {"image_id", "category_id", "bbox", "score", "segmentation"}
        assert e["image_id"] == 1
        assert e["category_id"] == 2
        assert isinstance(e["score"], float)
        assert isinstance(e["bbox"], list)
        assert len(e["bbox"]) == 4
        assert isinstance(e["segmentation"], dict)
        assert "counts" in e["segmentation"]
        assert isinstance(e["segmentation"]["counts"], str)


def test_score_formula_obj_times_presence():
    entries = queries_to_coco_results(
        _outputs(n=1, logits=0.0, presence=0.0),
        image_id=1,
        category_id=1,
        original_hw=(8, 8),
    )
    assert entries[0]["score"] == pytest.approx(0.25, abs=1e-6)


def test_box_denorm_cxcywh_to_xywh():
    # cxcywh (0.5, 0.5, 1.0, 1.0) on (H=100, W=200) → xywh [0, 0, 200, 100]
    boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]])
    entries = queries_to_coco_results(
        _outputs(n=1, boxes=boxes),
        image_id=1,
        category_id=1,
        original_hw=(100, 200),
    )
    assert entries[0]["bbox"] == pytest.approx([0.0, 0.0, 200.0, 100.0], abs=1e-4)


def test_mask_upsample_and_threshold_at_zero():
    # 4x4 logits: top-left quadrant positive, rest negative. Upsampled to 8x8.
    m = torch.full((1, 1, 4, 4), -5.0)
    m[..., :2, :2] = 5.0
    entries = queries_to_coco_results(
        _outputs(n=1, masks=m), image_id=1, category_id=1, original_hw=(8, 8)
    )
    decoded = mask_utils.decode(entries[0]["segmentation"])
    # bilinear-then-threshold-at-0 produces roughly the same quadrant ±1px.
    expected_area = 4 * 4
    assert abs(decoded.sum() - expected_area) <= 4


def test_mask_threshold_parameter_changes_output():
    # Use original_hw=(8, 8) so bilinear upsample is a real operation (4x4 → 8x8).
    m = torch.full((1, 1, 4, 4), 0.4)
    e0 = queries_to_coco_results(
        _outputs(n=1, masks=m),
        image_id=1,
        category_id=1,
        original_hw=(8, 8),
        mask_threshold=0.0,
    )
    e5 = queries_to_coco_results(
        _outputs(n=1, masks=m),
        image_id=1,
        category_id=1,
        original_hw=(8, 8),
        mask_threshold=0.5,
    )
    assert mask_utils.decode(e0[0]["segmentation"]).sum() == 64
    assert mask_utils.decode(e5[0]["segmentation"]).sum() == 0


def test_rle_roundtrip():
    m = torch.full((1, 1, 4, 4), 5.0)
    e = queries_to_coco_results(
        _outputs(n=1, masks=m), image_id=1, category_id=1, original_hw=(4, 4)
    )
    decoded = mask_utils.decode(e[0]["segmentation"])
    assert decoded.dtype == np.uint8
    assert decoded.shape == (4, 4)
    assert decoded.all()


def test_empty_queries_returns_empty_list():
    out = {
        "pred_logits": torch.zeros(1, 0, 1),
        "pred_boxes": torch.zeros(1, 0, 4),
        "pred_masks": torch.zeros(1, 0, 4, 4),
        "presence_logit_dec": torch.zeros(1, 1),
    }
    assert queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4)) == []


def test_batch_greater_than_one_raises():
    out = {
        "pred_logits": torch.zeros(2, 1, 1),
        "pred_boxes": torch.zeros(2, 1, 4),
        "pred_masks": torch.zeros(2, 1, 4, 4),
        "presence_logit_dec": torch.zeros(2, 1),
    }
    with pytest.raises(ValueError, match="batch=1"):
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))


def test_nonfinite_pred_logits_raises():
    out = _outputs(n=1)
    out["pred_logits"][0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))


def test_nonfinite_presence_logit_dec_raises():
    out = _outputs(n=1)
    out["presence_logit_dec"][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))


def test_nonfinite_pred_masks_raises():
    out = _outputs(n=1)
    out["pred_masks"][0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))


def test_nonfinite_pred_boxes_raises():
    out = _outputs(n=1)
    out["pred_boxes"][0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))


def test_bbox_clamp_left_edge_clips_width():
    # cx=5, bw=40, W=100, H=100 → x1 = max(0, 5-20) = 0, x2 = min(100, 5+20) = 25
    # So output x=0, bw=25.
    boxes = torch.tensor([[[5.0 / 100.0, 0.5, 40.0 / 100.0, 0.5]]])
    entries = queries_to_coco_results(
        _outputs(n=1, boxes=boxes),
        image_id=1,
        category_id=1,
        original_hw=(100, 100),
    )
    bbox = entries[0]["bbox"]
    assert bbox[0] == pytest.approx(0.0, abs=1e-4)  # x
    assert bbox[2] == pytest.approx(25.0, abs=1e-4)  # bw = x2 - x = 25 - 0


def test_bfloat16_inputs_run_without_error():
    out = {
        "pred_logits": torch.zeros(1, 2, 1, dtype=torch.bfloat16),
        "pred_boxes": torch.full((1, 2, 4), 0.5, dtype=torch.bfloat16),
        "pred_masks": torch.zeros(1, 2, 4, 4, dtype=torch.bfloat16),
        "presence_logit_dec": torch.zeros(1, 1, dtype=torch.bfloat16),
    }
    entries = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))
    assert len(entries) == 2
    for e in entries:
        assert np.isfinite(e["score"])


# --- score_and_topk_filter helper (DRY: shared by lite proxy + exact path) ---


def _inline_scores_and_keep(
    out: dict[str, torch.Tensor], max_dets: int | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Re-implement the ORIGINAL inline logic so the helper can be diffed against it."""
    pred_logits = out["pred_logits"]
    presence = out["presence_logit_dec"]
    p_obj = torch.sigmoid(pred_logits.float()).squeeze(-1).squeeze(0)
    p_presence = torch.sigmoid(presence.float()).reshape(())
    scores = p_obj * p_presence
    n = scores.shape[0]
    keep_idx: torch.Tensor | None = None
    if max_dets is not None and n > max_dets:
        kth = torch.topk(scores, max_dets).values.min()
        keep_idx = (scores >= kth).nonzero(as_tuple=False).squeeze(-1)
        scores = scores[keep_idx]
    return scores, keep_idx


def test_score_and_topk_filter_matches_inline_with_ties_at_kth():
    # 6 distinct high scores + 3 tied at 0.3; max_dets=8 makes the 8th-highest
    # score land INSIDE the tie → the >= threshold keeps all 3 tied (superset).
    scores = [0.9 - i * 0.001 for i in range(6)] + [0.3, 0.3, 0.3]  # 6 distinct + 3 ties
    out = _outputs_with_scores(scores)
    exp_scores, exp_keep = _inline_scores_and_keep(out, max_dets=8)
    got_scores, got_keep = score_and_topk_filter(out, max_dets=8)
    assert got_keep is not None and exp_keep is not None
    assert torch.equal(got_keep, exp_keep)
    assert torch.equal(got_scores, exp_scores)
    # superset semantics: 6 distinct + all 3 tied at 0.3 = 9 survivors (> max_dets=8).
    assert got_keep.shape[0] == 9


def test_score_and_topk_filter_no_filter_when_max_dets_none():
    out = _outputs_with_scores([0.9, 0.5, 0.1])
    exp_scores, exp_keep = _inline_scores_and_keep(out, max_dets=None)
    got_scores, got_keep = score_and_topk_filter(out, max_dets=None)
    assert got_keep is None
    assert exp_keep is None
    assert torch.equal(got_scores, exp_scores)


def test_score_and_topk_filter_no_filter_when_n_le_max_dets():
    out = _outputs_with_scores([0.9, 0.5, 0.1])
    got_scores, got_keep = score_and_topk_filter(out, max_dets=100)
    assert got_keep is None
    assert got_scores.shape[0] == 3


def test_score_and_topk_filter_nonfinite_raises():
    out = _outputs(n=1)
    out["pred_logits"][0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        score_and_topk_filter(out, max_dets=None)


# --- top-N filter tests ---


def _outputs_with_scores(scores: list[float], h: int = 4, w: int = 4) -> dict[str, torch.Tensor]:
    # Encode target post-sigmoid scores via pred_logits with presence fixed so
    # sigmoid(presence)=1 (large positive). score = sigmoid(logit) * ~1.
    n = len(scores)
    logits = torch.tensor(
        [[[torch.logit(torch.tensor(min(max(s, 1e-6), 1 - 1e-6))).item()] for s in scores]]
    )
    return {
        "pred_logits": logits,  # (1, n, 1)
        "pred_boxes": torch.full((1, n, 4), 0.5),
        "pred_masks": torch.full((1, n, h, w), -10.0),
        "presence_logit_dec": torch.full((1, 1), 20.0),  # sigmoid≈1
    }


def test_filter_no_op_when_n_le_cap():
    # N=3 <= cap=100 -> identical entries to unfiltered.
    out = _outputs_with_scores([0.9, 0.5, 0.1])
    base = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))
    filt = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100)
    assert filt == base


def test_filter_keeps_top_cap_by_score():
    # 105 queries, distinct descending scores; cap=100 -> exactly 100 survivors,
    # and they are the 100 highest scores.
    scores = [i / 200.0 for i in range(105, 0, -1)]  # 105 distinct, descending
    out = _outputs_with_scores(scores)
    filt = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100)
    assert len(filt) == 100
    kept = sorted((e["score"] for e in filt), reverse=True)
    expected = sorted(scores, reverse=True)[:100]
    assert kept == pytest.approx(expected, abs=1e-5)


def test_filter_boundary_ties_keep_superset():
    # 102 queries; scores tie exactly at the cap boundary (positions 100,101,102
    # all equal). The >= threshold keeps the SUPERSET (all 3 tied), never < cap.
    scores = [0.9 - i * 0.001 for i in range(99)] + [0.3, 0.3, 0.3]  # 99 distinct + 3 ties
    out = _outputs_with_scores(scores)
    filt = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100)
    # 99 above the tie + all 3 tied at 0.3 = 102 survivors (superset, not truncated to 100).
    assert len(filt) == 102


def test_filter_n_zero_returns_empty():
    out = {
        "pred_logits": torch.zeros(1, 0, 1),
        "pred_boxes": torch.zeros(1, 0, 4),
        "pred_masks": torch.zeros(1, 0, 4, 4),
        "presence_logit_dec": torch.zeros(1, 1),
    }
    assert (
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100)
        == []
    )


def test_filter_none_is_no_filter():
    # max_dets=None (default) preserves the predict/visualize contract: ALL queries.
    scores = [i / 200.0 for i in range(105, 0, -1)]
    out = _outputs_with_scores(scores)
    entries = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))
    assert len(entries) == 105  # unfiltered


def test_batched_rle_decodes_identically():
    # Distinct mask patterns per query; batched RLE must decode bit-identically.
    # Use 6x6 logit space upsampled to 6x6 so growing squares [1x1..5x5] fit cleanly.
    n = 5
    masks = torch.full((1, n, 6, 6), -10.0)
    for i in range(n):
        masks[0, i, : i + 1, : i + 1] = 10.0  # growing top-left square
    out = {
        "pred_logits": torch.zeros(1, n, 1),
        "pred_boxes": torch.full((1, n, 4), 0.5),
        "pred_masks": masks,
        "presence_logit_dec": torch.zeros(1, 1),
    }
    entries = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(6, 6))
    assert len(entries) == n
    for i, e in enumerate(entries):
        decoded = mask_utils.decode(e["segmentation"])
        assert decoded.sum() == (i + 1) * (i + 1)  # the growing square area
