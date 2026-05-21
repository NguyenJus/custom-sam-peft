"""Pure-function tests for eval/postprocess.py."""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch

from custom_sam_peft.eval.postprocess import queries_to_coco_results


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
