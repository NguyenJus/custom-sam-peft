"""Unit tests for the model-dependent eval viz pass (CPU-only, tiny stub)."""

from __future__ import annotations

import torch
from PIL import Image

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.visualize import render_eval_pair
from custom_sam_peft.models.matching import HungarianMatcher
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _example(class_id: int) -> Example:
    h = w = 8
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[:4, :4] = True
    return Example(
        image=torch.zeros(3, h, w),
        image_id="img_0",
        prompts=TextPrompts(classes=["cat", "dog"]),
        instances=[Instance(mask=mask, class_id=class_id, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
    )


def _matcher() -> HungarianMatcher:
    w = MatcherWeights()
    return HungarianMatcher(
        lambda_l1=w.lambda_l1, lambda_giou=w.lambda_giou, lambda_mask=w.lambda_mask
    )


def test_render_eval_pair_returns_hstacked_image() -> None:
    model = TinySam3Stub()
    ex = _example(class_id=0)
    out = render_eval_pair(
        model,
        ex,
        ["cat", "dog"],
        mask_threshold=0.0,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        matcher=_matcher(),
    )
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    # Hstacked: width >= 2 * source width (8 px each, plus legend/titles add height not width).
    assert out.width >= 16


def test_render_eval_pair_no_gt_class_draws_no_pred_for_that_class() -> None:
    # Image has a single 'cat' (class_id 0) GT; 'dog' (class_id 1) has no GT, so
    # the dog matcher target list is empty and no dog pred is drawn. The call must
    # not raise and must return a composite.
    model = TinySam3Stub()
    ex = _example(class_id=0)
    out = render_eval_pair(
        model, ex, ["cat", "dog"], mask_threshold=0.0,
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], matcher=_matcher(),
    )
    assert isinstance(out, Image.Image)
