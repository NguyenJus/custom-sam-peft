"""Sam3Wrapper.forward accepts a SupportPrompts container with strict validation."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.data.base import SupportPrompts, TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _wrapper() -> Sam3Wrapper:
    return Sam3Wrapper(TinySam3Stub(), mask_size=8)


def test_forward_accepts_none_support() -> None:
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = w(images, prompts, support=None)
    assert set(out) >= {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_forward_default_support_is_none() -> None:
    """Omitting `support` is equivalent to passing `support=None`."""
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = w(images, prompts)
    assert set(out) >= {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_forward_accepts_per_image_support_boxes() -> None:
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    # K=1, so SupportPrompts.boxes length == B*K == 2 (image-major / class-minor).
    support = SupportPrompts(boxes=[torch.tensor([[1.0, 2.0, 3.0, 4.0]]), None])
    out = w(images, prompts, support=support)
    assert "pred_masks" in out


def test_forward_accepts_support_with_none_boxes() -> None:
    """SupportPrompts(boxes=None) is equivalent to support=None."""
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = w(images, prompts, support=SupportPrompts(boxes=None))
    assert "pred_masks" in out


def test_forward_rejects_mismatched_support_boxes_length() -> None:
    w = _wrapper()
    images = torch.zeros(2, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    # K=1, so expected length is B*K=2; passing length 1 should fail.
    with pytest.raises(ValueError, match=r"len.*boxes"):
        w(images, prompts, support=SupportPrompts(boxes=[None]))


def test_forward_rejects_wrong_support_box_shape() -> None:
    w = _wrapper()
    images = torch.zeros(1, 3, 8, 8)
    prompts = [TextPrompts(classes=["cat"])]
    bad = torch.zeros(2, 5)
    with pytest.raises(ValueError, match=r"\(M_i, 4\)|shape"):
        w(images, prompts, support=SupportPrompts(boxes=[bad]))
