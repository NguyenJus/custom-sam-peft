"""Sam3Wrapper.forward accepts an optional SupportPrompts no-op seam.

After #88 the `box_hint` curriculum is removed: `support` is a reserved,
field-less seam that is accepted but ignored (the forward is text-only).
"""

from __future__ import annotations

import torch

from custom_sam_peft.data.base import TextPrompts
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
