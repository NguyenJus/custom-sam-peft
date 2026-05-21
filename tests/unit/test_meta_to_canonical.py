"""Unit tests for meta_to_canonical adapter (revised plan)."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.models.matching import CanonicalOutputs, meta_to_canonical


def _raw_outputs(b: int = 2, q: int = 3, h: int = 16) -> dict:
    """Hand-crafted dict that mimics Meta's forward_grounding output shape."""
    return {
        "pred_logits": torch.randn(b, q, 1),
        "pred_boxes": torch.rand(b, q, 4),
        "pred_masks": torch.randn(b, q, h, h),
        "presence_logit_dec": torch.randn(b, 1),
    }


def test_adapter_squeezes_trailing_dims() -> None:
    raw = _raw_outputs(b=2, q=3, h=16)
    canonical = meta_to_canonical(raw)
    assert isinstance(canonical, CanonicalOutputs)
    assert canonical.obj_logits.shape == (2, 3)
    assert canonical.pred_boxes.shape == (2, 3, 4)
    assert canonical.pred_masks.shape == (2, 3, 16, 16)
    assert canonical.img_presence.shape == (2,)


def test_adapter_preserves_values() -> None:
    raw = _raw_outputs(b=1, q=2, h=8)
    canonical = meta_to_canonical(raw)
    assert torch.equal(canonical.obj_logits, raw["pred_logits"].squeeze(-1))
    assert canonical.pred_boxes is raw["pred_boxes"]
    assert canonical.pred_masks is raw["pred_masks"]
    assert torch.equal(canonical.img_presence, raw["presence_logit_dec"].squeeze(-1))


def test_adapter_raises_on_missing_key() -> None:
    raw = _raw_outputs()
    del raw["pred_masks"]
    with pytest.raises(KeyError):
        meta_to_canonical(raw)
