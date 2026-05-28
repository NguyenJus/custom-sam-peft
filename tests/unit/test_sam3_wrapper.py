"""Unit tests for Sam3Wrapper using TinySam3Stub (no real model)."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.data.base import BoxPrompts, TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_wrapper_passes_through_single_class_text_prompts() -> None:
    stub = TinySam3Stub(num_queries=2, mask_size=16)
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = wrapper(image, prompts)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_wrapper_rejects_multi_class_text_prompts() -> None:
    """Multi-class prompts are now valid up to MULTIPLEX_CAP; over-cap is rejected."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    prompts = [TextPrompts(classes=too_many)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        wrapper(image, prompts)


def test_wrapper_rejects_mixed_prompt_variants() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [
        TextPrompts(classes=["cat"]),
        BoxPrompts(boxes=torch.zeros(1, 4), class_ids=torch.zeros(1, dtype=torch.long)),
    ]
    with pytest.raises(ValueError, match="same prompt variant"):
        wrapper(image, prompts)


def test_wrapper_rejects_batch_size_mismatch() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"])]  # B=2 images but 1 prompt
    with pytest.raises(ValueError, match="len\\(prompts\\)"):
        wrapper(image, prompts)


def test_sam3_wrapper_has_peft_model_slot() -> None:
    from torch import nn

    from custom_sam_peft.models.sam3 import Sam3Wrapper

    wrapper = Sam3Wrapper(nn.Identity(), mask_size=8)
    assert hasattr(wrapper, "peft_model")
    assert wrapper.peft_model is None


def test_multiplex_cap_constant_exists() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    assert MULTIPLEX_CAP == 16


def _imgs(b: int) -> torch.Tensor:
    return torch.zeros(b, 3, 8, 8)


def _default_wrapper() -> Sam3Wrapper:
    """Return a Sam3Wrapper with default (rgb, 3-channel) settings for _validate_inputs tests."""
    from torch import nn

    return Sam3Wrapper(nn.Identity(), mask_size=8)


def test_validate_inputs_accepts_K_between_1_and_cap() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    w = _default_wrapper()
    for k in (1, 5, MULTIPLEX_CAP):
        prompts = [TextPrompts(classes=[f"c{i}" for i in range(k)])] * 2
        w._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_rejects_K_zero() -> None:
    w = _default_wrapper()
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        w._validate_inputs(_imgs(1), [TextPrompts(classes=[])], None)


def test_validate_inputs_rejects_K_over_cap() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    w = _default_wrapper()
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        w._validate_inputs(_imgs(1), [TextPrompts(classes=too_many)], None)


def test_validate_inputs_rejects_mismatched_class_lists_across_batch() -> None:
    w = _default_wrapper()
    prompts = [TextPrompts(classes=["cat", "dog"]), TextPrompts(classes=["dog", "cat"])]
    with pytest.raises(ValueError, match=r"same.*class"):
        w._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_k1_still_passes() -> None:
    w = _default_wrapper()
    w._validate_inputs(
        _imgs(3),
        [TextPrompts(classes=["cat"]) for _ in range(3)],
        None,
    )


# ---------------------------------------------------------------------------
# Task 11: channel-adapter wiring tests
# ---------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402 — after module-level imports above

from custom_sam_peft.models.sam3 import _Sam3ImageAdapter  # noqa: E402


class _StubBackbone(nn.Module):
    def forward_image(self, images):  # pragma: no cover - shape probe
        return {"_chans": images.shape[1]}


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _StubBackbone()


def test_validate_inputs_accepts_configured_channels_rejects_wrong():
    w = Sam3Wrapper(
        _Sam3ImageAdapter(_StubModel(), channels=5, channel_semantics="freeform"),
        channels=5,
        channel_semantics="freeform",
    )
    # Accept: correct channel count, correct ndim
    w._validate_inputs(
        torch.zeros(1, 5, 8, 8),
        [TextPrompts(classes=["cat"])],
        None,
    )
    # Reject: wrong channel count (3 instead of 5)
    with pytest.raises(ValueError, match=r"\(B, 5, H, W\)"):
        w._validate_inputs(torch.zeros(1, 3, 8, 8), [TextPrompts(classes=["cat"])], None)
    # Reject: ndim != 4
    with pytest.raises(ValueError):
        w._validate_inputs(torch.zeros(5, 8, 8), [], None)


def test_rgb_adapter_is_none_zero_new_params():
    ad = _Sam3ImageAdapter(_StubModel(), channels=3, channel_semantics="rgb")
    assert ad.channel_adapter is None
    base = sum(p.numel() for p in _StubModel().parameters())
    total = sum(p.numel() for p in ad.parameters())
    assert total == base  # zero new params for rgb


def test_freeform_adapter_present_and_trainable():
    ad = _Sam3ImageAdapter(_StubModel(), channels=4, channel_semantics="freeform")
    assert isinstance(ad.channel_adapter, nn.Conv2d)
    assert any(p.requires_grad for p in ad.channel_adapter.parameters())
