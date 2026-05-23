"""Unit tests for Sam3Wrapper using TinySam3Stub (no real model)."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.data.base import BoxPrompts, TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_wrapper_passes_through_single_class_text_prompts() -> None:
    stub = TinySam3Stub(num_queries=2, mask_size=16)
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = wrapper(image, prompts)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_wrapper_rejects_multi_class_text_prompts() -> None:
    """Multi-class prompts are now valid up to MULTIPLEX_CAP; over-cap is rejected."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    prompts = [TextPrompts(classes=too_many)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        wrapper(image, prompts)


def test_wrapper_rejects_mixed_prompt_variants() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [
        TextPrompts(classes=["cat"]),
        BoxPrompts(boxes=torch.zeros(1, 4), class_ids=torch.zeros(1, dtype=torch.long)),
    ]
    with pytest.raises(ValueError, match="same prompt variant"):
        wrapper(image, prompts)


def test_wrapper_rejects_batch_size_mismatch() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, image_size=64, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"])]  # B=2 images but 1 prompt
    with pytest.raises(ValueError, match="len\\(prompts\\)"):
        wrapper(image, prompts)


def test_sam3_wrapper_has_peft_model_slot() -> None:
    from torch import nn

    from custom_sam_peft.models.sam3 import Sam3Wrapper

    wrapper = Sam3Wrapper(nn.Identity(), image_size=8, mask_size=8)
    assert hasattr(wrapper, "peft_model")
    assert wrapper.peft_model is None


def test_multiplex_cap_constant_exists() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    assert MULTIPLEX_CAP == 16


def _imgs(b: int) -> torch.Tensor:
    return torch.zeros(b, 3, 8, 8)


def test_validate_inputs_accepts_K_between_1_and_cap() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    for k in (1, 5, MULTIPLEX_CAP):
        prompts = [TextPrompts(classes=[f"c{i}" for i in range(k)])] * 2
        Sam3Wrapper._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_rejects_K_zero() -> None:
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        Sam3Wrapper._validate_inputs(_imgs(1), [TextPrompts(classes=[])], None)


def test_validate_inputs_rejects_K_over_cap() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        Sam3Wrapper._validate_inputs(_imgs(1), [TextPrompts(classes=too_many)], None)


def test_validate_inputs_rejects_mismatched_class_lists_across_batch() -> None:
    prompts = [TextPrompts(classes=["cat", "dog"]), TextPrompts(classes=["dog", "cat"])]
    with pytest.raises(ValueError, match=r"same.*class"):
        Sam3Wrapper._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_k1_still_passes() -> None:
    Sam3Wrapper._validate_inputs(
        _imgs(3),
        [TextPrompts(classes=["cat"]) for _ in range(3)],
        None,
    )
