"""Unit tests for _Sam3ImageAdapter.forward multiplex assembly.

We mock the inner model's backbone.forward_image / backbone.forward_text /
forward_grounding so the test exercises only the adapter's input shaping
(img_ids, text_ids, geometric_prompt column count).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.sam3 import (
    MULTIPLEX_CAP,
    _Sam3ImageAdapter,
    validate_forward_inputs,
)


def _make_fake_inner(captured: dict[str, object]) -> MagicMock:
    inner = MagicMock()
    inner.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
    inner.backbone.forward_image.return_value = {"feat": torch.zeros(1)}
    inner.backbone.forward_text.return_value = {"text_feat": torch.zeros(1)}

    def _grounding(*, backbone_out, find_input, find_target, geometric_prompt):
        captured["find_input"] = find_input
        captured["geometric_prompt"] = geometric_prompt
        # Return dummy outputs shaped (B·K, Q, *) — Q=2 here.
        n_rows = find_input.img_ids.shape[0]
        return {
            "pred_logits": torch.zeros(n_rows, 2, 1),
            "pred_boxes": torch.zeros(n_rows, 2, 4),
            "pred_masks": torch.zeros(n_rows, 2, 4, 4),
            "presence_logit_dec": torch.zeros(n_rows, 1),
        }

    inner.forward_grounding.side_effect = _grounding
    return inner


@pytest.mark.parametrize("b,k", [(1, 1), (2, 3), (4, 16)])
def test_adapter_builds_img_text_ids_image_major(b: int, k: int) -> None:
    captured: dict[str, object] = {}
    inner = _make_fake_inner(captured)
    adapter = _Sam3ImageAdapter(inner)

    images = torch.zeros(b, 3, 8, 8)
    classes = [f"c{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]

    out = adapter(images, prompts)

    find_input = captured["find_input"]
    # image-major / class-minor: img_ids = arange(B).repeat_interleave(K)
    assert torch.equal(
        find_input.img_ids,
        torch.arange(b).repeat_interleave(k),
    )
    # text_ids = arange(K).repeat(B)
    assert torch.equal(
        find_input.text_ids,
        torch.arange(k).repeat(b),
    )
    # output first dim is B·K
    assert out["pred_logits"].shape[0] == b * k


@pytest.mark.parametrize("b,k", [(2, 3), (4, 16)])
def test_adapter_calls_forward_text_once_with_k_names(b: int, k: int) -> None:
    captured: dict[str, object] = {}
    inner = _make_fake_inner(captured)
    adapter = _Sam3ImageAdapter(inner)

    classes = [f"c{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]
    adapter(torch.zeros(b, 3, 8, 8), prompts)

    # forward_text called exactly once with the K class names.
    assert inner.backbone.forward_text.call_count == 1
    args, _kwargs = inner.backbone.forward_text.call_args
    # First positional arg is the list of class names.
    assert args[0] == classes


# ---------------------------------------------------------------------------
# validate_forward_inputs free function (§5.3 / §8.5) — extracted from
# Sam3Wrapper._validate_inputs; identical K∈[1,CAP] + shared-class-list checks.
# ---------------------------------------------------------------------------


def _shared_prompts(b: int, k: int) -> list[TextPrompts]:
    classes = [f"c{i}" for i in range(k)]
    return [TextPrompts(classes=classes) for _ in range(b)]


@pytest.mark.parametrize("k", [1, 5, MULTIPLEX_CAP])
def test_validate_forward_inputs_accepts_valid_K(k: int) -> None:
    """Valid K ∈ [1, MULTIPLEX_CAP] with shared class lists passes."""
    validate_forward_inputs(torch.zeros(2, 3, 8, 8), _shared_prompts(2, k), 3)


def test_validate_forward_inputs_rejects_K_zero() -> None:
    """K==0 (empty class list) violates the [1, CAP] bound."""
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        validate_forward_inputs(torch.zeros(1, 3, 8, 8), [TextPrompts(classes=[])], 3)


def test_validate_forward_inputs_rejects_K_over_cap() -> None:
    """K > MULTIPLEX_CAP violates the [1, CAP] bound."""
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        validate_forward_inputs(torch.zeros(1, 3, 8, 8), [TextPrompts(classes=too_many)], 3)


def test_validate_forward_inputs_rejects_mismatched_class_lists() -> None:
    """All prompts in a batch must share the same class list in the same order."""
    prompts = [TextPrompts(classes=["cat", "dog"]), TextPrompts(classes=["dog", "cat"])]
    with pytest.raises(ValueError, match=r"same.*class"):
        validate_forward_inputs(torch.zeros(2, 3, 8, 8), prompts, 3)


def test_validate_forward_inputs_honors_channels_arg() -> None:
    """The channels argument drives the channel-count check (not hardcoded 3)."""
    validate_forward_inputs(torch.zeros(1, 5, 8, 8), [TextPrompts(classes=["cat"])], 5)
    with pytest.raises(ValueError, match=r"\(B, 5, H, W\)"):
        validate_forward_inputs(torch.zeros(1, 3, 8, 8), [TextPrompts(classes=["cat"])], 5)


@pytest.mark.parametrize("b,k", [(1, 1), (2, 3), (4, 16)])
def test_adapter_find_stage_ids_match_arange(b: int, k: int) -> None:
    """Rerouted FindStage img_ids/text_ids (via _multiplex) equal the arange ordering."""
    captured: dict[str, object] = {}
    inner = _make_fake_inner(captured)
    adapter = _Sam3ImageAdapter(inner)

    images = torch.zeros(b, 3, 8, 8)
    classes = [f"c{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]
    adapter(images, prompts)

    find_input = captured["find_input"]
    assert find_input.img_ids.dtype == torch.long
    assert find_input.text_ids.dtype == torch.long
    assert torch.equal(find_input.img_ids, torch.arange(b).repeat_interleave(k))
    assert torch.equal(find_input.text_ids, torch.arange(k).repeat(b))
