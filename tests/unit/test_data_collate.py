"""Tests for data/collate.py."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.data.collate import collate_batch


def _ex(image_id: str, shape: tuple[int, int, int] = (3, 64, 64)) -> Example:
    return Example(
        image=torch.zeros(shape, dtype=torch.float32),
        image_id=image_id,
        prompts=TextPrompts(classes=["a"]),
        instances=[
            Instance(
                mask=torch.zeros((shape[1], shape[2]), dtype=torch.bool),
                class_id=0,
                box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
            )
        ],
    )


def test_collate_stacks_images() -> None:
    batch = collate_batch([_ex("a"), _ex("b"), _ex("c")])
    assert batch["images"].shape == (3, 3, 64, 64)
    assert batch["images"].dtype == torch.float32


def test_collate_keeps_prompts_as_list() -> None:
    a = _ex("a")
    b = Example(
        image=torch.zeros((3, 64, 64)),
        image_id="b",
        prompts=TextPrompts(classes=["b-class"]),
        instances=[],
    )
    c = _ex("c")
    batch = collate_batch([a, b, c])
    assert isinstance(batch["prompts"], list)
    assert len(batch["prompts"]) == 3
    assert isinstance(batch["prompts"][0], TextPrompts)
    assert isinstance(batch["prompts"][1], TextPrompts)
    assert isinstance(batch["prompts"][2], TextPrompts)


def test_collate_keeps_instances_as_list_of_lists() -> None:
    a = _ex("a")
    b = Example(
        image=torch.zeros((3, 64, 64)),
        image_id="b",
        prompts=TextPrompts(classes=["a"]),
        instances=[],
    )
    batch = collate_batch([a, b])
    assert isinstance(batch["instances"], list)
    assert len(batch["instances"]) == 2
    assert len(batch["instances"][0]) == 1
    assert len(batch["instances"][1]) == 0


def test_collate_image_id_order_preserved() -> None:
    batch = collate_batch([_ex("z"), _ex("y"), _ex("x")])
    assert batch["image_ids"] == ["z", "y", "x"]


def test_collate_empty_batch_raises() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_batch([])


def test_collate_image_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError) as exc:
        collate_batch([_ex("a"), _ex("b", shape=(3, 32, 32))])
    msg = str(exc.value)
    assert "(3, 64, 64)" in msg
    assert "(3, 32, 32)" in msg


def test_collate_adds_semantic_key_none_under_instance() -> None:
    from custom_sam_peft.data.base import SemanticTarget  # noqa: F401

    exs = [
        Example(
            image=torch.zeros(3, 8, 8),
            image_id="a",
            prompts=TextPrompts(classes=["cat"]),
            instances=[],
        ),
        Example(
            image=torch.zeros(3, 8, 8),
            image_id="b",
            prompts=TextPrompts(classes=["cat"]),
            instances=[],
        ),
    ]
    out = collate_batch(exs)
    assert out["semantic"] == [None, None]
    assert out["instances"] == [[], []]


def test_collate_carries_semantic_targets() -> None:
    from custom_sam_peft.data.base import SemanticTarget

    tgt = SemanticTarget(torch.zeros(8, 8, dtype=torch.int64), ignore_index=255)
    exs = [
        Example(
            image=torch.zeros(3, 8, 8),
            image_id="a",
            prompts=TextPrompts(classes=["road"]),
            semantic=tgt,
        ),
        Example(
            image=torch.zeros(3, 8, 8),
            image_id="b",
            prompts=TextPrompts(classes=["road"]),
            semantic=tgt,
        ),
    ]
    out = collate_batch(exs)
    assert out["semantic"] == [tgt, tgt]
    assert out["instances"] == [[], []]
