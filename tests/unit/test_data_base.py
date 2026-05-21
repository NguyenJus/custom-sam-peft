"""Tests for data/base.py protocols and dataclasses."""

from __future__ import annotations

import torch

from custom_sam_peft.data.base import (
    BoxPrompts,
    Dataset,
    Example,
    Instance,
    TextPrompts,
    is_dataset,
)


def test_text_prompts_and_box_prompts_are_distinct_types() -> None:
    t = TextPrompts(classes=["cat", "dog"])
    b = BoxPrompts(
        boxes=torch.zeros((2, 4)),
        class_ids=torch.tensor([0, 1]),
    )
    assert isinstance(t, TextPrompts)
    assert isinstance(b, BoxPrompts)


def test_example_holds_image_prompts_and_instances() -> None:
    inst = Instance(
        mask=torch.zeros((4, 4), dtype=torch.bool),
        class_id=0,
        box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    ex = Example(
        image=torch.zeros((3, 4, 4)),
        image_id="img-1",
        prompts=TextPrompts(classes=["cat"]),
        instances=[inst],
    )
    assert ex.image_id == "img-1"
    assert ex.instances[0].class_id == 0


class _FakeDataset:
    def __init__(self) -> None:
        self._items = [
            Example(
                image=torch.zeros((3, 2, 2)),
                image_id=f"i-{i}",
                prompts=TextPrompts(classes=["a"]),
                instances=[],
            )
            for i in range(3)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> Example:
        return self._items[i]

    @property
    def class_names(self) -> list[str]:
        return ["a"]


def test_dataset_protocol_recognizes_conforming_class() -> None:
    ds: Dataset = _FakeDataset()
    assert len(ds) == 3
    assert ds[0].image_id == "i-0"
    assert ds.class_names == ["a"]
    assert is_dataset(ds) is True


def test_dataset_protocol_rejects_nonconforming() -> None:
    assert is_dataset(object()) is False


class _AlmostDataset:
    """Has __len__ and __getitem__ but no class_names — should not match Dataset."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, i: int) -> Example:  # pragma: no cover - never called
        raise NotImplementedError


def test_dataset_protocol_rejects_class_missing_class_names() -> None:
    assert is_dataset(_AlmostDataset()) is False
