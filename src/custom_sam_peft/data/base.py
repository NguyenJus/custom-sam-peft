"""Data protocols and dataclasses — the stable seam between data and trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class TextPrompts:
    """Open-vocabulary class names used as prompts for one image."""

    classes: list[str]


@dataclass(frozen=True)
class BoxPrompts:
    """Per-image box prompts and their target class ids.

    `boxes` is `(N, 4)` xyxy in pixel coords; converted to normalized cxcywh
    at the collator boundary before reaching the matcher/losses.
    """

    boxes: torch.Tensor  # (N, 4) xyxy, pixel coords
    class_ids: torch.Tensor  # (N,) int64


Prompts = TextPrompts | BoxPrompts


@dataclass(frozen=True)
class Instance:
    """Ground-truth instance for one mask in one image.

    `box` is `(4,)` xyxy in pixel coords; converted to normalized cxcywh at
    the collator boundary before reaching the matcher/losses.
    """

    mask: torch.Tensor  # (H, W) bool
    class_id: int
    box: torch.Tensor  # (4,) xyxy, pixel coords


@dataclass(frozen=True)
class Example:
    """One training/eval example."""

    image: torch.Tensor  # (3, H, W) normalized
    image_id: str
    prompts: Prompts
    instances: list[Instance]


@runtime_checkable
class Dataset(Protocol):
    """Read-only mapping from index to Example, plus a class vocabulary."""

    def __len__(self) -> int:
        pass

    def __getitem__(self, i: int) -> Example:
        pass

    @property
    def class_names(self) -> list[str]:
        pass


def is_dataset(obj: object) -> bool:
    """Structural check used by tests and CLI doctor."""
    return isinstance(obj, Dataset)
