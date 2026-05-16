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
    """Per-image box prompts and their target class ids."""

    boxes: torch.Tensor  # (N, 4) xyxy, pixel coords
    class_ids: torch.Tensor  # (N,) int64


Prompts = TextPrompts | BoxPrompts


@dataclass(frozen=True)
class Instance:
    """Ground-truth instance for one mask in one image."""

    mask: torch.Tensor  # (H, W) bool
    class_id: int
    box: torch.Tensor  # (4,) xyxy


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

    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Example: ...
    @property
    def class_names(self) -> list[str]: ...


def is_dataset(obj: object) -> bool:
    """Structural check used by tests and CLI doctor."""
    return isinstance(obj, Dataset)
