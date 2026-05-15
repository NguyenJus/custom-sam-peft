"""HuggingFace `datasets` adapter. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.data.base import Dataset, Example


class HFDataset:
    def __init__(self, name: str, split: str, prompt_mode: str) -> None:
        self.name = name
        self.split = split
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by spec: spec/data-loading")


@register("dataset", "hf")
def build_hf(cfg: dict[str, Any]) -> Dataset:
    return HFDataset(
        name=cfg["name"],
        split=cfg["split"],
        prompt_mode=cfg["prompt_mode"],
    )
