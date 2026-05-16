"""COCO instance-JSON dataset adapter. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.data.base import Dataset, Example


class COCODataset:
    """Read a COCO instance-segmentation JSON + image folder as a Dataset."""

    def __init__(self, annotations: str, images: str, prompt_mode: str) -> None:
        self.annotations = annotations
        self.images = images
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by spec: spec/data-loading")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by spec: spec/data-loading")


@register("dataset", "coco")
def build_coco(cfg: dict[str, Any]) -> Dataset:
    return COCODataset(
        annotations=cfg["annotations"],
        images=cfg["images"],
        prompt_mode=cfg["prompt_mode"],
    )
