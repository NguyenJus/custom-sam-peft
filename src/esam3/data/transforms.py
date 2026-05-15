"""Image + prompt augmentations. Implementation deferred to spec/data-loading."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import AugmentationsConfig


def build_train_transforms(cfg: AugmentationsConfig, image_size: int) -> Any:
    raise NotImplementedError("filled in by spec: spec/data-loading")


def build_eval_transforms(image_size: int) -> Any:
    raise NotImplementedError("filled in by spec: spec/data-loading")
