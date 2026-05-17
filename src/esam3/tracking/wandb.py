"""Weights & Biases tracker. Implementation deferred to spec/tracking.

Requires the [wandb] optional extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from esam3._registry import register


class WandBTracker:
    def __init__(self, project: str, entity: str | None) -> None:
        self.project = project
        self.entity = entity

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def close(self) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")


@register("tracker", "wandb")
def build_wandb(cfg: dict[str, Any]) -> WandBTracker:
    return WandBTracker(project=cfg.get("project", "esam3"), entity=cfg.get("entity"))
