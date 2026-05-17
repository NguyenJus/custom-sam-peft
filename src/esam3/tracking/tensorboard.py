"""TensorBoard tracker. Implementation deferred to spec/tracking.

Requires the [tensorboard] optional extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from esam3._registry import register


class TensorBoardTracker:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")

    def close(self) -> None:
        raise NotImplementedError("filled in by spec: spec/tracking")


@register("tracker", "tensorboard")
def build_tensorboard(cfg: dict[str, Any]) -> TensorBoardTracker:
    return TensorBoardTracker(log_dir=cfg.get("log_dir", "./runs"))
