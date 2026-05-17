"""No-op tracker. Selected via tracking.backend = "none"."""

from __future__ import annotations

from typing import Any

import numpy as np

from esam3._registry import register


class NoopTracker:
    """Tracker that drops all calls on the floor."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        return None

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        return None

    def close(self) -> None:
        return None


@register("tracker", "none")
def build_noop(_cfg: dict[str, Any]) -> NoopTracker:
    """Factory called by trainer's tracker-building dispatch."""
    return NoopTracker()
