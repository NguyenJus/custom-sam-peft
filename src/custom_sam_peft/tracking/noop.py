"""No-op tracker. Selected via tracking.backend = "none"."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TrainConfig


class NoopTracker:
    """Tracker that drops all calls on the floor."""

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        return None

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        return None

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        return None

    def close(self) -> None:
        return None


@register("tracker", "none")
def build_noop(_cfg: TrainConfig) -> NoopTracker:
    """Factory called by build_tracker for backend='none'."""
    return NoopTracker()
