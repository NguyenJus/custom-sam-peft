"""TensorBoardTracker — wraps torch.utils.tensorboard.SummaryWriter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.tracking.base import _validate_image

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class TensorBoardTracker:
    """Tracker backend writing to TensorBoard event files under run_dir."""

    wants_images = True

    def __init__(self, cfg: TrainConfig) -> None:
        self._cfg = cfg
        self._writer: SummaryWriter | None = None
        self._closed = False

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(run_dir))
        # Markdown code fence so TB's text tab renders the YAML monospaced.
        self._writer.add_text("config", "```yaml\n" + yaml.safe_dump(config) + "\n```", 0)

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        if self._writer is None:
            raise RuntimeError("start_run() must be called before log_scalars()")
        for tag, value in values.items():
            if not math.isfinite(value):
                continue
            self._writer.add_scalar(tag, value, step)

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        if self._writer is None:
            raise RuntimeError("start_run() must be called before log_images()")
        for tag, arr in images.items():
            _validate_image(tag, arr)
            self._writer.add_image(tag, arr, step, dataformats="HWC")

    def close(self) -> None:
        if self._closed or self._writer is None:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True


@register("tracker", "tensorboard")
def build_tensorboard(cfg: TrainConfig) -> TensorBoardTracker:
    return TensorBoardTracker(cfg)
