"""TensorBoardTracker — wraps torch.utils.tensorboard.SummaryWriter.

Requires the [tensorboard] optional extra. Eager ImportError surfaces the
missing dependency at construction (not at step 50).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from esam3._registry import register
from esam3.config.schema import TrainConfig
from esam3.tracking.base import _validate_image

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class TensorBoardTracker:
    """Tracker backend writing to TensorBoard event files under run_dir."""

    def __init__(self, cfg: TrainConfig) -> None:
        # Probe the `tensorboard` package directly — torch.utils.tensorboard
        # is part of torch itself and would import successfully even with the
        # extra missing; the SDK only fails when SummaryWriter is used.
        try:
            import tensorboard  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tracking.backend='tensorboard' requires the [tensorboard] extra. "
                "Install with: pip install 'efficient-sam3-finetuning[tensorboard]'"
            ) from e
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
