"""Tracker protocol — the stable seam between trainer and logging backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tracker(Protocol):
    """Minimal logging contract that every backend must implement.

    Lifecycle: ``__init__`` → ``start_run`` → ``log_*`` … → ``close``.
    """

    def start_run(
        self,
        run_dir: Path,
        config: dict[str, Any],
        resume_from: Path | None = None,
    ) -> None:
        pass

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        pass

    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None:
        pass

    def close(self) -> None:
        pass


def _validate_image(tag: str, arr: np.ndarray[Any, Any]) -> None:
    """Enforce the trainer's image contract: uint8 (H, W, 3).

    Both real backends call this; the message includes the tag so a failing
    image is identifiable in a multi-image log call.
    """
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image '{tag}' must be uint8 (H, W, 3); got dtype={arr.dtype} shape={tuple(arr.shape)}"
        )
