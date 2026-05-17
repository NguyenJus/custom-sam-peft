"""Tracker protocol — the stable seam between trainer and logging backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tracker(Protocol):
    """Minimal logging contract that every backend must implement."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None: ...
    def log_images(self, step: int, images: dict[str, np.ndarray[Any, Any]]) -> None: ...
    def close(self) -> None: ...
