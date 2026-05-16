"""Checkpoint save/load. Implementation deferred to spec/training-loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_adapter(model: Any, path: Path) -> None:
    raise NotImplementedError("filled in by spec: spec/training-loop")


def save_merged(model: Any, path: Path) -> None:
    raise NotImplementedError("filled in by spec: spec/training-loop")


def load_adapter(model: Any, path: Path) -> Any:
    raise NotImplementedError("filled in by spec: spec/training-loop")
