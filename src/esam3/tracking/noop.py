"""No-op tracker. Selected via tracking.backend = "none"."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.tracking.base import NoopTracker  # re-exported for public API

__all__ = ["NoopTracker", "build_noop"]


@register("tracker", "none")
def build_noop(_cfg: dict[str, Any]) -> NoopTracker:
    """Factory called by trainer's tracker-building dispatch."""
    return NoopTracker()
