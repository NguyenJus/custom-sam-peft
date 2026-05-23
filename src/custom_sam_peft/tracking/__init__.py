"""Tracking subsystem — Tracker Protocol, build_tracker factory."""

from __future__ import annotations

from typing import cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.tracking.base import Tracker

__all__ = ["Tracker", "build_tracker"]


def build_tracker(cfg: TrainConfig) -> Tracker:
    """Resolve cfg.tracking.backend to a concrete Tracker.

    Imports the chosen backend module lazily so missing optional extras only
    surface when that backend is actually requested. The @register decorator
    in each backend module wires the factory into _registry on first import.
    """
    backend = cfg.tracking.backend  # Literal["tensorboard", "wandb", "none"]
    if backend == "tensorboard":
        from custom_sam_peft.tracking import tensorboard as _tb  # noqa: F401
    elif backend == "wandb":
        from custom_sam_peft.tracking import wandb as _wb  # noqa: F401
    elif backend == "none":
        from custom_sam_peft.tracking import noop as _noop  # noqa: F401
    else:  # pragma: no cover — pydantic Literal rejects this at config-load
        raise ValueError(f"unknown tracking.backend: {backend!r}")
    factory = lookup("tracker", backend)
    return cast(Tracker, factory(cfg))
