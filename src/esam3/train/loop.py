"""Inner training step / epoch loop. Implementation deferred to spec/training-loop."""

from __future__ import annotations

from typing import Any

from esam3.config.schema import TrainHyperparams


def run_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    cfg: TrainHyperparams,
    step: int,
) -> int:
    """Run one epoch. Returns the updated global step counter."""
    raise NotImplementedError("filled in by spec: spec/training-loop")
