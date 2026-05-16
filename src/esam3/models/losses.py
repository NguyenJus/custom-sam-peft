"""SAM3.1 training losses. Implementation deferred to spec/model-loading."""

from __future__ import annotations

from typing import Any

import torch


def mask_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def box_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def objectness_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")


def total_loss(outputs: dict[str, Any], targets: dict[str, Any]) -> torch.Tensor:
    raise NotImplementedError("filled in by spec: spec/model-loading")
