"""Per-step scheduler stepping (spec §6.2).

Every schedule is a per-step LambdaLR (warmup is handled inside the LambdaLR
lr_lambda for all schedules), so the scheduler is stepped exactly once per
optimizer step. This keeps the call in one place so run_epoch/train_step stay
otherwise unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

PlateauOrLambda = torch.optim.lr_scheduler.LRScheduler


def step_per_train_step(scheduler: Any) -> None:
    """Advance the per-step LambdaLR for ONE optimizer step."""
    scheduler.step()
