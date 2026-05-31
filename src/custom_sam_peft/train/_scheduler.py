"""Mode-aware scheduler stepping (spec §6.2).

In plateau mode the warmup is a per-step linear ramp written directly to
param_groups; the ReduceLROnPlateau is stepped only at evals (via
LadderState.observe). In non-plateau modes the per-step LambdaLR is stepped
exactly as before. This keeps the branch in one place so run_epoch/train_step
stay otherwise unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

PlateauOrLambda = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau


def step_per_train_step(
    scheduler: Any,
    *,
    global_step: int,
    base_lr: float,
    warmup_steps: int,
    mode: str,
) -> None:
    """Advance the scheduler for ONE training step.

    - non-plateau modes: scheduler.step() (per-step LambdaLR), unchanged.
    - plateau mode: during warmup (global_step < warmup_steps) write
      param_groups LR = base_lr * (global_step + 1) / max(warmup_steps, 1);
      after warmup, no-op (the plateau scheduler is stepped only at evals).
    """
    if mode != "plateau":
        scheduler.step()
        return
    if global_step < warmup_steps:
        factor = (global_step + 1) / max(warmup_steps, 1)
        lr = base_lr * factor
        for group in scheduler.optimizer.param_groups:
            group["lr"] = lr
    # else: hold — ReduceLROnPlateau owns the LR from the first eval onward.
