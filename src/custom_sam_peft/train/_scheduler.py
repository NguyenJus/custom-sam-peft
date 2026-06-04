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


def rewind_to_step(scheduler: Any, step: int) -> None:
    """Rewind a per-step LambdaLR to ``step`` completed optimizer steps (#308).

    On resume the trainer re-walks the interrupted epoch from its start (no
    batch-skip), so the scheduler — whose LR is a pure function of its internal
    ``last_epoch`` counter — must be rewound to the epoch boundary
    (``start_epoch * steps_per_epoch``) before the re-walk. Otherwise it
    over-steps the schedule by the number of already-walked batches, drift that
    compounds across successive resumes.

    Establishes the pre-loop invariant a fresh start at ``step`` would have:
    ``last_epoch == step`` and ``lr == base_lr * lr_lambda(step)`` for every
    param group, so the first re-walked batch optimizes at the correct LR rather
    than the stale loaded-checkpoint LR.
    """
    lrs = [
        base * lmbda(step)
        for base, lmbda in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)
    ]
    scheduler.last_epoch = step
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs, strict=True):
        param_group["lr"] = lr
    scheduler._last_lr = lrs
