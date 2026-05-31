"""Scheduler split: per-step warmup vs per-eval plateau cut (spec §6, §14.3)."""

from __future__ import annotations

import torch

from custom_sam_peft.train._scheduler import step_per_train_step


def _opt(lr: float = 1e-4):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_warmup_ramp_then_hold_in_plateau_mode() -> None:
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    base_lr, warmup = 1e-4, 100
    # Mid-warmup step 49 → lr = base * 50/100.
    step_per_train_step(sched, global_step=49, base_lr=base_lr, warmup_steps=warmup, mode="plateau")
    assert abs(opt.param_groups[0]["lr"] - base_lr * 50 / 100) < 1e-12
    # After warmup, the per-step helper holds the LR (no write, no plateau step).
    opt.param_groups[0]["lr"] = base_lr
    step_per_train_step(
        sched, global_step=150, base_lr=base_lr, warmup_steps=warmup, mode="plateau"
    )
    assert opt.param_groups[0]["lr"] == base_lr


def test_plateau_scheduler_not_stepped_per_train_step() -> None:
    """In plateau mode the per-step helper never calls ReduceLROnPlateau.step()."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    calls = {"n": 0}
    orig = sched.step

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    sched.step = spy  # type: ignore[method-assign]
    for s in range(0, 300):
        step_per_train_step(sched, global_step=s, base_lr=1e-4, warmup_steps=100, mode="plateau")
    assert calls["n"] == 0  # plateau scheduler only ticks at evals


def test_lambda_lr_stepped_per_train_step_in_non_plateau_mode() -> None:
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 0.5)
    step_per_train_step(sched, global_step=0, base_lr=1e-4, warmup_steps=0, mode="cosine")
    # LambdaLR.step() applied the 0.5 multiplier.
    assert abs(opt.param_groups[0]["lr"] - 0.5e-4) < 1e-12


def test_param_groups_lr_matches_get_last_lr_for_lambda() -> None:
    """Regression: the param_groups read equals get_last_lr() for LambdaLR (§14.3)."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0 - s * 0.01)
    for _s in range(5):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - sched.get_last_lr()[0]) < 1e-12
