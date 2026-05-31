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


def test_cosine_mode_steps_lambda_lr_and_decays() -> None:
    """step_per_train_step with mode='cosine' on a decaying LambdaLR must call
    scheduler.step() so the LR actually falls below base_lr after warmup.

    This test would FAIL under the FIX-1 bug: if step_per_train_step received
    mode='plateau' (the requested schedule) instead of mode='cosine' (the
    effective/fallback schedule), it would take the plateau warmup/no-op branch
    and never call LambdaLR.step(), leaving the LR constant after warmup.
    """
    base_lr = 1e-3
    warmup = 5
    total_steps = 50

    import math

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    opt = _opt(lr=base_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # Drive many steps past warmup using mode="cosine" (effective/fallback schedule).
    for step in range(total_steps):
        step_per_train_step(
            sched, global_step=step, base_lr=base_lr, warmup_steps=warmup, mode="cosine"
        )

    final_lr = opt.param_groups[0]["lr"]
    # After 50 steps with cosine decay, LR must be well below base_lr.
    # At progress=1.0: lr_lambda = 0.5*(1+cos(pi)) = 0.0, so final_lr ~= 0.
    # We just assert it decayed significantly (< 10% of base_lr).
    assert final_lr < base_lr * 0.1, (
        f"LambdaLR cosine decay did not fire: final_lr={final_lr:.6f} >= "
        f"{base_lr * 0.1:.6f}. step_per_train_step may have used mode='plateau' "
        "instead of mode='cosine' (effective/fallback)."
    )
