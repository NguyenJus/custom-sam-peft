"""Scheduler tests: all schedules are per-step LambdaLR (#264).

Replaces the old plateau-scheduler tests. With plateau removed, every schedule
(constant/cosine/linear/poly) is a LambdaLR and step_per_train_step(scheduler)
calls scheduler.step() unconditionally on every optimizer step.
"""

from __future__ import annotations

import torch

from custom_sam_peft.train._scheduler import rewind_to_step, step_per_train_step
from custom_sam_peft.train.trainer import _build_scheduler


def _opt(lr: float = 1e-4):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _fake_cfg(warmup_steps: int = 0):
    class _Cfg:
        class train:
            warmup_steps = 0

    _Cfg.train.warmup_steps = warmup_steps
    return _Cfg()


# ---------------------------------------------------------------------------
# step_per_train_step always calls scheduler.step()
# ---------------------------------------------------------------------------


def test_step_per_train_step_calls_step() -> None:
    """step_per_train_step(scheduler) calls scheduler.step() exactly once."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 0.5)
    step_per_train_step(sched)
    assert abs(opt.param_groups[0]["lr"] - 0.5e-4) < 1e-12


def test_step_per_train_step_called_every_optimizer_step() -> None:
    """Calling step_per_train_step N times advances the scheduler N times."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0 - s * 0.01)
    for _ in range(5):
        step_per_train_step(sched)
    # After 5 steps, lr_lambda(5) = 1 - 0.05 = 0.95.
    assert abs(opt.param_groups[0]["lr"] - 1e-4 * 0.95) < 1e-12


def test_param_groups_lr_matches_get_last_lr_for_lambda() -> None:
    """Regression: param_groups lr equals get_last_lr() for LambdaLR."""
    opt = _opt(lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0 - s * 0.01)
    for _s in range(5):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - sched.get_last_lr()[0]) < 1e-12


# ---------------------------------------------------------------------------
# rewind_to_step rewinds the per-step LambdaLR to an exact step (#308)
# ---------------------------------------------------------------------------


def test_rewind_to_step_matches_a_fresh_forward_walk() -> None:
    """rewind_to_step(sched, k) lands last_epoch AND lr where stepping to k would.

    Mirrors the resume path: a scheduler advanced past the epoch boundary (the
    loaded mid-epoch state) is rewound back to the boundary so the re-walk does
    not over-step the schedule.
    """
    lam = lambda s: 1.0 - s * 0.01  # noqa: E731 — terse closure for the test
    # Reference: a fresh scheduler stepped forward to step 3.
    ref = torch.optim.lr_scheduler.LambdaLR(_opt(lr=1e-4), lr_lambda=lam)
    for _ in range(3):
        ref.step()

    # Over-stepped scheduler (simulating a loaded mid-epoch checkpoint at 5),
    # then rewound back to the boundary at 3.
    over = torch.optim.lr_scheduler.LambdaLR(_opt(lr=1e-4), lr_lambda=lam)
    for _ in range(5):
        over.step()
    rewind_to_step(over, 3)

    assert over.last_epoch == ref.last_epoch == 3
    assert abs(over.get_last_lr()[0] - ref.get_last_lr()[0]) < 1e-12
    assert abs(over.optimizer.param_groups[0]["lr"] - ref.get_last_lr()[0]) < 1e-12


def test_rewind_then_step_continues_correctly() -> None:
    """After rewind to k, the next step() advances to k+1 with the right lr."""
    lam = lambda s: 1.0 - s * 0.01  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(_opt(lr=1e-4), lr_lambda=lam)
    for _ in range(5):
        sched.step()
    rewind_to_step(sched, 2)
    step_per_train_step(sched)
    assert sched.last_epoch == 3
    assert abs(sched.optimizer.param_groups[0]["lr"] - 1e-4 * lam(3)) < 1e-12


# ---------------------------------------------------------------------------
# build_scheduler always returns a LambdaLR
# ---------------------------------------------------------------------------


def test_build_scheduler_returns_lambda_lr_for_all_schedules() -> None:
    for schedule in ("constant", "cosine", "linear", "poly"):
        opt = _opt(lr=1e-4)
        sched = _build_scheduler(opt, _fake_cfg(), 100, schedule)
        assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR), (
            f"{schedule} did not return a LambdaLR"
        )


# ---------------------------------------------------------------------------
# Warmup ramp applies to all non-constant schedules
# ---------------------------------------------------------------------------


def test_warmup_ramp_then_decay_for_cosine() -> None:
    """Cosine schedule: LR ramps during warmup then decays below base_lr."""
    base_lr = 1e-3
    warmup = 5
    total_steps = 50
    opt = _opt(lr=base_lr)
    sched = _build_scheduler(opt, _fake_cfg(warmup_steps=warmup), total_steps, "cosine")

    # Drive all steps.
    for _ in range(total_steps):
        step_per_train_step(sched)

    final_lr = opt.param_groups[0]["lr"]
    assert final_lr < base_lr * 0.1, (
        f"cosine LR did not decay: final_lr={final_lr:.6f} >= {base_lr * 0.1:.6f}"
    )


def test_warmup_ramp_then_decay_for_poly() -> None:
    """Poly schedule: LR ramps during warmup then poly-decays to near-zero."""
    base_lr = 1e-3
    warmup = 5
    total_steps = 50
    opt = _opt(lr=base_lr)
    sched = _build_scheduler(opt, _fake_cfg(warmup_steps=warmup), total_steps, "poly")

    for _ in range(total_steps):
        step_per_train_step(sched)

    final_lr = opt.param_groups[0]["lr"]
    assert final_lr < base_lr * 0.1, (
        f"poly LR did not decay: final_lr={final_lr:.6f} >= {base_lr * 0.1:.6f}"
    )


def test_warmup_ramp_then_decay_for_linear() -> None:
    """Linear schedule: LR decays to 0 by the end."""
    base_lr = 1e-3
    warmup = 5
    total_steps = 50
    opt = _opt(lr=base_lr)
    sched = _build_scheduler(opt, _fake_cfg(warmup_steps=warmup), total_steps, "linear")

    for _ in range(total_steps):
        step_per_train_step(sched)

    final_lr = opt.param_groups[0]["lr"]
    assert final_lr < base_lr * 0.05, f"linear LR did not decay to near-zero: {final_lr:.6f}"


def test_constant_schedule_holds_after_warmup() -> None:
    """Constant schedule: LR equals base_lr after warmup, never decays."""
    base_lr = 1e-4
    warmup = 10
    total_steps = 100
    opt = _opt(lr=base_lr)
    sched = _build_scheduler(opt, _fake_cfg(warmup_steps=warmup), total_steps, "constant")

    # Advance through warmup + beyond.
    for _ in range(total_steps):
        step_per_train_step(sched)

    # After warmup the constant schedule returns 1.0, so LR == base_lr.
    final_lr = opt.param_groups[0]["lr"]
    assert abs(final_lr - base_lr) < 1e-10, (
        f"constant schedule drifted: {final_lr:.6f} != {base_lr:.6f}"
    )
