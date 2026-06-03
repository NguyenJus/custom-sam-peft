"""Ladder counter + grace tests (#264). CPU-only, no model.

Tests cover:
  - basic improvement/counter behaviour
  - cold-mAP grace: counter never accrues while mAP == 0.0 (woken stays False)
  - warmup_floor_steps backstop: counter does not accrue below the step floor
    even after woken fires
  - combined: counter begins only when BOTH conditions are satisfied
  - early-stop fires after exactly stop_patience non-improving evals once awake
  - disabled early-stop never fires
  - None mAP is a no-op
  - state_dict / load_state_dict round-trip (including 'woken')
  - LR is a pure function of step (unaffected by mAP)
"""

from __future__ import annotations

import math

import torch

from custom_sam_peft.config.schema import (
    EarlyStopConfig,
    TrainHyperparams,
)
from custom_sam_peft.train.ladder import LadderState
from custom_sam_peft.train.trainer import _POLY_POWER, _build_scheduler


def _cfg(**train_kw: object):
    """A minimal object exposing cfg.train.early_stop."""

    class _Cfg:
        train = TrainHyperparams(epochs=1, **train_kw)  # type: ignore[arg-type]

    return _Cfg()


# ---------------------------------------------------------------------------
# Basic improvement / counter behaviour
# ---------------------------------------------------------------------------


def test_improvement_resets_counter() -> None:
    cfg = _cfg()
    ladder = LadderState()
    for i, m in enumerate([0.5, 0.6, 0.7], start=1001):
        d = ladder.observe(m, step=i, cfg=cfg)
        assert not d.should_stop
    assert ladder.evals_without_improvement == 0
    assert ladder.woken is True


def test_stop_fires_after_stop_patience_non_improving_evals() -> None:
    """After woken + step >= warmup_floor_steps, stop fires at exactly stop_patience."""
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10, warmup_floor_steps=1))
    ladder = LadderState()
    # Establish best with first above-zero eval (woken) at a step above the floor.
    ladder.observe(0.5, step=1, cfg=cfg)
    stop = None
    for i in range(2, 13):
        d = ladder.observe(0.5, step=i, cfg=cfg)
        if d.should_stop:
            stop = d
            break
    assert stop is not None
    assert stop.triggering_step == 11  # 10 non-improving evals after the first


def test_min_delta_boundary_is_strict() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10, warmup_floor_steps=1))
    ladder = LadderState()
    ladder.observe(0.500, step=1, cfg=cfg)
    # Exactly +min_delta (0.001) is NOT an improvement (strict >).
    d = ladder.observe(0.501, step=2, cfg=cfg)
    assert ladder.evals_without_improvement == 1
    # Just above is an improvement.
    ladder.observe(0.5021, step=3, cfg=cfg)
    assert ladder.evals_without_improvement == 0
    assert not d.should_stop


def test_observe_none_map_noops() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10, warmup_floor_steps=1))
    ladder = LadderState()
    ladder.observe(0.5, step=1, cfg=cfg)
    before = ladder.evals_without_improvement
    d = ladder.observe(None, step=2, cfg=cfg)
    assert not d.should_stop
    assert ladder.evals_without_improvement == before


# ---------------------------------------------------------------------------
# Acceptance criterion (a): cold-mAP grace — counter never climbs while mAP==0
# ---------------------------------------------------------------------------


def test_cold_map_never_stops_counter_stays_zero() -> None:
    """Feed mAP=0.0 for well more than stop_patience evals at various steps.

    With both steps below and above warmup_floor_steps: the counter must stay 0
    and woken must stay False because mAP never exceeds 0.0.
    """
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=5, warmup_floor_steps=3))
    ladder = LadderState()
    for step in range(0, 20):
        d = ladder.observe(0.0, step=step, cfg=cfg)
        assert not d.should_stop, f"should not stop at step={step}"
        assert ladder.evals_without_improvement == 0, f"counter accrued at step={step}"
    assert ladder.woken is False


def test_cold_then_nonzero_eventually_stops() -> None:
    """A run that stays cold then wakes stops after exactly stop_patience plateaued evals."""
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=5, warmup_floor_steps=1))
    ladder = LadderState()
    # Feed many zero evals — none should stop and counter stays 0.
    for step in range(0, 15):
        d = ladder.observe(0.0, step=step, cfg=cfg)
        assert not d.should_stop
        assert ladder.evals_without_improvement == 0
    assert ladder.woken is False
    # First non-zero eval wakes the run.
    ladder.observe(0.4, step=15, cfg=cfg)
    assert ladder.woken is True
    assert ladder.evals_without_improvement == 0
    # Now plateau: counter should climb and stop at stop_patience.
    stop = None
    for step in range(16, 22):
        d = ladder.observe(0.4, step=step, cfg=cfg)
        if d.should_stop:
            stop = d
            break
    assert stop is not None
    assert stop.triggering_step == 20  # 5 non-improving evals after the first 0.4


# ---------------------------------------------------------------------------
# Acceptance criterion (b): warmup_floor backstop
# ---------------------------------------------------------------------------


def test_warmup_floor_backstop_delays_counter() -> None:
    """mAP flickers >0 BEFORE warmup_floor_steps then plateaus.

    The counter must not begin accruing until step >= warmup_floor_steps.
    """
    floor = 10
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=5, warmup_floor_steps=floor))
    ladder = LadderState()
    # mAP > 0 at step 2 → woken=True but step < floor → grace NOT lifted.
    ladder.observe(0.3, step=2, cfg=cfg)
    assert ladder.woken is True
    # Steps 3-9: still below floor, counter must NOT accrue.
    for step in range(3, floor):
        d = ladder.observe(0.3, step=step, cfg=cfg)
        assert not d.should_stop
        assert ladder.evals_without_improvement == 0, f"counter accrued before floor at step={step}"
    # Steps at and above floor: counter NOW accrues.
    for step in range(floor, floor + 5):
        ladder.observe(0.3, step=step, cfg=cfg)
    assert ladder.evals_without_improvement == 5


def test_warmup_floor_alone_blocks_if_not_woken() -> None:
    """Even above warmup_floor_steps, counter stays 0 while mAP == 0.0."""
    floor = 2
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=3, warmup_floor_steps=floor))
    ladder = LadderState()
    for step in range(floor, floor + 10):
        d = ladder.observe(0.0, step=step, cfg=cfg)
        assert not d.should_stop
        assert ladder.evals_without_improvement == 0


# ---------------------------------------------------------------------------
# Acceptance criterion (c): counter ticks at every successful eval once awake
# ---------------------------------------------------------------------------


def test_counter_ticks_every_eval_once_awake() -> None:
    """Once grace is lifted the counter increments on each non-improving eval."""
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=20, warmup_floor_steps=1))
    ladder = LadderState()
    ladder.observe(0.5, step=1, cfg=cfg)
    for i in range(5):
        ladder.observe(0.5, step=2 + i, cfg=cfg)
    assert ladder.evals_without_improvement == 5


# ---------------------------------------------------------------------------
# Acceptance criterion (d): LR is a pure function of step (unaffected by mAP)
# ---------------------------------------------------------------------------


def test_lr_unaffected_by_map() -> None:
    """The poly LambdaLR scheduler produces the same LR regardless of mAP observations."""
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=3, warmup_floor_steps=1))
    opt1 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)

    train_cfg = cfg.train  # type: ignore[attr-defined]
    # total_steps = 50
    sched1 = _build_scheduler(opt1, type("T", (), {"train": train_cfg})(), 50, "poly")
    sched2 = _build_scheduler(opt2, type("T", (), {"train": train_cfg})(), 50, "poly")

    ladder_high = LadderState()
    ladder_zero = LadderState()

    for step in range(1, 20):
        sched1.step()
        sched2.step()
        ladder_high.observe(0.9, step=step, cfg=cfg)
        ladder_zero.observe(0.0, step=step, cfg=cfg)

    lr1 = opt1.param_groups[0]["lr"]
    lr2 = opt2.param_groups[0]["lr"]
    assert abs(lr1 - lr2) < 1e-12, f"LR should be the same regardless of mAP: {lr1} vs {lr2}"


# ---------------------------------------------------------------------------
# Early-stop disabled
# ---------------------------------------------------------------------------


def test_early_stop_disabled_never_stops() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(enabled=False, warmup_floor_steps=1))
    ladder = LadderState()
    ladder.observe(0.5, step=1, cfg=cfg)
    for i in range(2, 30):
        d = ladder.observe(0.5, step=i, cfg=cfg)
        assert not d.should_stop


# ---------------------------------------------------------------------------
# State dict / load_state_dict (including 'woken')
# ---------------------------------------------------------------------------


def test_state_dict_round_trip() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10, warmup_floor_steps=1))
    ladder = LadderState()
    ladder.observe(0.5, step=1, cfg=cfg)
    ladder.observe(0.5, step=2, cfg=cfg)
    d = ladder.state_dict()
    restored = LadderState()
    restored.load_state_dict(d)
    assert restored.best == ladder.best
    assert restored.evals_without_improvement == ladder.evals_without_improvement
    assert restored.woken == ladder.woken


def test_state_dict_round_trip_woken_false() -> None:
    """woken=False is also correctly persisted and restored."""
    ladder = LadderState()
    d = ladder.state_dict()
    assert d["woken"] is False
    restored = LadderState()
    restored.load_state_dict(d)
    assert restored.woken is False


def test_state_dict_round_trip_woken_true() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10, warmup_floor_steps=1))
    ladder = LadderState()
    ladder.observe(0.7, step=1, cfg=cfg)
    assert ladder.woken is True
    d = ladder.state_dict()
    assert d["woken"] is True
    restored = LadderState()
    restored.load_state_dict(d)
    assert restored.woken is True
    assert restored.best == ladder.best


def test_load_state_dict_without_woken_defaults_false() -> None:
    """Old checkpoints without 'woken' key default to False (backward compat)."""
    old_state = {"best": 0.5, "evals_without_improvement": 3}
    ladder = LadderState()
    ladder.load_state_dict(old_state)
    assert ladder.woken is False
    assert ladder.best == 0.5
    assert ladder.evals_without_improvement == 3


# ---------------------------------------------------------------------------
# Poly LR schedule shape
# ---------------------------------------------------------------------------


def test_poly_schedule_decays_below_base_lr() -> None:
    """After warmup, poly LambdaLR must be strictly below base_lr at the horizon."""
    warmup = 5
    total_steps = 50
    base_lr = 1e-3

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)

    class _FakeCfg:
        class train:
            warmup_steps = warmup

    sched = _build_scheduler(opt, _FakeCfg(), total_steps, "poly")  # type: ignore[arg-type]

    for _ in range(total_steps):
        sched.step()

    final_lr = opt.param_groups[0]["lr"]
    assert final_lr < base_lr * 0.1, f"poly LR did not decay: {final_lr:.6f} >= {base_lr * 0.1:.6f}"


def test_poly_schedule_warmup_ramp() -> None:
    """During warmup the LR ramps linearly from 0 toward base_lr.

    LambdaLR initialises with last_epoch=0 (base_lr), then each .step() call
    increments last_epoch and recomputes lr_lambda(last_epoch). So the first
    .step() call evaluates lr_lambda(1) = (1+1)/warmup.
    """
    warmup = 10
    total_steps = 100
    base_lr = 1e-3

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)

    class _FakeCfg:
        class train:
            warmup_steps = warmup

    sched = _build_scheduler(opt, _FakeCfg(), total_steps, "poly")  # type: ignore[arg-type]

    # First step(): last_epoch becomes 1 → lr_lambda(1) = (1+1)/10 = 0.2 → lr = 0.2e-3.
    sched.step()
    lr_at_step1 = opt.param_groups[0]["lr"]
    assert abs(lr_at_step1 - base_lr * (1 + 1) / warmup) < 1e-12, lr_at_step1

    # At warmup boundary (step warmup-1 → last_epoch=warmup-1): lr_lambda = warmup/warmup = 1.0.
    for _ in range(warmup - 2):
        sched.step()
    lr_at_warmup = opt.param_groups[0]["lr"]
    assert abs(lr_at_warmup - base_lr) < 1e-10, lr_at_warmup


# ---------------------------------------------------------------------------
# Regression: re-zeroed-after-woken vs never-woken cold grace (#264)
# ---------------------------------------------------------------------------


def test_counter_accrues_when_woken_and_above_floor_even_if_map_returns_zero() -> None:
    """mAP=0.0 AFTER waking is non-improving, NOT cold.

    Distinguishes "never woken" cold-grace (counter frozen) from "re-zeroed
    after woken" (counter accrues). We wake the ladder with a positive mAP below
    the floor, then feed mAP=0.0 at steps >= warmup_floor_steps: because the run
    is already woken and grace is lifted, the 0.0 evals are non-improving and the
    counter MUST climb.
    """
    floor = 5
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=20, warmup_floor_steps=floor))
    ladder = LadderState()
    # Wake early, below the floor → woken latches True, counter still frozen.
    ladder.observe(0.3, step=1, cfg=cfg)
    assert ladder.woken is True
    assert ladder.evals_without_improvement == 0
    # Now feed 0.0 at/above the floor: woken + above floor → counter accrues.
    for step in range(floor, floor + 4):
        ladder.observe(0.0, step=step, cfg=cfg)
    assert ladder.evals_without_improvement == 4, (
        "0.0 after waking must be counted as non-improving, not treated as cold"
    )


def test_cold_start_lr_is_analytic_poly_and_never_stops() -> None:
    """Drive a real poly LambdaLR while feeding mAP=0.0 each eval.

    Asserts (a) the ladder never stops and the counter stays 0 (never woken),
    AND (b) the scheduler LR at each step equals the analytic poly value
    base_lr * (1 - min(progress, 1))**_POLY_POWER after warmup — proving the LR
    is a pure function of step, unaffected by the cold metric (#264).
    """
    warmup = 3
    total_steps = 40
    base_lr = 1e-3

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)

    class _FakeCfg:
        class train:
            warmup_steps = warmup

    sched = _build_scheduler(opt, _FakeCfg(), total_steps, "poly")  # type: ignore[arg-type]

    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=5, warmup_floor_steps=0))
    ladder = LadderState()

    for step in range(1, 21):
        sched.step()  # advances last_epoch to `step`
        d = ladder.observe(0.0, step=step, cfg=cfg)
        assert not d.should_stop, f"cold run stopped at step={step}"
        assert ladder.evals_without_improvement == 0, f"counter accrued at step={step}"

        # Analytic poly value for this step (last_epoch == step).
        if step < warmup:
            expected_factor = (step + 1) / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            expected_factor = (1.0 - min(progress, 1.0)) ** _POLY_POWER
        expected_lr = base_lr * expected_factor
        actual_lr = opt.param_groups[0]["lr"]
        assert math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-15), (
            f"step={step}: LR {actual_lr} != analytic poly {expected_lr}"
        )

    assert ladder.woken is False
