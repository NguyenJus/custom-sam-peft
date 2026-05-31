"""Ladder counter + staircase tests (spec §6.3, §14.1). CPU-only, no model."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import (
    EarlyStopConfig,
    TrainHyperparams,
)
from custom_sam_peft.train.ladder import LadderState


def _cfg(**train_kw: object):
    """A minimal object exposing cfg.train.early_stop / cfg.train.lr_decay_on_plateau."""

    class _Cfg:
        train = TrainHyperparams(epochs=1, **train_kw)  # type: ignore[arg-type]

    return _Cfg()


def _plateau_scheduler(
    lr: float = 1e-4,
    *,
    patience: int = 5,
    factor: float = 0.1,
    min_lr: float = 1e-6,
    min_delta: float = 0.001,
):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=factor,
        patience=patience,
        threshold=min_delta,
        threshold_mode="abs",
        min_lr=min_lr,
    ), opt


def test_improvement_resets_both_counters() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler()
    ladder = LadderState()
    for i, m in enumerate([0.5, 0.6, 0.7], start=1):
        d = ladder.observe(m, step=i, scheduler=sched, cfg=cfg)
        assert not d.should_stop
    assert ladder.evals_without_improvement == 0
    assert opt.param_groups[0]["lr"] == 1e-4  # no cut


def test_rung1_staircase_one_cut_at_patience() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler(patience=5)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)  # establishes best
    # six more non-improving evals (steps 2..7): patience=5 fires after patience+1
    for i in range(2, 8):
        ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
    assert opt.param_groups[0]["lr"] == 1e-5  # one x0.1 cut


def test_rung2_independent_of_cut_stops_at_stop_patience() -> None:
    cfg = _cfg(early_stop=EarlyStopConfig(stop_patience=10))
    sched, _opt = _plateau_scheduler(patience=5)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)  # best
    stop = None
    for i in range(2, 12):  # ten non-improving evals → stop at the 10th
        d = ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
        if d.should_stop:
            stop = d
            break
    assert stop is not None
    assert stop.triggering_step == 11  # 10 non-improving evals after the first


def test_one_cut_before_stop_with_shipped_defaults() -> None:
    cfg = _cfg()  # patience=5, stop_patience=10, min_delta=0.001
    sched, opt = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    cut_lr = None
    stopped_at = None
    for i in range(2, 12):
        d = ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
        if opt.param_groups[0]["lr"] == 1e-5 and cut_lr is None:
            cut_lr = i
        if d.should_stop:
            stopped_at = i
            break
    # cut after 6 non-improving ReduceLROnPlateau steps (patience=5 fires at patience+1)
    assert cut_lr == 7
    assert stopped_at == 11  # stop after 10 non-improving rung-2 evals (independent counter)
    assert opt.param_groups[0]["lr"] == 1e-5  # exactly one cut


def test_min_lr_floor() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler(patience=1, min_lr=1e-6)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    for i in range(2, 30):
        ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
    assert opt.param_groups[0]["lr"] >= 1e-6


def test_min_delta_boundary_is_strict() -> None:
    cfg = _cfg()  # min_delta=0.001
    sched, _ = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.500, step=1, scheduler=sched, cfg=cfg)
    # Exactly +min_delta is NOT an improvement (strict >).
    d = ladder.observe(0.501, step=2, scheduler=sched, cfg=cfg)
    assert ladder.evals_without_improvement == 1
    # Just above is an improvement.
    ladder.observe(0.5021, step=3, scheduler=sched, cfg=cfg)
    assert ladder.evals_without_improvement == 0
    assert not d.should_stop


def test_shared_improvement_when_early_stop_disabled() -> None:
    """early_stop.enabled=False but plateau mode: rung-1 still cuts on min_delta/mAP (wart §5.4)."""
    cfg = _cfg(early_stop=EarlyStopConfig(enabled=False))
    sched, opt = _plateau_scheduler(patience=5)
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    stopped = False
    for i in range(2, 30):
        d = ladder.observe(0.5, step=i, scheduler=sched, cfg=cfg)
        stopped = stopped or d.should_stop
    assert opt.param_groups[0]["lr"] == 1e-6 or opt.param_groups[0]["lr"] < 1e-4  # cut(s) fired
    assert not stopped  # no early stop when disabled


def test_observe_none_map_noops() -> None:
    cfg = _cfg()
    sched, opt = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    before = ladder.evals_without_improvement
    d = ladder.observe(None, step=2, scheduler=sched, cfg=cfg)
    assert not d.should_stop
    assert ladder.evals_without_improvement == before
    assert opt.param_groups[0]["lr"] == 1e-4  # no cut on a None tick


def test_state_dict_round_trip() -> None:
    cfg = _cfg()
    sched, _ = _plateau_scheduler()
    ladder = LadderState()
    ladder.observe(0.5, step=1, scheduler=sched, cfg=cfg)
    ladder.observe(0.5, step=2, scheduler=sched, cfg=cfg)
    d = ladder.state_dict()
    restored = LadderState()
    restored.load_state_dict(d)
    assert restored.best == ladder.best
    assert restored.evals_without_improvement == ladder.evals_without_improvement
