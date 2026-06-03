"""Ladder state persists in training_state and restores on load (spec §8, §14.4)."""

from __future__ import annotations

from pathlib import Path

import torch

from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import (
    ResumeState,
    load_full_state,
    save_full_state,
)
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper
from tests.integration.test_trainer_evaluator_seam import _make_cfg


def test_ladder_round_trips_through_full_state(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)

    state_dir = tmp_path / "checkpoints" / "step_3"
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=opt,
        scheduler=sched,
        global_step=3,
        epoch=0,
        nan_streak=0,
        cfg=cfg,
        ladder={"best": 0.5, "evals_without_improvement": 2, "woken": True},
        best_metric_value=0.5,
        scheduler_kind="poly",
    )

    # Fresh objects to load into.
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg.peft)
    o2 = torch.optim.AdamW([p for p in w2.parameters() if p.requires_grad], lr=1e-4)
    s2 = torch.optim.lr_scheduler.LambdaLR(o2, lr_lambda=lambda s: 1.0)
    rs = load_full_state(state_dir, w2, o2, s2, cfg)

    assert isinstance(rs, ResumeState)
    assert rs.ladder == {"best": 0.5, "evals_without_improvement": 2, "woken": True}
    assert rs.best_metric_value == 0.5
    assert rs.scheduler_kind == "poly"


def test_old_checkpoint_without_ladder_loads(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    state_dir = tmp_path / "checkpoints" / "step_1"
    # Save WITHOUT the new args (defaults) — simulates a pre-#197 payload shape.
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        nan_streak=0,
        cfg=cfg,
    )
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg.peft)
    o2 = torch.optim.AdamW([p for p in w2.parameters() if p.requires_grad], lr=1e-4)
    s2 = torch.optim.lr_scheduler.LambdaLR(o2, lr_lambda=lambda s: 1.0)
    rs = load_full_state(state_dir, w2, o2, s2, cfg)
    assert rs.ladder is None
    assert rs.best_metric_value is None
    assert rs.scheduler_kind is None


def test_legacy_plateau_scheduler_kind_round_trips(tmp_path: Path) -> None:
    """A checkpoint with scheduler_kind='plateau' (legacy) loads without crashing.

    The trainer will log a warning and fall back to cfg.lr_schedule on resume,
    but the checkpoint load itself must succeed and return the stored kind.
    """
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    state_dir = tmp_path / "checkpoints" / "step_5"
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=opt,
        scheduler=sched,
        global_step=5,
        epoch=0,
        nan_streak=0,
        cfg=cfg,
        ladder={"best": 0.3, "evals_without_improvement": 1, "woken": True},
        best_metric_value=0.3,
        scheduler_kind="plateau",  # legacy, removed in #264
    )
    w2 = make_stub_wrapper(dim=8, working=True)
    apply_lora(w2, cfg.peft)
    o2 = torch.optim.AdamW([p for p in w2.parameters() if p.requires_grad], lr=1e-4)
    s2 = torch.optim.lr_scheduler.LambdaLR(o2, lr_lambda=lambda s: 1.0)
    rs = load_full_state(state_dir, w2, o2, s2, cfg)
    # The checkpoint load returns the stored kind; the trainer handles the fallback.
    assert rs.scheduler_kind == "plateau"
    assert rs.ladder is not None
