"""Save/load roundtrip for save_full_state <-> load_full_state.

Pins the serialization contract directly: every scalar and tensor in the
training state survives a save+load cycle bit-identically. The integration
test in tests/integration/test_train_resume.py exercises the same code-path
end-to-end via Trainer.fit but cannot assert bit-equality because the
trainer re-walks the interrupted epoch on resume (see
src/custom_sam_peft/train/checkpoint.py:7).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from custom_sam_peft.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import (
    ResumeState,
    load_full_state,
    save_full_state,
)
from custom_sam_peft.train.trainer import _build_optimizer, _build_scheduler
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _cfg(tmp_path: Path) -> TrainConfig:
    """Minimal TrainConfig — values don't matter for the roundtrip, only that
    the schema validates and cfg_hash is stable across save+load."""
    return TrainConfig(
        run=RunConfig(name="roundtrip", output_dir=str(tmp_path), seed=42),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations=str(tmp_path / "a.json"), images=str(tmp_path)),
            val=DataSplit(annotations=str(tmp_path / "a.json"), images=str(tmp_path)),
        ),
        peft=PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
        train=TrainHyperparams(
            epochs=2,
            batch_size=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )


def test_save_load_roundtrip_preserves_optimizer_scheduler_rng_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    wrapper = make_stub_wrapper(dim=8, working=True)
    apply_lora(wrapper, cfg.peft)

    trainable = [p for p in wrapper.model.parameters() if p.requires_grad]
    assert trainable, "stub LoRA wrapper has no trainable params"

    optimizer = _build_optimizer("adamw", trainable, cfg.train.learning_rate)
    scheduler = _build_scheduler(optimizer, cfg, total_steps=10)

    # Drive a few real optimizer steps so exp_avg / exp_avg_sq / step are
    # populated. The quadratic loss sum(p.pow(2).sum() for p in trainable)
    # yields nonzero gradients on every parameter without driving the stub's
    # forward — fast and bit-stable.
    for _ in range(3):
        optimizer.zero_grad()
        loss = sum(p.pow(2).sum() for p in trainable)
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Snapshot pre-save state.
    sd_pre = optimizer.state_dict()
    ssd_pre = scheduler.state_dict()
    rng_pre = torch.get_rng_state().clone()

    state_dir = tmp_path / "checkpoints" / "step_3"
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=3,
        epoch=0,
        nan_streak=0,
        box_hint_p=cfg.train.box_hint.p_start,
        cfg=cfg,
    )

    # Mutate RNG between save and load to prove load actually restores it
    # (not just preserves a quiescent process state).
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    # Fresh optimizer + scheduler (load_full_state mutates in-place).
    fresh_opt = _build_optimizer("adamw", trainable, cfg.train.learning_rate)
    fresh_sched = _build_scheduler(fresh_opt, cfg, total_steps=10)
    rs = load_full_state(state_dir, wrapper, fresh_opt, fresh_sched, cfg)

    # ResumeState fields.
    assert isinstance(rs, ResumeState)
    assert rs.start_step == 3
    assert rs.start_epoch == 0
    assert rs.nan_streak == 0
    assert rs.box_hint_p == pytest.approx(cfg.train.box_hint.p_start)

    sd_post = fresh_opt.state_dict()
    ssd_post = fresh_sched.state_dict()

    # Param groups: bit-equal on canonical scalar keys.
    assert len(sd_pre["param_groups"]) == len(sd_post["param_groups"])
    for g_pre, g_post in zip(sd_pre["param_groups"], sd_post["param_groups"], strict=True):
        for key in ("lr", "betas", "weight_decay", "eps"):
            if key in g_pre:
                assert g_pre[key] == g_post[key], f"param_group {key} drift"

    # Per-parameter state: bit-equal on step counter and running moments.
    assert set(sd_pre["state"]) == set(sd_post["state"]), "param IDs changed across roundtrip"
    for pid in sd_pre["state"]:
        st_pre = sd_pre["state"][pid]
        st_post = sd_post["state"][pid]
        if "step" in st_pre:
            step_pre = st_pre["step"]
            step_post = st_post["step"]
            if isinstance(step_pre, torch.Tensor):
                assert torch.equal(step_pre, step_post), f"param {pid}: step tensor drift"
            else:
                assert int(step_pre) == int(step_post), f"param {pid}: step drift"
        for mom_key in ("exp_avg", "exp_avg_sq"):
            if mom_key in st_pre:
                assert torch.equal(st_pre[mom_key], st_post[mom_key]), (
                    f"param {pid}: {mom_key} not bit-equal across save/load"
                )

    # Scheduler state: bit-equal on canonical keys.
    for key in ("last_epoch", "_step_count"):
        if key in ssd_pre:
            assert ssd_pre[key] == ssd_post[key], f"scheduler {key} drift"

    # CPU RNG: bit-equal to pre-save snapshot (load_full_state restored it
    # despite our intermediate manual_seed(12345)).
    assert torch.equal(torch.get_rng_state(), rng_pre), "CPU RNG not restored across load"
