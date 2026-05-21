"""Full-state roundtrip + LoRA/QLoRA dispatchers."""

from __future__ import annotations

import json
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
from custom_sam_peft.models.sam3 import Sam3Wrapper
from custom_sam_peft.peft_adapters.lora import apply_lora
from custom_sam_peft.train.checkpoint import (
    ResumeState,
    _has_linear4bit,
    load_full_state,
    save_adapter,
    save_full_state,
)
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _make_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run=RunConfig(name="test", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]),
        train=TrainHyperparams(epochs=1),
    )


def _trainable_optimizer(wrapper: Sam3Wrapper) -> torch.optim.Optimizer:
    params = [p for p in wrapper.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=1e-4)


def test_has_linear4bit_returns_false_for_lora(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    assert _has_linear4bit(wrapper) is False


def test_save_adapter_writes_lora_artifacts(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    out = tmp_path / "adapter"
    save_adapter(wrapper, out)
    assert (out / "adapter_config.json").exists()
    assert not (out / "custom_sam_peft_qlora.json").exists()


def test_save_full_state_writes_training_state_and_adapter(tmp_path: Path) -> None:
    wrapper = make_stub_wrapper(dim=8)
    cfg = _make_cfg(tmp_path)
    apply_lora(wrapper, cfg.peft)
    optimizer = _trainable_optimizer(wrapper)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    state_dir = tmp_path / "checkpoints" / "step_42"
    save_full_state(
        state_dir=state_dir,
        wrapper=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=42,
        epoch=1,
        nan_streak=0,
        box_hint_p=0.5,
        cfg=cfg,
    )

    assert (state_dir / "adapter" / "adapter_config.json").exists()
    state_file = state_dir / "training_state.pt"
    assert state_file.exists()
    state = torch.load(state_file, weights_only=False)
    assert state["global_step"] == 42
    assert state["epoch"] == 1
    assert state["box_hint_p"] == 0.5
    assert state["peft_method"] == "lora"
    assert "optimizer" in state and "scheduler" in state and "rng" in state
    assert "cfg_hash" in state


def test_load_full_state_restores_optimizer_and_step(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    w_a = make_stub_wrapper(dim=8)
    apply_lora(w_a, cfg.peft)
    opt_a = _trainable_optimizer(w_a)
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lr_lambda=lambda s: 1.0)
    for p in w_a.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt_a.step()
    state_dir = tmp_path / "checkpoints" / "step_5"
    save_full_state(state_dir, w_a, opt_a, sched_a, 5, 0, 0, 0.8, cfg)

    w_b = make_stub_wrapper(dim=8)
    apply_lora(w_b, cfg.peft)
    opt_b = _trainable_optimizer(w_b)
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_lambda=lambda s: 1.0)
    rs = load_full_state(state_dir, w_b, opt_b, sched_b, cfg)
    assert isinstance(rs, ResumeState)
    assert rs.start_step == 5
    assert rs.start_epoch == 0
    assert rs.box_hint_p == 0.8
    assert any(opt_b.state.values())


def test_load_full_state_raises_on_peft_method_mismatch(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    w_a = make_stub_wrapper(dim=8)
    apply_lora(w_a, cfg.peft)
    opt_a = _trainable_optimizer(w_a)
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lr_lambda=lambda s: 1.0)
    state_dir = tmp_path / "checkpoints" / "step_0"
    save_full_state(state_dir, w_a, opt_a, sched_a, 0, 0, 0, 1.0, cfg)

    (state_dir / "adapter" / "custom_sam_peft_qlora.json").write_text(
        json.dumps({"format_version": 1, "quant_type": "nf4", "compute_dtype": "bfloat16"})
    )

    w_b = make_stub_wrapper(dim=8)
    opt_b = _trainable_optimizer(w_b)
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_lambda=lambda s: 1.0)
    with pytest.raises(RuntimeError, match="peft_method"):
        load_full_state(state_dir, w_b, opt_b, sched_b, cfg)


def test_rng_state_restored_after_resume(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    w_a = make_stub_wrapper(dim=8)
    apply_lora(w_a, cfg.peft)
    opt_a = _trainable_optimizer(w_a)
    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lr_lambda=lambda s: 1.0)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    _ = random.random()
    _ = np.random.rand(3)
    _ = torch.rand(3)

    state_dir = tmp_path / "checkpoints" / "step_0"
    save_full_state(state_dir, w_a, opt_a, sched_a, 0, 0, 0, 1.0, cfg)
    expected_py = random.random()
    expected_np = np.random.rand(3).tolist()
    expected_torch = torch.rand(3).tolist()

    w_b = make_stub_wrapper(dim=8)
    apply_lora(w_b, cfg.peft)
    opt_b = _trainable_optimizer(w_b)
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_lambda=lambda s: 1.0)
    load_full_state(state_dir, w_b, opt_b, sched_b, cfg)
    assert random.random() == expected_py
    assert np.allclose(np.random.rand(3), expected_np)
    assert torch.allclose(torch.rand(3), torch.tensor(expected_torch))
