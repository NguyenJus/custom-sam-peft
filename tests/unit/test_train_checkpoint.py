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
    from custom_sam_peft.errors import CheckpointError

    with pytest.raises((RuntimeError, CheckpointError), match="peft_method"):
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


def test_channel_adapter_file_written_and_skipped_for_rgb(tmp_path: Path) -> None:
    """save/load channel-adapter helpers round-trip weights bit-for-bit (CPU stub).

    Also verifies the rgb (no-adapter) path is a complete no-op: no file written,
    no error on load.
    """
    import torch
    import torch.nn as nn

    from custom_sam_peft.train import checkpoint as C

    class _StubAdapterModel:
        # mirrors the real _Sam3ImageAdapter: holds .channel_adapter
        def __init__(self, adapter: nn.Conv2d | None) -> None:
            self.channel_adapter = adapter

    class _StubPeft:
        def save_pretrained(self, p: str) -> None:
            from pathlib import Path

            Path(p).mkdir(parents=True, exist_ok=True)
            (Path(p) / "adapter_model.safetensors").write_bytes(b"x")

    class _StubWrapper:
        # mirrors Sam3Wrapper: wrapper.model is the _Sam3ImageAdapter
        def __init__(self, adapter: nn.Conv2d | None) -> None:
            self.model = _StubAdapterModel(adapter)
            self.peft_model = _StubPeft()

    conv = nn.Conv2d(4, 3, 1)
    with torch.no_grad():
        conv.weight.normal_()
    w_has = _StubWrapper(conv)

    # helper-level round-trip (the core contract; real-state_dict bit-for-bit is GPU test G2)
    C._save_channel_adapter(w_has, tmp_path / "rt")  # type: ignore[arg-type]
    assert (tmp_path / "rt" / C._CHANNEL_ADAPTER_FILENAME).exists()
    fresh = nn.Conv2d(4, 3, 1)
    w_fresh = _StubWrapper(fresh)
    C._load_channel_adapter(w_fresh, tmp_path / "rt")  # type: ignore[arg-type]
    assert torch.allclose(fresh.weight, conv.weight)
    assert torch.allclose(fresh.bias, conv.bias)  # type: ignore[arg-type]

    # rgb: no adapter -> no file written, load is a no-op
    w_rgb = _StubWrapper(None)
    C._save_channel_adapter(w_rgb, tmp_path / "rgb")  # type: ignore[arg-type]
    assert not (tmp_path / "rgb" / C._CHANNEL_ADAPTER_FILENAME).exists()
    C._load_channel_adapter(w_rgb, tmp_path / "rgb")  # type: ignore[arg-type]  # no-op, no error


def test_save_adapter_calls_save_channel_adapter(tmp_path: Path) -> None:
    """save_adapter calls _save_channel_adapter after the PEFT dispatch (CPU stub).

    Uses monkeypatching to verify the call without a real PeftModel.
    """
    import torch.nn as nn

    from custom_sam_peft.train import checkpoint as C

    class _StubAdapterModel:
        def __init__(self, adapter: nn.Conv2d | None) -> None:
            self.channel_adapter = adapter

    class _StubPeft:
        def save_pretrained(self, p: str) -> None:
            from pathlib import Path

            Path(p).mkdir(parents=True, exist_ok=True)
            (Path(p) / "adapter_model.safetensors").write_bytes(b"x")

    class _StubWrapper:
        def __init__(self, adapter: nn.Conv2d | None) -> None:
            self.model = _StubAdapterModel(adapter)
            self.peft_model = _StubPeft()

    conv = nn.Conv2d(4, 3, 1)
    w = _StubWrapper(conv)

    calls: list[tuple[object, Path]] = []

    original = C._save_channel_adapter

    def _recording(wrapper: object, adapter_dir: Path) -> None:  # type: ignore[override]
        calls.append((wrapper, adapter_dir))
        original(wrapper, adapter_dir)  # type: ignore[arg-type]

    out = tmp_path / "adapter"

    # Monkeypatch _has_linear4bit to always return False so save_lora path is taken.
    orig_has_linear4bit = C._has_linear4bit
    C._has_linear4bit = lambda w: False  # type: ignore[assignment]
    C._save_channel_adapter = _recording  # type: ignore[assignment]
    try:
        C.save_adapter(w, out)  # type: ignore[arg-type]
    finally:
        C._has_linear4bit = orig_has_linear4bit  # type: ignore[assignment]
        C._save_channel_adapter = original  # type: ignore[assignment]

    assert len(calls) == 1, f"expected 1 call to _save_channel_adapter, got {len(calls)}"
    _, called_dir = calls[0]
    assert called_dir == out, f"expected dir={out}, got {called_dir}"
