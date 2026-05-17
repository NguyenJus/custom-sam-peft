"""Tests for esam3.peft_adapters.lora.apply_lora and helpers."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from esam3.config.schema import PEFTConfig
from esam3.peft_adapters.lora import (
    SCOPE_TARGETS,
    apply_lora,
    load_lora,
    merge_lora,
    save_lora,
)
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper


def _trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _total(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _lora_param_names(m: nn.Module) -> list[str]:
    return [n for n, _ in m.named_parameters() if "lora_" in n]


def test_apply_lora_default_scope_freezes_base() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    # Every non-LoRA param is frozen.
    for n, p in w.model.model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad, f"LoRA param {n} should be trainable"
        else:
            assert not p.requires_grad, f"Base param {n} should be frozen"


def test_apply_lora_vision_scope_matches_only_vision() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", scope="vision"))
    lora_names = _lora_param_names(w.model.model)
    assert lora_names, "expected LoRA params under vision scope"
    assert all("vision_encoder" in n for n in lora_names), lora_names
    assert not any("mask_decoder" in n for n in lora_names), lora_names


def test_apply_lora_vision_decoder_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", scope="vision_decoder"))
    lora_names = _lora_param_names(w.model.model)
    assert any("vision_encoder" in n for n in lora_names), lora_names
    assert any("mask_decoder" in n for n in lora_names), lora_names
    # Negative-control Linears must not be adapted.
    assert not any("neg_control" in n for n in lora_names), lora_names


def test_apply_lora_all_scope_includes_negative_controls() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", scope="all"))
    lora_names = _lora_param_names(w.model.model)
    assert any("neg_control" in n for n in lora_names), lora_names


def test_target_modules_overrides_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="all",  # would normally adapt everything
            target_modules=["vision_encoder.block0.attn.qkv"],
        ),
    )
    lora_names = _lora_param_names(w.model.model)
    # Exactly one Linear adapted → two LoRA params (lora_A, lora_B).
    qkv_lora = [n for n in lora_names if "vision_encoder.block0.attn.qkv" in n]
    assert len(qkv_lora) >= 2, qkv_lora
    other = [n for n in lora_names if "vision_encoder.block0.attn.qkv" not in n]
    assert not other, f"target_modules override should ignore scope; got {other}"


def test_apply_lora_no_match_raises() -> None:
    w = make_stub_wrapper()
    with pytest.raises(ValueError) as exc:
        apply_lora(w, PEFTConfig(method="lora", target_modules=["nonexistent.module"]))
    msg = str(exc.value)
    assert "nonexistent.module" in msg
    # Error should also surface at least one real Linear path to help debugging.
    assert "vision_encoder" in msg or "neg_control" in msg, msg


def test_apply_lora_idempotent_guard() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    with pytest.raises(RuntimeError, match="already applied"):
        apply_lora(w, PEFTConfig(method="lora"))


def test_apply_lora_trainable_ratio_under_default_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    ratio = _trainable(w.model.model) / _total(w.model.model)
    assert ratio < 0.20, f"trainable ratio {ratio:.2%} unexpectedly high on tiny stub"


def test_apply_lora_preserves_forward_signature() -> None:
    w = make_stub_wrapper()
    sig_before = inspect.signature(w.forward)
    apply_lora(w, PEFTConfig(method="lora"))
    sig_after = inspect.signature(w.forward)
    assert sig_before == sig_after
    assert list(sig_after.parameters) == ["images", "prompts"]


def test_apply_lora_sets_peft_model_handle() -> None:
    w = make_stub_wrapper()
    assert w.peft_model is None
    apply_lora(w, PEFTConfig(method="lora"))
    assert w.peft_model is not None
    # The handle is the same object that replaced wrapper.model.model.
    assert w.peft_model is w.model.model


def test_scope_targets_keys_match_lora_scope_literal() -> None:
    # Cheap guard: SCOPE_TARGETS must cover every literal value of LoraScope.
    assert set(SCOPE_TARGETS) == {"vision", "vision_decoder", "all"}


def test_save_load_lora_roundtrip(tmp_path: Path) -> None:
    w1 = make_stub_wrapper()
    apply_lora(w1, PEFTConfig(method="lora"))
    # Capture trained-side state-dict for the LoRA params.
    sd1 = {n: p.detach().clone() for n, p in w1.model.model.named_parameters() if "lora_" in n}
    assert sd1, "expected LoRA params on the saved wrapper"

    save_lora(w1, tmp_path)

    # adapter_config.json + adapter weights file should now exist.
    assert (tmp_path / "adapter_config.json").exists()
    weight_files = list(tmp_path.glob("adapter_model*"))
    assert weight_files, f"no adapter weight file in {tmp_path}; got {list(tmp_path.iterdir())}"

    # Fresh wrapper, load adapter, compare params bit-for-bit.
    w2 = make_stub_wrapper()
    load_lora(w2, tmp_path)
    sd2 = {n: p for n, p in w2.model.model.named_parameters() if "lora_" in n}
    assert set(sd1) == set(sd2), f"param-name mismatch: {set(sd1) ^ set(sd2)}"
    for name, t1 in sd1.items():
        assert torch.allclose(t1, sd2[name], atol=0.0), f"mismatch on {name}"


def test_load_lora_idempotent_guard(tmp_path: Path) -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    save_lora(w, tmp_path)
    with pytest.raises(RuntimeError, match="already has a PeftModel"):
        load_lora(w, tmp_path)


def test_save_lora_without_apply_raises(tmp_path: Path) -> None:
    w = make_stub_wrapper()
    with pytest.raises(RuntimeError, match="no PeftModel"):
        save_lora(w, tmp_path)


def test_merge_lora_unwraps_and_clears_handle() -> None:
    w = make_stub_wrapper()
    # Snapshot one pre-LoRA base weight so we can verify deltas folded in.
    pre = w.model.model.vision_encoder.block0.attn.qkv.weight.detach().clone()

    apply_lora(w, PEFTConfig(method="lora"))
    # Force a non-zero LoRA-B so merge changes the base.
    for n, p in w.model.model.named_parameters():
        if "lora_B" in n and "vision_encoder.block0.attn.qkv" in n:
            with torch.no_grad():
                p.add_(1.0)

    merge_lora(w)

    assert w.peft_model is None
    assert "Peft" not in type(w.model.model).__name__
    post = w.model.model.vision_encoder.block0.attn.qkv.weight.detach()
    assert not torch.allclose(pre, post), "merge_lora should have folded LoRA deltas into base"


def test_merge_lora_without_apply_raises() -> None:
    w = make_stub_wrapper()
    with pytest.raises(RuntimeError, match="no PeftModel"):
        merge_lora(w)


def test_apply_lora_registered_under_peft_lora() -> None:
    from esam3._registry import lookup

    fn = lookup("peft", "lora")
    assert fn is apply_lora


def test_resolve_targets_supports_custom_linear_types() -> None:
    """The new linear_types parameter lets qlora.py match Linear4bit modules."""
    from esam3.peft_adapters.lora import _resolve_targets

    class FakeLinear4bit(nn.Module):
        """Stand-in for bnb.nn.Linear4bit; not an nn.Linear subclass."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = nn.Module()
            self.vision_encoder.block0 = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn.qkv = FakeLinear4bit(8, 24)  # type: ignore[assignment]
            self.vision_encoder.block0.attn.proj = FakeLinear4bit(8, 8)  # type: ignore[assignment]

    base = Base()
    cfg = PEFTConfig(method="qlora", scope="vision")

    # Default linear_types=(nn.Linear,) finds nothing.
    with pytest.raises(ValueError, match=r"no nn\.Linear modules matched"):
        _resolve_targets(base, cfg)

    # Custom linear_types=(FakeLinear4bit,) finds the two attention modules.
    matched = _resolve_targets(base, cfg, linear_types=(FakeLinear4bit,))
    assert sorted(matched) == [
        "vision_encoder.block0.attn.proj",
        "vision_encoder.block0.attn.qkv",
    ]


def test_resolve_targets_default_still_filters_to_nn_linear() -> None:
    """Backward-compat guard: default behavior unchanged after adding linear_types."""
    from esam3.peft_adapters.lora import _resolve_targets

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = nn.Module()
            self.vision_encoder.block0 = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn = nn.Module()  # type: ignore[assignment]
            self.vision_encoder.block0.attn.qkv = nn.Linear(8, 24)  # type: ignore[assignment]
            self.vision_encoder.block0.attn.proj = nn.Linear(8, 8)  # type: ignore[assignment]

    matched = _resolve_targets(Base(), PEFTConfig(method="lora", scope="vision"))
    assert sorted(matched) == [
        "vision_encoder.block0.attn.proj",
        "vision_encoder.block0.attn.qkv",
    ]
