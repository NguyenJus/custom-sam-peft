"""Tests for custom_sam_peft.peft_adapters.lora.apply_lora and helpers."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.peft_adapters.lora import (
    SCOPE_TARGETS,
    apply_lora,
    load_lora,
    merge_lora,
    save_lora,
)
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _total(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _lora_param_names(m: nn.Module) -> list[str]:
    return [n for n, _ in m.named_parameters() if "lora_" in n]


def test_apply_lora_default_scope_freezes_base() -> None:
    w = make_stub_wrapper()
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    # Every non-LoRA param is frozen.
    for n, p in w.model.model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad, f"LoRA param {n} should be trainable"
        else:
            assert not p.requires_grad, f"Base param {n} should be frozen"


def test_apply_lora_vision_scope_matches_only_vision() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]))
    lora_names = _lora_param_names(w.model.model)
    assert lora_names, "expected LoRA params under vision scope"
    assert all("vision_trunk" in n for n in lora_names), lora_names
    assert not any("transformer_decoder" in n for n in lora_names), lora_names


def test_apply_lora_vision_decoder_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    lora_names = _lora_param_names(w.model.model)
    assert any("vision_trunk" in n for n in lora_names), lora_names
    assert any("transformer_decoder" in n for n in lora_names), lora_names
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
            target_modules=["vision_trunk.blocks.0.attn.qkv"],
        ),
    )
    lora_names = _lora_param_names(w.model.model)
    # Exactly one Linear adapted → two LoRA params (lora_A, lora_B).
    qkv_lora = [n for n in lora_names if "vision_trunk.blocks.0.attn.qkv" in n]
    assert len(qkv_lora) >= 2, qkv_lora
    other = [n for n in lora_names if "vision_trunk.blocks.0.attn.qkv" not in n]
    assert not other, f"target_modules override should ignore scope; got {other}"


def test_apply_lora_no_match_raises() -> None:
    w = make_stub_wrapper()
    with pytest.raises(ValueError) as exc:
        apply_lora(w, PEFTConfig(method="lora", target_modules=["nonexistent.module"]))
    msg = str(exc.value)
    assert "nonexistent.module" in msg
    # Error should also surface at least one real Linear path to help debugging.
    assert "vision_trunk" in msg or "neg_control" in msg, msg


def test_apply_lora_idempotent_guard() -> None:
    w = make_stub_wrapper()
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    with pytest.raises(RuntimeError, match="already applied"):
        apply_lora(
            w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
        )


def test_apply_lora_trainable_ratio_under_default_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    ratio = _trainable(w.model.model) / _total(w.model.model)
    assert ratio < 0.20, f"trainable ratio {ratio:.2%} unexpectedly high on tiny stub"


def test_apply_lora_preserves_forward_signature() -> None:
    w = make_stub_wrapper()
    sig_before = inspect.signature(w.forward)
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    sig_after = inspect.signature(w.forward)
    assert sig_before == sig_after
    assert list(sig_after.parameters) == ["images", "prompts", "box_hints"]


def test_apply_lora_sets_peft_model_handle() -> None:
    w = make_stub_wrapper()
    assert w.peft_model is None
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    assert w.peft_model is not None
    # The handle is the same object that replaced wrapper.model.model.
    assert w.peft_model is w.model.model


def test_scope_targets_keys_match_lora_scope_literal() -> None:
    # Cheap guard: SCOPE_TARGETS must cover every literal value of LoraScope.
    assert set(SCOPE_TARGETS) == {"vision", "vision_decoder", "all"}


def test_save_load_lora_roundtrip(tmp_path: Path) -> None:
    w1 = make_stub_wrapper()
    apply_lora(
        w1, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
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


def test_load_lora_keeps_lora_params_trainable(tmp_path: Path) -> None:
    w_a = make_stub_wrapper(dim=8)
    apply_lora(w_a, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]))
    save_lora(w_a, tmp_path / "adapter")

    w_b = make_stub_wrapper(dim=8)
    load_lora(w_b, tmp_path / "adapter")

    lora_params = [p for n, p in w_b.named_parameters() if "lora_" in n]
    assert lora_params, "expected at least one LoRA-named parameter after load_lora"
    assert all(p.requires_grad for p in lora_params), (
        "load_lora must leave LoRA params trainable for resume-then-train flows"
    )


def test_load_lora_on_already_wrapped_reloads_weights(tmp_path: Path) -> None:
    # When a wrapper already has a PeftModel (e.g. apply_lora was called before
    # load_full_state), load_lora reloads adapter weights rather than raising.
    w = make_stub_wrapper()
    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    save_lora(w, tmp_path)
    # Should not raise; should return the same wrapper.
    result = load_lora(w, tmp_path)
    assert result is w
    lora_params = [p for n, p in w.named_parameters() if "lora_" in n]
    assert lora_params, "expected LoRA params still present after reload"
    assert all(p.requires_grad for p in lora_params)


def test_save_lora_without_apply_raises(tmp_path: Path) -> None:
    w = make_stub_wrapper()
    with pytest.raises(RuntimeError, match="no PeftModel"):
        save_lora(w, tmp_path)


def test_merge_lora_unwraps_and_clears_handle() -> None:
    w = make_stub_wrapper()
    # Snapshot one pre-LoRA base weight so we can verify deltas folded in.
    pre = w.model.model.vision_trunk.blocks[0].attn.qkv.weight.detach().clone()

    apply_lora(
        w, PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"])
    )
    # Force a non-zero LoRA-B so merge changes the base.
    for n, p in w.model.model.named_parameters():
        if "lora_B" in n and "vision_trunk.blocks.0.attn.qkv" in n:
            with torch.no_grad():
                p.add_(1.0)

    merge_lora(w)

    assert w.peft_model is None
    assert "Peft" not in type(w.model.model).__name__
    post = w.model.model.vision_trunk.blocks[0].attn.qkv.weight.detach()
    assert not torch.allclose(pre, post), "merge_lora should have folded LoRA deltas into base"


def test_merge_lora_without_apply_raises() -> None:
    w = make_stub_wrapper()
    with pytest.raises(RuntimeError, match="no PeftModel"):
        merge_lora(w)


def test_apply_lora_registered_under_peft_lora() -> None:
    from custom_sam_peft._registry import lookup

    fn = lookup("peft", "lora")
    assert fn is apply_lora


def test_resolve_targets_supports_custom_linear_types() -> None:
    """The new linear_types parameter lets qlora.py match Linear4bit modules."""
    from custom_sam_peft.peft_adapters.lora import _resolve_targets

    class FakeLinear4bit(nn.Module):
        """Stand-in for bnb.nn.Linear4bit; not an nn.Linear subclass."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.vision_backbone = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk.blocks = nn.ModuleList(  # type: ignore[assignment]
                [nn.Module()]
            )
            self.backbone.vision_backbone.trunk.blocks[0].attn = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk.blocks[0].attn.qkv = FakeLinear4bit(8, 24)  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk.blocks[0].attn.proj = FakeLinear4bit(8, 8)  # type: ignore[assignment]

    base = Base()
    cfg = PEFTConfig(method="qlora", scope="vision")

    # Default linear_types=(nn.Linear,) finds nothing.
    with pytest.raises(ValueError, match=r"no Linear modules matched"):
        _resolve_targets(base, cfg)

    # Custom linear_types=(FakeLinear4bit,) finds the two attention modules,
    # and a mismatch under that override surfaces the correct type label.
    matched = _resolve_targets(base, cfg, linear_types=(FakeLinear4bit,))
    assert sorted(matched) == [
        "backbone.vision_backbone.trunk.blocks.0.attn.proj",
        "backbone.vision_backbone.trunk.blocks.0.attn.qkv",
    ]
    empty_cfg = PEFTConfig(method="qlora", scope="vision", target_modules=["does_not_match"])
    with pytest.raises(ValueError, match=r"no FakeLinear4bit modules matched"):
        _resolve_targets(base, empty_cfg, linear_types=(FakeLinear4bit,))


def test_resolve_targets_default_still_filters_to_nn_linear() -> None:
    """Backward-compat guard: default behavior unchanged after adding linear_types."""
    from custom_sam_peft.peft_adapters.lora import _resolve_targets

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.vision_backbone = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk.blocks = nn.ModuleList(  # type: ignore[assignment]
                [nn.Module()]
            )
            self.backbone.vision_backbone.trunk.blocks[0].attn = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk.blocks[0].attn.qkv = nn.Linear(8, 24)  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk.blocks[0].attn.proj = nn.Linear(8, 8)  # type: ignore[assignment]

    matched = _resolve_targets(Base(), PEFTConfig(method="lora", scope="vision"))
    assert sorted(matched) == [
        "backbone.vision_backbone.trunk.blocks.0.attn.proj",
        "backbone.vision_backbone.trunk.blocks.0.attn.qkv",
    ]


def test_scope_targets_match_real_sam3_module_naming() -> None:
    """Regression guard: the production SCOPE_TARGETS regexes match the real
    SAM 3.1 module-naming shape (sourced from sam3/model/{vitdet,necks,vl_combiner,decoder}.py).
    """
    from custom_sam_peft.peft_adapters.lora import SCOPE_TARGETS, _resolve_targets

    class _RealNamingStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.vision_backbone = nn.Module()  # type: ignore[assignment]
            self.backbone.vision_backbone.trunk = nn.Module()  # type: ignore[assignment]
            blocks: list[nn.Module] = []
            for _ in range(2):
                block = nn.Module()
                block.attn = nn.Module()  # type: ignore[assignment]
                block.attn.qkv = nn.Linear(8, 24)  # type: ignore[assignment]
                block.attn.proj = nn.Linear(8, 8)  # type: ignore[assignment]
                block.mlp = nn.Module()  # type: ignore[assignment]
                block.mlp.fc1 = nn.Linear(8, 16)  # type: ignore[assignment]
                block.mlp.fc2 = nn.Linear(16, 8)  # type: ignore[assignment]
                blocks.append(block)
            self.backbone.vision_backbone.trunk.blocks = nn.ModuleList(blocks)  # type: ignore[assignment]
            self.transformer = nn.Module()
            self.transformer.decoder = nn.Module()  # type: ignore[assignment]
            decoder_layers: list[nn.Module] = []
            for _ in range(2):
                layer = nn.Module()
                for kind in ("self_attn", "cross_attn", "ca_text"):
                    sub = nn.Module()
                    sub.out_proj = nn.Linear(8, 8)  # type: ignore[assignment]
                    setattr(layer, kind, sub)
                layer.linear1 = nn.Linear(8, 16)  # type: ignore[assignment]  # decoder FFN
                layer.linear2 = nn.Linear(16, 8)  # type: ignore[assignment]  # decoder FFN
                decoder_layers.append(layer)
            self.transformer.decoder.layers = nn.ModuleList(decoder_layers)  # type: ignore[assignment]

    stub = _RealNamingStub()

    vision = _resolve_targets(stub, PEFTConfig(method="lora", scope="vision"))
    assert sorted(vision) == [
        "backbone.vision_backbone.trunk.blocks.0.attn.proj",
        "backbone.vision_backbone.trunk.blocks.0.attn.qkv",
        "backbone.vision_backbone.trunk.blocks.1.attn.proj",
        "backbone.vision_backbone.trunk.blocks.1.attn.qkv",
    ]

    vision_decoder = _resolve_targets(stub, PEFTConfig(method="lora", scope="vision_decoder"))
    assert "transformer.decoder.layers.0.self_attn.out_proj" in vision_decoder
    assert "transformer.decoder.layers.0.cross_attn.out_proj" in vision_decoder
    assert "transformer.decoder.layers.0.ca_text.out_proj" in vision_decoder
    assert "transformer.decoder.layers.1.self_attn.out_proj" in vision_decoder
    # vision scope subset is included.
    assert set(vision).issubset(set(vision_decoder))
    # Decoder FFN linears ARE now adapted (sam3.model.decoder.TransformerDecoderLayer:64,67).
    assert "transformer.decoder.layers.0.linear1" in vision_decoder
    assert "transformer.decoder.layers.0.linear2" in vision_decoder
    assert "transformer.decoder.layers.1.linear1" in vision_decoder
    assert "transformer.decoder.layers.1.linear2" in vision_decoder
    # Vision-trunk MLP is intentionally NOT adapted under vision_decoder.
    assert all(".mlp." not in n for n in vision_decoder)

    # SCOPE_TARGETS still exposes only the three documented scopes.
    assert set(SCOPE_TARGETS) == {"vision", "vision_decoder", "all"}


def test_vision_decoder_scope_matches_decoder_ffn_linears() -> None:
    """Focused test: linear[12] pattern matches both LoRA and QLoRA paths.

    Under LoRA, decoder FFN modules are nn.Linear (default linear_types).
    Under QLoRA, they become Linear4bit — simulated here with a stand-in type
    by calling _resolve_targets with linear_types=(nn.Linear,) directly, since
    the regex is type-agnostic and type filtering is orthogonal.
    """
    from custom_sam_peft.peft_adapters.lora import _resolve_targets

    class FakeDecoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(64, 4)
            self.linear1 = nn.Linear(64, 256)
            self.linear2 = nn.Linear(256, 64)

    class FakeRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.decoder = nn.Module()  # type: ignore[assignment]
            self.transformer.decoder.layers = nn.ModuleList(  # type: ignore[assignment]
                [FakeDecoderLayer()]
            )

    root = FakeRoot()
    cfg = PEFTConfig(method="lora", scope="vision_decoder")

    # Default path (LoRA): linear_types=(nn.Linear,) — linear1/linear2 are matched.
    matched_lora = _resolve_targets(root, cfg, linear_types=(nn.Linear,))
    assert "transformer.decoder.layers.0.linear1" in matched_lora, matched_lora
    assert "transformer.decoder.layers.0.linear2" in matched_lora, matched_lora

    # QLoRA simulation: use a custom type to mirror the Linear4bit scenario.
    # linear1/linear2 are swapped to FakeLinear4bit; out_proj stays as nn.Linear
    # (matching MHA-exclusion behavior in qlora.py).
    class FakeLinear4bit(nn.Module):
        """Stand-in for bnb.nn.Linear4bit; not an nn.Linear subclass."""

        def __init__(self, in_f: int, out_f: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_f, in_f))

    class FakeDecoderLayerQuantized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(64, 4)
            # FFN linears are quantized (Linear4bit); out_proj inside MHA stays Linear.
            self.linear1 = FakeLinear4bit(64, 256)
            self.linear2 = FakeLinear4bit(256, 64)

    class FakeRootQuantized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.decoder = nn.Module()  # type: ignore[assignment]
            self.transformer.decoder.layers = nn.ModuleList(  # type: ignore[assignment]
                [FakeDecoderLayerQuantized()]
            )

    root_q = FakeRootQuantized()
    matched_qlora = _resolve_targets(root_q, cfg, linear_types=(FakeLinear4bit,))
    assert "transformer.decoder.layers.0.linear1" in matched_qlora, matched_qlora
    assert "transformer.decoder.layers.0.linear2" in matched_qlora, matched_qlora
