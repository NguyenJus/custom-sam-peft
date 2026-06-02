"""Unit coverage for the #230 target_parameters resolution axis."""

from __future__ import annotations

import pytest
from torch import nn

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.peft_adapters.lora import (
    SCOPE_TARGET_PARAMETERS,
    SCOPE_TARGETS,
    _resolve_target_parameters,
)

# ---------------------------------------------------------------------------
# Task 2.2: SCOPE_TARGET_PARAMETERS dict + vision_decoder_concept scope entry
# ---------------------------------------------------------------------------


def test_scope_target_parameters_has_concept_inproj_patterns() -> None:
    pats = SCOPE_TARGET_PARAMETERS["vision_decoder_concept"]
    assert any("ca_text" in p and "in_proj_weight" in p for p in pats)
    assert any("self_attn" in p and "in_proj_weight" in p for p in pats)
    assert not any("cross_attn" in p for p in pats), "cross_attn is RoPEAttention, not MHA"


def test_concept_scope_modules_equal_vision_decoder() -> None:
    assert SCOPE_TARGETS["vision_decoder_concept"] == SCOPE_TARGETS["vision_decoder"]


def test_legacy_scopes_have_no_parameter_targets() -> None:
    for scope in ("vision", "vision_decoder", "all"):
        assert SCOPE_TARGET_PARAMETERS.get(scope, []) == []


# ---------------------------------------------------------------------------
# Task 2.3: _resolve_target_parameters
# ---------------------------------------------------------------------------


class _MiniBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        layer = nn.Module()
        layer.ca_text = nn.MultiheadAttention(8, 2)
        layer.self_attn = nn.MultiheadAttention(8, 2)
        self.transformer.decoder.layers = nn.ModuleList([layer])


def test_resolve_empty_for_legacy_scope_returns_empty_no_error() -> None:
    base = _MiniBase()
    got = _resolve_target_parameters(base, PEFTConfig(method="lora", scope="vision_decoder"))
    assert got == []


def test_resolve_override_verbatim_precedence() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(
        method="lora",
        scope="vision_decoder",  # legacy scope (no scope params) ...
        target_parameters=[r"\.ca_text\.in_proj_weight$"],  # ... but override set
    )
    got = _resolve_target_parameters(base, cfg)
    assert got == ["transformer.decoder.layers.0.ca_text.in_proj_weight"]


def test_resolve_empty_list_override_is_valid() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(method="lora", scope="vision_decoder_concept", target_parameters=[])
    assert _resolve_target_parameters(base, cfg) == []


def test_resolve_concept_scope_matches_inproj_from_mini_base() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    got = _resolve_target_parameters(base, cfg)
    assert "transformer.decoder.layers.0.ca_text.in_proj_weight" in got
    assert "transformer.decoder.layers.0.self_attn.in_proj_weight" in got
    assert len(got) == 2


def test_resolve_non_empty_no_match_raises_valueerror() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(method="lora", target_parameters=["nonexistent.param.path$"])
    with pytest.raises(ValueError) as exc:
        _resolve_target_parameters(base, cfg)
    msg = str(exc.value)
    assert "nonexistent.param.path$" in msg  # patterns tried listed
    assert "in_proj_weight" in msg  # a real parameter name sampled


# ---------------------------------------------------------------------------
# Task 2.4: Wire target_parameters into apply_lora
# ---------------------------------------------------------------------------


def test_apply_lora_legacy_scope_passes_target_parameters_none() -> None:
    """Reproducibility: legacy scopes must build LoraConfig with target_parameters=None."""
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    captured: dict[str, object] = {}
    real_cfg = lora_mod.LoraConfig

    def _spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_cfg(*args, **kwargs)

    w = make_stub_wrapper(dim=8, working=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lora_mod, "LoraConfig", _spy)
        lora_mod.apply_lora(
            w,
            PEFTConfig(
                method="lora",
                scope="vision_decoder",
                target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"],
            ),
        )
    assert "target_parameters" in captured, (
        "apply_lora did not pass target_parameters to LoraConfig"
    )
    assert captured["target_parameters"] is None


# ---------------------------------------------------------------------------
# Task 2.5: Wire target_parameters into the QLoRA apply path
# ---------------------------------------------------------------------------


def test_qlora_and_lora_resolve_same_parameter_set() -> None:
    """§10.3: the parameter axis is mode-independent — same names for LoRA and QLoRA."""
    from custom_sam_peft.peft_adapters.lora import _resolve_target_parameters

    base = _MiniBase()
    lora_cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    qlora_cfg = PEFTConfig(method="qlora", scope="vision_decoder_concept")
    assert _resolve_target_parameters(base, lora_cfg) == _resolve_target_parameters(base, qlora_cfg)
    # And both resolve the two in_proj params on the mini base.
    got = _resolve_target_parameters(base, lora_cfg)
    assert any("ca_text.in_proj_weight" in n for n in got)
    assert any("self_attn.in_proj_weight" in n for n in got)


# ---------------------------------------------------------------------------
# Task 2.6: Flip default scope to vision_decoder_concept
# ---------------------------------------------------------------------------


def test_peftconfig_default_scope_is_concept() -> None:
    assert PEFTConfig(method="lora").scope == "vision_decoder_concept"


def test_lorascope_literal_includes_concept() -> None:
    import typing

    from custom_sam_peft.config.schema import LoraScope

    assert set(typing.get_args(LoraScope)) == {
        "vision",
        "vision_decoder",
        "vision_decoder_concept",
        "all",
    }


# ---------------------------------------------------------------------------
# Task 2.7: Expose MHA in_proj in the LoRA stub fixture
# ---------------------------------------------------------------------------


def test_fixture_exposes_mha_inproj_and_concept_patterns() -> None:
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_PATTERNS,
        FIXTURE_SCOPE_TARGET_PARAMETERS,
        make_stub_wrapper,
    )

    w = make_stub_wrapper(dim=8, working=False)
    base = w.model.model
    names = [n for n, _ in base.named_parameters()]
    assert any(n.endswith("ca_text.in_proj_weight") for n in names), names[:10]
    assert any(n.endswith("self_attn.in_proj_weight") for n in names), names[:10]
    # cross_attn must NOT be MHA (negative control for the parameter axis).
    assert not any("cross_attn.in_proj_weight" in n for n in names)
    # The concept fixture mappings exist.
    assert "vision_decoder_concept" in FIXTURE_SCOPE_PATTERNS
    assert "vision_decoder_concept" in FIXTURE_SCOPE_TARGET_PARAMETERS
