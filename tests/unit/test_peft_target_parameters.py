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


def _real_paths() -> list[str]:
    return [n for n, _ in _MiniBase().named_parameters()]


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


def test_resolve_non_empty_no_match_raises_valueerror() -> None:
    base = _MiniBase()
    cfg = PEFTConfig(method="lora", target_parameters=["nonexistent.param.path$"])
    with pytest.raises(ValueError) as exc:
        _resolve_target_parameters(base, cfg)
    msg = str(exc.value)
    assert "nonexistent.param.path$" in msg  # patterns tried listed
    assert "in_proj_weight" in msg  # a real parameter name sampled
