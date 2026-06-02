"""Unit coverage for the #230 target_parameters resolution axis."""

from __future__ import annotations

from custom_sam_peft.peft_adapters.lora import SCOPE_TARGET_PARAMETERS, SCOPE_TARGETS


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
