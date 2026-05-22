"""Unit coverage for PEFT scope → trainable-set wiring on a stub.

C2 per spec §6.2. The stub renames subtrees (vision_trunk vs the real
backbone.vision_backbone.trunk) so the production SCOPE_TARGETS regex
does not match; the test drives via FIXTURE_SCOPE_PATTERNS which mirrors
the same shape against the renamed paths. T5/T6 in tests/integration/
test_peft_{lora,qlora}_real.py cover real-module-name matching on GPU;
this file covers the scope→trainable-set logic on CPU.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.peft_adapters.lora import apply_lora
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def _lora_param_names(wrapper: object) -> list[str]:
    return [n for n, _ in wrapper.model.model.named_parameters() if "lora_" in n]


def test_scope_vision_targets_only_vision_subtree() -> None:
    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="vision",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision"],
        ),
    )
    names = _lora_param_names(w)
    assert names, "no lora_ params at scope='vision'"
    assert any("vision_trunk.blocks" in n for n in names), f"no vision-trunk LoRA: {names[:5]}"
    assert all("transformer_decoder" not in n for n in names), (
        f"transformer_decoder targets present at scope='vision': "
        f"{[n for n in names if 'transformer_decoder' in n][:5]}"
    )
    assert all("neg_control_" not in n for n in names), (
        f"neg_control_ targets present at scope='vision': "
        f"{[n for n in names if 'neg_control_' in n][:5]}"
    )


def test_scope_vision_decoder_targets_vision_and_decoder() -> None:
    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="vision_decoder",
            target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"],
        ),
    )
    names = _lora_param_names(w)
    assert any("vision_trunk.blocks" in n for n in names), (
        f"no vision-trunk LoRA at scope='vision_decoder': {names[:5]}"
    )
    assert any("transformer_decoder.layers" in n for n in names), (
        f"no transformer-decoder LoRA at scope='vision_decoder': {names[:5]}"
    )
    assert all("neg_control_" not in n for n in names), (
        f"neg_control_ targets present at scope='vision_decoder': "
        f"{[n for n in names if 'neg_control_' in n][:5]}"
    )


def test_scope_all_targets_every_linear() -> None:
    w = make_stub_wrapper(dim=8, working=False)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="all",
            target_modules=FIXTURE_SCOPE_PATTERNS["all"],
        ),
    )
    names = _lora_param_names(w)
    # Every Linear in the stub should pick up LoRA, including the negative
    # controls.
    assert any("vision_trunk.blocks" in n for n in names)
    assert any("transformer_decoder.layers" in n for n in names)
    assert any("neg_control_a" in n for n in names), (
        f"neg_control_a missing at scope='all': {names[:10]}"
    )
    assert any("neg_control_b" in n for n in names), (
        f"neg_control_b missing at scope='all': {names[:10]}"
    )


@pytest.mark.parametrize("scope", ["vision", "vision_decoder", "all"])
def test_scope_forward_backward_finite_grad(scope: str) -> None:
    """Wiring assertion: LoRA actually plugs into the gradient path.

    A scope mis-mapping (e.g. regex matches no real Linear) would still pass
    the parameter-name assertions above if the test only looked at names.
    Doing forward+backward and checking lora_A.grad is finite proves the
    parameter is in the gradient graph.
    """
    w = make_stub_wrapper(dim=8, working=True)
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope=scope,
            target_modules=FIXTURE_SCOPE_PATTERNS[scope],
        ),
    )
    # Sam3Wrapper._validate_inputs requires a prompts list matching batch size;
    # the stub's forward ignores prompts content but the outer wrapper validates.
    out = w(images=torch.randn(1, 3, 8, 8), prompts=[TextPrompts(classes=["x"])])
    loss = out["pred_masks"].sum()
    loss.backward()

    lora_a_params = [
        (n, p) for n, p in w.model.model.named_parameters() if "lora_A" in n and p.requires_grad
    ]
    assert lora_a_params, f"no lora_A params at scope={scope!r}"
    has_grad = [(n, p) for n, p in lora_a_params if p.grad is not None]
    assert has_grad, (
        f"no lora_A param received a gradient at scope={scope!r} — "
        f"the regex matched module names but the modules are not in the forward path"
    )
    for n, p in has_grad:
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {n} at scope={scope!r}"
