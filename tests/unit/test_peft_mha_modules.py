"""Unit coverage for the #230 SCOPE_MHA_MODULES resolution axis."""

from __future__ import annotations

import pytest
from torch import nn

from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.peft_adapters.lora import SCOPE_MHA_MODULES, SCOPE_TARGETS

# ---------------------------------------------------------------------------
# Task 2.2: SCOPE_MHA_MODULES dict + vision_decoder_concept scope entry
# ---------------------------------------------------------------------------


def test_scope_mha_modules_has_concept_mha_patterns() -> None:
    pats = SCOPE_MHA_MODULES["vision_decoder_concept"]
    assert any("ca_text" in p for p in pats)
    assert any("self_attn" in p for p in pats)
    assert not any("cross_attn" in p for p in pats), "cross_attn is RoPEAttention, not MHA"
    assert not any("in_proj_weight" in p for p in pats), "MHA axis names modules, not params"


def test_concept_scope_modules_de_overlap_vision_decoder() -> None:
    """Concept SCOPE_TARGETS drops the self_attn/ca_text out_proj alternatives (peft's
    MHA wrapper adapts out_proj internally; double-targeting must be avoided)."""
    concept = SCOPE_TARGETS["vision_decoder_concept"]
    # cross_attn.out_proj is kept (RoPEAttention -> genuine nn.Linear).
    assert any("cross_attn" in p and "out_proj" in p for p in concept)
    # self_attn / ca_text out_proj are NOT generic targets under the concept scope.
    assert not any(("self_attn" in p or "ca_text" in p) and "out_proj" in p for p in concept)
    # And it is NOT module-equal to vision_decoder anymore.
    assert SCOPE_TARGETS["vision_decoder_concept"] != SCOPE_TARGETS["vision_decoder"]


def test_legacy_scopes_have_no_mha_targets() -> None:
    for scope in ("vision", "vision_decoder", "all"):
        assert SCOPE_MHA_MODULES.get(scope, []) == []


# ---------------------------------------------------------------------------
# Task 2.3: _resolve_mha_modules
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


def test_resolve_mha_concept_scope_returns_both_modules() -> None:
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules

    got = _resolve_mha_modules(
        _MiniBase(), PEFTConfig(method="lora", scope="vision_decoder_concept")
    )
    assert got == [
        "transformer.decoder.layers.0.ca_text",
        "transformer.decoder.layers.0.self_attn",
    ]


def test_resolve_mha_empty_for_legacy_scope_returns_empty_no_error() -> None:
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules

    assert (
        _resolve_mha_modules(_MiniBase(), PEFTConfig(method="lora", scope="vision_decoder")) == []
    )


def test_resolve_mha_returns_empty_when_target_modules_overridden() -> None:
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules

    cfg = PEFTConfig(
        method="lora",
        scope="vision_decoder_concept",  # has MHA patterns ...
        target_modules=[r"\.ca_text$"],  # ... but override owns the module axis
    )
    assert _resolve_mha_modules(_MiniBase(), cfg) == []


def test_resolve_mha_non_empty_no_match_raises_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules

    # A non-empty pattern list that matches zero MHA modules must raise.
    monkeypatch.setitem(
        lora_mod.SCOPE_MHA_MODULES, "vision_decoder_concept", [r"\.nonexistent_mha$"]
    )
    cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    with pytest.raises(ValueError) as exc:
        _resolve_mha_modules(_MiniBase(), cfg)
    msg = str(exc.value)
    assert "nonexistent_mha" in msg  # patterns tried listed
    assert "ca_text" in msg or "self_attn" in msg  # a real MHA module name sampled


# ---------------------------------------------------------------------------
# Task 2.4: apply_lora wiring — no target_parameters kwarg; legacy unchanged
# ---------------------------------------------------------------------------


def test_apply_lora_never_passes_target_parameters_and_legacy_unioned_empty() -> None:
    """Reproducibility: legacy scopes build LoraConfig with no MHA union and never a
    target_parameters kwarg (the reverted axis)."""
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper

    captured: dict[str, object] = {}
    real_cfg = lora_mod.LoraConfig

    def _spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_cfg(*args, **kwargs)

    w = make_stub_wrapper(dim=8, working=False)
    cfg = PEFTConfig(
        method="lora",
        scope="vision_decoder",
        target_modules=FIXTURE_SCOPE_PATTERNS["vision_decoder"],
    )
    # An explicit target_modules override owns the module axis, so the scope's MHA
    # patterns are not unioned in — _resolve_mha_modules returns []. The LoraConfig's
    # target_modules is therefore exactly the generic-resolved concrete module names
    # (NOT the raw regex patterns), with nothing appended.
    expected_modules = lora_mod._resolve_targets(w.model.model, cfg)
    assert lora_mod._resolve_mha_modules(w.model.model, cfg) == []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lora_mod, "LoraConfig", _spy)
        lora_mod.apply_lora(w, cfg)
    assert "target_parameters" not in captured  # reverted axis is gone
    assert captured["target_modules"] == expected_modules  # no MHA union appended


# ---------------------------------------------------------------------------
# Task 2.5 (QLoRA parity) + schema default re-assertions
# ---------------------------------------------------------------------------


def test_qlora_and_lora_resolve_same_mha_set() -> None:
    """§10.3: the MHA axis is mode-independent — same module names for LoRA and QLoRA."""
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules

    base = _MiniBase()
    lora_cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    qlora_cfg = PEFTConfig(method="qlora", scope="vision_decoder_concept")
    assert _resolve_mha_modules(base, lora_cfg) == _resolve_mha_modules(base, qlora_cfg)
    got = _resolve_mha_modules(base, lora_cfg)
    assert any(n.endswith("ca_text") for n in got)
    assert any(n.endswith("self_attn") for n in got)


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
# Task 2.6: Fixture exposes FIXTURE_SCOPE_MHA_MODULES + de-overlapped concept patterns
# ---------------------------------------------------------------------------


def test_fixture_exposes_mha_modules_and_de_overlapped_concept_patterns() -> None:
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_MHA_MODULES,
        FIXTURE_SCOPE_PATTERNS,
        make_stub_wrapper,
    )

    w = make_stub_wrapper(dim=8, working=False)
    base = w.model.model
    # ca_text / self_attn are real nn.MultiheadAttention (in_proj_weight exists).
    names = [n for n, _ in base.named_parameters()]
    assert any(n.endswith("ca_text.in_proj_weight") for n in names), names[:10]
    assert any(n.endswith("self_attn.in_proj_weight") for n in names), names[:10]
    # cross_attn must NOT be MHA (negative control for the MHA axis).
    assert not any("cross_attn.in_proj_weight" in n for n in names)

    # The concept fixture mappings exist and are de-overlapped.
    assert "vision_decoder_concept" in FIXTURE_SCOPE_PATTERNS
    assert "vision_decoder_concept" in FIXTURE_SCOPE_MHA_MODULES
    concept_generic = FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"]
    assert not any(
        ("self_attn" in p or "ca_text" in p) and "out_proj" in p for p in concept_generic
    )
    concept_mha = FIXTURE_SCOPE_MHA_MODULES["vision_decoder_concept"]
    assert any("ca_text" in p for p in concept_mha)
    assert any("self_attn" in p for p in concept_mha)


# ---------------------------------------------------------------------------
# Task 2.7: CPU union path + all-scope hard test
# ---------------------------------------------------------------------------


def _lora_param_names(wrapper: object) -> list[str]:
    return [n for n, _ in wrapper.model.model.named_parameters() if "lora_" in n]


def test_concept_scope_unions_modules_and_mha_on_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.2: with production scope dicts monkeypatched to fixture prefixes, apply_lora's
    real union path attaches generic-module LoRA + MHA in_proj/out_proj LoRA, and NOT on
    cross_attn (negative control)."""
    import custom_sam_peft.peft_adapters.lora as lora_mod
    from tests.fixtures.tiny_sam3_lora_stub import (
        FIXTURE_SCOPE_MHA_MODULES,
        FIXTURE_SCOPE_PATTERNS,
        make_stub_wrapper,
    )

    monkeypatch.setitem(
        lora_mod.SCOPE_MHA_MODULES,
        "vision_decoder_concept",
        FIXTURE_SCOPE_MHA_MODULES["vision_decoder_concept"],
    )
    monkeypatch.setitem(
        lora_mod.SCOPE_TARGETS,
        "vision_decoder_concept",
        FIXTURE_SCOPE_PATTERNS["vision_decoder_concept"],
    )
    w = make_stub_wrapper(dim=8, working=False)
    lora_mod.apply_lora(w, PEFTConfig(method="lora", scope="vision_decoder_concept"))

    names = _lora_param_names(w)
    assert any("vision_trunk.blocks" in n for n in names)
    assert any("ca_text" in n and "lora" in n for n in names), names[:10]
    assert any("self_attn" in n and "lora" in n for n in names), names[:10]
    assert not any("cross_attn" in n and "in_proj" in n for n in names)
    # Trainable ratio is computed and sane. On this dim-8 stub the LoRA rank r=16 exceeds
    # the layer width, so the absolute ratio is large and NOT representative of the real
    # model; the real <5% budget is enforced on the full SAM 3.1 model by the GPU test
    # (§10.4). Here we only confirm LoRA attached and did not make the whole model trainable.
    base = w.model.model
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    assert 0 < trainable < total


def test_all_scope_never_reaches_mha_on_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """§4.3 HARD: the 'all' scope's .* lives only in _resolve_targets (nn.Linear); it can
    never reach an nn.MultiheadAttention module."""
    from custom_sam_peft.peft_adapters.lora import _resolve_mha_modules
    from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper

    base = make_stub_wrapper(dim=8, working=False).model.model
    assert _resolve_mha_modules(base, PEFTConfig(method="lora", scope="all")) == []
