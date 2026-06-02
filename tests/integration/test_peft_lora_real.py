"""End-to-end LoRA test against the real SAM 3.1 checkpoint.

Skipped automatically unless the .pt checkpoint is present AND a CUDA GPU
with compute capability >= 7.5 is available (SAM 3.1's PositionEmbeddingSine
hardcodes device="cuda", so a compatible GPU is required even for inspection).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.peft_adapters.lora import (
    _resolve_targets,
    apply_lora,
    load_lora,
    merge_lora,
    save_lora,
)

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_t4,
]

# #230 Phase 1 spike — go/no-go (recorded 2026-06-02):
#   Mechanism CHOSEN: target_parameters (peft 0.19.1 LoraConfig.target_parameters).
#   Plain-LoRA mechanism: GO — confirmed by a CPU toy-MHA probe (attach +
#     forward finite-grad on in_proj lora_A + merge_and_unload to non-Peft).
#   Real-model empirical run (this test): DEFERRED — no SAM 3.1 checkpoint on the
#     dev box; this requires_checkpoint test skips locally. Run on a
#     checkpoint-equipped GPU runner to confirm attach+merge on the real decoder.
#   QLoRA coexistence: mechanism sound (in_proj stays bf16/unquantized via the
#     MHA exclusion, so it is plain bf16 LoRA even in QLoRA mode); empirical
#     confirmation DEFERRED to the GPU+bnb runner.
#   => Phase 2 proceeds with the target_parameters mechanism (design default).

# --- #230 in_proj feasibility spike (§7). Phase 1 is production-code-free: it
# builds the PeftModel INLINE via peft.get_peft_model with target_parameters,
# because apply_lora does not wire target_parameters until Phase 2 (Task 2.4).
_INPROJ_PARAM_PATTERNS = [
    r"transformer\.decoder\.layers\.\d+\.ca_text\.in_proj_weight$",
    r"transformer\.decoder\.layers\.\d+\.self_attn\.in_proj_weight$",
]


def test_apply_lora_on_real_sam31_under_trainable_budget() -> None:
    w = load_sam31(ModelConfig())
    apply_lora(w, PEFTConfig(method="lora"))

    trainable = sum(p.numel() for p in w.model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.model.parameters())
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.2%} exceeds 5% budget"

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert any("vision_backbone" in n for n in lora_names), "no vision-trunk LoRA targets"
    assert any("transformer.decoder" in n for n in lora_names), (
        "no transformer-decoder LoRA targets"
    )


def test_save_load_roundtrip_on_real_sam31(tmp_path: Path) -> None:
    w1 = load_sam31(ModelConfig())
    apply_lora(w1, PEFTConfig(method="lora"))
    sd1 = {n: p.detach().clone() for n, p in w1.model.model.named_parameters() if "lora_" in n}
    save_lora(w1, tmp_path)

    w2 = load_sam31(ModelConfig())
    load_lora(w2, tmp_path)
    sd2 = {n: p for n, p in w2.model.model.named_parameters() if "lora_" in n}
    assert set(sd1) == set(sd2)
    for name, t1 in sd1.items():
        assert torch.allclose(t1, sd2[name], atol=0.0), f"mismatch on {name}"


def test_merge_lora_on_real_sam31() -> None:
    w = load_sam31(ModelConfig())
    apply_lora(w, PEFTConfig(method="lora"))
    merge_lora(w)
    assert w.peft_model is None
    assert "Peft" not in type(w.model.model).__name__


def test_apply_lora_vision_scope_targets_only_vision_backbone() -> None:
    """T5 per spec §6.1: scope='vision' attaches LoRA only to vision_backbone.

    The test asserts the production SCOPE_TARGETS['vision'] regex matches the
    real SAM 3.1 module names (a regression like Meta renaming vision_backbone
    to image_encoder would slip past C2 in tests/unit/test_peft_scope_coverage.py
    because that test uses a stub). Forward-free; cost is dominated by
    load_sam31 which is already paid by the other tests in this file.
    """
    w = load_sam31(ModelConfig())
    apply_lora(w, PEFTConfig(method="lora", scope="vision"))

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert lora_names, "no lora_ params after apply_lora(scope='vision')"
    assert any("vision_backbone" in n for n in lora_names), (
        f"no vision-trunk LoRA targets at scope='vision': {lora_names[:5]}"
    )
    assert all("transformer.decoder" not in n for n in lora_names), (
        f"transformer.decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'transformer.decoder' in n][:5]}"
    )
    assert all("mask_decoder" not in n for n in lora_names), (
        f"mask_decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'mask_decoder' in n][:5]}"
    )


def test_spike_inproj_lora_attaches_merges() -> None:
    """#230 §7.1: target_parameters LoRA on ca_text/self_attn in_proj_weight
    attaches and merges on the real SAM 3.1 decoder (plain LoRA).

    Route: INLINE get_peft_model (Phase 1 is production-code-free). The
    forward-grad assertion (§7.1 item 2) is deferred to Phase 2 Task 2.10's
    real-model forward; here we prove structural attach (lora params present +
    requires_grad) and merge. The mechanism (target_parameters on MHA in_proj)
    was confirmed GO by a CPU toy-MHA probe (attach + forward-grad + merge).
    """
    w = load_sam31(ModelConfig())
    base = cast(nn.Module, w.model.model)

    compiled = [re.compile(p) for p in _INPROJ_PARAM_PATTERNS]
    param_names = [n for n, _ in base.named_parameters() if any(c.search(n) for c in compiled)]
    assert param_names, "no ca_text/self_attn in_proj_weight parameters found on real decoder"

    module_names = _resolve_targets(base, PEFTConfig(method="lora", scope="vision_decoder"))

    for p in base.parameters():
        p.requires_grad = False

    pm = get_peft_model(
        cast(Any, base),
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=module_names,
            target_parameters=param_names,
            bias="none",
            task_type=None,
        ),
    )

    # 1) Attach: LoRA params exist for both in_proj parameters and are trainable.
    lora_named = [(n, p) for n, p in pm.named_parameters() if "lora_" in n]
    assert any("ca_text" in n for n, _ in lora_named), (
        f"no ca_text in_proj LoRA: {[n for n, _ in lora_named][:8]}"
    )
    assert any("self_attn" in n for n, _ in lora_named), (
        f"no self_attn in_proj LoRA: {[n for n, _ in lora_named][:8]}"
    )
    assert all(p.requires_grad for n, p in lora_named if "lora_A" in n)

    # Record observed trainable ratio for the Phase 2 §8.3 contract.
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pm.parameters())
    assert trainable / total < 0.05, f"trainable ratio {trainable / total:.2%} exceeds 5% budget"

    # 3) Merge: folds module + parameter adapters without raising; result is non-Peft.
    merged = pm.merge_and_unload()
    assert "Peft" not in type(merged).__name__
