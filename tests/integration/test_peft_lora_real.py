"""End-to-end LoRA test against the real SAM 3.1 checkpoint.

Skipped automatically unless the .pt checkpoint is present AND a CUDA GPU
with compute capability >= 7.5 is available (SAM 3.1's PositionEmbeddingSine
hardcodes device="cuda", so a compatible GPU is required even for inspection).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.peft_adapters.lora import (
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
#   Mechanism SELECTED: §7.3(a) lora.MultiheadAttention (NOT target_parameters).
#   The target_parameters route (peft 0.19.1 lora.ParamWrapper) hard-raises
#   ValueError: lora.ParamWrapper does not work with lora_dropout != 0,
#   which poisons LoraConfig construction when dropout=0.05 (our default).
#   The concept scope instead names the ca_text/self_attn nn.MultiheadAttention
#   modules in target_modules; peft dispatches them to lora.MultiheadAttention
#   (layer.py:2492), adapting BOTH in_proj_weight AND out_proj, with dropout.
#
#   Plain-LoRA mechanism: GO — confirmed on the real SAM 3.1 decoder via GPU
#     runner; attach + forward finite-grad + merge_and_unload all passed.
#   QLoRA coexistence: GO — confirmed on GPU runner; the MHA stays unquantized
#     (_mha_exclusion_types), so the bf16 in_proj LoRA and the Linear4bit module
#     LoRA coexist in one PeftModel with dropout=0.05 and merge_and_unload is
#     clean (only a benign NF4-rounding UserWarning).
#   => Production tests below drive apply_lora / merge_lora via PEFTConfig
#      (scope="vision_decoder_concept"), no inline get_peft_model calls.


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


def test_inproj_lora_attaches_merges_vision_decoder_concept() -> None:
    """#230 §10.4: vision_decoder_concept scope adapts ca_text/self_attn MHA
    in_proj_weight AND out_proj via peft's lora.MultiheadAttention path.

    Mechanism: §7.3(a) SELECTED — the concept scope names the ca_text/self_attn
    nn.MultiheadAttention modules in target_modules (via SCOPE_MHA_MODULES);
    peft dispatches them to lora.MultiheadAttention (layer.py:2492), adapting
    BOTH in_proj_weight (params ...ca_text.lora_A / ...self_attn.lora_A) AND
    out_proj (...ca_text.base_layer.out_proj.lora_A), with lora_dropout support.
    The LoraConfig carries dropout=0.05 (our default; ParamWrapper would crash).

    Asserts:
      1. LoRA params exist for ca_text in_proj (lora_A key contains 'ca_text'
         but not 'out_proj') and self_attn in_proj (same structure).
      2. LoRA params exist for ca_text out_proj via MHA wrapper
         ('ca_text.base_layer.out_proj.lora_A').
      3. The LoraConfig on the PeftModel carries lora_dropout=0.05.
      4. A forward pass runs without error.
      5. Trainable ratio stays under the 5% budget (§8.3 empirical confirmation).
      6. merge_lora folds both axes without error.
    """
    from peft import LoraConfig

    w = load_sam31(ModelConfig())
    cfg = PEFTConfig(method="lora", scope="vision_decoder_concept")
    apply_lora(w, cfg)

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]

    # 1. in_proj LoRA present for ca_text and self_attn.
    ca_text_inproj = [n for n in lora_names if "ca_text" in n and "out_proj" not in n]
    assert any("lora_A" in n for n in ca_text_inproj), (
        f"no ca_text in_proj lora_A: {lora_names[:12]}"
    )
    self_attn_inproj = [n for n in lora_names if "self_attn" in n and "out_proj" not in n]
    assert any("lora_A" in n for n in self_attn_inproj), (
        f"no self_attn in_proj lora_A: {lora_names[:12]}"
    )

    # 2. out_proj LoRA present for ca_text (via MHA wrapper base_layer).
    assert any("ca_text.base_layer.out_proj.lora_A" in n for n in lora_names), (
        f"no ca_text.base_layer.out_proj.lora_A: {[n for n in lora_names if 'ca_text' in n]}"
    )

    # 3. LoraConfig carries dropout=0.05.
    peft_cfg: LoraConfig = w.model.model.peft_config["default"]  # type: ignore[index]
    assert peft_cfg.lora_dropout == 0.05, f"expected lora_dropout=0.05, got {peft_cfg.lora_dropout}"

    # 4. Forward runs.
    from custom_sam_peft.data.base import TextPrompts
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    w.eval()
    images = torch.zeros(1, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, device="cuda", dtype=torch.float32)
    prompts = [TextPrompts(classes=["object"])]
    with torch.no_grad():
        out = w(images, prompts, support=None)
    assert out is not None

    # 5. Trainable ratio under 5% budget (§8.3 empirical confirmation).
    trainable = sum(p.numel() for p in w.model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.model.parameters())
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.2%} exceeds 5% budget"

    # 6. Merge folds both axes without error.
    merge_lora(w)
    assert w.peft_model is None
    assert "Peft" not in type(w.model.model).__name__
