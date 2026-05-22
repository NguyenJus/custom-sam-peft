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
from custom_sam_peft.peft_adapters.lora import apply_lora, load_lora, merge_lora, save_lora

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
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
