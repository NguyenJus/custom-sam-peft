"""End-to-end LoRA test against the real SAM 3.1 checkpoint.

Skipped automatically unless the .pt checkpoint is present AND a CUDA GPU
with compute capability >= 7.5 is available (SAM 3.1's PositionEmbeddingSine
hardcodes device="cuda", so a compatible GPU is required even for inspection).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from esam3.config.schema import ModelConfig, PEFTConfig
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import apply_lora, load_lora, merge_lora, save_lora

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
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
