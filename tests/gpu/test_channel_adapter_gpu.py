"""GPU-gated tests for the N-channel adapter: G1 forward, G2 checkpoint round-trip,
G3 export-bundle reload.

All three require:
  - A CUDA device with compute capability >= 7.5  (requires_compatible_gpu)
  - The real SAM 3.1 checkpoint at models/sam3.1/sam3.1_multiplex.pt  (requires_checkpoint)

Run explicitly:
    pytest -m gpu tests/gpu/test_channel_adapter_gpu.py -v
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def test_G1_real_forward_nchannel(tmp_path):
    """A freeform N-channel batch flows end-to-end; adapter feeds 3 ch to forward_image."""
    from custom_sam_peft.config.schema import ModelConfig
    from custom_sam_peft.data.base import TextPrompts
    from custom_sam_peft.models.sam3 import load_sam31

    n = 5
    wrapper = load_sam31(ModelConfig(), channels=n, channel_semantics="freeform").cuda()
    images = torch.randn(1, n, 1008, 1008, device="cuda", dtype=torch.bfloat16)
    prompts = [TextPrompts(classes=["thing"])]
    with torch.no_grad():
        out = wrapper(images, prompts, box_hints=None)
    assert "pred_masks" in out


def test_G2_checkpoint_roundtrip_adapter_weights(tmp_path):
    """save -> load restores channel-adapter weights bit-for-bit (real state_dict)."""
    from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
    from custom_sam_peft.models.sam3 import load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora
    from custom_sam_peft.train.checkpoint import _load_channel_adapter, _save_channel_adapter

    w = load_sam31(ModelConfig(), channels=4, channel_semantics="rgba").cuda()
    apply_lora(w, PEFTConfig(method="lora", r=4))
    with torch.no_grad():
        w.model.channel_adapter.weight.add_(torch.randn_like(w.model.channel_adapter.weight))
    before = w.model.channel_adapter.weight.detach().clone()
    _save_channel_adapter(w, tmp_path)
    with torch.no_grad():
        w.model.channel_adapter.weight.zero_()
    _load_channel_adapter(w, tmp_path)
    assert torch.equal(w.model.channel_adapter.weight.cpu(), before.cpu())


def test_G3_export_bundle_reload_adapter(tmp_path):
    """save_adapter then reload via load_sam31 + load_adapter restores adapter weights."""
    from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
    from custom_sam_peft.models.sam3 import load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora
    from custom_sam_peft.train.checkpoint import load_adapter, save_adapter

    w = load_sam31(ModelConfig(), channels=4, channel_semantics="rgba").cuda()
    apply_lora(w, PEFTConfig(method="lora", r=4))
    with torch.no_grad():
        w.model.channel_adapter.weight.normal_()
    before = w.model.channel_adapter.weight.detach().cpu().clone()
    save_adapter(w, tmp_path / "exp")
    w2 = load_sam31(ModelConfig(), channels=4, channel_semantics="rgba").cuda()
    load_adapter(w2, tmp_path / "exp")
    assert torch.equal(w2.model.channel_adapter.weight.detach().cpu(), before)
