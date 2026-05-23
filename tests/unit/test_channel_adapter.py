import torch

from custom_sam_peft.models.sam3 import _build_channel_adapter


def test_rgb_builds_no_adapter():
    assert _build_channel_adapter(channels=3, channel_semantics="rgb") is None


def test_freeform_3ch_builds_learned_adapter_not_passthrough():
    adapter = _build_channel_adapter(channels=3, channel_semantics="freeform")
    assert adapter is not None
    assert isinstance(adapter, torch.nn.Conv2d)
    assert adapter.in_channels == 3 and adapter.out_channels == 3
    # average_broadcast init for N=3 => weight == 1/3 everywhere, bias == 0
    assert torch.allclose(adapter.weight, torch.full_like(adapter.weight, 1.0 / 3.0))
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    assert adapter.weight.requires_grad and adapter.bias.requires_grad
