"""Integration test: load real SAM 3.1 checkpoint and run a forward pass.

Skipped automatically unless the .pt checkpoint is present AND a CUDA GPU
with compute capability >= 7.5 is available.
"""

from __future__ import annotations

import pytest
import torch

from esam3.config.schema import ModelConfig
from esam3.data.base import TextPrompts
from esam3.models.matching import meta_to_canonical
from esam3.models.sam3 import Sam3Wrapper, load_sam31


@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_returns_wrapper() -> None:
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    assert isinstance(wrapper, Sam3Wrapper)


@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_load_sam31_forward_to_canonical() -> None:
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
    wrapper = load_sam31(cfg)
    image = torch.zeros(1, 3, 1008, 1008, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        outputs = wrapper(image, [TextPrompts(classes=["cat"])])
    canonical = meta_to_canonical(outputs)
    assert canonical.obj_logits.dim() == 2  # (B, Q)
    assert canonical.pred_boxes.shape[-1] == 4
    assert canonical.pred_masks.shape[-1] == 288
    assert canonical.img_presence.dim() == 1  # (B,)
