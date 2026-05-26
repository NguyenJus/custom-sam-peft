"""Integration test: load real SAM 3.1 checkpoint and run a forward pass.

Skipped automatically unless the .pt checkpoint is present AND a CUDA GPU
with compute capability >= 6.0 is available.  The two cheaper tests use
gpu_local (fits a GTX 1080); the K=8 multiplex forward uses gpu_t4.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.config.schema import ModelConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.matching import meta_to_canonical
from custom_sam_peft.models.sam3 import Sam3Wrapper, load_sam31

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
]


@pytest.mark.gpu_local
def test_load_sam31_returns_wrapper() -> None:
    cfg = ModelConfig(device="cuda", dtype="bfloat16")
    wrapper = load_sam31(cfg)
    assert isinstance(wrapper, Sam3Wrapper)


@pytest.mark.gpu_local
def test_load_sam31_forward_to_canonical() -> None:
    cfg = ModelConfig(device="cuda", dtype="bfloat16")
    wrapper = load_sam31(cfg)
    # This is an inference smoke test, so use eval() + no_grad to disable
    # dropout / training-mode behavior.  (forward also works under train()
    # thanks to _patch_forward_grounding_skip_matching_on_none_target, but
    # the trainer exercises that path separately in tests/gpu/test_real_train_*.)
    wrapper.eval()
    image = torch.zeros(1, 3, 1008, 1008, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        outputs = wrapper(image, [TextPrompts(classes=["cat"])])
    canonical = meta_to_canonical(outputs)
    assert canonical.obj_logits.dim() == 2  # (B, Q)
    assert canonical.pred_boxes.shape[-1] == 4
    assert canonical.pred_masks.shape[-1] == 288
    assert canonical.img_presence.dim() == 1  # (B,)


@pytest.mark.gpu_t4
def test_load_sam31_multiplex_K8_forward() -> None:
    """Real K=8 multiplex forward emits pred_logits.shape[0] == B*8 and finite outputs.

    Per spec §13 AC 16. Confirms (B*K, ...) row layout end-to-end on real weights.
    """
    cfg = ModelConfig(device="cuda", dtype="bfloat16")
    wrapper = load_sam31(cfg)
    wrapper.eval()
    b = 2
    k = 8
    image = torch.zeros(b, 3, 1008, 1008, dtype=torch.bfloat16, device="cuda")
    classes = [f"class_{i}" for i in range(k)]
    prompts = [TextPrompts(classes=classes) for _ in range(b)]
    with torch.no_grad():
        outputs = wrapper(image, prompts)
    assert outputs["pred_logits"].shape[0] == b * k
    assert outputs["presence_logit_dec"].shape[0] == b * k
    assert outputs["pred_masks"].shape[0] == b * k
    assert outputs["pred_boxes"].shape[0] == b * k
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_boxes"]).all()
