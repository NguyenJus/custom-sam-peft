"""GPU regression: real K=16 multiplex forward at decide_eval_batch_size's
choice for image_size=1008 runs without OOM; peak <= 4x predicted_bytes.

The 4x ceiling is a conservative regression guard, not a tightness check -
see spec §9 for the calibration-constant note.
"""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.config.schema import ModelConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, load_sam31
from custom_sam_peft.presets import decide_eval_batch_size

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_t4,
]


def test_real_K16_forward_at_chosen_B_within_predicted_envelope() -> None:
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    bs, predicted_bytes, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)

    if predicted_bytes == 0:
        pytest.skip("CPU fallback — needs a compatible GPU")

    cfg = ModelConfig(device="cuda", dtype="bfloat16")
    wrapper = load_sam31(cfg)
    wrapper.eval()

    images = torch.zeros(
        bs, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    classes = [f"class_{i}" for i in range(MULTIPLEX_CAP)]
    prompts = [TextPrompts(classes=classes) for _ in range(bs)]

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = wrapper(images, prompts)
    peak = torch.cuda.max_memory_allocated()

    assert outputs["pred_logits"].shape[0] == bs * MULTIPLEX_CAP
    assert peak <= 4 * predicted_bytes, (
        f"peak={peak} > 4 * predicted_bytes={4 * predicted_bytes}; "
        "either forward_only_factor underestimates eval memory or this GPU "
        "is over the empirical envelope. See spec §9 calibration note."
    )
