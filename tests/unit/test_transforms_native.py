"""CPU-only: verify pad-only / native-res eval transform (downscale=False, design C §5.4)."""

import numpy as np
import torch

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.transforms import build_eval_transforms

_MODEL = "facebook/sam3.1"


def test_downscale_false_is_byte_identical_for_small_inputs():
    # 800x1008 — longest edge == image_size, so LongestMaxSize is a no-op either way.
    img = (np.random.RandomState(0).rand(800, 1008, 3) * 255).astype(np.uint8)
    t_down = build_eval_transforms(
        1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=True
    )
    t_native = build_eval_transforms(
        1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False
    )
    a = t_down(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    b = t_native(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    assert torch.allclose(a, b)


def test_downscale_false_does_not_shrink_oversized():
    # 1500x1500 — larger than image_size=1008.
    img = (np.random.RandomState(0).rand(1500, 1500, 3) * 255).astype(np.uint8)
    t_down = build_eval_transforms(
        1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=True
    )
    t_native = build_eval_transforms(
        1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False
    )
    a = t_down(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    b = t_native(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    # downscale=True: LongestMaxSize shrinks to 1008, PadIfNeeded pads to 1008x1008.
    assert a.shape == (3, 1008, 1008)
    # downscale=False: no LongestMaxSize, PadIfNeeded only pads UP to min — 1500>=1008
    # so no padding either; tensor stays 1500x1500.
    assert b.shape == (3, 1500, 1500)
