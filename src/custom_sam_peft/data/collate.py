"""Variable-shape batch collator for the data subsystem."""

from __future__ import annotations

from typing import Any

import torch

from custom_sam_peft.data.base import Example


def collate_batch(examples: list[Example]) -> dict[str, Any]:
    """Stack images, keep ragged prompts/instances as Python lists.

    Returns a dict with keys: "images" (B,3,H,W), "image_ids" (list[str]),
    "prompts" (list[Prompts]), "instances" (list[list[Instance]]).

    Raises:
        ValueError: empty input or mismatched image shapes across the batch.
    """
    if not examples:
        raise ValueError("collate_batch received empty batch")
    ref_shape = tuple(examples[0].image.shape)
    for ex in examples[1:]:
        shp = tuple(ex.image.shape)
        if shp != ref_shape:
            raise ValueError(f"collate_batch: image shape mismatch: {ref_shape} vs {shp}")
    images = torch.stack([ex.image for ex in examples], dim=0)
    return {
        "images": images,
        "image_ids": [ex.image_id for ex in examples],
        "prompts": [ex.prompts for ex in examples],
        "instances": [list(ex.instances) for ex in examples],
    }
