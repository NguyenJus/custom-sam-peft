"""A tiny `nn.Module` matching the (planned) SAM3.1 forward contract.

Used to unit-test trainer/eval/peft adapter logic without loading real weights.
The forward signature is intentionally loose — the contract gets pinned in
spec/model-loading.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinySam3Stub(nn.Module):
    """Returns deterministically-shaped random outputs given image + prompts."""

    def __init__(self, num_classes: int = 2, mask_size: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        # A single trainable param so optimizers have something to update.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, prompts: Any) -> dict[str, torch.Tensor]:
        del prompts  # ignored by the stub
        batch = image.shape[0] if image.ndim == 4 else 1
        return {
            "masks": torch.zeros(batch, 1, self.mask_size, self.mask_size) + self.dummy,
            "boxes": torch.zeros(batch, 1, 4) + self.dummy,
            "objectness": torch.zeros(batch, 1) + self.dummy,
            "class_logits": torch.zeros(batch, 1, self.num_classes) + self.dummy,
        }
