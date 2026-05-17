"""A tiny `nn.Module` that mimics Meta SAM 3.1's image-forward output dict.

Used to unit-test the Sam3Wrapper, meta_to_canonical adapter, and the loss
pipeline without loading the real ~3.5 GB checkpoint. Output keys match
Meta's `Sam3Image.forward_grounding` contract.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinySam3Stub(nn.Module):
    """Returns Meta-shaped output dict given image + prompts.

    Q = number of decoder queries (default 4 for fast tests).
    The stub is per-class (one prompt at a time), matching Sam3Wrapper's
    single-prompt forward contract.
    """

    def __init__(self, num_queries: int = 4, mask_size: int = 16) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.mask_size = mask_size
        # One trainable param so optimizers have something to update.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image: torch.Tensor,
        prompts: Any,
        box_hints: Any = None,
    ) -> dict[str, torch.Tensor]:
        del prompts, box_hints  # ignored by the stub
        b = image.shape[0] if image.ndim == 4 else 1
        q, m = self.num_queries, self.mask_size
        return {
            "pred_logits": torch.zeros(b, q, 1) + self.dummy,
            "pred_boxes": torch.zeros(b, q, 4) + self.dummy,
            "pred_masks": torch.zeros(b, q, m, m) + self.dummy,
            "presence_logit_dec": torch.zeros(b, 1) + self.dummy,
        }
