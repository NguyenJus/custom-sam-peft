"""A tiny `nn.Module` that mimics Meta SAM 3.1's image-forward output dict.

Used to unit-test the Sam3Wrapper, meta_to_canonical adapter, and the loss
pipeline without loading the real ~3.5 GB checkpoint. Output keys match
Meta's `Sam3Image.forward_grounding` contract.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from custom_sam_peft.data.base import TextPrompts


class TinySam3Stub(nn.Module):
    """Returns Meta-shaped output dict given image + prompts.

    Q = number of decoder queries (default 4 for fast tests).

    In multiplex mode (K > 1 classes per TextPrompts), the real SAM 3.1 model
    returns (B*K, Q, ...) shaped outputs (one row per image-class pair).  This
    stub replicates that contract: when prompts is a list of TextPrompts with
    K classes each, the output batch dimension is B*K.  For K=1 (legacy) the
    output batch dimension is B.
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
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs  # support= (outer-model path) is ignored
        b = image.shape[0] if image.ndim == 4 else 1
        # Multiplex: if prompts are TextPrompts with K classes each,
        # the real model expands the batch to B*K (one row per image-class slot).
        k = 1
        if isinstance(prompts, list) and prompts and isinstance(prompts[0], TextPrompts):
            k = len(prompts[0].classes)
        bk = b * k
        q, m = self.num_queries, self.mask_size
        return {
            "pred_logits": torch.zeros(bk, q, 1) + self.dummy,
            "pred_boxes": torch.zeros(bk, q, 4) + self.dummy,
            "pred_masks": torch.zeros(bk, q, m, m) + self.dummy,
            "presence_logit_dec": torch.zeros(bk, 1) + self.dummy,
        }
