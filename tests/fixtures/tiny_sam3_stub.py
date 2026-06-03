"""A tiny `nn.Module` that mimics Meta SAM 3.1's image-forward output dict.

Used to unit-test the Sam3Wrapper, meta_to_canonical adapter, and the loss
pipeline without loading the real ~3.5 GB checkpoint. Output keys match
Meta's `Sam3Image.forward_grounding` contract.

The class additionally exposes a tiny *traceable* sub-structure
(``backbone.forward_image``, ``backbone.forward_text``, ``forward_grounding``)
so the ONNX export shims (`_EncoderExport` / `_DecoderExport`,
`export/onnx.py`) can be traced by ``torch.onnx.export`` at tiny scale on CPU.
The original ``forward(image, prompts) -> 4-key dict`` contract is preserved
verbatim so all existing stub-based tests keep passing.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from custom_sam_peft.data.base import TextPrompts


class _TinyBackbone(nn.Module):
    """Traceable vision + text backbone with a single FPN level (L = 1).

    ``forward_image`` returns a Meta-shaped ``backbone_out`` dict carrying a
    single-level ``backbone_fpn`` list and matching ``vision_pos_enc`` list.
    ``forward_text`` consumes a Python ``list[str]`` (non-traceable, baked at
    export time) and returns a constant text-embedding tensor.
    """

    def __init__(self, dim: int = 4, feat_hw: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.feat_hw = feat_hw
        # A 3->dim stem so the encoder graph has real (traceable) weights.
        self.stem = nn.Conv2d(3, dim, kernel_size=1)

    def forward_image(self, images: Tensor) -> dict[str, Any]:
        """Return a single-level backbone_out dict (feats + positional enc)."""
        # Collapse spatial dims to a fixed feat_hw via adaptive pooling so the
        # traced graph has a fixed output shape independent of input H/W.
        feat = self.stem(images)
        feat = nn.functional.adaptive_avg_pool2d(feat, (self.feat_hw, self.feat_hw))
        b = feat.shape[0]
        pos = torch.zeros(b, self.dim, self.feat_hw, self.feat_hw, dtype=feat.dtype)
        return {"backbone_fpn": [feat], "vision_pos_enc": [pos]}

    def forward_text(self, classes: list[str], device: Any = None) -> dict[str, Tensor]:
        """Return a baked constant text embedding for K classes (non-traceable input)."""
        del device
        k = len(classes)
        embed = torch.zeros(k, self.dim)
        return {"text_embed": embed}


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
        # Traceable sub-structure used by the ONNX export shims.
        self.backbone = _TinyBackbone()

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

    def forward_grounding(
        self,
        *,
        backbone_out: dict[str, Any],
        find_input: Any,
        find_target: Any = None,
        geometric_prompt: Any = None,
    ) -> dict[str, Tensor]:
        """Traceable grounding core: emit the four-key dict for R = len(img_ids) rows."""
        del find_target, geometric_prompt
        feat = backbone_out["backbone_fpn"][0]
        # Number of output rows R = B*K is encoded by the multiplex img_ids length.
        img_ids = find_input.img_ids
        r = int(img_ids.shape[0])
        # Tie outputs to feat so the traced graph actually depends on the encoder.
        base = feat.reshape(feat.shape[0], -1).mean()
        q, m = self.num_queries, self.mask_size
        return {
            "pred_logits": torch.zeros(r, q, 1) + base,
            "pred_boxes": torch.zeros(r, q, 4) + base,
            "pred_masks": torch.zeros(r, q, m, m) + base,
            "presence_logit_dec": torch.zeros(r, 1) + base,
        }
