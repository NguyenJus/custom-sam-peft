"""Tiny stub mirroring SAM 3.1's attention module naming for LoRA tests.

This stub exists so SCOPE_TARGETS regex patterns can be exercised on CPU
without the real Meta checkpoint. It is structurally a Sam3Wrapper around a
two-layer fake model with a `vision_encoder` and `mask_decoder` subtree, plus
two negative-control Linear modules outside either subtree.

By default, forward() raises NotImplementedError — structural tests never
execute forward.  Pass ``working=True`` to get a wrapper with a real forward
path suitable for training-loop integration tests.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from esam3.models.sam3 import Sam3Wrapper


class _AttnBlock(nn.Module):
    """SAM 3.1-style block with fused qkv + output proj."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)  # type: ignore[assignment]
        self.attn.proj = nn.Linear(dim, dim)  # type: ignore[assignment]


class _DecoderAttn(nn.Module):
    """Separate q/k/v/out_proj as used in transformer-style decoders."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = _DecoderAttn(dim)
        self.cross_attn = _DecoderAttn(dim)


class TinySam3LoraStub(nn.Module):
    """Fake SAM 3.1 inner-base with realistic attention naming.

    When ``working=False`` (default), forward raises NotImplementedError.
    When ``working=True``, forward routes through ``vision_encoder.block0.attn.qkv``
    so that LoRA A/B matrices participate in the gradient graph.
    """

    def __init__(self, dim: int = 8, working: bool = False) -> None:
        super().__init__()
        self._working = working
        self._dim = dim
        self._num_queries = 4
        self._mask_size = 8
        self.vision_encoder = nn.Module()
        self.vision_encoder.block0 = _AttnBlock(dim)  # type: ignore[assignment]
        self.vision_encoder.block1 = _AttnBlock(dim)  # type: ignore[assignment]
        self.mask_decoder = nn.Module()
        self.mask_decoder.layer0 = _DecoderLayer(dim)  # type: ignore[assignment]
        # Negative controls: Linears outside any LoRA scope.
        self.neg_control_a = nn.Linear(dim, dim)
        self.neg_control_b = nn.Linear(dim, dim)

    def forward(self, images: Any = None, prompts: Any = None, **kwargs: Any) -> Any:
        if not self._working:
            raise NotImplementedError("TinySam3LoraStub.forward is intentionally not implemented")
        b = images.shape[0]  # type: ignore[union-attr]
        q, m = self._num_queries, self._mask_size
        flat = images.reshape(b, 3, -1).mean(dim=-1)  # type: ignore[union-attr]  # (B, 3)
        feat = torch.nn.functional.pad(flat, (0, self._dim - 3))  # (B, dim)
        feat = self.vision_encoder.block0.attn.qkv(feat)  # type: ignore[operator]  # (B, dim*3)
        scalar = feat.mean()
        return {
            "pred_logits": torch.zeros(b, q, 1) + scalar,
            "pred_boxes": torch.zeros(b, q, 4) + scalar,
            "pred_masks": torch.zeros(b, q, m, m) + scalar,
            "presence_logit_dec": torch.zeros(b, 1) + scalar,
        }


class _StubAdapter(nn.Module):
    """Minimal adapter mirroring _Sam3ImageAdapter's two-level model attribute."""

    def __init__(self, base: TinySam3LoraStub) -> None:
        super().__init__()
        self.model = base

    def forward(self, images: Any = None, prompts: Any = None, box_hints: Any = None) -> Any:
        return self.model(images, prompts)  # type: ignore[return-value]


def make_stub_wrapper(dim: int = 128, working: bool = False) -> Sam3Wrapper:
    """Build a Sam3Wrapper whose inner base is a TinySam3LoraStub.

    Args:
        dim: Linear layer width.
        working: When True the wrapper has a real forward path; when False
            (default) forward raises NotImplementedError.
    """
    base = TinySam3LoraStub(dim=dim, working=working)
    adapter = _StubAdapter(base)
    return Sam3Wrapper(adapter, image_size=8, mask_size=8)
