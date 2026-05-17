"""Tiny stub mirroring SAM 3.1's attention module naming for LoRA tests.

This stub exists so SCOPE_TARGETS regex patterns can be exercised on CPU
without the real Meta checkpoint. It is structurally a Sam3Wrapper around a
two-layer fake model with a `vision_encoder` and `mask_decoder` subtree, plus
two negative-control Linear modules outside either subtree.

forward() raises NotImplementedError — these tests never execute forward.
"""

from __future__ import annotations

from torch import Tensor, nn

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
    """Fake SAM 3.1 inner-base with realistic attention naming."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.vision_encoder = nn.Module()
        self.vision_encoder.block0 = _AttnBlock(dim)  # type: ignore[assignment]
        self.vision_encoder.block1 = _AttnBlock(dim)  # type: ignore[assignment]
        self.mask_decoder = nn.Module()
        self.mask_decoder.layer0 = _DecoderLayer(dim)  # type: ignore[assignment]
        # Negative controls: Linears outside any LoRA scope.
        self.neg_control_a = nn.Linear(dim, dim)
        self.neg_control_b = nn.Linear(dim, dim)

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise NotImplementedError("TinySam3LoraStub.forward is intentionally not implemented")


class _StubAdapter(nn.Module):
    """Minimal adapter mirroring _Sam3ImageAdapter's two-level model attribute."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.model = base

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise NotImplementedError("_StubAdapter.forward is intentionally not implemented")


def make_stub_wrapper(dim: int = 128) -> Sam3Wrapper:
    """Build a Sam3Wrapper whose inner base is a TinySam3LoraStub."""
    base = TinySam3LoraStub(dim=dim)
    adapter = _StubAdapter(base)
    return Sam3Wrapper(adapter, image_size=8, mask_size=8)
