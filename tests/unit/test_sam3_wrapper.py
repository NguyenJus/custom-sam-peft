"""Unit tests for Sam3Wrapper using TinySam3Stub (no real model)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.models.sam3 import Sam3Wrapper, _Sam3ImageAdapter
from custom_sam_peft.models.trunk_cache import (
    SPIKE_TRUNK_FWD_MS,
    TrunkFeatureCache,
    trunk_fingerprint,
)
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def test_wrapper_passes_through_single_class_text_prompts() -> None:
    stub = TinySam3Stub(num_queries=2, mask_size=16)
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"]), TextPrompts(classes=["cat"])]
    out = wrapper(image, prompts)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"}


def test_wrapper_rejects_multi_class_text_prompts() -> None:
    """Multi-class prompts are now valid up to MULTIPLEX_CAP; over-cap is rejected."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(1, 3, 64, 64)
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    prompts = [TextPrompts(classes=too_many)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        wrapper(image, prompts)


def test_wrapper_rejects_batch_size_mismatch() -> None:
    stub = TinySam3Stub()
    wrapper = Sam3Wrapper(stub, mask_size=16)
    image = torch.zeros(2, 3, 64, 64)
    prompts = [TextPrompts(classes=["cat"])]  # B=2 images but 1 prompt
    with pytest.raises(ValueError, match="len\\(prompts\\)"):
        wrapper(image, prompts)


def test_sam3_wrapper_has_peft_model_slot() -> None:
    from torch import nn

    from custom_sam_peft.models.sam3 import Sam3Wrapper

    wrapper = Sam3Wrapper(nn.Identity(), mask_size=8)
    assert hasattr(wrapper, "peft_model")
    assert wrapper.peft_model is None


def test_multiplex_cap_constant_exists() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    assert MULTIPLEX_CAP == 16


def _imgs(b: int) -> torch.Tensor:
    return torch.zeros(b, 3, 8, 8)


def _default_wrapper() -> Sam3Wrapper:
    """Return a Sam3Wrapper with default (rgb, 3-channel) settings for _validate_inputs tests."""
    from torch import nn

    return Sam3Wrapper(nn.Identity(), mask_size=8)


def test_validate_inputs_accepts_K_between_1_and_cap() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    w = _default_wrapper()
    for k in (1, 5, MULTIPLEX_CAP):
        prompts = [TextPrompts(classes=[f"c{i}" for i in range(k)])] * 2
        w._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_rejects_K_zero() -> None:
    w = _default_wrapper()
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        w._validate_inputs(_imgs(1), [TextPrompts(classes=[])], None)


def test_validate_inputs_rejects_K_over_cap() -> None:
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    w = _default_wrapper()
    too_many = [f"c{i}" for i in range(MULTIPLEX_CAP + 1)]
    with pytest.raises(ValueError, match="MULTIPLEX_CAP"):
        w._validate_inputs(_imgs(1), [TextPrompts(classes=too_many)], None)


def test_validate_inputs_rejects_mismatched_class_lists_across_batch() -> None:
    w = _default_wrapper()
    prompts = [TextPrompts(classes=["cat", "dog"]), TextPrompts(classes=["dog", "cat"])]
    with pytest.raises(ValueError, match=r"same.*class"):
        w._validate_inputs(_imgs(2), prompts, None)


def test_validate_inputs_k1_still_passes() -> None:
    w = _default_wrapper()
    w._validate_inputs(
        _imgs(3),
        [TextPrompts(classes=["cat"]) for _ in range(3)],
        None,
    )


# ---------------------------------------------------------------------------
# Task 11: channel-adapter wiring tests
# ---------------------------------------------------------------------------


class _StubBackbone(nn.Module):
    def forward_image(self, images):  # pragma: no cover - shape probe
        return {"_chans": images.shape[1]}


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _StubBackbone()


def test_validate_inputs_accepts_configured_channels_rejects_wrong():
    w = Sam3Wrapper(
        _Sam3ImageAdapter(_StubModel(), channels=5, channel_semantics="freeform"),
        channels=5,
        channel_semantics="freeform",
    )
    # Accept: correct channel count, correct ndim
    w._validate_inputs(
        torch.zeros(1, 5, 8, 8),
        [TextPrompts(classes=["cat"])],
        None,
    )
    # Reject: wrong channel count (3 instead of 5)
    with pytest.raises(ValueError, match=r"\(B, 5, H, W\)"):
        w._validate_inputs(torch.zeros(1, 3, 8, 8), [TextPrompts(classes=["cat"])], None)
    # Reject: ndim != 4
    with pytest.raises(ValueError):
        w._validate_inputs(torch.zeros(5, 8, 8), [], None)


def test_rgb_adapter_is_none_zero_new_params():
    ad = _Sam3ImageAdapter(_StubModel(), channels=3, channel_semantics="rgb")
    assert ad.channel_adapter is None
    base = sum(p.numel() for p in _StubModel().parameters())
    total = sum(p.numel() for p in ad.parameters())
    assert total == base  # zero new params for rgb


def test_freeform_adapter_present_and_trainable():
    ad = _Sam3ImageAdapter(_StubModel(), channels=4, channel_semantics="freeform")
    assert isinstance(ad.channel_adapter, nn.Conv2d)
    assert any(p.requires_grad for p in ad.channel_adapter.parameters())


# ---------------------------------------------------------------------------
# Trunk cache adapter integration tests (spec §6 / shared-interface contract)
# ---------------------------------------------------------------------------
#
# These tests drive _Sam3ImageAdapter with an attached TrunkFeatureCache via a
# CPU stub that provides forward_image / forward_text / forward_grounding.
# They verify the adapter-level assembly/tiling logic in sam3.py:358-446 that
# is the highest-risk untested surface (spec §6 §3 Integration).
# ---------------------------------------------------------------------------

# FPN shapes used by all stub tests: two levels each at (B, C, H, W) with B=1.
_FPN_SHAPES: list[tuple[int, ...]] = [(1, 2, 4, 4), (1, 2, 4, 4)]
# Fixed number of mask-decoder queries in the stub.
_NUM_Q: int = 2
# Mask size used by the stub.
_MASK_SZ: int = 8


def _make_fpn(b: int) -> list[torch.Tensor]:
    """Return FPN tensors with a known, reproducible seeded pattern."""
    g = torch.Generator()
    g.manual_seed(0)
    return [torch.randn(b, *shape[1:], generator=g) for shape in _FPN_SHAPES]


def _fp16_rt(fpn: list[torch.Tensor]) -> list[torch.Tensor]:
    """Simulate the fp16 round-trip applied by put_batch + _h2d_entry.

    put_batch casts each tensor to fp16; _h2d_entry casts back to model_dtype
    (float32 in CPU tests).  Returns CPU tensors for device-agnostic comparison.
    """
    return [t.detach().cpu().to(dtype=torch.float16).to(dtype=torch.float32) for t in fpn]


class _CacheStubBackbone(nn.Module):
    """Stub backbone for _Sam3ImageAdapter cache integration tests."""

    def __init__(self) -> None:
        super().__init__()
        # Single frozen parameter so next(model.parameters()) works.
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward_image(self, images: torch.Tensor) -> dict[str, Any]:
        b = images.shape[0]
        fpn = _make_fpn(b)
        pos_enc = [torch.ones(b, 1, 1, 1)]  # content-independent, deterministic
        return {
            "backbone_fpn": fpn,
            "vision_features": fpn[-1],
            "vision_pos_enc": pos_enc,
            "sam2_backbone_out": None,
        }

    def forward_text(self, classes: list[str], *, device: torch.device) -> dict[str, Any]:
        return {}


class _CacheStubModel(nn.Module):
    """Full model stub for _Sam3ImageAdapter: backbone + forward_grounding."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _CacheStubBackbone()

    def forward_grounding(
        self,
        backbone_out: dict[str, Any],
        find_input: Any,
        find_target: Any,
        geometric_prompt: Any,
    ) -> dict[str, Any]:
        bk = find_input.img_ids.shape[0]  # B*K from multiplex
        q, m = _NUM_Q, _MASK_SZ
        return {
            "pred_logits": torch.zeros(bk, q, 1),
            "pred_boxes": torch.zeros(bk, q, 4),
            "pred_masks": torch.zeros(bk, q, m, m),
            "presence_logit_dec": torch.zeros(bk, 1),
            # Stash backbone_out so tests can inspect what was passed to forward_grounding.
            "_backbone_out_under_test": backbone_out,
        }


def _make_adapter_with_cache(
    tmp_path: Path,
    b: int = 1,
) -> tuple[_Sam3ImageAdapter, TrunkFeatureCache]:
    """Build a _Sam3ImageAdapter backed by _CacheStubModel and a TrunkFeatureCache."""
    stub_model = _CacheStubModel()
    adapter = _Sam3ImageAdapter(stub_model, channels=3, channel_semantics="rgb")

    fp = trunk_fingerprint(
        checkpoint_id="test-ckpt",
        scope="decoder_concept",
        dtype="float32",
        image_size=1008,
    )
    cache = TrunkFeatureCache(
        cache_dir=tmp_path / "cache",
        fingerprint=fp,
        model_dtype=torch.float32,
        n_samples=b + 10,
        per_image_bytes=1024,
        trunk_fwd_ms=SPIKE_TRUNK_FWD_MS,
        free_disk_fraction=0.99,
        allow_slow_disk=True,  # skip throughput probe in unit tests
    )
    adapter.attach_trunk_cache(cache)
    return adapter, cache


def _prompts(b: int) -> list[TextPrompts]:
    return [TextPrompts(classes=["cat"]) for _ in range(b)]


def _images(b: int) -> torch.Tensor:
    return torch.zeros(b, 3, 8, 8)


def _uids(b: int, prefix: str = "img") -> list[str]:
    return [f"{prefix}:{i}" for i in range(b)]


# ---------------------------------------------------------------------------
# Test: Sam3Wrapper.trunk_cache property
# ---------------------------------------------------------------------------


def test_wrapper_trunk_cache_is_none_without_cache() -> None:
    """Sam3Wrapper.trunk_cache returns None for a stub (no _trunk_cache attr)."""
    wrapper = Sam3Wrapper(TinySam3Stub(), mask_size=_MASK_SZ)
    assert wrapper.trunk_cache is None


def test_wrapper_trunk_cache_returns_cache_when_attached(tmp_path: Path) -> None:
    """Sam3Wrapper.trunk_cache returns the cache after attach_trunk_cache."""
    stub_model = _CacheStubModel()
    adapter = _Sam3ImageAdapter(stub_model, channels=3, channel_semantics="rgb")
    fp = trunk_fingerprint(
        checkpoint_id="test", scope="decoder_concept", dtype="float32", image_size=1008
    )
    cache = TrunkFeatureCache(
        cache_dir=tmp_path / "c",
        fingerprint=fp,
        model_dtype=torch.float32,
        n_samples=5,
        per_image_bytes=1024,
        trunk_fwd_ms=SPIKE_TRUNK_FWD_MS,
        free_disk_fraction=0.99,
        allow_slow_disk=True,
    )
    adapter.attach_trunk_cache(cache)
    wrapper = Sam3Wrapper(adapter, mask_size=_MASK_SZ)
    assert wrapper.trunk_cache is cache


# ---------------------------------------------------------------------------
# Test: epoch-0 store → epoch-1 replay equivalence (B=1)
# ---------------------------------------------------------------------------


def test_adapter_epoch0_stores_epoch1_replays_b1(tmp_path: Path) -> None:
    """Spec §6 epoch-0-store / epoch-1-replay equivalence (B=1).

    The replayed backbone_fpn must match the fp16 round-trip of a fresh
    forward_image call (fp16 storage is lossy, so 'bit-identical' means up to
    the fp16 cast — NOT the original float32 values).
    """
    adapter, _cache = _make_adapter_with_cache(tmp_path, b=1)
    uids = _uids(1)
    imgs = _images(1)
    prompts_b1 = _prompts(1)

    # Epoch 0: cache miss — runs forward_image and stores.
    out0 = adapter(imgs, prompts_b1, sample_uids=uids)
    bo0 = out0["_backbone_out_under_test"]

    # vision_pos_enc must be present (tiled from cached copy after epoch 0).
    assert "vision_pos_enc" in bo0
    assert isinstance(bo0["vision_pos_enc"], list)
    assert len(bo0["vision_pos_enc"]) >= 1

    # Epoch 1: cache hit — assembles backbone_out from stored blobs.
    out1 = adapter(imgs, prompts_b1, sample_uids=uids)
    bo1 = out1["_backbone_out_under_test"]

    # backbone_fpn should match the fp16 round-trip of the original.
    # Move both to CPU for device-agnostic comparison (_h2d_entry may have
    # moved replayed tensors to CUDA when CUDA is available).
    expected_fpn = _fp16_rt(bo0["backbone_fpn"])
    for i, (replayed, expected) in enumerate(zip(bo1["backbone_fpn"], expected_fpn, strict=True)):
        assert torch.allclose(replayed.cpu().float(), expected.float(), atol=0.0), (
            f"FPN level {i}: replayed backbone_fpn differs from fp16 round-trip"
        )

    # vision_features must equal backbone_fpn[-1].
    assert torch.equal(bo1["vision_features"].cpu(), bo1["backbone_fpn"][-1].cpu())

    # vision_pos_enc must be re-attached with matching shape.
    assert bo1["vision_pos_enc"][0].shape == bo0["vision_pos_enc"][0].shape


# ---------------------------------------------------------------------------
# Test: epoch-0 store → epoch-1 replay with B=2 (batch merge)
# ---------------------------------------------------------------------------


def test_adapter_epoch0_stores_epoch1_replays_b2(tmp_path: Path) -> None:
    """Spec §6 epoch-0-store / epoch-1-replay: B=2 batch merge is correct.

    Verifies that per-image entries are torch.cat'd in the right order and
    that vision_pos_enc is tiled to B=2.
    """
    adapter, _cache = _make_adapter_with_cache(tmp_path, b=2)
    uids = _uids(2)
    imgs = _images(2)
    prompts_b2 = _prompts(2)

    # Epoch 0: store.
    out0 = adapter(imgs, prompts_b2, sample_uids=uids)
    bo0 = out0["_backbone_out_under_test"]
    original_fpn = [t.clone() for t in bo0["backbone_fpn"]]

    # Epoch 1: replay.
    out1 = adapter(imgs, prompts_b2, sample_uids=uids)
    bo1 = out1["_backbone_out_under_test"]

    # backbone_fpn[level] should be (B=2, C, H, W).
    for lvl, t in enumerate(bo1["backbone_fpn"]):
        assert t.shape[0] == 2, f"FPN level {lvl}: expected B=2, got {t.shape[0]}"

    # Replayed values match the fp16 round-trip of the originals.
    expected_fpn = _fp16_rt(original_fpn)
    for i, (replayed, expected) in enumerate(zip(bo1["backbone_fpn"], expected_fpn, strict=True)):
        assert torch.allclose(replayed.cpu().float(), expected.float(), atol=0.0), (
            f"FPN level {i}: B=2 replay differs from fp16 round-trip"
        )

    # vision_pos_enc must be tiled to B=2.
    assert bo1["vision_pos_enc"][0].shape[0] == 2, "vision_pos_enc not tiled to B=2"


# ---------------------------------------------------------------------------
# Test: OOM-style microbatch subset slicing (all-or-none per microbatch)
# ---------------------------------------------------------------------------


def test_adapter_microbatch_subset_uses_correct_uids(tmp_path: Path) -> None:
    """Spec §3 Integration: sample_uids sliced by micro_indices locks in correct entries.

    Simulates the OOM-microbatch path in loop.py where _forward_group is called
    with micro_indices=[0] and micro_indices=[1] separately from a B=2 batch.
    Verifies that each micro-forward independently stores and replays correctly.
    """
    uids_full = _uids(2)
    imgs_full = _images(2)

    # Two separate adapters sharing the same cache dir to simulate epoch
    # boundary storage + replay.
    adapter_store, cache_store = _make_adapter_with_cache(tmp_path, b=1)
    adapter_replay, cache_replay = _make_adapter_with_cache(tmp_path / "r", b=1)
    # Share the same cache dir and fingerprint so stored blobs are visible on replay.
    cache_replay._cache_dir = cache_store._cache_dir
    cache_replay._fingerprint = cache_store._fingerprint

    for i, uid in enumerate(uids_full):
        micro_imgs = imgs_full[i : i + 1]
        micro_uids = [uid]
        micro_prompts = [TextPrompts(classes=["cat"])]

        # Epoch 0: store micro-batch i.
        out_store = adapter_store(micro_imgs, micro_prompts, sample_uids=micro_uids)
        bo_store = out_store["_backbone_out_under_test"]

        # Epoch 1: replay micro-batch i via the replay adapter.
        out_replay = adapter_replay(micro_imgs, micro_prompts, sample_uids=micro_uids)
        bo_replay = out_replay["_backbone_out_under_test"]

        expected_fpn = _fp16_rt(bo_store["backbone_fpn"])
        for lvl, (replayed, expected) in enumerate(
            zip(bo_replay["backbone_fpn"], expected_fpn, strict=True)
        ):
            assert torch.allclose(replayed.cpu().float(), expected.float(), atol=0.0), (
                f"micro_idx={i} FPN level {lvl}: replay differs from fp16 round-trip"
            )


# ---------------------------------------------------------------------------
# Test: vision_pos_enc is tiled to the correct shape (contiguous)
# ---------------------------------------------------------------------------


def test_adapter_replay_pos_enc_is_contiguous(tmp_path: Path) -> None:
    """Replayed vision_pos_enc must be contiguous (no stride-0 aliasing).

    Verifies the .contiguous() fix in sam3.py forward (minor fix from review).
    """
    adapter, _cache = _make_adapter_with_cache(tmp_path, b=2)
    uids = _uids(2)
    imgs = _images(2)
    prompts_b2 = _prompts(2)

    # Epoch 0: store.
    adapter(imgs, prompts_b2, sample_uids=uids)
    # Epoch 1: replay.
    out1 = adapter(imgs, prompts_b2, sample_uids=uids)
    bo1 = out1["_backbone_out_under_test"]

    for lvl, p in enumerate(bo1["vision_pos_enc"]):
        # vision_pos_enc is on CPU (cached from pos_enc[0:1].detach().cpu())
        # and tiled via expand(...).contiguous().
        assert p.is_contiguous(), (
            f"vision_pos_enc level {lvl}: not contiguous after expand+contiguous"
        )


# ---------------------------------------------------------------------------
# Test: prefetch-then-cold-read fallthrough (miss-vs-hit branching)
# ---------------------------------------------------------------------------


def test_adapter_prefetch_result_consumed_on_replay(tmp_path: Path) -> None:
    """Spec §6 prefetch correctness: prefetch result is consumed on replay.

    After prefetch() is called for a set of uids, _get_prefetched should return
    the entries (once the thread finishes), and the replay should yield the same
    features as a cold get_batch.
    """
    adapter, cache = _make_adapter_with_cache(tmp_path, b=1)
    uids = _uids(1)
    imgs = _images(1)
    prompts_b1 = _prompts(1)

    # Epoch 0: store.
    out0 = adapter(imgs, prompts_b1, sample_uids=uids)
    bo0 = out0["_backbone_out_under_test"]

    # Manually schedule prefetch for the same uids.
    cache.prefetch(uids)
    # Wait for background thread to finish (tiny test blobs, no real delay).
    deadline = time.monotonic() + 5.0
    while cache._prefetch_result is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cache._prefetch_result is not None, "Prefetch thread did not complete in time"

    # Epoch 1 replay via adapter: should consume the prefetched result.
    out1 = adapter(imgs, prompts_b1, sample_uids=uids)
    bo1 = out1["_backbone_out_under_test"]

    expected_fpn = _fp16_rt(bo0["backbone_fpn"])
    for i, (replayed, expected) in enumerate(zip(bo1["backbone_fpn"], expected_fpn, strict=True)):
        assert torch.allclose(replayed.cpu().float(), expected.float(), atol=0.0), (
            f"Prefetch-replay FPN level {i} differs from fp16 round-trip"
        )

    # After consumption, _prefetch_result should be cleared.
    assert cache._prefetch_result is None, "_get_prefetched did not clear result"
