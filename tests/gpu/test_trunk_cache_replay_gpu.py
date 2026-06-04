"""Real-SAM3.1 GPU smoke test for the #300 trunk feature cache (pure replay).

Validates what the CPU-stub unit tests structurally cannot:

  - the REAL ``forward_image`` dict round-trips through the on-disk fp16 cache
    bit-identically (up to the fp16 cast) on every FPN level;
  - epoch-1 genuinely SKIPS the trunk (no second ``forward_image`` call);
  - the replayed ``backbone_out`` feeds ``forward_grounding`` end-to-end on a
    real CUDA device — this is a DEVICE regression guard: ``vision_pos_enc`` is
    cached on CPU but must be re-attached on the model device, or
    ``forward_grounding``'s ``vis_pos_enc[img_ids]`` raises a cross-device error
    (a bug a CPU-only suite cannot surface — everything is on cpu there);
  - the three §2 correctness guards and the §3.5 build-time throughput / free-disk
    activation guards all PASS on the real frozen ``decoder_concept`` model.

cite: spec docs/superpowers/specs/2026-06-04-trunk-feature-cache-300-design.md §1, §3, §3.5, §4, §6.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import ModelConfig, NormalizeConfig, PEFTConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.data.transforms import build_eval_transforms
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE, load_sam31
from custom_sam_peft.models.trunk_cache import (
    SPIKE_PER_IMAGE_BYTES,
    SPIKE_TRUNK_FWD_MS,
    TrunkFeatureCache,
    assert_aug_off,
    assert_rgb_input,
    assert_trunk_frozen,
    trunk_fingerprint,
)

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def _fp16_roundtrip(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().to(torch.float16).to(dtype)


def test_trunk_cache_real_replay_equivalence() -> None:
    cfg = ModelConfig(dtype="bfloat16")
    wrapper = load_sam31(cfg, channels=3, channel_semantics="rgb")
    # Default decoder_concept PEFT so the guards see the real frozen trunk state.
    lookup("peft", "lora")(wrapper, PEFTConfig(method="lora"))
    wrapper.eval()
    device = next(wrapper.parameters()).device
    model_dtype = next(wrapper.parameters()).dtype

    adapter = wrapper.model  # _Sam3ImageAdapter
    backbone = adapter.model.backbone  # type: ignore[union-attr]

    # --- §2 correctness guards must PASS on the real frozen model ---
    assert_trunk_frozen(wrapper)
    assert_rgb_input(adapter.channel_adapter)
    eval_tf = build_eval_transforms(
        SAM3_IMAGE_SIZE,
        model_name=cfg.name,
        normalize=NormalizeConfig(),
        channel_semantics="rgb",
    )
    assert_aug_off(eval_tf)

    # --- fixed deterministic input ---
    torch.manual_seed(0)
    images = torch.randn(1, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, dtype=torch.float32, device=device)
    prompts = [TextPrompts(classes=["object"])]
    uids = ["img0:full"]

    # --- reference: fresh forward_image, fp16 round-tripped ---
    with torch.no_grad():
        ref = backbone.forward_image(images)
    ref_fpn = [_fp16_roundtrip(t, model_dtype) for t in ref["backbone_fpn"]]

    with tempfile.TemporaryDirectory(prefix="trunk_cache_gpu_") as tmp:
        cache_dir = Path(tmp) / ".trunk_cache"
        fp = trunk_fingerprint(
            checkpoint_id=cfg.checkpoint_file,
            scope="decoder_concept",
            dtype=str(model_dtype),
            image_size=SAM3_IMAGE_SIZE,
        )
        # --- §3.5 build-time activation guards (throughput probe + free-disk fit) PASS ---
        cache = TrunkFeatureCache(
            cache_dir=cache_dir,
            fingerprint=fp,
            model_dtype=model_dtype,
            n_samples=1,
            per_image_bytes=SPIKE_PER_IMAGE_BYTES,
            trunk_fwd_ms=SPIKE_TRUNK_FWD_MS,
        )
        adapter.attach_trunk_cache(cache)

        # Count real forward_image calls to prove epoch-1 skips the trunk.
        calls = {"n": 0}
        orig = backbone.forward_image

        def _counting(imgs: torch.Tensor, *a: object, **k: object) -> dict:
            calls["n"] += 1
            return orig(imgs, *a, **k)

        backbone.forward_image = _counting  # type: ignore[method-assign]
        try:
            with torch.no_grad():
                out0 = wrapper(images, prompts, sample_uids=uids)  # epoch 0: miss -> store
            n0 = calls["n"]
            with torch.no_grad():
                out1 = wrapper(images, prompts, sample_uids=uids)  # epoch 1: hit -> replay
            n1 = calls["n"]
        finally:
            backbone.forward_image = orig  # type: ignore[method-assign]

        # epoch 0 ran the trunk; epoch 1 did NOT call forward_image again.
        assert n0 >= 1
        assert n1 == n0, "epoch-1 replay must skip the trunk (no new forward_image call)"

        # Replayed fpn == fp16 round-trip of the fresh recompute, every level.
        replayed = cache.get_batch(uids)
        assert replayed is not None
        rep_fpn = replayed[0]["backbone_fpn"]
        assert len(rep_fpn) == len(ref_fpn)
        for r, e in zip(rep_fpn, ref_fpn, strict=True):
            assert torch.equal(r[:1].float(), e[:1].float())

        # End-to-end grounding ran on the replayed features on a real GPU
        # (the device regression guard) and produced finite masks.
        for out in (out0, out1):
            assert "pred_masks" in out
            assert torch.isfinite(out["pred_masks"]).all()

        cache.teardown()
        assert not cache_dir.exists()
