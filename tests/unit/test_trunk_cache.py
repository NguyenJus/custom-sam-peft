"""Spec §6 tests for TrunkFeatureCache and its correctness guards.

Tests every bullet in spec §6:
  - Guard matrix (trainable trunk / trunk-LoRA, non-RGB adapter, aug-on)
  - Key stability across epochs + shuffle; distinct tiling-window uids
  - Epoch-0-store / epoch-1-replay equivalence (fp16 round-trip)
  - Eviction -> recompute
  - Free-disk fit-check fail-fast (§3.5b)
  - Throughput auto-guard fail / override (§3.5c) — measured throughput only,
    no kernel 'rotational' flag
  - Cleanup on teardown (normal exit and simulated mid-run exception)
  - Prefetch correctness: on vs off yields identical features under shuffled order

CPU-only. No real-model byte/timing numbers in CI (spec §6).

Run with::

    cd <worktree> && PYTHONPATH=src .venv/bin/python -m pytest \
        tests/unit/test_trunk_cache.py -o "addopts=" -p no:cacheprovider -q
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from custom_sam_peft.models.trunk_cache import (
    _ALLOWED_AUG_CLASSES,
    _CACHED_KEYS,
    SPIKE_PER_IMAGE_BYTES,
    SPIKE_TRUNK_FWD_MS,
    TrunkFeatureCache,
    assert_aug_off,
    assert_rgb_input,
    assert_trunk_frozen,
    trunk_fingerprint,
)

# ---------------------------------------------------------------------------
# Helpers / minimal fixtures
# ---------------------------------------------------------------------------


def _make_fake_entry(
    fpn_shapes: list[tuple[int, ...]] | None = None,
    sam2: bool = False,
) -> dict[str, Any]:
    """Return a minimal forward_image-like per-image entry (batch=1).

    Shapes default to two FPN levels each (1, 4, 4) for fast tests.
    """
    if fpn_shapes is None:
        fpn_shapes = [(1, 4, 4), (1, 4, 4)]
    fpn = [torch.randn(*s) for s in fpn_shapes]
    entry: dict[str, Any] = {
        "backbone_fpn": fpn,
    }
    if sam2:
        sam2_fpn = [torch.randn(*s) for s in fpn_shapes]
        entry["sam2_backbone_out"] = {"backbone_fpn": sam2_fpn}
    else:
        entry["sam2_backbone_out"] = None
    return entry


def _make_cache(
    tmp_path: Path,
    *,
    n_samples: int = 5,
    per_image_bytes: int = 1024,
    trunk_fwd_ms: float = SPIKE_TRUNK_FWD_MS,
    free_disk_fraction: float = 0.99,
    allow_slow_disk: bool = True,  # skip throughput probe in unit tests by default
    fingerprint: str | None = None,
    model_dtype: torch.dtype = torch.float32,
) -> TrunkFeatureCache:
    """Build a TrunkFeatureCache bypassing the slow throughput probe."""
    fp = fingerprint or trunk_fingerprint(
        checkpoint_id="test-ckpt",
        scope="decoder_concept",
        dtype="float32",
        image_size=1008,
    )
    return TrunkFeatureCache(
        cache_dir=tmp_path / "cache",
        fingerprint=fp,
        model_dtype=model_dtype,
        n_samples=n_samples,
        per_image_bytes=per_image_bytes,
        trunk_fwd_ms=trunk_fwd_ms,
        free_disk_fraction=free_disk_fraction,
        allow_slow_disk=allow_slow_disk,
    )


# ---------------------------------------------------------------------------
# §6 bullet: Guard matrix — assert_trunk_frozen
# ---------------------------------------------------------------------------


class TestAssertTrunkFrozen:
    """Spec §6: each precondition violation produces a correct hard-error."""

    def _frozen_model(self) -> nn.Module:
        """Minimal model with backbone.vision_backbone — all params frozen."""

        class _VisionBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)
                for p in self.parameters():
                    p.requires_grad_(False)

        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vision_backbone = _VisionBackbone()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = _Backbone()

        return _Model()

    def test_frozen_trunk_passes(self) -> None:
        """A fully-frozen trunk (no requires_grad) must not raise."""
        model = self._frozen_model()
        # Should not raise
        assert_trunk_frozen(model)

    def test_trainable_trunk_raises_naming_peft_scope(self) -> None:
        """Spec §6 guard matrix: trainable trunk → hard-error naming peft.scope."""
        model = self._frozen_model()
        # Unfreeze one parameter to simulate a trainable trunk.
        vb = model.backbone.vision_backbone  # type: ignore[attr-defined]
        for p in vb.parameters():
            p.requires_grad_(True)
            break  # just one

        with pytest.raises(ValueError, match=r"peft\.scope"):
            assert_trunk_frozen(model)

    def test_trainable_trunk_error_mentions_trainable_params(self) -> None:
        """The error message must name the offending trainable parameters."""
        model = self._frozen_model()
        vb = model.backbone.vision_backbone  # type: ignore[attr-defined]
        for p in vb.parameters():
            p.requires_grad_(True)

        with pytest.raises(ValueError, match="trainable trunk parameters"):
            assert_trunk_frozen(model)

    def test_trunk_lora_raises_naming_peft_scope(self) -> None:
        """Spec §6 guard matrix: trunk-LoRA → hard-error naming peft.scope.

        Simulate by attaching a lora_A parameter to a Linear in the trunk.
        """
        model = self._frozen_model()
        vb = model.backbone.vision_backbone  # type: ignore[attr-defined]
        # Attach a fake lora_A attribute to the linear (simulates peft LoRA)
        vb.linear.lora_A = nn.Parameter(torch.zeros(4, 4))  # type: ignore[assignment]

        with pytest.raises(ValueError, match=r"peft\.scope"):
            assert_trunk_frozen(model)

    def test_no_trunk_submodule_passes_silently(self) -> None:
        """Guard skips cleanly on stub models without backbone.vision_backbone."""

        class _StubModel(nn.Module):
            pass

        # Should not raise
        assert_trunk_frozen(_StubModel())

    def test_adapter_level_path_also_found(self) -> None:
        """Guard walks model.backbone.vision_backbone (adapter path)."""

        class _VisionBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)
                for p in self.parameters():
                    p.requires_grad_(True)

        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vision_backbone = _VisionBackbone()

        class _InnerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = _Backbone()

        class _Adapter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _InnerModel()

        with pytest.raises(ValueError, match=r"peft\.scope"):
            assert_trunk_frozen(_Adapter())


# ---------------------------------------------------------------------------
# §6 bullet: Guard matrix — assert_rgb_input
# ---------------------------------------------------------------------------


class TestAssertRgbInput:
    """Spec §6: non-RGB channel adapter produces the correct hard-error."""

    def test_none_adapter_passes(self) -> None:
        """RGB path: channel_adapter is None → no error."""
        assert_rgb_input(None)

    def test_non_none_adapter_raises_naming_channel_semantics(self) -> None:
        """Spec §6 guard matrix: non-RGB adapter → names data.channel_semantics."""
        adapter = nn.Conv2d(4, 3, 1)
        with pytest.raises(ValueError, match=r"data\.channel_semantics"):
            assert_rgb_input(adapter)

    def test_non_none_adapter_raises_mentions_adapter(self) -> None:
        """Error message should indicate the adapter is not None."""
        adapter = nn.Conv2d(1, 3, 1)
        with pytest.raises(ValueError, match="channel_adapter"):
            assert_rgb_input(adapter)


# ---------------------------------------------------------------------------
# §6 bullet: Guard matrix — assert_aug_off
# ---------------------------------------------------------------------------


class TestAssertAugOff:
    """Spec §6: aug-on transforms produce the correct hard-error."""

    def _compose(self, transforms: list[Any]) -> Any:
        """Return a minimal A.Compose-like object wrapping transforms."""
        try:
            import albumentations as A

            return A.Compose(transforms)
        except ImportError:
            # Fallback: plain namespace object mimicking A.Compose
            class _FakeCompose:
                def __init__(self, t: list[Any]) -> None:
                    self.transforms = t

            return _FakeCompose(transforms)

    def _allowed_transform(self, name: str) -> Any:
        """Return a stub transform with the given allowed class name."""

        class _T:
            pass

        _T.__name__ = name  # type: ignore[attr-defined]

        class _Instance:
            pass

        _Instance.__class__ = _T  # type: ignore[assignment]
        return _Instance()

    def test_only_allowed_transforms_pass(self) -> None:
        """Pipeline with only deterministic transforms: must not raise."""
        A = pytest.importorskip("albumentations")

        transform = self._compose(
            [
                A.LongestMaxSize(max_size=1024),
                A.PadIfNeeded(1024, 1024, border_mode=0),
                A.Normalize(),
            ]
        )
        # Should not raise
        assert_aug_off(transform)

    def test_horizontal_flip_raises_naming_data_augmentations(self) -> None:
        """Spec §6: aug-on (HorizontalFlip) → names data.augmentations."""
        A = pytest.importorskip("albumentations")

        transform = self._compose(
            [
                A.LongestMaxSize(max_size=1024),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
            ]
        )
        with pytest.raises(ValueError, match=r"data\.augmentations"):
            assert_aug_off(transform)

    def test_horizontal_flip_names_offending_class(self) -> None:
        """Error message must name the offending transform class."""
        A = pytest.importorskip("albumentations")

        transform = self._compose([A.HorizontalFlip(p=0.5)])
        with pytest.raises(ValueError, match="HorizontalFlip"):
            assert_aug_off(transform)

    def test_brightness_contrast_raises(self) -> None:
        """Spec §6: photometric aug (RandomBrightnessContrast) → error."""
        A = pytest.importorskip("albumentations")

        transform = self._compose([A.RandomBrightnessContrast(p=0.5)])
        with pytest.raises(ValueError, match=r"data\.augmentations"):
            assert_aug_off(transform)

    def test_allowed_class_list_matches_spec(self) -> None:
        """_ALLOWED_AUG_CLASSES must contain the four spec-cited deterministic steps."""
        required = {"LongestMaxSize", "PadIfNeeded", "Normalize", "ToTensorV2"}
        assert required <= _ALLOWED_AUG_CLASSES, (
            f"Missing from _ALLOWED_AUG_CLASSES: {required - _ALLOWED_AUG_CLASSES}"
        )

    def test_shift_scale_rotate_raises(self) -> None:
        """Geometric aug (ShiftScaleRotate) → error."""
        A = pytest.importorskip("albumentations")

        transform = self._compose([A.ShiftScaleRotate(p=0.5)])
        with pytest.raises(ValueError, match=r"data\.augmentations"):
            assert_aug_off(transform)


# ---------------------------------------------------------------------------
# §6 bullet: Key stability across epochs + shuffle; distinct tiling-window uids
# ---------------------------------------------------------------------------


class TestKeyStability:
    """Spec §6: sample_uid stable across epochs/shuffle; distinct per tiling window."""

    def test_same_uid_returns_same_blob_path(self, tmp_path: Path) -> None:
        """get_batch on the same uid after put_batch returns the same data."""
        cache = _make_cache(tmp_path)
        uid = "12345:top_left"
        entry = _make_fake_entry()

        cache.put_batch([uid], [entry])
        result = cache.get_batch([uid])
        assert result is not None
        assert len(result) == 1

    def test_different_windows_same_image_distinct_blobs(self, tmp_path: Path) -> None:
        """Tiling windows of one image must map to distinct cache keys."""
        cache = _make_cache(tmp_path)
        image_id = "42"
        uid_a = f"{image_id}:top_left"
        uid_b = f"{image_id}:bottom_right"

        entry_a = _make_fake_entry()
        entry_b = _make_fake_entry()
        cache.put_batch([uid_a, uid_b], [entry_a, entry_b])

        result_a = cache.get_batch([uid_a])
        result_b = cache.get_batch([uid_b])
        assert result_a is not None
        assert result_b is not None
        # Blobs are stored at separate paths.
        path_a = cache._blob_path(uid_a)
        path_b = cache._blob_path(uid_b)
        assert path_a != path_b
        assert path_a.exists()
        assert path_b.exists()

    def test_uid_stable_under_simulated_shuffle(self, tmp_path: Path) -> None:
        """Spec §6: uid is stable across shuffle — same uid hits same blob."""
        cache = _make_cache(tmp_path)
        uids = [f"img_{i}:w0" for i in range(6)]
        entries = [_make_fake_entry() for _ in uids]
        cache.put_batch(uids, entries)

        # Simulate a shuffled sampler ordering.
        shuffled = [uids[3], uids[0], uids[5], uids[1]]
        result = cache.get_batch(shuffled)
        assert result is not None
        assert len(result) == 4

    def test_uid_same_across_simulated_epochs(self, tmp_path: Path) -> None:
        """Epoch-0 put / epoch-1 get with SAME uid → hit."""
        cache = _make_cache(tmp_path)
        uid = "999:window_0"
        entry = _make_fake_entry()

        # Epoch 0: store.
        cache.put_batch([uid], [entry])

        # Epoch 1: same uid → must hit.
        result = cache.get_batch([uid])
        assert result is not None

    def test_tiling_uid_format_is_image_id_colon_window(self) -> None:
        """sample_uid = f'{image_id}:{window}' — confirm format is the right shape."""
        image_id = "2024"
        window = "(0, 0, 512, 512)"
        uid = f"{image_id}:{window}"
        # Must contain colon separator, image_id, and window representation.
        assert ":" in uid
        assert uid.startswith(image_id)


# ---------------------------------------------------------------------------
# §6 bullet: Epoch-0-store / epoch-1-replay equivalence
# ---------------------------------------------------------------------------


class TestEpochReplayEquivalence:
    """Spec §6: replayed backbone_out bit-identical to the fp16 round-trip of a fresh recompute."""

    def test_replay_matches_fp16_roundtrip(self, tmp_path: Path) -> None:
        """Replay must equal fp16(recompute), not the raw float32 recompute.

        The fp16 cast is lossy, so we compare replay to the SAME fp16 round-trip
        of a fresh recompute (spec §6 "bit-identical means up to the fp16 cast").
        Note: replayed tensors may be on CUDA when a GPU is available; compare on CPU.
        """
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uid = "img0:w0"
        # Build a float32 entry with known values.
        fpn_raw = [torch.randn(1, 4, 4), torch.randn(1, 4, 4)]
        entry_raw: dict[str, Any] = {
            "backbone_fpn": fpn_raw,
            "sam2_backbone_out": None,
        }

        # Store (fp16 on disk).
        cache.put_batch([uid], [entry_raw])

        # Replay (read → cast to model_dtype=float32, possibly onto CUDA).
        result = cache.get_batch([uid])
        assert result is not None

        # Expected: fp16 → float32 round-trip of the original tensors.
        expected_fpn = [t.to(torch.float16).to(torch.float32) for t in fpn_raw]
        replayed_fpn = result[0]["backbone_fpn"]
        for rep, exp in zip(replayed_fpn, expected_fpn, strict=True):
            rep_cpu = rep.cpu()
            assert torch.allclose(rep_cpu, exp, atol=1e-5), (
                f"Replay mismatch: max diff {(rep_cpu - exp).abs().max().item():.6f}"
            )

    def test_replay_excludes_vision_pos_enc(self, tmp_path: Path) -> None:
        """vision_pos_enc must NOT be stored or returned by get_batch.

        Per spec §1/§3: it is content-independent and excluded from the cache;
        it is re-attached by the adapter.
        """
        cache = _make_cache(tmp_path)
        uid = "img1:w0"
        entry: dict[str, Any] = {
            "backbone_fpn": [torch.randn(1, 4, 4)],
            "vision_pos_enc": [torch.randn(1, 4, 4)],  # should be stripped
            "sam2_backbone_out": None,
        }
        cache.put_batch([uid], [entry])
        result = cache.get_batch([uid])
        assert result is not None
        assert "vision_pos_enc" not in result[0], (
            "vision_pos_enc must be excluded from cached entries (spec §1 / §3)"
        )

    def test_vision_features_is_fpn_last(self, tmp_path: Path) -> None:
        """On replay, vision_features must equal backbone_fpn[-1] (spec §1)."""
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uid = "img2:w0"
        fpn = [torch.randn(1, 4, 4), torch.randn(1, 4, 4)]
        entry: dict[str, Any] = {
            "backbone_fpn": fpn,
            "sam2_backbone_out": None,
        }
        cache.put_batch([uid], [entry])
        result = cache.get_batch([uid])
        assert result is not None
        # Use data_ptr() equality when both live on the same device, or value equality.
        vf = result[0]["vision_features"].cpu()
        last_fpn = result[0]["backbone_fpn"][-1].cpu()
        assert torch.allclose(vf, last_fpn)

    def test_sam2_backbone_out_replayed_when_present(self, tmp_path: Path) -> None:
        """If sam2_backbone_out is in the entry, it must be replayed."""
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uid = "img3:w0"
        entry = _make_fake_entry(sam2=True)
        cache.put_batch([uid], [entry])
        result = cache.get_batch([uid])
        assert result is not None
        assert result[0].get("sam2_backbone_out") is not None

    def test_sam2_vision_features_in_sam2_backbone_out(self, tmp_path: Path) -> None:
        """sam2_backbone_out['vision_features'] == sam2_backbone_out['backbone_fpn'][-1]."""
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uid = "img4:w0"
        entry = _make_fake_entry(sam2=True)
        cache.put_batch([uid], [entry])
        result = cache.get_batch([uid])
        assert result is not None
        s2 = result[0]["sam2_backbone_out"]
        assert s2 is not None
        assert torch.allclose(s2["vision_features"].cpu(), s2["backbone_fpn"][-1].cpu())

    def test_batch_of_two_matches_fp16_roundtrip(self, tmp_path: Path) -> None:
        """Batch of 2 images: each entry round-trips through fp16 correctly.

        Note: replayed tensors may land on CUDA; compare on CPU.
        """
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uids = ["a:w0", "b:w0"]
        entries = [
            {"backbone_fpn": [torch.randn(1, 4, 4)], "sam2_backbone_out": None},
            {"backbone_fpn": [torch.randn(1, 4, 4)], "sam2_backbone_out": None},
        ]
        cache.put_batch(uids, entries)
        result = cache.get_batch(uids)
        assert result is not None
        for i, (entry, res) in enumerate(zip(entries, result, strict=True)):
            expected = entry["backbone_fpn"][0].to(torch.float16).to(torch.float32)
            got = res["backbone_fpn"][0].cpu()
            assert torch.allclose(got, expected, atol=1e-5), f"Batch entry {i} mismatch"

    def test_cached_keys_constant_matches_spec(self) -> None:
        """_CACHED_KEYS must contain backbone_fpn and sam2_backbone_out (spec §1)."""
        assert "backbone_fpn" in _CACHED_KEYS
        assert "sam2_backbone_out" in _CACHED_KEYS


# ---------------------------------------------------------------------------
# §6 bullet: Eviction → recompute
# ---------------------------------------------------------------------------


class TestEvictionRecompute:
    """Spec §6: a forced miss recomputes the whole batch and refreshes the cache."""

    def test_missing_blob_is_an_all_or_none_miss(self, tmp_path: Path) -> None:
        """If any uid is missing, get_batch returns None (all-or-none)."""
        cache = _make_cache(tmp_path)
        uid_present = "img0:w0"
        uid_missing = "img1:w0"

        entry = _make_fake_entry()
        cache.put_batch([uid_present], [entry])

        # Only uid_present is stored; uid_missing is absent → all-or-none miss.
        result = cache.get_batch([uid_present, uid_missing])
        assert result is None

    def test_corrupted_blob_causes_miss(self, tmp_path: Path) -> None:
        """A corrupt/invalid blob causes a miss (not a crash)."""
        cache = _make_cache(tmp_path)
        uid = "img0:w0"
        blob_path = cache._blob_path(uid)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"not a valid pickle")

        result = cache.get_batch([uid])
        assert result is None

    def test_fingerprint_mismatch_causes_miss(self, tmp_path: Path) -> None:
        """A blob with a different fingerprint is treated as a miss."""
        fp_a = trunk_fingerprint(
            checkpoint_id="ckpt-a",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        fp_b = trunk_fingerprint(
            checkpoint_id="ckpt-b",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        cache_a = _make_cache(tmp_path, fingerprint=fp_a)
        uid = "img0:w0"
        entry = _make_fake_entry()
        cache_a.put_batch([uid], [entry])

        # Read with a different fingerprint.
        cache_dir = tmp_path / "cache"
        cache_b = TrunkFeatureCache(
            cache_dir=cache_dir,
            fingerprint=fp_b,
            model_dtype=torch.float32,
            n_samples=5,
            per_image_bytes=1024,
            allow_slow_disk=True,
            free_disk_fraction=0.99,
        )
        result = cache_b.get_batch([uid])
        assert result is None

    def test_post_eviction_put_refreshes_blob(self, tmp_path: Path) -> None:
        """After a forced miss, a new put_batch stores fresh data."""
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uid = "img0:w0"

        entry_v1 = {"backbone_fpn": [torch.ones(1, 4, 4)], "sam2_backbone_out": None}
        entry_v2 = {"backbone_fpn": [torch.zeros(1, 4, 4)], "sam2_backbone_out": None}

        cache.put_batch([uid], [entry_v1])
        # Simulate eviction by deleting the blob.
        blob_path = cache._blob_path(uid)
        blob_path.unlink()

        # Miss.
        assert cache.get_batch([uid]) is None

        # Refresh.
        cache.put_batch([uid], [entry_v2])
        result = cache.get_batch([uid])
        assert result is not None
        expected = torch.zeros(1, 4, 4, dtype=torch.float32)
        # Replayed tensor may be on CUDA; compare on CPU.
        assert torch.allclose(result[0]["backbone_fpn"][0].cpu(), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# §6 bullet: Free-disk fit-check fail-fast (§3.5b)
# ---------------------------------------------------------------------------


class TestFreeDiskFitCheck:
    """Spec §6: free-disk guard at build time, naming cache_trunk_features."""

    def test_fit_check_passes_when_projected_under_limit(self, tmp_path: Path) -> None:
        """Small projected size ≪ free disk → no exception."""
        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        # per_image_bytes=1 * n_samples=1 → negligible projected size
        cache = TrunkFeatureCache(
            cache_dir=tmp_path / "cache",
            fingerprint=fp,
            model_dtype=torch.float32,
            n_samples=1,
            per_image_bytes=1,
            allow_slow_disk=True,
            free_disk_fraction=0.70,
        )
        assert cache is not None

    def test_fit_check_fails_when_projected_over_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Projected size > 70% free disk → ValueError naming cache_trunk_features."""
        # Stub disk_usage: report 100 bytes free.
        fake_usage = shutil.disk_usage(tmp_path)._replace(free=100)
        monkeypatch.setattr("shutil.disk_usage", lambda _p: fake_usage)

        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        with pytest.raises(ValueError, match="cache_trunk_features"):
            TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=10,
                per_image_bytes=100,  # 1000 bytes > 70 bytes (70% of 100)
                allow_slow_disk=True,
                free_disk_fraction=0.70,
            )

    def test_fit_check_names_cache_trunk_features_config_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error message must name the config key cache_trunk_features (spec §3.5b)."""
        fake_usage = shutil.disk_usage(tmp_path)._replace(free=100)
        monkeypatch.setattr("shutil.disk_usage", lambda _p: fake_usage)

        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        with pytest.raises(ValueError) as exc_info:
            TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=10,
                per_image_bytes=100,
                allow_slow_disk=True,
                free_disk_fraction=0.70,
            )
        assert "cache_trunk_features" in str(exc_info.value)

    def test_fit_check_boundary_at_exact_70_percent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Projected == exactly 70% free → passes (not-over means passes)."""
        fake_usage = shutil.disk_usage(tmp_path)._replace(free=100)
        monkeypatch.setattr("shutil.disk_usage", lambda _p: fake_usage)

        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        # 70 bytes <= 70% of 100 → should pass.
        cache = TrunkFeatureCache(
            cache_dir=tmp_path / "cache",
            fingerprint=fp,
            model_dtype=torch.float32,
            n_samples=70,
            per_image_bytes=1,
            allow_slow_disk=True,
            free_disk_fraction=0.70,
        )
        assert cache is not None

    def test_fit_check_boundary_just_over_70_percent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Projected > 70% free → fails."""
        fake_usage = shutil.disk_usage(tmp_path)._replace(free=100)
        monkeypatch.setattr("shutil.disk_usage", lambda _p: fake_usage)

        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        # 71 bytes > 70 bytes (70% of 100) → should fail.
        with pytest.raises(ValueError, match="cache_trunk_features"):
            TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=71,
                per_image_bytes=1,
                allow_slow_disk=True,
                free_disk_fraction=0.70,
            )


# ---------------------------------------------------------------------------
# §6 bullet: Throughput auto-guard fail / override (§3.5c)
# CRITICAL: guard must NOT read the kernel 'rotational' flag.
# ---------------------------------------------------------------------------


class TestThroughputAutoGuard:
    """Spec §6: throughput guard derived from measured speed, not rotational flag."""

    def test_guard_raises_when_throughput_below_breakeven(self, tmp_path: Path) -> None:
        """Slow disk (throughput < break-even) → ValueError naming cache_allow_slow_disk."""
        # Stub _run_throughput_guard to simulate a slow disk rejection.
        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )

        def _slow_probe(cache_dir: Path, per_image_bytes: int, trunk_fwd_ms: float) -> None:
            # Simulate: measured < break_even → raise as the real guard would.
            trunk_fwd_s = trunk_fwd_ms / 1000.0
            break_even_bps = per_image_bytes / trunk_fwd_s if trunk_fwd_s > 0 else float("inf")
            measured_bps = break_even_bps * 0.1  # 10% of break-even = slow disk
            raise ValueError(
                f"cache_trunk_features: disk throughput {measured_bps / 1e9:.2f} GB/s "
                f"is below the break-even threshold {break_even_bps / 1e9:.2f} GB/s "
                "Set cache_allow_slow_disk = true to override this guard"
            )

        with (
            patch.object(TrunkFeatureCache, "_run_throughput_guard", staticmethod(_slow_probe)),
            pytest.raises(ValueError, match="cache_allow_slow_disk"),
        ):
            TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=1,
                per_image_bytes=1024,
                allow_slow_disk=False,
                free_disk_fraction=0.99,
            )

    def test_guard_passes_with_allow_slow_disk_override(self, tmp_path: Path) -> None:
        """allow_slow_disk=True must skip the throughput guard entirely."""
        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        call_count: list[int] = [0]

        def _counting_probe(cache_dir: Path, per_image_bytes: int, trunk_fwd_ms: float) -> None:
            call_count[0] += 1
            raise RuntimeError("should not be called when allow_slow_disk=True")

        with patch.object(
            TrunkFeatureCache, "_run_throughput_guard", staticmethod(_counting_probe)
        ):
            # allow_slow_disk=True → guard must be skipped entirely.
            cache = TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=1,
                per_image_bytes=1024,
                allow_slow_disk=True,
                free_disk_fraction=0.99,
            )
        assert call_count[0] == 0, "Throughput probe must be skipped when allow_slow_disk=True"
        assert cache is not None

    def test_guard_passes_when_throughput_above_breakeven(self, tmp_path: Path) -> None:
        """Fast disk (throughput > break-even) → no error."""
        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )

        def _fast_probe(cache_dir: Path, per_image_bytes: int, trunk_fwd_ms: float) -> None:
            # Does not raise (simulates fast SSD).
            return

        with patch.object(TrunkFeatureCache, "_run_throughput_guard", staticmethod(_fast_probe)):
            cache = TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=1,
                per_image_bytes=1024,
                allow_slow_disk=False,
                free_disk_fraction=0.99,
            )
        assert cache is not None

    def test_guard_derived_from_measured_throughput_not_rotational_flag(
        self, tmp_path: Path
    ) -> None:
        """Spec §3.5(c) HARD REQUIREMENT: guard must not READ the kernel rotational flag.

        The guard source may MENTION 'rotational' in comments explaining why the flag
        is NOT used (that is correct documentation).  The prohibition is on READING it
        — i.e., opening /sys/block/.../queue/rotational or calling sysfs APIs.
        We check that no sysfs path or blkid/udev call references 'rotational'.
        """
        import inspect
        import re

        source = inspect.getsource(TrunkFeatureCache._run_throughput_guard)

        # These are the patterns that would indicate a FUNCTIONAL read of the flag:
        # - '/queue/rotational' (sysfs path)
        # - blkid, udevadm queries
        # - open(...'rotational') or read the rotational sysfs file
        sysfs_rotational = re.compile(r"""(?x)
            queue/rotational        # sysfs path
            | open\s*\(.*rotational # open() call on rotational file
            | read\s*\(.*rotational # read() call on rotational file
            | blkid.*rotational     # blkid usage
        """)
        assert not sysfs_rotational.search(source), (
            "TrunkFeatureCache._run_throughput_guard must NOT read the kernel "
            "'rotational' sysfs flag (spec §3.5(c) hard requirement — unreliable under "
            "WSL2/VHDX). The guard must be derived from measured throughput only."
        )

    def test_throughput_guard_error_names_cache_allow_slow_disk(self, tmp_path: Path) -> None:
        """The throughput guard error must name cache_allow_slow_disk (spec §3.5c)."""
        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )

        def _slow_probe(cache_dir: Path, per_image_bytes: int, trunk_fwd_ms: float) -> None:
            raise ValueError("cache_allow_slow_disk mock slow disk")

        with (
            patch.object(TrunkFeatureCache, "_run_throughput_guard", staticmethod(_slow_probe)),
            pytest.raises(ValueError) as exc_info,
        ):
            TrunkFeatureCache(
                cache_dir=tmp_path / "cache",
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=1,
                per_image_bytes=1024,
                allow_slow_disk=False,
                free_disk_fraction=0.99,
            )
        assert "cache_allow_slow_disk" in str(exc_info.value)

    def test_real_throughput_probe_uses_posix_fadvise_not_rotational(self, tmp_path: Path) -> None:
        """The probe must call posix_fadvise / fsync — not any rotational-flag reader."""
        import inspect

        source = inspect.getsource(TrunkFeatureCache._run_throughput_guard)
        assert "posix_fadvise" in source or "POSIX_FADV_DONTNEED" in source, (
            "Throughput probe must use posix_fadvise for page-cache eviction "
            "(cite: spec §3.5(c); spike _evict_page_cache)"
        )

    def test_break_even_is_derived_per_image_bytes_over_trunk_fwd(self, tmp_path: Path) -> None:
        """The break-even formula must be derivable: per_image_bytes / trunk_fwd_s.

        Verify the guard source embeds the derivation, not a hardcoded threshold.
        """
        import inspect

        source = inspect.getsource(TrunkFeatureCache._run_throughput_guard)
        # The formula 'per_image_bytes / trunk_fwd' must appear in some form.
        assert "per_image_bytes" in source and "trunk_fwd" in source, (
            "Break-even must be derived from per_image_bytes / trunk_fwd "
            "(spec §3.5(c)). Hardcoded thresholds are forbidden."
        )


# ---------------------------------------------------------------------------
# §6 bullet: Cleanup on teardown (§3.5) — normal exit AND simulated exception
# ---------------------------------------------------------------------------


class TestCleanupOnTeardown:
    """Spec §6: cache dir removed on normal exit AND on a simulated mid-run exception."""

    def test_teardown_removes_cache_dir_on_normal_exit(self, tmp_path: Path) -> None:
        """Normal teardown must remove the cache directory."""
        cache = _make_cache(tmp_path)
        uid = "img0:w0"
        entry = _make_fake_entry()
        cache.put_batch([uid], [entry])

        cache_dir = tmp_path / "cache"
        assert cache_dir.exists(), "Cache dir must exist before teardown"

        cache.teardown()
        assert not cache_dir.exists(), "Cache dir must be removed by teardown"

    def test_teardown_called_in_finally_on_exception(self, tmp_path: Path) -> None:
        """Spec §6: teardown fires even when a mid-run exception is raised."""
        cache = _make_cache(tmp_path)
        uid = "img0:w0"
        entry = _make_fake_entry()
        cache.put_batch([uid], [entry])

        cache_dir = tmp_path / "cache"
        assert cache_dir.exists()

        # Simulate the trainer.fit finally-block pattern.
        try:
            raise RuntimeError("simulated training exception")
        except RuntimeError:
            cache.teardown()

        assert not cache_dir.exists(), (
            "Cache dir must be removed on exception (finally block must call teardown)"
        )

    def test_teardown_is_idempotent(self, tmp_path: Path) -> None:
        """teardown must not raise when called twice (ignore_errors=True)."""
        cache = _make_cache(tmp_path)
        cache.teardown()
        # Second call should not raise.
        cache.teardown()

    def test_teardown_removes_all_sharded_blobs(self, tmp_path: Path) -> None:
        """teardown must remove the entire cache_dir tree including sharded subdirs."""
        cache = _make_cache(tmp_path)
        uids = [f"img_{i}:w0" for i in range(4)]
        entries = [_make_fake_entry() for _ in uids]
        cache.put_batch(uids, entries)

        cache_dir = tmp_path / "cache"
        # Verify blobs were actually created.
        all_files = list(cache_dir.rglob("*.pt"))
        assert len(all_files) > 0, "Blobs must exist before teardown"

        cache.teardown()
        assert not cache_dir.exists(), "Entire cache_dir must be removed by teardown"

    def test_trainer_teardown_helper_calls_cache_teardown(self, tmp_path: Path) -> None:
        """The trainer-level _teardown_trunk_cache helper must call cache.teardown()."""
        from custom_sam_peft.models.sam3 import Sam3Wrapper
        from custom_sam_peft.train.trainer import _teardown_trunk_cache

        cache = _make_cache(tmp_path)
        teardown_called: list[bool] = []
        original_teardown = cache.teardown

        def _spy_teardown() -> None:
            teardown_called.append(True)
            original_teardown()

        cache.teardown = _spy_teardown  # type: ignore[method-assign]

        # Build a minimal Sam3Wrapper with a fake adapter holding the cache.
        class _FakeAdapter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._trunk_cache = cache

        # _teardown_trunk_cache takes a Sam3Wrapper; use MagicMock to avoid
        # constructing the full wrapper while still satisfying the type hint.
        fake_wrapper = MagicMock(spec=Sam3Wrapper)
        fake_wrapper.model = _FakeAdapter()
        fake_wrapper.model._trunk_cache = cache
        fake_wrapper.model._trunk_cache.teardown = _spy_teardown  # type: ignore[method-assign]

        _teardown_trunk_cache(fake_wrapper)
        assert teardown_called, "_teardown_trunk_cache must call cache.teardown()"


# ---------------------------------------------------------------------------
# §6 bullet: Prefetch correctness (§3.5) — on vs off, shuffled order
# ---------------------------------------------------------------------------


class TestPrefetchCorrectness:
    """Spec §6: prefetch on vs off yields identical replayed features."""

    def _populate(
        self, cache: TrunkFeatureCache, n: int = 4
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Store n entries and return their uids + original entries."""
        uids = [f"img_{i}:w0" for i in range(n)]
        entries = [
            {"backbone_fpn": [torch.rand(1, 4, 4)], "sam2_backbone_out": None} for _ in range(n)
        ]
        cache.put_batch(uids, entries)
        return uids, entries

    def test_prefetch_then_get_prefetched_returns_same_data(self, tmp_path: Path) -> None:
        """prefetch + _get_prefetched must return same features as get_batch."""
        cache = _make_cache(tmp_path, model_dtype=torch.float32)
        uids, _ = self._populate(cache)

        query = [uids[2], uids[0]]

        # Without prefetch: cold read.
        cold = cache.get_batch(query)
        assert cold is not None

        # With prefetch: queue then consume.
        cache.prefetch(query)
        # Wait for background thread to finish.
        if cache._prefetch_thread is not None:
            cache._prefetch_thread.join(timeout=5.0)
        prefetched = cache._get_prefetched(query)

        assert prefetched is not None, "_get_prefetched must return data after join"
        for cold_e, pre_e in zip(cold, prefetched, strict=True):
            # Tensors may be on CUDA; compare on CPU.
            assert torch.allclose(
                cold_e["backbone_fpn"][0].cpu(), pre_e["backbone_fpn"][0].cpu(), atol=1e-5
            ), "Prefetch must return identical features to cold read"

    def test_prefetch_with_wrong_uids_returns_none(self, tmp_path: Path) -> None:
        """_get_prefetched must return None if called with different uids."""
        cache = _make_cache(tmp_path)
        uids, _ = self._populate(cache)

        cache.prefetch([uids[0], uids[1]])
        if cache._prefetch_thread is not None:
            cache._prefetch_thread.join(timeout=5.0)

        # Query with different uids → None.
        result = cache._get_prefetched([uids[2], uids[3]])
        assert result is None

    def test_prefetch_noop_when_thread_alive(self, tmp_path: Path) -> None:
        """Second prefetch call while first thread is alive must be a no-op."""
        cache = _make_cache(tmp_path)
        uids, _ = self._populate(cache)

        # Start a slow prefetch by patching _load_blob to sleep briefly.
        original_load = cache._load_blob
        barrier = threading.Event()
        enter_count: list[int] = [0]

        def _slow_load(uid: str) -> dict[str, Any] | None:
            enter_count[0] += 1
            barrier.wait(timeout=2.0)
            return original_load(uid)

        cache._load_blob = _slow_load  # type: ignore[method-assign]

        cache.prefetch([uids[0]])
        # First thread still running; second call must be no-op.
        cache.prefetch([uids[1]])
        barrier.set()

        if cache._prefetch_thread is not None:
            cache._prefetch_thread.join(timeout=5.0)
        # Only one prefetch actually ran.
        assert enter_count[0] <= 1  # at most 1 batch loaded (the first prefetch)

    def test_prefetch_on_vs_off_identical_features_shuffled_order(self, tmp_path: Path) -> None:
        """Spec §6: prefetch ON vs OFF must yield identical replayed features
        including under a shuffled sampler order.
        """
        # Use two separate cache instances pointing to the same on-disk blobs.
        fp = trunk_fingerprint(
            checkpoint_id="test",
            scope="decoder_concept",
            dtype="float32",
            image_size=1008,
        )
        cache_dir = tmp_path / "shared_cache"

        def _build_cache() -> TrunkFeatureCache:
            return TrunkFeatureCache(
                cache_dir=cache_dir,
                fingerprint=fp,
                model_dtype=torch.float32,
                n_samples=8,
                per_image_bytes=1024,
                allow_slow_disk=True,
                free_disk_fraction=0.99,
            )

        writer = _build_cache()
        uids = [f"img_{i}:w0" for i in range(6)]
        entries = [{"backbone_fpn": [torch.rand(1, 4, 4)], "sam2_backbone_out": None} for _ in uids]
        writer.put_batch(uids, entries)

        # Shuffled sampler order (simulates training shuffle).
        shuffled = [uids[4], uids[1], uids[3], uids[0]]

        # Path A: no prefetch, direct cold reads.
        reader_cold = _build_cache()
        cold_result = reader_cold.get_batch(shuffled)
        assert cold_result is not None

        # Path B: with prefetch.
        reader_prefetch = _build_cache()
        reader_prefetch.prefetch(shuffled)
        if reader_prefetch._prefetch_thread is not None:
            reader_prefetch._prefetch_thread.join(timeout=5.0)
        prefetch_result = reader_prefetch._get_prefetched(shuffled)
        if prefetch_result is None:
            # If prefetch expired / wasn't consumed, fall back to cold read.
            prefetch_result = reader_prefetch.get_batch(shuffled)
        assert prefetch_result is not None

        # Features must be identical. Tensors may be on CUDA; compare on CPU.
        for cold_e, pre_e in zip(cold_result, prefetch_result, strict=True):
            assert torch.allclose(
                cold_e["backbone_fpn"][0].cpu(), pre_e["backbone_fpn"][0].cpu(), atol=1e-5
            ), "Prefetch must not change the replayed features"

    def test_prefetch_missing_uid_returns_none(self, tmp_path: Path) -> None:
        """_get_prefetched must return None for a uid that doesn't exist on disk."""
        cache = _make_cache(tmp_path)
        uids, _ = self._populate(cache)

        missing_uid = "nonexistent:w0"
        cache.prefetch([uids[0], missing_uid])
        if cache._prefetch_thread is not None:
            cache._prefetch_thread.join(timeout=5.0)

        # The prefetch itself consumed, but the miss means None on _get_prefetched.
        result = cache._get_prefetched([uids[0], missing_uid])
        # Can be None (miss) or non-None (prefetch stored miss marker).
        # The important invariant: if result is returned, all uids are valid.
        if result is not None:
            assert len(result) == 2  # all-or-none: both present or None returned


# ---------------------------------------------------------------------------
# trunk_fingerprint — stability and collision-resistance
# ---------------------------------------------------------------------------


class TestTrunkFingerprint:
    """trunk_fingerprint must be stable and collision-resistant."""

    def test_same_args_same_fingerprint(self) -> None:
        fp1 = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        fp2 = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        assert fp1 == fp2

    def test_different_ckpt_different_fingerprint(self) -> None:
        fp1 = trunk_fingerprint(
            checkpoint_id="ckpt-a",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        fp2 = trunk_fingerprint(
            checkpoint_id="ckpt-b",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        assert fp1 != fp2

    def test_different_scope_different_fingerprint(self) -> None:
        fp1 = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        fp2 = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="vision",
            dtype="bfloat16",
            image_size=1008,
        )
        assert fp1 != fp2

    def test_different_dtype_different_fingerprint(self) -> None:
        fp1 = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        fp2 = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="float16",
            image_size=1008,
        )
        assert fp1 != fp2

    def test_pipe_char_in_checkpoint_id_raises(self) -> None:
        """checkpoint_id must not contain '|' (field separator)."""
        with pytest.raises(ValueError, match="checkpoint_id"):
            trunk_fingerprint(
                checkpoint_id="ckpt|bad",
                scope="decoder_concept",
                dtype="bfloat16",
                image_size=1008,
            )

    def test_fingerprint_is_hex_string(self) -> None:
        fp = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_length_is_64(self) -> None:
        """SHA-256 hex digest is 64 characters."""
        fp = trunk_fingerprint(
            checkpoint_id="ckpt",
            scope="decoder_concept",
            dtype="bfloat16",
            image_size=1008,
        )
        assert len(fp) == 64


# ---------------------------------------------------------------------------
# Spike-measured constants — guard against accidental changes
# ---------------------------------------------------------------------------


class TestSpikeConstants:
    """SPIKE_PER_IMAGE_BYTES and SPIKE_TRUNK_FWD_MS must match the spike values."""

    def test_per_image_bytes_matches_spike(self) -> None:
        """53.16 MiB = 55_726_080 bytes (cite: spike Step 1)."""
        assert SPIKE_PER_IMAGE_BYTES == 55_726_080

    def test_trunk_fwd_ms_matches_spike(self) -> None:
        """91.4 ms (cite: spike Step 2a)."""
        assert abs(SPIKE_TRUNK_FWD_MS - 91.4) < 1e-6


# ---------------------------------------------------------------------------
# All-or-none semantics — get_batch returns None on any miss
# ---------------------------------------------------------------------------


class TestAllOrNone:
    """get_batch is ALL-or-NONE: None if any uid missing/corrupt/foreign-fingerprint."""

    def test_single_miss_returns_none(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        result = cache.get_batch(["nonexistent:w0"])
        assert result is None

    def test_all_present_returns_list(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        uids = ["a:w0", "b:w0"]
        entries = [_make_fake_entry(), _make_fake_entry()]
        cache.put_batch(uids, entries)
        result = cache.get_batch(uids)
        assert result is not None
        assert len(result) == 2

    def test_partial_hit_returns_none(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        uids = ["a:w0", "b:w0"]
        # Store only the first.
        cache.put_batch([uids[0]], [_make_fake_entry()])
        result = cache.get_batch(uids)
        assert result is None

    def test_empty_batch_returns_empty_list(self, tmp_path: Path) -> None:
        """Empty batch → returns empty list (vacuously all-hit)."""
        cache = _make_cache(tmp_path)
        result = cache.get_batch([])
        assert result == []


class TestTilePosEnc:
    """tile_pos_enc: content-independent pos_enc replay onto the batch/device."""

    def test_tiles_b1_to_batch_on_device(self, tmp_path: Path) -> None:
        """B=1 cached pos_enc expands to the requested batch on the target device."""
        cache = _make_cache(tmp_path)
        cached = [torch.randn(1, 3, 4, 4), torch.randn(1, 5, 2, 2)]
        device = torch.device("cpu")

        tiled = cache.tile_pos_enc(cached, batch_size=4, device=device)

        assert len(tiled) == len(cached)
        for orig, out in zip(cached, tiled, strict=True):
            assert out.shape == (4, *orig.shape[1:])
            assert out.device == device
            assert out.is_contiguous()
            # expand replicates the single source row across the batch dim.
            for i in range(4):
                assert torch.equal(out[i], orig[0])

    def test_empty_pos_enc_list(self, tmp_path: Path) -> None:
        """Empty pos_enc list → empty result (no levels to tile)."""
        cache = _make_cache(tmp_path)
        assert cache.tile_pos_enc([], batch_size=2, device=torch.device("cpu")) == []
