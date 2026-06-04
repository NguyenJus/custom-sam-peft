"""Trunk feature cache for frozen ViT replay across training epochs.

Spec: docs/superpowers/specs/2026-06-04-trunk-feature-cache-300-design.md

Single-run, on-disk (SSD) residence: torch.save / fp16 per sample_uid, stored
under <output_dir>/.trunk_cache/. Deleted at teardown. Never RAM-resident,
never cross-run persistent.

Public API
----------
trunk_fingerprint(*, checkpoint_id, scope, dtype, image_size) -> str
assert_trunk_frozen(model) -> None
assert_rgb_input(channel_adapter) -> None
assert_aug_off(train_transform) -> None
TrunkFeatureCache(cache_dir, fingerprint, ...)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spike-measured constants (cite: docs/spikes/2026-06-04-trunk-feature-cache-disk-respike.md)
# ---------------------------------------------------------------------------

# Per-image fp16 bytes measured on real SAM 3.1 B=1 bf16 run.
# cite: spike Step 1 — 53.16 MiB/image (backbone_fpn only, no sam2_backbone_out)
SPIKE_PER_IMAGE_BYTES: int = 55_726_080  # 53.16 MiB in bytes

# Trunk forward wall-clock measured on RTX 5070 Ti.
# cite: spike Step 2a — trunk_fwd mean 91.4 ms
SPIKE_TRUNK_FWD_MS: float = 91.4

# Keys in the forward_image output dict that are content-dependent and cached.
# vision_pos_enc is content-INDEPENDENT (spatial grid only) and is excluded.
# cite: spec §1 / §3 "Stored value"
_CACHED_KEYS: frozenset[str] = frozenset({"backbone_fpn", "sam2_backbone_out"})

# Allowlist of Albumentations class names that are deterministic / do not
# affect trunk-input pixel values. Any transform NOT in this set causes
# assert_aug_off to raise.
# cite: spec §2 guard 3 — "LongestMaxSize+PadIfNeeded+Normalize+ToTensorV2
#   are the only allowed deterministic steps"
_ALLOWED_AUG_CLASSES: frozenset[str] = frozenset(
    {
        "LongestMaxSize",
        "PadIfNeeded",
        "Normalize",
        "ToTensorV2",
        "Compose",
        "BboxParams",
    }
)


# ---------------------------------------------------------------------------
# trunk_fingerprint
# ---------------------------------------------------------------------------


def trunk_fingerprint(
    *,
    checkpoint_id: str,
    scope: str,
    dtype: str,
    image_size: int,
) -> str:
    """Return a stable hex digest identifying the trunk configuration.

    Written into every on-disk blob and checked on every read to prevent a
    stale cache from being replayed against a different trunk.

    The fingerprint is a SHA-256 hex digest of the canonical string
    ``"<checkpoint_id>|<scope>|<dtype>|<image_size>"`` — deterministic,
    collision-resistant, and compact (~64 chars).

    Parameters
    ----------
    checkpoint_id:
        Identifies the model checkpoint, e.g. ``"facebook/sam3.1"`` or a local
        path stem.  Must not contain ``|`` (asserted).
    scope:
        LoRA scope literal, e.g. ``"decoder_concept"``.
    dtype:
        Model parameter dtype string, e.g. ``"bfloat16"``.
    image_size:
        Fixed trunk input size (e.g. 1008 for SAM 3.1).
    """
    for label, value in [
        ("checkpoint_id", checkpoint_id),
        ("scope", scope),
        ("dtype", dtype),
    ]:
        if "|" in value:
            raise ValueError(f"trunk_fingerprint: {label}={value!r} must not contain '|'")
    canonical = f"{checkpoint_id}|{scope}|{dtype}|{image_size}"
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Correctness guards (spec §2)
# ---------------------------------------------------------------------------


def assert_trunk_frozen(model: nn.Module) -> None:
    """Raise ValueError if any trunk parameter is trainable or a trunk LoRA exists.

    Guard 1 (spec §2): zero requires_grad params under the trunk AND no LoRA
    module attached to the trunk. Checks the ``backbone.vision_backbone`` subtree
    (SAM 3.1 naming). If the trunk submodule is absent, passes silently (stub
    models in tests may lack it — integration builds the real wrapper).

    Raises
    ------
    ValueError
        Names the trainable parameters / LoRA modules found and the config key
        ``peft.scope`` to change.
    """
    trunk: nn.Module | None = None
    # Walk the module to find the trunk — supports both the real model and
    # _Sam3ImageAdapter (which stores the inner model as .model).
    for attr in ("backbone", "model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            trunk_candidate = getattr(sub, "vision_backbone", None)
            if trunk_candidate is not None:
                trunk = trunk_candidate
                break
            # One more level: adapter.model.backbone.vision_backbone
            backbone = getattr(sub, "backbone", None)
            if backbone is not None:
                trunk_candidate = getattr(backbone, "vision_backbone", None)
                if trunk_candidate is not None:
                    trunk = trunk_candidate
                    break

    if trunk is None:
        # Stub / non-standard model — skip structural check
        return

    trainable: list[str] = [name for name, p in trunk.named_parameters() if p.requires_grad]
    lora_modules: list[str] = [
        name
        for name, m in trunk.named_modules()
        if type(m).__name__ in ("Linear", "MultiheadAttention")
        and any("lora_" in cname for cname, _ in m.named_parameters(recurse=False))
    ]

    errors: list[str] = []
    if trainable:
        errors.append(
            f"trainable trunk parameters: {trainable[:5]}"
            + (" (truncated)" if len(trainable) > 5 else "")
        )
    if lora_modules:
        errors.append(f"LoRA modules on trunk: {lora_modules}")

    if errors:
        raise ValueError(
            "cache_trunk_features requires a fully-frozen trunk (spec §2 guard 1). "
            + " | ".join(errors)
            + " — change peft.scope to 'decoder_concept' (or another non-trunk scope)"
        )


def assert_rgb_input(channel_adapter: nn.Module | None) -> None:
    """Raise ValueError if channel_adapter is not None.

    Guard 2 (spec §2): the trunk input must be RGB (channel_adapter is None).
    A trainable channel adapter drifts the trunk input every step, so caching
    is invalid.

    Raises
    ------
    ValueError
        Names ``data.channel_semantics`` as the config key to change.
    """
    if channel_adapter is not None:
        raise ValueError(
            "cache_trunk_features requires RGB input (spec §2 guard 2): "
            "channel_adapter is not None, meaning a trainable N->3 Conv2d is "
            "applied upstream of the trunk and drifts its input every step. "
            "Set data.channel_semantics = 'rgb' to disable the adapter."
        )


def assert_aug_off(train_transform: Any) -> None:
    """Raise ValueError if the built train transform contains stochastic augmentations.

    Guard 3 (spec §2): no geometric / photometric / jitter / resize-jitter steps.
    Inspects the BUILT A.Compose object; rejects any transform whose class name is
    not in ``_ALLOWED_AUG_CLASSES``. LongestMaxSize, PadIfNeeded, Normalize, and
    ToTensorV2 are the only allowed deterministic steps.

    Parameters
    ----------
    train_transform:
        The built ``albumentations.Compose`` object returned by
        ``build_train_transforms``.

    Raises
    ------
    ValueError
        Names the offending transform class and ``data.augmentations`` as the
        config key to change.
    """

    # Walk the transform tree collecting all leaf transform class names.
    def _collect(t: Any) -> list[str]:
        """Recursively collect class names from an albumentations pipeline."""
        names: list[str] = []
        cls_name = type(t).__name__
        # Compose / Sequential hold .transforms list
        transforms_list = getattr(t, "transforms", None)
        if transforms_list is not None:
            for child in transforms_list:
                names.extend(_collect(child))
        else:
            names.append(cls_name)
        return names

    all_names = _collect(train_transform)
    bad: list[str] = [n for n in all_names if n not in _ALLOWED_AUG_CLASSES]
    if bad:
        raise ValueError(
            "cache_trunk_features requires augmentation to be disabled (spec §2 guard 3): "
            f"found stochastic/non-deterministic transforms {bad}. "
            "Set all knobs in data.augmentations to 0 / False to silence this guard."
        )


# ---------------------------------------------------------------------------
# TrunkFeatureCache
# ---------------------------------------------------------------------------


class TrunkFeatureCache:
    """On-disk, per-sample-uid, single-run trunk feature cache.

    Lifecycle
    ---------
    1. Build-time (``__init__``): run free-disk fit-check (b) and throughput
       auto-guard (c).
    2. Epoch 0 (all-miss): ``put_batch`` stores fp16 features per uid.
    3. Epochs 1+ (all-hit): ``get_batch`` cold-reads, pins, and non-blocking
       H2D copies the cached entries.
    4. ``prefetch`` queues depth-1 background reads.
    5. ``teardown`` removes the cache directory.

    Parameters
    ----------
    cache_dir:
        Directory under which per-uid blobs are written (created if absent).
    fingerprint:
        Output of ``trunk_fingerprint``. Written into every blob; mismatches
        cause a miss (recompute).
    model_dtype:
        ``torch.dtype`` for the H2D cast on replay.
    n_samples:
        Total number of distinct sample_uids (post-tiling, from ``len(train_ds)``).
    per_image_bytes:
        Expected on-disk bytes per cached image. Used for the fit-check and
        throughput probe break-even. Default: ``SPIKE_PER_IMAGE_BYTES``
        (53.16 MiB; cite: spike Step 1).
    trunk_fwd_ms:
        Measured trunk forward time in ms. Used to derive the break-even
        throughput. Default: ``SPIKE_TRUNK_FWD_MS`` (91.4 ms; cite: spike Step 2a).
    free_disk_fraction:
        Refuse if projected cache > this fraction of free disk.
        Default 0.70 (cite: spec §3.5(b)).
    allow_slow_disk:
        Skip the throughput auto-guard failure.
        Default False (cite: spec §3.5(c)).
    prefetch_depth:
        Number of steps to prefetch ahead. Default 1 (cite: spec §3.5 prefetch).
    """

    def __init__(
        self,
        cache_dir: Path,
        fingerprint: str,
        *,
        model_dtype: torch.dtype,
        n_samples: int,
        per_image_bytes: int = SPIKE_PER_IMAGE_BYTES,
        trunk_fwd_ms: float = SPIKE_TRUNK_FWD_MS,
        free_disk_fraction: float = 0.70,  # cite: spec §3.5(b)
        allow_slow_disk: bool = False,  # cite: spec §3.5(c)
        prefetch_depth: int = 1,  # cite: spec §3.5 prefetch depth
    ) -> None:
        self._cache_dir = cache_dir
        self._fingerprint = fingerprint
        self._model_dtype = model_dtype
        self._per_image_bytes = per_image_bytes
        self._trunk_fwd_ms = trunk_fwd_ms
        self._free_disk_fraction = free_disk_fraction
        self._allow_slow_disk = allow_slow_disk
        self._prefetch_depth = prefetch_depth

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Guard (b): free-disk fit-check
        projected_bytes = n_samples * per_image_bytes
        disk_info = shutil.disk_usage(cache_dir)
        free_bytes = disk_info.free
        limit_bytes = free_disk_fraction * free_bytes
        if projected_bytes > limit_bytes:
            projected_gib = projected_bytes / (1024**3)
            free_gib = free_bytes / (1024**3)
            limit_gib = limit_bytes / (1024**3)
            raise ValueError(
                f"cache_trunk_features: projected cache size {projected_gib:.2f} GiB "
                f"({n_samples} samples x {per_image_bytes / (1024**2):.1f} MiB) "
                f"exceeds {free_disk_fraction:.0%} of free disk "
                f"({limit_gib:.2f} GiB of {free_gib:.2f} GiB free). "
                "Disable caching (cache_trunk_features = false), free disk space, or "
                "increase cache_free_disk_fraction."
            )

        # Guard (c): throughput auto-guard — probe cold read, compare to break-even.
        # MUST be derived from measured throughput, NEVER from the kernel rotational flag.
        # cite: spec §3.5(c); spike Step 2d derivation:
        #   break_even_bps = per_image_bytes / (trunk_fwd_ms / 1000)
        if not allow_slow_disk:
            self._run_throughput_guard(cache_dir, per_image_bytes, trunk_fwd_ms)

        # Prefetch state
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: dict[str, dict[str, Any] | None] | None = None
        self._prefetch_uids: list[str] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blob_path(self, uid: str) -> Path:
        """Return the blob path for a sample_uid, using a two-level directory
        shard (first 2 hex chars of sha256 of uid) to avoid huge flat dirs.
        """
        shard = hashlib.sha256(uid.encode()).hexdigest()[:2]
        return self._cache_dir / shard / f"{uid}.pt"

    @staticmethod
    def _run_throughput_guard(
        cache_dir: Path,
        per_image_bytes: int,
        trunk_fwd_ms: float,
    ) -> None:
        """Write a probe blob, evict its page cache, cold-read it, measure throughput.

        The break-even is derived: break_even_bps = per_image_bytes / trunk_fwd_s.
        If measured throughput < break_even_bps, raise ValueError naming
        cache_allow_slow_disk.

        Page-cache eviction via posix_fadvise(DONTNEED) — no root required.
        NEVER reads the kernel rotational flag (cite: spec §3.5(c)).
        """
        probe_path = cache_dir / ".throughput_probe.pt"

        # Write a probe tensor of the expected size.
        probe_tensor = torch.zeros(per_image_bytes // 2, dtype=torch.float16)
        torch.save({"probe": probe_tensor, "fingerprint": "probe"}, probe_path)
        del probe_tensor

        try:
            # fsync + DONTNEED to evict page cache (cite: spike _evict_page_cache)
            fd = os.open(str(probe_path), os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

            # Cold read timing
            t0 = time.perf_counter()
            _ = torch.load(probe_path, map_location="cpu", weights_only=True)
            elapsed_s = time.perf_counter() - t0

            # Derive break-even: feature_bytes / trunk_fwd_s
            # cite: spec §3.5(c); spike derivation: 53.16 MiB / 91.4 ms ≈ 0.57 GB/s
            trunk_fwd_s = trunk_fwd_ms / 1000.0
            break_even_bps = per_image_bytes / trunk_fwd_s if trunk_fwd_s > 0 else float("inf")

            actual_size = probe_path.stat().st_size
            measured_bps = actual_size / elapsed_s if elapsed_s > 0 else float("inf")

            _LOG.debug(
                "Throughput probe: %.2f GB/s measured, %.2f GB/s break-even",
                measured_bps / 1e9,
                break_even_bps / 1e9,
            )

            if measured_bps < break_even_bps:
                raise ValueError(
                    f"cache_trunk_features: disk throughput {measured_bps / 1e9:.2f} GB/s "
                    f"is below the break-even threshold "
                    f"{break_even_bps / 1e9:.2f} GB/s "
                    f"(= {per_image_bytes / (1024**2):.1f} MiB / {trunk_fwd_ms:.1f} ms). "
                    "A cold read costs more than recomputing the trunk. "
                    "Set cache_allow_slow_disk = true to override this guard, or "
                    "disable caching (cache_trunk_features = false)."
                )
        finally:
            probe_path.unlink(missing_ok=True)

    def _load_blob(self, uid: str) -> dict[str, Any] | None:
        """Load and validate a single cached blob. Returns None on any failure."""
        path = self._blob_path(uid)
        if not path.exists():
            return None
        try:
            data: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(data, dict):
                return None
            if data.get("fingerprint") != self._fingerprint:
                _LOG.debug("Fingerprint mismatch for uid %r — cache miss", uid)
                return None
            entry = data.get("entry")
            if not isinstance(entry, dict):
                return None
            return entry
        except Exception:
            _LOG.debug("Failed to load blob for uid %r — cache miss", uid, exc_info=True)
            return None

    def _h2d_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Pin CPU tensors and copy non-blocking to model_dtype on the appropriate device.

        Mirrors the pinned-copy H2D tail from #288 / transfer_binarize.
        If CUDA is unavailable (CPU-only tests), returns tensors on CPU in model_dtype.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def _move(t: torch.Tensor) -> torch.Tensor:
            if device.type == "cuda":
                pinned = t.pin_memory()
                return pinned.to(device=device, dtype=self._model_dtype, non_blocking=True)
            return t.to(dtype=self._model_dtype)

        def _process_fpn(fpn: list[torch.Tensor]) -> list[torch.Tensor]:
            return [_move(t) for t in fpn]

        out: dict[str, Any] = {}
        out["backbone_fpn"] = _process_fpn(entry["backbone_fpn"])
        out["vision_features"] = out["backbone_fpn"][-1]

        sam2 = entry.get("sam2_backbone_out")
        if sam2 is not None:
            sam2_fpn = _process_fpn(sam2["backbone_fpn"])
            out["sam2_backbone_out"] = {
                "backbone_fpn": sam2_fpn,
                "vision_features": sam2_fpn[-1],
            }
        else:
            out["sam2_backbone_out"] = None

        if device.type == "cuda":
            torch.cuda.synchronize()

        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(self, sample_uids: list[str]) -> list[dict[str, Any]] | None:
        """ALL-or-NONE read: return entries for all uids or None on any miss.

        On a hit, each entry is cold-read from disk, tensors are pinned, and a
        non-blocking H2D copy is issued (cite: spec §3.5 replay path).

        Returns
        -------
        list[dict] of length len(sample_uids) with each entry containing
        ``backbone_fpn``, ``vision_features``, and optionally
        ``sam2_backbone_out``; cast to ``model_dtype`` on the target device.
        None if any uid is missing, corrupt, or carries a foreign fingerprint.
        """
        entries: list[dict[str, Any]] = []
        for uid in sample_uids:
            entry = self._load_blob(uid)
            if entry is None:
                return None  # all-or-none
            entries.append(self._h2d_entry(entry))
        return entries

    def put_batch(
        self,
        sample_uids: list[str],
        entries: list[dict[str, Any]],
    ) -> None:
        """Store fp16 blobs for each sample_uid in entries.

        Entries must correspond to forward_image output dicts MINUS
        vision_pos_enc. Each entry is detached, batch-unbound (slice [i:i+1]),
        cast to fp16, and saved with the fingerprint header.

        Parameters
        ----------
        sample_uids:
            Stable per-sample identifiers (one per image in the batch).
        entries:
            List of per-image dicts as produced by unbound forward_image output.
            Each dict should contain ``backbone_fpn`` (list of tensors) and
            optionally ``sam2_backbone_out`` (dict with ``backbone_fpn`` list).
            The ``vision_features`` key (== backbone_fpn[-1]) is NOT stored
            separately — it is reconstructed from fpn[-1] on replay.
            ``vision_pos_enc`` must have been excluded by the caller.
        """
        for uid, entry in zip(sample_uids, entries, strict=True):
            blob_path = self._blob_path(uid)
            blob_path.parent.mkdir(parents=True, exist_ok=True)

            def _fp16(t: torch.Tensor) -> torch.Tensor:
                return t.detach().to(dtype=torch.float16, device="cpu").contiguous()

            stored_entry: dict[str, Any] = {
                "backbone_fpn": [_fp16(t) for t in entry["backbone_fpn"]],
            }

            sam2 = entry.get("sam2_backbone_out")
            if sam2 is not None:
                stored_entry["sam2_backbone_out"] = {
                    "backbone_fpn": [_fp16(t) for t in sam2["backbone_fpn"]],
                }
            else:
                stored_entry["sam2_backbone_out"] = None

            # Spec §1: only content-DEPENDENT keys may be persisted. The
            # content-INDEPENDENT vision_pos_enc is cached once on the wrapper
            # and must never reach disk (it would break the replay invariant).
            # _CACHED_KEYS is the single source of truth for that contract.
            extra = stored_entry.keys() - _CACHED_KEYS
            if extra:
                raise ValueError(
                    "trunk cache refuses to persist non-cacheable keys "
                    f"{sorted(extra)} (allowed: {sorted(_CACHED_KEYS)})"
                )

            blob = {"fingerprint": self._fingerprint, "entry": stored_entry}
            torch.save(blob, blob_path)

    def tile_pos_enc(
        self,
        cached_pos_enc: list[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Tile the once-cached, content-independent vision_pos_enc onto the batch.

        vision_pos_enc is the spatial grid only (content-INDEPENDENT), so it is
        computed once on epoch 0 and held on CPU on the wrapper to keep VRAM
        free. On replay it is expanded from B=1 to ``batch_size`` and moved onto
        the model ``device`` — forward_grounding indexes it with on-device
        img_ids, so it MUST land on the same device or the index op raises a
        cross-device error (caught only on a real GPU; CPU stubs put everything
        on cpu). ``.contiguous()`` gives a fresh stride layout matching a real
        forward_image, avoiding stride-0 aliasing under any downstream in-place op.

        This device move lives here (not in sam3.py) because it is part of the
        cache's H2D replay path; the §9.2 static guard allowlists trunk_cache.py
        for exactly these moves. cite: spec §1 vision_pos_enc handling.
        """
        return [p.expand(batch_size, *p.shape[1:]).contiguous().to(device) for p in cached_pos_enc]

    def prefetch(self, sample_uids: list[str]) -> None:
        """Depth-1 background prefetch: read next step's blobs in a daemon thread.

        The background thread loads all blobs for ``sample_uids`` and stores
        results in ``self._prefetch_result``. A subsequent ``get_batch`` call
        with the same uids picks up the prefetched data if ready.

        If a prior prefetch thread is still running, this call is a no-op
        (avoids queuing more than depth=1; cite: spec §3.5 prefetch depth=1).
        """
        with self._prefetch_lock:
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return  # prior step's prefetch still in flight
            self._prefetch_uids = list(sample_uids)
            self._prefetch_result = None

            def _worker() -> None:
                result: dict[str, dict[str, Any] | None] = {}
                for uid in sample_uids:
                    result[uid] = self._load_blob(uid)
                with self._prefetch_lock:
                    self._prefetch_result = result

            t = threading.Thread(target=_worker, daemon=True, name="trunk-cache-prefetch")
            self._prefetch_thread = t
        t.start()

    def _get_prefetched(self, sample_uids: list[str]) -> list[dict[str, Any]] | None:
        """Non-blocking probe: return prefetched entries if all ready, else None.

        If the prefetch thread finished and produced entries for exactly these
        uids, consume the result and return H2D-copied entries. Otherwise None
        (caller should cold-read normally).
        """
        with self._prefetch_lock:
            if self._prefetch_result is None:
                return None
            if self._prefetch_uids != list(sample_uids):
                return None
            result = self._prefetch_result
            self._prefetch_result = None
            self._prefetch_uids = None

        entries: list[dict[str, Any]] = []
        for uid in sample_uids:
            raw = result.get(uid)
            if raw is None:
                return None
            entries.append(self._h2d_entry(raw))
        return entries

    def teardown(self) -> None:
        """Remove the cache directory unconditionally (cite: spec §3.5 cleanup).

        Called in the trainer's ``finally`` block so an aborted or crashed run
        does not leak ~193 GiB of on-disk state.
        """
        shutil.rmtree(self._cache_dir, ignore_errors=True)
        _LOG.debug("TrunkFeatureCache: removed %s", self._cache_dir)
