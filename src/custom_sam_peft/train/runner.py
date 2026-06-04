"""End-to-end train pipeline. CLI is a thin wrapper over `run_training`."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from custom_sam_peft._registry import lookup
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.split_source import (
    _log_split_source,
    resolve_split_source,
    save_split_source,
)
from custom_sam_peft.data.subset import SubsetDataset, resolve_subset_indices
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.models.sam3 import load_sam31
from custom_sam_peft.tracking import build_tracker
from custom_sam_peft.train.trainer import Trainer

_LOG = logging.getLogger(__name__)


def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Returns the created path."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _apply_limit(inner: Dataset, cfg: TrainConfig, pipeline: str) -> Dataset:
    """Wrap `inner` in a `SubsetDataset` per `cfg.data.limit`, or return as-is."""
    lim_cfg = cfg.data.limit
    limit_val = lim_cfg.train if pipeline == "train" else lim_cfg.val
    if limit_val is None:
        return inner

    labels = None
    if lim_cfg.strategy == "stratified":
        labels = getattr(inner, "image_class_labels", None)

    indices = resolve_subset_indices(
        len(inner),
        limit_val,
        seed=lim_cfg.seed,
        strategy=lim_cfg.strategy,
        image_class_labels=labels,
    )
    _LOG.info(
        "data.limit applied: %s=%d/%d (strategy=%s, seed=%d)",
        pipeline,
        len(indices),
        len(inner),
        lim_cfg.strategy,
        lim_cfg.seed,
    )
    return SubsetDataset(inner, indices)


def _build_dataset_from_dict(
    data_cfg_dict: dict[str, Any], cfg: TrainConfig, pipeline: str
) -> Dataset:
    # Thread task into the data dict so format-specific builders (e.g. build_hf)
    # can branch on it without coupling to TrainConfig directly.
    data_cfg_dict = {**data_cfg_dict, "task": cfg.task}
    builder = lookup("dataset", cfg.data.format)
    inner = cast(Dataset, builder(data_cfg_dict, model_name=cfg.model.name, pipeline=pipeline))
    return _apply_limit(inner, cfg, pipeline)


def _build_dataset(cfg: TrainConfig, pipeline: str) -> Dataset:
    """Build a dataset without any auto-split injection — used by doctor."""
    return _build_dataset_from_dict(cfg.data.model_dump(), cfg, pipeline)


def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> EvalArtifacts:
    """Build datasets, load model + PEFT, build tracker, run Trainer.fit.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §6.4.
    """
    # On resume, continue in the run dir that owns the checkpoint
    # (checkpoints live at <run_dir>/checkpoints/step_N/), so resumed
    # artifacts (config.yaml, best/, metrics, split_source.json) stay in the
    # original folder. Fresh runs mint a new timestamped dir.
    resume_run_dir = resume_from.parent.parent if resume_from is not None else None
    run_dir = resume_run_dir if resume_run_dir is not None else make_run_dir(cfg)
    vs = resolve_split_source(cfg, run_dir=resume_run_dir)
    save_split_source(vs, run_dir)
    _log_split_source(vs)

    data_cfg_dict = cfg.data.model_dump()
    if vs.train_ids is not None:
        # Inject resolved ids whenever the split carved a train pool — both
        # auto_split (val+test) and test-only (mode="none", no val bucket).
        # Without this guard, test-only runs build train_ds from the full pool,
        # leaking held-out test images into training (spec §9 invariant: "test is
        # held out from training entirely").
        resolved: dict[str, list[str]] = {"train": list(vs.train_ids)}
        if vs.mode == "auto_split":
            assert vs.val_ids is not None  # noqa: S101
            resolved["eval"] = list(vs.val_ids)
        data_cfg_dict["_resolved_image_ids"] = resolved

    train_ds = _build_dataset_from_dict(data_cfg_dict, cfg, "train")
    val_ds: Dataset | None = (
        None if vs.mode == "none" else _build_dataset_from_dict(data_cfg_dict, cfg, "eval")
    )

    # Write subset.json when at least one side has a limit applied
    lim_cfg = cfg.data.limit
    if (lim_cfg.train is not None or lim_cfg.val is not None) and val_ds is not None:
        _write_subset_manifest(run_dir, train_ds, val_ds, cfg)

    wrapper: Any = load_sam31(
        cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
    )
    peft_factory = lookup("peft", cfg.peft.method)
    peft_factory(wrapper, cfg.peft)

    # Trunk feature cache setup (spec §3 Integration / §3.5 activation guard).
    # Fires AFTER load_sam31 + peft_factory so the guards see the final frozen state.
    if cfg.train.cache_trunk_features:
        _setup_trunk_cache(cfg, wrapper, train_ds, run_dir)

    tracker = build_tracker(cfg)
    trainer = Trainer(wrapper, train_ds, val_ds, tracker, cfg)
    return trainer.fit(run_dir=run_dir, resume_from=resume_from)


def _setup_trunk_cache(
    cfg: TrainConfig,
    wrapper: Any,
    train_ds: Dataset,
    run_dir: Path,
) -> None:
    """Fire §2 correctness guards and build+attach TrunkFeatureCache.

    Called from run_training when cfg.train.cache_trunk_features is True.
    Raises ValueError on any guard failure (fail-fast, build-time).

    Spec: §2 (three correctness guards), §3 Integration, §3.5 activation guard.
    """
    import torch

    from custom_sam_peft.models.trunk_cache import (
        TrunkFeatureCache,
        assert_aug_off,
        assert_rgb_input,
        assert_trunk_frozen,
        trunk_fingerprint,
    )

    # Peel the wrapper: Sam3Wrapper.model is _Sam3ImageAdapter.
    adapter = getattr(wrapper, "model", None)

    # Guard 1: trunk frozen (spec §2 guard 1) — checked on the full wrapper
    # so the walker can find backbone.vision_backbone through the adapter.
    assert_trunk_frozen(wrapper)

    # Guard 2: RGB input (spec §2 guard 2)
    channel_adapter = getattr(adapter, "channel_adapter", None)
    assert_rgb_input(channel_adapter)

    # Guard 3: aug off (spec §2 guard 3) — inspect the built train transform.
    # Use getattr to reach _transforms without coupling to a specific Dataset type.
    # SubsetDataset wraps an inner dataset; walk one level if needed.
    train_transform = getattr(train_ds, "_transforms", None)
    if train_transform is None:
        inner_ds = getattr(train_ds, "_inner", None)
        if inner_ds is not None:
            train_transform = getattr(inner_ds, "_transforms", None)
    if train_transform is not None:
        assert_aug_off(train_transform)
    else:
        # Spec §2 guard 3: fail-fast on ANY uncertainty about augmentation.
        # We cannot confirm aug-off for a non-standard dataset whose transform
        # is not accessible via _transforms — refuse rather than silently skip.
        # (Standard CocoDataset / SubsetDataset always expose _transforms.)
        raise ValueError(
            f"cache_trunk_features: could not locate _transforms on train dataset "
            f"{type(train_ds).__name__}; cannot verify the aug-off precondition "
            f"(spec §2 guard 3). Disable caching (cache_trunk_features = false) or "
            "expose a _transforms attribute on the dataset."
        )

    # Build fingerprint: identifies the exact trunk config so a stale cache
    # from a different run/scope cannot be replayed.
    fp = trunk_fingerprint(
        checkpoint_id=cfg.model.name or "unknown",
        scope=cfg.peft.scope,
        dtype=cfg.model.dtype,
        image_size=1008,  # SAM3_IMAGE_SIZE — always fixed for SAM 3.1
    )

    # Derive model_dtype for the cache's H2D cast.
    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = _dtype_map.get(cfg.model.dtype, torch.float32)

    cache_dir = run_dir / ".trunk_cache"
    cache = TrunkFeatureCache(
        cache_dir=cache_dir,
        fingerprint=fp,
        model_dtype=model_dtype,
        n_samples=len(train_ds),
        free_disk_fraction=cfg.train.cache_free_disk_fraction,
        allow_slow_disk=cfg.train.cache_allow_slow_disk,
    )

    # Attach the cache to the adapter.
    if adapter is not None and hasattr(adapter, "attach_trunk_cache"):
        adapter.attach_trunk_cache(cache)
    else:
        _LOG.warning(
            "cache_trunk_features: adapter %s has no attach_trunk_cache method; "
            "cache will not be used.",
            type(adapter).__name__,
        )

    _LOG.info(
        "TrunkFeatureCache attached: cache_dir=%s n_samples=%d fingerprint=%s...",
        cache_dir,
        len(train_ds),
        fp[:12],
    )


def _write_subset_manifest(
    run_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
) -> None:
    """Write <run_dir>/subset.json recording resolved indices per side."""
    lim_cfg = cfg.data.limit
    manifest: dict[str, Any] = {
        "limit": {
            "train": lim_cfg.train,
            "val": lim_cfg.val,
            "seed": lim_cfg.seed,
            "strategy": lim_cfg.strategy,
        }
    }
    if lim_cfg.train is not None and isinstance(train_ds, SubsetDataset):
        inner_len = len(train_ds._inner)
        manifest["train"] = {
            "n_total": inner_len,
            "n_kept": len(train_ds),
            "indices": train_ds._indices,
        }
    if lim_cfg.val is not None and isinstance(val_ds, SubsetDataset):
        inner_len = len(val_ds._inner)
        manifest["val"] = {
            "n_total": inner_len,
            "n_kept": len(val_ds),
            "indices": val_ds._indices,
        }
    (run_dir / "subset.json").write_text(json.dumps(manifest, indent=2))


# Canonical library alias: run_train(config) -> EvalArtifacts
run_train = run_training
