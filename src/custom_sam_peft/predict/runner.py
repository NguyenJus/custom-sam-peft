"""predict/runner.py — core inference loop for ``csp predict``.

Public API:
  PredictOptions — frozen dataclass; one field per CLI flag.
  PredictReport  — frozen dataclass; n_images, n_predictions, elapsed_sec.
  run_predict(opts: PredictOptions) -> PredictReport

Design notes (spec §2, §9):
  - ``load_sam31`` is called *after* the dry-run short-circuit.
  - ``peft_adapters`` is imported *only* inside the ``if opts.checkpoint is not None`` block,
    so the base-model-only hot path never triggers that import.
  - ``queries_to_coco_results`` is called per-image (hard-asserts batch==1).
  - v1 calls postprocess per-image even when --batch-size > 1 (spec §12, §13).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from custom_sam_peft.cli._progress import progress as P

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUILTIN_DEFAULT_MODEL = "facebook/sam3.1"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictOptions:
    """All options passed from the CLI shell to run_predict.

    Most fields carry NO field-level default — defaults live in the Phase 6
    CLI layer so that tests must construct PredictOptions explicitly.

    Exception: ``batch_size`` carries a field-level default of ``"auto"``
    for API-surface reasons (spec §13 AC 13).  Because Python dataclasses
    require fields-with-defaults to come after fields-without-defaults,
    ``batch_size`` is placed last.
    """

    images: Path
    prompts: str  # raw spec; resolved by parse_prompts inside run_predict
    output: Path
    checkpoint: Path | None
    merge_adapter: bool
    config: Path | None
    score_threshold: float
    top_k: int
    save_masks: Literal["rle", "png", "none"]
    visualize: bool
    device: Literal["auto", "cuda", "cpu"]
    dtype: Literal["auto", "bfloat16", "float32"]
    seed: int
    dry_run: bool
    verbose: bool
    batch_size: int | Literal["auto"] = "auto"


@dataclass(frozen=True)
class PredictReport:
    """Summary returned by run_predict on success."""

    n_images: int
    n_predictions: int
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Internal resolved-config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedConfig:
    """Post-resolution values; computed once by _resolve_config."""

    model_name: str
    image_size: int  # always SAM3_IMAGE_SIZE; kept for logging / backward-compat uses
    channels: int
    channel_semantics: str
    device: str  # "cuda" or "cpu" (auto already resolved)
    dtype: torch.dtype  # resolved torch.dtype
    dtype_str: str  # "bfloat16" or "float32"
    normalize_mean: list[float]
    normalize_std: list[float]


# ---------------------------------------------------------------------------
# Config resolution (spec §6)
# ---------------------------------------------------------------------------


def _resolve_config(opts: PredictOptions) -> _ResolvedConfig:
    """Resolve PredictOptions into concrete post-resolution values.

    Precedence for model.name (spec §6):
      1. adapter adapter_config.json:base_model_name_or_path (when checkpoint given)
         → emits WARN if it disagrees with config / builtin default.
      2. --config.model.name (when config given).
      3. Builtin default ("facebook/sam3.1").

    image_size is always SAM3_IMAGE_SIZE (1008); SAM 3.1 rescales inputs internally.
    """
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    # --- parse --config YAML (if given) ---
    config_model_name: str | None = None
    config_channels: int | None = None
    config_channel_semantics: str | None = None

    if opts.config is not None:
        try:
            import yaml

            raw: dict[str, Any] = yaml.safe_load(opts.config.read_text(encoding="utf-8")) or {}
            model_section = raw.get("model", {})
            if isinstance(model_section, dict):
                val = model_section.get("name")
                if val is not None:
                    config_model_name = str(val)
            data_section = raw.get("data", {})
            if isinstance(data_section, dict):
                val = data_section.get("channels")
                if val is not None:
                    config_channels = int(val)
                val = data_section.get("channel_semantics")
                if val is not None:
                    config_channel_semantics = str(val)
        except Exception as exc:
            logger.warning("Failed to parse --config %s: %s", opts.config, exc)

    # --- model name resolution ---
    adapter_model_name: str | None = None
    if opts.checkpoint is not None:
        from custom_sam_peft.predict.adapter_load import read_adapter_base_model_name

        adapter_model_name = read_adapter_base_model_name(opts.checkpoint)

    # Build comparison baseline (what --config or builtin says)
    fallback_model_name = config_model_name or _BUILTIN_DEFAULT_MODEL

    if adapter_model_name is not None:
        if adapter_model_name != fallback_model_name:
            logger.warning(
                "Adapter base_model_name_or_path %r disagrees with config/default %r; "
                "using adapter value (spec §6).",
                adapter_model_name,
                fallback_model_name,
            )
        model_name = adapter_model_name
    else:
        model_name = fallback_model_name

    # --- image_size (constant — SAM 3.1 always rescales to SAM3_IMAGE_SIZE) ---
    image_size = SAM3_IMAGE_SIZE

    # --- channels + channel_semantics ---
    channels = config_channels if config_channels is not None else 3
    channel_semantics = config_channel_semantics if config_channel_semantics is not None else "rgb"

    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTIC_NAMES

    if channel_semantics not in CHANNEL_SEMANTIC_NAMES:
        raise ValueError(
            f"data.channel_semantics={channel_semantics!r} in the predict --config is not "
            f"a valid semantic; expected one of {sorted(CHANNEL_SEMANTIC_NAMES)}."
        )

    # --- device resolution ---
    if opts.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = opts.device

    # --- dtype resolution ---
    if opts.dtype == "auto":
        dtype_str = "bfloat16" if device_str == "cuda" else "float32"
    else:
        dtype_str = opts.dtype

    dtype_torch = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    # --- normalization (resolve_normalization with fallback) ---
    from custom_sam_peft.config.schema import NormalizeConfig
    from custom_sam_peft.data.transforms import resolve_normalization

    mean, std = resolve_normalization(
        model_name, NormalizeConfig(), channel_semantics=channel_semantics
    )

    return _ResolvedConfig(
        model_name=model_name,
        image_size=image_size,
        channels=channels,
        channel_semantics=channel_semantics,
        device=device_str,
        dtype=dtype_torch,
        dtype_str=dtype_str,
        normalize_mean=mean,
        normalize_std=std,
    )


# ---------------------------------------------------------------------------
# Stable image-id hash (same scheme as eval/evaluator._int_image_id)
# ---------------------------------------------------------------------------


def _int_image_id(path: Path) -> int:
    """blake2s 8-byte digest of the absolute path string → stable int id."""
    return int(
        hashlib.blake2s(str(path.resolve()).encode("utf-8"), digest_size=8).hexdigest(),
        16,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_predict(opts: PredictOptions) -> PredictReport:
    """Run offline inference per spec §9.

    Steps (in exact order):
      1. resolve images + prompts
      2. preflight log
      3. dry-run short-circuit (BEFORE load_sam31)
      4. load_sam31 + move to device + cast dtype
      5. adapter load + optional merge
      6. build transforms
      7. VRAM hint (cuda + bs=1 + free > 12 GB)
      8. warmup
      9. forward loop (per-image, per-class; postprocess per-image)
     10. save_masks branch + visualize
     11. write predictions.json, image_id_map.json, run.json
     12. return PredictReport
    """
    # --- seeds (spec §12 determinism row) ---
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # ---------------------------------------------------------------------------
    # Step 1: resolve images + prompts
    # ---------------------------------------------------------------------------
    from custom_sam_peft.predict.inputs import parse_prompts, resolve_images

    image_paths = resolve_images(opts.images)
    prompts = parse_prompts(opts.prompts)

    # ---------------------------------------------------------------------------
    # Step 2: preflight log
    # ---------------------------------------------------------------------------
    rcfg = _resolve_config(opts)

    adapter_kind_str = "none"
    if opts.checkpoint is not None:
        from custom_sam_peft.predict.adapter_load import detect_adapter_kind

        adapter_kind_str = detect_adapter_kind(opts.checkpoint)

    logger.info(
        "predict: model=%s adapter=%s device=%s dtype=%s prompts=[%d] images=%d threshold=%.3f",
        rcfg.model_name,
        adapter_kind_str,
        rcfg.device,
        rcfg.dtype_str,
        len(prompts),
        len(image_paths),
        opts.score_threshold,
    )

    # ---------------------------------------------------------------------------
    # Step 3: dry-run short-circuit (spec §9 item 3, §10) — BEFORE load_sam31
    # ---------------------------------------------------------------------------
    if opts.dry_run:
        _print_dry_run_preview(image_paths, prompts, rcfg)
        return PredictReport(n_images=0, n_predictions=0, elapsed_sec=0.0)

    # ---------------------------------------------------------------------------
    # Step 4: load model (happens AFTER dry-run check)
    # ---------------------------------------------------------------------------
    from custom_sam_peft.config.schema import ModelConfig
    from custom_sam_peft.models.sam3 import load_sam31

    model_cfg = ModelConfig(name=rcfg.model_name)

    model: torch.nn.Module = load_sam31(
        model_cfg, channels=rcfg.channels, channel_semantics=rcfg.channel_semantics
    )
    model = model.to(rcfg.device, dtype=rcfg.dtype)
    model.eval()

    # ---------------------------------------------------------------------------
    # Step 5: adapter load + optional merge (lazy-import peft_adapters here only)
    # ---------------------------------------------------------------------------
    kind: str | None = None
    if opts.checkpoint is not None:
        from custom_sam_peft.predict import adapter_load as _al

        kind = _al.detect_adapter_kind(opts.checkpoint)
        model = _al.load_adapter(model, opts.checkpoint, kind)
        if opts.merge_adapter:
            model = _al.maybe_merge_adapter(model, merge=True)
        adapter_kind_str = kind

    # ---------------------------------------------------------------------------
    # Step 6: build transforms
    # ---------------------------------------------------------------------------
    from custom_sam_peft.config.schema import NormalizeConfig
    from custom_sam_peft.data.transforms import build_eval_transforms

    normalize_cfg = NormalizeConfig(mean=rcfg.normalize_mean, std=rcfg.normalize_std)
    transforms = build_eval_transforms(
        rcfg.image_size,
        model_name=rcfg.model_name,
        normalize=normalize_cfg,
        channel_semantics=rcfg.channel_semantics,
    )

    # ---------------------------------------------------------------------------
    # Step 6b: resolve batch_size (spec §13 AC 13)
    # ---------------------------------------------------------------------------
    if opts.batch_size == "auto":
        from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
        from custom_sam_peft.presets import decide_eval_batch_size

        bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
    else:
        bs = int(opts.batch_size)

    # ---------------------------------------------------------------------------
    # Step 7: VRAM hint (spec §6, §12)
    # ---------------------------------------------------------------------------
    if rcfg.device == "cuda" and bs == 1:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes > 12 * 1024**3:
                logger.info("free VRAM is >12 GB; consider --batch-size 4 or 8.")
        except RuntimeError:
            pass  # some older drivers fail

    # ---------------------------------------------------------------------------
    # Step 8: warmup (spec §9 item 8)
    # ---------------------------------------------------------------------------
    from custom_sam_peft.data.base import TextPrompts

    with torch.no_grad():
        _warmup_input = torch.zeros(
            1, rcfg.channels, rcfg.image_size, rcfg.image_size, device=rcfg.device, dtype=rcfg.dtype
        )
        _dummy_prompt = [TextPrompts(classes=["warmup"])]
        with contextlib.suppress(Exception):
            model(_warmup_input, _dummy_prompt, box_hints=None)

    # ---------------------------------------------------------------------------
    # Step 9: forward loop — flat (image_chunk, class_group) iteration
    # ---------------------------------------------------------------------------
    from custom_sam_peft.eval.evaluator import _row_outputs
    from custom_sam_peft.eval.postprocess import queries_to_coco_results
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    def _chunked(seq: list[Any], n: int) -> list[list[Any]]:
        return [seq[i : i + n] for i in range(0, len(seq), n)]

    all_predictions: list[dict[str, object]] = []
    id_to_path: dict[int, Path] = {}
    id_to_stem: dict[int, str] = {}
    originals: dict[int, tuple[int, int]] = {}
    n_successful = 0
    t_start = time.perf_counter()

    # Initialise progress inner bar to the real image count (no-op outside a session).
    P.reset_inner(total=len(image_paths))
    log_every_n = max(1, len(image_paths) // 50)
    images_processed = 0

    for chunk_paths in _chunked(list(image_paths), bs):
        chunk_t0 = time.perf_counter()

        # --- open + transform each image in the chunk ---
        imgs: list[torch.Tensor] = []
        metas: list[tuple[int, int, int]] = []  # (image_id, orig_h, orig_w)
        chunk_paths_ok: list[Path] = []  # parallel to metas; used for verbose logging

        for img_path in chunk_paths:
            try:
                from custom_sam_peft.data.io import read_image as _read_image

                img_np = _read_image(img_path, rcfg.channels)
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", img_path, exc)
                continue

            orig_h, orig_w = img_np.shape[0], img_np.shape[1]
            image_id = _int_image_id(img_path)
            id_to_path[image_id] = img_path.resolve()
            id_to_stem[image_id] = img_path.stem
            originals[image_id] = (orig_h, orig_w)

            transformed = transforms(image=img_np, bboxes=[], class_labels=[], instance_idx=[])
            imgs.append(transformed["image"].to(rcfg.device, dtype=rcfg.dtype))
            metas.append((image_id, orig_h, orig_w))
            chunk_paths_ok.append(img_path)

        if not imgs:
            continue

        img_batch = torch.stack(imgs, dim=0)  # (B, C, H, W)

        # --- flat inner loop over class groups ---
        for group in _chunked(prompts, MULTIPLEX_CAP):
            K_g = len(group)
            prompts_g = [TextPrompts(classes=list(group)) for _ in metas]
            try:
                with torch.no_grad():
                    outputs = model(img_batch, prompts_g, box_hints=None)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.error(
                        "OOM: consider --no-merge-adapter (QLoRA), --batch-size 1, or --device cpu"
                    )
                raise

            # postprocess each (image, class) row from the multiplexed output
            for r in range(len(metas) * K_g):
                ii, kk = divmod(r, K_g)
                image_id, orig_h, orig_w = metas[ii]
                class_idx_one_based = prompts.index(group[kk]) + 1
                entries = queries_to_coco_results(
                    _row_outputs(outputs, r),
                    image_id=image_id,
                    category_id=class_idx_one_based,
                    original_hw=(orig_h, orig_w),
                    mask_threshold=0.0,
                )
                entries = [e for e in entries if cast(float, e["score"]) >= opts.score_threshold]
                entries.sort(key=lambda e: cast(float, e["score"]), reverse=True)
                entries = entries[: opts.top_k]
                all_predictions.extend(entries)

        n_successful += len(metas)
        images_processed += len(metas)

        # Verbose: emit one INFO line per image in the chunk.
        if opts.verbose:
            chunk_latency_ms = (time.perf_counter() - chunk_t0) * 1000.0
            per_image_ms = chunk_latency_ms / max(len(metas), 1)
            for img_path in chunk_paths_ok:
                logger.info(
                    "image %d/%d %s (%.1f ms)",
                    images_processed - len(metas) + chunk_paths_ok.index(img_path) + 1,
                    len(image_paths),
                    img_path.name,
                    per_image_ms,
                )

        # Progress ticks: one advance per image in this chunk.
        for _ in metas:
            P.advance_inner()

        if images_processed % log_every_n == 0 or images_processed == len(image_paths):
            elapsed_so_far = max(time.perf_counter() - t_start, 1e-9)
            P.update_postfix(
                done=f"{images_processed}/{len(image_paths)}",
                it_s=images_processed / elapsed_so_far,
            )

    elapsed_sec = time.perf_counter() - t_start

    # If zero images succeeded, raise (CLI converts to exit 1 per spec §10)
    if n_successful == 0:
        logger.error("All images failed to load; no predictions written.")
        raise RuntimeError("all images failed")

    # ---------------------------------------------------------------------------
    # Step 10: save_masks branch + visualize
    # ---------------------------------------------------------------------------
    opts.output.mkdir(parents=True, exist_ok=True)

    from custom_sam_peft.predict.writers import (
        write_image_id_map,
        write_predictions,
        write_run_json,
    )

    write_predictions(
        all_predictions,
        opts.output,
        save_masks=opts.save_masks,
        originals=originals,
        id_to_stem=id_to_stem if opts.save_masks == "png" else None,
    )

    if opts.visualize:
        from custom_sam_peft.predict.visualize import write_visualization

        # Group predictions by image_id
        by_image: dict[int, list[dict[str, object]]] = {}
        for entry in all_predictions:
            iid = int(cast(int, entry["image_id"]))
            by_image.setdefault(iid, []).append(entry)

        for image_id, img_path in id_to_path.items():
            img_entries = by_image.get(image_id, [])
            write_visualization(img_path, img_entries, opts.output, prompts=prompts)

    # ---------------------------------------------------------------------------
    # Step 11: write sidecars
    # ---------------------------------------------------------------------------
    write_image_id_map(id_to_path, opts.output)

    run_meta: dict[str, Any] = {
        "model": rcfg.model_name,
        "checkpoint": str(opts.checkpoint) if opts.checkpoint is not None else None,
        "adapter_kind": adapter_kind_str if opts.checkpoint is not None else None,
        "merge_adapter": opts.merge_adapter,
        "prompts": prompts,
        "score_threshold": opts.score_threshold,
        "top_k": opts.top_k,
        "mask_threshold": 0.0,
        "device": rcfg.device,
        "dtype": rcfg.dtype_str,
        "image_size": rcfg.image_size,
        "batch_size": opts.batch_size,
        "seed": opts.seed,
        "n_images": n_successful,
        "n_predictions": len(all_predictions),
        "elapsed_sec": elapsed_sec,
    }
    write_run_json(run_meta, opts.output)

    # ---------------------------------------------------------------------------
    # Step 12: return PredictReport
    # ---------------------------------------------------------------------------
    return PredictReport(
        n_images=n_successful,
        n_predictions=len(all_predictions),
        elapsed_sec=elapsed_sec,
    )


# ---------------------------------------------------------------------------
# Dry-run preview printer
# ---------------------------------------------------------------------------


def _print_dry_run_preview(
    image_paths: list[Path],
    prompts: list[str],
    rcfg: _ResolvedConfig,
) -> None:
    """Print the dry-run preview to stdout (spec §9 item 3).

    Uses bare ``print`` (T201 ignored for this file via per-file-ignore) to keep
    output grep-friendly and avoid Rich's word-wrap on long filesystem paths.
    """
    print("=== csp predict --dry-run ===")
    print()
    print("Resolved config:")
    print(f"  model:      {rcfg.model_name}")
    print(f"  device:     {rcfg.device}")
    print(f"  dtype:      {rcfg.dtype_str}")
    print(f"  image_size: {rcfg.image_size}")
    print(f"  normalize:  mean={rcfg.normalize_mean}  std={rcfg.normalize_std}")
    print()
    print(f"Prompts ({len(prompts)}):")
    for i, p in enumerate(prompts, start=1):
        print(f"  [{i}] {p}")
    print()
    n_total = len(image_paths)
    shown = image_paths[:10]
    print(f"Images ({n_total} total; showing first {len(shown)}):")
    for img_path in shown:
        print(f"  {img_path}")
    if n_total > 10:
        print(f"  ... and {n_total - 10} more")
    print()
    print("(dry-run: model not loaded, no files written)")
