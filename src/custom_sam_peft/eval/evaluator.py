"""Evaluator — runs a model over a dataset and returns a MetricsReport.

See docs/superpowers/specs/2026-05-17-eval-design.md for the contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
import pycocotools.mask as mask_utils
import torch
from pycocotools.coco import COCO

from custom_sam_peft import profiling
from custom_sam_peft.cli._progress import progress as P
from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.data.base import Dataset, Example, TextPrompts
from custom_sam_peft.data.tiling import (
    EVAL_OVERLAP,
    iter_windows,
    tiling_engaged,
)
from custom_sam_peft.eval.metrics import MetricsReport, coco_max_dets_cap, compute_coco_map
from custom_sam_peft.eval.postprocess import (
    _upsample_mask_logits,
    queries_to_coco_results,
    score_and_topk_filter,
)
from custom_sam_peft.eval.proxy_map import ProxyEntry, dense_iou_matrix, proxy_map_from_iou
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE
from custom_sam_peft.oom import OomDecision, OomLadder, is_cuda_oom
from custom_sam_peft.paths import predictions_path
from custom_sam_peft.runtime import Runtime, to_device

_LOG = logging.getLogger(__name__)


def _lite_exact_map_hatch() -> bool:
    """Whether the ``CSP_LITE_EXACT_MAP`` escape hatch forces lite -> exact mAP.

    Mirrors ``profiling.py``'s ``CSP_PROFILE`` truthiness: disabled for ``""``,
    ``"0"``, ``"false"``, ``"False"``; any other non-empty value enables it.
    Read live (not module-import-time) so tests / runs can toggle the env var.
    """
    return os.environ.get("CSP_LITE_EXACT_MAP", "0") not in ("", "0", "false", "False")


def _proxy_entry_from_outputs(
    row_outputs: dict[str, torch.Tensor],
    gt_bin: torch.Tensor,
    image_id: int,
    cat_idx: int,
    original_hw: tuple[int, int],
    mask_threshold: float,
    max_dets: int,
) -> ProxyEntry:
    """Build one ``(image, category)`` ProxyEntry from a single forward row.

    Uses the SHARED ``score_and_topk_filter`` + ``_upsample_mask_logits`` helpers
    so the score/topk/upsample/binarize is byte-identical to the exact RLE path —
    the only difference is binarization stays ON-DEVICE (no ``.cpu()``) and the
    dense IoU is a matmul instead of an RLE encode. Emits an entry even when there
    are zero predictions or zero GT (so npig is complete for the AP aggregation).

    ``gt_bin`` is the stacked ``(M, H, W)`` bool GT for this (image, category) on
    the model device. ``original_hw`` is the (tile-local) prediction resolution.
    """
    scores, keep_idx = score_and_topk_filter(row_outputs, max_dets)
    pred_masks = row_outputs["pred_masks"]
    n = pred_masks.shape[1]
    if n == 0:
        m = 0
        pred_bin = torch.zeros(
            (0, original_hw[0], original_hw[1]), dtype=torch.bool, device=gt_bin.device
        )
    else:
        masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
        if keep_idx is not None:
            masks_logits = masks_logits[keep_idx]
        with profiling.bucket("eval.mask_upsample"):
            masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (M, H, W)
        # SAME binarization as the exact path (masks_up > mask_threshold) but
        # kept on-device — bit-identical masks, no logit-resolution binarize.
        pred_bin = masks_up > mask_threshold
        m = pred_bin.shape[0]
    with profiling.bucket("eval.proxy_iou"):
        iou = dense_iou_matrix(pred_bin, gt_bin)  # (m, M) on device
    iou_np = iou.cpu().numpy()
    scores_np = scores.detach().cpu().numpy() if m else np.zeros((0,), dtype=np.float32)
    return ProxyEntry(
        image_id=image_id,
        category_id=cat_idx + 1,
        iou=iou_np,
        scores=scores_np,
    )


def _gt_masks_for_category(
    instances: Sequence[Any],
    cat_idx: int,
    hw: tuple[int, int],
    runtime: Runtime,
    crop: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Stack GT masks for one category into ``(M, H, W)`` bool on the device.

    Selects this example's instances with ``class_id == cat_idx``. On the tiling
    path ``crop=(y0, x0, h, w)`` crops each full-image mask to the tile window
    exactly like ``_build_coco_gt_with_tiling`` (instances with no pixels in the
    window are dropped). ``hw`` is the (tile-local) prediction resolution that the
    stacked masks must match. Returns an empty ``(0, H, W)`` tensor when no GT.
    Device moves route through ``runtime.to_device`` (§9.2 static guard); the
    masks are already bool (``Instance.mask``), so only the device changes.
    """
    masks: list[torch.Tensor] = []
    for inst in instances:
        if int(inst.class_id) != cat_idx:
            continue
        mask = inst.mask
        if crop is not None:
            y0, x0, h, w = crop
            mask = mask[y0 : y0 + h, x0 : x0 + w]
            if not bool(mask.any()):
                continue
        masks.append(to_device(mask, runtime))
    if not masks:
        return torch.zeros((0, hw[0], hw[1]), dtype=torch.bool, device=runtime.device)
    return torch.stack(masks, dim=0)


def _row_outputs(outputs: dict[str, torch.Tensor], r: int) -> dict[str, torch.Tensor]:
    """Slice multiplex outputs at row r, preserving the batch dim (size 1).

    Non-tensor entries (e.g. sam3's ``prev_encoder_out`` nested dict or
    ``encoder_hidden_states``) are dropped silently. The only consumer
    (``queries_to_coco_results``) needs just the tensor prediction keys
    (``pred_logits``, ``pred_boxes``, ``pred_masks``, ``presence_logit_dec``).
    """
    return {k: v[r : r + 1] for k, v in outputs.items() if isinstance(v, torch.Tensor)}


def _int_image_id(image_id: str) -> int:
    """Stable int hash of a string image_id (blake2s, 8-byte digest)."""
    return int(hashlib.blake2s(image_id.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _mask_to_rle(mask: torch.Tensor) -> Any:
    """Convert a (H, W) bool tensor to a pycocotools RLE dict."""
    with profiling.bucket("eval.gt_rle_encode"):
        arr = mask.cpu().numpy().astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(arr))
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _tile_image_id(image_id: str, tile_idx: int) -> int:
    """Stable int id for a (image_id, tile_idx) pair.

    Uses the same blake2s scheme as _int_image_id so COCO evaluation alignment
    between per-tile predictions and per-tile GT is guaranteed to use the same id.
    """
    return int(
        hashlib.blake2s(f"{image_id}:{tile_idx}".encode(), digest_size=8).hexdigest(),
        16,
    )


def _build_coco_gt_with_tiling(
    examples: Sequence[Example],
    dataset: Dataset,
) -> COCO:
    """Build an in-memory COCO ground-truth, tiling large images (spec §5.4).

    For images where ``tiling_engaged`` is True, the GT is decomposed into
    non-overlapping tiles (EVAL_OVERLAP=0.0) and each tile's GT instances are
    cropped to tile-local coordinates.  The tile-local image_id is produced by
    ``_tile_image_id(ex.image_id, tile_idx)``.  Small images are added as-is
    using ``_int_image_id`` (direct path, byte-for-byte unchanged).
    """
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    seen_ids: dict[int, str] = {}
    ann_id = 1

    for ex in examples:
        orig_h, orig_w = int(ex.image.shape[-2]), int(ex.image.shape[-1])

        if not tiling_engaged(orig_h, orig_w):
            # Direct path — unchanged.
            int_id = _int_image_id(ex.image_id)
            prior = seen_ids.get(int_id)
            if prior is not None and prior != ex.image_id:
                raise RuntimeError(
                    f"image_id hash collision: {ex.image_id!r} and {prior!r} both hash to {int_id}"
                )
            seen_ids[int_id] = ex.image_id
            images.append({"id": int_id, "height": orig_h, "width": orig_w})
            for inst in ex.instances:
                rle = _mask_to_rle(inst.mask)
                area = int(mask_utils.area(rle))
                x1, y1, x2, y2 = (float(v) for v in inst.box.tolist())
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": int_id,
                        "category_id": int(inst.class_id) + 1,
                        "iscrowd": 0,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": area,
                        "segmentation": rle,
                    }
                )
                ann_id += 1
        else:
            # Tiling path — decompose into non-overlapping tiles.
            windows = iter_windows(orig_h, orig_w, tile=SAM3_IMAGE_SIZE, overlap=EVAL_OVERLAP)
            for t_idx, win in enumerate(windows):
                tile_int_id = _tile_image_id(ex.image_id, t_idx)
                images.append({"id": tile_int_id, "height": win.h, "width": win.w})
                for inst in ex.instances:
                    # Crop the instance mask to the tile window.
                    inst_mask_np = inst.mask.cpu().numpy()  # (orig_h, orig_w) bool
                    tile_mask_np = inst_mask_np[win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w]
                    if not tile_mask_np.any():
                        continue  # instance has no pixels in this tile; skip
                    tile_mask_t = torch.from_numpy(tile_mask_np)
                    rle = _mask_to_rle(tile_mask_t)
                    area = int(mask_utils.area(rle))
                    # Derive tile-local bbox from the cropped mask.
                    rows = np.any(tile_mask_np, axis=1)
                    cols = np.any(tile_mask_np, axis=0)
                    y_min = int(np.argmax(rows))
                    y_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
                    x_min = int(np.argmax(cols))
                    x_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": tile_int_id,
                            "category_id": int(inst.class_id) + 1,
                            "iscrowd": 0,
                            "bbox": [
                                float(x_min),
                                float(y_min),
                                float(x_max - x_min),
                                float(y_max - y_min),
                            ],
                            "area": area,
                            "segmentation": rle,
                        }
                    )
                    ann_id += 1

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(dataset.class_names)]
    gt = COCO()
    gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    gt.createIndex()
    return gt


def _build_coco_gt_from_examples(
    examples: Sequence[Example], dataset: Dataset
) -> tuple[COCO, dict[str, int]]:
    """Build an in-memory COCO ground-truth from a pre-fetched list of Examples.

    Returns the COCO object and a ``str_image_id -> int_image_id`` map.
    Raises RuntimeError on int-id collision.
    """
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    seen_ids: dict[int, str] = {}
    str_to_int: dict[str, int] = {}
    ann_id = 1

    for ex in examples:
        int_id = _int_image_id(ex.image_id)
        prior = seen_ids.get(int_id)
        if prior is not None and prior != ex.image_id:
            raise RuntimeError(
                f"image_id hash collision: {ex.image_id!r} and {prior!r} both hash to {int_id}"
            )
        seen_ids[int_id] = ex.image_id
        str_to_int[ex.image_id] = int_id
        h, w = ex.image.shape[-2:]
        images.append({"id": int_id, "height": int(h), "width": int(w)})
        for inst in ex.instances:
            rle = _mask_to_rle(inst.mask)
            area = int(mask_utils.area(rle))
            x1, y1, x2, y2 = (float(v) for v in inst.box.tolist())
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int_id,
                    "category_id": int(inst.class_id) + 1,  # 1-indexed for COCO
                    "iscrowd": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": area,
                    "segmentation": rle,
                }
            )
            ann_id += 1

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(dataset.class_names)]
    gt = COCO()
    gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    gt.createIndex()
    return gt, str_to_int


class Evaluator:
    """Compute COCO metrics for a model on a dataset."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self._last_predictions: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Private helpers (decomposed from evaluate)
    # ------------------------------------------------------------------

    def _iter_predictions(
        self, model: Any, examples: Sequence[Example], dataset: Dataset
    ) -> list[dict[str, object]]:
        """Run the forward loop and return raw COCO-format prediction entries.

        Puts the model into eval mode for the duration of the loop and restores
        its training state on exit. Iterates flat over (image_chunk, class_group)
        pairs using the shared OomLadder for B-then-K OOM recovery. Moves dataset
        images to the model's device before each forward via runtime.to_device
        (§3 seam discipline). The dataset yields CPU tensors; passing them straight
        to a CUDA-resident model raises a device mismatch inside the first Conv2d.
        Falls back to CPU for parameterless / non-nn.Module test stubs.

        cfg.batch_size is already resolved to an int by run_eval (T10 wires the
        "auto" resolution). The OomLadder halves B then K stickily on
        torch.cuda.OutOfMemoryError; RETRY_B discards chunk_buf and restarts the
        image-chunk at the smaller B; RETRY_K resumes from the current class index
        at the smaller K, keeping chunk_buf; FLOOR_RETRY retries once; TERMINAL
        raises RuntimeError. Buffer-and-commit ensures no dup/drop on any path.

        When tiling_engaged(orig_h, orig_w) is True for an example, the image
        tensor is sliced into non-overlapping tiles (EVAL_OVERLAP=0.0 per spec
        §5.4/§13.5) and the forward is run per tile.  Predictions carry tile-local
        image_ids (from _tile_image_id) and tile-local original_hw — they
        align with _build_coco_gt_with_tiling's tile-local GT entries, so
        compute_coco_map sees paired (pred, GT) at the same coordinate frame.
        No stitched full-image mask is ever materialized.  Tiled images bypass
        the batch-chunk loop and are processed one image at a time (batch_size=1
        per tile, shared OomLadder).
        """
        cfg = self.cfg

        # Lazy import: predict/__init__ -> runner -> evaluator would otherwise create
        # a cycle at module load. tiling_preprocess itself imports only numpy/torch.
        from custom_sam_peft.predict.tiling_preprocess import preprocess_tile

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()

        try:
            _p = next(model.parameters())
            param_device = _p.device
            param_dtype = _p.dtype
        except (StopIteration, AttributeError):
            param_device = torch.device("cpu")
            param_dtype = torch.float32
        eval_runtime = Runtime(device=param_device, dtype=param_dtype)

        # Replace the old state dict with the shared ladder. effective_K starts at
        # min(MULTIPLEX_CAP, n_classes); micro_batch_size at the resolved cfg.batch_size.
        n_classes = len(dataset.class_names)
        max_dets_cap = coco_max_dets_cap()  # top-N cap, derived from COCOeval maxDets
        ladder = OomLadder(
            micro_batch_size=int(cfg.batch_size),
            effective_K=min(MULTIPLEX_CAP, n_classes) if n_classes else 1,
        )

        # Pad-only transform for the tiling branch (design C, spec §5.4): each
        # native-res tile crop is padded raw-0 THEN normalized via the SHARED
        # preprocess_tile helper, byte-identical to predict. The dataset already
        # built this transform (downscale=False for the eval pipeline) from the same
        # model/normalize/channel-semantics config predict uses — reuse it to avoid
        # both an unreachable rebuild (the evaluator holds only an EvalConfig) and
        # any drift from predict's preprocessing.
        pad_only_transform = getattr(dataset, "tile_transform", None)

        predictions: list[dict[str, object]] = []
        img_idx_global = 0
        try:
            with torch.no_grad(), P.push_subtask("eval", total=len(examples)) as sub:
                i = 0
                while i < len(examples):
                    bs = ladder.micro_batch_size
                    image_chunk = list(examples[i : i + bs])
                    # Split chunk into direct-path and tiling-path examples.
                    # Tiled examples are processed immediately, one image at a time.
                    direct_chunk: list[Example] = []
                    for ex in image_chunk:
                        orig_h = int(ex.image.shape[-2])
                        orig_w = int(ex.image.shape[-1])
                        if tiling_engaged(orig_h, orig_w):
                            # --- Tiling path (spec §5.4): non-overlapping tiles ---
                            windows = iter_windows(
                                orig_h,
                                orig_w,
                                tile=SAM3_IMAGE_SIZE,
                                overlap=EVAL_OVERLAP,
                            )
                            if ex.image_native is None:
                                raise ValueError(
                                    "eval tiling path requires Example.image_native "
                                    "(native-res raw pixels) for the per-tile pad-only "
                                    f"preprocess, but it is None for image_id={ex.image_id!r}. "
                                    "This is a dataset wiring bug: the eval/val dataset "
                                    "must populate image_native for oversized examples "
                                    "(design C, spec §5.4)."
                                )
                            if pad_only_transform is None:
                                raise ValueError(
                                    "eval tiling path requires the dataset's pad-only "
                                    "tile_transform, but it is absent for image_id="
                                    f"{ex.image_id!r}. The eval/val dataset must expose "
                                    "tile_transform (build_eval_transforms(..., "
                                    "downscale=False)) for design-C per-tile preprocess."
                                )
                            for t_idx, win in enumerate(windows):
                                tile_int_id = _tile_image_id(ex.image_id, t_idx)
                                tile_hw = (win.h, win.w)
                                # Crop the NATIVE-RES raw pixels for this window and run
                                # the SHARED pad-only preprocess (pad raw-0 -> normalize),
                                # byte-identical to predict's _predict_one_tile.
                                crop_np = ex.image_native[
                                    win.y0 : win.y0 + win.h,
                                    win.x0 : win.x0 + win.w,
                                ]
                                tile_batch = preprocess_tile(
                                    crop_np,
                                    pad_only_transform,
                                    device=eval_runtime.device,
                                    dtype=eval_runtime.dtype,
                                ).unsqueeze(0)  # (1, C, 1008, 1008), already on device
                                j = 0
                                while j < n_classes:
                                    K_g = min(ladder.effective_K, n_classes - j)
                                    group = dataset.class_names[j : j + K_g]
                                    prompts_g = [TextPrompts(classes=list(group))]
                                    try:
                                        with profiling.bucket("eval.forward"):
                                            outputs = cast(
                                                "dict[str, torch.Tensor]",
                                                model(tile_batch, prompts_g, support=None),
                                            )
                                        profiling.incr("eval.forwards")
                                    except RuntimeError as oom_exc:
                                        if not is_cuda_oom(oom_exc):
                                            raise
                                        decision = ladder.on_oom()
                                        # A per-tile forward is already B=1, so halving
                                        # B (RETRY_B) cannot shrink THIS forward. Drain
                                        # the B rung without re-running the identical
                                        # forward, mirroring the direct path's
                                        # restart-and-advance: keep applying on_oom()
                                        # until the ladder progresses to K reduction
                                        # (RETRY_K), the single FLOOR_RETRY, or TERMINAL.
                                        # The ladder is monotonic (B then K only
                                        # decrease, FLOOR_RETRY fires once), so this
                                        # loop is bounded and always reaches a non-B rung.
                                        while decision is OomDecision.RETRY_B:
                                            decision = ladder.on_oom()
                                        if decision is OomDecision.RETRY_K:
                                            continue
                                        if decision is OomDecision.FLOOR_RETRY:
                                            continue
                                        raise RuntimeError(
                                            "eval OOM at classes_per_forward=1 on a"
                                            " single tile; use a larger GPU."
                                        ) from None
                                    for kk in range(K_g):
                                        cat_idx = dataset.class_names.index(group[kk])
                                        entries = queries_to_coco_results(
                                            _row_outputs(outputs, kk),
                                            tile_int_id,
                                            cat_idx + 1,
                                            tile_hw,
                                            cfg.mask_threshold,
                                            max_dets=max_dets_cap,
                                        )
                                        predictions.extend(entries)
                                    j += K_g
                        else:
                            direct_chunk.append(ex)

                    # --- Direct path (unchanged): batch forward over direct_chunk ---
                    if direct_chunk:
                        images_t = to_device(
                            torch.stack([ex.image for ex in direct_chunk]), eval_runtime
                        )
                        chunk_buf: list[dict[str, object]] = []
                        restart_chunk = False
                        j = 0  # class index into dataset.class_names
                        while j < n_classes:
                            K_g = min(ladder.effective_K, n_classes - j)
                            group = dataset.class_names[j : j + K_g]
                            prompts_g = [TextPrompts(classes=list(group)) for _ in direct_chunk]
                            try:
                                with profiling.bucket("eval.forward"):
                                    outputs = cast(
                                        "dict[str, torch.Tensor]",
                                        model(images_t, prompts_g, support=None),
                                    )
                                profiling.incr("eval.forwards")
                                if profiling.is_enabled():
                                    # Report the forward OUTPUT dtype (bf16), not the
                                    # input image dtype (always fp32) — capturing the
                                    # input mislabels the compute dtype, the exact
                                    # confusion #250 (d06cd96) corrected.
                                    profiling.note(
                                        eval_forward_dtype=str(
                                            outputs["pred_masks"].dtype
                                            if isinstance(outputs.get("pred_masks"), torch.Tensor)
                                            else "unknown"
                                        ),
                                        n_classes=n_classes,
                                        model_input_hw=tuple(images_t.shape[-2:]),
                                    )
                            except RuntimeError as oom_exc:
                                # OOM may surface as a non-OutOfMemoryError RuntimeError
                                # on this card (see oom.is_cuda_oom). (#208)
                                if not is_cuda_oom(oom_exc):
                                    raise
                                decision = ladder.on_oom()
                                if decision is OomDecision.RETRY_B:
                                    # Image set per forward changed: discard the buffer
                                    # and restart this image-chunk at the smaller B.
                                    restart_chunk = True
                                    break
                                if decision is OomDecision.RETRY_K:
                                    # Resume from the SAME class index at the smaller K_g
                                    # (recomputed at the top of the loop). Completed
                                    # K-groups' rows in chunk_buf stay valid.
                                    continue
                                if decision is OomDecision.FLOOR_RETRY:
                                    continue  # retry the same forward once
                                raise RuntimeError(
                                    "eval OOM at batch_size=1 and classes_per_forward=1; "
                                    "use a larger GPU or smaller image_size."
                                ) from None
                            for r in range(len(direct_chunk) * K_g):
                                ii, kk = divmod(r, K_g)
                                ex = direct_chunk[ii]
                                original_hw = (
                                    int(ex.image.shape[-2]),
                                    int(ex.image.shape[-1]),
                                )
                                int_id = _int_image_id(ex.image_id)
                                cat_idx = dataset.class_names.index(group[kk])
                                entries = queries_to_coco_results(
                                    _row_outputs(outputs, r),
                                    int_id,
                                    cat_idx + 1,
                                    original_hw,
                                    cfg.mask_threshold,
                                    max_dets=max_dets_cap,
                                )
                                chunk_buf.extend(entries)
                            j += K_g  # advance by the ACTUAL group length
                        if restart_chunk:
                            continue  # re-enter outer while at smaller B; i unchanged
                        # Completed every class group for this image-chunk: commit once.
                        predictions.extend(chunk_buf)

                    i += len(image_chunk)
                    img_idx_global += len(image_chunk)
                    for _ in range(len(image_chunk)):
                        sub.advance()
                    sub.update_postfix(it_s=float(img_idx_global))
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        return predictions

    def _iter_proxy_iou(
        self, model: Any, examples: Sequence[Example], dataset: Dataset
    ) -> tuple[list[ProxyEntry], int]:
        """Lite dense-IoU proxy forward loop (#269).

        Sibling of ``_iter_predictions`` sharing the SAME OomLadder + tiling /
        direct scaffold; the only difference is the per-(image, category) emit
        step, which builds a ``ProxyEntry`` (on-device IoU matmul) via the SHARED
        ``score_and_topk_filter`` / ``_upsample_mask_logits`` helpers instead of
        RLE-encoding COCO entries. Masks are therefore bit-identical to the exact
        path. One ProxyEntry is emitted per VISITED (image, category) even when
        there are zero preds or zero GT (so per-category ``npig`` is complete).

        Returns ``(entries, n_predictions)`` where ``n_predictions`` is the total
        survivor count across all entries (informational for the report).
        """
        cfg = self.cfg

        from custom_sam_peft.predict.tiling_preprocess import preprocess_tile

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()

        try:
            _p = next(model.parameters())
            param_device = _p.device
            param_dtype = _p.dtype
        except (StopIteration, AttributeError):
            param_device = torch.device("cpu")
            param_dtype = torch.float32
        eval_runtime = Runtime(device=param_device, dtype=param_dtype)

        n_classes = len(dataset.class_names)
        max_dets_cap = coco_max_dets_cap()
        ladder = OomLadder(
            micro_batch_size=int(cfg.batch_size),
            effective_K=min(MULTIPLEX_CAP, n_classes) if n_classes else 1,
        )
        pad_only_transform = getattr(dataset, "tile_transform", None)

        entries: list[ProxyEntry] = []

        def _emit(
            row_outputs: dict[str, torch.Tensor],
            instances: Sequence[Any],
            image_id: int,
            cat_idx: int,
            hw: tuple[int, int],
            crop: tuple[int, int, int, int] | None,
            into: list[ProxyEntry],
        ) -> None:
            # Build the CPU ProxyEntry NOW (the IoU matmul + .cpu() releases this
            # group's GPU forward output as the class-group loop advances) and
            # append it to ``into``. The direct path buffers into a chunk-local
            # list so an OOM RETRY_B can discard it wholesale without dropping or
            # duplicating committed groups — WITHOUT pinning every group's GPU
            # output for the whole chunk (the proxy's point is to stay GPU-light).
            gt_bin = _gt_masks_for_category(instances, cat_idx, hw, eval_runtime, crop)
            entry = _proxy_entry_from_outputs(
                row_outputs,
                gt_bin,
                image_id,
                cat_idx,
                hw,
                cfg.mask_threshold,
                max_dets_cap,
            )
            into.append(entry)

        try:
            with torch.no_grad(), P.push_subtask("eval", total=len(examples)) as sub:
                i = 0
                while i < len(examples):
                    bs = ladder.micro_batch_size
                    image_chunk = list(examples[i : i + bs])
                    direct_chunk: list[Example] = []
                    for ex in image_chunk:
                        orig_h = int(ex.image.shape[-2])
                        orig_w = int(ex.image.shape[-1])
                        if tiling_engaged(orig_h, orig_w):
                            windows = iter_windows(
                                orig_h, orig_w, tile=SAM3_IMAGE_SIZE, overlap=EVAL_OVERLAP
                            )
                            if ex.image_native is None:
                                raise ValueError(
                                    "eval tiling path requires Example.image_native "
                                    f"for image_id={ex.image_id!r}."
                                )
                            if pad_only_transform is None:
                                raise ValueError(
                                    "eval tiling path requires the dataset's pad-only "
                                    f"tile_transform; absent for image_id={ex.image_id!r}."
                                )
                            for t_idx, win in enumerate(windows):
                                tile_int_id = _tile_image_id(ex.image_id, t_idx)
                                tile_hw = (win.h, win.w)
                                crop = (win.y0, win.x0, win.h, win.w)
                                crop_np = ex.image_native[
                                    win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w
                                ]
                                tile_batch = preprocess_tile(
                                    crop_np,
                                    pad_only_transform,
                                    device=eval_runtime.device,
                                    dtype=eval_runtime.dtype,
                                ).unsqueeze(0)
                                j = 0
                                while j < n_classes:
                                    K_g = min(ladder.effective_K, n_classes - j)
                                    group = dataset.class_names[j : j + K_g]
                                    prompts_g = [TextPrompts(classes=list(group))]
                                    try:
                                        with profiling.bucket("eval.forward"):
                                            outputs = cast(
                                                "dict[str, torch.Tensor]",
                                                model(tile_batch, prompts_g, support=None),
                                            )
                                        profiling.incr("eval.forwards")
                                    except RuntimeError as oom_exc:
                                        if not is_cuda_oom(oom_exc):
                                            raise
                                        decision = ladder.on_oom()
                                        while decision is OomDecision.RETRY_B:
                                            decision = ladder.on_oom()
                                        if decision is OomDecision.RETRY_K:
                                            continue
                                        if decision is OomDecision.FLOOR_RETRY:
                                            continue
                                        raise RuntimeError(
                                            "eval OOM at classes_per_forward=1 on a"
                                            " single tile; use a larger GPU."
                                        ) from None
                                    for kk in range(K_g):
                                        cat_idx = dataset.class_names.index(group[kk])
                                        _emit(
                                            _row_outputs(outputs, kk),
                                            ex.instances,
                                            tile_int_id,
                                            cat_idx,
                                            tile_hw,
                                            crop,
                                            entries,
                                        )
                                    j += K_g
                        else:
                            direct_chunk.append(ex)

                    if direct_chunk:
                        images_t = to_device(
                            torch.stack([ex.image for ex in direct_chunk]), eval_runtime
                        )
                        chunk_buf: list[ProxyEntry] = []
                        restart_chunk = False
                        j = 0
                        while j < n_classes:
                            K_g = min(ladder.effective_K, n_classes - j)
                            group = dataset.class_names[j : j + K_g]
                            prompts_g = [TextPrompts(classes=list(group)) for _ in direct_chunk]
                            try:
                                with profiling.bucket("eval.forward"):
                                    outputs = cast(
                                        "dict[str, torch.Tensor]",
                                        model(images_t, prompts_g, support=None),
                                    )
                                profiling.incr("eval.forwards")
                            except RuntimeError as oom_exc:
                                if not is_cuda_oom(oom_exc):
                                    raise
                                decision = ladder.on_oom()
                                if decision is OomDecision.RETRY_B:
                                    restart_chunk = True
                                    break
                                if decision is OomDecision.RETRY_K:
                                    continue
                                if decision is OomDecision.FLOOR_RETRY:
                                    continue
                                raise RuntimeError(
                                    "eval OOM at batch_size=1 and classes_per_forward=1; "
                                    "use a larger GPU or smaller image_size."
                                ) from None
                            for r in range(len(direct_chunk) * K_g):
                                ii, kk = divmod(r, K_g)
                                ex = direct_chunk[ii]
                                original_hw = (
                                    int(ex.image.shape[-2]),
                                    int(ex.image.shape[-1]),
                                )
                                int_id = _int_image_id(ex.image_id)
                                cat_idx = dataset.class_names.index(group[kk])
                                # Build the entry NOW into the chunk-local buffer so
                                # this group's GPU output is freed as j advances; the
                                # buffer is committed once the chunk completes all
                                # class groups (mirrors the exact path's
                                # buffer-and-commit so an OOM RETRY_B drops nothing).
                                _emit(
                                    _row_outputs(outputs, r),
                                    ex.instances,
                                    int_id,
                                    cat_idx,
                                    original_hw,
                                    None,
                                    chunk_buf,
                                )
                            j += K_g
                        if restart_chunk:
                            continue
                        entries.extend(chunk_buf)

                    i += len(image_chunk)
                    for _ in range(len(image_chunk)):
                        sub.advance()
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        n_predictions = sum(int(e.scores.shape[0]) for e in entries)
        return entries, n_predictions

    def _aggregate_metrics(
        self,
        predictions: list[dict[str, object]],
        gt: COCO,
        dataset: Dataset,
    ) -> MetricsReport:
        """Compute a MetricsReport from raw predictions and ground-truth COCO data."""
        cfg = self.cfg

        if profiling.is_enabled():
            profiling.note(n_images=len(gt.imgs))
        with profiling.bucket("eval.coco_aggregate"):
            report = compute_coco_map(
                predictions=predictions,
                ground_truth=gt,
                iou_thresholds=cfg.iou_thresholds,
                include_per_class=(cfg.mode == "full"),
            )

        if cfg.mode == "full":
            skipped = sum(1 for name in dataset.class_names if name not in report.per_class)
            if skipped:
                _LOG.info(
                    "eval: skipped %d/%d classes with no GT instances",
                    skipped,
                    len(dataset.class_names),
                )

        return report

    def _maybe_save_predictions(
        self,
        preds: list[dict[str, object]],
        run_dir: Path | None,
        *,
        split: str = "val",
    ) -> None:
        """Write predictions to disk when configured and ``run_dir`` is given.

        Uses ``paths.predictions_path`` for the canonical output path.
        Skipped in lite mode regardless of ``cfg.save_predictions``.
        """
        if run_dir is None:
            return
        if not (self.cfg.save_predictions and self.cfg.mode == "full"):
            return
        out_path = predictions_path(run_dir, split=split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(preds))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: Literal[False] = False,
    ) -> MetricsReport:
        pass

    @overload
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: Literal[True],
    ) -> tuple[MetricsReport, list[float], list[int] | None]:
        pass

    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: bool = False,
    ) -> MetricsReport | tuple[MetricsReport, list[float], list[int] | None]:
        """Run the model over the dataset and return a MetricsReport.

        Pure compute — no disk I/O. Restores the model's training/eval state
        after the forward loop.

        Two metric backends, chosen by ``cfg.mode``:

        - ``mode == "full"`` (standalone / final report): the exact pycocotools
          RLE + COCOeval path. ``overall["mAP"]`` is true COCO mAP.
        - ``mode == "lite"`` (in-training validation): a fast GPU dense-IoU AP
          PROXY (#269). ``overall["mAP"]`` is a monotone-calibrated ranking
          proxy, NOT exact COCO mAP — its absolute units differ from full mode,
          so the lite and full ``mAP`` curves are not directly comparable. The
          proxy emits ``mAP`` (+ ``mAP_50`` / ``mAP_75`` iff those thresholds are
          in ``cfg.iou_thresholds``) so the in-loop control consumers
          (best-checkpoint selection and early-stop) are untouched.
          Setting the ``CSP_LITE_EXACT_MAP`` env var (profiling-style truthiness:
          any non-empty value other than ``"0"``/``"false"``/``"False"``) forces
          lite back through the exact COCO path — an escape hatch, not a config
          knob.

        When ``return_per_example_iou=True``, also returns a list of per-example
        MEAN IoU values across ``cfg.iou_thresholds`` aligned with dataset indices.
        Per-example IoU is an exact-path artifact (derived from the RLE
        predictions, used only for viz sample selection), so requesting it ALWAYS
        routes through the exact path — even in lite mode. The only caller that
        does so is the once-per-run close-out / standalone eval; the trainer's
        periodic in-loop eval does not, so it keeps the fast proxy. The default
        ``False`` preserves the previous return type for backward compatibility
        (e.g. `custom_sam_peft eval` CLI, mid-training eval).
        """
        # Reset predictions at the start so evaluate_and_save never writes
        # stale data from a prior call that may have failed mid-run.
        self._last_predictions = []

        cfg = self.cfg
        # eval.total wraps the whole forward+aggregate body so the serial
        # dataset-load fraction (eval.dataset_load) and the unbucketed residual
        # (per-chunk to_device, GT build) are visible against a real wall-time
        # denominator — the gate measurement for #265. Strict no-op when
        # CSP_PROFILE is unset.
        with profiling.bucket("eval.total"):
            n_total = len(dataset)
            n = n_total if cfg.mode == "full" else min(cfg.lite_max_images, n_total)
            # Serial single-threaded materialization of all examples before the
            # GPU forward loop — the prefetch target of #265. Timed separately
            # so its fraction of eval.total decides whether bounded prefetch
            # workers are worth adding (Amdahl-capped ~1.1-1.4x upside).
            with profiling.bucket("eval.dataset_load"):
                examples = [dataset[i] for i in range(n)]
            profiling.incr("eval.examples_loaded", by=n)

            # Implicit lite=proxy / full=exact split (#269): exact when full OR
            # when the CSP_LITE_EXACT_MAP escape hatch is set. Also exact whenever
            # return_per_example_iou is requested — per-example IoU is an
            # exact-path artifact (derived from the RLE predictions for viz sample
            # selection), and the only caller that requests it is the once-per-run
            # close-out / standalone eval, NOT the trainer's periodic in-loop eval.
            # So the proxy still covers the hot loop (the speedup that matters)
            # while close-out keeps its pre-#269 exact behaviour transparently.
            use_exact = cfg.mode == "full" or _lite_exact_map_hatch() or return_per_example_iou

            if not use_exact:
                # --- Lite dense-IoU AP proxy path (#269) ---
                proxy_entries, n_preds = self._iter_proxy_iou(model, examples, dataset)
                overall = proxy_map_from_iou(
                    proxy_entries, list(cfg.iou_thresholds), coco_max_dets_cap()
                )
                self._last_predictions = []
                return MetricsReport(
                    overall=overall,
                    per_class={},
                    n_images=len(examples),
                    n_predictions=n_preds,
                )

            # Use tiling-aware GT builder: large images are decomposed into
            # non-overlapping tiles (EVAL_OVERLAP=0.0); small images use the
            # direct path unchanged (spec §5.4).
            gt = _build_coco_gt_with_tiling(examples, dataset)

            predictions = self._iter_predictions(model, examples, dataset)
            report = self._aggregate_metrics(predictions, gt, dataset)
            self._maybe_save_predictions(predictions, run_dir=None)
            self._last_predictions = predictions

            if not return_per_example_iou:
                return report
            # For per_example_iou, build a direct-path GT (full-image) for the
            # IoU computation which works on full-image entries and is only used
            # for visualization sample selection — not the metric itself.
            gt_direct, _ = _build_coco_gt_from_examples(examples, dataset)
            gt_counts = [len(ex.instances) for ex in examples]
            per_example_iou = self._compute_per_example_iou(examples, predictions, gt_direct)
            return report, per_example_iou, gt_counts

    def _full_image_pred_rles(
        self,
        ex: Example,
        orig_h: int,
        orig_w: int,
        preds_by_image: dict[int, list[dict[str, object]]],
    ) -> list[Any]:
        """Return full-image predicted-mask RLEs for one example.

        Direct path: the example's predictions were stored under the full-image
        id, so their RLEs are already full-image — return them as-is.

        Tiling path: predictions were stored under TILE-LOCAL ids with masks in
        tile-local coordinates. Each tile-local prediction is placed onto its own
        zero-initialised (orig_h, orig_w) canvas at the window's (win.y0, win.x0)
        offset and emitted as a separate full-image RLE — one RLE per tile-level
        prediction, not one merged RLE per image. IoU evaluation later takes the
        union/max over these per-tile predictions, so correctness does NOT depend
        on tiles being strictly disjoint (overlap=0.0 is typical but not required).
        The windows are recomputed deterministically from the example dims —
        iter_windows is deterministic and matched the windows used during
        accumulation — so the tile_id -> window reverse map is exact.
        """
        if not tiling_engaged(orig_h, orig_w):
            int_id = _int_image_id(ex.image_id)
            return [p["segmentation"] for p in preds_by_image.get(int_id, [])]

        windows = iter_windows(orig_h, orig_w, tile=SAM3_IMAGE_SIZE, overlap=EVAL_OVERLAP)
        full_rles: list[Any] = []
        for t_idx, win in enumerate(windows):
            tile_id = _tile_image_id(ex.image_id, t_idx)
            for p in preds_by_image.get(tile_id, []):
                tile_mask = mask_utils.decode(p["segmentation"]).astype(bool)  # (win.h, win.w)
                canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
                canvas[win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w] = tile_mask[
                    : win.h, : win.w
                ]
                rle = mask_utils.encode(np.asfortranarray(canvas))
                rle["counts"] = rle["counts"].decode("ascii")
                full_rles.append(rle)
        return full_rles

    def _compute_per_example_iou(
        self,
        examples: Sequence[Example],
        predictions: list[dict[str, object]],
        gt: COCO,
    ) -> list[float]:
        """Compute mean IoU per example across self.cfg.iou_thresholds.

        The 'IoU' here is segmentation IoU between the best-matched predicted
        mask and any GT mask in the same image (greedy match, max IoU). For an
        example with no GT instances, IoU is 0.0 if it has predictions, else 1.0
        (vacuous match — consistent with COCO's empty-image handling). Examples
        skipped during model inference are marked NaN; pick_samples treats NaN
        as -inf for ranking and they are eligible only as 'worst' picks.

        Tiled examples (tiling_engaged True) have their predictions stored under
        tile-local ids; ``_full_image_pred_rles`` reconstructs the disjoint
        tile masks back onto a full-image canvas so the IoU is computed against
        the SAME full-image GT and with the IDENTICAL definition as the direct
        path (the only difference is where the predicted masks come from).
        """
        out: list[float] = []
        # Group predictions by image_id for cheap lookup. For tiled examples the
        # keys are tile-local ids; for direct examples they are full-image ids.
        preds_by_image: dict[int, list[dict[str, object]]] = {}
        for entry in predictions:
            preds_by_image.setdefault(int(entry["image_id"]), []).append(entry)  # type: ignore[call-overload]

        for ex in examples:
            int_id = _int_image_id(ex.image_id)
            orig_h, orig_w = int(ex.image.shape[-2]), int(ex.image.shape[-1])
            gt_anns = gt.imgToAnns.get(int_id, [])
            # Full-image predicted RLEs: as-stored on the direct path, reconstructed
            # from disjoint tiles on the tiling path.
            pred_rles = self._full_image_pred_rles(ex, orig_h, orig_w, preds_by_image)

            if not gt_anns and not pred_rles:
                out.append(1.0)  # vacuous match
                continue
            if not gt_anns or not pred_rles:
                out.append(0.0)
                continue

            # Build (n_pred, n_gt) IoU matrix for this example.
            gt_rles = [a["segmentation"] for a in gt_anns]
            iscrowd = [0] * len(gt_rles)
            with profiling.bucket("eval.pair_iou"):
                iou_mat = mask_utils.iou(pred_rles, gt_rles, iscrowd)
            # max-IoU greedy: for each GT, the best predicted IoU; mean over thresholds.
            # Spec §6.1: "the MEAN IoU across the eval's IoU thresholds [0.5, …, 0.95]".
            # We compute the per-GT best-pred IoU once, then average across thresholds:
            # at threshold t, the per-GT-IoU is the best-pred IoU if >= t else 0, so the
            # threshold-mean reduces to mean_t(best_iou >= t) which is the cdf at the
            # discrete thresholds. Use that as the example score.
            if iou_mat.size == 0:
                out.append(0.0)
                continue
            best_per_gt = np.asarray(iou_mat).max(axis=0)  # (n_gt,)
            thresholds = np.asarray(self.cfg.iou_thresholds)
            # Mean over (gt, thresholds) of (best_per_gt[g] >= thresholds[t]).
            hit = best_per_gt[:, None] >= thresholds[None, :]
            out.append(float(hit.mean()))

        return out

    def evaluate_and_save(self, model: Any, dataset: Dataset, output_dir: Path) -> MetricsReport:
        """Call ``evaluate``, write ``metrics.json``, and optionally save predictions.

        Predictions are written via ``_maybe_save_predictions`` using the canonical
        path from ``paths.predictions_path`` — only when ``cfg.save_predictions=True``
        AND ``cfg.mode == "full"``. In lite mode, predictions are never persisted
        regardless of ``cfg.save_predictions``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.evaluate(model, dataset)

        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "task": "instance",
                    "overall": report.overall,
                    "per_class": report.per_class,
                    "n_images": report.n_images,
                    "n_predictions": report.n_predictions,
                },
                indent=2,
            )
        )

        self._maybe_save_predictions(self._last_predictions, run_dir=output_dir)

        return report
