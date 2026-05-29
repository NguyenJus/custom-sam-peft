"""Eval GT-vs-Pred qualitative visualization (final/standalone eval path only).

Owns: variety-weighted image selection, config-aware denormalization, GT-instance
to render-entry conversion, the per-image matched render pair, the compositor, and
the top-level write_eval_visualizations entry point. Reuses predict/visualize.py for
the shared single-panel renderer, palette, and color map.

n-channel rule (§7.1): for inputs with more than 3 channels, only the first 3
denormalized channels are rendered as RGB (best-effort preview, not a faithful
multi-spectral visualization).

Spec: docs/superpowers/specs/2026-05-29-eval-visualize-design.md.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image, ImageDraw, ImageFont

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.base import Dataset, Example, Instance, TextPrompts
from custom_sam_peft.data.transforms import resolve_normalization
from custom_sam_peft.eval.evaluator import _row_outputs
from custom_sam_peft.eval.postprocess import queries_to_coco_results
from custom_sam_peft.models.matching import HungarianMatcher, meta_to_canonical
from custom_sam_peft.predict.visualize import color_for_class, render_overlay
from custom_sam_peft.runtime import Runtime, to_device

_LOG = logging.getLogger(__name__)


def _spread_indices(sorted_indices: list[int], k: int) -> list[int]:
    """Pick k evenly spaced elements from sorted_indices (preserving order)."""
    if k <= 0 or not sorted_indices:
        return []
    if k >= len(sorted_indices):
        return list(sorted_indices)
    # Evenly spaced positions across [0, len-1].
    positions = [round(j * (len(sorted_indices) - 1) / (k - 1)) for j in range(k)] if k > 1 else [0]
    seen: set[int] = set()
    out: list[int] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            out.append(sorted_indices[p])
    # Back-fill if rounding collided (keep k distinct positions when possible).
    j = 0
    while len(out) < k and j < len(sorted_indices):
        if sorted_indices[j] not in out:
            out.append(sorted_indices[j])
        j += 1
    return out


def pick_samples(
    per_example_iou: Sequence[float],
    dataset: Dataset,
    count: int,
) -> list[int]:
    """Return up to `count` dataset indices, variety-weighted toward high IoU.

    Filters to candidates with >=1 GT instance (excludes no-GT images), ranks by
    per_example_iou (NaN -> -inf, eligible only as 'worst'), and picks a
    good/median/worst spread per spec §5.3. Returns <= count indices when the
    candidate pool is smaller than count. Indices are returned in descending-IoU
    order so the written composites are filename-stable and roughly best-to-worst.
    """
    # Candidate filter: >=1 GT instance. per_example_iou is index-aligned to the
    # dataset slice the metrics pass evaluated (full or lite).
    candidates = [i for i in range(len(per_example_iou)) if len(dataset[i].instances) > 0]
    if not candidates:
        return []

    def rank_key(i: int) -> float:
        v = per_example_iou[i]
        return -math.inf if (v is None or math.isnan(v)) else float(v)

    ranked = sorted(candidates, key=rank_key, reverse=True)  # descending IoU

    if len(ranked) <= count:
        return ranked  # small-pool rule: take all, already descending

    good = round(0.5 * count)
    worst = min(2, max(1, round(0.2 * count)))
    median = count - good - worst

    n = len(ranked)
    good_slice = ranked[:good] if good > 0 else []
    worst_slice = ranked[n - worst :] if worst > 0 else []
    # Median band: the middle region between the good and worst slices.
    mid_lo = good
    mid_hi = n - worst
    median_pool = ranked[mid_lo:mid_hi]

    picked_good = _spread_indices(good_slice, good)
    picked_median = _spread_indices(median_pool, median)
    picked_worst = _spread_indices(worst_slice, worst)

    # Disjoint by construction (slices don't overlap). De-dup defensively and
    # back-fill from the next band if a band came up short.
    chosen: list[int] = []
    for idx in [*picked_good, *picked_median, *picked_worst]:
        if idx not in chosen:
            chosen.append(idx)
    if len(chosen) < count:
        for idx in ranked:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == count:
                break

    # Return in descending-IoU order.
    chosen.sort(key=rank_key, reverse=True)
    return chosen[:count]


def denormalize_to_rgb(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Image.Image:
    """Invert normalization and return a PIL RGB image (first 3 channels when C>3).

    pixel = normalized * std + mean, clamped to [0, 1], scaled to [0, 255], uint8,
    transposed (C, H, W) -> (H, W, C). For C>3 inputs only the first 3 channels are
    rendered as RGB (the corresponding first-3 mean/std are used).
    """
    c = image.shape[0]
    n = min(c, 3)
    chans = image[:n].float()
    m = torch.tensor([float(x) for x in mean[:n]]).view(n, 1, 1)
    s = torch.tensor([float(x) for x in std[:n]]).view(n, 1, 1)
    pixel = (chans * s + m).clamp(0.0, 1.0)
    arr = (pixel * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()  # (H, W, n)
    if n < 3:
        # Pad to 3 channels by repeating the last channel (e.g. grayscale -> RGB).
        arr = (
            np.repeat(arr[:, :, :1], 3, axis=2)
            if n == 1
            else np.concatenate([arr, arr[:, :, -1:].repeat(3 - n, axis=2)], axis=2)
        )
    return Image.fromarray(arr, mode="RGB")


def _mask_to_rle(mask: torch.Tensor) -> dict[str, object]:
    """(H, W) bool/uint8 mask -> pycocotools RLE dict with ASCII counts.

    Mirrors eval/postprocess.py::_logits_to_rle's encode + ascii-decode.
    """
    arr = np.asfortranarray(mask.cpu().numpy().astype(np.uint8))
    rle: dict[str, object] = mask_utils.encode(arr)
    counts = rle["counts"]
    rle["counts"] = counts.decode("ascii") if isinstance(counts, bytes) else counts
    return rle


def gt_instances_to_entries(instances: list[Instance]) -> list[dict[str, object]]:
    """Convert GT Instances to render_overlay entry dicts (no score key).

    category_id = class_id + 1 (1-indexed); bbox = xyxy -> xywh; segmentation = RLE
    of inst.mask. No `score` key (GT carries no score; the renderer labels the class
    name only).
    """
    entries: list[dict[str, object]] = []
    for inst in instances:
        x1, y1, x2, y2 = (float(v) for v in inst.box.tolist())
        entries.append(
            {
                "category_id": int(inst.class_id) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": _mask_to_rle(inst.mask),
            }
        )
    return entries


_TITLE_BAR_H = 18
_LEGEND_ROW_H = 16
_LEGEND_SWATCH = 12
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_image_id(image_id: str) -> str:
    """Replace any char outside [A-Za-z0-9._-] with '_' (path separators, ':',
    spaces, URL chars). Yields a single-segment, filesystem-safe filename stem."""
    return _SANITIZE_RE.sub("_", image_id)


def _compose_pair(
    gt_panel: Image.Image,
    pred_panel: Image.Image,
    *,
    class_names_present: list[str],
) -> Image.Image:
    """Hstack `Ground Truth | Prediction` with panel titles and a shared per-class
    color legend (the union of classes present in either panel). The same class is
    the same color in both panels because both call color_for_class."""
    font = ImageFont.load_default()
    panel_h = max(gt_panel.height, pred_panel.height)
    panel_w = gt_panel.width + pred_panel.width
    legend_h = _LEGEND_ROW_H * (len(class_names_present) + 1) if class_names_present else 0
    total_h = _TITLE_BAR_H + panel_h + legend_h
    canvas = Image.new("RGB", (panel_w, total_h), color=(255, 255, 255))

    # Titles.
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Ground Truth", fill=(0, 0, 0), font=font)
    draw.text((gt_panel.width + 4, 4), "Prediction", fill=(0, 0, 0), font=font)

    # Panels below the title bar.
    canvas.paste(gt_panel, (0, _TITLE_BAR_H))
    canvas.paste(pred_panel, (gt_panel.width, _TITLE_BAR_H))

    # Legend below the panels.
    if class_names_present:
        y = _TITLE_BAR_H + panel_h
        draw.text((4, y), "Legend:", fill=(0, 0, 0), font=font)
        y += _LEGEND_ROW_H
        for name in class_names_present:
            color = color_for_class(name)
            draw.rectangle([4, y, 4 + _LEGEND_SWATCH, y + _LEGEND_SWATCH], fill=color)
            draw.text((4 + _LEGEND_SWATCH + 4, y), name, fill=(0, 0, 0), font=font)
            y += _LEGEND_ROW_H
    return canvas


def _matched_pred_entries(
    model: Any,
    example: Example,
    class_names: list[str],
    *,
    mask_threshold: float,
    matcher: HungarianMatcher,
    runtime: Runtime,
) -> list[dict[str, object]]:
    """Per-class K=1 forward + mask-only Hungarian match; return the matched-query
    COCO entries (1:1 with GT masks) aggregated across all classes. Draws ONLY
    matched preds (no unmatched/extra detections).
    """
    h, w = int(example.image.shape[-2]), int(example.image.shape[-1])
    images_1 = to_device(example.image.unsqueeze(0), runtime)  # (1, C, H, W)
    out_entries: list[dict[str, object]] = []
    for cls_idx, class_name in enumerate(class_names):
        targets = [inst for inst in example.instances if int(inst.class_id) == cls_idx]
        if not targets:
            continue  # no GT for this class → nothing matched/drawn
        outputs = model(images_1, [TextPrompts(classes=[class_name])], support=None)
        canonical = meta_to_canonical(outputs)
        # matcher returns per-image [(query_idx, target_idx)]; one image here.
        query_idx, _target_idx = matcher(canonical, [targets])[0]
        # All-query COCO entries for this class, then keep only matched query rows.
        all_entries = queries_to_coco_results(
            _row_outputs(outputs, 0),
            0,  # image_id is irrelevant for rendering (entries are per-image)
            cls_idx + 1,
            (h, w),
            mask_threshold,
        )
        for q in query_idx.tolist():
            if 0 <= q < len(all_entries):
                out_entries.append(all_entries[q])
    return out_entries


def render_eval_pair(
    model: Any,
    example: Example,
    class_names: list[str],
    *,
    mask_threshold: float,
    mean: Sequence[float],
    std: Sequence[float],
    matcher: HungarianMatcher,
) -> Image.Image:
    """Return the hstacked `Ground Truth | Prediction` composite for one image.

    GT panel: denormalized source + GT instance overlays (no score). Pred panel:
    denormalized source + the Hungarian mask-only matched 1:1 preds per class,
    aggregated across classes (matched preds only). Both panels use the same
    color_for_class mapping so a class is the same color in both.
    """
    try:
        param_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        param_device = torch.device("cpu")
    runtime = Runtime(device=param_device, dtype=torch.float32)

    source = denormalize_to_rgb(example.image, mean, std)

    gt_entries = gt_instances_to_entries(example.instances)
    gt_panel = render_overlay(source, gt_entries, prompts=class_names)

    pred_entries = _matched_pred_entries(
        model,
        example,
        class_names,
        mask_threshold=mask_threshold,
        matcher=matcher,
        runtime=runtime,
    )
    pred_panel = render_overlay(source, pred_entries, prompts=class_names)

    # Legend = union of classes present in either panel.
    present_ids = {
        int(e["category_id"])
        for e in (*gt_entries, *pred_entries)
        if isinstance(e["category_id"], (int, float))
    }
    names_present = [class_names[c - 1] for c in sorted(present_ids) if 0 < c <= len(class_names)]
    return _compose_pair(gt_panel, pred_panel, class_names_present=names_present)


def write_eval_visualizations(
    model: Any,
    dataset: Dataset,
    output_dir: Path,
    *,
    per_example_iou: Sequence[float],
    count: int,
    mask_threshold: float,
    model_name: str,
    normalize: NormalizeConfig | None,
    channel_semantics: str,
) -> list[Path]:
    """Phase-2 viz pass. Selects `count` variety-weighted images (§5), renders a
    GT-vs-Pred composite per image (§7.4-7.5), writes PNGs under
    output_dir/visualizations/, and returns the written paths. Memory-bounded:
    processes and frees one image at a time. Per-image failures are caught and
    logged at WARNING; never raises for a single bad image.
    """
    selected = pick_samples(per_example_iou, dataset, count)
    if not selected:
        _LOG.info("eval visualize: no GT-bearing images to visualize; skipping.")
        return []

    mean, std = resolve_normalization(
        model_name,
        normalize,  # type: ignore[arg-type]
        channel_semantics=channel_semantics,
    )
    w = MatcherWeights()
    matcher = HungarianMatcher(
        lambda_l1=w.lambda_l1, lambda_giou=w.lambda_giou, lambda_mask=w.lambda_mask
    )

    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    written: list[Path] = []
    try:
        with torch.no_grad():
            for idx in selected:
                example = None
                try:
                    example = dataset[idx]
                    composite = render_eval_pair(
                        model,
                        example,
                        list(dataset.class_names),
                        mask_threshold=mask_threshold,
                        mean=mean,
                        std=std,
                        matcher=matcher,
                    )
                    out_path = vis_dir / f"{_sanitize_image_id(example.image_id)}.png"
                    composite.save(out_path)
                    written.append(out_path)
                except Exception:
                    image_id = example.image_id if example is not None else "<unavailable>"
                    _LOG.warning(
                        "eval visualize: failed to render image_id=%r (idx=%d); skipping.",
                        image_id,
                        idx,
                        exc_info=True,
                    )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    return written
