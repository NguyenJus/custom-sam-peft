"""Results bundler — writes ``runs/<id>/summary.md`` + ``samples/*.png``.

Three public functions in dependency order:

1. ``pick_samples`` — pure: index selection from per-example IoU + overall mAP.
2. ``render_overlay`` — pure: image + pred/gt masks → PIL image with caption.
3. ``write_bundle`` — composes the above, runs per-sample re-inference, writes disk.

The orchestrator (``custom_sam_peft run``) assembles a frozen ``BundleContext`` and
calls ``write_bundle(ctx, …)``. The notebook does not import this module.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §6.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_sam_peft.data.base import Dataset, Example, TextPrompts
from custom_sam_peft.eval.metrics import MetricsReport

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BundleContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleContext:
    """All run-context fields the bundler needs, assembled by `custom_sam_peft run`."""

    run_dir: Path
    config_path: Path
    start_ts: datetime
    end_ts: datetime
    preset_label: str | None
    per_example_iou: list[float]
    merged_dir: Path | None
    merged_export_error: str | None


# ---------------------------------------------------------------------------
# pick_samples
# ---------------------------------------------------------------------------


def _bracket(mAP: float) -> tuple[int, int, int]:
    """Return (best, median, worst) triple for the spec brackets at n_val=6."""
    if math.isnan(mAP) or mAP < 0.4:
        return (1, 1, 4)
    if mAP < 0.7:
        return (2, 2, 2)
    return (4, 1, 1)


def _score(per_example_iou: list[float]) -> list[float]:
    """Replace NaN with -inf for ranking (NaN sorts as worst)."""
    return [(-math.inf if math.isnan(x) else x) for x in per_example_iou]


def pick_samples(
    per_example_iou: list[float],
    overall_mAP: float,
    n_val: int,
) -> list[int]:
    """Pick up to 6 val indices to render, partitioned by bracket.

    Returns the concatenation (best…, median…, worst…). Tie-break by index asc.
    See spec §6.1 for the bracket table.
    """
    if n_val == 0:
        return []
    if len(per_example_iou) != n_val:
        raise ValueError(f"len(per_example_iou)={len(per_example_iou)} != n_val={n_val}")

    cap = min(6, n_val)
    b, m, w = _bracket(overall_mAP)

    if cap < 6:
        ratios = [b / 6.0, m / 6.0, w / 6.0]
        picks = [int(r * cap) for r in ratios]
        while sum(picks) < cap:
            picks[2] += 1  # top up with 'worst'
        b, m, w = picks

    scores = _score(per_example_iou)
    nan_count = sum(1 for x in per_example_iou if math.isnan(x))
    if nan_count:
        _LOG.warning("bundle: %d val examples had NaN IoU; treated as worst", nan_count)

    indexed = list(enumerate(scores))
    by_desc = sorted(indexed, key=lambda kv: (-kv[1], kv[0]))  # best
    by_asc = sorted(indexed, key=lambda kv: (kv[1], kv[0]))  # worst

    best_idx = [i for i, _ in by_desc[:b]]
    best_set = set(best_idx)
    # Exclude best indices when picking worst so ties don't double-count.
    worst_candidates = [(i, s) for i, s in by_asc if i not in best_set]
    worst_idx = [i for i, _ in worst_candidates[:w]]
    used = set(best_idx) | set(worst_idx)

    finite = [s for s in scores if math.isfinite(s)]
    median = float(np.median(finite)) if finite else 0.0

    # median: closest-to-median by |score - median|, excluding used + NaN-only.
    eligible = [(i, s) for i, s in indexed if i not in used and math.isfinite(s)]
    eligible.sort(key=lambda kv: (abs(kv[1] - median), kv[0]))
    median_idx = [i for i, _ in eligible[:m]]

    # If median is short (e.g. all-finite eligibility exhausted), fall back to
    # any remaining unused indices in ascending order.
    if len(median_idx) < m:
        remaining = [i for i, _ in indexed if i not in used and i not in median_idx]
        median_idx.extend(remaining[: m - len(median_idx)])

    return best_idx + median_idx + worst_idx


# ---------------------------------------------------------------------------
# render_overlay
# ---------------------------------------------------------------------------


_PRED_RGBA = (255, 0, 255, 96)
_GT_RGBA = (0, 255, 255, 96)


def render_overlay(
    image: Image.Image,
    predicted_mask: np.ndarray[Any, np.dtype[np.bool_]],
    ground_truth_mask: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    caption: str,
) -> Image.Image:
    """Return a single PNG-able RGB image with prediction + GT overlaid.

    Visual contract:
      - Prediction in semi-transparent magenta (255, 0, 255, 96).
      - GT in semi-transparent cyan (0, 255, 255, 96).
      - Caption text at the bottom-left, white on a black 50%-opacity strip.
    """
    expected_hw = (image.size[1], image.size[0])  # PIL is (W, H); numpy is (H, W)
    if predicted_mask.shape != expected_hw or ground_truth_mask.shape != expected_hw:
        raise ValueError(
            f"mask shape mismatch: image={expected_hw}, "
            f"pred={predicted_mask.shape}, gt={ground_truth_mask.shape}"
        )

    base = image.convert("RGBA")
    pred_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gt_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    pred_pixels = np.zeros((expected_hw[0], expected_hw[1], 4), dtype=np.uint8)
    pred_pixels[predicted_mask.astype(bool)] = _PRED_RGBA
    pred_layer = Image.fromarray(pred_pixels, mode="RGBA")

    gt_pixels = np.zeros((expected_hw[0], expected_hw[1], 4), dtype=np.uint8)
    gt_pixels[ground_truth_mask.astype(bool)] = _GT_RGBA
    gt_layer = Image.fromarray(gt_pixels, mode="RGBA")

    composed = Image.alpha_composite(Image.alpha_composite(base, gt_layer), pred_layer)

    draw = ImageDraw.Draw(composed, mode="RGBA")
    text_w = max(60, min(base.size[0], 8 * len(caption)))
    strip_h = min(16, max(1, base.size[1] // 4))
    y0 = base.size[1] - strip_h
    draw.rectangle([(0, y0), (text_w, y0 + strip_h)], fill=(0, 0, 0, 128))
    draw.text((4, y0 + 2), caption, fill=(255, 255, 255, 255))

    return composed.convert("RGB")


# ---------------------------------------------------------------------------
# write_bundle
# ---------------------------------------------------------------------------


def _reinfer_one_example(
    model_wrapper: Any,
    val_dataset: Dataset,
    idx: int,
) -> tuple[Image.Image, np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Re-run inference for a single example and return (image, pred, gt).

    `image`: source image as RGB PIL (already-resized, the same view fed into
    the model). `pred`: HxW bool (model's binarized mask, union over GT classes
    for that example). `gt`: HxW bool (union of all GT instance masks for the
    example). Raises if the model forward errors — caught one level up.
    """
    ex: Example = val_dataset[idx]
    # The collator/wrapper expects a batched image and one TextPrompts per image.
    classes = list(getattr(val_dataset, "class_names", []))
    if not classes:
        raise RuntimeError(f"val_dataset has no class_names; cannot prompt example {idx}")

    image_chw = ex.image  # (3, H, W) normalized — already on the model's device path
    h, w = int(image_chw.shape[-2]), int(image_chw.shape[-1])
    with torch.no_grad():
        outputs = model_wrapper(
            image_chw.unsqueeze(0),
            [TextPrompts(classes=classes)],
            box_hints=None,
        )
    # Outputs include `pred_masks` of shape (1, Q, H, W) — take union over queries
    # thresholded at 0.0 (same as Evaluator's default).
    pred_masks_logits = outputs["pred_masks"][0]
    pred_union = (pred_masks_logits > 0.0).any(dim=0).cpu().numpy().astype(bool)

    gt_union = np.zeros((h, w), dtype=bool)
    for inst in ex.instances:
        m = inst.mask.cpu().numpy().astype(bool)
        # Pad/truncate to the expected (h, w) if necessary; trust the dataset.
        gt_union |= m

    # Source image — denormalize to display-friendly RGB.
    arr = image_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(arr, mode="RGB")
    return image_pil, pred_union, gt_union


def _bracket_label(picks: list[int], composition: tuple[int, int, int]) -> list[str]:
    """Return per-index bracket label aligned with `picks`."""
    b, m, _w = composition
    labels: list[str] = []
    for i in range(len(picks)):
        if i < b:
            labels.append("best")
        elif i < b + m:
            labels.append("median")
        else:
            labels.append("worst")
    return labels


def _format_duration(start: datetime, end: datetime) -> str:
    delta = end - start
    secs = int(delta.total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _hardware_lines() -> tuple[str, float | None]:
    if not torch.cuda.is_available():
        return "(no CUDA)", None
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / (1024**3)


def write_bundle(
    ctx: BundleContext,
    metrics_report: MetricsReport,
    val_dataset: Dataset,
    model_wrapper: Any,
) -> None:
    """Write `ctx.run_dir/summary.md` and `ctx.run_dir/samples/*.png`.

    Idempotent: re-runs overwrite. Failure modes:
      - Per-sample inference raises → that PNG is skipped; WARNING logged;
        "skipped samples" note in summary.md. Bundle does not abort.
      - All other errors propagate.
    """
    samples_dir = ctx.run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale samples from prior runs.
    for stale in samples_dir.glob("*.png"):
        stale.unlink()

    mAP = float(metrics_report.overall.get("mAP", float("nan")))
    n_val = len(val_dataset)
    indices = pick_samples(ctx.per_example_iou, mAP, n_val)
    composition = _bracket(mAP)
    if n_val < 6 and n_val > 0:
        # Re-derive the prorated composition for caption purposes.
        ratios = [composition[0] / 6.0, composition[1] / 6.0, composition[2] / 6.0]
        picks = [int(r * n_val) for r in ratios]
        while sum(picks) < n_val:
            picks[2] += 1
        composition = (picks[0], picks[1], picks[2])

    edge_notes: list[str] = []
    if n_val == 0:
        edge_notes.append("empty val: no samples rendered (n_val == 0)")
    elif n_val < 6:
        edge_notes.append(
            f"capped: n_val={n_val} < 6 → rendered {len(indices)} samples per prorated composition"
        )
    if math.isnan(mAP):
        edge_notes.append("NaN mAP: classified as 'poor' bracket")
    if ctx.merged_export_error is not None:
        edge_notes.append(f"export-merge failed: {ctx.merged_export_error} — bundle continued")

    skipped: list[tuple[int, str]] = []
    labels = _bracket_label(indices, composition)
    per_bracket_ordinal: dict[str, int] = {"best": 0, "median": 0, "worst": 0}
    sample_filenames: list[str] = []

    for picked_idx, bracket in zip(indices, labels, strict=True):
        ordinal = per_bracket_ordinal[bracket]
        per_bracket_ordinal[bracket] += 1
        iou = ctx.per_example_iou[picked_idx]
        caption = f"{bracket} @ IoU={iou:.2f}"
        try:
            image, pred, gt = _reinfer_one_example(model_wrapper, val_dataset, picked_idx)
            png = render_overlay(image, pred, gt, caption=caption)
            fname = f"{ordinal}_{bracket}.png"
            png.save(samples_dir / fname)
            sample_filenames.append(fname)
        except Exception as exc:
            _LOG.warning("bundle: skipped sample idx=%d (%s): %s", picked_idx, bracket, exc)
            skipped.append((picked_idx, type(exc).__name__))

    if skipped:
        details = ", ".join(f"{i} raised {cls}" for i, cls in skipped)
        edge_notes.append(f"skipped samples: {details} — see log")

    # ---- summary.md -----------------------------------------------------
    headline = f"# {ctx.config_path.parent.name} — {mAP:.4f}"
    gpu_name, vram_gb = _hardware_lines()
    vram_line = f"- VRAM: {vram_gb:.1f} GB" if vram_gb is not None else "- VRAM: (n/a)"
    preset_line = f"- Applied: {ctx.preset_label or 'manual'}"

    adapter_path = (ctx.run_dir / "adapter").resolve()
    try:
        adapter_rel = adapter_path.relative_to(ctx.run_dir.resolve())
    except ValueError:
        adapter_rel = adapter_path

    if ctx.merged_export_error is not None:
        merged_line = f"- Merged:  FAILED — {ctx.merged_export_error} — see logs"
    elif ctx.merged_dir is None:
        merged_line = "- Merged:  skipped (cfg.export.merge=false)"
    else:
        try:
            merged_rel = ctx.merged_dir.resolve().relative_to(ctx.run_dir.resolve())
            merged_line = f"- Merged:  {merged_rel}"
        except ValueError:
            merged_line = f"- Merged:  {ctx.merged_dir}"

    samples_md = "\n".join(f"![{fn}](samples/{fn})" for fn in sample_filenames)
    edges_md = "\n".join(f"- {line}" for line in edge_notes) if edge_notes else ""

    config_rel = ctx.config_path.name

    body = (
        f"{headline}\n\n"
        f"## Run\n"
        f"- Start:  {ctx.start_ts.isoformat()}\n"
        f"- End:    {ctx.end_ts.isoformat()}\n"
        f"- Duration: {_format_duration(ctx.start_ts, ctx.end_ts)}\n\n"
        f"## Hardware\n"
        f"- GPU:  {gpu_name}\n"
        f"{vram_line}\n\n"
        f"## Preset\n"
        f"{preset_line}\n\n"
        f"## Outputs\n"
        f"- Adapter: {adapter_rel}\n"
        f"{merged_line}\n"
        f"- Config:  {config_rel}\n\n"
        f"## Samples\n"
        f"{samples_md}\n"
    )
    if edges_md:
        body += f"\n## Edge cases\n{edges_md}\n"

    (ctx.run_dir / "summary.md").write_text(body)
