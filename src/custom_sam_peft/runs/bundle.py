"""Results bundler — writes ``runs/<id>/summary.md`` + ``samples/*.png``.

Four public functions in dependency order:

1. ``run_export`` — library: load adapter, export adapter or merged model to disk.
2. ``pick_samples`` — pure: index selection from per-example IoU + overall mAP.
3. ``render_overlay`` — pure: image + pred/gt masks → PIL image with caption.
4. ``write_bundle`` — composes the above, runs per-sample re-inference, writes disk.

The orchestrator (``custom_sam_peft run``) assembles a frozen ``BundleContext`` and
calls ``write_bundle(ctx, …)``. The notebook does not import this module.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §6.
"""

from __future__ import annotations

import json
import logging
import math
import zipfile as _zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_sam_peft import paths
from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.data.base import Dataset, Example, TextPrompts
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.peft_adapters import method_pretty_name
from custom_sam_peft.presets import PresetDecision
from custom_sam_peft.train.types import OomEvent

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# run_export
# ---------------------------------------------------------------------------


def run_export(
    cfg: TrainConfig,
    checkpoint: Path,
    *,
    merge: bool = False,
    output: Path | None = None,
) -> Path:
    """Load an adapter from *checkpoint* and export it.

    When *merge* is True, LoRA deltas are folded into the base weights and the
    merged ``state_dict`` is written to *output* (default: ``<run_dir>/merged``).
    When *merge* is False, the raw adapter files are copied to *output*
    (required when not merging, to avoid overwriting the source).

    Returns the path to the written output directory.
    """
    from custom_sam_peft.models.sam3 import load_sam31
    from custom_sam_peft.train.checkpoint import load_adapter, save_adapter, save_merged

    run_dir = checkpoint.parent
    wrapper = load_sam31(cfg.model)
    load_adapter(wrapper, checkpoint)

    if merge:
        out = output if output is not None else (run_dir / "merged")
        save_merged(wrapper, out)
    else:
        if output is None:
            raise ValueError(
                "output is required when not merging (refusing to overwrite source checkpoint)"
            )
        out = output
        save_adapter(wrapper, out)

    return out


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
    preset: PresetDecision
    per_example_iou: list[float]
    merged_dir: Path | None
    merged_export_error: str | None
    oom_events: tuple[OomEvent, ...]


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


def _preset_block(preset: PresetDecision) -> str:
    ckpt_word = "on" if preset.gradient_checkpointing else "off"
    method_pretty = method_pretty_name(preset.method)
    used_gib = preset.predicted_bytes / (1024**3)
    total_gib = (preset.budget_bytes + preset.headroom_bytes) / (1024**3)
    headroom_gib = preset.headroom_bytes / (1024**3)
    if preset.provenance == "calibrated":
        date_str = preset.calibrated_at[:10] if preset.calibrated_at else "unknown"
        cache_name = Path(preset.cache_path).name if preset.cache_path else "(unknown)"
        source_line = f"- Source: calibrated {date_str} (cache: {cache_name})"
    else:
        source_line = "- Source: analytic estimate"
    return (
        f"- Method: {method_pretty} r={preset.r}, batch={preset.batch_size}, "
        f"grad_accum={preset.grad_accum_steps}, gradient_checkpointing={ckpt_word}, bf16\n"
        f"- GPU:    {preset.gpu_name} ({total_gib:.1f} GiB)\n"
        f"- Budget: {used_gib:.1f} / {total_gib:.1f} GiB used ({headroom_gib:.1f} GiB headroom)\n"
        f"{source_line}"
    )


def _oom_edge_note(events: tuple[OomEvent, ...]) -> str | None:
    """Return the OOM-summary line for `## Edge cases`, or None when there were none."""
    if not events:
        return None
    final_mb = events[-1].new_micro_batch_size
    ckpt_event = next((e for e in events if e.action == "grad_ckpt_enabled"), None)
    base = f"OOM retries: {len(events)} — final micro_batch={final_mb}"
    if ckpt_event is not None:
        base += f", gradient_checkpointing enabled at step {ckpt_event.step}"
    return base


def _write_summary_no_val(ctx: BundleContext) -> None:
    """Spec §7.5: write summary.md only; no samples directory.

    Headline is `# <run-name> — no-val` instead of `# <run-name> — <mAP>`.
    """
    gpu_name, vram_gb = _hardware_lines()
    vram_line = f"- VRAM: {vram_gb:.1f} GB" if vram_gb is not None else "- VRAM: (n/a)"

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

    config_rel = ctx.config_path.name

    headline = f"# {ctx.config_path.parent.name} — no-val"
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
        f"{_preset_block(ctx.preset)}\n\n"
        f"## Outputs\n"
        f"- Adapter: {adapter_rel}\n"
        f"{merged_line}\n"
        f"- Config:  {config_rel}\n\n"
        f"## Validation\n"
        f"No validation set; this run did not produce mAP or per-example IoU.\n"
        f"Tracker scalars and training-loss curve are at the configured TB run dir.\n"
    )
    edge_lines: list[str] = []
    if ctx.merged_export_error is not None:
        edge_lines.append(f"- export-merge failed: {ctx.merged_export_error}")
    oom_line = _oom_edge_note(ctx.oom_events)
    if oom_line is not None:
        edge_lines.append(f"- {oom_line}")
    if edge_lines:
        body += "\n## Edge cases\n" + "\n".join(edge_lines) + "\n"

    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    (ctx.run_dir / "summary.md").write_text(body)


def _collect_artifacts(
    ctx: BundleContext,
    metrics_report: MetricsReport | None,
    val_dataset: Dataset | None,
    model_wrapper: Any,
) -> list[Path]:
    """Render sample overlays + write ``summary.md``; return all artifact paths.

    No-val mode: when val_dataset is None, writes summary.md only with the
    "no-val" headline and skips the samples/ directory.

    Artifacts collected (in order):
      - ``summary.md`` at ``ctx.run_dir``
      - Per-sample PNGs under ``ctx.run_dir/samples/``

    Idempotent: re-runs overwrite. Failure modes:
      - Per-sample inference raises → that PNG is skipped; WARNING logged;
        "skipped samples" note in summary.md. Collection does not abort.
      - All other errors propagate.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.5.
    """
    if val_dataset is None:
        _write_summary_no_val(ctx)
        return []
    samples_dir = ctx.run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale samples from prior runs.
    for stale in samples_dir.glob("*.png"):
        stale.unlink()

    assert metrics_report is not None  # noqa: S101 — val_dataset present implies report present
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

    oom_line = _oom_edge_note(ctx.oom_events)
    if oom_line is not None:
        edge_notes.append(oom_line)

    # ---- summary.md -----------------------------------------------------
    headline = f"# {ctx.config_path.parent.name} — {mAP:.4f}"
    gpu_name, vram_gb = _hardware_lines()
    vram_line = f"- VRAM: {vram_gb:.1f} GB" if vram_gb is not None else "- VRAM: (n/a)"
    preset_block = _preset_block(ctx.preset)

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
        f"{preset_block}\n\n"
        f"## Outputs\n"
        f"- Adapter: {adapter_rel}\n"
        f"{merged_line}\n"
        f"- Config:  {config_rel}\n\n"
        f"## Samples\n"
        f"{samples_md}\n"
    )
    if edges_md:
        body += f"\n## Edge cases\n{edges_md}\n"

    summary_path = ctx.run_dir / "summary.md"
    summary_path.write_text(body)

    artifacts: list[Path] = [summary_path]
    artifacts.extend(samples_dir / fn for fn in sample_filenames)
    return artifacts


def _write_manifest(run_dir: Path, artifacts: list[Path]) -> Path:
    """Write a JSON manifest of *artifacts* to the artifacts sub-directory.

    The manifest lists each artifact as a relative path from *run_dir*. Returns
    the path to the written manifest file.
    """
    manifest_path = paths.artifact_path(run_dir, name="manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in artifacts:
        try:
            entries.append(str(p.relative_to(run_dir)))
        except ValueError:
            entries.append(str(p))
    manifest_path.write_text(json.dumps({"artifacts": entries}, indent=2))
    return manifest_path


def _zip_bundle(run_dir: Path, artifacts: list[Path], manifest: Path) -> Path:
    """Zip *artifacts* + *manifest* into ``bundle_path(run_dir)``.

    All files are stored relative to *run_dir* so the archive is portable.
    Returns the path to the written zip file.
    """
    out = paths.bundle_path(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_files = [*list(artifacts), manifest]
    with _zipfile.ZipFile(out, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
        for p in all_files:
            if not p.exists():
                continue
            try:
                arcname = str(p.relative_to(run_dir))
            except ValueError:
                arcname = p.name
            zf.write(p, arcname)
    _LOG.info("bundle: wrote %s (%d files)", out, len(zf.namelist()))
    return out


def write_bundle(
    ctx: BundleContext,
    metrics_report: MetricsReport | None,
    val_dataset: Dataset | None,
    model_wrapper: Any,
) -> None:
    """Write sample overlays, ``summary.md``, a manifest, and a zip bundle.

    No-val mode: when val_dataset is None, writes summary.md only with the
    "no-val" headline and skips the samples/ directory.

    Idempotent: re-runs overwrite. See ``_collect_artifacts`` for failure-mode
    semantics (per-sample errors are warned and skipped; other errors propagate).

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §7.5.
    """
    artifacts = _collect_artifacts(ctx, metrics_report, val_dataset, model_wrapper)
    manifest = _write_manifest(ctx.run_dir, artifacts)
    _zip_bundle(ctx.run_dir, artifacts, manifest)
