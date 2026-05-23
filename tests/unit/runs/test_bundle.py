"""Tests for src/custom_sam_peft/runs/bundle.py."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from custom_sam_peft.presets import PresetDecision
from custom_sam_peft.runs.bundle import (
    BundleContext,
    pick_samples,
    render_overlay,
    write_bundle,
)
from custom_sam_peft.train.types import OomEvent

# ---- pick_samples --------------------------------------------------------


@pytest.mark.parametrize(
    "mAP, ious, expected_composition",
    [
        # mAP >= 0.7 → 4 best + 1 median + 1 worst (n_val=6)
        (0.80, [0.1, 0.4, 0.5, 0.6, 0.85, 0.95], (4, 1, 1)),
        # 0.4 <= mAP < 0.7 → 2 best + 2 median + 2 worst (n_val=6)
        (0.50, [0.1, 0.2, 0.4, 0.5, 0.85, 0.9], (2, 2, 2)),
        # mAP < 0.4 → 1 best + 1 median + 4 worst (n_val=6)
        (0.10, [0.0, 0.1, 0.2, 0.3, 0.5, 0.95], (1, 1, 4)),
        # NaN mAP → 'poor' bracket
        (float("nan"), [0.0, 0.1, 0.2, 0.3, 0.5, 0.95], (1, 1, 4)),
    ],
)
def test_pick_samples_brackets_at_n6(
    mAP: float, ious: list[float], expected_composition: tuple[int, int, int]
) -> None:
    picks = pick_samples(ious, mAP, n_val=6)
    assert len(picks) == 6
    # Slot-bucketing tracks the construction order — best…, median…, worst….
    b, m, w = expected_composition
    assert b + m + w == 6
    # Verify the highest-IoU index is in the 'best' slot region.
    best_region = picks[:b]
    worst_region = picks[b + m :]
    assert max(ious) in [ious[i] for i in best_region]
    assert min(ious) in [ious[i] for i in worst_region]


def test_pick_samples_empty_returns_empty() -> None:
    assert pick_samples([], 0.5, n_val=0) == []


def test_pick_samples_n_lt_6_caps_and_topups_with_worst() -> None:
    # n_val=2, mAP=0.5 (mid bracket 2/2/2) → cap 2; floor 0/0/0 → topup worst → (0, 0, 2)
    picks = pick_samples([0.7, 0.1], 0.5, n_val=2)
    assert len(picks) == 2
    # Both indices land in 'worst' (idx asc tiebreak).
    assert picks == [1, 0]  # 0.1 < 0.7 so idx 1 first


def test_pick_samples_n_val_1_landed_in_worst_for_poor_bracket() -> None:
    picks = pick_samples([0.42], 0.1, n_val=1)
    assert picks == [0]


def test_pick_samples_identical_ious_tiebreak_by_index_asc() -> None:
    picks = pick_samples([0.5] * 6, 0.5, n_val=6)
    # All identical → best/worst sort stable by index asc; median fills the remainder.
    assert sorted(picks) == [0, 1, 2, 3, 4, 5]


def test_pick_samples_all_zero_ious() -> None:
    picks = pick_samples([0.0] * 6, 0.0, n_val=6)  # 'poor' bracket
    assert sorted(picks) == [0, 1, 2, 3, 4, 5]


def test_pick_samples_nan_iou_treated_as_minus_inf() -> None:
    # idx 2 is NaN; should NOT appear in 'best' but must be eligible for 'worst'.
    ious = [0.9, 0.5, float("nan"), 0.1, 0.2, 0.6]
    picks = pick_samples(ious, 0.10, n_val=6)
    # Poor bracket = 1 best + 1 median + 4 worst → best is idx 0 (0.9).
    assert picks[0] == 0
    # NaN-IoU index must be present in worst region (treated as -inf, sorts as worst).
    assert 2 in picks[2:]


def test_pick_samples_ordering_best_then_median_then_worst() -> None:
    ious = [0.9, 0.4, 0.5, 0.3, 0.1, 0.85]
    picks = pick_samples(ious, 0.50, n_val=6)  # 2/2/2 bracket
    # First 2 = best (sorted desc by IoU, ties by idx asc): 0.9 → idx 0, 0.85 → idx 5.
    assert picks[:2] == [0, 5]
    # Last 2 = worst (sorted asc): 0.1 → 4, 0.3 → 3.
    assert picks[4:] == [4, 3]


# ---- render_overlay ------------------------------------------------------


def test_render_overlay_returns_rgb_image_of_input_size() -> None:
    img = Image.new("RGB", (32, 24), (10, 10, 10))
    pred = np.zeros((24, 32), dtype=bool)
    pred[:, :16] = True
    gt = np.zeros((24, 32), dtype=bool)
    gt[:, 16:] = True
    out = render_overlay(img, pred, gt, caption="best @ IoU=0.83")
    assert out.mode == "RGB"
    assert out.size == img.size


def test_render_overlay_recolours_prediction_and_gt() -> None:
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    pred = np.zeros((16, 16), dtype=bool)
    pred[:, :8] = True  # left half = prediction (magenta)
    gt = np.zeros((16, 16), dtype=bool)
    gt[:, 8:] = True  # right half = GT (cyan)
    out = render_overlay(img, pred, gt, caption="x")
    arr = np.asarray(out)
    # Left half should pick up magenta (high R, low G, high B).
    left = arr[8, 4]
    # Right half should pick up cyan (low R, high G, high B).
    right = arr[8, 12]
    assert left[0] > 30 and left[2] > 30  # magenta has R + B
    assert right[1] > 30 and right[2] > 30  # cyan has G + B
    assert left[1] < left[0]  # less green than red in magenta
    assert right[0] < right[1]  # less red than green in cyan


def test_render_overlay_raises_on_shape_mismatch() -> None:
    img = Image.new("RGB", (16, 16))
    pred = np.zeros((16, 16), dtype=bool)
    gt = np.zeros((15, 16), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        render_overlay(img, pred, gt, caption="x")


# ---- write_bundle --------------------------------------------------------


def _make_metrics(mAP: float) -> MagicMock:
    r = MagicMock()
    r.overall = {"mAP": mAP, "mAP_50": mAP, "mAP_75": mAP}
    r.per_class = {}
    r.n_images = 3
    r.n_predictions = 3
    return r


def _make_decision() -> PresetDecision:
    return PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        gradient_checkpointing=False,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * 1024**3),
        predicted_bytes=int(38.4 * 1024**3),
        budget_bytes=39 * 1024**3,
        image_size=1008,
        gpu_name="NVIDIA A100-SXM4-40GB",
        provenance="calibrated",
        cache_path="/tmp/.custom_sam_peft_calibration.json",  # noqa: S108
        calibrated_at="2026-05-18",
    )


def _make_ctx(tmp_path: Path, **overrides: object) -> BundleContext:
    base = BundleContext(
        run_dir=tmp_path / "run",
        config_path=tmp_path / "config.yaml",
        start_ts=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
        end_ts=datetime(2026, 5, 18, 12, 5, 0, tzinfo=UTC),
        preset=_make_decision(),
        per_example_iou=[0.1, 0.5, 0.9],
        merged_dir=None,
        merged_export_error=None,
        oom_events=(),
    )
    base.run_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text("run: {name: r}\n")
    return replace(base, **overrides)


def _make_dataset(n: int) -> MagicMock:
    ds = MagicMock()
    ds.__len__ = lambda self: n
    return ds


def test_write_bundle_writes_summary_and_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(tmp_path)
    val_ds = _make_dataset(3)
    model = MagicMock()
    report = _make_metrics(mAP=0.42)

    # Monkeypatch the per-example inference helper to deterministic 1-pixel masks.
    def _fake_render(image: Image.Image, pred: object, gt: object, *, caption: str) -> Image.Image:
        return Image.new("RGB", (8, 8), (1, 2, 3))

    monkeypatch.setattr("custom_sam_peft.runs.bundle.render_overlay", _fake_render)

    # Monkeypatch the per-sample re-inference to a no-op that yields blank masks.
    def _fake_run_one(
        _model: object, _ds: object, _idx: int
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        return Image.new("RGB", (8, 8)), np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=bool)

    monkeypatch.setattr("custom_sam_peft.runs.bundle._reinfer_one_example", _fake_run_one)

    write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=model)
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "0.4200" in summary
    assert "## Run" in summary
    assert "## Hardware" in summary
    assert "## Preset" in summary
    assert "## Outputs" in summary
    assert "## Samples" in summary
    pngs = sorted((ctx.run_dir / "samples").glob("*.png"))
    assert len(pngs) >= 1
    # Names embed bracket label and ordinal.
    names = [p.name for p in pngs]
    assert any("worst" in n for n in names)


def test_write_bundle_empty_val_writes_summary_with_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    val_ds = _make_dataset(0)
    report = _make_metrics(mAP=0.0)
    write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "empty val" in summary.lower()
    assert (ctx.run_dir / "samples").is_dir()
    assert list((ctx.run_dir / "samples").glob("*.png")) == []


def test_write_bundle_merge_failure_recorded_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(
        tmp_path,
        merged_dir=None,
        merged_export_error="ValueError: rank mismatch",
    )
    val_ds = _make_dataset(0)
    report = _make_metrics(mAP=0.0)
    write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "FAILED" in summary
    assert "rank mismatch" in summary


def test_write_bundle_skipped_sample_logged_and_summary_notes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _make_ctx(tmp_path)
    val_ds = _make_dataset(3)
    report = _make_metrics(mAP=0.42)

    def _explode(
        _model: object, _ds: object, _idx: int
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        raise RuntimeError("forward kaboom")

    monkeypatch.setattr("custom_sam_peft.runs.bundle._reinfer_one_example", _explode)

    with caplog.at_level("WARNING"):
        write_bundle(ctx, report, val_dataset=val_ds, model_wrapper=MagicMock())

    summary = (ctx.run_dir / "summary.md").read_text()
    assert "skipped samples" in summary.lower()
    assert "forward kaboom" in caplog.text or any(
        "forward kaboom" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): no-val bundle path
# ---------------------------------------------------------------------------


def test_write_bundle_no_val_writes_summary_only(tmp_path: Path) -> None:
    """Spec §7.5: write_bundle(val_dataset=None, metrics_report=None) writes summary.md only."""
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=MagicMock())
    summary_path = ctx.run_dir / "summary.md"
    assert summary_path.is_file()
    summary = summary_path.read_text()
    assert "no-val" in summary.lower() or "no validation" in summary.lower()
    # No samples directory should be created in no-val mode.
    samples_dir = ctx.run_dir / "samples"
    assert not samples_dir.exists() or not any(samples_dir.glob("*.png"))


def test_write_bundle_no_val_headline_says_no_val(tmp_path: Path) -> None:
    """Spec §7.5: headline reads '... — no-val' (no mAP number)."""
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    first_line = summary.splitlines()[0]
    assert first_line.startswith("# ")
    assert "no-val" in first_line.lower()


def test_write_bundle_no_val_contains_no_validation_set_line(tmp_path: Path) -> None:
    """Spec §7.5: summary body contains 'No validation set'."""
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    write_bundle(ctx, metrics_report=None, val_dataset=None, model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "No validation set" in summary


def _fake_reinfer(*_a: object, **_k: object) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    return (
        Image.new("RGB", (8, 8)),
        np.zeros((8, 8), bool),
        np.zeros((8, 8), bool),
    )


def test_write_bundle_preset_block_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _make_ctx(tmp_path, per_example_iou=[])
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        _fake_reinfer,
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "## Preset" in summary
    assert "- Method: LoRA r=32" in summary
    assert "batch=2" in summary
    assert "grad_accum=8" in summary
    assert "gradient_checkpointing=off" in summary
    assert "- GPU:    NVIDIA A100-SXM4-40GB" in summary
    assert "38.4 / " in summary  # used/total GiB
    assert "calibrated" in summary.lower()
    assert "2026-05-18" in summary


def test_write_bundle_preset_block_analytic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = replace(_make_decision(), provenance="analytic", cache_path=None, calibrated_at=None)
    ctx = _make_ctx(tmp_path, per_example_iou=[], preset=decision)
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        _fake_reinfer,
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "- Source: analytic estimate" in summary


def test_write_bundle_oom_edge_note_with_ckpt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = (
        OomEvent(
            step=10,
            action="microbatch_halved",
            new_micro_batch_size=4,
            new_gradient_checkpointing=False,
        ),
        OomEvent(
            step=20,
            action="microbatch_halved",
            new_micro_batch_size=2,
            new_gradient_checkpointing=False,
        ),
        OomEvent(
            step=412,
            action="grad_ckpt_enabled",
            new_micro_batch_size=2,
            new_gradient_checkpointing=True,
        ),
    )
    ctx = _make_ctx(tmp_path, per_example_iou=[], oom_events=events)
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        _fake_reinfer,
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "OOM retries: 3" in summary
    assert "final micro_batch=2" in summary
    assert "gradient_checkpointing enabled at step 412" in summary


def test_write_bundle_oom_edge_note_no_ckpt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = (
        OomEvent(
            step=10,
            action="microbatch_halved",
            new_micro_batch_size=4,
            new_gradient_checkpointing=False,
        ),
    )
    ctx = _make_ctx(tmp_path, per_example_iou=[], oom_events=events)
    monkeypatch.setattr(
        "custom_sam_peft.runs.bundle._reinfer_one_example",
        _fake_reinfer,
    )
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "OOM retries: 1" in summary
    # The "gradient_checkpointing enabled at step X" clause must be omitted.
    assert "gradient_checkpointing enabled" not in summary
