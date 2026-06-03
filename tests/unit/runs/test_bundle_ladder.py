"""Bundle surfaces best-as-final adapter + ladder events (spec §7.6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from custom_sam_peft.presets import PresetDecision
from custom_sam_peft.runs.bundle import BundleContext, write_bundle
from custom_sam_peft.train.ladder import LadderEvents


def _make_preset() -> PresetDecision:
    return PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        classes_per_forward=8,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * 1024**3),
        predicted_bytes=int(38.4 * 1024**3),
        budget_bytes=39 * 1024**3,
        gpu_name="NVIDIA A100-SXM4-40GB",
        provenance="calibrated",
        cache_path="/tmp/.custom_sam_peft_calibration.json",  # noqa: S108
        calibrated_at="2026-05-18",
    )


def _ctx(run_dir: Path, **kw) -> BundleContext:  # type: ignore[type-arg]
    return BundleContext(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        start_ts=datetime.now(UTC),
        end_ts=datetime.now(UTC),
        preset=_make_preset(),
        per_example_iou=[],
        merged_dir=None,
        merged_export_error=None,
        oom_events=(),
        **kw,
    )


def test_bundle_context_accepts_ladder_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best").mkdir()
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.8, "global_step": 6})
    )
    events = LadderEvents(stop_reason="early stop: 10 evals")
    ctx = _ctx(run_dir, ladder_events=events)
    # No-val path → summary.md only; must not raise and must mention the stop.
    write_bundle(ctx, None, val_dataset=None, model_wrapper=None)
    body = (run_dir / "summary.md").read_text()
    assert "best checkpoint" in body or "best/" in body
    assert "early stop" in body


def test_bundle_default_ladder_events_renders_unchanged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    ctx = _ctx(run_dir)  # ladder_events defaults to None
    write_bundle(ctx, None, val_dataset=None, model_wrapper=None)
    body = (run_dir / "summary.md").read_text()
    assert "early stop" not in body  # nothing rendered for a no-ladder run
