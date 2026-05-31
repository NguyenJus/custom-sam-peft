"""Task 15: _orchestrate reuses close_out's single eval; run_eval must NOT be called.

The run orchestrator drops its own eval + export-merge phases (close_out inside
fit() already ran the one full eval + export-merge on the best weights). This
test asserts that run_eval is never called on the normal (no-val) path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import custom_sam_peft.cli.run_cmd as run_cmd
from custom_sam_peft.cli._progress import ProgressMode
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.presets import PresetDecision


def _write_min_config(tmp_path: Path) -> Path:
    """Minimal valid config.yaml that TrainConfig can load without real data."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "run:\n  name: test-single-eval\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def _build_artifacts(run_dir: Path) -> EvalArtifacts:
    """Build an EvalArtifacts stub as close_out would return it (no-val path)."""
    return EvalArtifacts(
        checkpoint_path=run_dir / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=None,  # no-val
        per_example_iou=None,  # no-val
        final_weights="best",
        ladder_events=None,
        oom_events=(),
        time_limit_stop=None,
    )


def test_orchestrate_does_not_call_run_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval must not be called on the normal run path (close_out owns the single eval)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write val_source.json with mode=none so no model rebuild happens.
    (run_dir / "val_source.json").write_text(json.dumps({"mode": "none"}))

    # Build the EvalArtifacts that run_training returns.
    artifacts = _build_artifacts(run_dir)

    # Monkeypatch run_training -> return artifacts immediately.
    monkeypatch.setattr(run_cmd, "run_training", lambda cfg, resume_from=None: artifacts)

    # run_eval is no longer imported in run_cmd (it was removed); confirm it is absent.
    eval_calls: list[Any] = []
    assert not hasattr(run_cmd, "run_eval"), (
        "run_eval must NOT be imported in run_cmd — the orchestrator no longer calls it"
    )

    # Monkeypatch _load_preset_or_fallback -> stub (avoids CUDA-required decide_preset).
    stub_preset = PresetDecision(
        method="lora",
        r=4,
        batch_size=1,
        grad_accum_steps=1,
        dtype="bfloat16",
        headroom_bytes=0,
        predicted_bytes=0,
        budget_bytes=0,
        gpu_name="stub",
        provenance="analytic",
        cache_path=None,
        calibrated_at=None,
    )
    monkeypatch.setattr(run_cmd, "_load_preset_or_fallback", lambda cfg: stub_preset)

    # Monkeypatch write_bundle -> no-op.
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: None)

    # Monkeypatch load_sam31 / load_adapter -> no-op (should not be reached on no-val path).
    sam_calls: list[Any] = []
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: sam_calls.append(1) or object())
    monkeypatch.setattr(run_cmd, "load_adapter", lambda *a, **k: None)

    # Load a real TrainConfig from the minimal yaml.
    from custom_sam_peft.config.loader import load_config

    cfg_path = _write_min_config(tmp_path)
    cfg = load_config(cfg_path)

    rc = run_cmd._orchestrate(
        cfg=cfg,
        resume=None,
        mode=ProgressMode.OFF,
        config_path=cfg_path,
    )

    assert rc == 0, f"_orchestrate returned {rc}"
    assert len(eval_calls) == 0, (
        f"run_eval was called {len(eval_calls)} time(s) — expected 0. "
        "The orchestrator must reuse close_out's results, not re-run eval."
    )
    # On no-val path, model should not be rebuilt.
    assert len(sam_calls) == 0, (
        f"load_sam31 was called on no-val path ({len(sam_calls)} time(s)); "
        "should only be called when vs.mode != 'none'."
    )
