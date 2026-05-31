"""End-to-end CLI integration tests for `custom_sam_peft run`."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.presets import PresetDecision

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Minimal valid TrainConfig YAML — used by the skip-init guard test so run_init's
# side-effect produces a config that load_config can parse.
_MINIMAL_CONFIG = """\
run:
  name: t
  output_dir: /tmp/runs
  seed: 0
data:
  format: coco
  train:
    annotations: t.json
    images: t/
  val:
    annotations: v.json
    images: v/
peft:
  method: lora
train:
  epochs: 1
export:
  merge: false
"""


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def _make_cfg_yaml(tmp_path: Path, *, merge: bool = False) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
run: {{name: t, output_dir: {tmp_path / "runs"}, seed: 0}}
data:
  format: coco
  train: {{annotations: t.json, images: t/}}
  val: {{annotations: v.json, images: v/}}
peft: {{method: lora}}
train: {{epochs: 1}}
export: {{merge: {str(merge).lower()}}}
"""
    )
    return cfg


def _make_preset_decision() -> PresetDecision:
    return PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        classes_per_forward=8,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * 1024**3),
        predicted_bytes=int(38.4 * 1024**3),
        budget_bytes=(39 * 1024**3),
        gpu_name="StubGPU",
        provenance="analytic",
        cache_path=None,
        calibrated_at=None,
    )


def _write_preset_sidecar(tmp_path: Path) -> PresetDecision:
    d = PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        classes_per_forward=8,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * 1024**3),
        predicted_bytes=int(38.4 * 1024**3),
        budget_bytes=(39 * 1024**3),
        gpu_name="StubGPU",
        provenance="analytic",
        cache_path=None,
        calibrated_at=None,
    )
    (tmp_path / "preset.json").write_text(d.to_json())
    return d


def _make_train_result(
    run_dir: Path,
    *,
    final_metrics: object = None,
    per_example_iou: list[float] | None = None,
    merged_export_error: str | None = None,
) -> EvalArtifacts:
    """Build a minimal EvalArtifacts that satisfies _orchestrate's expectations."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter").mkdir(exist_ok=True)
    # _orchestrate reads val_source.json after training to decide val mode.
    (run_dir / "val_source.json").write_text(
        '{"mode": "explicit", "fraction_requested": null, "seed_used": null, '
        '"realized_fraction": null, "n_train": null, "n_val": null, '
        '"per_class_counts": null, "missing_in_val": null, '
        '"train_ids": null, "val_ids": null}'
    )
    return EvalArtifacts(
        checkpoint_path=run_dir / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=final_metrics,  # type: ignore[arg-type]
        per_example_iou=per_example_iou,
        oom_events=(),
        time_limit_stop=None,
        final_weights="best",
        ladder_events=None,
        merged_export_error=merged_export_error,
    )


def _patch_phases(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_dir: Path,
    train_raises: Exception | None = None,
    bundle_raises: Exception | None = None,
    train_result: EvalArtifacts | None = None,
) -> dict[str, object]:
    """Patch every phase entry point. Return a dict that records calls.

    The new architecture: run_training returns EvalArtifacts (which already
    contains final_metrics, per_example_iou, and merged_export_error from
    close_out). _orchestrate reuses those results — no second eval, no separate
    merge phase.
    """
    captured: dict[str, object] = {"order": []}

    fake_report = MagicMock(overall={"mAP": 0.42})
    fake_per_ex = [0.1, 0.5, 0.9]

    if train_result is None:
        train_result = _make_train_result(
            run_dir,
            final_metrics=fake_report,
            per_example_iou=fake_per_ex,
        )

    # Stub _load_preset_or_fallback so tests that don't write a sidecar don't hit CUDA.
    fake_preset = _make_preset_decision()
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._load_preset_or_fallback",
        lambda _cfg: fake_preset,
    )

    def _train(cfg: object, *, resume_from: object = None) -> object:
        captured["order"].append("train")  # type: ignore[union-attr]
        captured["resume_from"] = resume_from
        if train_raises is not None:
            raise train_raises
        return train_result

    def _write_bundle(
        ctx: object, report: object, *, val_dataset: object, model_wrapper: object
    ) -> None:
        captured["order"].append("bundle")  # type: ignore[union-attr]
        captured["bundle_ctx"] = ctx
        if bundle_raises is not None:
            raise bundle_raises

    monkeypatch.setattr("custom_sam_peft.cli.run_cmd.run_training", _train)
    monkeypatch.setattr("custom_sam_peft.cli.run_cmd.write_bundle", _write_bundle)
    monkeypatch.setattr("custom_sam_peft.cli.run_cmd.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr("custom_sam_peft.cli.run_cmd.load_adapter", lambda *_a, **_kw: None)
    # Build a stub val_dataset.
    fake_ds = MagicMock(__len__=lambda self: 3, class_names=["a"])

    def _build_val(_cfg: object, _vs: object) -> object:
        captured["order"].append("build_val")  # type: ignore[union-attr]
        return fake_ds

    monkeypatch.setattr("custom_sam_peft.cli.run_cmd._build_val_dataset", _build_val)
    return captured


def test_run_help_exits_zero() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Train + eval" in _plain(result.output)


def test_run_full_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pipeline: training succeeds, bundle is written, exit 0.

    close_out (inside run_training) already ran the eval; _orchestrate must
    NOT run a second eval — it reuses train_result.final_metrics/per_example_iou.
    """
    run_dir = tmp_path / "runs" / "r"
    captured = _patch_phases(monkeypatch, run_dir=run_dir)
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    order = captured["order"]
    assert order[0] == "train"
    assert order[-1] == "bundle"
    # No separate "eval" phase — eval happened inside run_training/close_out.
    assert "eval" not in order
    ctx = captured["bundle_ctx"]
    assert ctx.merged_dir is None
    assert ctx.merged_export_error is None
    # per_example_iou comes verbatim from EvalArtifacts — no second eval.
    assert ctx.per_example_iou == [0.1, 0.5, 0.9]


def test_run_train_failure_skips_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch,
        run_dir=tmp_path / "runs" / "r",
        train_raises=RuntimeError("kaboom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "kaboom" in _plain(result.output) or "kaboom" in (result.stderr or "")
    order = captured["order"]
    assert "bundle" not in order


def test_run_eval_failure_skips_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval now happens inside run_training (close_out). An eval/training exception
    makes run_training raise, which exits 1 and skips the bundle phase."""
    captured = _patch_phases(
        monkeypatch,
        run_dir=tmp_path / "runs" / "r",
        train_raises=RuntimeError("eval-boom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "bundle" not in captured["order"]


def test_run_merge_failure_still_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Merge failure is soft-failed INSIDE close_out (Part 1 fix) and surfaced via
    EvalArtifacts.merged_export_error.  _orchestrate must still build the bundle
    (exit 0) and surface the merge error through BundleContext."""
    run_dir = tmp_path / "runs" / "r"
    # Simulate close_out having soft-failed the merge: no merged/ dir on disk,
    # merged_export_error set to the error string.
    train_result = _make_train_result(
        run_dir,
        final_metrics=MagicMock(overall={"mAP": 0.42}),
        per_example_iou=[0.1, 0.5, 0.9],
        merged_export_error="rank mismatch",
    )
    captured = _patch_phases(monkeypatch, run_dir=run_dir, train_result=train_result)
    cfg = _make_cfg_yaml(tmp_path, merge=True)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "bundle" in captured["order"]
    ctx = captured["bundle_ctx"]
    assert ctx.merged_dir is None
    assert "rank mismatch" in (ctx.merged_export_error or "")


def test_run_bundle_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "runs" / "r"
    _patch_phases(
        monkeypatch,
        run_dir=run_dir,
        bundle_raises=RuntimeError("bundle-boom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    # run_dir and adapter remain on disk.
    assert run_dir.exists()
    assert (run_dir / "adapter").exists()


def test_run_reads_preset_sidecar_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _write_preset_sidecar(tmp_path)
    # Do NOT stub _load_preset_or_fallback for this test — we want the real sidecar path.
    # We must un-stub it after _patch_phases has set it; easier to call _patch_phases
    # first, then override the stub with the real function.
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    import custom_sam_peft.cli.run_cmd as _run_cmd

    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._load_preset_or_fallback",
        _run_cmd._load_preset_or_fallback,
    )
    cfg = _make_cfg_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)  # so run_cmd resolves preset.json relative to cwd
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset == expected
    assert ctx.oom_events == ()


def test_run_synthesizes_analytic_preset_when_sidecar_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Stub _fallback_preset so we don't need CUDA in this test.
    fake_decision = _write_preset_sidecar(tmp_path)  # writes & returns a PresetDecision
    (tmp_path / "preset.json").unlink()  # remove the sidecar so the fallback path runs
    monkeypatch.setattr("custom_sam_peft.cli.run_cmd._fallback_preset", lambda cfg: fake_decision)
    # Also restore the real _load_preset_or_fallback so the fallback path actually runs.
    import custom_sam_peft.cli.run_cmd as _run_cmd

    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._load_preset_or_fallback",
        _run_cmd._load_preset_or_fallback,
    )
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset.provenance == "analytic"


def test_run_consumes_train_tuple_verbatim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_orchestrate reuses EvalArtifacts.final_metrics and .per_example_iou verbatim —
    no second eval is triggered. The per_example_iou in BundleContext must be exactly
    what run_training returned."""
    run_dir = tmp_path / "runs" / "r"

    # Use a distinctive per_example_iou list to verify verbatim pass-through.
    sentinel_iou = [0.11, 0.22, 0.33]
    fake_report = MagicMock(overall={"mAP": 0.77})
    train_result = _make_train_result(
        run_dir,
        final_metrics=fake_report,
        per_example_iou=sentinel_iou,
    )
    captured = _patch_phases(monkeypatch, run_dir=run_dir, train_result=train_result)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
run: {{name: t, output_dir: {tmp_path / "runs"}, seed: 0}}
data:
  format: coco
  train: {{annotations: t.json, images: t/}}
  val: {{annotations: v.json, images: v/}}
model: {{dtype: float16}}
peft: {{method: qlora, r: 32}}
train: {{epochs: 1, batch_size: 4, grad_accum_steps: 4}}
export: {{merge: false}}
"""
    )

    result = runner.invoke(app, ["run", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    # No second eval — only "train" and "bundle" phases ran.
    assert "eval" not in captured["order"]
    # Bundle receives the exact iou list from EvalArtifacts (verbatim, no re-eval).
    ctx = captured["bundle_ctx"]
    assert ctx.per_example_iou == sentinel_iou


def test_run_skip_init_guard_warns_and_autoinits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run with no usable config warns 'not initialized', auto-inits (formula, no
    probe), then proceeds. Spec §6.2."""
    monkeypatch.chdir(tmp_path)
    called: dict[str, object] = {}
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._orchestrate",
        lambda *a, **k: called.setdefault("ran", True) or 0,
    )
    # Patch run_init so the guard does not need a GPU.
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd.run_init",
        lambda *a, **k: (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG),
    )
    result = runner.invoke(app, ["run", "--config", "config.yaml"])
    assert "not initialized" in result.output.lower()
    assert called.get("ran") is True
