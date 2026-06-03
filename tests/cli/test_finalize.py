"""run --finalize: rebuild + close_out, no training (spec §11, §14.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import custom_sam_peft.cli.run_cmd as run_cmd
from custom_sam_peft.eval._artifacts import EvalArtifacts


def _make_paused_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "paused-run"
    (run_dir / "checkpoints" / "step_5").mkdir(parents=True)
    (run_dir / "best" / "adapter").mkdir(parents=True)
    (run_dir / "best" / "best.json").write_text(
        json.dumps({"metric": "mAP", "value": 0.8, "global_step": 5})
    )
    (run_dir / "config.yaml").write_text("run:\n  name: paused\n")
    (run_dir / "val_source.json").write_text('{"mode": "none"}')
    return run_dir / "checkpoints" / "step_5"


def test_finalize_calls_close_out_no_training(tmp_path: Path, monkeypatch) -> None:
    resume = _make_paused_run(tmp_path)
    run_dir = resume.parent.parent

    called = {"close_out": 0, "train": 0}
    artifacts = EvalArtifacts(
        checkpoint_path=run_dir / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=None,
        per_example_iou=None,
        final_weights="best",
    )
    monkeypatch.setattr(
        run_cmd, "close_out", lambda *a, **k: (called.__setitem__("close_out", 1), artifacts)[1]
    )
    monkeypatch.setattr(run_cmd, "run_training", lambda *a, **k: called.__setitem__("train", 1))
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: object())
    monkeypatch.setattr(run_cmd, "load_adapter", lambda *a, **k: None)
    monkeypatch.setattr(run_cmd, "load_config", lambda p, **kw: _SavedCfg())
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: None)
    monkeypatch.setattr(run_cmd, "_load_preset_or_fallback", lambda c: object())

    rc = run_cmd._finalize(
        _SavedCfg(),
        resume,
        config_path=run_dir / "config.yaml",
    )
    assert rc == 0
    assert called["close_out"] == 1
    assert called["train"] == 0  # NO training


def test_finalize_requires_resume(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("run:\n  name: x\n")
    monkeypatch.setattr(run_cmd, "load_config", lambda p, **kw: _SavedCfg())
    with pytest.raises(typer.Exit) as exc:
        run_cmd.run(
            config_arg=cfg_path,
            resume=None,
            time_limit=None,
            finalize=True,
            override=[],
            verbose=False,
            progress=run_cmd.Progress.off,
            visualize=False,
        )
    assert exc.value.exit_code == 1


def test_finalize_rejects_time_limit(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("run:\n  name: x\n")
    monkeypatch.setattr(run_cmd, "load_config", lambda p, **kw: _SavedCfg())
    with pytest.raises(typer.Exit) as exc:
        run_cmd.run(
            config_arg=cfg_path,
            resume="__latest__",
            time_limit="1h",
            finalize=True,
            override=[],
            verbose=False,
            progress=run_cmd.Progress.off,
            visualize=False,
        )
    assert exc.value.exit_code == 1


def test_finalize_resolves_latest(tmp_path: Path, monkeypatch) -> None:
    resume = _make_paused_run(tmp_path)
    cfg_path = resume.parent.parent / "config.yaml"

    monkeypatch.setattr(run_cmd, "find_latest_checkpoint", lambda cfg: resume)
    captured = {"resume": None}

    def fake_finalize(cfg, resume_path, *, config_path):
        captured["resume"] = resume_path
        return 0

    monkeypatch.setattr(run_cmd, "_finalize", fake_finalize)
    monkeypatch.setattr(run_cmd, "load_config", lambda p, **kw: _SavedCfg())

    with pytest.raises(typer.Exit) as exc:
        run_cmd.run(
            config_arg=cfg_path,
            resume="__latest__",
            time_limit=None,
            finalize=True,
            override=[],
            verbose=False,
            progress=run_cmd.Progress.off,
            visualize=False,
        )
    assert exc.value.exit_code == 0
    assert captured["resume"] == resume  # resolved via find_latest_checkpoint


class _SavedCfg:
    class export:
        merge = False

    class model:
        pass

    class data:
        channels = 3
        channel_semantics = "rgb"

    class eval:
        visualize = False

        @staticmethod
        def model_copy(*, update=None):
            return _SavedCfg.eval

    class run:
        name = "paused"

    def model_copy(self, *, update=None):
        return _SavedCfg()
