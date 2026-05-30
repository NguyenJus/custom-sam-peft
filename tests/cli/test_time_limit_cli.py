"""CLI integration for --time-limit (spec §11.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.eval._artifacts import EvalArtifacts, TimeLimitStop

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    """A minimal valid config the loader accepts (no real data load triggered:
    run_train is patched in these tests, so dataset paths are never opened)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: tl\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def _stop_artifacts(run_dir: Path) -> EvalArtifacts:
    return EvalArtifacts(
        checkpoint_path=run_dir / "checkpoints" / "step_5" / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=None,
        time_limit_stop=TimeLimitStop(
            stop_step=5,
            stop_epoch=0,
            total_epochs=1,
            checkpoint_dir=run_dir / "checkpoints" / "step_5",
            duration_label="2h",
            best_dir=None,
            best_map=None,
        ),
    )


def test_train_bad_time_limit_exits_1_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import train_cmd

    called = {"run": False}
    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **k: called.__setitem__("run", True))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["train", "--config", str(cfg), "--time-limit", "10x"])
    assert result.exit_code == 1
    assert "invalid --time-limit" in result.output
    assert called["run"] is False


def test_train_time_limit_overrides_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import train_cmd

    seen: dict[str, Any] = {}

    def fake_run_train(cfg: Any, **k: Any) -> EvalArtifacts:
        seen["time_limit"] = cfg.train.time_limit
        return _stop_artifacts(tmp_path / "run")

    monkeypatch.setattr(train_cmd, "run_train", fake_run_train)
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["train", "--config", str(cfg), "--time-limit", "2h"])
    assert result.exit_code == 0
    assert seen["time_limit"] == "2h"


def test_train_stop_prints_message_and_skips_eval_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import train_cmd

    run_dir = tmp_path / "run"
    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **k: _stop_artifacts(run_dir))
    eval_called = {"n": 0}
    export_called = {"n": 0}
    monkeypatch.setattr(train_cmd, "run_eval", lambda *a, **k: eval_called.__setitem__("n", 1))
    monkeypatch.setattr(train_cmd, "run_export", lambda *a, **k: export_called.__setitem__("n", 1))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(
        app, ["train", "--config", str(cfg), "--time-limit", "2h", "--eval", "--export"]
    )
    assert result.exit_code == 0
    assert "Time limit (2h) reached" in result.output
    assert eval_called["n"] == 0
    assert export_called["n"] == 0


def test_run_bad_time_limit_exits_1_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import run_cmd

    called = {"run": False}
    monkeypatch.setattr(run_cmd, "run_training", lambda *a, **k: called.__setitem__("run", True))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["run", "--config", str(cfg), "--time-limit", "10x"])
    assert result.exit_code == 1
    assert "invalid --time-limit" in result.output
    assert called["run"] is False


def test_run_time_limit_overrides_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import run_cmd

    seen: dict[str, Any] = {}

    def fake_run_training(cfg: Any, **k: Any) -> Any:
        seen["time_limit"] = cfg.train.time_limit
        return _stop_artifacts(tmp_path / "run")

    monkeypatch.setattr(run_cmd, "run_training", fake_run_training)
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["run", "--config", str(cfg), "--time-limit", "2h"])
    assert result.exit_code == 0
    assert seen["time_limit"] == "2h"


def test_run_stop_short_circuits_before_eval_export_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import run_cmd

    run_dir = tmp_path / "run"
    (run_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_cmd, "run_training", lambda *a, **k: _stop_artifacts(run_dir))
    phase_calls = {"val": 0, "load": 0, "eval": 0, "merged": 0, "bundle": 0}
    monkeypatch.setattr(
        "custom_sam_peft.data.val_source.load_val_source",
        lambda *a, **k: phase_calls.__setitem__("val", 1),
    )
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: phase_calls.__setitem__("load", 1))
    monkeypatch.setattr(run_cmd, "run_eval", lambda *a, **k: phase_calls.__setitem__("eval", 1))
    monkeypatch.setattr(
        run_cmd, "save_merged", lambda *a, **k: phase_calls.__setitem__("merged", 1)
    )
    monkeypatch.setattr(
        run_cmd, "write_bundle", lambda *a, **k: phase_calls.__setitem__("bundle", 1)
    )
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["run", "--config", str(cfg), "--time-limit", "2h"])
    assert result.exit_code == 0
    assert "Time limit (2h) reached" in result.output
    # No phase after train ran:
    assert phase_calls["val"] == 0
    assert phase_calls["load"] == 0
    assert phase_calls["eval"] == 0
    assert phase_calls["merged"] == 0
    assert phase_calls["bundle"] == 0
