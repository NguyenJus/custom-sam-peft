"""CLI integration for host-RAM-floor stop (mirrors test_time_limit_cli.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.eval._artifacts import EvalArtifacts, HostRamStop

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    """A minimal valid config the loader accepts (no real data load triggered:
    run_train / run_training is patched in these tests)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: hr\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def _ram_stop_artifacts(run_dir: Path) -> EvalArtifacts:
    return EvalArtifacts(
        checkpoint_path=run_dir / "checkpoints" / "step_5" / "adapter",
        peft_method="lora",
        run_dir=run_dir,
        final_metrics=None,
        host_ram_stop=HostRamStop(
            stop_step=5,
            stop_epoch=0,
            total_epochs=1,
            checkpoint_dir=run_dir / "checkpoints" / "step_5",
            available_gb=1.2,
            floor_gb=4.0,
            best_dir=None,
            best_map=None,
        ),
    )


# ---------------------------------------------------------------------------
# format_host_ram_message unit tests (mirror test_time_limit_message.py)
# ---------------------------------------------------------------------------


def _stop(*, best: bool) -> HostRamStop:
    return HostRamStop(
        stop_step=4120,
        stop_epoch=3,
        total_epochs=10,
        checkpoint_dir=Path("runs/x/checkpoints/step_4120"),
        available_gb=1.5,
        floor_gb=4.0,
        best_dir=Path("runs/x/best") if best else None,
        best_map=0.612 if best else None,
    )


def test_message_has_resume_command_train() -> None:
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(
        _stop(best=False), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "custom-sam-peft train --config configs/run.yaml --resume __latest__" in msg


def test_message_has_resume_command_run() -> None:
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(
        _stop(best=False), subcommand="run", config_path=Path("configs/run.yaml")
    )
    assert "custom-sam-peft run --config configs/run.yaml --resume __latest__" in msg


def test_message_with_best_includes_best_lines() -> None:
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(
        _stop(best=True), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "Best so far" in msg
    assert "best" in msg
    assert "0.612" in msg
    assert "Use best as-is" in msg


def test_message_without_best_omits_best_lines() -> None:
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(
        _stop(best=False), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "Best so far" not in msg
    assert "Use best as-is" not in msg
    assert "--resume __latest__" in msg


def test_message_includes_floor_and_available() -> None:
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(_stop(best=False), subcommand="train", config_path=Path("c.yaml"))
    assert "4.0 GB" in msg  # floor
    assert "1.50 GB" in msg  # available


def test_message_epoch_rendered_one_based() -> None:
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(_stop(best=False), subcommand="train", config_path=Path("c.yaml"))
    assert "(epoch 4/10)" in msg  # stop_epoch 3 (zero-based) -> "4/10"


def test_message_lower_memory_guidance_factual() -> None:
    """Resume guidance must NOT claim mid-run config edits are auto-applied."""
    from custom_sam_peft.cli._host_ram import format_host_ram_message

    msg = format_host_ram_message(_stop(best=False), subcommand="train", config_path=Path("c.yaml"))
    # Must mention editing config and resuming — factual guidance only.
    assert "edit config" in msg
    assert "resume" in msg.lower()
    # Must NOT claim auto-apply.
    assert "auto-appl" not in msg.lower()


# ---------------------------------------------------------------------------
# train CLI: host_ram_stop short-circuits eval/export
# ---------------------------------------------------------------------------


def test_train_host_ram_stop_prints_message_and_skips_eval_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import train_cmd

    run_dir = tmp_path / "run"
    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **k: _ram_stop_artifacts(run_dir))
    eval_called = {"n": 0}
    export_called = {"n": 0}
    monkeypatch.setattr(train_cmd, "run_eval", lambda *a, **k: eval_called.__setitem__("n", 1))
    monkeypatch.setattr(train_cmd, "run_export", lambda *a, **k: export_called.__setitem__("n", 1))
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["train", "--config", str(cfg), "--eval", "--export"])
    assert result.exit_code == 0
    assert "Host RAM floor" in result.output
    assert eval_called["n"] == 0
    assert export_called["n"] == 0


# ---------------------------------------------------------------------------
# run CLI: host_ram_stop short-circuits before model reload / bundle
# ---------------------------------------------------------------------------


def test_run_host_ram_stop_short_circuits_before_eval_export_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import run_cmd

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_cmd, "run_training", lambda *a, **k: _ram_stop_artifacts(run_dir))
    phase_calls = {"val": 0, "load": 0, "bundle": 0}
    monkeypatch.setattr(
        "custom_sam_peft.data.split_source.load_split_source",
        lambda *a, **k: phase_calls.__setitem__("val", 1),
    )
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: phase_calls.__setitem__("load", 1))
    monkeypatch.setattr(
        run_cmd, "write_bundle", lambda *a, **k: phase_calls.__setitem__("bundle", 1)
    )
    cfg = _write_min_config(tmp_path)

    from custom_sam_peft.cli.main import app

    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Host RAM floor" in result.output
    assert phase_calls["val"] == 0
    assert phase_calls["load"] == 0
    assert phase_calls["bundle"] == 0
