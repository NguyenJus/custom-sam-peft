"""Exit-message formatter tests (spec §11.6)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.cli._time_limit import format_time_limit_message
from custom_sam_peft.eval._artifacts import TimeLimitStop


def _stop(*, best: bool, label: str = "2h30m") -> TimeLimitStop:
    return TimeLimitStop(
        stop_step=4120,
        stop_epoch=3,
        total_epochs=10,
        checkpoint_dir=Path("runs/x/checkpoints/step_4120"),
        duration_label=label,
        best_dir=Path("runs/x/best") if best else None,
        best_map=0.612 if best else None,
    )


def test_message_has_resume_command_train() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "custom-sam-peft train --config configs/run.yaml --resume __latest__" in msg


def test_message_with_best_includes_best_lines() -> None:
    msg = format_time_limit_message(
        _stop(best=True), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "Best so far" in msg
    assert "best" in msg  # the best/ path
    assert "0.612" in msg
    assert "Use best as-is" in msg


def test_message_without_best_omits_best_lines() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="train", config_path=Path("configs/run.yaml")
    )
    assert "Best so far" not in msg
    assert "Use best as-is" not in msg
    assert "--resume __latest__" in msg  # resume still present


def test_message_subcommand_run() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="run", config_path=Path("configs/run.yaml")
    )
    assert "custom-sam-peft run --config configs/run.yaml --resume __latest__" in msg


def test_message_duration_from_format_seconds() -> None:
    a = format_time_limit_message(
        _stop(best=False, label="2h30m"), subcommand="train", config_path=Path("c.yaml")
    )
    assert "(2h30m)" in a


def test_message_epoch_rendered_one_based() -> None:
    msg = format_time_limit_message(
        _stop(best=False), subcommand="train", config_path=Path("c.yaml")
    )
    assert "(epoch 4/10)" in msg  # stop_epoch 3 (zero-based) -> "4/10"
