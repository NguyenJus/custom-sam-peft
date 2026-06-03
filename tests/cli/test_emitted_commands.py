"""Emitted copy-paste commands use the positional config form (§6.5)."""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft.cli._interactive import _launch_command


def test_launch_command_positional_train() -> None:
    assert _launch_command(Path("config.yaml"), "train") == "custom-sam-peft train config.yaml"


def test_launch_command_positional_run() -> None:
    assert _launch_command(Path("config.yaml"), "run") == "custom-sam-peft run config.yaml"


def test_launch_command_eval_uses_flag() -> None:
    assert (
        _launch_command(Path("config.yaml"), "eval") == "custom-sam-peft eval --config config.yaml"
    )
