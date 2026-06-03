"""Cross-command flag-consistency guard (parser introspection, CPU-only).

Tightened phase by phase. Phase 2 asserts the post-additive vocabulary: -v on all
eight commands, --dry-run on train/run/eval/predict, run --override, eval --split enum.
Spec §4.7.
"""

from __future__ import annotations

import click
import pytest
import typer.main

from custom_sam_peft.cli._options import Progress
from custom_sam_peft.cli.main import app

_GROUP = typer.main.get_command(app)


def _command(name: str) -> click.Command:
    cmd = _GROUP.get_command(None, name)  # type: ignore[attr-defined]
    assert cmd is not None, f"no such command: {name}"
    return cmd


def _opt(cmd: click.Command, name: str) -> click.Option | None:
    for p in cmd.params:
        if isinstance(p, click.Option) and p.name == name:
            return p
    return None


# Commands that carry --progress (doctor/init/calibrate intentionally lack it, §5.3).
_PROGRESS_CMDS = ["train", "run", "eval", "export", "predict"]
# After Phase 2, -v/--verbose is on all eight commands.
_ALL_CMDS = ["train", "run", "eval", "export", "init", "doctor", "predict", "calibrate"]
# --dry-run is on these four after Phase 2.
_DRY_RUN_CMDS = ["train", "run", "eval", "predict"]


@pytest.mark.parametrize("name", _PROGRESS_CMDS)
def test_progress_is_progress_enum(name: str) -> None:
    opt = _opt(_command(name), "progress")
    assert opt is not None, f"{name} missing --progress"
    # Typer renders an Enum-typed option with a click.Choice of the enum values.
    assert isinstance(opt.type, click.Choice)
    assert set(opt.type.choices) == {m.value for m in Progress}


@pytest.mark.parametrize("name", _ALL_CMDS)
def test_verbose_present_all(name: str) -> None:
    opt = _opt(_command(name), "verbose")
    assert opt is not None, f"{name} missing -v/--verbose"
    assert "-v" in opt.opts or "-v" in opt.secondary_opts


@pytest.mark.parametrize("name", _DRY_RUN_CMDS)
def test_dry_run_present(name: str) -> None:
    assert _opt(_command(name), "dry_run") is not None, f"{name} missing --dry-run"


def test_train_has_override() -> None:
    assert _opt(_command("train"), "override") is not None


def test_run_has_override_after_phase2() -> None:
    assert _opt(_command("run"), "override") is not None


def test_eval_split_is_split_enum() -> None:
    from custom_sam_peft.cli._options import Split

    opt = _opt(_command("eval"), "split")
    assert opt is not None
    assert isinstance(opt.type, click.Choice)
    assert set(opt.type.choices) == {m.value for m in Split}
