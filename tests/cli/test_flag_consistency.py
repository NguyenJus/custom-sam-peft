"""Cross-command flag-consistency guard (parser introspection, CPU-only).

Tightened phase by phase. Phase 1 asserts only the currently-true vocabulary.
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


# Commands that carry --progress after Phase 1.
_PROGRESS_CMDS = ["train", "run", "eval", "export", "predict"]
# Commands that carry -v/--verbose after Phase 1.
_VERBOSE_CMDS = ["train", "run", "eval", "export", "predict"]


@pytest.mark.parametrize("name", _PROGRESS_CMDS)
def test_progress_is_progress_enum(name: str) -> None:
    opt = _opt(_command(name), "progress")
    assert opt is not None, f"{name} missing --progress"
    # Typer renders an Enum-typed option with a click.Choice of the enum values.
    assert isinstance(opt.type, click.Choice)
    assert set(opt.type.choices) == {m.value for m in Progress}


@pytest.mark.parametrize("name", _VERBOSE_CMDS)
def test_verbose_present(name: str) -> None:
    opt = _opt(_command(name), "verbose")
    assert opt is not None, f"{name} missing -v/--verbose"
    assert "-v" in opt.opts or "-v" in opt.secondary_opts


def test_train_has_override() -> None:
    assert _opt(_command("train"), "override") is not None
