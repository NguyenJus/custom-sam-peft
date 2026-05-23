"""Tests for bare --eval / --export flags on train / eval and the run alias.

Spec: docs/superpowers/specs/2026-05-18-simplify-ux-design.md §7.2 (user-locked:
bare flags only — not --with-eval / --with-export / --then-eval / --then-export).
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
runner = CliRunner()


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def test_train_supports_bare_eval_flag() -> None:
    """train --help must advertise --eval (bare, no --with-eval / --then-eval)."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "--eval" in output, f"--eval not found in train --help output:\n{output}"


def test_train_supports_bare_export_flag() -> None:
    """train --help must advertise --export (bare, no --with-export / --then-export)."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "--export" in output, f"--export not found in train --help output:\n{output}"


def test_run_is_alias_for_train_eval_export() -> None:
    """run --help must mention it is an alias or reference train --eval --export."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output).lower()
    assert "alias" in output or "train --eval --export" in output, (
        f"run --help does not mention alias or train --eval --export:\n{_plain(result.output)}"
    )


def test_no_with_flags_exist() -> None:
    """No --with-eval / --with-export / --then-eval / --then-export flags anywhere in CLI."""
    forbidden = ("--with-eval", "--with-export", "--then-eval", "--then-export")
    for cmd in ("train", "eval", "run"):
        result = runner.invoke(app, [cmd, "--help"])
        output = _plain(result.output)
        for flag in forbidden:
            assert flag not in output, (
                f"Forbidden flag {flag!r} found in `{cmd} --help` output:\n{output}"
            )
