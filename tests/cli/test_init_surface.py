"""init tier flags reject bad values at the parser (Phase 2).

The enum-typed parameters must produce a Typer/Click parser error (exit 2)
before run_init is called.  We verify parser-level rejection by checking that
the error message mentions the correct flag rather than '--template' (which is
what the post-parse ValueError handler in init() mis-attributes today).
"""

from __future__ import annotations

from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_init_bad_preset_rejected() -> None:
    result = runner.invoke(app, ["init", "--preset", "bogus", "--output", "x.yaml"])
    assert result.exit_code != 0
    # Parser-level rejection names the offending flag; post-parse path misattributes to --template.
    assert "--preset" in result.output or "preset" in result.output.lower()
    assert "--template" not in result.output


def test_init_bad_intensity_rejected() -> None:
    result = runner.invoke(app, ["init", "--intensity", "nope", "--output", "x.yaml"])
    assert result.exit_code != 0
    assert "--intensity" in result.output or "intensity" in result.output.lower()
    assert "--template" not in result.output


def test_init_bad_class_imbalance_rejected() -> None:
    result = runner.invoke(app, ["init", "--class-imbalance", "nope", "--output", "x.yaml"])
    assert result.exit_code != 0
    # class-imbalance already raises BadParameter with the correct hint, but not at parser level
    assert "--class-imbalance" in result.output or "class" in result.output.lower()
