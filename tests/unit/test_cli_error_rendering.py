"""Tests for the four-part CLI error renderer.

Spec §7.3 / Task 6.3: CustomSamPeftError subclasses carry expected/found/fix fields;
cli/main._render_error assembles them into the four-part user-facing message.

Design note
-----------
``CliRunner.invoke(app, ...)`` invokes ``app()`` directly, not ``main()``.
The ``main()`` wrapper (which calls ``_render_error`` and writes to stderr) never
fires during test invocations.  We therefore test in two complementary ways:

1. **Unit tests for ``_render_error``** — verify the four-part format string.
2. **Integration tests via ``runner.invoke``** — verify that the ``ConfigError``
   raised by a bad-config path carries the correct ``expected``/``found``/``fix``
   fields, and that rendering those fields produces the expected output.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from custom_sam_peft.cli.main import _render_error, app
from custom_sam_peft.errors import (
    CheckpointError,
    ConfigError,
    CustomSamPeftError,
    DataError,
    ModelError,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Unit tests — _render_error
# ---------------------------------------------------------------------------


def test_render_error_base_message_only() -> None:
    """Without extra fields, only the message and traceback hint appear."""
    err = CustomSamPeftError("something went wrong")
    rendered = _render_error(err)
    assert "something went wrong" in rendered
    assert "Rerun with -v for full traceback." in rendered
    assert "Expected:" not in rendered
    assert "Found:" not in rendered
    assert "Fix:" not in rendered


def test_render_error_all_four_parts() -> None:
    """All four parts appear when expected/found/fix are set."""
    err = CustomSamPeftError(
        "bad thing happened",
        expected="a good thing",
        found="a bad thing",
        fix="do the right thing",
    )
    rendered = _render_error(err)
    assert "bad thing happened" in rendered
    assert "Expected: a good thing" in rendered
    assert "Found: a bad thing" in rendered
    assert "Fix: do the right thing" in rendered
    assert "Rerun with -v for full traceback." in rendered
    assert "-v" in rendered


def test_render_error_partial_fields() -> None:
    """Only populated fields appear in the output."""
    err = CustomSamPeftError("oops", expected="something", fix="do this")
    rendered = _render_error(err)
    assert "Expected: something" in rendered
    assert "Found:" not in rendered
    assert "Fix: do this" in rendered


def test_render_error_config_error() -> None:
    """ConfigError with all four fields renders correctly."""
    err = ConfigError(
        "config not found",
        field_path="<path>",
        expected="an existing YAML file",
        found="'/no/such/file.yaml' (does not exist or is not a file)",
        fix="create the file or pass the correct path with --config",
    )
    rendered = _render_error(err)
    assert "config not found" in rendered
    assert "<path>" in rendered
    assert "Expected: an existing YAML file" in rendered
    assert "Found:" in rendered
    assert "Fix:" in rendered
    assert "Rerun with -v for full traceback." in rendered


def test_render_error_checkpoint_error() -> None:
    """CheckpointError with expected/found/fix renders all four parts."""
    err = CheckpointError(
        "peft_method mismatch",
        expected="adapter dir matching peft_method='lora'",
        found="adapter dir appears to be 'qlora'",
        fix="ensure --resume points to the correct checkpoint directory",
    )
    rendered = _render_error(err)
    assert "peft_method mismatch" in rendered
    assert "Expected:" in rendered
    assert "Found:" in rendered
    assert "Fix:" in rendered


def test_render_error_data_error() -> None:
    """DataError (no kwargs) renders with message and traceback hint only."""
    err = DataError("dataset not found")
    rendered = _render_error(err)
    assert "dataset not found" in rendered
    assert "Rerun with -v for full traceback." in rendered


def test_render_error_model_error_with_fields() -> None:
    """ModelError with fields renders correctly."""
    err = ModelError("model build failed", expected="valid model config", fix="check model.type")
    rendered = _render_error(err)
    assert "Expected: valid model config" in rendered
    assert "Fix: check model.type" in rendered


# ---------------------------------------------------------------------------
# Integration tests — bad config path trips loader → ConfigError with fields
# ---------------------------------------------------------------------------


def test_config_error_renders_four_parts(tmp_path: Path) -> None:
    """A non-existent config path raises a ConfigError with all four fields populated.

    The four-part rendering is verified by calling _render_error on the caught
    exception, since CliRunner invokes app() not main().
    """
    bad_config = tmp_path / "bad.yaml"
    # Deliberately do NOT create the file

    result = runner.invoke(app, ["train", "--config", str(bad_config)])

    assert result.exit_code == 1

    # The exception must be a ConfigError with all four fields
    exc = result.exception
    assert isinstance(exc, ConfigError), f"expected ConfigError, got {type(exc)}"
    assert exc.expected is not None, "expected field should be set"
    assert exc.found is not None, "found field should be set"
    assert exc.fix is not None, "fix field should be set"

    # The four-part renderer must produce all four parts
    out = _render_error(exc)
    assert "Expected:" in out
    assert "Found:" in out
    assert "Fix:" in out
    assert "-v" in out


def test_invalid_yaml_config_error_has_four_fields(tmp_path: Path) -> None:
    """Invalid YAML raises ConfigError with expected/found/fix."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":\tinvalid: yaml: {{\n")

    result = runner.invoke(app, ["train", "--config", str(bad_yaml)])

    assert result.exit_code == 1
    exc = result.exception
    assert isinstance(exc, ConfigError)
    assert exc.expected is not None
    assert exc.found is not None
    assert exc.fix is not None
    out = _render_error(exc)
    assert "Expected:" in out
    assert "Found:" in out
    assert "Fix:" in out
    assert "-v" in out


def test_schema_validation_config_error_has_four_fields(tmp_path: Path) -> None:
    """A config that parses as YAML but fails schema validation raises four-field ConfigError."""
    bad_config = tmp_path / "bad.yaml"
    # Valid YAML, but missing required fields → pydantic ValidationError → ConfigError
    bad_config.write_text("data:\n  format: coco\n")

    result = runner.invoke(app, ["train", "--config", str(bad_config)])

    assert result.exit_code == 1
    exc = result.exception
    assert isinstance(exc, ConfigError)
    assert exc.expected is not None
    assert exc.found is not None
    assert exc.fix is not None
    out = _render_error(exc)
    assert "Expected:" in out
    assert "Fix:" in out
    assert "-v" in out


# ---------------------------------------------------------------------------
# Backwards-compatibility guard: existing error constructors still work
# ---------------------------------------------------------------------------


def test_config_error_backwards_compat() -> None:
    """ConfigError still works without the new kwargs (backwards compat)."""
    err = ConfigError("old-style error", field_path="some.field")
    assert err.field_path == "some.field"
    assert "some.field" in str(err)
    assert err.expected is None
    assert err.found is None
    assert err.fix is None


def test_base_error_backwards_compat() -> None:
    """CustomSamPeftError still works with message-only construction."""
    err = CustomSamPeftError("plain message")
    assert str(err) == "plain message"
    assert err.expected is None
    assert err.found is None
    assert err.fix is None
