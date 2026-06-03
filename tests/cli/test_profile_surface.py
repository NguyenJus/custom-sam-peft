"""CLI surface tests for the `csp profile` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_profiler() -> Any:
    """Reset + disable the module-global profiler around every test so a test
    that enables it (and may raise before its manual cleanup) cannot leak
    _ENABLED=True into later tests in the same pytest process."""
    from custom_sam_peft import profiling

    profiling.disable()
    profiling.reset()
    yield
    profiling.disable()
    profiling.reset()


def _write_config(path: Path, tmp_path: Path) -> None:
    path.write_text(
        "run:\n  name: prof\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )


# ---------------------------------------------------------------------------
# Registration / help
# ---------------------------------------------------------------------------


def test_profile_command_registered() -> None:
    """The 'profile' subcommand must appear in the top-level help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "profile" in result.output


def test_profile_help_works() -> None:
    """csp profile --help must exit 0 and mention eval/profiling."""
    result = runner.invoke(app, ["profile", "--help"])
    assert result.exit_code == 0, result.output
    # Key words from the docstring
    assert "profile" in result.output.lower()


def test_profile_no_args_returns_nonzero() -> None:
    """Without --config or --checkpoint the command must fail gracefully."""
    result = runner.invoke(app, ["profile"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Dry-run path (no model, no real eval)
# ---------------------------------------------------------------------------


def test_profile_dry_run(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, tmp_path)
    result = runner.invoke(app, ["profile", "--config", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output


# ---------------------------------------------------------------------------
# Monkeypatched run_eval: enable() called + snapshot written
# ---------------------------------------------------------------------------


def test_profile_enables_profiler_and_dumps_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify enable() is called and snapshot JSON is written, without a real model."""
    from custom_sam_peft import profiling
    from custom_sam_peft.cli import profile_cmd

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, tmp_path)

    enabled_at_call: list[bool] = []

    def fake_run_eval(cfg: Any, **kwargs: Any) -> Any:
        enabled_at_call.append(profiling.is_enabled())
        # Simulate some profiled work by recording a bucket directly.
        profiling.note(n_images=3)
        profiling.incr("eval.forwards", by=2)

        class _R:
            overall: ClassVar[dict[str, float]] = {"mAP": 0.42}

        return _R()

    monkeypatch.setattr(profile_cmd, "run_eval", fake_run_eval)

    # Reset profiler state so earlier test runs don't bleed in.
    profiling.disable()
    profiling.reset()

    result = runner.invoke(app, ["profile", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output

    # profiling.enable() must have been called before run_eval
    assert enabled_at_call == [True]

    # Snapshot JSON must be present and parseable
    snap = tmp_path / "profile_snapshot.json"
    assert snap.is_file(), f"profile_snapshot.json not written to {tmp_path}"
    data = json.loads(snap.read_text())
    assert "buckets" in data
    assert "meta" in data
    assert data["meta"].get("n_images") == 3

    # Table header must appear in output
    assert "TOTAL(timed)" in result.output

    # Restore profiler to a clean disabled state for subsequent tests.
    profiling.disable()
    profiling.reset()
