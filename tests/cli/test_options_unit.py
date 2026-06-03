"""Unit tests for cli/_options.py helpers (CPU-only, no model load)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from custom_sam_peft.cli._options import Progress, Split, discover_config, merge_cli_overrides


def test_progress_enum_values() -> None:
    assert [m.value for m in Progress] == ["auto", "on", "off", "plain"]


def test_split_enum_values() -> None:
    assert [m.value for m in Split] == ["val", "test"]


def test_discover_config_finds_sibling(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ckpt = run_dir / "checkpoints" / "step_5" / "adapter"
    ckpt.mkdir(parents=True)
    cfg = run_dir / "config.yaml"
    cfg.write_text("run:\n  name: x\n")
    assert discover_config(ckpt).resolve() == cfg.resolve()


def test_discover_config_raises_when_absent(tmp_path: Path) -> None:
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()
    with pytest.raises(typer.BadParameter):
        discover_config(ckpt)


def test_merge_appends_name_and_output_dir() -> None:
    out = merge_cli_overrides(["train.epochs=10"], name="my-run", output_dir=Path("runs/exp1"))
    assert out == ["train.epochs=10", "run.name=my-run", "run.output_dir=runs/exp1"]


def test_merge_noop_when_no_convenience_flags() -> None:
    assert merge_cli_overrides(["a.b=c"], name=None, output_dir=None) == ["a.b=c"]


def test_merge_conflict_on_name_raises() -> None:
    with pytest.raises(typer.BadParameter):
        merge_cli_overrides(["run.name=bar"], name="foo", output_dir=None)


def test_merge_conflict_on_output_dir_raises() -> None:
    with pytest.raises(typer.BadParameter):
        merge_cli_overrides(["run.output_dir=x"], name=None, output_dir=Path("y"))
