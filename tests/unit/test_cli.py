"""Tests for the Typer CLI skeleton."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from esam3.cli.main import app

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s: str) -> str:
    """Strip ANSI escape sequences so substring asserts are terminal-independent."""
    return _ANSI.sub("", s)


runner = CliRunner()


def test_root_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in _plain(result.stdout)
    assert "eval" in _plain(result.stdout)
    assert "export" in _plain(result.stdout)
    assert "init" in _plain(result.stdout)
    assert "doctor" in _plain(result.stdout)


def test_train_help_exits_zero() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in _plain(result.stdout)


def test_eval_help_exits_zero() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--config" in _plain(result.stdout)
    assert "--checkpoint" in _plain(result.stdout)


def test_export_help_exits_zero() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint" in _plain(result.stdout)


def test_init_help_exits_zero() -> None:
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


def test_doctor_runs_and_prints_not_implemented() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "not yet implemented" in _plain(result.stdout).lower()


def test_train_with_valid_config_prints_not_implemented(tmp_path: object) -> None:
    # Depends on Task 16 example configs being committed.
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "configs" / "examples" / "coco_text_lora.yaml"
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "not yet implemented" in _plain(result.stdout).lower()


def test_eval_command_with_split_test_missing_data_test(tmp_path: Path) -> None:
    """`esam3 eval --split test` errors when data.test is None."""
    from esam3.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
  prompt_mode: text
peft: {method: lora}
train: {epochs: 1}
"""
    )
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path), "--split", "test"],
    )
    assert result.exit_code != 0
    assert "data.test" in result.output


def test_eval_command_save_predictions_flag_parses(monkeypatch: object, tmp_path: Path) -> None:
    """--save-predictions / --no-save-predictions override cfg.eval.save_predictions."""
    import esam3.cli.eval_cmd as eval_cmd
    from esam3.cli.main import app

    captured: dict[str, bool | None] = {}

    def fake_run(
        *,
        config: Path,
        checkpoint: Path,
        split: str,
        output: Path | None,
        save_predictions: bool | None,
    ) -> None:
        captured["save_predictions"] = save_predictions

    monkeypatch.setattr(eval_cmd, "_run_eval", fake_run)

    local_runner = CliRunner()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("placeholder")
    local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path), "--save-predictions"],
    )
    assert captured["save_predictions"] is True
    local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path), "--no-save-predictions"],
    )
    assert captured["save_predictions"] is False


def test_eval_command_rejects_qlora_method(tmp_path: Path) -> None:
    """esam3 eval --checkpoint errors when peft.method is not lora."""
    from esam3.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
  prompt_mode: text
peft: {method: qlora}
train: {epochs: 1}
"""
    )
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "qlora" in _plain(result.output).lower() or "only lora" in _plain(result.output).lower()
