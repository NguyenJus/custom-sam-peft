"""Tests for the Typer CLI skeleton."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

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


def test_doctor_runs_and_reports_environment() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "torch" in _plain(result.stdout).lower()


def test_train_invokes_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """train CLI parses config and delegates to run_training."""
    from unittest.mock import MagicMock

    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.cli.main import app

    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "configs" / "examples" / "coco_text_lora.yaml"

    fake_result = MagicMock(
        run_dir=tmp_path / "r",
        adapter_path=tmp_path / "r" / "adapter",
        final_metrics=None,
        time_limit_stop=None,
    )
    called: dict[str, object] = {}

    def fake_run(cfg_obj, *, resume_from=None):
        called["cfg"] = cfg_obj
        called["resume_from"] = resume_from
        return fake_result

    monkeypatch.setattr(train_cmd, "run_train", fake_run)

    local_runner = CliRunner()
    result = local_runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "run_dir=" in _plain(result.stdout)
    assert called["resume_from"] is None


def test_eval_command_with_split_test_missing_data_test(tmp_path: Path) -> None:
    """`custom_sam_peft eval --split test` errors when data.test is None."""
    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
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
    from unittest.mock import MagicMock

    import custom_sam_peft.cli.eval_cmd as eval_cmd
    from custom_sam_peft.cli.main import app

    captured: dict[str, bool | None] = {}

    def fake_run(
        cfg,
        *,
        checkpoint: Path,
        split: str,
        output_dir: Path | None,
        save_predictions: bool | None,
        visualize: bool | None = None,
    ):
        captured["save_predictions"] = save_predictions
        return MagicMock(overall={})

    monkeypatch.setattr(eval_cmd, "run_eval", fake_run)

    local_runner = CliRunner()
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
peft: {method: lora}
train: {epochs: 1}
"""
    )
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


def test_eval_command_accepts_qlora_method(tmp_path: Path) -> None:
    """custom_sam_peft eval --checkpoint no longer rejects peft.method=qlora.

    The command will fail for other reasons (no real checkpoint on disk), but the
    failure must NOT be the old 'only LoRA adapters' guard. QLoRA is now accepted
    and dispatched via QloraAdapter.load_from_disk.
    """
    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
peft: {method: qlora}
train: {epochs: 1}
"""
    )
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path)],
    )
    # Must NOT contain the old rejection message.
    assert "checkpoint loading currently supports only LoRA" not in _plain(result.output)
    assert "only lora" not in _plain(result.output).lower()


# ---------------------------------------------------------------------------
# --resume flag tests (train + run)
# ---------------------------------------------------------------------------


def _make_train_cfg_file(tmp_path: Path) -> Path:
    """Write a minimal valid training config and return its path."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: myrun, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
peft: {method: lora}
train: {epochs: 1}
"""
    )
    return cfg_path


def test_train_resume_no_flag_forwards_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train with no --resume flag passes resume_from=None."""
    from unittest.mock import MagicMock

    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.cli.main import app

    cfg_path = _make_train_cfg_file(tmp_path)
    fake_result = MagicMock(
        run_dir=tmp_path / "r",
        checkpoint_path=tmp_path / "r" / "adapter",
        final_metrics=None,
        time_limit_stop=None,
    )
    called: dict[str, object] = {}

    def fake_run(cfg_obj, *, resume_from=None):
        called["resume_from"] = resume_from
        return fake_result

    monkeypatch.setattr(train_cmd, "run_train", fake_run)
    local_runner = CliRunner()
    result = local_runner.invoke(app, ["train", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert called["resume_from"] is None


def test_train_resume_explicit_path_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train --resume /explicit/path forwards Path('/explicit/path') as resume_from."""
    from pathlib import Path as _Path
    from unittest.mock import MagicMock

    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.cli.main import app

    cfg_path = _make_train_cfg_file(tmp_path)
    fake_result = MagicMock(
        run_dir=tmp_path / "r",
        checkpoint_path=tmp_path / "r" / "adapter",
        final_metrics=None,
        time_limit_stop=None,
    )
    called: dict[str, object] = {}

    def fake_run(cfg_obj, *, resume_from=None):
        called["resume_from"] = resume_from
        return fake_result

    monkeypatch.setattr(train_cmd, "run_train", fake_run)
    local_runner = CliRunner()
    result = local_runner.invoke(
        app, ["train", "--config", str(cfg_path), "--resume", "/explicit/path"]
    )
    assert result.exit_code == 0, result.output
    assert called["resume_from"] == _Path("/explicit/path")


def test_train_resume_latest_calls_find_latest_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train --resume (no value) calls find_latest_checkpoint and passes result to run_train."""
    from pathlib import Path as _Path
    from unittest.mock import MagicMock

    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.cli.main import app

    cfg_path = _make_train_cfg_file(tmp_path)
    fake_result = MagicMock(
        run_dir=tmp_path / "r",
        checkpoint_path=tmp_path / "r" / "adapter",
        final_metrics=None,
        time_limit_stop=None,
    )
    resolved_ckpt = _Path(tmp_path / "myrun-2026-01-01T00-00-00" / "checkpoints" / "step_10")
    called: dict[str, object] = {}

    def fake_run(cfg_obj, *, resume_from=None):
        called["resume_from"] = resume_from
        return fake_result

    def fake_find_latest(cfg_obj):
        called["find_cfg"] = cfg_obj
        return resolved_ckpt

    monkeypatch.setattr(train_cmd, "run_train", fake_run)
    monkeypatch.setattr(train_cmd, "find_latest_checkpoint", fake_find_latest)
    local_runner = CliRunner()
    result = local_runner.invoke(app, ["train", "--config", str(cfg_path), "--resume"])
    assert result.exit_code == 0, result.output
    assert called["resume_from"] == resolved_ckpt


def test_train_resume_latest_no_checkpoint_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """train --resume with no checkpoints found exits non-zero with error message."""
    from unittest.mock import MagicMock

    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.cli.main import app
    from custom_sam_peft.errors import CheckpointError

    cfg_path = _make_train_cfg_file(tmp_path)
    fake_result = MagicMock(
        run_dir=tmp_path / "r",
        checkpoint_path=tmp_path / "r" / "adapter",
        final_metrics=None,
        time_limit_stop=None,
    )

    monkeypatch.setattr(train_cmd, "run_train", lambda *a, **kw: fake_result)
    monkeypatch.setattr(
        train_cmd,
        "find_latest_checkpoint",
        lambda cfg: (_ for _ in ()).throw(CheckpointError("no checkpoint found for myrun in /tmp")),
    )
    local_runner = CliRunner()
    result = local_runner.invoke(app, ["train", "--config", str(cfg_path), "--resume"])
    assert result.exit_code != 0


def test_run_resume_no_flag_forwards_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run with no --resume flag passes resume_from=None."""
    from custom_sam_peft.cli import run_cmd
    from custom_sam_peft.cli.main import app

    cfg_path = _make_train_cfg_file(tmp_path)
    called: dict[str, object] = {}

    def fake_orchestrate(cfg_obj, resume, mode, *, visualize=None, config_path=None):
        called["resume"] = resume
        return 0

    monkeypatch.setattr(run_cmd, "_orchestrate", fake_orchestrate)
    local_runner = CliRunner()
    result = local_runner.invoke(app, ["run", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert called["resume"] is None


def test_run_resume_explicit_path_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run --resume /explicit/path forwards Path('/explicit/path')."""
    from pathlib import Path as _Path

    from custom_sam_peft.cli import run_cmd
    from custom_sam_peft.cli.main import app

    cfg_path = _make_train_cfg_file(tmp_path)
    called: dict[str, object] = {}

    def fake_orchestrate(cfg_obj, resume, mode, *, visualize=None, config_path=None):
        called["resume"] = resume
        return 0

    monkeypatch.setattr(run_cmd, "_orchestrate", fake_orchestrate)
    local_runner = CliRunner()
    result = local_runner.invoke(
        app, ["run", "--config", str(cfg_path), "--resume", "/explicit/path"]
    )
    assert result.exit_code == 0, result.output
    assert called["resume"] == _Path("/explicit/path")


def test_run_resume_latest_calls_find_latest_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run --resume (no value) calls find_latest_checkpoint and passes result to _orchestrate."""
    from pathlib import Path as _Path

    from custom_sam_peft.cli import run_cmd
    from custom_sam_peft.cli.main import app

    cfg_path = _make_train_cfg_file(tmp_path)
    resolved_ckpt = _Path(tmp_path / "myrun-2026-01-01T00-00-00" / "checkpoints" / "step_10")
    called: dict[str, object] = {}

    def fake_orchestrate(cfg_obj, resume, mode, *, visualize=None, config_path=None):
        called["resume"] = resume
        return 0

    def fake_find_latest(cfg_obj):
        called["find_cfg"] = cfg_obj
        return resolved_ckpt

    monkeypatch.setattr(run_cmd, "_orchestrate", fake_orchestrate)
    monkeypatch.setattr(run_cmd, "find_latest_checkpoint", fake_find_latest)
    local_runner = CliRunner()
    result = local_runner.invoke(app, ["run", "--config", str(cfg_path), "--resume"])
    assert result.exit_code == 0, result.output
    assert called["resume"] == resolved_ckpt


def test_run_resume_latest_no_checkpoint_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run --resume with no checkpoints found exits non-zero."""
    from custom_sam_peft.cli import run_cmd
    from custom_sam_peft.cli.main import app
    from custom_sam_peft.errors import CheckpointError

    cfg_path = _make_train_cfg_file(tmp_path)

    monkeypatch.setattr(run_cmd, "_orchestrate", lambda *a, **kw: 0)
    monkeypatch.setattr(
        run_cmd,
        "find_latest_checkpoint",
        lambda cfg: (_ for _ in ()).throw(CheckpointError("no checkpoint found for myrun in /tmp")),
    )
    local_runner = CliRunner()
    result = local_runner.invoke(app, ["run", "--config", str(cfg_path), "--resume"])
    assert result.exit_code != 0
