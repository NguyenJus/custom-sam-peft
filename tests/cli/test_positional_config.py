"""train/run accept positional config + hidden --config alias (Phase 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_min_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "run:\n  name: pc\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )
    return cfg


def _patch_train(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], tmp_path: Path) -> None:
    from custom_sam_peft.cli import train_cmd
    from custom_sam_peft.eval._artifacts import EvalArtifacts

    def fake_run_train(cfg: Any, **k: Any) -> EvalArtifacts:
        seen["name"] = cfg.run.name
        return EvalArtifacts(
            checkpoint_path=tmp_path / "adapter",
            peft_method="lora",
            run_dir=tmp_path,
            final_metrics=None,
        )

    monkeypatch.setattr(train_cmd, "run_train", fake_run_train)


def _patch_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import run_cmd

    def fake_orchestrate(cfg: Any, *a: Any, **k: Any) -> int:
        return 0

    monkeypatch.setattr(run_cmd, "_orchestrate", fake_orchestrate)


def test_train_positional_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    _patch_train(monkeypatch, seen, tmp_path)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", str(cfg)])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "pc"


def test_train_config_alias_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    _patch_train(monkeypatch, seen, tmp_path)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "pc"


def test_train_no_config_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0


def test_run_positional_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["run", str(cfg)])
    assert result.exit_code == 0, result.output


def test_run_config_alias_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch)
    cfg = _write_min_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output


def test_run_no_config_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
