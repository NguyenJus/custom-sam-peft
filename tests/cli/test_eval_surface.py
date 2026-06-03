"""CLI surface for eval: Split enum validation + config discovery (Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


def _write_config(path: Path, tmp_path: Path) -> None:
    path.write_text(
        "run:\n  name: ev\n  output_dir: " + str(tmp_path) + "\n"
        "data:\n  format: coco\n"
        "  train:\n    annotations: a\n    images: i\n"
        "  val:\n    annotations: a\n    images: i\n"
        "peft:\n  method: lora\n"
        "train:\n  epochs: 1\n"
    )


def test_eval_bad_split_rejected_by_parser(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    _write_config(cfg, tmp_path)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output or "split" in result.output.lower()


def test_eval_discovers_sibling_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import eval_cmd

    run_dir = tmp_path / "run"
    ckpt = run_dir / "checkpoints" / "step_5" / "adapter"
    ckpt.mkdir(parents=True)
    _write_config(run_dir / "config.yaml", tmp_path)

    seen: dict[str, Any] = {}

    def fake_run_eval(cfg: Any, **k: Any) -> Any:
        seen["name"] = cfg.run.name

        class _R:
            overall: ClassVar[dict[str, float]] = {"mAP": 0.0}

        return _R()

    monkeypatch.setattr(eval_cmd, "run_eval", fake_run_eval)
    result = runner.invoke(app, ["eval", "--checkpoint", str(ckpt)])
    assert result.exit_code == 0, result.output
    assert seen["name"] == "ev"


def test_eval_baseline_without_config_still_raises(tmp_path: Path) -> None:
    # No --checkpoint, no --config: baseline eval still requires --config.
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0
    assert "config" in result.output.lower()
