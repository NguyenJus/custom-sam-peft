"""predict CLI surface: removed flags error; merge derived from adapter kind."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize(
    "flag",
    [["--device", "cuda"], ["--dtype", "float32"], ["--seed", "1"], ["--merge-adapter"]],
)
def test_removed_predict_flags_error(flag: list[str], tmp_path: Path) -> None:
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    result = runner.invoke(
        app,
        [
            "predict",
            "--images",
            str(imgs),
            "--prompts",
            "a",
            "--output",
            str(tmp_path / "o"),
            *flag,
        ],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def _make_lora_ckpt(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    return d


def test_merge_derived_lora_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import predict_cmd

    seen: dict[str, Any] = {}

    def fake_run_predict(opts: Any) -> Any:
        seen["merge"] = opts.merge_adapter

        class _R:
            n_images = 0
            n_predictions = 0
            elapsed_sec = 0.0

        return _R()

    monkeypatch.setattr(predict_cmd, "run_predict", fake_run_predict)
    monkeypatch.setattr(predict_cmd, "detect_adapter_kind", lambda p: "lora")
    ckpt = _make_lora_ckpt(tmp_path / "ckpt")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    result = runner.invoke(
        app,
        [
            "predict",
            "--images",
            str(imgs),
            "--prompts",
            "a",
            "--output",
            str(tmp_path / "o"),
            "--checkpoint",
            str(ckpt),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["merge"] is True


def test_merge_derived_qlora_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import predict_cmd

    seen: dict[str, Any] = {}

    def fake_run_predict(opts: Any) -> Any:
        seen["merge"] = opts.merge_adapter

        class _R:
            n_images = 0
            n_predictions = 0
            elapsed_sec = 0.0

        return _R()

    monkeypatch.setattr(predict_cmd, "run_predict", fake_run_predict)
    monkeypatch.setattr(predict_cmd, "detect_adapter_kind", lambda p: "qlora")
    ckpt = _make_lora_ckpt(tmp_path / "ckpt")
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    result = runner.invoke(
        app,
        [
            "predict",
            "--images",
            str(imgs),
            "--prompts",
            "a",
            "--output",
            str(tmp_path / "o"),
            "--checkpoint",
            str(ckpt),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["merge"] is False
