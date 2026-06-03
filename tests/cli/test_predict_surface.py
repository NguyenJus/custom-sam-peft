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


def _make_full_bundle(d: Path) -> Path:
    """A bundle dir with all required sidecars + graphs (files are stubs; CLI only checks names)."""
    d.mkdir(parents=True)
    for name in (
        "image_encoder.onnx",
        "decoder.onnx",
        "preprocessor.json",
        "model_card.json",
        "prompts.txt",
    ):
        (d / name).write_text('{"include": "all"}' if name.endswith(".json") else "x")
    return d


def test_use_onnx_and_checkpoint_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--use-onnx + --checkpoint raises BadParameter about mutual exclusion (spec §4.3)."""
    monkeypatch.setattr("custom_sam_peft.cli.predict_cmd.run_predict", lambda opts: None)
    bundle = _make_full_bundle(tmp_path / "bundle")
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
            "--use-onnx",
            str(bundle),
            "--checkpoint",
            str(ckpt),
        ],
    )
    assert result.exit_code != 0
    # Rich truncates/word-wraps the long BadParameter message; the tail survives.
    collapsed = " ".join(result.output.split()).lower()
    assert "adapter merged in" in collapsed


def test_use_onnx_missing_files_lists_them(tmp_path: Path) -> None:
    """--use-onnx pointing at an incomplete bundle reports the missing file(s) (spec §4.3)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "decoder.onnx").write_text("x")  # everything else missing
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
            "--use-onnx",
            str(bundle),
        ],
    )
    assert result.exit_code != 0
    collapsed = " ".join(result.output.split())
    assert "missing" in collapsed.lower()
    # At least one missing sidecar name should appear (Rich may wrap long names).
    assert any(part in collapsed for part in ("preprocessor", "model_card", "prompts.txt"))


def test_use_onnx_passed_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete bundle passes validation and use_onnx reaches PredictOptions (spec §4.3)."""
    from custom_sam_peft.cli import predict_cmd

    seen: dict[str, Any] = {}

    def fake_run_predict(opts: Any) -> Any:
        seen["use_onnx"] = opts.use_onnx

        class _R:
            n_images = 0
            n_predictions = 0
            elapsed_sec = 0.0

        return _R()

    monkeypatch.setattr(predict_cmd, "run_predict", fake_run_predict)
    bundle = _make_full_bundle(tmp_path / "bundle")
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
            "--use-onnx",
            str(bundle),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["use_onnx"] == bundle


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
