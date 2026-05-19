"""End-to-end CLI integration tests for `esam3 run`."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from esam3.cli.main import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def _make_cfg_yaml(tmp_path: Path, *, merge: bool = False, bbox: bool = False) -> Path:
    cfg = tmp_path / "config.yaml"
    prompt = "bbox" if bbox else "text"
    cfg.write_text(
        f"""
run: {{name: t, output_dir: {tmp_path / "runs"}, seed: 0}}
data:
  format: coco
  train: {{annotations: t.json, images: t/}}
  val: {{annotations: v.json, images: v/}}
  prompt_mode: {prompt}
peft: {{method: lora}}
train: {{epochs: 1}}
export: {{merge: {str(merge).lower()}}}
"""
    )
    return cfg


def _patch_phases(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_dir: Path,
    train_raises: Exception | None = None,
    eval_raises: Exception | None = None,
    merge_raises: Exception | None = None,
    bundle_raises: Exception | None = None,
) -> dict[str, object]:
    """Patch every phase entry point. Return a dict that records calls."""
    captured: dict[str, object] = {"order": []}

    fake_result = MagicMock(
        run_dir=run_dir,
        adapter_path=run_dir / "adapter",
        merged_path=None,
        final_metrics=None,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter").mkdir(exist_ok=True)

    fake_report = MagicMock(overall={"mAP": 0.42})
    fake_per_ex = [0.1, 0.5, 0.9]

    def _train(cfg: object, *, resume_from: object = None) -> object:
        captured["order"].append("train")  # type: ignore[union-attr]
        captured["resume_from"] = resume_from
        if train_raises is not None:
            raise train_raises
        return fake_result

    def _eval(
        cfg: object,
        *,
        checkpoint: object,
        output_dir: object,
        val_dataset: object,
        model: object,
        return_per_example_iou: bool,
        **_kw: object,
    ) -> object:
        captured["order"].append("eval")  # type: ignore[union-attr]
        captured["return_per_example_iou"] = return_per_example_iou
        if eval_raises is not None:
            raise eval_raises
        return fake_report, fake_per_ex

    def _save_merged(_wrapper: object, _path: object) -> None:
        captured["order"].append("merge")  # type: ignore[union-attr]
        if merge_raises is not None:
            raise merge_raises

    def _write_bundle(
        ctx: object, report: object, *, val_dataset: object, model_wrapper: object
    ) -> None:
        captured["order"].append("bundle")  # type: ignore[union-attr]
        captured["bundle_ctx"] = ctx
        if bundle_raises is not None:
            raise bundle_raises

    monkeypatch.setattr("esam3.cli.run_cmd.run_training", _train)
    monkeypatch.setattr("esam3.cli.run_cmd.run_eval", _eval)
    monkeypatch.setattr("esam3.cli.run_cmd.save_merged", _save_merged)
    monkeypatch.setattr("esam3.cli.run_cmd.write_bundle", _write_bundle)
    monkeypatch.setattr("esam3.cli.run_cmd.load_sam31", lambda _m: MagicMock())
    monkeypatch.setattr("esam3.cli.run_cmd.load_adapter", lambda *_a, **_kw: None)
    # Build a stub val_dataset.
    fake_ds = MagicMock(__len__=lambda self: 3, class_names=["a"])

    def _build_val(_cfg: object) -> object:
        captured["order"].append("build_val")  # type: ignore[union-attr]
        return fake_ds

    monkeypatch.setattr("esam3.cli.run_cmd._build_val_dataset", _build_val)
    return captured


def test_run_help_exits_zero() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Train + eval" in _plain(result.output)


def test_run_full_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    # Every phase called in order.
    order = captured["order"]
    assert order[0] == "train"
    # build_val may run before or after train depending on impl, but bundle is last.
    assert order[-1] == "bundle"
    assert "eval" in order
    assert captured["return_per_example_iou"] is True
    ctx = captured["bundle_ctx"]
    assert ctx.merged_dir is None
    assert ctx.merged_export_error is None
    assert ctx.per_example_iou == [0.1, 0.5, 0.9]


def test_run_train_failure_skips_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch,
        run_dir=tmp_path / "runs" / "r",
        train_raises=RuntimeError("kaboom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "kaboom" in _plain(result.output) or "kaboom" in (result.stderr or "")
    order = captured["order"]
    assert "eval" not in order
    assert "merge" not in order
    assert "bundle" not in order


def test_run_eval_failure_skips_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch,
        run_dir=tmp_path / "runs" / "r",
        eval_raises=RuntimeError("eval-boom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "merge" not in captured["order"]
    assert "bundle" not in captured["order"]


def test_run_merge_failure_still_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(
        monkeypatch,
        run_dir=tmp_path / "runs" / "r",
        merge_raises=ValueError("rank mismatch"),
    )
    cfg = _make_cfg_yaml(tmp_path, merge=True)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "merge" in captured["order"]
    assert "bundle" in captured["order"]
    ctx = captured["bundle_ctx"]
    assert ctx.merged_dir is None
    assert "rank mismatch" in (ctx.merged_export_error or "")


def test_run_bundle_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "runs" / "r"
    _patch_phases(
        monkeypatch,
        run_dir=run_dir,
        bundle_raises=RuntimeError("bundle-boom"),
    )
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code != 0
    # run_dir and adapter remain on disk.
    assert run_dir.exists()
    assert (run_dir / "adapter").exists()


def test_run_rejects_bbox_prompt_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path, bbox=True)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 2
    assert "train" not in captured["order"]
    assert "eval" not in captured["order"]
    assert "bbox" in _plain(result.output).lower()


def test_run_passes_preset_label_env_var_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ESAM3_PRESET_LABEL", "auto: 12-24GB tier")
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset_label == "auto: 12-24GB tier"


def test_run_preset_label_absent_yields_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ESAM3_PRESET_LABEL", raising=False)
    captured = _patch_phases(monkeypatch, run_dir=tmp_path / "runs" / "r")
    cfg = _make_cfg_yaml(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    ctx = captured["bundle_ctx"]
    assert ctx.preset_label is None
