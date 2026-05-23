"""custom_sam_peft export wires save_adapter / save_merged."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    """A run-dir-shaped tree: {run_dir}/config.yaml + {run_dir}/adapter/."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "adapter").mkdir()
    (run_dir / "config.yaml").write_text(
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
    return run_dir


def _patch_export(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    import custom_sam_peft.cli.export_cmd as export_cmd

    def _fake_run_export(
        cfg: object,
        checkpoint: object,
        *,
        merge: bool = False,
        output: object = None,
    ) -> object:
        if merge:
            out = output if output is not None else (checkpoint.parent / "merged")  # type: ignore[union-attr]
            captured["saved_merged_to"] = out
        else:
            if output is None:
                raise ValueError(
                    "output is required when not merging (refusing to overwrite source checkpoint)"
                )
            out = output
            captured["saved_adapter_to"] = out
        return out

    monkeypatch.setattr(export_cmd, "run_export", _fake_run_export)


def test_export_auto_discovers_config(fake_run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    _patch_export(monkeypatch, captured)
    out = fake_run_dir.parent / "exported_adapter"
    result = CliRunner().invoke(
        app,
        ["export", "--checkpoint", str(fake_run_dir / "adapter"), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert captured["saved_adapter_to"] == out


def test_export_no_merge_requires_output(
    fake_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_export(monkeypatch, {})
    result = CliRunner().invoke(app, ["export", "--checkpoint", str(fake_run_dir / "adapter")])
    assert result.exit_code != 0
    assert "--output" in result.output or "output" in result.output.lower()


def test_export_merge_defaults_output_to_run_dir_merged(
    fake_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    _patch_export(monkeypatch, captured)
    result = CliRunner().invoke(
        app, ["export", "--checkpoint", str(fake_run_dir / "adapter"), "--merge"]
    )
    assert result.exit_code == 0, result.output
    assert captured["saved_merged_to"] == fake_run_dir / "merged"


def test_export_config_not_found(tmp_path: Path) -> None:
    ckpt = tmp_path / "lonely_adapter"
    ckpt.mkdir()
    result = CliRunner().invoke(
        app, ["export", "--checkpoint", str(ckpt), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code != 0
    assert "config" in result.output.lower()
