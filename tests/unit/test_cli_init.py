"""esam3 init writes a template that reloads cleanly through load_config."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from esam3.cli.main import app
from esam3.config.loader import load_config

runner = CliRunner()


def _make_data_paths(tmp_path: Path) -> None:
    """Touch the four data paths the template references so load_config validates."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "train.json").write_text("{}")
    (tmp_path / "data" / "val.json").write_text("{}")
    (tmp_path / "data" / "train").mkdir(exist_ok=True)
    (tmp_path / "data" / "val").mkdir(exist_ok=True)


def test_init_writes_lora_template(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--template", "coco-text-lora", "--output", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    cfg = load_config(out)
    assert cfg.peft.method == "lora"


def test_init_writes_qlora_template(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--template", "coco-text-qlora", "--output", str(out)],
    )
    assert result.exit_code == 0
    cfg = load_config(out)
    assert cfg.peft.method == "qlora"


def test_init_refuses_clobber(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    out.write_text("existing\n")
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code != 0
    assert out.read_text() == "existing\n"


def test_init_force_overwrites(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    out.write_text("existing\n")
    result = runner.invoke(app, ["init", "--output", str(out), "--force"])
    assert result.exit_code == 0
    assert "existing" not in out.read_text()
    assert yaml.safe_load(out.read_text())["peft"]["method"] == "lora"


def test_init_unknown_template_rejected(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--template", "hf-text"])
    assert result.exit_code != 0
    assert "hf-text" in result.output or "unknown" in result.output.lower()
