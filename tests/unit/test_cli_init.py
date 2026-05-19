"""esam3 init writes a template that reloads cleanly through load_config."""

from __future__ import annotations

from pathlib import Path

import pytest
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


# ---------------------------------------------------------------------------
# spec/hf-utils — download-weights flag matrix
# ---------------------------------------------------------------------------


def test_init_no_download_weights_skips_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-download-weights: skip download, print remediation hint."""
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"

    calls: list[object] = []
    monkeypatch.setattr(
        "esam3.cli.init_cmd.download_model",
        lambda *a, **kw: calls.append((a, kw)),
    )

    result = runner.invoke(
        app,
        ["init", "--output", str(out), "--no-download-weights"],
    )
    assert result.exit_code == 0, result.output
    assert calls == []  # MUST NOT download
    assert "skipping" in result.output.lower() or "fetched on first" in result.output


def test_init_short_circuits_when_weights_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File already at <local_dir>/<checkpoint_file>: don't download."""
    _make_data_paths(tmp_path)
    # Materialize the resolved checkpoint path BEFORE the CLI runs. The
    # template's local_dir is relative ("models/sam3.1"), so we chdir into
    # tmp_path so the resolved Path is tmp_path/models/sam3.1/sam3.1_multiplex.pt.
    weights = tmp_path / "models" / "sam3.1" / "sam3.1_multiplex.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    calls: list[object] = []
    monkeypatch.setattr(
        "esam3.cli.init_cmd.download_model",
        lambda *a, **kw: calls.append((a, kw)),
    )

    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert calls == []
    out_lower = result.output.lower()
    assert "already present" in out_lower or "skipping download" in out_lower


def test_init_non_tty_skips_with_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No flag + non-TTY stdin: skip; print hint pointing at --download-weights."""
    _make_data_paths(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("esam3.cli.init_cmd.sys.stdin.isatty", lambda: False)

    calls: list[object] = []
    monkeypatch.setattr(
        "esam3.cli.init_cmd.download_model",
        lambda *a, **kw: calls.append((a, kw)),
    )

    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert calls == []
    assert "--download-weights" in result.output or "fetched on first" in result.output


def test_init_download_weights_yes_triggers_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--download-weights --yes: download silently with revision pass-through."""
    _make_data_paths(tmp_path)
    monkeypatch.chdir(tmp_path)

    calls: list[dict[str, object]] = []

    def _fake_dl(name: str, local_dir: Path, *, revision: str | None = None) -> Path:
        calls.append({"name": name, "local_dir": local_dir, "revision": revision})
        return Path(local_dir)

    monkeypatch.setattr("esam3.cli.init_cmd.download_model", _fake_dl)

    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--output", str(out), "--download-weights", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["name"] == "facebook/sam3.1"
    assert calls[0]["local_dir"] == Path("models/sam3.1")
    assert calls[0]["revision"] is None


def test_init_download_failure_surfaces_as_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_model RuntimeError → CLI exit code != 0 + message printed."""
    _make_data_paths(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _boom(*a: object, **kw: object) -> Path:
        raise RuntimeError("could not download 'facebook/sam3.1': the repo is gated.")

    monkeypatch.setattr("esam3.cli.init_cmd.download_model", _boom)

    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--output", str(out), "--download-weights", "--yes"],
    )
    assert result.exit_code != 0
    assert "gated" in result.output.lower()
