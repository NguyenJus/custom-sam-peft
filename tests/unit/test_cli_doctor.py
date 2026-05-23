"""custom_sam_peft doctor formats run_doctor output."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


runner = CliRunner()


def test_doctor_table_output_includes_torch() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "torch" in text.lower()
    assert "python" in text.lower()


def test_doctor_json_round_trips() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "torch_version" in blob
    assert "optional_deps" in blob
    assert "core_versions" in blob


# ---------------------------------------------------------------------------
# spec/hf-utils — HuggingFace auth rendering
# ---------------------------------------------------------------------------


def test_doctor_table_includes_hf_auth_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("custom_sam_peft.diagnostics.huggingface_hub.get_token", lambda: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "HuggingFace auth" in text or "token source" in text


def test_doctor_json_reports_env_token_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert blob["hf_auth"]["token_source"] == "env"
    assert blob["hf_auth"]["has_token"] is True


def test_doctor_json_reports_cache_token_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "custom_sam_peft.diagnostics.huggingface_hub.get_token", lambda: "cache-tok"
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert blob["hf_auth"]["token_source"] == "cache"


def test_doctor_json_reports_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("custom_sam_peft.diagnostics.huggingface_hub.get_token", lambda: None)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert blob["hf_auth"]["token_source"] == "none"
    assert blob["hf_auth"]["has_token"] is False
    assert any("no HuggingFace token" in i for i in blob["issues"]), blob["issues"]


def test_doctor_json_round_trips_hf_auth_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--json` always includes the hf_auth field with both sub-keys."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("custom_sam_peft.diagnostics.huggingface_hub.get_token", lambda: None)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "hf_auth" in blob
    assert set(blob["hf_auth"].keys()) == {"token_source", "has_token"}


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): doctor --config / Data table
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, *, val: bool, val_split: bool) -> Path:
    """Write a minimal valid TrainConfig YAML to disk; return its path."""
    import yaml

    data_block: dict[str, object] = {
        "format": "coco",
        "train": {"annotations": str(tmp_path / "t.json"), "images": str(tmp_path / "imgs")},
        "prompt_mode": "text",
        "image_size": 32,
    }
    # Create the referenced files so the loader doesn't fail at path resolve.
    (tmp_path / "t.json").write_text("{}")
    (tmp_path / "imgs").mkdir(exist_ok=True)
    if val:
        data_block["val"] = {
            "annotations": str(tmp_path / "v.json"),
            "images": str(tmp_path / "vimgs"),
        }
        (tmp_path / "v.json").write_text("{}")
        (tmp_path / "vimgs").mkdir(exist_ok=True)
    if val_split:
        data_block["val_split"] = {"fraction": 0.2, "seed": 5}
    cfg = {
        "run": {"name": "doc", "output_dir": str(tmp_path / "runs"), "seed": 11},
        "data": data_block,
        "peft": {"method": "lora"},
        "train": {"epochs": 1},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_doctor_config_auto_split_renders_data_table(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, val=False, val_split=True)
    result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Data" in text
    assert "auto_split" in text
    assert "0.200" in text  # fraction formatted as 3 decimals
    assert "5" in text  # seed


def test_doctor_config_explicit_renders_data_table(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, val=True, val_split=False)
    result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Data" in text
    assert "explicit" in text


def test_doctor_config_none_renders_data_table(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, val=False, val_split=False)
    result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Data" in text
    assert "none" in text


def test_doctor_without_config_does_not_call_enumerate_or_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §7.7: doctor must never invoke the splitter or enumerate items."""

    def _must_not_run(*_a: object, **_kw: object) -> object:
        raise AssertionError("doctor must not call this")

    monkeypatch.setattr("custom_sam_peft.data.val_source._enumerate_coco_items", _must_not_run)
    monkeypatch.setattr("custom_sam_peft.data.splitter.stratified_split", _must_not_run)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
