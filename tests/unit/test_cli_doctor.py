"""custom_sam_peft doctor formats run_doctor output."""

from __future__ import annotations

import json
import re

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
# spec/domain-aware-augmentation-presets — doctor --config
# ---------------------------------------------------------------------------


def _write_doctor_config(tmp_path) -> str:
    """Write a minimal valid TrainConfig YAML for doctor --config tests."""
    import yaml

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text("{}")
    (data_dir / "val.json").write_text("{}")
    (data_dir / "train").mkdir()
    (data_dir / "val").mkdir()
    cfg = {
        "run": {"name": "doctor-cfg"},
        "model": {"name": "facebook/sam3.1"},
        "data": {
            "format": "coco",
            "train": {
                "annotations": str(data_dir / "train.json"),
                "images": str(data_dir / "train"),
            },
            "val": {"annotations": str(data_dir / "val.json"), "images": str(data_dir / "val")},
            "augmentations": {"preset": "medical", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_doctor_with_config_renders_resolved_augmentations(tmp_path) -> None:
    cfg_path = _write_doctor_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", cfg_path])
    assert result.exit_code == 0, result.output
    text = _plain(result.stdout)
    assert "Resolved augmentations" in text
    assert "preset" in text
    assert "medical" in text
    assert "intensity" in text


def test_doctor_with_config_renders_normalization(tmp_path) -> None:
    cfg_path = _write_doctor_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", cfg_path])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Normalization" in text
    assert "mean" in text
    assert "std" in text
    assert "resolution path" in text


def test_doctor_json_no_config_no_resolved_block() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "resolved_config" not in blob


def test_doctor_json_with_config_has_resolved_block(tmp_path) -> None:
    cfg_path = _write_doctor_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", cfg_path, "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "resolved_config" in blob
    rc = blob["resolved_config"]
    assert set(rc.keys()) == {"augmentations", "normalize", "loss"}
    assert rc["augmentations"]["preset"] == "medical"
    assert rc["augmentations"]["intensity"] == "medium"
    assert set(rc["augmentations"]["resolved"].keys()) == {
        "hflip",
        "vflip",
        "rotate90",
        "rotate_arbitrary",
        "color_jitter",
        "stain_jitter",
        "blur",
        "gauss_noise",
    }
    assert isinstance(rc["augmentations"]["steps"], list)
    assert rc["normalize"]["model_name"] == "facebook/sam3.1"
    assert rc["normalize"]["resolution_path"] in {"processor", "table-fallback", "config-fallback"}


# ---------------------------------------------------------------------------
# spec/domain-aware-loss-presets — doctor --config resolved-losses (Phase F)
# ---------------------------------------------------------------------------


def test_doctor_with_config_renders_resolved_losses(tmp_path) -> None:
    """Spec §10.2: --config renders a 'Resolved losses' table."""
    from typer.testing import CliRunner

    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    CliRunner().invoke(
        app,
        ["init", "--preset", "medical", "--class-imbalance", "moderate", "--output", str(cfg_path)],
    )
    res = CliRunner().invoke(app, ["doctor", "--config", str(cfg_path)])
    assert res.exit_code == 0, res.output
    text = _plain(res.output)
    assert "Resolved losses" in text
    assert "preset" in text and "medical" in text
    assert "class_imbalance" in text and "moderate" in text
    assert "term_classes" in text
    assert "FocalTverskyLoss" in text  # from med/moderate row


def test_doctor_json_with_config_has_loss_block(tmp_path) -> None:
    import json

    from typer.testing import CliRunner

    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    CliRunner().invoke(
        app,
        [
            "init",
            "--preset",
            "natural",
            "--class-imbalance",
            "balanced",
            "--output",
            str(cfg_path),
        ],
    )
    res = CliRunner().invoke(app, ["doctor", "--config", str(cfg_path), "--json"])
    assert res.exit_code == 0
    body = json.loads(_plain(res.output))
    assert "resolved_config" in body
    assert "loss" in body["resolved_config"]
    loss = body["resolved_config"]["loss"]
    assert loss["preset"] == "natural"
    assert loss["class_imbalance"] == "balanced"
    assert set(loss["resolved"].keys()) == {
        "mask_family",
        "box_family",
        "obj_family",
        "presence_family",
        "w_mask",
        "w_box",
        "w_obj",
        "w_presence",
        "focal_gamma",
        "focal_alpha",
        "tversky_alpha",
        "tversky_gamma",
        "boundary_weight",
    }


def test_doctor_json_without_config_no_loss_block() -> None:
    """Spec §10.2: with no --config, output has no loss block."""
    import json

    from typer.testing import CliRunner

    from custom_sam_peft.cli.main import app

    res = CliRunner().invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0
    body = json.loads(_plain(res.output))
    if "resolved_config" in body:
        assert "loss" not in body["resolved_config"]


def test_doctor_no_config_json_shape_unchanged() -> None:
    """Without --config, JSON output has the same top-level shape as the legacy run."""
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    # Top-level keys are the DoctorReport dataclass fields; resolved_config is absent.
    assert "torch_version" in blob
    assert "hf_auth" in blob
    assert "resolved_config" not in blob
