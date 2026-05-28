"""Tests for the YAML config loader and --override merging."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.config.loader import ConfigError, apply_overrides, load_config


def _write_minimal_yaml(p: Path) -> Path:
    p.write_text(
        """
run:
  name: t
model:
  name: facebook/sam3.1
data:
  format: coco
  train: { annotations: train.json, images: train/ }
  val: { annotations: val.json, images: val/ }
peft:
  method: lora
train:
  epochs: 3
""".lstrip()
    )
    return p


def test_load_config_returns_validated_train_config(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    cfg = load_config(cfg_file)
    assert cfg.run.name == "t"
    assert cfg.train.epochs == 3
    assert cfg.peft.method == "lora"


def test_paths_resolved_relative_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative paths in config are resolved against the process CWD, not the config file's dir."""
    # Config lives in a subdir; CWD is tmp_path (simulating project root)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg_file = _write_minimal_yaml(config_dir / "c.yaml")

    monkeypatch.chdir(tmp_path)
    cfg = load_config(cfg_file)

    assert Path(cfg.data.train.annotations).is_absolute()
    # Paths anchor at CWD (tmp_path), not at config's parent (config_dir)
    assert Path(cfg.data.train.annotations) == (tmp_path / "train.json").resolve()
    assert Path(cfg.data.val.images) == (tmp_path / "val").resolve()
    assert "configs" not in str(cfg.data.train.annotations)


def test_apply_overrides_modifies_nested_key() -> None:
    base = {"a": {"b": {"c": 1}}}
    apply_overrides(base, ["a.b.c=42"])
    assert base == {"a": {"b": {"c": 42}}}


def test_apply_overrides_parses_int_float_bool_null() -> None:
    base: dict[str, object] = {"x": {}}
    apply_overrides(base, ["x.i=7", "x.f=1.5", "x.t=true", "x.f2=false", "x.n=null"])
    assert base["x"] == {"i": 7, "f": 1.5, "t": True, "f2": False, "n": None}


def test_apply_overrides_creates_missing_intermediate_keys() -> None:
    base: dict[str, object] = {}
    apply_overrides(base, ["deeply.nested.key=value"])
    assert base == {"deeply": {"nested": {"key": "value"}}}


def test_load_config_with_override(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    cfg = load_config(cfg_file, overrides=["train.epochs=99", "peft.r=8"])
    assert cfg.train.epochs == 99
    assert cfg.peft.r == 8


def test_invalid_config_raises_config_error(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("run: { name: t }\n")
    with pytest.raises(ConfigError):
        load_config(cfg_file)


def test_malformed_override_raises(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    with pytest.raises(ConfigError, match="malformed override"):
        load_config(cfg_file, overrides=["not_an_assignment"])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_override_empty_payload_is_empty_string() -> None:
    base: dict[str, object] = {}
    apply_overrides(base, ["a.b="])
    assert base == {"a": {"b": ""}}


def test_override_empty_key_segment_raises() -> None:
    with pytest.raises(ConfigError, match="empty key segment"):
        apply_overrides({}, ["=oops"])
    with pytest.raises(ConfigError, match="empty key segment"):
        apply_overrides({}, ["a..b=1"])


def test_override_traversing_non_dict_raises() -> None:
    base: dict[str, object] = {"a": "string-not-dict"}
    with pytest.raises(ConfigError, match="non-dict"):
        apply_overrides(base, ["a.b.c=1"])


def test_paths_resolved_relative_to_cwd_not_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths in config must resolve relative to CWD, not the config file's directory.

    Regression test for the bug where running
        train --config configs/foo.yaml
    would resolve data/coco/... relative to configs/ instead of the project root.
    """
    # Simulate: project root is tmp_path/project, config lives in a subdir
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_subdir = project_root / "configs" / "examples"
    config_subdir.mkdir(parents=True)

    # Write a config with relative paths — these paths are relative to project_root (CWD),
    # NOT relative to config_subdir
    cfg_file = config_subdir / "my_config.yaml"
    cfg_file.write_text(
        """
run:
  name: subdir-test
  output_dir: ./runs
model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
data:
  format: coco
  train: { annotations: data/train.json, images: data/train/ }
  val: { annotations: data/val.json, images: data/val/ }
peft:
  method: lora
train:
  epochs: 1
""".lstrip()
    )

    # CWD is the project root, NOT the config subdir
    monkeypatch.chdir(project_root)

    cfg = load_config(cfg_file)

    # Paths should be absolute and anchored at project_root (CWD), not at config_subdir
    assert Path(cfg.data.train.annotations).is_absolute()
    assert Path(cfg.data.train.annotations) == (project_root / "data" / "train.json").resolve()
    assert Path(cfg.data.train.images) == (project_root / "data" / "train").resolve()
    assert Path(cfg.data.val.annotations) == (project_root / "data" / "val.json").resolve()
    assert Path(cfg.data.val.images) == (project_root / "data" / "val").resolve()
    assert Path(cfg.run.output_dir) == (project_root / "runs").resolve()

    # Crucially: paths must NOT include the config subdir in their ancestry
    assert "configs" not in str(cfg.data.train.annotations)
    assert "configs" not in str(cfg.run.output_dir)
