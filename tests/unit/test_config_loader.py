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
  prompt_mode: bbox
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


def test_paths_resolved_relative_to_config_file(tmp_path: Path) -> None:
    cfg_file = _write_minimal_yaml(tmp_path / "c.yaml")
    cfg = load_config(cfg_file)
    assert Path(cfg.data.train.annotations).is_absolute()
    assert Path(cfg.data.train.annotations) == (tmp_path / "train.json").resolve()
    assert Path(cfg.data.val.images) == (tmp_path / "val").resolve()


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
