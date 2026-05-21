"""Load + validate YAML configs into a TrainConfig.

Responsibilities:
  - Load YAML.
  - Apply `--override key.subkey=value` flags onto the dict.
  - Resolve every path in DataConfig relative to the config file's directory.
  - Validate via pydantic; surface errors as ConfigError.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from custom_sam_peft.config.schema import TrainConfig

_PATH_KEYS: tuple[tuple[str, ...], ...] = (
    ("data", "train", "annotations"),
    ("data", "train", "images"),
    ("data", "val", "annotations"),
    ("data", "val", "images"),
    ("run", "output_dir"),
)


class ConfigError(ValueError):
    """Raised when a config cannot be loaded, parsed, or validated."""


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
) -> TrainConfig:
    """Load YAML at `path`, apply overrides, resolve paths, return TrainConfig."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    if overrides:
        apply_overrides(raw, overrides)

    _resolve_paths(raw, base_dir=p.parent.resolve())

    try:
        return TrainConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e


def apply_overrides(target: dict[str, Any], overrides: Sequence[str]) -> None:
    """Mutate `target` in place: each override is `dotted.key=value`.

    Values are parsed YAML-style (`true`/`null`/numbers map to Python types).
    A bare RHS (`key=`) is interpreted as the empty string.
    """
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"malformed override (expected key=value): {ov!r}")
        key, _, raw_value = ov.partition("=")
        keys = key.split(".")
        if not key or any(not k for k in keys):
            raise ConfigError(f"malformed override (empty key segment): {ov!r}")
        node = target
        for k in keys[:-1]:
            existing = node.get(k)
            if existing is None:
                existing = {}
                node[k] = existing
            elif not isinstance(existing, dict):
                raise ConfigError(
                    f"override {ov!r} traverses non-dict at '{k}' (have {type(existing).__name__})"
                )
            node = existing
        node[keys[-1]] = _parse_scalar(raw_value)


def _parse_scalar(s: str) -> Any:
    """YAML-style scalar parsing for override values; empty string passes through."""
    if s == "":
        return ""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> None:
    for key_path in _PATH_KEYS:
        node: Any = raw
        for k in key_path[:-1]:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if not isinstance(node, dict):
            continue
        leaf = key_path[-1]
        val = node.get(leaf)
        if isinstance(val, str):
            candidate = Path(val)
            if not candidate.is_absolute():
                node[leaf] = str((base_dir / candidate).resolve())
