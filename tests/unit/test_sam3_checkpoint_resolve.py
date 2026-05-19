"""Tests for esam3.models.sam3._resolve_checkpoint_path.

Auto-download on miss, strict re-check, and the local_dir=None error path.
``download_model`` is patched at the consumer's import site
(``esam3.models.sam3.download_model``) because ``_resolve_checkpoint_path``
will ``from esam3.utils.huggingface import download_model`` at module top
and bind the name into ``esam3.models.sam3``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.config.schema import ModelConfig
from esam3.models.sam3 import _resolve_checkpoint_path


def test_resolve_checkpoint_returns_path_when_file_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Short-circuit: file already on disk; download_model not called."""
    (tmp_path / "ckpt.pt").write_bytes(b"x")
    cfg = ModelConfig(local_dir=str(tmp_path), checkpoint_file="ckpt.pt")

    calls: list[tuple[object, ...]] = []

    def _fake_dl(*args: object, **kwargs: object) -> Path:
        calls.append((args, kwargs))
        return tmp_path

    monkeypatch.setattr("esam3.models.sam3.download_model", _fake_dl)

    out = _resolve_checkpoint_path(cfg)
    assert out == tmp_path / "ckpt.pt"
    assert calls == []  # MUST NOT call download_model when file already exists


def test_resolve_checkpoint_auto_downloads_on_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing file: download_model called with (name, local_dir, revision=cfg.revision)."""
    cfg = ModelConfig(
        name="facebook/sam3.1",
        local_dir=str(tmp_path),
        checkpoint_file="ckpt.pt",
        revision=None,
    )

    calls: list[dict[str, object]] = []

    def _fake_dl(name: str, local_dir: Path, *, revision: str | None = None) -> Path:
        calls.append({"name": name, "local_dir": local_dir, "revision": revision})
        (Path(local_dir) / "ckpt.pt").write_bytes(b"x")  # simulate fetch
        return Path(local_dir)

    monkeypatch.setattr("esam3.models.sam3.download_model", _fake_dl)

    out = _resolve_checkpoint_path(cfg)
    assert out == tmp_path / "ckpt.pt"
    assert calls == [
        {"name": "facebook/sam3.1", "local_dir": Path(str(tmp_path)), "revision": None}
    ]


def test_resolve_checkpoint_passes_revision_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = ModelConfig(
        name="facebook/sam3.1",
        local_dir=str(tmp_path),
        checkpoint_file="ckpt.pt",
        revision="abc123",
    )

    calls: list[dict[str, object]] = []

    def _fake_dl(name: str, local_dir: Path, *, revision: str | None = None) -> Path:
        calls.append({"revision": revision})
        (Path(local_dir) / "ckpt.pt").write_bytes(b"x")
        return Path(local_dir)

    monkeypatch.setattr("esam3.models.sam3.download_model", _fake_dl)

    _resolve_checkpoint_path(cfg)
    assert calls[0]["revision"] == "abc123"


def test_resolve_checkpoint_raises_when_download_leaves_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict re-check: download "succeeded" but the expected file is absent."""
    cfg = ModelConfig(
        name="facebook/sam3.1",
        local_dir=str(tmp_path),
        checkpoint_file="ckpt.pt",
    )

    def _fake_dl(name: str, local_dir: Path, *, revision: str | None = None) -> Path:
        # Intentionally do NOT create ckpt.pt — simulate revision-without-file.
        return Path(local_dir)

    monkeypatch.setattr("esam3.models.sam3.download_model", _fake_dl)

    with pytest.raises(FileNotFoundError, match="still missing"):
        _resolve_checkpoint_path(cfg)


def test_resolve_checkpoint_local_dir_none_hints_at_esam3_init() -> None:
    cfg = ModelConfig(local_dir=None)
    with pytest.raises(FileNotFoundError) as excinfo:
        _resolve_checkpoint_path(cfg)
    msg = str(excinfo.value)
    assert "local_dir is None" in msg
    assert "esam3 init" in msg
