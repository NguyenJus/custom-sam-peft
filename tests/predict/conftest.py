"""Shared fixtures for tests/predict/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """A nested directory with allowed and disallowed image files."""
    sub = tmp_path / "sub"
    sub.mkdir()

    # allowed extensions
    Image.new("RGB", (4, 4)).save(tmp_path / "a.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "b.png")
    Image.new("RGB", (4, 4)).save(sub / "c.jpeg")
    Image.new("RGB", (4, 4)).save(sub / "d.bmp")

    # disallowed — should be filtered out
    (tmp_path / "e.gif").write_bytes(b"GIF89a")
    (tmp_path / "notes.txt").write_text("hello")

    return tmp_path


@pytest.fixture()
def single_png(tmp_path: Path) -> Path:
    p = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(p)
    return p


@pytest.fixture()
def txt_manifest(tmp_path: Path) -> Path:
    """A .txt manifest with comments, blank lines, and relative + absolute paths."""
    img_abs = tmp_path / "abs.jpg"
    Image.new("RGB", (4, 4)).save(img_abs)

    rel_dir = tmp_path / "rel_imgs"
    rel_dir.mkdir()
    img_rel = rel_dir / "rel.png"
    Image.new("RGB", (4, 4)).save(img_rel)

    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        f"# this is a comment\n\n  # indented comment\n{img_abs}\nrel_imgs/rel.png\n\n",
        encoding="utf-8",
    )
    return manifest


@pytest.fixture()
def json_manifest(tmp_path: Path) -> Path:
    img1 = tmp_path / "x.jpg"
    img2 = tmp_path / "y.png"
    Image.new("RGB", (4, 4)).save(img1)
    Image.new("RGB", (4, 4)).save(img2)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([str(img1), str(img2)]),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture()
def prompts_file(tmp_path: Path) -> Path:
    p = tmp_path / "classes.txt"
    p.write_text("cat\ndog\nperson\n", encoding="utf-8")
    return p
