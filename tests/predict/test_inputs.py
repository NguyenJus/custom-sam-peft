"""Tests for predict/inputs.py — resolve_images."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from PIL import Image

from custom_sam_peft.predict.inputs import resolve_images


def test_resolve_images_directory_recursive(image_dir: Path) -> None:
    """Recursive dir walk collects only allowed extensions, including subdirs."""
    result = resolve_images(str(image_dir))
    names = {p.name for p in result}
    # allowed files
    assert "a.jpg" in names
    assert "b.png" in names
    assert "c.jpeg" in names
    assert "d.bmp" in names
    # disallowed filtered out
    assert "e.gif" not in names
    assert "notes.txt" not in names


def test_resolve_images_glob_recursive(tmp_path: Path) -> None:
    """Glob with ** returns sorted absolute paths for matching files."""
    sub = tmp_path / "sub"
    sub.mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "a.jpg")
    Image.new("RGB", (4, 4)).save(sub / "b.jpg")

    glob_spec = str(tmp_path / "**" / "*.jpg")
    result = resolve_images(glob_spec)

    assert len(result) == 2
    # result must be sorted by absolute-path string
    paths_str = [str(p.resolve()) for p in result]
    assert paths_str == sorted(paths_str)


def test_resolve_images_single_file(single_png: Path) -> None:
    """A single image-extension file is returned as a one-element list."""
    result = resolve_images(str(single_png))
    assert result == [single_png.resolve()]


def test_resolve_images_txt_manifest(txt_manifest: Path) -> None:
    """TXT manifest: comments and blank lines skipped; relative paths resolve to manifest parent."""
    result = resolve_images(str(txt_manifest))
    names = {p.name for p in result}
    assert "abs.jpg" in names
    assert "rel.png" in names
    # no comments or blanks in the resolved list
    assert len(result) == 2


def test_resolve_images_txt_manifest_relative_to_manifest_parent(tmp_path: Path) -> None:
    """Relative entries in a .txt manifest resolve against manifest's parent, not cwd."""
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.new("RGB", (4, 4)).save(imgs / "photo.png")

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("imgs/photo.png\n", encoding="utf-8")

    result = resolve_images(str(manifest))
    assert len(result) == 1
    assert result[0] == (tmp_path / "imgs" / "photo.png").resolve()


def test_resolve_images_json_manifest(json_manifest: Path) -> None:
    """JSON manifest decoding to list[str] is parsed correctly."""
    result = resolve_images(str(json_manifest))
    names = {p.name for p in result}
    assert "x.jpg" in names
    assert "y.png" in names


def test_resolve_images_json_manifest_non_list_raises(tmp_path: Path) -> None:
    """JSON manifest that decodes to a non-list raises an error."""
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps({"paths": ["/some/img.jpg"]}), encoding="utf-8")

    with pytest.raises((typer.BadParameter, ValueError, TypeError)):
        resolve_images(str(bad_manifest))


def test_resolve_images_extension_allowlist(tmp_path: Path) -> None:
    """Files outside the allowed extension set are filtered out."""
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")  # allowed
    (tmp_path / "b.gif").write_bytes(b"GIF89a")  # not allowed
    (tmp_path / "c.mp4").write_bytes(b"\x00")  # not allowed

    result = resolve_images(str(tmp_path))
    assert len(result) == 1
    assert result[0].name == "a.png"


def test_resolve_images_extension_case_insensitive(tmp_path: Path) -> None:
    """Extension matching is case-insensitive (e.g. .JPG is treated as .jpg)."""
    p = tmp_path / "img.JPG"
    Image.new("RGB", (4, 4)).save(p)
    result = resolve_images(str(tmp_path))
    assert len(result) == 1


def test_resolve_images_rgba_to_rgb_implicit(tmp_path: Path) -> None:
    """RGBA image is included in resolved list and can be converted to RGB."""
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 4)).save(p)
    result = resolve_images(str(tmp_path))
    assert len(result) == 1
    # PIL.Image.open(...).convert("RGB") must succeed
    img = Image.open(result[0]).convert("RGB")
    assert img.mode == "RGB"


def test_resolve_images_unreadable_warn_and_skip(tmp_path: Path) -> None:
    """A corrupt (zero-byte) file with a valid extension is still in the resolved list.

    resolve_images is path-level only; the WARN-and-skip behavior fires in the
    runner's per-image load loop. The path must be present so the runner can
    decide whether to skip it.
    """
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_bytes(b"")  # zero bytes — corrupt JPEG
    valid = tmp_path / "ok.png"
    Image.new("RGB", (4, 4)).save(valid)

    result = resolve_images(str(tmp_path))
    names = {p.name for p in result}
    assert "bad.jpg" in names
    assert "ok.png" in names


def test_resolve_images_zero_result_raises(tmp_path: Path) -> None:
    """An empty dir (no allowed-ext files) raises typer.BadParameter."""
    (tmp_path / "notes.txt").write_text("nothing here")

    with pytest.raises(typer.BadParameter) as exc_info:
        resolve_images(str(tmp_path))

    assert "no images resolved from" in str(exc_info.value)


def test_resolve_images_sort_determinism(tmp_path: Path) -> None:
    """Returned list is sorted by absolute-path string regardless of fs ordering."""
    # Create files whose names don't sort the same as creation order
    paths = []
    for name in ["z.png", "a.png", "m.jpg"]:
        p = tmp_path / name
        Image.new("RGB", (4, 4)).save(p)
        paths.append(p)

    result = resolve_images(str(tmp_path))
    resolved_strs = [str(p.resolve()) for p in result]
    assert resolved_strs == sorted(resolved_strs)
