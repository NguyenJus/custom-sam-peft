"""Input and prompt resolution for csp predict."""

from __future__ import annotations

import glob as _glob
import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


def _is_allowed(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMAGE_EXTS


def resolve_images(spec: str | Path) -> list[Path]:
    """Resolve *spec* to a sorted list of image paths.

    Accepted forms:
    - Directory: recursive walk, collect files with allowed extensions.
    - Glob string (contains ``*`` or ``?``): ``glob.glob(spec, recursive=True)``.
    - Single image file: extension must be in the allowlist.
    - ``.txt`` manifest: one path per line; ``#``-prefixed and blank lines skipped;
      relative entries resolve against the manifest's parent directory.
    - ``.json`` manifest: must decode to a ``list[str]``.

    Empty result raises ``typer.BadParameter``.
    """
    p = Path(spec)
    paths: list[Path]

    if p.is_dir():
        paths = [f for f in p.rglob("*") if f.is_file() and _is_allowed(f)]

    elif isinstance(spec, str) and ("*" in spec or "?" in spec):
        raw = _glob.glob(spec, recursive=True)
        paths = [Path(r) for r in raw if _is_allowed(Path(r))]

    elif p.is_file() and p.suffix.lower() == ".txt":
        paths = _load_txt_manifest(p)

    elif p.is_file() and p.suffix.lower() == ".json":
        paths = _load_json_manifest(p)

    elif p.is_file():
        paths = [] if not _is_allowed(p) else [p]

    else:
        # Treat as a glob string even if it doesn't contain * or ?
        raw = _glob.glob(str(spec), recursive=True)
        paths = [Path(r) for r in raw if _is_allowed(Path(r))]

    result = sorted(set(paths), key=lambda x: str(x.resolve()))

    if not result:
        raise typer.BadParameter(f"no images resolved from {spec}")

    return result


def _load_txt_manifest(manifest: Path) -> list[Path]:
    parent = manifest.parent
    paths: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip()
        if not line or line.startswith("#"):
            continue
        entry = Path(raw_line.strip())
        resolved = entry if entry.is_absolute() else parent / entry
        if _is_allowed(resolved):
            paths.append(resolved)
    return paths


def _load_json_manifest(manifest: Path) -> list[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"JSON manifest must decode to a list of strings, got {type(data).__name__}"
        )
    parent = manifest.parent
    paths: list[Path] = []
    for item in data:
        if not isinstance(item, str):
            raise TypeError(f"JSON manifest entries must be strings, got {type(item).__name__}")
        entry = Path(item)
        resolved = entry if entry.is_absolute() else parent / entry
        if _is_allowed(resolved):
            paths.append(resolved)
    return paths


def parse_prompts(spec: str | Path) -> list[str]:
    """Resolve *spec* to a deduplicated, ordered list of class-name strings.

    *spec* may be either a comma-separated string or a path to a UTF-8 file
    with one class name per line.

    Empty result raises ``typer.BadParameter``.
    """
    p = Path(spec)
    if p.is_file():
        raw_entries = p.read_text(encoding="utf-8").splitlines()
    else:
        raw_entries = str(spec).split(",")

    seen: dict[str, None] = {}
    for entry in raw_entries:
        stripped = entry.strip()
        if stripped and stripped not in seen:
            seen[stripped] = None

    result = list(seen.keys())

    if not result:
        raise typer.BadParameter("--prompts must resolve to at least one non-empty class name")

    return result
