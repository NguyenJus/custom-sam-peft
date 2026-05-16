"""Architectural guard: the data layer must not import TrainConfig.

The data layer accepts a `dict[str, Any]` plus `model_name: str` and
`pipeline: Literal['train','eval']` — not the full TrainConfig. Verified
via static AST walk over `src/esam3/data/`.
"""

from __future__ import annotations

import ast
from pathlib import Path

_DATA_FILES = ("coco.py", "hf.py", "transforms.py", "collate.py", "base.py")
_FORBIDDEN_NAMES = frozenset({"TrainConfig"})


def _find_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "esam3.config.schema":
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "esam3.config.schema":
                    found.append("<module>")
    return found


def test_data_layer_does_not_import_train_config() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "src" / "esam3" / "data"
    offenders: dict[str, list[str]] = {}
    for fname in _DATA_FILES:
        fp = data_dir / fname
        if not fp.is_file():
            continue
        tree = ast.parse(fp.read_text(encoding="utf-8"))
        names = _find_imports(tree)
        bad = sorted(set(names) & _FORBIDDEN_NAMES)
        if bad:
            offenders[fname] = bad
    assert offenders == {}, f"data layer imports forbidden names: {offenders}"
