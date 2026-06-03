"""Bundle sidecar loaders for the ONNX predict path (spec §6, §8.3).

Reads the three load-bearing sidecars an ONNX bundle ships:
``preprocessor.json``, ``model_card.json``, and ``prompts.txt``. This module
imports ONLY ``json`` + ``pathlib`` — no torch, no onnxruntime — so it can be
loaded inside the torch-free ORT inference core (spec §8.4, §10.10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREPROCESSOR_FILE = "preprocessor.json"
MODEL_CARD_FILE = "model_card.json"
PROMPTS_FILE = "prompts.txt"


def load_preprocessor(bundle_dir: Path) -> dict[str, Any]:
    """Load and parse ``preprocessor.json`` from a bundle directory (spec §6.1)."""
    data: dict[str, Any] = json.loads(
        (Path(bundle_dir) / PREPROCESSOR_FILE).read_text(encoding="utf-8")
    )
    return data


def load_model_card(bundle_dir: Path) -> dict[str, Any]:
    """Load and parse ``model_card.json`` from a bundle directory (spec §6.2)."""
    data: dict[str, Any] = json.loads(
        (Path(bundle_dir) / MODEL_CARD_FILE).read_text(encoding="utf-8")
    )
    return data


def load_prompts(bundle_dir: Path) -> list[str]:
    """Load ``prompts.txt`` as newline-delimited class names, file order preserved (spec §6.3)."""
    text = (Path(bundle_dir) / PROMPTS_FILE).read_text(encoding="utf-8")
    # Strip a single trailing newline, then split; do NOT re-sort (index = category order).
    return text.rstrip("\n").split("\n") if text.rstrip("\n") else []
