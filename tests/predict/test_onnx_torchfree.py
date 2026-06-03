"""Torch-free guarantees and sidecar loaders for the ONNX predict path (spec §10.10, §8.4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from custom_sam_peft.predict import onnx_bundle


@pytest.fixture()
def bundle_dir(tmp_path: Path) -> Path:
    """A minimal bundle dir carrying the three load-bearing sidecars."""
    d = tmp_path / "bundle"
    d.mkdir()
    (d / onnx_bundle.PREPROCESSOR_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_size": 1008,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "max_pixel_value": 255.0,
                "channels": 3,
                "channel_semantics": "rgb",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (d / onnx_bundle.MODEL_CARD_FILE).write_text(
        json.dumps({"schema_version": 1, "name": "facebook/sam3.1", "include": "all"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    # Deliberately unsorted, trailing newline — order must be preserved verbatim.
    (d / onnx_bundle.PROMPTS_FILE).write_text("zebra\ncat\ndog\n", encoding="utf-8")
    return d


def test_load_preprocessor_returns_dict(bundle_dir: Path) -> None:
    """load_preprocessor parses preprocessor.json into a dict."""
    pp = onnx_bundle.load_preprocessor(bundle_dir)
    assert pp["image_size"] == 1008
    assert pp["mean"] == [0.5, 0.5, 0.5]
    assert pp["channel_semantics"] == "rgb"


def test_load_model_card_returns_dict(bundle_dir: Path) -> None:
    """load_model_card parses model_card.json into a dict."""
    card = onnx_bundle.load_model_card(bundle_dir)
    assert card["name"] == "facebook/sam3.1"
    assert card["include"] == "all"


def test_load_prompts_preserves_order(bundle_dir: Path) -> None:
    """load_prompts returns lines in file order without re-sorting, no trailing blank."""
    prompts = onnx_bundle.load_prompts(bundle_dir)
    assert prompts == ["zebra", "cat", "dog"]


def test_onnx_bundle_is_torch_free() -> None:
    """onnx_bundle imports only json + pathlib — never torch or onnxruntime."""
    src = Path(onnx_bundle.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "import onnxruntime" not in src


def test_subprocess_ort_core_import_is_torch_free() -> None:
    """A child process loads the inference-core modules + references _OrtCore, no torch (§10.10).

    The modules are loaded directly from their source files via ``importlib.util`` so the
    eager ``custom_sam_peft/__init__.py`` train-chain import (which pulls torch) is bypassed:
    the guarantee under test is that the inference-core *module code* (``onnx_bundle``,
    ``_multiplex``, and ``_OrtCore`` in ``onnx_session``) introduces no torch import, with
    ``import torch`` kept lazy inside ``OnnxSam3Session`` (never at module top).
    """
    root = Path(onnx_bundle.__file__).resolve().parent.parent  # .../src/custom_sam_peft
    child = f"""
import importlib.util, sys
from pathlib import Path

root = Path({str(root)!r})


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_load("_mx", "models/_multiplex.py")
_load("_bundle", "predict/onnx_bundle.py")
sess = _load("_sess", "predict/onnx_session.py")
core = getattr(sess, "_OrtCore")
assert core is not None
assert "torch" not in sys.modules, sorted(k for k in sys.modules if "torch" in k)
print("OK")
"""
    result = subprocess.run(  # noqa: S603 - trusted: sys.executable with a literal probe script
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
