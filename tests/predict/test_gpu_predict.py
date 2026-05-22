"""GPU integration tests for ``csp predict`` — gated by mark.gpu.

All four tests require:
  - A CUDA device with compute capability >= 7.5 (requires_compatible_gpu)
  - The real SAM 3.1 checkpoint at models/sam3.1/sam3.1_multiplex.pt (requires_checkpoint)

These tests are excluded from default pytest collection / CI and are intended
to be run explicitly:
    pytest -m gpu tests/predict/test_gpu_predict.py -v

Mirrors the module-level mark pattern from tests/gpu/test_real_train_overfits.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.predict.runner import PredictOptions, run_predict
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _RecordingTracker

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

# ---------------------------------------------------------------------------
# Paths to example YAML configs used to drive training fixtures
# ---------------------------------------------------------------------------

_LORA_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_lora.yaml"
_QLORA_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "examples" / "gpu_smoke_qlora.yaml"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_image(tmp_path: Path, *, size: int = 1024) -> Path:
    """Write a synthetic RGB PNG of (size x size) to tmp_path/images/synthetic.png."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "synthetic.png"
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(img_path)
    return img_path


def _make_base_opts(
    tmp_path: Path,
    *,
    images: Path,
    prompts: str = "cat,dog",
    checkpoint: Path | None = None,
    merge_adapter: bool = True,
    batch_size: int = 1,
) -> PredictOptions:
    """Construct a PredictOptions suitable for GPU integration tests."""
    return PredictOptions(
        images=images,
        prompts=prompts,
        output=tmp_path / "out",
        checkpoint=checkpoint,
        merge_adapter=merge_adapter,
        config=None,
        score_threshold=0.0,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cuda",
        dtype="bfloat16",
        batch_size=batch_size,
        seed=42,
        dry_run=False,
        verbose=False,
    )


def _assert_predictions_decodable(out_dir: Path, orig_h: int, orig_w: int) -> None:
    """Assert predictions.json exists and every RLE decodes to (orig_h, orig_w)."""
    pred_file = out_dir / "predictions.json"
    assert pred_file.exists(), f"predictions.json not written to {out_dir}"

    entries = json.loads(pred_file.read_text())
    assert isinstance(entries, list), "predictions.json must be a JSON array"

    for entry in entries:
        seg = entry.get("segmentation")
        assert seg is not None, f"entry missing 'segmentation': {entry}"
        # Ensure counts is bytes for pycocotools.decode
        decode_rle: dict[str, Any] = dict(seg)
        counts = decode_rle["counts"]
        if isinstance(counts, str):
            decode_rle["counts"] = counts.encode("ascii")
        m = mask_utils.decode(decode_rle)
        assert m.shape == (orig_h, orig_w), (
            f"decoded mask shape {m.shape} != expected ({orig_h}, {orig_w})"
        )
        assert m.dtype == np.uint8, f"decoded mask dtype {m.dtype} != uint8"


def _train_and_get_adapter(
    tmp_path: Path,
    config_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run a short training loop and return the path to the saved adapter directory."""
    cfg = load_config(
        config_path,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path / 'train_out'}",
        ],
    )
    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)
    run_training(cfg)

    # Locate the adapter checkpoint saved during training
    run_dirs = sorted((tmp_path / "train_out").glob("*"))
    assert run_dirs, f"No run directory found under {tmp_path / 'train_out'}"
    # The adapter dir is the run dir itself (PEFT saves adapter_config.json at root)
    adapter_dir = run_dirs[-1]
    # If there are step-subdirs, pick the latest one that has adapter_config.json
    candidates = [adapter_dir, *sorted(adapter_dir.glob("**/adapter_config.json"))]
    for candidate in candidates:
        if candidate.name == "adapter_config.json":
            return candidate.parent
    # Fallback: return run dir and let the test fail with a clear message
    return adapter_dir


# ---------------------------------------------------------------------------
# Test 1: Base model (no adapter), real SAM 3.1, synthetic 1024x1024 image
# ---------------------------------------------------------------------------


def test_predict_base_model_cuda(tmp_path: Path) -> None:
    """Real facebook/sam3.1 load + warmup + one synthetic 1024x1024 image + two text prompts.

    Asserts predictions.json is written and every RLE decodes to (1024, 1024) uint8.
    """
    img_path = _make_synthetic_image(tmp_path, size=1024)
    opts = _make_base_opts(tmp_path, images=img_path.parent, prompts="cat,dog")
    run_predict(opts)
    _assert_predictions_decodable(tmp_path / "out", orig_h=1024, orig_w=1024)


# ---------------------------------------------------------------------------
# Test 2: LoRA adapter trained on-the-fly via run_training
# ---------------------------------------------------------------------------


def test_predict_lora_adapter_cuda(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train a tiny LoRA on tiny_coco, then run csp predict with --checkpoint.

    Exercises merge_lora on a real PEFT wrapper.
    Asserts predictions.json written and all RLE masks decodable.
    """
    adapter_dir = _train_and_get_adapter(tmp_path, _LORA_CONFIG, tiny_coco_dir, monkeypatch)

    img_path = _make_synthetic_image(tmp_path, size=1024)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat,dog",
        checkpoint=adapter_dir,
        merge_adapter=True,
    )
    run_predict(opts)
    _assert_predictions_decodable(tmp_path / "out", orig_h=1024, orig_w=1024)


# ---------------------------------------------------------------------------
# Test 3: QLoRA adapter with --no-merge-adapter (skip 4-bit dequant-merge path)
# ---------------------------------------------------------------------------


def test_predict_qlora_no_merge_cuda(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train a tiny QLoRA on tiny_coco, then run csp predict with --no-merge-adapter.

    Intentionally does NOT exercise merge_and_unload so the 4-bit dequant path
    is skipped. Asserts predictions.json written and all RLE masks decodable.
    """
    adapter_dir = _train_and_get_adapter(tmp_path, _QLORA_CONFIG, tiny_coco_dir, monkeypatch)

    img_path = _make_synthetic_image(tmp_path, size=1024)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat,dog",
        checkpoint=adapter_dir,
        merge_adapter=False,  # --no-merge-adapter: skip merge_and_unload
    )
    run_predict(opts)
    _assert_predictions_decodable(tmp_path / "out", orig_h=1024, orig_w=1024)


# ---------------------------------------------------------------------------
# Test 4: VRAM hint log emitted when free VRAM > 12 GB and batch_size == 1
# ---------------------------------------------------------------------------


def test_predict_vram_hint_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On cuda with >12 GB free VRAM + batch_size=1, runner logs the VRAM hint at INFO.

    Skipped when free VRAM <= 12 GB (hint would not fire).
    Logger: custom_sam_peft.predict.runner
    Expected string: "free VRAM is >12 GB; consider --batch-size 4 or 8."
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes <= 12 * 1024**3:
        pytest.skip(reason="free VRAM is not >12 GB; hint would not fire")

    img_path = _make_synthetic_image(tmp_path, size=1024)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat",
        checkpoint=None,
        batch_size=1,
    )

    with caplog.at_level(logging.INFO, logger="custom_sam_peft.predict.runner"):
        run_predict(opts)

    assert "free VRAM is >12 GB" in caplog.text, (
        f"Expected VRAM hint not found in caplog.text. Captured:\n{caplog.text}"
    )
