"""Predict-fits-8GB validation — #142 / R11.

Proves the min_gpu_qlora-class QLoRA adapter can run prediction within the
8 GB / CC 7.5 small-card budget (PREDICT_8GB_BUDGET_GB = 7.0 GB).

Strategy:
  1. Train a tiny QLoRA adapter on the 2-image tiny_coco fixture using
     configs/examples/min_gpu_qlora.yaml (the narrowest decoder-only scope,
     fp16, 8 GB-honest config).
  2. Run batch=1 predict with --no-merge-adapter (keeps 4-bit base resident —
     the lowest-footprint path a real 8 GB-card user would choose).
  3. Measure peak VRAM around the run_predict() call.
  4. Assert peak <= PREDICT_8GB_BUDGET_GB * 1024**3.
  5. Assert predictions.json is written and decodable.

Markers: gpu_t4 (CC >= 7.5, <= 16 GB) — auto-skips on CPU.
Requires the real SAM 3.1 checkpoint and bitsandbytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.predict.budget import PREDICT_8GB_BUDGET_GB
from custom_sam_peft.predict.runner import PredictOptions, run_predict
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

pytestmark = [
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

_MIN_GPU_QLORA_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "examples" / "min_gpu_qlora.yaml"
)

# ---------------------------------------------------------------------------
# Helpers (replicated from test_gpu_predict.py)
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
        dtype="float16",
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

    run_dirs = sorted((tmp_path / "train_out").glob("*"))
    assert run_dirs, f"No run directory found under {tmp_path / 'train_out'}"
    adapter_dir = run_dirs[-1]
    candidates = [adapter_dir, *sorted(adapter_dir.glob("**/adapter_config.json"))]
    for candidate in candidates:
        if candidate.name == "adapter_config.json":
            return candidate.parent
    return adapter_dir


# ---------------------------------------------------------------------------
# Test: predict-fits-8GB (#142 / R11)
# ---------------------------------------------------------------------------


@pytest.mark.gpu_t4
@pytest.mark.requires_bnb
@pytest.mark.skipif(
    not _bnb_available(), reason="bitsandbytes not installed; QLoRA training unavailable"
)
def test_predict_fits_8gb(
    tmp_path: Path,
    tiny_coco_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate that min_gpu_qlora predict peak VRAM fits the 8 GB small-card budget.

    Trains a tiny QLoRA adapter (min_gpu_qlora.yaml, decoder-only fp16) on the
    2-image tiny_coco fixture, then runs batch=1 predict with --no-merge-adapter
    (the lowest-footprint path for a real 8 GB / CC 7.5 user).

    Asserts:
      - Peak VRAM during run_predict() <= PREDICT_8GB_BUDGET_GB * 1024**3 (7.0 GB).
      - predictions.json is written and every RLE mask decodes correctly.

    Measured on RTX 5070 Ti (CC 12.0, 16 GB) — the 8 GB assertion is valid because
    fp16 + no-merge-adapter memory usage is device-architecture-independent at the
    model/batch scale tested here. (#142 / R11)
    """
    adapter_dir = _train_and_get_adapter(
        tmp_path, _MIN_GPU_QLORA_CONFIG, tiny_coco_dir, monkeypatch
    )

    img_path = _make_synthetic_image(tmp_path, size=1024)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat,dog",
        checkpoint=adapter_dir,
        merge_adapter=False,  # --no-merge-adapter: keep 4-bit base resident
        batch_size=1,
    )

    torch.cuda.reset_peak_memory_stats()
    run_predict(opts)
    peak_bytes = torch.cuda.max_memory_allocated()

    budget_bytes = PREDICT_8GB_BUDGET_GB * 1024**3
    peak_gb = peak_bytes / 1024**3
    assert peak_bytes <= budget_bytes, (
        f"Predict peak VRAM {peak_gb:.2f} GB exceeds 8 GB small-card budget "
        f"{PREDICT_8GB_BUDGET_GB:.1f} GB. This config is NOT safe for 8 GB cards. "
        f"(#142 / R11)"
    )

    _assert_predictions_decodable(tmp_path / "out", orig_h=1024, orig_w=1024)
