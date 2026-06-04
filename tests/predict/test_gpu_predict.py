"""GPU integration tests for ``csp predict`` — all gpu_t4 tier.

All four tests require:
  - A CUDA device with compute capability >= 7.5 (requires_compatible_gpu)
  - The real SAM 3.1 checkpoint at models/sam3.1/sam3.1_multiplex.pt (requires_checkpoint)

These tests are excluded from default pytest collection / CI and are intended
to be run explicitly on a gpu_t4-capable device (CC 7.5 floor: Tesla T4 /
RTX 5070 Ti). bf16 is coerced to fp16 below CC 8.0.

Mirrors the module-level mark pattern from tests/gpu/test_real_train_overfits.py.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE
from custom_sam_peft.predict.runner import PredictOptions, run_predict
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _bnb_available, _RecordingTracker

pytestmark = [
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


def _make_synthetic_image(tmp_path: Path, *, size: int = SAM3_IMAGE_SIZE) -> Path:
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
# Test 1: Base model (no adapter), real SAM 3.1, synthetic 1008x1008 image
# ---------------------------------------------------------------------------


@pytest.mark.gpu_t4
def test_predict_base_model_cuda(tmp_path: Path) -> None:
    """Real facebook/sam3.1 load + warmup + one synthetic 1008x1008 image + two text prompts.

    Asserts predictions.json is written and every RLE decodes to (1008, 1008) uint8.
    """
    img_path = _make_synthetic_image(tmp_path, size=SAM3_IMAGE_SIZE)
    opts = _make_base_opts(tmp_path, images=img_path.parent, prompts="cat,dog")
    run_predict(opts)
    _assert_predictions_decodable(tmp_path / "out", orig_h=SAM3_IMAGE_SIZE, orig_w=SAM3_IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Test 2: LoRA adapter trained on-the-fly via run_training
# ---------------------------------------------------------------------------


@pytest.mark.gpu_t4
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

    img_path = _make_synthetic_image(tmp_path, size=SAM3_IMAGE_SIZE)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat,dog",
        checkpoint=adapter_dir,
        merge_adapter=True,
    )
    run_predict(opts)
    _assert_predictions_decodable(tmp_path / "out", orig_h=SAM3_IMAGE_SIZE, orig_w=SAM3_IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Test 3: QLoRA adapter with --no-merge-adapter (skip 4-bit dequant-merge path)
# ---------------------------------------------------------------------------


@pytest.mark.gpu_t4
@pytest.mark.requires_bnb
@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
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

    img_path = _make_synthetic_image(tmp_path, size=SAM3_IMAGE_SIZE)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat,dog",
        checkpoint=adapter_dir,
        merge_adapter=False,  # --no-merge-adapter: skip merge_and_unload
    )
    run_predict(opts)
    _assert_predictions_decodable(tmp_path / "out", orig_h=SAM3_IMAGE_SIZE, orig_w=SAM3_IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Test 4: VRAM hint log emitted when free VRAM > 12 GB and batch_size == 1
# ---------------------------------------------------------------------------


@pytest.mark.gpu_t4
def test_predict_vram_hint_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On cuda + batch_size=1, the runner emits the >12 GB VRAM hint iff the free
    VRAM measured *after model load* exceeds 12 GB.

    The runner checks free VRAM at Step 7, after SAM 3.1 is resident on the GPU,
    so a pre-load reading in the test disagrees with the runner on any card where
    free VRAM straddles 12 GB across the model load (e.g. the 16 GB 5070 Ti, #209).
    We therefore assert conditionally against the free value the runner itself
    logs at its gate point — the exact number the hint is gated on.

    Logger: custom_sam_peft.predict.runner
    Expected string: "free VRAM is >12 GB; consider --batch-size 4 or 8."
    """
    img_path = _make_synthetic_image(tmp_path, size=SAM3_IMAGE_SIZE)
    opts = _make_base_opts(
        tmp_path,
        images=img_path.parent,
        prompts="cat",
        checkpoint=None,
        batch_size=1,
    )

    with caplog.at_level(logging.DEBUG, logger="custom_sam_peft.predict.runner"):
        run_predict(opts)

    # Recover the free-VRAM reading the runner took at its gate point (post-load).
    match = re.search(r"VRAM hint check: free=(\d+) bytes", caplog.text)
    assert match is not None, (
        f"Runner did not log its VRAM-hint free-VRAM reading. Captured:\n{caplog.text}"
    )
    free_bytes = int(match.group(1))
    hint_logged = "free VRAM is >12 GB" in caplog.text

    if free_bytes > 12 * 1024**3:
        assert hint_logged, (
            f"free VRAM {free_bytes / 1024**3:.2f} GB > 12 GB but hint was not logged. "
            f"Captured:\n{caplog.text}"
        )
    else:
        assert not hint_logged, (
            f"free VRAM {free_bytes / 1024**3:.2f} GB <= 12 GB but hint was logged. "
            f"Captured:\n{caplog.text}"
        )
