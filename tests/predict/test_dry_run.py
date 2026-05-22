"""Tests for the --dry-run short-circuit in predict/runner.py.

All tests are CPU-only; no real model is loaded.
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pytest
from PIL import Image as PILImage

from custom_sam_peft.predict.runner import PredictOptions, run_predict

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_image_dir(tmp_path: Path, n: int = 3) -> Path:
    """Write *n* synthetic PNG images into tmp_path and return it."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(n):
        PILImage.new("RGB", (32, 32), color=(i * 30, i * 20, i * 10)).save(
            img_dir / f"img_{i:03d}.png"
        )
    return img_dir


def _make_opts(
    tmp_path: Path,
    *,
    images: Path | None = None,
    dry_run: bool = True,
    prompts: str = "cat,dog",
    n_images: int = 3,
) -> PredictOptions:
    if images is None:
        images = _make_image_dir(tmp_path, n=n_images)
    return PredictOptions(
        images=images,
        prompts=prompts,
        output=tmp_path / "out",
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.3,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        batch_size=1,
        seed=0,
        dry_run=dry_run,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Test 1: dry-run short-circuits before model load
# ---------------------------------------------------------------------------


def test_dry_run_short_circuits_before_model_load(tmp_path: Path) -> None:
    """With dry_run=True, load_sam31 is never called even if it would raise."""
    opts = _make_opts(tmp_path, dry_run=True)

    # Patch load_sam31 at its source module — the runner imports it lazily inside
    # the function body, so we patch the canonical location.
    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31",
        side_effect=RuntimeError("load_sam31 should NOT be called in dry-run mode"),
    ):
        # Should return without raising (dry-run exits gracefully)
        report = run_predict(opts)

    # dry-run returns a stub report (n_images, n_predictions=0, elapsed_sec)
    assert report is not None


# ---------------------------------------------------------------------------
# Test 2: dry-run prints first 10 images and all prompts
# ---------------------------------------------------------------------------


def test_dry_run_prints_first_10_images_all_prompts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout contains the first 10 resolved image paths and every prompt."""
    # Create 12 images so we can check the "first 10" truncation
    opts = _make_opts(tmp_path, prompts="cat,dog,person", n_images=12)

    # Dry run shouldn't even reach load_sam31, but patch it anyway as a guard
    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31",
        side_effect=RuntimeError("load_sam31 should NOT be called"),
    ):
        run_predict(opts)

    captured = capsys.readouterr().out

    # All prompts must appear
    for p in ["cat", "dog", "person"]:
        assert p in captured, f"Prompt {p!r} missing from dry-run stdout"

    # At least 10 image references should appear — count filenames present
    n_found = sum(1 for i in range(12) if f"img_{i:03d}.png" in captured)
    assert n_found >= 10, f"Expected >=10 image paths in dry-run output; found {n_found}"


# ---------------------------------------------------------------------------
# Test 3: dry-run writes nothing to filesystem
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Output directory remains untouched (not even created) during dry-run."""
    out_dir = tmp_path / "out"
    opts = _make_opts(tmp_path)
    opts = PredictOptions(
        images=opts.images,
        prompts=opts.prompts,
        output=out_dir,
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.3,
        top_k=100,
        save_masks="rle",
        visualize=False,
        device="cpu",
        dtype="float32",
        batch_size=1,
        seed=0,
        dry_run=True,
        verbose=False,
    )

    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31",
        side_effect=RuntimeError("should not be called"),
    ):
        run_predict(opts)

    assert not out_dir.exists(), "dry-run must not create the output directory"


# ---------------------------------------------------------------------------
# Test 4: dry-run prints resolved config
# ---------------------------------------------------------------------------


def test_dry_run_prints_resolved_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Stdout contains the resolved config block (model name, device, dtype, image_size)."""
    opts = _make_opts(tmp_path, dry_run=True)

    with mock.patch(
        "custom_sam_peft.models.sam3.load_sam31",
        side_effect=RuntimeError("should not be called"),
    ):
        run_predict(opts)

    captured = capsys.readouterr().out

    # Key resolved-config fields should be printed
    assert "facebook/sam3.1" in captured, "Resolved model name missing from dry-run output"
    assert "cpu" in captured, "Device missing from dry-run output"
    assert "float32" in captured, "Dtype missing from dry-run output"
    # image_size (default 1024) should appear
    assert "1024" in captured, "image_size missing from dry-run output"
