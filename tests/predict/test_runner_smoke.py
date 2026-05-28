"""Smoke tests for predict/runner.py — end-to-end with a stub nn.Module.

All tests are CPU-only; load_sam31 is monkeypatched to return a stub module.

The stub matches the contract of queries_to_coco_results exactly:
  pred_logits:       (1, Q, 1)
  pred_boxes:        (1, Q, 4)   normalized cxcywh in [0, 1]
  pred_masks:        (1, Q, H_low, W_low)
  presence_logit_dec:(1, 1)
"""

from __future__ import annotations

import json
import logging
import sys
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image as PILImage

from custom_sam_peft.predict.runner import PredictOptions, PredictReport, run_predict

# ---------------------------------------------------------------------------
# Stub nn.Module — lives here so conftest.py provides it as a fixture
# ---------------------------------------------------------------------------

Q = 4  # number of queries
H_LOW = 16  # low-res mask height
W_LOW = 16  # low-res mask width
HIGH_SCORE = 0.9  # logit that maps to high probability after sigmoid
LOW_SCORE = -10.0  # logit that maps to near-zero score


class _StubSamModule(torch.nn.Module):
    """Minimal stub whose forward returns tensors matching the multiplex contract.

    For a batch of B images with K_g classes each, the output batch dim is B * K_g
    (matching the real SAM 3.1 multiplex forward shape expected by _row_outputs).
    """

    def __init__(
        self,
        n_queries: int = Q,
        score_logit: float = HIGH_SCORE,
        presence_logit: float = HIGH_SCORE,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.score_logit = score_logit
        self.presence_logit = presence_logit
        self.forward_call_count = 0

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[Any],
        support: Any = None,
    ) -> dict[str, torch.Tensor]:
        self.forward_call_count += 1
        b = images.shape[0]
        # For multiplex TextPrompts, K_g is the number of classes; output dim = B * K_g.
        from custom_sam_peft.data.base import TextPrompts as _TP

        k_g = len(prompts[0].classes) if prompts and isinstance(prompts[0], _TP) else 1
        total = b * k_g
        return {
            "pred_logits": torch.full((total, self.n_queries, 1), self.score_logit),
            "pred_boxes": torch.full((total, self.n_queries, 4), 0.5),
            "pred_masks": torch.zeros(total, self.n_queries, H_LOW, W_LOW),
            "presence_logit_dec": torch.full((total, 1), self.presence_logit),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_dir(tmp_path: Path, n: int = 1) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(n):
        PILImage.new("RGB", (64, 64), color=(i * 30, 100, 200)).save(img_dir / f"img_{i:03d}.png")
    return img_dir


def _make_opts(
    tmp_path: Path,
    *,
    images: Path | None = None,
    prompts: str = "cat",
    checkpoint: Path | None = None,
    score_threshold: float = 0.0,
    top_k: int = 100,
    save_masks: str = "rle",
    visualize: bool = False,
    merge_adapter: bool = True,
    batch_size: int | str = 1,
    seed: int = 42,
    dry_run: bool = False,
    verbose: bool = False,
    n_images: int = 1,
) -> PredictOptions:
    if images is None:
        images = _make_image_dir(tmp_path, n=n_images)
    return PredictOptions(
        images=images,
        prompts=prompts,
        output=tmp_path / "out",
        checkpoint=checkpoint,
        merge_adapter=merge_adapter,
        config=None,
        score_threshold=score_threshold,
        top_k=top_k,
        save_masks=save_masks,  # type: ignore[arg-type]
        visualize=visualize,
        device="cpu",
        dtype="float32",
        batch_size=batch_size,  # type: ignore[arg-type]
        seed=seed,
        dry_run=dry_run,
        verbose=verbose,
    )


def _patch_load(stub: torch.nn.Module) -> mock.MagicMock:
    """Return a patcher that replaces load_sam31 with a factory returning stub.

    load_sam31 is imported lazily inside run_predict, so we patch it at its
    source module (custom_sam_peft.models.sam3) rather than runner's namespace.
    """

    def _factory(cfg: Any, **kwargs: Any) -> torch.nn.Module:
        return stub

    return mock.patch("custom_sam_peft.models.sam3.load_sam31", side_effect=_factory)


# ---------------------------------------------------------------------------
# 1. test_run_predict_smoke_end_to_end_cpu
# ---------------------------------------------------------------------------


def test_run_predict_smoke_end_to_end_cpu(tmp_path: Path) -> None:
    """One 64x64 image + two prompts -> all three output files written; report populated."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, prompts="cat,dog", save_masks="rle")

    with _patch_load(stub):
        report = run_predict(opts)

    out_dir = tmp_path / "out"
    assert (out_dir / "predictions.json").exists(), "predictions.json not written"
    assert (out_dir / "image_id_map.json").exists(), "image_id_map.json not written"
    assert (out_dir / "run.json").exists(), "run.json not written"

    predictions = json.loads((out_dir / "predictions.json").read_text())
    assert isinstance(predictions, list)
    # 2 prompts x Q=4 queries = 8 entries (score_threshold=0.0, top_k=100)
    assert len(predictions) == 2 * Q

    assert isinstance(report, PredictReport)
    assert report.n_images == 1
    assert report.n_predictions == len(predictions)
    assert report.elapsed_sec >= 0.0


# ---------------------------------------------------------------------------
# 2. test_run_predict_base_model_only_no_peft_import
# ---------------------------------------------------------------------------


def test_run_predict_base_model_only_no_peft_import(tmp_path: Path) -> None:
    """With checkpoint=None, custom_sam_peft.peft_adapters must NOT be imported."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, checkpoint=None)

    # Remove peft_adapters from sys.modules so a fresh import would be detectable
    saved = {k: v for k, v in sys.modules.items() if "peft_adapters" in k}
    for k in saved:
        del sys.modules[k]

    sentinel = object()
    # Place a sentinel so any import attempt would reveal a different object
    sys.modules["custom_sam_peft.peft_adapters"] = sentinel  # type: ignore[assignment]

    try:
        with _patch_load(stub):
            run_predict(opts)

        # Verify the sentinel is still the exact same object (not replaced)
        assert sys.modules.get("custom_sam_peft.peft_adapters") is sentinel, (
            "peft_adapters was imported despite checkpoint=None"
        )
    finally:
        del sys.modules["custom_sam_peft.peft_adapters"]
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 3. test_run_predict_warmup_runs_one_forward
# ---------------------------------------------------------------------------


def test_run_predict_warmup_runs_one_forward(tmp_path: Path) -> None:
    """The stub module is called at least once for warmup before per-image loop."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, prompts="cat")

    with _patch_load(stub):
        run_predict(opts)

    # warmup = 1 call + per-image per-class calls
    # With 1 image and 1 prompt: total = 1 (warmup) + 1 (inference) = 2
    assert stub.forward_call_count >= 2, (
        f"Expected at least 2 forward calls (1 warmup + 1 inference); got {stub.forward_call_count}"
    )


# ---------------------------------------------------------------------------
# 4. test_run_predict_vram_hint_not_logged_on_cpu
# ---------------------------------------------------------------------------


def test_run_predict_vram_hint_not_logged_on_cpu(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """On device='cpu', the VRAM hint must NOT be logged."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path)

    with caplog.at_level(logging.INFO), _patch_load(stub):
        run_predict(opts)

    vram_msgs = [r for r in caplog.records if "vram" in r.getMessage().lower()]
    assert len(vram_msgs) == 0, (
        f"Unexpected VRAM hint on CPU: {[r.getMessage() for r in vram_msgs]}"
    )


# ---------------------------------------------------------------------------
# 5. test_run_predict_seed_recorded
# ---------------------------------------------------------------------------


def test_run_predict_seed_recorded(tmp_path: Path) -> None:
    """run.json must contain the seed value supplied in PredictOptions."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, seed=7)

    with _patch_load(stub):
        run_predict(opts)

    run_json = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run_json["seed"] == 7


# ---------------------------------------------------------------------------
# 6. test_run_predict_top_k_filtering_applied
# ---------------------------------------------------------------------------


def test_run_predict_top_k_filtering_applied(tmp_path: Path) -> None:
    """With top_k=2 and Q=4 queries, only top 2 per (image, class) kept."""
    stub = _StubSamModule(n_queries=Q)
    # two prompts → max 2 * top_k entries
    opts = _make_opts(tmp_path, prompts="cat,dog", score_threshold=0.0, top_k=2)

    with _patch_load(stub):
        report = run_predict(opts)

    preds = json.loads((tmp_path / "out" / "predictions.json").read_text())
    # 2 prompts x top_k=2 = at most 4 entries
    assert len(preds) <= 4, f"Expected at most 4 predictions with top_k=2; got {len(preds)}"
    assert report.n_predictions == len(preds)


# ---------------------------------------------------------------------------
# 7. test_run_predict_score_threshold_applied
# ---------------------------------------------------------------------------


def test_run_predict_score_threshold_applied(tmp_path: Path) -> None:
    """When all stub scores are below threshold, n_predictions==0, file written empty."""
    # LOW_SCORE logit → sigmoid ≈ 0, well below threshold=0.3
    stub = _StubSamModule(score_logit=LOW_SCORE, presence_logit=LOW_SCORE)
    opts = _make_opts(tmp_path, score_threshold=0.3, top_k=100)

    with _patch_load(stub):
        report = run_predict(opts)

    preds = json.loads((tmp_path / "out" / "predictions.json").read_text())
    assert preds == [], f"Expected empty predictions list; got {len(preds)} entries"
    assert report.n_predictions == 0


# ---------------------------------------------------------------------------
# 8. test_run_predict_save_masks_none
# ---------------------------------------------------------------------------


def test_run_predict_save_masks_none(tmp_path: Path) -> None:
    """With save_masks='none', no segmentation field and no masks/ dir."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, save_masks="none", score_threshold=0.0)

    with _patch_load(stub):
        run_predict(opts)

    preds = json.loads((tmp_path / "out" / "predictions.json").read_text())
    for entry in preds:
        assert "segmentation" not in entry, "save_masks='none' must drop segmentation"

    masks_dir = tmp_path / "out" / "masks"
    assert not masks_dir.exists(), "masks/ dir must not be created with save_masks='none'"


# ---------------------------------------------------------------------------
# 9. test_run_predict_save_masks_png
# ---------------------------------------------------------------------------


def test_run_predict_save_masks_png(tmp_path: Path) -> None:
    """With save_masks='png', mask PNGs are written and entries carry mask_png."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, save_masks="png", score_threshold=0.0)

    with _patch_load(stub):
        run_predict(opts)

    preds = json.loads((tmp_path / "out" / "predictions.json").read_text())
    assert len(preds) > 0, "Need at least one prediction for mask PNG check"

    masks_dir = tmp_path / "out" / "masks"
    assert masks_dir.is_dir(), "masks/ dir must exist with save_masks='png'"

    for entry in preds:
        assert "mask_png" in entry, "Entry missing mask_png with save_masks='png'"
        png_path = tmp_path / "out" / entry["mask_png"]
        assert png_path.exists(), f"mask PNG file not found: {png_path}"
        assert "segmentation" not in entry, "segmentation should be removed with save_masks='png'"


# ---------------------------------------------------------------------------
# 10. test_run_predict_visualize
# ---------------------------------------------------------------------------


def test_run_predict_visualize(tmp_path: Path) -> None:
    """With visualize=True, per-image overlay PNGs are written."""
    stub = _StubSamModule()
    opts = _make_opts(tmp_path, visualize=True, score_threshold=0.0)

    with _patch_load(stub):
        run_predict(opts)

    vis_dir = tmp_path / "out" / "visualizations"
    assert vis_dir.is_dir(), "visualizations/ dir must exist when visualize=True"
    vis_pngs = list(vis_dir.glob("*.png"))
    assert len(vis_pngs) >= 1, "At least one visualization PNG must be written"


# ---------------------------------------------------------------------------
# 11. test_run_predict_unreadable_image_warns_and_skips
# ---------------------------------------------------------------------------


def test_run_predict_unreadable_image_warns_and_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt image produces a WARN log and is skipped; remaining images succeed."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Good image
    PILImage.new("RGB", (64, 64)).save(img_dir / "good.png")
    # Corrupt image (not a real image file)
    (img_dir / "corrupt.png").write_bytes(b"NOTANIMAGE")

    stub = _StubSamModule()
    opts = _make_opts(tmp_path, images=img_dir, score_threshold=0.0)

    with caplog.at_level(logging.WARNING), _patch_load(stub):
        report = run_predict(opts)

    # Should have warned about the corrupt file
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) >= 1, "Expected at least one warning for corrupt image"

    # Should have processed the good image
    assert report.n_images >= 1, "At least one image (good.png) should have been processed"


# ---------------------------------------------------------------------------
# 12. test_run_predict_every_image_fails_exits_1
# ---------------------------------------------------------------------------


def test_run_predict_every_image_fails_exits_1(tmp_path: Path) -> None:
    """When every image is corrupt, RuntimeError is raised (CLI converts to exit 1)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Corrupt images only
    for name in ["a.png", "b.png"]:
        (img_dir / name).write_bytes(b"NOTANIMAGE")

    stub = _StubSamModule()
    opts = _make_opts(tmp_path, images=img_dir)

    with _patch_load(stub), pytest.raises(RuntimeError, match="all images failed"):
        run_predict(opts)


# ---------------------------------------------------------------------------
# 13. test_predict_options_batch_size_default_auto
# ---------------------------------------------------------------------------


def test_predict_options_batch_size_default_auto() -> None:
    """PredictOptions.batch_size dataclass default is 'auto'."""
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(PredictOptions)}
    assert fields["batch_size"].default == "auto", (
        f"Expected PredictOptions.batch_size default == 'auto'; "
        f"got {fields['batch_size'].default!r}"
    )


# ---------------------------------------------------------------------------
# 14. test_run_predict_resolves_auto
# ---------------------------------------------------------------------------


def test_run_predict_resolves_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run_predict resolves 'auto' once at entry via decide_eval_batch_size."""
    call_args: list[tuple] = []

    def _fake_decide(classes_per_forward: int = 16):
        call_args.append((classes_per_forward,))
        return 1, 0, "analytic"  # (batch_size, predicted_bytes, provenance)

    monkeypatch.setattr(
        "custom_sam_peft.presets.decide_eval_batch_size",
        _fake_decide,
    )

    stub = _StubSamModule()
    opts = _make_opts(tmp_path, batch_size="auto", prompts="cat")

    with _patch_load(stub):
        run_predict(opts)

    assert len(call_args) == 1, (
        f"Expected decide_eval_batch_size to be called exactly once; called {len(call_args)} times"
    )


# ---------------------------------------------------------------------------
# 15. test_run_predict_flat_loop_iterates_image_chunks_x_groups
# ---------------------------------------------------------------------------


class _MultiplexStubSamModule(torch.nn.Module):
    """Stub whose forward returns tensors shaped (B * K_g, Q, ...) for multiplex.

    When each TextPrompts carries K_g classes, the output batch dim is B * K_g.
    """

    def __init__(
        self,
        n_queries: int = Q,
        score_logit: float = HIGH_SCORE,
        presence_logit: float = HIGH_SCORE,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.score_logit = score_logit
        self.presence_logit = presence_logit
        self.call_count = 0
        self.prompts_per_call: list[Any] = []

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[Any],
        support: Any = None,
    ) -> dict[str, torch.Tensor]:
        self.call_count += 1
        self.prompts_per_call.append(prompts)
        b = images.shape[0]
        # For TextPrompts, K_g is the number of classes in the shared prompt.
        from custom_sam_peft.data.base import TextPrompts as _TP

        k_g = len(prompts[0].classes) if prompts and isinstance(prompts[0], _TP) else 1
        total = b * k_g
        return {
            "pred_logits": torch.full((total, self.n_queries, 1), self.score_logit),
            "pred_boxes": torch.full((total, self.n_queries, 4), 0.5),
            "pred_masks": torch.zeros(total, self.n_queries, H_LOW, W_LOW),
            "presence_logit_dec": torch.full((total, 1), self.presence_logit),
        }


def test_run_predict_flat_loop_iterates_image_chunks_x_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Flat (image_chunk, group) iteration; warmup is still single-image / single-class."""
    # 4 images, 3 prompts, batch_size=2 -> ceil(4/2)=2 chunks x 1 group = 2 inference calls.
    # Total = 1 (warmup) + 2 (inference) = 3.
    stub = _MultiplexStubSamModule()
    opts = _make_opts(
        tmp_path,
        batch_size=2,
        prompts="cat,dog,bird",
        score_threshold=0.0,
        n_images=4,
    )

    with _patch_load(stub):
        report = run_predict(opts)

    # warmup=1 + 2 inference chunks = 3 total calls
    assert stub.call_count == 3, (
        f"Expected 3 forward calls (1 warmup + 2 batch); got {stub.call_count}"
    )

    # Each inference call (calls 2 and 3) should have prompts with 3 classes each
    inference_calls = stub.prompts_per_call[1:]  # skip warmup
    for call_prompts in inference_calls:
        assert len(call_prompts[0].classes) == 3, (
            f"Expected each inference call to pass 3 classes; got {len(call_prompts[0].classes)}"
        )

    assert report.n_images == 4
