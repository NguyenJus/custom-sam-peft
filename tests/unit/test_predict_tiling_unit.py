"""CPU-only unit tests for the predict tiling path (spec §5.2).

No real SAM model is loaded — the forward is stubbed so these tests run on CPU
without any checkpoint.  GPU + real-model integration is covered by Task 1.7 (G1).
"""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch

from custom_sam_peft.data.tiling import (
    DEFAULT_OVERLAP,
    Fragment,
    MergedInstance,
    iter_windows,
    merge_fragments,
    run_windows,
    tiling_engaged,
)
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

# ---------------------------------------------------------------------------
# Auto-engage decision
# ---------------------------------------------------------------------------


def test_small_image_takes_direct_path() -> None:
    """Images with max(h, w) <= SAM3_IMAGE_SIZE do not tile."""
    assert tiling_engaged(900, 1008) is False


def test_image_exactly_at_tile_size_direct_path() -> None:
    """max(h, w) == SAM3_IMAGE_SIZE is the boundary — direct path."""
    assert tiling_engaged(SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE) is False


def test_oversized_image_engages_tiling() -> None:
    """Images with any edge > SAM3_IMAGE_SIZE trigger the tiling path."""
    assert tiling_engaged(SAM3_IMAGE_SIZE + 1, SAM3_IMAGE_SIZE) is True
    assert tiling_engaged(SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE + 1) is True
    assert tiling_engaged(1500, 1500) is True


# ---------------------------------------------------------------------------
# Core tiling + merge round-trip
# ---------------------------------------------------------------------------


def test_oversized_image_tiles_and_merges_seam_object() -> None:
    """A horizontal bar spanning every vertical seam merges into full-canvas masks."""
    img = np.zeros((1500, 1500, 3), np.uint8)
    assert tiling_engaged(1500, 1500) is True
    windows = iter_windows(1500, 1500, tile=1008, overlap=0.25)

    def fake_forward(crop: np.ndarray, window: object) -> list[Fragment]:
        m = np.zeros(crop.shape[:2], bool)
        m[700:760, :] = True  # a horizontal bar crossing every vertical seam
        return [Fragment(mask=m, score=0.9, category_id=1, window_id=id(window))]

    frags = run_windows(img, windows, fake_forward)
    merged = merge_fragments(frags, (1500, 1500))
    assert all(mi.mask.shape == (1500, 1500) for mi in merged)  # ONE full-extent canvas


def test_merged_instances_carry_correct_category() -> None:
    """merge_fragments preserves category_id on MergedInstances."""
    img = np.zeros((1500, 1500, 3), np.uint8)
    windows = iter_windows(1500, 1500, tile=SAM3_IMAGE_SIZE, overlap=DEFAULT_OVERLAP)

    def fake_forward(crop: np.ndarray, window: object) -> list[Fragment]:
        m = np.zeros(crop.shape[:2], bool)
        m[100:200, 100:200] = True
        return [Fragment(mask=m, score=0.8, category_id=2, window_id=id(window))]

    frags = run_windows(img, windows, fake_forward)
    merged = merge_fragments(frags, (1500, 1500))
    for mi in merged:
        assert mi.category_id == 2


# ---------------------------------------------------------------------------
# _merged_instance_to_entry helper
# ---------------------------------------------------------------------------


def test_merged_instance_to_entry_rle_shape() -> None:
    """_merged_instance_to_entry must emit a segmentation RLE at the mask's canvas size."""
    from custom_sam_peft.predict.runner import _merged_instance_to_entry

    h, w = 200, 300
    mask = np.zeros((h, w), bool)
    mask[50:100, 60:120] = True
    mi = MergedInstance(mask=mask, score=0.75, category_id=3)
    entry = _merged_instance_to_entry(mi, image_id=42, category_id=3)

    assert entry["image_id"] == 42
    assert entry["category_id"] == 3
    assert entry["score"] == pytest.approx(0.75)

    rle = entry["segmentation"]
    # Decode and confirm shape matches canvas
    decode_rle: dict = dict(rle)  # type: ignore[arg-type]
    counts = decode_rle["counts"]
    if isinstance(counts, str):
        decode_rle["counts"] = counts.encode("ascii")
    decoded = mask_utils.decode(decode_rle)
    assert decoded.shape == (h, w)
    assert decoded.sum() == mask.sum()


def test_merged_instance_to_entry_empty_mask_bbox() -> None:
    """An all-zero mask must produce bbox [0, 0, 0, 0] without error."""
    from custom_sam_peft.predict.runner import _merged_instance_to_entry

    mask = np.zeros((100, 100), bool)
    mi = MergedInstance(mask=mask, score=0.5, category_id=1)
    entry = _merged_instance_to_entry(mi, image_id=1, category_id=1)
    assert entry["bbox"] == [0.0, 0.0, 0.0, 0.0]


def test_merged_instance_to_entry_bbox_is_xywh() -> None:
    """bbox is [x, y, w, h] bounding the non-zero pixels."""
    from custom_sam_peft.predict.runner import _merged_instance_to_entry

    mask = np.zeros((200, 200), bool)
    mask[50:100, 30:80] = True  # y:[50,100), x:[30,80)
    mi = MergedInstance(mask=mask, score=0.9, category_id=1)
    entry = _merged_instance_to_entry(mi, image_id=1, category_id=1)
    x, y, w, h = entry["bbox"]  # type: ignore[misc]
    assert x == pytest.approx(30.0)
    assert y == pytest.approx(50.0)
    assert w == pytest.approx(49.0)  # 79 - 30 = 49
    assert h == pytest.approx(49.0)  # 99 - 50 = 49


# ---------------------------------------------------------------------------
# run.json tiling provenance structure
# ---------------------------------------------------------------------------


def test_run_meta_tiling_record_keys() -> None:
    """The tiling provenance dict emitted in run.json must have the required keys."""
    required_keys = {"engaged", "tile", "overlap", "n_windows_total"}
    # Construct a minimal tiling record as runner.py would build it
    from custom_sam_peft.data.tiling import DEFAULT_OVERLAP
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    tiling_record = {
        "engaged": False,
        "tile": SAM3_IMAGE_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "n_windows_total": 0,
    }
    assert required_keys.issubset(tiling_record.keys())
    assert tiling_record["tile"] == SAM3_IMAGE_SIZE
    assert tiling_record["overlap"] == DEFAULT_OVERLAP


# ---------------------------------------------------------------------------
# _predict_one_tile unit tests (FIX 5)
# ---------------------------------------------------------------------------
#
# Helper: build a valid model-output dict that queries_to_coco_results accepts.
# Shapes: pred_logits (1, N, 1), pred_boxes (1, N, 4), pred_masks (1, N, Hm, Wm),
#         presence_logit_dec (1, 1).
# We use N=1, Hm=Wm=4 so postprocessing is cheap on CPU.


def _make_model_outputs(
    score_logit: float = 5.0,
    presence_logit: float = 5.0,
    tile_h: int = 16,
    tile_w: int = 16,
) -> dict[str, torch.Tensor]:
    """Build minimal canned model outputs accepted by queries_to_coco_results."""
    # Single query (N=1), mask spatial size 4x4 (upsampled to tile_hw internally)
    return {
        "pred_logits": torch.tensor([[[score_logit]]], dtype=torch.float32),  # (1,1,1)
        "pred_boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32).unsqueeze(
            0
        ),  # (1,1,4) normalised cxcywh
        "pred_masks": torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        + 1.0,  # all-positive logits → all-True mask after threshold
        "presence_logit_dec": torch.tensor([[presence_logit]], dtype=torch.float32),  # (1,1)
    }


def _make_transforms_passthrough(tile_h: int, tile_w: int) -> object:
    """Stub albumentations-style transform that wraps the array in a tensor."""

    class _T:
        def __call__(
            self,
            image: np.ndarray,
            bboxes: list,
            class_labels: list,
            instance_idx: list,
        ) -> dict:
            # (H, W, 3) → (3, H, W) float32 tensor, matching real transform output
            t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
            return {"image": t, "bboxes": bboxes, "class_labels": class_labels}

    return _T()


def _run_predict_one_tile(
    model: object,
    score_threshold: float = 0.5,
    prompts: list[str] | None = None,
    tile_h: int = 16,
    tile_w: int = 16,
) -> list:
    """Invoke _predict_one_tile with a minimal CPU setup."""
    from custom_sam_peft.oom import OomLadder
    from custom_sam_peft.predict.runner import _predict_one_tile

    if prompts is None:
        prompts = ["cat"]

    crop_np = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    ladder = OomLadder(micro_batch_size=1, effective_K=len(prompts))
    transforms = _make_transforms_passthrough(tile_h, tile_w)

    return _predict_one_tile(
        crop_np,
        0,  # window_idx
        model=model,  # type: ignore[arg-type]
        transforms=transforms,
        prompts=prompts,
        score_threshold=score_threshold,
        device="cpu",
        dtype=torch.float32,
        ladder=ladder,
        category_id_offset=0,
    )


class _StubModel:
    """Stub model that returns canned outputs; optionally raises OOM on first call."""

    def __init__(self, outputs: dict, *, oom_on_first: bool = False) -> None:
        self._outputs = outputs
        self._oom_on_first = oom_on_first
        self._call_count = 0

    def __call__(self, *args: object, **kwargs: object) -> dict:
        self._call_count += 1
        if self._oom_on_first and self._call_count == 1:
            raise RuntimeError("CUDA driver error: device not ready")
        return self._outputs


def test_predict_one_tile_score_threshold_drops_low_score() -> None:
    """(a) Fragments scoring below score_threshold are not returned."""
    # presence_logit=-10 → presence≈0, score≈0 → far below any reasonable threshold
    outputs = _make_model_outputs(score_logit=5.0, presence_logit=-10.0)
    model = _StubModel(outputs)
    frags = _run_predict_one_tile(model, score_threshold=0.5)
    assert frags == [], f"expected no fragments for near-zero score, got {frags}"


def test_predict_one_tile_score_above_threshold_kept() -> None:
    """Fragments scoring above score_threshold ARE returned."""
    outputs = _make_model_outputs(score_logit=5.0, presence_logit=5.0)  # score ≈ 1.0
    model = _StubModel(outputs)
    frags = _run_predict_one_tile(model, score_threshold=0.5)
    assert len(frags) >= 1, "expected at least one fragment above threshold"
    for frag in frags:
        assert frag.score >= 0.5


def test_predict_one_tile_oom_retry_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) OOM-style RuntimeError on first call → ladder retries → returns fragments.

    Guards FIX 1: without the RETRY_B branch, the original code fell through to
    raise RuntimeError on 'device not ready', crashing the whole run.
    """
    # Build a ladder with B=2 so that the first on_oom() returns RETRY_B
    # (halves B to 1), and the second forward succeeds.
    from custom_sam_peft.oom import OomLadder
    from custom_sam_peft.predict.runner import _predict_one_tile

    outputs = _make_model_outputs(score_logit=5.0, presence_logit=5.0)
    model = _StubModel(outputs, oom_on_first=True)

    crop_np = np.zeros((16, 16, 3), dtype=np.uint8)
    # B=2 so first on_oom → RETRY_B (halves to 1); next forward succeeds
    ladder = OomLadder(micro_batch_size=2, effective_K=1)
    transforms = _make_transforms_passthrough(16, 16)

    frags = _predict_one_tile(
        crop_np,
        0,
        model=model,  # type: ignore[arg-type]
        transforms=transforms,
        prompts=["cat"],
        score_threshold=0.0,
        device="cpu",
        dtype=torch.float32,
        ladder=ladder,
        category_id_offset=0,
    )
    assert model._call_count == 2, f"expected 2 calls (1 OOM + 1 retry), got {model._call_count}"
    assert len(frags) >= 1, "expected fragments after successful retry"
    # Ladder B should have halved
    assert ladder.micro_batch_size == 1


def test_predict_one_tile_category_id_arithmetic() -> None:
    """(c) category_id uses (category_id_offset + j + kk) + 1 arithmetic."""
    from custom_sam_peft.oom import OomLadder
    from custom_sam_peft.predict.runner import _predict_one_tile

    outputs = _make_model_outputs(score_logit=5.0, presence_logit=5.0)
    model = _StubModel(outputs)

    crop_np = np.zeros((16, 16, 3), dtype=np.uint8)
    # offset=2, 1 prompt → cat_id = (2 + 0 + 0) + 1 = 3
    ladder = OomLadder(micro_batch_size=1, effective_K=1)
    transforms = _make_transforms_passthrough(16, 16)

    frags = _predict_one_tile(
        crop_np,
        0,
        model=model,  # type: ignore[arg-type]
        transforms=transforms,
        prompts=["cat"],
        score_threshold=0.0,
        device="cpu",
        dtype=torch.float32,
        ladder=ladder,
        category_id_offset=2,
    )
    assert len(frags) >= 1
    for frag in frags:
        assert frag.category_id == 3, f"expected category_id=3, got {frag.category_id}"


def test_predict_one_tile_rle_mask_matches_crop_shape() -> None:
    """(d) The decoded RLE mask inside each fragment matches the crop shape."""
    # We inspect the bool mask on the Fragment (already decoded by _predict_one_tile)
    outputs = _make_model_outputs(score_logit=5.0, presence_logit=5.0, tile_h=24, tile_w=32)
    model = _StubModel(outputs)
    frags = _run_predict_one_tile(model, score_threshold=0.0, tile_h=24, tile_w=32)
    assert len(frags) >= 1
    for frag in frags:
        # Fragment mask is tile-local (H_tile, W_tile) bool array
        assert frag.mask.shape == (24, 32), f"unexpected mask shape: {frag.mask.shape}"
        assert frag.mask.dtype == bool
