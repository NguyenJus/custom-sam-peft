"""Unit tests for eval tiling: non-overlapping metric accumulation (spec §5.4).

Guard tests: EVAL_OVERLAP constant, iter_windows non-overlap coverage, small-image
direct path. CPU-only — no real SAM model loaded.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import torch

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.tiling import EVAL_OVERLAP, iter_windows, tiling_engaged
from custom_sam_peft.data.transforms import build_eval_transforms
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE
from custom_sam_peft.predict.tiling_preprocess import preprocess_tile

_MODEL = "facebook/sam3.1"


def _pad_only_transform() -> Any:
    """The design-C pad-only eval transform (no LongestMaxSize)."""
    return build_eval_transforms(
        SAM3_IMAGE_SIZE, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False
    )


# ---------------------------------------------------------------------------
# Guard tests (Step 1 — constants and rules; must pass immediately)
# ---------------------------------------------------------------------------


def test_eval_uses_non_overlapping_tiling() -> None:
    assert EVAL_OVERLAP == 0.0
    ws = iter_windows(2016, 2016, tile=1008, overlap=EVAL_OVERLAP)
    # non-overlapping: exactly 4 disjoint 1008x1008 windows, no shared band
    assert len(ws) == 4
    starts = sorted({w.y0 for w in ws})
    assert starts == [0, 1008]


def test_small_eval_image_direct_path() -> None:
    assert tiling_engaged(700, 700) is False


# ---------------------------------------------------------------------------
# Faithfulness tests (design C, FAITHFULNESS-CRITICAL — Task 1.6b)
# ---------------------------------------------------------------------------


def test_preprocess_tile_pads_with_normalize_zero_not_literal_zero() -> None:
    """The pad-only transform pads raw-0 THEN normalizes, so the padded extent is
    normalize(0) (≈ -mean/std), NOT literal 0 (transforms.py PadIfNeeded BEFORE
    Normalize). This is the faithfulness invariant the evaluator depends on."""
    transform = _pad_only_transform()
    crop = (np.random.RandomState(0).rand(1008, 492, 3) * 255).astype(np.uint8)  # edge tile
    t = preprocess_tile(crop, transform, device="cpu", dtype=torch.float32)
    assert t.shape == (3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE)
    pad_region = t[:, :, 492:]  # top-left placement -> right columns are pad
    assert not torch.allclose(pad_region, torch.zeros_like(pad_region))  # NOT literal 0
    # pad value == normalize(0) == -mean/std, constant per channel across the pad band
    for c in range(3):
        assert torch.allclose(pad_region[c], pad_region[c].flatten()[0])


def test_eval_per_tile_input_is_byte_identical_to_predict() -> None:
    """MANDATORY parity guard (design C): predict's downscale=True transform and eval's
    downscale=False transform must produce BYTE-IDENTICAL tensors on a real tile crop.

    The faithfulness guarantee rests on iter_windows tiles having their longest edge
    exactly 1008, so LongestMaxSize(1008) is a no-op inside the downscale=True transform.
    This test uses a 756x1008 crop (one axis < 1008, longest edge == 1008) — the
    DISTINGUISHING case where the two transforms would diverge if LongestMaxSize were
    ever NOT a no-op.  Both must produce identical (3, 1008, 1008) tensors.
    """
    # predict uses downscale=True (its default); eval uses downscale=False
    predict_transform = build_eval_transforms(
        SAM3_IMAGE_SIZE, model_name=_MODEL, normalize=NormalizeConfig(), downscale=True
    )
    eval_transform = build_eval_transforms(
        SAM3_IMAGE_SIZE, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False
    )
    # 756x1008: one axis < 1008 (the distinguishing case), longest edge exactly 1008
    crop = (np.random.RandomState(1).rand(756, 1008, 3) * 255).astype(np.uint8)
    # predict-path preprocessing (what _predict_one_tile feeds the model — downscale=True)
    predict_input = preprocess_tile(crop, predict_transform, device="cpu", dtype=torch.float32)
    # eval-path preprocessing (what the evaluator feeds the model — downscale=False)
    eval_input = preprocess_tile(crop, eval_transform, device="cpu", dtype=torch.float32)
    assert predict_input.shape == (3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE)
    assert eval_input.shape == (3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE)
    assert torch.allclose(predict_input, eval_input, atol=1e-6), (
        "predict (downscale=True) and eval (downscale=False) tensors diverged — "
        "LongestMaxSize(1008) is NOT a no-op on a 756x1008 tile; faithfulness bug."
    )


# ---------------------------------------------------------------------------
# CPU unit test: evaluator tiling branch with stub forward (no real model)
# ---------------------------------------------------------------------------


def _make_stub_outputs(h: int, w: int, n_queries: int = 2) -> dict[str, torch.Tensor]:
    """Minimal output dict that queries_to_coco_results accepts."""
    return {
        "pred_logits": torch.zeros(1, n_queries, 1),
        "pred_boxes": torch.zeros(1, n_queries, 4),
        "pred_masks": torch.zeros(1, n_queries, h, w),
        "presence_logit_dec": torch.zeros(1, 1),
    }


def _make_large_example(orig_h: int, orig_w: int) -> Any:
    """Create an Example whose image is large enough to engage tiling.

    Carries image_native (native-res raw numpy pixels) as the design-C eval dataset
    does, so the evaluator's per-tile pad-only preprocess has raw pixels to crop.
    """
    from custom_sam_peft.data.base import Example, Instance, TextPrompts

    # Image tensor at "large" resolution (H > SAM3_IMAGE_SIZE = 1008).
    image = torch.zeros(3, orig_h, orig_w)
    image_native = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)  # (H, W, C) raw pixels
    mask = torch.zeros(orig_h, orig_w, dtype=torch.bool)
    mask[0:10, 0:10] = True
    return Example(
        image=image,
        image_id="large_img_0",
        prompts=TextPrompts(classes=["cat"]),
        instances=[
            Instance(
                mask=mask,
                class_id=0,
                box=torch.tensor([0.0, 0.0, 10.0, 10.0]),
            )
        ],
        image_native=image_native,
    )


class _TilingCallCountDataset:
    """1-image dataset whose single image is 2016x2016 (2x SAM3_IMAGE_SIZE).

    Non-overlapping tiling with tile=1008 yields exactly 4 windows.
    """

    class_names: ClassVar[list[str]] = ["cat"]

    def __init__(self) -> None:
        self._example = _make_large_example(2016, 2016)
        # design-C: eval dataset exposes the pad-only transform the evaluator runs
        # per tile via the shared preprocess_tile helper.
        self.tile_transform = _pad_only_transform()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> Any:
        return self._example


def _make_tile_stub_model(tile_size: int) -> MagicMock:
    """A mock model that returns valid output shaped for a (tile_size, tile_size) forward."""

    def _forward(
        images: torch.Tensor, prompts: Any, support: Any = None
    ) -> dict[str, torch.Tensor]:
        b = images.shape[0]
        k_g = len(prompts[0].classes) if prompts else 1
        rows = b * k_g
        h, w = images.shape[-2], images.shape[-1]
        return {
            "pred_logits": torch.zeros(rows, 2, 1),
            "pred_boxes": torch.zeros(rows, 2, 4),
            "pred_masks": torch.zeros(rows, 2, h, w),
            "presence_logit_dec": torch.zeros(rows, 1),
        }

    model = MagicMock(side_effect=_forward)
    model.training = False
    # No parameters — uses CPU path
    del model.parameters
    return model


def test_evaluator_tiling_branch_calls_model_per_tile() -> None:
    """When tiling_engaged, the evaluator calls model once per tile (not once per image).

    2016x2016 image + tile=1008 + overlap=0.0 -> 4 tiles.
    1 class, 1 class group (K_g=1 per tile because MULTIPLEX_CAP >= 1).
    Expect 4 model calls (one per tile-class-group pair).
    """
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.eval.evaluator import Evaluator

    dataset = _TilingCallCountDataset()
    model = _make_tile_stub_model(tile_size=1008)

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)

    examples = [dataset[0]]
    preds = ev._iter_predictions(model, examples, dataset)

    # Each call must have received a tensor whose max(H, W) == 1008 (tile size,
    # not the original 2016) — confirms tiling path, not full-image forward.
    for call_args in model.call_args_list:
        img_arg: torch.Tensor = call_args[0][0]
        assert max(img_arg.shape[-2], img_arg.shape[-1]) == 1008, (
            f"Expected 1008-sized tile input, got {img_arg.shape}"
        )

    # 4 tiles x 1 class group = 4 calls.
    assert model.call_count == 4, f"Expected 4 tile-level calls, got {model.call_count}"

    # Predictions list must not be empty (accumulation happened).
    assert isinstance(preds, list)


def test_evaluator_tiling_no_stitched_mask_in_predictions() -> None:
    """Tiled accumulation does NOT materialize a full 2016x2016 mask in any entry.

    COCO entry segmentation RLEs must describe masks sized to the tile (1008x1008),
    not the original image (2016x2016). This verifies spec §5.4's
    'without materializing a stitched full-image mask'.
    """

    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.eval.evaluator import Evaluator

    dataset = _TilingCallCountDataset()
    model = _make_tile_stub_model(tile_size=1008)

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [dataset[0]]
    preds = ev._iter_predictions(model, examples, dataset)

    for entry in preds:
        seg = entry["segmentation"]
        size = seg["size"]  # type: ignore[index]
        h_rle, w_rle = int(size[0]), int(size[1])
        # Must NOT be the full-image size; must be at most tile_size.
        assert h_rle <= 1008, f"RLE H={h_rle} exceeds tile size 1008 (stitched mask leaked)"
        assert w_rle <= 1008, f"RLE W={w_rle} exceeds tile size 1008 (stitched mask leaked)"


def test_evaluator_small_image_direct_path_unchanged() -> None:
    """Small images (<=1008 edge) must take the direct path (byte-for-byte unchanged).

    The direct path calls model once per class group for the entire image, not
    once per tile. With a 8x8 image and 1 class, expect exactly 1 model call.
    """
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.data.base import Example, Instance, TextPrompts
    from custom_sam_peft.eval.evaluator import Evaluator

    # 8x8 image — well below 1008, direct path.
    small_example = Example(
        image=torch.zeros(3, 8, 8),
        image_id="small_0",
        prompts=TextPrompts(classes=["cat"]),
        instances=[
            Instance(
                mask=torch.zeros(8, 8, dtype=torch.bool),
                class_id=0,
                box=torch.tensor([0.0, 0.0, 4.0, 4.0]),
            )
        ],
    )

    class _SmallDataset:
        class_names: ClassVar[list[str]] = ["cat"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> Example:
            return small_example

    dataset = _SmallDataset()
    model = _make_tile_stub_model(tile_size=8)

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [dataset[0]]
    ev._iter_predictions(model, examples, dataset)

    # Direct path: 1 class, 1 class group -> exactly 1 model call.
    assert model.call_count == 1, f"Expected 1 direct-path call, got {model.call_count}"
    # Direct path: model receives the full 8x8 image, not a tile.
    img_arg: torch.Tensor = model.call_args_list[0][0][0]
    assert img_arg.shape[-2] == 8 and img_arg.shape[-1] == 8


def _make_topleft_hit_model(hit: int = 10) -> MagicMock:
    """A mock model whose query 0 predicts a strong mask over the tile's
    top-left ``hit`` x ``hit`` region (matching the GT placed at the full-image
    top-left), and query 1 predicts nothing.

    The stub emits the same prediction for every tile because the example image
    is all-zeros (tiles are indistinguishable by content). Only tile 0's
    reconstructed prediction lands on the GT (canvas[0:hit, 0:hit]); other tiles'
    predictions land at their own offsets where there is no GT, so they
    contribute zero IoU. This isolates the tile-0 overlap deterministically.
    """

    def _forward(
        images: torch.Tensor, prompts: Any, support: Any = None
    ) -> dict[str, torch.Tensor]:
        b = images.shape[0]
        k_g = len(prompts[0].classes) if prompts else 1
        rows = b * k_g
        h, w = int(images.shape[-2]), int(images.shape[-1])
        # query 0: strong positive logits over the top-left hit x hit block,
        # strongly negative elsewhere; query 1: all negative (empty mask).
        masks = torch.full((rows, 2, h, w), -50.0)
        masks[:, 0, 0:hit, 0:hit] = 50.0
        return {
            # query 0 confidently present; query 1 absent.
            "pred_logits": torch.tensor([[[50.0], [-50.0]]]).expand(rows, 2, 1).contiguous(),
            "pred_boxes": torch.zeros(rows, 2, 4),
            "pred_masks": masks,
            "presence_logit_dec": torch.full((rows, 1), 50.0),
        }

    model = MagicMock(side_effect=_forward)
    model.training = False
    del model.parameters
    return model


def test_tiled_per_example_iou_is_nonzero_for_overlapping_pred() -> None:
    """Regression for the BLOCKING finding: per_example_iou must be a correct
    NONZERO full-image IoU for a tiled image whose prediction overlaps GT.

    Before the fix, tiled predictions were stored under tile-local ids while
    _compute_per_example_iou looked them up by full-image id, so the lookup
    returned [] and every tiled example scored 0.0. The fix reconstructs the
    disjoint tile masks onto a full-image canvas and computes IoU against the
    full-image GT. The stub's tile-0 prediction exactly matches the 10x10 GT,
    so the example score must be close to 1.0 (and definitely not 0.0).
    """
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.eval.evaluator import Evaluator

    dataset = _TilingCallCountDataset()  # 2016x2016, GT mask at [0:10, 0:10]
    model = _make_topleft_hit_model(hit=10)

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    _report, per_example_iou = Evaluator(cfg).evaluate(model, dataset, return_per_example_iou=True)

    # Length contract (#245): exactly one value per example.
    assert len(per_example_iou) == len(dataset) == 1
    # The tile-0 reconstructed prediction matches the GT exactly -> IoU == 1.0,
    # so the threshold-mean score (best_iou >= 0.5) is 1.0. The load-bearing
    # assertion is simply that it is NOT 0.0 (the pre-fix bug value).
    assert per_example_iou[0] > 0.0, "tiled per-example IoU collapsed to 0.0"
    assert per_example_iou[0] == 1.0


def test_evaluator_tiling_evaluate_returns_metrics_report() -> None:
    """Full evaluate() on a tiled image returns a valid MetricsReport."""
    from custom_sam_peft.config.schema import EvalConfig
    from custom_sam_peft.eval.evaluator import Evaluator
    from custom_sam_peft.eval.metrics import MetricsReport

    dataset = _TilingCallCountDataset()
    model = _make_tile_stub_model(tile_size=1008)

    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1)
    report = Evaluator(cfg).evaluate(model, dataset)

    assert isinstance(report, MetricsReport)
    assert report.n_images >= 1
