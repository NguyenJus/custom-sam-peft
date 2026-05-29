"""Unit tests for eval/visualize.py pure primitives (CPU-only, no model)."""

from __future__ import annotations

import math

import torch

from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.visualize import pick_samples


class _FakeDataset:
    """Index-aligned dataset whose examples carry the requested #GT instances."""

    def __init__(self, gt_counts: list[int]) -> None:
        self._examples = []
        for i, n in enumerate(gt_counts):
            insts = [
                Instance(
                    mask=torch.zeros(4, 4, dtype=torch.bool),
                    class_id=0,
                    box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
                )
                for _ in range(n)
            ]
            self._examples.append(
                Example(
                    image=torch.zeros(3, 4, 4),
                    image_id=f"img_{i}",
                    prompts=TextPrompts(classes=["a"]),
                    instances=insts,
                )
            )
        self.class_names = ["a"]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]


def _bands(n: int) -> tuple[int, int, int]:
    good = round(0.5 * n)
    worst = min(2, max(1, round(0.2 * n)))
    median = n - good - worst
    return good, median, worst


def test_band_sizes_n10() -> None:
    ds = _FakeDataset([1] * 30)
    iou = [i / 30 for i in range(30)]  # all distinct, all GT-bearing
    picked = pick_samples(iou, ds, 10)
    assert len(picked) == 10
    assert _bands(10) == (5, 3, 2)


def test_band_sizes_various_n() -> None:
    for n, (g, m, w) in [(1, _bands(1)), (2, _bands(2)), (5, _bands(5)), (20, _bands(20))]:
        assert g + m + w == n
        assert w <= 2  # worst cap


def test_worst_cap_large_n() -> None:
    ds = _FakeDataset([1] * 50)
    iou = [i / 50 for i in range(50)]
    picked = pick_samples(iou, ds, 20)
    assert len(picked) == 20
    _g, _m, w = _bands(20)
    assert w == 2  # capped despite round(0.2*20)=4


def test_gt_filter_excludes_no_gt_images() -> None:
    # idx 0 has the highest IoU but NO GT → must never be selected.
    ds = _FakeDataset([0, 1, 1, 1, 1])
    iou = [1.0, 0.9, 0.8, 0.7, 0.6]
    picked = pick_samples(iou, ds, 4)
    assert 0 not in picked
    assert set(picked) <= {1, 2, 3, 4}


def test_small_pool_returns_all_candidates() -> None:
    ds = _FakeDataset([1, 1, 1])  # 3 GT-bearing candidates
    iou = [0.3, 0.2, 0.1]
    picked = pick_samples(iou, ds, 10)
    assert sorted(picked) == [0, 1, 2]
    assert len(picked) <= 10


def test_indices_unique_across_bands() -> None:
    ds = _FakeDataset([1] * 12)
    iou = [i / 12 for i in range(12)]
    picked = pick_samples(iou, ds, 10)
    assert len(picked) == len(set(picked))  # no index in two bands


def test_nan_sorts_to_bottom_worst_only() -> None:
    # idx 2 is NaN → ranked -inf → only ever a "worst" pick, never "good".
    ds = _FakeDataset([1, 1, 1, 1, 1, 1])
    iou = [0.9, 0.8, math.nan, 0.6, 0.5, 0.4]
    picked = pick_samples(iou, ds, 6)  # pool == N → all returned
    assert 2 in picked  # eligible as worst
    # With N < pool, the top "good" band must not include the NaN index.
    picked2 = pick_samples(iou, ds, 2)
    _g, _, _w = _bands(2)  # (1, 0, 1)
    assert picked2[0] != 2  # highest-IoU first, never the NaN


def test_returned_in_descending_iou_order() -> None:
    ds = _FakeDataset([1] * 6)
    iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    picked = pick_samples(iou, ds, 6)
    vals = [iou[i] for i in picked]
    assert vals == sorted(vals, reverse=True)


def test_denormalize_to_rgb_round_trip() -> None:
    import numpy as np
    from PIL import Image

    from custom_sam_peft.eval.visualize import denormalize_to_rgb

    # Known uint8 image → normalize with mean/std → denorm → expect round-trip.
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)  # (H, W, C)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = (orig.astype(np.float32) / 255.0 - np.asarray(mean)) / np.asarray(std)
    tensor = torch.from_numpy(norm).permute(2, 0, 1)  # (C, H, W)
    img = denormalize_to_rgb(tensor, mean, std)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (7, 5)  # (W, H)
    back = np.asarray(img)
    assert np.abs(back.astype(int) - orig.astype(int)).max() <= 2  # rounding tolerance


def test_denormalize_to_rgb_n_channel_uses_first_3() -> None:
    from custom_sam_peft.eval.visualize import denormalize_to_rgb

    tensor = torch.zeros(5, 4, 6)  # C=5, H=4, W=6
    mean = [0.5] * 5
    std = [0.5] * 5
    img = denormalize_to_rgb(tensor, mean, std)
    assert img.mode == "RGB"
    assert img.size == (6, 4)  # (W, H); only first 3 channels rendered


def test_denormalize_to_rgb_grayscale_padded() -> None:
    from custom_sam_peft.eval.visualize import denormalize_to_rgb

    img = denormalize_to_rgb(torch.zeros(1, 4, 6), [0.5], [0.5])
    assert img.mode == "RGB"
    assert img.size == (6, 4)


def test_denormalize_to_rgb_two_channel_padded() -> None:
    from custom_sam_peft.eval.visualize import denormalize_to_rgb

    img = denormalize_to_rgb(torch.zeros(2, 4, 6), [0.5, 0.5], [0.5, 0.5])
    assert img.mode == "RGB"
    assert img.size == (6, 4)


def test_gt_instances_to_entries_conversion() -> None:
    import pycocotools.mask as mask_utils

    from custom_sam_peft.data.base import Instance
    from custom_sam_peft.eval.visualize import gt_instances_to_entries

    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[1:5, 2:6] = True
    inst = Instance(mask=mask, class_id=2, box=torch.tensor([2.0, 1.0, 6.0, 5.0]))
    entries = gt_instances_to_entries([inst])
    assert len(entries) == 1
    e = entries[0]
    assert e["category_id"] == 3  # class_id + 1
    assert e["bbox"] == [2.0, 1.0, 4.0, 4.0]  # xyxy -> xywh
    assert "score" not in e  # GT carries no score
    # segmentation decodes back to the input mask.
    decoded = mask_utils.decode(e["segmentation"])  # (H, W) uint8
    assert decoded.shape == (8, 8)
    assert bool((torch.from_numpy(decoded).bool() == mask).all())


def test_compose_pair_hstacks_with_titles_and_legend() -> None:
    from PIL import Image

    from custom_sam_peft.eval.visualize import _compose_pair

    gt = Image.new("RGB", (40, 30), color=(10, 10, 10))
    pred = Image.new("RGB", (40, 30), color=(20, 20, 20))
    composite = _compose_pair(gt, pred, class_names_present=["cat", "dog"])
    # Width is at least the sum of the two panels (hstacked), height >= panel height.
    assert composite.width >= gt.width + pred.width
    assert composite.height >= gt.height


def test_sanitize_image_id() -> None:
    from custom_sam_peft.eval.visualize import _sanitize_image_id

    assert _sanitize_image_id("img_0") == "img_0"
    assert _sanitize_image_id("a/b/c") == "a_b_c"
    assert _sanitize_image_id("http://x/y.jpg") == "http___x_y.jpg"
    assert "/" not in _sanitize_image_id("nested/path:weird name")
    assert "\\" not in _sanitize_image_id("win\\path")
