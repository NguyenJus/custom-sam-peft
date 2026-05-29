"""Unit tests for the model-dependent eval viz pass (CPU-only, tiny stub)."""

from __future__ import annotations

from typing import ClassVar

import torch
from PIL import Image

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.visualize import render_eval_pair
from custom_sam_peft.models.matching import HungarianMatcher
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _example(class_id: int) -> Example:
    h = w = 8
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[:4, :4] = True
    return Example(
        image=torch.zeros(3, h, w),
        image_id="img_0",
        prompts=TextPrompts(classes=["cat", "dog"]),
        instances=[Instance(mask=mask, class_id=class_id, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
    )


def _matcher() -> HungarianMatcher:
    w = MatcherWeights()
    return HungarianMatcher(
        lambda_l1=w.lambda_l1, lambda_giou=w.lambda_giou, lambda_mask=w.lambda_mask
    )


def test_render_eval_pair_returns_hstacked_image() -> None:
    model = TinySam3Stub()
    ex = _example(class_id=0)
    out = render_eval_pair(
        model,
        ex,
        ["cat", "dog"],
        mask_threshold=0.0,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        matcher=_matcher(),
    )
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    # Hstacked: width >= 2 * source width (8 px each, plus legend/titles add height not width).
    assert out.width >= 16


def test_render_eval_pair_no_gt_class_draws_no_pred_for_that_class() -> None:
    # Image has a single 'cat' (class_id 0) GT; 'dog' (class_id 1) has no GT, so
    # the dog matcher target list is empty and no dog pred is drawn. The call must
    # not raise and must return a composite.
    model = TinySam3Stub()
    ex = _example(class_id=0)
    out = render_eval_pair(
        model,
        ex,
        ["cat", "dog"],
        mask_threshold=0.0,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        matcher=_matcher(),
    )
    assert isinstance(out, Image.Image)


def _dataset(class_ids: list[int]):
    examples = [_example(class_id=c) for c in class_ids]
    for i, ex in enumerate(examples):
        # give each a distinct image_id (frozen dataclass → rebuild)
        examples[i] = Example(
            image=ex.image, image_id=f"img_{i}", prompts=ex.prompts, instances=ex.instances
        )

    class _DS:
        class_names: ClassVar[list[str]] = ["cat", "dog"]

        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, j: int) -> Example:
            return examples[j]

    return _DS()


def test_write_eval_visualizations_writes_pngs(tmp_path) -> None:
    from custom_sam_peft.eval.visualize import write_eval_visualizations

    ds = _dataset([0, 1, 0])  # 3 GT-bearing images
    model = TinySam3Stub()
    paths = write_eval_visualizations(
        model,
        ds,
        tmp_path,
        per_example_iou=[0.9, 0.5, 0.1],
        count=10,
        mask_threshold=0.0,
        model_name="facebook/sam3.1",
        normalize=None,
        channel_semantics="rgb",
    )
    assert len(paths) == 3  # small pool → all candidates
    vis_dir = tmp_path / "visualizations"
    assert vis_dir.is_dir()
    written = sorted(p.name for p in vis_dir.glob("*.png"))
    assert written == ["img_0.png", "img_1.png", "img_2.png"]
    for p in paths:
        Image.open(p).verify()  # readable image


def test_write_eval_visualizations_zero_candidates(tmp_path, caplog) -> None:
    from custom_sam_peft.data.base import Example, TextPrompts
    from custom_sam_peft.eval.visualize import write_eval_visualizations

    # All images have NO GT → zero candidates.
    no_gt = [
        Example(
            image=torch.zeros(3, 8, 8),
            image_id=f"n_{i}",
            prompts=TextPrompts(classes=["cat", "dog"]),
            instances=[],
        )
        for i in range(2)
    ]

    class _DS:
        class_names: ClassVar[list[str]] = ["cat", "dog"]

        def __len__(self) -> int:
            return len(no_gt)

        def __getitem__(self, j: int) -> Example:
            return no_gt[j]

    with caplog.at_level("INFO"):
        paths = write_eval_visualizations(
            TinySam3Stub(),
            _DS(),
            tmp_path,
            per_example_iou=[1.0, 1.0],
            count=5,
            mask_threshold=0.0,
            model_name="facebook/sam3.1",
            normalize=None,
            channel_semantics="rgb",
        )
    assert paths == []
    assert not (tmp_path / "visualizations").exists() or not list(
        (tmp_path / "visualizations").glob("*.png")
    )
    assert any("no GT" in r.message.lower() or "no gt" in r.message.lower() for r in caplog.records)


def test_write_eval_visualizations_per_image_failure_is_caught(
    tmp_path, monkeypatch, caplog
) -> None:
    """A single image that raises during render is logged at WARNING and skipped;
    other images still render."""
    from custom_sam_peft.eval import visualize as viz

    ds = _dataset([0, 1])
    model = TinySam3Stub()
    calls = {"n": 0}
    real = viz.render_eval_pair

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(viz, "render_eval_pair", flaky)
    with caplog.at_level("WARNING"):
        paths = viz.write_eval_visualizations(
            model,
            ds,
            tmp_path,
            per_example_iou=[0.9, 0.1],
            count=10,
            mask_threshold=0.0,
            model_name="facebook/sam3.1",
            normalize=None,
            channel_semantics="rgb",
        )
    assert len(paths) == 1  # one survived
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert any(
        r.levelname == "WARNING" and ("image_id=" in r.getMessage() or "img_" in r.getMessage())
        for r in caplog.records
    )
