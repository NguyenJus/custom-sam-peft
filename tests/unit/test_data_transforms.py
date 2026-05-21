"""Tests for data/transforms.py."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.transforms import build_eval_transforms, resolve_normalization


@contextmanager
def _patch_proc_to_imagenet() -> Iterator[None]:
    """Patch AutoImageProcessor so resolve_normalization falls back to ImageNet defaults."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")
    with patch("transformers.AutoImageProcessor", mock_aip):
        yield


def test_resolve_normalization_uses_image_processor_when_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_proc = SimpleNamespace(image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    mock_aip.from_pretrained.assert_called_once_with("facebook/sam3.1", local_files_only=True)
    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )


def test_resolve_normalization_falls_back_on_oserror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    assert any(re.search(r"AutoImageProcessor cache miss", rec.message) for rec in caplog.records)


def test_resolve_normalization_falls_back_on_attribute_error() -> None:
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = SimpleNamespace()  # missing image_mean/image_std

    with patch("transformers.AutoImageProcessor", mock_aip):
        mean, _std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]


def test_eval_transforms_resizes_to_square() -> None:
    with _patch_proc_to_imagenet():
        compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    masks = [np.ones((40, 80), dtype=np.uint8)]
    out = compose(image=img, bboxes=[[0.0, 0.0, 80.0, 40.0]], masks=masks, class_labels=[0])
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 64, 64)
    assert out["image"].dtype == torch.float32
    bx = out["bboxes"][0]
    assert 0 <= bx[0] <= 1 and 0 <= bx[1] <= 1
    assert 60 <= bx[2] <= 64 and 28 <= bx[3] <= 36
    assert out["masks"][0].shape == (64, 64)


def test_eval_transforms_pad_position_top_left() -> None:
    """The right/bottom region should be zero-padded (top-left preserves original)."""
    with _patch_proc_to_imagenet():
        compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    img = np.full((32, 64, 3), 255, dtype=np.uint8)
    out = compose(image=img, bboxes=[], masks=[], class_labels=[])
    top_row = out["image"][0, 0, :]
    bottom_row = out["image"][0, 60, :]
    assert top_row.mean().item() > 0
    assert bottom_row.mean().item() < 0


import random

from custom_sam_peft.config.schema import AugmentationsConfig
from custom_sam_peft.data.transforms import build_train_transforms


def test_train_transforms_deterministic_with_seeded_global_rng() -> None:
    """With albumentations 2.x, determinism is controlled via compose.set_random_seed()."""
    aug = AugmentationsConfig(hflip=True, color_jitter=0.1)

    def run() -> torch.Tensor:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        with _patch_proc_to_imagenet():
            compose = build_train_transforms(aug, 64, model_name="x", normalize=NormalizeConfig())
        compose.set_random_seed(0)
        img = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
        return compose(image=img, bboxes=[], masks=[], class_labels=[])["image"]

    a = run()
    b = run()
    assert torch.equal(a, b)


def test_train_transforms_hflip_disabled() -> None:
    aug = AugmentationsConfig(hflip=False, color_jitter=0.0)
    with _patch_proc_to_imagenet():
        compose = build_train_transforms(aug, 64, model_name="x", normalize=NormalizeConfig())
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:, :8, 0] = 200  # strong left column marker
    flips = 0
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        out = compose(image=img.copy(), bboxes=[], masks=[], class_labels=[])
        left = out["image"][0, :32, :8].mean().item()
        right = out["image"][0, :32, 56:64].mean().item()
        if right > left:
            flips += 1
    assert flips == 0


def test_train_transforms_color_jitter_zero_preserves_color() -> None:
    with _patch_proc_to_imagenet():
        eval_compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
        train_compose = build_train_transforms(
            AugmentationsConfig(hflip=False, color_jitter=0.0),
            64,
            model_name="x",
            normalize=NormalizeConfig(),
        )
    img = np.random.RandomState(0).randint(0, 256, size=(40, 60, 3), dtype=np.uint8)
    e_out = eval_compose(image=img, bboxes=[], masks=[], class_labels=[])
    t_out = train_compose(image=img, bboxes=[], masks=[], class_labels=[])
    assert torch.allclose(e_out["image"], t_out["image"], atol=1e-5)
