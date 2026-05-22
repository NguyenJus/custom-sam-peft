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
import transformers as _transformers

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.transforms import build_eval_transforms, resolve_normalization

# Pre-warm the transformers lazy module so patch("transformers.AutoImageProcessor", ...)
# works correctly. The _LazyModule caches attribute resolutions on first access; without
# this warm-up, patch's getattr call both triggers the lazy load AND caches the result in
# a way that makes subsequent patches unreliable across tests.
_ = _transformers.AutoImageProcessor


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
    """Path 1, model not in KNOWN_PROCESSOR_STATS: returns processor values, logs INFO."""
    fake_proc = SimpleNamespace(image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("some/other-backbone", NormalizeConfig())

    mock_aip.from_pretrained.assert_called_once_with("some/other-backbone", local_files_only=True)
    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )
    # No WARN records on the happy path.
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_resolve_normalization_falls_back_on_oserror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 2: OSError + model in table -> table fallback, exactly one WARN."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    assert "known-good stats" in warn_records[0].getMessage()


def test_resolve_normalization_falls_back_on_attribute_error() -> None:
    """Path 2: AttributeError + model in table -> table fallback."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = SimpleNamespace()  # missing image_mean/image_std

    with patch("transformers.AutoImageProcessor", mock_aip):
        mean, _std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]


def test_resolve_normalization_processor_loads_no_table_entry_no_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, model NOT in table: returns processor values, no WARN, INFO present."""
    fake_proc = SimpleNamespace(image_mean=[0.7, 0.7, 0.7], image_std=[0.2, 0.2, 0.2])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("some/unknown-backbone", NormalizeConfig())

    assert mean == [0.7, 0.7, 0.7]
    assert std == [0.2, 0.2, 0.2]
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )


def test_resolve_normalization_processor_loads_matches_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, model in table, values within 1e-3 of table entry: no WARN, INFO present."""
    fake_proc = SimpleNamespace(
        image_mean=[0.4855, 0.4555, 0.4055],  # within 1e-3 of [0.485, 0.456, 0.406]
        image_std=[0.2295, 0.2245, 0.2255],  # within 1e-3 of [0.229, 0.224, 0.225]
    )
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.4855, 0.4555, 0.4055]
    assert std == [0.2295, 0.2245, 0.2255]
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )


def test_resolve_normalization_processor_loads_diverges_from_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 1, in-table divergence > 1e-3: returns proc values + one WARN naming both vectors."""
    fake_proc = SimpleNamespace(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    # Table is a sentinel, not a gate: processor values are returned.
    assert mean == [0.5, 0.5, 0.5]
    assert std == [0.5, 0.5, 0.5]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    # Both vectors must appear in the single WARN message.
    assert "0.5" in msg
    assert "0.485" in msg
    assert "0.229" in msg


def test_resolve_normalization_processor_fails_model_in_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 2: OSError + model in table -> returns table values, one WARN naming the fallback."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        # User's NormalizeConfig is intentionally distinct from the table to confirm table wins.
        user_norm = NormalizeConfig(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mean, std = resolve_normalization("facebook/sam3.1", user_norm)

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    assert "known-good stats" in msg
    assert "0.485" in msg


def test_resolve_normalization_processor_fails_model_not_in_table(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path 3: OSError + not in table -> returns user's fallback, one WARN naming YAML values."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.transforms")
        user_norm = NormalizeConfig(mean=[0.3, 0.3, 0.3], std=[0.2, 0.2, 0.2])
        mean, std = resolve_normalization("some/unknown-backbone", user_norm)

    assert mean == [0.3, 0.3, 0.3]
    assert std == [0.2, 0.2, 0.2]
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1
    msg = warn_records[0].getMessage()
    assert "no known-good entry" in msg
    assert "0.3" in msg


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


from pathlib import Path

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.config.schema import DataConfig, ModelConfig


def test_shipped_yamls_match_schema_defaults() -> None:
    """All four shipped YAMLs resolve normalize / image_size / gradient_checkpointing
    to the schema's default values — i.e. the YAML echoes are consistent with the
    schema as the source of truth.
    """
    repo_root = Path(__file__).resolve().parents[2]
    yaml_paths = [
        repo_root / "configs" / "examples" / "coco_text_lora.yaml",
        repo_root / "configs" / "examples" / "coco_text_qlora.yaml",
        repo_root / "src" / "custom_sam_peft" / "cli" / "templates" / "coco_text_lora.yaml",
        repo_root / "src" / "custom_sam_peft" / "cli" / "templates" / "coco_text_qlora.yaml",
    ]
    schema_image_size = DataConfig.model_fields["image_size"].default
    schema_grad_ckpt = ModelConfig.model_fields["gradient_checkpointing"].default
    # NormalizeConfig defaults are constructed via default_factory; build an
    # instance to read them.
    from custom_sam_peft.config.schema import NormalizeConfig

    schema_mean = NormalizeConfig().mean
    schema_std = NormalizeConfig().std

    for p in yaml_paths:
        assert p.is_file(), p
        cfg = load_config(p)
        assert cfg.data.image_size == schema_image_size, p
        assert cfg.model.gradient_checkpointing == schema_grad_ckpt, p
        assert cfg.data.normalize.mean == schema_mean, p
        assert cfg.data.normalize.std == schema_std, p
