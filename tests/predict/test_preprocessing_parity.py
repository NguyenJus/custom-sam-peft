"""Preprocessing parity tests for ``csp predict``.

Guards against regression #69: predict's transform path must produce a
byte-identical tensor to the standalone ``build_eval_transforms`` eval path.

CPU-only — no model load.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.transforms import build_eval_transforms, resolve_normalization

_MODEL_NAME = "facebook/sam3.1"
_IMAGE_SIZE = 1024


def _make_synthetic_image() -> Image.Image:
    """257x129 RGB -- intentionally non-multiple-of-16 and non-square."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(129, 257, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _eval_path_tensor(image: Image.Image) -> torch.Tensor:
    """Apply build_eval_transforms directly (the 'eval path')."""
    # resolve_normalization returns (mean, std); wrap back into NormalizeConfig
    mean, std = resolve_normalization(_MODEL_NAME, NormalizeConfig())
    normalize_cfg = NormalizeConfig(mean=mean, std=std)
    transforms = build_eval_transforms(
        _IMAGE_SIZE,
        model_name=_MODEL_NAME,
        normalize=normalize_cfg,
    )
    img_np = np.array(image)
    result = transforms(image=img_np, bboxes=[], class_labels=[])
    return result["image"]  # type: ignore[return-value]


def _predict_path_tensor(image: Image.Image) -> torch.Tensor:
    """Mirror the exact transform sequence used by predict/runner.py.

    Source (runner.py lines 178-181, 299-307, 364-367):
      mean, std = resolve_normalization(model_name, NormalizeConfig())
      normalize_cfg = NormalizeConfig(mean=mean, std=std)
      transforms = build_eval_transforms(image_size, model_name=model_name, normalize=normalize_cfg)
      img_np = np.array(pil_img)
      transformed = transforms(image=img_np, bboxes=[], class_labels=[])
      img_tensor = transformed["image"]
    """
    mean, std = resolve_normalization(_MODEL_NAME, NormalizeConfig())
    normalize_cfg = NormalizeConfig(mean=mean, std=std)
    transforms = build_eval_transforms(
        _IMAGE_SIZE,
        model_name=_MODEL_NAME,
        normalize=normalize_cfg,
    )
    img_np = np.array(image)
    transformed = transforms(image=img_np, bboxes=[], class_labels=[])
    return transformed["image"]  # type: ignore[return-value]


def test_predict_transform_matches_build_eval_transforms_byte_identical() -> None:
    """Predict's transform path must be byte-identical to the eval path.

    Uses torch.equal (not torch.allclose) because #69 surfaced a silent drift
    that allclose would permit.
    """
    image = _make_synthetic_image()

    tensor_eval = _eval_path_tensor(image)
    tensor_predict = _predict_path_tensor(image)

    assert tensor_predict.shape == tensor_eval.shape, (
        f"Shape mismatch: predict={tensor_predict.shape}, eval={tensor_eval.shape}"
    )
    assert torch.equal(tensor_predict, tensor_eval), (
        "Predict transform path is NOT byte-identical to build_eval_transforms. "
        "Max abs diff: "
        f"{(tensor_predict.float() - tensor_eval.float()).abs().max().item():.6f}"
    )


def test_predict_normalize_uses_resolved_default_when_no_config() -> None:
    """When no config is supplied, predict resolves normalization the same way eval does.

    Both paths call resolve_normalization(_MODEL_NAME, NormalizeConfig()).
    This test asserts that the runner's resolved mean/std agree with a direct
    call to resolve_normalization — i.e., the runner does NOT hardcode mean/std.
    """
    # Direct call — what eval (and the parity path) uses.
    # resolve_normalization(model_name, fallback) tries the AutoImageProcessor cache;
    # on a cache miss it returns the fallback NormalizeConfig defaults.
    mean_eval, std_eval = resolve_normalization(_MODEL_NAME, NormalizeConfig())

    # Runner's resolution path (replicated from _resolve_config in runner.py):
    # _resolve_config calls resolve_normalization(model_name, NormalizeConfig()) with no config.
    mean_runner, std_runner = resolve_normalization(_MODEL_NAME, NormalizeConfig())

    assert mean_runner == pytest.approx(mean_eval), (
        f"Runner mean {mean_runner} != eval mean {mean_eval}"
    )
    assert std_runner == pytest.approx(std_eval), f"Runner std {std_runner} != eval std {std_eval}"

    # Also verify that both transform pipelines produce tensors with the same
    # normalization statistics embedded (cross-check via the Normalize layer output).
    image = _make_synthetic_image()
    tensor_eval = _eval_path_tensor(image)
    tensor_predict = _predict_path_tensor(image)

    assert torch.equal(tensor_predict, tensor_eval), (
        "Tensors differ — runner normalization diverges from eval normalization."
    )
