"""Shared per-tile preprocessing for predict AND eval (design C, REQUIRED contract).

Both the predict tiled path (``_predict_one_tile``) and the eval tiled branch
(``Evaluator._iter_predictions``) route every native-resolution tile crop through
the single :func:`preprocess_tile` helper, so the per-tile model input is
byte-identical across both paths (spec §5.4, faithfulness-critical).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def preprocess_tile(
    crop_np: np.ndarray[Any, Any],
    transform: Any,
    *,
    device: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply the pad-only transform to a native-res tile crop, return a device tensor.

    The pad-only Albumentations transform pads with raw ``0`` BEFORE ``Normalize``
    (``transforms.py`` ``PadIfNeeded`` -> ``Normalize`` order), so the padded extent
    becomes ``normalize(0)`` (= ``-mean/std``, a NON-zero per-channel constant), NOT
    literal ``0``. Padding an already-normalized tensor with literal ``0`` would feed
    the model a DIFFERENT input than predict/train; running this shared helper on the
    raw numpy crop in both paths guarantees byte-identical per-tile inputs.

    Returns the ``(C, image_size, image_size)`` tensor on *device* with *dtype*.
    """
    out = transform(image=crop_np, bboxes=[], class_labels=[], instance_idx=[])
    return out["image"].to(device, dtype=dtype)
