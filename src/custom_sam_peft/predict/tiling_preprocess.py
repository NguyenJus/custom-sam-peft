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

from custom_sam_peft.runtime import Runtime, to_device


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

    The device move routes through ``runtime.to_device`` (the single sanctioned
    device-move seam, §3 static guard); the dtype cast is a separate pure-dtype
    cast. Result is identical to a combined device-and-dtype move.
    """
    out = transform(image=crop_np, bboxes=[], class_labels=[], instance_idx=[])
    moved = to_device(out["image"], Runtime(device=device, dtype=dtype))
    return moved.to(dtype)
