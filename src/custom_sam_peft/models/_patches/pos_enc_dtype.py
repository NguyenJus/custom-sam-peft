"""Patch: wrap PositionEmbeddingSine._encode_xy to honor input dtype.

See src/custom_sam_peft/models/sam3.py::_patch_pos_enc_dtype for full rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

from torch import nn

logger = logging.getLogger(__name__)


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Wrap every PositionEmbeddingSine._encode_xy to honor input dtype.

    Per-instance MethodType replacement; idempotent.
    See ``_patch_pos_enc_dtype`` in models/sam3.py for the full rationale.
    """
    from types import MethodType

    from sam3.model.position_encoding import PositionEmbeddingSine

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, PositionEmbeddingSine):
            continue
        if getattr(submodule, "_custom_sam_peft_pos_enc_dtype_patched", False):
            continue
        original = submodule._encode_xy

        def _encode_xy_dtype_aware(self, x, y, _orig=original):  # type: ignore[no-untyped-def]
            pos_x, pos_y = _orig(x, y)
            return pos_x.to(dtype=x.dtype), pos_y.to(dtype=x.dtype)

        submodule._encode_xy = MethodType(_encode_xy_dtype_aware, submodule)
        submodule._custom_sam_peft_pos_enc_dtype_patched = True  # idempotency marker
        patched_count += 1

    logger.info(
        "Patched %d PositionEmbeddingSine._encode_xy callsites for dtype awareness.",
        patched_count,
    )
