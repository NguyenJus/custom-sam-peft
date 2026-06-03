"""Single source of truth for B*K multiplex row ordering.

The B images x K classes multiplex assembles ``B*K`` rows in image-major /
class-minor order, consumed by both the torch adapter
(``sam3.py`` ``_Sam3ImageAdapter.forward``) and the ONNX decoder feed.
This module imports ONLY numpy — it must stay torch-free so the ORT inference
core can build identical index arrays without importing torch.
"""

from __future__ import annotations

import numpy as np

_Int64Array = np.ndarray[tuple[int], np.dtype[np.int64]]


def multiplex_index_arrays(b: int, k: int) -> tuple[_Int64Array, _Int64Array]:
    """Return (img_ids, text_ids) int64 arrays for B*K image-major / class-minor rows."""
    img_ids = np.repeat(np.arange(b, dtype=np.int64), k)
    text_ids = np.tile(np.arange(k, dtype=np.int64), b)
    return img_ids, text_ids
