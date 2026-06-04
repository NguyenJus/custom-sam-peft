"""GPU regression test for the pinned-buffer non-blocking mask transfer (#288).

Asserts bit-identity between ``_binarize_to_host`` on a real CUDA tensor and the
baseline ``(masks_up > thr).cpu().numpy()`` across a sequence of shapes that
exercise both buffer growth (realloc) and reuse-without-stale-tail.

Run explicitly:
    pytest -m gpu_t4 tests/gpu/test_postprocess_pinned_transfer_gpu.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
]


def test_binarize_to_host_cuda_matches_baseline() -> None:
    """Pinned-buffer D2H transfer is bit-identical to the pageable baseline.

    The M sequence (4, 16, 64, 8, 1) over 32x32 exercises:
      - grow M (4 → 16 → 64): forces buffer realloc;
      - shrink M (64 → 8 → 1): forces buffer reuse; the [:numel] slice in
        ``view_for`` must prevent any stale tail from the larger M from leaking.
    """
    from custom_sam_peft.eval.postprocess import _binarize_to_host

    thr = 0.0
    for m in (4, 16, 64, 8, 1):
        masks_up = torch.randn(m, 32, 32, device="cuda")
        got = _binarize_to_host(masks_up, thr)
        expected = (masks_up > thr).cpu().numpy()
        assert got.dtype == np.bool_, f"expected dtype np.bool_, got {got.dtype}"
        assert np.array_equal(got, expected), (
            f"pinned transfer result differs from baseline at M={m}"
        )
