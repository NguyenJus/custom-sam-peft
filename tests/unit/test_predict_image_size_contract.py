"""CPU regression test for the predict default-image-size (1008 RoPE) contract.

Bug class 3 — documented in docs/testing/gpu-audit-2026-05-24.md:
  On a real GPU run, a config-less predict call resized the input to 1024, but
  SAM 3.1's RoPE positional embeddings expect the canonical size 1008, causing a
  ``freqs_cis.shape == (H, W)`` assertion to trip at runtime.

  The pre-fix code had a constant ``_BUILTIN_DEFAULT_IMAGE_SIZE = 1024`` that was
  used as the resolved image size.  The fix removed that constant and replaced the
  resolution path with ``image_size = SAM3_IMAGE_SIZE`` (imported from
  ``custom_sam_peft.models.sam3``), where ``SAM3_IMAGE_SIZE = 1008``.

  This file locks in both:
    1. The value of ``SAM3_IMAGE_SIZE`` (must be 1008, not 1024).
    2. The end-to-end resolution path: ``_resolve_config`` with no image-size
       override must produce a ``_ResolvedConfig`` whose ``image_size`` equals
       ``SAM3_IMAGE_SIZE`` (1008).

  These are pure CPU tests — no CUDA, no checkpoint loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE
from custom_sam_peft.predict.runner import PredictOptions, _resolve_config, _ResolvedConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_OPTS = PredictOptions(
    images=Path("."),
    prompts="cat",
    output=Path("."),
    checkpoint=None,
    merge_adapter=False,
    config=None,
    score_threshold=0.5,
    top_k=10,
    save_masks="none",
    visualize=False,
    device="cpu",
    dtype="float32",
    seed=0,
    dry_run=False,
    verbose=False,
    use_onnx=None,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSam3ImageSizeConstant:
    """SAM3_IMAGE_SIZE must be 1008 — the value the RoPE table is built for."""

    def test_sam3_image_size_is_1008(self) -> None:
        """SAM3_IMAGE_SIZE == 1008.

        Bug class 3 root cause: the pre-fix constant was 1024, which is
        incompatible with SAM 3.1's RoPE positional embeddings.  This assertion
        fails immediately if a future change accidentally reverts the constant.
        """
        assert SAM3_IMAGE_SIZE == 1008, (
            f"SAM3_IMAGE_SIZE must be 1008 (SAM 3.1 RoPE grid size), got {SAM3_IMAGE_SIZE}. "
            "Changing this to 1024 will cause freqs_cis shape mismatches at inference time."
        )


class TestResolveConfigImageSize:
    """_resolve_config must wire image_size = SAM3_IMAGE_SIZE (1008) by default."""

    def test_resolved_config_image_size_equals_sam3_image_size(self) -> None:
        """Config-less _resolve_config returns image_size == SAM3_IMAGE_SIZE.

        This exercises the full resolution path (not just the constant) so that
        a regression in _resolve_config — e.g., restoring a hardcoded 1024
        fallback — would be caught on CPU without any GPU run.

        Pre-fix behaviour: image_size was resolved from _BUILTIN_DEFAULT_IMAGE_SIZE
        (= 1024), so this assertion would have failed with 1024 != 1008.
        """
        rcfg: _ResolvedConfig = _resolve_config(_MINIMAL_OPTS)

        assert rcfg.image_size == SAM3_IMAGE_SIZE, (
            f"_resolve_config must set image_size to SAM3_IMAGE_SIZE ({SAM3_IMAGE_SIZE}), "
            f"got {rcfg.image_size}.  Bug class 3 (RoPE shape mismatch) would reoccur."
        )

    def test_resolved_config_image_size_is_literal_1008(self) -> None:
        """Resolved image_size == 1008 (literal guard — independent of the constant).

        Separately asserts the numeric value so that a change to *both*
        SAM3_IMAGE_SIZE and _resolve_config to a new, mutually consistent but
        wrong value (e.g., both set to 1024) would still fail here.

        If SAM 3.1's RoPE table is ever rebuilt for a different resolution, this
        test must be updated together with the model checkpoint and SAM3_IMAGE_SIZE.
        """
        rcfg: _ResolvedConfig = _resolve_config(_MINIMAL_OPTS)

        assert rcfg.image_size == 1008, (
            f"Resolved image_size must be the literal 1008, got {rcfg.image_size}. "
            "Update this test only when the SAM 3.1 RoPE grid is deliberately changed."
        )

    def test_resolved_config_is_frozen_dataclass(self) -> None:
        """_ResolvedConfig is a frozen dataclass — image_size cannot be mutated post-build.

        Guards against a future refactor that unfreezes the dataclass and allows
        downstream code to silently overwrite image_size.
        """
        rcfg: _ResolvedConfig = _resolve_config(_MINIMAL_OPTS)

        with pytest.raises((AttributeError, TypeError)):
            rcfg.image_size = 1024  # type: ignore[misc]
