"""Regression test for _row_outputs non-tensor bug class (bug class 2).

Guards against a crash in the predict/eval path where ``_row_outputs`` iterated
ALL keys in the model output dict and applied ``v[r:r+1]`` unconditionally.
sam3's ``forward_grounding`` returns non-tensor entries (e.g. a nested
``prev_encoder_out`` dict, ``encoder_hidden_states`` as a list) — slicing those
with a ``slice`` key raised ``KeyError(slice)`` / ``TypeError``.

The fix (``if isinstance(v, torch.Tensor)`` guard in the comprehension) is
already in source. This file locks in that contract so bug class 2 can never
escape to a GPU run again.

Reference: docs/testing/gpu-audit-2026-05-24.md — bug class 2.

These are pure CPU tests — no checkpoint loading, no CUDA required.
"""

from __future__ import annotations

import torch

from custom_sam_peft.eval.evaluator import _row_outputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MULTIPLEX = 3  # row-count in the fake outputs dict; any value >= 2 works

_TENSOR_KEYS: list[str] = [
    "pred_logits",
    "pred_boxes",
    "pred_masks",
    "presence_logit_dec",
]

_NONTENSOR_KEYS: dict[str, object] = {
    # Nested dict — realistic sam3 prev_encoder_out shape
    "prev_encoder_out": {
        "encoder_embedding": None,
        "vision_features": None,
        "backbone_fpn": [],
    },
    # Non-tensor sequence — realistic encoder_hidden_states
    "encoder_hidden_states": [torch.randn(2, 4, 8) for _ in range(2)],
}


def _build_outputs() -> dict[str, object]:
    """Build a realistic mixed outputs dict with MULTIPLEX leading rows."""
    outputs: dict[str, object] = {
        "pred_logits": torch.randn(MULTIPLEX, 1, 10),
        "pred_boxes": torch.randn(MULTIPLEX, 1, 4),
        "pred_masks": torch.randn(MULTIPLEX, 1, 1, 16, 16),
        "presence_logit_dec": torch.randn(MULTIPLEX, 1, 10),
    }
    outputs.update(_NONTENSOR_KEYS)
    return outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRowOutputsNonTensor:
    """_row_outputs must filter non-tensor entries and slice tensor entries correctly."""

    def test_does_not_raise_on_mixed_outputs(self) -> None:
        """_row_outputs must not raise when the dict contains non-tensor values.

        Pre-fix behaviour: ``v[r:r+1]`` over a nested dict raised
        ``KeyError(slice(r, r+1, None))``; over a list it would return a
        sub-list rather than raise, but the semantics were wrong.
        Post-fix: the ``isinstance(v, torch.Tensor)`` guard prevents either.
        """
        outputs = _build_outputs()
        # Must not raise — that is the primary contract this test guards.
        result = _row_outputs(outputs, r=0)  # type: ignore[arg-type]
        assert result is not None

    def test_non_tensor_keys_are_dropped(self) -> None:
        """Non-tensor entries must be absent from the returned dict."""
        outputs = _build_outputs()
        result = _row_outputs(outputs, r=0)  # type: ignore[arg-type]

        for key in _NONTENSOR_KEYS:
            assert key not in result, f"Non-tensor key {key!r} should be dropped by _row_outputs"

    def test_tensor_keys_are_present(self) -> None:
        """All tensor prediction keys must be present in the returned dict."""
        outputs = _build_outputs()
        result = _row_outputs(outputs, r=0)  # type: ignore[arg-type]

        for key in _TENSOR_KEYS:
            assert key in result, f"Tensor key {key!r} should be present in _row_outputs result"

    def test_row_0_batch_dim_preserved_at_size_1(self) -> None:
        """Row 0 slice: each tensor's leading dim must be 1 (batch dim preserved)."""
        outputs = _build_outputs()
        result = _row_outputs(outputs, r=0)  # type: ignore[arg-type]

        for key in _TENSOR_KEYS:
            assert result[key].shape[0] == 1, (
                f"{key}: expected leading dim 1 after row-slice, got shape {result[key].shape}"
            )

    def test_row_1_batch_dim_preserved_at_size_1(self) -> None:
        """Row 1 slice: same contract, different row index."""
        outputs = _build_outputs()
        result = _row_outputs(outputs, r=1)  # type: ignore[arg-type]

        for key in _TENSOR_KEYS:
            assert result[key].shape[0] == 1, (
                f"{key}: expected leading dim 1 after row-slice at r=1, "
                f"got shape {result[key].shape}"
            )

    def test_sliced_values_match_direct_index(self) -> None:
        """Returned tensors must equal outputs[key][r:r+1] for r=0 and r=1."""
        outputs = _build_outputs()

        for r in (0, 1):
            result = _row_outputs(outputs, r=r)  # type: ignore[arg-type]
            for key in _TENSOR_KEYS:
                expected = outputs[key][r : r + 1]  # type: ignore[index]
                assert torch.equal(result[key], expected), (
                    f"{key} at r={r}: result does not match outputs[{key!r}][{r}:{r}+1]"
                )

    def test_only_tensor_keys_returned(self) -> None:
        """The returned dict must contain ONLY entries from the tensor keys."""
        outputs = _build_outputs()
        result = _row_outputs(outputs, r=0)  # type: ignore[arg-type]

        extra = set(result.keys()) - set(_TENSOR_KEYS)
        assert not extra, f"Unexpected keys in result (should only have tensor keys): {extra}"
