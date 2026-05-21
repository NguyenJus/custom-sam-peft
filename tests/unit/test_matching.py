"""Unit tests for HungarianMatcher (revised plan — no class cost)."""

from __future__ import annotations

import torch

from custom_sam_peft.data.base import Instance
from custom_sam_peft.models.matching import CanonicalOutputs, HungarianMatcher


def _make_outputs(q: int = 4, mask_size: int = 16) -> CanonicalOutputs:
    return CanonicalOutputs(
        obj_logits=torch.zeros(1, q),
        pred_boxes=torch.zeros(1, q, 4),
        pred_masks=torch.zeros(1, q, mask_size, mask_size),
        img_presence=torch.zeros(1),
    )


def _instance(box: list[float], mask_size: int = 16) -> Instance:
    return Instance(
        mask=torch.zeros(mask_size, mask_size),
        class_id=0,
        box=torch.tensor(box, dtype=torch.float32),
    )


def test_matcher_empty_targets_returns_empty_pairs() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = _make_outputs(q=4)
    indices = matcher(outputs, [[]])
    assert len(indices) == 1
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 0
    assert tgt_idx.numel() == 0


def test_matcher_returns_one_match_per_target() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = _make_outputs(q=4)
    targets = [[_instance([0.5, 0.5, 0.1, 0.1]), _instance([0.2, 0.2, 0.1, 0.1])]]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2
    assert sorted(tgt_idx.tolist()) == [0, 1]
    assert len(set(pred_idx.tolist())) == 2


def test_matcher_handles_more_targets_than_queries() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = _make_outputs(q=2)
    targets = [
        [
            _instance([0.1, 0.1, 0.1, 0.1]),
            _instance([0.3, 0.3, 0.1, 0.1]),
            _instance([0.5, 0.5, 0.1, 0.1]),
        ]
    ]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2


def test_matcher_accepts_bf16_predictions() -> None:
    """Matcher must accept bf16 pred_boxes/pred_masks (fp32 targets).

    When the wrapper is cast to bf16 (the LoRA/QLoRA compute_dtype) its
    ``pred_boxes`` and ``pred_masks`` are bf16, but ``torch.cdist`` (used for
    the L1 cost) implements neither CPU nor CUDA kernels for bf16:
    ``NotImplementedError: "cdist_cuda" not implemented for 'BFloat16'`` (and
    the symmetric error on CPU).  The matcher is ``@torch.no_grad()`` so it
    can safely upcast to fp32 internally — this test pins that contract.
    """
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = CanonicalOutputs(
        obj_logits=torch.zeros(1, 4, dtype=torch.bfloat16),
        pred_boxes=torch.tensor(
            [
                [
                    [0.5, 0.5, 0.1, 0.1],
                    [0.2, 0.2, 0.1, 0.1],
                    [0.8, 0.8, 0.1, 0.1],
                    [0.1, 0.9, 0.1, 0.1],
                ]
            ],
            dtype=torch.bfloat16,
        ),
        pred_masks=torch.zeros(1, 4, 16, 16, dtype=torch.bfloat16),
        img_presence=torch.zeros(1, dtype=torch.bfloat16),
    )
    targets = [[_instance([0.5, 0.5, 0.1, 0.1]), _instance([0.2, 0.2, 0.1, 0.1])]]
    indices = matcher(outputs, targets)
    pred_idx, tgt_idx = indices[0]
    assert pred_idx.numel() == 2
    assert tgt_idx.numel() == 2
    assert sorted(tgt_idx.tolist()) == [0, 1]


def test_matcher_batched() -> None:
    matcher = HungarianMatcher(lambda_l1=5.0, lambda_giou=2.0, lambda_mask=5.0)
    outputs = CanonicalOutputs(
        obj_logits=torch.zeros(2, 3),
        pred_boxes=torch.zeros(2, 3, 4),
        pred_masks=torch.zeros(2, 3, 16, 16),
        img_presence=torch.zeros(2),
    )
    targets = [
        [_instance([0.5, 0.5, 0.1, 0.1])],
        [_instance([0.2, 0.2, 0.1, 0.1]), _instance([0.7, 0.7, 0.1, 0.1])],
    ]
    indices = matcher(outputs, targets)
    assert len(indices) == 2
    assert indices[0][0].numel() == 1
    assert indices[1][0].numel() == 2
