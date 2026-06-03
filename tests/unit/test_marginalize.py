# tests/unit/test_marginalize.py
"""marginalize_group + build_semantic_logits + semantic_argmax (§6)."""

from __future__ import annotations

import torch

from custom_sam_peft.models.semantic import (
    build_semantic_logits,
    marginalize_group,
    semantic_argmax,
)


def _stub_outputs(b, k, q=4, h=8, w=8):
    n = b * k
    return {
        "pred_logits": torch.randn(n, q, 1),
        "pred_masks": torch.randn(n, q, h, w),
        "presence_logit_dec": torch.randn(n, 1),
        "semantic_seg": torch.randn(n, 1, h, w),
    }


def test_marginalize_max_shape():
    out = _stub_outputs(2, 3)
    fg = marginalize_group(out, 2, 3, query_reduce="max", source="marginalize")
    assert fg.shape == (2, 3, 8, 8)  # (b, k, H, W) per-concept LOGITS


def test_marginalize_sum_shape():
    out = _stub_outputs(2, 3)
    fg = marginalize_group(out, 2, 3, query_reduce="sum", source="marginalize")
    assert fg.shape == (2, 3, 8, 8)


def test_semantic_seg_source_shape():
    out = _stub_outputs(2, 3)
    fg = marginalize_group(out, 2, 3, query_reduce="max", source="semantic_seg")
    assert fg.shape == (2, 3, 8, 8)  # surfaced directly from out["semantic_seg"]


def test_build_semantic_logits_prepends_background():
    # one group covering all K=3 concepts; b=2.
    fg = torch.randn(2, 3, 8, 8)
    sem = build_semantic_logits([fg], background_logit=0.0)
    assert sem.shape == (2, 4, 8, 8)  # K+1 channels
    assert torch.allclose(sem[:, 0], torch.zeros(2, 8, 8))  # bg channel == background_logit


def test_argmax_background_wins_when_all_fg_negative():
    # All concept logits very negative -> argmax picks bg channel 0.
    fg = torch.full((1, 2, 4, 4), -10.0)
    sem = build_semantic_logits([fg], background_logit=0.0)
    labels = semantic_argmax(sem)
    assert labels.shape == (1, 4, 4)
    assert torch.all(labels == 0)


def test_argmax_concept_wins_when_fg_high():
    fg = torch.full((1, 2, 4, 4), -10.0)
    fg[:, 1] = 10.0  # concept 1 (channel 2) dominates
    sem = build_semantic_logits([fg], background_logit=0.0)
    labels = semantic_argmax(sem)
    assert torch.all(labels == 2)  # channel index 2 == concept dense_id 1 + 1


def test_multigroup_concat_preserves_concept_order():
    # two groups of 2 concepts each -> K=4 total, concat along concept axis.
    g0 = torch.randn(1, 2, 4, 4)
    g1 = torch.randn(1, 2, 4, 4)
    sem = build_semantic_logits([g0, g1], background_logit=0.0)
    assert sem.shape == (1, 5, 4, 4)  # 4 concepts + bg
