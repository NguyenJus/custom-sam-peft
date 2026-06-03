# tests/unit/test_semantic_loss.py
"""SemanticLoss.forward shapes + ignore_index + degenerate identities (§7.4)."""

from __future__ import annotations

import torch

from custom_sam_peft.config.schema import SemanticLossConfig
from custom_sam_peft.data.base import SemanticTarget
from custom_sam_peft.models.losses.semantic_compose import build_semantic_loss
from custom_sam_peft.models.losses.semantic_presets import (
    _SEM_TERM_CLASS_NAMES,
    resolve,
)


def _loss(preset="natural", ci="balanced", **ov):
    cfg = SemanticLossConfig(preset=preset, class_imbalance=ci, overrides=ov or {})
    return build_semantic_loss(resolve(cfg))


def test_forward_returns_ce_region_total():
    loss = _loss()
    B, K, H, W = 2, 3, 16, 16
    logits = torch.randn(B, K + 1, H, W, requires_grad=True)
    tgts = [
        SemanticTarget(torch.randint(0, K + 1, (H, W), dtype=torch.int64), ignore_index=255)
        for _ in range(B)
    ]
    out = loss(logits, tgts)
    assert set(out.keys()) == {"ce", "region", "total"}
    out["total"].backward()  # gradients flow
    assert logits.grad is not None


def test_fully_ignored_image_finite_loss():
    loss = _loss()
    B, K, H, W = 1, 2, 8, 8
    logits = torch.randn(B, K + 1, H, W, requires_grad=True)
    labels = torch.full((H, W), 255, dtype=torch.int64)  # all void
    out = loss(logits, [SemanticTarget(labels, ignore_index=255)])
    assert torch.isfinite(out["total"])


def test_gt_downsampled_to_logit_res_nearest():
    # GT at full res (32) vs logits at 16 -> loss downsamples GT, no crash, finite.
    loss = _loss()
    logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    labels = torch.randint(0, 3, (32, 32), dtype=torch.int64)
    out = loss(logits, [SemanticTarget(labels, ignore_index=255)])
    assert torch.isfinite(out["total"])


def test_sem_term_class_names_match_compose_registry():
    # §7.3 sync test: every sem_family resolves to a known compose term.
    from custom_sam_peft.models.losses.semantic_compose import SEM_FAMILY_BUILDERS

    assert set(_SEM_TERM_CLASS_NAMES.keys()) == set(SEM_FAMILY_BUILDERS.keys())
