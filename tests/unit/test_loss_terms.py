"""Tests for the 15 loss-term classes (spec §11.2)."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.models.losses.terms import box as box_terms
from custom_sam_peft.models.losses.terms import mask as mask_terms
from custom_sam_peft.models.losses.terms import obj as obj_terms
from custom_sam_peft.models.losses.terms import presence as presence_terms

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Mask axis
# ---------------------------------------------------------------------------

_MASK_CLASSES = [
    mask_terms.BCELoss,
    mask_terms.DiceLoss,
    mask_terms.DiceBCELoss,
    mask_terms.FocalBCELoss,
    mask_terms.FocalDiceLoss,
    mask_terms.TverskyLoss,
    mask_terms.FocalTverskyLoss,
    mask_terms.BoundaryLoss,
]


@pytest.fixture
def mask_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.randn(2, 16, 16, requires_grad=True)
    tgt = torch.zeros(2, 16, 16)
    tgt[:, 4:12, 4:12] = 1
    return pred, tgt


@pytest.mark.parametrize("cls", _MASK_CLASSES)
def test_mask_forward_finite_and_backprops(cls: type, mask_batch) -> None:
    pred, tgt = mask_batch
    term = cls()
    val = term(pred, tgt)
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    pred.grad = None


def test_mask_dice_equiv_tversky_alpha_half(mask_batch) -> None:
    pred, tgt = mask_batch
    dice = mask_terms.DiceLoss()(pred, tgt)
    tversky = mask_terms.TverskyLoss(tversky_alpha=0.5)(pred, tgt)
    assert torch.allclose(dice, tversky, atol=1e-5)


def test_mask_focal_tversky_equiv_tversky_at_gamma_one(mask_batch) -> None:
    pred, tgt = mask_batch
    tversky = mask_terms.TverskyLoss(tversky_alpha=0.7)(pred, tgt)
    ft = mask_terms.FocalTverskyLoss(tversky_alpha=0.7, tversky_gamma=1.0)(pred, tgt)
    assert torch.allclose(tversky, ft, atol=1e-5)


def test_mask_focal_bce_equiv_bce_at_gamma_zero(mask_batch) -> None:
    pred, tgt = mask_batch
    bce = mask_terms.BCELoss()(pred, tgt)
    # focal_alpha=0.5 → flat alpha_t weighting (0.5 for both classes); γ=0 kills focal weight
    focal = mask_terms.FocalBCELoss(focal_gamma=0.0, focal_alpha=0.5)(pred, tgt)
    # The alpha_t=0.5 scaling halves the per-pixel CE; multiply by 2 to compare.
    assert torch.allclose(bce, 2.0 * focal, atol=1e-5)


def test_mask_boundary_zero_weight_equals_dice(mask_batch) -> None:
    pred, tgt = mask_batch
    dice = mask_terms.DiceLoss()(pred, tgt)
    boundary = mask_terms.BoundaryLoss(boundary_weight=0.0)(pred, tgt)
    assert torch.allclose(dice, boundary, atol=1e-5)


def test_mask_boundary_finite_under_extreme_imbalance() -> None:
    """All-zero target (no positives) — Kervadec branch must not produce NaN."""
    pred = torch.randn(2, 16, 16, requires_grad=True)
    tgt = torch.zeros(2, 16, 16)
    val = mask_terms.BoundaryLoss(boundary_weight=0.2)(pred, tgt)
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_mask_upsample_when_shapes_differ() -> None:
    pred = torch.randn(2, 8, 8, requires_grad=True)
    tgt = torch.zeros(2, 16, 16)
    tgt[:, 4:12, 4:12] = 1
    val = mask_terms.DiceBCELoss()(pred, tgt)  # auto-upsamples pred to 16x16
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# Box axis
# ---------------------------------------------------------------------------

_BOX_CLASSES = [box_terms.L1GIoULoss, box_terms.GIoUOnlyLoss, box_terms.CIoULoss]


@pytest.fixture
def box_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]], requires_grad=True)
    tgt = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3]])
    return pred, tgt


@pytest.mark.parametrize("cls", _BOX_CLASSES)
def test_box_forward_finite_and_backprops(cls: type, box_batch) -> None:
    pred, tgt = box_batch
    val = cls()(pred, tgt)
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("cls", _BOX_CLASSES)
def test_box_empty_input_returns_zero(cls: type) -> None:
    empty = torch.zeros((0, 4))
    val = cls()(empty, empty)
    assert val.item() == 0.0


def test_box_giou_only_disjoint_boxes_finite() -> None:
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]], requires_grad=True)
    tgt = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    val = box_terms.GIoUOnlyLoss()(pred, tgt)
    assert torch.isfinite(val).item()


# ---------------------------------------------------------------------------
# Obj axis
# ---------------------------------------------------------------------------


@pytest.fixture
def obj_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ol = torch.randn(2, 8, requires_grad=True)
    mm = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    return ol, mm


@pytest.mark.parametrize("cls", [obj_terms.BCELoss, obj_terms.FocalBCELoss])
def test_obj_forward_finite_and_backprops(cls: type, obj_batch) -> None:
    ol, mm = obj_batch
    val = cls()(ol, mm)
    assert torch.isfinite(val).item()
    val.backward()
    assert ol.grad is not None


# ---------------------------------------------------------------------------
# Presence axis
# ---------------------------------------------------------------------------


@pytest.fixture
def presence_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ip = torch.randn(4, requires_grad=True)
    ht = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    return ip, ht


@pytest.mark.parametrize("cls", [presence_terms.BCELoss, presence_terms.FocalBCELoss])
def test_presence_forward_finite_and_backprops(cls: type, presence_batch) -> None:
    ip, ht = presence_batch
    val = cls()(ip, ht)
    assert torch.isfinite(val).item()
    val.backward()
    assert ip.grad is not None
