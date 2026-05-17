"""Unit tests for per-component losses + total_loss in models/losses.py."""

from __future__ import annotations

import torch

from esam3.models.losses import mask_loss


def test_mask_loss_zero_on_perfect_match() -> None:
    pred = torch.full((2, 32, 32), -10.0)
    pred[:, :16, :] = 10.0
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_mask_loss_positive_when_wrong() -> None:
    pred = torch.zeros(2, 32, 32)
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.item() > 0.0


def test_mask_loss_upsamples_pred_to_target_resolution() -> None:
    pred = torch.zeros(2, 16, 16)
    target = torch.zeros(2, 32, 32)
    loss = mask_loss(pred, target)
    assert torch.isfinite(loss)
