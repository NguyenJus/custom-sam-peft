"""Unit tests for per-component losses + total_loss in models/losses.py."""

from __future__ import annotations

import torch

from custom_sam_peft.models.losses import box_loss, mask_loss, objectness_loss, presence_loss


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


def test_box_loss_zero_on_perfect_match() -> None:
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss = box_loss(pred, target)
    assert loss.item() < 1e-4


def test_box_loss_positive_when_offset() -> None:
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    loss = box_loss(pred, target)
    assert loss.item() > 0.5


def test_objectness_loss_zero_when_predictions_agree() -> None:
    obj_logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(obj_logits, matched)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_objectness_loss_high_when_predictions_invert() -> None:
    obj_logits = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(obj_logits, matched)
    assert loss.item() > 1.0


def test_presence_loss_zero_when_agree() -> None:
    img_presence = torch.tensor([10.0, -10.0, 10.0])
    image_has_target = torch.tensor([True, False, True])
    loss = presence_loss(img_presence, image_has_target)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_presence_loss_high_when_inverted() -> None:
    img_presence = torch.tensor([-10.0, 10.0, -10.0])
    image_has_target = torch.tensor([True, False, True])
    loss = presence_loss(img_presence, image_has_target)
    assert loss.item() > 1.0


def _stub_outputs(b: int = 1, q: int = 4, h: int = 16) -> dict:
    return {
        "pred_logits": torch.zeros(b, q, 1),
        "pred_boxes": torch.zeros(b, q, 4),
        "pred_masks": torch.zeros(b, q, h, h),
        "presence_logit_dec": torch.zeros(b, 1),
    }


def test_total_loss_returns_all_components() -> None:
    from custom_sam_peft.config.schema import LossConfig
    from custom_sam_peft.data.base import Instance
    from custom_sam_peft.models.losses import total_loss

    raw = _stub_outputs()
    targets = [
        [
            Instance(
                mask=torch.zeros(32, 32),
                class_id=0,
                box=torch.tensor([0.5, 0.5, 0.2, 0.2]),
            )
        ]
    ]
    losses = total_loss(raw, targets, LossConfig())
    assert set(losses.keys()) == {"total", "mask", "box", "obj", "presence"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_total_loss_total_equals_weighted_sum() -> None:
    from custom_sam_peft.config.schema import LossConfig
    from custom_sam_peft.data.base import Instance
    from custom_sam_peft.models.losses import total_loss

    raw = _stub_outputs()
    targets = [
        [
            Instance(
                mask=torch.zeros(32, 32),
                class_id=0,
                box=torch.tensor([0.5, 0.5, 0.2, 0.2]),
            )
        ]
    ]
    cfg = LossConfig()
    losses = total_loss(raw, targets, cfg)
    expected = (
        cfg.w_mask * losses["mask"]
        + cfg.w_box * losses["box"]
        + cfg.w_obj * losses["obj"]
        + cfg.w_presence * losses["presence"]
    )
    assert torch.allclose(losses["total"], expected, atol=1e-6)


def test_total_loss_handles_empty_targets() -> None:
    from custom_sam_peft.config.schema import LossConfig
    from custom_sam_peft.models.losses import total_loss

    raw = _stub_outputs()
    losses = total_loss(raw, [[]], LossConfig())
    # No matches → mask + box are zero; obj + presence are still finite (no-object supervision).
    assert losses["mask"].item() == 0.0
    assert losses["box"].item() == 0.0
    assert torch.isfinite(losses["obj"])
    assert torch.isfinite(losses["presence"])
