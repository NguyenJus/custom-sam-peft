"""Mask-axis loss term classes (spec §8.1).

All classes are uniform-signature: forward(pred_logits, target) where
pred_logits and target are (N, H, W). If spatial shapes differ, pred_logits
is bilinear-upsampled to the target resolution (matches the pre-#112
mask_loss behavior).

Every class accepts the full hyperparameter pack as keyword-only kwargs and
silently ignores irrelevant ones. This keeps build_loss_bundle uniform.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate

_EPS = 1.0  # matches pre-#112 _dice_loss


def _align(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    return interpolate(
        pred[:, None],
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _dice(p: Tensor, t: Tensor) -> Tensor:
    """Soft Dice loss.

    Uses the TP-normalized form (TP + ε) / (TP + 0.5·FN + 0.5·FP + ε) so
    that Dice(p, t) == TverskyLoss(alpha=0.5)(p, t) exactly — this ensures the
    degenerate-case identity test holds (spec §8.1).
    """
    p = p.flatten(1)
    t = t.flatten(1)
    tp = (p * t).sum(-1)
    fn = ((1.0 - p) * t).sum(-1)
    fp = (p * (1.0 - t)).sum(-1)
    num = tp + _EPS
    den = tp + 0.5 * fn + 0.5 * fp + _EPS
    return (1.0 - num / den).mean()


def _focal_bce_per_pixel(logits: Tensor, targets: Tensor, gamma: float, alpha: float) -> Tensor:
    p = logits.sigmoid()
    ce = binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


class _MaskTermBase(nn.Module):
    """Accept the full hyperparameter pack and stash it; subclasses use what they need."""

    def __init__(
        self,
        *,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        tversky_alpha: float = 0.5,
        tversky_gamma: float = 1.0,
        boundary_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_gamma = float(tversky_gamma)
        self.boundary_weight = float(boundary_weight)


class BCELoss(_MaskTermBase):
    """Pixel-wise binary cross-entropy loss on mask logits."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        return binary_cross_entropy_with_logits(pred, target.float())


class DiceLoss(_MaskTermBase):
    """Soft Dice loss on mask predictions."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        return _dice(pred.sigmoid(), target.float())


class DiceBCELoss(_MaskTermBase):
    """Today's `mask_loss`: 0.5*Dice + 0.5*BCE."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        bce = binary_cross_entropy_with_logits(pred, target.float())
        dice = _dice(pred.sigmoid(), target.float())
        return 0.5 * dice + 0.5 * bce


class FocalBCELoss(_MaskTermBase):
    """Focal binary cross-entropy loss that down-weights easy mask pixels."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        return _focal_bce_per_pixel(pred, target.float(), self.focal_gamma, self.focal_alpha)


class FocalDiceLoss(_MaskTermBase):
    """Equal-weighted combination of focal BCE and soft Dice for mask predictions."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        fbce = _focal_bce_per_pixel(pred, target.float(), self.focal_gamma, self.focal_alpha)
        dice = _dice(pred.sigmoid(), target.float())
        return 0.5 * dice + 0.5 * fbce


def _tversky_index(p: Tensor, t: Tensor, alpha: float) -> Tensor:
    p = p.flatten(1)
    t = t.flatten(1)
    tp = (p * t).sum(-1)
    fn = ((1.0 - p) * t).sum(-1)
    fp = (p * (1.0 - t)).sum(-1)
    return (tp + _EPS) / (tp + alpha * fn + (1.0 - alpha) * fp + _EPS)


class TverskyLoss(_MaskTermBase):
    """Tversky loss that penalises false negatives and false positives asymmetrically."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        ti = _tversky_index(pred.sigmoid(), target.float(), self.tversky_alpha)
        return (1.0 - ti).mean()


class FocalTverskyLoss(_MaskTermBase):
    """Focal Tversky loss that applies an additional power focusing to difficult examples."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        ti = _tversky_index(pred.sigmoid(), target.float(), self.tversky_alpha)
        return ((1.0 - ti).pow(self.tversky_gamma)).mean()


def _signed_distance_transform(
    target_np: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Signed distance transform for one (H, W) uint8/bool mask.

    Positive inside the object, negative outside. Computed via scipy
    distance_transform_edt on the binary mask and its complement.
    """
    mask = target_np.astype(bool)
    if not mask.any():
        # All-zero target: distance is + everywhere outside (i.e. positive everywhere
        # outside the object, which doesn't exist); use the EDT of the complement and
        # negate so the SDT is non-positive (pushing predictions away costs nothing).
        return np.asarray(-distance_transform_edt(~mask), dtype=np.float32)
    if mask.all():
        return np.asarray(distance_transform_edt(mask), dtype=np.float32)
    pos = distance_transform_edt(mask)
    neg = distance_transform_edt(~mask)
    return np.asarray(pos - neg, dtype=np.float32)


def _kervadec_boundary(pred_sigmoid: Tensor, target: Tensor) -> Tensor:
    """Kervadec et al. 2019 boundary loss: integral of pred * SDT(target).

    SDT is computed on CPU per image (scipy), then moved to pred.device.
    Detached from autograd — gradient flows only through pred.
    """
    batch_sdts = []
    target_cpu = target.detach().to(torch.uint8).cpu().numpy()
    for i in range(target_cpu.shape[0]):
        batch_sdts.append(_signed_distance_transform(target_cpu[i]))
    sdt = torch.from_numpy(np.stack(batch_sdts)).to(pred_sigmoid.device, pred_sigmoid.dtype)
    # Normalize by spatial size so the magnitude is comparable to Dice's [0, 1] range.
    return (pred_sigmoid * sdt).mean()


class BoundaryLoss(_MaskTermBase):
    """boundary_weight * Kervadec + (1 - boundary_weight) * Dice.

    boundary_weight=0 degenerates to plain Dice; boundary_weight=1 is pure Kervadec.
    """

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        p_sig = pred.sigmoid()
        dice_term = _dice(p_sig, target.float())
        if self.boundary_weight <= 0.0:
            return dice_term
        boundary_term = _kervadec_boundary(p_sig, target)
        return self.boundary_weight * boundary_term + (1.0 - self.boundary_weight) * dice_term
