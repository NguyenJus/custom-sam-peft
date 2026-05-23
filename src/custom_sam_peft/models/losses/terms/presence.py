"""Presence-axis loss term classes (spec §8.4).

forward(img_presence, image_has_target) where img_presence is (B,) and
image_has_target is (B,) bool.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits


class _PresenceTermBase(nn.Module):
    def __init__(
        self,
        *,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **_unused: float,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)


class BCELoss(_PresenceTermBase):
    """Today's presence_loss."""

    def forward(self, img_presence: Tensor, image_has_target: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(img_presence, image_has_target.float())


class FocalBCELoss(_PresenceTermBase):
    """Focal BCE presence loss that down-weights easy image-level predictions."""

    def forward(self, img_presence: Tensor, image_has_target: Tensor) -> Tensor:
        p = img_presence.sigmoid()
        t = image_has_target.float()
        ce = binary_cross_entropy_with_logits(img_presence, t, reduction="none")
        p_t = p * t + (1.0 - p) * (1.0 - t)
        alpha_t = self.focal_alpha * t + (1.0 - self.focal_alpha) * (1.0 - t)
        return (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce).mean()
