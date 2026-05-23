"""Obj-axis loss term classes (spec §8.3).

forward(obj_logits, matched_mask) where obj_logits is (B, Q) and matched_mask
is (B, Q) bool — True for queries assigned to some target.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits


class _ObjTermBase(nn.Module):
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


class BCELoss(_ObjTermBase):
    """Binary cross-entropy objectness loss over matched query assignments."""

    def forward(self, obj_logits: Tensor, matched_mask: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(obj_logits, matched_mask.float())


class FocalBCELoss(_ObjTermBase):
    """Sigmoid focal BCE — today's objectness_loss."""

    def forward(self, obj_logits: Tensor, matched_mask: Tensor) -> Tensor:
        p = obj_logits.sigmoid()
        ce = binary_cross_entropy_with_logits(
            obj_logits,
            matched_mask.float(),
            reduction="none",
        )
        p_t = p * matched_mask + (1.0 - p) * (1.0 - matched_mask.float())
        alpha_t = self.focal_alpha * matched_mask + (1.0 - self.focal_alpha) * (
            1.0 - matched_mask.float()
        )
        return (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce).mean()
