"""15 term classes across 4 axes (mask / box / obj / presence).

This `__init__` re-exports the public class objects with axis-prefixed names
(MaskBCELoss, ObjBCELoss, …) to avoid the same-name collision between axes.
The composer in `models/losses/compose.py` imports each axis module directly
and does not rely on these aliases.
"""

from custom_sam_peft.models.losses.terms import box as _box
from custom_sam_peft.models.losses.terms import mask as _mask
from custom_sam_peft.models.losses.terms import obj as _obj
from custom_sam_peft.models.losses.terms import presence as _presence

# Mask axis (8)
MaskBCELoss = _mask.BCELoss
MaskDiceLoss = _mask.DiceLoss
MaskDiceBCELoss = _mask.DiceBCELoss
MaskFocalBCELoss = _mask.FocalBCELoss
MaskFocalDiceLoss = _mask.FocalDiceLoss
MaskTverskyLoss = _mask.TverskyLoss
MaskFocalTverskyLoss = _mask.FocalTverskyLoss
MaskBoundaryLoss = _mask.BoundaryLoss

# Box axis (3)
BoxL1GIoULoss = _box.L1GIoULoss
BoxGIoUOnlyLoss = _box.GIoUOnlyLoss
BoxCIoULoss = _box.CIoULoss

# Obj axis (2)
ObjBCELoss = _obj.BCELoss
ObjFocalBCELoss = _obj.FocalBCELoss

# Presence axis (2)
PresenceBCELoss = _presence.BCELoss
PresenceFocalBCELoss = _presence.FocalBCELoss

__all__ = [
    "BoxCIoULoss",
    "BoxGIoUOnlyLoss",
    "BoxL1GIoULoss",
    "MaskBCELoss",
    "MaskBoundaryLoss",
    "MaskDiceBCELoss",
    "MaskDiceLoss",
    "MaskFocalBCELoss",
    "MaskFocalDiceLoss",
    "MaskFocalTverskyLoss",
    "MaskTverskyLoss",
    "ObjBCELoss",
    "ObjFocalBCELoss",
    "PresenceBCELoss",
    "PresenceFocalBCELoss",
]
