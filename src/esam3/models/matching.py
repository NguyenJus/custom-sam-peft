"""Adapter + Hungarian matcher for SAM 3.1 training.

`meta_to_canonical` is the SINGLE point in the codebase that knows Meta's
native output dict key names. If Meta renames a field, only this function
breaks. Filled in by Task 5 once the actual key names are inspected against
a real `Sam3Wrapper` forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CanonicalOutputs:
    """Output of `meta_to_canonical`. Used by the matcher and losses.

    Shapes:
      class_logits: (B, Q, C+1)   # last index = "no-object"
      pred_boxes:   (B, Q, 4)     # normalized cx,cy,w,h in [0, 1]
      pred_masks:   (B, Q, 288, 288)
      presence:     (B, Q)        # objectness logit
    """

    class_logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    presence: Tensor


def meta_to_canonical(outputs: dict) -> CanonicalOutputs:
    """Convert Meta sam3's native output dict to CanonicalOutputs.

    Implementation deferred to Task 5 (requires inspection of real Meta output).
    """
    raise NotImplementedError("filled in by Task 5 of spec/model-loading")
