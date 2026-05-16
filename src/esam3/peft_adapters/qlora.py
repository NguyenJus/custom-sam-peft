"""QLoRA adapter (4-bit base + LoRA). Implementation deferred to spec/peft-qlora.

Requires the [qlora] optional extra (bitsandbytes).
"""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.config.schema import PEFTConfig


@register("peft", "qlora")
def apply_qlora(model: Any, cfg: PEFTConfig) -> Any:
    """Quantize the base model to nf4 and wrap with LoRA, returning the wrapped module."""
    raise NotImplementedError("filled in by spec: spec/peft-qlora")
