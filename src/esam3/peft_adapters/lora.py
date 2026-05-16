"""LoRA adapter via huggingface/peft. Implementation deferred to spec/peft-lora."""

from __future__ import annotations

from typing import Any

from esam3._registry import register
from esam3.config.schema import PEFTConfig


@register("peft", "lora")
def apply_lora(model: Any, cfg: PEFTConfig) -> Any:
    """Wrap `model` with a LoRA PeftModel, returning the wrapped module."""
    raise NotImplementedError("filled in by spec: spec/peft-lora")
