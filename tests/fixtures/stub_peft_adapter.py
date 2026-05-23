"""Stub PEFT adapter registered via @register("peft", "stub").

Used exclusively by tests/integration/test_peft_extensibility.py to prove
the registry+protocol surface is open for extension without modifying src/.

The stub does nothing to the model (returns it untouched), satisfies the
PEFTMethod protocol, and never raises.
"""

from __future__ import annotations

from pathlib import Path

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.models.sam3 import Sam3Wrapper

# ---------------------------------------------------------------------------
# PEFTMethod protocol implementation for the stub
# ---------------------------------------------------------------------------


class StubPEFTMethod:
    """PEFTMethod implementation for the no-op stub adapter."""

    def recommended_optimizer(self) -> str:
        """Return 'adamw' — same as LoRA; stub does no quantization."""
        return "adamw"

    def disables_outer_autocast(self) -> bool:
        """Return False — stub does not modify autocast behaviour."""
        return False

    def detect_method_from_checkpoint(self, adapter_dir: Path) -> str:
        """Return 'stub' — no on-disk marker needed for a no-op adapter."""
        return "stub"

    def supports_checkpoint_load_from_disk(self) -> bool:
        """Return True — stub has no checkpoint state to load."""
        return True


# ---------------------------------------------------------------------------
# Factory function — registered side effect fires on import
# ---------------------------------------------------------------------------


@register("peft", "stub")
def apply_stub(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """No-op PEFT factory: return the wrapper untouched.

    This is the minimal valid factory signature (same as apply_lora).
    The dispatcher can call it; training proceeds with the unmodified model.
    """
    return wrapper
