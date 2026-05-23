"""PEFT adapter package.

Documented seam: trainers, evaluators, and checkpoint code interact with
PEFT adapters through the ``PEFTMethod`` protocol below. They must not
branch on ``cfg.peft.method`` strings.

Registered factories:
  ``lookup("peft", "lora")``  → ``apply_lora``
  ``lookup("peft", "qlora")`` → ``apply_qlora``

For method-dispatch decisions (optimizer, autocast, checkpoint detection)
call the appropriate ``LoraAdapter`` or ``QloraAdapter`` instance methods
instead of testing ``cfg.peft.method``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from custom_sam_peft.errors import CheckpointError

_QLORA_META_FILENAME = "custom_sam_peft_qlora.json"


@runtime_checkable
class PEFTMethod(Protocol):
    """Protocol for PEFT adapter implementations registered via @register("peft", ...).

    Trainers, evaluators, and checkpoint code call these methods instead of
    branching on cfg.peft.method strings. Each registered adapter (lora.py,
    qlora.py) must implement this interface.
    """

    def recommended_optimizer(self) -> str:
        """Return the optimizer name to use when cfg.train.optimizer == 'auto'.

        Returns 'adamw8bit' for QLoRA (requires bitsandbytes), 'adamw' for LoRA.
        """
        ...

    def disables_outer_autocast(self) -> bool:
        """Return True if outer torch.autocast must NOT be used during training.

        QLoRA returns True: sam3's internal autocast(enabled=False) regions
        produce bf16/fp32 collisions under an outer autocast scope.
        LoRA returns False: outer autocast is safe.
        """
        ...

    def detect_method_from_checkpoint(self, adapter_dir: Path) -> str:
        """Inspect adapter_dir and return the canonical method string.

        QLoRA: checks for custom_sam_peft_qlora.json presence → returns 'qlora'.
        LoRA: absence of the JSON marker → returns 'lora'.
        Raises CheckpointError on ambiguous or corrupted state.
        """
        ...

    def supports_checkpoint_load_from_disk(self) -> bool:
        """Return True if this method can load a checkpoint from disk without
        a pre-loaded model wrapper.

        LoRA returns True. QLoRA returns False (requires a live wrapper with
        quantized base; disk-only load is deferred to a follow-up PR).
        """
        ...


class LoraAdapter:
    """PEFTMethod implementation for LoRA (plain full-precision fine-tuning)."""

    def recommended_optimizer(self) -> str:
        return "adamw"

    def disables_outer_autocast(self) -> bool:
        return False

    def detect_method_from_checkpoint(self, adapter_dir: Path) -> str:
        meta = adapter_dir / _QLORA_META_FILENAME
        if meta.exists():
            raise CheckpointError(
                f"detect_method_from_checkpoint: found {_QLORA_META_FILENAME} in "
                f"{adapter_dir!r} but LoraAdapter was used — checkpoint appears to be QLoRA"
            )
        return "lora"

    def supports_checkpoint_load_from_disk(self) -> bool:
        return True


class QloraAdapter:
    """PEFTMethod implementation for QLoRA (4-bit base + LoRA)."""

    def recommended_optimizer(self) -> str:
        return "adamw8bit"

    def disables_outer_autocast(self) -> bool:
        return True

    def detect_method_from_checkpoint(self, adapter_dir: Path) -> str:
        meta = adapter_dir / _QLORA_META_FILENAME
        if not meta.exists():
            raise CheckpointError(
                f"detect_method_from_checkpoint: {_QLORA_META_FILENAME} not found in "
                f"{adapter_dir!r} but QloraAdapter was used — checkpoint appears to be LoRA"
            )
        return "qlora"

    def supports_checkpoint_load_from_disk(self) -> bool:
        return False


def method_pretty_name(method: str) -> str:
    """Return a human-readable display name for the given peft.method string.

    Centralises the lora→"LoRA", qlora→"QLoRA" mapping inside peft_adapters/
    so no other module needs to branch on the method string for display purposes.

    Raises ValueError for unknown method strings.
    """
    if method == "lora":
        return "LoRA"
    if method == "qlora":
        return "QLoRA"
    raise ValueError(f"Unknown peft.method {method!r}; expected 'lora' or 'qlora'.")


def make_peft_method(method: str) -> PEFTMethod:
    """Return the PEFTMethod instance for the given peft.method string.

    This is the single factory that maps the string from cfg.peft.method to
    a protocol instance. Call it once during run setup (e.g. in Trainer.__init__
    or run_eval) and pass the instance through rather than passing cfg.peft.method.

    Raises ValueError for unknown method strings.
    """
    if method == "lora":
        return LoraAdapter()
    if method == "qlora":
        return QloraAdapter()
    raise ValueError(
        f"Unknown peft.method {method!r}; expected 'lora' or 'qlora'. "
        "Register additional adapters via @register('peft', '<name>') and "
        "add a branch here."
    )
