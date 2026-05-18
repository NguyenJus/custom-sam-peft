"""Shared predicates used by both unit and integration QLoRA tests.

Kept here (not in `tests/integration/`) so the lightweight unit suite can
import the predicate without pulling in the integration module's heavy
top-level imports (torch, sam3, esam3.peft_adapters.qlora, etc.).
"""

from __future__ import annotations

from torch import nn

_LORA_ADAPTER_PATH_TOKENS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


def has_plain_nn_linear(module: nn.Module) -> bool:
    """True if any nn.Linear remains in the BASE tree (NOT under a LoRA adapter path).

    Subclasses of nn.Linear (e.g. bnb.nn.Linear4bit) are ignored via `type(m) is`.
    Plain nn.Linear modules whose qualified name from `named_modules()` contains
    any token in `_LORA_ADAPTER_PATH_TOKENS` are also ignored: they belong to
    LoRA's full-precision adapter, not the base.
    """
    for name, m in module.named_modules():
        if type(m) is not nn.Linear:
            continue
        if any(tok in name for tok in _LORA_ADAPTER_PATH_TOKENS):
            continue
        return True
    return False
