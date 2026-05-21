"""Shared predicates used by both unit and integration QLoRA tests.

Kept here (not in `tests/integration/`) so the lightweight unit suite can
import the predicate without pulling in the integration module's heavy
top-level imports (torch, sam3, custom_sam_peft.peft_adapters.qlora, etc.).
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


def _mha_exclusion_types() -> tuple[type[nn.Module], ...]:
    """Mirror of ``custom_sam_peft.peft_adapters.qlora._mha_exclusion_types``.

    Duplicated here (not imported from qlora.py) so this module stays
    importable in CPU unit environments that don't have sam3/bitsandbytes.
    The torch built-in MHA is always available; sam3's custom class is
    imported lazily with try/except.
    """
    types: tuple[type[nn.Module], ...] = (nn.MultiheadAttention,)
    try:
        from sam3.model.model_misc import MultiheadAttention as _Sam3CustomMHA

        types = (*types, _Sam3CustomMHA)
    except ImportError:
        pass
    return types


def has_plain_nn_linear(module: nn.Module) -> bool:
    """True if any nn.Linear remains in the BASE tree (NOT under a LoRA adapter
    path AND NOT under an MHA-style module that legitimately retains plain
    ``nn.Linear`` children after ``apply_qlora``).

    Subclasses of nn.Linear (e.g. bnb.nn.Linear4bit) are ignored via ``type(m) is``.

    Two categories of ``nn.Linear`` are excluded from the "plain Linear present"
    check:

    1. **LoRA adapter paths** — qualified names containing any token in
       ``_LORA_ADAPTER_PATH_TOKENS`` (e.g. ``lora_A``, ``lora_B``).  These are
       the full-precision LoRA adapters injected by PEFT and intentionally remain
       as plain ``nn.Linear``.

    2. **MHA descendants** — ``nn.Linear`` modules whose ancestor in
       ``named_modules()`` is an instance of any type returned by
       ``_mha_exclusion_types()``.  ``apply_qlora`` deliberately skips these
       because both ``torch.nn.MultiheadAttention`` and
       ``sam3.model.model_misc.MultiheadAttention`` access their internal
       ``out_proj.weight`` as a raw tensor via ``F.linear``, bypassing
       ``Linear4bit.__call__``.  Quantizing them would cause a dtype mismatch
       on the first forward pass.
    """
    mha_types = _mha_exclusion_types()

    # Collect the qualified-name prefixes of every MHA-typed module so we can
    # test whether a given Linear's name falls under one of them.  We walk
    # named_modules() once here and once again below; two passes keep the logic
    # readable without needing parent-pointer attributes (nn.Module lacks those).
    mha_prefixes: list[str] = []
    for name, m in module.named_modules():
        # Append a trailing "." so that a sibling named "<name>x" is not
        # accidentally matched.  An empty name means the root module itself
        # is an MHA — all of its children are descendants, so we skip the
        # prefix check entirely for that case (handled in the loop below).
        if isinstance(m, mha_types) and name:
            mha_prefixes.append(name + ".")

    for name, m in module.named_modules():
        if type(m) is not nn.Linear:
            continue
        if any(tok in name for tok in _LORA_ADAPTER_PATH_TOKENS):
            continue
        if any(name.startswith(pfx) for pfx in mha_prefixes):
            continue
        return True
    return False
