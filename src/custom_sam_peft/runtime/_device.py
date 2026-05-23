"""Single device-move helper. The data collator is the ONLY caller."""

from __future__ import annotations

from typing import Any

import torch

from custom_sam_peft.runtime._runtime import Runtime


def to_device(obj: Any, runtime: Runtime) -> Any:
    """Recursively move tensors in `obj` onto `runtime.device`.

    The §9.2 static guard test enforces that this is the only place
    `.to(device)` runs outside the runtime/ module itself.
    """
    if torch.is_tensor(obj):
        return obj.to(runtime.device)
    if isinstance(obj, dict):
        return {k: to_device(v, runtime) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, runtime) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj
