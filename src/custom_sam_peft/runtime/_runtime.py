"""Runtime value object — single source of device + dtype truth."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from custom_sam_peft.errors import ConfigError

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Runtime:
    """Carries device, dtype, and rank-awareness fields.

    `is_primary` and `world_size` are §2 seam scaffolding: they always
    have values (True / 1) today but exist so that future DDP / FSDP
    work has somewhere to plumb rank info without touching every call
    site. The §10 dead-code sweep MUST NOT remove them.
    """

    device: torch.device
    dtype: torch.dtype
    is_primary: bool = True
    world_size: int = 1

    @classmethod
    def from_config(cls, *, device: str, dtype: str) -> Runtime:
        """Resolve device/dtype strings to torch types once.

        Downstream code receives a Runtime and never re-parses these
        strings.
        """
        try:
            resolved_dtype = _DTYPE_MAP[dtype.lower()]
        except KeyError as e:
            raise ConfigError(
                f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_MAP)}",
                field_path="runtime.dtype",
                expected=f"one of {sorted(_DTYPE_MAP)}",
                found=repr(dtype),
                fix="update runtime.dtype in your config to one of the supported values",
            ) from e
        return cls(device=torch.device(device), dtype=resolved_dtype)
