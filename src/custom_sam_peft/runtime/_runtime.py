"""Runtime value object — single source of device + dtype truth."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from custom_sam_peft.errors import ConfigError

logger = logging.getLogger(__name__)

_dtype_coercion_warned = False

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


def coerce_dtype_for_capability(
    dtype: torch.dtype,
    *,
    capability: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> torch.dtype:
    """Coerce bfloat16 -> float16 on hardware below compute capability 8.0.

    bf16 is not natively supported below CC 8.0 (Ampere), so we run those cards
    in float16. Only bfloat16 is touched; float16/float32 pass through. Emits a
    one-time warning per process when a coercion happens.

    Pass ``capability`` directly (CPU-testable) or a ``device`` to read it from.
    """
    global _dtype_coercion_warned
    if dtype is not torch.bfloat16:
        return dtype
    if capability is None:
        if device is not None and device.type == "cuda":
            capability = torch.cuda.get_device_capability(device)
        else:
            return dtype  # CPU / unknown: leave bf16 alone (autocast path handles CPU)
    if capability >= (8, 0):
        return dtype
    if not _dtype_coercion_warned:
        logger.warning(
            "Requested bfloat16 on a device with compute capability %s (< 8.0, "
            "below the CC 8.0 / Ampere floor for native bf16); coercing to float16.",
            capability,
        )
        _dtype_coercion_warned = True
    return torch.float16
