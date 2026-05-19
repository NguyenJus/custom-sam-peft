"""VRAM-tier preset patch generator.

The notebook GENERATE cell calls `pick_preset()` to derive PEFT method,
LoRA rank, batch size, grad-accum steps, gradient checkpointing, and
dtype from the current GPU's VRAM. `preset_label()` produces the matching
short string the orchestrator forwards to the bundler via env var.

Replacement plan: see logs/TODO.md / issue #36 — algorithmic derivation
will replace this table-driven helper in a future spec.
"""

from __future__ import annotations

import torch

_GB = 1024**3

_CUDA_HINT = (
    "pick_preset() requires CUDA; got cpu-only torch. "
    "In Colab: Runtime → Change runtime type → GPU. "
    "On RunPod: deploy a GPU pod."
)


def _device_total_bytes() -> int:
    return int(torch.cuda.get_device_properties(0).total_memory)


def _tier_for_gb(total_gb: float) -> str:
    if total_gb < 12.0:
        return "<12GB"
    if total_gb < 24.0:
        return "12-24GB"
    if total_gb < 48.0:
        return "24-48GB"
    return "≥48GB"


def pick_preset() -> dict[str, dict[str, object]]:
    """Return a config-patch dict keyed by the current GPU's VRAM tier.

    Raises:
        RuntimeError: torch.cuda.is_available() is False.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)

    total_gb = _device_total_bytes() / _GB
    tier = _tier_for_gb(total_gb)

    if tier == "<12GB":
        return {
            "peft": {"method": "qlora", "r": 8},
            "train": {"batch_size": 1, "grad_accum_steps": 16},
            "model": {"gradient_checkpointing": True, "dtype": "bfloat16"},
        }
    if tier == "12-24GB":
        return {
            "peft": {"method": "qlora", "r": 16},
            "train": {"batch_size": 1, "grad_accum_steps": 8},
            "model": {"gradient_checkpointing": True, "dtype": "bfloat16"},
        }
    if tier == "24-48GB":
        return {
            "peft": {"method": "lora", "r": 16},
            "train": {"batch_size": 2, "grad_accum_steps": 4},
            "model": {"gradient_checkpointing": False, "dtype": "bfloat16"},
        }
    # ≥48GB
    return {
        "peft": {"method": "lora", "r": 32},
        "train": {"batch_size": 4, "grad_accum_steps": 2},
        "model": {"gradient_checkpointing": False, "dtype": "bfloat16"},
    }


def preset_label(total_bytes: int | None = None) -> str:
    """Return a short tier label like 'auto: 12-24GB tier'.

    If `total_bytes` is None, reads device 0's total memory (requires CUDA).
    """
    if total_bytes is None:
        if not torch.cuda.is_available():
            raise RuntimeError(_CUDA_HINT)
        total_bytes = _device_total_bytes()
    return f"auto: {_tier_for_gb(total_bytes / _GB)} tier"
