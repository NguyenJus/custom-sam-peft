"""Adapter detection and load dispatch for csp predict.

Public API:
  AdapterKind            — Literal["lora", "qlora"]
  detect_adapter_kind(checkpoint_dir) -> AdapterKind
  load_adapter(model, checkpoint_dir, kind) -> nn.Module
  maybe_merge_adapter(model, *, merge) -> nn.Module
  read_adapter_base_model_name(checkpoint_dir) -> str | None

All peft_adapters imports are lazy (inside functions) so that the
base-model-only hot path never imports peft_adapters (spec §2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import typer
from torch import nn

logger = logging.getLogger(__name__)

AdapterKind = Literal["lora", "qlora"]

_QLORA_SENTINEL = "custom_sam_peft_qlora.json"
_LORA_CONFIG = "adapter_config.json"


def detect_adapter_kind(checkpoint_dir: Path) -> AdapterKind:
    """Return "qlora" if the QLoRA sentinel file is present, else "lora".

    Raises typer.BadParameter if adapter_config.json is also absent (i.e.
    the directory does not look like any known adapter checkpoint).
    """
    if (checkpoint_dir / _QLORA_SENTINEL).is_file():
        return "qlora"
    if (checkpoint_dir / _LORA_CONFIG).is_file():
        return "lora"
    raise typer.BadParameter(
        f"--checkpoint must contain adapter_config.json (checked: {checkpoint_dir})"
    )


def load_adapter(model: nn.Module, checkpoint_dir: Path, kind: AdapterKind) -> nn.Module:
    """Dispatch to the correct adapter loader based on *kind*.

    peft_adapters is imported lazily to keep it off the base-model hot path.
    The peft_adapters API expects Sam3Wrapper but we accept the wider nn.Module
    type here so callers don't need to import Sam3Wrapper.
    """
    if kind == "qlora":
        from custom_sam_peft.peft_adapters import qlora as _qlora

        return _qlora.load_qlora(model, checkpoint_dir)  # type: ignore[arg-type]
    else:
        from custom_sam_peft.peft_adapters import lora as _lora

        return _lora.load_lora(model, checkpoint_dir)  # type: ignore[arg-type]


def maybe_merge_adapter(model: nn.Module, *, merge: bool) -> nn.Module:
    """Merge LoRA deltas into the base model if *merge* is True.

    When merge=True on a QLoRA model, the merge is NOT auto-disabled — the
    caller (user) opts in explicitly (spec §6: "the user makes the call").
    """
    if not merge:
        return model
    from custom_sam_peft.peft_adapters import lora as _lora

    return _lora.merge_lora(model)  # type: ignore[arg-type]


def read_adapter_base_model_name(checkpoint_dir: Path) -> str | None:
    """Read base_model_name_or_path from adapter_config.json, or return None.

    Returns None if the file is absent or the key is missing.
    """
    config_path = checkpoint_dir / _LORA_CONFIG
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as fh:
        data: dict[str, object] = json.load(fh)
    value = data.get("base_model_name_or_path")
    return str(value) if value is not None else None
