"""LoRA adapter for SAM 3.1 via HuggingFace peft.

Public entry points:
  apply_lora(wrapper, cfg) -> Sam3Wrapper   # freeze base, inject LoRA
  save_lora(wrapper, dirpath) -> None       # persist adapter weights
  load_lora(wrapper, dirpath) -> Sam3Wrapper  # restore from disk
  merge_lora(wrapper) -> Sam3Wrapper        # fold adapters into base

SCOPE_TARGETS maps the LoraScope literal to a list of regex patterns matched
against `base.named_modules()` paths. It is the SINGLE point that encodes
SAM 3.1's attention module naming; if Meta renames modules, only this dict
(and a few stub fixtures) needs to change.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from esam3._registry import register
from esam3.config.schema import PEFTConfig
from esam3.models.sam3 import Sam3Wrapper

logger = logging.getLogger(__name__)


SCOPE_TARGETS: dict[str, list[str]] = {
    "vision": [r"vision_encoder\..*\.attn\.(qkv|proj)$"],
    "vision_decoder": [
        r"vision_encoder\..*\.attn\.(qkv|proj)$",
        r"mask_decoder\..*\.(self_attn|cross_attn)\.(q|k|v|out)_proj$",
    ],
    # TODO(task-7): replace `r".*"` — currently matches every nn.Linear in the
    # tree (including MLP / feedforward projections that are NOT adaptation
    # targets). Pin to attention-only patterns once SCOPE_TARGETS is verified
    # against the real SAM 3.1 module names.
    "all": [r".*"],
}


def _resolve_targets(base: nn.Module, cfg: PEFTConfig) -> list[str]:
    patterns = cfg.target_modules if cfg.target_modules is not None else SCOPE_TARGETS[cfg.scope]
    compiled = [re.compile(p) for p in patterns]
    matched: list[str] = []
    linears: list[str] = []
    for name, module in base.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        linears.append(name)
        if any(c.search(name) for c in compiled):
            matched.append(name)
    if not matched:
        sample = ", ".join(linears[:50]) if linears else "<no nn.Linear modules found>"
        raise ValueError(
            f"apply_lora: no nn.Linear modules matched patterns {patterns}. "
            f"Linear modules actually present (first 50): {sample}"
        )
    return matched


@register("peft", "lora")
def apply_lora(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """Freeze SAM 3.1 base and inject LoRA adapters; mutate `wrapper` in place.

    Returns the same wrapper instance for fluent use. After return:
      * every base parameter has requires_grad=False
      * LoRA A/B matrices on matched modules have requires_grad=True
      * wrapper.peft_model is the resulting PeftModel
      * wrapper.model.model is the PeftModel-wrapped base
    """
    if wrapper.peft_model is not None:
        raise RuntimeError("LoRA already applied to this wrapper")

    base = cast(nn.Module, wrapper.model.model)
    matched_names = _resolve_targets(base, cfg)

    for p in base.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=matched_names,
        bias=cfg.bias,
        task_type=None,
    )
    peft_base = cast(PeftModel, get_peft_model(cast(Any, base), lora_cfg))

    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base

    trainable = sum(p.numel() for p in peft_base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_base.parameters())
    ratio = trainable / total if total else 0.0
    logger.info(
        "LoRA: trainable=%d (%.2f%%) of %d (scope=%s, n_targets=%d)",
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
        len(matched_names),
    )
    if ratio > 0.10:
        logger.warning(
            "LoRA trainable ratio %.2f%% exceeds 10%%; "
            "likely a misconfigured scope or target_modules.",
            100 * ratio,
        )
    return wrapper


def save_lora(wrapper: Sam3Wrapper, dirpath: str | Path) -> None:
    """Write LoRA adapter weights + LoraConfig JSON to `dirpath`."""
    if wrapper.peft_model is None:
        raise RuntimeError("save_lora: wrapper has no PeftModel; call apply_lora first")
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    wrapper.peft_model.save_pretrained(str(dirpath))


def load_lora(wrapper: Sam3Wrapper, dirpath: str | Path) -> Sam3Wrapper:
    """Apply a previously-saved LoRA adapter to `wrapper`; mutate in place."""
    if wrapper.peft_model is not None:
        raise RuntimeError("load_lora: wrapper already has a PeftModel attached")
    base = cast(nn.Module, wrapper.model.model)
    for p in base.parameters():
        p.requires_grad = False
    peft_base = PeftModel.from_pretrained(base, str(dirpath))
    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base
    return wrapper


def merge_lora(wrapper: Sam3Wrapper) -> Sam3Wrapper:
    """Fold LoRA deltas into the base weights and unwrap PeftModel; mutate in place."""
    if wrapper.peft_model is None:
        raise RuntimeError("merge_lora: wrapper has no PeftModel; call apply_lora first")
    merged: Any = wrapper.peft_model.merge_and_unload()
    wrapper.model.model = merged
    wrapper.peft_model = None
    return wrapper
