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


# Real SAM 3.1 attention naming, verified against
# sam3/model/{vitdet.py,necks.py,vl_combiner.py,decoder.py,model_misc.py}.
# `meta_to_canonical` and SCOPE_TARGETS are the two single-points-of-contact
# for SAM 3.1's surface naming; if Meta renames modules, only these change.
SCOPE_TARGETS: dict[str, list[str]] = {
    # ViT vision trunk: fused qkv + output projection per block.
    "vision": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
    ],
    # Vision trunk + transformer decoder attention output projections.
    # MultiheadAttentionWrapper exposes only `out_proj` as nn.Linear; its
    # in_proj_weight/q,k,v_proj_weight are bare Parameters and not LoRA-targetable.
    "vision_decoder": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
    ],
    # Every nn.Linear in the tree. Existing intentional over-match; narrowing
    # is deferred (see TODO history in PRs #4 / #7).
    "all": [r".*"],
}


def _resolve_targets(
    base: nn.Module,
    cfg: PEFTConfig,
    linear_types: tuple[type, ...] = (nn.Linear,),
) -> list[str]:
    patterns = cfg.target_modules if cfg.target_modules is not None else SCOPE_TARGETS[cfg.scope]
    compiled = [re.compile(p) for p in patterns]
    matched: list[str] = []
    linears: list[str] = []
    for name, module in base.named_modules():
        if not isinstance(module, linear_types):
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
    """Apply a previously-saved LoRA adapter to `wrapper`; mutate in place.

    If `wrapper` already has a PeftModel (i.e. apply_lora was called before
    this, as in the normal resume-from-checkpoint flow), the saved adapter
    weights are reloaded into the existing PeftModel instead of re-wrapping.
    """
    dirpath = str(dirpath)
    if wrapper.peft_model is not None:
        # Resume path: wrapper is already LoRA-wrapped; reload weights only.
        wrapper.peft_model.load_adapter(dirpath, "default", is_trainable=True)
        return wrapper
    base = cast(nn.Module, wrapper.model.model)
    for p in base.parameters():
        p.requires_grad = False
    peft_base = PeftModel.from_pretrained(base, dirpath, is_trainable=True)
    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base
    return wrapper


def merge_lora(wrapper: Sam3Wrapper) -> Sam3Wrapper:
    """Fold LoRA deltas into the base weights and unwrap PeftModel; mutate in place.

    For QLoRA wrappers, this dequantizes the 4-bit base to compute_dtype during
    folding; the resulting module is no longer 4-bit-quantized.
    """
    if wrapper.peft_model is None:
        raise RuntimeError("merge_lora: wrapper has no PeftModel; call apply_lora first")
    merged: Any = wrapper.peft_model.merge_and_unload()
    wrapper.model.model = merged
    wrapper.peft_model = None
    return wrapper
