"""LoRA adapter for SAM 3.1 via HuggingFace peft.

Public entry points:
  apply_lora(wrapper, cfg) -> Sam3Wrapper   # freeze base, inject LoRA
  save_lora(wrapper, dirpath) -> None       # persist adapter weights
  load_lora(wrapper, dirpath) -> Sam3Wrapper  # restore from disk
  merge_lora(wrapper) -> Sam3Wrapper        # fold adapters into base

SCOPE_TARGETS and SCOPE_MHA_MODULES together encode SAM 3.1's attention naming;
if Meta renames modules, only these dicts (and a few stub fixtures) need to change.
SCOPE_TARGETS maps the LoraScope literal to a list of regex patterns matched
against nn.Linear modules in `base.named_modules()`. SCOPE_MHA_MODULES maps the
LoraScope literal to a list of regex patterns matched against nn.MultiheadAttention
modules, whose names are unioned into target_modules so peft dispatches them to its
lora.MultiheadAttention layer (adapting both in_proj_weight and out_proj, with dropout).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import PEFTConfig
from custom_sam_peft.models.sam3 import Sam3Wrapper

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
    # Vision trunk + transformer decoder attention projections + decoder FFN linears.
    # MultiheadAttentionWrapper exposes only `out_proj` as nn.Linear; its
    # in_proj_weight/q,k,v_proj_weight are bare Parameters and not LoRA-targetable.
    # NOTE: The out_proj pattern only takes effect under plain LoRA. Under QLoRA,
    # _mha_exclusion_types in qlora.py keeps MHA children (including out_proj) as
    # nn.Linear rather than quantizing them, so _resolve_targets(...,
    # linear_types=(Linear4bit,)) returns zero out_proj matches.
    # The linear[12] pattern matches in BOTH modes: TransformerDecoderLayer.linear1
    # and .linear2 (sam3.model.decoder.TransformerDecoderLayer:64,67) are bare
    # nn.Linear FFN modules outside any MHA, so they get quantized under QLoRA and
    # remain as nn.Linear under LoRA — targetable in either case.
    "vision_decoder": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
        r"transformer\.decoder\.layers\.\d+\.linear[12]$",
    ],
    # vision_decoder's generic-module set MINUS the self_attn/ca_text out_proj
    # alternatives (peft's lora.MultiheadAttention adapts those out_proj internally when
    # the MHA module is targeted via SCOPE_MHA_MODULES; double-targeting must be avoided).
    # cross_attn is a RoPEAttention (genuine nn.Linear out_proj), so it stays generic.
    # See spec #230 §4.2, §5.1.
    "vision_decoder_concept": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer\.decoder\.layers\.\d+\.cross_attn\.out_proj$",
        r"transformer\.decoder\.layers\.\d+\.linear[12]$",
    ],
    # vision_decoder_concept MINUS the ViT trunk pattern — trunk stays frozen
    # (no LoRA adapters, all trunk base params keep requires_grad=False; autograd
    # skips the trunk subgraph automatically). New default scope (schema.py).
    # QLoRA note: cross_attn.out_proj is a genuine nn.Linear on a RoPEAttention
    # (not an nn.MultiheadAttention wrapper), so it stays targetable in both modes.
    # linear1/linear2 are bare nn.Linear FFN modules and quantize under QLoRA.
    # self_attn/ca_text MHA modules are adapted via SCOPE_MHA_MODULES exactly as
    # in vision_decoder_concept. QLoRA behavior is therefore identical to
    # vision_decoder_concept minus the trunk.
    "decoder_concept": [
        r"transformer\.decoder\.layers\.\d+\.cross_attn\.out_proj$",
        r"transformer\.decoder\.layers\.\d+\.linear[12]$",
    ],
    # Every nn.Linear in the tree. Existing intentional over-match; narrowing
    # is deferred (see TODO history in PRs #4 / #7).
    "all": [r".*"],
}

# Parallel to SCOPE_TARGETS: scope -> regexes matched against nn.MultiheadAttention
# modules in named_modules(). Naming an MHA module makes peft dispatch it to its
# lora.MultiheadAttention layer, which adapts BOTH in_proj_weight and out_proj (with
# dropout support). Only the concept scope populates it; absent scopes carry no MHA
# targets (reproducibility for vision/vision_decoder/all). This is the second
# single-point-of-contact for SAM 3.1 surface naming alongside SCOPE_TARGETS.
SCOPE_MHA_MODULES: dict[str, list[str]] = {
    "vision_decoder_concept": [
        r"transformer\.decoder\.layers\.\d+\.ca_text$",
        r"transformer\.decoder\.layers\.\d+\.self_attn$",
    ],
    # decoder_concept: identical to vision_decoder_concept — trunk carries no MHA
    # targets regardless, so the MHA surface is unchanged between the two scopes.
    "decoder_concept": [
        r"transformer\.decoder\.layers\.\d+\.ca_text$",
        r"transformer\.decoder\.layers\.\d+\.self_attn$",
    ],
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
        type_label = "/".join(t.__name__ for t in linear_types)
        sample = ", ".join(linears[:50]) if linears else f"<no {type_label} modules found>"
        raise ValueError(
            f"apply_lora: no {type_label} modules matched patterns {patterns}. "
            f"{type_label} modules actually present (first 50): {sample}"
        )
    return matched


def _resolve_mha_modules(base: nn.Module, cfg: PEFTConfig) -> list[str]:
    """Resolve scope MHA-module patterns against the nn.MultiheadAttention modules.

    Precedence mirrors _resolve_targets:
      * cfg.target_modules is not None -> return [] (the user's explicit module
        override owns the module axis; the scope's MHA patterns do not apply).
      * else -> SCOPE_MHA_MODULES.get(cfg.scope, []) matched against the
        nn.MultiheadAttention modules in base.named_modules().

    Returns the full matched MHA module names (e.g.
    'transformer.decoder.layers.0.ca_text') to union into target_modules so peft
    dispatches them to lora.MultiheadAttention (adapting in_proj_weight + out_proj).
    Returns [] when the resolved pattern list is empty (legacy scopes) or when
    target_modules is overridden -- NOT an error. Raises ValueError only when a
    NON-EMPTY pattern list matches zero MHA modules (a typo or SAM rename), mirroring
    _resolve_targets' no-match error so the in_proj surface never silently trains
    nothing.
    """
    if cfg.target_modules is not None:
        return []
    patterns = SCOPE_MHA_MODULES.get(cfg.scope, [])
    if not patterns:
        return []
    compiled = [re.compile(p) for p in patterns]
    mha_names = [
        name for name, module in base.named_modules() if isinstance(module, nn.MultiheadAttention)
    ]
    matched = [name for name in mha_names if any(c.search(name) for c in compiled)]
    if not matched:
        no_mha = "<no nn.MultiheadAttention modules found>"
        sample = ", ".join(mha_names[:50]) if mha_names else no_mha
        raise ValueError(
            f"apply_lora: no nn.MultiheadAttention modules matched SCOPE_MHA_MODULES "
            f"patterns {patterns}. MHA modules actually present (first 50): {sample}"
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
    mha_names = _resolve_mha_modules(base, cfg)
    target_modules = matched_names + [n for n in mha_names if n not in matched_names]

    for p in base.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=target_modules,
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
        "LoRA: trainable=%d (%.2f%%) of %d (scope=%s, n_targets=%d, n_mha_targets=%d)",
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
        len(matched_names),
        len(mha_names),
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
