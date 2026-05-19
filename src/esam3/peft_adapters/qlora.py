"""QLoRA adapter for SAM 3.1: 4-bit base + LoRA via huggingface/peft.

Public entry points:
  apply_qlora(wrapper, cfg) -> Sam3Wrapper   # quantize base, inject LoRA
  save_qlora(wrapper, dirpath) -> None       # persist adapter + quant metadata
  load_qlora(wrapper, dirpath) -> Sam3Wrapper  # restore from disk

Requires the [qlora] optional extra (bitsandbytes). bitsandbytes is imported
lazily inside apply_qlora / load_qlora so LoRA-only users are unaffected.

Isolation contract: this module imports from lora.py (for _resolve_targets +
SCOPE_TARGETS) but lora.py never imports from qlora.py. lora.py never imports
bitsandbytes.

esam3_qlora.json format (v1):
  {"format_version": 1, "quant_type": "nf4", "compute_dtype": "bfloat16"}
Bump format_version whenever fields change shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from esam3._registry import register
from esam3.config.schema import Dtype, PEFTConfig, QLoRAConfig
from esam3.models.sam3 import Sam3Wrapper
from esam3.peft_adapters.lora import _resolve_targets

logger = logging.getLogger(__name__)


_QLORA_META_FILE = "esam3_qlora.json"
_QLORA_META_VERSION = 1


def _import_bnb() -> Any:
    """Lazy import of bitsandbytes with a helpful ImportError on absence."""
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "QLoRA requires bitsandbytes. Install with: "
            "pip install 'efficient-sam3-finetuning[qlora]'"
        ) from e
    return bnb


def _torch_dtype(name: Dtype) -> torch.dtype:
    """Map the schema's Dtype literal to a torch.dtype."""
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _collect_linear_names(base: nn.Module) -> list[str]:
    """Return the fully-qualified names of every nn.Linear in `base`."""
    return [n for n, m in base.named_modules() if isinstance(m, nn.Linear)]


def _resolve_parent(base: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Walk `dotted_name` to find the immediate parent module and final attr."""
    parts = dotted_name.split(".")
    parent: nn.Module = base
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _replace_with_bnb_linear4bit(base: nn.Module, names: list[str], qcfg: QLoRAConfig) -> None:
    """In-place swap: nn.Linear -> bnb.nn.Linear4bit for every name in `names`."""
    bnb = _import_bnb()
    compute_dtype = _torch_dtype(qcfg.compute_dtype)
    for name in names:
        parent, attr = _resolve_parent(base, name)
        old = cast(nn.Linear, getattr(parent, attr))
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            quant_type=qcfg.quant_type,
            compute_dtype=compute_dtype,
        )
        new.load_state_dict(old.state_dict())
        new = new.to(old.weight.device)  # quantization fires on .to(cuda)
        setattr(parent, attr, new)


@register("peft", "qlora")
def apply_qlora(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """Quantize SAM 3.1 base to 4-bit and inject LoRA adapters; mutate in place.

    After return:
      * every nn.Linear in the base has been replaced by bnb.nn.Linear4bit
      * norm layers upcast to fp32 (kbit-training recipe)
      * LoRA A/B matrices on matched attention modules have requires_grad=True
      * all 4-bit base weights have requires_grad=False
      * wrapper.peft_model is the resulting PeftModel
    """
    if wrapper.peft_model is not None:
        raise RuntimeError("QLoRA already applied to this wrapper")

    bnb = _import_bnb()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base = cast(nn.Module, wrapper.model.model)

    quant_names = _collect_linear_names(base)
    if not quant_names:
        raise ValueError("apply_qlora: no nn.Linear modules found in base; cannot quantize")

    _replace_with_bnb_linear4bit(base, quant_names, cfg.qlora)

    lora_target_names = _resolve_targets(base, cfg, linear_types=(bnb.nn.Linear4bit,))

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=lora_target_names,
        bias=cfg.bias,
        task_type=None,
    )

    base = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        base,
        use_gradient_checkpointing=getattr(base, "is_gradient_checkpointing", False),
    )
    # peft 0.19's LoRA dispatcher (peft.tuners.lora.model:226) gates on
    # `model.is_loaded_in_4bit` to route to bnb.Linear4bit wrappers whose
    # merge() already handles dequant→add→repack correctly.  apply_qlora
    # replaces Linears manually so the flag is never set; set it now before
    # get_peft_model so peft dispatches to the bnb path, not the generic
    # Linear path whose merge() blindly does `weight.data += delta` on
    # packed 4-bit storage (shape mismatch → RuntimeError).
    base.is_loaded_in_4bit = True
    peft_base = get_peft_model(base, lora_cfg)

    from peft import PeftModel as _PeftModel

    wrapper.model.model = peft_base
    wrapper.peft_model = cast(_PeftModel, peft_base)

    trainable = sum(p.numel() for p in peft_base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_base.parameters())
    ratio = trainable / total if total else 0.0
    logger.info(
        "QLoRA: %d Linears -> Linear4bit; trainable=%d (%.2f%%) of %d "
        "(lora_scope=%s, n_lora_targets=%d, quant_type=%s, compute_dtype=%s)",
        len(quant_names),
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
        len(lora_target_names),
        cfg.qlora.quant_type,
        cfg.qlora.compute_dtype,
    )
    if ratio > 0.10:
        logger.warning(
            "QLoRA trainable ratio %.2f%% exceeds 10%%; "
            "likely a misconfigured scope or target_modules.",
            100 * ratio,
        )
    return wrapper


def _infer_quant_type_from_wrapper(wrapper: Sam3Wrapper) -> str:
    """Read the quant_type from the first Linear4bit module in the wrapped base.

    In current bitsandbytes (the version installed on Colab alongside torch >= 2.4),
    `quant_type` lives on the Params4bit weight (`module.weight.quant_type`), not on
    the Linear4bit module. The legacy attribute `module.quant_type` is also checked
    as a fallback for older bnb builds the original tests were written against.
    """
    bnb = _import_bnb()
    if wrapper.peft_model is None:
        raise RuntimeError("_infer_quant_type_from_wrapper: wrapper.peft_model is None")
    for module in wrapper.peft_model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            # Primary: bnb >= the Params4bit-quant_type refactor.
            weight = getattr(module, "weight", None)
            qt = getattr(weight, "quant_type", None) if weight is not None else None
            if isinstance(qt, str):
                return qt
            # Fallback: legacy bnb where Linear4bit carried quant_type directly.
            qt_legacy = getattr(module, "quant_type", None)
            if isinstance(qt_legacy, str):
                return qt_legacy
            raise RuntimeError(
                "save_qlora: could not infer quant_type from Linear4bit module. "
                f"module repr: {module!r}; "
                f"bnb.__version__={getattr(bnb, '__version__', '<unknown>')}; "
                "expected `module.weight.quant_type` (current) or `module.quant_type` (legacy) "
                "to be a str."
            )
    raise RuntimeError(
        "save_qlora: wrapper.peft_model contains no Linear4bit modules; "
        "this should not happen after apply_qlora"
    )


def _infer_compute_dtype_from_wrapper(wrapper: Sam3Wrapper) -> str:
    """Read the compute_dtype from the first Linear4bit module in the wrapped base."""
    bnb = _import_bnb()
    if wrapper.peft_model is None:
        raise RuntimeError("_infer_compute_dtype_from_wrapper: wrapper.peft_model is None")
    for module in wrapper.peft_model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            dt = module.compute_dtype
            if dt == torch.bfloat16:
                return "bfloat16"
            if dt == torch.float16:
                return "float16"
            raise RuntimeError(
                f"save_qlora: unexpected Linear4bit.compute_dtype={dt!r}; "
                "schema supports bfloat16 | float16 only"
            )
    raise RuntimeError("save_qlora: wrapper.peft_model contains no Linear4bit modules")


def save_qlora(wrapper: Sam3Wrapper, dirpath: str | Path) -> None:
    """Write LoRA adapter weights + esam3_qlora.json (quant metadata) to `dirpath`."""
    if wrapper.peft_model is None:
        raise RuntimeError("save_qlora: wrapper has no PeftModel; call apply_qlora first")
    out = Path(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    wrapper.peft_model.save_pretrained(str(out))
    meta = {
        "format_version": _QLORA_META_VERSION,
        "quant_type": _infer_quant_type_from_wrapper(wrapper),
        "compute_dtype": _infer_compute_dtype_from_wrapper(wrapper),
    }
    (out / _QLORA_META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def load_qlora(wrapper: Sam3Wrapper, dirpath: str | Path) -> Sam3Wrapper:
    """Reconstruct a QLoRA wrapper from a saved directory; mutate in place."""
    if wrapper.peft_model is not None:
        raise RuntimeError("load_qlora: wrapper already has a PeftModel attached")

    src = Path(dirpath)
    meta_path = src / _QLORA_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"load_qlora: {_QLORA_META_FILE} not found in {src}. "
            "If this is a LoRA-only checkpoint, call load_lora instead."
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("format_version") != _QLORA_META_VERSION:
        raise ValueError(
            f"load_qlora: unsupported {_QLORA_META_FILE} format_version "
            f"{meta.get('format_version')!r}; expected {_QLORA_META_VERSION}"
        )
    qcfg = QLoRAConfig(
        quant_type=meta["quant_type"],
        compute_dtype=meta["compute_dtype"],
    )

    from peft import PeftModel, prepare_model_for_kbit_training

    base = cast(nn.Module, wrapper.model.model)
    quant_names = _collect_linear_names(base)
    if not quant_names:
        raise ValueError("load_qlora: no nn.Linear modules found in base; cannot quantize")
    _replace_with_bnb_linear4bit(base, quant_names, qcfg)
    base = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        base,
        use_gradient_checkpointing=getattr(base, "is_gradient_checkpointing", False),
    )
    # Mirror the flag set in apply_qlora so peft's LoRA dispatcher routes to
    # bnb.Linear4bit wrappers on the restored model.  Without this, load_qlora
    # would succeed but a subsequent merge_lora call would hit the same
    # packed-weight shape mismatch as described in apply_qlora above.
    base.is_loaded_in_4bit = True
    peft_base = PeftModel.from_pretrained(base, str(src))
    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base
    return wrapper
