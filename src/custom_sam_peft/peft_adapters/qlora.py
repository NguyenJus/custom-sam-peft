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

custom_sam_peft_qlora.json format (v1):
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

from custom_sam_peft._registry import register
from custom_sam_peft.config.schema import Dtype, PEFTConfig, QLoRAConfig
from custom_sam_peft.models.sam3 import Sam3Wrapper
from custom_sam_peft.peft_adapters.lora import _resolve_targets

logger = logging.getLogger(__name__)


_QLORA_META_FILE = "custom_sam_peft_qlora.json"
_QLORA_META_VERSION = 1


def _import_bnb() -> Any:
    """Lazy import of bitsandbytes with a helpful ImportError on absence."""
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "QLoRA requires bitsandbytes. Install with: pip install 'custom-sam-peft[qlora]'"
        ) from e
    return bnb


def _torch_dtype(name: Dtype) -> torch.dtype:
    """Map the schema's Dtype literal to a torch.dtype."""
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _mha_exclusion_types() -> tuple[type[nn.Module], ...]:
    """Module types whose internal ``nn.Linear`` children MUST be skipped during
    4-bit quantization because their ``forward`` extracts ``out_proj.weight`` as
    a raw tensor and calls ``F.linear`` on it directly, bypassing the
    ``Linear4bit.__call__`` dispatch that bitsandbytes needs to dequantize.

    Two known types ship with the project:
      - ``torch.nn.MultiheadAttention`` (PyTorch built-in; sam3's decoder uses
        it at ``sam3/model/decoder.py:54,59``).  Bypass site:
        ``torch.nn.functional.multi_head_attention_forward`` line 6637.
      - ``sam3.model.model_misc.MultiheadAttention`` (sam3 custom; instantiated
        at ``sam3/model_builder.py:226`` as ``cross_attend_prompt`` and threaded
        into the decoder via ``cross_attention`` factory).  Bypass site:
        ``sam3.model.model_misc.multi_head_attention_forward`` line 432.

    sam3's custom class is imported lazily (try/except) so CPU unit tests can
    run without sam3 installed.  In a production load (where sam3 is always
    importable), both types are unconditionally in the exclusion set.

    Audit notes for future maintainers (re-evaluate every sam3 version bump):
      - ``sam3.sam.transformer.Attention`` is safe — its ``forward`` calls
        ``self.out_proj(out)`` via module dispatch (no bypass).
      - ``sam3.model.vitdet.Attention`` is safe — uses ``self.qkv(x)`` /
        ``self.proj(x)`` via module dispatch.
      - ``out_proj.weight`` references in ``video_tracking_multiplex.py`` and
        ``sam3_tracker_base.py`` are ``.shape[0]`` lookups, not forward
        dispatch.
    """
    types: tuple[type[nn.Module], ...] = (nn.MultiheadAttention,)
    try:
        from sam3.model.model_misc import MultiheadAttention as _Sam3CustomMHA

        types = (*types, _Sam3CustomMHA)
    except ImportError:
        # sam3 not importable (CPU unit-test environments may omit it); the
        # torch built-in alone is still excluded.
        pass
    return types


def _collect_linear_names(base: nn.Module) -> list[str]:
    """Every ``nn.Linear`` in ``base``, excluding children of MHA-style modules.

    The exclusion is mandatory.  Both ``torch.nn.MultiheadAttention`` and
    ``sam3.model.model_misc.MultiheadAttention`` implement their ``forward``
    by extracting ``out_proj.weight`` as a raw tensor and passing it to
    ``F.linear`` directly — bypassing ``Linear4bit.__call__`` and so bypassing
    bitsandbytes' dequant kernel.  Quantizing those children causes the first
    forward to raise ``RuntimeError: self and mat2 must have the same dtype,
    but got Float and Byte`` (the "Byte" is bnb's uint8-packed 4-bit storage).

    See ``_mha_exclusion_types`` for the full audit and the analogous fix in
    ``transformers.utils.bitsandbytes`` (``lm_head`` exclusion).

    Trade-off this introduces: out_proj remains ``nn.Linear`` (fp32/bf16) in
    QLoRA mode, so ``_resolve_targets(..., linear_types=(Linear4bit,))`` no
    longer matches it and LoRA is NOT injected on out_proj under QLoRA.
    In LoRA mode (no quantization) out_proj is still a plain ``nn.Linear``
    and LoRA targets it normally.  The asymmetry is the right one: MHA's
    in_proj path also stays unquantized (``in_proj_weight`` is a raw
    ``Parameter``, not a ``Linear`` submodule), so the bulk of decoder
    finetuning lives at FFN Linears outside MHA and is unaffected.
    """
    mha_types = _mha_exclusion_types()
    mha_prefixes = {name for name, mod in base.named_modules() if isinstance(mod, mha_types)}

    def _under_mha(linear_name: str) -> bool:
        return any(linear_name == p or linear_name.startswith(p + ".") for p in mha_prefixes)

    return [n for n, m in base.named_modules() if isinstance(m, nn.Linear) and not _under_mha(n)]


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


def _quantize_base(model: nn.Module, cfg: PEFTConfig) -> nn.Module:
    """Replace every eligible nn.Linear in *model* with bnb.nn.Linear4bit in place.

    Returns *model* (same object) after the in-place swap so callers can chain.
    Raises ``ValueError`` when no eligible linear modules are found.
    """
    quant_names = _collect_linear_names(model)
    if not quant_names:
        raise ValueError("apply_qlora: no nn.Linear modules found in base; cannot quantize")
    _replace_with_bnb_linear4bit(model, quant_names, cfg.qlora)
    return model


def _freeze_non_adapter(model: nn.Module) -> None:
    """Set ``requires_grad = False`` on every parameter of *model* in place.

    We do NOT call peft's ``prepare_model_for_kbit_training`` here: that
    helper upcasts every non-``Params4bit`` bf16/fp16 parameter to fp32
    (peft/utils/other.py lines 181-186) under the assumption that outer
    ``torch.autocast`` will be on at training time to handle dtype routing
    back to compute_dtype.  This codebase deliberately avoids outer autocast
    because sam3 has internal ``with torch.amp.autocast(enabled=False)``
    regions (notably ``sam3/model/decoder.py::forward_ffn``) that re-trigger
    bf16/fp32 collisions whenever an outer scope is active — see
    ``src/custom_sam_peft/models/sam3.py::_patch_pos_enc_dtype`` for the
    canonical record of that constraint (PR #13).

    Without outer autocast the upcast is fatal at every raw-Parameter forward
    site that bypasses module dispatch.  Confirmed callsites against the QLoRA
    path:
      - ``torch.nn.MultiheadAttention`` and
        ``sam3.model.model_misc.MultiheadAttention`` each call
        ``F.linear(act, in_proj_weight, ...)`` and
        ``F.linear(act, out_proj.weight, ...)`` directly.  bf16 activation x
        fp32 weight raises ``RuntimeError: mat1 and mat2 must have the same
        dtype, but got BFloat16 and Float``.
      - ``sam3.model.model_misc.LayerScale.forward`` does ``x * self.gamma``;
        promotes bf16 activation to fp32, which then collides with the next
        ``Linear4bit`` (expects compute dtype input).  Same pattern in
        ``sam3/model/memory.py:141``.

    The audit is not exhaustive across every sam3 release, so we take the
    systemic fix (skip the upcast entirely) instead of patching each site.
    The trade-off: ``nn.LayerNorm`` weights stay in compute_dtype (bf16)
    rather than fp32.  The kbit-training recipe in the QLoRA paper recommends
    fp32 LayerNorms for gradient stability on long runs; our smoke tier is a
    50-step overfit (loss converges in tens of steps) so this is fine.  Long
    production runs may want a future ``cfg.qlora.upcast_norms`` knob (out of
    scope for issue #44; track as follow-up).

    ``prepare_model_for_kbit_training`` also does base-param freezing and
    (conditionally) gradient-checkpointing setup.  Freezing is already handled
    by ``peft.get_peft_model`` (the LoraConfig path freezes base params and
    marks the new lora_A/B adapters trainable).  We do the explicit loop here
    too as belt-and-suspenders so the contract is visible at the call site
    rather than implicit in peft.  Gradient checkpointing was being passed as
    ``False`` anyway (sam3's top-level model has no ``set_grad_checkpointing``),
    so nothing is lost on that axis.
    """
    for param in model.parameters():
        param.requires_grad = False


def _inject_lora_adapters(model: nn.Module, cfg: PEFTConfig) -> nn.Module:
    """Wrap *model* in a PeftModel with LoRA adapters on its Linear4bit layers.

    Sets ``model.is_loaded_in_4bit = True`` before calling ``get_peft_model``
    so that peft 0.19's LoRA dispatcher (peft.tuners.lora.model:226) routes to
    the bnb.Linear4bit merge path (dequant→add→repack) rather than the generic
    Linear path whose ``merge()`` blindly does ``weight.data += delta`` on
    packed 4-bit storage, causing a shape-mismatch RuntimeError.

    Returns the resulting ``PeftModel`` (a new object wrapping *model*).
    """
    bnb = _import_bnb()

    from peft import LoraConfig, get_peft_model

    lora_target_names = _resolve_targets(model, cfg, linear_types=(bnb.nn.Linear4bit,))
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=lora_target_names,
        bias=cfg.bias,
        task_type=None,
    )
    model.is_loaded_in_4bit = True  # type: ignore[assignment]
    return get_peft_model(model, lora_cfg)  # type: ignore[arg-type]


@register("peft", "qlora")
def apply_qlora(wrapper: Sam3Wrapper, cfg: PEFTConfig) -> Sam3Wrapper:
    """Quantize SAM 3.1 base to 4-bit and inject LoRA adapters; mutate in place.

    After return:
      * every nn.Linear in the base has been replaced by bnb.nn.Linear4bit
      * LoRA A/B matrices on matched attention modules have requires_grad=True
      * all 4-bit base weights have requires_grad=False
      * wrapper.peft_model is the resulting PeftModel
    """
    if wrapper.peft_model is not None:
        raise RuntimeError("QLoRA already applied to this wrapper")

    from peft import PeftModel as _PeftModel

    base = cast(nn.Module, wrapper.model.model)
    _quantize_base(base, cfg)
    _freeze_non_adapter(base)
    peft_base = _inject_lora_adapters(base, cfg)

    wrapper.model.model = peft_base
    wrapper.peft_model = cast(_PeftModel, peft_base)

    trainable = sum(p.numel() for p in peft_base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_base.parameters())
    ratio = trainable / total if total else 0.0
    logger.info(
        "QLoRA: trainable=%d (%.2f%%) of %d (lora_scope=%s, quant_type=%s, compute_dtype=%s)",
        trainable,
        100 * ratio,
        total,
        cfg.scope if cfg.target_modules is None else "<override>",
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
    """Write LoRA adapter weights + custom_sam_peft_qlora.json (quant metadata) to `dirpath`."""
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

    from peft import PeftModel

    base = cast(nn.Module, wrapper.model.model)
    quant_names = _collect_linear_names(base)
    if not quant_names:
        raise ValueError("load_qlora: no nn.Linear modules found in base; cannot quantize")
    _replace_with_bnb_linear4bit(base, quant_names, qcfg)
    # See apply_qlora: we deliberately skip ``prepare_model_for_kbit_training``
    # to avoid the fp32 upcast that collides with bf16 activations at sam3's
    # raw-Parameter forward sites (MHA in_proj / out_proj, LayerScale gamma,
    # etc.).  Freeze base params explicitly to mirror what the helper did for us.
    for param in base.parameters():
        param.requires_grad = False
    # Mirror the flag set in apply_qlora so peft's LoRA dispatcher routes to
    # bnb.Linear4bit wrappers on the restored model.  Without this, load_qlora
    # would succeed but a subsequent merge_lora call would hit the same
    # packed-weight shape mismatch as described in apply_qlora above.
    base.is_loaded_in_4bit = True  # type: ignore[assignment]
    peft_base = PeftModel.from_pretrained(base, str(src))
    wrapper.model.model = peft_base
    wrapper.peft_model = peft_base
    return wrapper
