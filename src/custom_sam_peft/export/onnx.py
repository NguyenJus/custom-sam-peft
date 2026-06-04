"""ONNX export orchestrator + tracers + sidecar writers (spec §5, §6).

``run_export_onnx`` merges the trained adapter (mandatory; ONNX cannot represent
LoRA/QLoRA deltas), traces SAM 3.1 into a two-file SAM-family bundle
(``image_encoder.onnx`` + ``decoder.onnx``) with load-bearing JSON/txt sidecars,
and optionally fp16-casts. The whole bundle is staged to a sibling temp dir and
promoted only on success.

Tracing uses ``torch.onnx.export(..., dynamo=False)`` (the legacy TorchScript
exporter) per the locked decision (spec §2): tracing, not dynamo, for v1.

The ``--check`` torch-vs-ORT parity verification (spec §7) composes the ORT
bundle exactly as ``csp predict --use-onnx`` wires it (reusing
``predict._OrtCore``) and compares it against the merged torch
``Sam3Wrapper.forward`` on synthetic input; on drift it raises
``ExportParityError`` and leaves NO promoted bundle.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

import custom_sam_peft
from custom_sam_peft._provenance import git_sha
from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.models._multiplex import multiplex_index_arrays
from custom_sam_peft.models.sam3 import (
    SAM3_IMAGE_SIZE,
    Sam3Wrapper,
    load_sam31,
)
from custom_sam_peft.peft_adapters import discover_method_from_checkpoint
from custom_sam_peft.train.checkpoint import _hash_cfg, load_adapter

if TYPE_CHECKING:
    from custom_sam_peft.config.schema import TrainConfig

logger = logging.getLogger(__name__)

# --- Bundle file names (spec §5) ---
ENCODER_FILE = "image_encoder.onnx"
DECODER_FILE = "decoder.onnx"
PREPROCESSOR_FILE = "preprocessor.json"
PROMPTS_FILE = "prompts.txt"
MODEL_CARD_FILE = "model_card.json"

PREPROCESSOR_SCHEMA_VERSION = 1
MODEL_CARD_SCHEMA_VERSION = 1

# Parity-check contract (consumed by a later --check phase; spec §7).
_PARITY_KEYS = ("pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec")
_PARITY_TOL = {"fp32": (1e-3, 1e-3), "fp16": (1e-2, 1e-2)}  # (atol, rtol)

# Trace batch size: B=2 genuinely exercises the dynamic batch axis (B=1 would
# silently pass a batch-hardcoded graph). Spec §5.2/§5.3.
_TRACE_B = 2


def run_export_onnx(
    cfg: TrainConfig,
    checkpoint: Path,
    *,
    output: Path,
    opset: int,
    fp16: bool,
    include: str,  # "encoder" | "decoder" | "all"
    dynamic_axes: bool,
    check: bool,
) -> Path:
    """Merge adapter, trace SAM 3.1 into a two-file ONNX bundle + sidecars at ``output``.

    Returns the bundle directory. Raises on QLoRA+fp16-off, missing CUDA for
    QLoRA, or (in a later phase) parity drift (``--check``).
    """
    output = Path(output)
    staging = output.with_name(output.name + ".tmp-onnx")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        wrapper, method, export_dtype = _merge_and_cast(cfg, checkpoint, fp16=fp16)
        del method  # method drives the merge guards only; not needed downstream.
        class_names = _resolve_class_names(cfg)

        adapter = wrapper.model
        merged = cast(nn.Module, adapter.model)
        channel_adapter = cast("nn.Module | None", adapter.channel_adapter)

        if include in ("encoder", "all"):
            _trace_encoder(
                merged,
                channel_adapter,
                staging / ENCODER_FILE,
                channels=cfg.data.channels,
                opset=opset,
                export_dtype=export_dtype,
                dynamic_axes=dynamic_axes,
            )
        if include in ("decoder", "all"):
            _trace_decoder(
                merged,
                staging / DECODER_FILE,
                class_names=class_names,
                channels=cfg.data.channels,
                channel_adapter=channel_adapter,
                opset=opset,
                export_dtype=export_dtype,
                dynamic_axes=dynamic_axes,
            )

        _write_preprocessor(staging, cfg)
        _write_prompts(staging, class_names)

        # Parity is verified against the STAGING bundle BEFORE model_card.json is
        # written (so a drift abort never leaves a card claiming parity held) and
        # BEFORE promotion (so a failed --check leaves no loadable bundle). On
        # drift _run_parity_check raises ExportParityError; the except clause
        # below removes staging and re-raises (spec §7).
        parity_checked = False
        if check:
            # _OrtCore reads include from model_card.json; write a provisional card
            # so the composed bundle wires identically to predict-side load. It is
            # fully overwritten by the LAST _write_model_card below (real
            # parity_checked value), so a drift abort leaves no parity-true card.
            _write_model_card(
                staging,
                cfg,
                opset=opset,
                fp16=fp16,
                include=include,
                dynamic_axes=dynamic_axes,
                parity_checked=False,
            )
            _run_parity_check(staging, wrapper, cfg, fp16=fp16, include=include)
            parity_checked = True

        _write_model_card(  # written LAST: its presence signals a complete bundle.
            staging,
            cfg,
            opset=opset,
            fp16=fp16,
            include=include,
            dynamic_axes=dynamic_axes,
            parity_checked=parity_checked,
        )

        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(staging), str(output))
        return output
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _merge_and_cast(
    cfg: TrainConfig,
    checkpoint: Path,
    *,
    fp16: bool,
) -> tuple[Sam3Wrapper, str, torch.dtype]:
    """Load adapter, merge deltas (mandatory), cast to export precision.

    Returns ``(wrapper, method, export_dtype)``. ``method`` is ``"lora"`` or
    ``"qlora"``; ``export_dtype`` is ``torch.float16`` or ``torch.float32``. The
    wrapper is ``.eval()``, requires_grad=False, single uniform dtype.
    """
    from custom_sam_peft.peft_adapters.lora import merge_lora

    method = discover_method_from_checkpoint(checkpoint)

    if method == "qlora" and not fp16:
        raise ValueError(
            "QLoRA adapters dequantize to fp16/bf16; exporting them to fp32 ONNX "
            "(--fp16 off) upcasts the full merged model and can OOM on memory-tight "
            "machines. Fix: re-run with --fp16."
        )
    if method == "qlora" and not torch.cuda.is_available():
        raise RuntimeError(
            "ONNX export of a QLoRA adapter requires a CUDA device (4-bit dequantize "
            "runs on GPU). No CUDA device is visible. Fix: export on a GPU machine, "
            "or re-train/save a LoRA (non-quantized) adapter."
        )

    device = torch.device("cuda" if (method == "qlora" or torch.cuda.is_available()) else "cpu")
    wrapper = load_sam31(
        cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
    )
    wrapper = wrapper.to(device)
    wrapper = load_adapter(wrapper, checkpoint)
    merge_lora(wrapper)
    adapter = wrapper.model
    merged = cast(nn.Module, adapter.model)

    # --- RoPE complex-op swap (spec §Hook points in _merge_and_cast, issue #279) ---
    # All four steps run BEFORE the fp16/fp32 cast so freqs_cis is still complex64
    # and both equivalence guards compare on fp32 buffers.

    # Step 1: VE-RoPE detection — raises VeRopeUnsupportedError if any module uses
    # use_ve_rope=True (no real-valued equivalent; spec §Error handling).
    # The scan is folded into _patch_encoder_rope_for_export (single pass over modules,
    # checked and raised before any mutation; per spec §Implementation note).

    # Step 2: Whole-encoder pre-patch reference capture.
    # Deterministic synthetic image: raw floats, SAM3_IMAGE_SIZE, cfg.data.channels,
    # batch _TRACE_B (reuses _run_parity_check input conventions; spec §Whole-encoder guard).
    _rope_gen = torch.Generator(device=next(merged.parameters()).device).manual_seed(0)
    _rope_img = torch.rand(
        _TRACE_B,
        cfg.data.channels,
        SAM3_IMAGE_SIZE,
        SAM3_IMAGE_SIZE,
        generator=_rope_gen,
        device=next(merged.parameters()).device,
    )
    with torch.no_grad():
        encoder_ref_pre_patch = _torch_encoder_feats(wrapper, _rope_img)

    # Step 3: Perform the swap + per-module equivalence guards.
    _n_rope_patched = _patch_encoder_rope_for_export(merged)

    # Step 4: Whole-encoder post-patch guard (belt-and-suspenders).
    # Compares against the pre-patch reference at the fp32 _PARITY_TOL band.
    # This guard always runs pre-cast on fp32 buffers regardless of --fp16 flag
    # (spec §Whole-encoder guard).
    with torch.no_grad():
        encoder_out_post_patch = _torch_encoder_feats(wrapper, _rope_img)
    _rope_atol, _rope_rtol = _PARITY_TOL["fp32"]  # fp32 band; spec §Whole-encoder guard
    for _key in sorted(encoder_ref_pre_patch):
        _ref_t = encoder_ref_pre_patch[_key]
        _out_t = encoder_out_post_patch[_key]
        if not torch.allclose(_out_t, _ref_t, atol=_rope_atol, rtol=_rope_rtol):
            _max_delta = float((_out_t - _ref_t).abs().max())
            raise RopeEquivalenceError(
                f"Whole-encoder RoPE equivalence guard FAILED on key {_key!r}. "
                f"max|Δ|={_max_delta:.6g} exceeds fp32 band "
                f"(atol={_rope_atol}, rtol={_rope_rtol}). "
                "Aborting export."
            )
    # --- end RoPE swap ---

    if fp16:
        merged.half()
        export_dtype = torch.float16
    else:
        merged.float()
        export_dtype = torch.float32
    merged.eval()
    for p in merged.parameters():
        p.requires_grad_(False)
    if method == "qlora":
        adapter.model = merged.to("cpu")
        channel_adapter = cast("nn.Module | None", adapter.channel_adapter)
        if channel_adapter is not None:
            adapter.channel_adapter = channel_adapter.to("cpu").to(export_dtype)
    return wrapper, method, export_dtype


def _resolve_class_names(cfg: TrainConfig) -> list[str]:
    """Build the train dataset and read its class_names; raise if empty."""
    from custom_sam_peft.train.runner import _build_dataset

    dataset = _build_dataset(cfg, "train")
    class_names = list(getattr(dataset, "class_names", []))
    if not class_names:
        raise RuntimeError(
            "dataset has no class_names; cannot bake the ONNX decoder's text "
            "embedding or write prompts.txt. Check cfg.data.train."
        )
    return class_names


# ---------------------------------------------------------------------------
# --check parity (spec §7)
# ---------------------------------------------------------------------------


class ExportParityError(RuntimeError):
    """torch-vs-ORT parity failed; the ONNX bundle was NOT promoted."""


class VeRopeUnsupportedError(RuntimeError):
    """A vitdet Attention module uses VisionRotaryEmbeddingVE (use_ve_rope=True).

    This variant has no real-valued equivalent and cannot be lowered to ONNX.
    It is a separate, harder blocker than the standard complex-RoPE path.
    The SAM 3.1 default is use_ve_rope=False (encoder uses standard RoPE).
    We refuse to silently leave an untraceable op rather than skip it.
    Export is aborted before any trace or cast.
    """


class RopeEquivalenceError(RuntimeError):
    """The original-vs-real RoPE equivalence guard failed.

    Raised by either the per-module guard (atol/rtol 1e-5; spec §Per-module guard)
    or the whole-encoder guard (fp32 _PARITY_TOL band; spec §Whole-encoder guard).
    Export is aborted before any trace or cast.
    """


def _patch_encoder_rope_for_export(merged: nn.Module) -> int:
    """Swap every encoder Attention module from complex RoPE to the real-valued path.

    Walks ``merged.modules()`` and patches every ``sam3.model.vitdet.Attention``
    instance that uses complex RoPE (``use_rope=True, use_rope_real=False,
    use_ve_rope=False``) by:

    1. Snapshotting the current ``freqs_cis`` (complex64) for the per-module guard.
    2. Setting ``use_rope_real=True``.
    3. Calling ``module._setup_rope_freqs()`` — sam3's own code re-registers
       ``freqs_cis`` and additionally registers ``freqs_cis_real`` / ``freqs_cis_imag``
       (cosθ / sinθ buffers).  No learned weights are touched.
    4. Running a per-module equivalence guard at tight tolerance (atol/rtol 1e-5;
       spec §Per-module guard — spike showed bit-exact at fp32).

    Raises ``VeRopeUnsupportedError`` (before any mutation) if any Attention module
    uses ``use_ve_rope=True`` — that variant has no real-valued equivalent.

    Raises ``RopeEquivalenceError`` if the per-module guard fails.

    The leftover complex ``freqs_cis`` buffer is left in place (inert after the swap;
    survives ``.half()``; ``_apply_rope`` asserts non-None so it must NOT be None).

    Returns the number of modules patched.  No count>0 assertion — the TinySam3Stub
    and any decoder-only model legitimately yield 0.
    """
    # Lazy import: keeps sam3 out of module-level scope (consistent with the
    # existing lazy-import pattern in this file, e.g. merge_lora imported inside
    # _merge_and_cast).
    from sam3.model.vitdet import Attention
    from sam3.sam.rope import apply_rotary_enc, apply_rotary_enc_real

    # --- First pass: fail loud on VE-RoPE before any mutation ---
    for module in merged.modules():
        if isinstance(module, Attention) and module.use_rope and module.use_ve_rope:
            raise VeRopeUnsupportedError(
                f"Module {module!r} uses VisionRotaryEmbeddingVE (use_ve_rope=True). "
                "This variant has no real-valued equivalent and cannot be lowered to "
                "ONNX. It is a separate, harder blocker than the standard complex-RoPE "
                "path. The SAM 3.1 default is use_ve_rope=False (encoder uses standard "
                "RoPE). Aborting export."
            )

    n = 0
    for module in merged.modules():
        if not isinstance(module, Attention):
            continue
        if not (module.use_rope and not module.use_rope_real and not module.use_ve_rope):
            continue

        # Step 1: snapshot the complex freqs_cis BEFORE any mutation.
        # Used by the per-module guard to compare original-vs-real paths.
        freqs_cis_snapshot = module.freqs_cis.detach().clone()  # complex64

        # Step 2 + 3: flip the flag and regenerate the RoPE table via sam3's own code.
        # _setup_rope_freqs re-registers freqs_cis AND registers freqs_cis_real /
        # freqs_cis_imag (vitdet.py:549-552).  No learned weights are touched.
        module.use_rope_real = True
        module._setup_rope_freqs()

        # Step 4: per-module equivalence guard.
        # Build deterministic q, k in the module's head shape (B, H, L, head_dim).
        # Spec §Per-module guard: use (B=1, H=num_heads, L=input_size[0]*input_size[1],
        # head_dim=head_dim); deterministic generator.
        if module.input_size is None:
            raise RuntimeError(
                f"Attention module {module!r} has use_rope=True but input_size is None; "
                "cannot build the per-module RoPE equivalence guard."
            )
        L = module.input_size[0] * module.input_size[1]
        # Build q/k on the module's device so the elementwise multiply in
        # apply_rotary_enc / apply_rotary_enc_real never sees a CPU/CUDA mismatch.
        # freqs_cis_real was just regenerated by _setup_rope_freqs(), so its
        # .device is authoritative (mirrors the whole-encoder guard's convention
        # of using next(merged.parameters()).device).
        rope_device = module.freqs_cis_real.device
        gen = torch.Generator(device=rope_device).manual_seed(0)
        q = torch.randn(1, module.num_heads, L, module.head_dim, generator=gen, device=rope_device)
        k = torch.randn(1, module.num_heads, L, module.head_dim, generator=gen, device=rope_device)

        # Reference: original complex path using the pre-patch snapshot.
        # Signature: apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False).
        with torch.no_grad():
            q_ref, k_ref = apply_rotary_enc(q, k, freqs_cis=freqs_cis_snapshot)

        # Candidate: the real path via the freshly regenerated buffers.
        # Signature: apply_rotary_enc_real(xq, xk, freqs_cis_real, freqs_cis_imag).
        with torch.no_grad():
            q_real, k_real = apply_rotary_enc_real(
                q,
                k,
                freqs_cis_real=module.freqs_cis_real,
                freqs_cis_imag=module.freqs_cis_imag,
            )

        # Tight tolerance: spike showed bit-exact at fp32; 1e-5 is generous
        # (spec §Spike findings, §Per-module guard).
        _ROPE_GUARD_ATOL = 1e-5  # spec §Per-module guard
        _ROPE_GUARD_RTOL = 1e-5  # spec §Per-module guard
        q_ok = torch.allclose(q_real, q_ref, atol=_ROPE_GUARD_ATOL, rtol=_ROPE_GUARD_RTOL)
        k_ok = torch.allclose(k_real, k_ref, atol=_ROPE_GUARD_ATOL, rtol=_ROPE_GUARD_RTOL)
        if not q_ok or not k_ok:
            max_delta_q = float((q_real - q_ref).abs().max())
            max_delta_k = float((k_real - k_ref).abs().max())
            raise RopeEquivalenceError(
                f"Per-module RoPE equivalence guard FAILED for {module!r}. "
                f"max|Δq|={max_delta_q:.6g}, max|Δk|={max_delta_k:.6g} "
                f"(atol={_ROPE_GUARD_ATOL}, rtol={_ROPE_GUARD_RTOL}). "
                "The regenerated real-valued RoPE table does not match the original "
                "complex path at the documented tolerance. Aborting export."
            )

        n += 1

    logger.info("_patch_encoder_rope_for_export: patched %d Attention module(s)", n)
    return n


def _run_parity_check(
    staging_dir: Path,
    wrapper: Sam3Wrapper,
    cfg: TrainConfig,
    *,
    fp16: bool,
    include: str,
) -> None:
    """Compare the composed ORT bundle against the merged torch forward (spec §7).

    Runs on ``CPUExecutionProvider`` over deterministic synthetic input and
    raises :class:`ExportParityError` on the first drifting / shape-mismatched
    ``_PARITY_KEYS`` entry. For partial ``--include`` the parity is scoped to the
    present graph against its torch intermediate (spec §7.3).
    """
    import numpy as np

    from custom_sam_peft.data.base import TextPrompts
    from custom_sam_peft.predict.onnx_session import _OrtCore

    # --- Deterministic synthetic input (spec §7.1) ---
    gen = torch.Generator(device="cpu").manual_seed(0)
    channels = cfg.data.channels
    b = _TRACE_B
    size = SAM3_IMAGE_SIZE
    dtype = torch.float16 if fp16 else torch.float32
    images = torch.rand(b, channels, size, size, generator=gen).to(dtype)  # RAW floats
    classes = _resolve_class_names(cfg)
    k = min(len(classes), 2) or 1
    prompt_classes = list(classes[:k]) or ["object"]
    prompts = [TextPrompts(classes=list(prompt_classes)) for _ in range(b)]

    band = "fp16" if fp16 else "fp32"
    atol, rtol = _PARITY_TOL[band]

    # --- Torch reference: the SAME merged module at the SAME export dtype ---
    # The merged wrapper may be on CUDA (LoRA path); feed the reference on its device.
    # np_img below is taken from the CPU copy, so the ORT (CPU-EP) side is unaffected.
    with torch.no_grad():
        ref = wrapper.forward(images.to(next(wrapper.parameters()).device), prompts)

    np_img = images.detach().cpu().numpy()
    core = _OrtCore(staging_dir, ["CPUExecutionProvider"])

    # --- ORT side: compose exactly as csp predict --use-onnx wires it (§7.2/§7.3) ---
    if include in ("encoder", "all"):
        vision_feats = core.run_encoder(np_img)
    else:  # include == "decoder": feed torch-produced encoder feats as the boundary.
        vision_feats = _torch_encoder_feats(wrapper, images)

    if include == "encoder":
        # No decoder graph: scope parity to the encoder feats vs the torch backbone
        # intermediate at the same encoder<->decoder boundary (spec §7.3).
        ort_out = vision_feats
        ref_out = _torch_encoder_feats(wrapper, images)
        keys = sorted(ort_out)
    else:
        ort_out = core.run_decoder(vision_feats, prompt_classes)
        ref_out = {key: ref[key] for key in _PARITY_KEYS}
        keys = list(_PARITY_KEYS)

    for key in keys:
        ort_arr = np.asarray(ort_out[key]).astype(np.float32)
        ref_arr = ref_out[key].detach().cpu().to(torch.float32).numpy()
        if ort_arr.shape != ref_arr.shape:
            _raise_parity(staging_dir, key, ref_arr, ort_arr, band, atol, rtol, shape_mismatch=True)
        if not np.allclose(ort_arr, ref_arr, atol=atol, rtol=rtol):
            _raise_parity(staging_dir, key, ref_arr, ort_arr, band, atol, rtol)


def _torch_encoder_feats(wrapper: Sam3Wrapper, images: Tensor) -> dict[str, Any]:
    """Run the torch encoder shim to produce the named vision-feature boundary tensors."""
    adapter = wrapper.model
    merged = cast(nn.Module, adapter.model)
    channel_adapter = cast("nn.Module | None", adapter.channel_adapter)
    shim = _EncoderExport(merged, channel_adapter).eval()
    with torch.no_grad():
        outs = shim(images)
    names = _encoder_output_names(len(outs))
    return dict(zip(names, outs, strict=True))


def _raise_parity(
    staging_dir: Path,
    key: str,
    ref_arr: Any,
    ort_arr: Any,
    band: str,
    atol: float,
    rtol: float,
    *,
    shape_mismatch: bool = False,
) -> None:
    """Build the diagnostic ExportParityError for the first drifting/mismatched key."""
    import numpy as np

    if shape_mismatch:
        detail = f"shape mismatch torch={ref_arr.shape} vs ort={ort_arr.shape}"
        worst = "n/a"
        vals = "n/a"
    else:
        diff = np.abs(ort_arr - ref_arr)
        flat_idx = int(np.argmax(diff))
        worst = str(np.unravel_index(flat_idx, diff.shape))
        max_abs = float(diff.flat[flat_idx])
        detail = f"max-abs-diff={max_abs:.6g}"
        vals = f"torch={float(ref_arr.flat[flat_idx]):.6g} ort={float(ort_arr.flat[flat_idx]):.6g}"
    hint = (
        "fp16: retry without --fp16 to isolate quantization drift"
        if band == "fp16"
        else "likely an unsupported-op / tracing mismatch: bump --opset or file an issue"
    )
    raise ExportParityError(
        f"torch-vs-ORT parity failed on key {key!r}: {detail}; worst-index={worst}; "
        f"{vals}; band={band} (atol={atol}, rtol={rtol}). Hint: {hint}."
    )


# ---------------------------------------------------------------------------
# Tracers (spec §5.2 / §5.3)
# ---------------------------------------------------------------------------


class _EncoderExport(nn.Module):
    """Vision path shim: optional N->3 channel adapter + backbone.forward_image.

    Flattens the FPN ``backbone_fpn`` feats + ``vision_pos_enc`` to a fixed count
    of output tensors (the encoder<->decoder boundary).
    """

    def __init__(self, merged: nn.Module, channel_adapter: nn.Module | None) -> None:
        super().__init__()
        self.merged = merged
        self.channel_adapter = channel_adapter

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        """Return flattened (feats..., pos...) for the FPN levels."""
        x = self.channel_adapter(images) if self.channel_adapter is not None else images
        backbone = cast(Any, self.merged).backbone
        backbone_out = backbone.forward_image(x)
        feats = list(backbone_out["backbone_fpn"])
        pos = list(backbone_out["vision_pos_enc"])
        return (*feats, *pos)


class _DecoderExport(nn.Module):
    """Decoder shim: bake the training-class text embedding as a constant.

    Rebuilds ``backbone_out`` from the encoder's flattened vision feats, attaches
    the baked text outputs, builds a FindStage via ``multiplex_index_arrays``
    semantics + a zero geometric prompt, and runs the grounding core.
    """

    def __init__(
        self,
        merged: nn.Module,
        baked_text: dict[str, Tensor],
        b: int,
        k: int,
        n_levels: int,
    ) -> None:
        super().__init__()
        self.merged = merged
        self.b = b
        self.k = k
        self.n_levels = n_levels
        # Bake the text outputs as constant buffers so the graph carries them.
        self._text_keys: list[str] = []
        for i, (key, val) in enumerate(baked_text.items()):
            buf_name = f"_baked_text_{i}"
            self.register_buffer(buf_name, val.detach())
            self._text_keys.append(key)

    def _baked_text(self) -> dict[str, Tensor]:
        return {key: getattr(self, f"_baked_text_{i}") for i, key in enumerate(self._text_keys)}

    def forward(self, *vision_feats: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the export-only grounding core; return the four SAM3 output keys."""
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt

        feats = list(vision_feats[: self.n_levels])
        pos = list(vision_feats[self.n_levels :])
        backbone_out: dict[str, Any] = {"backbone_fpn": feats, "vision_pos_enc": pos}
        backbone_out.update(self._baked_text())

        ii, ti = multiplex_index_arrays(self.b, self.k)
        n_cols = self.b * self.k
        device = feats[0].device
        model_dtype = feats[0].dtype
        find_input = FindStage(
            img_ids=torch.from_numpy(ii).to(device=device),
            text_ids=torch.from_numpy(ti).to(device=device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        gp = Prompt(
            box_embeddings=torch.zeros(0, n_cols, 4, device=device, dtype=model_dtype),
            box_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
            point_embeddings=torch.zeros(0, n_cols, 2, device=device, dtype=model_dtype),
            point_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
        )
        out = cast(Any, self.merged).forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_masks"],
            out["presence_logit_dec"],
        )


def _trace_encoder(
    merged: nn.Module,
    channel_adapter: nn.Module | None,
    dest: Path,
    *,
    channels: int,
    opset: int,
    export_dtype: torch.dtype,
    dynamic_axes: bool,
) -> None:
    """Trace ``_EncoderExport`` to ``dest`` (spec §5.2)."""
    shim = _EncoderExport(merged, channel_adapter).eval()
    # Trace inputs must live on the merged model's device: the LoRA path leaves the
    # model on CUDA (fp16 conv is unsupported on CPU), so CPU dummies would mismatch.
    # ONNX graphs are device-neutral, so tracing on CUDA still yields a CPU-loadable graph.
    images = torch.zeros(
        _TRACE_B,
        channels,
        SAM3_IMAGE_SIZE,
        SAM3_IMAGE_SIZE,
        dtype=export_dtype,
        device=next(merged.parameters()).device,
    )
    with torch.no_grad():
        outs = shim(images)
    n_outputs = len(outs)
    output_names = _encoder_output_names(n_outputs)
    axes = None
    if dynamic_axes:
        axes = {"images": {0: "batch"}}
        for name in output_names:
            axes[name] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            shim,
            (images,),
            str(dest),
            opset_version=opset,
            dynamo=False,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=axes,
        )


def _trace_decoder(
    merged: nn.Module,
    dest: Path,
    *,
    class_names: list[str],
    channels: int,
    channel_adapter: nn.Module | None,
    opset: int,
    export_dtype: torch.dtype,
    dynamic_axes: bool,
) -> None:
    """Trace ``_DecoderExport`` to ``dest`` (spec §5.3); K baked from class_names."""
    k = len(class_names)
    # Produce the encoder feats once (the encoder<->decoder boundary) to feed the
    # decoder shim with correctly-shaped dummies, and bake the text embedding.
    enc = _EncoderExport(merged, channel_adapter).eval()
    # Trace inputs on the merged model's device (see _trace_encoder for why).
    images = torch.zeros(
        _TRACE_B,
        channels,
        SAM3_IMAGE_SIZE,
        SAM3_IMAGE_SIZE,
        dtype=export_dtype,
        device=next(merged.parameters()).device,
    )
    with torch.no_grad():
        vision_feats = enc(images)
    n_levels = len(vision_feats) // 2

    device = vision_feats[0].device
    baked_text = cast(Any, merged).backbone.forward_text(list(class_names), device=device)
    baked_text = {key: val.detach() for key, val in baked_text.items()}

    shim = _DecoderExport(merged, baked_text, _TRACE_B, k, n_levels).eval()
    input_names = _encoder_output_names(len(vision_feats))
    output_names = list(_PARITY_KEYS)
    axes = None
    if dynamic_axes:
        axes = {name: {0: "batch"} for name in input_names}
        for name in output_names:
            axes[name] = {0: "rows"}
    with torch.no_grad():
        torch.onnx.export(
            shim,
            tuple(vision_feats),
            str(dest),
            opset_version=opset,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=axes,
        )


def _encoder_output_names(n_outputs: int) -> list[str]:
    """Stable names for the flattened (feats..., pos...) encoder outputs."""
    n_levels = n_outputs // 2
    feats = [f"feat_{i}" for i in range(n_levels)]
    pos = [f"pos_{i}" for i in range(n_levels)]
    return feats + pos


# ---------------------------------------------------------------------------
# Sidecar writers (spec §6)
# ---------------------------------------------------------------------------


def _write_json(dest: Path, record: dict[str, Any]) -> None:
    """Serialize ``record`` as pretty JSON with a trailing newline."""
    dest.write_text(json.dumps(record, indent=2) + "\n")


def _write_preprocessor(bundle_dir: Path, cfg: TrainConfig) -> None:
    """Write ``preprocessor.json`` per the §6.1 field-sourcing table (all reuse)."""
    from custom_sam_peft.data.transforms import resolve_normalization_with_path

    normalize = cfg.data.normalize or NormalizeConfig()
    mean, std, norm_path = resolve_normalization_with_path(
        cfg.model.name, normalize, channel_semantics=cfg.data.channel_semantics
    )

    if cfg.data.channel_semantics != "rgb" and cfg.data.normalize is None:
        logger.warning(
            "channel_semantics=%r but data.normalize is unset: the bundle's "
            "preprocessor.json will ship the schema-default mean/std (the "
            "'config-fallback' path), which is almost certainly wrong for a "
            "non-rgb model. Set data.normalize.mean/std explicitly.",
            cfg.data.channel_semantics,
        )

    # cv2 constants stored as STRING names (build-dependent ints); predict-side
    # maps name->cv2 int. Cross-ref transforms.py:215-226.
    record = {
        "schema_version": PREPROCESSOR_SCHEMA_VERSION,
        "image_size": SAM3_IMAGE_SIZE,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "max_pixel_value": float(normalize.max_pixel_value),
        "normalization_path": norm_path,
        "channels": cfg.data.channels,
        "channel_semantics": cfg.data.channel_semantics,
        "resize_interpolation": "INTER_LINEAR",
        "mask_interpolation": "INTER_NEAREST",
        "pad_position": "top_left",
        "border_mode": "BORDER_CONSTANT",
        "border_fill_value": 0,
    }
    _write_json(bundle_dir / PREPROCESSOR_FILE, record)


def _write_prompts(bundle_dir: Path, class_names: list[str]) -> None:
    """Write ``prompts.txt``: one training class per line, order preserved (§6.3)."""
    if not class_names:
        raise RuntimeError("prompts.txt requires a non-empty class list")
    (bundle_dir / PROMPTS_FILE).write_text("\n".join(class_names) + "\n", encoding="utf-8")


def _write_model_card(
    bundle_dir: Path,
    cfg: TrainConfig,
    *,
    opset: int,
    fp16: bool,
    include: str,
    dynamic_axes: bool,
    parity_checked: bool,
) -> None:
    """Write ``model_card.json`` LAST (§6.2); presence signals a complete bundle."""
    record = {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "name": cfg.model.name,
        "base": cfg.model.name,
        "training_config_hash": _hash_cfg(cfg),
        "opset": opset,
        "fp16": fp16,
        "include": include,
        "dynamic_axes": dynamic_axes,
        "parity_checked": parity_checked,
        "git_sha": git_sha(),
        "version": custom_sam_peft.__version__,
        "exported_at": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    }
    _write_json(bundle_dir / MODEL_CARD_FILE, record)
