"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md.

Revised by docs/superpowers/plans/2026-05-16-model-loading-revised.md to match
Meta's open-vocab head: one prompt class per forward call. Trainer loops over
the fixed class vocabulary externally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from peft import PeftModel

import sam3
import torch
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import BoxPrompts, Prompts, TextPrompts
from esam3.utils.huggingface import download_model

logger = logging.getLogger(__name__)


def _build_geometric_prompt(
    box_hints: list[Tensor | None],
    image_size: int,
    device: torch.device,
) -> Prompt | None:
    """Translate per-image box hints to Meta's ``Prompt`` container.

    Layout pinned by docs/superpowers/plans/2026-05-17-training-loop-notes.md:
      - ``Prompt.box_embeddings``: ``(N_boxes, B, 4)`` float, normalized cxcywh in ``[0, 1]``.
      - ``Prompt.box_mask``: ``(B, N_boxes)`` bool, ``True`` = padded (PyTorch key-padding).
      - ``box_labels`` left ``None`` (defaults to all-positive in the encoder).

    Coordinate space: input boxes are absolute pixel xyxy; output is normalized
    cxcywh in ``[0, 1]`` relative to ``image_size``.  The wrapper contract
    assumes square images (``H == W == image_size``).

    Padding is right-padded; pad slots filled with zeros (encoder filters via
    the mask).  Returns ``None`` when every entry is ``None``; the adapter
    substitutes Meta's zero-length-seq dummy (``Prompt(box_embeddings=zeros(0,
    B, 4), box_mask=zeros(B, 0))``) in that case.
    """
    if all(h is None for h in box_hints):
        return None

    b = len(box_hints)
    n_max = max((h.shape[0] for h in box_hints if h is not None), default=0)
    if n_max == 0:
        # All tensors present but empty (edge case — treat as all-None dummy).
        return None

    # Normalize scale: pixel xyxy → normalized xyxy → cxcywh.
    scale = torch.tensor(
        [image_size, image_size, image_size, image_size], dtype=torch.float32, device=device
    )

    # box_embeddings: (N_max, B, 4), box_mask: (B, N_max)
    box_embeddings = torch.zeros(n_max, b, 4, dtype=torch.float32, device=device)
    box_mask = torch.ones(b, n_max, dtype=torch.bool, device=device)  # True = padded

    for i, h in enumerate(box_hints):
        if h is None:
            # Entire row stays masked (all True) and zero-filled.
            continue
        if h.ndim != 2 or h.shape[-1] != 4:
            raise ValueError(f"box_hints[{i}] must have shape (M_i, 4); got {tuple(h.shape)}")
        n_i = h.shape[0]
        if n_i == 0:
            continue
        h_dev = h.to(device=device, dtype=torch.float32)
        norm_xyxy = h_dev / scale  # (n_i, 4), normalized xyxy
        cxcywh = box_xyxy_to_cxcywh(norm_xyxy)  # (n_i, 4), normalized cxcywh
        box_embeddings[:n_i, i, :] = cxcywh
        box_mask[i, :n_i] = False  # real hints are NOT padded

    return Prompt(box_embeddings=box_embeddings, box_mask=box_mask, box_labels=None)


class Sam3Wrapper(nn.Module):
    """Thin wrapper around Meta's SAM 3.1 model.

    Contract:
      - ``forward(images, prompts, box_hints=None)`` accepts a batch of B images
        and a list of B ``Prompts`` objects, one per image.
      - ``box_hints``: optional list of length B.  Each element is either ``None``
        (no geometric hint for that image) or a ``(M_i, 4)`` float tensor of
        absolute pixel xyxy boxes.  ``box_hints`` must not be combined with
        ``BoxPrompts`` (they carry conflicting localization signals).
      - All prompts in a batch MUST be the same variant (TextPrompts XOR
        BoxPrompts); the wrapper raises on mixed batches.
      - For TextPrompts, each image's prompt MUST contain exactly one class
        name; the trainer is responsible for looping over the fixed class
        vocabulary and accumulating losses across classes.
      - Returns Meta's native output dict unchanged.
    """

    def __init__(self, model: nn.Module, image_size: int = 1008, mask_size: int = 288) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size
        self.peft_model: PeftModel | None = None

    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(images, prompts, box_hints)
        out: dict[str, Any] = self.model(images, prompts, box_hints=box_hints)
        return out

    @staticmethod
    def _validate_inputs(
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be (B, 3, H, W); got shape {tuple(images.shape)}")
        b = images.shape[0]
        if len(prompts) != b:
            raise ValueError(f"len(prompts)={len(prompts)} must equal batch size {b}")
        if not prompts:
            return
        first = type(prompts[0])
        for p in prompts:
            if type(p) is not first:
                raise ValueError(
                    "All prompts in a batch must be the same prompt variant "
                    "(TextPrompts or BoxPrompts), not mixed."
                )
            if isinstance(p, TextPrompts) and len(p.classes) != 1:
                raise ValueError(
                    f"TextPrompts must contain exactly one class per forward "
                    f"call (got {len(p.classes)}). Loop over the class vocabulary "
                    f"externally."
                )

        if box_hints is not None:
            if first is BoxPrompts:
                raise ValueError(
                    "box_hints must not be combined with BoxPrompts prompts. "
                    "BoxPrompts already carry localization information."
                )
            if len(box_hints) != b:
                raise ValueError(f"len(box_hints)={len(box_hints)} must equal batch size {b}")
            for i, h in enumerate(box_hints):
                if h is None:
                    continue
                if h.ndim != 2 or h.shape[-1] != 4:
                    raise ValueError(
                        f"box_hints[{i}] must have shape (M_i, 4); got {tuple(h.shape)}"
                    )


def _resolve_checkpoint_path(cfg: ModelConfig) -> Path:
    """Return the local checkpoint path, auto-downloading from the Hub on miss.

    - ``local_dir=None`` raises ``FileNotFoundError`` with an `esam3 init` hint.
    - File present: return it (no Hub contact).
    - File missing: ``download_model(cfg.name, local_dir, revision=cfg.revision)``,
      then re-check. If the file is STILL missing post-download (e.g. the user
      pinned a revision that doesn't contain it), raise ``FileNotFoundError``
      with a precise diagnostic.
    """
    if cfg.local_dir is None:
        raise FileNotFoundError(
            f"ModelConfig.local_dir is None. Set it to a directory for "
            f"{cfg.checkpoint_file}, or run `esam3 init` to scaffold one."
        )
    local_dir = Path(cfg.local_dir)
    path = local_dir / cfg.checkpoint_file
    if path.exists():
        return path
    logger.info(
        "SAM 3.1 checkpoint missing at %s; auto-downloading %s",
        path,
        cfg.name,
    )
    download_model(cfg.name, local_dir, revision=cfg.revision)
    if not path.exists():
        raise FileNotFoundError(
            f"Downloaded {cfg.name} into {local_dir} but {cfg.checkpoint_file} "
            f"is still missing. Check that the repo (revision={cfg.revision!r}) "
            f"contains that file."
        )
    return path


class _Sam3ImageAdapter(nn.Module):
    """Adapt raw Sam3Image to the (images, prompts, box_hints) calling convention.

    Sam3Image's training-mode forward (``forward_grounding``) expects
    ``(backbone_out, find_input, find_target, geometric_prompt)``, none of which
    are raw image tensors or our ``Prompts`` dataclasses.  This adapter holds the
    inner ``Sam3Image`` and orchestrates the conversion.

    The ``box_hints`` kwarg routes per-image absolute-pixel xyxy box hints
    through ``_build_geometric_prompt`` into Meta's ``Prompt`` container.  When
    every entry is ``None`` (or the kwarg itself is ``None``), the builder
    returns ``None`` and we substitute Meta's zero-length-seq dummy.

    ``image_size`` must match the wrapper's image_size; ``load_sam31`` plumbs
    it through the constructor.
    """

    def __init__(self, model: nn.Module, image_size: int = 1008) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        if not all(isinstance(p, TextPrompts) for p in prompts):
            raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
        text_prompts = cast(list[TextPrompts], prompts)
        class_names = [p.classes[0] for p in text_prompts]
        if len(set(class_names)) > 1:
            raise ValueError(
                "All prompts in a batch must share the same class name "
                "(SAM 3.1 forward_grounding runs one text prompt per call); "
                f"got {class_names}"
            )
        device = images.device
        b = images.shape[0]
        model_dtype = next(self.model.parameters()).dtype
        backbone_out = self.model.backbone.forward_image(images)  # type: ignore[union-attr, operator]
        text_outputs = self.model.backbone.forward_text(  # type: ignore[union-attr, operator]
            [class_names[0]], device=device
        )
        backbone_out.update(text_outputs)
        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long),
            text_ids=torch.zeros(b, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        gp = _build_geometric_prompt(
            box_hints if box_hints is not None else [None] * b,
            self.image_size,
            device,
        )
        if gp is None:
            gp = Prompt(
                box_embeddings=torch.zeros(0, b, 4, device=device, dtype=model_dtype),
                box_mask=torch.zeros(b, 0, device=device, dtype=torch.bool),
                point_embeddings=torch.zeros(0, b, 2, device=device, dtype=model_dtype),
                point_mask=torch.zeros(b, 0, device=device, dtype=torch.bool),
            )
        outputs: dict[str, Tensor] = self.model.forward_grounding(  # type: ignore[operator]
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
        return outputs


def _patch_pos_enc_dtype(model: nn.Module) -> None:
    """Wrap every PositionEmbeddingSine._encode_xy to honor input dtype.

    sam3's ``PositionEmbeddingSine._encode_xy``
    (sam3/model/position_encoding.py:60-77) builds its frequency table as
    ``dim_t = torch.arange(..., dtype=torch.float32, ...)`` regardless of the
    input dtype.  Downstream broadcasts produce fp32 output, which then feeds a
    bf16-weight ``points_pos_enc_project`` Linear in
    ``PointGeometryEncoder._encode_points`` (sam3/model/geometry_encoders.py:623)
    and raises ``RuntimeError: mat1 and mat2 must have the same dtype`` on Colab
    T4 with ``ModelConfig(dtype="bfloat16")``.  This is true even for zero-length
    point sequences because ``F.linear`` validates dtypes regardless of seq len.

    We wrap each ``_encode_xy`` method to cast its (pos_x, pos_y) outputs to the
    dtype of the input ``x`` tensor BEFORE returning.  The bound method is
    replaced via ``MethodType`` on each ``PositionEmbeddingSine`` instance so the
    patch persists across forward calls and survives ``.to(device)`` /
    ``.to(dtype)`` (only parameters move; methods do not).

    This is a localized stop-gap.  The right long-term fix is upstream in
    sam3's pos-enc to honor input dtype directly (tracked as a follow-up
    in logs/TODO.md).  Re-evaluate every sam3 version bump.

    Notes:
    - We use a per-instance ``MethodType`` replacement (NOT class-level
      monkey-patch) to avoid affecting other consumers of sam3 in the same
      process.
    - We do NOT introduce any ``torch.autocast`` scope; doing so re-triggered
      the bf16-vs-fp32 collision inside ``sam3/model/decoder.py::forward_ffn``'s
      ``with torch.amp.autocast(enabled=False)`` region during PR #13's v2 work.
      The cast-on-return approach side-steps that entirely.
    """
    from types import MethodType

    from sam3.model.position_encoding import PositionEmbeddingSine

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, PositionEmbeddingSine):
            continue
        if getattr(submodule, "_esam3_pos_enc_dtype_patched", False):
            continue
        original = submodule._encode_xy

        def _encode_xy_dtype_aware(self, x, y, _orig=original):  # type: ignore[no-untyped-def]
            pos_x, pos_y = _orig(x, y)
            return pos_x.to(dtype=x.dtype), pos_y.to(dtype=x.dtype)

        submodule._encode_xy = MethodType(_encode_xy_dtype_aware, submodule)
        submodule._esam3_pos_enc_dtype_patched = True  # idempotency marker
        patched_count += 1

    logger.info(
        "Patched %d PositionEmbeddingSine._encode_xy callsites for dtype awareness.",
        patched_count,
    )


def _patch_roi_align_dtype() -> None:
    """Wrap ``torchvision.ops.roi_align`` to cast ``boxes`` to the input tensor's dtype.

    sam3's ``SequenceGeometryEncoder._encode_boxes``
    (sam3/model/geometry_encoders.py:651-653) calls ``torchvision.ops.roi_align``
    with ``rois`` hard-cast to fp32 via ``.float()``, while ``img_feats`` is bf16
    when the model is loaded under ``ModelConfig(dtype="bfloat16")``.  torchvision's
    C++ kernel requires both arguments to share dtype, so this raises a
    ``RuntimeError`` on Colab T4.  We cannot modify the sam3 source (installed
    package), and we cannot wrap the call in ``torch.autocast`` because that
    re-triggers the bf16-vs-fp32 collision inside
    ``sam3/model/decoder.py::forward_ffn``'s ``with torch.amp.autocast(enabled=False)``
    region — the same constraint that drove the cast-on-output approach adopted in
    PR #13.

    This function installs a thin module-level wrapper on ``torchvision.ops.roi_align``
    that coerces ``boxes`` (both the list-of-tensors form and the single-tensor form)
    to ``input.dtype`` before delegating to the original kernel.  The patch is
    idempotent: repeated calls are no-ops once the sentinel attribute is set.

    Notes:
    - This is a module-level monkey-patch (not class/instance-level) because the
      call site we are working around is inside sam3's installed package, not a
      submodule of the model we can traverse.
    - We do NOT introduce any ``torch.autocast`` scope; doing so re-triggered the
      bf16-vs-fp32 collision inside ``sam3/model/decoder.py::forward_ffn``'s
      ``with torch.amp.autocast(enabled=False)`` region during PR #13's v2 work.
      The cast-before-call approach side-steps that entirely.
    - Re-evaluate every sam3 version bump; track long-term fix in logs/TODO.md.
    """
    import torchvision.ops as tvo  # type: ignore[import-untyped]

    if getattr(tvo, "_esam3_roi_align_dtype_patched", False):
        return
    _original = tvo.roi_align

    def _roi_align_dtype_aware(input, boxes, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(boxes, (list, tuple)):
            boxes = type(boxes)(
                b.to(dtype=input.dtype) if b.dtype != input.dtype else b for b in boxes
            )
        elif hasattr(boxes, "dtype") and boxes.dtype != input.dtype:
            boxes = boxes.to(dtype=input.dtype)
        return _original(input, boxes, *args, **kwargs)

    tvo.roi_align = _roi_align_dtype_aware
    tvo._esam3_roi_align_dtype_patched = True
    logger.info(
        "Patched torchvision.ops.roi_align for dtype awareness (boxes cast to input dtype)."
    )


def _patch_encode_prompt_dtype(model: nn.Module) -> None:
    """Cast ``_encode_prompt``'s returned ``prompt`` to the model's parameter dtype.

    sam3's ``SAM3Image._encode_prompt``
    (sam3/model/sam3_image.py:196-198) builds a fallback ``visual_prompt_embed``
    via ``torch.zeros((0, *geo_feats.shape[1:]), device=...)`` with NO ``dtype=``
    argument, so it defaults to ``torch.float32``.  Even though the tensor is
    zero-length in dim 0, the immediately-following
    ``torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)`` triggers
    PyTorch's type-promotion rule and returns an fp32 ``prompt`` when the
    model is loaded under ``ModelConfig(dtype="bfloat16")``.  The fp32 prompt
    then flows into ``TransformerEncoderFusion`` as ``memory``; the encoder
    layer's ``cross_attn_image`` does ``key = memory`` and feeds it to a
    Linear with bf16 weight, producing
    ``RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16``.

    We cannot modify sam3 source (installed package), and we cannot wrap the
    call in ``torch.autocast`` for the same reason as the prior two patches:
    sam3's ``decoder.py::forward_ffn`` contains an explicit
    ``with torch.amp.autocast(enabled=False)`` region that re-triggers the
    bf16/fp32 collision.

    The patch rebinds ``_encode_prompt`` per instance via ``MethodType`` and
    casts the returned ``prompt`` (and ``prompt_mask`` is left alone — it's
    boolean) to the model's parameter dtype before returning.  When the model
    is loaded in fp32 the cast is a no-op.

    Notes:
    - We use a per-instance ``MethodType`` replacement (NOT class-level
      monkey-patch) so we don't affect other consumers of sam3 in the same
      process.
    - Idempotency sentinel mirrors ``_patch_pos_enc_dtype``.
    - Re-evaluate every sam3 version bump; track upstream fix in logs/TODO.md.
    """
    from types import MethodType

    if not hasattr(model, "_encode_prompt"):
        return
    if getattr(model, "_esam3_encode_prompt_dtype_patched", False):
        return
    original = model._encode_prompt
    target_dtype = next(model.parameters()).dtype

    def _encode_prompt_dtype_aware(self, *args, _orig=original, _dtype=target_dtype, **kwargs):  # type: ignore[no-untyped-def]
        prompt, prompt_mask, backbone_out = _orig(*args, **kwargs)
        if prompt.dtype != _dtype:
            prompt = prompt.to(dtype=_dtype)
        return prompt, prompt_mask, backbone_out

    model._encode_prompt = MethodType(_encode_prompt_dtype_aware, model)  # type: ignore[assignment]
    model._esam3_encode_prompt_dtype_patched = True  # type: ignore[assignment]
    logger.info(
        "Patched SAM3Image._encode_prompt for dtype awareness (prompt cast to %s).",
        target_dtype,
    )


# Modules that own a `weight` parameter and require their floating-point
# input to match that weight's dtype. We hook the forward pre-call on each
# instance to cast input[0] in-flight. Embedding is intentionally excluded
# (integer input). Attention-style fused modules (e.g. nn.MultiheadAttention)
# are excluded because they take multiple tensor inputs with non-uniform
# dtype expectations; sam3's attention is built from raw Linears anyway.
_DTYPE_SENSITIVE_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.LayerNorm,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)


def _patch_module_input_dtype(model: nn.Module) -> None:
    """Install a generic fp-input-dtype backstop on every dtype-sensitive submodule.

    Several places in sam3 build activation tensors with hardcoded fp32
    (e.g. ``torch.arange(..., dtype=torch.float32)`` in
    ``model_misc.gen_sineembed_for_position`` at sam3/model/model_misc.py:915,
    ``.float()`` in ``get_valid_ratio`` at sam3/model/model_misc.py:910).
    When the model is loaded under ``ModelConfig(dtype="bfloat16")``, those
    fp32 tensors flow into bf16-weighted ``nn.Linear``/``nn.LayerNorm``/conv
    modules and raise ``RuntimeError: mat1 and mat2 must have the same
    dtype, but got Float and BFloat16`` on Colab T4.  We've already patched
    three specific producers (``_patch_pos_enc_dtype``,
    ``_patch_roi_align_dtype``, ``_patch_encode_prompt_dtype``); this is the
    generic backstop that catches any remaining or future cascading site by
    coercing dtype at the *consumer* boundary.

    The hook fires before each forward call, casts a floating-point first
    positional input to the module's first-parameter dtype, and returns the
    rewritten args tuple.  Non-tensor inputs and integer/bool tensors are
    passed through untouched (preserves ``nn.Embedding`` semantics, though
    Embedding is excluded from the iterated module-type set anyway).
    Parameter-free submodules (no ``.parameters()``) are skipped.

    The patch is idempotent per instance via a sentinel attribute.  Hooks
    survive ``.to(dtype=)`` / ``.to(device)`` calls because they are
    attached to the module, not its parameters.

    Notes:
    - This is a *consumer-side* defense.  We retain the producer-side
      patches above for sites we already know about, partly to keep the
      precision close to source and partly so the existing test suite for
      those producer patches keeps documenting the upstream bug surface.
      The two layers compose without redundancy in the happy path: the
      consumer hook is a no-op once the producer already matches.
    - We do NOT use ``torch.autocast`` because that re-triggers the bf16/fp32
      collision in ``sam3/model/decoder.py::forward_ffn``'s
      ``with torch.amp.autocast(enabled=False)`` region (PR #13 constraint).
    - Re-evaluate every sam3 version bump; track upstream fix in logs/TODO.md.
    """

    def _input_dtype_hook(module: nn.Module, args: tuple[Any, ...]):  # type: ignore[no-untyped-def]
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None
        try:
            target_dtype = next(module.parameters()).dtype
        except StopIteration:
            return None
        if x.dtype == target_dtype:
            return None
        return (x.to(dtype=target_dtype), *args[1:])

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, _DTYPE_SENSITIVE_MODULE_TYPES):
            continue
        if getattr(submodule, "_esam3_module_input_dtype_patched", False):
            continue
        submodule.register_forward_pre_hook(_input_dtype_hook)
        submodule._esam3_module_input_dtype_patched = True  # type: ignore[assignment]
        patched_count += 1

    logger.info(
        "Patched %d dtype-sensitive modules (Linear/LayerNorm/Conv) with input-dtype hook.",
        patched_count,
    )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's `sam3` package and wrap it for our trainer.

    Returns a `Sam3Wrapper` whose `forward(images, prompts, box_hints=None)` returns Meta's
    native per-class output dict (`pred_logits`, `pred_boxes`, `pred_masks`,
    `presence_logit_dec`).
    """
    ckpt_path = _resolve_checkpoint_path(cfg)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw_model = sam3.build_sam3_image_model(
        device=device,
        eval_mode=False,  # training mode — gradients flow.
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

    if cfg.gradient_checkpointing:
        if hasattr(raw_model, "set_grad_checkpointing"):
            raw_model.set_grad_checkpointing(True)
        else:
            logger.warning(
                "Meta sam3 model has no `set_grad_checkpointing`; "
                "gradient_checkpointing=True is a no-op on this revision."
            )

    if cfg.dtype == "bfloat16":
        raw_model = raw_model.to(dtype=torch.bfloat16)
    elif cfg.dtype == "float16":
        raw_model = raw_model.to(dtype=torch.float16)

    # Cast PositionEmbeddingSine._encode_xy outputs to input dtype to avoid
    # fp32 inputs feeding bf16 Linear weights in the geometry encoder.
    # See _patch_pos_enc_dtype for full rationale.
    _patch_pos_enc_dtype(raw_model)

    # Cast roi_align boxes to input dtype to avoid fp32 rois fed to bf16 img_feats
    # in sam3's geometry encoder. See _patch_roi_align_dtype for full rationale.
    _patch_roi_align_dtype()

    # Cast _encode_prompt's `prompt` to model dtype — sam3 builds a fallback
    # fp32 visual_prompt_embed via torch.zeros() without dtype=, and
    # torch.cat type-promotes the concatenated prompt to fp32 even when
    # txt_feats/geo_feats are bf16. See _patch_encode_prompt_dtype.
    _patch_encode_prompt_dtype(raw_model)

    # Generic backstop: cast fp inputs to weight dtype at every
    # nn.Linear/LayerNorm/Conv* in the model. Catches any remaining or
    # future cascading fp32 producer site we haven't patched directly.
    # See _patch_module_input_dtype for full rationale.
    _patch_module_input_dtype(raw_model)

    adapter = _Sam3ImageAdapter(raw_model, image_size=1008)
    return Sam3Wrapper(adapter, image_size=1008, mask_size=288)
