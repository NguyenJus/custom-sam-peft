"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md.

Revised by docs/superpowers/plans/2026-05-16-model-loading-revised.md to match
Meta's open-vocab head: one prompt class per forward call. Trainer loops over
the fixed class vocabulary externally.
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from peft import PeftModel

import sam3
import torch
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from torch import Tensor, nn

from custom_sam_peft.config.schema import ModelConfig
from custom_sam_peft.data.base import BoxPrompts, Prompts, TextPrompts
from custom_sam_peft.utils.huggingface import download_model

logger = logging.getLogger(__name__)

# Keys absent from the released sam3.1_multiplex.pt checkpoint that are
# harmless to ignore.  The released checkpoint was built from a 3-scale
# (scale_factors=[4.0, 2.0, 1.0]) tri-backbone neck, so convs[3] (the
# scale=0.5 FPN head) was never trained and is absent.  Our
# build_sam3_image_model path instantiates a 4-scale neck
# (scale_factors=[4.0, 2.0, 1.0, 0.5]), so PyTorch reports convs[3]'s
# four parameters as missing when we load_state_dict.  They are safe to
# ignore because:
#   • SAM3VLBackbone is built with scalp=1 (model_builder.py:122), which
#     causes vl_combiner.py:91-95 to do sam3_features[:-1], dropping the
#     output of convs[3] entirely before it reaches the transformer.
#   • Sam3Image uses num_feature_levels=1 and reads backbone_fpn[-1],
#     which after the scalp=1 trim is convs[2]'s output — never convs[3].
#   • convs[3]'s randomly-initialised weights never receive gradients and
#     never influence any output or loss.
#
# Cross-references to re-check on every sam3 version bump:
#   • sam3/.../necks.py:36-95  — Sam3DualViTDetNeck.convs construction
#   • sam3/.../model_builder.py:122  — scalp=1 in _create_vl_backbone
#   • sam3/.../vl_combiner.py:91-95  — the sam3_features[:-1] trim
_KNOWN_MISSING_KEYS: frozenset[str] = frozenset(
    {
        "backbone.vision_backbone.convs.3.conv_1x1.weight",
        "backbone.vision_backbone.convs.3.conv_1x1.bias",
        "backbone.vision_backbone.convs.3.conv_3x3.weight",
        "backbone.vision_backbone.convs.3.conv_3x3.bias",
    }
)

# Regex matching the print line that sam3's _load_checkpoint emits when
# missing_keys is non-empty (model_builder.py:557-561):
#   f"loaded {checkpoint_path} and found missing and/or unexpected keys:\n{missing_keys=}"
# The group captures only the list repr (\[.*?\] with re.DOTALL to allow
# multi-line list formatting) so that any output *after* the list is not
# consumed by the match.  The leading .+ for the path is intentionally greedy
# so it stays on the first line; re.DOTALL only affects the list group.
_SAM3_MISSING_KEYS_RE = re.compile(
    r"loaded .+ and found missing and/or unexpected keys:\nmissing_keys=(\[.*?\])",
    re.DOTALL,
)


def _classify_missing_keys(
    missing: set[str],
    unexpected: set[str],
) -> Literal["ok", "fail"]:
    """Classify a (missing_keys, unexpected_keys) pair from load_state_dict.

    Returns ``"ok"`` when:
      - ``unexpected`` is empty, AND
      - ``missing`` is a subset of ``_KNOWN_MISSING_KEYS`` (i.e. the released
        checkpoint may have shipped some of those keys in a newer release, so a
        smaller-than-known missing set is still safe).

    Returns ``"fail"`` in all other cases.  A ``"fail"`` result means the
    caller should raise ``RuntimeError`` with the diff so no checkpoint
    regression slips through silently.

    Note on "subset vs equals": if a future sam3 release starts shipping
    convs[3] weights, the missing set shrinks (possibly to empty).  That is
    strictly safer — the released checkpoint now fully initialises the neck —
    so we accept it.  What we never accept is keys outside the known set going
    missing (could mean a renamed or restructurally different checkpoint) or
    any unexpected key appearing (could mean a model-architecture forward
    incompatibility).
    """
    if unexpected:
        return "fail"
    if not missing.issubset(_KNOWN_MISSING_KEYS):
        return "fail"
    return "ok"


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
      - ``forward`` supports both training (``model.train()``) and inference
        (``model.eval()``) modes.  The internal ``_Sam3ImageAdapter``
        hardcodes ``find_target=None`` when calling sam3's
        ``forward_grounding``; sam3's training-mode side-effect that would
        otherwise call ``back_convert(None)`` is neutralized by
        ``_patch_forward_grounding_skip_matching_on_none_target`` (installed
        by ``load_sam31``).  The trainer runs its own ``HungarianMatcher`` in
        ``custom_sam_peft.models.losses.total_loss``; ``out["indices"]`` written by
        sam3's matching call is never read by us.
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

    - ``local_dir=None`` raises ``FileNotFoundError`` with an `custom_sam_peft init` hint.
    - File present: return it (no Hub contact).
    - File missing: ``download_model(cfg.name, local_dir, revision=cfg.revision)``,
      then re-check. If the file is STILL missing post-download (e.g. the user
      pinned a revision that doesn't contain it), raise ``FileNotFoundError``
      with a precise diagnostic.
    """
    if cfg.local_dir is None:
        raise FileNotFoundError(
            f"ModelConfig.local_dir is None. Set it to a directory for "
            f"{cfg.checkpoint_file}, or run `custom_sam_peft init` to scaffold one."
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
        if getattr(submodule, "_custom_sam_peft_pos_enc_dtype_patched", False):
            continue
        original = submodule._encode_xy

        def _encode_xy_dtype_aware(self, x, y, _orig=original):  # type: ignore[no-untyped-def]
            pos_x, pos_y = _orig(x, y)
            return pos_x.to(dtype=x.dtype), pos_y.to(dtype=x.dtype)

        submodule._encode_xy = MethodType(_encode_xy_dtype_aware, submodule)
        submodule._custom_sam_peft_pos_enc_dtype_patched = True  # idempotency marker
        patched_count += 1

    logger.info(
        "Patched %d PositionEmbeddingSine._encode_xy callsites for dtype awareness.",
        patched_count,
    )


def _patch_roi_align_dtype() -> None:
    """Wrap ``torchvision.ops.roi_align`` to handle bf16-incompatible kernels and dtype skew.

    Two cooperating problems:

    1) **sam3 passes mismatched dtypes**: ``SequenceGeometryEncoder._encode_boxes``
       (``sam3/model/geometry_encoders.py:651-653``) calls
       ``torchvision.ops.roi_align`` with ``rois`` hard-cast to fp32 via ``.float()``,
       while ``img_feats`` is bf16 when the model is loaded under
       ``ModelConfig(dtype="bfloat16")``.  torchvision's C++ kernel requires both
       arguments to share dtype, so this raises a ``RuntimeError`` on Colab T4.

    2) **torchvision's CUDA roi_align kernel does not implement bfloat16**: even
       after matching dtypes, calling the kernel with bf16 input + bf16 boxes
       raises ``NotImplementedError: "roi_align_forward_kernel" not implemented for
       'BFloat16'`` (observed on torchvision 0.x shipped with torch 2.10 on Colab
       T4).  fp16 is supported; bf16 is not.  So simply casting boxes down to
       input dtype isn't sufficient — bf16 inputs need to be upcast to fp32 for
       the kernel call, then cast back.

    The wrapper handles both:
      - When ``input.dtype`` is bf16: upcast input and boxes to fp32, run kernel,
        cast output back to bf16.  Trivial precision cost (roi_align is a 7x7
        pooling; fp32 is more accurate than bf16 anyway).
      - When ``input.dtype`` is fp32/fp16: keep the original "match boxes to input"
        behavior so sam3's hardcoded ``.float()`` rois don't crash against fp16 input.

    We patch BOTH ``torchvision.ops.roi_align`` (the package-level re-export, used
    by sam3's functional call) AND ``torchvision.ops.roi_align.roi_align`` (the
    submodule-local name used by ``torchvision.ops.RoIAlign.forward``).  Without
    the submodule patch, ``RoIAlign`` instances (e.g.,
    ``sam3/model/decoder.py:289``) keep calling the un-patched original.  Single
    sentinel covers both rebindings (idempotent re-apply).

    Notes:
    - We do NOT introduce any ``torch.autocast`` scope; doing so re-triggers the
      bf16-vs-fp32 collision inside ``sam3/model/decoder.py::forward_ffn``'s
      ``with torch.amp.autocast(enabled=False)`` region — the same constraint
      that drove the cast-on-output approach adopted in PR #13.
    - The right long-term fix is upstream: either sam3 stops hardcoding
      ``.float()`` on rois, or torchvision adds a bf16 kernel.  Re-evaluate every
      torchvision/sam3 version bump.
    """
    import sys

    import torchvision.ops as tvo  # type: ignore[import-untyped]

    # ``torchvision.ops.__init__`` re-exports the ``roi_align`` FUNCTION under the
    # same name as the submodule, so ``import torchvision.ops.roi_align`` resolves
    # to the function, not the module.  Reach the actual submodule via
    # ``sys.modules`` so we can patch its module-local ``roi_align`` symbol — the
    # one that ``torchvision.ops.RoIAlign.forward`` resolves at call time.
    tvo_ra_mod = sys.modules["torchvision.ops.roi_align"]

    if getattr(tvo, "_custom_sam_peft_roi_align_dtype_patched", False):
        return
    _original = tvo.roi_align

    def _roi_align_dtype_aware(input, boxes, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_dtype = input.dtype
        # torchvision's CUDA roi_align kernel doesn't implement bf16.  Upcast
        # both input and boxes to fp32 for the kernel; cast output back below.
        if original_dtype == torch.bfloat16:
            input = input.float()
            if isinstance(boxes, (list, tuple)):
                boxes = type(boxes)(b.float() for b in boxes)
            elif hasattr(boxes, "float"):
                boxes = boxes.float()
        else:
            # input dtype is kernel-supported; match boxes to input.
            if isinstance(boxes, (list, tuple)):
                boxes = type(boxes)(
                    b.to(dtype=input.dtype) if b.dtype != input.dtype else b for b in boxes
                )
            elif hasattr(boxes, "dtype") and boxes.dtype != input.dtype:
                boxes = boxes.to(dtype=input.dtype)
        out = _original(input, boxes, *args, **kwargs)
        if out.dtype != original_dtype:
            out = out.to(dtype=original_dtype)
        return out

    tvo.roi_align = _roi_align_dtype_aware
    # Also patch the submodule-local symbol so torchvision.ops.RoIAlign.forward
    # (which looks up `roi_align` in its own module namespace) routes through
    # the wrapper too.
    tvo_ra_mod.roi_align = _roi_align_dtype_aware
    tvo._custom_sam_peft_roi_align_dtype_patched = True
    logger.info(
        "Patched torchvision.ops.roi_align (functional + RoIAlign class) "
        "for bf16-safe execution; bf16 inputs run via fp32 upcast."
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
    if getattr(model, "_custom_sam_peft_encode_prompt_dtype_patched", False):
        return
    original = model._encode_prompt
    target_dtype = next(model.parameters()).dtype

    def _encode_prompt_dtype_aware(self, *args, _orig=original, _dtype=target_dtype, **kwargs):  # type: ignore[no-untyped-def]
        prompt, prompt_mask, backbone_out = _orig(*args, **kwargs)
        if prompt.dtype != _dtype:
            prompt = prompt.to(dtype=_dtype)
        return prompt, prompt_mask, backbone_out

    model._encode_prompt = MethodType(_encode_prompt_dtype_aware, model)  # type: ignore[assignment]
    model._custom_sam_peft_encode_prompt_dtype_patched = True  # type: ignore[assignment]
    logger.info(
        "Patched SAM3Image._encode_prompt for dtype awareness (prompt cast to %s).",
        target_dtype,
    )


def _patch_text_pool_dtype() -> None:
    """Replace sam3's text-pooling helpers to honor prompt dtype instead of fp32.

    Two sites in sam3 build a validity mask via ``(~prompt_mask).float()`` and
    then do ``prompt * is_valid``, which promotes a bf16 ``prompt`` to fp32:

      - ``sam3.model.encoder.pool_text_feat`` (module-level function at
        ``encoder.py:583-595``).  Called from the encoder's text-pooling path.
      - ``sam3.model.model_misc.DotProductScoring.mean_pool_text`` (method on
        the class at ``model_misc.py:734-741``).  Called from
        ``DotProductScoring.forward`` before
        ``self.prompt_proj(pooled_prompt)``.

    Under QLoRA the pooled-prompt then flows into a ``Linear4bit`` whose
    compute_dtype is bf16, and the following ``torch.matmul`` at
    ``model_misc.py:761`` crashes with
    ``RuntimeError: expected scalar type BFloat16 but found Float``
    against the bf16 ``proj_hs``.  Under plain LoRA, the bf16 model handles
    fp32 input less violently but the result is still a silent precision
    cliff.

    Replace both with versions that cast ``is_valid`` to ``prompt.dtype``
    instead of fp32.  The computation is otherwise byte-identical (mean
    pool over valid tokens).  Module-level / class-level rebind (not
    per-instance) so the fix applies to every ``Sam3Image`` built in the
    process.  Idempotent via sentinels.

    Notes:
    - This is one of a small family of "hardcoded fp32 producer inside
      sam3's forward" patches; see also ``_patch_encode_prompt_dtype``,
      ``_patch_pos_enc_dtype``, ``_patch_roi_align_dtype``.  Each handles
      a specific site that an outer ``torch.autocast`` would otherwise
      paper over (autocast is off-limits in this codebase per PR #13;
      ``decoder.py::forward_ffn`` has an internal
      ``with torch.amp.autocast(enabled=False)`` region).
    - sam3 imports are lazy/try-except so CPU unit tests can exercise the
      patch behavior without the full sam3 install.
    - Re-evaluate every sam3 version bump.  Long-term fix is upstream:
      ``(~prompt_mask).to(dtype=prompt.dtype)`` instead of ``.float()``.
    """
    # encoder.pool_text_feat (module-level function)
    try:
        import sam3.model.encoder as _encoder_mod

        if not getattr(_encoder_mod, "_custom_sam_peft_pool_text_feat_dtype_patched", False):

            def _pool_text_feat_dtype_aware(prompt, prompt_mask, pool_with_mask):  # type: ignore[no-untyped-def]
                if not pool_with_mask:
                    return prompt.mean(dim=0)
                if prompt_mask.dim() != 2:
                    raise ValueError(
                        f"pool_text_feat: prompt_mask.dim() must be 2; got {prompt_mask.dim()}"
                    )
                is_valid = (~prompt_mask).to(dtype=prompt.dtype).permute(1, 0)[..., None]
                num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
                pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
                return pooled_text

            _encoder_mod.pool_text_feat = _pool_text_feat_dtype_aware
            _encoder_mod._custom_sam_peft_pool_text_feat_dtype_patched = True  # type: ignore[attr-defined]
            logger.info("Patched sam3.model.encoder.pool_text_feat for dtype-aware text pooling.")
    except ImportError:
        pass

    # DotProductScoring.mean_pool_text (method on a class)
    try:
        import sam3.model.model_misc as _mm_mod

        DotProductScoring = _mm_mod.DotProductScoring
        if not getattr(DotProductScoring, "_custom_sam_peft_mean_pool_text_dtype_patched", False):

            def _mean_pool_text_dtype_aware(self, prompt, prompt_mask):  # type: ignore[no-untyped-def]
                is_valid = (~prompt_mask).to(dtype=prompt.dtype).permute(1, 0)[..., None]
                num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
                pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
                return pooled_prompt

            DotProductScoring.mean_pool_text = _mean_pool_text_dtype_aware
            DotProductScoring._custom_sam_peft_mean_pool_text_dtype_patched = True  # type: ignore[attr-defined]
            logger.info(
                "Patched sam3.model.model_misc.DotProductScoring.mean_pool_text "
                "for dtype-aware text pooling."
            )
    except ImportError:
        pass


def _patch_addmm_act_grad_safe() -> None:
    """Make sam3's ``addmm_act`` fused kernel grad-aware so LoRA training works.

    sam3 ships an inference-only fused matmul helper
    (``sam3/perflib/fused.py::addmm_act``) used by every ViT-Det MLP block's
    ``forward`` (``sam3/model/vitdet.py:71`` calls it as
    ``addmm_act(type(self.act), self.fc1, x)``).  The helper hard-rejects
    grad-enabled callers and explicitly detaches ``linear.weight`` /
    ``linear.bias`` — both correct optimizations at inference, both fatal
    for LoRA fine-tuning:

        def addmm_act(activation, linear, mat1):
            if torch.is_grad_enabled():
                raise ValueError("Expected grad to be disabled.")
            self = linear.bias.detach()
            mat2 = linear.weight.detach()
            ...

    LoRA adapters attach to ``fc1`` (and other backbone Linears).  Their
    backward path requires grad to flow through ``linear.weight`` so the
    chain rule reaches the adapter matrices, but every MLP block hits
    ``addmm_act`` first and raises ``ValueError("Expected grad to be
    disabled.")``.  ``train_step``'s ``except ValueError`` clause counts
    each raise as a non-finite micro-step, and after ``nan_abort_after``
    consecutive raises the trainer aborts.  Net effect: SAM 3.1 fine-tuning
    on GPU is impossible against this sam3 commit without this patch —
    every gpu-marked training test fails this way (caught during the
    manual GPU pass for issue #44).

    Fix: replace ``addmm_act`` with a wrapper that branches on
    ``torch.is_grad_enabled()``:
      - grad-enabled  (training)  : ``F.{relu,gelu}(linear(mat1))`` —
        plain, grad-tracking ``linear`` + activation.  Negligible perf
        cost on T4 since cuBLAS already fuses addmm internally.
      - grad-disabled (inference) : delegate to sam3's original fused
        kernel for full perf parity.

    The patch must update BOTH ``sam3.perflib.fused.addmm_act`` (the
    definition) AND ``sam3.model.vitdet.addmm_act`` (vitdet does
    ``from sam3.perflib.fused import addmm_act`` at import time, copying
    the function object into its own module namespace; later updates to
    ``perflib.fused.addmm_act`` alone do NOT reach vitdet's binding).

    Notes:
    - Module-level monkey-patch (not per-instance) since the call site is
      inside sam3's installed package, not a sub-module of our wrapper.
    - Activation support mirrors sam3's original kernel (ReLU and GELU
      class- or functional-form only); other activations raise the same
      ``ValueError`` as upstream — surfaces incompatibility loudly if
      future sam3 versions add new activations.
    - Idempotency sentinel mirrors the other ``_patch_*`` helpers.
    - The right long-term fix is upstream: sam3's ``addmm_act`` should
      branch on grad state itself.  Re-evaluate every sam3 version bump.
    """
    import sam3.model.vitdet as _vd
    import sam3.perflib.fused as _pf
    import torch.nn.functional as F

    if getattr(_pf, "_custom_sam_peft_addmm_act_grad_safe_patched", False):
        return

    _orig = _pf.addmm_act

    def _addmm_act_grad_safe(activation, linear, mat1):  # type: ignore[no-untyped-def]
        if not torch.is_grad_enabled():
            return _orig(activation, linear, mat1)
        x = linear(mat1)
        if activation in (nn.ReLU, F.relu):
            return F.relu(x)
        if activation in (nn.GELU, F.gelu):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    _pf.addmm_act = _addmm_act_grad_safe
    _vd.addmm_act = _addmm_act_grad_safe
    _pf._custom_sam_peft_addmm_act_grad_safe_patched = True  # type: ignore[attr-defined]
    logger.info(
        "Patched sam3.perflib.fused.addmm_act (and vitdet binding) for "
        "grad-aware forward; LoRA backbone fine-tuning now works on this "
        "sam3 commit."
    )


def _patch_forward_grounding_skip_matching_on_none_target(model: nn.Module) -> None:
    """Neutralize sam3's training-mode matching side-effect when ``find_target`` is ``None``.

    sam3's ``Sam3Image.forward_grounding`` (``sam3/model/sam3_image.py:440-496``)
    runs an extra side-effect when the model is in train mode (or interactive
    eval mode)::

        if self.training or self.num_interactive_steps_val > 0:
            self._compute_matching(out, self.back_convert(find_target))

    The call writes ``out["indices"]`` (and into each ``aux_outputs[*]``) using
    sam3's own matcher on a ``BatchedFindTarget``-shaped ``find_target``.  Our
    ``_Sam3ImageAdapter`` does not carry a ``BatchedFindTarget`` and passes
    ``find_target=None``; ``back_convert(None)`` then dereferences
    ``None.boxes`` and raises ``AttributeError``
    (``sam3/model/sam3_image.py:610``).  The crash is silent during inference
    because the gate above is ``False`` in eval mode (the inspection-tier GPU
    tests pass on a plain forward in eval), and only surfaces under
    training-mode calls from ``train_step`` / ``run_epoch`` and the
    eval/forward/train dance in ``Trainer._log_image_panel``.

    Our trainer runs its OWN ``HungarianMatcher`` against
    ``list[list[Instance]]`` targets inside
    ``custom_sam_peft.models.losses.total_loss``, and never reads ``out["indices"]``
    written by sam3's matching call (grep verified — the only ``.indices`` hit
    in ``src/`` is ``torch.topk(...).indices`` in ``trainer._log_image_panel``,
    unrelated).  The upstream side-effect is pure waste from our perspective
    AND it crashes on the ``None`` we pass.  This patch short-circuits both
    halves of that side-effect when ``find_target is None``:

      - ``back_convert(targets)``        : returns ``None`` when
        ``targets is None``; delegates to the original implementation
        otherwise.
      - ``_compute_matching(out, targets)`` : no-op when ``targets is None``;
        delegates to the original implementation otherwise.

    Behavior preserved: ``self.training`` stays ``True``; every other
    training-mode branch inside ``forward_grounding`` (DAC at
    ``sam3_image.py:266,310,397``, aux-output population at lines
    ``341,355,358,364,371,377,383``, the o2m head at ``366``, and activation
    checkpointing in the seg head at ``407``) fires normally.  The only
    suppressed work is the side-effect on ``out["indices"]`` that we replace
    via our own matcher downstream.  Non-``None`` ``find_target`` paths are
    untouched, so eval-time runs with ``num_interactive_steps_val > 0`` and a
    real target keep sam3-native matching intact.

    Notes:
    - Per-instance ``MethodType`` rebind (mirrors the other ``_patch_*``
      helpers); idempotency sentinel
      ``_custom_sam_peft_skip_matching_on_none_target_patched``.
    - Models without ``back_convert`` / ``_compute_matching`` are skipped
      (no-op), so unit tests with stand-ins that omit the methods do not
      raise.
    - The right long-term fix is upstream: ``forward_grounding`` should
      tolerate ``find_target=None`` natively.  Re-evaluate every sam3
      version bump.
    """
    from types import MethodType

    if getattr(model, "_custom_sam_peft_skip_matching_on_none_target_patched", False):
        return
    if not hasattr(model, "back_convert") or not hasattr(model, "_compute_matching"):
        return

    orig_back_convert = model.back_convert
    orig_compute_matching = model._compute_matching

    def _back_convert_none_safe(self, targets, _orig=orig_back_convert):  # type: ignore[no-untyped-def]
        if targets is None:
            return None
        return _orig(targets)

    def _compute_matching_none_safe(self, out, targets, _orig=orig_compute_matching):  # type: ignore[no-untyped-def]
        if targets is None:
            return
        return _orig(out, targets)

    model.back_convert = MethodType(_back_convert_none_safe, model)  # type: ignore[assignment]
    model._compute_matching = MethodType(_compute_matching_none_safe, model)  # type: ignore[assignment]
    model._custom_sam_peft_skip_matching_on_none_target_patched = True  # type: ignore[assignment]
    logger.info(
        "Patched sam3.Sam3Image.{back_convert,_compute_matching} to short-circuit "
        "on find_target=None; training-mode forward_grounding now bypasses sam3's "
        "internal matching side-effect (we run our own matcher in losses.total_loss)."
    )


def _patch_mha_input_dtype(model: nn.Module) -> None:
    """Cast ``query``/``key``/``value`` of every MHA module to the MHA's weight dtype.

    ``_patch_module_input_dtype`` only hooks ``nn.Linear`` / ``nn.LayerNorm`` /
    ``nn.Conv*`` and only casts the first positional arg.  It deliberately
    skips ``nn.MultiheadAttention`` because MHA takes three tensor inputs
    (query, key, value) with a shared dtype expectation that the simple
    args[0]-only hook can't honor.  But that leaves MHA modules completely
    unprotected against upstream dtype leaks: any fp32 tensor reaching the
    MHA's ``F.linear(input, in_proj_weight, ...)`` path (inside torch's
    ``_in_projection_packed`` or sam3's ``multi_head_attention_forward``)
    collides with the bf16 weight and raises
    ``RuntimeError: mat1 and mat2 must have the same dtype, but got Float
    and BFloat16``.

    Surfaced in the QLoRA release-tier test after the prior dtype-routing
    fixes: with the model fully bf16 (our skip of
    ``prepare_model_for_kbit_training``) and the weight side clean, an
    upstream fp32 promotion (likely a positional-embedding cast or a
    transient mixed-precision op in the decoder pipeline) ended up feeding
    MHA with fp32 query/key while value/weight were bf16.

    This hook is the symmetric backstop: cast every floating-point
    query/key/value input to the MHA's first-parameter dtype (which is the
    compute_dtype the MHA was set up for).  Both positional and keyword
    forms are handled.  Other args (masks, scalars, ``need_weights``, etc.)
    pass through untouched.

    Applies to both ``torch.nn.MultiheadAttention`` and
    ``sam3.model.model_misc.MultiheadAttention`` (the same custom class
    already enumerated in ``custom_sam_peft.peft_adapters.qlora._mha_exclusion_types``).
    sam3's custom MHA is imported lazily so the patch degrades gracefully
    when sam3 is unavailable (CPU unit tests).

    Notes:
    - Per-instance ``register_forward_pre_hook`` (with ``with_kwargs=True``);
      idempotency sentinel ``_custom_sam_peft_mha_input_dtype_patched``.
    - Hook fires before sam3's ``with torch.amp.autocast(enabled=False)``
      regions (which live INSIDE the MHA call), so we don't collide with
      the constraint that drives every other ``_patch_*_dtype`` helper.
    - Re-evaluate every sam3 version bump; track long-term fix in upstream
      sam3 (or wait for MHA to natively support mixed-dtype inputs).
    """
    mha_types: tuple[type[nn.Module], ...] = (nn.MultiheadAttention,)
    try:
        from sam3.model.model_misc import MultiheadAttention as _Sam3CustomMHA

        mha_types = (*mha_types, _Sam3CustomMHA)
    except ImportError:
        pass

    def _mha_input_dtype_hook(module, args, kwargs):  # type: ignore[no-untyped-def]
        try:
            target_dtype = next(module.parameters()).dtype
        except StopIteration:
            return None
        # Positional: args[0..2] are query/key/value in both torch and sam3 MHA.
        new_args = list(args)
        for i in range(min(3, len(new_args))):
            t = new_args[i]
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != target_dtype:
                new_args[i] = t.to(dtype=target_dtype)
        # Keyword: same three names. Other kwargs (masks, scalars) untouched.
        new_kwargs = dict(kwargs)
        for name in ("query", "key", "value"):
            t = new_kwargs.get(name)
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != target_dtype:
                new_kwargs[name] = t.to(dtype=target_dtype)
        return tuple(new_args), new_kwargs

    patched_count = 0
    for submodule in model.modules():
        if not isinstance(submodule, mha_types):
            continue
        if getattr(submodule, "_custom_sam_peft_mha_input_dtype_patched", False):
            continue
        submodule.register_forward_pre_hook(_mha_input_dtype_hook, with_kwargs=True)
        submodule._custom_sam_peft_mha_input_dtype_patched = True  # type: ignore[assignment]
        patched_count += 1

    logger.info(
        "Patched %d MultiheadAttention modules with query/key/value input-dtype hook.",
        patched_count,
    )


# Modules that own a `weight` parameter and require their floating-point
# input to match that weight's dtype. We hook the forward pre-call on each
# instance to cast input[0] in-flight. Embedding is intentionally excluded
# (integer input). Attention-style fused modules (nn.MultiheadAttention and
# sam3.model.model_misc.MultiheadAttention) are handled separately by
# _patch_mha_input_dtype, which casts query/key/value (three positional
# tensors) rather than just args[0].
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
        if getattr(submodule, "_custom_sam_peft_module_input_dtype_patched", False):
            continue
        submodule.register_forward_pre_hook(_input_dtype_hook)
        submodule._custom_sam_peft_module_input_dtype_patched = True  # type: ignore[assignment]
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

    # sam3's _load_checkpoint (model_builder.py:539-561) calls print() to
    # stdout when load_state_dict reports missing keys.  We capture stdout
    # during the build call so we can inspect the message and either suppress
    # the known-harmless convs.3 noise or raise loudly on anything unexpected.
    _captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(_captured_stdout):
        raw_model = sam3.build_sam3_image_model(
            device=device,
            eval_mode=False,  # training mode — gradients flow.
            checkpoint_path=str(ckpt_path),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )

    # Process any captured output from sam3's checkpoint loader.
    # sam3's _load_checkpoint (model_builder.py:557-561) calls:
    #   print(f"loaded {checkpoint_path} and found "
    #         f"missing and/or unexpected keys:\n{missing_keys=}")
    # That produces a two-line block; _SAM3_MISSING_KEYS_RE (re.DOTALL) matches
    # across the embedded newline so we search the full captured text at once.
    _stdout_text = _captured_stdout.getvalue()
    if _stdout_text:
        _remaining = _stdout_text
        while _remaining:
            _m = _SAM3_MISSING_KEYS_RE.search(_remaining)
            if _m is None:
                # No (further) missing-keys block — pass the rest through.
                sys.stdout.write(_remaining)
                break
            # Pass through any text that precedes the match unchanged.
            _before = _remaining[: _m.start()]
            if _before:
                sys.stdout.write(_before)
            # Parse the missing_keys list from the repr.
            # ast.literal_eval rejects anything that is not a Python literal,
            # so a maliciously crafted checkpoint whose state_dict key names
            # contain embedded code cannot cause arbitrary code execution here.
            _raw_repr = _m.group(1).strip()
            try:
                _missing: set[str] = set(ast.literal_eval(_raw_repr))
            except (ValueError, SyntaxError):  # pragma: no cover — defensive; repr is always a list
                sys.stdout.write(_m.group(0))
                _remaining = _remaining[_m.end() :]
                continue
            _verdict = _classify_missing_keys(_missing, unexpected=set())
            if _verdict == "ok":
                logger.debug(
                    "Suppressed known-harmless missing-keys noise from sam3 checkpoint "
                    "loader (convs.3 weights absent from released 3-scale checkpoint; "
                    "scalp=1 in vl_combiner drops convs[3] output — no impact on "
                    "training or inference).  Missing: %s",
                    sorted(_missing),
                )
            else:
                raise RuntimeError(
                    f"sam3 checkpoint load_state_dict reported unexpected deviation.\n"
                    f"  missing keys  : {sorted(_missing)}\n"
                    f"  keys outside known-harmless set: "
                    f"{sorted(_missing - _KNOWN_MISSING_KEYS)}\n"
                    f"Re-check sam3 version and update _KNOWN_MISSING_KEYS in "
                    f"src/custom_sam_peft/models/sam3.py if intentional."
                )
            _remaining = _remaining[_m.end() :]

    if cfg.gradient_checkpointing:
        if hasattr(raw_model, "set_grad_checkpointing"):
            raw_model.set_grad_checkpointing(True)
        else:
            # sam3 ships per-ViT-Det-block use_act_checkpoint flags but
            # flipping them on under non-reentrant torch.utils.checkpoint
            # produces a recompute-vs-original metadata mismatch
            # (CheckpointError). Likely interaction with sam3's internal
            # `with torch.amp.autocast(enabled=False)` regions and the
            # global no-outer-autocast constraint. Tracked in issue #60.
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

    # Replace sam3's text-pooling helpers (encoder.pool_text_feat and
    # DotProductScoring.mean_pool_text) so their internal (~mask).float()
    # promotion is replaced by (~mask).to(prompt.dtype). Otherwise bf16
    # prompts get promoted to fp32 mid-forward and crash downstream
    # Linear4bit / matmul ops with "expected BFloat16 but found Float".
    # See _patch_text_pool_dtype for full rationale.
    _patch_text_pool_dtype()

    # Make sam3's inference-only addmm_act fused kernel grad-aware so the
    # ViT-Det backbone supports LoRA fine-tuning. Without this, every
    # MLP block raises ValueError on grad-enabled forward and training
    # aborts after nan_abort_after consecutive non-finite micro-steps.
    # See _patch_addmm_act_grad_safe for full rationale.
    _patch_addmm_act_grad_safe()

    # Neutralize sam3's training-mode matching side-effect when find_target
    # is None. _Sam3ImageAdapter passes find_target=None; sam3's
    # forward_grounding would otherwise call back_convert(None) -> crash.
    # We run our own HungarianMatcher in losses.total_loss; sam3's matcher
    # output (out["indices"]) is never read by us.
    # See _patch_forward_grounding_skip_matching_on_none_target for full rationale.
    _patch_forward_grounding_skip_matching_on_none_target(raw_model)

    # Generic backstop: cast fp inputs to weight dtype at every
    # nn.Linear/LayerNorm/Conv* in the model. Catches any remaining or
    # future cascading fp32 producer site we haven't patched directly.
    # See _patch_module_input_dtype for full rationale.
    _patch_module_input_dtype(raw_model)

    # MHA-specific backstop: same idea as _patch_module_input_dtype but for
    # nn.MultiheadAttention and sam3.model.model_misc.MultiheadAttention,
    # which take three tensor inputs (query/key/value) and so are excluded
    # from the generic hook. Without this, any upstream fp32 promotion
    # reaches MHA's internal F.linear and crashes against the bf16 weight.
    # See _patch_mha_input_dtype for full rationale.
    _patch_mha_input_dtype(raw_model)

    adapter = _Sam3ImageAdapter(raw_model, image_size=1008)
    return Sam3Wrapper(adapter, image_size=1008, mask_size=288)
