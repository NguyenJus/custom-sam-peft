"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md.

Revised by docs/superpowers/plans/2026-05-16-model-loading-revised.md to match
Meta's open-vocab head: supports 1..MULTIPLEX_CAP class prompts per forward call.
All prompts in a batch must share the same class list in the same order.
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
    n_cols: int,
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
    if len(box_hints) != n_cols:
        raise ValueError(f"len(box_hints)={len(box_hints)} must equal n_cols={n_cols}")

    if all(h is None for h in box_hints):
        return None

    b = n_cols
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


# SAM 3.1's multiplex forward is trained at K ≤ 16 class prompts per call.
# This is a model property, not a tunable. Trainer/evaluator/predict cite
# this constant for chunking; see docs/superpowers/specs/2026-05-23-multiplex-forward-design.md §4.
MULTIPLEX_CAP: int = 16


class Sam3Wrapper(nn.Module):
    """Thin wrapper around Meta's SAM 3.1 model.

    Contract:
      - ``forward(images, prompts, box_hints=None)`` accepts a batch of B images
        and a list of B ``Prompts`` objects, one per image.
      - ``box_hints``: optional flat list of length ``B·K``, ordered image-major /
        class-minor (i.e. all K class slots for image 0, then all K class slots for
        image 1, …).  Each element is either ``None`` (no geometric hint for that
        slot) or a ``(M_i, 4)`` float tensor of absolute pixel xyxy boxes.
        ``box_hints`` must not be combined with ``BoxPrompts`` (they carry
        conflicting localization signals).  For the common K=1 case the list
        length equals B and the ordering is trivially image-major.
      - All prompts in a batch MUST be the same variant (TextPrompts XOR
        BoxPrompts); the wrapper raises on mixed batches.
      - For TextPrompts, each image's prompt may contain 1..MULTIPLEX_CAP
        class names; all prompts in a batch must share the same class list
        in the same order (multiplex forward assumes a shared K-prompt
        vocabulary).
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

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 1008,
        mask_size: int = 288,
        *,
        channels: int = 3,
        channel_semantics: str = "rgb",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size
        self.channels = channels
        self.channel_semantics = channel_semantics
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

    def _validate_inputs(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must be (B, {self.channels}, H, W); got shape {tuple(images.shape)}"
            )
        if images.shape[1] != self.channels:
            raise ValueError(
                f"images must be (B, {self.channels}, H, W); got "
                f"{images.shape[1]} channels in shape {tuple(images.shape)}"
            )
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
            if isinstance(p, TextPrompts) and not (1 <= len(p.classes) <= MULTIPLEX_CAP):
                raise ValueError(
                    f"TextPrompts must contain 1..MULTIPLEX_CAP (={MULTIPLEX_CAP}) classes per "
                    f"call (got {len(p.classes)}). Configure "
                    f"train.multiplex.classes_per_forward to bound K."
                )

        # After the per-prompt loop, enforce shared class list for TextPrompts.
        if first is TextPrompts:
            ref = tuple(cast(TextPrompts, prompts[0]).classes)
            for p in prompts[1:]:
                if tuple(cast(TextPrompts, p).classes) != ref:
                    raise ValueError(
                        "All TextPrompts in a batch must carry the same class "
                        "list in the same order (multiplex forward assumes a "
                        "shared K-prompt vocabulary)."
                    )

        if box_hints is not None:
            if first is BoxPrompts:
                raise ValueError(
                    "box_hints must not be combined with BoxPrompts prompts. "
                    "BoxPrompts already carry localization information."
                )
            # For TextPrompts, box_hints must be length B*K (image-major / class-minor).
            # For other prompt types (currently only BoxPrompts, guarded above), length B.
            if first is TextPrompts and prompts:
                k = len(cast(TextPrompts, prompts[0]).classes)
                expected_len = b * k
            else:
                expected_len = b
            if len(box_hints) != expected_len:
                raise ValueError(
                    f"len(box_hints)={len(box_hints)} must equal batch size x classes "
                    f"({b}x{expected_len // b if b else 1}={expected_len})"
                )
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


def _build_channel_adapter(channels: int, channel_semantics: str) -> nn.Conv2d | None:
    """Build the N->3 channel adapter per the semantic profile (spec §5.2/§5.3).

    Returns None for semantic=='rgb' (passthrough, zero new params). Otherwise a
    fully-trainable Conv2d(channels, 3, 1) initialized per profile.adapter_init.
    """
    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS

    profile = CHANNEL_SEMANTICS[channel_semantics]
    if not profile.use_adapter:
        return None
    conv = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
    bias: Tensor = cast(Tensor, conv.bias)  # bias=True guarantees non-None
    with torch.no_grad():
        conv.weight.zero_()
        bias.zero_()
        if profile.adapter_init == "average_broadcast":
            conv.weight.fill_(1.0 / channels)
        elif profile.adapter_init == "identity_passthrough":
            # Identity on first 3 input channels, zero on the rest.
            for o in range(3):
                if o < channels:
                    conv.weight[o, o, 0, 0] = 1.0
        else:  # pragma: no cover - registry guards this
            raise ValueError(f"unknown adapter_init: {profile.adapter_init!r}")
    conv.weight.requires_grad_(True)
    bias.requires_grad_(True)
    return conv


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

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 1008,
        *,
        channels: int = 3,
        channel_semantics: str = "rgb",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.channel_semantics = channel_semantics
        self.channel_adapter = _build_channel_adapter(channels, channel_semantics)
        # Cast the adapter to match the inner model's parameter dtype so that a
        # bf16/fp16 model doesn't raise dtype mismatch on the first Conv2d call.
        # _apply_dtype casts only the raw inner model; the adapter is built after
        # that step and would otherwise default to float32.
        if self.channel_adapter is not None:
            _first_param = next(model.parameters(), None)
            if _first_param is not None:
                self.channel_adapter = self.channel_adapter.to(dtype=_first_param.dtype)

    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        if not all(isinstance(p, TextPrompts) for p in prompts):
            raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
        text_prompts = cast(list[TextPrompts], prompts)
        # Sam3Wrapper._validate_inputs guarantees a shared class list across the batch.
        classes = list(text_prompts[0].classes)
        k = len(classes)
        device = images.device
        b = images.shape[0]
        model_dtype = next(self.model.parameters()).dtype

        if self.channel_adapter is not None:
            # Cast input to the adapter's weight dtype first (mirrors the inner
            # model's module_input_dtype patch, which doesn't cover our adapter).
            # Only the input tensor is cast — the module itself is never re-cast
            # inside forward, which would break the optimizer's parameter tracking.
            images = images.to(dtype=self.channel_adapter.weight.dtype)
            images = self.channel_adapter(images)  # (B, N, H, W) -> (B, 3, H, W)

        backbone_out = self.model.backbone.forward_image(images)  # type: ignore[union-attr, operator]
        text_outputs = self.model.backbone.forward_text(  # type: ignore[union-attr, operator]
            classes, device=device
        )
        backbone_out.update(text_outputs)

        # Multiplex assembly: B images x K classes -> B*K rows, image-major / class-minor.
        # img_ids  = [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]  (each repeated K times)
        # text_ids = [0,1,...,K-1, 0,1,...,K-1, ..., 0,1,...,K-1]  (repeated B times)
        n_cols = b * k
        find_input = FindStage(
            img_ids=torch.arange(b, device=device, dtype=torch.long).repeat_interleave(k),
            text_ids=torch.arange(k, device=device, dtype=torch.long).repeat(b),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        gp = _build_geometric_prompt(
            box_hints if box_hints is not None else [None] * n_cols,
            n_cols=n_cols,
            image_size=self.image_size,
            device=device,
        )
        if gp is None:
            gp = Prompt(
                box_embeddings=torch.zeros(0, n_cols, 4, device=device, dtype=model_dtype),
                box_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
                point_embeddings=torch.zeros(0, n_cols, 2, device=device, dtype=model_dtype),
                point_mask=torch.zeros(n_cols, 0, device=device, dtype=torch.bool),
            )
        outputs: dict[str, Tensor] = self.model.forward_grounding(  # type: ignore[operator]
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=gp,
        )
        return outputs


# ---------------------------------------------------------------------------
# Patch delegation wrappers — thin shims that keep the original _patch_*
# names importable from this module (used by existing per-patch unit tests)
# while the real implementations live under models/_patches/.
# ---------------------------------------------------------------------------


def _patch_pos_enc_dtype(model: nn.Module) -> None:
    """Delegate to models/_patches/pos_enc_dtype.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import pos_enc_dtype as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(model, Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _patch_roi_align_dtype() -> None:
    """Delegate to models/_patches/roi_align_dtype.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import roi_align_dtype as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(nn.Module(), Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _patch_encode_prompt_dtype(model: nn.Module) -> None:
    """Delegate to models/_patches/encode_prompt_dtype.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import encode_prompt_dtype as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(model, Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _patch_text_pool_dtype() -> None:
    """Delegate to models/_patches/text_pool_dtype.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import text_pool_dtype as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(nn.Module(), Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _is_linear4bit(module: nn.Module) -> bool:
    """Return True when *module* is a bitsandbytes ``Linear4bit`` instance.

    The bitsandbytes import is lazy so this helper stays usable on CPU test
    environments where bitsandbytes is not installed.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    Linear4bit = getattr(bnb.nn, "Linear4bit", None)
    return Linear4bit is not None and isinstance(module, Linear4bit)


def _apply_activation(activation: Any, x: torch.Tensor) -> torch.Tensor:
    """Apply *activation* to *x*, matching sam3's supported activation set.

    sam3's ``addmm_act`` (``sam3/perflib/fused.py``) supports exactly four
    forms — ``torch.nn.functional.relu``, ``torch.nn.ReLU``,
    ``torch.nn.functional.gelu``, ``torch.nn.GELU``.  We mirror that set;
    anything else raises ``ValueError`` loudly (same contract as upstream).
    """
    import torch.nn.functional as F

    if activation in (nn.ReLU, F.relu):
        return F.relu(x)
    if activation in (nn.GELU, F.gelu):
        return F.gelu(x)
    raise ValueError(f"_addmm_act_grad_safe: unsupported activation {activation!r}")


def _patch_addmm_act_grad_safe() -> None:
    """Delegate to models/_patches/addmm_act_grad_safe.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import addmm_act_grad_safe as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(nn.Module(), Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _patch_forward_grounding_skip_matching_on_none_target(model: nn.Module) -> None:
    """Delegate to models/_patches/forward_grounding_skip_matching.apply."""
    from custom_sam_peft.models._patches import forward_grounding_skip_matching as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(model, Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _patch_mha_input_dtype(model: nn.Module) -> None:
    """Delegate to models/_patches/mha_input_dtype.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import mha_input_dtype as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(model, Runtime(device=torch.device("cpu"), dtype=torch.float32))


# Modules that own a `weight` parameter and require their floating-point
# input to match that weight's dtype. Referenced by models/_patches/module_input_dtype.py.
_DTYPE_SENSITIVE_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.LayerNorm,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)


def _patch_module_input_dtype(model: nn.Module) -> None:
    """Delegate to models/_patches/module_input_dtype.apply (see that module for rationale)."""
    from custom_sam_peft.models._patches import module_input_dtype as _m
    from custom_sam_peft.runtime._runtime import Runtime

    _m.apply(model, Runtime(device=torch.device("cpu"), dtype=torch.float32))


def _locate_weights(cfg: ModelConfig) -> Path:
    """Resolve the checkpoint path from config (HF / local / cache).

    Delegates to ``_resolve_checkpoint_path``; exists as a named seam so
    ``load_sam31`` reads as a linear orchestration shell.
    """
    return _resolve_checkpoint_path(cfg)


def _construct_raw_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate Sam3Image from the checkpoint, suppressing known-harmless stdout noise.

    sam3's ``_load_checkpoint`` (model_builder.py:539-561) calls ``print()`` to
    stdout when ``load_state_dict`` reports missing keys.  We capture stdout
    during the build call, filter out the known-harmless convs.3 noise, and
    re-emit only unrecognised content.  Unrecognised missing keys raise
    ``RuntimeError`` loudly so checkpoint regressions don't slip through silently.
    """
    ckpt_path = _locate_weights(cfg)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    assert isinstance(raw_model, nn.Module)  # noqa: S101
    return raw_model


def _apply_dtype(model: nn.Module, cfg: ModelConfig) -> nn.Module:
    """Cast model parameters to the dtype specified in config (in-place, returns model).

    Returns the model reference (which may be a new object for non-in-place casts,
    though ``nn.Module.to`` modifies in-place and returns self).
    """
    if cfg.dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    elif cfg.dtype == "float16":
        model = model.to(dtype=torch.float16)
    return model


def _apply_patches(model: nn.Module) -> None:
    """Apply all dtype-correctness patches to *model* via ``Sam3Patches.apply``.

    Builds a ``Runtime`` from the model's current parameter dtype, then
    delegates to ``Sam3Patches.apply`` which iterates ``_ALL_PATCHES``
    (populated in Task 5.7 with one entry per patch module under
    ``models/_patches/``).
    """
    from custom_sam_peft.runtime import Sam3Patches
    from custom_sam_peft.runtime._runtime import Runtime

    _dtype_str = "float32"
    try:
        target_dtype = next(model.parameters()).dtype
        _dtype_map_inv = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        _dtype_str = _dtype_map_inv.get(target_dtype, "float32")
    except StopIteration:
        pass
    runtime = Runtime.from_config(device="cpu", dtype=_dtype_str)
    Sam3Patches.apply(model, runtime)


def _freeze_base(model: nn.Module, peft_method: Any) -> None:
    """Freeze base model parameters, leaving adapter params trainable.

    Currently a no-op seam: freezing is done by the PEFT adapter factories
    (``peft_adapters.lora``, ``peft_adapters.qlora``) after ``load_sam31``
    returns.  This helper exists so the orchestration shell in ``load_sam31``
    has a named step for the future case where we want to centralize freezing
    here (e.g. for a "no-PEFT fine-tune" mode).

    Precondition: ``peft_method`` is ``None`` when called from the current
    ``load_sam31`` path; the argument is reserved for callers that may pass a
    ``PEFTMethod`` in the future.
    """
    # No-op: PEFT adapters handle freezing post-load.


def load_sam31(
    cfg: ModelConfig,
    *,
    channels: int = 3,
    channel_semantics: str = "rgb",
) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's `sam3` package and wrap it for our trainer.

    Returns a `Sam3Wrapper` whose `forward(images, prompts, box_hints=None)` returns Meta's
    native per-class output dict (`pred_logits`, `pred_boxes`, `pred_masks`,
    `presence_logit_dec`).
    """
    raw_model = _construct_raw_model(cfg)
    raw_model = _apply_dtype(raw_model, cfg)
    _apply_patches(raw_model)
    _freeze_base(raw_model, peft_method=None)

    adapter = _Sam3ImageAdapter(
        raw_model, image_size=1008, channels=channels, channel_semantics=channel_semantics
    )
    return Sam3Wrapper(
        adapter,
        image_size=1008,
        mask_size=288,
        channels=channels,
        channel_semantics=channel_semantics,
    )
