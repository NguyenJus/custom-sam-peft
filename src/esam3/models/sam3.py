"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md.

Revised by docs/superpowers/plans/2026-05-16-model-loading-revised.md to match
Meta's open-vocab head: one prompt class per forward call. Trainer loops over
the fixed class vocabulary externally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import PeftModel

import sam3
import torch
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.geometry_encoders import Prompt
from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import BoxPrompts, Prompts, TextPrompts

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
    if cfg.local_dir is None:
        raise FileNotFoundError(
            "ModelConfig.local_dir is None and Hub fetch is not implemented. "
            f"Set local_dir to a directory containing {cfg.checkpoint_file}. "
            f"To download: `huggingface-cli download {cfg.name} --local-dir models/sam3.1`."
        )
    path = Path(cfg.local_dir) / cfg.checkpoint_file
    if not path.exists():
        raise FileNotFoundError(
            f"SAM 3.1 checkpoint not found at {path}. "
            f"Run: huggingface-cli download {cfg.name} --local-dir {cfg.local_dir}"
        )
    return path


def _resolve_bpe_path(cfg: ModelConfig) -> Path:
    """The BPE merges file is shipped alongside the checkpoint in the HF repo."""
    if cfg.local_dir is None:
        raise FileNotFoundError("ModelConfig.local_dir is None; cannot resolve BPE path.")
    path = Path(cfg.local_dir) / "merges.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"SAM 3.1 BPE merges file not found at {path}. Re-download the checkpoint "
            f"directory from {cfg.name}."
        )
    return path


class _Sam3ImageAdapter(nn.Module):
    """Adapt raw Sam3Image to the (images, prompts, **kwargs) calling convention.

    Sam3Image's training-mode forward (``forward_grounding``) expects
    ``(backbone_out, find_input, find_target, geometric_prompt)``, none of which
    are raw image tensors or our ``Prompts`` dataclasses.  This adapter holds the
    inner ``Sam3Image`` and orchestrates the conversion.

    TODO (Task 4 / DONE_WITH_CONCERNS): The body below raises ``NotImplementedError``
    because building ``backbone_out``, ``find_input``, and ``find_target`` from
    ``(images, list[TextPrompts])`` requires non-trivial text tokenization via
    Meta's ``FindStage`` / ``TextTokenizer`` pipeline that was not fully pinned by
    Task 0.  Task 0's notes pin ``forward_grounding`` at ``sam3_image.py:440-446``
    and the ``_get_dummy_prompt`` zero-hint form at ``sam3_image.py:547-553``
    (use these when ``box_hints`` contains all ``None``).

    The ``box_hints`` kwarg is accepted so that ``Sam3Wrapper.forward`` can pass
    it through without breaking the interface.  When this body is completed,
    build the geometric prompt via ``_build_geometric_prompt`` and pass it as
    ``geometric_prompt`` to ``self.model.forward_grounding(...)``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        # IMPLEMENTOR: complete this body to drive Meta's text-grounding path.
        # Contract pinned in docs/superpowers/plans/2026-05-17-training-loop-notes.md
        # ("Where the slot lives" + Note A + Note B + Sources). Required steps:
        #   1. Encode images → backbone_out via self.model.image_encoder(images).
        #   2. Tokenize per-image class names ([p.classes[0] for p in prompts])
        #      through Meta's text tokenizer/encoder (the same path
        #      `SAM3Image.forward` uses to populate `find_input` at
        #      sam3_image.py:576-580). Inspect sam3.model.data_misc.FindStage
        #      for the dtype/device contract — see notes file §Sources.
        #   3. Build `find_input` (FindStage with input_text + input_text_mask)
        #      and `find_target` (training-time labels; pass an empty/None
        #      target for inference paths). The find_target shape depends on
        #      whether mask supervision is active — refer to the trainer's
        #      call site for the live contract.
        #   4. Build `geometric_prompt`:
        #        gp = _build_geometric_prompt(box_hints or [None]*B,
        #                                     self.image_size, images.device)
        #        if gp is None:
        #            gp = Prompt(
        #                box_embeddings=torch.zeros(0, B, 4, device=images.device),
        #                box_mask=torch.zeros(B, 0, device=images.device, dtype=torch.bool),
        #            )  # Meta's zero-length dummy (sam3_image.py:547-553)
        #   5. Call self.model.forward_grounding(backbone_out, find_input,
        #      find_target, geometric_prompt=gp) and return its dict.
        raise NotImplementedError(
            "Sam3Image high-level forward entrypoint not yet pinned; complete this "
            "function using the contract pinned in Task 0's notes file. See the "
            "step-by-step recipe in this function's inline comment above."
        )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's `sam3` package and wrap it for our trainer.

    Returns a `Sam3Wrapper` whose `forward(images, prompts, box_hints=None)` returns Meta's
    native per-class output dict (`pred_logits`, `pred_boxes`, `pred_masks`,
    `presence_logit_dec`).
    """
    ckpt_path = _resolve_checkpoint_path(cfg)
    bpe_path = _resolve_bpe_path(cfg)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw_model = sam3.build_sam3_image_model(
        bpe_path=str(bpe_path),
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

    adapter = _Sam3ImageAdapter(raw_model)
    return Sam3Wrapper(adapter, image_size=1008, mask_size=288)
