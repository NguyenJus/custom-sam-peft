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
from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import Prompts, TextPrompts

logger = logging.getLogger(__name__)


class Sam3Wrapper(nn.Module):
    """Thin wrapper around Meta's SAM 3.1 model.

    Contract:
      - `forward(images, prompts)` accepts a batch of B images and a list of
        B `Prompts` objects, one per image.
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

    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Any]:
        self._validate_prompts(images, prompts)
        out: dict[str, Any] = self.model(images, prompts)
        return out

    @staticmethod
    def _validate_prompts(images: Tensor, prompts: list[Prompts]) -> None:
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
    """Adapt raw Sam3Image to the (images, prompts) calling convention used by Sam3Wrapper.

    Sam3Image's training-mode forward (`forward_grounding`) expects
    `(backbone_out, find_input, find_target, geometric_prompt)`, none of which are
    raw image tensors or our `Prompts` dataclasses. This adapter holds the inner
    `Sam3Image` and orchestrates the conversion based on what Meta's high-level
    methods (inspected in Step 1) expose. If Meta exposes `predict_inst` or
    similar that takes raw images, prefer that path here.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Tensor]:
        # IMPLEMENTOR: based on Step 1's inspection, fill this in.
        # The simplest case: Sam3Image exposes a method that takes
        # `(images, list_of_class_names)` and returns the per-image dict.
        #
        # For TextPrompts (which is the only supported case per Sam3Wrapper):
        #   class_names = [p.classes[0] for p in prompts]
        #   return self.model.<entrypoint>(images, class_names)
        #
        # If no such method exists, build the lower-level Sam3 inputs here
        # (backbone_out via self.model.image_encoder(images), find_input from
        # tokenized class names, etc.). Keep the function body small — if it
        # exceeds ~30 lines, factor out helpers in this same file.
        raise NotImplementedError(
            "Sam3Image high-level forward entrypoint not yet pinned; complete this "
            "function after running Step 1's inspection in your local environment."
        )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's `sam3` package and wrap it for our trainer.

    Returns a `Sam3Wrapper` whose `forward(images, prompts)` returns Meta's
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
