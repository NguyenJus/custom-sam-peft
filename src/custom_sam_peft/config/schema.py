"""Pydantic v2 schema for custom_sam_peft training configurations.

This module is the source of truth for every default and constraint. The
loader merges YAML + CLI overrides into a plain dict, then validates once
against TrainConfig.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

Dtype = Literal["bfloat16", "float16"]
DataFormat = Literal["coco", "hf"]
PromptMode = Literal["text", "bbox"]
PEFTMethod = Literal["lora", "qlora"]
QuantType = Literal["nf4", "fp4"]
# "auto" resolves at trainer construction (src/custom_sam_peft/train/trainer.py:45-49):
# adamw8bit if peft.method == "qlora" else adamw.
Optimizer = Literal["adamw", "adamw8bit", "auto"]
LRSchedule = Literal["constant", "cosine", "linear"]
TrackerBackend = Literal["tensorboard", "wandb", "none"]
TextPromptMode = Literal["present", "all", "present_plus_negatives", "sampled_fixed_k"]
LoraScope = Literal["vision", "vision_decoder", "all"]
EvalMode = Literal["full", "lite"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Strict):
    name: str
    output_dir: str = "./runs"
    seed: int = 42


class ModelConfig(_Strict):
    name: str = "facebook/sam3.1"
    local_dir: str | None = "models/sam3.1"
    checkpoint_file: str = "sam3.1_multiplex.pt"
    revision: str | None = None
    gradient_checkpointing: bool = (
        False  # TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed
    )
    dtype: Dtype = "bfloat16"
    device: str | None = None


class DataSplit(_Strict):
    annotations: str = Field(min_length=1)
    images: str = Field(min_length=1)


class AugmentationsConfig(_Strict):
    hflip: bool = True
    color_jitter: float = Field(default=0.1, ge=0.0, le=1.0)


class TextPromptConfig(_Strict):
    """How TextPrompts.classes is populated for each image when prompt_mode='text'.

    - present:                Use exactly the categories present in the image's
                              annotations (post-iscrowd filter). Default.
    - all:                    Use the full dataset class vocabulary every time.
    - present_plus_negatives: Use the present categories plus N randomly-sampled
                              negative class names per image.
    - sampled_fixed_k:        Use exactly k class names: all positives, plus
                              negatives sampled to reach k. If positives exceed
                              k, positives are truncated (kept in dense-id
                              ascending order). Deterministic given (seed, image_id).
    """

    mode: TextPromptMode = "present"
    negatives_per_image: int = Field(
        default=0,
        ge=0,
        description=(
            "How many randomly-sampled negative class names to add per image when "
            "mode='present_plus_negatives'. Bounded above by TextPrompts' multiplex "
            "cap of 16 (k field). Example configs ship 4, which leaves headroom for "
            "typical COCO present-class counts (~3-7 per image)."
        ),
    )
    k: int = Field(default=16, ge=1, le=16)


class NormalizeConfig(_Strict):
    """Normalization stats used as a user-controllable fallback for image preprocessing.

    Resolution is delegated to
    :func:`custom_sam_peft.data.transforms.resolve_normalization`, which consults
    three sources in order:

      1. ``AutoImageProcessor.from_pretrained(model.name, local_files_only=True)``
         (succeeds when the HF cache is populated). Emits INFO.
      2. On ``OSError/AttributeError/ValueError``: look up ``model.name`` in
         :data:`custom_sam_peft.data.transforms.KNOWN_PROCESSOR_STATS`. If
         present, return the table values (emits WARNING).
      3. Otherwise, return the (mean, std) here (emits WARNING — verify these
         are correct for the backbone).

    Defaults are ImageNet stats, matching ``facebook/sam3.1``'s
    ``Sam3ImageProcessor`` and the ``KNOWN_PROCESSOR_STATS`` entry. Users with a
    non-SAM3 backbone should override these and the YAML's ``data.normalize``
    block accordingly.
    """

    mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406], min_length=3, max_length=3
    )
    std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225], min_length=3, max_length=3
    )

    @model_validator(mode="after")
    def _check_ranges(self) -> NormalizeConfig:
        for m in self.mean:
            if not (0.0 <= m <= 1.0):
                raise ValueError(f"normalize.mean values must be in [0, 1]; got {m}")
        for s in self.std:
            if s <= 0.0:
                raise ValueError(f"normalize.std values must be > 0; got {s}")
        return self


class HFFieldMap(_Strict):
    """Optional overrides for HuggingFace dataset field names.

    Defaults match a conventional schema: top-level `image`, nested `objects.bbox`,
    `objects.category`, optional `objects.segmentation`; class names from the
    top-level `categories` feature.
    """

    image: str = "image"
    bbox: str = "objects.bbox"
    category: str = "objects.category"
    segmentation: str | None = "objects.segmentation"
    categories_feature: str = "categories"
    bbox_format: Literal["xywh", "xyxy"] = "xyxy"


class HFDatasetConfig(_Strict):
    """HuggingFace dataset specification (used when DataConfig.format == 'hf')."""

    name: str = Field(min_length=1)
    split_train: str = "train"
    split_val: str = "validation"
    field_map: HFFieldMap = Field(default_factory=HFFieldMap)


class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit
    test: DataSplit | None = None
    hf: HFDatasetConfig | None = None
    prompt_mode: PromptMode
    image_size: PositiveInt = 1008  # SAM3.1's native input; see models/sam3.py:192,304,1202-1203.
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    text_prompt: TextPromptConfig = Field(default_factory=TextPromptConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)

    @model_validator(mode="after")
    def _check_format_specific(self) -> DataConfig:
        if self.format == "hf" and self.hf is None:
            raise ValueError("data.hf is required when data.format == 'hf'")
        return self


class QLoRAConfig(_Strict):
    quant_type: QuantType = "nf4"
    compute_dtype: Dtype = "bfloat16"


class PEFTConfig(_Strict):
    method: PEFTMethod
    r: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    scope: LoraScope = "vision_decoder"
    target_modules: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of module name patterns to adapt. When None, "
            "apply_lora uses SCOPE_TARGETS[scope]. When set, scope is ignored."
        ),
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)


class MatcherWeights(_Strict):
    """Per-term cost weights for the Hungarian matcher.

    v0 defaults are mask-only (box terms = 0) because v0 trains text-only
    with no box supervision. Users who later want box supervision can
    override these in config.
    """

    lambda_l1: float = Field(default=0.0, ge=0.0)
    lambda_giou: float = Field(default=0.0, ge=0.0)
    lambda_mask: PositiveFloat = 5.0


class BoxHintSchedule(_Strict):
    """Linear-decay schedule for per-image probability of feeding GT boxes
    as a localization hint alongside the text prompt.

    p(t) = max(p_end, p_start + (p_end - p_start) * t / decay_steps)
    where t = global_step. Applied per-image via Bernoulli(p(t)) over each
    image's GT boxes for the currently-prompted class.

    early_stop_p_threshold is consumed by a future early-stopping mechanism
    (not by the training-loop spec): a run MUST NOT terminate early while
    current p(t) >= this value. Recorded here so the constraint is
    co-located with the schedule it gates.
    """

    p_start: float = Field(default=1.0, ge=0.0, le=1.0)
    p_end: float = Field(default=0.0, ge=0.0, le=1.0)
    decay_steps: PositiveInt = 5000
    early_stop_p_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_monotone(self) -> BoxHintSchedule:
        if self.p_end > self.p_start:
            raise ValueError(
                f"BoxHintSchedule must decay: p_end ({self.p_end}) > p_start ({self.p_start})"
            )
        return self


class LossConfig(_Strict):
    """Loss-mix weights and focal CE params for SAM 3.1 training.

    No `w_cls`: discrimination across classes comes from running one forward
    pass per class prompt. `w_presence` weights the image-level
    "any-instance-of-this-class-present?" supervision applied to
    `presence_logit_dec`.
    """

    w_mask: PositiveFloat = 1.0
    w_box: float = Field(default=0.0, ge=0.0)
    w_obj: PositiveFloat = 1.0
    w_presence: PositiveFloat = 1.0
    matcher_weights: MatcherWeights = Field(default_factory=MatcherWeights)
    focal_gamma: PositiveFloat = 2.0
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0)


class TrainHyperparams(_Strict):
    epochs: PositiveInt
    batch_size: PositiveInt = 1
    grad_accum_steps: PositiveInt = 8
    optimizer: Optimizer = "auto"
    lr: PositiveFloat = 1.0e-4
    lr_schedule: LRSchedule = "cosine"
    warmup_steps: int = Field(default=100, ge=0)
    max_grad_norm: PositiveFloat = 1.0
    eval_every: PositiveInt = 500
    save_every: PositiveInt = 1000
    loss: LossConfig = Field(default_factory=LossConfig)
    box_hint: BoxHintSchedule = Field(default_factory=BoxHintSchedule)
    log_every: PositiveInt = 50
    nan_abort_after: PositiveInt = 20
    num_workers: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 1),
        ge=0,
        description="DataLoader workers. 0 disables multiprocessing.",
    )


class EvalConfig(_Strict):
    metrics: list[str] = Field(default_factory=lambda: ["mAP", "mAP_50", "mAP_75", "per_class_AP"])
    iou_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    mode: EvalMode = "full"
    lite_max_images: PositiveInt = 64
    mask_threshold: float = 0.0
    save_predictions: bool = False


class WandbConfig(_Strict):
    project: str = "custom_sam_peft"
    entity: str | None = None


class TrackingConfig(_Strict):
    backend: TrackerBackend = "tensorboard"
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class ExportConfig(_Strict):
    merge: bool = False


class TrainConfig(_Strict):
    """Top-level config produced by the loader."""

    run: RunConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    peft: PEFTConfig
    train: TrainHyperparams
    eval: EvalConfig = Field(default_factory=EvalConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
