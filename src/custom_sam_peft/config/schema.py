"""Pydantic v2 schema for custom_sam_peft training configurations.

This module is the source of truth for every default and constraint. The
loader merges YAML + CLI overrides into a plain dict, then validates once
against TrainConfig.

Internal sub-configs (MatcherWeights, WandbConfig, ExportConfig)
have been moved to config._internal per audit Section G. They are re-exported
here for backward compatibility. New code should import from
config._internal directly.

LossConfig is now defined here (Pydantic model) as part of the #112 schema
break; the former dataclass LossConfig in _internal.py has been deleted.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from custom_sam_peft.config._internal import (
    ExportConfig,
    MatcherWeights,
    WandbConfig,
)
from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS

__all__ = [  # noqa: RUF022
    # User-facing Pydantic models
    "AugmentationOverrides",
    "AugmentationsConfig",
    "BoxHintSchedule",
    "ClassImbalance",
    "DataConfig",
    "DataSplit",
    "EvalConfig",
    "HFDatasetConfig",
    "HFFieldMap",
    "LimitConfig",
    "LossConfig",
    "LossOverrides",
    "ModelConfig",
    "MultiplexConfig",
    "NormalizeConfig",
    "PEFTConfig",
    "QLoRAConfig",
    "RunConfig",
    "TextPromptConfig",
    "TrackingConfig",
    "TrainConfig",
    "TrainHyperparams",
    "ValSplitConfig",
    # Type aliases
    "BoxFamily",
    "DataFormat",
    "Dtype",
    "EvalMode",
    "Intensity",
    "LoraScope",
    "LRSchedule",
    "MaskFamily",
    "ObjFamily",
    "Optimizer",
    "PEFTMethod",
    "PresenceFamily",
    "Preset",
    "PromptMode",
    "QuantType",
    "SubsetStrategy",
    "TextPromptMode",
    "TrackerBackend",
    # Internal classes re-exported for backward compatibility (audit Section G)
    # These are dataclasses, not Pydantic models. Import from config._internal
    # directly in new code.
    "ExportConfig",
    "MatcherWeights",
    "WandbConfig",
]

Dtype = Literal["bfloat16", "float16"]
DataFormat = Literal["coco", "hf"]
PromptMode = Literal["text", "bbox"]
PEFTMethod = Literal["lora", "qlora"]
QuantType = Literal["nf4", "fp4"]
# "auto" resolves at trainer construction via peft_method.recommended_optimizer()
# (src/custom_sam_peft/train/trainer.py): adamw8bit for QLoRA, adamw for LoRA.
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
    gradient_checkpointing: bool = (
        False  # TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed
    )
    dtype: Dtype = "bfloat16"
    # --- advanced ---
    revision: str | None = None
    device: str | None = None


class DataSplit(_Strict):
    annotations: str = Field(min_length=1)
    images: str = Field(min_length=1)


Preset = Literal["natural", "medical", "satellite", "microscopy", "none", "custom"]
Intensity = Literal["safe", "medium", "aggressive"]

ClassImbalance = Literal["balanced", "moderate", "severe"]
MaskFamily = Literal[
    "bce", "dice", "dice_bce", "focal_bce", "focal_dice", "focal_tversky", "boundary"
]
BoxFamily = Literal["l1_giou", "giou_only", "ciou"]
ObjFamily = Literal["focal_bce", "bce"]
PresenceFamily = Literal["bce", "focal_bce"]


class LossOverrides(_Strict):
    """Per-knob overrides. All None → inherit from (preset, class_imbalance).

    Setting any field to a non-None value replaces just that field in the
    resolved table. Extra keys are rejected (extra="forbid"); typos surface
    at config-load time.
    """

    # Term selection (4 axes)
    mask_family: MaskFamily | None = None
    box_family: BoxFamily | None = None
    obj_family: ObjFamily | None = None
    presence_family: PresenceFamily | None = None

    # Weights (4)
    w_mask: PositiveFloat | None = None
    w_box: float | None = Field(default=None, ge=0.0)
    w_obj: PositiveFloat | None = None
    w_presence: PositiveFloat | None = None

    # Focal params (2)
    focal_gamma: PositiveFloat | None = None
    focal_alpha: float | None = Field(default=None, ge=0.0, le=1.0)

    # Tversky params (2)
    tversky_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_gamma: PositiveFloat | None = None

    # Boundary blend coefficient (1)
    boundary_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    # Matcher contract (internal sub-model; accepts dict or MatcherWeights instance)
    matcher_weights: MatcherWeights | None = None

    @field_validator("matcher_weights", mode="before")
    @classmethod
    def _coerce_matcher_weights(cls, v: object) -> MatcherWeights | None:
        if v is None or isinstance(v, MatcherWeights):
            return v
        if isinstance(v, dict):
            return MatcherWeights(**v)
        raise TypeError(f"matcher_weights must be a dict or MatcherWeights, got {type(v).__name__}")


class LossConfig(_Strict):
    preset: Preset = "natural"
    class_imbalance: ClassImbalance = "balanced"
    overrides: LossOverrides = Field(default_factory=LossOverrides)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class AugmentationOverrides(_Strict):
    """Per-knob overrides. All None → inherit from (preset, intensity).

    Setting any field to a non-None value replaces just that field in the
    resolved table. Extra keys are rejected (extra="forbid"); typos surface
    at config-load time.
    """

    hflip: bool | None = None
    vflip: bool | None = None
    rotate90: bool | None = None
    rotate_arbitrary: float | None = Field(default=None, ge=0.0)
    color_jitter: float | None = Field(default=None, ge=0.0)
    stain_jitter: float | None = Field(default=None, ge=0.0)
    blur: float | None = Field(default=None, ge=0.0)
    gauss_noise: float | None = Field(default=None, ge=0.0)


class AugmentationsConfig(_Strict):
    preset: Preset = "natural"
    intensity: Intensity = "medium"
    overrides: AugmentationOverrides = Field(default_factory=AugmentationOverrides)


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

    # --- advanced --- (all normalize fields override the AutoImageProcessor-derived stats)
    mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406], min_length=1, max_length=16
    )
    std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225], min_length=1, max_length=16
    )
    max_pixel_value: float = Field(
        default=255.0,
        gt=0.0,
        description=(
            "Divisor applied by A.Normalize before subtracting mean / dividing by "
            "std. Default 255.0 assumes uint8 input. For float multi-band input "
            "(e.g. SAR/height already in [0,1]), set this to your data's max (e.g. "
            "1.0); mean/std must be expressed in the same units. See spec §7.2."
        ),
    )

    @model_validator(mode="after")
    def _check_ranges(self) -> NormalizeConfig:
        if len(self.mean) != len(self.std):
            raise ValueError(
                f"normalize.mean has {len(self.mean)} entries but normalize.std has "
                f"{len(self.std)}; mean and std must have the same length."
            )
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


class ValSplitConfig(_Strict):
    """Auto-split parameters. Used when DataConfig.val_split is set.

    Carves data.train into train+val deterministically. In v0:
      - stratification is always-on Sechidis multi-label iterative;
        not configurable.
      - split unit is always 'image'; not configurable. Splitting by
        annotation can leak the same image into both sides.

    Spec: docs/superpowers/specs/2026-05-22-data-no-val-auto-split-design.md §3.1.
    """

    fraction: float = Field(default=0.1, gt=0.0, le=0.5)
    seed: int | None = None  # None → inherit run.seed at resolve time


SubsetStrategy = Literal["random", "stratified", "first_n"]


class LimitConfig(_Strict):
    """Optional per-split dataset size limits.

    Each limit is either:
    - None  — no limit (use all samples)
    - int   — absolute sample count (must be >= 1)
    - float — fraction of the split (must be in (0.0, 1.0])

    Booleans are explicitly rejected: Pydantic v2 coerces bool → int,
    so we check isinstance(v, bool) BEFORE the numeric range check.
    """

    train: int | float | None = None
    val: int | float | None = None
    seed: int = 42
    strategy: SubsetStrategy = "random"

    @model_validator(mode="before")
    @classmethod
    def _check_limits(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for name in ("train", "val"):
            v = data.get(name)
            if v is None:
                continue
            if isinstance(v, bool):
                raise ValueError(f"limit.{name} must not be a bool; got {v!r}")
            if isinstance(v, int):
                if v < 1:
                    raise ValueError(f"limit.{name} must be >= 1 when an int; got {v!r}")
            elif isinstance(v, float) and not (0.0 < v <= 1.0):
                raise ValueError(f"limit.{name} must be in (0.0, 1.0] when a float; got {v!r}")
        return data


class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit | None = None
    val_split: ValSplitConfig | None = None
    prompt_mode: PromptMode
    image_size: PositiveInt = 1008  # SAM3.1's native input; see models/sam3.py:192,304,1202-1203.
    channels: int = Field(
        default=3,
        ge=1,
        le=16,
        description=(
            "Number of input image channels (1..16). The N->3 channel adapter "
            "(a 1x1 conv inserted before the frozen SAM3.1 patch-embed) bridges "
            "N channels down to the pretrained 3-channel stem. The cap of 16 is "
            "deliberate: beyond ~16 channels the 3-channel bottleneck becomes "
            "lossy, at which point a future 'bridge B' (replacing the patch-embed "
            "with an in_chans=N stem; issue follow-up) would be warranted instead. "
            "Explicit only — no auto-detection."
        ),
    )
    channel_semantics: Literal["rgb", "rgba", "grayscale", "freeform"] = Field(
        default="rgb",
        description=(
            "How the input channels are interpreted (independent of the channel "
            "COUNT in `channels`). Drives the channel adapter (build + init), the "
            "normalization default, and the augmentation regime. See the "
            "CHANNEL_SEMANTICS registry (src/custom_sam_peft/data/channel_semantics.py) "
            "for the per-semantic profile. Default 'rgb' reproduces today's behavior "
            "exactly. Add new semantics by adding a registry entry."
        ),
    )
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    text_prompt: TextPromptConfig = Field(default_factory=TextPromptConfig)
    normalize: NormalizeConfig | None = None
    limit: LimitConfig = Field(default_factory=LimitConfig)
    # --- advanced ---
    test: DataSplit | None = None
    hf: HFDatasetConfig | None = None

    @model_validator(mode="after")
    def _check_format_specific(self) -> DataConfig:
        if self.format == "hf" and self.hf is None:
            raise ValueError("data.hf is required when data.format == 'hf'")
        return self

    @model_validator(mode="after")
    def _check_channels_semantics_normalize(self) -> DataConfig:
        profile = CHANNEL_SEMANTICS[self.channel_semantics]

        # (a) semantic <-> channels match
        if self.channels not in profile.allowed_channels:
            allowed = sorted(profile.allowed_channels)
            allowed_str = f"{allowed[0]}" if len(allowed) == 1 else f"{allowed[0]}..{allowed[-1]}"
            raise ValueError(
                f"data.channel_semantics={self.channel_semantics!r} requires "
                f"data.channels={allowed_str}, but data.channels={self.channels}."
            )

        # (b) resolve normalize: explicit wins; else profile default; freeform requires explicit.
        if self.normalize is None:
            if profile.normalize_default is None:
                raise ValueError(
                    f"data.channel_semantics={self.channel_semantics!r} requires explicit "
                    f"data.normalize.mean/std (one value per channel; no default exists for "
                    f"freeform). Provide N={self.channels} mean and {self.channels} std values."
                )
            mean, std = profile.normalize_default
            self.normalize = NormalizeConfig(mean=list(mean), std=list(std))

        # (c) length cross-check (after default materialization)
        if len(self.normalize.mean) != self.channels or len(self.normalize.std) != self.channels:
            raise ValueError(
                f"data.normalize.mean has {len(self.normalize.mean)} entries but "
                f"data.channels={self.channels}; provide exactly {self.channels} per-channel "
                f"mean values (and {self.channels} std values)."
            )
        return self

    @model_validator(mode="after")
    def _check_val_modes(self) -> DataConfig:
        if self.val is not None and self.val_split is not None:
            raise ValueError(
                "data.val and data.val_split are mutually exclusive. "
                "Set one to provide a validation set, neither for no-val mode."
            )
        return self

    @model_validator(mode="after")
    def _check_hf_split_val_compat(self) -> DataConfig:
        if (
            self.format == "hf"
            and self.val_split is not None
            and self.hf is not None
            and self.hf.split_val != "validation"
        ):
            raise ValueError(
                "data.hf.split_val cannot be customized when data.val_split is set; "
                "auto-split carves the val set from data.hf.split_train. "
                "Remove split_val or remove val_split."
            )
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
    # --- advanced ---
    target_modules: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of module name patterns to adapt. When None, "
            "apply_lora uses SCOPE_TARGETS[scope]. When set, scope is ignored."
        ),
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)


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
    # early_stop_p_threshold demoted (audit Section E): no active src consumer;
    # retained as seam scaffolding for a future early-stopping mechanism.
    # See follow-up issue (Section J4). Not user-settable from YAML.

    @model_validator(mode="after")
    def _check_monotone(self) -> BoxHintSchedule:
        if self.p_end > self.p_start:
            raise ValueError(
                f"BoxHintSchedule must decay: p_end ({self.p_end}) > p_start ({self.p_start})"
            )
        return self


class MultiplexConfig(_Strict):
    """Multiplex forward knobs.

    classes_per_forward: number of class prompts per multiplex forward pass.
    Capped at SAM 3.1's MULTIPLEX_CAP=16 (in src/custom_sam_peft/models/sam3.py).
    Setting 1 reduces to the legacy per-class regime within the same code path.
    """

    classes_per_forward: int = Field(default=16, ge=1, le=16)


class TrainHyperparams(_Strict):
    epochs: PositiveInt
    batch_size: PositiveInt = 1
    grad_accum_steps: PositiveInt = 8
    optimizer: Optimizer = "auto"
    learning_rate: PositiveFloat = 1.0e-4
    lr_schedule: LRSchedule = "cosine"
    warmup_steps: int = Field(default=100, ge=0)
    save_every: PositiveInt = 1000
    box_hint: BoxHintSchedule = Field(default_factory=BoxHintSchedule)
    log_every: PositiveInt = 50
    # --- advanced ---
    max_grad_norm: PositiveFloat = 1.0
    eval_every: PositiveInt = 500
    loss: LossConfig = Field(default_factory=LossConfig)
    nan_abort_after: PositiveInt = 20
    num_workers: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 1),
        ge=0,
        description="DataLoader workers. 0 disables multiprocessing.",
    )
    multiplex: MultiplexConfig = Field(default_factory=MultiplexConfig)


class EvalConfig(_Strict):
    # --- advanced --- (all eval fields are optional overrides; section defaults are usable as-is)
    iou_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    mode: EvalMode = "full"
    lite_max_images: PositiveInt = 64
    mask_threshold: float = 0.0
    save_predictions: bool = False
    batch_size: PositiveInt | Literal["auto"] = "auto"


# WandbConfig, ExportConfig moved to config._internal (audit Section G).
# They are re-exported from this module for backward compatibility.
# New code should import from custom_sam_peft.config._internal directly.


class TrackingConfig(_Strict):
    backend: TrackerBackend = "tensorboard"
    # --- advanced ---
    wandb: WandbConfig = Field(default_factory=WandbConfig)


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
