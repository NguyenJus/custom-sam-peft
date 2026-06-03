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

from custom_sam_peft.config._duration import parse_duration_to_seconds
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
    "SemanticDataConfig",
    "SemanticLossConfig",
    "SemanticLossOverrides",
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
    "SemMaskFamily",
    "Optimizer",
    "PEFTMethod",
    "PresenceFamily",
    "Preset",
    "QuantType",
    "SubsetStrategy",
    "Task",
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
DataFormat = Literal["coco", "hf", "mask_png"]  # mask_png is semantic-only (§4.2)
Task = Literal["instance", "semantic"]  # cite: #113 — extensible (panoptic out of scope)
PEFTMethod = Literal["lora", "qlora"]
QuantType = Literal["nf4", "fp4"]
# "auto" resolves at trainer construction via peft_method.recommended_optimizer()
# (src/custom_sam_peft/train/trainer.py): adamw8bit for QLoRA, adamw for LoRA.
Optimizer = Literal["adamw", "adamw8bit", "auto"]
LRSchedule = Literal["constant", "cosine", "linear", "poly"]
TrackerBackend = Literal["local", "tensorboard", "wandb", "none"]
TextPromptMode = Literal["present", "all", "present_plus_negatives", "sampled_fixed_k"]
LoraScope = Literal["vision", "vision_decoder", "vision_decoder_concept", "all"]
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
    dtype: Dtype = "bfloat16"
    # --- advanced ---
    revision: str | None = None
    device: str | None = None


class DataSplit(_Strict):
    """A pair of paths identifying one data split.

    For ``mask_png`` format: ``annotations`` is reinterpreted as the
    **label-map PNG directory** and ``images`` as the image directory
    (no JSON annotation file is used).
    """

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


SemMaskFamily = Literal["ce_dice", "focal_dice", "focal_tversky", "boundary", "ce", "dice"]


class SemanticLossOverrides(_Strict):
    """Per-knob overrides; None -> inherit from (preset, class_imbalance)."""

    sem_family: SemMaskFamily | None = None
    w_ce: PositiveFloat | None = None
    w_region: PositiveFloat | None = None  # weight on the Dice/Tversky/Boundary term
    focal_gamma: PositiveFloat | None = None
    focal_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_gamma: PositiveFloat | None = None
    boundary_weight: float | None = Field(default=None, ge=0.0, le=1.0)


class SemanticLossConfig(_Strict):
    preset: Preset = "natural"  # reuse #112 Preset verbatim
    class_imbalance: ClassImbalance = "balanced"  # reuse #112 axis verbatim
    overrides: SemanticLossOverrides = Field(default_factory=SemanticLossOverrides)
    # --- argmax / background / reduction knobs (§4.5, §6.2) ---
    background_logit: float = 0.0  # cite: degenerate logit boundary (sigmoid(0)=0.5)
    background_class_name: str | None = None  # tbd: #113 — custom explicit-bg name
    query_reduce: Literal["max", "sum"] = "max"  # tbd: #113 — see §6.2
    source: Literal["marginalize", "semantic_seg"] = "marginalize"  # cite: §3.3 / OQ-1


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
    """How TextPrompts.classes is populated for each image.

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
    k: int = Field(default=16, ge=1, le=16)  # cite: models/sam3.py:MULTIPLEX_CAP


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

    Defaults to ``[0.5, 0.5, 0.5]`` / ``[0.5, 0.5, 0.5]`` — matching
    ``facebook/sam3.1``'s ``Sam3ImageProcessor`` (empirically verified 2026-05-30)
    and the ``KNOWN_PROCESSOR_STATS`` entry. Users with a non-SAM3 backbone should
    override these and the YAML's ``data.normalize`` block accordingly.
    """

    # --- advanced --- (all normalize fields override the AutoImageProcessor-derived stats)
    mean: list[float] = Field(
        default_factory=lambda: [0.5, 0.5, 0.5],
        min_length=1,
        max_length=16,  # cite: empirically verified 2026-05-30 (Sam3ImageProcessor)
    )
    std: list[float] = Field(
        default_factory=lambda: [0.5, 0.5, 0.5],
        min_length=1,
        max_length=16,  # cite: empirically verified 2026-05-30 (Sam3ImageProcessor)
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
    label_map: str | None = None  # cite: #113 -- HF feature holding the (H,W) label image


class HFDatasetConfig(_Strict):
    """HuggingFace dataset specification (used when DataConfig.format == 'hf')."""

    name: str = Field(min_length=1)
    split_train: str = "train"
    split_val: str | None = None
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


class SemanticDataConfig(_Strict):
    """Semantic-segmentation data parameters. Required when task == 'semantic'.

    Lives under DataConfig.semantic. None for instance datasets.
    """

    class_map: str | None = Field(
        default=None,
        description=(
            "Path to a JSON file mapping integer pixel value -> class name, e.g. "
            '{"0": "background", "1": "road", "2": "building"}. The set of NAMES '
            "(excluding any explicit background, see §4.5) is the prompted concept "
            "vocabulary AND the dataset class_names, in ascending-pixel-value order. "
            "Required for data.format: mask_png (the only class-name source). "
            "Optional for data.format: hf — when absent the class vocabulary is "
            "derived from the HF dataset's label feature ClassLabel.names."
        ),
    )
    ignore_index: int = Field(
        default=255,  # cite: PASCAL VOC / Cityscapes void convention (255)
        description=(
            "Pixel value in the label map treated as void/unlabeled. Excluded from "
            "both loss and metrics. Not a class. Default 255 is the de-facto standard."
        ),
    )
    label_suffix: str = Field(
        default="_labelIds.png",  # tbd: #113 — Cityscapes-style; override per dataset
        description=(
            "Filename suffix that maps an image file to its label-map PNG (mask_png "
            "format only). image 'aachen_000000.png' -> label "
            "'aachen_000000{label_suffix}'. Set to '.png' for same-stem pairing."
        ),
    )


class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit | None = None
    val_split: ValSplitConfig | None = None
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
    semantic: SemanticDataConfig | None = None  # required when task == 'semantic'
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
            and self.hf.split_val is not None
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
    use_double_quant: bool = False


class PEFTConfig(_Strict):
    method: PEFTMethod
    r: PositiveInt = 16  # cite: LoRA (Hu 2021) arXiv:2106.09685 §4.1; alpha=2r convention
    alpha: PositiveInt = 32  # cite: LoRA (Hu 2021) §4.1 "we simply set alpha to the first r we try"
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)  # tbd: #191 (LoRA varies 0.0-0.1)
    scope: LoraScope = "vision_decoder_concept"
    # tbd: #230 (project-chosen SAM 3.1 concept scope; default flipped from
    #      vision_decoder so the shipped default can learn niche TEXT concepts —
    #      vision_decoder freezes ca_text/self_attn in_proj. Reproducibility: a config
    #      without an explicit peft.scope now additionally adapts ca_text/self_attn
    #      in_proj; configs pinning vision/vision_decoder/all are unaffected. See
    #      research note §4, §7.)
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


class MultiplexConfig(_Strict):
    """Multiplex forward knobs.

    classes_per_forward: number of class prompts per multiplex forward pass.
    Capped at SAM 3.1's MULTIPLEX_CAP=16 (in src/custom_sam_peft/models/sam3.py).
    Setting 1 reduces to the legacy per-class regime within the same code path.
    """

    classes_per_forward: int = Field(default=16, ge=1, le=16)  # cite: models/sam3.py:MULTIPLEX_CAP


class EarlyStopConfig(_Strict):
    """Early-stop knobs gating the no-improvement counter.

    monitor/min_delta define the improvement test (mAP > best + min_delta,
    strict). The counter that drives early stop only accrues once BOTH grace
    conditions are satisfied:

      - adaptive baseline (PRIMARY): the first eval producing a non-zero mAP
        "wakes" the run. Until then mAP is pinned at 0.0 (cold) and the counter
        never climbs — self-scaling, no magic threshold.
      - warmup_floor_steps (BACKSTOP): a fixed floor in optimizer steps below
        which the counter may not accrue regardless of mAP.

    A model that never produces a non-zero mAP trains to the horizon."""

    enabled: bool = True
    # issue: on by default (research §7, issue acceptance criteria).
    monitor: Literal["mAP"] = "mAP"
    # existing best-metric key (trainer.py _best_metric_key). Exposed as a seam;
    # only mAP is validated/wired for now.
    min_delta: PositiveFloat = 0.001
    stop_patience: PositiveInt = 10
    warmup_floor_steps: int = Field(default=1000, ge=0)
    # cite: Detectron2 SOLVER.WARMUP_ITERS (default 1000) — backstop grace floor
    # in optimizer steps before the no-improvement counter may accrue (#264).
    # 0 disables the backstop (adaptive-baseline-only grace).


class TrainHyperparams(_Strict):
    epochs: PositiveInt
    batch_size: PositiveInt = 1
    grad_accum_steps: PositiveInt = 8
    optimizer: Optimizer = "auto"
    learning_rate: PositiveFloat = 1.0e-4
    lr_schedule: LRSchedule = "poly"
    # cite: WarmupPolyLR (Detectron2) / PolyLR (MMSegmentation) — the fixed-horizon
    #       poly-decay norm (#264).
    warmup_steps: int = Field(default=100, ge=0)
    save_every: PositiveInt | None = Field(
        default=None,
        description=(
            "Save a checkpoint every N global steps. "
            "None (default) means auto-resolve at runtime to steps_per_epoch, "
            "so one checkpoint per epoch."
        ),
    )
    log_every: PositiveInt = 50
    # --- advanced ---
    max_grad_norm: PositiveFloat = 1.0
    eval_every: PositiveInt | None = Field(
        default=None,
        description=(
            "Run mid-run evaluation every N global steps. "
            "None (default) means auto-resolve at runtime to steps_per_epoch, "
            "so one evaluation per epoch."
        ),
    )
    time_limit: str | int | None = Field(
        default=None,
        description=(
            "Wall-clock budget for this invocation. Accepts a human duration "
            '("2h30m", "90m", "3600s") or bare seconds (3600). None (default) '
            "means unlimited. The budget is per-run: --resume restarts the clock."
        ),
    )

    @field_validator("time_limit", mode="before")
    @classmethod
    def _validate_time_limit(cls, v: object) -> object:
        """Validate (don't rewrite) the duration. Stored verbatim; parsed in fit().

        mode="before" so a raw bool is rejected by parse_duration_to_seconds
        before Pydantic's lax bool->int coercion would mask it.
        """
        if v is None:
            return v
        parse_duration_to_seconds(v)  # type: ignore[arg-type]  # runtime branches on str|int|bool
        return v

    host_ram_floor_gb: float = Field(
        default=2.0,
        description=(
            "Available host-RAM floor (GB). When psutil.virtual_memory().available "
            "drops below this value at any training step, a full resumable checkpoint "
            "is flushed and training stops gracefully (exit 0). A value <= 0 disables "
            "the guard. On by default (2.0 GB) — host-RAM OOM triggers SIGKILL which "
            "cannot be caught in Python, so we must stop proactively."
        ),
    )
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)

    loss: LossConfig = Field(default_factory=LossConfig)
    semantic_loss: SemanticLossConfig = Field(default_factory=SemanticLossConfig)
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
    mask_threshold: float = 0.0  # cite: degenerate-case (logit boundary; sigmoid(0)=0.5)
    save_predictions: bool = False
    batch_size: PositiveInt | Literal["auto"] = "auto"
    visualize: bool = True
    visualize_count: PositiveInt = 10


# WandbConfig, ExportConfig moved to config._internal (audit Section G).
# They are re-exported from this module for backward compatibility.
# New code should import from custom_sam_peft.config._internal directly.


class TrackingConfig(_Strict):
    backend: TrackerBackend = "local"
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
    task: Task = "instance"  # cite: #113 — default preserves the instance path exactly

    @model_validator(mode="after")
    def _check_task_data_compat(self) -> TrainConfig:
        if self.task == "semantic":
            if self.data.format == "coco":
                raise ValueError(
                    "task: semantic does not support data.format: coco (instance JSON). "
                    "Use data.format: mask_png or hf with a semantic field map."
                )
            if self.data.semantic is None:
                raise ValueError("task: semantic requires data.semantic (class_map, ignore_index).")
            if self.data.format == "mask_png" and self.data.semantic.class_map is None:
                raise ValueError(
                    "data.format: mask_png requires data.semantic.class_map "
                    "(a JSON pixel-value -> class-name map)."
                )
            if self.eval.iou_thresholds != EvalConfig().iou_thresholds:
                raise ValueError(
                    "eval.iou_thresholds is inert under task: semantic (mIoU has no "
                    "threshold sweep). Remove it."
                )
            if self.eval.mask_threshold != EvalConfig().mask_threshold:
                raise ValueError(
                    "eval.mask_threshold is inert under task: semantic (argmax, not "
                    "per-mask binarize). Remove it."
                )
        else:  # instance
            if self.data.semantic is not None:
                raise ValueError("data.semantic is only valid when task: semantic.")
            if self.data.format == "mask_png":
                raise ValueError("data.format: mask_png requires task: semantic.")
        return self
