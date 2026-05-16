"""Pydantic v2 schema for esam3 training configurations.

This module is the source of truth for every default and constraint. The
loader merges YAML + CLI overrides into a plain dict, then validates once
against TrainConfig.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

Dtype = Literal["bfloat16", "float16"]
DataFormat = Literal["coco", "hf"]
PromptMode = Literal["text", "bbox"]
PEFTMethod = Literal["lora", "qlora"]
QuantType = Literal["nf4", "fp4"]
Optimizer = Literal["adamw", "adamw8bit"]
LRSchedule = Literal["constant", "cosine", "linear"]
TrackerBackend = Literal["tensorboard", "wandb", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Strict):
    name: str
    output_dir: str = "./runs"
    seed: int = 42


class ModelConfig(_Strict):
    name: str = "facebook/sam3.1"
    revision: str | None = None
    gradient_checkpointing: bool = True
    dtype: Dtype = "bfloat16"


class DataSplit(_Strict):
    annotations: str = Field(min_length=1)
    images: str = Field(min_length=1)


class AugmentationsConfig(_Strict):
    hflip: bool = True
    color_jitter: float = Field(default=0.1, ge=0.0, le=1.0)


class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit
    prompt_mode: PromptMode
    image_size: PositiveInt = 1024
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)


class QLoRAConfig(_Strict):
    quant_type: QuantType = "nf4"
    compute_dtype: Dtype = "bfloat16"


class PEFTConfig(_Strict):
    method: PEFTMethod
    r: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)


class TrainHyperparams(_Strict):
    epochs: PositiveInt
    batch_size: PositiveInt = 1
    grad_accum_steps: PositiveInt = 8
    optimizer: Optimizer = "adamw"
    lr: PositiveFloat = 1.0e-4
    lr_schedule: LRSchedule = "cosine"
    warmup_steps: int = Field(default=100, ge=0)
    max_grad_norm: PositiveFloat = 1.0
    eval_every: PositiveInt = 500
    save_every: PositiveInt = 1000


class EvalConfig(_Strict):
    metrics: list[str] = Field(default_factory=lambda: ["mAP", "mAP_50", "mAP_75", "per_class_AP"])
    iou_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )


class WandbConfig(_Strict):
    project: str = "esam3"
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
