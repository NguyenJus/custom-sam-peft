"""Import every @register site once so the registry is populated.

CLI entry imports this module at load time; library callers (notebooks)
may import it explicitly if they need plugin lookup."""

from custom_sam_peft.data import coco, hf
from custom_sam_peft.peft_adapters import lora, qlora
from custom_sam_peft.tracking import noop, tensorboard, wandb

__all__ = ["coco", "hf", "lora", "noop", "qlora", "tensorboard", "wandb"]
