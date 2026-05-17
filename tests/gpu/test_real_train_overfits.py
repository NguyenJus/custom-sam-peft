"""50-step LoRA overfit on tiny_coco with box-hint curriculum.

Gated by `@pytest.mark.gpu`, `@requires_compatible_gpu`, and
`@requires_checkpoint`. Not in CI by default. Run with:
    pytest -m gpu tests/gpu/test_real_train_overfits.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.config.schema import (
    AugmentationsConfig,
    BoxHintSchedule,
    DataConfig,
    DataSplit,
    ModelConfig,
    NormalizeConfig,
    PEFTConfig,
    RunConfig,
    TextPromptConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.data.coco import COCODataset
from esam3.data.transforms import build_train_transforms
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import apply_lora
from esam3.tracking.noop import NoopTracker
from esam3.train.trainer import Trainer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def _ds(tiny_coco_dir: Path) -> COCODataset:
    # NOTE: build_train_transforms signature is (aug_cfg, image_size, ...) — see Task 9 notes.
    transforms = build_train_transforms(
        AugmentationsConfig(hflip=True, color_jitter=0.0),
        1008,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


def test_overfits_in_50_steps(tmp_path: Path, tiny_coco_dir: Path) -> None:
    ds = _ds(tiny_coco_dir)
    cfg = TrainConfig(
        run=RunConfig(name="gpu-smoke", output_dir=str(tmp_path), seed=0),
        model=ModelConfig(dtype="bfloat16", gradient_checkpointing=True),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            prompt_mode="text",
            image_size=1008,
        ),
        peft=PEFTConfig(method="lora", scope="vision_decoder"),
        train=TrainHyperparams(
            epochs=25,
            batch_size=1,
            grad_accum_steps=1,
            lr=5e-4,
            lr_schedule="constant",
            warmup_steps=0,
            save_every=50,
            log_every=10,
            box_hint=BoxHintSchedule(p_start=1.0, p_end=0.0, decay_steps=25),
            num_workers=0,
        ),
    )

    class _RecordingTracker(NoopTracker):
        def __init__(self) -> None:
            self.scalars: list[tuple[int, dict[str, float]]] = []

        def log_scalars(self, step: int, values: dict[str, float]) -> None:
            self.scalars.append((step, values))

        def log_images(self, step: int, images: dict[str, object]) -> None:
            pass

        def close(self) -> None:
            pass

    tracker = _RecordingTracker()
    wrapper = load_sam31(cfg.model).cuda()
    apply_lora(wrapper, cfg.peft)
    trainer = Trainer(wrapper, ds, ds, tracker, cfg)
    trainer.fit()

    losses = [s["loss/total"] for _, s in tracker.scalars if s["loss/total"] > 0]
    assert losses, "expected at least one logged scalar window"
    assert losses[-1] <= 0.7 * losses[0], (
        f"expected ≥30% loss drop; got start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
