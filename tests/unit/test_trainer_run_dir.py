"""End-to-end Trainer.fit() on the stub: verify run-dir layout."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from esam3.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.data.base import Example, Instance, TextPrompts
from esam3.peft_adapters.lora import apply_lora
from esam3.tracking.noop import NoopTracker
from esam3.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper


class _TinyTextDataset:
    """Two-example dataset with text prompts, suitable for the stub wrapper."""

    def __init__(self) -> None:
        self._examples = [
            Example(
                image=torch.zeros(3, 8, 8),
                image_id=f"img{i}",
                prompts=TextPrompts(classes=["A"]),
                instances=[
                    Instance(
                        mask=torch.zeros(8, 8, dtype=torch.bool),
                        class_id=0,
                        box=torch.tensor([1.0, 1.0, 5.0, 5.0]),
                    )
                ],
            )
            for i in range(2)
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]

    @property
    def class_names(self) -> list[str]:
        return ["A"]


def test_fit_creates_expected_layout(tmp_path: Path) -> None:
    ds = _TinyTextDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="layout-test", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
        train=TrainHyperparams(
            epochs=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )
    apply_lora(wrapper, cfg.peft)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    result = trainer.fit()
    rd = result.run_dir
    assert rd.exists()
    assert (rd / "config.yaml").exists()
    assert (rd / "adapter" / "adapter_config.json").exists()
    assert (rd / "metrics.json").exists()
    assert (rd / "checkpoints").exists()
    assert result.final_metrics is None
    assert result.merged_path is None
    payload = json.loads((rd / "metrics.json").read_text())
    assert "global_step" in payload
