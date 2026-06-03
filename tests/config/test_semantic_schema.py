# tests/config/test_semantic_schema.py  (new file alongside the existing schema tests)
"""Schema coverage for the #113 task axis + semantic data config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import TrainConfig


def _base_cfg(**over):
    """Minimal valid instance config dict; callers override task/data/etc.

    Adjustments from task template to match the real schema:
    - DataConfig uses top-level `train`/`val` fields (not nested under `splits`)
    - TrainHyperparams requires `epochs` (PositiveInt)
    """
    cfg = {
        "run": {"name": "t", "output_dir": "runs/t"},
        "data": {
            "format": "coco",
            "train": {"images": "img", "annotations": "ann.json"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1},
    }
    cfg.update(over)
    return cfg


def test_task_defaults_to_instance():
    cfg = TrainConfig.model_validate(_base_cfg())
    assert cfg.task == "instance"


def test_semantic_rejects_coco_format():
    with pytest.raises(ValidationError, match=r"does not support data\.format: coco"):
        TrainConfig.model_validate(
            _base_cfg(
                task="semantic",
                data={
                    "format": "coco",
                    "train": {"images": "img", "annotations": "ann.json"},
                },
            )
        )


def test_semantic_requires_data_semantic():
    with pytest.raises(ValidationError, match=r"requires data\.semantic"):
        TrainConfig.model_validate(
            _base_cfg(
                task="semantic",
                data={
                    "format": "mask_png",
                    "train": {"images": "img", "annotations": "labels"},
                },
            )
        )


def test_instance_rejects_data_semantic():
    with pytest.raises(ValidationError, match=r"data\.semantic is only valid"):
        TrainConfig.model_validate(
            _base_cfg(
                data={
                    "format": "hf",
                    "train": {"images": "x", "annotations": "y"},
                    "semantic": {"class_map": "cm.json"},
                    "hf": {"name": "some/dataset"},
                },
            )
        )


def test_instance_rejects_mask_png_format():
    with pytest.raises(ValidationError, match="mask_png requires task: semantic"):
        TrainConfig.model_validate(
            _base_cfg(
                data={
                    "format": "mask_png",
                    "train": {"images": "img", "annotations": "labels"},
                },
            )
        )


def test_semantic_mask_png_valid():
    cfg = TrainConfig.model_validate(
        _base_cfg(
            task="semantic",
            data={
                "format": "mask_png",
                "train": {"images": "img", "annotations": "labels"},
                "semantic": {"class_map": "cm.json"},
            },
        )
    )
    assert cfg.task == "semantic"
    assert cfg.data.semantic is not None
    assert cfg.data.semantic.ignore_index == 255  # default void
    assert cfg.data.semantic.label_suffix == "_labelIds.png"


def test_semantic_rejects_nondefault_eval_iou_thresholds():
    with pytest.raises(ValidationError, match="iou_thresholds"):
        TrainConfig.model_validate(
            _base_cfg(
                task="semantic",
                data={
                    "format": "mask_png",
                    "train": {"images": "img", "annotations": "labels"},
                    "semantic": {"class_map": "cm.json"},
                },
                eval={"iou_thresholds": [0.5, 0.75]},
            )
        )
