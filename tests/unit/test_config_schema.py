"""Tests for the pydantic config schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import TrainConfig


def _minimal_dict() -> dict[str, object]:
    return {
        "run": {"name": "test-run", "output_dir": "./runs", "seed": 42},
        "model": {"name": "facebook/sam3.1"},
        "data": {
            "format": "coco",
            "train": {"annotations": "data/train.json", "images": "data/train/"},
            "val": {"annotations": "data/val.json", "images": "data/val/"},
            "prompt_mode": "bbox",
            "image_size": 1024,
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 10},
        "eval": {},
        "tracking": {"backend": "tensorboard"},
        "export": {"merge": False},
    }


def test_full_config_validates() -> None:
    cfg = TrainConfig.model_validate(_minimal_dict())
    assert cfg.run.name == "test-run"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.peft.method == "lora"
    assert cfg.train.batch_size == 1
    assert cfg.train.grad_accum_steps == 8
    assert cfg.train.optimizer == "adamw"
    assert cfg.tracking.backend == "tensorboard"


def test_invalid_dtype_rejected() -> None:
    d = _minimal_dict()
    d["model"]["dtype"] = "float32"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_prompt_mode_rejected() -> None:
    d = _minimal_dict()
    d["data"]["prompt_mode"] = "points"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_peft_method_rejected() -> None:
    d = _minimal_dict()
    d["peft"]["method"] = "ia3"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_data_format_rejected() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "yolo"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_optimizer_rejected() -> None:
    d = _minimal_dict()
    d["train"]["optimizer"] = "sgd"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_invalid_tracker_backend_rejected() -> None:
    d = _minimal_dict()
    d["tracking"]["backend"] = "mlflow"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_negative_lr_rejected() -> None:
    d = _minimal_dict()
    d["train"]["lr"] = -1.0  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_zero_epochs_rejected() -> None:
    d = _minimal_dict()
    d["train"]["epochs"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_unknown_top_level_key_rejected() -> None:
    d = _minimal_dict()
    d["extra_section"] = {}
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_qlora_subconfig_defaults() -> None:
    d = _minimal_dict()
    d["peft"]["method"] = "qlora"  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.peft.qlora.quant_type == "nf4"
    assert cfg.peft.qlora.compute_dtype == "bfloat16"


def test_all_public_submodels_are_importable() -> None:
    """Smoke check that every documented sub-model is a public attribute of schema."""
    from esam3.config import schema

    expected = {
        "AugmentationsConfig",
        "DataConfig",
        "DataSplit",
        "EvalConfig",
        "ExportConfig",
        "ModelConfig",
        "PEFTConfig",
        "QLoRAConfig",
        "RunConfig",
        "TrackingConfig",
        "TrainConfig",
        "TrainHyperparams",
        "WandbConfig",
    }
    missing = {n for n in expected if not hasattr(schema, n)}
    assert missing == set(), f"missing public sub-models: {missing}"
