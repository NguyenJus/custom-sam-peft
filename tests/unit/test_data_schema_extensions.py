"""Tests for the new data-loading config schema additions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import TextPromptConfig


def test_text_prompt_config_defaults() -> None:
    cfg = TextPromptConfig()
    assert cfg.mode == "present"
    assert cfg.negatives_per_image == 0
    assert cfg.k == 16


def test_text_prompt_config_k_bounded() -> None:
    with pytest.raises(ValidationError):
        TextPromptConfig(k=17)
    with pytest.raises(ValidationError):
        TextPromptConfig(k=0)


from esam3.config.schema import NormalizeConfig


def test_normalize_config_defaults() -> None:
    cfg = NormalizeConfig()
    assert cfg.mean == [0.485, 0.456, 0.406]
    assert cfg.std == [0.229, 0.224, 0.225]
    assert len(cfg.mean) == 3 and len(cfg.std) == 3


def test_normalize_config_validation_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.1, 0.2], std=[0.1, 0.1, 0.1])


def test_normalize_config_validation_rejects_nonpositive_std() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.1, 0.1, 0.1], std=[0.0, 0.1, 0.1])


def test_normalize_config_validation_rejects_mean_out_of_range() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[1.5, 0.1, 0.1], std=[0.1, 0.1, 0.1])


from esam3.config.schema import HFFieldMap


def test_hf_field_map_defaults() -> None:
    fm = HFFieldMap()
    assert fm.image == "image"
    assert fm.bbox == "objects.bbox"
    assert fm.category == "objects.category"
    assert fm.segmentation == "objects.segmentation"
    assert fm.categories_feature == "categories"
    assert fm.bbox_format == "xyxy"


def test_hf_field_map_segmentation_can_be_none() -> None:
    fm = HFFieldMap(segmentation=None)
    assert fm.segmentation is None


def test_hf_field_map_rejects_invalid_bbox_format() -> None:
    with pytest.raises(ValidationError):
        HFFieldMap(bbox_format="cxcywh")  # type: ignore[arg-type]


from esam3.config.schema import HFDatasetConfig


def test_hf_dataset_config_required_name() -> None:
    with pytest.raises(ValidationError):
        HFDatasetConfig()  # type: ignore[call-arg]


def test_hf_dataset_config_defaults() -> None:
    cfg = HFDatasetConfig(name="my-org/my-ds")
    assert cfg.name == "my-org/my-ds"
    assert cfg.split_train == "train"
    assert cfg.split_val == "validation"
    assert cfg.field_map.bbox == "objects.bbox"


def test_hf_dataset_config_name_min_length() -> None:
    with pytest.raises(ValidationError):
        HFDatasetConfig(name="")


from pathlib import Path

from esam3.config.schema import DataConfig, TrainConfig


def _minimal_data(format: str = "coco") -> dict[str, object]:
    return {
        "format": format,
        "train": {"annotations": "a.json", "images": "imgs/"},
        "val": {"annotations": "a.json", "images": "imgs/"},
        "prompt_mode": "bbox",
    }


def test_data_config_accepts_coco_without_hf() -> None:
    cfg = DataConfig.model_validate(_minimal_data("coco"))
    assert cfg.hf is None
    assert cfg.text_prompt.mode == "present"
    assert cfg.normalize.mean == [0.485, 0.456, 0.406]


def test_data_config_requires_hf_when_format_hf() -> None:
    with pytest.raises(ValidationError) as exc:
        DataConfig.model_validate(_minimal_data("hf"))
    assert "data.hf" in str(exc.value)


def test_data_config_accepts_hf_with_hf_block() -> None:
    d = _minimal_data("hf")
    d["hf"] = {"name": "cppe-5"}
    cfg = DataConfig.model_validate(d)
    assert cfg.hf is not None
    assert cfg.hf.name == "cppe-5"


def test_data_config_accepts_text_prompt_override() -> None:
    d = _minimal_data("coco")
    d["text_prompt"] = {"mode": "present_plus_negatives", "negatives_per_image": 3}
    cfg = DataConfig.model_validate(d)
    assert cfg.text_prompt.mode == "present_plus_negatives"
    assert cfg.text_prompt.negatives_per_image == 3


def test_existing_example_yaml_still_validates() -> None:
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    for name in ("coco_text_lora.yaml", "coco_text_qlora.yaml"):
        p = repo_root / "configs" / "examples" / name
        raw = yaml.safe_load(p.read_text())
        TrainConfig.model_validate(raw)
