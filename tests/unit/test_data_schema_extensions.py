"""Tests for the new data-loading config schema additions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import TextPromptConfig


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


from custom_sam_peft.config.schema import NormalizeConfig


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


from custom_sam_peft.config.schema import HFFieldMap


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


from custom_sam_peft.config.schema import HFDatasetConfig


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

from custom_sam_peft.config.schema import DataConfig, TrainConfig


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


# ---------------------------------------------------------------------------
# Task 4 + 5: channels / channel_semantics / normalize cross-validation
# ---------------------------------------------------------------------------


def _make_data(**kw):
    base = dict(
        format="coco",
        train={"annotations": "a.json", "images": "imgs"},
        prompt_mode="text",
    )
    base.update(kw)
    return DataConfig.model_validate(base)


def test_channels_defaults_to_three_and_semantic_rgb():
    d = _make_data()
    assert d.channels == 3
    assert d.channel_semantics == "rgb"


def test_channels_accepts_1_and_16_rejects_0_and_17():
    from pydantic import ValidationError as PydanticValidationError

    _make_data(channels=1, channel_semantics="grayscale")
    _make_data(
        channels=16, channel_semantics="freeform", normalize={"mean": [0.5] * 16, "std": [0.2] * 16}
    )
    with pytest.raises(PydanticValidationError):
        _make_data(channels=0)
    with pytest.raises(PydanticValidationError):
        _make_data(channels=17)


def test_channel_semantics_membership():
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        _make_data(channel_semantics="hyperspectral")


def test_semantic_channels_mismatch_rejected():
    with pytest.raises(ValueError, match=r"channel_semantics='rgba' requires .*channels=4"):
        _make_data(channels=3, channel_semantics="rgba")
    with pytest.raises(ValueError, match=r"channel_semantics='grayscale' requires .*channels=1"):
        _make_data(channels=3, channel_semantics="grayscale")


def test_rgb_default_fills_imagenet_when_omitted():
    d = _make_data()  # rgb, channels=3, no normalize
    assert d.normalize.mean == [0.485, 0.456, 0.406]
    assert d.normalize.std == [0.229, 0.224, 0.225]


def test_rgba_default_fills_imagenet_plus_alpha_len4():
    d = _make_data(channels=4, channel_semantics="rgba")
    assert d.normalize.mean == [0.485, 0.456, 0.406, 0.5]
    assert len(d.normalize.mean) == 4 == len(d.normalize.std)


def test_grayscale_default_fills_luminance_len1():
    d = _make_data(channels=1, channel_semantics="grayscale")
    assert d.normalize.mean == [0.449]
    assert d.normalize.std == [0.226]


def test_freeform_without_explicit_stats_rejected():
    with pytest.raises(ValueError, match=r"channel_semantics='freeform' requires explicit"):
        _make_data(channels=5, channel_semantics="freeform")


def test_freeform_with_explicit_stats_ok():
    d = _make_data(
        channels=5,
        channel_semantics="freeform",
        normalize={"mean": [0.1, 0.2, 0.3, 0.4, 0.5], "std": [0.1] * 5},
    )
    assert len(d.normalize.mean) == 5


def test_normalize_length_must_match_channels():
    with pytest.raises(ValueError, match=r"normalize\.mean has 3 entries but .*channels=5"):
        _make_data(
            channels=5,
            channel_semantics="freeform",
            normalize={"mean": [0.1, 0.2, 0.3], "std": [0.1, 0.2, 0.3]},
        )


def test_explicit_normalize_wrong_length_rejected_for_named_semantic():
    with pytest.raises(ValueError, match=r"data\.normalize\.mean has 4 entries but .*channels=3"):
        _make_data(
            channels=3, channel_semantics="rgb", normalize={"mean": [0.1] * 4, "std": [0.2] * 4}
        )
