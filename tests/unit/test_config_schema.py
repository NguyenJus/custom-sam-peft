"""Tests for the pydantic config schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import NormalizeConfig, TrainConfig


def _minimal_dict() -> dict[str, object]:
    return {
        "run": {"name": "test-run", "output_dir": "./runs", "seed": 42},
        "model": {"name": "facebook/sam3.1"},
        "data": {
            "format": "coco",
            "train": {"annotations": "data/train.json", "images": "data/train/"},
            "val": {"annotations": "data/val.json", "images": "data/val/"},
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
    assert cfg.train.optimizer == "auto"
    assert cfg.tracking.backend == "tensorboard"


def test_invalid_dtype_rejected() -> None:
    d = _minimal_dict()
    d["model"]["dtype"] = "float32"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_prompt_mode_rejected_by_schema() -> None:
    """Spec #126 §6: any prompt_mode key (regardless of value) fails at load.

    The schema is the sole gate; `_Strict`'s extra="forbid" rejects the key
    with a Pydantic ValidationError of type "extra_forbidden".
    """
    d = _minimal_dict()
    # _minimal_dict() now constructs a payload without prompt_mode (Step 2).
    # Add it back as an extra key and assert it is rejected.
    for value in ("text", "bbox", "something_else"):
        d["data"]["prompt_mode"] = value  # type: ignore[index]
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig.model_validate(d)
        errors = exc_info.value.errors()
        assert any(
            e["type"] == "extra_forbidden" and e["loc"][-1] == "prompt_mode" for e in errors
        ), f"expected extra_forbidden on data.prompt_mode for value={value!r}; got {errors}"


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


def test_negative_learning_rate_rejected() -> None:
    d = _minimal_dict()
    d["train"]["learning_rate"] = -1.0  # type: ignore[index]
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


def test_data_image_size_key_rejected() -> None:
    """data.image_size must be rejected — removed field; guard against silent re-introduction."""
    d = _minimal_dict()
    d["data"]["image_size"] = 1008  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_qlora_subconfig_defaults() -> None:
    d = _minimal_dict()
    d["peft"]["method"] = "qlora"  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.peft.qlora.quant_type == "nf4"
    assert cfg.peft.qlora.compute_dtype == "bfloat16"


def test_all_public_submodels_are_importable() -> None:
    """Smoke check that every user-facing sub-model is a public attribute of schema.

    Note: ExportConfig, WandbConfig, LossConfig, MatcherWeights have been moved
    to config._internal (audit Section G) and are re-exported from schema for
    backward compatibility. They are excluded from the user-facing set here.
    """
    from custom_sam_peft.config import schema

    # User-facing Pydantic models (audit Section G: user-set fields)
    expected_user_facing = {
        "AugmentationsConfig",
        "DataConfig",
        "DataSplit",
        "EvalConfig",
        "HFDatasetConfig",
        "HFFieldMap",
        "ModelConfig",
        "NormalizeConfig",
        "PEFTConfig",
        "QLoRAConfig",
        "RunConfig",
        "TextPromptConfig",
        "TrackingConfig",
        "TrainConfig",
        "TrainHyperparams",
    }
    missing = {n for n in expected_user_facing if not hasattr(schema, n)}
    assert missing == set(), f"missing user-facing sub-models: {missing}"

    # Internal classes re-exported for backward compatibility
    expected_internal_reexported = {"ExportConfig", "LossConfig", "MatcherWeights", "WandbConfig"}
    missing_reexport = {n for n in expected_internal_reexported if not hasattr(schema, n)}
    assert missing_reexport == set(), (
        f"internal classes no longer re-exported from schema (update consumers): {missing_reexport}"
    )


def test_peft_defaults_include_scope_and_bias() -> None:
    d = _minimal_dict()
    cfg = TrainConfig.model_validate(d)
    assert cfg.peft.scope == "vision_decoder"
    assert cfg.peft.bias == "none"
    assert cfg.peft.target_modules is None


def test_peft_scope_invalid_value_rejected() -> None:
    d = _minimal_dict()
    d["peft"]["scope"] = "encoder"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_peft_bias_invalid_value_rejected() -> None:
    d = _minimal_dict()
    d["peft"]["bias"] = "some"  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_peft_target_modules_accepts_explicit_list() -> None:
    d = _minimal_dict()
    d["peft"]["target_modules"] = ["vision_encoder.block0.attn.qkv"]  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.peft.target_modules == ["vision_encoder.block0.attn.qkv"]


def test_peft_target_modules_and_scope_both_set_validates() -> None:
    # Pydantic does not enforce precedence; apply_lora does. Both should validate.
    d = _minimal_dict()
    d["peft"]["scope"] = "all"  # type: ignore[index]
    d["peft"]["target_modules"] = ["foo"]  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.peft.scope == "all"
    assert cfg.peft.target_modules == ["foo"]


# ---------------------------------------------------------------------------
# Task 1: EvalConfig extensions + DataConfig.test
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_data_config_dict() -> dict:
    return {
        "format": "coco",
        "train": {"annotations": "t.json", "images": "t/"},
        "val": {"annotations": "v.json", "images": "v/"},
    }


def test_eval_config_defaults_extended() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    c = EvalConfig()
    assert c.mode == "full"
    assert c.lite_max_images == 64
    assert c.mask_threshold == 0.0
    assert c.save_predictions is False


def test_eval_config_mode_literal_validated() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    EvalConfig(mode="lite")
    with pytest.raises(ValidationError):
        EvalConfig(mode="medium")  # type: ignore[arg-type]


def test_eval_config_lite_max_images_must_be_positive() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    with pytest.raises(ValidationError):
        EvalConfig(lite_max_images=0)


def test_data_config_test_defaults_to_none(minimal_data_config_dict: dict) -> None:
    from custom_sam_peft.config.schema import DataConfig

    cfg = DataConfig(**minimal_data_config_dict)
    assert cfg.test is None


def test_data_config_test_accepts_data_split(minimal_data_config_dict: dict) -> None:
    from custom_sam_peft.config.schema import DataConfig

    minimal_data_config_dict["test"] = {"annotations": "a.json", "images": "img/"}
    cfg = DataConfig(**minimal_data_config_dict)
    assert cfg.test is not None
    assert cfg.test.annotations == "a.json"


# ---------------------------------------------------------------------------
# spec/data-no-val-auto-split (#71): optional val + val_split + validators
# ---------------------------------------------------------------------------


def test_val_null_validates() -> None:
    """data.val: null resolves to no-val mode; must not raise."""
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val is None
    assert cfg.data.val_split is None


def test_val_omitted_validates() -> None:
    """Omitting data.val entirely also resolves to no-val mode."""
    d = _minimal_dict()
    del d["data"]["val"]  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val is None
    assert cfg.data.val_split is None


def test_val_and_val_split_mutually_exclusive() -> None:
    d = _minimal_dict()
    d["data"]["val_split"] = {"fraction": 0.1}  # type: ignore[index]
    # val is still present from _minimal_dict.
    with pytest.raises(ValidationError, match="mutually exclusive"):
        TrainConfig.model_validate(d)


def test_val_split_fraction_above_half_rejected() -> None:
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.6}  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_val_split_fraction_zero_or_negative_rejected() -> None:
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.0}  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)
    d["data"]["val_split"] = {"fraction": -0.1}  # type: ignore[index]
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(d)


def test_hf_split_val_custom_with_val_split_rejected() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {  # type: ignore[index]
        "name": "tiny/dataset",
        "split_train": "train",
        "split_val": "custom_val",
    }
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.1}  # type: ignore[index]
    with pytest.raises(ValidationError, match="split_val cannot be customized"):
        TrainConfig.model_validate(d)


def test_hf_split_val_default_with_val_split_validates() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {"name": "tiny/dataset"}  # default split_val=None
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.1, "seed": 7}  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val_split is not None
    assert cfg.data.val_split.fraction == 0.1
    assert cfg.data.val_split.seed == 7


def test_hf_split_val_set_without_val_split_validates() -> None:
    """spec §12.3: HF + named split_val (no val/val_split) is the explicit opt-in."""
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {"name": "tiny/dataset", "split_val": "myval"}  # type: ignore[index]
    d["data"]["val"] = None  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val_split is None
    assert cfg.data.hf is not None
    assert cfg.data.hf.split_val == "myval"


def test_neither_val_nor_val_split_validates() -> None:
    """Spec §3.3: neither set → resolves to no-val mode (WARN at resolve, not validation)."""
    d = _minimal_dict()
    d["data"]["val"] = None  # type: ignore[index]
    # val_split is not present in _minimal_dict.
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val is None
    assert cfg.data.val_split is None


# ---------------------------------------------------------------------------
# spec/domain-aware-augmentation-presets (#75): preset/intensity/overrides
# ---------------------------------------------------------------------------


def test_augmentations_default_preset_and_intensity() -> None:
    from custom_sam_peft.config.schema import AugmentationsConfig

    cfg = AugmentationsConfig()
    assert cfg.preset == "natural"
    assert cfg.intensity == "medium"


def test_augmentation_overrides_rejects_unknown_keys() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationOverrides

    with pytest.raises(ValidationError):
        AugmentationOverrides.model_validate({"hfilp": True})  # typo


def test_augmentations_preset_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationsConfig

    with pytest.raises(ValidationError):
        AugmentationsConfig.model_validate({"preset": "mediacl"})  # typo


def test_augmentations_intensity_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationsConfig

    with pytest.raises(ValidationError):
        AugmentationsConfig.model_validate({"intensity": "medum"})  # typo


def test_augmentations_overrides_default_factory_isolation() -> None:
    """Two AugmentationsConfig() instances must not share a single overrides object."""
    from custom_sam_peft.config.schema import AugmentationsConfig

    a = AugmentationsConfig()
    b = AugmentationsConfig()
    assert a.overrides is not b.overrides


def test_augmentations_overrides_all_none_by_default() -> None:
    from custom_sam_peft.config.schema import AugmentationsConfig

    dumped = AugmentationsConfig().overrides.model_dump()
    assert all(v is None for v in dumped.values())
    assert set(dumped.keys()) == {
        "hflip",
        "vflip",
        "rotate90",
        "rotate_arbitrary",
        "color_jitter",
        "stain_jitter",
        "blur",
        "gauss_noise",
    }


def test_augmentation_overrides_rejects_negative_floats() -> None:
    """Field(ge=0.0) on float overrides catches negative sigma at load time."""
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationOverrides

    with pytest.raises(ValidationError):
        AugmentationOverrides.model_validate({"stain_jitter": -0.1})


# ---------------------------------------------------------------------------
# spec/domain-aware-loss-presets (#112): LossConfig / LossOverrides
# ---------------------------------------------------------------------------


def test_loss_config_defaults() -> None:
    from custom_sam_peft.config.schema import LossConfig

    cfg = LossConfig()
    assert cfg.preset == "natural"
    assert cfg.class_imbalance == "balanced"
    assert cfg.overrides.model_dump() == {
        "mask_family": None,
        "box_family": None,
        "obj_family": None,
        "presence_family": None,
        "w_mask": None,
        "w_box": None,
        "w_obj": None,
        "w_presence": None,
        "focal_gamma": None,
        "focal_alpha": None,
        "tversky_alpha": None,
        "tversky_gamma": None,
        "boundary_weight": None,
        "matcher_weights": None,
    }


def test_loss_config_class_imbalance_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import LossConfig

    with pytest.raises(ValidationError):
        LossConfig(class_imbalance="moderete")  # type: ignore[arg-type]


def test_loss_config_preset_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import LossConfig

    with pytest.raises(ValidationError):
        LossConfig(preset="medecal")  # type: ignore[arg-type]


def test_loss_overrides_rejects_unknown_keys() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import LossOverrides

    with pytest.raises(ValidationError):
        LossOverrides(mask_familty="dice_bce")  # type: ignore[call-arg]


def test_loss_overrides_default_factory_isolation() -> None:
    from custom_sam_peft.config.schema import LossConfig

    a = LossConfig()
    b = LossConfig()
    assert a.overrides is not b.overrides


def test_loss_overrides_family_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import LossOverrides

    with pytest.raises(ValidationError):
        LossOverrides(mask_family="focle_bce")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        LossOverrides(box_family="diou")  # type: ignore[arg-type]


def test_loss_overrides_matcher_weights_dict_coerced() -> None:
    from custom_sam_peft.config._internal import MatcherWeights
    from custom_sam_peft.config.schema import LossOverrides

    o = LossOverrides(matcher_weights={"lambda_mask": 7.0})  # type: ignore[arg-type]
    assert isinstance(o.matcher_weights, MatcherWeights)
    assert o.matcher_weights.lambda_mask == 7.0


def test_loss_overrides_w_box_zero_allowed() -> None:
    from custom_sam_peft.config.schema import LossOverrides

    LossOverrides(w_box=0.0)  # ge=0.0; no exception


def test_loss_overrides_w_mask_zero_rejected() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import LossOverrides

    with pytest.raises(ValidationError):
        LossOverrides(w_mask=0.0)  # PositiveFloat


# ---------------------------------------------------------------------------
# SAM 3.1 multiplex config (#122): MultiplexConfig
# ---------------------------------------------------------------------------


def test_multiplex_config_defaults() -> None:
    from custom_sam_peft.config.schema import MultiplexConfig

    cfg = MultiplexConfig()
    assert cfg.classes_per_forward == 16


def test_multiplex_config_validates_range() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import MultiplexConfig

    with pytest.raises(ValidationError):
        MultiplexConfig(classes_per_forward=0)
    with pytest.raises(ValidationError):
        MultiplexConfig(classes_per_forward=17)


def test_train_hyperparams_has_multiplex_default() -> None:
    from custom_sam_peft.config.schema import MultiplexConfig, TrainHyperparams

    th = TrainHyperparams(epochs=1)
    assert isinstance(th.multiplex, MultiplexConfig)
    assert th.multiplex.classes_per_forward == 16


def test_normalize_accepts_length_one_and_sixteen():
    NormalizeConfig(mean=[0.5], std=[0.2])
    NormalizeConfig(mean=[0.5] * 16, std=[0.2] * 16)


def test_normalize_rejects_length_zero():
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[], std=[])


def test_normalize_rejects_length_seventeen():
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.5] * 17, std=[0.2] * 17)


def test_normalize_max_pixel_value_default_and_override():
    assert NormalizeConfig().max_pixel_value == 255.0
    assert NormalizeConfig(max_pixel_value=1.0).max_pixel_value == 1.0


def test_normalize_max_pixel_value_must_be_positive():
    with pytest.raises(ValidationError):
        NormalizeConfig(max_pixel_value=0.0)
    with pytest.raises(ValidationError):
        NormalizeConfig(max_pixel_value=-1.0)


def test_normalize_keeps_per_value_range_checks():
    with pytest.raises(ValueError, match=r"normalize\.mean values must be in"):
        NormalizeConfig(mean=[1.5], std=[0.2])
    with pytest.raises(ValueError, match=r"normalize\.std values must be > 0"):
        NormalizeConfig(mean=[0.5], std=[0.0])


def test_eval_config_visualize_defaults() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    cfg = EvalConfig()
    assert cfg.visualize is True
    assert cfg.visualize_count == 10


def test_eval_config_visualize_count_must_be_positive() -> None:
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import EvalConfig

    with pytest.raises(ValidationError):
        EvalConfig(visualize_count=0)


def test_box_hint_field_rejected_by_schema() -> None:
    """After #88 removal, train.box_hint is no longer a valid field."""
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import TrainHyperparams

    with pytest.raises(ValidationError):
        TrainHyperparams(epochs=1, box_hint={"p_start": 1.0, "p_end": 0.0})
