import pytest

from custom_sam_peft.errors import (
    CheckpointError,
    ConfigError,
    CustomSamPeftError,
    DataError,
    ModelError,
)
from custom_sam_peft.errors import (
    EnvironmentError as CSPEnvironmentError,
)


def test_base_class_exists():
    assert issubclass(CustomSamPeftError, Exception)


@pytest.mark.parametrize(
    "subclass",
    [ConfigError, DataError, ModelError, CheckpointError, CSPEnvironmentError],
)
def test_subclasses_inherit_base(subclass):
    assert issubclass(subclass, CustomSamPeftError)


def test_config_error_carries_field_path():
    err = ConfigError("bad value", field_path="data.train.path")
    assert err.field_path == "data.train.path"
    assert "data.train.path" in str(err)


def test_environment_error_carries_precondition():
    err = CSPEnvironmentError("missing checkpoint", precondition="checkpoint_present")
    assert err.precondition == "checkpoint_present"
    assert "checkpoint_present" in str(err)


def test_subclasses_can_be_caught_at_base():
    with pytest.raises(CustomSamPeftError):
        raise ConfigError("x", field_path="a.b")


# --- New field tests (Task 6.3) ---


def test_base_error_default_fields_are_none():
    err = CustomSamPeftError("bare message")
    assert err.expected is None
    assert err.found is None
    assert err.fix is None


def test_base_error_carries_optional_fields():
    err = CustomSamPeftError(
        "something failed",
        expected="a thing",
        found="another thing",
        fix="do this",
    )
    assert err.expected == "a thing"
    assert err.found == "another thing"
    assert err.fix == "do this"


def test_config_error_carries_all_new_fields():
    err = ConfigError(
        "bad value",
        field_path="data.train.path",
        expected="an existing directory",
        found="/no/such/dir (does not exist)",
        fix="create the directory or update data.train.path",
    )
    assert err.field_path == "data.train.path"
    assert err.expected == "an existing directory"
    assert err.found == "/no/such/dir (does not exist)"
    assert err.fix == "create the directory or update data.train.path"


def test_config_error_new_fields_default_to_none():
    err = ConfigError("old style", field_path="a.b")
    assert err.expected is None
    assert err.found is None
    assert err.fix is None


def test_environment_error_carries_all_new_fields():
    err = CSPEnvironmentError(
        "no GPU",
        precondition="cuda_available",
        expected="CUDA-capable GPU",
        found="no GPU detected",
        fix="run on a machine with a GPU or use --device cpu",
    )
    assert err.precondition == "cuda_available"
    assert err.expected == "CUDA-capable GPU"
    assert err.found == "no GPU detected"
    assert err.fix == "run on a machine with a GPU or use --device cpu"


def test_environment_error_new_fields_default_to_none():
    err = CSPEnvironmentError("missing dep", precondition="bitsandbytes_installed")
    assert err.expected is None
    assert err.found is None
    assert err.fix is None


def test_data_error_carries_fields():
    err = DataError("dataset missing", expected="COCO JSON", fix="download the dataset")
    assert err.expected == "COCO JSON"
    assert err.fix == "download the dataset"
    assert err.found is None


def test_model_error_carries_fields():
    err = ModelError("build failed", found="weights file missing")
    assert err.found == "weights file missing"
    assert err.expected is None
    assert err.fix is None


def test_checkpoint_error_carries_fields():
    err = CheckpointError(
        "mismatch",
        expected="lora checkpoint",
        found="qlora checkpoint",
        fix="use --resume with the correct checkpoint path",
    )
    assert err.expected == "lora checkpoint"
    assert err.found == "qlora checkpoint"
    assert err.fix == "use --resume with the correct checkpoint path"
