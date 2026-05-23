import pytest
import torch

from custom_sam_peft.runtime import Runtime, to_device


def test_runtime_fields_default_world_size_1():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    assert rt.world_size == 1
    assert rt.is_primary is True


def test_runtime_from_config_resolves_bfloat16():
    rt = Runtime.from_config(device="cpu", dtype="bfloat16")
    assert rt.dtype is torch.bfloat16


def test_runtime_from_config_resolves_float16():
    rt = Runtime.from_config(device="cpu", dtype="float16")
    assert rt.dtype is torch.float16


def test_runtime_from_config_rejects_unknown_dtype():
    from custom_sam_peft.errors import ConfigError

    with pytest.raises(ConfigError):
        Runtime.from_config(device="cpu", dtype="quadruple")


def test_to_device_moves_tensor():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    x = torch.zeros(3)
    y = to_device(x, rt)
    assert y.device == torch.device("cpu")


def test_to_device_recurses_into_dict():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    batch = {"img": torch.zeros(3), "label": torch.ones(2)}
    out = to_device(batch, rt)
    assert out["img"].device == torch.device("cpu")
    assert out["label"].device == torch.device("cpu")


def test_to_device_recurses_into_list():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    batch = [torch.zeros(3), torch.ones(2)]
    out = to_device(batch, rt)
    assert out[0].device == torch.device("cpu")


def test_to_device_passes_through_non_tensor():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    assert to_device("hello", rt) == "hello"
    assert to_device(42, rt) == 42
