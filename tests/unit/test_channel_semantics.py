import dataclasses

import pytest

from custom_sam_peft.data.channel_semantics import (
    CHANNEL_SEMANTIC_NAMES,
    CHANNEL_SEMANTICS,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def test_registry_has_four_shipped_semantics():
    assert set(CHANNEL_SEMANTICS) == {"rgb", "rgba", "grayscale", "freeform"}
    assert set(CHANNEL_SEMANTIC_NAMES) == set(CHANNEL_SEMANTICS)


def test_rgb_profile_passthrough_imagenet():
    p = CHANNEL_SEMANTICS["rgb"]
    assert p.allowed_channels == {3}
    assert p.use_adapter is False
    assert p.photometric is True
    assert p.normalize_default == (_IMAGENET_MEAN, _IMAGENET_STD)


def test_rgba_profile_identity_passthrough_alpha_default():
    p = CHANNEL_SEMANTICS["rgba"]
    assert p.allowed_channels == {4}
    assert p.use_adapter is True
    assert p.adapter_init == "identity_passthrough"
    assert p.photometric is True
    mean, std = p.normalize_default
    assert mean == (*_IMAGENET_MEAN, 0.5)
    assert std == (*_IMAGENET_STD, 0.5)


def test_grayscale_profile_luminance_default():
    p = CHANNEL_SEMANTICS["grayscale"]
    assert p.allowed_channels == {1}
    assert p.use_adapter is True
    assert p.adapter_init == "average_broadcast"
    assert p.photometric is True
    assert p.normalize_default == ((0.449,), (0.226,))


def test_freeform_profile_no_default_range_channels():
    p = CHANNEL_SEMANTICS["freeform"]
    assert set(p.allowed_channels) == set(range(1, 17))
    assert p.use_adapter is True
    assert p.adapter_init == "average_broadcast"
    assert p.photometric is False
    assert p.normalize_default is None


def test_profile_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CHANNEL_SEMANTICS["rgb"].use_adapter = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "semantic,channel,ok",
    [
        ("rgb", 3, True),
        ("rgb", 4, False),
        ("rgba", 4, True),
        ("rgba", 3, False),
        ("grayscale", 1, True),
        ("grayscale", 3, False),
        ("freeform", 1, True),
        ("freeform", 16, True),
        ("freeform", 17, False),
    ],
)
def test_allowed_channels_membership(semantic, channel, ok):
    assert (channel in CHANNEL_SEMANTICS[semantic].allowed_channels) is ok
