"""Channel-semantics registry: decouples channel COUNT from channel SEMANTICS.

The semantic profile (NOT the count) drives whether a channel adapter is built,
how it is initialized, the normalize default, and the augmentation regime.
Add a new semantic by adding ONE entry here (spec §1.3, §14 note); the adapter,
normalize, and augmentation logic read the profile flags, never a hardcoded name.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ChannelSemanticsProfile:
    """Per-semantic treatment profile. See spec §1.2 / §1.3."""

    allowed_channels: Collection[int]
    use_adapter: bool  # False only for rgb (passthrough)
    adapter_init: Literal["average_broadcast", "identity_passthrough"]
    photometric: bool  # True for rgb/rgba/grayscale; False for freeform
    # (mean, std) tuple, or None when explicit stats are required (freeform).
    normalize_default: tuple[tuple[float, ...], tuple[float, ...]] | None


CHANNEL_SEMANTICS: dict[str, ChannelSemanticsProfile] = {
    "rgb": ChannelSemanticsProfile(
        allowed_channels=frozenset({3}),
        use_adapter=False,
        adapter_init="average_broadcast",  # unused when use_adapter=False (passthrough)
        photometric=True,
        normalize_default=(_IMAGENET_MEAN, _IMAGENET_STD),
    ),
    "rgba": ChannelSemanticsProfile(
        allowed_channels=frozenset({4}),
        use_adapter=True,
        adapter_init="identity_passthrough",
        photometric=True,
        normalize_default=((*_IMAGENET_MEAN, 0.5), (*_IMAGENET_STD, 0.5)),
    ),
    "grayscale": ChannelSemanticsProfile(
        allowed_channels=frozenset({1}),
        use_adapter=True,
        adapter_init="average_broadcast",
        photometric=True,
        normalize_default=((0.449,), (0.226,)),
    ),
    "freeform": ChannelSemanticsProfile(
        allowed_channels=range(1, 17),
        use_adapter=True,
        adapter_init="average_broadcast",
        photometric=False,
        normalize_default=None,
    ),
}

# Tuple of the registry keys, for the schema Literal and membership checks.
CHANNEL_SEMANTIC_NAMES: tuple[str, ...] = tuple(CHANNEL_SEMANTICS)
