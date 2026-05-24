"""Task 16 / spec §10.1 C10: channel-adapter params are collected by the trainable set.

Collection works automatically via requires_grad; this test guards against a future
refactor silently dropping the adapter params from the optimizer's parameter set.
"""

from __future__ import annotations

import torch.nn as nn

from custom_sam_peft.models.sam3 import _Sam3ImageAdapter


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        for p in self.parameters():
            p.requires_grad_(False)  # base frozen


def test_C10_optimizer_includes_channel_adapter_params() -> None:
    adapter = _Sam3ImageAdapter(_StubModel(), channels=4, channel_semantics="freeform")
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    ca_params = set(map(id, adapter.channel_adapter.parameters()))
    assert ca_params.issubset(set(map(id, trainable)))
    assert len(trainable) == 2  # exactly the adapter weight + bias


def test_C10_rgb_has_no_trainable_adapter_params() -> None:
    adapter = _Sam3ImageAdapter(_StubModel(), channels=3, channel_semantics="rgb")
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    assert trainable == []  # rgb: no adapter, base frozen
