"""Unit tests for the vit_act_checkpoint patch — CPU-only, synthetic modules.

The patch iterates an ``nn.Module`` tree and sets ``use_act_checkpoint=True``
on every submodule already exposing that attribute (sam3's ViT-Det blocks).
The contract is attribute-level and sam3-agnostic, so the tests use synthetic
stand-ins rather than instantiating a full sam3 model.

GPU-only behavior NOT covered here: the deterministic-autocast wrap added in
the Phase-1 fix (Fix A item b) and the recompute-determinism guarantee. Those
are verified in tests/gpu/test_grad_checkpointing.py on a real T4.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from custom_sam_peft.models._patches import vit_act_checkpoint
from custom_sam_peft.runtime._runtime import Runtime

_CPU_RUNTIME = Runtime(device=torch.device("cpu"), dtype=torch.float32)


class _FakeViTDetBlock(nn.Module):
    """Stand-in for a sam3 ViT-Det block exposing the use_act_checkpoint flag."""

    def __init__(self) -> None:
        super().__init__()
        self.use_act_checkpoint = False
        self.lin = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FakeNonCheckpointable(nn.Module):
    """Stand-in for a module that does NOT expose the checkpoint flag."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)


class _FakeModel(nn.Module):
    def __init__(self, n_blocks: int = 4, with_non: bool = True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeViTDetBlock() for _ in range(n_blocks)])
        if with_non:
            self.other = _FakeNonCheckpointable()


def test_flips_use_act_checkpoint_on_every_exposing_block() -> None:
    model = _FakeModel(n_blocks=3)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is False
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is True


def test_skips_modules_without_the_attribute() -> None:
    model = _FakeModel(n_blocks=2)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    assert not hasattr(model.other, "use_act_checkpoint")


def test_idempotent_double_apply() -> None:
    model = _FakeModel(n_blocks=2)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is True
        assert getattr(blk, vit_act_checkpoint._SENTINEL_ATTR, False) is True


def test_warns_when_no_exposing_modules(caplog: pytest.LogCaptureFixture) -> None:
    model = nn.Linear(2, 2)  # no use_act_checkpoint anywhere in the tree
    _logger = "custom_sam_peft.models._patches.vit_act_checkpoint"
    with caplog.at_level(logging.WARNING, logger=_logger):
        vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    assert any("ZERO" in rec.message for rec in caplog.records), [r.message for r in caplog.records]


def test_logs_positive_count(caplog: pytest.LogCaptureFixture) -> None:
    model = _FakeModel(n_blocks=5)
    with caplog.at_level(logging.INFO, logger="custom_sam_peft.models._patches.vit_act_checkpoint"):
        vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    messages = [rec.message for rec in caplog.records]
    assert any("5" in m and "checkpoint" in m.lower() for m in messages), messages
