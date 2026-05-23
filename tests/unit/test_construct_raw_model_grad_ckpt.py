"""CPU test: the static entry point invokes the activation-checkpoint patch.

The real sam3 model is unavailable in a CPU-deterministic way, so we monkeypatch
sam3.build_sam3_image_model to return a synthetic ViT-Det stand-in and spy on
_patch_enable_vit_act_checkpoint to confirm _construct_raw_model calls it when
cfg.gradient_checkpointing is True (and does NOT call it when False).

Monkeypatching notes:
- ``sam3.build_sam3_image_model`` is called in _construct_raw_model as
  ``sam3.build_sam3_image_model(...)`` where ``sam3`` is the top-level
  ``import sam3`` module bound into custom_sam_peft.models.sam3.  We reach
  it via ``sam3_mod.sam3.build_sam3_image_model``.
- ``_locate_weights`` is patched at ``sam3_mod._locate_weights`` so no
  real checkpoint resolution is attempted.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

import custom_sam_peft.models.sam3 as sam3_mod
from custom_sam_peft.config.schema import ModelConfig


class _FakeBlk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_act_checkpoint = False
        self.lin = nn.Linear(2, 2)


class _FakeRawModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlk(), _FakeBlk()])
        # No set_grad_checkpointing — forces the patch branch.


def _install_fake_build(monkeypatch, model: nn.Module) -> None:
    monkeypatch.setattr(sam3_mod, "_locate_weights", lambda cfg: Path("/tmp/fake.pt"))  # noqa: S108
    monkeypatch.setattr(sam3_mod.sam3, "build_sam3_image_model", lambda **kw: model)


def _cfg(grad_ckpt: bool) -> ModelConfig:
    return ModelConfig(gradient_checkpointing=grad_ckpt, device="cpu")


def test_construct_raw_model_invokes_patch_when_enabled(monkeypatch) -> None:
    model = _FakeRawModel()
    _install_fake_build(monkeypatch, model)
    calls: list[nn.Module] = []
    monkeypatch.setattr(sam3_mod, "_patch_enable_vit_act_checkpoint", lambda m: calls.append(m))
    out = sam3_mod._construct_raw_model(_cfg(True))
    assert calls == [out], "patch not invoked exactly once on the raw model"


def test_construct_raw_model_skips_patch_when_disabled(monkeypatch) -> None:
    model = _FakeRawModel()
    _install_fake_build(monkeypatch, model)
    calls: list[nn.Module] = []
    monkeypatch.setattr(sam3_mod, "_patch_enable_vit_act_checkpoint", lambda m: calls.append(m))
    sam3_mod._construct_raw_model(_cfg(False))
    assert calls == [], "patch invoked when gradient_checkpointing=False"
