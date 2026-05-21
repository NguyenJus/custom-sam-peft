"""Tests for src/custom_sam_peft/presets.py — VRAM-tier patch generator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.presets import pick_preset, preset_label

_GB = 1024**3


@pytest.fixture
def _force_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _stub_props(monkeypatch: pytest.MonkeyPatch, total_bytes: int) -> None:
    props = MagicMock(total_memory=total_bytes, name="StubGPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)


def test_pick_preset_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        pick_preset()


@pytest.mark.parametrize(
    "total_bytes, expected_method, expected_r, expected_bs, expected_ga, expected_ckpt",
    [
        (int(11.9 * _GB), "qlora", 8, 1, 16, True),  # <12 tier (upper edge)
        (int(12.0 * _GB), "qlora", 16, 1, 8, True),  # 12-24 tier (lower edge, inclusive)
        (int(23.9 * _GB), "qlora", 16, 1, 8, True),  # 12-24 tier (upper edge)
        (int(24.0 * _GB), "lora", 16, 2, 4, False),  # 24-48 tier (lower edge, inclusive)
        (int(47.9 * _GB), "lora", 16, 2, 4, False),  # 24-48 tier (upper edge)
        (int(48.0 * _GB), "lora", 32, 4, 2, False),  # ≥48 tier
    ],
)
def test_pick_preset_tiers(
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    total_bytes: int,
    expected_method: str,
    expected_r: int,
    expected_bs: int,
    expected_ga: int,
    expected_ckpt: bool,
) -> None:
    _stub_props(monkeypatch, total_bytes)
    patch = pick_preset()
    assert patch["peft"]["method"] == expected_method
    assert patch["peft"]["r"] == expected_r
    assert patch["train"]["batch_size"] == expected_bs
    assert patch["train"]["grad_accum_steps"] == expected_ga
    assert patch["model"]["gradient_checkpointing"] is expected_ckpt
    assert patch["model"]["dtype"] == "bfloat16"


@pytest.mark.parametrize(
    "total_bytes, must_contain",
    [
        (int(11.0 * _GB), "<12GB"),
        (int(16.0 * _GB), "12-24GB"),
        (int(40.0 * _GB), "24-48GB"),
        (int(80.0 * _GB), "≥48GB"),
    ],
)
def test_preset_label_format(
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    total_bytes: int,
    must_contain: str,
) -> None:
    _stub_props(monkeypatch, total_bytes)
    label = preset_label()
    assert "auto:" in label
    assert must_contain in label


def test_preset_label_with_explicit_total_bytes() -> None:
    # Does not need CUDA when explicit bytes provided.
    assert "12-24GB" in preset_label(total_bytes=int(16 * _GB))
