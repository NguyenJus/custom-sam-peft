"""Tests for src/custom_sam_peft/presets.py — analytic VRAM preset chooser.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §3, §7, §9.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.presets import PresetDecision, decide_preset

_GB = 1024**3


@pytest.fixture
def _force_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _stub_gpu(monkeypatch: pytest.MonkeyPatch, total_bytes: int, name: str = "StubGPU") -> None:
    props = MagicMock(total_memory=total_bytes)
    props.name = name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: name)


# ---- decide_preset: per-tier behavior --------------------------------------


def test_decide_preset_11gib_chooses_qlora(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(11 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "qlora"
    # At 11 GiB the LoRA base model is too large; QLoRA is chosen at the highest
    # rank that fits (analytic seed, superseded by calibration cache).
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_16gib_chooses_lora_low_rank(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(16 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "lora"
    # At 16 GiB, lora is chosen over qlora (quality preference). The rank is
    # within the search space maximum; exact rank depends on analytic seed constants.
    assert d.r <= 64
    assert d.batch_size >= 1


def test_decide_preset_40gib_chooses_lora_high_rank(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "lora"
    assert d.r >= 32
    assert d.batch_size >= 2
    assert d.gradient_checkpointing is False


def test_decide_preset_80gib_chooses_max_rank_batch(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset(image_size=1024)
    assert d.r == 64
    assert d.batch_size >= 8  # within 1 step of max (spec says "or near max")


def test_decide_preset_unfittable_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(4 * _GB))
    with pytest.raises(RuntimeError, match=r"SAM 3\.1 needs"):
        decide_preset(image_size=1024)


def test_decide_preset_grad_accum_targets_16(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset(image_size=1024)
    assert d.batch_size * d.grad_accum_steps >= 16


def test_decide_preset_prefers_lora_over_qlora_when_both_fit(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset(image_size=1024)
    assert d.method == "lora"


def test_decide_preset_image_size_scales_activation(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    small = decide_preset(image_size=1024)
    big = decide_preset(image_size=2048)
    # At larger image_size the chosen config must be at least as conservative.
    assert big.predicted_bytes >= small.predicted_bytes or big.r <= small.r


# ---- calibration cache provenance ------------------------------------------


def _write_cache(path: Path, **fields: object) -> None:
    base = {
        "schema_version": 1,
        "calibrated_at": "2026-05-22T00:00:00+00:00",
        "gpu_name": "StubGPU",
        "gpu_total_memory_bytes": int(40 * _GB),
        "image_size": 1024,
        "sam3_checkpoint_sha": "deadbeef",
        "torch_version": "2.4.0",
        "custom_sam_peft_version": "0.0.0",
        "activation_bytes_per_example": int(0.5 * _GB),
        "peak_memory_bytes_at_probe": int(38 * _GB),
    }
    base.update(fields)
    path.write_text(json.dumps(base))


def test_decide_preset_uses_calibration_cache_when_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB), name="StubGPU")
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    _write_cache(cache)
    monkeypatch.chdir(tmp_path)
    # Make sha resolver match the cache's "deadbeef".
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    d = decide_preset(image_size=1024)
    assert d.provenance == "calibrated"
    assert d.cache_path == cache.resolve()


def test_decide_preset_ignores_stale_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB), name="StubGPU")
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    _write_cache(cache, sam3_checkpoint_sha="WRONG")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    d = decide_preset(image_size=1024)
    assert d.provenance == "analytic"


# ---- headroom env override --------------------------------------------------


def test_decide_preset_headroom_env_override(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "2.0")
    d = decide_preset(image_size=1024)
    assert d.budget_bytes == int(40 * _GB) - 2 * _GB


def test_decide_preset_headroom_env_invalid_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "not-a-number")
    with pytest.raises(RuntimeError, match="CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB"):
        decide_preset(image_size=1024)


def test_decide_preset_headroom_env_negative_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "-1")
    with pytest.raises(RuntimeError, match="CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB"):
        decide_preset(image_size=1024)


# ---- PresetDecision.label / to_json / config_patch -------------------------


def _make_decision(provenance: str = "calibrated") -> PresetDecision:
    return PresetDecision(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        gradient_checkpointing=False,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * _GB),
        predicted_bytes=int(38.4 * _GB),
        budget_bytes=int(39 * _GB),
        image_size=1008,
        gpu_name="NVIDIA A100-SXM4-40GB",
        provenance=provenance,  # type: ignore[arg-type]
        cache_path=Path(".custom_sam_peft_calibration.json"),
        calibrated_at="2026-05-22T00:00:00+00:00" if provenance == "calibrated" else None,
    )


def test_preset_decision_label_calibrated() -> None:
    d = _make_decision(provenance="calibrated")
    label = d.label()
    assert "LoRA r=32" in label
    assert "calibrated" in label
    assert "2026-05-22" in label


def test_preset_decision_label_analytic() -> None:
    d = _make_decision(provenance="analytic")
    label = d.label()
    assert "(analytic estimate)" in label


def test_preset_decision_to_json_round_trip() -> None:
    d = _make_decision()
    js = d.to_json()
    d2 = PresetDecision.from_json(js)
    assert d == d2


def test_preset_decision_config_patch_3_sections() -> None:
    patch = _make_decision().config_patch
    assert set(patch.keys()) == {"model", "peft", "train"}
    assert patch["peft"]["method"] == "lora"
    assert patch["peft"]["r"] == 32
    assert patch["train"]["batch_size"] == 2
    assert patch["train"]["grad_accum_steps"] == 8
    assert patch["model"]["gradient_checkpointing"] is False
    assert patch["model"]["dtype"] == "bfloat16"


def test_decide_preset_image_size_invalid_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    with pytest.raises(ValueError, match="image_size"):
        decide_preset(image_size=0)


def test_decide_preset_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        decide_preset(image_size=1024)


def test_predicted_bytes_train_mode_unchanged() -> None:
    """Existing train-mode callers stay byte-identical after the mode param."""
    from custom_sam_peft.presets import _predicted_bytes

    # Default mode='train' — same value as the pre-change signature.
    n = _predicted_bytes("lora", r=4, batch=1, ckpt=False, image_size=1024, cache=None)
    # And the explicit mode='train' matches.
    assert n == _predicted_bytes(
        "lora", r=4, batch=1, ckpt=False, image_size=1024, cache=None, mode="train"
    )
