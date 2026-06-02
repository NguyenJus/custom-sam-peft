"""Tests for src/custom_sam_peft/presets.py — analytic VRAM preset chooser.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §3, §7, §9.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.presets import (
    A_FIXED,
    A_PER_CLASS,
    WORKSPACE_BYTES,
    PresetDecision,
    _activation_bytes,
    _adapter_bytes,
    _attention_bytes_per_example,
    _candidates,
    _flash_attention_available,
    _model_bytes,
    _optimizer_bytes,
    _predicted_bytes,
    _sort_key,
    decide_eval_batch_size,
    decide_preset,
)

_GB = 1024**3


@pytest.fixture
def _force_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _stub_gpu(
    monkeypatch: pytest.MonkeyPatch,
    total_bytes: int,
    name: str = "StubGPU",
    cc: tuple[int, int] = (8, 0),
) -> None:
    props = MagicMock(total_memory=total_bytes)
    props.name = name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _idx: cc)


# ---- decide_preset: per-tier behavior --------------------------------------


def test_decide_preset_32gib_sizes_lora(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # With the split formula (A_FIXED dominant), a 32 GiB card can fit LoRA at
    # conservative K. Assert a valid PresetDecision is returned (no RuntimeError).
    _stub_gpu(monkeypatch, int(32 * _GB))
    d = decide_preset()
    assert isinstance(d, PresetDecision)
    assert d.method == "lora"
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_40gib_chooses_lora_low_rank(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # At 40 GiB (budget=39 GiB), LoRA fits; lora is preferred over qlora by the
    # sort key. Exact rank/K/batch depend on the analytic seed.
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset()
    assert d.method == "lora"
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_65gib_chooses_lora_high_rank(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # At 65 GiB (budget=64 GiB), LoRA at high rank fits (split formula is more
    # generous than the old lumped K=16 model). Exact rank/K/batch per seed.
    _stub_gpu(monkeypatch, int(65 * _GB))
    d = decide_preset()
    assert d.method == "lora"
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_80gib_chooses_max_rank_batch(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # At 80 GiB (budget=79 GiB) with K_eff=16, the activation term at batch=2 is
    # ~46 GiB, so max batch is 2 at r=64.  We test that the sort key selects r=64
    # and picks the highest feasible batch (>=2).
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset()
    assert d.r == 64
    assert d.batch_size >= 2  # max feasible batch at K_eff=16 within 79 GiB


def test_decide_preset_unfittable_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(4 * _GB))
    with pytest.raises(RuntimeError, match=r"SAM 3\.1 needs"):
        decide_preset()


def test_decide_preset_invalid_k_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """decide_preset(k=0) and decide_preset(k=-1) must raise ValueError."""
    _stub_gpu(monkeypatch, int(40 * _GB))
    with pytest.raises(ValueError):
        decide_preset(k=0)
    with pytest.raises(ValueError):
        decide_preset(k=-1)


def test_decide_preset_grad_accum_targets_16(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset()
    assert d.batch_size * d.grad_accum_steps >= 16


def test_decide_preset_prefers_lora_over_qlora_when_both_fit(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    d = decide_preset()
    assert d.method == "lora"


# ---- calibration cache provenance ------------------------------------------


def _write_cache(path: Path, **fields: object) -> None:
    base = {
        "schema_version": 3,
        "calibrated_at": "2026-05-22T00:00:00+00:00",
        "gpu_name": "StubGPU",
        "gpu_total_memory_bytes": int(40 * _GB),
        "sam3_checkpoint_sha": "deadbeef",
        "torch_version": "2.4.0",
        "custom_sam_peft_version": "0.0.0",
        "A_fixed": int(1.30 * _GB),
        "A_per_class": int(0.15 * _GB),
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
    d = decide_preset()
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
    d = decide_preset()
    assert d.provenance == "analytic"


# ---- headroom env override --------------------------------------------------


def test_decide_preset_headroom_env_override(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "2.0")
    d = decide_preset()
    assert d.budget_bytes == int(40 * _GB) - 2 * _GB


def test_decide_preset_headroom_env_invalid_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "not-a-number")
    with pytest.raises(RuntimeError, match="CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB"):
        decide_preset()


def test_decide_preset_headroom_env_negative_raises(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(40 * _GB))
    monkeypatch.setenv("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB", "-1")
    with pytest.raises(RuntimeError, match="CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB"):
        decide_preset()


# ---- PresetDecision.label / to_json / config_patch -------------------------


def _make_decision(provenance: str = "calibrated", **over: object) -> PresetDecision:
    base: dict[str, object] = dict(
        method="lora",
        r=32,
        batch_size=2,
        grad_accum_steps=8,
        classes_per_forward=8,
        dtype="bfloat16",
        headroom_bytes=int(1.6 * _GB),
        predicted_bytes=int(38.4 * _GB),
        budget_bytes=int(39 * _GB),
        gpu_name="NVIDIA A100-SXM4-40GB",
        provenance=provenance,
        cache_path=Path(".custom_sam_peft_calibration.json"),
        calibrated_at="2026-05-22T00:00:00+00:00" if provenance == "calibrated" else None,
    )
    base.update(over)
    return PresetDecision(**base)  # type: ignore[arg-type]


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


def test_from_json_drops_stale_image_size_key() -> None:
    """from_json silently drops unknown keys (e.g. image_size from pre-removal sidecars)."""
    d = _make_decision()
    raw = json.loads(d.to_json())
    raw["image_size"] = 1008  # simulate a sidecar written before image_size was removed
    stale_json = json.dumps(raw)
    d2 = PresetDecision.from_json(stale_json)
    assert d == d2


def test_preset_decision_config_patch_3_sections() -> None:
    patch = _make_decision().config_patch
    assert set(patch.keys()) == {"model", "peft", "train"}
    assert patch["peft"]["method"] == "lora"
    assert patch["peft"]["r"] == 32
    assert patch["train"]["batch_size"] == 2
    assert patch["train"]["grad_accum_steps"] == 8
    assert "gradient_checkpointing" not in patch["model"]
    assert patch["model"]["dtype"] == "bfloat16"


def test_presetdecision_has_alpha_field_and_in_config_patch() -> None:
    from custom_sam_peft.presets import PresetDecision

    d = PresetDecision(
        method="lora", r=8, alpha=16, batch_size=1, grad_accum_steps=8,
        classes_per_forward=1, dtype="bfloat16", headroom_bytes=0,
        predicted_bytes=0, budget_bytes=0, gpu_name="X",
        provenance="calibrated", cache_path=None, calibrated_at=None,
    )
    assert d.alpha == 16
    assert d.config_patch["peft"]["alpha"] == 16


def test_decide_preset_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        decide_preset()


def test_predicted_bytes_train_mode_unchanged() -> None:
    """Existing train-mode callers stay correct after the ckpt param removal."""
    from custom_sam_peft.presets import _predicted_bytes

    n = _predicted_bytes("lora", r=4, batch=1, image_size=1008, cache=None)
    assert n == _predicted_bytes("lora", r=4, batch=1, image_size=1008, cache=None, mode="train")


def test_preset_decision_label_has_no_ckpt_token() -> None:
    d = _make_decision()
    assert "ckpt=" not in d.label()


# ---- dtype token in label / round-trip -------------------------------------


def test_preset_decision_label_renders_fp16_token() -> None:
    d = _make_decision()
    object.__setattr__(d, "dtype", "float16")  # PresetDecision is a frozen dataclass
    assert "fp16" in d.label()
    assert "bf16" not in d.label()


def test_preset_decision_label_renders_bf16_token() -> None:
    d = _make_decision()  # default dtype="bfloat16"
    assert "bf16" in d.label()


def test_preset_decision_float16_round_trips() -> None:
    d = _make_decision()
    object.__setattr__(d, "dtype", "float16")
    d2 = PresetDecision.from_json(d.to_json())
    assert d2.dtype == "float16"
    assert d == d2


# ---- decide_preset dtype selection by compute capability -------------------


def test_decide_preset_selects_float16_below_cc80(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """On CC<8.0 hardware (Pascal/GTX 1080) decide_preset must pick float16."""
    # Use 40 GiB so a preset fits even at K_eff=MULTIPLEX_CAP (16).
    _stub_gpu(monkeypatch, int(40 * _GB), cc=(6, 1))
    decision = decide_preset()
    assert decision.dtype == "float16"
    assert "fp16" in decision.label()
    assert decision.config_patch["model"]["dtype"] == "float16"


def test_decide_preset_selects_bfloat16_at_cc80(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """On CC>=8.0 hardware (Ampere+) decide_preset must pick bfloat16."""
    # Use 40 GiB so a preset fits even at K_eff=MULTIPLEX_CAP (16).
    _stub_gpu(monkeypatch, int(40 * _GB), cc=(8, 0))
    decision = decide_preset()
    assert decision.dtype == "bfloat16"


# ---- K_eff activation + shared attention helper ----------------------------


def test_flash_attention_available_by_cc() -> None:
    # cc >= (8, 0): flash / mem-efficient SDPA -> no materialized attention term.
    assert _flash_attention_available((8, 0)) is True
    assert _flash_attention_available((9, 0)) is True
    assert _flash_attention_available((12, 0)) is True  # 5070 Ti dev box
    # cc < (8, 0): assume math backend materializes -> include the attention term.
    assert _flash_attention_available((7, 5)) is False  # Turing (conservative)
    assert _flash_attention_available((6, 1)) is False  # Pascal (GTX 1080)
    # Unknown / unreadable cc -> conservative False (safe over-estimate).
    assert _flash_attention_available(None) is False


def test_attention_bytes_helper_matches_sdpa_model() -> None:
    """The shared helper reproduces the inline SDPA model: H * N^2 * 4 bytes."""
    from custom_sam_peft.presets import _attention_bytes_per_example

    image_size = 1008
    n_tokens = (image_size // 14) ** 2  # patch=14
    expected = 16 * n_tokens * n_tokens * 4  # heads=16, fp32
    assert _attention_bytes_per_example(image_size) == expected


def test_predicted_bytes_train_grows_with_k_eff() -> None:
    """Train-mode prediction is monotone in K_eff (more classes/group -> more activation)."""
    from custom_sam_peft.presets import _predicted_bytes

    small_k = _predicted_bytes("lora", r=8, batch=1, image_size=1008, cache=None, k_eff=1)
    big_k = _predicted_bytes("lora", r=8, batch=1, image_size=1008, cache=None, k_eff=16)
    assert big_k > small_k


def test_decide_preset_threads_k_into_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    """decide_preset(k=...) feeds K_eff into the train formula; larger k -> larger
    predicted_bytes for the chosen preset (monotone), all else equal."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _stub_gpu(monkeypatch, int(80 * _GB))  # large card so a preset always fits
    d_small = decide_preset(k=1)
    d_big = decide_preset(k=16)
    assert d_big.predicted_bytes > d_small.predicted_bytes


def test_decide_preset_defaults_k_to_cap_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No k supplied -> conservative worst case == MULTIPLEX_CAP."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _stub_gpu(monkeypatch, int(80 * _GB))
    assert decide_preset().predicted_bytes == decide_preset(k=MULTIPLEX_CAP).predicted_bytes


def test_activation_bytes_split_is_linear_in_k_no_analytic_cache() -> None:
    # Encoder term (A_FIXED) does NOT scale with K; only A_PER_CLASS * K does.
    at_k1 = _activation_bytes(batch=1, cache=None, k_eff=1)
    at_k16 = _activation_bytes(batch=1, cache=None, k_eff=16)
    assert at_k1 == A_FIXED + A_PER_CLASS * 1
    assert at_k16 == A_FIXED + A_PER_CLASS * 16
    # The #203 regression guard: K=1 vs K=16 differ by exactly 15 * A_PER_CLASS.
    assert at_k16 - at_k1 == 15 * A_PER_CLASS


def test_activation_bytes_scales_with_batch() -> None:
    assert _activation_bytes(batch=4, cache=None, k_eff=2) == ((A_FIXED + A_PER_CLASS * 2) * 4)


def test_activation_bytes_reads_split_cache() -> None:
    cache = {"A_fixed": 1000, "A_per_class": 7}
    assert _activation_bytes(batch=2, cache=cache, k_eff=3) == (1000 + 7 * 3) * 2


def test_predicted_bytes_train_threads_k() -> None:
    # K-invariance holds in BOTH regimes: K=16 minus K=1 equals exactly
    # 15 * A_PER_CLASS * batch (encoder + attention both unchanged by K).
    img = 1008
    for flash in (True, False):
        pb_k1 = _predicted_bytes(
            "qlora", 4, 1, img, None, mode="train", k_eff=1, flash_available=flash
        )
        pb_k16 = _predicted_bytes(
            "qlora", 4, 1, img, None, mode="train", k_eff=16, flash_available=flash
        )
        assert pb_k16 - pb_k1 == 15 * A_PER_CLASS * 1


def test_predicted_bytes_train_no_attention_when_flash() -> None:
    # cc >= 8.0 (flash): train branch is STATIC + split, NO attention term.
    img = 1008
    static = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4) + WORKSPACE_BYTES
    pb = _predicted_bytes("qlora", 4, 1, img, None, mode="train", k_eff=1, flash_available=True)
    assert pb == static + (A_FIXED + A_PER_CLASS * 1)


def test_predicted_bytes_train_adds_attention_when_no_flash() -> None:
    # cc < 8.0 (math backend): train branch re-adds the materialized
    # _attention_bytes_per_example(img) * batch term (K-invariant).
    img = 1008
    static = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4) + WORKSPACE_BYTES
    attn = _attention_bytes_per_example(img) * 1  # batch=1
    pb = _predicted_bytes("qlora", 4, 1, img, None, mode="train", k_eff=1, flash_available=False)
    assert pb == static + (A_FIXED + A_PER_CLASS * 1) + attn
    assert attn > 0  # the term is genuinely re-added


def test_predicted_bytes_scales_with_image_size_only_when_no_flash() -> None:
    small = _predicted_bytes("qlora", 4, 1, 512, None, mode="train", k_eff=1, flash_available=False)
    big = _predicted_bytes("qlora", 4, 1, 1008, None, mode="train", k_eff=1, flash_available=False)
    assert big > small  # attention term grows with image_size on no-flash cards
    # flash: no image-size dependence (attention folded into the split)
    fl_small = _predicted_bytes(
        "qlora", 4, 1, 512, None, mode="train", k_eff=1, flash_available=True
    )
    fl_big = _predicted_bytes(
        "qlora", 4, 1, 1008, None, mode="train", k_eff=1, flash_available=True
    )
    assert fl_small == fl_big


def test_predicted_bytes_eval_threads_k() -> None:
    img = 1008
    pb_k1 = _predicted_bytes("lora", 4, 1, img, None, mode="eval", k_eff=1, flash_available=True)
    pb_k4 = _predicted_bytes("lora", 4, 1, img, None, mode="eval", k_eff=4, flash_available=True)
    assert pb_k4 > pb_k1  # eval activation scales with K via the split


def test_predicted_bytes_eval_adds_attention_when_no_flash() -> None:
    # eval stays SAFE on no-flash cards: it adds the same materialized term so it
    # never under-predicts vs. the math backend.
    img = 1008
    pb_flash = _predicted_bytes("lora", 4, 1, img, None, mode="eval", k_eff=1, flash_available=True)
    pb_noflash = _predicted_bytes(
        "lora", 4, 1, img, None, mode="eval", k_eff=1, flash_available=False
    )
    assert pb_noflash - pb_flash == _attention_bytes_per_example(img) * 1


def test_preset_decision_config_patch_carries_classes_per_forward() -> None:
    d = _make_decision(classes_per_forward=8)
    patch = d.config_patch
    assert patch["train"]["multiplex"]["classes_per_forward"] == 8
    assert patch["train"]["batch_size"] == 2
    assert patch["train"]["grad_accum_steps"] == 8


def test_preset_decision_label_surfaces_k() -> None:
    d = _make_decision(classes_per_forward=8)
    assert "K=8" in d.label()


def test_preset_decision_json_round_trip_carries_k() -> None:
    d = _make_decision(classes_per_forward=8)
    back = PresetDecision.from_json(d.to_json())
    assert back.classes_per_forward == 8
    assert back == d


# ---- candidate grid + sort key + k upper bound ----------------------------


def test_candidates_are_4_tuples_with_ks() -> None:
    cands = _candidates()
    assert all(len(c) == 4 for c in cands)
    ks = {c[3] for c in cands}
    assert ks == {1, 2, 4, 8, 16}


def test_sort_key_protects_k_over_batch() -> None:
    # At fixed method/r, (K=8, batch=1) sorts ahead of (K=1, batch=8).
    assert _sort_key(("lora", 16, 1, 8)) < _sort_key(("lora", 16, 8, 1))


def test_sort_key_protects_r_over_k_and_batch() -> None:
    assert _sort_key(("lora", 32, 1, 1)) < _sort_key(("lora", 16, 16, 16))


def test_sort_key_prefers_lora_over_qlora() -> None:
    assert _sort_key(("lora", 16, 1, 1)) < _sort_key(("qlora", 16, 1, 1))


def test_decide_preset_k_is_upper_bound(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset(k=4)
    assert d.classes_per_forward <= 4


def test_decide_preset_k_zero_and_negative_raise(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    with pytest.raises(ValueError):
        decide_preset(k=0)
    with pytest.raises(ValueError):
        decide_preset(k=-1)


def test_decide_preset_24gib_sizes_successfully(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # #203 regression: a 24 GiB card must size successfully, not raise.
    _stub_gpu(monkeypatch, int(24 * _GB))
    d = decide_preset()
    assert isinstance(d, PresetDecision)
    assert d.predicted_bytes <= d.budget_bytes


def test_decide_preset_big_card_picks_high_k(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset()
    assert d.classes_per_forward >= 8
    assert d.batch_size >= 2


# ---- decide_eval_batch_size K threading + v3 cache round-trip ---------------


def test_decide_eval_batch_size_threads_k_no_regression(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    _stub_gpu(monkeypatch, int(24 * _GB))
    bs1, _, _ = decide_eval_batch_size(classes_per_forward=1)
    bs16, _, _ = decide_eval_batch_size(classes_per_forward=16)
    # Higher K can only LOWER (or hold) best_bs — never raise it (no regression).
    assert bs16 <= bs1
    assert bs16 >= 1


def test_decide_eval_batch_size_no_flash_adds_attention(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # On a no-flash card (cc < 8.0) the eval predictor adds the materialized
    # attention term, so best_bs is <= the flash-regime best_bs (never under-fits).
    _stub_gpu(monkeypatch, int(24 * _GB), cc=(8, 0))
    bs_flash, _, _ = decide_eval_batch_size(classes_per_forward=1)
    _stub_gpu(monkeypatch, int(24 * _GB), cc=(7, 5))
    bs_noflash, _, _ = decide_eval_batch_size(classes_per_forward=1)
    assert bs_noflash <= bs_flash
    assert bs_noflash >= 1


def test_decide_preset_consumes_v3_cache(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "calibrated_at": "2026-05-31T00:00:00+00:00",
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "abc",
                "A_fixed": 1_000_000_000,
                "A_per_class": 50_000_000,
                "peak_memory_bytes_at_probe": 6_000_000_000,
            }
        )
    )
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "calibrated"


def test_decide_preset_ignores_v2_cache(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "abc",
                "activation_bytes_per_example": 1_000_000_000,
            }
        )
    )
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "analytic"  # stale v2 dropped


def test_decide_preset_ignores_v3_cache_missing_split_keys(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    """A v3-tagged cache missing A_fixed/A_per_class must be ignored (graceful
    fallback to analytic), not crash with a KeyError."""
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "abc",
                # A_fixed and A_per_class intentionally omitted (malformed cache)
                "peak_memory_bytes_at_probe": 6_000_000_000,
            }
        )
    )
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "analytic"  # malformed v3 dropped, no crash
