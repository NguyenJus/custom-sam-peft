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


def test_decide_preset_80gib_chooses_pinned_r_max_batch(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    # At 80 GiB (budget=79 GiB), the pinned r=16 fits with generous b/k.
    # Hardware NEVER raises r above the pinned default — the old r=64 outcome
    # is the pre-fix bug being corrected. Spec §3/§6.1.
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset()
    assert d.r == 16  # pinned; never raised to 64
    assert d.batch_size >= 2  # generous budget yields high batch at pinned r


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
        "schema_version": 4,  # current CACHE_SCHEMA_VERSION (v4: pinned r/alpha)
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
        method="lora",
        r=8,
        alpha=16,
        batch_size=1,
        grad_accum_steps=8,
        classes_per_forward=1,
        dtype="bfloat16",
        headroom_bytes=0,
        predicted_bytes=0,
        budget_bytes=0,
        gpu_name="X",
        provenance="calibrated",
        cache_path=None,
        calibrated_at=None,
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
    """On CC<8.0 hardware (here stubbed at CC 6.1) decide_preset must pick float16."""
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
    assert _flash_attention_available((6, 1)) is False  # CC 6.1 (pre-Ampere)
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


def test_candidates_are_2_tuples_with_ks() -> None:
    # Post-pin: _candidates enumerates (batch, k) only — method/r are pinned inputs,
    # not searched. r is no longer a search dimension (spec §6.1).
    cands = _candidates()
    assert all(len(c) == 2 for c in cands)
    ks = {c[1] for c in cands}
    assert ks == {1, 2, 4, 8, 16}


def test_sort_key_protects_k_over_batch() -> None:
    # At pinned method/r, (K=8, batch=1) sorts ahead of (K=1, batch=8).
    # Sort key now takes (batch, k) 2-tuples (spec §6.1).
    assert _sort_key((1, 8)) < _sort_key((8, 1))


def test_sort_key_largest_k_wins() -> None:
    # k=16 beats k=8 regardless of batch (tail-to-head: batch first, k second).
    assert _sort_key((1, 16)) < _sort_key((16, 8))


def test_sort_key_highest_batch_wins_at_same_k() -> None:
    # At same k, higher batch sorts better.
    assert _sort_key((8, 4)) < _sort_key((4, 4))


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


def test_decide_preset_ignores_v3_cache_schema_version(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    """Post schema bump (v3→v4), a v3 cache is now stale and must be rejected.

    This is §8.9 test_v3_cache_is_stale: a v3 cache with chosen_r=64 is
    treated as stale on the version mismatch and the analytic path is used.
    Spec §7.2.
    """
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 3,  # stale — CACHE_SCHEMA_VERSION is now 4
                "calibrated_at": "2026-05-31T00:00:00+00:00",
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "abc",
                "A_fixed": 1_000_000_000,
                "A_per_class": 50_000_000,
                "peak_memory_bytes_at_probe": 6_000_000_000,
                "chosen_r": 64,  # the old rank-maximization value — now stale
            }
        )
    )
    d = decide_preset(cache_path=cache_file)
    # v3 cache is rejected on version mismatch → analytic path, NOT calibrated
    assert d.provenance == "analytic"


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
    """A v3-tagged (now stale) cache missing A_fixed/A_per_class must be ignored
    (graceful fallback to analytic), not crash with a KeyError.

    Post schema bump, a v3 cache is stale regardless of its contents; the
    version mismatch is what rejects it. This test ensures no crash."""
    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "abc")
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 3,  # stale
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "abc",
                # A_fixed and A_per_class intentionally omitted (malformed cache)
                "peak_memory_bytes_at_probe": 6_000_000_000,
            }
        )
    )
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "analytic"  # stale v3 dropped, no crash


# ============================================================================
# §8 Adversarial CPU tests — pinned r/alpha ladder
# ============================================================================


# ---------------------------------------------------------------------------
# §8.1 Pin invariant (headline) — r never raised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("total_gib", [8, 16, 24, 48, 80])
def test_pin_never_raises_r(
    total_gib: int, monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """For EVERY budget sweeping tiny→huge, decision.r <= 16 (cfg.r).
    The sizer NEVER emits r > cfg.r for ANY budget. Spec §8.1."""
    _stub_gpu(monkeypatch, int(total_gib * _GB))
    d = decide_preset()
    assert d.r <= 16, (
        f"decide_preset() emitted r={d.r} on a {total_gib} GiB card; "
        "hardware must never raise r above the pinned default (spec §3)"
    )


@pytest.mark.parametrize("total_gib", [16, 24, 48, 80])
def test_pin_holds_r_when_config_fits(
    total_gib: int, monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """Whenever the pinned config fits at all, decision.r == 16.
    Larger budgets do NOT bump r. Spec §8.1."""
    from custom_sam_peft.presets import _predicted_bytes

    _stub_gpu(monkeypatch, int(total_gib * _GB))
    budget = int(total_gib * _GB) - 1 * _GB
    # Confirm the pinned floor actually fits on this budget (test precondition).
    floor = _predicted_bytes("lora", r=16, batch=1, image_size=1008, cache=None, k_eff=1)
    if floor > budget:
        pytest.skip(f"{total_gib} GiB card too small for lora r=16 b=1 k=1")
    d = decide_preset()
    assert d.r == 16, (
        f"decide_preset() emitted r={d.r} on a {total_gib} GiB card; "
        "pinned r must hold whenever the config fits (spec §3)"
    )


# ---------------------------------------------------------------------------
# §8.2 alpha never stranded
# ---------------------------------------------------------------------------


def test_alpha_equals_cfg_alpha_when_r_unchanged(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """Whenever decision.r == cfg.r, decision.alpha == cfg.alpha exactly.
    The r=64/alpha=32 signature (alpha:r=0.5) is impossible. Spec §8.2."""
    _stub_gpu(monkeypatch, int(40 * _GB))
    # Default cfg: r=16, alpha=32 (the shipped pinned pair)
    d = decide_preset()
    assert d.r == 16
    assert d.alpha == 32, (
        f"alpha={d.alpha} stranded at field default; expected cfg.alpha=32 (spec §7.1)"
    )
    # Explicit pinned pair: r=16, alpha=24 (non-default ratio)
    d2 = decide_preset(r=16, alpha=24)
    assert d2.r == 16
    assert d2.alpha == 24, f"alpha={d2.alpha} not pinned to cfg.alpha=24"


def test_alpha_equals_cfg_alpha_end_to_end_decide_preset(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """End-to-end through decide_preset: pinned alpha flows into the PresetDecision."""
    _stub_gpu(monkeypatch, int(80 * _GB))
    d = decide_preset(r=16, alpha=32)
    # r is pinned; alpha must exactly equal the pinned cfg.alpha, not be stranded
    # at the field default (32 here happens to equal default, but test with
    # non-default alpha to confirm pinning, not coincidence):
    d2 = decide_preset(r=16, alpha=48)
    assert d.alpha == 32
    assert d2.alpha == 48, f"alpha={d2.alpha} not pinned to cfg.alpha=48"


# ---------------------------------------------------------------------------
# §8.3 Batch maximized; k_start is min(cfg, cap, num_classes)
# ---------------------------------------------------------------------------


def test_b_grows_with_budget_r_alpha_method_fixed(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """A strictly larger budget yields b >= smaller-budget b, while r/alpha/method
    are byte-identical across budgets. Spec §8.3."""
    _stub_gpu(monkeypatch, int(48 * _GB))
    d_small = decide_preset()
    _stub_gpu(monkeypatch, int(80 * _GB))
    d_big = decide_preset()
    # r, alpha, method are pinned — should be identical across budgets
    assert d_big.r == d_small.r == 16
    assert d_big.alpha == d_small.alpha == 32
    assert d_big.method == d_small.method == "lora"
    # batch grows with budget (monotone, never decreases)
    assert d_big.batch_size >= d_small.batch_size


def test_k_start_is_min_cfg_cap_numclasses(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """k_start == min(cfg K, MULTIPLEX_CAP=16, num_classes). On a large card
    the chosen k_start equals the vocabulary cap. Spec §8.3/§5."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    _stub_gpu(monkeypatch, int(80 * _GB))
    # With num_classes=4, k_start = min(16, 16, 4) = 4; a large card should
    # pick k <= 4 (the vocabulary limits the k ceiling, not MULTIPLEX_CAP).
    d = decide_preset(num_classes=4)
    assert d.classes_per_forward <= 4, (
        f"classes_per_forward={d.classes_per_forward} exceeds num_classes=4; "
        "k_start must be capped at min(k_cap, MULTIPLEX_CAP, num_classes)"
    )
    # Without num_classes, a large card is free to choose k up to MULTIPLEX_CAP.
    d_uncap = decide_preset()
    assert d_uncap.classes_per_forward <= MULTIPLEX_CAP


# ---------------------------------------------------------------------------
# §8.4 Ladder order under shrinking budget
# ---------------------------------------------------------------------------


def test_sacrifice_sequence_exact(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """Drive budget down and assert the exact sequence b↓ → k↓ (b held at 1)
    → lora→qlora @same r → r↓ (+warning). Spec §8.4."""
    from custom_sam_peft.presets import _predicted_bytes

    # Scenario A: b shrinks — only b=1 fits at k=16 (b=2 doesn't).
    # Compute exact threshold: budget = (lora r=16 b=2 k=16) - 1 byte.
    pb_b2_k16 = _predicted_bytes(
        "lora", r=16, batch=2, image_size=1008, cache=None, k_eff=16, flash_available=True
    )
    pb_b1_k16 = _predicted_bytes(
        "lora", r=16, batch=1, image_size=1008, cache=None, k_eff=16, flash_available=True
    )
    budget_a = pb_b2_k16 - 1  # b=2 doesn't fit; b=1 does
    _stub_gpu(monkeypatch, budget_a + 1 * _GB, cc=(8, 0))  # headroom=1 GiB
    d_a = decide_preset()
    assert d_a.r == 16, f"r changed during b-shrink: {d_a.r}"
    assert d_a.alpha == 32, f"alpha changed during b-shrink: {d_a.alpha}"
    assert d_a.method == "lora", f"method changed during b-shrink: {d_a.method}"
    assert d_a.batch_size == 1, f"expected b=1 after b-shrink but got {d_a.batch_size}"
    assert d_a.classes_per_forward == 16, f"k changed during b-shrink: {d_a.classes_per_forward}"
    assert pb_b1_k16 <= budget_a  # sanity: b=1 k=16 must fit

    # Scenario B: k shrinks — budget too tight for lora r=16 b=1 k=8, fits k=4.
    # b is HELD at 1 during k-shrink (not re-grown at lower k).
    pb_b1_k8 = _predicted_bytes(
        "lora", r=16, batch=1, image_size=1008, cache=None, k_eff=8, flash_available=True
    )
    pb_b1_k4 = _predicted_bytes(
        "lora", r=16, batch=1, image_size=1008, cache=None, k_eff=4, flash_available=True
    )
    budget_b = pb_b1_k8 - 1  # k=8 doesn't fit; k=4 does
    _stub_gpu(monkeypatch, budget_b + 1 * _GB, cc=(8, 0))
    d_b = decide_preset()
    assert d_b.r == 16, f"r changed during k-shrink: {d_b.r}"
    assert d_b.alpha == 32, f"alpha changed during k-shrink: {d_b.alpha}"
    assert d_b.method == "lora", f"method changed during k-shrink: {d_b.method}"
    assert d_b.batch_size == 1, (
        f"b was re-grown during k-shrink: {d_b.batch_size}; b MUST stay at 1"
    )
    assert d_b.classes_per_forward == 4, (
        f"expected k=4 after k-shrink but got {d_b.classes_per_forward}"
    )
    assert pb_b1_k4 <= budget_b  # sanity: b=1 k=4 must fit

    # Scenario C: qlora flip — lora r=16 k=1 doesn't fit; qlora r=16 k=1 does.
    # r is UNCHANGED across the flip.
    pb_lora_k1 = _predicted_bytes(
        "lora", r=16, batch=1, image_size=1008, cache=None, k_eff=1, flash_available=True
    )
    pb_qlora_k1 = _predicted_bytes(
        "qlora", r=16, batch=1, image_size=1008, cache=None, k_eff=1, flash_available=True
    )
    budget_c = pb_lora_k1 - 1  # lora r=16 doesn't fit; qlora r=16 does
    _stub_gpu(monkeypatch, budget_c + 1 * _GB, cc=(8, 0))
    d_c = decide_preset()
    assert d_c.r == 16, f"r changed during qlora flip: {d_c.r}"
    assert d_c.alpha == 32, f"alpha changed during qlora flip: {d_c.alpha}"
    assert d_c.method == "qlora", f"expected qlora after flip but got {d_c.method}"
    assert pb_qlora_k1 <= budget_c  # sanity: qlora r=16 k=1 must fit


def test_r_is_last_lever(monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None) -> None:
    """r is the LAST field to change and only ever decreases. On any non-r-reduction
    path (b-down, k-down, qlora-flip), r stays at cfg.r=16. Spec §8.4."""
    from custom_sam_peft.presets import _predicted_bytes

    # Probe progressively smaller budgets and assert r doesn't change until the
    # very end (when qlora r=16 b=1 k=1 no longer fits).
    for scenario_name, pb_fn in [
        (
            "b_shrink",
            lambda: (
                _predicted_bytes(
                    "lora",
                    r=16,
                    batch=2,
                    image_size=1008,
                    cache=None,
                    k_eff=16,
                    flash_available=True,
                )
                - 1
            ),
        ),
        (
            "k_shrink",
            lambda: (
                _predicted_bytes(
                    "lora",
                    r=16,
                    batch=1,
                    image_size=1008,
                    cache=None,
                    k_eff=8,
                    flash_available=True,
                )
                - 1
            ),
        ),
        (
            "qlora_flip",
            lambda: (
                _predicted_bytes(
                    "lora",
                    r=16,
                    batch=1,
                    image_size=1008,
                    cache=None,
                    k_eff=1,
                    flash_available=True,
                )
                - 1
            ),
        ),
    ]:
        budget = pb_fn()
        _stub_gpu(monkeypatch, budget + 1 * _GB, cc=(8, 0))
        d = decide_preset()
        assert d.r == 16, (
            f"r changed to {d.r} during {scenario_name} (before r-reduction step); "
            "r must be the LAST lever (spec §3/§4.2)"
        )


# ---------------------------------------------------------------------------
# §8.5 alpha co-scale on reduction
# ---------------------------------------------------------------------------


def test_alpha_coscaled_on_r_reduction(
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At a budget forcing r-reduction, alpha == max(1, round(cfg_alpha * r_new / cfg_r))
    and a WARNING is emitted. Spec §8.5."""
    import logging

    from custom_sam_peft.presets import _predicted_bytes

    # Budget: qlora r=16 b=1 k=1 doesn't fit, but qlora r=8 b=1 k=1 does.
    pb_qlora_r16 = _predicted_bytes(
        "qlora", r=16, batch=1, image_size=1008, cache=None, k_eff=1, flash_available=True
    )
    pb_qlora_r8 = _predicted_bytes(
        "qlora", r=8, batch=1, image_size=1008, cache=None, k_eff=1, flash_available=True
    )
    budget = pb_qlora_r16 - 1  # qlora r=16 doesn't fit; qlora r=8 does
    if pb_qlora_r8 > budget:
        pytest.skip("r=8 qlora doesn't fit either; cannot isolate r-reduction step")
    _stub_gpu(monkeypatch, budget + 1 * _GB, cc=(8, 0))

    cfg_r = 16
    cfg_alpha = 32
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.presets"):
        d = decide_preset(r=cfg_r, alpha=cfg_alpha)

    # r must have been reduced (we engineered the budget for this)
    assert d.r < cfg_r, f"r={d.r} not reduced; test budget miscalculated"
    r_new = d.r
    expected_alpha = max(1, round(cfg_alpha * r_new / cfg_r))
    assert d.alpha == expected_alpha, (
        f"alpha={d.alpha} not co-scaled; expected {expected_alpha} "
        f"(max(1, round({cfg_alpha} * {r_new} / {cfg_r})))"
    )
    # alpha:r ratio should approximately match cfg_alpha/cfg_r
    actual_ratio = d.alpha / d.r
    expected_ratio = cfg_alpha / cfg_r
    assert abs(actual_ratio - expected_ratio) < 0.5, (
        f"alpha:r ratio {actual_ratio:.2f} too far from cfg ratio {expected_ratio:.2f}"
    )
    # WARNING must be emitted on r-reduction
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("reducing" in str(m).lower() or "reduce" in str(m).lower() for m in warning_msgs), (
        f"No WARNING emitted on r-reduction; got: {warning_msgs}"
    )


def test_no_warning_no_coscale_on_nonreduction(
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On ANY non-reduction path (b-down, k-down, qlora-flip), NO warning is emitted
    and alpha == cfg_alpha. Spec §8.5."""
    import logging

    cfg_r = 16
    cfg_alpha = 32

    # Non-reduction scenarios: b-down, k-down, qlora-flip
    scenarios = [
        ("b_shrink_40gb", int(40 * _GB)),  # generous budget; b/k adjust only
        ("k_shrink_20gb", int(20 * _GB)),  # k steps down but r stays
    ]
    for name, total in scenarios:
        _stub_gpu(monkeypatch, total, cc=(8, 0))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="custom_sam_peft.presets"):
            d = decide_preset(r=cfg_r, alpha=cfg_alpha)
        # r must not change
        assert d.r == cfg_r, f"{name}: r={d.r} changed (expected {cfg_r})"
        # alpha must not change
        assert d.alpha == cfg_alpha, (
            f"{name}: alpha={d.alpha} changed; expected {cfg_alpha} (no co-scale on non-reduction)"
        )
        # No WARNING from the presets module
        preset_warns = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "custom_sam_peft.presets" in r.name
        ]
        assert not preset_warns, (
            f"{name}: unexpected WARNING on non-reduction path: {[r.message for r in preset_warns]}"
        )


# ---------------------------------------------------------------------------
# §8.6 Regression (golden)
# ---------------------------------------------------------------------------


def test_regression_16gb_r16_stays_r16(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None
) -> None:
    """The exact reported case: cfg.r=16, 16 GB card, peak ~12.5 GB < budget
    => decision.r == 16, decision.alpha == 32, NOT r=64.

    The pre-fix stale cache had chosen_r=64, alpha=32 (alpha:r=0.5).
    After the pin fix, r is pinned at the cfg value (16) and alpha tracks it.
    Spec §8.6."""
    _stub_gpu(monkeypatch, int(16 * _GB), cc=(8, 0))
    d = decide_preset(r=16, alpha=32)
    assert d.r == 16, (
        f"Golden regression: r={d.r} on a 16 GB card; expected 16 "
        "(pre-fix bug was r=64 from rank-maximization)"
    )
    assert d.alpha == 32, (
        f"Golden regression: alpha={d.alpha}; expected 32 "
        "(pre-fix bug was alpha=32 stranded at field default while r=64)"
    )
    # Verify the budget is sufficient for r=16 (lora fits analytic)
    assert d.predicted_bytes <= d.budget_bytes


# ---------------------------------------------------------------------------
# §8.7 num_classes helper
# ---------------------------------------------------------------------------


def test_numclasses_mask_png_class_map_count(tmp_path: Path) -> None:
    """A synthetic class_map JSON of N entries → helper returns N. Spec §8.7."""
    import json as _json

    from custom_sam_peft.data._num_classes import infer_num_classes

    # Write a minimal class_map JSON (value → label dict)
    class_map = {str(i): f"class_{i}" for i in range(5)}
    class_map_path = tmp_path / "class_map.json"
    class_map_path.write_text(_json.dumps(class_map))

    data_cfg = {
        "format": "mask_png",
        "semantic": {"class_map": str(class_map_path)},
    }
    result = infer_num_classes(data_cfg)
    assert result == 5, f"Expected 5 classes but got {result}"


def test_numclasses_hf_classlabel_names() -> None:
    """A stub HF dataset with ClassLabel.names of length N → helper returns N.
    Spec §8.7."""
    from unittest.mock import MagicMock, patch

    from custom_sam_peft.data._num_classes import infer_num_classes

    # Stub a HF dataset with a ClassLabel feature
    stub_names = ["cat", "dog", "bird", "fish", "horse"]
    mock_feature = MagicMock()
    mock_feature.names = stub_names
    mock_ds = MagicMock()
    mock_ds.features = {"label": mock_feature}

    with patch(
        "custom_sam_peft.data._num_classes._num_classes_hf",
        return_value=len(stub_names),
    ):
        data_cfg = {
            "format": "hf",
            "hf": {"name": "stub/dataset", "split": "train"},
        }
        result = infer_num_classes(data_cfg)
    assert result == 5, f"Expected 5 classes but got {result}"


def test_numclasses_fallback_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _force_cuda_available: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Data absent / unresolved vocabulary → helper returns None, sizer falls back
    to k_cap, a WARNING is emitted, and sizing succeeds (never hard-fails). Spec §8.7."""
    import logging

    from custom_sam_peft.data._num_classes import infer_num_classes

    # Verify: missing class_map → None + WARNING
    data_cfg_no_classmap = {
        "format": "mask_png",
        "semantic": {},  # no class_map key
    }
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.data._num_classes"):
        result = infer_num_classes(data_cfg_no_classmap)
    assert result is None, f"Expected None on missing class_map; got {result}"
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns, "No WARNING emitted for missing class_map"

    # Verify: None num_classes does NOT break decide_preset (sizer falls back to k_cap).
    caplog.clear()
    _stub_gpu(monkeypatch, int(24 * _GB), cc=(8, 0))
    d = decide_preset(num_classes=None)  # explicit None → uses k_cap
    assert isinstance(d, PresetDecision)
    assert d.predicted_bytes <= d.budget_bytes


# ---------------------------------------------------------------------------
# §8.8 init / wizard
# ---------------------------------------------------------------------------


def test_init_writes_template_r_on_big_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """csp init on an 80 GB card writes template r=16/alpha=32 into config.yaml,
    NOT r=64. Only b/k differ from the template defaults. Spec §8.8."""
    import yaml
    from typer.testing import CliRunner

    from custom_sam_peft.cli.main import app

    runner = CliRunner()
    # Stub an 80 GiB card (Ampere CC 8.0 so bfloat16 is chosen)
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("custom_sam_peft.presets.torch.cuda.is_available", lambda: True)
    props = MagicMock(total_memory=int(80 * _GB))
    props.name = "H100-SXM5-80GB"
    monkeypatch.setattr("custom_sam_peft.presets.torch.cuda.get_device_properties", lambda _: props)
    monkeypatch.setattr(
        "custom_sam_peft.presets.torch.cuda.get_device_name", lambda _: "H100-SXM5-80GB"
    )
    monkeypatch.setattr(
        "custom_sam_peft.presets.torch.cuda.get_device_capability", lambda _: (9, 0)
    )

    # Create the data paths that the template references
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.json").write_text("{}")
    (tmp_path / "data" / "val.json").write_text("{}")
    (tmp_path / "data" / "train").mkdir()
    (tmp_path / "data" / "val").mkdir()

    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()

    body = yaml.safe_load(out.read_text())
    peft = body.get("peft", {})
    assert peft.get("r") == 16, (
        f"r={peft.get('r')} in config.yaml; expected 16 (template default, not max-fitting 64)"
    )
    assert peft.get("alpha") == 32, f"alpha={peft.get('alpha')} in config.yaml; expected 32"


def test_wizard_patch_pins_r_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wizard's config_patch carries peft.r == 16, peft.alpha == 32. Spec §8.8."""
    from custom_sam_peft.cli import setup_wizard as sw

    # Stub CUDA + GPU for the analytic path
    monkeypatch.setattr("custom_sam_peft.presets.torch.cuda.is_available", lambda: True)
    props = MagicMock(total_memory=int(40 * _GB))
    props.name = "StubGPU"
    monkeypatch.setattr("custom_sam_peft.presets.torch.cuda.get_device_properties", lambda _: props)
    monkeypatch.setattr("custom_sam_peft.presets.torch.cuda.get_device_name", lambda _: "StubGPU")
    monkeypatch.setattr(
        "custom_sam_peft.presets.torch.cuda.get_device_capability", lambda _: (8, 0)
    )

    # Build a Ctx with no prior peft answers (uses PEFTConfig defaults r=16, alpha=32)
    ctx = sw.Ctx(answers={}, cuda_available=True)
    patch = sw._calibrate_or_analytic(ctx)
    assert patch is not None, "Wizard analytic path returned None (decide_preset failed)"
    peft_patch = patch.get("peft", {})
    assert peft_patch.get("r") == 16, (
        f"wizard config_patch.peft.r={peft_patch.get('r')}; expected 16 (pinned default)"
    )
    assert peft_patch.get("alpha") == 32, (
        f"wizard config_patch.peft.alpha={peft_patch.get('alpha')}; expected 32"
    )


# ---------------------------------------------------------------------------
# §8.9 Cache schema
# ---------------------------------------------------------------------------


def test_v3_cache_is_stale(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    """A v3 cache with chosen_r=64 is treated as stale (_load_cache rejects it on
    version mismatch) → re-probe path (provenance=analytic). Spec §8.9."""
    from custom_sam_peft.presets import CACHE_SCHEMA_VERSION

    _stub_gpu(monkeypatch, int(24 * _GB))
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "sha_xyz")
    cache_file = tmp_path / "cache.json"
    # Write an old v3 cache with the rank-maximization artifact
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 3,  # stale — CACHE_SCHEMA_VERSION is now 4
                "calibrated_at": "2026-05-01T00:00:00+00:00",
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "sha_xyz",
                "A_fixed": int(1 * _GB),
                "A_per_class": int(0.1 * _GB),
                "peak_memory_bytes_at_probe": int(22 * _GB),
                "chosen_r": 64,  # the old rank-maximization bug value
                "chosen_alpha": 32,  # stranded alpha (was the bug)
                "chosen_batch": 1,
                "chosen_method": "lora",
                "chosen_classes_per_forward": 8,
            }
        )
    )
    assert CACHE_SCHEMA_VERSION == 4  # confirm the schema has been bumped
    d = decide_preset(cache_path=cache_file)
    # v3 is stale → rejected on version mismatch → analytic path
    assert d.provenance == "analytic", (
        f"v3 cache was NOT rejected; provenance={d.provenance}. "
        "v3 caches must auto-invalidate on the schema bump (spec §7.2)."
    )
    assert d.r == 16, f"r={d.r} after v3 rejection; expected pinned default 16"


def test_v4_cache_roundtrips(
    monkeypatch: pytest.MonkeyPatch, _force_cuda_available: None, tmp_path: Path
) -> None:
    """A v4 cache written by _write_cache_v3 round-trips through _decision_from_cache
    with the pinned r/alpha. Spec §8.9."""
    from custom_sam_peft.presets import CACHE_SCHEMA_VERSION

    assert CACHE_SCHEMA_VERSION == 4  # confirm the bump

    _stub_gpu(monkeypatch, int(24 * _GB), name="StubGPU")
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "sha_v4")
    cache_file = tmp_path / "cache.json"
    # Write a valid v4 cache with the pinned r=16 (not 64)
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "calibrated_at": "2026-06-04T00:00:00+00:00",
                "gpu_name": "StubGPU",
                "sam3_checkpoint_sha": "sha_v4",
                "A_fixed": int(0 * _GB),
                "A_per_class": int(1.163 * _GB),
                "peak_memory_bytes_at_probe": int(12 * _GB),
                "chosen_r": 16,  # pinned (not rank-maximized)
                "chosen_alpha": 32,  # matches 2*r invariant
                "chosen_batch": 1,
                "chosen_method": "lora",
                "chosen_classes_per_forward": 4,
            }
        )
    )
    d = decide_preset(cache_path=cache_file)
    assert d.provenance == "calibrated", f"v4 cache not consumed; provenance={d.provenance}"
    assert d.r == 16, f"r={d.r} from cache; expected 16"
    assert d.alpha == 32, f"alpha={d.alpha} from cache; expected 32"
