"""Tests for src/custom_sam_peft/cli/calibrate_cmd.py — calibration probe CLI.

All `models.sam3.load_sam31`, `peft_adapters.lora.apply_lora`, and
`torch.cuda.max_memory_allocated` are monkeypatched — these tests run on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from custom_sam_peft.cli.main import app

_GB = 1024**3
runner = CliRunner()


def _write_config(path: Path, *, method: str, r: int, k: int) -> None:
    """Write a minimal valid TrainConfig YAML with the given peft/multiplex settings."""
    content = f"""\
run:
  name: calibration-test
  output_dir: ./runs
  seed: 42

model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  dtype: bfloat16

data:
  format: coco
  train:
    annotations: data/placeholder/annotations.json
    images: data/placeholder/images

peft:
  method: {method}
  r: {r}

train:
  epochs: 1
  batch_size: 1
  grad_accum_steps: 8
  multiplex:
    classes_per_forward: {k}

tracking:
  backend: none
"""
    path.write_text(content)


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    peak: int = int(38 * _GB),
    gpu_name: str = "NVIDIA A100-SXM4-40GB",
    total: int = int(40 * _GB),
    sha: str = "deadbeef",
    tmp_path: Path | None = None,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = MagicMock(total_memory=total)
    props.name = gpu_name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: gpu_name)
    # Ampere (8, 0) → bfloat16; needed for decide_preset dtype resolution.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _idx: (8, 0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peak)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd._run_probe",
        lambda **kw: peak,
    )
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd._sam3_checkpoint_sha",
        lambda: sha,
    )
    if tmp_path is not None:
        _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)


def test_calibrate_writes_cache_with_schema_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["schema_version"] == 3
    assert {"A_fixed", "A_per_class"}.issubset(data.keys())
    assert "activation_bytes_per_example" not in data


def test_calibrate_k1_oom_exits_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "_run_probe",
        lambda **kw: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")),
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 5
    assert "GPU too small" in result.output


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), True),
        (RuntimeError("CUDA error: out of memory"), True),
        (RuntimeError("CUDA driver error: device not ready"), True),
        (RuntimeError("cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED"), True),
        (RuntimeError("shape '[4, 3]' is invalid for input of size 5"), False),
        (ValueError("bad config"), False),
        (KeyboardInterrupt(), False),
    ],
)
def test_is_cuda_oom_matches_clean_and_dirty_oom(exc: BaseException, expected: bool) -> None:
    """The matcher recognizes both torch.cuda.OutOfMemoryError and the dirty-OOM
    RuntimeError variants (e.g. WSL2/sm_120 'device not ready'), but NOT genuine
    non-OOM errors. (#208)"""
    from custom_sam_peft.cli.calibrate_cmd import _is_cuda_oom

    assert _is_cuda_oom(exc) is expected


def test_calibrate_k1_dirty_oom_exits_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A K=1 probe that OOMs as a dirty 'device not ready' RuntimeError (not
    torch.cuda.OutOfMemoryError) is still recognized as GPU-too-small. (#208)"""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "_run_probe",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("CUDA driver error: device not ready")),
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 5
    assert "GPU too small" in result.output


def test_calibrate_non_oom_runtimeerror_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine (non-OOM) RuntimeError in a probe must NOT be misread as
    GPU-too-small (exit 5); it surfaces as a probe failure (exit 4). (#208)"""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "_run_probe",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("einsum dimension mismatch")),
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 4
    assert "GPU too small" not in result.output


def test_calibrate_checkpoint_missing_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "_run_probe",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("checkpoint missing")),
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 3
    assert "SAM 3.1 checkpoint not found" in result.output


def test_calibrate_cache_fresh_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh v3 cache: exits 0, no re-probe, config rewritten from calibrated provenance."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    # decide_preset validates the cache via presets._current_sam3_checkpoint_sha;
    # _patch_probe only rebinds the calibrate_cmd alias, so patch the presets-side
    # too (in production they are the same function) — otherwise decide_preset
    # rejects the cache on sha mismatch and falls back to analytic provenance.
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "deadbeef")
    # Guard: no probes should fire on the cache-fresh path.
    monkeypatch.setattr(
        calibrate_cmd,
        "_run_probe",
        lambda **kw: (_ for _ in ()).throw(AssertionError("cache-fresh path must not probe")),
    )
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    # Seed a confirmed v3 cache with the chosen_* keys so _decision_from_cache
    # returns the empirical config (provenance="calibrated") on the fresh-cache path.
    cache.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "calibrated_at": "2026-05-22T00:00:00+00:00",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "gpu_total_memory_bytes": int(40 * _GB),
                "sam3_checkpoint_sha": "deadbeef",
                "torch_version": "2.4.0",
                "custom_sam_peft_version": "0.0.0",
                "A_fixed": int(1 * _GB),
                "A_per_class": 50 * 1024**2,
                "peak_memory_bytes_at_probe": int(38 * _GB),
                "chosen_method": "lora",
                "chosen_r": 16,
                "chosen_batch": 4,
                "chosen_classes_per_forward": 8,
            }
        )
    )
    mtime_before = cache.stat().st_mtime
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    assert "cache fresh" in result.output
    # The cache file must NOT be rewritten (mtime unchanged).
    assert cache.stat().st_mtime == mtime_before
    # The config sizing block MUST be rewritten from the empirical cached values.
    cfg_path = tmp_path / "config.yaml"
    body = cfg_path.read_text()
    assert "# calibrated" in body
    from custom_sam_peft.config.loader import load_config

    cfg = load_config(cfg_path)  # still a valid config
    # The rewritten sizing must reflect the empirical decision (provenance="calibrated"),
    # not the analytic formula.
    assert cfg.peft.r == 16
    assert cfg.train.batch_size == 4


def test_calibrate_force_overwrites_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"stale": true}')
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads(cache.read_text())
    assert data.get("schema_version") == 3
    assert {"A_fixed", "A_per_class"}.issubset(data.keys())
    assert "activation_bytes_per_example" not in data


def test_calibrate_non_cuda_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 2
    assert "CUDA" in result.output


def test_calibrate_negative_activation_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny peak (smaller than fixed overhead) makes _derive_split clamp A_fixed to 0.

    Under Amendment 2: a negative A_fixed clamps SILENTLY (no warning) — it is the
    expected dev-GPU outcome. Only a negative A_per_class (broken differential) emits
    a WARNING. With a constant tiny_peak both K=1 and K=4 probes return the same
    value, so a_per_class = 0 (no warning). Verify exit 0 and v3 cache with clamped
    zeros (spec §2.1).
    """
    from custom_sam_peft.cli import calibrate_cmd

    tiny_peak = 10 * 1024**2  # 10 MiB — much smaller than any overhead estimate
    _patch_probe(monkeypatch, peak=tiny_peak, tmp_path=tmp_path)
    # Both Stage-1 probes return the tiny peak; _derive_split computes a_per_class=0
    # (differential is 0) and a_fixed<0 -> clamped to 0 SILENTLY (Amendment 2).
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: tiny_peak)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["schema_version"] == 3
    assert data["A_fixed"] == 0
    assert data["A_per_class"] == 0


def test_calibrate_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"prior": true}')
    # Force the os.replace step to fail; the prior cache must still exist.
    monkeypatch.setattr(
        "custom_sam_peft.cli.calibrate_cmd.os.replace",
        lambda _src, _dst: (_ for _ in ()).throw(OSError("disk full")),
    )
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 6
    # The original file content survives the failed write.
    assert json.loads(cache.read_text()) == {"prior": True}


def test_calibrate_probes_at_config_r_and_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-3 confirm probe runs at the decision's (method, r, batch, K).

    The config's k_cap is respected as an upper bound: classes_per_forward from the
    config caps K throughout the analytic aim and Stage-3 confirm. Stage 1 always
    probes at qlora/r4/K=1,4 regardless of the config.
    """
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "config.yaml", method="qlora", r=32, k=8)
    calls: list[dict] = []

    def _fake_probe(*, method: str, r: int, k_eff: int, batch: int) -> int:
        calls.append({"method": method, "r": r, "k_eff": k_eff, "batch": batch})
        return _synthetic_peak(method=method, r=r, k_eff=k_eff, batch=batch)

    monkeypatch.setattr("custom_sam_peft.cli.calibrate_cmd._run_probe", _fake_probe)
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    # Stage 1 probes at qlora/r4 (fixed); Stage 3 probes at the analytic aim.
    stage1_calls = [c for c in calls if c["method"] == "qlora" and c["r"] == 4]
    assert len(stage1_calls) >= 1  # at least K=1 derive probe ran
    # The Stage-3 probe(s) must respect k_cap=8 (never exceed the config's k).
    assert all(c["k_eff"] <= 8 for c in calls), f"k_eff exceeded k_cap=8: {calls}"


def test_calibrate_rewrites_config_in_place_annotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a successful probe, the config's sizing block is re-annotated
    '# calibrated <date>' and re-loads via load_config. Spec §5.3."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, method="lora", r=16, k=16)
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    body = cfg_path.read_text()
    assert "# calibrated" in body
    from custom_sam_peft.config.loader import load_config

    assert load_config(cfg_path) is not None  # still valid


def test_calibrate_auto_inits_when_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config -> warn + auto-init (formula, no probe) then probe it. Spec §5.4."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "config.yaml").exists()
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").exists()
    assert "not initialized" in result.output.lower() or "auto" in result.output.lower()


def _synthetic_peak(*, method: str, r: int, k_eff: int, batch: int) -> int:
    """Deterministic peak following the split model, for confirm-and-climb tests.

    Overhead is the FLASH-BASELINE STATIC with NO attention term, matching the
    predictor / _derive_split in the FLASH regime (Amendment 2 / spec §2.1). The
    Stage-1 derive test therefore runs under a flash cc stub (cc=(8,0) set by
    _patch_probe) so _derive_split subtracts STATIC only and the solve is exact.
    """
    from custom_sam_peft.presets import (
        WORKSPACE_BYTES,
        _adapter_bytes,
        _model_bytes,
        _optimizer_bytes,
    )

    a_fixed = 1_000_000_000
    a_per_class = 50_000_000
    overhead = _model_bytes(method) + _adapter_bytes(r) + _optimizer_bytes(r) + WORKSPACE_BYTES
    activation = (a_fixed + a_per_class * k_eff) * batch
    return int(overhead + activation)


def test_run_calibration_stage1_solves_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)  # sets cuda stubs + writes config
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    calibrate_cmd.run_calibration(config=tmp_path / "config.yaml", output=out, force=True)
    data = json.loads(out.read_text())
    assert data["schema_version"] == 3
    # A_per_class solved from the two synthetic K=1/K=4 peaks (closed form).
    assert abs(data["A_per_class"] - 50_000_000) < 1_000_000
    assert "activation_bytes_per_example" not in data


def test_run_calibration_climbs_k_then_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    # Big card: synthetic peaks fit a wide grid -> climb should grow K then batch.
    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="BigGPU", total=int(80 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    assert decision.classes_per_forward >= 8
    data = json.loads(out.read_text())
    # Recorded peak is the final fitting probe's measured value, not the placeholder.
    assert data["peak_memory_bytes_at_probe"] > 10 * _GB


def test_run_calibration_shrinks_on_injected_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)
    calls: list[dict] = []

    def _probe(**kw):
        calls.append(kw)
        # OOM whenever batch>1 or K>2 (forces shrink batch-first then K).
        if kw["batch"] > 1 or kw["k_eff"] > 2:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    # Empirical (method, r, batch, k, peak) tuple drives the decision.
    assert decision.batch_size == 1
    assert decision.classes_per_forward <= 2
    assert decision.method == "lora"  # r/method not yet sacrificed here


def test_run_calibration_reduces_r_on_under_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    def _probe(**kw):
        # OOM for EVERY (batch, K) at the aimed r; fits only at a lower r.
        if kw["r"] > 16:
            raise torch.cuda.OutOfMemoryError("synthetic")
        if kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    assert decision.r <= 16  # r reduced to fit (full sacrifice order)
    import yaml

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["peft"]["r"] == decision.r  # written config matches, not the aimed r


def test_run_calibration_flips_to_qlora_when_lora_exhausts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)

    def _probe(**kw):
        # Every LoRA config OOMs (even r=_RS[0], batch=1, K=ks[0]); QLoRA fits.
        if kw["method"] == "lora":
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    assert decision.method == "qlora"


def test_run_calibration_decision_is_empirical_not_analytic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct regression guard for Correction B: when the real probe under-fits the
    analytic aim, the returned decision AND the written config equal the empirically
    fitting config, NOT the analytic aim."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    # The analytic aim (config A) over-predicts headroom and picks a high r/K/batch
    # that the real probe rejects; only a lower config (B) fits empirically.
    def _probe(**kw):
        if kw["r"] > 8 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    decision = calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=out, force=True
    )
    # Decision is the empirically-fitting config B, not the analytic aim A.
    assert decision.r == 8
    assert decision.batch_size == 1
    assert decision.classes_per_forward == 1
    import yaml

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["peft"]["r"] == 8
    assert cfg["train"]["batch_size"] == 1
    # Recorded peak is the real measured peak of config B.
    data = json.loads(out.read_text())
    assert data["peak_memory_bytes_at_probe"] == _synthetic_peak(
        method="lora", r=8, k_eff=1, batch=1
    )


def test_run_calibration_cache_fresh_returns_empirical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache-fresh re-run preserves the prior probe's empirical config (Correction B
    for the cached path): it must NOT revert to the OOM-prone analytic aim, and must
    NOT re-probe."""
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="SmallGPU", total=int(16 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    # First call (force=True): the probe UNDER-fits the analytic aim, so the empirical
    # config is lower-r/batch/K than the aim would pick.
    def _probe(**kw):
        if kw["r"] > 8 or kw["batch"] > 1 or kw["k_eff"] > 1:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".custom_sam_peft_calibration.json"
    first = calibrate_cmd.run_calibration(config=tmp_path / "config.yaml", output=out, force=True)
    data = json.loads(out.read_text())
    # The confirmed cache persists the empirically-chosen sizing.
    assert data["chosen_method"] == first.method == "lora"
    assert data["chosen_r"] == first.r == 8
    assert data["chosen_batch"] == first.batch_size == 1
    assert data["chosen_classes_per_forward"] == first.classes_per_forward == 1

    # Second call (force=False, cache fresh): must NOT probe and must return the SAME
    # empirical config — never the analytic aim (r=64/...).
    def _raise_if_probed(**kw):
        raise AssertionError("cache-fresh path must not re-probe")

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _raise_if_probed)
    monkeypatch.setattr(
        calibrate_cmd,
        "_derive_split",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache-fresh path must not derive")),
    )
    second = calibrate_cmd.run_calibration(config=tmp_path / "config.yaml", output=out, force=False)
    assert (second.method, second.r, second.batch_size, second.classes_per_forward) == (
        "lora",
        8,
        1,
        1,
    )
    assert second.provenance == "calibrated"


def test_run_calibration_probe_count_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="BigGPU", total=int(80 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)
    count = {"n": 0}

    def _probe(**kw):
        count["n"] += 1
        return _synthetic_peak(**kw)

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    calibrate_cmd.run_calibration(
        config=tmp_path / "config.yaml", output=tmp_path / "c.json", force=True
    )
    # New bound covers the larger walk: batch down + K down + two full r-descents
    # (LoRA then QLoRA after the method flip), plus the 2 derive probes.
    # Mirrors the _confirm_and_climb max_probes formula + 2 for _derive_split.
    assert count["n"] <= (
        len(calibrate_cmd._BATCHES)
        + len(calibrate_cmd._KS)
        + 2 * len(calibrate_cmd._RS)  # two r-descents: LoRA + QLoRA
        + 2  # _confirm_and_climb slack
        + 2  # derive probes (_derive_split runs 2 probes before Stage 3)
    )


def test_run_calibration_k1_oom_raises_gpu_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_sam_peft.cli import calibrate_cmd

    _patch_probe(monkeypatch, tmp_path=tmp_path)
    _write_config(tmp_path / "config.yaml", method="lora", r=16, k=16)

    def _probe(**kw):
        raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(calibrate_cmd._GpuTooSmall):
        calibrate_cmd.run_calibration(
            config=tmp_path / "config.yaml", output=tmp_path / "c.json", force=True
        )


def test_calibrate_non_default_output_uses_calibrated_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I2: calibrate --output <non-default> writes cache at the custom path.

    When --output points to a non-default path, run_calibration must write the v3
    cache there and the returned decision must have provenance="calibrated".
    """
    from custom_sam_peft.cli import calibrate_cmd

    custom_cache = tmp_path / "my_custom_cache.json"
    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(calibrate_cmd, "_run_probe", lambda **kw: _synthetic_peak(**kw))
    monkeypatch.chdir(tmp_path)

    # Intercept _load_cache (called from inside decide_preset) to capture which
    # cache_path it receives. This avoids patching the lazily-imported decide_preset.
    captured_cache_path: dict[str, object] = {}
    from custom_sam_peft import presets as _presets_mod

    orig_load_cache = _presets_mod._load_cache

    def _spy_load_cache(gpu_name: str, cache_path: Path | None = None) -> tuple[object, object]:
        captured_cache_path["path"] = cache_path
        return orig_load_cache(gpu_name, cache_path=cache_path)

    monkeypatch.setattr("custom_sam_peft.presets._load_cache", _spy_load_cache)

    result = runner.invoke(
        app, ["calibrate", "--output", str(custom_cache), "--config", "config.yaml"]
    )
    assert result.exit_code == 0, result.output
    assert custom_cache.is_file(), "cache was not written to custom path"
    data = json.loads(custom_cache.read_text())
    assert data["schema_version"] == 3
    # _load_cache (and therefore decide_preset) must have received the non-default path.
    assert captured_cache_path.get("path") == custom_cache, (
        f"_load_cache received cache_path={captured_cache_path.get('path')!r}, "
        f"expected {custom_cache}"
    )


def test_run_calibration_large_aim_all_probes_oom_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: big GPU -> high analytic aim (batch=16) -> every Stage-3 probe OOMs
    -> must raise _GpuTooSmall, NOT silently return a non-fitting config (peak=0).

    With total=100*_GB the analytic aim is (lora, r=64, batch=16, k=16). The shrink
    walk requires 31 probes before the else:raise branch is reachable.

    Buggy bound (max_probes=29): loop exits at probes=29 with fits=False, returns a
    config with peak=0 instead of raising. Fixed bound (max_probes=35>=32): loop stays
    open at probes=31, the else:raise fires.
    """
    from custom_sam_peft.cli import calibrate_cmd

    # 100 GiB GPU: analytic aim lands at (lora, r=64, batch=16, k=16), forcing the
    # full 31-probe shrink walk before the else:raise is reachable.
    _patch_probe(monkeypatch, tmp_path=tmp_path, gpu_name="BigGPU", total=int(100 * _GB))
    _write_config(tmp_path / "config.yaml", method="lora", r=64, k=16)

    def _probe(**kw):
        # Let the two _derive_split probes succeed (qlora/r=4/K=1 and qlora/r=4/K=4).
        # OOM everything else so the Stage-3 shrink must walk the full ladder to
        # exhaustion and hit the else:raise branch.
        if kw["method"] == "qlora" and kw["r"] == 4:
            return _synthetic_peak(**kw)
        raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(calibrate_cmd, "_run_probe", _probe)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(calibrate_cmd._GpuTooSmall):
        calibrate_cmd.run_calibration(
            config=tmp_path / "config.yaml",
            output=tmp_path / "c.json",
            force=True,
        )
