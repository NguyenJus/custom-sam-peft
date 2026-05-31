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


def test_calibrate_writes_cache_with_schema_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    assert cache.is_file()
    data = json.loads(cache.read_text())
    expected_keys = {
        "schema_version",
        "calibrated_at",
        "gpu_name",
        "gpu_total_memory_bytes",
        "sam3_checkpoint_sha",
        "torch_version",
        "custom_sam_peft_version",
        "activation_bytes_per_example",
        "peak_memory_bytes_at_probe",
    }
    assert expected_keys.issubset(data.keys())
    assert data["schema_version"] == 2
    assert "image_size" not in data
    assert data["sam3_checkpoint_sha"] == "deadbeef"


def test_calibrate_cache_fresh_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Write a config too: the fresh-cache path now reads load_config before checking
    # freshness so it can derive k_eff for the config rewrite.
    _patch_probe(monkeypatch, tmp_path=tmp_path)
    # decide_preset validates the cache via presets._current_sam3_checkpoint_sha;
    # _patch_probe only rebinds the calibrate_cmd alias, so patch the presets-side
    # too (in production they are the same function) — otherwise decide_preset
    # rejects the cache on sha mismatch and falls back to analytic provenance.
    monkeypatch.setattr("custom_sam_peft.presets._current_sam3_checkpoint_sha", lambda: "deadbeef")
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "calibrated_at": "2026-05-22T00:00:00+00:00",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "gpu_total_memory_bytes": int(40 * _GB),
                "sam3_checkpoint_sha": "deadbeef",
                "torch_version": "2.4.0",
                "custom_sam_peft_version": "0.0.0",
                "activation_bytes_per_example": int(1 * _GB),
                "peak_memory_bytes_at_probe": int(38 * _GB),
            }
        )
    )
    mtime_before = cache.stat().st_mtime
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0, result.output
    assert "cache fresh" in result.output
    # The cache file must NOT be rewritten (mtime unchanged).
    assert cache.stat().st_mtime == mtime_before
    # The config sizing block MUST be rewritten from the cached values.
    cfg_path = tmp_path / "config.yaml"
    body = cfg_path.read_text()
    assert "# calibrated" in body
    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
    from custom_sam_peft.presets import decide_preset

    cfg = load_config(cfg_path)  # still a valid config
    # The rewritten sizing must reflect the *calibrated* decision (cache consumed),
    # not the analytic formula — otherwise the fresh-cache path silently ignored it.
    k_eff = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    calibrated = decide_preset(k=k_eff, cache_path=cache)
    assert calibrated.provenance == "calibrated"
    assert cfg.train.batch_size == calibrated.batch_size
    assert cfg.peft.r == calibrated.r
    assert cfg.train.grad_accum_steps == calibrated.grad_accum_steps


def test_calibrate_force_overwrites_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch, tmp_path=tmp_path)
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    cache.write_text('{"stale": true}')
    result = runner.invoke(app, ["calibrate", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads(cache.read_text())
    assert data.get("schema_version") == 2


def test_calibrate_non_cuda_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 2
    assert "CUDA" in result.output


def test_calibrate_negative_activation_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # peak much smaller than model+adapter+opt → negative raw activation.
    _patch_probe(monkeypatch, peak=10 * 1024**2, tmp_path=tmp_path)  # 10 MiB peak — tiny
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    assert data["activation_bytes_per_example"] == 0
    # The warning lands on stderr; CliRunner merges it into .output when mix_stderr=True (default).
    assert "negative" in result.output.lower() or "clamp" in result.output.lower()


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
    """calibrate reads cfg.peft.r/method and cfg.train.multiplex.classes_per_forward
    and passes them to the probe. Spec §5.1."""
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / "config.yaml", method="qlora", r=32, k=8)
    captured: dict[str, object] = {}

    def _fake_probe(*, method: str, r: int, k_eff: int, batch: int) -> int:
        captured.update(method=method, r=r, k_eff=k_eff, batch=batch)
        return int(38 * _GB)

    monkeypatch.setattr("custom_sam_peft.cli.calibrate_cmd._run_probe", _fake_probe)
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    assert captured == {"method": "qlora", "r": 32, "k_eff": 8, "batch": 1}


def test_calibrate_rewrites_config_in_place_annotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a successful probe, the config's sizing block is re-annotated
    '# calibrated <date>' and re-loads via load_config. Spec §5.3."""
    _patch_probe(monkeypatch)
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
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "config.yaml").exists()
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").exists()
    assert "not initialized" in result.output.lower() or "auto" in result.output.lower()


def _synthetic_peak(*, method: str, r: int, k_eff: int, batch: int) -> int:
    """Deterministic peak following the split model, for confirm-and-climb tests."""
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE
    from custom_sam_peft.presets import (
        WORKSPACE_BYTES,
        _adapter_bytes,
        _attention_bytes_per_example,
        _model_bytes,
        _optimizer_bytes,
    )

    a_fixed = 1_000_000_000
    a_per_class = 50_000_000
    overhead = (
        _model_bytes(method)
        + _adapter_bytes(r)
        + _optimizer_bytes(r)
        + WORKSPACE_BYTES
        + _attention_bytes_per_example(SAM3_IMAGE_SIZE) * batch
    )
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
    # New bound covers the larger walk (batch + K + r + method flip + the 2 derive
    # probes); mirror the _confirm_and_climb max_probes formula.
    assert count["n"] <= (
        len(calibrate_cmd._BATCHES)
        + len(calibrate_cmd._KS)
        + len(calibrate_cmd._RS)
        + 2  # derive probes
        + 2  # method flip + slack
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
    """I2: calibrate --output <non-default> threads the path to decide_preset.

    When --output points to a non-default path, decide_preset must receive that path
    as cache_path so provenance reflects the just-written probe rather than falling
    back to an absent default cache.
    """
    custom_cache = tmp_path / "my_custom_cache.json"
    _patch_probe(monkeypatch, tmp_path=tmp_path)
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
    # _load_cache (and therefore decide_preset) must have received the non-default path.
    assert captured_cache_path.get("path") == custom_cache, (
        f"_load_cache received cache_path={captured_cache_path.get('path')!r}, "
        f"expected {custom_cache}"
    )
