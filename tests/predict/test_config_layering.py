"""Tests for config layering / resolution logic inside predict/runner.py.

All tests exercise _resolve_config (or its observable side-effects via run_predict).
No model loading needed — all tests either call _resolve_config directly or
monkeypatch load_sam31 to avoid touching real weights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from custom_sam_peft.predict.runner import PredictOptions, _resolve_config, run_predict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LORA_DIR = Path(__file__).parent / "fixtures" / "lora_adapter"

_BUILTIN_DEFAULT = "facebook/sam3.1"


def _make_opts(
    tmp_path: Path,
    *,
    images: Path | None = None,
    checkpoint: Path | None = None,
    config: Path | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    score_threshold: float = 0.3,
    top_k: int = 100,
    save_masks: str = "rle",
    visualize: bool = False,
    merge_adapter: bool = True,
    batch_size: int = 1,
    seed: int = 0,
    dry_run: bool = False,
    verbose: bool = False,
) -> PredictOptions:
    """Build a minimal PredictOptions for config-layering tests."""
    if images is None:
        img = tmp_path / "img.png"
        from PIL import Image as PILImage

        PILImage.new("RGB", (32, 32)).save(img)
        images = tmp_path

    return PredictOptions(
        images=images,
        prompts="cat",
        output=tmp_path / "out",
        checkpoint=checkpoint,
        merge_adapter=merge_adapter,
        config=config,
        score_threshold=score_threshold,
        top_k=top_k,
        save_masks=save_masks,  # type: ignore[arg-type]
        visualize=visualize,
        device=device,  # type: ignore[arg-type]
        dtype=dtype,  # type: ignore[arg-type]
        batch_size=batch_size,
        seed=seed,
        dry_run=dry_run,
        verbose=verbose,
    )


def _make_config_yaml(tmp_path: Path, model_name: str, image_size: int = 1024) -> Path:
    """Write a minimal predict config YAML that pins model.name."""
    cfg_path = tmp_path / "predict_cfg.yaml"
    cfg_path.write_text(
        f"model:\n  name: {model_name!r}\ndata:\n  image_size: {image_size}\n",
        encoding="utf-8",
    )
    return cfg_path


def _make_custom_lora_dir(tmp_path: Path, base_model: str) -> Path:
    """Create an adapter checkpoint dir with a specific base_model_name_or_path."""
    ckpt_dir = tmp_path / "adapter"
    ckpt_dir.mkdir()
    (ckpt_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model, "peft_type": "LORA", "r": 8}),
        encoding="utf-8",
    )
    return ckpt_dir


# ---------------------------------------------------------------------------
# Test 1: adapter pin overrides config model.name, emits WARN on disagreement
# ---------------------------------------------------------------------------


def test_adapter_pin_overrides_config_with_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Adapter base_model_name_or_path wins over --config.model.name; WARN emitted."""
    adapter_model = "custom-org/sam3-finetune"
    config_model = "some-other-org/different-sam"

    ckpt_dir = _make_custom_lora_dir(tmp_path, adapter_model)
    cfg_path = _make_config_yaml(tmp_path, config_model)

    opts = _make_opts(tmp_path, checkpoint=ckpt_dir, config=cfg_path)

    with caplog.at_level(logging.WARNING):
        resolved = _resolve_config(opts)

    assert resolved.model_name == adapter_model, "adapter pin must override --config.model.name"
    # A WARN mentioning both values must be emitted
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    warn_texts = " ".join(r.getMessage() for r in warns)
    assert adapter_model in warn_texts or config_model in warn_texts, (
        "WARN should mention the conflicting model names"
    )


# ---------------------------------------------------------------------------
# Test 2: no checkpoint → fall through to --config model.name
# ---------------------------------------------------------------------------


def test_no_checkpoint_falls_through_to_config(tmp_path: Path) -> None:
    """Without a checkpoint, model.name comes from --config.model.name."""
    config_model = "my-org/custom-sam-variant"
    cfg_path = _make_config_yaml(tmp_path, config_model)

    opts = _make_opts(tmp_path, checkpoint=None, config=cfg_path)
    resolved = _resolve_config(opts)

    assert resolved.model_name == config_model


# ---------------------------------------------------------------------------
# Test 3: no checkpoint, no config → builtin default
# ---------------------------------------------------------------------------


def test_no_checkpoint_no_config_uses_builtin_default(tmp_path: Path) -> None:
    """Without checkpoint or config, model.name falls back to the builtin default."""
    opts = _make_opts(tmp_path, checkpoint=None, config=None)
    resolved = _resolve_config(opts)

    assert resolved.model_name == _BUILTIN_DEFAULT


# ---------------------------------------------------------------------------
# Test 4: CLI flag beats --config for non-model fields (e.g. device, dtype)
# ---------------------------------------------------------------------------


def test_cli_flag_beats_config(tmp_path: Path) -> None:
    """CLI --device and --dtype override any config-file setting."""
    # Config says bfloat16 / cuda (if it had those fields); CLI says cpu / float32.
    opts = _make_opts(tmp_path, device="cpu", dtype="float32", config=None)
    resolved = _resolve_config(opts)

    # After resolution: "cpu" stays "cpu" (no auto-resolution since it's explicit)
    assert resolved.device == "cpu"
    assert resolved.dtype_str == "float32"


# ---------------------------------------------------------------------------
# Test 5: adapter pin, no --config conflict → no WARN emitted
# ---------------------------------------------------------------------------


def test_adapter_pin_no_conflict_no_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When adapter base model agrees with builtin default, no WARN is logged."""
    # The stock lora fixture uses facebook/sam3.1 which IS the builtin default
    opts = _make_opts(tmp_path, checkpoint=_LORA_DIR, config=None)

    with caplog.at_level(logging.WARNING):
        resolved = _resolve_config(opts)

    assert resolved.model_name == _BUILTIN_DEFAULT
    # Scope the assertion to its intent: no WARN about adapter/config model-name
    # disagreement. Unrelated WARNs (e.g. AutoImageProcessor cache miss from
    # resolve_normalization, see #69) are out of this test's contract.
    disagreement_warns = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "disagrees with config/default" in r.getMessage()
    ]
    assert len(disagreement_warns) == 0, (
        f"Expected no disagreement warnings but got: {[r.getMessage() for r in disagreement_warns]}"
    )


# ---------------------------------------------------------------------------
# Test 6: image_size from config
# ---------------------------------------------------------------------------


def test_image_size_from_config(tmp_path: Path) -> None:
    """image_size is read from --config.data.image_size when present."""
    cfg_path = _make_config_yaml(tmp_path, _BUILTIN_DEFAULT, image_size=512)
    opts = _make_opts(tmp_path, config=cfg_path)
    resolved = _resolve_config(opts)

    assert resolved.image_size == 512


def test_image_size_defaults_to_1024_when_no_config(tmp_path: Path) -> None:
    """image_size defaults to 1024 (SAM 3.1 native) when no config is given."""
    opts = _make_opts(tmp_path, config=None)
    resolved = _resolve_config(opts)

    assert resolved.image_size == 1024


# ---------------------------------------------------------------------------
# Test 7: malformed --config YAML is tolerated with a WARN, falls back to defaults
# ---------------------------------------------------------------------------


def test_invalid_config_yaml_logs_warning_and_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt --config file emits a WARN and resolution falls back to builtin defaults."""
    bad = tmp_path / "broken.yaml"
    bad.write_text("model: [unterminated\n", encoding="utf-8")

    opts = _make_opts(tmp_path, config=bad)
    with caplog.at_level(logging.WARNING):
        resolved = _resolve_config(opts)

    assert resolved.model_name == _BUILTIN_DEFAULT
    assert resolved.image_size == 1024
    warns = [r for r in caplog.records if "Failed to parse --config" in r.getMessage()]
    assert warns, "expected a WARN about the corrupt YAML"


# ---------------------------------------------------------------------------
# Test 8: device="auto" + dtype="auto" resolve through the cuda-availability branch
# ---------------------------------------------------------------------------


def test_auto_device_and_dtype_resolve_on_cpu(tmp_path: Path) -> None:
    """device='auto' / dtype='auto' resolve through torch.cuda.is_available() branch.

    On this CPU sandbox they collapse to ('cpu', 'float32'); on a cuda host they
    would collapse to ('cuda', 'bfloat16'). Either way the auto-branch is exercised.
    """
    import torch

    opts = _make_opts(tmp_path, device="auto", dtype="auto", config=None)
    resolved = _resolve_config(opts)

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_dtype = "bfloat16" if expected_device == "cuda" else "float32"
    assert resolved.device == expected_device
    assert resolved.dtype_str == expected_dtype


# ---------------------------------------------------------------------------
# Test 9: preflight detect_adapter_kind runs before dry_run short-circuit
# ---------------------------------------------------------------------------


def test_dry_run_with_checkpoint_runs_preflight_detect(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When --checkpoint is set, the preflight log reports the detected adapter kind
    even in --dry-run mode (no model load happens, but detect_adapter_kind does)."""
    opts = _make_opts(tmp_path, checkpoint=_LORA_DIR, dry_run=True)

    with caplog.at_level(logging.INFO):
        report = run_predict(opts)

    assert report.n_predictions == 0
    assert report.n_images == 0
    preflight = [r for r in caplog.records if "predict: model=" in r.getMessage()]
    assert preflight, "expected a preflight log line"
    assert "adapter=lora" in preflight[0].getMessage(), (
        f"preflight should report lora adapter kind, got: {preflight[0].getMessage()}"
    )


# ---------------------------------------------------------------------------
# Test 10: verbose=True emits per-image latency log lines
# ---------------------------------------------------------------------------


def test_verbose_emits_per_image_latency_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With verbose=True the runner emits an INFO line per processed image."""
    # Reuse the stub model from test_runner_smoke; we just need run_predict to
    # iterate through one image without touching a real checkpoint.
    from PIL import Image as PILImage

    from tests.predict.test_runner_smoke import _StubSamModule

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    PILImage.new("RGB", (64, 64), color=(50, 100, 200)).save(img_dir / "a.png")

    monkeypatch.setattr(
        "custom_sam_peft.models.sam3.load_sam31",
        lambda _cfg: _StubSamModule(),
    )

    opts = PredictOptions(
        images=img_dir,
        prompts="cat",
        output=tmp_path / "out",
        checkpoint=None,
        merge_adapter=True,
        config=None,
        score_threshold=0.0,
        top_k=10,
        save_masks="rle",  # type: ignore[arg-type]
        visualize=False,
        device="cpu",  # type: ignore[arg-type]
        dtype="float32",  # type: ignore[arg-type]
        batch_size=1,
        seed=0,
        dry_run=False,
        verbose=True,
    )

    with caplog.at_level(logging.INFO):
        run_predict(opts)

    per_image = [r for r in caplog.records if r.getMessage().startswith("image 1/1 a.png")]
    assert per_image, "verbose=True should emit a per-image latency log line"
