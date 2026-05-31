"""`custom-sam-peft calibrate` — probe peak VRAM at the config's (method, r, k, batch).

Writes `./.custom_sam_peft_calibration.json` (schema_version=2). Read by
`custom_sam_peft.presets._load_cache` so `decide_preset()` produces a tight,
GPU-accurate config instead of an analytic estimate.

Spec: docs/superpowers/specs/2026-05-28-vram-calibration-reassess-design.md §4-§5.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from custom_sam_peft.presets import PresetDecision

import torch
import typer

from custom_sam_peft import __version__ as _PKG_VERSION
from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.presets import (
    _CUDA_HINT,
    A_PER_CLASS,
    CACHE_FILENAME,
    CACHE_SCHEMA_VERSION,
    WORKSPACE_BYTES,
    _adapter_bytes,
    _attention_bytes_per_example,
    _headroom_bytes,
    _model_bytes,
    _optimizer_bytes,
)
from custom_sam_peft.presets import (
    _current_sam3_checkpoint_sha as _sam3_checkpoint_sha,
)

# Search grid mirrors presets._candidates Ks/batches/rs (spec §3/§4). The climb is
# bounded by these so a model error cannot loop (spec §4 "bounded probe count").
# The full sacrifice order on OOM is batch -> K -> r -> method (LoRA->QLoRA), so the
# climb walks _RS down (and flips method) to keep training fitting the GPU (spec §4).
_KS: tuple[int, ...] = (1, 2, 4, 8, 16)
_BATCHES: tuple[int, ...] = tuple(range(1, 17))
_RS: tuple[int, ...] = (8, 16, 24, 32, 48, 64)


class _CalibrationError(Exception):
    """Base for calibration failures; carries the CLI exit code."""

    exit_code = 4  # default: probe failure


class _GpuTooSmall(_CalibrationError):
    exit_code = 5


class _CheckpointMissing(_CalibrationError):
    exit_code = 3


class _CacheWriteFailed(_CalibrationError):
    exit_code = 6


def _cache_calibrated_at(output: Path) -> str | None:
    try:
        return json.loads(output.read_text()).get("calibrated_at")
    except (OSError, json.JSONDecodeError):
        return None


def _cache_is_fresh(path: Path, gpu_name: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        data.get("schema_version") == CACHE_SCHEMA_VERSION
        and data.get("gpu_name") == gpu_name
        and data.get("sam3_checkpoint_sha") == _sam3_checkpoint_sha()
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """tmp + os.replace; preserves prior file on failure."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def _run_probe(*, method: str, r: int, k_eff: int, batch: int) -> int:
    """Run one forward+backward at the config's (method, r, k_eff, batch).

    Returns peak bytes. CUDA only. Spec §5.1.
    """
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE, load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    k_eff = max(1, min(k_eff, MULTIPLEX_CAP))
    model_cfg = ModelConfig()
    # No DataConfig in scope; rgb default is the documented exception (spec §5.4).
    wrapper = load_sam31(model_cfg, channels=3, channel_semantics="rgb")
    apply_lora(wrapper, PEFTConfig(method=method, r=r))

    device = next(wrapper.parameters()).device
    images = torch.zeros(
        batch, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, dtype=torch.bfloat16, device=device
    )
    from custom_sam_peft.data.base import TextPrompts

    # K_eff distinct synthetic class prompts per image (not a single "thing").
    prompts = [TextPrompts(classes=[f"class_{j}" for j in range(k_eff)]) for _ in range(batch)]

    torch.cuda.reset_peak_memory_stats()
    out = wrapper(images, prompts, support=None)
    loss = torch.zeros((), device=device, dtype=torch.float32)
    for t in out.values():
        if isinstance(t, torch.Tensor):
            loss = loss + t.float().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    return int(torch.cuda.max_memory_allocated())


def _apply_config_rewrite(config: Path, *, decision: PresetDecision) -> None:
    """Rewrite the config's sizing block from an already-chosen PresetDecision.

    The caller passes the authoritative decision: the EMPIRICAL confirm-and-climb
    result for `calibrate` (Correction B), or the analytic decide_preset result for
    the cache-fresh path / init_cmd. This helper no longer re-derives sizing from the
    cache — it persists exactly what `decision` carries. Emits a WARNING on failure
    (OSError/ValueError/RuntimeError) and silently returns — the cache stays the
    authoritative output.
    """
    try:
        from custom_sam_peft.cli._config_rewrite import _rewrite_sizing_block

        annotation = f"# calibrated {datetime.now(UTC).date().isoformat()}"
        _rewrite_sizing_block(
            config,
            method=decision.method,
            r=decision.r,
            batch_size=decision.batch_size,
            grad_accum_steps=decision.grad_accum_steps,
            dtype=decision.dtype,
            annotation=annotation,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(
            f"WARNING: config rewrite failed (cache intact, config unchanged): {exc}",
            err=True,
        )


def _derive_split(method: str, r: int, batch: int) -> tuple[int, int]:
    """Stage 1: two cheap probes (K=1, K=4) -> (A_fixed, A_per_class).

    Raises _GpuTooSmall iff the K=1 probe OOMs. A K=4-probe OOM degrades to the
    analytic A_PER_CLASS seed (single-point A_fixed from peak_K1). Spec §4 Stage 1.
    """
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    try:
        peak_k1 = _run_probe(method="qlora", r=4, k_eff=1, batch=1)
    except torch.cuda.OutOfMemoryError as exc:
        raise _GpuTooSmall("K=1 probe OOMed — GPU too small") from exc

    fixed_overhead = (
        _model_bytes("qlora")
        + _adapter_bytes(4)
        + _optimizer_bytes(4)
        + WORKSPACE_BYTES
        + _attention_bytes_per_example(SAM3_IMAGE_SIZE) * 1
    )
    try:
        peak_k4 = _run_probe(method="qlora", r=4, k_eff=4, batch=1)
        a_per_class = int((peak_k4 - peak_k1) / (4 - 1))
    except torch.cuda.OutOfMemoryError:
        typer.echo(
            "WARNING: K=4 probe OOMed; falling back to analytic A_per_class seed",
            err=True,
        )
        a_per_class = A_PER_CLASS
    a_fixed = int(peak_k1 - fixed_overhead - a_per_class)

    if a_fixed < 0 or a_per_class < 0:
        typer.echo(
            f"WARNING: clamped negative split (A_fixed={a_fixed}, "
            f"A_per_class={a_per_class}); overhead model may need re-derivation",
            err=True,
        )
        a_fixed = max(0, a_fixed)
        a_per_class = max(0, a_per_class)
    return a_fixed, a_per_class


def _write_cache_v3(
    output: Path,
    *,
    gpu_name: str,
    total: int,
    a_fixed: int,
    a_per_class: int,
    peak: int,
    method: str | None = None,
    r: int | None = None,
    batch: int | None = None,
    classes_per_forward: int | None = None,
) -> None:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "calibrated_at": datetime.now(UTC).isoformat(),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total,
        "sam3_checkpoint_sha": _sam3_checkpoint_sha(),
        "torch_version": torch.__version__,
        "custom_sam_peft_version": _PKG_VERSION,
        "A_fixed": int(a_fixed),
        "A_per_class": int(a_per_class),
        "peak_memory_bytes_at_probe": int(peak),
    }
    # The empirically-chosen sizing (Correction B). ADDITIVE optional v3 keys: ABSENT
    # on the Stage-2 pre-probe placeholder write (peak=0, no chosen_* args); PRESENT
    # on the FINAL post-confirm write (all four passed from the _confirm_and_climb
    # tuple). Persisting them lets the cache-fresh early-return reconstruct the
    # authoritative empirical decision instead of re-deriving the analytic aim.
    if method is not None:
        payload["chosen_method"] = method
    if r is not None:
        payload["chosen_r"] = int(r)
    if batch is not None:
        payload["chosen_batch"] = int(batch)
    if classes_per_forward is not None:
        payload["chosen_classes_per_forward"] = int(classes_per_forward)
    try:
        _atomic_write_json(output, payload)
    except OSError as exc:
        raise _CacheWriteFailed(str(exc)) from exc


def _decision_from_cache(output: Path, k_cap: int) -> PresetDecision | None:
    """Reconstruct the AUTHORITATIVE empirical decision from a confirmed v3 cache.

    Returns the PresetDecision recorded by the last confirm-and-climb (provenance
    "calibrated") when the cache carries the `chosen_*` keys. Returns None when they
    are absent — a placeholder-only (pre-probe) or legacy cache holds no empirical
    record, so the caller must fall back to the analytic `decide_preset`. This is the
    cache-fresh dual of Correction B: a probe's empirical result, once written, stays
    authoritative across re-runs and never reverts to the analytic aim.
    """
    from custom_sam_peft.presets import PresetDecision

    try:
        data = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "chosen_method" not in data:
        return None
    method = data["chosen_method"]
    r = int(data["chosen_r"])
    batch = int(data["chosen_batch"])
    k = min(int(data["chosen_classes_per_forward"]), k_cap)
    peak = int(data["peak_memory_bytes_at_probe"])
    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)
    cc = torch.cuda.get_device_capability(0)
    dtype = "float16" if cc < (8, 0) else "bfloat16"
    headroom = _headroom_bytes()
    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=max(1, 16 // batch),
        classes_per_forward=k,
        dtype=dtype,  # type: ignore[arg-type]
        headroom_bytes=headroom,
        predicted_bytes=peak,
        budget_bytes=total - headroom,
        gpu_name=gpu_name,
        provenance="calibrated",
        cache_path=output,
        calibrated_at=_cache_calibrated_at(output),
    )


def run_calibration(*, config: Path, output: Path, force: bool) -> PresetDecision:
    """Three-stage model-guided calibration. Returns the chosen PresetDecision.

    Stage 1 derive -> Stage 2 analytic aim -> Stage 3 confirm-and-climb. Writes the
    v3 cache and rewrites the config sizing block. Spec §4.
    """
    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
    from custom_sam_peft.presets import decide_preset

    if not config.exists():
        from custom_sam_peft.cli.init_cmd import run_init

        typer.echo(
            f"WARNING: {config} not initialized — auto-init (formula, no probe) then probe.",
            err=True,
        )
        run_init("coco-text-lora", config, force=False)

    cfg = load_config(config)
    method = cfg.peft.method
    r = cfg.peft.r
    k_cap = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    batch = cfg.train.batch_size

    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)

    if not force and _cache_is_fresh(output, gpu_name):
        # No probe runs this invocation. Prefer the EMPIRICAL record persisted by a
        # prior confirm-and-climb (the `chosen_*` cache keys) so a re-run never
        # reverts a probe-reduced config back to the OOM-prone analytic aim. Only when
        # the cache holds no empirical record (placeholder-only / legacy cache) does
        # analytic `decide_preset` become the correct/only source (Correction B path b).
        typer.echo("cache fresh — exiting")
        decision = _decision_from_cache(output, k_cap)
        if decision is None:
            decision = decide_preset(k=k_cap, cache_path=output)
        _apply_config_rewrite(config, decision=decision)
        return decision

    try:
        a_fixed, a_per_class = _derive_split(method, r, batch)
    except FileNotFoundError as exc:
        raise _CheckpointMissing(str(exc)) from exc

    # Stage 2 + Stage 3 (filled in Tasks 2.2-2.3). Placeholder: write the split
    # cache from Stage 1 and aim analytically. NOTE: this analytic return is replaced
    # in Task 2.2 by the EMPIRICAL PresetDecision built from the confirm-and-climb
    # result (Correction B) — do not keep the analytic decide_preset return here past
    # Task 2.2.
    peak = a_fixed + a_per_class  # placeholder; replaced by Stage-3 measured peak
    _write_cache_v3(
        output, gpu_name=gpu_name, total=total, a_fixed=a_fixed, a_per_class=a_per_class, peak=peak
    )
    decision = decide_preset(k=k_cap, cache_path=output)  # placeholder; Task 2.2 makes empirical
    _apply_config_rewrite(config, decision=decision)
    return decision


def calibrate(
    output: Path = typer.Option(Path(CACHE_FILENAME), "--output", help="Cache file path."),
    force: bool = typer.Option(False, "--force", help="Re-probe even if the cache is fresh."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Training config YAML path."),
) -> None:
    """Probe peak VRAM at the config's (method, r, k, batch) and cache the result."""
    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE

    if not torch.cuda.is_available():
        typer.echo(f"ERROR: {_CUDA_HINT}", err=True)
        raise typer.Exit(code=2)

    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)

    if not config.exists():
        typer.echo(
            f"WARNING: {config} not initialized — auto-init (formula, no probe) then probe.",
            err=True,
        )
        from custom_sam_peft.cli.init_cmd import run_init

        run_init("coco-text-lora", config, force=False)

    cfg = load_config(config)
    method = cfg.peft.method
    r = cfg.peft.r
    k_eff = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    batch = cfg.train.batch_size

    if not force and _cache_is_fresh(output, gpu_name):
        typer.echo("cache fresh — exiting")
        _apply_config_rewrite(config, k_eff=k_eff, cache_path=output)
        raise typer.Exit(code=0)

    try:
        peak = _run_probe(method=method, r=r, k_eff=k_eff, batch=batch)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: SAM 3.1 checkpoint not found: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except torch.cuda.OutOfMemoryError as exc:
        typer.echo(
            "ERROR: calibration probe OOMed at config's sizing — GPU too small",
            err=True,
        )
        raise typer.Exit(code=5) from exc
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"ERROR: probe failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    overhead = (
        _model_bytes(method)
        + _adapter_bytes(r)
        + _optimizer_bytes(r)
        + WORKSPACE_BYTES
        + _attention_bytes_per_example(SAM3_IMAGE_SIZE) * batch
    )
    activation = peak - overhead
    if activation < 0:
        typer.echo(
            f"WARNING: negative activation ({activation} bytes); "
            "clamping to 0 — constants may need recalibration",
            err=True,
        )
        activation = 0

    # Store per-(example*K_eff) so _activation_per_example * k_eff reconstructs it.
    activation_per_example = int(activation / max(1, batch * k_eff))

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "calibrated_at": datetime.now(UTC).isoformat(),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total,
        "sam3_checkpoint_sha": _sam3_checkpoint_sha(),
        "torch_version": torch.__version__,
        "custom_sam_peft_version": _PKG_VERSION,
        "activation_bytes_per_example": int(activation_per_example),
        "peak_memory_bytes_at_probe": int(peak),
    }
    try:
        _atomic_write_json(output, payload)
    except OSError as exc:
        typer.echo(f"ERROR: cache write failed: {exc}", err=True)
        raise typer.Exit(code=6) from exc

    # Rewrite the config's sizing block in place with calibrated values.
    # Pass cache_path=output so decide_preset reads the freshly-written cache
    # (provenance="calibrated") even when --output is non-default.
    _apply_config_rewrite(config, k_eff=k_eff, cache_path=output)

    def _gib(b: int) -> float:
        return b / (1024**3)

    typer.echo(f"GPU:        {gpu_name} (SAM3_IMAGE_SIZE={SAM3_IMAGE_SIZE})")
    typer.echo(f"Peak:       {_gib(peak):.1f} GiB")
    typer.echo(f"Activation: {_gib(activation_per_example):.2f} GiB/example")
    typer.echo(f"Cache:      {output}")
