"""`custom-sam-peft calibrate` — probe peak VRAM at the config's (method, r, k, batch).

Writes `./.custom_sam_peft_calibration.json` (schema_version=2). Read by
`custom_sam_peft.presets._load_cache` so `decide_preset()` produces a tight,
GPU-accurate config instead of an analytic estimate.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §4-§5.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import typer

from custom_sam_peft import __version__ as _PKG_VERSION
from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.presets import (
    _CUDA_HINT,
    CACHE_FILENAME,
    CACHE_SCHEMA_VERSION,
    WORKSPACE_BYTES,
    _adapter_bytes,
    _attention_bytes_per_example,
    _model_bytes,
    _optimizer_bytes,
)
from custom_sam_peft.presets import (
    _current_sam3_checkpoint_sha as _sam3_checkpoint_sha,
)


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
    apply_lora(wrapper, PEFTConfig(method=method, r=r))  # type: ignore[arg-type]

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

    if not force and _cache_is_fresh(output, gpu_name):
        typer.echo("cache fresh — exiting")
        raise typer.Exit(code=0)

    cfg = load_config(config)
    method = cfg.peft.method
    r = cfg.peft.r
    k_eff = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    batch = cfg.train.batch_size

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
        typer.echo(f"ERROR: LoRA stub attach failed: {exc}", err=True)
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

    def _gib(b: int) -> float:
        return b / (1024**3)

    typer.echo(f"GPU:        {gpu_name} (SAM3_IMAGE_SIZE={SAM3_IMAGE_SIZE})")
    typer.echo(f"Peak:       {_gib(peak):.1f} GiB")
    typer.echo(f"Activation: {_gib(activation_per_example):.2f} GiB/example")
    typer.echo(f"Cache:      {output}")
