"""`custom-sam-peft calibrate` — thin wrapper over `run_calibration`.

Three-stage probe: (1) derive split (two probes → `A_fixed`/`A_per_class`),
(2) analytic aim via `decide_preset`, (3) confirm-and-climb. Writes a v3 cache
(schema_version=3) at `./.custom_sam_peft_calibration.json`.

Spec: docs/superpowers/specs/2026-05-28-vram-calibration-reassess-design.md §4-§5.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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
    _flash_attention_available,
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
        return cast("str | None", json.loads(output.read_text()).get("calibrated_at"))
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


# A CUDA out-of-memory condition does NOT always raise torch.cuda.OutOfMemoryError.
# On some driver/arch/runtime combinations (observed: WSL2 + sm_120 + CUDA 13) an
# exhausted allocation surfaces as a generic RuntimeError carrying a lower-level
# driver/library status string instead — e.g. "CUDA driver error: device not ready"
# or "CUBLAS_STATUS_ALLOC_FAILED". The calibration climb must recognize these as
# "does not fit" (stop climbing / shrink) rather than abort the whole probe. (#208)
_OOM_SIGNATURES: tuple[str, ...] = (
    "out of memory",
    "device not ready",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)


def _is_cuda_oom(exc: BaseException) -> bool:
    """True iff *exc* is a CUDA out-of-memory condition.

    Matches the clean ``torch.cuda.OutOfMemoryError`` AND the dirty-OOM
    ``RuntimeError`` variants that some driver/arch/runtime stacks raise instead
    (see ``_OOM_SIGNATURES``). Pure function — unit-testable without a GPU.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(sig in msg for sig in _OOM_SIGNATURES)
    return False


def _run_probe(*, method: str, r: int, k_eff: int, batch: int) -> int:
    """Run one forward+backward at the config's (method, r, k_eff, batch).

    Returns peak bytes. CUDA only. Spec §5.1.
    """
    from custom_sam_peft.data.base import TextPrompts
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE, load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    k_eff = max(1, min(k_eff, MULTIPLEX_CAP))
    model_cfg = ModelConfig()
    wrapper = out = loss = images = None
    try:
        # No DataConfig in scope; rgb default is the documented exception (spec §5.4).
        wrapper = load_sam31(model_cfg, channels=3, channel_semantics="rgb")
        apply_lora(wrapper, PEFTConfig(method=method, r=r))

        device = next(wrapper.parameters()).device
        images = torch.zeros(
            batch, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, dtype=torch.bfloat16, device=device
        )
        # K_eff distinct synthetic class prompts per image (not a single "thing").
        prompts = [TextPrompts(classes=[f"class_{j}" for j in range(k_eff)]) for _ in range(batch)]

        torch.cuda.reset_peak_memory_stats()
        out = wrapper(images, prompts, support=None)
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for t in out.values():
            if isinstance(t, torch.Tensor):
                loss = loss + t.float().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        # Synchronize so the backward's kernels have all completed before we read
        # the peak and tear down — otherwise a kernel could still be allocating
        # (inflating the next probe) or could surface its error only later.
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    finally:
        # Release this probe's model + activations before the next ladder probe.
        # nn.Module plus the dtype forward-hooks form reference cycles, so the
        # model is NOT freed by refcounting when this function returns — it
        # lingers until a sporadic cyclic gc.collect(), and the CUDA caching
        # allocator never returns reserved blocks to the driver without
        # empty_cache(). Across the 4-6 probe ladder this accumulates to the
        # <=16GB ceiling and the next allocation dies as "CUDA driver error:
        # device not ready" (a dirty OOM on this WSL2/sm_120 path). Freeing per
        # probe also keeps each peak measurement clean: otherwise a probe's peak
        # is inflated by the prior probe's still-resident model. (#208)
        del wrapper, out, loss, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
            classes_per_forward=decision.classes_per_forward,
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

    The inverted overhead is REGIME-MATCHED (Amendment 2, spec §2.1): it must equal
    exactly what the _predicted_bytes branch adds for THIS card's SDPA regime, so the
    split reproduces the measured peak. overhead = STATIC + (attn(1) if not flash else
    0), flash = _flash_attention_available(cc). On the cc=12.0 dev box flash=True ->
    subtract STATIC only -> portable flash-baseline seeds. A_fixed clamps to >=0; a
    clamped-to-zero A_fixed is the EXPECTED dev-GPU outcome (encoder activation <
    model-weight conservatism margin), not an error.
    """
    try:
        peak_k1 = _run_probe(method="qlora", r=4, k_eff=1, batch=1)
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise
        raise _GpuTooSmall("K=1 probe OOMed — GPU too small") from exc

    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    # Regime-matched overhead (Amendment 2 / spec §2.1): STATIC + conditional
    # materialized attention, the SAME quantity the predictor adds for this card.
    cc = torch.cuda.get_device_capability(0)
    flash = _flash_attention_available(cc)
    static = _model_bytes("qlora") + _adapter_bytes(4) + _optimizer_bytes(4) + WORKSPACE_BYTES
    overhead = static + (0 if flash else _attention_bytes_per_example(SAM3_IMAGE_SIZE))
    try:
        peak_k4 = _run_probe(method="qlora", r=4, k_eff=4, batch=1)
        a_per_class = int((peak_k4 - peak_k1) / (4 - 1))
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise
        typer.echo(
            "WARNING: K=4 probe OOMed; falling back to analytic A_per_class seed",
            err=True,
        )
        a_per_class = A_PER_CLASS
    a_fixed = int(peak_k1 - overhead - a_per_class)

    # Warn ONLY on a negative A_per_class (a genuinely broken differential). A
    # negative A_fixed clamps to 0 silently — it is the expected outcome and must NOT
    # block the cache write (Amendment 2 / spec §2.1).
    if a_per_class < 0:
        typer.echo(
            f"WARNING: clamped negative A_per_class={a_per_class}; "
            "two-point differential looks broken — re-derive on a real GPU",
            err=True,
        )
        a_per_class = max(0, a_per_class)
    a_fixed = max(0, a_fixed)
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
        method=method,
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


def _confirm_and_climb(
    *, method: Literal["lora", "qlora"], r: int, batch: int, k: int, budget: int, k_cap: int
) -> tuple[Literal["lora", "qlora"], int, int, int, int]:
    """Stage 3: probe the aim, then climb (K then batch, at the fitting method/r) on
    headroom, or shrink down the FULL sacrifice order on OOM. Returns the empirical
    (method, r, batch, k, measured_peak). Bounded by the grid.

    Shrink order on OOM/over-budget, one probe per step (spec §4):
      1. batch > 1            -> batch -= 1
      2. else k > ks[0]       -> k = ks[ks.index(k) - 1]
      3. else r > _RS[0]      -> r = _RS[_RS.index(r) - 1]  (batch/K at their mins)
      4. else method == lora  -> method = "qlora"; r = _RS[-1]
         (QLoRA's NF4 base is far cheaper; re-try from the highest r and let the loop
          shrink r to the best-fitting QLoRA r — preserving accuracy where possible)
      5. else (qlora, r=_RS[0], batch=1, K=ks[0]) still OOM -> raise _GpuTooSmall.

    Climb-up NEVER raises r and NEVER flips method on a probe — accuracy levers are
    not raised on the strength of a probe (spec §4). It grows K to the next grid
    value first, then batch, at the fitting method/r only.
    """
    ks = [x for x in _KS if x <= k_cap]
    # Worst-case shrink-to-raise walk: confirm + batch down + K down + r down (LoRA)
    # + method flip + r down again (QLoRA), then one more iteration to hit the raise.
    # Two full r-descents -> 2*len(_RS). Slack +2 keeps the raise reachable (>=32).
    # Use len(_KS) — the unfiltered grid — as a safe upper bound on len(ks).
    max_probes = len(_BATCHES) + len(_KS) + 2 * len(_RS) + 2
    probes = 0

    def _probe_fits(m: str, rr: int, b: int, kk: int) -> tuple[bool, int]:
        nonlocal probes
        probes += 1
        try:
            peak = _run_probe(method=m, r=rr, k_eff=kk, batch=b)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            return False, 0
        return peak <= budget, peak

    # Confirm the Stage-2 aim; shrink down the full sacrifice order until it fits.
    fits, peak = _probe_fits(method, r, batch, k)
    while not fits and probes < max_probes:
        if batch > 1:
            batch -= 1
        elif k > ks[0]:
            k = ks[ks.index(k) - 1]
        elif r > _RS[0]:
            r = _RS[_RS.index(r) - 1]  # batch and K already at their minimums
        elif method == "lora":
            method = "qlora"  # cheaper NF4 base; retry from the highest r
            r = _RS[-1]
        else:
            # qlora, r=_RS[0], batch=1, K=ks[0] and still OOM -> GPU too small.
            raise _GpuTooSmall(
                "no config fits down to (qlora, r="
                f"{_RS[0]}, batch=1, K={ks[0]}) — candidate space exhausted"
            )
        fits, peak = _probe_fits(method, r, batch, k)

    # Climb: grow K to the next grid value first, then batch, at the fitting
    # method/r only (never raise r or flip method on a probe).
    best = (method, r, batch, k, peak)
    while probes < max_probes:
        if k < ks[-1]:
            cand_b, cand_k = batch, ks[ks.index(k) + 1]
        elif batch < _BATCHES[-1]:
            cand_b, cand_k = batch + 1, k
        else:
            break  # grid max reached
        fits, peak = _probe_fits(method, r, cand_b, cand_k)
        if not fits:
            break
        batch, k, best = cand_b, cand_k, (method, r, cand_b, cand_k, peak)
    return best


def run_calibration(*, config: Path, output: Path, force: bool) -> PresetDecision:
    """Three-stage model-guided calibration. Returns the chosen PresetDecision.

    Stage 1 derive -> Stage 2 analytic aim -> Stage 3 confirm-and-climb. Writes the
    v3 cache and rewrites the config sizing block. Spec §4.
    """
    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
    from custom_sam_peft.presets import PresetDecision, decide_preset

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

    # Stage 2 — analytic aim over the full grid using the derived split. This is the
    # ONLY analytic use in the probe path (the aim that the probe then confirms).
    _write_cache_v3(
        output, gpu_name=gpu_name, total=total, a_fixed=a_fixed, a_per_class=a_per_class, peak=0
    )
    aim = decide_preset(k=k_cap, cache_path=output)
    budget = aim.budget_bytes  # decide_preset already computed total - headroom

    # Stage 3 — confirm + climb/shrink down the full sacrifice order (bounded).
    # Returns the EMPIRICAL config; _GpuTooSmall is raised inside on full exhaustion.
    method, r, batch, k, peak = _confirm_and_climb(
        method=aim.method,
        r=aim.r,
        batch=aim.batch_size,
        k=aim.classes_per_forward,
        budget=budget,
        k_cap=k_cap,
    )

    # Persist the measured peak AND the empirically-chosen sizing (Correction B). The
    # chosen_* keys make this confirmed config authoritative on every later cache-fresh
    # read, so a re-run never reverts to the analytic aim.
    _write_cache_v3(
        output,
        gpu_name=gpu_name,
        total=total,
        a_fixed=a_fixed,
        a_per_class=a_per_class,
        peak=peak,
        method=method,
        r=r,
        batch=batch,
        classes_per_forward=k,
    )

    # Build the AUTHORITATIVE decision from the empirical tuple (Correction B): the
    # config rewrite and the returned PresetDecision both use THESE values, not a
    # re-derived analytic decide_preset.
    cc = torch.cuda.get_device_capability(0)
    dtype = "float16" if cc < (8, 0) else "bfloat16"
    headroom = _headroom_bytes()
    decision = PresetDecision(
        method=method,
        r=r,
        batch_size=batch,
        grad_accum_steps=max(1, 16 // batch),
        classes_per_forward=k,
        dtype=dtype,  # type: ignore[arg-type]
        headroom_bytes=headroom,
        predicted_bytes=peak,  # the real measured peak
        budget_bytes=total - headroom,
        gpu_name=gpu_name,
        provenance="calibrated",
        cache_path=output,
        calibrated_at=_cache_calibrated_at(output),
    )
    _apply_config_rewrite(config, decision=decision)
    return decision


def calibrate(
    output: Path = typer.Option(Path(CACHE_FILENAME), "--output", help="Cache file path."),
    force: bool = typer.Option(False, "--force", help="Re-probe even if the cache is fresh."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Training config YAML path."),
) -> None:
    """Probe peak VRAM at the config's (method, r, k, batch) and cache the result."""
    if not torch.cuda.is_available():
        typer.echo(f"ERROR: {_CUDA_HINT}", err=True)
        raise typer.Exit(code=2)
    try:
        decision = run_calibration(config=config, output=output, force=force)
    except _GpuTooSmall as exc:
        typer.echo(f"ERROR: {exc} — calibration probe OOMed; GPU too small", err=True)
        raise typer.Exit(code=5) from exc
    except _CheckpointMissing as exc:
        typer.echo(f"ERROR: SAM 3.1 checkpoint not found: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except _CacheWriteFailed as exc:
        typer.echo(f"ERROR: cache write failed: {exc}", err=True)
        raise typer.Exit(code=6) from exc
    except _CalibrationError as exc:
        typer.echo(f"ERROR: probe failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"ERROR: probe failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    typer.echo(decision.label())
