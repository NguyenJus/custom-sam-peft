#!/usr/bin/env python3
"""Trunk-feature-cache feasibility spike (issue #300, Part A).

Measures three quantities needed before any cache wiring lands:

  1. Per-image feature bytes in fp16 — WITH and WITHOUT the sam2_backbone_out
     path (sam2_backbone_out is None in typical SAM 3.1 deployments; included
     for completeness).  Excludes vision_pos_enc (content-independent, not
     cached per spec §1).

  2. Trunk-forward wall-clock as a FRACTION of a full train_step, measured via
     the permanent CSP_PROFILE=1 bucket-timer harness (eval.forward /
     train.forward buckets).  H2D copy time for one image's cached (CPU-pinned)
     features is measured separately.  Prints net-win estimate:
       net_win_per_step = trunk_fwd_mean - h2d_copy_mean

  3. Break-even table — total cache size vs the 16 GB host-RAM budget and vs
     disk-I/O headroom, for a range of dataset sizes.

This script is a MEASUREMENT TOOL ONLY.  It does NOT modify training, does NOT
write a cache, and runs forward passes only (no backward, no optimizer).

Residence choice (RAM / disk / hybrid) is left to the human reading the output.

Usage
-----
Minimal (default config, auto-resolves checkpoint)::

    CSP_PROFILE=1 uv run python scripts/spike_trunk_cache_feasibility.py

Override model checkpoint (e.g. specific local file)::

    CSP_PROFILE=1 uv run python scripts/spike_trunk_cache_feasibility.py \\
        --checkpoint models/sam3.1/sam3.1_multiplex.pt \\
        --dtype bfloat16 \\
        --batch 1 \\
        --warmup 2 \\
        --iters 10 \\
        --dataset-sizes 100 500 1000 5000 10000

Dump a profile snapshot for post-hoc inspection::

    CSP_PROFILE=1 uv run python scripts/spike_trunk_cache_feasibility.py \\
        --snapshot-out /tmp/spike_snapshot.json

Notes
-----
- Run with CSP_PROFILE=1 so bucket() accumulates timing in the permanent
  profiling facility.  Without it the harness still runs but prints wall-clock
  from an internal perf_counter fallback.
- Requires the real SAM 3.1 checkpoint (auto-downloaded from HF if absent in
  the default models/sam3.1/ local dir).
- bitsandbytes is NOT required (QLoRA path is not exercised here).
- The script never calls backward() or step() — VRAM stays at inference-mode
  peak.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure-arithmetic break-even calculator (no GPU, no model).
# ---------------------------------------------------------------------------


def bytes_per_image_fp16(
    fpn_shapes: list[tuple[int, int, int]],
    *,
    sam2_fpn_shapes: list[tuple[int, int, int]] | None = None,
) -> dict[str, int]:
    """Return fp16 byte counts for one cached image (batch-unbound, detach'd).

    Parameters
    ----------
    fpn_shapes:
        List of (C, H, W) per FPN level for the main backbone_fpn pyramid.
        vision_features is backbone_fpn[-1] — NOT double-counted.
    sam2_fpn_shapes:
        Optional list of (C, H, W) per FPN level for sam2_backbone_out.
        Pass None (or []) when sam2_backbone_out is None.

    Returns
    -------
    dict with keys:
        "backbone_fpn_bytes"   — bytes for the main FPN pyramid (fp16)
        "sam2_fpn_bytes"       — bytes for sam2 pyramid if present, else 0
        "total_bytes"          — sum of both (no double-counting)
        "total_bytes_no_sam2"  — backbone_fpn_bytes only (sam2 absent path)
    """
    bytes_per_elem = 2  # fp16

    fpn_bytes = sum(c * h * w * bytes_per_elem for c, h, w in fpn_shapes)

    sam2_bytes = 0
    if sam2_fpn_shapes:
        sam2_bytes = sum(c * h * w * bytes_per_elem for c, h, w in sam2_fpn_shapes)

    return {
        "backbone_fpn_bytes": fpn_bytes,
        "sam2_fpn_bytes": sam2_bytes,
        "total_bytes": fpn_bytes + sam2_bytes,
        "total_bytes_no_sam2": fpn_bytes,
    }


_GB = 1024**3
_MB = 1024**2
_HOST_RAM_BUDGET_GB = 16.0
_DISK_WARN_THRESHOLD_GB = 50.0  # HDD saturation crash risk starts here


def breakeven_table(
    bytes_without_sam2: int,
    bytes_with_sam2: int,
    dataset_sizes: list[int],
    *,
    host_ram_budget_gb: float = _HOST_RAM_BUDGET_GB,
    disk_warn_threshold_gb: float = _DISK_WARN_THRESHOLD_GB,
) -> str:
    """Return a formatted break-even table as a string.

    Computes total cache size for each dataset size (both WITH and WITHOUT the
    sam2_backbone_out path), checks against the 16 GB host-RAM budget, and
    flags disk headroom against the known HDD-saturation crash risk.

    Parameters
    ----------
    bytes_without_sam2:
        Per-image fp16 bytes for the backbone_fpn-only path.
    bytes_with_sam2:
        Per-image fp16 bytes when sam2_backbone_out is also cached.
    dataset_sizes:
        List of image counts to evaluate (e.g. [100, 500, 1000, 5000]).
    host_ram_budget_gb:
        Host-RAM budget in GiB (default 16).
    disk_warn_threshold_gb:
        Disk-size threshold in GiB above which a HDD-saturation warning is
        emitted (default 50).
    """
    lines: list[str] = []
    lines.append("\n=== Break-even table: cache size vs 16 GB host-RAM budget ===\n")
    lines.append(f"  Per-image bytes (no sam2) : {bytes_without_sam2 / _MB:>8.2f} MiB")
    lines.append(f"  Per-image bytes (w/ sam2) : {bytes_with_sam2 / _MB:>8.2f} MiB")
    lines.append(f"  Host-RAM budget           : {host_ram_budget_gb:.0f} GiB")
    lines.append(f"  HDD-saturation warn level : {disk_warn_threshold_gb:.0f} GiB\n")

    header = (
        f"  {'N images':>10}  {'GB no-sam2':>12}  {'RAM fit?':>9}"
        f"  {'GB w/-sam2':>12}  {'RAM fit?':>9}  Disk risk"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for n in dataset_sizes:
        gb_no = n * bytes_without_sam2 / _GB
        gb_wi = n * bytes_with_sam2 / _GB

        fits_no = "YES" if gb_no <= host_ram_budget_gb else "NO "
        fits_wi = "YES" if gb_wi <= host_ram_budget_gb else "NO "

        disk_flag = ""
        if gb_no >= disk_warn_threshold_gb or gb_wi >= disk_warn_threshold_gb:
            disk_flag = "WARN: HDD-saturation crash risk on this box"

        lines.append(
            f"  {n:>10}  {gb_no:>12.3f}  {fits_no:>9}  {gb_wi:>12.3f}  {fits_wi:>9}  {disk_flag}"
        )

    lines.append("")
    lines.append(
        "  NOTE: disk residence is EXPLICITLY weighed against HDD-saturation crash"
        " risk — sessions\n"
        "  on this box crash from disk-I/O saturation, not RAM/VRAM OOM."
        " Prefer RAM-cap if it fits,\n"
        "  hybrid if only no-sam2 fits, disk only if the dataset is too large"
        " for RAM AND disk I/O is\n"
        "  confirmed non-saturating on a representative run."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone break-even CLI entry point (no GPU required).
# ---------------------------------------------------------------------------


def _breakeven_main(argv: list[str] | None = None) -> int:
    """Standalone CLI: compute break-even table from supplied byte counts."""
    parser = argparse.ArgumentParser(
        prog="spike_trunk_cache_feasibility --breakeven",
        description=(
            "Pure-arithmetic break-even table (no GPU). "
            "Supply measured per-image byte counts and dataset sizes."
        ),
    )
    parser.add_argument(
        "--bytes-no-sam2",
        type=int,
        required=True,
        metavar="BYTES",
        help="Per-image fp16 bytes for backbone_fpn-only path (from spike run).",
    )
    parser.add_argument(
        "--bytes-with-sam2",
        type=int,
        default=None,
        metavar="BYTES",
        help=(
            "Per-image fp16 bytes when sam2_backbone_out is also cached. "
            "Defaults to --bytes-no-sam2 when omitted."
        ),
    )
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000, 10000, 50000],
        metavar="N",
        help="Dataset image counts to evaluate (default: 100 500 1000 5000 10000 50000).",
    )
    parser.add_argument(
        "--ram-budget-gb",
        type=float,
        default=_HOST_RAM_BUDGET_GB,
        metavar="GB",
        help=f"Host-RAM budget in GiB (default: {_HOST_RAM_BUDGET_GB}).",
    )
    parser.add_argument(
        "--disk-warn-gb",
        type=float,
        default=_DISK_WARN_THRESHOLD_GB,
        metavar="GB",
        help=f"HDD-saturation warn threshold in GiB (default: {_DISK_WARN_THRESHOLD_GB}).",
    )
    args = parser.parse_args(argv)

    b_no = args.bytes_no_sam2
    b_wi = args.bytes_with_sam2 if args.bytes_with_sam2 is not None else b_no

    print(  # noqa: T201
        breakeven_table(
            b_no,
            b_wi,
            args.dataset_sizes,
            host_ram_budget_gb=args.ram_budget_gb,
            disk_warn_threshold_gb=args.disk_warn_gb,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# GPU spike harness — requires real SAM 3.1 model + CUDA.
# ---------------------------------------------------------------------------


def _measure_trunk_bytes(
    forward_image_out: dict,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Extract per-image fp16 byte counts from a forward_image output dict.

    Excludes vision_pos_enc (content-independent, not cached per spec §1).
    Divides batch tensors by batch_size to get per-image figures.

    Returns a dict with:
        fpn_shapes      — list of (C, H, W) per FPN level
        fpn_bytes       — total backbone_fpn bytes in fp16 (per image)
        sam2_fpn_shapes — list of (C, H, W) if sam2_backbone_out present, else []
        sam2_fpn_bytes  — total sam2 pyramid bytes in fp16, else 0
        total_bytes     — fpn_bytes + sam2_fpn_bytes
        total_bytes_no_sam2 — fpn_bytes only
    """

    bytes_per_fp16 = 2

    fpn_tensors: list = forward_image_out["backbone_fpn"]
    fpn_shapes = []
    fpn_bytes = 0
    for t in fpn_tensors:
        # t is (B, C, H, W) — divide by B for per-image
        c, h, w = t.shape[1], t.shape[2], t.shape[3]
        fpn_shapes.append((c, h, w))
        fpn_bytes += c * h * w * bytes_per_fp16

    sam2_out = forward_image_out.get("sam2_backbone_out")
    sam2_fpn_shapes: list[tuple[int, int, int]] = []
    sam2_fpn_bytes = 0
    if sam2_out is not None:
        for t in sam2_out["backbone_fpn"]:
            c, h, w = t.shape[1], t.shape[2], t.shape[3]
            sam2_fpn_shapes.append((c, h, w))
            sam2_fpn_bytes += c * h * w * bytes_per_fp16

    return {
        "fpn_shapes": fpn_shapes,
        "fpn_bytes": fpn_bytes,
        "sam2_fpn_shapes": sam2_fpn_shapes,
        "sam2_fpn_bytes": sam2_fpn_bytes,
        "total_bytes": fpn_bytes + sam2_fpn_bytes,
        "total_bytes_no_sam2": fpn_bytes,
    }


def _pin_copy_tensors(
    forward_image_out: dict,
    *,
    batch_size: int,
) -> list[dict]:
    """Detach, move to CPU-pinned memory, and split per-image (like a real cache would).

    Returns a list of dicts — one per image in the batch — where each dict
    contains the cached tensors (backbone_fpn levels + sam2_backbone_out if
    present), all on CPU-pinned memory.

    vision_pos_enc is excluded (content-independent, not stored per spec §1).
    """
    import torch

    per_image: list[dict] = []
    for i in range(batch_size):
        entry: dict = {}
        # backbone_fpn: list of (B, C, H, W) -> slice to (1, C, H, W) then pin
        entry["backbone_fpn"] = [
            t[i : i + 1].detach().to(dtype=torch.float16).cpu().pin_memory()
            for t in forward_image_out["backbone_fpn"]
        ]
        # vision_features is backbone_fpn[-1] — do NOT duplicate; store a
        # reference so replay can reconstruct via fpn[-1].
        # sam2_backbone_out (if present)
        sam2_out = forward_image_out.get("sam2_backbone_out")
        if sam2_out is not None:
            entry["sam2_backbone_out"] = {
                "backbone_fpn": [
                    t[i : i + 1].detach().to(dtype=torch.float16).cpu().pin_memory()
                    for t in sam2_out["backbone_fpn"]
                ],
                "vision_features": None,  # will be fpn[-1] on replay
            }
        per_image.append(entry)
    return per_image


def _h2d_copy_one(
    pinned_entry: dict,
    device,
) -> None:
    """Non-blocking H2D copy of one cached image entry (simulates cache replay)."""
    import torch

    for t in pinned_entry["backbone_fpn"]:
        t.to(device, non_blocking=True)
    sam2 = pinned_entry.get("sam2_backbone_out")
    if sam2 is not None:
        for t in sam2["backbone_fpn"]:
            t.to(device, non_blocking=True)
    torch.cuda.synchronize()


def _fmt_bytes(n: int) -> str:
    if n >= _GB:
        return f"{n / _GB:.3f} GiB"
    if n >= _MB:
        return f"{n / _MB:.2f} MiB"
    return f"{n / 1024:.1f} KiB"


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.2f} ms"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spike_trunk_cache_feasibility",
        description=(
            "Trunk-feature-cache feasibility spike (issue #300, Part A). "
            "Measures per-image feature bytes and trunk-forward wall-clock "
            "vs H2D copy time on the real SAM 3.1 model."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to sam3.1_multiplex.pt. "
            "Defaults to models/sam3.1/sam3.1_multiplex.pt (auto-downloaded from HF if absent)."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype (default: bfloat16, matching the training default).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="B",
        help="Batch size for trunk-forward timing (default: 1, matching typical train batch).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        metavar="N",
        help="Number of warm-up iterations before timing (default: 3).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        metavar="N",
        help="Number of timed iterations for mean/median (default: 10).",
    )
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000, 10000, 50000],
        metavar="N",
        help=(
            "Dataset image counts for the break-even table "
            "(default: 100 500 1000 5000 10000 50000)."
        ),
    )
    parser.add_argument(
        "--snapshot-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: dump a CSP_PROFILE JSON snapshot to this path.",
    )
    parser.add_argument(
        "--breakeven",
        action="store_true",
        help=(
            "Run in pure-arithmetic break-even mode (no GPU). "
            "Requires --bytes-no-sam2. Ignores all GPU-related flags."
        ),
    )
    parser.add_argument(
        "--bytes-no-sam2",
        type=int,
        default=None,
        metavar="BYTES",
        help="(--breakeven mode only) per-image fp16 bytes for backbone_fpn-only path.",
    )
    parser.add_argument(
        "--bytes-with-sam2",
        type=int,
        default=None,
        metavar="BYTES",
        help="(--breakeven mode only) per-image fp16 bytes when sam2_backbone_out is cached.",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Pure break-even mode (no GPU).
    # ------------------------------------------------------------------
    if args.breakeven:
        if args.bytes_no_sam2 is None:
            sys.exit("error: --breakeven requires --bytes-no-sam2")
        b_no = args.bytes_no_sam2
        b_wi = args.bytes_with_sam2 if args.bytes_with_sam2 is not None else b_no
        print(  # noqa: T201
            breakeven_table(b_no, b_wi, args.dataset_sizes)
        )
        return 0

    # ------------------------------------------------------------------
    # GPU spike path — late imports so script stays importable on CPU-only envs.
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError as exc:
        sys.exit(f"error: torch not available: {exc}")

    if not torch.cuda.is_available():
        sys.exit(
            "error: CUDA is not available. "
            "This spike requires the GPU box with real SAM 3.1 weights. "
            "For pure break-even arithmetic only, use --breakeven --bytes-no-sam2 <N>."
        )

    try:
        import custom_sam_peft.profiling as prof
        from custom_sam_peft.config.schema import ModelConfig
        from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE, load_sam31
    except ImportError as exc:
        sys.exit(
            f"error: cannot import custom_sam_peft: {exc}\n"
            "Run via: uv run python scripts/spike_trunk_cache_feasibility.py ..."
        )

    # Enable the permanent profiling harness for this run.
    prof.enable()
    prof.reset()

    # ------------------------------------------------------------------
    # Build ModelConfig from CLI args.
    # ------------------------------------------------------------------
    model_cfg_kwargs: dict = {"dtype": args.dtype}
    if args.checkpoint is not None:
        # Override the local_dir / checkpoint_file pair from a full path.
        ckpt = args.checkpoint.resolve()
        model_cfg_kwargs["local_dir"] = str(ckpt.parent)
        model_cfg_kwargs["checkpoint_file"] = ckpt.name

    model_cfg = ModelConfig(**model_cfg_kwargs)

    print(f"\n[spike] Loading SAM 3.1 (dtype={args.dtype})...")  # noqa: T201
    wrapper = load_sam31(model_cfg, channels=3, channel_semantics="rgb")
    wrapper.eval()

    device = next(wrapper.parameters()).device
    print(f"[spike] Model on device={device}, dtype={args.dtype}")  # noqa: T201

    B = args.batch
    images = torch.zeros(
        B,
        3,
        SAM3_IMAGE_SIZE,
        SAM3_IMAGE_SIZE,
        dtype=torch.float32,
        device=device,
    )

    # Get a reference to the inner _Sam3ImageAdapter's backbone for direct calls.
    adapter = wrapper.model  # _Sam3ImageAdapter
    backbone = adapter.model.backbone  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Step 1: Measure per-image feature bytes.
    # ------------------------------------------------------------------
    print("\n[spike] === Step 1: per-image feature bytes ===")  # noqa: T201

    with torch.no_grad():
        raw_out = backbone.forward_image(images)

    byte_info = _measure_trunk_bytes(raw_out, batch_size=B)
    fpn_shapes: list[tuple[int, int, int]] = byte_info["fpn_shapes"]  # type: ignore[assignment]
    sam2_fpn_shapes: list[tuple[int, int, int]] = byte_info["sam2_fpn_shapes"]  # type: ignore[assignment]

    print(f"  backbone_fpn levels: {len(fpn_shapes)}")  # noqa: T201
    for i, (c, h, w) in enumerate(fpn_shapes):
        level_bytes = c * h * w * 2
        print(f"    level[{i}]: ({c}, {h}, {w}) -> {_fmt_bytes(level_bytes)}")  # noqa: T201
    print(f"  backbone_fpn total (per image, fp16): {_fmt_bytes(byte_info['fpn_bytes'])}")  # noqa: T201

    if sam2_fpn_shapes:
        print(f"  sam2_backbone_out levels: {len(sam2_fpn_shapes)}")  # noqa: T201
        for i, (c, h, w) in enumerate(sam2_fpn_shapes):
            level_bytes = c * h * w * 2
            print(f"    sam2 level[{i}]: ({c}, {h}, {w}) -> {_fmt_bytes(level_bytes)}")  # noqa: T201
        print(  # noqa: T201
            f"  sam2_backbone_out total (per image, fp16): "
            f"{_fmt_bytes(byte_info['sam2_fpn_bytes'])}"
        )
    else:
        print("  sam2_backbone_out: None (not present in this deployment)")  # noqa: T201

    print(  # noqa: T201
        f"\n  TOTAL per image (no sam2): {_fmt_bytes(byte_info['total_bytes_no_sam2'])}"
    )
    print(  # noqa: T201
        f"  TOTAL per image (w/ sam2): {_fmt_bytes(byte_info['total_bytes'])}"
    )

    # ------------------------------------------------------------------
    # Step 2a: Trunk-forward timing (repeated iters, warmup first).
    # ------------------------------------------------------------------
    print(f"\n[spike] === Step 2a: trunk forward_image timing (B={B}) ===")  # noqa: T201
    print(f"  warmup={args.warmup} iters, timed={args.iters} iters")  # noqa: T201

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            torch.cuda.synchronize()
            _ = backbone.forward_image(images)
            torch.cuda.synchronize()

    # Timed iters — reuse prof.bucket("spike.trunk_fwd") for accumulation.
    trunk_times: list[float] = []
    with torch.no_grad():
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with prof.bucket("spike.trunk_fwd"):
                _ = backbone.forward_image(images)
            torch.cuda.synchronize()
            trunk_times.append(time.perf_counter() - t0)

    trunk_mean = statistics.mean(trunk_times)
    trunk_median = statistics.median(trunk_times)
    print(f"  trunk_fwd mean  : {_fmt_ms(trunk_mean)}")  # noqa: T201
    print(f"  trunk_fwd median: {_fmt_ms(trunk_median)}")  # noqa: T201

    # ------------------------------------------------------------------
    # Step 2b: H2D copy timing — one pinned image entry to GPU.
    # ------------------------------------------------------------------
    print("\n[spike] === Step 2b: H2D copy timing (1 pinned image -> GPU) ===")  # noqa: T201

    # Build the pinned cache entry from the last forward pass output.
    with torch.no_grad():
        last_out = backbone.forward_image(images)
    pinned_entries = _pin_copy_tensors(last_out, batch_size=B)
    pinned_one = pinned_entries[0]

    # Warmup H2D
    for _ in range(args.warmup):
        _h2d_copy_one(pinned_one, device)

    # Timed H2D iters
    h2d_times: list[float] = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with prof.bucket("spike.h2d_copy"):
            _h2d_copy_one(pinned_one, device)
        h2d_times.append(time.perf_counter() - t0)

    h2d_mean = statistics.mean(h2d_times)
    h2d_median = statistics.median(h2d_times)
    print(f"  h2d_copy mean  : {_fmt_ms(h2d_mean)}")  # noqa: T201
    print(f"  h2d_copy median: {_fmt_ms(h2d_median)}")  # noqa: T201

    # ------------------------------------------------------------------
    # Step 2c: Full train_step timing (for trunk-as-fraction-of-step).
    # NOTE: We run a MOCK train_step (forward only, no loss/backward/optim)
    # because the real train_step needs a full DataConfig + batch dict with
    # instances.  The trunk fraction is measured as trunk_fwd / wrapper_fwd.
    # ------------------------------------------------------------------
    print(f"\n[spike] === Step 2c: full wrapper.forward timing (B={B}, mock classes) ===")  # noqa: T201

    from custom_sam_peft.data.base import TextPrompts

    prompts = [TextPrompts(classes=["__spike_class__"])] * B

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            torch.cuda.synchronize()
            _ = wrapper(images, prompts)
            torch.cuda.synchronize()

    fwd_times: list[float] = []
    with torch.no_grad():
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with prof.bucket("spike.wrapper_fwd"):
                _ = wrapper(images, prompts)
            torch.cuda.synchronize()
            fwd_times.append(time.perf_counter() - t0)

    fwd_mean = statistics.mean(fwd_times)
    fwd_median = statistics.median(fwd_times)
    print(f"  wrapper_fwd mean  : {_fmt_ms(fwd_mean)}")  # noqa: T201
    print(f"  wrapper_fwd median: {_fmt_ms(fwd_median)}")  # noqa: T201

    trunk_fraction_mean = trunk_mean / fwd_mean if fwd_mean > 0 else float("nan")
    trunk_fraction_median = trunk_median / fwd_median if fwd_median > 0 else float("nan")
    print(f"\n  trunk_fwd / wrapper_fwd (mean)  : {trunk_fraction_mean:.1%}")  # noqa: T201
    print(f"  trunk_fwd / wrapper_fwd (median): {trunk_fraction_median:.1%}")  # noqa: T201

    # ------------------------------------------------------------------
    # Net win summary.
    # ------------------------------------------------------------------
    net_win_mean = trunk_mean - h2d_mean
    net_win_median = trunk_median - h2d_median
    go_nogo_mean = "GO" if net_win_mean > 0 else "NO-GO"
    go_nogo_median = "GO" if net_win_median > 0 else "NO-GO"

    print("\n[spike] === Net-win estimate (per replayed step) ===")  # noqa: T201
    print("  net_win = trunk_fwd - h2d_copy")  # noqa: T201
    print(f"  mean  : {_fmt_ms(net_win_mean)}  -> {go_nogo_mean}")  # noqa: T201
    print(f"  median: {_fmt_ms(net_win_median)}  -> {go_nogo_median}")  # noqa: T201
    print(  # noqa: T201
        "\n  NOTE: this is the per-step saving ON REPLAYED STEPS (epochs 1+).\n"
        "  Total saving = net_win_per_step * steps_per_epoch * (epochs - 1)."
    )

    # ------------------------------------------------------------------
    # Step 3: Break-even table.
    # ------------------------------------------------------------------
    print(  # noqa: T201
        breakeven_table(
            byte_info["total_bytes_no_sam2"],
            byte_info["total_bytes"],
            args.dataset_sizes,
        )
    )

    # ------------------------------------------------------------------
    # Snapshot dump.
    # ------------------------------------------------------------------
    prof.note(
        spike_trunk_fwd_mean_s=trunk_mean,
        spike_trunk_fwd_median_s=trunk_median,
        spike_h2d_copy_mean_s=h2d_mean,
        spike_h2d_copy_median_s=h2d_median,
        spike_wrapper_fwd_mean_s=fwd_mean,
        spike_wrapper_fwd_median_s=fwd_median,
        spike_net_win_mean_s=net_win_mean,
        spike_fpn_bytes_per_image=byte_info["fpn_bytes"],
        spike_sam2_fpn_bytes_per_image=byte_info["sam2_fpn_bytes"],
        spike_total_bytes_no_sam2=byte_info["total_bytes_no_sam2"],
        spike_total_bytes_with_sam2=byte_info["total_bytes"],
        spike_batch_size=B,
        spike_dtype=args.dtype,
        spike_iters=args.iters,
        spike_warmup=args.warmup,
    )

    if args.snapshot_out is not None:
        out_path = prof.dump(args.snapshot_out)
        print(f"\n[spike] Snapshot written to {out_path}")  # noqa: T201
    else:
        print(  # noqa: T201
            "\n[spike] Tip: pass --snapshot-out /tmp/spike_snapshot.json to dump "
            "a CSP_PROFILE snapshot for post-hoc inspection with "
            "scripts/attribute_profile.py."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
