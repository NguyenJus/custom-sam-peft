#!/usr/bin/env python
"""TEMPORARY standalone eval profiler driver (issue #250, Phase 1).

Runs ONE representative eval with CSP_EVAL_PROFILE=1 and prints the per-bucket
breakdown + metadata for the attribution report. Removed in Phase 2.

Usage (on the RTX 5070 Ti, in its own process for GPU memory isolation):
    CSP_EVAL_PROFILE=1 uv run python scripts/profile_eval_250.py \
        --config <eval.yaml> [--checkpoint <ckpt>]

The --config must point at a real eval config (full mode, real val split, real
image size) — a representative run, not a stub.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    if os.environ.get("CSP_EVAL_PROFILE", "0") in ("", "0"):
        raise SystemExit("set CSP_EVAL_PROFILE=1 before running this profiler")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    args = ap.parse_args()

    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.eval import _profile
    from custom_sam_peft.eval.runner import run_eval

    _profile.reset()
    cfg = load_config(args.config)
    run_eval(cfg, checkpoint=args.checkpoint, split=args.split)

    buckets, meta = _profile.snapshot()
    total = sum(buckets.values()) or 1.0
    print("\n=== issue #250 eval profile ===")  # noqa: T201
    print(f"metadata: {meta}")  # noqa: T201
    print(f"{'bucket':<22}{'seconds':>12}{'% of timed':>14}")  # noqa: T201
    order = [
        "forward",
        "mask_upsample",
        "transfer_binarize",
        "rle_encode",
        "coco_aggregate",
    ]
    for name in order:
        s = buckets.get(name, 0.0)
        print(f"{name:<22}{s:>12.4f}{100 * s / total:>13.1f}%")  # noqa: T201
    print(f"{'TOTAL(timed)':<22}{total:>12.4f}{100.0:>13.1f}%")  # noqa: T201


if __name__ == "__main__":
    main()
