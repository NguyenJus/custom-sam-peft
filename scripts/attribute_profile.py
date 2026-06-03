#!/usr/bin/env python3
"""Standalone attribution reader for CSP_PROFILE snapshots (issue #256 / #273).

Reads one or two snapshot JSON files (produced by ``CSP_PROFILE=1`` + ``csp profile``
or ``profiling.dump()``) and prints / writes a Markdown attribution report.

Usage
-----
Single snapshot::

    python scripts/attribute_profile.py snapshot.json

With baseline comparison::

    python scripts/attribute_profile.py snapshot.json --baseline baseline.json

Write report to file::

    python scripts/attribute_profile.py snapshot.json --out report.md

Options::

    --baseline PATH     Baseline snapshot for regression detection.
    --out PATH          Write Markdown report to PATH instead of stdout.
    --rle-threshold F   Share threshold for RLE GO verdict (default: 0.05).
    --share-tolerance F Regression share-delta threshold (default: 0.05).

This script is intentionally **not** wired into the ``csp`` CLI so the CLI
surface stays unchanged (per spec §9).  It imports only from
``custom_sam_peft.profiling_report`` (pure Python, no GPU) and the stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_snap(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.exit(f"error: cannot read snapshot {path}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attribute_profile",
        description="Turn a CSP_PROFILE JSON snapshot into a ranked attribution report.",
    )
    parser.add_argument("snapshot", type=Path, help="Path to profile_snapshot.json")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help="Baseline snapshot for regression detection.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write Markdown report to PATH (default: print to stdout).",
    )
    parser.add_argument(
        "--rle-threshold",
        type=float,
        default=0.05,
        metavar="F",
        help="RLE share threshold for GO verdict (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--share-tolerance",
        type=float,
        default=0.05,
        metavar="F",
        help="Regression share-delta threshold (default: 0.05 = 5pp).",
    )
    args = parser.parse_args(argv)

    # Late import — keeps the script importable even without the package on sys.path
    # when the user runs it from the repo root via ``uv run``.
    try:
        from custom_sam_peft.profiling_report import (
            attribute_snapshot,
            compare_snapshots,
            render_report,
        )
    except ImportError as exc:
        sys.exit(
            f"error: cannot import custom_sam_peft.profiling_report: {exc}\n"
            "Run via: uv run python scripts/attribute_profile.py ..."
        )

    snap = _load_snap(args.snapshot)
    data = attribute_snapshot(snap, rle_threshold=args.rle_threshold)

    regression_flags = None
    if args.baseline is not None:
        baseline_snap = _load_snap(args.baseline)
        regression_flags = compare_snapshots(
            baseline_snap,
            snap,
            share_tolerance=args.share_tolerance,
        )

    report = render_report(data, regression_flags=regression_flags)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)  # noqa: T201
    else:
        print(report)  # noqa: T201

    return 0


if __name__ == "__main__":
    sys.exit(main())
