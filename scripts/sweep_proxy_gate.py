#!/usr/bin/env python
"""Checkpoint-sweep proxy-vs-exact gate for the dense-IoU AP proxy (#277).

Sweeps banked checkpoints through lite eval twice each (exact via
``CSP_LITE_EXACT_MAP=1`` and proxy with env unset) on the SAME lite val
subset, then evaluates the §8.2 pre-enablement gate:

  - Spearman rho >= rho_threshold (default 0.95; cite: spike §8.2 step 3)
  - No min_delta-relevant adjacent inversion
  - min_delta scale check: reports frac_exact, frac_proxy, ratio

OUT OF SCOPE for this script: launching GPU runs, re-profiling speedup, or
executing the sweep against real checkpoints without explicit user go-ahead
(Phase-0-gated; see the spec).  The script is the harness; the user kicks it
off manually after Phase 0 completes and checkpoints are banked.

Usage
-----
    python scripts/sweep_proxy_gate.py \\
        --run-dir runs/<run_id> \\
        --config config/my_run.yaml \\
        --split val \\
        --lite-max-images 64 \\
        --output sweep_gate_result.json

    # Or point directly at the checkpoints directory:
    python scripts/sweep_proxy_gate.py \\
        --checkpoints runs/<run_id>/checkpoints \\
        --config config/my_run.yaml \\
        --output sweep_gate_result.json

The script prints a PASS/FAIL summary to stdout and writes a JSON result file
when --output is given.

CSP_PROFILE=1 is compatible with this script: the existing profiling harness
instruments the eval passes and its output is preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep banked checkpoints through lite eval and evaluate the §8.2 gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run-dir",
        type=Path,
        metavar="DIR",
        help="Run output directory (contains checkpoints/ subdirectory).",
    )
    src.add_argument(
        "--checkpoints",
        type=Path,
        metavar="DIR",
        help="Direct path to the checkpoints/ directory.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="YAML",
        help="Training config YAML used for the reference run.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate on (default: val).",
    )
    parser.add_argument(
        "--lite-max-images",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override eval.lite_max_images for the sweep. "
            "Must be the same value for BOTH passes to guarantee the same subset. "
            "Default: use the value from the config."
        ),
    )
    parser.add_argument(
        "--rho-threshold",
        type=float,
        default=0.95,
        metavar="RHO",
        help="Spearman rho threshold for PASS (default: 0.95; cite: spike §8.2 step 3).",
    )
    parser.add_argument(
        "--scale-ratio-threshold",
        type=float,
        default=2.0,  # tbd: harvest-time tunable; not derived analytically
        metavar="RATIO",
        help=(
            "Flag material_divergence when frac_proxy/frac_exact exceeds this "
            "value (default: 2.0; tbd: tune at harvest time)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="JSON",
        help="Write the full GateResult as JSON to this path.",
    )
    return parser


def _load_cfg(config_path: Path, lite_max_images: int | None):
    """Load TrainConfig from YAML, optionally overriding lite_max_images."""
    import yaml

    from custom_sam_peft.config.schema import TrainConfig

    raw = yaml.safe_load(config_path.read_text())
    cfg = TrainConfig.model_validate(raw)
    if lite_max_images is not None:
        cfg = cfg.model_copy(
            update={"eval": cfg.eval.model_copy(update={"lite_max_images": lite_max_images})}
        )
    return cfg


def _make_eval_fn(cfg, split: str):
    """Return an eval_fn(checkpoint, exact) -> (mAP, mAP_50 | None).

    Imports run_eval lazily so the module is importable without GPU.
    """
    from custom_sam_peft.eval.runner import run_eval

    # Force lite mode for the sweep regardless of the run's eval.mode setting.
    sweep_cfg = cfg.model_copy(
        update={"eval": cfg.eval.model_copy(update={"mode": "lite", "visualize": False})}
    )

    def eval_fn(checkpoint: Path, exact: bool) -> tuple[float, float | None]:
        # CSP_LITE_EXACT_MAP is already set/unset by run_sweep before calling us.
        report = run_eval(sweep_cfg, checkpoint=checkpoint, split=split)
        mAP = report.overall.get("mAP", 0.0)
        mAP_50 = report.overall.get("mAP_50")
        return (mAP, mAP_50)

    return eval_fn


def _gate_result_to_dict(result) -> dict:
    """Serialize GateResult to a JSON-compatible dict."""

    def _record_dict(r) -> dict:
        d = {
            "checkpoint": str(r.checkpoint),
            "step": r.step,
            "exact_map": r.exact_map,
            "proxy_map": r.proxy_map,
        }
        if r.exact_map_50 is not None:
            d["exact_map_50"] = r.exact_map_50
        if r.proxy_map_50 is not None:
            d["proxy_map_50"] = r.proxy_map_50
        return d

    def _maybe_float(v: float) -> object:
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v

    sc = result.scale
    scale_dict = None
    if sc is not None:
        scale_dict = {
            "min_delta": sc.min_delta,
            "exact_range": sc.exact_range,
            "proxy_range": sc.proxy_range,
            "frac_exact": _maybe_float(sc.frac_exact),
            "frac_proxy": _maybe_float(sc.frac_proxy),
            "ratio": _maybe_float(sc.ratio),
            "material_divergence": sc.material_divergence,
            "scale_ratio_threshold": sc.scale_ratio_threshold,
        }

    return {
        "passed": result.passed,
        "n_total": len(result.all_records),
        "n_gated": result.n_gated,
        "rho": _maybe_float(result.rho),
        "rho_threshold": result.rho_threshold,
        "rho_passed": result.rho_passed,
        "n_inversions": result.n_inversions,
        "inversions_passed": result.inversions_passed,
        "scale": scale_dict,
        "gated_records": [_record_dict(r) for r in result.gated_records],
        "all_records": [_record_dict(r) for r in result.all_records],
    }


def _print_summary(result, min_delta: float) -> None:
    """Print a human-readable PASS/FAIL summary to stdout."""
    verdict = "PASS" if result.passed else "FAIL"
    print(f"\n=== Proxy-vs-Exact Gate: {verdict} ===")  # noqa: T201
    print(f"  Checkpoints swept : {len(result.all_records)}")  # noqa: T201
    print(f"  Non-zero-exact    : {result.n_gated} (gate domain)")  # noqa: T201

    rho_str = f"{result.rho:.4f}" if math.isfinite(result.rho) else str(result.rho)
    rho_pass = "PASS" if result.rho_passed else "FAIL"
    print(  # noqa: T201
        f"  Spearman rho      : {rho_str} >= {result.rho_threshold} → {rho_pass}"
    )

    inv_pass = "PASS" if result.inversions_passed else "FAIL"
    print(  # noqa: T201
        f"  Inversions        : {result.n_inversions} (min_delta={min_delta}) → {inv_pass}"
    )

    sc = result.scale
    if sc is not None:

        def _fmt(v: float) -> str:
            return f"{v:.6f}" if math.isfinite(v) else str(v)

        print(  # noqa: T201
            f"  Scale check       : frac_exact={_fmt(sc.frac_exact)}"
            f"  frac_proxy={_fmt(sc.frac_proxy)}"
            f"  ratio={_fmt(sc.ratio)}"
            f"  material_divergence={sc.material_divergence}"
        )
        if sc.material_divergence:
            print(  # noqa: T201
                "  *** RECOMMENDATION: ratio exceeds threshold "
                f"({sc.ratio:.2f} > {sc.scale_ratio_threshold}). "
                "Consider recalibrating min_delta before trusting the proxy "
                "as a control input (spike §7b). Do NOT ship a silently "
                "rescaled early-stop sensitivity."
            )

    if result.n_inversions > 0:
        print("  Inversion details:")  # noqa: T201
        for a, b in result.inversion_pairs:
            print(  # noqa: T201
                f"    {a.checkpoint.name}: exact={a.exact_map:.4f} proxy={a.proxy_map:.4f}"
                f" → {b.checkpoint.name}: exact={b.exact_map:.4f} proxy={b.proxy_map:.4f}"
                f" (exact_delta={b.exact_map - a.exact_map:.4f})"
            )
    print()  # noqa: T201


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve checkpoints directory.
    checkpoints_dir = args.run_dir / "checkpoints" if args.run_dir is not None else args.checkpoints

    print(f"Loading config: {args.config}")  # noqa: T201
    cfg = _load_cfg(args.config, args.lite_max_images)
    min_delta = float(cfg.train.early_stop.min_delta)

    print(f"Checkpoints dir: {checkpoints_dir}")  # noqa: T201

    from custom_sam_peft.eval.proxy_gate import evaluate_gate
    from custom_sam_peft.eval.sweep import discover_checkpoints, run_sweep

    checkpoints = discover_checkpoints(checkpoints_dir)
    print(f"Found {len(checkpoints)} checkpoint(s).")  # noqa: T201

    if not checkpoints:
        print("ERROR: no step_*.pt checkpoint directories found.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    eval_fn = _make_eval_fn(cfg, split=args.split)

    print(  # noqa: T201
        f"Sweeping {len(checkpoints)} checkpoints x 2 passes "
        f"(lite_max_images={cfg.eval.lite_max_images}, split={args.split}) ..."
    )
    records = run_sweep(checkpoints_dir, eval_fn)

    result = evaluate_gate(
        records,
        min_delta=min_delta,
        rho_threshold=args.rho_threshold,
        scale_ratio_threshold=args.scale_ratio_threshold,
    )

    _print_summary(result, min_delta=min_delta)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_gate_result_to_dict(result), indent=2))
        print(f"Result written to: {args.output}")  # noqa: T201

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
