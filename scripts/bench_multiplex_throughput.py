#!/usr/bin/env python
"""Benchmark K=1 vs K=16 multiplex training throughput. NOT in CI.

Usage:
    python scripts/bench_multiplex_throughput.py [--image-size 1008] [--batch 4]
        [--n-classes 80] [--n-steps 5] [--device cuda]

Spec: docs/superpowers/specs/2026-05-23-multiplex-forward-design.md §9.
"""

from __future__ import annotations

import argparse
import time

import torch


def _build_inputs(batch: int, n_classes: int, image_size: int, device, dtype):
    """Build synthetic images + class names + per-image TextPrompts."""
    images = torch.zeros(batch, 3, image_size, image_size, dtype=dtype, device=device)
    class_names = [f"class_{i}" for i in range(n_classes)]
    return images, class_names


def _bench_one(
    wrapper,
    images,
    class_names: list[str],
    classes_per_forward: int,
    n_steps: int,
    TextPrompts,
) -> float:
    """Return mean wall-clock seconds per micro-step.

    A "step" simulates one training/eval iteration over all class_names: when
    classes_per_forward=1 that is len(class_names) forwards per step; when
    classes_per_forward=16 it is ceil(len/16) forwards per step.

    ``TextPrompts`` is passed explicitly to avoid a closure/import-order issue.
    """
    # Chunk class_names into groups of size classes_per_forward.
    groups = [
        class_names[i : i + classes_per_forward]
        for i in range(0, len(class_names), classes_per_forward)
    ]

    batch = images.shape[0]

    def _run_groups():
        for group in groups:
            prompts = [TextPrompts(classes=list(group)) for _ in range(batch)]
            _ = wrapper(images, prompts)

    # Warm up once (don't count in timing).
    with torch.no_grad():
        _run_groups()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time n_steps real steps.
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            _run_groups()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark K=1 vs K=16 multiplex throughput.")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-classes", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    from custom_sam_peft.config.schema import ModelConfig
    from custom_sam_peft.data.base import TextPrompts
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, load_sam31

    cfg = ModelConfig(
        device=args.device,
        gradient_checkpointing=False,
        dtype=args.dtype,
    )
    wrapper = load_sam31(cfg)
    wrapper.eval()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    images, class_names = _build_inputs(args.batch, args.n_classes, args.image_size, device, dtype)

    sec_k1 = _bench_one(wrapper, images, class_names, 1, args.n_steps, TextPrompts)
    sec_k16 = _bench_one(wrapper, images, class_names, MULTIPLEX_CAP, args.n_steps, TextPrompts)

    print(  # noqa: T201 — intentional: this script's only output mechanism
        f"K=1:  {sec_k1:.3f} s/step  ({args.batch / sec_k1:.2f} img/s)\n"
        f"K=16: {sec_k16:.3f} s/step ({args.batch / sec_k16:.2f} img/s)\n"
        f"speedup: {sec_k1 / sec_k16:.1f}x"
    )


if __name__ == "__main__":
    main()
