"""Re-derive presets.py split activation seeds from probes on the local GPU.

Maintainer-only. Run on the 16 GB dev card:

    uv run python scripts/_derive_preset_constants.py --method qlora --r 4 --batch 1

Runs two cheap probes (K=1, K=4) and prints the two-point split:
    A_per_class = (peak_K4 - peak_K1) / (4 - 1)
    A_fixed     = clamp(peak_K1 - overhead - A_per_class, min=0)
where overhead = STATIC + (attn(1) if not flash else 0) is REGIME-MATCHED
(Amendment 2): STATIC = model + adapter + optimizer + workspace, flash =
_flash_attention_available(cc) from the live card, attn(1) =
_attention_bytes_per_example(1008). This is the SAME overhead the predictor adds, so
the printed seeds reproduce the measured peak and are a portable flash-baseline (on a
cc>=8.0 card flash=True -> subtract STATIC only). Measured natively at SAM 3.1's fixed
1008px (no image-size scale term). A clamped A_FIXED=0 is the expected dev-GPU result.
Prints copy-paste-ready `A_FIXED = ...` / `A_PER_CLASS = ...` lines for presets.py.
Not imported by the package or the test suite. Spec §2.1/§6.
"""

from __future__ import annotations

import argparse

import torch

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.presets import (
    WORKSPACE_BYTES,
    _adapter_bytes,
    _attention_bytes_per_example,
    _flash_attention_available,
    _model_bytes,
    _optimizer_bytes,
)

_GB = 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=16)
    # --k is ignored: the two-point split always probes K=1 and K=4.
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--method", choices=["lora", "qlora"], default="lora")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE, load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    image_size = SAM3_IMAGE_SIZE

    def _probe_peak(k_eff: int) -> int:
        wrapper = load_sam31(ModelConfig(), channels=3, channel_semantics="rgb")
        apply_lora(wrapper, PEFTConfig(method=args.method, r=args.r))
        device = next(wrapper.parameters()).device
        images = torch.zeros(
            args.batch, 3, image_size, image_size, dtype=torch.bfloat16, device=device
        )
        prompts = [
            TextPrompts(classes=[f"class_{j}" for j in range(k_eff)]) for _ in range(args.batch)
        ]
        torch.cuda.reset_peak_memory_stats()
        out = wrapper(images, prompts, support=None)
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for t in out.values():
            if isinstance(t, torch.Tensor):
                loss = loss + t.float().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return int(torch.cuda.max_memory_allocated())

    peak_k1 = _probe_peak(min(1, MULTIPLEX_CAP))
    peak_k4 = _probe_peak(min(4, MULTIPLEX_CAP))

    # Regime-matched overhead (Amendment 2 / spec §2.1): STATIC + conditional
    # materialized attention, the SAME quantity the predictor adds for this card. On
    # a cc>=8.0 (flash) card the attention is folded into the empirical split -> no
    # term; on cc<8.0 the math-backend score matrix is subtracted off so the seeds
    # normalize to the portable flash-baseline. Inverting it makes the printed seeds
    # reproduce the measured peak.
    cc = torch.cuda.get_device_capability(0)
    flash = _flash_attention_available(cc)
    static = (
        _model_bytes(args.method)
        + _adapter_bytes(args.r)
        + _optimizer_bytes(args.r)
        + WORKSPACE_BYTES
    )
    overhead = static + (0 if flash else _attention_bytes_per_example(image_size))
    a_per_class = int((peak_k4 - peak_k1) / (4 - 1))
    a_fixed = int(peak_k1 - overhead - a_per_class)

    if args.batch != 1:
        # Split is per-image; normalize the activation by batch before clamping.
        # attn() is also per-image, so divide the (per-image) attention into overhead.
        per_img_overhead = static / args.batch + (
            0 if flash else _attention_bytes_per_example(image_size)
        )
        a_per_class = int(a_per_class / args.batch)
        a_fixed = int((peak_k1) / args.batch - per_img_overhead - a_per_class)

    # Clamp A_FIXED to >=0. A negative residual (encoder activation below the
    # model-weight conservatism margin in STATIC) clamps to 0 — the expected,
    # cited dev-GPU outcome (spec §2.1/§6), not an error.
    clamped = a_fixed < 0
    a_fixed = max(0, a_fixed)

    print(f"peak K=1:          {peak_k1 / _GB:.2f} GiB")  # noqa: T201
    print(f"peak K=4:          {peak_k4 / _GB:.2f} GiB")  # noqa: T201
    print(f"flash regime:      {flash} (cc={cc})")  # noqa: T201
    print(f"overhead (subtr.): {overhead / _GB:.2f} GiB")  # noqa: T201
    if clamped:
        print(  # noqa: T201
            "note: A_FIXED residual was negative -> clamped to 0 (encoder activation "
            "below STATIC model-weight margin; expected, spec §2.1)"
        )
    print(  # noqa: T201
        f"A_FIXED = {a_fixed}  # {a_fixed / _GB:.3f} GiB (encoder, per image @1008px)"
    )
    print(  # noqa: T201
        f"A_PER_CLASS = {a_per_class}  # {a_per_class / _GB:.3f} GiB (decoder, per class @1008px)"
    )


if __name__ == "__main__":
    main()
