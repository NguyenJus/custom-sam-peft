"""Re-derive presets.py constants from probes / real-model inspection on the local GPU.

Maintainer-only. Run on the 16 GB dev card.

--- Split activation seeds (original purpose) ---

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

--- Per-scope adapter dimension sums (§3 extension) ---

Also prints a copy-paste-ready ADAPTER_DIM_SUM_BY_SCOPE = {...} block for presets.py
by calling apply_lora on the real model (via load_sam31) for each LoraScope value and
summing lora_A.in_features + lora_B.out_features over every injected peft LoRA adapter.
This is exact by construction and captures peft's real MHA treatment (both in_proj and
out_proj) with zero hand-derivation. The --scope-survey flag enables this path.

Why NOT a live count at estimate time (do not re-attempt):
  - The vendored ViT __init__ calls .item() on a meta tensor:
    sam3/model/vitdet.py:878: `dpr = [x.item() for x in torch.linspace(...)]`
    -> "Tensor.item() cannot be called on meta tensors".
  - Complex-RoPE initialisation is a further likely blocker on meta.
  - A real CPU build of the ~5B-param model does not fit the 16 GB box.
Hence the offline derivation via the maintainer GPU script is the chosen path.

Not imported by the package or the test suite. Spec §2.1/§3/§6.
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


def _survey_adapter_dim_sums() -> None:
    """For each LoraScope value, call apply_lora on the real model and sum
    lora_A.in_features + lora_B.out_features over every injected peft LoRA
    adapter. Prints a copy-paste-ready ADAPTER_DIM_SUM_BY_SCOPE = {...} block
    for presets.py.

    Requires a GPU and the real SAM 3.1 checkpoint. Do NOT call at import time
    or from the package — this is a maintainer-only offline derivation.

    Why NOT a live meta-device count: see module docstring.
    """
    import gc
    import typing

    from custom_sam_peft.config.schema import LoraScope
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE as _  # noqa: F401 (import check)
    from custom_sam_peft.models.sam3 import load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    scope_values: tuple[str, ...] = typing.get_args(LoraScope)

    results: dict[str, int] = {}
    for scope_val in scope_values:
        print(f"\n[scope-survey] probing scope={scope_val!r} ...")  # noqa: T201
        wrapper = load_sam31(ModelConfig(), channels=3, channel_semantics="rgb")
        apply_lora(wrapper, PEFTConfig(method="lora", scope=scope_val))  # type: ignore[arg-type]
        dim_sum = 0
        for _name, mod in wrapper.named_modules():
            # peft (0.19.x) stores lora_A / lora_B as nn.ModuleDict keyed by adapter
            # name, each value an inner nn.Linear (lora_A: in_features->r;
            # lora_B: r->out_features). Iterate the dict values to reach them.
            lora_a = getattr(mod, "lora_A", None)
            lora_b = getattr(mod, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue
            for child_a, child_b in zip(lora_a.values(), lora_b.values(), strict=False):
                if hasattr(child_a, "in_features") and hasattr(child_b, "out_features"):
                    dim_sum += child_a.in_features + child_b.out_features
        results[scope_val] = dim_sum
        print(f"[scope-survey]   dim_sum={dim_sum:,}")  # noqa: T201
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    today = __import__("datetime").date.today().isoformat()
    print("\n# --- copy-paste block for presets.py ADAPTER_DIM_SUM_BY_SCOPE ---")  # noqa: T201
    print(f"# derived: _derive_preset_constants.py {today}")  # noqa: T201
    print("ADAPTER_DIM_SUM_BY_SCOPE: dict[str, int] = {")  # noqa: T201
    for scope_val in scope_values:
        v = results[scope_val]
        print(f'    "{scope_val}": {v:_},  # derived: _derive_preset_constants.py {today}')  # noqa: T201
    print("}")  # noqa: T201
    print("# --- end copy-paste block ---")  # noqa: T201


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive presets.py constants from GPU probes / real-model inspection."
    )
    ap.add_argument("--r", type=int, default=16)
    # --k is ignored: the two-point split always probes K=1 and K=4.
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--method", choices=["lora", "qlora"], default="lora")
    ap.add_argument(
        "--scope",
        default="decoder_concept",
        help=(
            "LoraScope value used when computing the static overhead term for the "
            "split-activation derivation. Must match the scope the probe was "
            "trained with. Spec §3."
        ),
    )
    ap.add_argument(
        "--scope-survey",
        action="store_true",
        default=False,
        help=(
            "For each LoraScope value, call apply_lora on the real model and sum "
            "lora_A.in_features + lora_B.out_features. Prints a copy-paste-ready "
            "ADAPTER_DIM_SUM_BY_SCOPE block for presets.py. Requires GPU + real "
            "SAM 3.1 checkpoint. Spec §3."
        ),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    if args.scope_survey:
        _survey_adapter_dim_sums()
        return

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
        + _adapter_bytes(args.r, args.scope)
        + _optimizer_bytes(args.r, args.scope)
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
    print(f"scope:             {args.scope}")  # noqa: T201
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
