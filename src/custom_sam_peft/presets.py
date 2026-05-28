"""Algorithmic VRAM-tier PEFT preset chooser.

Analytic memory model + optional calibration cache. Public surface is
`decide_preset()` + `PresetDecision`.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from custom_sam_peft.config.schema import ModelConfig
from custom_sam_peft.peft_adapters import method_pretty_name

_LOG = logging.getLogger(__name__)

_GB = 1024**3
_MIB = 1024**2

_CUDA_HINT = (
    "decide_preset() requires CUDA; got cpu-only torch. "
    "In Colab: Runtime → Change runtime type → GPU. "
    "On RunPod: deploy a GPU pod."
)

# === CONSTANTS — see spec §3 ================================================
# These ride with the SAM 3.1 checkpoint identity. If Meta ships a new
# checkpoint, re-derive via scripts/_derive_preset_constants.py and update.

MODEL_PARAMS = 5_000_000_000  # SAM 3.1 base parameter count — analytic seed for full model stack
# (vision encoder ~762M + text encoder ~302M + decoder/neck ~50M,
#  plus retained activations and optimizer state heuristic; superseded
#  by calibration cache. Re-derive via scripts/_derive_preset_constants.py)
LORA_LAYERS = 96  # vision_decoder scope, count of nn.Linear LoRA targets (from _resolve_targets)
D_IN = 768  # avg input feature dim across LoRA targets
D_OUT = 768  # avg output feature dim across LoRA targets
Q_OVERHEAD = 64 * _MIB  # bnb NF4 per-block scale + zero-point overhead
WORKSPACE_BYTES = 256 * _MIB  # cuDNN workspace + autograd graph + tmp buffers (spec §3)
BASE_ACTIVATION_AT_1024 = int(1.5 * _GB)  # seed; superseded by calibration cache

# Forward-only memory is roughly 1/4 of the train-step probe (train captures
# forward + backward + retained graph; eval captures only forward, no graph).
# Spec §8.
forward_only_factor: float = 0.25

# === CALIBRATION CACHE =====================================================

CACHE_FILENAME = ".custom_sam_peft_calibration.json"
CACHE_SCHEMA_VERSION = 2


# === PresetDecision ========================================================


@dataclass(frozen=True)
class PresetDecision:
    """The chosen preset plus all the context needed to render it.

    Fields after `dtype` are diagnostic — the bundler renders them into
    `## Preset`, and `label()` flattens the whole thing onto one line.

    Spec: design §7.
    """

    method: Literal["lora", "qlora"]
    r: int
    batch_size: int
    grad_accum_steps: int
    dtype: Literal["bfloat16", "float16"]
    headroom_bytes: int
    predicted_bytes: int
    budget_bytes: int
    gpu_name: str
    provenance: Literal["calibrated", "analytic"]
    cache_path: Path | None
    calibrated_at: str | None  # ISO 8601 string when provenance == "calibrated", None otherwise

    @property
    def config_patch(self) -> dict[str, dict[str, object]]:
        """The 3-section dict the deep-merge consumer expects."""
        return {
            "model": {"dtype": self.dtype},
            "peft": {"method": self.method, "r": self.r},
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
        }

    def label(self) -> str:
        method = method_pretty_name(self.method)
        used_gib = self.predicted_bytes / _GB
        total_gib = (self.budget_bytes + self.headroom_bytes) / _GB
        if self.provenance == "calibrated":
            date_str = self.calibrated_at[:10] if self.calibrated_at else "unknown"
            suffix = f"(calibrated {date_str})"
        else:
            suffix = "(analytic estimate)"
        dtype_token = "fp16" if self.dtype == "float16" else "bf16"
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"grad_accum={self.grad_accum_steps} {dtype_token} — "
            f"fits in {used_gib:.1f}/{total_gib:.1f} GiB on {self.gpu_name} {suffix}"
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["cache_path"] = None if self.cache_path is None else str(self.cache_path)
        # calibrated_at is str | None — JSON handles it directly
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> PresetDecision:
        d = json.loads(s)
        d["cache_path"] = None if d["cache_path"] is None else Path(d["cache_path"])
        # calibrated_at is str | None — pass through as-is
        # Drop fields that may exist in old sidecars but no longer in the dataclass.
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


# === Memory model ==========================================================


def _bytes_per_param_for_method(method: str) -> float:
    return 2.0 if method == "lora" else 0.5  # bf16/fp16 (2.0) vs NF4 (0.5)


def _model_bytes(method: str) -> int:
    base = int(MODEL_PARAMS * _bytes_per_param_for_method(method))
    return base + (Q_OVERHEAD if method == "qlora" else 0)


def _adapter_bytes(r: int) -> int:
    # LORA_LAYERS * r * (D_IN + D_OUT) * 2 bytes (bf16 adapter weights).
    return LORA_LAYERS * r * (D_IN + D_OUT) * 2


def _optimizer_bytes(r: int) -> int:
    # AdamW state on the bf16 adapter — fp32 m, fp32 v, fp32 master copy.
    # Adapter weights are 2 B/param; state is 8 B/param -> 4x adapter_bytes.
    return _adapter_bytes(r) * 4


def _activation_per_example(image_size: int, cache: dict[str, Any] | None) -> int:
    if cache is not None:
        return int(cache["activation_bytes_per_example"])
    return int(BASE_ACTIVATION_AT_1024 * (image_size / 1024) ** 2)


def _activation_bytes(image_size: int, batch: int, cache: dict[str, Any] | None) -> int:
    per = _activation_per_example(image_size, cache)
    return int(per * batch)


def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    image_size: int,
    cache: dict[str, Any] | None,
    mode: Literal["train", "eval"] = "train",
) -> int:
    if mode == "train":
        return (
            _model_bytes(method)
            + _adapter_bytes(r)
            + _optimizer_bytes(r)
            + _activation_bytes(image_size, batch, cache)
            + WORKSPACE_BYTES
        )
    # mode == "eval": no optimizer, no adapter bytes; activations x forward_only_factor.
    activations = int(_activation_bytes(image_size, batch, cache) * forward_only_factor)
    return _model_bytes(method) + activations + WORKSPACE_BYTES


# === Calibration cache I/O =================================================


def _current_sam3_checkpoint_sha() -> str:
    """Hash the configured SAM 3.1 checkpoint file. Public for monkeypatching."""
    cfg = ModelConfig()
    if cfg.local_dir is None:
        return ""
    ckpt = Path(cfg.local_dir) / cfg.checkpoint_file
    if not ckpt.is_file():
        return ""
    h = hashlib.sha256()
    with ckpt.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache(gpu_name: str) -> tuple[dict[str, Any] | None, Path | None]:
    """Return (cache_dict, absolute_cache_path) iff the cache matches."""
    cache_path = Path(CACHE_FILENAME).resolve()
    if not cache_path.is_file():
        return None, None
    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _LOG.warning("calibration cache unreadable (%s); falling through to analytic", exc)
        return None, None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        _LOG.warning(
            "calibration cache schema_version=%r != %d; ignoring",
            data.get("schema_version"),
            CACHE_SCHEMA_VERSION,
        )
        return None, None
    if (
        data.get("gpu_name") != gpu_name
        or data.get("sam3_checkpoint_sha") != _current_sam3_checkpoint_sha()
    ):
        return None, None
    return data, cache_path


# === Headroom + budget =====================================================


def _headroom_bytes() -> int:
    raw = os.environ.get("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB")
    if raw is None:
        return 1 * _GB
    try:
        gib = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB must be a non-negative float"
        ) from exc
    if gib < 0 or math.isnan(gib):
        raise RuntimeError("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB must be a non-negative float")
    return int(gib * _GB)


# === Search space =========================================================


def _candidates() -> list[tuple[str, int, int]]:
    methods = ("lora", "qlora")
    rs = (8, 16, 24, 32, 48, 64)
    batches = tuple(range(1, 17))
    return [(m, r, b) for m in methods for r in rs for b in batches]


def _sort_key(c: tuple[str, int, int]) -> tuple[int, int, int]:
    method, r, batch = c
    return (
        0 if method == "lora" else 1,
        -r,
        -batch,
    )


# === Public entry point ====================================================


def decide_preset() -> PresetDecision:
    """Pick the largest configuration that fits within the VRAM budget.

    Raises:
      RuntimeError: CUDA unavailable, env-var malformed, or no candidate fits.

    Spec: design §3 + §7.
    """
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    image_size = SAM3_IMAGE_SIZE
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    decided_dtype: Literal["bfloat16", "float16"] = "float16" if cc < (8, 0) else "bfloat16"

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, cache_path = _load_cache(gpu_name)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )
    calibrated_at: str | None = str(cache["calibrated_at"]) if cache is not None else None

    feasible = []
    for method, r, batch in _candidates():
        pb = _predicted_bytes(method, r, batch, image_size, cache)
        if pb <= budget:
            feasible.append((method, r, batch, pb))

    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1. Use a larger GPU."
        )

    feasible.sort(key=lambda t: _sort_key(t[:3]))
    method, r, batch, predicted = feasible[0]
    grad_accum = max(1, 16 // batch)

    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        dtype=decided_dtype,
        headroom_bytes=headroom,
        predicted_bytes=predicted,
        budget_bytes=budget,
        gpu_name=gpu_name,
        provenance=provenance,
        cache_path=cache_path,
        calibrated_at=calibrated_at,
    )


def decide_eval_batch_size(
    classes_per_forward: int = 16,
) -> tuple[int, int, Literal["calibrated", "analytic"]]:
    """Pick the largest forward-only batch size that fits within the eval VRAM budget.

    Returns (batch_size, predicted_bytes, provenance).

    On CPU: returns (1, 0, "analytic") and logs once.
    Spec: design §8.

    Note: ``classes_per_forward`` is accepted for API stability but does **not**
    currently affect the returned batch size.  K (classes per forward) is folded
    into ``forward_only_factor`` empirically (spec §8) rather than computed from
    this parameter.  It is reserved for a future K-aware tuning pass (spec §12
    follow-up).  Pass ``MULTIPLEX_CAP`` from ``custom_sam_peft.models.sam3`` so
    that callsites remain correct when K-awareness is wired in.
    """
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP as _CAP
    from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

    image_size = SAM3_IMAGE_SIZE
    # Guard against mis-use: classes_per_forward must be in [1, MULTIPLEX_CAP].
    # Import lazily to avoid a circular dependency at module load time.
    if not (1 <= classes_per_forward <= _CAP):
        raise ValueError(f"classes_per_forward must be in [1, {_CAP}]; got {classes_per_forward}")
    if not torch.cuda.is_available():
        _LOG.info("eval.batch_size=auto on CPU -> falling back to 1")
        return 1, 0, "analytic"

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, _ = _load_cache(gpu_name)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )

    best_bs = 1
    best_predicted = _predicted_bytes(
        "lora", r=4, batch=1, image_size=image_size, cache=cache, mode="eval"
    )
    for batch in range(1, 65):  # B in [1, 64]
        pb = _predicted_bytes(
            "lora", r=4, batch=batch, image_size=image_size, cache=cache, mode="eval"
        )
        if pb <= budget:
            best_bs = batch
            best_predicted = pb

    # Attention-memory ceiling: SDPA math fallback materialises the full B*H*N*N
    # score matrix in float32 (4 bytes) even when inputs are bf16.  At SAM 3.1's
    # patch_size=14, image_size=1008 -> N=5184 tokens, H=16 heads, the analytic
    # model can return bs~35 on a 24 GiB card, causing a 56 GiB allocation (OOM).
    # Constants: SAM 3.1 hiera-large vision backbone, from sam3/model_builder.py.
    _SAM3_PATCH = 14  # vision backbone patch size
    _SAM3_HEADS = 16  # vision backbone attention heads
    _n_tokens = (image_size // _SAM3_PATCH) ** 2
    # fp32 (4 bytes): worst case when SDPA math backend upcasts bf16 inputs.
    _attn_per_example = _SAM3_HEADS * _n_tokens * _n_tokens * 4
    # Model weights and forward activations are ALREADY resident when SDPA runs;
    # subtract them from the budget before solving for the attention-bound bs.
    attn_budget = budget - _model_bytes("lora") - WORKSPACE_BYTES
    _act_per_example = int(_activation_per_example(image_size, cache) * forward_only_factor)
    _per_example = _attn_per_example + _act_per_example
    attn_cap = max(1, attn_budget // _per_example) if attn_budget > 0 else 1
    if attn_cap < best_bs:
        _LOG.warning(
            "eval auto-batch capped %d -> %d by SDPA attention memory "
            "(H*N^2*fp32=%.1f GiB SDPA/image at image_size=%d; issue #162)",
            best_bs,
            attn_cap,
            _attn_per_example / _GB,
            image_size,
        )
        best_bs = attn_cap
        best_predicted = _predicted_bytes(
            "lora", r=4, batch=best_bs, image_size=image_size, cache=cache, mode="eval"
        )

    return best_bs, best_predicted, provenance
