"""Algorithmic VRAM-tier PEFT preset chooser.

Analytic memory model + optional calibration cache. Public surface is
`decide_preset()` + `PresetDecision`.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md.
"""

from __future__ import annotations

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
CKPT_FACTOR = (
    0.3  # activation reduction with gradient_checkpointing on (spec §3, ~sqrt(num_layers))
)
BASE_ACTIVATION_AT_1024 = int(1.5 * _GB)  # seed; superseded by calibration cache

# === CALIBRATION CACHE =====================================================

CACHE_FILENAME = ".custom_sam_peft_calibration.json"
CACHE_SCHEMA_VERSION = 1


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
    gradient_checkpointing: bool
    dtype: Literal["bfloat16"]
    headroom_bytes: int
    predicted_bytes: int
    budget_bytes: int
    image_size: int
    gpu_name: str
    provenance: Literal["calibrated", "analytic"]
    cache_path: Path | None
    calibrated_at: str | None  # ISO 8601 string when provenance == "calibrated", None otherwise

    @property
    def config_patch(self) -> dict[str, dict[str, object]]:
        """The 3-section dict the deep-merge consumer expects."""
        return {
            "model": {
                "gradient_checkpointing": self.gradient_checkpointing,
                "dtype": self.dtype,
            },
            "peft": {"method": self.method, "r": self.r},
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
        }

    def label(self) -> str:
        ckpt = "on" if self.gradient_checkpointing else "off"
        method = method_pretty_name(self.method)
        used_gib = self.predicted_bytes / _GB
        total_gib = (self.budget_bytes + self.headroom_bytes) / _GB
        if self.provenance == "calibrated":
            date_str = self.calibrated_at[:10] if self.calibrated_at else "unknown"
            suffix = f"(calibrated {date_str})"
        else:
            suffix = "(analytic estimate)"
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"grad_accum={self.grad_accum_steps} ckpt={ckpt} bf16 — "
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
        return cls(**d)


# === Memory model ==========================================================


def _bytes_per_param_for_method(method: str) -> float:
    return 2.0 if method == "lora" else 0.5  # bf16 vs NF4


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


def _activation_bytes(image_size: int, batch: int, ckpt: bool, cache: dict[str, Any] | None) -> int:
    per = _activation_per_example(image_size, cache)
    factor = CKPT_FACTOR if ckpt else 1.0
    return int(per * batch * factor)


def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    ckpt: bool,
    image_size: int,
    cache: dict[str, Any] | None,
) -> int:
    return (
        _model_bytes(method)
        + _adapter_bytes(r)
        + _optimizer_bytes(r)
        + _activation_bytes(image_size, batch, ckpt, cache)
        + WORKSPACE_BYTES
    )


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


def _load_cache(image_size: int, gpu_name: str) -> tuple[dict[str, Any] | None, Path | None]:
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
        or int(data.get("image_size", -1)) != image_size
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


def _candidates() -> list[tuple[str, int, int, bool]]:
    methods = ("lora", "qlora")
    rs = (8, 16, 24, 32, 48, 64)
    batches = tuple(range(1, 17))
    ckpts = (False, True)
    return [(m, r, b, c) for m in methods for r in rs for b in batches for c in ckpts]


def _sort_key(c: tuple[str, int, int, bool]) -> tuple[int, int, int, int]:
    method, r, batch, ckpt = c
    return (
        0 if method == "lora" else 1,
        -r,
        -batch,
        0 if not ckpt else 1,
    )


# === Public entry point ====================================================


def decide_preset(image_size: int) -> PresetDecision:
    """Pick the largest configuration that fits within the VRAM budget.

    Raises:
      ValueError: image_size invalid.
      RuntimeError: CUDA unavailable, env-var malformed, or no candidate fits.

    Spec: design §3 + §7.
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, cache_path = _load_cache(image_size, gpu_name)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )
    calibrated_at: str | None = str(cache["calibrated_at"]) if cache is not None else None

    feasible = []
    for method, r, batch, ckpt in _candidates():
        pb = _predicted_bytes(method, r, batch, ckpt, image_size, cache)
        if pb <= budget:
            feasible.append((method, r, batch, ckpt, pb))

    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        # Compute minimum-needed at QLoRA r=4 batch=1 ckpt=on for the error msg.
        min_needed = _predicted_bytes("qlora", 4, 1, True, image_size, cache)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1 ckpt=on. Use a larger GPU."
        )

    feasible.sort(key=lambda t: _sort_key(t[:4]))
    method, r, batch, ckpt, predicted = feasible[0]
    grad_accum = max(1, 16 // batch)

    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        gradient_checkpointing=ckpt,
        dtype="bfloat16",
        headroom_bytes=headroom,
        predicted_bytes=predicted,
        budget_bytes=budget,
        image_size=image_size,
        gpu_name=gpu_name,
        provenance=provenance,
        cache_path=cache_path,
        calibrated_at=calibrated_at,
    )
