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

MODEL_PARAMS = 5_000_000_000  # SAM 3.1 base parameter count — analytic seed
# (vision encoder ~762M + text encoder ~302M + decoder/neck ~50M,
#  plus retained activations and optimizer state heuristic; superseded
#  by calibration cache. Re-derive via scripts/_derive_preset_constants.py)
LORA_LAYERS = 96  # vision_decoder scope; nn.Linear LoRA targets (_resolve_targets)
D_IN = 768  # avg input feature dim across LoRA targets
D_OUT = 768  # avg output feature dim across LoRA targets
Q_OVERHEAD = 64 * _MIB  # bnb NF4 per-block scale + zero-point overhead
WORKSPACE_BYTES = 256 * _MIB  # cuDNN workspace + autograd graph + tmp buffers (spec §3)
# Split activation seeds — PORTABLE FLASH-BASELINE, measured natively at SAM 3.1's
# fixed SAM3_IMAGE_SIZE=1008 (no image-size scale term). Spec §2/§2.1/§6.
#   predicted_peak = STATIC + (A_FIXED + A_PER_CLASS * K) * batch
#                    + (_attention_bytes_per_example(1008) * batch  if cc < 8.0 else 0)
# A_FIXED   — K-invariant vision-encoder (hiera-large) activation, per image.
# A_PER_CLASS — decoder / mask-head activation, per (image x class), two-point split.
A_FIXED = 0  # 0.00 GiB encoder activation per image @1008px (clamped flash residual)
A_PER_CLASS = 1_248_840_021  # 1.163 GiB decoder activation per class @1008px

# Forward-only memory is roughly 1/4 of the train-step probe (train captures
# forward + backward + retained graph; eval captures only forward, no graph).
# Spec §8.
forward_only_factor: float = 0.25

# === CALIBRATION CACHE =====================================================

CACHE_FILENAME = ".custom_sam_peft_calibration.json"
CACHE_SCHEMA_VERSION = 4  # v4: pinned r/alpha (r no longer searched); drops rank-maximization


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
    classes_per_forward: int
    dtype: Literal["bfloat16", "float16"]
    headroom_bytes: int
    predicted_bytes: int
    budget_bytes: int
    gpu_name: str
    provenance: Literal["calibrated", "analytic"]
    cache_path: Path | None
    calibrated_at: str | None  # ISO 8601 string when provenance == "calibrated", None otherwise
    alpha: int = 32  # cite: PEFTConfig.alpha default (defaults-provenance.md:85); set explicitly
    # by every constructor; default is a defensive fallback. Co-scaled alongside r
    # on VRAM-driven rank reduction; never raised above the pinned config alpha.

    @property
    def config_patch(self) -> dict[str, dict[str, object]]:
        """The 3-section dict the deep-merge consumer expects."""
        return {
            "model": {"dtype": self.dtype},
            "peft": {"method": self.method, "r": self.r, "alpha": self.alpha},
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "multiplex": {"classes_per_forward": self.classes_per_forward},
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
            f"K={self.classes_per_forward} grad_accum={self.grad_accum_steps} "
            f"{dtype_token} — "
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


# SAM 3.1 vision backbone (hiera-large), from sam3/model_builder.py. Shared by
# the train-branch formula and decide_eval_batch_size's SDPA ceiling so both
# cite one definition (spec §3.2).
# _attention_bytes_per_example is the dominant activation term at SAM 3.1's
# 1008px image; only the A_PER_CLASS term scales with k_eff in the train branch
# (see _activation_bytes). Spec §2.
_SAM3_PATCH = 14  # vision backbone patch size
_SAM3_HEADS = 16  # vision backbone attention heads

# === Memory model ==========================================================


def _bytes_per_param_for_method(method: str) -> float:
    return 2.0 if method == "lora" else 0.5


def _model_bytes(method: str) -> int:
    base = int(MODEL_PARAMS * _bytes_per_param_for_method(method))
    return base + (Q_OVERHEAD if method == "qlora" else 0)


def _adapter_bytes(r: int) -> int:
    # LORA_LAYERS * r * (D_IN + D_OUT) * 2 bytes (bf16 adapter weights).
    return LORA_LAYERS * r * (D_IN + D_OUT) * 2


def _optimizer_bytes(r: int) -> int:
    # AdamW state on the bf16 adapter — fp32 m + fp32 v (two fp32 moments).
    # Adapter weights are 2 B/param; state is 8 B/param -> 4x adapter_bytes.
    return _adapter_bytes(r) * 4


def _attention_bytes_per_example(image_size: int) -> int:
    """Per-example SDPA score-matrix bytes: H * N^2 * 4 (fp32 math upcast).

    At SAM 3.1's image_size=1008, patch=14 -> N=5184 tokens, so this term is the
    dominant activation contributor and is exactly what the train formula omitted
    (the 10-vs-22 GiB miss). Spec §3.2.
    """
    n_tokens = (image_size // _SAM3_PATCH) ** 2
    return _SAM3_HEADS * n_tokens * n_tokens * 4


def _flash_attention_available(cc: tuple[int, int] | None) -> bool:
    """True iff the GPU's compute capability gets FlashAttention-2 / mem-efficient
    SDPA (cc >= 8.0), so the encoder self-attention never materializes the N*N score
    matrix.

    cc < 8.0 (pre-Ampere, e.g. Turing 7.5) commonly falls back to the SDPA math
    backend, which materializes the full H*N^2 fp32 score matrix; on those cards the
    predictor re-adds _attention_bytes_per_example for safety (Amendment 2, spec §2.1).

    Conservative default: an unknown / unreadable cc returns False (assume no flash ->
    include the attention term -> safe over-estimate). Turing (7.5) is deliberately
    treated as no-flash even though it usually gets mem-efficient SDPA — over-
    estimating is always safe for cards below the CC 8.0 floor.
    """
    if cc is None:
        return False
    return cc >= (8, 0)


def _activation_bytes(batch: int, cache: dict[str, Any] | None, k_eff: int = 1) -> int:
    """Split activation bytes: (A_FIXED + A_PER_CLASS * K) * batch.

    A_FIXED (K-invariant vision-encoder activation) does NOT scale with K; only the
    A_PER_CLASS decoder term does. Measured natively at SAM 3.1's fixed 1008px, so
    there is no image-size scale term. Reads the split from a v3 cache when present.
    Spec §2.
    """
    if cache is not None:
        a_fixed = int(cache["A_fixed"])
        a_per_class = int(cache["A_per_class"])
    else:
        a_fixed = A_FIXED
        a_per_class = A_PER_CLASS
    return int((a_fixed + a_per_class * k_eff) * batch)


def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    image_size: int,
    cache: dict[str, Any] | None,
    mode: Literal["train", "eval"] = "train",
    k_eff: int = 1,
    flash_available: bool = True,
) -> int:
    # cc-aware attention term (Amendment 2, spec §2.1). On a flash card
    # (cc >= 8.0) the forward attention is folded into the empirical split, so no
    # separate term. On a no-flash card (cc < 8.0) SDPA commonly falls back to the
    # math backend, which materializes the full H*N^2 fp32 score matrix; that worst
    # case is NOT in the flash-baseline split, so it is re-added for safety. The term
    # is K-INVARIANT (* batch, never * k_eff), so it cannot re-trigger #203.
    attn = 0 if flash_available else _attention_bytes_per_example(image_size) * batch
    if mode == "train":
        # STATIC + split + conditional attention.
        return (
            _model_bytes(method)
            + _adapter_bytes(r)
            + _optimizer_bytes(r)
            + _activation_bytes(batch, cache, k_eff=k_eff)
            + WORKSPACE_BYTES
            + attn
        )
    # mode == "eval": no optimizer, no adapter bytes; activations x forward_only_factor.
    # K is threaded through the split; decide_eval_batch_size passes its
    # classes_per_forward as k_eff and the regime flag as flash_available. The
    # conditional attention term keeps eval SAFE on no-flash cards. The independent
    # SDPA CAP stays in decide_eval_batch_size (always-on ceiling). Spec §2.1/§6.
    activations = int(_activation_bytes(batch, cache, k_eff=k_eff) * forward_only_factor)
    return _model_bytes(method) + activations + WORKSPACE_BYTES + attn


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


def _load_cache(
    gpu_name: str, cache_path: Path | None = None
) -> tuple[dict[str, Any] | None, Path | None]:
    """Return (cache_dict, absolute_cache_path) iff the cache matches.

    Args:
        gpu_name: the GPU name reported by torch.cuda.get_device_name(0).
        cache_path: path to the calibration cache file. Defaults to
            ``Path(CACHE_FILENAME).resolve()`` (the fixed default location).
            Pass an explicit path when the caller writes the cache to a non-default
            location (e.g. ``calibrate --output``).
    """
    cache_path = Path(CACHE_FILENAME).resolve() if cache_path is None else cache_path
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
    if "A_fixed" not in data or "A_per_class" not in data:
        _LOG.warning("calibration cache missing split keys (A_fixed/A_per_class); ignoring")
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


def _candidates() -> list[tuple[int, int]]:
    """Enumerate the (batch, k) search space. method/r are pinned inputs, not searched.

    Spec §6.1: drop the r dimension; enumerate only (b, k). Tail-to-head =
    sacrifice order (give up batch first, then K). Spec §3/§4.
    """
    batches = tuple(range(1, 17))
    ks = (1, 2, 4, 8, 16)
    return [(b, k) for b in batches for k in ks]


def _sort_key(c: tuple[int, int]) -> tuple[int, int]:
    """Sort (batch, k) candidates highest-first: largest K wins, then largest batch.

    Tail-to-head = sacrifice order: give up batch first, then K. method/r are
    pinned inputs and are not part of the sort key. Spec §3/§6.1.
    """
    batch, k = c
    return (-k, -batch)


# === Public entry point ====================================================


def decide_preset(
    k: int | None = None,
    cache_path: Path | None = None,
    *,
    method: Literal["lora", "qlora"] = "lora",  # cite: PEFTConfig.method default
    r: int = 16,  # cite: PEFTConfig.r default (defaults-provenance.md:84)
    alpha: int = 32,  # cite: PEFTConfig.alpha default (defaults-provenance.md:85)
    num_classes: int | None = None,
) -> PresetDecision:
    """Pick the largest (b, k) configuration that fits within the VRAM budget.

    ``method``, ``r``, and ``alpha`` are pinned accuracy choices — hardware never
    raises them. Only ``b`` (batch) and ``k`` (classes_per_forward) are autosized.
    ``r`` is reduced as a last resort with a co-scaled ``alpha`` and a user warning.

    Sacrifice order (cheapest to most damaging):
        b↓ → k↓ (b held at 1) → lora→qlora @ same r → r↓ (+WARNING, alpha co-scaled)

    Args:
      k: upper bound on the K (classes-per-forward) search. When None, uses
         MULTIPLEX_CAP. Callers with a config in scope pass
         cfg.train.multiplex.classes_per_forward as the cap. Spec §3.
      cache_path: path to the calibration cache file. Defaults to
         ``Path(CACHE_FILENAME).resolve()`` (the fixed default location). Pass an
         explicit path when the calibration cache was written to a non-default
         location (e.g. ``calibrate --output``), so provenance reflects the
         just-written probe rather than a stale/absent default cache.
      method: pinned PEFT method; defaults to the PEFTConfig.method default "lora".
      r: pinned LoRA rank; defaults to the PEFTConfig.r default 16
         (defaults-provenance.md:84).
      alpha: pinned LoRA alpha; defaults to the PEFTConfig.alpha default 32
         (defaults-provenance.md:85). Co-scaled alongside r on VRAM-driven rank
         reduction: alpha = max(1, round(alpha * r_new / r)).
      num_classes: best-effort class count from the dataset vocabulary. When
         provided, k_start = min(cfg K cap, MULTIPLEX_CAP, num_classes) — capping
         K at the actual vocabulary never probes more K than needed. None falls
         back to min(cfg K cap, MULTIPLEX_CAP).

    Raises:
      RuntimeError: CUDA unavailable, env-var malformed, or no candidate fits.

    Spec: design §4, §6.1, §7.
    """
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE

    image_size = SAM3_IMAGE_SIZE
    # `k` is the UPPER BOUND on the K search (default MULTIPLEX_CAP). A user who
    # pins a lower classes_per_forward is respected as a cap. Spec §3.
    k_cap = MULTIPLEX_CAP if k is None else min(k, MULTIPLEX_CAP)
    if k_cap < 1:
        raise ValueError(f"k must be >= 1 when provided; got {k}")
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)

    # Snapshot cfg_r / cfg_alpha for co-scaling on r-reduction (§4.1 step 4).
    cfg_r = r
    cfg_alpha = alpha

    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    decided_dtype: Literal["bfloat16", "float16"] = "float16" if cc < (8, 0) else "bfloat16"
    flash = _flash_attention_available(cc)

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, cache_path = _load_cache(gpu_name, cache_path=cache_path)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )
    calibrated_at: str | None = str(cache["calibrated_at"]) if cache is not None else None

    # k_start: min of cfg K cap, MULTIPLEX_CAP, and dataset vocabulary (§4.1/§5).
    # None num_classes → treat as unconstrained (fall through to min(k_cap, CAP)).
    if num_classes is not None and num_classes >= 1:
        k_start = min(k_cap, MULTIPLEX_CAP, num_classes)
    else:
        k_start = min(k_cap, MULTIPLEX_CAP)
    # k_start must be in the K grid; clamp to the highest grid value <= k_start.
    _K_GRID = (16, 8, 4, 2, 1)
    k_start = max((v for v in _K_GRID if v <= k_start), default=1)

    def _fits_bk(cur_method: str, cur_r: int, batch: int, k_cand: int) -> tuple[bool, int]:
        """Return (fits, predicted_bytes)."""
        pb = _predicted_bytes(
            cur_method, cur_r, batch, image_size, cache, k_eff=k_cand, flash_available=flash
        )
        return pb <= budget, pb

    def _search_bk(cur_method: str, cur_r: int, k_init: int) -> tuple[int, int, int] | None:
        """Run §4.1 steps 1-2: largest b at k_init, then step k down with b=1.

        Returns (batch, k, predicted_bytes) or None if infeasible even at b=1, k=1.
        """
        # Step 1: largest b in 16..1 that fits at k_init.
        for b in range(16, 0, -1):
            fits, pb = _fits_bk(cur_method, cur_r, b, k_init)
            if fits:
                return b, k_init, pb
        # Step 2: b=1 fails at k_init; step k DOWN the grid, b HELD at 1.
        for k_try in _K_GRID:
            if k_try >= k_init:
                continue  # only smaller values
            fits, pb = _fits_bk(cur_method, cur_r, 1, k_try)
            if fits:
                return 1, k_try, pb
        return None

    # === §4.1 Ladder =========================================================
    # Steps 1-2: pinned (method, r) at k_start.
    result = _search_bk(method, r, k_start)
    if result is not None:
        batch, k_chosen, predicted = result
        grad_accum = max(1, math.ceil(16 / batch))
        return PresetDecision(
            method=method,
            r=r,
            alpha=alpha,
            batch_size=batch,
            grad_accum_steps=grad_accum,
            classes_per_forward=k_chosen,
            dtype=decided_dtype,
            headroom_bytes=headroom,
            predicted_bytes=predicted,
            budget_bytes=budget,
            gpu_name=gpu_name,
            provenance=provenance,
            cache_path=cache_path,
            calibrated_at=calibrated_at,
        )

    # Step 3: lora → qlora at the SAME r (NF4 base is far cheaper; preserves rank).
    if method == "lora":
        method = "qlora"
        result = _search_bk(method, r, k_start)
        if result is not None:
            batch, k_chosen, predicted = result
            grad_accum = max(1, math.ceil(16 / batch))
            return PresetDecision(
                method=method,
                r=r,
                alpha=alpha,
                batch_size=batch,
                grad_accum_steps=grad_accum,
                classes_per_forward=k_chosen,
                dtype=decided_dtype,
                headroom_bytes=headroom,
                predicted_bytes=predicted,
                budget_bytes=budget,
                gpu_name=gpu_name,
                provenance=provenance,
                cache_path=cache_path,
                calibrated_at=calibrated_at,
            )

    # Step 4: still infeasible → reduce r DOWN the grid, co-scaling alpha, with warning.
    _R_GRID = (48, 32, 24, 16, 8)  # descending; only values < the pinned r are tried
    for r_next in _R_GRID:
        if r_next >= r:
            continue
        r = r_next
        alpha = max(1, round(cfg_alpha * r / cfg_r))
        _LOG.warning(
            "decide_preset(): GPU too small for pinned r=%d; reducing to r=%d "
            "(alpha co-scaled to %d to preserve alpha:r ratio). "
            "Accuracy may be affected. Spec §4.1 step 4.",
            cfg_r,
            r,
            alpha,
        )
        # Restart steps 1-2 (method is already "qlora" from step 3 if lora failed).
        result = _search_bk(method, r, k_start)
        if result is not None:
            batch, k_chosen, predicted = result
            grad_accum = max(1, math.ceil(16 / batch))
            return PresetDecision(
                method=method,
                r=r,
                alpha=alpha,
                batch_size=batch,
                grad_accum_steps=grad_accum,
                classes_per_forward=k_chosen,
                dtype=decided_dtype,
                headroom_bytes=headroom,
                predicted_bytes=predicted,
                budget_bytes=budget,
                gpu_name=gpu_name,
                provenance=provenance,
                cache_path=cache_path,
                calibrated_at=calibrated_at,
            )

    # Step 5: exhausted — preserve the existing message shape (presets.py:413-423).
    budget_gib = budget / _GB
    headroom_gib = headroom / _GB
    min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache, k_eff=1, flash_available=flash)
    raise RuntimeError(
        f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
        f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
        f"batch=1 K=1. Use a larger GPU."
    )


def decide_eval_batch_size(
    classes_per_forward: int = 16,
) -> tuple[int, int, Literal["calibrated", "analytic"]]:
    """Pick the largest forward-only batch size that fits within the eval VRAM budget.

    Returns (batch_size, predicted_bytes, provenance).

    On CPU: returns (1, 0, "analytic") and logs once.
    Spec: design §8.

    ``classes_per_forward`` is now threaded through the split activation formula
    as k_eff, so higher K can only lower (or hold) the returned batch size —
    never raise it (no regression guarantee). Spec §6.
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
    cc = torch.cuda.get_device_capability(0)
    flash = _flash_attention_available(cc)

    headroom = _headroom_bytes()
    budget = total - headroom

    cache, _ = _load_cache(gpu_name)
    provenance: Literal["calibrated", "analytic"] = (
        "calibrated" if cache is not None else "analytic"
    )

    best_bs = 1
    best_predicted = _predicted_bytes(
        "lora",
        r=4,
        batch=1,
        image_size=image_size,
        cache=cache,
        mode="eval",
        k_eff=classes_per_forward,
        flash_available=flash,
    )
    for batch in range(1, 65):  # B in [1, 64]
        pb = _predicted_bytes(
            "lora",
            r=4,
            batch=batch,
            image_size=image_size,
            cache=cache,
            mode="eval",
            k_eff=classes_per_forward,
            flash_available=flash,
        )
        if pb <= budget:
            best_bs = batch
            best_predicted = pb

    # Attention-memory ceiling: SDPA math fallback materialises the full B*H*N*N
    # score matrix in float32 (4 bytes) even when inputs are bf16.  At SAM 3.1's
    # patch_size=14, image_size=1008 -> N=5184 tokens, H=16 heads, the analytic
    # model can return bs~35 on a 24 GiB card, causing a 56 GiB allocation (OOM).
    # Attention-memory ceiling via the shared helper so the train term and this
    # eval cap cite one definition (spec §3.2 / issue #162).
    # This SDPA ceiling is UNCONDITIONAL (always-on, cc-unaware): it is a conservative
    # cap that can only lower best_bs (the safe direction). Making it cc-aware would
    # only raise best_bs on flash cards — the unsafe direction. Spec §2.1/§7.
    _attn_per_example = _attention_bytes_per_example(image_size)
    # Model weights and CUDA workspace are fixed overhead, independent of batch
    # size; subtract them from the budget once. Attention scores AND forward
    # activations both scale per-example, so they share the divisor below.
    # Counting activations in the divisor (not just attention) makes the cap
    # conservative — it can only lower bs, which is safe for OOM prevention
    # (issue #162).
    attn_budget = budget - _model_bytes("lora") - WORKSPACE_BYTES
    _act_per_example = int(
        _activation_bytes(batch=1, cache=cache, k_eff=classes_per_forward) * forward_only_factor
    )
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
            "lora",
            r=4,
            batch=best_bs,
            image_size=image_size,
            cache=cache,
            mode="eval",
            k_eff=classes_per_forward,
            flash_available=flash,
        )

    return best_bs, best_predicted, provenance
