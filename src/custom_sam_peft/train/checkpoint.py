"""Checkpoint save/load for the training loop.

Persists adapter weights via the appropriate PEFT module (LoRA vs QLoRA
detected by Linear4bit-presence) and a sibling `training_state.pt` carrying
optimizer / scheduler / RNG / step / epoch / box_hint_p.

Resume granularity is epoch-boundary: the trainer re-walks the interrupted
epoch (RNG-restored shuffling replays the same order). See
docs/superpowers/specs/2026-05-17-training-loop-design.md §7 for rationale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from custom_sam_peft.config.schema import TrainConfig
from custom_sam_peft.errors import CheckpointError
from custom_sam_peft.models.sam3 import Sam3Wrapper
from custom_sam_peft.paths import CHECKPOINTS_SUBDIR
from custom_sam_peft.peft_adapters import make_peft_method
from custom_sam_peft.peft_adapters.lora import load_lora, merge_lora, save_lora
from custom_sam_peft.peft_adapters.qlora import load_qlora, save_qlora

_LOG = logging.getLogger(__name__)
_TRAINING_STATE_FILENAME = "training_state.pt"
_FORMAT_VERSION = 1
_CHANNEL_ADAPTER_FILENAME = "channel_adapter.pt"


def _wrapper_channel_adapter(wrapper: Sam3Wrapper) -> Any:
    """Return the channel adapter module if present, else None.

    The adapter lives on _Sam3ImageAdapter (== wrapper.model), a sibling of the raw
    SAM model — outside the PeftModel, so PEFT save/load never touches it (spec §5.1).
    """
    return getattr(wrapper.model, "channel_adapter", None)


def _save_channel_adapter(wrapper: Sam3Wrapper, adapter_dir: Path) -> None:
    """Dump the channel adapter's state_dict to channel_adapter.pt. No-op when None (rgb)."""
    ca = _wrapper_channel_adapter(wrapper)
    if ca is None:
        return
    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ca.state_dict(), adapter_dir / _CHANNEL_ADAPTER_FILENAME)


def _load_channel_adapter(wrapper: Sam3Wrapper, adapter_dir: Path) -> None:
    """Restore the channel adapter from channel_adapter.pt. No-op when the file is
    absent (rgb) or the wrapper has no adapter."""
    ca = _wrapper_channel_adapter(wrapper)
    path = adapter_dir / _CHANNEL_ADAPTER_FILENAME
    if ca is None:
        if path.exists():
            _LOG.warning(
                "Found %s but the loaded model has no channel adapter; the saved "
                "channel-adapter weights will NOT be restored. Was the model built "
                "with the right data.channels/data.channel_semantics?",
                path.name,
            )
        return
    if not path.exists():
        return
    state = torch.load(path, weights_only=True, map_location="cpu")
    ca.load_state_dict(state)


@dataclass(frozen=True)
class ResumeState:
    start_step: int
    start_epoch: int
    nan_streak: int
    box_hint_p: float


def _has_linear4bit(wrapper: Sam3Wrapper) -> bool:
    """True iff wrapper.peft_model contains any bnb.nn.Linear4bit module.

    Lazy-imports bitsandbytes; returns False on ImportError so LoRA-only
    builds don't depend on bnb being installed."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    if wrapper.peft_model is None:
        return False
    bnb_any: Any = bnb
    return any(isinstance(m, bnb_any.nn.Linear4bit) for m in wrapper.peft_model.modules())


def _hash_cfg(cfg: TrainConfig) -> str:
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_adapter(wrapper: Sam3Wrapper, path: Path) -> None:
    """LoRA vs QLoRA dispatch by Linear4bit-presence."""
    if wrapper.peft_model is None:
        raise RuntimeError("save_adapter: wrapper has no PeftModel; call apply_lora/qlora first")
    if _has_linear4bit(wrapper):
        save_qlora(wrapper, path)
    else:
        save_lora(wrapper, path)
    _save_channel_adapter(wrapper, path)


def load_adapter(wrapper: Sam3Wrapper, path: Path) -> Sam3Wrapper:
    """LoRA vs QLoRA dispatch via the canonical peft_adapters discovery seam."""
    from custom_sam_peft.peft_adapters import discover_method_from_checkpoint

    if discover_method_from_checkpoint(path) == "qlora":
        load_qlora(wrapper, path)
    else:
        load_lora(wrapper, path)
    _load_channel_adapter(wrapper, path)
    return wrapper


def save_merged(wrapper: Sam3Wrapper, path: Path) -> None:
    """Fold LoRA/QLoRA deltas into the base then dump the merged state_dict.

    For QLoRA wrappers, merge_lora dequantizes the 4-bit base to
    compute_dtype during folding; the resulting module is no longer 4-bit.

    The merged state_dict (wrapper.model.state_dict()) already contains
    channel_adapter.* keys because channel_adapter is a submodule of
    wrapper.model (_Sam3ImageAdapter). The extra channel_adapter.pt written
    here is symmetry with the save_adapter path so callers can reload the
    channel adapter without needing the full merged model file.
    """
    if wrapper.peft_model is None:
        raise RuntimeError("save_merged: wrapper has no PeftModel; call apply_lora/qlora first")
    merge_lora(wrapper)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.model.state_dict(), path / "pytorch_model.bin")
    _save_channel_adapter(wrapper, path)


def save_full_state(
    state_dir: Path,
    wrapper: Sam3Wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    epoch: int,
    nan_streak: int,
    box_hint_p: float,
    cfg: TrainConfig,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    save_adapter(wrapper, state_dir / "adapter")
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "global_step": global_step,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        "box_hint_p": float(box_hint_p),
        "nan_streak": int(nan_streak),
        "peft_method": cfg.peft.method,
        "cfg_hash": _hash_cfg(cfg),
    }
    torch.save(payload, state_dir / _TRAINING_STATE_FILENAME)


def load_full_state(
    state_dir: Path,
    wrapper: Sam3Wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: TrainConfig,
) -> ResumeState:
    state_file = state_dir / _TRAINING_STATE_FILENAME
    if not state_file.exists():
        raise FileNotFoundError(
            f"load_full_state: {state_file} not found. "
            f"Pass the step subdirectory produced by save_full_state "
            f"(e.g. paths.checkpoint_path(run_dir, step=N).parent)."
        )
    state = torch.load(state_file, weights_only=False)
    if state.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"load_full_state: unsupported format_version "
            f"{state.get('format_version')!r}; expected {_FORMAT_VERSION}"
        )

    adapter_dir = state_dir / "adapter"
    saved_method = state["peft_method"]
    saved_peft = make_peft_method(saved_method)
    try:
        detected_method = saved_peft.detect_method_from_checkpoint(adapter_dir)
    except CheckpointError as exc:
        raise CheckpointError(
            f"load_full_state: peft_method mismatch — training_state.pt says "
            f"{saved_method!r} but adapter dir contents are inconsistent: {exc}",
            expected=f"adapter dir consistent with peft_method={saved_method!r}",
            found=f"inconsistent adapter dir at {adapter_dir!r}",
            fix="use the checkpoint directory produced by the same training run, or delete the checkpoint and retrain",  # noqa: E501
        ) from exc
    if saved_method != detected_method:
        raise CheckpointError(
            f"load_full_state: peft_method mismatch — training_state.pt says "
            f"{saved_method!r} but adapter dir contents say {detected_method!r}",
            expected=f"adapter dir matching peft_method={saved_method!r}",
            found=f"adapter dir appears to be {detected_method!r}",
            fix="ensure --resume points to the checkpoint directory from the correct training run",
        )
    load_adapter(wrapper, adapter_dir)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(cast(torch.ByteTensor, rng["torch_cpu"]))
    if rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    if state["cfg_hash"] != _hash_cfg(cfg):
        _LOG.warning(
            "load_full_state: cfg_hash mismatch — resumed run uses a different "
            "config than the saved checkpoint. Proceeding anyway."
        )

    return ResumeState(
        start_step=int(state["global_step"]),
        start_epoch=int(state["epoch"]),
        nan_streak=int(state["nan_streak"]),
        box_hint_p=float(state["box_hint_p"]),
    )


def find_latest_checkpoint(cfg: TrainConfig) -> Path:
    """Return <run_dir>/checkpoints/step_N for the newest run matching cfg.run.name.

    "Newest" is determined by lexicographic sort of the run directory name, which is
    valid because make_run_dir writes an ISO-style timestamp suffix.

    Raises:
        CheckpointError: if no matching checkpoint can be found in cfg.run.output_dir.
    """
    output_dir = Path(cfg.run.output_dir)
    prefix = f"{cfg.run.name}-"

    # Collect run dirs whose name starts with the run name prefix and contain
    # at least one checkpoints/step_* directory.
    candidates: list[Path] = []
    if output_dir.is_dir():
        for entry in output_dir.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix):
                continue
            ckpt_dir = entry / CHECKPOINTS_SUBDIR
            if not ckpt_dir.is_dir():
                continue
            step_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
            if step_dirs:
                candidates.append(entry)

    # Sort descending by run dir name (lex — works because timestamp is ISO-style).
    candidates.sort(key=lambda p: p.name, reverse=True)

    for run_dir in candidates:
        ckpt_dir = run_dir / CHECKPOINTS_SUBDIR
        step_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
        if not step_dirs:
            continue

        # Pick the step dir with the largest integer suffix.
        def _step_num(p: Path) -> int:
            try:
                return int(p.name.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1

        best = max(step_dirs, key=_step_num)
        return best

    raise CheckpointError(
        f"find_latest_checkpoint: no checkpoint found for run name "
        f"{cfg.run.name!r} in output_dir {cfg.run.output_dir!r}.",
        expected=f"at least one {prefix}<timestamp>/{CHECKPOINTS_SUBDIR}/step_<N>/ directory",
        found=f"no matching directories under {cfg.run.output_dir!r}",
        fix=(
            "run `csp train --config <cfg>` first to produce a checkpoint, "
            "or pass an explicit path to --resume"
        ),
    )
