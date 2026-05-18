"""Inner training step + epoch loop.

`train_step` runs the per-batch class-vocabulary loop with per-class backward
(O(forward) memory regardless of class count), Bernoulli box-hint sampling,
and NaN-skip policy. `run_epoch` handles cadence: scalar logging every
`log_every` micro-steps and full-state checkpoints (plus image panels) every
`save_every`.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from esam3.config.schema import BoxHintSchedule, TrainConfig
from esam3.data.base import Instance, TextPrompts
from esam3.models.losses import total_loss
from esam3.models.sam3 import Sam3Wrapper
from esam3.tracking.base import Tracker

_LOG = logging.getLogger(__name__)


@dataclass
class StepResult:
    losses: dict[str, float]
    p_t: float
    n_hint_applied: int
    n_classes: int
    grad_norm: float | None
    skipped: bool
    nan_streak: int
    images_processed: int

    @classmethod
    def empty(cls, p_t: float, nan_streak: int = 0) -> StepResult:
        return cls(
            losses={"mask": 0.0, "box": 0.0, "obj": 0.0, "presence": 0.0, "total": 0.0},
            p_t=p_t,
            n_hint_applied=0,
            n_classes=0,
            grad_norm=None,
            skipped=True,
            nan_streak=nan_streak,
            images_processed=0,
        )


def _box_hint_p(global_step: int, cfg: BoxHintSchedule) -> float:
    if global_step >= cfg.decay_steps:
        return cfg.p_end
    frac = global_step / cfg.decay_steps
    return cfg.p_start + (cfg.p_end - cfg.p_start) * frac


def _autocast_ctx(cfg: TrainConfig) -> Any:
    if cfg.peft.method == "qlora":
        return contextlib.nullcontext()
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_step(
    model: Sam3Wrapper,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: TrainConfig,
    class_names: list[str],
    global_step: int,
    nan_streak: int,
) -> StepResult:
    device = next(model.parameters()).device
    images: Tensor = batch["images"].to(device)
    prompts = batch["prompts"]
    targets: list[list[Instance]] = batch["instances"]
    B = images.shape[0]
    p_t = _box_hint_p(global_step, cfg.train.box_hint)

    classes_in_batch = sorted({c for p in prompts for c in p.classes})
    if not classes_in_batch:
        _LOG.warning("train_step: batch has no class prompts; skipping (data condition)")
        return StepResult.empty(p_t=p_t, nan_streak=nan_streak)

    accum: dict[str, float] = {"mask": 0.0, "box": 0.0, "obj": 0.0, "presence": 0.0, "total": 0.0}
    finite_class_count = 0
    n_hint_applied = 0

    for c in classes_in_batch:
        prompts_c = [TextPrompts(classes=[c]) for _ in range(B)]
        c_dense = class_names.index(c)
        targets_c = [[inst for inst in targets[i] if inst.class_id == c_dense] for i in range(B)]
        hints_c: list[Tensor | None] = []
        for i in range(B):
            if targets_c[i] and random.random() < p_t:
                hints_c.append(torch.stack([inst.box for inst in targets_c[i]]).to(device))
                n_hint_applied += 1
            else:
                hints_c.append(None)

        class_losses: dict[str, Tensor] | None = None
        class_scaled: Tensor | None = None
        try:
            with _autocast_ctx(cfg):
                out = model(images, prompts_c, box_hints=hints_c)
                class_losses = total_loss(out, targets_c, cfg.train.loss)
            class_scaled = class_losses["total"] / (
                len(classes_in_batch) * cfg.train.grad_accum_steps
            )
            is_finite = bool(torch.isfinite(class_scaled))
        except ValueError as exc:
            # Hungarian matcher raises ValueError on non-finite cost matrices;
            # treat as a NaN-class skip. Other exceptions (RuntimeError for OOM,
            # shape mismatches, dtype errors, device mismatches) must propagate.
            _LOG.warning("train_step: class %r raised %s; treating as non-finite.", c, exc)
            is_finite = False

        if is_finite and class_scaled is not None and class_losses is not None:
            class_scaled.backward()  # type: ignore[no-untyped-call]
            finite_class_count += 1
            for k in ("mask", "box", "obj", "presence", "total"):
                accum[k] += float(class_losses[k].detach())

    skipped = finite_class_count == 0
    new_streak = nan_streak + 1 if skipped else 0
    if new_streak >= cfg.train.nan_abort_after:
        raise RuntimeError(f"Training aborted: {new_streak} consecutive non-finite micro-steps.")

    grad_norm: float | None = None
    if (global_step + 1) % cfg.train.grad_accum_steps == 0 and not skipped:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.train.max_grad_norm,
            )
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return StepResult(
        losses={k: v / max(finite_class_count, 1) for k, v in accum.items()},
        p_t=p_t,
        n_hint_applied=n_hint_applied,
        n_classes=len(classes_in_batch),
        grad_norm=grad_norm,
        skipped=skipped,
        nan_streak=new_streak,
        images_processed=B,
    )


@dataclass
class _ScalarWindow:
    n: int = 0
    cumulative_skipped: int = 0
    sums: dict[str, float] = field(
        default_factory=lambda: {
            "loss/total": 0.0,
            "loss/mask": 0.0,
            "loss/box": 0.0,
            "loss/obj": 0.0,
            "loss/presence": 0.0,
            "box_hint/applied": 0.0,
            "throughput/img_s": 0.0,
            "grad_norm": 0.0,
        }
    )
    grad_norm_n: int = 0
    last_p_t: float = 0.0
    last_lr: float = 0.0
    images_in_window: int = 0
    wall_t0: float = field(default_factory=time.perf_counter)

    def update(self, r: StepResult, lr: float) -> None:
        self.n += 1
        if r.skipped:
            self.cumulative_skipped += 1
            return
        self.sums["loss/total"] += r.losses["total"]
        self.sums["loss/mask"] += r.losses["mask"]
        self.sums["loss/box"] += r.losses["box"]
        self.sums["loss/obj"] += r.losses["obj"]
        self.sums["loss/presence"] += r.losses["presence"]
        denom = max(r.n_classes * max(r.images_processed, 1), 1)
        self.sums["box_hint/applied"] += r.n_hint_applied / denom
        self.images_in_window += r.images_processed
        if r.grad_norm is not None:
            self.sums["grad_norm"] += r.grad_norm
            self.grad_norm_n += 1
        self.last_p_t = r.p_t
        self.last_lr = lr

    def flush(self) -> dict[str, float]:
        n = max(self.n - 0, 1)
        elapsed = max(time.perf_counter() - self.wall_t0, 1e-9)
        out = {
            "loss/total": self.sums["loss/total"] / n,
            "loss/mask": self.sums["loss/mask"] / n,
            "loss/box": self.sums["loss/box"] / n,
            "loss/obj": self.sums["loss/obj"] / n,
            "loss/presence": self.sums["loss/presence"] / n,
            "lr": self.last_lr,
            "box_hint/p": self.last_p_t,
            "box_hint/applied": self.sums["box_hint/applied"] / n,
            "grad_norm": (self.sums["grad_norm"] / self.grad_norm_n if self.grad_norm_n else 0.0),
            "throughput/img_s": self.images_in_window / elapsed,
            "skipped_steps": float(self.cumulative_skipped),
        }
        cum_skipped = self.cumulative_skipped
        self.__init__()  # type: ignore[misc]
        self.cumulative_skipped = cum_skipped
        return out


def run_epoch(
    model: Sam3Wrapper,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tracker: Tracker,
    cfg: TrainConfig,
    run_dir: Path,
    epoch: int,
    global_step: int,
    nan_streak: int,
    class_names: list[str],
    val_ds: Any,
    on_checkpoint: Callable[[int, int, float, int], None],
    on_eval: Callable[[int], None],
) -> tuple[int, int]:
    """Drive one epoch. `on_checkpoint(global_step, epoch, p_t, nan_streak)`
    is called at every `save_every` boundary; the trainer wires it to the
    checkpoint + image-panel routines. `on_eval(global_step)` is called at
    every `eval_every` boundary for lite mid-run evaluation."""
    window = _ScalarWindow()
    for batch in loader:
        result = train_step(
            model,
            batch,
            optimizer,
            scheduler,
            cfg,
            class_names=class_names,
            global_step=global_step,
            nan_streak=nan_streak,
        )
        nan_streak = result.nan_streak
        global_step += 1
        window.update(result, lr=float(scheduler.get_last_lr()[0]))
        if global_step % cfg.train.log_every == 0:
            tracker.log_scalars(global_step, window.flush())
        if global_step % cfg.train.save_every == 0:
            on_checkpoint(global_step, epoch, result.p_t, nan_streak)
        if global_step > 0 and global_step % cfg.train.eval_every == 0:
            on_eval(global_step)
    return global_step, nan_streak
