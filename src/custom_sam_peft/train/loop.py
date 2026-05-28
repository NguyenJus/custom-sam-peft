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
from typing import Any, cast

import torch
from torch import Tensor

from custom_sam_peft.cli._progress import progress as P
from custom_sam_peft.config.schema import BoxHintSchedule, TrainConfig
from custom_sam_peft.data.base import Instance, TextPrompts
from custom_sam_peft.models.losses import total_loss
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, Sam3Wrapper
from custom_sam_peft.peft_adapters import PEFTMethod, make_peft_method
from custom_sam_peft.runtime import Runtime, to_device
from custom_sam_peft.runtime._runtime import coerce_dtype_for_capability
from custom_sam_peft.tracking.base import Tracker
from custom_sam_peft.train.types import OomEvent

_LOG = logging.getLogger(__name__)

# Module-level flag: logs the auto-chunk INFO message only once per process run.
# Reset via _reset_auto_chunk_log() in tests.
_AUTO_CHUNK_LOGGED: bool = False


def _reset_auto_chunk_log() -> None:
    """Test helper: reset the _AUTO_CHUNK_LOGGED flag between test cases."""
    global _AUTO_CHUNK_LOGGED
    _AUTO_CHUNK_LOGGED = False


def _chunked(seq: list[str], n: int) -> list[list[str]]:
    """Split seq into consecutive chunks of size ≤ n. Preserves order."""
    if n <= 0:
        raise ValueError(f"_chunked: n must be positive; got {n}")
    return [seq[i : i + n] for i in range(0, len(seq), n)]


@dataclass
class OomState:
    """Mutable state the OOM ladder reads/writes across steps.

    Held by the Trainer for the lifetime of a `fit()` call. The trainer's
    inner per-class loss block calls `_train_step_with_oom_ladder` once per
    step; on OOM the helper halves `micro_batch_size` in place (sticky) and
    appends to `pending_oom_events`.
    """

    step: int = 0
    micro_batch_size: int = 1
    pending_oom_events: list[OomEvent] = field(default_factory=list)


def _train_step_with_oom_ladder(
    model: Any,
    batch: Any,
    state: Any,  # _State (test) | OomState (prod)
    *,
    forward_call: Callable[[Any, Any], torch.Tensor],
) -> torch.Tensor:
    """Run one optimizer-step's worth of microbatches; ladder OOM downward.

    Caller is responsible for `optimizer.zero_grad()` (once, outside this
    helper) and `optimizer.step()` (once, after this helper returns).

    Spec §6 invariants:
      - microbatch shrink is sticky
      - optimizer.zero_grad never called mid-microbatch (helper does not call it)
      - mid-step OOM replays from i=0 at the smaller size

    Returns the final detached loss tensor of the last successful microbatch.
    """
    n = len(batch)
    last_loss: torch.Tensor | None = None
    while True:
        try:
            mb = state.micro_batch_size
            n_micro = (n + mb - 1) // mb
            for i in range(n_micro):
                start = i * mb
                end = min(start + mb, n)
                micro = batch[start:end]
                loss = forward_call(model, micro)
                # Caller divides by grad_accum_steps separately; we divide by
                # n_micro here so the gradient magnitude matches the pre-ladder
                # path. Outer loop must NOT divide again.
                (loss / n_micro).backward()
                last_loss = loss.detach()
            return last_loss if last_loss is not None else torch.tensor(0.0)
        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if state.micro_batch_size > 1:
                state.micro_batch_size //= 2
                state.pending_oom_events.append(
                    OomEvent(
                        step=state.step,
                        action="microbatch_halved",
                        new_micro_batch_size=state.micro_batch_size,
                    )
                )
                _LOG.warning(
                    "OOM at step %d — halving micro_batch_size to %d",
                    state.step,
                    state.micro_batch_size,
                )
                continue
            raise RuntimeError(
                f"OOM at step {state.step} after micro_batch=1. Use a larger GPU."
            ) from oom_err


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


def _autocast_ctx(cfg: TrainConfig, peft_method: PEFTMethod) -> Any:
    if peft_method.disables_outer_autocast():
        return contextlib.nullcontext()
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    requested = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16
    dtype = coerce_dtype_for_capability(
        requested, device=torch.device("cuda", torch.cuda.current_device())
    )
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
    peft_method: PEFTMethod | None = None,
    runtime: Runtime | None = None,
    oom_state: OomState | None = None,
) -> StepResult:
    _peft_method: PEFTMethod = (
        peft_method if peft_method is not None else make_peft_method(cfg.peft.method)
    )
    # Device moves are routed through runtime.to_device (§3 seam discipline).
    # If no runtime was passed, synthesize one from the model's parameter device.
    if runtime is None:
        param_device = next(model.parameters()).device
        runtime = Runtime(device=param_device, dtype=torch.float32)
    images: Tensor = to_device(batch["images"], runtime)
    prompts = batch["prompts"]
    targets: list[list[Instance]] = batch["instances"]
    B = images.shape[0]
    p_t = _box_hint_p(global_step, cfg.train.box_hint)

    classes_in_batch = sorted({c for p in prompts for c in p.classes})
    if not classes_in_batch:
        _LOG.warning("train_step: batch has no class prompts; skipping (data condition)")
        return StepResult.empty(p_t=p_t, nan_streak=nan_streak)

    # Build per-group chunks.  effective_K is capped at MULTIPLEX_CAP so the
    # wrapper's validation never fires (it rejects K > MULTIPLEX_CAP).
    effective_K = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    groups = _chunked(classes_in_batch, effective_K)
    G = len(groups)

    global _AUTO_CHUNK_LOGGED
    if len(classes_in_batch) > MULTIPLEX_CAP and not _AUTO_CHUNK_LOGGED:
        _LOG.info(
            "multiplex auto-chunk: classes_in_batch=%d > MULTIPLEX_CAP=%d -> %d groups",
            len(classes_in_batch),
            MULTIPLEX_CAP,
            G,
        )
        _AUTO_CHUNK_LOGGED = True

    accum: dict[str, float] = {"mask": 0.0, "box": 0.0, "obj": 0.0, "presence": 0.0, "total": 0.0}
    finite_group_count = 0
    n_hint_applied = 0

    if oom_state is not None:
        oom_state.step = global_step

    for group in groups:
        # Build per-group prompts (all B images share the same class list).
        prompts_g = [TextPrompts(classes=list(group)) for _ in range(B)]

        # Build image-major / class-minor hints and targets.
        # hints_g[i*K_g + j] covers image i, class group[j].
        hints_g: list[Tensor | None] = []
        targets_g: list[list[Instance]] = []
        for i in range(B):
            for c in group:
                c_dense = class_names.index(c)
                row_targets = [inst for inst in targets[i] if inst.class_id == c_dense]
                targets_g.append(row_targets)
                if row_targets and random.random() < p_t:  # noqa: S311 — training sampling probability, not security-sensitive
                    box_tensor = torch.stack([inst.box for inst in row_targets])
                    hints_g.append(to_device(box_tensor, runtime))
                    n_hint_applied += 1
                else:
                    hints_g.append(None)

        group_losses: dict[str, Tensor] | None = None
        group_scaled: Tensor | None = None
        is_finite = False
        try:
            if oom_state is not None:
                # OOM ladder (Pattern B): treat the batch indices as the microbatch
                # sequence so the ladder can halve B on OOM. The forward_call
                # receives a list of image indices (one microbatch slice at a time)
                # and returns the per-microbatch loss divided only by
                # (G * grad_accum_steps). The helper applies / n_micro via
                # `(loss / n_micro).backward()`, so the closure must NOT include
                # n_micro in its denominator — doing so would double-scale gradients
                # whenever n_micro > 1 (i.e., after any OOM halving).
                #
                # The flat hint/target lists are image-major / class-minor with
                # K_g slots per image. When the ladder halves B from mb to mb//2,
                # we pass micro_indices [0..mb//2-1] and the closure slices the
                # flat hint list as hints_g[i*K_g : (i+1)*K_g] for each micro image.
                # This keeps the flat-row layout correct across all microbatch sizes.
                K_g = len(group)
                _last_group_losses: list[dict[str, Tensor]] = []

                def _forward_group(
                    _model: Any,
                    micro_indices: list[int],
                    _prompts_g: list[Any] = prompts_g,
                    _targets_g: list[list[Instance]] = targets_g,
                    _hints_g: list[Tensor | None] = hints_g,
                    _K_g: int = K_g,
                    _G: int = G,
                    _grad_accum: int = cfg.train.grad_accum_steps,
                    _losses_out: list[dict[str, Tensor]] = _last_group_losses,
                    _pm: PEFTMethod = _peft_method,
                ) -> Tensor:
                    # Slice prompts and flat hint/target rows for this microbatch.
                    micro_prompts = [_prompts_g[i] for i in micro_indices]
                    # For each image i in micro_indices, take its K_g consecutive rows.
                    micro_targets = [
                        _targets_g[i * _K_g + j] for i in micro_indices for j in range(_K_g)
                    ]
                    micro_hints = [
                        _hints_g[i * _K_g + j] for i in micro_indices for j in range(_K_g)
                    ]
                    micro_imgs = images[micro_indices]
                    with _autocast_ctx(cfg, _pm):
                        micro_out = _model(micro_imgs, micro_prompts, box_hints=micro_hints)
                        micro_grp_losses = total_loss(micro_out, micro_targets, cfg.train.loss)
                    _losses_out.clear()
                    _losses_out.append(micro_grp_losses)
                    # Divide only by G and grad_accum — NOT by n_micro.
                    # The ladder helper applies / n_micro in (loss / n_micro).backward().
                    return cast(Tensor, micro_grp_losses["total"]) / (_G * _grad_accum)

                image_indices = list(range(B))
                _train_step_with_oom_ladder(
                    model, image_indices, oom_state, forward_call=_forward_group
                )
                # Use the last microbatch's losses for scalar logging.
                group_losses = _last_group_losses[0] if _last_group_losses else None
                if group_losses is not None:
                    group_scaled_val = group_losses["total"] / (G * cfg.train.grad_accum_steps)
                    is_finite = bool(torch.isfinite(group_scaled_val))
            else:
                with _autocast_ctx(cfg, _peft_method):
                    out = model(images, prompts_g, box_hints=hints_g)
                    group_losses = total_loss(out, targets_g, cfg.train.loss)
                group_scaled = group_losses["total"] / (G * cfg.train.grad_accum_steps)
                is_finite = bool(torch.isfinite(group_scaled))
        except ValueError as exc:
            # Hungarian matcher raises ValueError on non-finite cost matrices;
            # treat as a NaN-group skip. Other exceptions (RuntimeError for OOM,
            # shape mismatches, dtype errors, device mismatches) must propagate.
            _LOG.warning(
                "train_step: group %r raised %s; treating as non-finite.", list(group), exc
            )
            is_finite = False

        if is_finite and group_losses is not None:
            if oom_state is None and group_scaled is not None:
                group_scaled.backward()  # type: ignore[no-untyped-call]
            # (when oom_state is not None, backward already happened in the ladder)
            finite_group_count += 1
            for k in ("mask", "box", "obj", "presence", "total"):
                accum[k] += float(group_losses[k].detach())

    # Step is skipped only when EVERY group is non-finite.
    skipped = finite_group_count == 0
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
        losses={k: v / max(finite_group_count, 1) for k, v in accum.items()},
        p_t=p_t,
        n_hint_applied=n_hint_applied,
        n_classes=len(classes_in_batch),  # K_total: contract unchanged
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
    on_checkpoint: Callable[[int, int, float, int], None],
    on_eval: Callable[[int], None],
    peft_method: PEFTMethod | None = None,
    runtime: Runtime | None = None,
    oom_state: OomState | None = None,
) -> tuple[int, int]:
    """Drive one epoch. `on_checkpoint(global_step, epoch, p_t, nan_streak)`
    is called at every `save_every` boundary; the trainer wires it to the
    checkpoint + image-panel routines. `on_eval(global_step)` is called at
    every `eval_every` boundary for lite mid-run evaluation."""
    _peft_method: PEFTMethod = (
        peft_method if peft_method is not None else make_peft_method(cfg.peft.method)
    )
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
            peft_method=_peft_method,
            runtime=runtime,
            oom_state=oom_state,
        )
        nan_streak = result.nan_streak
        global_step += 1
        window.update(result, lr=float(scheduler.get_last_lr()[0]))
        P.advance_inner()
        if global_step % cfg.train.log_every == 0:
            scalars = window.flush()
            tracker.log_scalars(global_step, scalars)
            P.update_postfix(
                loss=scalars.get("loss/total", 0.0),
                lr=scalars.get("lr", 0.0),
                it_s=scalars.get("throughput/img_s", 0.0),
            )
        if global_step % cfg.train.save_every == 0:
            on_checkpoint(global_step, epoch, result.p_t, nan_streak)
        if global_step > 0 and global_step % cfg.train.eval_every == 0:
            on_eval(global_step)
    return global_step, nan_streak
