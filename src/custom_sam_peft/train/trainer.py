"""Trainer — public training entrypoint. Step body lives in train/loop.py."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from custom_sam_peft.cli._progress import progress as P
from custom_sam_peft.config.schema import Optimizer, TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.collate import collate_batch
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.models.sam3 import Sam3Wrapper
from custom_sam_peft.tracking.base import Tracker
from custom_sam_peft.train.checkpoint import (
    ResumeState,
    load_full_state,
    save_adapter,
    save_full_state,
    save_merged,
)
from custom_sam_peft.train.loop import OomState, _box_hint_p, run_epoch
from custom_sam_peft.train.types import OomEvent
from custom_sam_peft.train.visualize import render_mask_panel

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    adapter_path: Path
    merged_path: Path | None
    final_metrics: MetricsReport | None  # None if end-of-run eval raises
    oom_events: tuple[OomEvent, ...] = ()


def _resolve_optimizer_name(cfg: TrainConfig) -> Optimizer:
    requested = cfg.train.optimizer
    if requested != "auto":
        return requested
    return "adamw8bit" if cfg.peft.method == "qlora" else "adamw"


def _build_optimizer(
    name: Optimizer, params: list[torch.nn.Parameter], lr: float
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "adamw8bit requires bitsandbytes. Install with: "
                "pip install 'custom-sam-peft[qlora]'"
            ) from e
        bnb_any: Any = bnb
        return bnb_any.optim.AdamW8bit(params, lr=lr)  # type: ignore[no-any-return]
    raise ValueError(f"unknown optimizer name: {name!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = cfg.train.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        if cfg.train.lr_schedule == "constant":
            return 1.0
        if cfg.train.lr_schedule == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(seed: int) -> Any:
    def init(worker_id: int) -> None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)

    return init


def _maybe_use_file_system_sharing(num_workers: int) -> str | None:
    """Switch torch multiprocessing sharing strategy to ``file_system``.

    PyTorch's Linux default is ``file_descriptor``: one FD per shared tensor
    storage between the DataLoader worker and the main process. With many
    workers shipping many tensors per sample, the per-process FD limit
    (commonly ``ulimit -n = 1024``) is easy to exhaust and surfaces as
    ``RuntimeError: unable to open shared memory object ... Too many open files``.

    Returns the new strategy if it was switched, else ``None``.
    """
    if num_workers <= 0:
        return None
    import torch.multiprocessing as torch_mp

    if torch_mp.get_sharing_strategy() != "file_descriptor":  # type: ignore[no-untyped-call]
        return None
    torch_mp.set_sharing_strategy("file_system")  # type: ignore[no-untyped-call]
    return "file_system"


class Trainer:
    def __init__(
        self,
        model: Sam3Wrapper,
        train_ds: Dataset,
        val_ds: Dataset,
        tracker: Tracker,
        cfg: TrainConfig,
    ) -> None:
        if cfg.data.prompt_mode == "bbox":
            raise ValueError(
                "prompt_mode='bbox' is not supported for training in v0; v0 trains "
                "text-only with optional GT-box hints sampled per-image. See "
                "logs/TODO.md for the deferred spec."
            )
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tracker = tracker
        self.cfg = cfg
        self._optimizer_name = _resolve_optimizer_name(cfg)
        if cfg.train.optimizer == "auto":
            _LOG.info(
                "optimizer=auto resolved to %s (peft.method=%s)",
                self._optimizer_name,
                cfg.peft.method,
            )

    def fit(self, *, run_dir: Path | None = None, resume_from: Path | None = None) -> RunResult:
        cfg = self.cfg
        _seed_everything(cfg.run.seed)

        if run_dir is None:
            from datetime import UTC, datetime

            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"

        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
        self.tracker.start_run(run_dir, cfg.model_dump(mode="json"), resume_from)

        device = next(self.model.parameters()).device
        pin = device.type == "cuda"
        new_strategy = _maybe_use_file_system_sharing(cfg.train.num_workers)
        if new_strategy is not None:
            _LOG.info(
                "torch mp sharing_strategy=%s (avoid EMFILE under many workers)", new_strategy
            )
        train_loader: DataLoader[Any] = DataLoader(
            self.train_ds,  # type: ignore[arg-type]
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=cfg.train.num_workers,
            pin_memory=pin,
            persistent_workers=cfg.train.num_workers > 0,
            worker_init_fn=_worker_init_fn(cfg.run.seed) if cfg.train.num_workers > 0 else None,
        )
        val_examples = [self.val_ds[i] for i in range(min(4, len(self.val_ds)))]

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = _build_optimizer(self._optimizer_name, trainable, cfg.train.lr)
        total_steps = cfg.train.epochs * max(len(train_loader), 1)
        scheduler = _build_scheduler(optimizer, cfg, total_steps)

        rs = ResumeState(
            start_step=0,
            start_epoch=0,
            nan_streak=0,
            box_hint_p=cfg.train.box_hint.p_start,
        )
        if resume_from is not None:
            rs = load_full_state(resume_from, self.model, optimizer, scheduler, cfg)
        global_step = rs.start_step
        nan_streak = rs.nan_streak
        start_epoch = rs.start_epoch

        class_names = self.train_ds.class_names
        oom_state = OomState(micro_batch_size=cfg.train.batch_size)

        def on_checkpoint(step: int, epoch: int, p_t: float, streak: int) -> None:
            state_dir = run_dir / "checkpoints" / f"step_{step}"
            save_full_state(
                state_dir=state_dir,
                wrapper=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=step,
                epoch=epoch,
                nan_streak=streak,
                box_hint_p=p_t,
                cfg=cfg,
            )
            self._log_image_panel(val_examples, class_names, step)

        def on_eval(step: int) -> None:
            try:
                lite_cfg = cfg.eval.model_copy(update={"mode": "lite", "save_predictions": False})
                report = Evaluator(lite_cfg).evaluate(self.model, self.val_ds)
                self.tracker.log_scalars(step, report.overall)
            except Exception:
                _LOG.warning("lite eval failed at step %d; skipping.", step, exc_info=True)

        full_report: MetricsReport | None = None
        merged_path: Path | None = None
        try:
            for epoch in range(start_epoch, cfg.train.epochs):
                total_batches = max(len(train_loader), 1)
                P.reset_inner(total=total_batches)
                global_step, nan_streak = run_epoch(
                    self.model,
                    train_loader,
                    optimizer,
                    scheduler,
                    self.tracker,
                    cfg,
                    run_dir,
                    epoch,
                    global_step,
                    nan_streak,
                    class_names,
                    self.val_ds,
                    on_checkpoint,
                    on_eval,
                    oom_state=oom_state,
                )
                P.advance_outer()

            adapter_path = run_dir / "adapter"
            save_adapter(self.model, adapter_path)
            if cfg.export.merge:
                merged_path = run_dir / "merged"
                save_merged(self.model, merged_path)

            full_report = Evaluator(cfg.eval).evaluate(self.model, self.val_ds)
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "overall": full_report.overall,
                        "per_class": full_report.per_class,
                        "n_images": full_report.n_images,
                        "n_predictions": full_report.n_predictions,
                        "global_step": global_step,
                        "epoch": cfg.train.epochs - 1,
                        "box_hint_p_final": _box_hint_p(global_step, cfg.train.box_hint),
                    },
                    indent=2,
                )
            )
        finally:
            self.tracker.close()

        return RunResult(
            run_dir=run_dir,
            adapter_path=run_dir / "adapter",
            merged_path=merged_path,
            final_metrics=full_report,
            oom_events=tuple(oom_state.pending_oom_events),
        )

    def _log_image_panel(
        self,
        val_examples: list[Any],
        class_names: list[str],
        global_step: int,
    ) -> None:
        if not val_examples:
            return
        self.model.eval()
        try:
            # Resolve the model's device once and move dataset images onto it
            # before each forward. See parallel rationale in
            # custom_sam_peft/eval/evaluator.py: dataset tensors are CPU; a CUDA model
            # otherwise crashes on the first Conv2d with a device-mismatch
            # RuntimeError. Falls back to CPU for parameterless / non-nn.Module
            # test stubs.
            try:
                device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
            with torch.no_grad():
                panels: list[np.ndarray[Any, Any]] = []
                for ex in val_examples:
                    if not ex.prompts.classes:
                        continue
                    c = ex.prompts.classes[0]
                    image = ex.image.permute(1, 2, 0).cpu().numpy()
                    image = (
                        (image - image.min()) / max(image.max() - image.min(), 1e-9) * 255
                    ).astype(np.uint8)
                    out = self.model(
                        ex.image.unsqueeze(0).to(device),
                        [ex.prompts.__class__(classes=[c])],
                        box_hints=None,
                    )
                    obj = out["pred_logits"].squeeze(-1).sigmoid().squeeze(0)
                    masks = out["pred_masks"].squeeze(0)
                    K = min(10, masks.shape[0])
                    top = torch.topk(obj, K).indices
                    sel = masks[top].sigmoid()
                    pred = (sel.max(dim=0).values >= 0.5).float().cpu().numpy()
                    if pred.shape != image.shape[:2]:
                        from torch.nn.functional import interpolate

                        pred_t = torch.tensor(pred)[None, None].float()
                        pred = interpolate(pred_t, size=image.shape[:2], mode="nearest")[
                            0, 0
                        ].numpy()
                    gt = [
                        inst.mask.cpu().numpy()
                        for inst in ex.instances
                        if class_names[inst.class_id] == c
                    ]
                    panels.append(render_mask_panel(image, gt, pred, class_name=c))
                if panels:
                    panel = np.concatenate(panels, axis=0)
                    self.tracker.log_images(global_step, {"val_panels": panel})
        except Exception:
            _LOG.warning(
                "_log_image_panel failed at step %d; skipping panel.", global_step, exc_info=True
            )
        finally:
            self.model.train()
