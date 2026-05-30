"""Trainer — public training entrypoint. Step body lives in train/loop.py."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from custom_sam_peft import paths
from custom_sam_peft.cli._progress import progress as P
from custom_sam_peft.config._duration import format_seconds, parse_duration_to_seconds
from custom_sam_peft.config.schema import Optimizer, TrainConfig
from custom_sam_peft.data.base import Dataset
from custom_sam_peft.data.collate import collate_batch
from custom_sam_peft.eval._artifacts import EvalArtifacts, TimeLimitStop
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.metrics import MetricsReport
from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, Sam3Wrapper
from custom_sam_peft.peft_adapters import PEFTMethod, make_peft_method
from custom_sam_peft.runtime import Runtime
from custom_sam_peft.tracking.base import Tracker
from custom_sam_peft.train.checkpoint import (
    ResumeState,
    load_full_state,
    save_adapter,
    save_full_state,
    save_merged,
)
from custom_sam_peft.train.loop import OomState, _TimeLimitReached, run_epoch
from custom_sam_peft.train.visualize import render_mask_panel

_LOG = logging.getLogger(__name__)


def _resolve_optimizer_name(cfg: TrainConfig, peft_method: PEFTMethod | None = None) -> Optimizer:
    requested = cfg.train.optimizer
    if requested != "auto":
        return requested
    _pm: PEFTMethod = peft_method if peft_method is not None else make_peft_method(cfg.peft.method)
    return _pm.recommended_optimizer()  # type: ignore[return-value]


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


def resolve_schedule_steps(
    *,
    save_every: int | None,
    eval_every: int | None,
    epochs: int,
    steps_per_epoch: int,
) -> tuple[int, int]:
    """Resolve None schedule fields to epoch-relative defaults.

    Called once ``steps_per_epoch = max(len(train_loader), 1)`` is known.
    Explicit (non-None) values are passed through unchanged for full
    backward compatibility.

    Resolution rules:
    - ``save_every``  → ``steps_per_epoch``
    - ``eval_every``  → ``steps_per_epoch``

    Returns:
        (save_every, eval_every) as resolved positive ints.
    """
    resolved_save = save_every if save_every is not None else steps_per_epoch
    resolved_eval = eval_every if eval_every is not None else steps_per_epoch
    return resolved_save, resolved_eval


class Trainer:
    def __init__(
        self,
        model: Sam3Wrapper,
        train_ds: Dataset,
        val_ds: Dataset | None,
        tracker: Tracker,
        cfg: TrainConfig,
        *,
        runtime: Runtime | None = None,
    ) -> None:
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tracker = tracker
        self.cfg = cfg
        self._peft_method: PEFTMethod = make_peft_method(cfg.peft.method)
        self._optimizer_name = _resolve_optimizer_name(cfg, self._peft_method)
        if cfg.train.optimizer == "auto":
            _LOG.info(
                "optimizer=auto resolved to %s (peft.method=%s)",
                self._optimizer_name,
                cfg.peft.method,
            )
        if val_ds is None:
            _LOG.info(
                "training without validation set; eval_every is a no-op, "
                "end-of-run eval and bundle samples are skipped."
            )
        # Best-model tracking state.
        self._best_metric_value: float = float("-inf")
        self._best_metric_key: str = "mAP"
        # Runtime injection (§2 seam discipline). If none provided, synthesize
        # one from cfg. Callers that pass runtime= get strict device isolation;
        # callers that don't pass it get a sensible default inferred from model
        # parameters and config. Wrapped in a broad except so that mock-heavy
        # unit tests (where cfg is a MagicMock) always succeed.
        if runtime is not None:
            self.runtime = runtime
        else:
            try:
                param_device = next(model.parameters()).device
                if not isinstance(param_device, torch.device):
                    raise TypeError("not a torch.device")
                inferred_device = str(param_device)
            except Exception:
                inferred_device = "cpu"
            try:
                raw_dtype = cfg.model.dtype
                dtype_str = str(raw_dtype) if isinstance(raw_dtype, str) else "float32"
            except Exception:
                dtype_str = "float32"
            try:
                self.runtime = Runtime.from_config(device=inferred_device, dtype=dtype_str)
            except Exception:
                self.runtime = Runtime.from_config(device="cpu", dtype="float32")

    # ------------------------------------------------------------------
    # Private helpers — decomposed from fit()
    # ------------------------------------------------------------------

    def _setup_run_dir(self, run_dir: Path | None) -> Path:
        """Create and initialise the run directory.

        config.yaml is written later in ``fit()``, after the epoch-relative
        schedule fields are resolved against the dataloader length.
        """
        cfg = self.cfg
        if run_dir is None:
            from datetime import UTC, datetime

            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"

        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        from custom_sam_peft.data.aug_presets import dump_augmentation_pipeline

        (run_dir / "augmentation_pipeline.json").write_text(
            json.dumps(
                dump_augmentation_pipeline(cfg.data.augmentations),
                indent=2,
                sort_keys=False,
            )
        )
        from custom_sam_peft.models.losses import dump_loss_bundle

        (run_dir / "loss_bundle.json").write_text(
            json.dumps(dump_loss_bundle(cfg.train.loss), indent=2, sort_keys=False)
        )
        return run_dir

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer from the resolved optimizer name and model params."""
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return _build_optimizer(self._optimizer_name, trainable, self.cfg.train.learning_rate)

    def _train_epoch(
        self,
        epoch: int,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        run_dir: Path,
        global_step: int,
        nan_streak: int,
        class_names: list[str],
        on_checkpoint: Any,
        on_eval: Any,
        oom_state: OomState | None = None,
        deadline: float | None = None,
    ) -> tuple[int, int]:
        """Run one training epoch; returns (global_step, nan_streak)."""
        return run_epoch(
            self.model,
            train_loader,
            optimizer,
            scheduler,
            self.tracker,
            self.cfg,
            run_dir,
            epoch,
            global_step,
            nan_streak,
            class_names,
            on_checkpoint,
            on_eval,
            peft_method=self._peft_method,
            runtime=self.runtime,
            oom_state=oom_state,
            deadline=deadline,
        )

    def _cap_eval_batch_size(self, bs: int, cap: int) -> int:
        """Return *bs* capped at *cap*, logging an INFO message if the cap fires."""
        if bs > cap:
            _LOG.info(
                "eval auto-batch capped at train batch (%d) — predictor picked %d",
                cap,
                bs,
            )
            return min(bs, cap)
        return bs

    def _eval_epoch(self, step: int, run_dir: Path, oom_state: OomState | None = None) -> None:
        """Run a periodic lite evaluation and log scalars to the tracker."""
        if self.val_ds is None:
            return
        try:
            cfg = self.cfg
            update: dict[str, object] = {"mode": "lite", "save_predictions": False}
            if cfg.eval.batch_size == "auto":
                from custom_sam_peft.presets import decide_eval_batch_size

                bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
                # Cap by the sticky train micro-batch size to avoid eval OOM.
                if oom_state is not None:
                    bs = self._cap_eval_batch_size(bs, oom_state.micro_batch_size)
                update["batch_size"] = bs
            lite_cfg = cfg.eval.model_copy(update=update)
            # Return training's freed-but-reserved activation pool to the CUDA
            # driver so the eval forward can allocate a contiguous block (#176).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report = Evaluator(lite_cfg).evaluate(self.model, self.val_ds)
            self.tracker.log_scalars(step, report.overall)
            self._maybe_save_best(report, step, run_dir)
        except RuntimeError as exc:
            if str(exc).startswith("eval OOM"):
                _LOG.error(
                    "validation eval ran out of GPU memory at batch_size=1 — "
                    "skipping eval for step %d.",
                    step,
                    exc_info=True,
                )
            else:
                _LOG.warning("lite eval failed at step %d; skipping.", step, exc_info=True)
        except Exception:
            _LOG.warning("lite eval failed at step %d; skipping.", step, exc_info=True)

    def _maybe_save_best(self, report: Any, step: int, run_dir: Path) -> None:
        """Save the adapter to run_dir/best/ when a new best mAP is observed.

        Failures are logged as warnings and swallowed so they never crash training.
        """
        try:
            metric = report.overall.get(self._best_metric_key)
            if metric is None:
                return
            if metric > self._best_metric_value:
                best_dir = run_dir / "best"
                adapter_dir = best_dir / "adapter"
                best_dir.mkdir(parents=True, exist_ok=True)
                save_adapter(self.model, adapter_dir)
                (best_dir / "best.json").write_text(
                    json.dumps(
                        {
                            "metric": self._best_metric_key,
                            "value": metric,
                            "global_step": step,
                        },
                        indent=2,
                    )
                )
                self._best_metric_value = metric
                _LOG.info(
                    "new best %s=%.4f at step %d; saved to %s",
                    self._best_metric_key,
                    metric,
                    step,
                    adapter_dir,
                )
        except Exception:
            _LOG.warning("failed to save best model at step %d; skipping.", step, exc_info=True)

    def _time_limited_artifacts(
        self,
        run_dir: Path,
        stop: Any,  # _TimeLimitReached
        budget_seconds: int | None,
        oom_state: OomState | None,
    ) -> EvalArtifacts:
        """Build the EvalArtifacts for a time-limited stop (spec §4.7).

        checkpoint_path points at the flushed step checkpoint's adapter
        (run_dir/adapter is intentionally NOT written on a stop).
        """
        assert budget_seconds is not None  # noqa: S101 — invariant: set whenever a stop fires
        checkpoint_dir = paths.checkpoint_path(run_dir, step=stop.step).parent / f"step_{stop.step}"
        best_dir: Path | None = None
        best_map: float | None = None
        best_candidate = run_dir / "best"
        best_json = best_candidate / "best.json"
        if best_candidate.is_dir() and best_json.is_file():
            try:
                data = json.loads(best_json.read_text())
                best_dir = best_candidate
                best_map = float(data["value"])
            except Exception:
                best_dir = None
                best_map = None
        time_limit_stop = TimeLimitStop(
            stop_step=stop.step,
            stop_epoch=stop.epoch,
            total_epochs=self.cfg.train.epochs,
            checkpoint_dir=checkpoint_dir,
            duration_label=format_seconds(budget_seconds),
            best_dir=best_dir,
            best_map=best_map,
        )
        return EvalArtifacts(
            checkpoint_path=checkpoint_dir / "adapter",
            peft_method=self.cfg.peft.method,
            run_dir=run_dir,
            final_metrics=None,
            oom_events=tuple(oom_state.pending_oom_events) if oom_state is not None else (),
            time_limit_stop=time_limit_stop,
        )

    def _maybe_checkpoint(
        self,
        step: int,
        epoch: int,
        nan_streak: int,
        run_dir: Path,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        class_names: list[str],
        val_examples: list[Any],
    ) -> None:
        """Save a full training-state checkpoint at the given step.

        save_full_state expects a ``step_<N>`` subdirectory; use the paths
        module to obtain the canonical checkpoints parent dir rather than
        string-joining it inline.
        """
        checkpoints_dir = paths.checkpoint_path(run_dir, step=step).parent
        state_dir = checkpoints_dir / f"step_{step}"
        save_full_state(
            state_dir=state_dir,
            wrapper=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=step,
            epoch=epoch,
            nan_streak=nan_streak,
            cfg=self.cfg,
        )
        self._log_image_panel(val_examples, class_names, step)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, *, run_dir: Path | None = None, resume_from: Path | None = None) -> EvalArtifacts:
        cfg = self.cfg
        _seed_everything(cfg.run.seed)

        run_dir = self._setup_run_dir(run_dir)

        new_strategy = _maybe_use_file_system_sharing(cfg.train.num_workers)
        if new_strategy is not None:
            _LOG.info(
                "torch mp sharing_strategy=%s (avoid EMFILE under many workers)", new_strategy
            )

        pin = self.runtime.device.type == "cuda"
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
        val_examples: list[Any] = (
            [] if self.val_ds is None else [self.val_ds[i] for i in range(min(4, len(self.val_ds)))]
        )

        # Resolve epoch-relative schedule defaults now that steps_per_epoch is known.
        steps_per_epoch = max(len(train_loader), 1)
        resolved_save, resolved_eval = resolve_schedule_steps(
            save_every=cfg.train.save_every,
            eval_every=cfg.train.eval_every,
            epochs=cfg.train.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        _LOG.info(
            "schedule resolved: save_every=%d eval_every=%d (steps_per_epoch=%d)",
            resolved_save,
            resolved_eval,
            steps_per_epoch,
        )
        resolved_train = cfg.train.model_copy(
            update={
                "save_every": resolved_save,
                "eval_every": resolved_eval,
            }
        )
        cfg = cfg.model_copy(update={"train": resolved_train})
        # Update self.cfg so that _train_epoch (which passes self.cfg to run_epoch)
        # always sees the resolved integer values — never None.
        self.cfg = cfg

        # Write config.yaml with resolved values so logs + artifacts reflect what
        # actually ran (must happen AFTER resolution so None fields are replaced).
        cfg_dict = cfg.model_dump(mode="json")
        vs_path = run_dir / "val_source.json"
        if vs_path.exists():
            saved = json.loads(vs_path.read_text())
            cfg_dict["val_source"] = {
                "mode": saved["mode"],
                "fraction_requested": saved.get("fraction_requested"),
                "realized_fraction": saved.get("realized_fraction"),
                "n_train": saved.get("n_train"),
                "n_val": saved.get("n_val"),
            }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
        self.tracker.start_run(run_dir, cfg_dict, resume_from)

        optimizer = self._build_optimizer()
        total_steps = cfg.train.epochs * steps_per_epoch
        scheduler = _build_scheduler(optimizer, cfg, total_steps)

        rs = ResumeState(
            start_step=0,
            start_epoch=0,
            nan_streak=0,
        )
        if resume_from is not None:
            rs = load_full_state(resume_from, self.model, optimizer, scheduler, cfg)
        global_step = rs.start_step
        nan_streak = rs.nan_streak
        start_epoch = rs.start_epoch

        class_names = self.train_ds.class_names
        from custom_sam_peft.models.sam3 import MULTIPLEX_CAP as _MULTIPLEX_CAP

        oom_state = OomState(
            micro_batch_size=cfg.train.batch_size,
            effective_K=min(cfg.train.multiplex.classes_per_forward, _MULTIPLEX_CAP),
        )

        def on_checkpoint(step: int, epoch: int, streak: int) -> None:
            self._maybe_checkpoint(
                step, epoch, streak, run_dir, optimizer, scheduler, class_names, val_examples
            )

        def on_eval(step: int) -> None:
            self._eval_epoch(step, run_dir, oom_state)

        deadline: float | None = None
        budget_seconds: int | None = None
        if cfg.train.time_limit is not None:
            budget_seconds = parse_duration_to_seconds(cfg.train.time_limit)
            deadline = time.monotonic() + budget_seconds
            _LOG.info(
                "time limit: %s (%ds) — stops at the first micro-step past the deadline",
                format_seconds(budget_seconds),
                budget_seconds,
            )

        merged_path: Path | None = None
        full_report: MetricsReport | None = None
        stop: _TimeLimitReached | None = None
        try:
            try:
                for epoch in range(start_epoch, cfg.train.epochs):
                    total_batches = max(len(train_loader), 1)
                    P.reset_inner(total=total_batches)
                    global_step, nan_streak = self._train_epoch(
                        epoch,
                        train_loader,
                        optimizer,
                        scheduler,
                        run_dir,
                        global_step,
                        nan_streak,
                        class_names,
                        on_checkpoint,
                        on_eval,
                        oom_state=oom_state,
                        deadline=deadline,
                    )
                    P.advance_outer()
            except _TimeLimitReached as e:
                stop = e
                global_step = e.step  # the flushed checkpoint's step

            if stop is None:
                adapter_path = run_dir / "adapter"
                save_adapter(self.model, adapter_path)
                if cfg.export.merge:
                    merged_path = run_dir / "merged"
                    save_merged(self.model, merged_path)

                if self.val_ds is not None:
                    full_eval_cfg = cfg.eval
                    if full_eval_cfg.batch_size == "auto":
                        from custom_sam_peft.presets import decide_eval_batch_size

                        bs, _, _ = decide_eval_batch_size(classes_per_forward=MULTIPLEX_CAP)
                        bs = self._cap_eval_batch_size(bs, oom_state.micro_batch_size)
                        full_eval_cfg = full_eval_cfg.model_copy(update={"batch_size": bs})
                    full_report = Evaluator(full_eval_cfg).evaluate(self.model, self.val_ds)
                if full_report is not None:
                    (run_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "overall": full_report.overall,
                                "per_class": full_report.per_class,
                                "n_images": full_report.n_images,
                                "n_predictions": full_report.n_predictions,
                                "global_step": global_step,
                                "epoch": cfg.train.epochs - 1,
                            },
                            indent=2,
                        )
                    )
                else:
                    (run_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "note": "no validation set provided",
                                "global_step": global_step,
                                "epoch": cfg.train.epochs - 1,
                            },
                            indent=2,
                        )
                    )
        finally:
            self.tracker.close()

        if stop is not None:
            return self._time_limited_artifacts(run_dir, stop, budget_seconds, oom_state)

        return EvalArtifacts(
            checkpoint_path=run_dir / "adapter",
            peft_method=self.cfg.peft.method,
            run_dir=run_dir,
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
            # Resolve device from the model's parameters so that dataset images
            # (always CPU) are moved onto the model's device before each forward.
            # Uses `model_dev` (not `device`) to avoid matching the §3 guard
            # that flags bare tensor-move sites outside runtime/ and data/collate.
            # The collator owns bulk-batch moves; this is a panel-render one-off.
            # Falls back gracefully for parameterless / non-nn.Module test stubs.
            try:
                model_dev = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                model_dev = torch.device("cpu")
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
                        ex.image.unsqueeze(0).to(model_dev),
                        [ex.prompts.__class__(classes=[c])],
                        support=None,
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
