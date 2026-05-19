# esam3 — Training Loop Design

**Status:** Draft (2026-05-17)
**Parent spec:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md) §11 step 5
**Scope:** Implements `train/trainer.py`, `train/loop.py`, `train/checkpoint.py`, plus a new `train/visualize.py`. Extends `models/sam3.py` with `box_hints` support and adds schema fields under `TrainHyperparams`. Depends on completed specs 1–4 (data, model, lora, qlora).

---

## 1. Goals

A single-device PyTorch training loop that finetunes SAM 3.1 on niche datasets using **text prompts only** at inference time, supervising on **masks + objectness + image-presence** (no box supervision, no box prediction emission). During training, ground-truth boxes are fed as an optional prompt-side localization hint with a Bernoulli probability `p(t)` that linearly decays from `p_start` to `p_end` over `decay_steps` global steps, so the model is weaned off the box crutch by the end of training.

### Behavioral targets

- Single-GPU runs on 12–16 GB VRAM with QLoRA + grad checkpointing.
- Reproducible resume from full state (model + optimizer + scheduler + RNG + box-hint p + step + epoch).
- bf16 / fp16 mixed precision via PyTorch autocast (LoRA path); QLoRA path relies on bnb internal compute dtype, no autocast wrapper.
- Skip non-finite micro-steps and abort after `nan_abort_after` consecutive failures.
- Per-class outer loop honors SAM 3.1's "one class per forward" contract; multiplex relaxation is mechanically future-proof.

### Non-goals (filed in §11)

- Evaluation call site (lands in spec/eval).
- `data.prompt_mode='bbox'` training (rejected at fit-time).
- Early-stopping callback.
- Multi-GPU / DDP / FSDP.
- Cosine / exponential box-hint schedules.
- Multiplex (multi-class-per-forward) forward path.

---

## 2. Module map

| File | Disposition | Purpose |
| --- | --- | --- |
| `src/esam3/train/trainer.py` | Implement | `Trainer.fit()` lifecycle: build optimizer / scheduler / dataloaders, drive epochs, save checkpoints, write `metrics.json`, return `RunResult`. ~150 LOC. |
| `src/esam3/train/loop.py` | Implement | `train_step()` per-batch class loop, autocast, grad accumulation, box-hint sampling, NaN policy. `run_epoch()` cadence (save_every / log_every / image panel). ~120 LOC. |
| `src/esam3/train/checkpoint.py` | Implement | `save_full_state` / `load_full_state` (training state) and `save_adapter` / `save_merged` / `load_adapter` dispatchers (LoRA vs QLoRA). ~80 LOC. |
| `src/esam3/train/visualize.py` | New | `render_mask_panel(image, gt_masks, pred_mask, class_name)` → image / GT-overlay / pred-overlay strip. ~40 LOC. Pure rendering. |
| `src/esam3/models/sam3.py` | Extend | `Sam3Wrapper.forward(images, prompts, box_hints=None)`; `_Sam3ImageAdapter.forward` threads hints into Meta's `geometric_prompt`; new `_build_geometric_prompt` helper. |
| `src/esam3/config/schema.py` | Extend | New `BoxHintSchedule` block under `TrainHyperparams.box_hint`; new `log_every`, `nan_abort_after` fields; new optional `Trainer`-consumed default for `LossConfig.w_box` and `MatcherWeights.lambda_l1` / `lambda_giou` (defaults flipped to `0.0`). |
| `src/esam3/data/base.py` | No change | Box hints sourced from existing `Instance.box`. |
| `configs/examples/coco_text_lora.yaml`, `coco_bbox_qlora.yaml` | Update | Drop box-loss weights, add `box_hint` block. Rename `coco_bbox_qlora.yaml` to `coco_text_qlora.yaml` (bbox prompt-mode no longer trains). |

---

## 3. Schema additions (`config/schema.py`)

```python
class BoxHintSchedule(_Strict):
    """Linear-decay schedule for the per-image probability of feeding GT
    boxes as a localization hint alongside the text prompt.

    p(t) = max(p_end, p_start + (p_end - p_start) * t / decay_steps)
    where t = global_step. Decayed once per training step and applied
    per-image via Bernoulli(p(t)) over each image's GT boxes for the
    currently-prompted class.

    early_stop_p_threshold is consumed by a future early-stopping
    mechanism (not by this spec): a run MUST NOT terminate early while
    current p(t) >= this value. Recorded here so the constraint is
    co-located with the schedule it gates.
    """
    p_start: float = Field(default=1.0, ge=0.0, le=1.0)
    p_end: float = Field(default=0.0, ge=0.0, le=1.0)
    decay_steps: PositiveInt = 5000
    early_stop_p_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_monotone(self) -> BoxHintSchedule:
        if self.p_end > self.p_start:
            raise ValueError(
                f"BoxHintSchedule must decay: p_end ({self.p_end}) > "
                f"p_start ({self.p_start})"
            )
        return self


class TrainHyperparams(_Strict):
    # ... existing fields unchanged ...
    box_hint: BoxHintSchedule = Field(default_factory=BoxHintSchedule)
    log_every: PositiveInt = 50           # micro-steps between tracker.log_scalars
    nan_abort_after: PositiveInt = 20     # consecutive non-finite micro-steps before raising
    num_workers: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 1),
        ge=0,
        description="DataLoader workers. 0 disables multiprocessing.",
    )
```

**`LossConfig` / `MatcherWeights` default flips.** `LossConfig.w_box` default becomes `0.0`; `MatcherWeights.lambda_l1` and `lambda_giou` defaults become `0.0`. Existing fields are not removed — users who later want box supervision can override. The matcher becomes mask-only by default for v0.

**Runtime policies** (not schema constraints):

- **QLoRA optimizer coercion.** In `Trainer.__init__`: if `cfg.peft.method == "qlora"` and `cfg.train.optimizer == "adamw"` (the schema default), the trainer logs `"qlora + adamw default → switching to adamw8bit"` and binds the bnb optimizer. A pinned non-default `optimizer` value is honored verbatim. Detection that the value is "default" is done by re-instantiating `TrainHyperparams(**{k: v for k, v in cfg.train.model_dump().items() if k != "optimizer"})` and comparing — i.e., we can't tell schema-default from user-typed-`adamw`, so the coercion uses a sentinel: a new `Optimizer | Literal["auto"] = "auto"` value on `TrainHyperparams.optimizer`, with `"auto"` meaning "trainer picks". Default schema value flips from `"adamw"` to `"auto"`.
- **`prompt_mode='bbox'` rejection.** In `Trainer.__init__`: if `cfg.data.prompt_mode == "bbox"`, raise `ConfigError("prompt_mode='bbox' is not supported for training in v0; v0 trains text-only with optional GT-box hints sampled per-image. See logs/TODO.md for the deferred spec.")`. Schema literal stays `text|bbox` so non-training tools still validate cleanly.

---

## 4. `Sam3Wrapper` extension (`models/sam3.py`)

### 4.1 Signature

```python
class Sam3Wrapper(nn.Module):
    def forward(
        self,
        images: Tensor,                            # (B, 3, H, W)
        prompts: list[Prompts],                    # len B
        box_hints: list[Tensor | None] | None = None,  # len B; each (M_i, 4) xyxy pixel, or None
    ) -> dict[str, Any]: ...
```

### 4.2 Validation (renamed `_validate_inputs`)

1. Existing rules: `images.ndim == 4`; `len(prompts) == B`; uniform prompt variant; `TextPrompts.classes` length exactly 1.
2. `box_hints is not None` → `len(box_hints) == B`; each entry is either `None` or a `(M_i, 4)` float tensor in **xyxy pixel coords at `image_size`**.
3. `box_hints` is only valid alongside `TextPrompts`. Combining with `BoxPrompts` raises (`BoxPrompts` already carries boxes; hinting is undefined there).

**Coordinate-space contract.** `box_hints[i]` arrives in **xyxy pixel coordinates at the wrapper's `image_size`** (1008 by default). This matches `Instance.box`, which is xyxy in the *post-augmentation* image space — boxes ride through the Albumentations pipeline alongside the image (per the data-loading spec), so they already live in the resized image's pixel coordinate system. The trainer does not rescale.

### 4.3 `_Sam3ImageAdapter.forward` and `_build_geometric_prompt`

The adapter:

1. Runs the backbone once: `backbone_out = self.model.image_encoder(images)`.
2. Tokenizes the per-image class name into Meta's `find_input` / `find_target`.
3. Calls `geometric_prompt = _build_geometric_prompt(box_hints, image_size, device)` (returns `None` when no hints are present, else a tensor in Meta's expected layout).
4. Calls `self.model.forward_grounding(backbone_out, find_input, find_target, geometric_prompt)`.

**`_build_geometric_prompt(box_hints, image_size, device)` is the single point of contact** for Meta's `geometric_prompt` tensor layout. The exact shape is **pinned in Implementation Plan Step 0** (see §10). The function is documented with the canonical layout so future Meta-side renames touch one place.

### 4.4 Fallback if Meta's slot is incompatible

If Plan Step 0 (slot inspection) reveals that `forward_grounding`'s `geometric_prompt` slot cannot accept box prompts in the text path, the implementation **halts** and asks the spec owner before proceeding. There is no silent fallback to "no hints"; either the curriculum ships as designed or the spec is renegotiated.

---

## 5. `Trainer.fit()` (`train/trainer.py`)

```python
class Trainer:
    def __init__(
        self,
        model: Sam3Wrapper,
        train_ds: Dataset,
        val_ds: Dataset,
        tracker: Tracker,
        cfg: TrainConfig,
    ) -> None: ...

    def fit(self, resume_from: Path | None = None) -> RunResult: ...
```

### 5.1 Lifecycle

1. **Pre-flight guards:** reject `cfg.data.prompt_mode == "bbox"`; coerce `cfg.train.optimizer == "auto"` based on `cfg.peft.method` (qlora → `adamw8bit`, lora → `adamw`).
2. **Run dir:** `runs/{cfg.run.name}-{YYYYMMDD-HHMMSS}/`. Create subdirs `adapter/`, `checkpoints/`. Write resolved `config.yaml` (post-override merge) at the top level.
3. **Seeding:** seed `random`, `numpy`, `torch.{cpu,cuda}`, set `torch.use_deterministic_algorithms(False)` (gradient checkpointing + bnb both have non-deterministic kernels; full determinism is unrealistic). The deterministic-resume invariant uses RNG state restore, not algorithm determinism.
4. **DataLoaders:** `DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=cfg.train.num_workers, pin_memory=<true on cuda>, persistent_workers=<true if num_workers > 0>, worker_init_fn=<seeds each worker from cfg.run.seed + worker_id>)`. Same for `val_ds` with `shuffle=False`. `num_workers` schema default: `min(4, os.cpu_count() or 1)` resolved at schema-instantiation time via a `default_factory`.
5. **Optimizer:** built over trainable params (`p for p in model.parameters() if p.requires_grad`). `adamw` → `torch.optim.AdamW`; `adamw8bit` → `bitsandbytes.optim.AdamW8bit` (lazy-imported with the same helpful ImportError pattern as `peft_adapters/qlora.py`).
6. **LR scheduler:** linear warmup for `cfg.train.warmup_steps`, then `cfg.train.lr_schedule` (`constant` / `cosine` / `linear`) over `total_steps - warmup_steps`. `total_steps = epochs * ceil(len(train_loader) / grad_accum_steps)`. Single composed `LambdaLR`.
7. **Resume:** if `resume_from is not None`, call `load_full_state(resume_from, wrapper, optimizer, scheduler)` → `ResumeState(start_step, start_epoch, nan_streak, box_hint_p)`. Otherwise zero-initialize those values.
8. **Epoch loop:**

   ```python
   class_names = train_ds.class_names      # propagated to train_step for dense-id lookup
   for epoch in range(start_epoch, cfg.train.epochs):
       global_step, nan_streak = run_epoch(
           model, train_loader, optimizer, scheduler, tracker,
           cfg, run_dir, epoch, global_step, nan_streak,
           class_names, val_ds,
       )
   ```

   `class_names` is the **train** split's `class_names` property; the dataset spec guarantees both splits share the same dense id space when reading the same COCO annotations.
9. **Final adapter:** call `save_adapter(model, run_dir / "adapter")`. Dispatch on LoRA vs QLoRA via Linear4bit-presence detection.
10. **Optional merge:** if `cfg.export.merge`, call `save_merged(model, run_dir / "merged")` (calls `merge_lora` then dumps the resulting module).
11. **`metrics.json`:** dump the final scalar window (the last `log_every` mean values) plus `{"global_step": ..., "epoch": ..., "box_hint_p_final": ...}`. **`final_metrics: None`** in `RunResult` because eval is deferred (spec/eval will overwrite this behavior).
12. **`tracker.close()`** in a `try/finally` so a mid-training exception still flushes the backend.
13. **Return `RunResult(run_dir, adapter_path, merged_path, final_metrics=None)`.**

### 5.2 `RunResult` (unchanged shape, defined in `trainer.py`)

```python
@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    adapter_path: Path
    merged_path: Path | None
    final_metrics: MetricsReport | None   # always None in this spec; spec/eval sets it
```

---

## 6. Step body (`train/loop.py`)

### 6.1 `train_step` — per-batch class-loop with per-class backward

```python
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
    images = batch["images"].to(device)
    prompts: list[Prompts] = batch["prompts"]
    targets: list[list[Instance]] = batch["instances"]
    B = images.shape[0]
    p_t = _box_hint_p(global_step, cfg.train.box_hint)

    classes_in_batch = sorted({c for p in prompts for c in p.classes})
    if not classes_in_batch:
        # Defensive: all images in batch had empty prompts.classes. Data layer
        # should never produce this for COCO (crowd-only images are dropped at
        # ingest), but HF datasets might. Log + skip this batch without bumping
        # nan_streak — this is a data condition, not a numerical failure.
        logger.warning("train_step: batch has no class prompts; skipping")
        return StepResult.empty(global_step=global_step, p_t=_box_hint_p(global_step, cfg.train.box_hint))

    accum = {"mask": 0.0, "obj": 0.0, "presence": 0.0, "total": 0.0}
    finite_class_count = 0
    n_hint_applied = 0

    for c in classes_in_batch:
        prompts_c = [TextPrompts(classes=[c])] * B
        c_dense = class_names.index(c)
        targets_c = [
            [inst for inst in targets[i] if inst.class_id == c_dense]
            for i in range(B)
        ]
        hints_c: list[Tensor | None] = []
        for i in range(B):
            if targets_c[i] and random.random() < p_t:
                hints_c.append(torch.stack([inst.box for inst in targets_c[i]]).to(device))
                n_hint_applied += 1
            else:
                hints_c.append(None)

        with _autocast_ctx(cfg):
            out = model(images, prompts_c, box_hints=hints_c)
            losses = total_loss(out, targets_c, cfg.train.loss)

        scaled = losses["total"] / (len(classes_in_batch) * cfg.train.grad_accum_steps)
        if torch.isfinite(scaled):
            scaled.backward()
            finite_class_count += 1
            for k in ("mask", "obj", "presence", "total"):
                accum[k] += float(losses[k].detach())

    skipped = finite_class_count == 0
    nan_streak = nan_streak + 1 if skipped else 0
    if nan_streak >= cfg.train.nan_abort_after:
        raise RuntimeError(
            f"Training aborted: {nan_streak} consecutive non-finite micro-steps."
        )

    grad_norm = None
    if (global_step + 1) % cfg.train.grad_accum_steps == 0 and not skipped:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.train.max_grad_norm,
        ).item()
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
        nan_streak=nan_streak,
        images_processed=B,
    )
```

### 6.2 Memory invariant — why per-class backward

Each class-forward retains its own activations. Accumulating losses and backwarding once would hold `C × forward` activations in memory. Calling `.backward()` after each class-forward frees those activations immediately while accumulating gradients into `.grad` buffers; the optimizer step still happens every `grad_accum_steps` micro-steps. Net memory is `O(forward)` regardless of `C`. Critical on a 12-16 GB GPU.

### 6.3 NaN policy

- A **single** non-finite class-forward is skipped (no backward) but does not by itself count as a "skipped step".
- A **micro-step** is "skipped" only when *every* class-forward in the union was non-finite (`finite_class_count == 0`).
- `nan_streak` increments on skipped micro-steps and resets on the first finite one.
- `cfg.train.nan_abort_after` consecutive skipped micro-steps → `RuntimeError`. Default `20`.

### 6.4 Autocast policy

```python
def _autocast_ctx(cfg: TrainConfig):
    if cfg.peft.method == "qlora":
        # bnb Linear4bit handles compute_dtype internally; autocast wrapper is
        # not needed and can degrade numerics.
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
    )
```

No `GradScaler`: bf16 doesn't need it, and fp16 + LoRA is a degraded path users opt into knowingly.

### 6.5 `run_epoch` cadence

```python
def run_epoch(model, loader, optimizer, scheduler, tracker, cfg, run_dir,
              global_step, nan_streak, val_ds) -> tuple[int, int]:
    window = _ScalarWindow()
    for batch in loader:
        result = train_step(...)
        window.update(result)
        global_step += 1
        if global_step % cfg.train.log_every == 0:
            tracker.log_scalars(global_step, window.flush())
        if global_step % cfg.train.save_every == 0:
            _save_checkpoint(model, optimizer, scheduler, global_step,
                             epoch, nan_streak, result.p_t, run_dir, cfg)
            _log_image_panel(model, val_ds, class_names, global_step, tracker)
        nan_streak = result.nan_streak
    return global_step, nan_streak
```

`_ScalarWindow` accumulates the keys listed in §6.7 over `log_every` micro-steps and emits mean-of-finite values.

### 6.6 Box-hint schedule

```python
def _box_hint_p(global_step: int, cfg: BoxHintSchedule) -> float:
    if global_step >= cfg.decay_steps:
        return cfg.p_end
    frac = global_step / cfg.decay_steps
    return cfg.p_start + (cfg.p_end - cfg.p_start) * frac
```

(Note `p_end < p_start` so the interpolation decreases monotonically; the `max(p_end, ...)` clamp from §3 is implicit because we early-return `p_end` past `decay_steps`.)

### 6.7 Scalar log payload (per `log_every` window)

```python
{
  "loss/total":        mean,
  "loss/mask":         mean,
  "loss/obj":          mean,
  "loss/presence":     mean,
  "lr":                scheduler.get_last_lr()[0],
  "box_hint/p":        result.p_t,           # value at last micro-step in window
  "box_hint/applied":  mean(n_hint_applied / (n_classes * B)),
  "grad_norm":         mean(grad_norm where not None),
  "throughput/img_s":  sum(images) / wall_time,
  "skipped_steps":     cumulative_skipped,
}
```

---

## 7. Checkpoint, resume, run-dir layout (`train/checkpoint.py`)

### 7.1 Run-dir layout

```text
runs/coco-cats-lora-20260518-143012/
├── config.yaml                      # frozen resolved TrainConfig (post-override)
├── adapter/                         # final LoRA/QLoRA adapter
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── esam3_qlora.json             # only if method=qlora
├── merged/                          # final merged base (only if cfg.export.merge)
├── checkpoints/
│   ├── step_1000/
│   │   ├── adapter/
│   │   └── training_state.pt
│   └── step_2000/...
├── metrics.json                     # final scalar dump; eval omitted in v0
└── train.log                        # logging.getLogger("esam3") mirror
```

### 7.2 `save_full_state`

```python
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
    adapter_dir = state_dir / "adapter"
    save_adapter(wrapper, adapter_dir)   # dispatches LoRA vs QLoRA
    torch.save(
        {
            "format_version": 1,
            "global_step": global_step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            "box_hint_p": box_hint_p,
            "nan_streak": nan_streak,
            "peft_method": cfg.peft.method,
            "cfg_hash": _hash_cfg(cfg),
        },
        state_dir / "training_state.pt",
    )
```

`_hash_cfg` is a sha256 of the canonical-JSON dump of `cfg.model_dump()`.

### 7.3 `load_full_state`

```python
@dataclass(frozen=True)
class ResumeState:
    start_step: int
    start_epoch: int
    nan_streak: int
    box_hint_p: float

def load_full_state(
    state_dir: Path,
    wrapper: Sam3Wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: TrainConfig,
) -> ResumeState: ...
```

1. Read `training_state.pt`.
2. Detect adapter type from sibling `adapter/` (presence of `esam3_qlora.json` → qlora). Cross-check against `state["peft_method"]`; mismatch is a hard error.
3. Apply adapter via `load_lora` or `load_qlora` to the (already-loaded) wrapper.
4. `optimizer.load_state_dict(state["optimizer"])`; `scheduler.load_state_dict(state["scheduler"])`.
5. Restore RNG states (CPU + CUDA where present).
6. `cfg_hash` mismatch → log a warning naming the divergent top-level keys (e.g., "lr changed: old=1e-4, new=5e-5"). Don't fail; resumes with tweaks are legitimate.
7. **Granularity choice (re-walk interrupted epoch):**
   - `start_epoch = state["epoch"]` — the trainer's outer loop re-enters the interrupted epoch from its start.
   - `start_step = state["global_step"]` — the global step counter resumes at the saved value. The next optimizer step increments it to `state["global_step"] + 1`.
   - RNG-state restore means the dataloader replays the same order it did the first time, so examples between the original checkpoint-step and the original epoch-end get re-trained. The model receives *new* gradient updates over the re-walked data (it is not a no-op). This is a deliberate trade: bounded by `save_every`, fully deterministic, no per-batch position to persist. A resumed run is *not* bit-identical to an uninterrupted one — it is "deterministic given the save point", which is the strongest reproducibility guarantee we can offer without saving dataloader iteration state.
8. Return `ResumeState(start_step, start_epoch, nan_streak, box_hint_p)`.

### 7.4 Adapter-only dispatchers

```python
def _has_linear4bit(wrapper: Sam3Wrapper) -> bool:
    """True if wrapper's PEFT base contains any bnb.nn.Linear4bit module.

    Lazy-imports bitsandbytes; returns False on ImportError (LoRA-only
    builds shouldn't depend on bnb being installed)."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    assert wrapper.peft_model is not None
    return any(isinstance(m, bnb.nn.Linear4bit) for m in wrapper.peft_model.modules())


def save_adapter(wrapper: Sam3Wrapper, path: Path) -> None:
    """LoRA vs QLoRA detection by Linear4bit-presence."""
    if wrapper.peft_model is None:
        raise RuntimeError("save_adapter: wrapper has no PeftModel")
    if _has_linear4bit(wrapper):
        save_qlora(wrapper, path)
    else:
        save_lora(wrapper, path)


def load_adapter(wrapper: Sam3Wrapper, path: Path) -> Sam3Wrapper:
    """LoRA vs QLoRA detection by esam3_qlora.json presence at `path`."""
    if (path / "esam3_qlora.json").exists():
        return load_qlora(wrapper, path)
    return load_lora(wrapper, path)


def save_merged(wrapper: Sam3Wrapper, path: Path) -> None:
    """merge_lora then dump base state_dict to `path / 'pytorch_model.bin'`.

    QLoRA merging dequantizes the base to compute_dtype during folding;
    the resulting module is no longer 4-bit-quantized."""
```

---

## 8. Image logging (`train/visualize.py`)

```python
def render_mask_panel(
    image: np.ndarray,           # (H, W, 3) uint8, un-normalized
    gt_masks: list[np.ndarray],  # (H, W) bool, all GT instances for the viz class
    pred_mask: np.ndarray,       # (H, W) float in [0,1], top-K merged
    class_name: str,
) -> np.ndarray:                 # (H, 3*W, 3) uint8 — image | GT-overlay | pred-overlay
```

**Cadence.** Every `cfg.train.save_every` global steps (immediately after the checkpoint write). Fixed mini-val slice `val_ds[0:min(4, len(val_ds))]`, chosen at `Trainer.__init__`. Same examples every panel so progress is visible run-to-run.

**Class selection.** For each example, use the first class name in `example.prompts.classes`. Deterministic — no RNG. Examples whose `prompts.classes` is empty are skipped from the panel slice (the panel composes whichever of the up-to-4 examples have at least one class).

**Prediction merge.** Run in `model.eval()` + `torch.no_grad()`, *no box hints* (pure-text inference is the target capability). Take top-K queries by `obj_logits.sigmoid()` with `K=10` (hard-coded for v0), threshold each mask at 0.5, union to a single `(H, W)` float mask via max. Restore `model.train()` after.

**Composition.** Concatenate the up-to-4 example panels vertically into one tall ndarray and pass to `tracker.log_images(global_step, {"val_panels": panel})`.

---

## 9. Error handling

| Condition | Behavior |
| --- | --- |
| `cfg.data.prompt_mode == "bbox"` | `ConfigError` from `Trainer.__init__` with the documented message. |
| `cfg.peft.method == "qlora"` and bnb missing | Lazy import in `apply_qlora` / `AdamW8bit` factory raises with the install hint already present in `peft_adapters/qlora.py`. Trainer doesn't need to duplicate. |
| Non-finite class-forward | Skip that class's backward; don't update accum. |
| Non-finite micro-step (all classes) | Increment `nan_streak`. Log `WARNING` with first-non-finite component. |
| `nan_streak >= nan_abort_after` | `RuntimeError`. `tracker.close()` still runs via `try/finally`. |
| `resume_from` points to a dir with no `training_state.pt` | `FileNotFoundError` with a hint pointing at expected `checkpoints/step_*/training_state.pt`. |
| `peft_method` mismatch between `training_state.pt` and adapter dir | `RuntimeError`. |
| `cfg_hash` mismatch on resume | `WARNING`. Names the divergent top-level keys. Run proceeds. |
| `cfg.export.merge` and method=qlora | Allowed. `save_merged` dequantizes base via `merge_lora` (already implemented in `peft_adapters/lora.py`). |

---

## 10. Implementation plan (numbered)

This section is consumed by the writing-plans skill.

**Step 0 (verification, blocking).** Inspect Meta's `Sam3Image.forward_grounding` source and pin the `geometric_prompt` tensor layout: shape, dtype, coordinate space (pixel vs normalized; cxcywh vs xyxy), padding convention for "no hint" entries. Write a unit test `tests/unit/test_geometric_prompt_builder.py` that constructs the expected tensor for {all-None, all-hinted, mixed} cases and asserts shape/dtype. If Meta's slot is incompatible with text-path prompting, halt and renegotiate the spec.

**Step 1.** Schema additions (§3): `BoxHintSchedule`, `box_hint`, `log_every`, `nan_abort_after`, `num_workers`; flip `LossConfig.w_box` / `MatcherWeights.lambda_l1` / `lambda_giou` defaults to `0.0`; change `TrainHyperparams.optimizer` default to `"auto"`. Update `Optimizer` literal to `"adamw" | "adamw8bit" | "auto"`. Unit tests for validators (`p_end > p_start` rejected; defaults shape).

**Step 2.** Extend `Sam3Wrapper.forward` with `box_hints`; implement `_build_geometric_prompt`; update `_validate_inputs`. Unit tests for the validator and builder.

**Step 3.** Implement `train/checkpoint.py`: `save_adapter`, `load_adapter`, `save_merged`, `save_full_state`, `load_full_state`, `ResumeState`. Unit roundtrip test on a stub wrapper.

**Step 4.** Implement `train/visualize.py: render_mask_panel`. Unit test for output shape/dtype/no-NaN.

**Step 5.** Implement `train/loop.py`: `_box_hint_p`, `_autocast_ctx`, `_ScalarWindow`, `train_step`, `run_epoch`. Unit tests for: class-loop dispatch, box-hint sampling under patched RNG, NaN policy (single-class skip vs full skip vs abort), and `_box_hint_p` math.

**Step 6.** Implement `train/trainer.py`: `Trainer.__init__` guards (bbox rejection, qlora coercion), run-dir creation, dataloader / optimizer / scheduler construction, epoch driver, `fit()` end-to-end, `metrics.json` writer. Unit tests for the guards.

**Step 7.** Integration test `tests/integration/test_train_end_to_end.py` (CPU + stub wrapper): full fit on `tiny_coco`, layout assertions.

**Step 8.** Integration test `tests/integration/test_train_resume.py`: save mid-run, resume, assert identical end-state.

**Step 9.** Update example configs (`configs/examples/coco_text_lora.yaml`, rename `coco_bbox_qlora.yaml` → `coco_text_qlora.yaml`); update `README.md` quickstart and `ARCHITECTURE.md` to mention the box-hint curriculum and the v0 text-only training scope.

**Step 10 (manual, not in CI).** GPU smoke `tests/gpu/test_real_train_overfits.py`: 50-step LoRA overfit on `tiny_coco` with the box-hint curriculum; assert `loss/total` drops by ≥30%. QLoRA parameterized variant.

---

## 11. Out of scope / deferred (TODO entries)

Each entry below is added to `logs/TODO.md` as part of Plan Step 1.

1. **Eval call site.** Trainer never calls `Evaluator.evaluate()`. `cfg.train.eval_every` is dormant. `spec/eval` will (a) build / inject an `Evaluator`, (b) call it at `eval_every` boundaries, (c) populate `RunResult.final_metrics` and `metrics.json`.
2. **`prompt_mode='bbox'` training.** Rejected at fit-time. File `spec/bbox-prompt-training` for the future bbox-as-primary-prompt path.
3. **Early-stopping callback.** Whatever future spec adds it MUST gate termination on `current_box_hint_p < cfg.train.box_hint.early_stop_p_threshold`. Documented; not implemented.
4. **Multi-GPU / DDP / FSDP.** Deferred to a Ray Train spec. Single-device assumption is localized to `_resolve_device()` and `wrapper.to(device)`.
5. **Multiplex (multi-class-per-forward).** Forward-compat invariants to preserve so the migration stays mechanical. The C=1 assumption today lives in **three** colocated places, all of which relax cleanly:
   - `Sam3Wrapper._validate_inputs` enforces `len(classes) == 1` per image. Multiplex weakens this to `len(classes) >= 1` — same shape, weaker bound.
   - `train_step` builds `prompts_c = [TextPrompts(classes=[c])] * B` inside its outer class loop. Multiplex replaces the outer loop with a single forward passing `[TextPrompts(classes=classes_in_batch)] * B`.
   - `box_hints: list[Tensor | None]` is the C=1 slice of the future `list[list[Tensor | None]]` shape — extension is a dimension growth, not a rename. The `_build_geometric_prompt` helper is the single tensor-layout point of contact.
   - Image panel composition extends by adding columns per class.
6. **Cosine / exponential box-hint schedules.** Schema accepts linear only today; add `BoxHintSchedule.shape: Literal["linear", "cosine", "exp"]` when needed.
7. **Image-panel K parameter.** Hard-coded `K=10` top-query merge for prediction visualization; promote to config if it ever matters.
8. **Determinism flag.** `torch.use_deterministic_algorithms` is left off because gradient checkpointing + bnb both have non-deterministic kernels. Resume reproducibility comes from RNG-state restore, not algorithmic determinism. Document this clearly in `ARCHITECTURE.md`.

---

## 12. Testing strategy

### 12.1 Unit (`tests/unit/`, fast, CPU, every commit)

| Test | Asserts |
| --- | --- |
| `test_box_hint_schedule.py` | `_box_hint_p` math at t=0, t=decay_steps/2, t=decay_steps, t>>decay_steps; validator rejects `p_end > p_start`. |
| `test_train_step_class_loop.py` | Stub wrapper records forwards. Batch of {img0: classes=[A,B], img1: classes=[A]} → exactly 2 wrapper calls; per-class targets filtered by `class_id`. |
| `test_train_step_box_hint_sampling.py` | Patched `random.random()` sequence drives Bernoulli; assert hints[i] is GT-stack or None per the patched flips at current `p_t`. |
| `test_train_step_nan_policy.py` | Stub returns NaN for one class → that class skipped, no abort; stub returns NaN for all classes in N consecutive micro-steps → `RuntimeError` at N=nan_abort_after; streak resets on first finite step. |
| `test_trainer_rejects_bbox_mode.py` | `prompt_mode='bbox'` → `ConfigError` from `__init__`. |
| `test_trainer_qlora_optimizer_coercion.py` | qlora + optimizer="auto" → `adamw8bit`; lora + "auto" → `adamw`; qlora + "adamw" → `adamw` (honored). |
| `test_checkpoint_save_load_roundtrip.py` | Build trainer, run 2 steps, save_full_state, load_full_state into fresh trainer; assert step/epoch/rng/optimizer/box_hint_p match. QLoRA variant `xfail` without bitsandbytes. |
| `test_run_dir_layout.py` | After `fit()` on stub: `config.yaml`, `adapter/`, `metrics.json`, `train.log`, expected `checkpoints/step_*` subdirs. |
| `test_geometric_prompt_builder.py` | All-None → sentinel; mixed → padded tensor in Meta's documented layout; pixel-coord preserved. |
| `test_visualize_panel.py` | `(H, 3*W, 3)` uint8, no NaN, empty-GT case handled. |

### 12.2 Integration (`tests/integration/`, `@pytest.mark.integration`, CPU + stub)

- `test_train_end_to_end.py`: `tiny_coco` + stub wrapper + LoRA, 4 steps, `save_every=2`, `epochs=1` → expected layout, parseable `metrics.json`, ≥1 checkpoint, finite scalars, panel written.
- `test_train_resume.py`: 4 steps, save state at step 2, kill, resume; assert resumed run produces the same end-state as 4 uninterrupted steps with the same seed.

### 12.3 GPU smoke (`tests/gpu/test_real_train_overfits.py`, `@pytest.mark.gpu`, manual / nightly)

50-step LoRA overfit on `tiny_coco` with `box_hint.p_start=1.0, p_end=0.0, decay_steps=25`. Assert `loss/total` at step 50 ≤ 0.7 × `loss/total` at step 1. QLoRA parameterized variant, skipped without bitsandbytes.

### 12.4 Coverage gate

Maintain ≥80% coverage on `src/esam3/train` after this spec lands.
