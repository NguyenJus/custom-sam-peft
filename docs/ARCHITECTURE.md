# custom_sam_peft Architecture

This is the one-page reference for how `custom_sam_peft` is wired together. The
full design rationale lives in `docs/superpowers/specs/`.

**Prompt invariant:** Text is the only prompt — the model takes one or more text (class) prompts and segments all matching instances. Training is text-only (the `box_hint` localization-hint curriculum was removed in #88). `SupportPrompts` is retained as a reserved extension seam (see [#126](https://github.com/NguyenJus/custom-sam-peft/issues/126) §12) for future hints (masks / points); it carries no fields today and is never used at inference.

## Module map

```text
src/custom_sam_peft/
  _registry.py         plugin registry: register(kind, name) + lookup
  config/
    schema.py          pydantic v2 — defaults + validation contract
    loader.py          load YAML + apply --override + resolve paths
  data/
    base.py            Example, Prompts (= TextPrompts), SupportPrompts, Dataset protocol
    coco.py / hf.py    @register("dataset", ...) adapters (call with pipeline + model_name kwargs)
    transforms.py      image + prompt augmentation
    collate.py         batch collator (variable-shape per image)
  models/
    sam3.py            HF SAM3.1 loader + forward wrapper
    losses.py          mask + box + objectness losses
  peft_adapters/
    lora.py / qlora.py @register("peft", ...) methods
  train/
    trainer.py         Trainer.fit() -> RunResult
    loop.py            inner step / epoch loop
    checkpoint.py      adapter + merged save/load
  eval/
    metrics.py         MetricsReport + COCO mAP
    evaluator.py       Evaluator.evaluate(model, dataset)
  tracking/
    base.py            Tracker protocol
    noop.py / tensorboard.py / wandb.py   @register("tracker", ...)
  cli/
    main.py            Typer entry point
    {train,eval,export,init,doctor}_cmd.py
```

## Data flow (one training run)

```text
my.yaml ─► config.loader.load() ─► TrainConfig
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
       build_dataset()           load_sam31()              build_tracker()
       (coco | hf)               (model + dtype)           (tb | wandb | none)
            │                          │
            │                   apply_lora / apply_qlora
            │                          │
            └──────────────┬───────────┘
                           ▼
         train.Trainer(model, train_ds, val_ds, tracker, cfg).fit()
                           │
                           ▼
              runs/{run_id}/adapter/        ← always
              runs/{run_id}/merged/         ← if cfg.export.merge
              runs/{run_id}/metrics.json    ← final eval
```

## Stable seams

These are the only interfaces a forking developer should expect to remain
stable across patch releases. Everything else is internal.

- `custom_sam_peft.data.base.Dataset` — `__len__`, `__getitem__(i) -> Example`, `class_names`.
- `custom_sam_peft.tracking.base.Tracker` — `log_scalars`, `log_images`, `close`.
- `custom_sam_peft.train.trainer.Trainer.fit() -> RunResult`.
- `custom_sam_peft.eval.evaluator.Evaluator.evaluate(model, dataset) -> MetricsReport`.

## Determinism and reproducibility

Gradient checkpointing and bitsandbytes contain non-deterministic CUDA kernels.
Resume reproducibility comes from RNG-state restore (Python `random`, NumPy, PyTorch CPU+CUDA), not from algorithmic determinism.
`torch.use_deterministic_algorithms` is intentionally left at the default (False) because enabling it would conflict with the above.
Bit-identical resume is therefore NOT guaranteed; the integration test `test_resume_matches_uninterrupted` asserts only finiteness and adapter weight preservation, not equality.

## Adding a new pluggable surface

The registry pattern is used for three kinds: `dataset`, `peft`, `tracker`.
A new implementation is one file plus a decorator:

```python
# src/custom_sam_peft/data/my_format.py
from custom_sam_peft._registry import register

@register("dataset", "my_format")
def build_my_format(cfg: dict) -> Dataset:
    return MyDataset(**cfg)
```

Plus one test that imports the module and calls `lookup("dataset", "my_format")`.
No edits to dispatch code.
