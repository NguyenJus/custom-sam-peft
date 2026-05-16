# esam3 Architecture

This is the one-page reference for how `esam3` is wired together. The
full design rationale lives in `docs/superpowers/specs/`.

## Module map

```
src/esam3/
  _registry.py         plugin registry: register(kind, name) + lookup
  config/
    schema.py          pydantic v2 — defaults + validation contract
    loader.py          load YAML + apply --override + resolve paths
  data/
    base.py            Example, Prompts (TextPrompts | BoxPrompts), Dataset protocol
    coco.py / hf.py    @register("dataset", ...) adapters
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

```
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

- `esam3.data.base.Dataset` — `__len__`, `__getitem__(i) -> Example`, `class_names`.
- `esam3.tracking.base.Tracker` — `log_scalars`, `log_images`, `close`.
- `esam3.train.trainer.Trainer.fit() -> RunResult`.
- `esam3.eval.evaluator.Evaluator.evaluate(model, dataset) -> MetricsReport`.

## Adding a new pluggable surface

The registry pattern is used for three kinds: `dataset`, `peft`, `tracker`.
A new implementation is one file plus a decorator:

```python
# src/esam3/data/my_format.py
from esam3._registry import register

@register("dataset", "my_format")
def build_my_format(cfg: dict) -> Dataset:
    return MyDataset(**cfg)
```

Plus one test that imports the module and calls `lookup("dataset", "my_format")`.
No edits to dispatch code.
