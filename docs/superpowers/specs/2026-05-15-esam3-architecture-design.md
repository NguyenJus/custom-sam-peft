# esam3 — Architecture & Scaffolding Design

**Status:** Approved (2026-05-15)
**Scope:** Overarching architecture for the `efficient-sam3-finetuning` package, plus the v0 scaffolding plan. Subsystem implementations are deferred to per-subsystem specs (see Section 8).

---

## 1. Goals & v0 Scope

A Python package for parameter-efficient finetuning of SAM3.1 on niche image instance-segmentation datasets, runnable on a single consumer GPU.

**v0 scope (locked):**

| Dimension | v0 | Deferred |
|---|---|---|
| Model | SAM3.1 only | SAM3 |
| Input prompts | Text, bounding boxes | Points, masks |
| Data | Static images | Video |
| Output | Instance segmentation | Semantic segmentation |
| Distribution | Single machine | Ray Train, Argo workflows |
| PEFT methods | LoRA, QLoRA | IA3, prefix tuning, others |
| Dataset formats | COCO instance JSON, HuggingFace `datasets` | Custom adapters |
| Tracking | Pluggable: TensorBoard, W&B, none | — |
| Hardware target | 12–16GB consumer GPU | — |
| Interface | CLI + Python library (library is source of truth) | — |
| License | Apache-2.0 | — |

---

## 2. Architectural Approach

Thin custom PyTorch training loop wrapping a `SAM3FinetuneModule` (model + LoRA + losses). No `transformers.Trainer`, no PyTorch Lightning. Rationale: SAM3.1's training contract (text/box prompts → instance masks + boxes + objectness losses) doesn't fit Trainer's classification/seq2seq assumptions, and Lightning's distribution layer competes with Ray Train, which is a future requirement.

HuggingFace `peft` provides LoRA injection. HuggingFace `datasets` is one supported dataset format. `pycocotools` provides COCO ingest and mAP metrics.

---

## 3. Package Layout

`src/`-layout. Import name `esam3`, PyPI distribution `efficient-sam3-finetuning`.

```
src/esam3/
  __init__.py
  _registry.py        # plugin registry: @register(kind, name) + lookup(kind, name)
  config/
    schema.py         # pydantic v2 models: TrainConfig, DataConfig, ModelConfig, PEFTConfig, EvalConfig, TrackingConfig, ExportConfig
    loader.py         # load_config(path, overrides) -> TrainConfig
  data/
    base.py           # Example, Prompts (TextPrompts | BoxPrompts), Instance, Dataset protocol
    coco.py           # COCO instance JSON adapter (@register)
    hf.py             # HuggingFace datasets adapter (@register)
    transforms.py     # image + prompt augmentations
    collate.py        # variable-shape batch collator
  models/
    sam3.py           # SAM3.1 loader from HF; forward wrapper taking (image, prompts)
    losses.py         # mask + box + objectness losses
  peft_adapters/      # named to avoid clashing with the `peft` library
    lora.py           # LoRA via huggingface/peft (@register)
    qlora.py          # 4-bit base via bitsandbytes + LoRA (@register)
  train/
    trainer.py        # Trainer.fit() — public entrypoint, returns RunResult
    loop.py           # inner step / epoch loop
    checkpoint.py     # save adapter, optional merged export, resume
  eval/
    metrics.py        # COCO mAP, mAP@.5/.75, per-class AP
    evaluator.py      # Evaluator.evaluate(model, dataset) -> MetricsReport
  tracking/
    base.py           # Tracker protocol
    tensorboard.py    # @register
    wandb.py          # @register
    noop.py           # @register
  cli/
    main.py           # `esam3` Typer entry point
    train_cmd.py      # `esam3 train --config ...`
    eval_cmd.py       # `esam3 eval --config ... --checkpoint ...`
    export_cmd.py     # `esam3 export --checkpoint ... [--merge]`
    init_cmd.py       # `esam3 init --template ...`
    doctor_cmd.py     # `esam3 doctor`

tests/
  unit/               # per-module tests, fast, CPU
  integration/        # @pytest.mark.integration — stub-model end-to-end
  gpu/                # @pytest.mark.gpu — real SAM3.1 overfit smoke
  fixtures/
    tiny_coco/        # 2 images + COCO JSON
    tiny_sam3_stub.py # stub nn.Module matching SAM3.1's forward contract
  conftest.py

configs/examples/
  coco_text_lora.yaml
  coco_bbox_qlora.yaml

docs/
  superpowers/specs/  # design docs (this file)

logs/
  log.md              # append-only activity log
  TODO.md             # append-only deferred-work log

pyproject.toml
LICENSE                # Apache-2.0
README.md
ARCHITECTURE.md
.pre-commit-config.yaml
.github/workflows/ci.yml
```

**Boundary rules:**

- `data/`, `models/`, `peft_adapters/`, `train/`, `eval/`, `tracking/` are independent modules with narrow protocols between them. Each is independently testable.
- `Trainer` depends only on protocols (`Dataset`, `Tracker`) — never on concrete adapter classes. This is the seam where Ray Train slots in later.
- The CLI is a thin shell. Every command parses config → calls a library function → prints a result. No training logic in CLI files.
- `peft_adapters` (not `peft`) avoids shadowing the `peft` PyPI library at import.

---

## 4. Data Flow

```
my.yaml ─► config.loader.load() ─► TrainConfig
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     data.build_dataset()      models.load_sam31()         tracking.build_tracker()
     (COCO | HF adapter)       returns nn.Module           (TB | W&B | noop)
            │                          │
            │                  peft_adapters.apply()
            │                  (LoRA or QLoRA wrap)
            │                          │
            └──────────────┬───────────┘
                           ▼
                  train.Trainer(model, train_ds, val_ds, tracker, cfg).fit()
                           │
                           ▼
              runs/{run_id}/adapter/        ← always (LoRA weights + config)
              runs/{run_id}/merged/         ← if cfg.export.merge
              runs/{run_id}/metrics.json    ← final eval report
```

Each `build_*` helper looks the implementation up via `_registry.lookup(kind, name)`, so adding a new dataset adapter / PEFT method / tracker is one file plus a `@register` decorator — no edits to dispatch code.

---

## 5. Core Protocols (Stable Seams)

These are the package's stable internal interfaces. All other types may change without notice. Documented in `ARCHITECTURE.md`.

```python
# data/base.py
@dataclass
class TextPrompts:
    classes: list[str]            # per-image class vocabulary

@dataclass
class BoxPrompts:
    boxes: torch.Tensor           # (N, 4) xyxy in pixel coords
    class_ids: torch.Tensor       # (N,)

Prompts = TextPrompts | BoxPrompts

@dataclass
class Instance:
    mask: torch.Tensor            # (H, W) bool
    class_id: int
    box: torch.Tensor             # (4,) xyxy

@dataclass
class Example:
    image: torch.Tensor           # (3, H, W) normalized
    image_id: str
    prompts: Prompts
    instances: list[Instance]     # ground truth

class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Example: ...
    @property
    def class_names(self) -> list[str]: ...

# tracking/base.py
class Tracker(Protocol):
    def log_scalars(self, step: int, values: dict[str, float]) -> None: ...
    def log_images(self, step: int, images: dict[str, np.ndarray]) -> None: ...
    def close(self) -> None: ...

# train/trainer.py
@dataclass
class RunResult:
    run_dir: Path
    adapter_path: Path
    merged_path: Path | None
    final_metrics: MetricsReport | None

class Trainer:
    def __init__(self, model, train_ds, val_ds, tracker, cfg: TrainConfig): ...
    def fit(self) -> RunResult: ...

# eval/evaluator.py
@dataclass
class MetricsReport:
    overall: dict[str, float]           # {"mAP": .., "mAP_50": .., "mAP_75": ..}
    per_class: dict[str, dict[str, float]]   # class_name -> {"AP": .., "AP_50": ..}
    n_images: int
    n_predictions: int

class Evaluator:
    def evaluate(self, model, dataset: Dataset) -> MetricsReport: ...
```

**Design notes:**

- `Prompts` is a tagged union per-example (not per-batch). COCO images have many boxes but one text vocabulary per image; the collator branches on type.
- Ground-truth `Instance.class_id` lives on the example so eval computes per-class AP without a side table.
- `Tracker.log_scalars` takes `step` explicitly — no hidden state. `noop` is a normal implementation, not a conditional.
- `class_names` lives on the Dataset (data-derived), not the config.
- `RunResult` makes library use testable; the integration smoke test asserts on its fields.

---

## 6. Memory Strategy (12–16GB target)

Aggressive memory-saving defaults, degraded predictably via config.

- **Mixed precision:** `bfloat16` autocast default. `float16` selectable. No fp32 training path.
- **Gradient checkpointing:** on by default on the SAM3.1 encoder; configurable off.
- **Frozen base, trainable adapters only.** LoRA in attention projections; everything else frozen.
- **QLoRA path:** base loaded `nf4` via bitsandbytes; adapters in bf16. QLoRA + grad-checkpointing is the 12GB recipe.
- **Image resolution:** config-driven; defaults to SAM3.1's native input.
- **Batch size:** default `1` with `gradient_accumulation_steps: 8`. Effective batch behavior independent of GPU.
- **Optimizer:** `adamw` (torch) by default. `adamw8bit` (bitsandbytes) selectable, requires the `[qlora]` extra — automatically available to QLoRA users.
- **No DDP/FSDP in v0** — single device. Multi-GPU deferred to a Ray Train spec.

---

## 7. CLI & Config

### CLI (Typer)

```
esam3 train   --config PATH [--override key=val ...] [--resume PATH]
esam3 eval    --config PATH --checkpoint PATH [--split val|test]
esam3 export  --checkpoint PATH [--merge] [--output PATH]
esam3 init    --template (coco-text|coco-bbox|hf-text)
esam3 doctor
```

Every command: parse config → validate → call library function → print result. CLI files contain no training logic.

### Config Schema (pydantic v2)

```yaml
run:
  name: "coco-cats-lora"
  output_dir: "./runs"
  seed: 42

model:
  name: "facebook/sam3.1"
  revision: null
  gradient_checkpointing: true
  dtype: "bfloat16"                # bfloat16 | float16

data:
  format: "coco"                   # coco | hf
  train:
    annotations: "data/train.json"
    images: "data/train/"
  val:
    annotations: "data/val.json"
    images: "data/val/"
  prompt_mode: "bbox"              # text | bbox
  image_size: 1024
  augmentations:
    hflip: true
    color_jitter: 0.1

peft:
  method: "lora"                   # lora | qlora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
  qlora:
    quant_type: "nf4"
    compute_dtype: "bfloat16"

train:
  epochs: 10
  batch_size: 1
  grad_accum_steps: 8
  optimizer: "adamw"               # adamw | adamw8bit (adamw8bit requires [qlora] extra)
  lr: 1.0e-4
  lr_schedule: "cosine"
  warmup_steps: 100
  max_grad_norm: 1.0
  eval_every: 500
  save_every: 1000

eval:
  metrics: ["mAP", "mAP_50", "mAP_75", "per_class_AP"]
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

tracking:
  backend: "tensorboard"           # tensorboard | wandb | none
  wandb:
    project: "esam3"
    entity: null

export:
  merge: false
```

**Rules:**

- All paths resolved relative to the config file's directory (not CWD).
- `--override key.subkey=value` flags merge over the file; validation runs once on the merged dict; errors cite the source.
- Pydantic models in `esam3.config.schema` are the contract. Defaults are declared there, nowhere else.
- `esam3 doctor` reports CUDA visibility, installed extras (`wandb`, `bitsandbytes`), free VRAM, and whether SAM3.1 weights are cached. Cheap to run.

---

## 8. Code Quality & Readability

The repo prioritizes legibility for the maintainer and forking developers.

- `README.md` — what / why / 60-second quickstart.
- `ARCHITECTURE.md` — one page: module map, data flow diagram, the Section 5 protocols documented as stable seams.
- `LICENSE` — Apache-2.0.
- Type hints everywhere; `mypy --strict` on `src/esam3`.
- Module + public-class docstrings (one-liner + Args/Returns where non-obvious). No function-by-function noise.
- `ruff` (lint + format), `pytest` + `pytest-cov`.
- `pre-commit` running ruff locally on commit.
- `.github/workflows/ci.yml` — ruff + mypy + pytest on push/PR. CPU-only.
- Optional deps grouped: `[wandb]`, `[qlora]`, `[dev]`. QLoRA opt-in because bitsandbytes is platform-finicky.
- Plugin-style `_registry.py` for the three pluggable surfaces (dataset adapters, PEFT methods, trackers). New adapter = one file + `@register` + one test.
- `logs/log.md` and `logs/TODO.md` exist from scaffolding with header comments explaining the append-only convention. Committed.

---

## 9. Testing Strategy

Three tiers. All CPU except the smoke tier.

- **Unit (fast, CPU, every commit).** One test module per source module. Heavy mocking at the SAM3.1 boundary — `data/`, `peft_adapters/`, `train/loop.py`, `eval/metrics.py`, `tracking/`, `config/` never load real weights. Use the `tiny_sam3_stub` fixture: an `nn.Module` with SAM3.1's forward signature returning random tensors of correct shape. Coverage gate 80% on `src/esam3`.

- **Integration (CPU, `@pytest.mark.integration`).** End-to-end with stub model + `tests/fixtures/tiny_coco/`. Verifies: config loads → dataset iterates → trainer runs one step → checkpoint writes → eval produces a `MetricsReport`. Run in CI.

- **GPU smoke (`@pytest.mark.gpu`, manual / nightly).** Load real SAM3.1, LoRA-finetune ~50 steps on tiny_coco, assert train loss at step 50 is ≥30% lower than step 1 (overfit on 2 images should easily clear this). Skipped when CUDA absent. Not in CI by default; documented in README as `pytest -m gpu`.

**Fixtures:** `tests/fixtures/tiny_coco/` (2 images + COCO JSON, KBs, committed); `tiny_sam3_stub.py` matching SAM3.1's forward contract; conftest fixtures `tmp_run_dir`, `stub_model`, `tiny_coco_dataset`, `noop_tracker`.

**Not tested:** SAM3.1 internals (Meta's problem), the W&B SDK (we test our adapter against a fake `Tracker`), CUDA / bitsandbytes (covered by GPU smoke).

---

## 10. Scaffolding (the First Implementation Plan)

Scaffolding produces a repo where the package imports cleanly, the CLI runs `--help`, tests pass on CPU, CI is green, and every public surface from Sections 3–9 exists as a typed stub or signature. **No training/eval/data-loading logic — those are subsequent plans.**

**In scope for scaffolding:**

- `pyproject.toml`: project metadata, Python 3.13, dep groups (`[dev]`, `[wandb]`, `[qlora]`). Core deps declared (torch, transformers, peft, datasets, pydantic, typer, pyyaml, pycocotools, numpy, rich). No deps imported yet inside `src/esam3` except in dedicated module stubs.
- Full `src/esam3/` tree (Section 3) with:
  - `config/schema.py` — pydantic models fully implemented (no point stubbing the contract).
  - `config/loader.py` — implemented (YAML load + `--override` merge + validate). Small, isolated, easy to test now.
  - `data/base.py`, `tracking/base.py` — Protocols + dataclasses defined.
  - `_registry.py` — implemented (it's tiny and used by every plugin module).
  - All other modules — class skeletons with full type signatures, docstrings, and `raise NotImplementedError("filled in by spec: <name>")` bodies.
  - `cli/` — Typer app wired; each command parses config and prints "not yet implemented".
- `tests/` with `conftest.py`, `tiny_coco/` fixture, `tiny_sam3_stub.py`, and tests for: config loading + validation + override merging; registry register/lookup; `esam3 --help` exits 0; every public module imports without raising.
- `logs/log.md`, `logs/TODO.md` with header comments.
- `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `.gitignore` additions.
- `README.md` rewrite (quickstart + status), `ARCHITECTURE.md`, `LICENSE` (Apache-2.0).
- `configs/examples/coco_text_lora.yaml`, `coco_bbox_qlora.yaml` — valid against the schema.

**Exit criteria:** `uv sync --all-extras --dev && ruff check && mypy && pytest && esam3 --help` all pass cleanly.

---

## 11. Next-Step Subsystem Specs

Each gets its own brainstorm → spec → plan → implementation cycle. Suggested build order:

1. **`spec/data-loading`** — `data/coco.py`, `data/hf.py`, `transforms.py`, `collate.py`. Defines how text and bbox prompts are encoded for SAM3.1. Most error-prone module; first so trainer + eval can consume it.
2. **`spec/model-loading`** — `models/sam3.py` (HF load + forward wrapper), `models/losses.py`. Includes gradient-checkpointing toggle.
3. **`spec/peft-lora`** — `peft_adapters/lora.py`. Smaller of the two PEFT specs.
4. **`spec/peft-qlora`** — `peft_adapters/qlora.py`. bitsandbytes-dependent; isolated from `lora.py`.
5. **`spec/training-loop`** — `train/trainer.py`, `train/loop.py`, `train/checkpoint.py`. Depends on 1–3.
6. **`spec/eval`** — `eval/metrics.py`, `eval/evaluator.py` (COCO mAP + per-class AP). Depends on 1, 2.
7. **`spec/tracking`** — `tracking/{tensorboard,wandb,noop}.py`. Independent of 1–6.
8. **`spec/cli`** — wire CLI commands to implemented library functions; implement `esam3 doctor` and `esam3 init`.
9. **`spec/smoke-test`** — GPU overfit-on-2-images integration test; first end-to-end validation that real SAM3.1 trains.

**Deferred to v1+ (not on the v0 path):** SAM3 (vs 3.1), point/mask prompts, semantic segmentation output, video, Ray Train distribution, Argo workflows.
