<!-- markdownlint-disable MD024 -->
# Configuration Schema

Regenerate this document whenever a config field is added, removed, or changed.

Every field a user might set in a YAML config is listed below.
The genuinely internal dataclasses (`MatcherWeights`) cannot be set from user YAML and are not
listed here. `WandbConfig` and `ExportConfig` are internal dataclasses but are surfaced as
`tracking.wandb.*` and `export.merge` and are documented in those sections. `LossConfig` is a
full Pydantic model and is documented under `train.loss.*`.

Layer legend:

- **common** — set this field in almost every config.
- **advanced** — leave at the default unless you have a specific reason to change it.

---

## `run`

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `run.name` | str | (required) | common | Unique name for this training run; used as the run-directory prefix. |
| `run.output_dir` | str | `"./runs"` | common | Root directory where per-run subdirectories are written. |
| `run.seed` | int | `42` | common | Global RNG seed for reproducibility (Python, NumPy, PyTorch). |

---

## `model`

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `model.name` | str | `"facebook/sam3.1"` | common | HuggingFace model ID or local path for the SAM 3.1 base checkpoint. |
| `model.local_dir` | str \| null | `"models/sam3.1"` | common | Local directory to cache/load the model from (passed to `snapshot_download`). |
| `model.checkpoint_file` | str | `"sam3.1_multiplex.pt"` | common | Filename of the SAM 3.1 weights file inside `local_dir`. |
| `model.dtype` | `"bfloat16"` \| `"float16"` | `"bfloat16"` | common | Floating-point precision for model weights and activations. |
| `model.revision` | str \| null | `null` | advanced | HuggingFace revision (branch, tag, or commit SHA) to pin the model download. |
| `model.device` | str \| null | `null` | advanced | Override the target device (e.g. `"cuda:1"`); `null` auto-selects the first available GPU. |

---

## `data`

Top-level data source and prompt configuration.

### Common fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.format` | `"coco"` \| `"hf"` | (required) | common | Dataset format — COCO instance JSON or HuggingFace datasets. |
| `data.train` | DataSplit | (required) | common | Paths to the training split. See `DataSplit` sub-fields below. |
| `data.val` | DataSplit \| null | `null` | common | Paths to an explicit validation split. Mutually exclusive with `data.val_split`; set one, or neither for no-val mode. |
| `data.val_split` | ValSplitConfig \| null | `null` | common | Auto-split parameters: carves `data.train` into train+val. Mutually exclusive with `data.val`. See `ValSplitConfig` sub-fields below. |
| `data.channels` | int [1, 16] | `3` | common | Number of input image channels. The N→3 channel adapter (a 1×1 conv before the frozen SAM 3.1 patch-embed) bridges N channels to the pretrained 3-channel stem. Explicit only — no auto-detection. |
| `data.channel_semantics` | `"rgb"` \| `"rgba"` \| `"grayscale"` \| `"freeform"` | `"rgb"` | common | How the input channels are interpreted. Drives the channel adapter, normalization default, and augmentation regime. Must be compatible with `data.channels`. `"freeform"` requires explicit `data.normalize.mean`/`std`. |
| `data.text_prompt.mode` | `"present"` \| `"all"` \| `"present_plus_negatives"` \| `"sampled_fixed_k"` | `"present"` | common | How the per-image class vocabulary is built for text prompts. |

### ValSplitConfig sub-fields (used by `data.val_split`)

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.val_split.fraction` | float (0, 0.5] | `0.1` | common | Fraction of training images carved out as validation. |
| `data.val_split.seed` | int \| null | `null` | advanced | RNG seed for the split; `null` inherits `run.seed` at resolve time. |

### DataSplit sub-fields (used by `data.train`, `data.val`, and `data.test`)

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `.annotations` | str (non-empty) | (required) | common | Path to the COCO-format annotations JSON file (or HF split name). |
| `.images` | str (non-empty) | (required) | common | Path to the image directory (or HF split name, ignored for HF format). |

### Augmentation fields

Augmentation is controlled by a `(preset, intensity)` pair that resolves to a full augmentation
table in code (`src/custom_sam_peft/data/augmentations.py`). Per-knob `overrides` replace
individual resolved values without affecting the rest.

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.augmentations.preset` | `"natural"` \| `"medical"` \| `"satellite"` \| `"microscopy"` \| `"none"` \| `"custom"` | `"natural"` | common | Augmentation preset; selects the base policy table for the domain. |
| `data.augmentations.intensity` | `"safe"` \| `"medium"` \| `"aggressive"` | `"medium"` | common | Intensity tier within the chosen preset. |
| `data.augmentations.overrides.hflip` | bool \| null | `null` | advanced | Override horizontal flip; `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.vflip` | bool \| null | `null` | advanced | Override vertical flip; `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.rotate90` | bool \| null | `null` | advanced | Override 90-degree rotation; `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.rotate_arbitrary` | float (≥0) \| null | `null` | advanced | Override arbitrary-angle rotation strength; `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.color_jitter` | float (≥0) \| null | `null` | advanced | Override color-jitter strength; `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.stain_jitter` | float (≥0) \| null | `null` | advanced | Override stain-jitter strength (histology datasets); `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.blur` | float (≥0) \| null | `null` | advanced | Override blur strength; `null` inherits from `(preset, intensity)`. |
| `data.augmentations.overrides.gauss_noise` | float (≥0) \| null | `null` | advanced | Override Gaussian-noise strength; `null` inherits from `(preset, intensity)`. |

### Normalization fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.normalize.mean` | list[float] (len 1..16, each in [0, 1]) | `[0.5, 0.5, 0.5]` | advanced | Per-channel image mean (field-level fallback; effective default is injected from `channel_semantics` — see `channel_semantics.py`). Length must equal `data.channels`. |
| `data.normalize.std` | list[float] (len 1..16, each >0) | `[0.5, 0.5, 0.5]` | advanced | Per-channel image std (field-level fallback; effective default is injected from `channel_semantics` — see `channel_semantics.py`). Length must equal `data.channels` and `normalize.mean`. |
| `data.normalize.max_pixel_value` | float (>0) | `255.0` | advanced | Divisor applied before mean/std normalization. Default `255.0` assumes uint8 input. For float multi-band input already in [0, 1] set this to `1.0`; mean/std must be expressed in the same units. |

### Dataset-limit fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.limit.train` | int (≥1) \| float (0, 1] \| null | `null` | advanced | Cap the training split. Int = absolute count; float = fraction; `null` = no limit. |
| `data.limit.val` | int (≥1) \| float (0, 1] \| null | `null` | advanced | Cap the validation split. Same type rules as `data.limit.train`. |
| `data.limit.seed` | int | `42` | advanced | RNG seed for random/stratified subsetting. |
| `data.limit.strategy` | `"random"` \| `"stratified"` \| `"first_n"` | `"random"` | advanced | How samples are selected when a limit is applied. |

### Text-prompt advanced fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.text_prompt.negatives_per_image` | int (≥0) | `0` | advanced | Negative class names added per image when `mode="present_plus_negatives"`. |
| `data.text_prompt.k` | int [1, 16] | `16` | advanced | Target total class count for `mode="sampled_fixed_k"`. |

### Advanced data fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.test` | DataSplit \| null | `null` | advanced | Optional held-out test split; evaluated separately from the val loop. |
| `data.hf` | HFDatasetConfig \| null | `null` | advanced | HuggingFace dataset config; required when `data.format == "hf"`. |

### HFDatasetConfig sub-fields (used when `data.format == "hf"`)

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `data.hf.name` | str (non-empty) | (required) | advanced | HuggingFace dataset identifier (e.g. `"rafaelpadilla/coco2017"`). |
| `data.hf.split_train` | str | `"train"` | advanced | HF split name for the training partition. |
| `data.hf.split_val` | str \| null | `null` | advanced | HF split name used as the validation set. Cannot be set when `data.val_split` is set — auto-split carves val from `split_train`. |
| `data.hf.field_map.image` | str | `"image"` | advanced | HF feature key containing the PIL image. |
| `data.hf.field_map.bbox` | str | `"objects.bbox"` | advanced | Dotted key path to the bounding-box list within each example. |
| `data.hf.field_map.category` | str | `"objects.category"` | advanced | Dotted key path to the category-id list within each example. |
| `data.hf.field_map.segmentation` | str \| null | `"objects.segmentation"` | advanced | Dotted key path to the segmentation mask list; `null` disables mask loading. |
| `data.hf.field_map.categories_feature` | str | `"categories"` | advanced | Top-level HF feature key for the class-vocabulary mapping. |
| `data.hf.field_map.bbox_format` | `"xywh"` \| `"xyxy"` | `"xyxy"` | advanced | Bounding-box coordinate format used by the dataset. |

---

## `peft`

LoRA / QLoRA adapter configuration.

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `peft.method` | `"lora"` \| `"qlora"` | (required) | common | PEFT method — `"lora"` for standard LoRA, `"qlora"` for 4-bit quantized LoRA. |
| `peft.r` | int (>0) | `16` | common | LoRA rank — number of low-rank decomposition dimensions. |
| `peft.alpha` | int (>0) | `32` | common | LoRA scaling factor; effective scale = `alpha / r`. |
| `peft.dropout` | float [0, 1) | `0.05` | common | Dropout probability applied to LoRA layers during training. |
| `peft.scope` | `"vision"` \| `"vision_decoder"` \| `"all"` | `"vision_decoder"` | common | Which parts of SAM 3.1 receive LoRA adapters. |
| `peft.target_modules` | list[str] \| null | `null` | advanced | Explicit list of module-name patterns to adapt; overrides `scope` when set. |
| `peft.bias` | `"none"` \| `"all"` \| `"lora_only"` | `"none"` | advanced | Which bias terms to train alongside the LoRA weights. |
| `peft.qlora.quant_type` | `"nf4"` \| `"fp4"` | `"nf4"` | advanced | 4-bit quantization type (applies when `peft.method == "qlora"`). |
| `peft.qlora.compute_dtype` | `"bfloat16"` \| `"float16"` | `"bfloat16"` | advanced | Dtype for dequantized compute; should match `model.dtype` in practice (applies when `peft.method == "qlora"`). |
| `peft.qlora.use_double_quant` | bool | `false` | advanced | Enable double quantization to further reduce memory (applies when `peft.method == "qlora"`). |

---

## `train`

Training hyperparameters and schedule.

### Common fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `train.epochs` | int (>0) | (required) | common | Number of complete passes through the training dataset. |
| `train.batch_size` | int (>0) | `1` | common | Number of images per gradient-accumulation micro-batch. |
| `train.grad_accum_steps` | int (>0) | `8` | common | Gradient accumulation steps before one optimizer step; effective batch = `batch_size × grad_accum_steps`. |
| `train.optimizer` | `"adamw"` \| `"adamw8bit"` \| `"auto"` | `"auto"` | common | Optimizer; `"auto"` selects `adamw8bit` for QLoRA and `adamw` for LoRA. |
| `train.learning_rate` | float (>0) | `1.0e-4` | common | Peak learning rate after warm-up. |
| `train.lr_schedule` | `"constant"` \| `"cosine"` \| `"linear"` \| `"plateau"` | `"plateau"` | common | Learning-rate schedule. `"plateau"` reduces LR on a validation-mAP plateau (rung 1) and is paired with early stop (rung 2); falls back to `"cosine"` with a warning when no val set is present. |
| `train.warmup_steps` | int (≥0) | `100` | common | Steps over which the LR linearly warms up from 0. |
| `train.save_every` | int (>0) \| null | `null` | common | Checkpoint every N optimizer steps. `null` auto-resolves to `steps_per_epoch` (one checkpoint per epoch). |
| `train.log_every` | int (>0) | `50` | common | Log scalar metrics every N optimizer steps. |

### Loss fields

Loss is controlled by a `(preset, class_imbalance)` pair that resolves to a full loss table in
code (`src/custom_sam_peft/train/loss/`). Per-knob `overrides` replace individual resolved values.

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `train.loss.preset` | `"natural"` \| `"medical"` \| `"satellite"` \| `"microscopy"` \| `"none"` \| `"custom"` | `"natural"` | common | Loss preset; selects the base policy table for the domain. |
| `train.loss.class_imbalance` | `"balanced"` \| `"moderate"` \| `"severe"` | `"balanced"` | common | Class-imbalance tier within the chosen preset. |
| `train.loss.overrides.mask_family` | `"bce"` \| `"dice"` \| `"dice_bce"` \| `"focal_bce"` \| `"focal_dice"` \| `"focal_tversky"` \| `"boundary"` \| null | `null` | advanced | Override mask loss family; `null` inherits from `(preset, class_imbalance)`. |
| `train.loss.overrides.box_family` | `"l1_giou"` \| `"giou_only"` \| `"ciou"` \| null | `null` | advanced | Override box loss family; `null` inherits from `(preset, class_imbalance)`. |
| `train.loss.overrides.obj_family` | `"focal_bce"` \| `"bce"` \| null | `null` | advanced | Override objectness loss family; `null` inherits from `(preset, class_imbalance)`. |
| `train.loss.overrides.presence_family` | `"bce"` \| `"focal_bce"` \| null | `null` | advanced | Override presence loss family; `null` inherits from `(preset, class_imbalance)`. |
| `train.loss.overrides.w_mask` | float (>0) \| null | `null` | advanced | Override mask loss weight; `null` inherits. |
| `train.loss.overrides.w_box` | float (≥0) \| null | `null` | advanced | Override box loss weight; `null` inherits. |
| `train.loss.overrides.w_obj` | float (>0) \| null | `null` | advanced | Override objectness loss weight; `null` inherits. |
| `train.loss.overrides.w_presence` | float (>0) \| null | `null` | advanced | Override presence loss weight; `null` inherits. |
| `train.loss.overrides.focal_gamma` | float (>0) \| null | `null` | advanced | Override focal-loss gamma; `null` inherits. |
| `train.loss.overrides.focal_alpha` | float [0, 1] \| null | `null` | advanced | Override focal-loss alpha; `null` inherits. |
| `train.loss.overrides.tversky_alpha` | float [0, 1] \| null | `null` | advanced | Override Tversky alpha; `null` inherits. |
| `train.loss.overrides.tversky_gamma` | float (>0) \| null | `null` | advanced | Override Tversky gamma; `null` inherits. |
| `train.loss.overrides.boundary_weight` | float [0, 1] \| null | `null` | advanced | Override boundary blend coefficient; `null` inherits. |
| `train.loss.overrides.matcher_weights` | MatcherWeights \| null | `null` | advanced | Advanced escape hatch to override Hungarian-matcher cost weights (`lambda_l1`, `lambda_giou`, `lambda_mask`). Accepts a dict. |

### Advanced fields

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `train.max_grad_norm` | float (>0) | `1.0` | advanced | Maximum gradient norm for gradient clipping. |
| `train.eval_every` | int (>0) \| null | `null` | advanced | Run a validation eval every N optimizer steps. `null` auto-resolves to `steps_per_epoch` (one eval per epoch). |
| `train.time_limit` | str \| int \| null | `null` | advanced | Wall-clock budget for this invocation. Accepts a human duration (`"2h30m"`, `"90m"`, `"3600s"`) or bare seconds (`3600`). `null` = unlimited. On expiry a resumable checkpoint is flushed and training exits 0. Budget is per-run: `--resume` restarts the clock. |
| `train.host_ram_floor_gb` | float | `2.0` | advanced | Host-RAM floor in GB. When available RAM drops below this value a resumable checkpoint is flushed and training stops gracefully. `<=0` disables the guard. |
| `train.nan_abort_after` | int (>0) | `20` | advanced | Abort training after this many consecutive steps with NaN loss. |
| `train.num_workers` | int (≥0) | min(4, cpu_count) | advanced | DataLoader worker processes. `0` disables multiprocessing. |
| `train.multiplex.classes_per_forward` | int [1, 16] | `16` | advanced | Class prompts per multiplex forward pass. `1` = legacy per-class regime. |
| `train.lr_decay_on_plateau.patience` | int (>0) | `5` | advanced | Non-improving evals before one LR cut (plateau mode only, rung 1). |
| `train.lr_decay_on_plateau.factor` | float (0, 1) | `0.1` | advanced | LR multiplier applied on each cut (plateau mode only). |
| `train.lr_decay_on_plateau.min_lr` | float (>0) | `1e-6` | advanced | LR floor; cuts never go below this value (plateau mode only). |
| `train.early_stop.enabled` | bool | `true` | advanced | Stop after `stop_patience` consecutive non-improving evals (rung 2). Works under any `lr_schedule` (unlike the rung-1 `lr_decay_on_plateau.*` knobs, which need `plateau`); requires a validation set for the mAP signal. |
| `train.early_stop.monitor` | `"mAP"` | `"mAP"` | advanced | Monitored metric (only `"mAP"` is wired). See §5.4 wart note below. |
| `train.early_stop.min_delta` | float (>0) | `0.001` | advanced | Shared improvement threshold for BOTH rungs (see §5.4 wart note below). |
| `train.early_stop.stop_patience` | int (>0) | `10` | advanced | Non-improving evals before early stop fires. |

**§5.4 wart — shared `monitor`/`min_delta` across both rungs:** `early_stop.monitor` and
`early_stop.min_delta` live under the `early_stop` sub-block but configure the definition of
"improvement" for **both** rung 1 (LR decay) and rung 2 (early stop). Even when
`early_stop.enabled=false`, these two fields still drive the rung-1 LR-decay threshold:
`monitor` selects the metric fed to `ReduceLROnPlateau` and `min_delta` sets its `threshold`
argument. This coupling is a known design wart (#197, §5.4) — the fields are named after the
early-stop rung but have cross-rung effect.

---

## `eval`

Evaluation configuration. All fields are optional — the section defaults are usable as-is.

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `eval.iou_thresholds` | list[float] | `[0.50, 0.55, …, 0.95]` | advanced | IoU thresholds at which to compute mask AP (COCO standard: 10 thresholds from 0.50 to 0.95). |
| `eval.mode` | `"full"` \| `"lite"` | `"full"` | advanced | `"lite"` evaluates only the first `lite_max_images` images (fast smoke check). |
| `eval.lite_max_images` | int (>0) | `64` | advanced | Maximum images evaluated in `"lite"` mode. |
| `eval.mask_threshold` | float | `0.0` | advanced | Sigmoid threshold above which a mask pixel is considered foreground. |
| `eval.batch_size` | int (>0) \| `"auto"` | `"auto"` | advanced | Evaluation batch size. `"auto"` selects a size based on available VRAM. |
| `eval.save_predictions` | bool | `false` | advanced | Persist per-image COCO-format prediction JSON to the run directory. |
| `eval.visualize` | bool | `true` | advanced | Write a Ground Truth \| Prediction composite PNG per sampled image under `<output>/visualizations/`. Disable per-command with `--no-visualize`. |
| `eval.visualize_count` | int (>0) | `10` | advanced | Number of images to sample for visualization. |

---

## `tracking`

Experiment tracking backend configuration.

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `tracking.backend` | `"local"` \| `"tensorboard"` \| `"wandb"` \| `"none"` | `"local"` | common | Which tracking backend to use for logging scalars and images. `local` writes `metrics.jsonl` with no heavy deps; `tensorboard` requires the `[tensorboard]` extra. |
| `tracking.wandb.project` | str | `"custom_sam_peft"` | advanced | Weights & Biases project name (only used when `tracking.backend == "wandb"`). |
| `tracking.wandb.entity` | str \| null | `null` | advanced | Weights & Biases team/user entity; `null` uses the default W&B entity. |

---

## `export`

Export options (adapter merge).

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `export.merge` | bool | `false` | advanced | Merge LoRA weights into the base model and save full weights on export. |

---

## CLI flags

### `run` / `train` close-out behaviour

The normal `run` and `train` paths close out on the **best** checkpoint (not the last step):
`close_out` restores `best/adapter` into the model before running the final eval and writing
`run_dir/adapter`, so the exported adapter always holds the best weights seen during training.

### `run --finalize`

Productionize a **paused** (time-limited) run with **no training**:

- Rebuilds the model from the checkpoint specified by `--resume` (a path or `__latest__`).
- Restores the best weights from `run_dir/best/adapter` (falls back to the checkpoint's own
  adapter when `best/` does not exist).
- Runs one full eval via `close_out` and writes all artifacts: `adapter/`, optional `merged/`,
  `metrics.json`, and the run bundle (`summary.md`, `bundle.json`).
- **Requires `--resume`** (a checkpoint path or `__latest__`); rejects `--time-limit` (no
  training happens).
- Uses the run's **saved `config.yaml`** (inside `run_dir/`) for fidelity — not the `--config`
  file passed to `run`. This means `eval.visualize` (and all other eval/export settings) are
  governed by the original run config, not the invoking config.

```shell
csp run --config cfg.yaml --resume runs/my-run/checkpoints/step_1000 --finalize
# or resolve the latest checkpoint automatically:
csp run --config cfg.yaml --resume __latest__ --finalize
```
