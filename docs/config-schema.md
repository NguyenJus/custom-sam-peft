<!-- markdownlint-disable MD024 -->
# Configuration Schema (v0.7.0)

> Generated as part of the v0.7.0 hardening pass (issue #26).
> Re-generate when adding or removing a config field.

Every field a user might set in a YAML config is listed below.
Fields under `config/_internal.py` (`LossConfig`, `MatcherWeights`, `WandbConfig`, `ExportConfig`)
are **not** listed here — they are internal-only and cannot be set from user YAML.

Layer legend:

- **common** — set this field in almost every config.
- **advanced** — leave at the default unless you have a specific reason to change it.

---

## `run`

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `run.name` | str | (required) | common | Unique name for this training run; used as the run-directory prefix. | Audit §E: 4/4 example configs set it; always user-specific. |
| `run.output_dir` | str | `"./runs"` | common | Root directory where per-run subdirectories are written. | Audit §E: 4/4 examples set it; needed to redirect output on remote machines. |
| `run.seed` | int | `42` | common | Global RNG seed for reproducibility (Python, NumPy, PyTorch). | Audit §E: 4/4 examples set it; reproducibility is always relevant. |

---

## `model`

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `model.name` | str | `"facebook/sam3.1"` | common | HuggingFace model ID or local path for the SAM 3.1 base checkpoint. | Audit §E: 4/4 examples + notebook set it; base model identity is fundamental. |
| `model.local_dir` | str \| null | `"models/sam3.1"` | common | Local directory to cache/load the model from (passed to `snapshot_download`). | Audit §E: 4/4 examples set it; air-gapped or pre-downloaded setups need this. |
| `model.checkpoint_file` | str | `"sam3.1_multiplex.pt"` | common | Filename of the SAM 3.1 weights file inside `local_dir`. | Audit §E: 4/4 examples set it; checkpoint file name may differ across releases. |
| `model.gradient_checkpointing` | bool | `true` | common | Enable gradient checkpointing to trade compute for VRAM during training. | Audit §E: 4/4 examples + notebook (via preset) set it; critical for VRAM-constrained GPUs. |
| `model.dtype` | `"bfloat16"` \| `"float16"` | `"bfloat16"` | common | Floating-point precision for model weights and activations. | Audit §E: 4/4 examples + notebook (via preset) set it; dtype choice depends on GPU capability. |
| `model.revision` | str \| null | `null` | advanced | HuggingFace revision (branch, tag, or commit SHA) to pin the model download. | Audit §E: 0 non-test hits; useful for reproducibility across checkpoint releases. |
| `model.device` | str \| null | `null` | advanced | Override the target device (e.g. `"cuda:1"`); `null` auto-selects the first available GPU. | Audit §E: 0 non-test hits; only needed for multi-GPU manual assignment. |

---

## `data`

Top-level data source and prompt configuration.

### Common fields

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `data.format` | `"coco"` \| `"hf"` | (required) | common | Dataset format — COCO instance JSON or HuggingFace datasets. | Audit §E: 4/4 examples + notebook set it; determines which dataset adapter is used. |
| `data.train` | DataSplit | (required) | common | Paths to the training split. See `DataSplit` sub-fields below. | Audit §E: 4/4 examples set it; without a train split there is nothing to train on. |
| `data.val` | DataSplit | (required) | common | Paths to the validation split. See `DataSplit` sub-fields below. | Audit §E: 4/4 examples set it; validation is always run during training. |
| `data.prompt_mode` | `"text"` \| `"bbox"` | (required) | common | Whether to prompt SAM with text class names or ground-truth bounding boxes. | Audit §E: 4/4 examples set it; determines the entire prompt-construction path. |
| `data.image_size` | int (>0) | `1024` | common | Target image size (square) fed to SAM. Must match the model's expected input. | Audit §E: 4/4 examples set it; changing it changes the entire input pipeline. |
| `data.augmentations.hflip` | bool | `true` | common | Apply random horizontal flip augmentation during training. | Audit §E: 4/4 examples set it; augmentation is standard for object-detection datasets. |
| `data.augmentations.color_jitter` | float [0, 1] | `0.1` | common | Strength of color-jitter augmentation (brightness, contrast, saturation). | Audit §E: 4/4 examples set it; tuned alongside learning rate for dataset-specific best results. |
| `data.text_prompt.mode` | `"present"` \| `"all"` \| `"present_plus_negatives"` \| `"sampled_fixed_k"` | `"present"` | common | How the per-image class vocabulary is built when `prompt_mode="text"`. | Audit §E: 2/4 examples set it; controls whether negative classes appear in the prompt. |
| `data.normalize.mean` | list[float] (len=3) | `[0.485, 0.456, 0.406]` | advanced | Per-channel image mean fallback when `AutoImageProcessor` cannot be loaded. | Audit §E: 2/4 examples set it; only needed when the HF processor is unavailable offline. |
| `data.normalize.std` | list[float] (len=3) | `[0.229, 0.224, 0.225]` | advanced | Per-channel image std fallback when `AutoImageProcessor` cannot be loaded. | Audit §E: 2/4 examples set it; same as above — fallback normalization only. |

### DataSplit sub-fields (used by `data.train`, `data.val`, and `data.test`)

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `.annotations` | str (non-empty) | (required) | common | Path to the COCO-format annotations JSON file (or HF split name). | Audit §E: 4/4 examples set it; the annotation path is dataset-specific. |
| `.images` | str (non-empty) | (required) | common | Path to the image directory (or HF split name, ignored for HF format). | Audit §E: 4/4 examples set it; the image directory is dataset-specific. |

### Advanced data fields

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `data.test` | DataSplit \| null | `null` | advanced | Optional held-out test split; evaluated separately from the val loop. | Audit §E: 0 non-test hits; most training runs use `val` only; test split is optional. |
| `data.hf` | HFDatasetConfig \| null | `null` | advanced | HuggingFace dataset config; required when `data.format == "hf"`. | Audit §E: 0 non-test hits; only needed for HF datasets (COCO is the default format). |
| `data.text_prompt.negatives_per_image` | int (≥0) | `0` | advanced | Number of randomly-sampled negative class names added to the text prompt. | Audit §E: 2/4 examples set it; only meaningful when `text_prompt.mode` uses negatives. |
| `data.text_prompt.k` | int [1, 16] | `16` | advanced | Target total class count for `sampled_fixed_k` text-prompt mode. | Audit §E: 0 non-test hits; only meaningful in `sampled_fixed_k` mode. |

### HFDatasetConfig sub-fields (used when `data.format == "hf"`)

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `data.hf.name` | str (non-empty) | (required) | advanced | HuggingFace dataset identifier (e.g. `"rafaelpadilla/coco2017"`). | Audit §E: 0 non-test hits; only used when `data.format == "hf"`. |
| `data.hf.split_train` | str | `"train"` | advanced | HF split name for the training partition. | Audit §E: 0 non-test hits; defaults match the standard HF split naming convention. |
| `data.hf.split_val` | str | `"validation"` | advanced | HF split name for the validation partition. | Audit §E: 0 non-test hits; same as above. |
| `data.hf.field_map.image` | str | `"image"` | advanced | HF feature key containing the PIL image. | Audit §E: 0 non-test hits; only needed when the dataset uses non-standard field names. |
| `data.hf.field_map.bbox` | str | `"objects.bbox"` | advanced | Dotted key path to the bounding-box list within each example. | Audit §E: 0 non-test hits; same rationale as `field_map.image`. |
| `data.hf.field_map.category` | str | `"objects.category"` | advanced | Dotted key path to the category-id list within each example. | Audit §E: 0 non-test hits; same rationale. |
| `data.hf.field_map.segmentation` | str \| null | `"objects.segmentation"` | advanced | Dotted key path to the segmentation mask list; `null` disables mask loading. | Audit §E: 0 non-test hits; mask loading is optional depending on dataset. |
| `data.hf.field_map.categories_feature` | str | `"categories"` | advanced | Top-level HF feature key for the class-vocabulary mapping. | Audit §E: 0 non-test hits; only needed for non-standard schema datasets. |
| `data.hf.field_map.bbox_format` | `"xywh"` \| `"xyxy"` | `"xyxy"` | advanced | Bounding-box coordinate format used by the dataset. | Audit §E: 0 non-test hits; only matters when the dataset uses `xywh` instead of `xyxy`. |

---

## `peft`

LoRA / QLoRA adapter configuration.

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `peft.method` | `"lora"` \| `"qlora"` | (required) | common | PEFT method — `"lora"` for standard LoRA, `"qlora"` for 4-bit quantized LoRA. | Audit §E: 4/4 examples + notebook set it; method choice drives quantization and optimizer selection. |
| `peft.r` | int (>0) | `16` | common | LoRA rank — number of low-rank decomposition dimensions. | Audit §E: 4/4 examples + notebook set it; rank is the primary PEFT quality/efficiency knob. |
| `peft.alpha` | int (>0) | `32` | common | LoRA scaling factor; effective scale = `alpha / r`. | Audit §E: 2/4 examples set it; alpha must be co-tuned with `r`. |
| `peft.dropout` | float [0, 1) | `0.05` | common | Dropout probability applied to LoRA layers during training. | Audit §E: 2/4 examples set it; small regularization effect on adapter weights. |
| `peft.scope` | `"vision"` \| `"vision_decoder"` \| `"all"` | `"vision_decoder"` | common | Which parts of SAM 3.1 receive LoRA adapters. | Audit §E: 2/4 examples set it; scope is a primary memory/quality trade-off. |
| `peft.target_modules` | list[str] \| null | `null` | advanced | Explicit list of module-name patterns to adapt; overrides `scope` when set. | Audit §E: 0 non-test hits; only needed for surgical adapter placement beyond scope presets. |
| `peft.bias` | `"none"` \| `"all"` \| `"lora_only"` | `"none"` | advanced | Which bias terms to train alongside the LoRA weights. | Audit §E: 0 non-test hits; non-default bias training is rarely needed. |
| `peft.qlora.quant_type` | `"nf4"` \| `"fp4"` | `"nf4"` | advanced | 4-bit quantization type used by QLoRA (requires `peft.method == "qlora"`). | Audit §E: 0 non-test hits; nf4 is the standard recommended by the QLoRA paper. |
| `peft.qlora.compute_dtype` | `"bfloat16"` \| `"float16"` | `"bfloat16"` | advanced | Dtype for dequantized compute (must match `model.dtype` in practice). | Audit §E: 0 non-test hits; only matters when switching between bf16 and fp16 GPU families. |

---

## `train`

Training hyperparameters and schedule.

### Common fields

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `train.epochs` | int (>0) | (required) | common | Number of complete passes through the training dataset. | Audit §E: 4/4 examples set it; the most fundamental hyperparameter — no universal default. |
| `train.batch_size` | int (>0) | `1` | common | Number of images per gradient-accumulation micro-batch. | Audit §E: 4/4 examples + notebook (via preset) set it; constrained by VRAM. |
| `train.grad_accum_steps` | int (>0) | `8` | common | Gradient accumulation steps before one optimizer step; effective batch = `batch_size × grad_accum_steps`. | Audit §E: 4/4 examples + notebook (via preset) set it; used to simulate large batches with limited VRAM. |
| `train.optimizer` | `"adamw"` \| `"adamw8bit"` \| `"auto"` | `"auto"` | common | Optimizer; `"auto"` selects `adamw8bit` for QLoRA and `adamw` for LoRA. | Audit §E: 4/4 examples set it; `"auto"` is almost always correct. |
| `train.learning_rate` | float (>0) | `1.0e-4` | common | Peak learning rate after warm-up. | Audit §E: 4/4 examples set it; the primary training quality knob. |
| `train.lr_schedule` | `"constant"` \| `"cosine"` \| `"linear"` | `"cosine"` | common | Learning-rate decay schedule applied after warm-up. | Audit §E: 2/4 examples set it; cosine is the recommended default. |
| `train.warmup_steps` | int (≥0) | `100` | common | Number of optimizer steps over which the LR linearly warms up from 0. | Audit §E: 4/4 examples set it; warm-up prevents early-step divergence. |
| `train.save_every` | int (>0) | `1000` | common | Save a checkpoint every N optimizer steps. | Audit §E: 4/4 examples set it; controls checkpoint frequency and disk usage. |
| `train.log_every` | int (>0) | `50` | common | Log scalar metrics every N optimizer steps. | Audit §E: 4/4 examples set it; determines monitoring granularity. |
| `train.box_hint.p_start` | float [0, 1] | `1.0` | common | Starting probability of feeding a GT bounding box hint alongside the text prompt. | Audit §E: 4/4 examples set it; controls how much box supervision the model receives early in training. |
| `train.box_hint.p_end` | float [0, 1] | `0.0` | common | Ending probability after the linear-decay schedule completes. | Audit §E: 4/4 examples set it; `0.0` means pure text-prompt inference after warm-up. |
| `train.box_hint.decay_steps` | int (>0) | `5000` | common | Number of global steps over which `p` decays from `p_start` to `p_end`. | Audit §E: 4/4 examples set it; should match or exceed `warmup_steps`. |

### Advanced fields

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `train.max_grad_norm` | float (>0) | `1.0` | advanced | Maximum gradient norm for gradient clipping. | Audit §E: 2/4 examples set it; 1.0 is the standard safe default; only tune if gradients explode. |
| `train.eval_every` | int (>0) | `500` | advanced | Run a validation evaluation pass every N optimizer steps. | Audit §E: 2/4 examples set it; more frequent eval helps but slows training. |
| `train.loss` | LossConfig | (internal defaults) | advanced | Loss-mix weights (w_mask, w_obj, w_presence) — see source for sub-fields. Internal-only fields are hardcoded in `_internal.py`. | Audit §E: 2/4 examples reference it; loss weights are rarely tuned; defaults work for all v0 datasets. |
| `train.nan_abort_after` | int (>0) | `20` | advanced | Abort training if NaN losses appear in more than this many consecutive steps. | Audit §E: 2/4 examples set it; NaN detection is a safety rail; 20 steps is a safe margin. |
| `train.num_workers` | int (≥0) | min(4, cpu_count) | advanced | Number of DataLoader worker processes; `0` disables multiprocessing. | Audit §E: 2/4 examples set it; auto-selected from CPU count; only tune on memory-limited hosts. |

---

## `eval`

Evaluation configuration. All fields are optional — the section defaults are usable as-is.

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `eval.iou_thresholds` | list[float] | `[0.50, 0.55, …, 0.95]` | advanced | IoU thresholds at which to compute mask AP (COCO standard: 10 thresholds from 0.50 to 0.95). | Audit §E: 2/4 examples set it; COCO standard is correct for almost all benchmarks. |
| `eval.mode` | `"full"` \| `"lite"` | `"full"` | advanced | `"lite"` evaluates only the first `lite_max_images` images (fast smoke check). | Audit §E: 0 non-test hits; `"lite"` is only useful for debugging or very large val sets. |
| `eval.lite_max_images` | int (>0) | `64` | advanced | Maximum images evaluated in `"lite"` mode. | Audit §E: 0 non-test hits; only meaningful when `eval.mode == "lite"`. |
| `eval.mask_threshold` | float | `0.0` | advanced | Sigmoid threshold above which a mask pixel is considered foreground. | Audit §E: 0 non-test hits; 0.0 uses SAM's raw sigmoid output; only tune for precision/recall trade-offs. |
| `eval.save_predictions` | bool | `false` | advanced | Persist per-image COCO-format prediction JSON to the run directory. | Audit §E: 0 non-test hits; useful for detailed error analysis but adds disk usage. |

---

## `tracking`

Experiment tracking backend configuration.

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `tracking.backend` | `"tensorboard"` \| `"wandb"` \| `"none"` | `"tensorboard"` | common | Which tracking backend to use for logging scalars and images. | Audit §E: 4/4 examples set it; backend depends on team tooling. |
| `tracking.wandb.project` | str | `"custom_sam_peft"` | advanced | Weights & Biases project name (only used when `tracking.backend == "wandb"`). | Audit §E: 0 non-test hits; only needed when using W&B. |
| `tracking.wandb.entity` | str \| null | `null` | advanced | Weights & Biases team/user entity; `null` uses the default W&B entity. | Audit §E: 0 non-test hits; only needed for W&B team workspaces. |

---

## `export`

Export options (adapter merge).

| Field | Type | Default | Layer | Description | YAGNI rationale |
| --- | --- | --- | --- | --- | --- |
| `export.merge` | bool | `false` | advanced | Merge LoRA weights into the base model and save full weights on export. | Audit §E: 0 non-test hits; merged weights are larger but self-contained; only needed for deployment. |
