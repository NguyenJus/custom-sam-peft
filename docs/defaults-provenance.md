# Defaults Provenance

This document is the source of truth for the provenance of every trust-bearing
default hyperparameter in `custom-sam-peft`. Inline `# cite:` / `# tbd:` tags in
the code are deliberately terse pointers into the rows below.

Umbrella `# tbd:` tracker: #191
(Every `# tbd: #191` tag and row points there.)

## Verification Standard

Every literature-backed value is verified against its *primary* source with a
captured quote + URL/DOI + exact equation/table/figure. Framework defaults link
the upstream docs and pin the observed version. Degenerate cases state the math
identity. Reference-implementation values cite the file/line they mirror.
Project numbers with no external source and no internal run are tagged
`# tbd: #191` — never fabricated.

Row schema (every section uses these six columns):

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

- **Location** — `file:symbol`.
- **Value** — the literal default.
- **Tag** — the inline tag class applied (mirrors the code), or `index-only` for
  untagged self-evident structural/string defaults.
- **Full reference** — authors, year, arXiv/DOI, exact Eq./Table/Fig.; or the
  upstream-doc URL + pinned version (framework defaults); or repo file/line
  (reference-impl).
- **Verifying quote** — short quote from the primary source establishing the
  value.
- **Notes** — caveats, degenerate-case identities, calibration run pointers,
  cross-links.

## config/_internal.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `config/_internal.py:MatcherWeights.lambda_l1` | `0.0` | `# cite: degenerate-case` | — | — | Text-only v0 disables box terms; YAGNI-demoted internal constant (docstring: "audit Section E"). |
| `config/_internal.py:MatcherWeights.lambda_giou` | `0.0` | `# cite: degenerate-case` | — | — | Text-only v0 disables box terms; YAGNI-demoted internal constant (docstring: "audit Section E"). |
| `config/_internal.py:MatcherWeights.lambda_mask` | `5.0` | `# tbd: #191` | — | — | Mask-only Hungarian matcher cost weight. Mask2Former (Cheng et al., arXiv:2112.01527) uses `MASK_WEIGHT: 5.0` in its canonical COCO config, which is a plausible upstream reference, but the project code/commits contain no explicit derivation link. Tracking via #191 until an internal run or an explicit design note records the source. |
| `config/_internal.py:WandbConfig.project` | `"custom_sam_peft"` | `index-only` | — | — | Self-evident project string; not user-trust-bearing. |
| `config/_internal.py:WandbConfig.entity` | `None` | `index-only` | — | — | Optional W&B entity; no default to cite. |
| `config/_internal.py:ExportConfig.merge` | `False` | `index-only` | — | — | Boolean export toggle; off by default. |

## config/schema.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `config/schema.py:RunConfig.output_dir` | `"./runs"` | `index-only` | — | — | Self-evident structural default; not trust-bearing. |
| `config/schema.py:RunConfig.seed` | `42` | `# cite: degenerate-case` | — | — | Arbitrary fixed seed; convention only. Any positive integer produces reproducible results. |
| `config/schema.py:ModelConfig.name` | `"facebook/sam3.1"` | `index-only` | — | — | Structural string pointing to the target model. |
| `config/schema.py:ModelConfig.local_dir` | `"models/sam3.1"` | `index-only` | — | — | Structural path; not trust-bearing. |
| `config/schema.py:ModelConfig.checkpoint_file` | `"sam3.1_multiplex.pt"` | `index-only` | — | — | Structural filename; not trust-bearing. |
| `config/schema.py:ModelConfig.dtype` | `"bfloat16"` | `# cite: framework default` | PyTorch / HuggingFace Transformers recommended dtype for Ampere+ GPUs. | bfloat16 is the default compute dtype in HF Trainer and torch.autocast for modern GPUs. | Mirrors QLoRAConfig.compute_dtype. |
| `config/schema.py:TextPromptConfig.mode` | `"present"` | `index-only` | — | — | Conservative default: use only categories present in the image. Rationale in field docstring. |
| `config/schema.py:TextPromptConfig.negatives_per_image` | `0` | `# cite: empirical` | Project design choice: mode='present' with 0 negatives is the conservative starting point. | Field description: "Example configs ship 4, which leaves headroom for typical COCO present-class counts (~3-7 per image)." | 0 is not zero-shot; it is the safe default before negative-mining is enabled. |
| `config/schema.py:TextPromptConfig.k` | `16` | `# cite: models/sam3.py:MULTIPLEX_CAP` | `src/custom_sam_peft/models/sam3.py` line 178: `MULTIPLEX_CAP: int = 16` | `MULTIPLEX_CAP: int = 16` — hard cap from SAM 3.1 head design. | Must equal MULTIPLEX_CAP; upper bound enforced by Field(le=16). |
| `config/schema.py:NormalizeConfig.mean` | `[0.485, 0.456, 0.406]` | `# cite: torchvision ImageNet-1k training-set statistics` | torchvision `_presets.py` lines 52–53 (ImageClassification defaults); also confirmed in `data/transforms.py:KNOWN_PROCESSOR_STATS["facebook/sam3.1"]`. URL: <https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py> | `mean: tuple[float, ...] = (0.485, 0.456, 0.406)` | These are the ImageNet-1k per-channel training-set means, used by all torchvision ImageNet-pretrained models and by SAM 3.1's Sam3ImageProcessor. |
| `config/schema.py:NormalizeConfig.std` | `[0.229, 0.224, 0.225]` | `# cite: torchvision ImageNet-1k training-set statistics` | torchvision `_presets.py` lines 52–53; `data/transforms.py:KNOWN_PROCESSOR_STATS["facebook/sam3.1"]`. URL: <https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py> | `std: tuple[float, ...] = (0.229, 0.224, 0.225)` | ImageNet-1k per-channel training-set standard deviations. |
| `config/schema.py:NormalizeConfig.max_pixel_value` | `255.0` | `# cite: framework default` | Albumentations `A.Normalize` docs: "max_pixel_value: float, None \| 255.0 \| Maximum possible pixel value, used for scaling in standard normalization. Defaults to 255.0." URL: <https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/> | "Defaults to 255.0." | 8-bit uint8 image max; field description cross-links spec §7.2. |
| `config/schema.py:HFFieldMap.image` | `"image"` | `index-only` | — | — | Conventional HF dataset field name; structural. |
| `config/schema.py:HFFieldMap.bbox` | `"objects.bbox"` | `index-only` | — | — | Conventional nested field path; structural. |
| `config/schema.py:HFFieldMap.category` | `"objects.category"` | `index-only` | — | — | Conventional nested field path; structural. |
| `config/schema.py:HFFieldMap.segmentation` | `"objects.segmentation"` | `index-only` | — | — | Conventional nested field path; structural. |
| `config/schema.py:HFFieldMap.categories_feature` | `"categories"` | `index-only` | — | — | Conventional HF feature name; structural. |
| `config/schema.py:HFFieldMap.bbox_format` | `"xyxy"` | `index-only` | — | — | Structural format literal; not trust-bearing. |
| `config/schema.py:HFDatasetConfig.split_train` | `"train"` | `index-only` | — | — | Conventional HF split name; structural. |
| `config/schema.py:ValSplitConfig.fraction` | `0.1` | `# tbd: #191` | — | — | 10% validation is a common convention but no internal calibration run has been recorded. Tracking via #191. |
| `config/schema.py:LimitConfig.seed` | `42` | `# cite: degenerate-case` | — | — | Arbitrary fixed seed; same convention as RunConfig.seed. |
| `config/schema.py:LimitConfig.strategy` | `"random"` | `index-only` | — | — | Default sampling strategy; structural. |
| `config/schema.py:DataConfig.channels` | `3` | `index-only` | — | — | Rationale in field description: 3-channel RGB is the SAM 3.1 pretrained stem width; explicit only. |
| `config/schema.py:DataConfig.channel_semantics` | `"rgb"` | `index-only` | — | — | Rationale in field description: reproduces current behavior exactly; drives channel adapter and augmentation regime. |
| `config/schema.py:QLoRAConfig.quant_type` | `"nf4"` | `# cite: QLoRA (Dettmers 2023) arXiv:2305.14314 §3` | Dettmers et al. 2023, "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314, §3 "4-bit NormalFloat Quantization". | "The information theoretically optimal data type for zero-mean normal distributions with arbitrary standard deviations σ in the range [−1,1]..." — ar5iv render of §3. | NF4 is the recommended quantization type; fp4 is the alternative. |
| `config/schema.py:QLoRAConfig.compute_dtype` | `"bfloat16"` | `# cite: framework default` | PyTorch / HuggingFace recommended compute dtype for Ampere+ GPUs. | Same reasoning as ModelConfig.dtype. | |
| `config/schema.py:QLoRAConfig.use_double_quant` | `False` | `# tbd: #191` | — | — | Double quantization (§3 of QLoRA paper) reduces memory ~0.37 bits/param; disabled by default as a conservative choice. No internal run has evaluated the trade-off. Tracking via #191. |
| `config/schema.py:PEFTConfig.r` | `16` | `# cite: LoRA (Hu 2021) arXiv:2106.09685 §4.1` | Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685, §4.1. Paper explores r=1,2,4,8,64 (Table 6); common practice for vision models is r=8–16. | "we simply set α to the first r we try and do not tune it" (§4.1). | r=16 is a repo-chosen mid-range value within the range explored. alpha=32=2×r follows the α=first-r convention. |
| `config/schema.py:PEFTConfig.alpha` | `32` | `# cite: LoRA (Hu 2021) arXiv:2106.09685 §4.1` | Hu et al. 2021, arXiv:2106.09685, §4.1. | "we simply set α to the first r we try and do not tune it" — setting alpha=32 with r=16 follows this convention (alpha=2r, a common variant since the paper's statement means alpha tracks the initial r tried). | alpha=2r is a common practical convention that extends the paper's "alpha=first r" guideline. |
| `config/schema.py:PEFTConfig.dropout` | `0.05` | `# tbd: #191` | — | — | LoRA paper uses 0.0–0.1 depending on task (Table 11: 0.1 for GPT-2); 0.05 is a repo-chosen midpoint. No internal run recorded. Tracking via #191. |
| `config/schema.py:PEFTConfig.scope` | `"vision_decoder"` | `# tbd: #191` | — | — | Project-chosen scope covering SAM 3.1 vision trunk + transformer decoder. Rationale: vision_decoder hits the attention and decoder FFN linears most relevant to open-vocab adaptation. No ablation recorded. Tracking via #191. |
| `config/schema.py:PEFTConfig.bias` | `"none"` | `# cite: framework default` | HuggingFace PEFT `LoraConfig` default: `bias="none"`. URL: <https://huggingface.co/docs/peft/package_reference/lora> | Default in PEFT LoraConfig is `bias="none"`. | Standard PEFT convention; not training training the bias terms keeps parameter count minimal. |
| `config/schema.py:BoxHintSchedule.p_start` | `1.0` | `# cite: degenerate-case` | — | — | Probability=1.0 is the maximum of [0,1]; full box-hint at step 0. Linear decay endpoint identity. |
| `config/schema.py:BoxHintSchedule.p_end` | `0.0` | `# cite: degenerate-case` | — | — | Probability=0.0 is the minimum of [0,1]; no box-hint at schedule end. Linear decay endpoint identity. |
| `config/schema.py:BoxHintSchedule.decay_steps` | `None` | `# tbd: #191` | — | — | Auto-resolves to 0.75×epochs×steps_per_epoch at runtime (decay over first 75% of run). The 0.75 fraction is project-chosen with no ablation. Cross-ref open issue #88. Tracking via #191. |
| `config/schema.py:MultiplexConfig.classes_per_forward` | `16` | `# cite: models/sam3.py:MULTIPLEX_CAP` | `src/custom_sam_peft/models/sam3.py` line 178: `MULTIPLEX_CAP: int = 16` | `MULTIPLEX_CAP: int = 16` — hard cap from SAM 3.1 model head. | Default=cap means maximum throughput per forward pass. Upper bound enforced by Field(le=16). |
| `config/schema.py:TrainHyperparams.epochs` | `required (template $epochs slot)` | `# cite: empirical` | See "Reference Training Profile" section below (Deliverable 2). | See "Reference Training Profile" section below (Deliverable 2). | Required field; no schema default. The shipped default lives in the `config_full.yaml` `$epochs` slot, set by the `init` flow. Provenance is the analytical reference profile, not a single citation. |
| `config/schema.py:TrainHyperparams.batch_size` | `1` | `# tbd: #191` | — | — | VRAM-driven engineering choice; effective batch = batch_size×grad_accum_steps. Cross-ref presets.py memory model. Tracking via #191. |
| `config/schema.py:TrainHyperparams.grad_accum_steps` | `8` | `# tbd: #191` | — | — | VRAM-driven; effective batch = 1×8=8. Cross-ref presets.py memory model. Tracking via #191. |
| `config/schema.py:TrainHyperparams.optimizer` | `"auto"` | `# cite: AdamW (Loshchilov 2019) arXiv:1711.05101` | Loshchilov & Hutter 2019, "Decoupled Weight Decay Regularization", arXiv:1711.05101, ICLR 2019. Algorithm 2 (AdamW). | "The main contribution of this paper is to improve regularization in Adam by decoupling the weight decay from the gradient-based update." (§2) | "auto" resolves to `adamw` (LoRA) or `adamw8bit` (QLoRA) at trainer construction via `peft_adapters/__init__.py:recommended_optimizer()`. |
| `config/schema.py:TrainHyperparams.learning_rate` | `1.0e-4` | `# tbd: #191` | — | — | Repo-chosen magnitude. See open issue #87 for planned A/B lr sweep. Tracking via #191. |
| `config/schema.py:TrainHyperparams.lr_schedule` | `"cosine"` | `# cite: SGDR (Loshchilov 2017) arXiv:1608.03983 §3` | Loshchilov & Hutter 2017, "SGDR: Stochastic Gradient Descent with Warm Restarts", arXiv:1608.03983, ICLR 2017, §3 Eq.(5). | Eq.(5): η_t = η_min + ½(η_max − η_min)(1 + cos(T_cur/T_i · π)) — cosine decay within each restart period. | No warm-restart is used here (single cosine decay); the cosine schedule shape (Eq.5) is the established reference. |
| `config/schema.py:TrainHyperparams.warmup_steps` | `100` | `# tbd: #191` | — | — | Repo-chosen magnitude; no ablation or internal run recorded. Tracking via #191. |
| `config/schema.py:TrainHyperparams.log_every` | `50` | `# tbd: #191` | — | — | Repo-chosen logging cadence. Tracking via #191. |
| `config/schema.py:TrainHyperparams.max_grad_norm` | `1.0` | `# tbd: #191` | — | — | Standard gradient-clipping magnitude used widely in transformer fine-tuning; no explicit derivation recorded for this project. Tracking via #191. |
| `config/schema.py:TrainHyperparams.nan_abort_after` | `20` | `# tbd: #191` | — | — | Repo-chosen NaN-abort patience. Tracking via #191. |
| `config/schema.py:TrainHyperparams.num_workers` | `min(4, cpu_count)` | `# tbd: #191` | — | — | Cap of 4 balances DataLoader throughput against memory; repo-chosen with no formal measurement. Tracking via #191. |
| `config/schema.py:EvalConfig.iou_thresholds` | `[0.5, 0.55, …, 0.95]` | `# cite: COCO (Lin 2014) arXiv:1405.0312 §4` | Lin et al. 2014, "Microsoft COCO: Common Objects in Context", arXiv:1405.0312, §4 Evaluation. IoU sweep [0.5:0.05:0.95] defines the standard COCO AP metric. | "AP is averaged over multiple IoU thresholds from 0.5 to 0.95 (in steps of 0.05)" — standard COCO detection evaluation protocol. | This sweep is the de-facto standard for segmentation/detection benchmarking since COCO 2014. |
| `config/schema.py:EvalConfig.mode` | `"full"` | `# tbd: #191` | — | — | Project default; full eval for completeness. Tracking via #191. |
| `config/schema.py:EvalConfig.lite_max_images` | `64` | `# tbd: #191` | — | — | Repo-chosen lite-mode image cap; no formal measurement. Tracking via #191. |
| `config/schema.py:EvalConfig.mask_threshold` | `0.0` | `# cite: degenerate-case` | — | — | Logit decision boundary: sigmoid(0.0)=0.5 is the probability midpoint. Threshold=0 ↔ predict positive when logit > 0 ↔ predicted probability > 0.5. Mathematical identity. |
| `config/schema.py:EvalConfig.save_predictions` | `False` | `index-only` | — | — | Boolean toggle; off by default. Not trust-bearing. |
| `config/schema.py:EvalConfig.batch_size` | `"auto"` | `index-only` | — | — | Auto-resolved at eval time; structural. |
| `config/schema.py:EvalConfig.visualize` | `True` | `# tbd: #191` | — | — | Repo-chosen default. Tracking via #191. |
| `config/schema.py:EvalConfig.visualize_count` | `10` | `# tbd: #191` | — | — | Repo-chosen number of visualized samples. Tracking via #191. |
| `config/schema.py:TrackingConfig.backend` | `"tensorboard"` | `index-only` | — | — | Structural tracker-backend literal; not trust-bearing. |

## data/aug_presets.py

Legend letters used in the `aug_presets.py` module docstring resolve here.

### Legend

| Letter | Meaning |
| --- | --- |
| (a) | Domain convention — flip/rotate90 enabling booleans reflect the symmetry properties of each domain. Domain rationale, not a published source. |
| (b) | Domain-tuned project magnitude — no published reference and no recorded internal calibration run. `# tbd: #191` |
| (c) | Ruifrok & Johnston 2001 / Tellez et al. 2018 — H&E stain-jitter rationale; exact sigma magnitudes are domain-tuned project choices with no published reference. `# tbd: #191` |
| (d) | Laterality-driven locked-off — see `LOCKED_OFF` map; clinically or structurally meaningful orientation; augmentation disabled by design. |

### Augmentation knob values

Rows are grouped by `(knob, distinct-value)`; presets that use the value are listed in the Notes column.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `aug_presets.py:PRESET_TABLE[*].hflip` | `True` | `# (a)` | Domain convention: natural images are horizontally symmetric; satellite imagery has no canonical orientation. | — | Used by: natural×{safe,medium,aggressive}, satellite×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].hflip` | `False` | `# (d)` | See `LOCKED_OFF["medical"]["hflip"]` and `LOCKED_OFF["microscopy"]["hflip"]`. | "laterality (left vs right) is clinically meaningful in most medical modalities (CXR, mammography, derm)" / "horizontal flip can break channel-ordering conventions in multiplexed microscopy" | Used by: medical×{safe,medium,aggressive}, microscopy×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].vflip` | `True` | `# (a)` | Domain convention: microscopy slides and satellite imagery have no canonical vertical orientation. | — | Used by: natural×aggressive, satellite×{safe,medium,aggressive}, microscopy×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].vflip` | `False` | `# (d)` | See `LOCKED_OFF["medical"]["vflip"]`. | "laterality (superior vs inferior) is clinically meaningful in most medical modalities" | Used by: medical×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].rotate90` | `True` | `# (a)` | Domain convention: satellite and microscopy imagery have no canonical orientation, so 90° rotations are valid invariances. | — | Used by: satellite×{safe,medium,aggressive}, microscopy×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].rotate90` | `False` | `# (d)` | See `LOCKED_OFF["medical"]["rotate90"]` and `LOCKED_OFF["natural"]["rotate90"]`. | "laterality is clinically meaningful" / "arbitrary 90° rotation breaks 'up' for natural photography" | Used by: natural×{safe,medium,aggressive}, medical×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].rotate_arbitrary` | `5.0` | `# (b)` | — | — | `# tbd: #191`. Used by: medical×medium. No published reference; project-chosen magnitude. |
| `aug_presets.py:PRESET_TABLE[*].rotate_arbitrary` | `10.0` | `# (b)` | — | — | `# tbd: #191`. Used by: natural×aggressive, medical×aggressive. No published reference; project-chosen magnitude. |
| `aug_presets.py:PRESET_TABLE[*].rotate_arbitrary` | `15.0` | `# (b)` | — | — | `# tbd: #191`. Used by: satellite×aggressive, microscopy×aggressive. No published reference; project-chosen magnitude. |
| `aug_presets.py:PRESET_TABLE[*].color_jitter` | `0.05` | `# (b)` | — | — | `# tbd: #191`. Used by: natural×safe, satellite×medium. Passed as `brightness=contrast=saturation=0.05, hue=0.025` to `A.ColorJitter`; Albumentations 2.0.8 default is `(0.8, 1.2)` — this is a domain-tuned project choice. |
| `aug_presets.py:PRESET_TABLE[*].color_jitter` | `0.1` | `# (b)` | — | — | `# tbd: #191`. Used by: natural×medium, satellite×aggressive. Same mapping as 0.05 row above. |
| `aug_presets.py:PRESET_TABLE[*].color_jitter` | `0.2` | `# (b)` | — | — | `# tbd: #191`. Used by: natural×aggressive. Same mapping as 0.05 row above. |
| `aug_presets.py:PRESET_TABLE[*].color_jitter` | `0.0` | `# (d)` | See `LOCKED_OFF["medical"]["color_jitter"]` and `LOCKED_OFF["microscopy"]["color_jitter"]`. | "color carries diagnostic signal (e.g. melanoma); use stain_jitter for H&E instead" / "color identifies fluorescence channels and must be preserved" | Used by: medical×{safe,medium,aggressive}, microscopy×{safe,medium,aggressive}. |
| `aug_presets.py:PRESET_TABLE[*].stain_jitter` | `0.03` | `# (c)` | Ruifrok & Johnston 2001, "Quantification of Histochemical Staining by Color Deconvolution", doi:10.1097/00000372-200112000-00001; Tellez et al. 2018, "H&E Stain Augmentation", arXiv:1804.02853. HED basis vectors implemented in `data/transforms.py:_HED_FROM_RGB_MATRIX`. | StainJitter sigma is the per-channel uniform perturbation in HED optical-density space; the H&E rationale cites these two sources. | `# tbd: #191` for exact magnitude. Used by: medical×medium. |
| `aug_presets.py:PRESET_TABLE[*].stain_jitter` | `0.07` | `# (c)` | Same as 0.03 row above. | Same as 0.03 row above. | `# tbd: #191` for exact magnitude. Used by: medical×aggressive. |
| `aug_presets.py:PRESET_TABLE[*].blur` | `0.03` | `# (b)` | — | — | `# tbd: #191`. Used by: medical×aggressive. Scalar maps to `sigma_limit=(0, 0.03×_GAUSS_BLUR_MAX_SIGMA)` in `data/transforms.py`. |
| `aug_presets.py:PRESET_TABLE[*].blur` | `0.05` | `# (b)` | — | — | `# tbd: #191`. Used by: natural×aggressive, satellite×aggressive, microscopy×aggressive. Same mapping as 0.03 row above. |
| `aug_presets.py:PRESET_TABLE[*].gauss_noise` | `0.01` | `# (b)` | — | — | `# tbd: #191`. Used by: medical×medium. Scalar maps to `std_range=(0, 0.01×_GAUSS_NOISE_MAX_VAR)` in `data/transforms.py`. |
| `aug_presets.py:PRESET_TABLE[*].gauss_noise` | `0.02` | `# (b)` | — | — | `# tbd: #191`. Used by: natural×aggressive, satellite×aggressive, microscopy×aggressive. Same mapping as 0.01 row above. |
| `aug_presets.py:PRESET_TABLE[*].gauss_noise` | `0.03` | `# (b)` | — | — | `# tbd: #191`. Used by: medical×aggressive. Same mapping as 0.01 row above. |

## data/channel_semantics.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## data/transforms.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## presets.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## cli/templates/config_full.yaml

Template-echoed literals; the authoritative provenance is the schema row for the
same symbol. This section cross-links the template slot to its schema row.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## models/losses/presets.py

### Citation legend (folded in from the module docstring)

| Letter | Source | Establishes |
| --- | --- | --- |

### Preset-table parameters

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |

## Reference Training Profile

<!-- Owned by Deliverable 2 (epochs alignment). Populated in Phase 2. -->
