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
| `config/schema.py:TextPromptConfig.k` | `16` | `# cite: models/sam3.py:MULTIPLEX_CAP` | `src/custom_sam_peft/models/sam3.py` line 116: `MULTIPLEX_CAP: int = 16` | `MULTIPLEX_CAP: int = 16` — hard cap from SAM 3.1 head design. | Must equal MULTIPLEX_CAP; upper bound enforced by Field(le=16). |
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
| `config/schema.py:MultiplexConfig.classes_per_forward` | `16` | `# cite: models/sam3.py:MULTIPLEX_CAP` | `src/custom_sam_peft/models/sam3.py` line 116: `MULTIPLEX_CAP: int = 16` | `MULTIPLEX_CAP: int = 16` — hard cap from SAM 3.1 model head. | Default=cap means maximum throughput per forward pass. Upper bound enforced by Field(le=16). |
| `config/schema.py:TrainHyperparams.epochs` | `required (template $epochs slot)` | `# cite: SAMed (Zhang & Liu 2023)` / `# tbd: #193` | See "Reference Training Profile" section below. | See "Reference Training Profile" section below. | Required field; no schema default. The shipped default lives in the `config_full.yaml` `$epochs` slot, set by the `init` flow. Provenance is the SAMed convergence anchor (see Reference Training Profile), not a single inline citation. |
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
| `data/channel_semantics.py:_IMAGENET_MEAN` | `(0.485, 0.456, 0.406)` | `# cite: ImageNet-1k stats (torchvision)` | torchvision `_presets.py` lines 52–53 (ImageClassification defaults). URL: <https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py> | `mean: tuple[float, ...] = (0.485, 0.456, 0.406)` | ImageNet-1k per-channel training-set means; same values verified in `config/schema.py:NormalizeConfig.mean`. Used by `rgb` and `rgba` profiles (unpacked via `*_IMAGENET_MEAN`). |
| `data/channel_semantics.py:_IMAGENET_STD` | `(0.229, 0.224, 0.225)` | `# cite: ImageNet-1k stats (torchvision)` | torchvision `_presets.py` lines 52–53. URL: <https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py> | `std: tuple[float, ...] = (0.229, 0.224, 0.225)` | ImageNet-1k per-channel training-set standard deviations; same values verified in `config/schema.py:NormalizeConfig.std`. Used by `rgb` and `rgba` profiles. |
| `data/channel_semantics.py:CHANNEL_SEMANTICS["rgb"].normalize_default` | `(_IMAGENET_MEAN, _IMAGENET_STD)` | `# cite: ImageNet-1k stats (torchvision)` | (See `_IMAGENET_MEAN`/`_IMAGENET_STD` rows above.) | — | Passthrough RGB profile; inherits the two module-level constants directly. |
| `data/channel_semantics.py:CHANNEL_SEMANTICS["rgba"].normalize_default` | `((*_IMAGENET_MEAN, 0.5), (*_IMAGENET_STD, 0.5))` | `# cite: degenerate-case (neutral alpha)` | — | — | The appended `0.5` for both mean and std is a degenerate-case neutral value: for an alpha channel in [0,1], mean=0.5 and std=0.5 map to zero-centred, unit-range output ((x−0.5)/0.5 ∈ [−1,1]). No published source; mathematical identity. |
| `data/channel_semantics.py:CHANNEL_SEMANTICS["grayscale"].normalize_default` | `((0.449,), (0.226,))` | `# cite: torchvision grayscale-ImageNet` | torchvision single-channel ImageNet convention: mean of RGB means = (0.485+0.456+0.406)/3 = 0.4490; mean of RGB stds = (0.229+0.224+0.225)/3 = 0.2260. URL: <https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py> | (Arithmetic from `_IMAGENET_MEAN`/`_IMAGENET_STD` above.) | 0.449 ≈ mean(0.485, 0.456, 0.406) = 0.4490 (rounded); 0.226 = mean(0.229, 0.224, 0.225) = 0.2260 exactly. Standard torchvision grayscale-ImageNet single-channel convention. |
| `data/channel_semantics.py:CHANNEL_SEMANTICS["freeform"].normalize_default` | `None` | `index-only` | — | — | Index-only: `None` signals that explicit user-supplied stats are required. No default to cite. |

## data/transforms.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `data/transforms.py:KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` | `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` | `# cite: ImageNet stats (HF Sam3ImageProcessor)` | ImageNet-1k per-channel training statistics; confirmed by 2026-05-21 audit of `AutoImageProcessor.from_pretrained("facebook/sam3.1").image_mean/image_std`. Same values in `config/schema.py:NormalizeConfig` and `data/channel_semantics.py:_IMAGENET_MEAN/_IMAGENET_STD`. | `mean: (0.485, 0.456, 0.406)`, `std: (0.229, 0.224, 0.225)` — as returned by HF `Sam3ImageProcessor` and matching torchvision `_presets.py` ImageNet-1k defaults. | Ratified by 2026-05-21 config-defaults audit (supersedes 2026-05-16 spec's `[0.5, 0.5, 0.5]` claim). Cross-ref open issue #86 (empirical processor-stats verification). |
| `data/transforms.py:_STATS_DIVERGENCE_ATOL` | `1e-3` | `# cite: empirical (tolerance chosen to catch [0.5,0.5,0.5] drift)` | Project engineering choice — inline comment rationale: "Loose enough to absorb float-serialization noise; tight enough to catch a real change (e.g. `[0.5, 0.5, 0.5]` diverges by >=0.014 per channel)." | — | The `[0.5, 0.5, 0.5]` reference (0.014 delta per channel) establishes that 1e-3 provides a 14× safety margin over the known bad value while absorbing sub-LSB float-serialization noise. |
| `data/transforms.py:_HED_FROM_RGB_MATRIX` | `[[0.65,0.70,0.29],[0.07,0.99,0.11],[0.27,0.57,0.78]]` | `# cite: Ruifrok & Johnston 2001` | Ruifrok & Johnston 2001, "Quantification of histochemical staining by color deconvolution", Anal Quant Cytol Histol 23(4):291–299. PMID 11531144. doi:[10.1097/00000372-200112000-00001](https://doi.org/10.1097/00000372-200112000-00001) | Table 1 stain OD vectors (2-decimal representation): H = [0.65, 0.70, 0.29], E = [0.07, 0.99, 0.11], DAB = [0.27, 0.57, 0.78]. | Rows are the published OD (optical density) basis vectors for H, E, DAB from Table 1 of the paper, rounded to 2 decimal places. Matches the standard 2-decimal form widely reproduced in color-deconvolution implementations. The matrix `_HED_FROM_RGB_MATRIX` is used as the forward map (HED→OD, i.e. `rgb_from_hed` direction); its inverse `_HED_FROM_RGB_INV` is the deconvolution transform (OD→HED). The variable name reflects the math role of the inverse, not the literal matrix. |
| `data/transforms.py:_GAUSS_NOISE_MAX_VAR` | `0.05` | `# tbd: #191` | — | — | Magnitude→Albumentations projection ceiling for `std_range` in `A.GaussNoise`; spec §8.1 reference is internal only. Tracking via #191. |
| `data/transforms.py:_GAUSS_BLUR_MAX_SIGMA` | `3.0` | `# tbd: #191` | — | — | Magnitude→Albumentations projection ceiling for `sigma_limit` in `A.GaussianBlur`; spec §8.1 reference is internal only. Tracking via #191. |

## presets.py

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `presets.py:MODEL_PARAMS` | `5_000_000_000` | `# cite: scripts/_derive_preset_constants.py` | `scripts/_derive_preset_constants.py` in repo root — SAM 3.1 checkpoint parameter count derivation script. Existing inline comment: "Re-derive via scripts/_derive_preset_constants.py". | Script confirmed present at `scripts/_derive_preset_constants.py`. | Analytic seed; superseded by calibration cache. Vision encoder ~762 M + text encoder ~302 M + decoder/neck ~50 M. |
| `presets.py:LORA_LAYERS` | `96` | `# cite: empirical (#148/#179 VRAM calibration)` | GitHub issue #148 "Reassess calibration mode" (CLOSED) and PR #179 "feat(presets): VRAM calibration reassessment — K\_eff activation, SDPA attn term, opt-in live probe" (MERGED). | Count of nn.Linear LoRA targets in the `vision_decoder` scope, derived from `_resolve_targets` during calibration runs in #148/#179. | empirical calibration; not a paper constant. |
| `presets.py:D_IN` | `768` | `# cite: empirical (#148/#179 VRAM calibration)` | Same as LORA\_LAYERS — average input feature dim across LoRA targets measured during #148/#179 calibration runs. | — | empirical calibration. |
| `presets.py:D_OUT` | `768` | `# cite: empirical (#148/#179 VRAM calibration)` | Same as LORA\_LAYERS — average output feature dim across LoRA targets measured during #148/#179 calibration runs. | — | empirical calibration. |
| `presets.py:Q_OVERHEAD` | `64 MiB` | `# cite: empirical (#148/#179 VRAM calibration)` | bitsandbytes NF4 per-block scale + zero-point overhead; magnitude calibrated in #148/#179. | — | empirical calibration. NF4 stores per-block quantization metadata (scale/offset); the 64 MiB figure was set during VRAM probe runs. |
| `presets.py:WORKSPACE_BYTES` | `256 MiB` | `# cite: empirical (#148/#179 VRAM calibration)` | cuDNN workspace + autograd graph + tmp buffers headroom; calibrated in #148/#179 (spec §3). | — | empirical calibration. |
| `presets.py:BASE_ACTIVATION_AT_1024` | `int(1.5 * GB)` | `# cite: empirical (#148/#179 VRAM calibration)` | Analytic seed for per-example activation bytes at image\_size=1024; calibrated in #148/#179. Superseded by calibration cache when present. | — | empirical calibration seed. |
| `presets.py:forward_only_factor` | `0.25` | `# cite: empirical (#148/#179 VRAM calibration)` | Forward-only eval memory is ~1/4 of the train-step probe (train = forward + backward + retained graph; eval = forward only, no retained graph). Calibrated in #148/#179 (spec §8). | — | empirical calibration. Note: K (classes\_per\_forward) is folded into this factor empirically rather than computed analytically (spec §8 / decide\_eval\_batch\_size docstring). |
| `presets.py:_SAM3_PATCH` | `14` | `# cite: sam3/model_builder.py` | `sam3/model_builder.py` line 82: `patch_size=14` in the hiera-large backbone constructor. Existing block comment: "SAM 3.1 vision backbone (hiera-large), from sam3/model\_builder.py." | `patch_size=14` — hiera-large vision backbone constructor argument. | Reference-implementation value; patch size governs token count N=(image\_size//patch)^2 used in `_attention_bytes_per_example`. |
| `presets.py:_SAM3_HEADS` | `16` | `# cite: sam3/model_builder.py` | `sam3/model_builder.py` line 85: `num_heads=16` in the hiera-large backbone constructor. Same block comment as `_SAM3_PATCH`. | `num_heads=16` — hiera-large vision backbone constructor argument. | Reference-implementation value; head count H used in `_attention_bytes_per_example`: H\*N^2\*4 bytes. |
| `presets.py:_bytes_per_param_for_method (2.0)` | `2.0 B/param` | `# cite: framework default` | PyTorch dtype sizes: `torch.bfloat16` and `torch.float16` are 16-bit = 2 bytes per element. URL: <https://pytorch.org/docs/stable/tensors.html> | "torch.bfloat16: 16-bit Brain floating point" / "torch.float16: 16-bit half-precision floating point" — each is 2 bytes. | Standard bf16/fp16 dtype width; not project-specific. |
| `presets.py:_bytes_per_param_for_method (0.5)` | `0.5 B/param` | `# cite: framework default` | bitsandbytes NF4 quantization: 4-bit storage = 0.5 bytes per parameter. bitsandbytes docs / QLoRA paper (Dettmers 2023, arXiv:2305.14314 §3): "4-bit NormalFloat Quantization". | "4-bit NormalFloat" — 4 bits per parameter = 0.5 bytes. | Standard NF4 storage width; not project-specific. |
| `presets.py:_optimizer_bytes (*4 literal)` | `4× adapter_bytes` | `# cite: framework default` | AdamW optimizer state = fp32 first moment m + fp32 second moment v + fp32 master copy = 3 × 4 B/param = 12 B/param for a bf16 2 B/param adapter → ratio = 12/2 = 6×. However, if master copy is omitted (mixed-precision AdamW without separate master weights), state = m + v = 2 × 4 B = 8 B/param → ratio = 8/2 = 4×. The `*4` literal implements the 8 B/param (m+v only) variant. Loshchilov & Hutter 2019, arXiv:1711.05101. PyTorch `torch.optim.AdamW` stores m and v in fp32 by default. | "AdamW state on the bf16 adapter — fp32 m, fp32 v, fp32 master copy. Adapter weights are 2 B/param; state is 8 B/param -> 4x adapter\_bytes." (presets.py inline comment). | No `ADAMW_STATE_MULT` symbol exists in the codebase (`rg -n 'ADAMW_STATE_MULT' src/` returns nothing) — the issue/plan mislabeled the literal; it is just `* 4` directly in `_optimizer_bytes`. The `*4` = 8 B/param ÷ 2 B/param (bf16 adapter) = the m+v fp32 state ratio. |
| `presets.py:CACHE_SCHEMA_VERSION` | `2` | `index-only` | — | — | Internal cache versioning integer; not trust-bearing. Incremented when the cache JSON schema changes in a backward-incompatible way. |

## cli/templates/config_full.yaml

Template-echoed literals; the authoritative provenance is the schema row for the
same symbol. This section cross-links the template slot to its schema row.

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `config_full.yaml:run.seed` | `42` | `cross-link` | See `config/schema.py:RunConfig.seed` row. | — | Template echo of the schema default. |
| `config_full.yaml:model.dtype` | `bfloat16` | `cross-link` | See `config/schema.py:ModelConfig.dtype` row. | — | Template echo of the schema default. |
| `config_full.yaml:data.text_prompt.mode` | `present_plus_negatives` | `cross-link` | See `config/schema.py:TextPromptConfig.mode` row. | — | DIFFERS from schema default (`present`). Template ships `present_plus_negatives` to pair with `negatives_per_image: 4`; schema default is the conservative `present` (0 negatives). |
| `config_full.yaml:data.text_prompt.negatives_per_image` | `4` | `cross-link` | See `config/schema.py:TextPromptConfig.negatives_per_image` row. | — | DIFFERS from schema default (`0`). Template ships `4` per the field-description rationale: "leaves headroom for typical COCO present-class counts (~3-7 per image)". Schema default is `0` (conservative starting point before negative-mining is enabled). |
| `config_full.yaml:data.normalize.mean` | `[0.485, 0.456, 0.406]` | `cross-link` | See `config/schema.py:NormalizeConfig.mean` row. | — | Template echo of the schema default. |
| `config_full.yaml:data.normalize.std` | `[0.229, 0.224, 0.225]` | `cross-link` | See `config/schema.py:NormalizeConfig.std` row. | — | Template echo of the schema default. |
| `config_full.yaml:peft.r` | `16` | `cross-link` | See `config/schema.py:PEFTConfig.r` row. | — | Template echo of the schema default. |
| `config_full.yaml:peft.alpha` | `32` | `cross-link` | See `config/schema.py:PEFTConfig.alpha` row. | — | Template echo of the schema default. |
| `config_full.yaml:peft.dropout` | `0.05` | `cross-link` | See `config/schema.py:PEFTConfig.dropout` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.epochs` | `$epochs` | `cross-link` | See `config/schema.py:TrainHyperparams.epochs` row + "Reference Training Profile". | — | Placeholder filled by the `init` flow (default `160`; see Reference Training Profile). |
| `config_full.yaml:train.batch_size` | `1` | `cross-link` | See `config/schema.py:TrainHyperparams.batch_size` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.grad_accum_steps` | `8` | `cross-link` | See `config/schema.py:TrainHyperparams.grad_accum_steps` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.learning_rate` | `1.0e-4` | `cross-link` | See `config/schema.py:TrainHyperparams.learning_rate` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.lr_schedule` | `cosine` | `cross-link` | See `config/schema.py:TrainHyperparams.lr_schedule` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.warmup_steps` | `100` | `cross-link` | See `config/schema.py:TrainHyperparams.warmup_steps` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.max_grad_norm` | `1.0` | `cross-link` | See `config/schema.py:TrainHyperparams.max_grad_norm` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.log_every` | `50` | `cross-link` | See `config/schema.py:TrainHyperparams.log_every` row. | — | Template echo of the schema default. |
| `config_full.yaml:train.nan_abort_after` | `20` | `cross-link` | See `config/schema.py:TrainHyperparams.nan_abort_after` row. | — | Template echo of the schema default. |
| `config_full.yaml:eval.iou_thresholds` | `[0.5, 0.55, …, 0.95]` | `cross-link` | See `config/schema.py:EvalConfig.iou_thresholds` row. | — | Template echo of the schema default. |
| `config_full.yaml:tracking.backend` | `tensorboard` | `cross-link` | See `config/schema.py:TrackingConfig.backend` row. | — | Template echo of the schema default. |
| `config_full.yaml:export.merge` | `false` | `cross-link` | See `config/_internal.py:ExportConfig.merge` row. | — | Template echo of the schema default. |

## models/losses/presets.py

### Citation legend (folded in from the module docstring)

| Letter | Source | Establishes |
| --- | --- | --- |
| A | Issue #112 body — draft preset × class_imbalance table in the original brainstorming issue. | Values lifted verbatim from the project's design table; not an external literature source. |
| B | Preserved pre-#112 hardcoded defaults from `models/losses.py` trainer behavior. | Structural continuity of existing behavior before the preset system was introduced. |
| C | Lin et al. 2017 "Focal Loss for Dense Object Detection", arXiv:1708.02002, Table 1. | γ=2.0, α=0.25 from the RetinaNet best-performing configuration in Table 1. |
| D | Abraham & Khan 2019 "A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation", arXiv:1810.07842, §2.1. | Focal-Tversky exponent: paper trains with γ_paper=4/3 so (1-TI)^(1/γ_paper) = (1-TI)^0.75; code uses `tversky_gamma=0.75` directly as the exponent. Paper's best α=0.7 (FP weight), β=0.3 (FN weight) in their notation (§2.1: "we train all models with α=0.7 and β=0.3"). |
| E | Salehi et al. 2017 "Tversky loss function for image segmentation using 3D fully convolutional deep networks", arXiv:1706.05721, Table 1. | Best FN-penalization weight on MS lesion segmentation: paper's β=0.7 (FN weight), α=0.3 (FP weight). Code convention: `tversky_alpha` weights FN (= paper's β), so `tversky_alpha=0.7` corresponds to Salehi's best β=0.7. Verifying quote (Experiments): "the best results were obtained from the FCN trained with β=0.7, which performed much better than the FCN trained with the Dice loss layer". |
| F | Degenerate-case identity: α=0.5 reduces Tversky to Dice; γ=1.0 reduces Focal-Tversky to Tversky. | Mathematical identity — no external citation required. |
| G | Alias-of-medical — microscopy copies medical presets (spec §5.2). | Placeholder pending a real microscopy user/dataset justification. `# tbd: #191` (see issue #120). |
| H | Kervadec et al. 2019 "Boundary loss for highly unbalanced segmentation", arXiv:1812.07032, MIDL 2019 (also IEEE TMI 2021 doi:10.1109/TMI.2021.3113078). | Boundary loss blend coefficient; paper uses ~0.2 as a representative blending weight for the boundary loss term alongside a region loss. |

### Preset-table parameters

| Location | Value | Tag | Full reference | Verifying quote | Notes |
| --- | --- | --- | --- | --- | --- |
| `models/losses/presets.py:PRESET_TABLE[("natural","moderate")].focal_gamma` | `2.5` | `# tbd: #191` | — | — | Escalated above Lin et al.'s γ=2.0 in the issue #112 draft table (legend A) as a moderate-imbalance step; no external paper specifies γ=2.5 and no internal calibration run has been recorded. |
| `models/losses/presets.py:PRESET_TABLE[("natural","severe")].focal_gamma` | `3.0` | `# tbd: #191` | — | — | Severe-imbalance step from the issue #112 draft table (legend A); no external paper specifies γ=3.0 and no internal calibration run has been recorded. |
| `models/losses/presets.py:PRESET_TABLE[("medical","moderate")].focal_gamma` | `2.5` | `# tbd: #191` | — | — | Same rationale as `("natural","moderate").focal_gamma`; issue #112 design table only. |
| `models/losses/presets.py:PRESET_TABLE[("medical","severe")].focal_gamma` | `3.0` | `# tbd: #191` | — | — | Same rationale as `("natural","severe").focal_gamma`; issue #112 design table only. |
| `models/losses/presets.py:PRESET_TABLE[("satellite","moderate")].focal_gamma` | `2.5` | `# tbd: #191` | — | — | Same rationale as `("natural","moderate").focal_gamma`; issue #112 design table only. |
| `models/losses/presets.py:PRESET_TABLE[("satellite","severe")].focal_gamma` | `3.0` | `# tbd: #191` | — | — | Same rationale as `("natural","severe").focal_gamma`; issue #112 design table only. |
| `models/losses/presets.py:PRESET_TABLE[("medical","moderate")].tversky_alpha` | `0.7` | `# cite: (A,E)` | Salehi et al. 2017, "Tversky loss function for image segmentation using 3D fully convolutional deep networks", arXiv:1706.05721, Table 1. | Experiments: "the best results were obtained from the FCN trained with β=0.7, which performed much better than the FCN trained with the Dice loss layer". β=0.7 is the FN-penalization weight in Salehi's notation; code's `tversky_alpha` is the FN weight (`tp + alpha*fn + (1-alpha)*fp`), so `tversky_alpha=0.7` = Salehi's β=0.7. | Naming-convention note: Salehi's α (FP weight) = 0.3; their β (FN weight) = 0.7 = this code's `tversky_alpha`. Abraham & Khan 2019 use the opposite convention (their α=FP, β=FN), but also report best FN weight of 0.3 in their convention (= FP-emphasis), which is inconsistent with the 0.7 FN weight here. The Salehi source is the correct citation for `tversky_alpha=0.7` as an FN-penalization weight. |
| `models/losses/presets.py:PRESET_TABLE[("satellite","severe")].tversky_alpha` | `0.7` | `# cite: (A,E)` | Salehi et al. 2017, arXiv:1706.05721 (same as above). | Experiments: "the best results were obtained from the FCN trained with β=0.7 ...". | Same as medical/moderate row above; microscopy/moderate and microscopy/severe inherit this via alias (legend G). |
| `models/losses/presets.py:PRESET_TABLE[("medical","severe")].tversky_alpha` | `0.8` | `# tbd: #191` | — | — | Further FN-bias escalation from the issue #112 design table (legend A); no external paper specifies α=0.8 as an FN-penalization weight and no internal calibration run has been recorded. |

## Reference Training Profile

The shipped training defaults form the following reference profile:

- `batch_size = 1`, `grad_accum_steps = 8` → effective batch = 8 (schema defaults in `config/schema.py:TrainHyperparams`).
- `epochs = 160` (shipped default, set by the `init` flow via the `config_full.yaml` `$epochs` slot).
- `eval.mode = "full"` (schema default in `config/schema.py:EvalConfig.mode`).

### Anchor: SAMed (Zhang & Liu, 2023)

The `epochs = 160` value is anchored to a **convergence figure**, not a runtime budget. The closest published analog to this repo's use case is **SAMed** — LoRA fine-tuning of SAM (LoRA rank 4, AdamW) on a small medical dataset — which is the same regime this repo targets (PEFT/LoRA on SAM, small dataset).

**Primary source:** Zhang & Liu, 2023, "Customized Segment Anything Model for Medical Image Segmentation", arXiv:2304.13785.

Key verifying quotes:

- Sec 4.2: "We adopt early stop at 14880 iterations (160 epochs)."
- Sec 4.1: "the training set contains 2212 axial slices" (18 cases — a small dataset directly comparable to this repo's target regime).
- Abstract: "After finetuning only 160 epochs on Synapse multi-organ segmentation dataset (Synapse), SAMed achieves 81.88 DSC."

The 160-epoch figure is therefore a **convergence anchor** drawn from the published LoRA-on-SAM small-dataset literature, not an arbitrary budget choice.

### Convergence vs. runtime tradeoff

At 160 epochs the run no longer fits the original "≤30 min on a 16 GB free-tier Colab T4" window that the earlier framing assumed: 160 epochs is ~16× the previous 10-epoch default, so the run exceeds that window by a wide margin. This is an order-of-magnitude inference from the epoch ratio, **not** a measured figure. The "≤30 min" budget framing is therefore **dropped** in favor of convergence. This reflects the standing design priority for this project: **final accuracy ≫ training speed** — a speed-only benefit is not a sufficient reason to reduce epoch count.

There is **no citable T4 per-step wall-clock figure** in the literature for this configuration, so the actual wall-clock is left unmeasured: `# tbd: #193` (empirical T4 confirmation of the reference profile). No runtime figure is stated as a measured or completed claim here.

For empirical GPU-test budget questions — including the 2-image overfit smoke-test — see **issue #195** (2-image overfit GPU smoke-test speed/convergence).
