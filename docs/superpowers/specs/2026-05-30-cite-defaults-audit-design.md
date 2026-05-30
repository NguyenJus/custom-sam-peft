# Design: Literature-Cite Every Default Hyperparameter (Issue #120)

## Overview

Every default hyperparameter baked into `custom-sam-peft` should be traceable
to a published source, a reference implementation, or an explicit empirical
justification. Today only one table carries provenance — the loss-preset table
at `src/custom_sam_peft/models/losses/presets.py` (112 inline `# cite:` tags
plus an A–H letter-legend in its module docstring). Every other default —
schema `Field` defaults, the augmentation preset table, channel-normalization
statistics, transform constants, and the VRAM-model coefficients — ships
uncited.

This audit produces two coupled deliverables:

1. **Inline provenance** — a concise `# cite:` / `# tbd:` tag on every
   *trust-bearing* default in the in-scope files, plus firming-up of the
   remaining `# citation needed` cells in the loss-preset table.
2. **A central index** — `docs/defaults-provenance.md`, one row per default,
   grouped by file, carrying the full reference (authors, year, arXiv/DOI,
   exact equation/table/figure) and a verifying quote. This file is the source
   of truth; the inline tags are deliberately terse pointers into it.

A third, narrowly-scoped change rides along because it is the one default whose
provenance *is* a recipe rather than a citation: the shipped `epochs` default.
We document a **reference training profile** (dataset assumption, batch/grad-
accum, epochs, eval mode) bounded by a ≤30-minute free-Colab T4 budget, derived
analytically, and align the shipped `epochs` default and the end-to-end
integration test to it.

This is a **documentation-and-provenance** effort. With the single sanctioned
exception of the `epochs` alignment, **no numeric default is re-tuned**.

## Goals

- Attach a concise, correct inline tag to every trust-bearing default in the
  in-scope files.
- Resolve every `# citation needed` comment in
  `models/losses/presets.py` to a real cite or a `# tbd: #<issue>` tracker —
  zero `# citation needed` remain.
- Build `docs/defaults-provenance.md` with one verified row per default,
  grouped by file, folding in the existing loss-preset A–H legend so the whole
  tree resolves from one document. It passes the project markdownlint gate.
- Document a reference T4 training profile and set the shipped `epochs` default
  (the `config_full.yaml` template the `init` flow renders) to its value.
- Make the end-to-end integration test exercise the *real default
  hyperparameters* (lr / scheduler / optimizer / PEFT config), with epoch count
  reduced solely for CI runtime and a comment saying so.

## Non-Goals

- **Re-tuning any value** except the sanctioned `epochs` alignment. Verification
  records what a value *is*, not what it *should be*.
- **Inline-tagging self-evident structural / string defaults** (e.g. the
  `HFFieldMap` field-name maps like `image="image"`, `output_dir="./runs"`,
  `model.name="facebook/sam3.1"`, `checkpoint_file`). These are *indexed* in the
  doc but carry no inline tag.
- **Citing `configs/examples/*.yaml`** (8 files: `coco_text_lora.yaml`,
  `gpu_smoke_lora.yaml`, etc.). These are intentionally-varied smoke/
  illustrative configs that inherit the template's rationale; no per-file cites.
- **A CI "no-uncited-default" enforcement hook.** Filed as a follow-up issue.
- **Adding new defaults.**
- **Inventing a citation for a project-chosen number.** No external source AND
  no internal run → `# tbd:` + the umbrella tracking sub-issue. Never fabricate.
- **Running a real T4 benchmark.** The epoch figure is derived analytically; an
  empirical T4 confirmation is filed as a `# tbd:` follow-up, not claimed done.

## Reality Check vs. the Issue Body

The issue body (#120) is stale in several places; all are resolved here. These
were verified against the tree on the audit branch:

- The issue lists **#112** (loss-preset table) and **#75** (augmentation preset
  table) as *incoming dependencies*. **Both are CLOSED / merged.** Both tables
  exist in final form (`models/losses/presets.py`, `data/aug_presets.py`).
  There is no blocking dependency; this audit cites the *shipped* artifacts.
- Issues **#148 / #179** (VRAM calibration) are also CLOSED / merged. Their
  calibration methodology is the documented provenance for the empirical VRAM
  coefficients in `presets.py`.
- The issue's framing assumes the VRAM model exposes a runtime/throughput
  estimator. **It does not.** `presets.py` is `decide_preset()` plus a
  *memory-only* model (`_predicted_bytes`, `_activation_bytes`,
  `_attention_bytes_per_example`); there is **no `estimate_runtime`,
  no `estimate_vram`, and no wall-clock / T4 30-minute budget assertion**
  anywhere in `src/` or `tests/gpu/`. The training-recipe derivation (§2.2) is
  re-grounded on what actually exists.

Current state confirmed in the tree (file:line where useful):

- `models/losses/presets.py` — 112 `# cite:` tags + A–H legend (docstring lines
  13–27); **9** in-table `# citation needed` cells: `focal_gamma` at lines 116,
  131, 162, 177, 208, 223 and `tversky_alpha` at lines 164, 179, 225 (plus the
  one docstring mention at line 25). All must be firmed or converted.
- `data/aug_presets.py` — **0** cites; 12 cells (4 presets × 3 intensities:
  `safe|medium|aggressive`), 8 knobs/cell (`hflip`, `vflip`, `rotate90`,
  `rotate_arbitrary`, `color_jitter`, `stain_jitter`, `blur`, `gauss_noise`).
- `config/schema.py` — ~60 user-facing defaults across `RunConfig`,
  `ModelConfig`, `TextPromptConfig`, `NormalizeConfig`, `PEFTConfig`,
  `QLoRAConfig`, `BoxHintSchedule`, `MultiplexConfig`, `TrainHyperparams`,
  `EvalConfig`, `TrackingConfig`, `ValSplitConfig`, `LimitConfig`, `DataConfig`.
- `config/_internal.py` — `MatcherWeights` (`lambda_l1=0.0`, `lambda_giou=0.0`,
  `lambda_mask=5.0`), `WandbConfig` (`project="custom_sam_peft"`), `ExportConfig`
  (`merge=False`). (Note: the issue's `MatcherWeights.lambda_mask=5.0` is real;
  its mentioned `lambda_bbox`/`lambda_dice`/`no_object_weight`,
  `BoxHintSchedule.decay_steps=5000`, `warmup_steps=100`,
  `TrainHyperparams.learning_rate` field name etc. partly differ — see §1.4.)
- `data/channel_semantics.py` — `_IMAGENET_MEAN`, `_IMAGENET_STD` (lines 15–16),
  four `CHANNEL_SEMANTICS` profiles with `normalize_default` tuples (rgb, rgba,
  grayscale `(0.449,)/(0.226,)`, freeform=None).
- `data/transforms.py` — `KNOWN_PROCESSOR_STATS["facebook/sam3.1"]` (line 60),
  `_STATS_DIVERGENCE_ATOL=1e-3`, `_HED_FROM_RGB_MATRIX` (Ruifrok & Johnston
  2001 basis, lines 76–83), `_GAUSS_NOISE_MAX_VAR=0.05`,
  `_GAUSS_BLUR_MAX_SIGMA=3.0`.
- `presets.py` — VRAM-model constants `MODEL_PARAMS=5e9`, `LORA_LAYERS=96`,
  `D_IN=768`, `D_OUT=768`, `Q_OVERHEAD=64 MiB`, `WORKSPACE_BYTES=256 MiB`,
  `BASE_ACTIVATION_AT_1024=1.5 GiB`, `forward_only_factor=0.25`,
  `_SAM3_PATCH=14`, `_SAM3_HEADS=16`, `_bytes_per_param_for_method` (2.0 bf16 /
  0.5 NF4), `_optimizer_bytes` 4× multiplier, `CACHE_SCHEMA_VERSION=2`.
  **No pre-existing `# cite:` tags exist in this file** (contrary to any
  assumption that `ADAMW_STATE_MULT` is already cited — that symbol does not
  exist; the AdamW state cost is the inline `*4` in `_optimizer_bytes`).
- The only file currently carrying `# cite:` tags is
  `models/losses/presets.py`.
- The user-facing template is `src/custom_sam_peft/cli/templates/
  config_full.yaml` (the only template; there is no `config_minimal.yaml`). Its
  `epochs:` slot is a `$epochs` placeholder filled by the `init` flow; the
  literal values (`r: 16`, `alpha: 32`, `dropout: 0.05`, `learning_rate:
  1.0e-4`, `lr_schedule: cosine`, `warmup_steps: 100`, `max_grad_norm: 1.0`,
  `negatives_per_image: 4`, ImageNet `mean`/`std`) are echoed in the template.

## Deliverable 1 — Inline Provenance + Central Index

### 1.1 Tag taxonomy (inline, concise)

Inline tags are intentionally terse and non-authoritative. The *full* reference
lives in `docs/defaults-provenance.md`. Allowed tag classes:

| Tag form | Meaning |
| --- | --- |
| `# cite: <paper> §/Eq.N` | Value backed by a published paper at an exact location. |
| `# cite: <repo>:<path>` | Value mirrors a reference implementation (file/line). |
| `# cite: empirical (<dataset>, run <id>)` | Value from an internal calibration/training run. |
| `# cite: degenerate-case` | Value is a math identity / limiting case (e.g. Tversky α=0.5 ≡ Dice). |
| `# cite: framework default` | Value is the upstream library default (Albumentations / torch / HF). |
| `# tbd: #<sub-issue>` | No external source and no internal run; tracked by a sub-issue. |

Dense tables (`aug_presets.py`) use a **per-file letter-legend** in the module
docstring — exactly like the loss-preset A–H legend — and cells reference legend
letters instead of repeating long tags. Self-evident structural/string defaults
(Non-Goals) carry no inline tag but still get an index row.

### 1.2 Central doc schema — `docs/defaults-provenance.md`

The doc is the source of truth. Structure:

- **Top section — Verification Standard.** States the rule: every literature-
  backed value is verified against its *primary* source with a captured quote +
  URL/DOI + exact equation/table/figure; framework defaults link upstream docs;
  degenerate cases state the math identity; reference-impl values cite file/line;
  unsourced project numbers are `# tbd:` with a tracker — never fabricated.
- **One section per in-scope file**, in audit order (below).
- **One table row per default**, columns:

  | Column | Content |
  | --- | --- |
  | Location | `file:symbol` (e.g. `config/schema.py:TrainHyperparams.learning_rate`). |
  | Value | The literal default. |
  | Tag | The inline tag class applied (mirrors the code). |
  | Full reference | Authors, year, arXiv/DOI, exact Eq./Table/Fig. |
  | Verifying quote | Short quote from the primary source establishing the value. |
  | Notes | Caveats, degenerate-case identities, calibration run pointers. |

- **Loss-preset legend fold-in.** The A–H legend currently in
  `models/losses/presets.py` (docstring lines 13–27) is reproduced (and
  expanded with quotes/DOIs) in this doc so the entire tree resolves from one
  file.

### 1.3 Interface contract between the two deliverables

The two deliverables meet at a single, explicit contract so a later
implementation phase can build on an earlier one without re-reading its code:

1. **The row schema** in §1.2 is the contract for all provenance work. Every
   inline tag added anywhere in the tree has exactly one corresponding doc row
   in that schema; every doc row names a real `file:symbol`. A phase that audits
   a file produces (a) the inline tags and (b) the matching doc-section rows
   together.
2. **The `epochs` row** is the contract between Deliverable 1 and Deliverable 2.
   The `config/schema.py:TrainHyperparams.epochs` row's "Full reference" /
   "Notes" cell points at the **Reference Training Profile** section of the same
   doc (Deliverable 2). Deliverable 2 owns that section and the numeric value;
   Deliverable 1 owns the row that links to it. Note `epochs` is a *required*
   field on `TrainHyperparams` (no schema default); its "default" lives in the
   `config_full.yaml` template's `$epochs` slot. This lets the audit phase and
   the training-recipe phase proceed against a fixed seam.

### 1.4 Per-file audit plan

Files listed in suggested audit order. "Tag" column gives the *expected*
dominant tag class; the implementer verifies each value and may downgrade to
`# tbd:` where verification fails.

#### `config/_internal.py` — matcher & internal dataclasses

- `MatcherWeights.lambda_mask=5.0`: `# cite: <repo>:<path>` / `# cite:` → DETR
  (Carion et al. 2020, arXiv:2005.12872) Hungarian-matcher cost weighting (the
  mask/dice cost weight); cross-check the Meta SAM3 / DETR reference impl for the
  ratio. `lambda_l1=0.0`, `lambda_giou=0.0`: `# cite: degenerate-case` — text-
  only v0 disables box terms (the docstring already explains the YAGNI demote).
- `WandbConfig.project="custom_sam_peft"`, `entity=None`, `ExportConfig.merge=
  False`: self-evident structural/boolean defaults — index row, no inline tag.

#### `config/schema.py` — user-facing defaults (~60)

- **PEFT** (`PEFTConfig.r=16`, `alpha=32`, `dropout=0.05`): `# cite:` → LoRA
  (Hu et al. 2021, arXiv:2106.09685) for rank/alpha/dropout conventions (note
  α=2r convention). `scope="vision_decoder"`, `bias="none"`,
  `QLoRAConfig.quant_type="nf4"`, `use_double_quant=False`: `# cite:` → QLoRA
  (Dettmers et al. 2023, arXiv:2305.14314) for NF4 / double-quant; framework
  default where applicable.
- **Optimizer/schedule** (`learning_rate=1.0e-4`, `lr_schedule="cosine"`,
  `optimizer="auto"`, `warmup_steps=100`, `max_grad_norm=1.0`): mix of
  `# cite:` (cosine — Loshchilov & Hutter 2017 SGDR, arXiv:1608.03983; AdamW
  resolved by `auto` — Loshchilov & Hutter 2019, arXiv:1711.05101) and
  `# cite: empirical` / `# tbd:` for repo-chosen magnitudes (lr, warmup).
- **Batch/accum** (`batch_size=1`, `grad_accum_steps=8`): `# cite: empirical` /
  `# tbd:` — VRAM-driven engineering choice (see `presets.py` model).
- **Eval** (`EvalConfig.iou_thresholds=[0.5:0.05:0.95]`, `lite_max_images=64`,
  `mask_threshold=0.0`, `mode="full"`, `visualize=True`, `visualize_count=10`):
  IoU sweep → `# cite:` COCO detection eval protocol (Lin et al. 2014,
  arXiv:1405.0312); the rest `# cite: empirical` / framework / `# tbd:`.
- **Box-hint schedule** (`BoxHintSchedule.p_start=1.0`, `p_end=0.0`,
  `decay_steps=None`): `# cite: degenerate-case` / `# cite: empirical` — the 0.75
  decay fraction is a project choice (documented in the field description).
- **Misc** (`nan_abort_after=20`, `log_every=50`, `num_workers=min(4,cpu)`,
  `MultiplexConfig.classes_per_forward=16`, `TextPromptConfig.k=16`,
  `negatives_per_image=0`, `ValSplitConfig.fraction=0.1`, `seed=42`,
  `NormalizeConfig.max_pixel_value=255.0`): `classes_per_forward`/`k` cap 16 →
  `# cite: <repo>:<path>` SAM 3.1 `MULTIPLEX_CAP=16` (`models/sam3.py`);
  `max_pixel_value=255.0` → `# cite: framework default` (8-bit); `seed=42` →
  convention; remainder `# cite: empirical` / `# tbd:`.
- **`NormalizeConfig.mean/std` defaults** (`[0.485,0.456,0.406]` /
  `[0.229,0.224,0.225]`): `# cite:` → ImageNet-1k stats (torchvision provenance;
  also mirrors `KNOWN_PROCESSOR_STATS` and the `facebook/sam3.1` processor).
- **`epochs`** (required field, no schema default — see §1.3): the *template*
  `$epochs` slot is the contract row linking to Deliverable 2.
- **Structural/string** (`output_dir="./runs"`, `model.name="facebook/sam3.1"`,
  `local_dir`, `checkpoint_file`, `dtype="bfloat16"`, `channel_semantics="rgb"`,
  `format` literals, `HFFieldMap.*`, `bbox_format="xyxy"`, `split_train=
  "train"`, `TrackingConfig.backend="tensorboard"`): index row, no inline tag,
  except where the value is trust-bearing (`dtype="bfloat16"` →
  `# cite: framework default`; `channels=3` cap 16 → see field description).

#### `data/aug_presets.py` — 12-cell preset × intensity table

- Add a **per-file letter-legend** in the module docstring (Albumentations
  defaults; flip/rotate/photometric magnitude conventions; project-empirical;
  laterality-driven locked-off rationale already in `LOCKED_OFF`).
- Tag each of the 8 knobs per cell via legend letters. Expected dominant
  classes: `# cite: framework default` (Albumentations) for probabilities/ranges
  matching the library, `# cite: empirical` / `# tbd:` for domain-tuned
  magnitudes (medical/satellite/microscopy `rotate_arbitrary`, `stain_jitter`,
  `gauss_noise`, `blur`). `stain_jitter` ties to Ruifrok & Johnston 2001 /
  Tellez et al. 2018 (see `transforms.py`). Highest-priority file (0 cites).

#### `data/channel_semantics.py` — normalization stats

- `_IMAGENET_MEAN=(0.485,0.456,0.406)`, `_IMAGENET_STD=(0.229,0.224,0.225)`:
  `# cite:` → ImageNet-1k training-set statistics (torchvision provenance).
- `grayscale` profile `((0.449,),(0.226,))`: `# cite:` → torchvision grayscale-
  ImageNet convention (single-channel mean/std). `rgba` extra channel `0.5`:
  `# cite: degenerate-case` → neutral [0,1]→[-1,1] zero-centering for the alpha
  channel. `freeform` `normalize_default=None`: index row (requires explicit
  user stats; no default to cite).

#### `data/transforms.py` — processor stats, HED matrix, projection constants

- `KNOWN_PROCESSOR_STATS["facebook/sam3.1"]`: `# cite:` → ImageNet stats
  (mirrors the HF `Sam3ImageProcessor`; the inline comment already ratifies this
  via the 2026-05-21 audit). `_STATS_DIVERGENCE_ATOL=1e-3`:
  `# cite: empirical` (tolerance chosen to catch `[0.5,0.5,0.5]` drift).
- `_HED_FROM_RGB_MATRIX`: `# cite:` → Ruifrok & Johnston 2001, "Quantification of
  histochemical staining by color deconvolution" (Anal Quant Cytol Histol
  23(4):291–299); quote the published H/E/DAB RGB absorbance vectors.
- `_GAUSS_NOISE_MAX_VAR=0.05`, `_GAUSS_BLUR_MAX_SIGMA=3.0`:
  `# cite: empirical` / `# tbd:` — magnitude→Albumentations projection ceilings
  (project choices; the spec §8.1 they reference is internal).

#### `presets.py` — VRAM model constants

- `MODEL_PARAMS=5e9`: `# cite: <repo>:<path>` → SAM 3.1 checkpoint parameter
  count (the inline comment cites `scripts/_derive_preset_constants.py`).
- `_bytes_per_param_for_method` 2.0 / 0.5, `_optimizer_bytes` `*4`:
  `# cite: framework default` (bf16/fp16 = 2 B; NF4 = 0.5 B; AdamW fp32 m+v+
  master = 8 B/param → 4× the bf16 adapter). The `*4` is the AdamW-state cost
  the issue called `ADAMW_STATE_MULT`; it is an inline literal, tagged here.
- `LORA_LAYERS=96`, `D_IN=768`, `D_OUT=768`, `Q_OVERHEAD`, `WORKSPACE_BYTES`,
  `BASE_ACTIVATION_AT_1024`, `forward_only_factor=0.25`:
  `# cite: empirical` → calibration in #148 / #179 (capture run IDs / PR links).
- `_SAM3_PATCH=14`, `_SAM3_HEADS=16`: `# cite: <repo>:<path>` → SAM 3.1 vision
  backbone (`sam3/model_builder.py`, hiera-large), already noted inline.

#### `models/losses/presets.py` — firm-up only

- Convert every `# citation needed` cell (`focal_gamma` at lines 116/131/162/
  177/208/223; `tversky_alpha` at 164/179/225) and the docstring mention (line
  25) to a real `# cite:` (legend letter) or `# tbd: #<umbrella>`:
  - The `focal_gamma` escalation (2.5 / 3.0) above Lin et al.'s γ=2.0 has no
    direct paper value → `# tbd: #<umbrella>` unless an internal run justifies
    it (`# cite: empirical`).
  - `tversky_alpha=0.7/0.8` "for this exact value" → verify against Salehi et
    al. 2017 (α=0.7 best on MS lesions) where it matches legend (E); 0.8 has no
    paper value → `# tbd: #<umbrella>`.
- Add doc rows for the preset params (or reference the folded-in A–H legend).
  No re-tuning of any loss hyperparameter.

### 1.5 Verification depth (DEEP VERIFICATION)

"Deep verification" = nailing each value's *actual* provenance, **not**
inventing papers for project numbers. Per tag class:

- `# cite: <paper>` → captured quote + DOI/arXiv + exact Eq./Table/Fig.
- `# cite: framework default` → confirm it truly is the
  Albumentations/torch/HF default and link the upstream docs (pin version).
- `# cite: degenerate-case` → state the math identity (e.g. Tversky α=0.5 ≡
  Dice; (0.5) channel-mean ≡ [-1,1] mapping).
- `# cite: <repo>:<path>` → the reference-impl file/line (Meta SAM3 backbone /
  multiplex cap; DETR / Carion 2020 for matcher λ; HF processor for stats).
- No external source AND no internal run → `# tbd:` + the umbrella sub-issue.
  **Never fabricate.**

Known-provenance starting points to verify (not to assume): LoRA (Hu et al.
2021) for PEFT r/α/dropout; QLoRA (Dettmers et al. 2023) for NF4/double-quant;
DETR (Carion et al. 2020) for matcher cost weights; ImageNet mean/std; COCO IoU
sweep [0.5:0.05:0.95] (Lin et al. 2014); focal loss (Lin et al. 2017); Tversky
(Salehi et al. 2017); focal-Tversky (Abraham & Khan 2019); boundary loss
(Kervadec et al. 2019); AdamW (Loshchilov & Hutter 2019); cosine/SGDR
(Loshchilov & Hutter 2017); Ruifrok & Johnston 2001 + Tellez et al. 2018 (HED).

## Deliverable 2 — Reference Training Profile (the `epochs` default)

### 2.1 What ships

A documented **reference training profile** written into
`docs/defaults-provenance.md` specifying:

- assumed dataset size,
- `batch_size` / `grad_accum_steps` (the shipped `1` / `8`),
- `epochs` (the derived value),
- eval mode (`full` vs `lite`),
- the budget claim: **train + eval ≤ 30 min on a 16 GB free-Colab T4**.

The shipped `epochs` value in `src/custom_sam_peft/cli/templates/
config_full.yaml` — currently the `$epochs` placeholder filled by the `init`
flow — is set to (or confirmed at) the profile's value. Because `epochs` is a
*required* `TrainHyperparams` field (no schema default), the alignment work is:
(a) the `init` flow's default `$epochs` substitution and (b) the documented
profile. `configs/examples/*.yaml` are out of scope (Non-Goals); the GPU-test
docstring reference to "shipped 50-step budget (epochs=25)" in
`tests/gpu/test_real_train_qlora_resume.py` is updated only if the derived
value changes.

### 2.2 Derivation method (analytical — no live T4 run, no runtime model)

A real T4 run cannot be executed in this environment, and — verified — the repo
has **no runtime/throughput model** (only the memory model in `presets.py`). So
the epoch figure is **derived analytically** from what actually exists plus
literature:

1. **Memory feasibility** — `presets.py` (`decide_preset` / `_predicted_bytes`)
   confirms the reference batch/grad-accum fits a 16 GB T4 with the
   `forward_only_factor` eval margin. This bounds *whether* a step fits, not
   *how long* it takes.
2. **Existing step-budget evidence** — `tests/gpu/test_real_train_qlora_resume.py`
   documents a "shipped 50-step budget (epochs=25)" on a 2-image dataset
   (batch=1, grad_accum=2 in that test). This is the closest existing
   wall-clock-adjacent reference and anchors the steps↔epochs arithmetic.
3. **Convergence literature** — deep-verified PEFT/LoRA fine-tuning convergence
   behavior (Hu et al. 2021 and related), used to argue the epoch count sits in
   a reasonable convergence regime for the assumed dataset size, not merely a
   budget artifact.

The doc records the arithmetic: assumed dataset size → steps/epoch at the
shipped batch×grad-accum → a *stated, sourced per-step-time assumption* (since
no in-repo timing model exists, the assumption and its basis — e.g. the #179
calibration notes or a cited T4 throughput figure — are written explicitly) →
total epochs that fit the 30-min budget. If the derivation confirms the
existing `25`, the value is unchanged and simply *documented*; if it lands
elsewhere, the template/`init` default moves to the derived value and the
GPU-test docstring is updated to match.

### 2.3 Empirical confirmation = follow-up, not a claim

The doc states plainly that the budget figure is analytical and rests on an
explicit per-step-time assumption (no in-repo runtime model). An **empirical T4
confirmation** is recorded as a `# tbd: #<follow-up>` and filed as a follow-up
issue — never asserted as completed.

### 2.4 Integration-test alignment (RESOLVED interpretation)

**Resolved decision.** The owner comment "integration tests should reflect this
epoch" is interpreted as: the end-to-end integration test should **exercise the
real default hyperparameters** (lr / scheduler / optimizer / PEFT config), with
**epoch count reduced solely for CI runtime** (not run til-convergence). It is
explicitly *not* interpreted as "run the integration test for the full
reference epochs."

Concretely (real files, verified):

- `tests/integration/test_train_then_eval.py` (`_build_cfg`, line 24) and
  `tests/integration/test_train_end_to_end.py` build configs that set
  `PEFTConfig(method="lora")` and `TrainHyperparams(epochs=1, …)` and otherwise
  **rely on schema defaults** for `learning_rate`, `lr_schedule`, `optimizer`,
  `warmup_steps`, `max_grad_norm`, and `peft.r/alpha/dropout/scope`. This means
  the practical default path is *already* exercised implicitly. The implementer:
  (a) confirms these configs do not override any practical hyperparameter away
  from its schema default (so the green run genuinely tests the default path),
  and (b) adds a **code comment** stating epochs are truncated to 1 purely for
  CI speed while the rest is the real default path. If any test pins a
  hyperparameter to a non-default value, it is reverted to the default (or the
  override is justified in the comment). Tests stay green.
- These integration tests are the end-to-end "default path" guard; there is **no
  `tests/cpu/` directory and no `test_smoke.py`** in this tree (the issue/brief
  naming was stale). The fast-truncation comment requirement therefore applies
  to the `epochs=1` integration configs above; no separate CPU-smoke file needs
  editing.

## Verification & Acceptance

Concrete acceptance criteria:

1. **Inline coverage** — every trust-bearing default in
   `config/schema.py`, `config/_internal.py`, `data/aug_presets.py`,
   `data/channel_semantics.py`, `data/transforms.py`, `presets.py`, and the
   user-facing `cli/templates/config_full.yaml` carries a concise inline tag
   (or, for dense `aug_presets.py`, a legend-letter reference) **and** a
   corresponding `docs/defaults-provenance.md` row. Self-evident
   structural/string defaults are indexed but untagged (Non-Goals).
2. **Loss-preset firm-up** — `grep -c "citation needed"
   src/custom_sam_peft/models/losses/presets.py` returns **0**; each former
   `# citation needed` (9 cells + the docstring mention) is now a real `# cite:`
   or a `# tbd: #<issue>`.
3. **Central doc** — `docs/defaults-provenance.md` exists, opens with the
   verification-standard section, is grouped by file with the row schema in
   §1.2, folds in the loss-preset legend, and **passes the project markdownlint
   gate**: CI runs `npx --yes markdownlint-cli2 --config
   .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"` (ci.yml line 99).
   Run that exact command locally before commit.
4. **Epochs alignment** — the shipped `epochs` default (the `config_full.yaml`
   `$epochs` slot filled by `init`) equals the documented reference-profile
   value; the profile (dataset assumption, batch/grad-accum, epochs, eval mode,
   ≤30-min-T4 budget, explicit per-step-time assumption) is written in the doc;
   the `tests/gpu/test_real_train_qlora_resume.py` docstring matches if the
   value changed.
5. **Integration tests** — `tests/integration/test_train_then_eval.py` and
   `test_train_end_to_end.py` run the practical-default-hyperparam path and are
   **green**; they carry the CI-truncation comment.
6. **No re-tuning** — diff review confirms no numeric default changed except the
   sanctioned `epochs` alignment.

## Risks & Mitigations

- **Unverifiable values.** Some repo numbers have no published source and no
  recorded run (`focal_gamma` 2.5/3.0 escalation, `tversky_alpha=0.8`, several
  aug magnitudes, lr/warmup). *Mitigation:* downgrade to `# tbd: #<umbrella>`
  rather than fabricate. Acceptance criterion 2 is satisfied by conversion, not
  invention.
- **No in-repo runtime model.** The 30-min T4 budget cannot be computed from the
  code (memory-only model). *Mitigation:* the doc states an explicit, sourced
  per-step-time assumption and labels the budget analytical; the empirical
  confirmation is a `# tbd:` follow-up.
- **T4 figure is analytical, not empirical.** *Mitigation:* state explicitly;
  file the empirical confirmation as a follow-up issue.
- **Framework-default drift.** "Albumentations/torch/HF default" claims age out
  across library versions. *Mitigation:* link the upstream doc *and* pin the
  version/commit observed in the verifying quote.
- **Reference-impl ambiguity.** DETR matcher weights and SAM3 backbone constants
  may differ between forks. *Mitigation:* cite the specific repo path the value
  mirrors, not a generic "DETR uses these."
- **Markdownlint on wide tables.** Provenance tables are wide. *Mitigation:*
  confirm MD013 (line-length) handling in `.config/markdownlint-cli2.jsonc`
  before authoring; wrap or disable per-block as that config dictates, and run
  the gate locally (the orchestrator owns the commit + lint step).
- **Tag/doc divergence.** Inline tags and doc rows can fall out of sync.
  *Mitigation:* the §1.3 contract requires tags and rows be produced together
  per file; acceptance criterion 1 checks both sides.

## Follow-up Issues (out of scope, file via `gh issue create`)

1. **CI "no-uncited-default" enforcement hook** — a lint/CI check that flags any
   new trust-bearing default lacking an inline tag + doc row. Explicitly
   deferred by the issue's own out-of-scope list.
2. **Empirical T4 confirmation** — run the reference profile on a real
   free-Colab T4 and replace the analytical budget claim (and its per-step-time
   assumption) with a measured one; referenced by the `# tbd:` in the
   training-profile doc section.
3. **Umbrella `# tbd:` tracker** — one issue collecting every default that
   resolved to `# tbd:` (unsourced project numbers: the `focal_gamma`
   escalation, `tversky_alpha=0.8`, aug magnitudes, lr/warmup, etc.), so they
   can be sourced or re-tuned later. All `# tbd: #<issue>` tags point here.

## Phasing Note (for the planner)

The natural seam is the §1.3 contract: the **provenance audit** (Deliverable 1 —
the row schema + all inline tags + doc sections) is one feature block; the
**reference training profile + epochs alignment + integration-test comment**
(Deliverable 2) is a second, smaller block that consumes the `epochs` doc row
defined by the first. The planner should phase accordingly, with the row schema
and the `epochs` row as the explicit phase-boundary interface.
