# Semantic segmentation as a task mode (`task: semantic`)

**Status:** Draft (2026-06-02)
**Tracking:** [#113](https://github.com/NguyenJus/custom-sam-peft/issues/113) — *feat: semantic segmentation task mode (fixed-class, head-free over SAM3 text grounding)*
**Release:** pre-1.0 minor bump (new top-level task axis + data formats + eval metrics → MINOR).
**Status of design:** locked spine (marginalization), one to-verify optimization (`semantic_seg` surfacing) resolved below.

This spec is written to be implemented cold. Every architectural claim was
verified against the actual code (`src/custom_sam_peft/`) and the released
SAM3 package (`.venv/lib/python3.12/site-packages/sam3/`) at commit `1f7cfaf`.
Every new or changed default hyperparameter carries a `# cite:` or `# tbd:` tag,
mirroring `src/custom_sam_peft/config/schema.py` and
`src/custom_sam_peft/models/losses/presets.py`.

**Builds on:**

- [`2026-05-23-domain-aware-loss-presets-design.md`](2026-05-23-domain-aware-loss-presets-design.md) (#112) — the `(preset, class_imbalance, overrides)` resolver pattern, the `models/losses/` package with `terms/` + `compose.py` + `presets.py`, the `PRESET_TABLE` shape, the `LOCKED_OFF` WARN convention, the `run_dir/loss_bundle.json` sidecar, and the `csp doctor --config` "Resolved losses" table. The semantic loss subsystem **mirrors** this structure with a parallel table; it does not redefine `Preset`/`ClassImbalance`.
- [`2026-05-22-domain-aware-augmentation-presets-design.md`](2026-05-22-domain-aware-augmentation-presets-design.md) (#75) — the `Preset` literal, sidecar shape, doctor-table pattern.
- The `#111` channel work (`data.channels` / `data.channel_semantics` / `CHANNEL_SEMANTICS`) — carries over unchanged; semantic targets are channel-count-agnostic (they describe the label map, not the image).

---

## 1. Goals

1. Add a top-level `task: Literal["instance", "semantic"] = "instance"` to `TrainConfig`. The instance path is the current behavior and **must be byte-for-byte unchanged** when `task` is omitted or `"instance"`.
2. Train a **fixed-class multi-class semantic segmenter** on a user's own dataset. Each pixel is assigned exactly one of the dataset's classes (single-label argmax) or `none/background`.
3. Stay **maximally faithful to SAM3 — ZERO invented heads.** The dense per-concept foreground map is produced by **mask-classification → semantic marginalization** over the outputs the existing grounding forward already produces. No new decoder, no new mask head added by us.
4. Deliver the class vocabulary through SAM3's **text-prompt** mechanism (the dataset's class names as concepts), reusing the entire `TextPrompts` + K≤16 multiplex + auto-chunk + OOM-ladder machinery already in the instance path.
5. Provide a **domain-aware, preset-driven semantic loss** mirroring the #112 instance system (same `Preset` + `ClassImbalance` axes), NOT a single fixed CE+Dice.
6. Provide a **semantic evaluator** computing mIoU, pixel accuracy, and per-class IoU (respecting `ignore_index`), replacing COCO mAP under `task: semantic`.
7. Ship **two data adapters** (`mask_png`, semantic HF), a `SemanticTarget` dataclass, collate support, and full-lifecycle CLI branching (`train`/`eval`/`predict`/`export`/`doctor`).
8. 100% CPU-testable schema/data/loss/eval-math layers via stubs; GPU tests only for the real forward (mirrors `tests/` conventions).

## 2. Non-goals (state explicitly)

- **Panoptic segmentation.** The `task` Literal is kept trivially extensible (`Literal["instance", "semantic"]`), but panoptic is out of scope.
- **Multi-label / overlapping-pixel semantic** (a pixel belonging to >1 class). v1 is strict single-label argmax.
- **3D / volumetric** semantic (#110).
- **Auto-converting instance ↔ semantic datasets.** A semantic config requires a semantic data source.
- **Open-vocab inference on concepts outside the training dataset as a guaranteed/tested capability.** Open-vocab is an *inherited SAM3 property* of the text-grounding forward, not a v1 train/eval target. Framing is classic fixed-class semantic segmentation implemented over SAM3 text grounding.
- **A user-supplied semantic loss callable / user-defined presets** (v1.1, same as #112).

## 3. Verified facts about SAM3 internals (read before implementing)

These were verified by reading `.venv/lib/python3.12/site-packages/sam3/` and
the repo forward path. They are the load-bearing premises of §6.

### 3.1 Our forward path

`Sam3Wrapper.forward(images, prompts)` (`src/custom_sam_peft/models/sam3.py:158`)
→ `_Sam3ImageAdapter.forward` (`:310`) → `model.forward_grounding(...)`
(`sam3/model/sam3_image.py:440`). The adapter builds a **B·K column** multiplex:
`img_ids = arange(B).repeat_interleave(K)`, `text_ids = arange(K).repeat(B)`
(image-major / class-minor). All B images in a batch share the same K-class list
in the same order (enforced by `Sam3Wrapper._validate_inputs`, `:198`).

### 3.2 Output dict shapes (one forward, N = B·K columns)

From `meta_to_canonical` (`src/custom_sam_peft/models/matching.py`) and the
sam3 decoder (`sam3_image.py:_update_scores_and_boxes`, `_run_segmentation_heads`):

| Key | Shape | Meaning |
|-----|-------|---------|
| `pred_logits` | `(N, Q, 1)` | per-query text-image similarity logit (objectness for this column's concept) |
| `pred_masks` | `(N, Q, H_m, W_m)` | per-query instance mask **logits**; `H_m = W_m = 288` at 1008-px input |
| `pred_boxes` | `(N, Q, 4)` | normalized cxcywh (unused by semantic) |
| `presence_logit_dec` | `(N, 1)` | image-level "does column N's concept appear" logit |

`N = B·K`. Column `n` corresponds to image `n // K` and concept `n % K`. `Q` is
the decoder's per-column query count (number of candidate masks per concept).

### 3.3 The `semantic_seg` head: present in the checkpoint, but single-channel

- `build_sam3_image_model` instantiates a `UniversalSegmentationHead`
  (`sam3/model_builder.py:233`/`:710`; class at `sam3/model/maskformer_segmentation.py:234`).
- Its `forward` (`:282`) returns `{"pred_masks", "semantic_seg", "presence_logit"}`, where
  `semantic_seg = self.semantic_seg_head(pixel_embed)` and
  `self.semantic_seg_head = nn.Conv2d(pixel_decoder.out_dim, 1, kernel_size=1)`
  (`:277`) — **ONE output channel**. So `semantic_seg` is a single concept-agnostic
  foreground map per multiplex column, NOT a per-class map.
- **Correction to the issue's framing:** `semantic_seg` is **NOT discarded**.
  In `sam3_image.py:_run_segmentation_heads` (`:412–422`), keys in
  `segmentation_head.instance_keys = ["pred_masks"]` (`maskformer_segmentation.py:92`)
  get the o2o-slice treatment; every **other** key (including `semantic_seg`) falls
  through the `else` branch to `out[k] = v` and **is propagated** to our output dict.
  So `out["semantic_seg"]` IS available to us — it was the issue's premise that was
  slightly off, not the head's existence.
- **Checkpoint contains trained weights.** Verified by loading
  `models/sam3.1/sam3.1_multiplex.pt`: keys
  `detector.segmentation_head.semantic_seg_head.{weight,bias}` exist with shapes
  `(1,256,1,1)` / `(1,)`, weight `mean ≈ -0.051, std ≈ 0.598` (non-trivial, trained —
  not zero/identity init).

**Design consequence (resolved here, not left open):** because `semantic_seg` is
only 1 channel (per column = per concept, exactly what we want a per-concept
foreground map to be), surfacing it is a **legitimate alternative** to
marginalization, and its weights ARE trained. BUT it is a *learned end-to-end*
foreground that was trained under SAM3's own (unknown to us) supervision, whereas
marginalization is derived purely from the query outputs we already validate and
postprocess. The **robust spine is marginalization** (§6.2); surfacing `semantic_seg`
is an **optional, config-gated alternative reduction** (§6.4) — both produce a
`(B, K, H_m, W_m)` per-concept logit stack, so they are interchangeable behind the
same interface. Marginalization is the default; `semantic_seg` surfacing is opt-in
and flagged for empirical comparison (Open Question OQ-1).

### 3.4 PEFT reach

`peft.scope` default is `vision_decoder_concept` (`schema.py:496`), which adapts
the vision encoder + grounding decoder + concept/text in_proj — the same modules
the semantic forward uses (it IS the same forward). No new scope is needed. The
marginalization adds zero trainable params. If `semantic_seg` surfacing is used
(§6.4), the `semantic_seg_head` Conv2d (257 params) may optionally be unfrozen and
trained directly (§9).

---

## 4. The task axis and config schema

All changes are in `src/custom_sam_peft/config/schema.py` unless noted.

### 4.1 `task` field

```python
Task = Literal["instance", "semantic"]  # new type alias near DataFormat (~line 90)

class TrainConfig(_Strict):
    run: RunConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    peft: PEFTConfig
    train: TrainHyperparams
    eval: EvalConfig = Field(default_factory=EvalConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    task: Task = "instance"  # cite: #113 — default preserves the instance path exactly
```

Place `task` last so existing positional/dict construction is unaffected. Add
`"Task"` to `__all__`.

### 4.2 `DataFormat` and the semantic data source

```python
DataFormat = Literal["coco", "hf", "mask_png"]  # add mask_png (~line 90)
```

`coco` stays instance-only. `hf` serves BOTH tasks (an HF dataset may expose
instance boxes/masks OR per-pixel label maps; the field map disambiguates, §5.4).
`mask_png` is **semantic-only** (paired image dir + label-map PNG dir).

New nested config for the semantic source:

```python
class SemanticDataConfig(_Strict):
    """Semantic-segmentation data parameters. Required when task == 'semantic'.

    Lives under DataConfig.semantic. None for instance datasets.
    """

    class_map: str = Field(
        min_length=1,
        description=(
            "Path to a JSON file mapping integer pixel value -> class name, e.g. "
            '{"0": "background", "1": "road", "2": "building"}. The set of NAMES '
            "(excluding any explicit background, see §4.5) is the prompted concept "
            "vocabulary AND the dataset class_names, in ascending-pixel-value order."
        ),
    )
    ignore_index: int = Field(
        default=255,  # cite: PASCAL VOC / Cityscapes void convention (255)
        description=(
            "Pixel value in the label map treated as void/unlabeled. Excluded from "
            "both loss and metrics. Not a class. Default 255 is the de-facto standard."
        ),
    )
    label_suffix: str = Field(
        default="_labelIds.png",  # tbd: #113 — Cityscapes-style; override per dataset
        description=(
            "Filename suffix that maps an image file to its label-map PNG (mask_png "
            "format only). image 'aachen_000000.png' -> label "
            "'aachen_000000{label_suffix}'. Set to '.png' for same-stem pairing."
        ),
    )
```

Add to `DataConfig` (`~line 383`):

```python
    semantic: SemanticDataConfig | None = None  # required when task == 'semantic'
```

For `mask_png`, `DataSplit.annotations` is reinterpreted as the **label-map PNG
directory** and `DataSplit.images` as the image directory (no JSON file). Document
this reuse in `DataSplit`'s docstring; do not add a new split type.

### 4.3 Cross-field validation (task ↔ data)

`task` lives on `TrainConfig`, but the data invariants must be checked where both
are visible. Add a `@model_validator(mode="after")` on **`TrainConfig`** (it sees
both `self.task` and `self.data`):

```python
@model_validator(mode="after")
def _check_task_data_compat(self) -> TrainConfig:
    if self.task == "semantic":
        if self.data.format == "coco":
            raise ValueError(
                "task: semantic does not support data.format: coco (instance JSON). "
                "Use data.format: mask_png or hf with a semantic field map."
            )
        if self.data.semantic is None:
            raise ValueError("task: semantic requires data.semantic (class_map, ignore_index).")
        # Instance-only knobs that are INERT under semantic — reject if set non-default
        # so users aren't misled (see §4.4 for the exact list).
        _reject_inert_instance_knobs(self)
    else:  # instance
        if self.data.semantic is not None:
            raise ValueError("data.semantic is only valid when task: semantic.")
        if self.data.format == "mask_png":
            raise ValueError("data.format: mask_png requires task: semantic.")
    return self
```

`_reject_inert_instance_knobs` is a module-level helper (see §4.4). Keep all
validators pure (no I/O).

### 4.4 Which instance knobs become inert under `task: semantic`

The semantic path drops the Hungarian matcher and the per-instance loss families.
The following are **inert** and must be rejected when set to a non-default value
under `task: semantic` (raise `ValueError` naming the knob and pointing at the
semantic equivalent):

| Inert knob | Why inert | Semantic equivalent |
|------------|-----------|---------------------|
| `train.loss.overrides.matcher_weights` | no matcher | (none) |
| `train.loss.overrides.box_family` / `w_box` | no boxes in semantic | (none) |
| `train.loss.overrides.obj_family` / `w_obj` | objectness folds into marginalization | (none) |
| `train.loss.overrides.presence_family` / `w_presence` | no per-image presence loss | (none) |
| `eval.iou_thresholds` | no mAP sweep | (mIoU has no threshold sweep) |
| `eval.mask_threshold` | argmax, not per-mask binarize | (none) |

**Implementation note:** rather than reject `train.loss` (which is the *instance*
`LossConfig`), the semantic path uses a **separate** `train.semantic_loss` subtree
(§7.2). So under `task: semantic`, `train.loss` is simply **ignored** (document it;
do not error on its presence since it has a default factory). Reject only the
fields a user would plausibly set *expecting* them to apply: `eval.iou_thresholds`
and `eval.mask_threshold` when explicitly overridden. The matcher/box/obj/presence
overrides live under `train.loss` and are already ignored — call this out in the
config docs rather than erroring, to keep the validator simple. **Decision:** the
validator errors ONLY on `data.format`/`data.semantic` mismatches (§4.3) and on
explicit non-default `eval.iou_thresholds`/`eval.mask_threshold` under semantic.
Everything else is documented-as-ignored. (Rationale: minimize validator blast
radius; the #112 `LossConfig` and `EvalConfig` defaults are harmless when unused.)

### 4.5 `none/background` semantics (pin exactly)

Two background-ish concepts must be reconciled:

1. **`ignore_index` (void/unlabeled).** Pixels with this label value are excluded
   from BOTH loss and metrics. Never predicted, never scored. Default 255.
2. **`none/background` argmax channel.** The model emits one foreground logit per
   prompted concept → `(B, K, H, W)`. We **prepend** a synthetic background channel
   so the argmax has a "matches no prompted concept" option → `(B, K+1, H, W)`,
   `argmax` over the `K+1` axis, channel 0 = background.

Reconciliation rules (config-driven, override allowed):

- **Background channel logit.** Default: a learned-free constant `bg_logit = 0.0`
  (`sigmoid(0) = 0.5`), i.e. a pixel is background unless some concept's
  foreground probability exceeds 0.5. # cite: degenerate-case logit boundary
  (mirrors `eval.mask_threshold = 0.0` convention, `schema.py:638`).
  Exposed as `train.semantic_loss.background_logit: float = 0.0` for override.
- **Explicit dataset background class.** If the dataset's `class_map` *contains* a
  class literally named one of `{"background", "bg", "none", "unlabeled"}`
  (case-insensitive), that class is treated as the background and is **NOT prompted
  as a concept** (it is not a SAM3 text concept). Its pixels map to the synthetic
  channel-0 instead. Document the recognized name set; allow
  `train.semantic_loss.background_class_name: str | None = None` to name a custom one.
  If no such class exists, the synthetic background channel is purely
  "no concept fired".
- **Interaction with `ignore_index`:** `ignore_index` always wins. A pixel that is
  `ignore_index` is dropped before argmax-vs-GT comparison regardless of background.

This yields the GT label encoding the loss/eval expect (§5.2): an `(H, W)` int64
tensor whose values are in `{0 = background, 1..K = concept dense id + 1,
ignore_index}`.

---

## 5. Data layer

All paths under `src/custom_sam_peft/data/`.

### 5.1 `SemanticTarget` and `Example` (in `data/base.py`)

```python
@dataclass(frozen=True)
class SemanticTarget:
    """Dense per-pixel class labels for one image (semantic task).

    `labels` holds class ids in {0..K} where 0 == background and 1..K == concept
    dense_id + 1 (the +1 makes room for the background channel). Pixels equal to
    `ignore_index` are void: excluded from loss and metrics.
    """

    labels: torch.Tensor      # (H, W) int64, values in {0..K} ∪ {ignore_index}
    ignore_index: int         # carried so collate/loss/eval need no extra plumbing
```

`Example` carries instances XOR a semantic target. Keep `instances` for the
instance path; add `semantic` (default `None`). Exactly one is populated per task:

```python
@dataclass(frozen=True)
class Example:
    image: torch.Tensor          # (C, H, W) normalized  (C from data.channels)
    image_id: str
    prompts: Prompts             # TextPrompts; for semantic, the K concept names
    instances: list[Instance] = field(default_factory=list)  # populated iff task == instance
    semantic: SemanticTarget | None = None                   # populated iff task == semantic
```

> **Interface contract:** `instances` becomes defaulted (was required). The
> instance path constructs `Example(image, image_id, prompts, instances=...)` with
> `semantic` left `None` — unchanged behavior. The `Dataset` protocol and
> `class_names` property are untouched. **Blast-radius note** (per memory
> "Required-field blast radius"): grep every `Example(` constructor (coco.py,
> hf.py, all tests, predict/eval stubs) and confirm none rely on `instances` being
> positional-required; the default keeps them valid, but a full suite run gates
> "done".

### 5.2 Concept ordering, dense ids, and GT encoding (single source of truth)

The prompted concept order **defines** the dense ids and therefore the `(K+1, H, W)`
logit-channel ↔ GT-label correspondence. Pin it once:

- `dataset.class_names` = concept names in **ascending class_map pixel-value order**,
  with any explicit background class removed (§4.5). `len(class_names) == K`.
- Concept `dense_id = i` (0-based) → prompted as `class_names[i]` → produces logit
  channel `i + 1` in the `(K+1, H, W)` stack (channel 0 = background).
- GT `SemanticTarget.labels[y, x]`:
  - `0` if the pixel's class_map value names the background class (or no class),
  - `i + 1` if the pixel's class_map value maps to `class_names[i]`,
  - `ignore_index` if the pixel value is the configured void value.

This identical convention is used by the data adapters (§5.3/§5.4), the loss
(§7.4), the evaluator (§8), and predict (§10.2). It is the **interface contract**
between data and everything downstream.

### 5.3 `mask_png` adapter (`data/mask_png.py`, new) — REQUIRED

```python
@register("dataset", "mask_png")
def build_mask_png(cfg: dict, *, model_name: str, pipeline: Literal["train","eval"]) -> Dataset
```

Mirrors `build_coco` (`data/coco.py:325`): selects the split sub-dict, builds
train/eval transforms via `build_train_transforms`/`build_eval_transforms`
(these already handle albumentations for masks — pass the label map as an
additional mask target so geometric augs stay aligned; see §5.5), and returns a
`MaskPngDataset`.

`MaskPngDataset`:
- `__init__(images_dir, labels_dir, *, class_map_path, ignore_index, label_suffix,
  transforms, text_prompt, channels)`.
- Loads `class_map` JSON → builds `class_names` (ascending pixel value, drop bg) +
  `value_to_label` map (pixel value → {0, i+1, ignore_index}) per §5.2.
- Enumerates `images_dir`; pairs each image to `labels_dir / (stem + label_suffix)`.
  Missing pair → `FileNotFoundError` listing the first few.
- `__getitem__`: read image (via `data/io.read_image`, `channels`), read label PNG
  (single-channel uint8/uint16, NO normalization — raw class indices), remap pixel
  values → GT labels per §5.2, run transforms (image + label as aligned mask),
  build `TextPrompts(classes=class_names)` (semantic ALWAYS prompts the full fixed
  vocabulary — text-prompt `mode` is forced to `all`; see §5.6), pack
  `Example(image, image_id, prompts, semantic=SemanticTarget(labels, ignore_index))`.
- `class_names` property → the K concept names.
- `image_class_labels` property (for stratified subset, mirrors coco.py): the set
  of present GT class ids per image (excluding background + ignore).

### 5.4 Semantic HF adapter (`data/semantic_hf.py`, new) — REQUIRED

A new builder registered under the SAME `hf` format but dispatched on `task`:
the `@register("dataset", "hf")` entry in `hf.py` (`:363`) gains a `task` branch.
**Decision:** keep one registry key `hf`; inside `build_hf`, if the caller's
`task == "semantic"` (thread `task` into the builder via the cfg dict, see §6.5
wiring) construct a `SemanticHFDataset` from `data/semantic_hf.py`; else the
existing instance `HFDataset`. This avoids a second `DataFormat` literal and keeps
the format axis (`coco`/`hf`/`mask_png`) orthogonal to the task axis.

`SemanticHFDataset` consumes HF datasets exposing a per-pixel label map (a
`label`/`annotation` image feature, e.g. `scene_parse_150`, ADE20K-style). Add to
`HFFieldMap` (`schema.py:303`):

```python
    label_map: str | None = None  # cite: #113 — HF feature holding the (H,W) label image
```

When `task == "semantic"`, `label_map` is required (validate in the semantic
builder, not the schema, to keep `HFFieldMap` task-agnostic). The class vocabulary
comes from the HF dataset's label feature names (`ClassLabel`/`names`), or from
`data.semantic.class_map` if provided as an override. Same `SemanticTarget`
materialization and §5.2 encoding as `mask_png`.

### 5.5 Transforms — label-map alignment

`build_train_transforms` / `build_eval_transforms` (`data/transforms.py`) already
pass an albumentations pipeline that transforms `masks`. For semantic, pass the
single `(H, W)` label map as ONE mask target. Two requirements:

1. Geometric augs (flip/rotate/resize) must use **nearest-neighbor** interpolation
   for the label map (never bilinear — it would invent fractional class ids).
   Albumentations applies nearest to mask targets by default; verify the resize
   step uses `cv2.INTER_NEAREST` for masks. # cite: standard semantic-seg practice.
2. The label map is resized to `SAM3_IMAGE_SIZE` (1008) like the image. The loss
   downsamples GT to logit resolution at compute time (§7.4); the evaluator
   upsamples logits to GT resolution (§8). Keep the stored target at image
   resolution so eval IoU is measured at full res.

No color/photometric aug touches the label map (it is not an image).

### 5.6 Text-prompt mode under semantic

Semantic ALWAYS prompts the full fixed class vocabulary every image (mode `all`),
because every pixel must be classifiable against every concept. Force this in the
semantic adapters (ignore `data.text_prompt.mode`); emit a one-time INFO if the
user set a non-`all` mode under `task: semantic`. K = number of concepts; when
K > MULTIPLEX_CAP (16) the existing multiplex auto-chunk (§6) handles it.

### 5.7 Collate (`data/collate.py`)

`collate_batch` adds a `"semantic"` key alongside `"instances"`:

```python
return {
    "images": images,
    "image_ids": [...],
    "prompts": [...],
    "instances": [list(ex.instances) for ex in examples],   # [] under semantic
    "semantic": [ex.semantic for ex in examples],            # [None]*B under instance
}
```

Image-shape consistency check is unchanged. The semantic targets are kept as a
ragged Python list (per-image `(H_i, W_i)` may differ if not resized; after §5.5
they are all `SAM3_IMAGE_SIZE`). No stacking — the loss handles per-image.

> **Phase A interface contract (config + data):** exposes `SemanticTarget`,
> `Example.semantic`, the `(K+1)`-channel ↔ GT-label convention (§5.2), the
> `mask_png`/`semantic_hf` builders (returning `Dataset` with `class_names` of
> length K), the collate `"semantic"` key, and the `task`/`data.semantic` schema.
> Consumes nothing downstream. Fully CPU-testable with synthetic label PNGs.

---

## 6. Model forward and marginalization

New module: `src/custom_sam_peft/models/semantic.py` (pure functions; the model
itself is unchanged — `load_sam31` is reused as-is).

### 6.1 Per-forward column layout (recap)

One `Sam3Wrapper.forward(images, [TextPrompts(class_names)]*B)` over a K-class
group yields the §3.2 dict with `N = B·K` columns, column `n` = (image `n//K`,
concept `n%K`). The semantic forward runs the SAME multiplex/auto-chunk loop the
trainer/evaluator already use (§6.5), one K-group at a time, and concatenates the
per-group per-concept maps along the concept axis.

### 6.2 Marginalization (PRIMARY, head-free)

For one column (image `b`, concept `k`) with `pred_logits[n] ∈ (Q,1)`,
`pred_masks[n] ∈ (Q, H_m, W_m)`, `presence_logit_dec[n] ∈ (1,)`:

```text
obj_q      = sigmoid(pred_logits[n, :, 0])              # (Q,)
mask_q     = sigmoid(pred_masks[n])                     # (Q, H_m, W_m)
presence   = sigmoid(presence_logit_dec[n, 0])          # scalar
fg_prob(b,k) = presence * Σ_q obj_q · mask_q            # (H_m, W_m)  in [0, +)
```

This is the standard mask-classification → semantic marginalization
(MaskFormer-style: `Σ_q class_prob_q · mask_prob_q`). The `presence` factor
gates concepts the model judges absent from the image (folds objectness/presence
in, replacing the instance path's separate obj/presence losses). `# cite:`
MaskFormer (Cheng et al. 2021, arXiv:2107.06278) §3.4 "semantic inference" =
`Σ_q softmax(class)_q · mask_q`; we use sigmoid (per-concept binary) not softmax
because SAM3's `pred_logits` are independent per-concept text-image similarities,
not a closed softmax over classes.

Because `fg_prob` is a probability-weighted sum it is in `[0, ∞)` but practically
≤ presence·(effective query count). The loss and argmax operate on a **per-concept
foreground logit** = `logit(clamp(fg_prob, eps, 1-eps))` so it composes with a
background logit channel. **Decision:** to avoid a `Σ_q` that can exceed 1, define
`fg_prob = presence * max_q (obj_q · mask_q)` as the v1 reduction (a soft-max over
queries is the natural per-pixel "best matching mask for this concept") rather than
a sum. # tbd: #113 — sum-vs-max reduction; default `max` (bounded in [0,1], clean
`logit`); expose `train.semantic_loss.query_reduce: Literal["max","sum"] = "max"`.
Document that `sum` matches the issue's literal formula and may help for
co-occurring instances of one class; `max` is the safer default for argmax.

Stack over K concepts → `(B, K, H_m, W_m)` foreground logits. Prepend background
(§4.5) → `(B, K+1, H_m, W_m)`. This is the semantic logit volume consumed by loss
(§7) and argmax (§6.3).

**Strictly head-free:** consumes only `pred_logits`, `pred_masks`,
`presence_logit_dec` — keys we already produce and validate.

### 6.3 Argmax → label map (inference / eval)

```text
sem_logits  = (B, K+1, H, W)   # H,W = logit res (288) or upsampled to GT res
label_map   = sem_logits.argmax(dim=1)   # (B, H, W) int64 in {0..K}
```

Channel 0 = background. For eval/predict, upsample `sem_logits` bilinearly to the
GT/original resolution BEFORE argmax (matches `postprocess._upsample_mask_logits`,
`eval/postprocess.py:33`).

### 6.4 OPTIONAL alternative reduction: surface `semantic_seg` (config-gated)

When `train.semantic_loss.source: Literal["marginalize","semantic_seg"] = "marginalize"`
is `"semantic_seg"`, the per-concept foreground logit is taken directly from the
column's `out["semantic_seg"]` (shape `(N, 1, H_s, W_s)`, §3.3) instead of the
§6.2 marginalization. Stack over K → `(B, K, H_s, W_s)`; same background-prepend +
argmax. The `semantic_seg_head` weights are trained (§3.3); optionally unfreeze them
(§9). This is opt-in and is the subject of OQ-1 (empirical comparison). The
marginalize path is the default and the only path required for v1 acceptance.

### 6.5 Where the forward loop lives — train vs eval vs predict

The B·K multiplex chunking + OOM-ladder loop is implemented THREE times today
(train `loop.py:train_step`, `eval/evaluator.py:_iter_predictions`,
`predict/runner.py`). The semantic forward **reuses each in place** by branching on
task, NOT by adding a fourth loop. The shared helper in `models/semantic.py`:

```python
def marginalize_group(
    outputs: dict[str, Tensor], b: int, k: int, *, query_reduce: str, source: str,
) -> Tensor:
    """(N=b*k columns) -> (b, k, H, W) per-concept foreground LOGITS for this group."""
```

Each call site calls `marginalize_group` on its existing per-group `outputs` and
concatenates groups along the concept axis. This keeps the auto-chunk / OOM-ladder
/ NaN-skip scaffolding centralized per call site.

> **Phase C interface contract (forward/marginalization + train branch):** exposes
> `marginalize_group(outputs, b, k) -> (b,k,H,W)` and the `(B, K+1, H, W)` semantic
> logit volume builder (background-prepend). Consumes Phase A's `class_names` order
> (§5.2) and the SAM3 output dict (§3.2). Produces the tensor the Phase B loss
> consumes. CPU-testable against `tiny_sam3_stub` outputs.

---

## 7. Loss — domain-aware semantic preset system

New subpackage `src/custom_sam_peft/models/losses/semantic/` (parallel to the
instance terms), OR new modules in the existing package. **Decision:** add a
parallel `SemanticLossConfig` schema subtree and a
`models/losses/semantic_presets.py` + `models/losses/semantic_compose.py`, reusing
the existing per-pixel term math from `terms/mask.py` where possible (Dice, Focal,
Tversky, Boundary are all per-pixel and class-agnostic). Rationale below.

### 7.1 Parallel subtree vs extend `LossConfig` — decision

**Recommend a parallel `train.semantic_loss: SemanticLossConfig` subtree**, NOT an
extension of `train.loss` (`LossConfig`). Justification:

- `LossConfig`'s axes are `mask_family`/`box_family`/`obj_family`/`presence_family`
  — **four instance axes that don't exist in semantic** (no box, no per-query obj,
  no per-image presence as a loss). Cramming a `semantic_family` into the same
  model would leave 3 of 4 axes permanently inert, contradicting the schema's
  strict-config ethos.
- The two are mutually exclusive by task. A separate subtree gated by `task` keeps
  each config minimal and each resolver focused (matches the §4.4 "inert knob"
  cleanliness goal and the project's "simplicity" priority).
- It reuses the #112 *machinery* (resolver shape, `PRESET_TABLE` keying, sidecar,
  doctor table) without overloading the #112 *data model*.

`train.loss` retains its default factory and is simply unused under `task:
semantic` (§4.4). `train.semantic_loss` has a default factory and is unused under
`task: instance`.

### 7.2 `SemanticLossConfig` schema (`config/schema.py`)

```python
SemMaskFamily = Literal["ce_dice", "focal_dice", "focal_tversky", "boundary", "ce", "dice"]

class SemanticLossOverrides(_Strict):
    """Per-knob overrides; None -> inherit from (preset, class_imbalance)."""
    sem_family:      SemMaskFamily | None = None
    w_ce:            PositiveFloat | None = None
    w_region:        PositiveFloat | None = None   # weight on the Dice/Tversky/Boundary term
    focal_gamma:     PositiveFloat | None = None
    focal_alpha:     float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_alpha:   float | None = Field(default=None, ge=0.0, le=1.0)
    tversky_gamma:   PositiveFloat | None = None
    boundary_weight: float | None = Field(default=None, ge=0.0, le=1.0)

class SemanticLossConfig(_Strict):
    preset:          Preset         = "natural"        # reuse #112 Preset verbatim
    class_imbalance: ClassImbalance = "balanced"       # reuse #112 axis verbatim
    overrides:       SemanticLossOverrides = Field(default_factory=SemanticLossOverrides)
    # --- argmax / background / reduction knobs (§4.5, §6.2) ---
    background_logit: float = 0.0          # cite: degenerate logit boundary (sigmoid(0)=0.5)
    background_class_name: str | None = None   # tbd: #113 — custom explicit-bg name
    query_reduce: Literal["max", "sum"] = "max"   # tbd: #113 — see §6.2
    source: Literal["marginalize", "semantic_seg"] = "marginalize"  # cite: §3.3 / OQ-1
```

Add `SemanticLossConfig`/`SemanticLossOverrides`/`SemMaskFamily` to `__all__`; add
`semantic_loss: SemanticLossConfig = Field(default_factory=SemanticLossConfig)` to
`TrainHyperparams` (`~line 621`, next to `loss`).

### 7.3 Semantic preset table (`models/losses/semantic_presets.py`)

Mirror `presets.py`: a `SEMANTIC_PRESET_TABLE: dict[(Preset, ClassImbalance), dict]`,
a `resolve(cfg: SemanticLossConfig) -> ResolvedSemanticLoss` frozen dataclass, a
`LOCKED_OFF` WARN map, a `dump_semantic_loss_bundle(cfg) -> dict` sidecar helper,
and `_SEM_TERM_CLASS_NAMES`. Pure-Python (no torch) so `csp doctor` imports it
without torch.

**Families** (per-pixel multi-class; base is multi-class CE over the `(K+1, H, W)`
logits, region term over per-class soft masks):

| `sem_family` | Definition | When |
|--------------|------------|------|
| `ce_dice` | `w_ce·CE + w_region·Dice` | natural default |
| `focal_dice` | `w_ce·FocalCE + w_region·Dice` | moderate imbalance |
| `focal_tversky` | `w_ce·FocalCE + w_region·FocalTversky` | severe imbalance / medical lesions |
| `boundary` | `w_ce·CE + w_region·(boundary_weight·Kervadec + (1-bw)·Dice)` | thin structures / satellite |
| `ce` | `CE` only | balanced, debugging |
| `dice` | `Dice` only | (rare) |

`CE` = multi-class cross-entropy with `ignore_index` (`F.cross_entropy(logits,
labels, ignore_index=...)`). `FocalCE` = multi-class focal CE (γ, α). The region
terms (Dice/Tversky/Boundary) are computed **per class** over the one-hot vs
softmax — see §7.4. They reuse the math in `terms/mask.py` (`_dice`,
`_tversky_index`, `_focal_bce_per_pixel` generalized, `_kervadec_boundary`) by
calling them per-class (the helpers operate on `(N, H, W)` and already mean-reduce).

**Per-domain defaults (every cell tagged):** keyed identically to #112. The
weights default to **CE/region = 0.2/0.8 from SAMed** for the `natural`/balanced
baseline; per-domain cells adjust. Legend (extends #112's):

- `(S)` SAMed (Zhang & Liu 2023, arXiv:2304.13785) §3.3 — `CE/Dice = 0.2/0.8`.
- `(C)` Lin et al. 2017 (focal) — γ=2.0, α=0.25.
- `(D)` Abraham & Khan 2019 (Focal-Tversky) — γ=0.75 best on ISIC.
- `(E)` Salehi et al. 2017 (Tversky) — β=0.7 (FN weight).
- `(H)` Kervadec et al. 2019 (boundary) — blend ~0.2.
- `(F)` degenerate identity (α=0.5 → Dice; γ=1.0 → Tversky).
- `(G)` alias-of-medical (microscopy copies medical).

```python
# Representative cells (planner fills all 12 + microscopy alias, mirroring presets.py).
("natural",   "balanced"): {"sem_family":"ce_dice",       "w_ce":0.2,"w_region":0.8, ...},  # cite: (S)
("natural",   "moderate"): {"sem_family":"focal_dice",    "w_ce":0.2,"w_region":0.8,"focal_gamma":2.0, ...},  # (S,C)
("natural",   "severe"):   {"sem_family":"focal_dice",    "w_ce":0.2,"w_region":0.8,"focal_gamma":3.0, ...},  # w (S); γ tbd:#191
("medical",   "balanced"): {"sem_family":"focal_dice",    "w_ce":0.2,"w_region":0.8,"focal_gamma":2.0, ...},  # (S,C)
("medical",   "moderate"): {"sem_family":"focal_tversky", "w_ce":0.2,"w_region":0.8,"tversky_alpha":0.7,"tversky_gamma":0.75, ...}, # (S,E,D)
("medical",   "severe"):   {"sem_family":"boundary",      "w_ce":0.2,"w_region":0.8,"boundary_weight":0.2, ...}, # (S,H)
("satellite", "balanced"): {"sem_family":"ce_dice",       "w_ce":0.2,"w_region":0.8, ...},  # (S)
("satellite", "moderate"): {"sem_family":"boundary",      "w_ce":0.2,"w_region":0.8,"boundary_weight":0.2, ...}, # (S,H) thin roads/edges
("satellite", "severe"):   {"sem_family":"focal_tversky", "w_ce":0.2,"w_region":0.8,"tversky_alpha":0.7,"tversky_gamma":0.75, ...}, # (S,E,D)
# microscopy.* = dict(medical.*)  # cite: (G)
```

Every `focal_gamma` escalation beyond 2.0 with no literature source → `# tbd: #191`
(matches presets.py convention). `tversky_alpha=0.8`-style unsourced values →
`# tbd: #191`. The `0.2/0.8` CE/region split is `# cite: (S)` for every cell unless
a domain-specific source justifies otherwise (none in v1 → keep `(S)`).

`LOCKED_OFF` (semantic): under `preset: medical`, overriding `sem_family` to `ce`
or `dice` (away from focal/tversky/boundary) WARNs (rare-positive underweighting);
under `preset: natural`, overriding to `focal_tversky`/`boundary` WARNs. Same
"override-wins, warning-is-the-contract" rule as #112.

### 7.4 `SemanticLoss` term + compose (`models/losses/semantic_compose.py`)

```python
class SemanticLoss(torch.nn.Module):
    """Multi-class semantic loss over (B, K+1, H, W) logits vs (B, H, W) int64 labels."""
    def forward(self, sem_logits: Tensor, targets: list[SemanticTarget]) -> dict[str, Tensor]:
        # returns {"ce": .., "region": .., "total": ..}
```

Contract:
- `sem_logits`: `(B, K+1, H_l, W_l)` (logit res, 288). `targets[b].labels`:
  `(H_g, W_g)` int64. **Downsample GT** to `(H_l, W_l)` with **nearest** (never
  bilinear) for the loss; or **upsample logits** to GT res — choose downsample-GT
  (cheaper, std practice). `# cite:` standard seg-loss practice.
- `ignore_index` plumbed through EVERY reduction: `F.cross_entropy(..., ignore_index=ii)`;
  the region terms mask out ignore pixels before computing per-class Dice/Tversky
  (build a `valid = labels != ii` mask; zero both pred and one-hot there).
- Region terms operate per class `c ∈ {0..K}`: `pred_c = softmax(sem_logits)[:, c]`,
  `tgt_c = (labels == c) & valid`, then call `terms/mask.py` helpers on
  `(B, H, W)` per class and mean over classes. Background channel (0) is included
  in CE always; include it in the region term too (it is a real class for argmax)
  unless `w_region` semantics say otherwise — **decision: include background in
  region term** for consistency with CE.
- Returns `{"ce", "region", "total"}` with `total = w_ce·ce + w_region·region`.

`build_semantic_loss(resolved: ResolvedSemanticLoss) -> SemanticLoss` mirrors
`build_loss_bundle`. A back-compat-style entry is NOT needed (semantic is new).

> **Phase B interface contract (loss):** exposes `SemanticLossConfig` schema,
> `resolve`/`SEMANTIC_PRESET_TABLE`/`dump_semantic_loss_bundle`, and
> `SemanticLoss.forward((B,K+1,H,W) logits, list[SemanticTarget]) -> {"ce","region","total"}`.
> Consumes Phase A's `SemanticTarget` + §5.2 encoding. Produces the loss-key set
> `{"ce","region","total"}` that Phase C's train branch logs. Fully CPU-testable
> (synthetic logits + label tensors; degenerate-identity tests like #112).

---

## 8. Eval — semantic metrics

New module `src/custom_sam_peft/eval/semantic_evaluator.py` (gated on task) +
extensions to `eval/metrics.py`.

### 8.1 `SemanticEvaluator`

```python
class SemanticEvaluator:
    def __init__(self, cfg: EvalConfig) -> None: ...
    def evaluate(self, model, dataset, *, return_per_example_iou: bool = False)
        -> MetricsReport | tuple[MetricsReport, list[float]]: ...
    def evaluate_and_save(self, model, dataset, output_dir: Path) -> MetricsReport: ...
```

Same public surface as `Evaluator` (so `run_eval` and the trainer's mid-run eval
can dispatch on task with no caller rewrite). Reuses the
`_iter_predictions`-style multiplex forward loop structure + OOM ladder from
`evaluator.py:_iter_predictions` (`:121`), but instead of
`queries_to_coco_results` it calls `marginalize_group` (§6.5), accumulates a
running confusion-matrix over `(K+1)` classes (background included), and skips
`ignore_index` pixels.

**Metric computation (streaming confusion matrix, `eval/metrics.py`):**

```python
@dataclass(frozen=True)
class SemanticMetrics:
    mean_iou: float
    pixel_accuracy: float
    per_class_iou: dict[str, float]   # class_name (incl "background") -> IoU

def compute_semantic_metrics(
    confusion: np.ndarray,    # (K+1, K+1) int64, rows=GT, cols=pred
    class_names: list[str],   # len K; index 0 reported as "background"
) -> SemanticMetrics:
    # IoU_c = TP_c / (TP_c + FP_c + FN_c); mIoU = mean over classes with GT support;
    # pixel_acc = trace / total. Classes with no GT pixels are omitted from mIoU
    # (mirrors COCO per-class "skip no-GT" at evaluator.py:255).
```

`ignore_index` pixels are never added to the confusion matrix (excluded from every
metric). The confusion matrix is built per image and summed (O(K²) memory, tiny).

### 8.2 `MetricsReport` — shared contract for both tasks

`MetricsReport` (`eval/metrics.py:17`) currently has `overall`, `per_class`,
`n_images`, `n_predictions`. **Keep it as the shared report** and populate it
task-appropriately so the persistence sites (`metrics.json` writes in
`evaluator.py:414`, `runner.py:212/236`) need no shape change:

- Semantic `overall = {"mIoU": .., "pixel_acc": ..}` (instead of
  `{"mAP", "mAP_50", "mAP_75"}`).
- Semantic `per_class = {class_name: {"IoU": ..}}` (instead of `{"AP": ..}`).
- `n_predictions` → repurpose as `n_pixels_scored` for semantic (document it).

This avoids a parallel report type and keeps `metrics.json` schema-stable per task.
Add a `task` field to the JSON written (so a reader knows which keys to expect):
`{"task": "semantic", "overall": {...}, "per_class": {...}, ...}`.

`per_example_iou` (used by viz ranking, `runner.py:210`): for semantic, return each
image's mean-IoU over present classes (the worst-image picker still works).

> **Phase D interface contract (eval):** exposes `SemanticEvaluator` (same public
> methods as `Evaluator`), `compute_semantic_metrics`, and the task-tagged
> `MetricsReport`/`metrics.json`. Consumes Phase C's `marginalize_group` + §5.2.
> CPU-testable: streaming-confusion math against synthetic label maps; one GPU
> test asserting the real forward + mIoU on a tiny mask_png fixture.

---

## 9. PEFT

No new scope. `peft.scope: vision_decoder_concept` (default, `schema.py:496`)
already adapts the vision encoder + grounding decoder + concept/text in_proj, which
IS the forward the semantic path uses (§3.4). Marginalization adds zero trainable
params.

If `train.semantic_loss.source == "semantic_seg"` (§6.4), the
`semantic_seg_head` (Conv2d, 257 params) may be unfrozen so the foreground head
adapts to the user's classes. **Decision:** v1 keeps it FROZEN even under
`source: semantic_seg` (the marginalize spine is the trainable path; surfacing is
an inference-time alternative for comparison). Unfreezing it is a follow-up tied to
OQ-1. Document this in `peft_adapters`'s scope notes; no code change needed for v1
beyond reading the pre-trained head.

---

## 10. CLI / full lifecycle

Each command branches on `cfg.task`. The instance branch is the current code path,
untouched.

### 10.1 `train` / `run` (`cli/train_cmd.py`, `cli/run_cmd.py`, `train/runner.py`, `train/trainer.py`, `train/loop.py`)

- `train/runner.py:_build_dataset_from_dict` (`:65`) already dispatches on
  `cfg.data.format` via `lookup("dataset", format)`. Thread `task` into the cfg
  dict (or pass as kwarg) so the `hf` builder picks instance vs semantic (§5.4).
  `mask_png` is unambiguous.
- **`train_step` (`loop.py:212`) gains a task branch.** Shared: device moves,
  `classes_in_batch` collection, the `while True` K-replay loop, `_chunked`
  grouping, the per-group `_autocast_ctx`, the OOM ladder
  (`_train_step_with_oom_ladder` / `oom_state.on_oom`), NaN-skip, grad-accum,
  `clip_grad_norm_`, scheduler step. **Branched:** under `task: semantic`, the
  per-group body builds per-image `TextPrompts(class_names)` (full vocab, no
  per-class target gather — there is no Hungarian matching), runs the forward,
  calls `marginalize_group` → `(B, K_g, H, W)`, and on the LAST group concatenates
  groups → `(B, K+1, H, W)` to call `SemanticLoss.forward(sem_logits,
  batch["semantic"])`. Because semantic loss needs ALL concepts together (argmax is
  over the full K+1), the per-group accumulation differs from instance: instance
  sums per-group losses; semantic must **collect per-group concept-logit slices and
  compute one loss over the assembled `(B, K+1, H, W)`**. Spell this out in the
  plan — it is the one place the loop topology genuinely differs (instance =
  per-group-independent loss+backward; semantic = assemble-then-loss). The OOM
  ladder still applies per forward; on K-rung shrink, the assembled stack is rebuilt
  from the already-computed groups (they are detached logit slices, cheap to hold).
  **Decision:** hold per-group `marginalize_group` outputs (detached from the graph
  is WRONG — we need gradients). Instead, accumulate the *graph-connected* per-group
  slices in a list and `torch.cat` before the single semantic backward. Memory: K+1
  channels at 288² × B is small (≪ the per-query mask tensors), so holding them
  across groups is acceptable. If a user's K is very large and this is tight, the
  OOM ladder's existing B/K shrink still recovers.
- **`StepResult` / `_ScalarWindow` / logging generalize to per-task loss keys.**
  Today both hardcode `{"mask","box","obj","presence","total"}` (`loop.py:191`,
  `:461`). Generalize: `StepResult.losses` is already `dict[str, float]`; change
  `StepResult.empty` and `_ScalarWindow` to accept the **task's loss-key set**
  (`{"ce","region","total"}` for semantic) rather than a hardcoded list. Pass the
  key set down from the trainer (it knows `cfg.task`). `_ScalarWindow.update`/`flush`
  iterate `self.sums` keys generically (`loss/<k>` for k in the task's keys). The
  instance key set is unchanged → instance logging is byte-identical. **Blast
  radius:** any test asserting exact `StepResult.empty()` keys (grep
  `StepResult.empty`, `loss/mask`) must be updated to the parametrized form.
- The loss-bundle sidecar: `train/trainer.py` writes `run_dir/loss_bundle.json`
  (#112). Under semantic, write `run_dir/semantic_loss_bundle.json` via
  `dump_semantic_loss_bundle(cfg.train.semantic_loss)` next to it. Add `task` to the
  persisted `run_dir/config.yaml` (round-trips already, per commit `1f7cfaf`).

### 10.2 `eval` (`cli/eval_cmd.py`, `eval/runner.py`)

`run_eval` (`eval/runner.py:60`) constructs `Evaluator(eval_cfg)` (`:173`). Branch:
`SemanticEvaluator` when `cfg.task == "semantic"`. The `evaluate`/`evaluate_and_save`
signatures match (§8.1), so the surrounding metrics.json/viz wiring is reused. The
`eval.batch_size == "auto"` resolution (`:156`) and OOM cap carry over. `--split`
handling unchanged.

### 10.3 `predict` (`cli/predict_cmd.py`, `predict/runner.py`, `predict/visualize.py`, `predict/writers.py`)

Predict already prompts arbitrary concepts (`--prompts`) and runs the multiplex
loop with the OOM ladder (`predict/runner.py:446`). For `task: semantic`:

- Under `--config` with `task: semantic`, default the prompts to the dataset's
  `class_names` if `--prompts` omitted (else use the user's concepts — open-vocab
  inference is *available* here, caveat-documented per §2).
- Replace the per-query COCO-style result entries with `marginalize_group` →
  `(N_imgs, K+1, H, W)` → argmax → an `(H, W)` int64 **label map** per image.
- **Output:** a **colorized label-map PNG** per image (deterministic palette keyed
  by concept index; background = black) + a raw single-channel index PNG. New writer
  in `predict/writers.py` (`write_semantic_label_map`) and a viz overlay in
  `predict/visualize.py`. The `predictions.json` per-image entry becomes
  `{"image_id", "label_map_path", "concepts": class_names}` (document the schema
  shift under semantic). `--score-threshold`/`--top-k`/`--save-masks` are
  instance-only → ignored (one-time INFO) under semantic.

### 10.4 `export` (`cli/export_cmd.py`)

Export bundles the adapter + config (no head to export — marginalization is
inference-time math). The exported config carries `task: semantic` and
`semantic_loss`. **Decision:** export is task-agnostic at the artifact level (it
ships the LoRA adapter + `config.yaml`); the only change is that the config now has
a `task` field which round-trips. If `source: semantic_seg` with an unfrozen head
were ever supported (post-v1), the head weights would join the export bundle —
note this as a future seam, no v1 code.

### 10.5 `csp doctor` (`cli/doctor_cmd.py`)

`doctor --config` renders resolved tables. Add a `task`-aware block:

- Always show `task` (a one-row "Task" line in the runtime/config summary).
- When `task: semantic`: render a **"Resolved semantic losses"** table (mirroring
  the "Resolved losses" table at `doctor_cmd.py:134`) from
  `resolve(cfg.train.semantic_loss)` + `dump_semantic_loss_bundle`, and a one-line
  "Head: marginalization (head-free)" or "Head: semantic_seg (surfaced)" per
  `source`. Suppress the instance "Resolved losses" table (it is inert).
- When `task: instance`: unchanged — instance "Resolved losses" table shown,
  semantic table suppressed.
- The `--json` block gains `"task"` and (semantic) a `"semantic_loss"` sub-key from
  `dump_semantic_loss_bundle`, replacing the inert `"loss"` block.

The doctor "Dataset" table (`:78`) gains `format` already; add a `task` row.

> **Phase E interface contract (CLI/predict/export/doctor):** consumes Phases A–D
> (datasets, marginalization, loss bundle, evaluator). Each command reads
> `cfg.task` and dispatches. No new cross-module exports; this is the wiring phase.

---

## 11. Testing strategy (mirror `tests/`)

The repo tests CPU-stubbed logic exhaustively and gates real-model behavior behind
GPU tests run via `scripts/run_gpu_tests.sh` (per memory: never bare `pytest
tests/`; the full real-model suite freezes a 16 GB box). Use `-o "addopts="` on
CPU-only dirs to bypass the global `--cov-fail-under=80` for inner-loop runs.

**CPU (no model, no GPU) — the bulk:**

- `schema`: `task` default, `task: semantic` ↔ `data.format` cross-validation
  (reject coco; require `data.semantic`), `SemanticLossConfig` strict-extra,
  inert-knob rejection (§4.4).
- `data`: `MaskPngDataset` against a synthetic temp tree (2 images + 2 label PNGs +
  class_map.json), `SemanticTarget` §5.2 encoding (background, ignore_index, dense
  ids), nearest-interp augmentation alignment, collate `"semantic"` key,
  `class_names` ordering. Semantic HF against a tiny in-memory HF `Dataset` with a
  label feature.
- `loss`: `SemanticLoss.forward` shapes, `ignore_index` exclusion (a fully-ignored
  image → zero/finite loss), degenerate identities (Tversky α=0.5 == Dice; FocalCE
  γ=0 == CE), `SEMANTIC_PRESET_TABLE` resolve + override-WARN + sidecar dump. Sync
  test: `_SEM_TERM_CLASS_NAMES` matches compose registry (mirrors #112's
  `test_term_class_names_match_compose_registry`).
- `marginalize`: `marginalize_group` against `tiny_sam3_stub`-shaped outputs
  (`(N, Q, 1)`/`(N, Q, H, W)`/`(N, 1)`) → assert `(B, K, H, W)` and the
  background-prepend → argmax label correctness. Both `query_reduce` modes and both
  `source` modes (semantic_seg path uses a stubbed `out["semantic_seg"]`).
- `eval`: `compute_semantic_metrics` on hand-built confusion matrices (mIoU,
  pixel-acc, per-class, ignore-exclusion, no-GT-class skip). Task-tagged
  `metrics.json` shape.
- `train`: `train_step` semantic branch with a CPU stub model returning fixed
  output dicts — assert it produces `{"ce","region","total"}`, backward runs, and
  the instance branch is unchanged (regression test on the existing instance
  `train_step`). `_ScalarWindow`/`StepResult` parametrized key set.
- `cli`: `doctor --config` semantic table, predict prompt-defaulting, eval
  dispatch (all with stubbed model/dataset).

**GPU (`tests/gpu/`, run via `scripts/run_gpu_tests.sh`, gated on the 5070 Ti):**

- One end-to-end semantic `train_step` on the real SAM3 wrapper over a 2-image,
  3-class synthetic mask_png fixture → assert finite loss + a backward + one
  optimizer step.
- One `SemanticEvaluator.evaluate` on the same fixture → assert a finite mIoU in
  `[0,1]` and a populated `per_class`.
- Keep GPU tests minimal (one process, real-model freeze risk per memory).

CPU tests must not import torch-heavy modules unnecessarily; keep
`semantic_presets.py` torch-free (like `presets.py`) so `doctor` and schema tests
stay light.

---

## 12. Backward compatibility

- `task` defaults to `"instance"`. A config without `task` validates and runs
  exactly as today. The §4.3 validator's instance branch only errors on
  semantic-only knobs being set (none are, by default).
- `Example.instances` becomes defaulted but stays first among the optionals; every
  existing `Example(image, image_id, prompts, instances=...)` call is unchanged.
  `Example.semantic` defaults `None`.
- `collate_batch` adds a `"semantic"` key (`[None]*B` under instance); existing
  consumers reading `"instances"` are unaffected.
- `StepResult`/`_ScalarWindow` key sets are parametrized but default to the instance
  keys; instance logging/JSON is byte-identical.
- `MetricsReport` is reused (not replaced); instance `metrics.json` keys are
  unchanged. The new `"task"` field in the JSON is additive.
- `DataFormat` gains `mask_png` (additive). `coco`/`hf` instance paths untouched.
- No back-compat shims for semantic (it is net-new). No pre-1.0 schema break beyond
  the additive `task` field.

**Full-suite gate (per memory "Required-field blast radius"):** because
`Example.instances` defaulting and the `StepResult` key parametrization touch
broadly-consumed types, run the FULL CPU suite (not just new tests) before "done",
and grep all `Example(` / `StepResult.empty(` / `loss/mask` / `collate_batch(`
call sites.

---

## 13. Phasing (for the planner)

Natural seams, each independently reviewable, with the interface contracts called
out inline above:

- **Phase A — config + data** (§4, §5). Exposes `task`/`data.semantic` schema,
  `SemanticTarget`, `Example.semantic`, `mask_png`/`semantic_hf` builders, collate,
  the §5.2 encoding. CPU-only. Contract at §5.7.
- **Phase B — semantic loss** (§7). Exposes `SemanticLossConfig`,
  `SEMANTIC_PRESET_TABLE`, `resolve`, `SemanticLoss`, sidecar. CPU-only. Depends on
  A's `SemanticTarget`. Contract at §7.4.
- **Phase C — forward/marginalization + train branch** (§6, §10.1). Exposes
  `marginalize_group`, the `(B,K+1,H,W)` builder, the `train_step` semantic branch,
  parametrized `StepResult`/`_ScalarWindow`. Depends on A (encoding) + B (loss).
  CPU-stub + one GPU smoke. Contract at §6.5.
- **Phase D — eval** (§8). Exposes `SemanticEvaluator`, `compute_semantic_metrics`,
  task-tagged report. Depends on A + C. CPU + one GPU. Contract at §8.2.
- **Phase E — CLI/predict/export/doctor** (§10). Wires `cfg.task` dispatch across
  commands. Depends on A–D. Mostly CPU (stubbed). Contract at §10.5.

A natural session boundary is A+B (data+loss, both pure/CPU), then C, then D+E.

---

## 14. Open questions / risks

- **OQ-1 (semantic_seg vs marginalize).** The released checkpoint has trained
  `semantic_seg_head` weights (§3.3), but it is single-channel and trained under
  SAM3's own supervision. Whether surfacing it (§6.4) beats marginalization on a
  user's fixed classes is **unverified** — it needs a real GPU comparison. v1 ships
  `marginalize` as default and `semantic_seg` as opt-in for this experiment. Risk:
  low (marginalize is the validated spine).
- **OQ-2 (`query_reduce` max vs sum).** §6.2 — `max` is the bounded, argmax-clean
  default; `sum` matches the issue's literal formula and may help dense
  co-occurring instances. Tagged `# tbd: #113`; both implemented, default `max`.
- **OQ-3 (per-domain weight escalations).** Several `focal_gamma`/`tversky_alpha`
  escalations have no literature source (inherited #191 `# tbd:` situation). They
  are tagged, not silent. The `0.2/0.8` CE/region baseline is solidly cited to
  SAMed `(S)`.
- **Risk — train-loop topology.** The semantic loss needs all K concepts assembled
  before one backward (unlike instance's per-group-independent backward), so the
  per-group slices must be held graph-connected across groups (§10.1). For large K
  with large B this is more memory than the instance path holds; the existing OOM
  ladder's B/K shrink is the safety net, but the assembled stack is the new peak —
  the planner should add a GPU test at K=16 to confirm headroom on the 5070 Ti.
- **Risk — augmentation nearest-interp.** A bilinear resize of the label map would
  silently corrupt class ids (§5.5). Add an explicit CPU test asserting
  nearest-interp on the label target.

---

## 15. Citations

| Tag | Source | Used for |
|-----|--------|----------|
| MaskFormer | Cheng et al. 2021, arXiv:2107.06278 §3.4 | semantic marginalization `Σ_q class_q · mask_q` (§6.2) |
| `(S)` SAMed | Zhang & Liu 2023, arXiv:2304.13785 §3.3 | CE/region = 0.2/0.8 baseline weights (§7.3) |
| `(C)` Focal | Lin et al. 2017, RetinaNet | γ=2.0, α=0.25 (§7.3) |
| `(D)` Focal-Tversky | Abraham & Khan 2019 | γ=0.75 (§7.3) |
| `(E)` Tversky | Salehi et al. 2017 | β/α=0.7 FN weight (§7.3) |
| `(H)` Boundary | Kervadec et al. 2019 | blend ~0.2 (§7.3) |
| `(F)` Degenerate identity | (math) | α=0.5→Dice, γ=1.0→Tversky (§7.3) |
| `(G)` Alias-of-medical | (project) | microscopy copies medical (§7.3) |
| VOC/Cityscapes void | PASCAL VOC, Cityscapes | `ignore_index = 255` (§4.2) |
| `MULTIPLEX_CAP` | `src/custom_sam_peft/models/sam3.py:116` | K ≤ 16 per forward |
| `SAM3_IMAGE_SIZE` | `src/custom_sam_peft/models/sam3.py:111` | 1008-px internal res; 288 mask res |
| Checkpoint inspection | `models/sam3.1/sam3.1_multiplex.pt` (loaded 2026-06-02) | `semantic_seg_head` (1,256,1,1) trained weights exist (§3.3) |

All `# tbd:` tags reference #191 (the standing "uncited escalation" issue, per
`presets.py`) or #113 (this feature's own design choices), matching the repo's
cite-or-tbd discipline.
