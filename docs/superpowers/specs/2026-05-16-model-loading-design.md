# Model-Loading Design (spec/model-loading)

**Status:** approved (brainstorming)
**Parent spec:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md) §11 step 2
**Sibling spec:** [`2026-05-16-data-loading-design.md`](2026-05-16-data-loading-design.md)
**Scope:** `src/esam3/models/sam3.py`, `src/esam3/models/losses.py`, **new** `src/esam3/models/matching.py`, additions to `src/esam3/config/schema.py`, fixes to `configs/examples/*.yaml`, test fixture updates.

---

## 1. Purpose

Provide a single loader entry point `load_sam31(cfg) -> Sam3Wrapper` plus the loss surface
(`total_loss(outputs, targets, cfg) -> dict`) the trainer needs to begin image-only finetuning
of SAM 3.1 on COCO with text or box prompts. The loader must:

- Source weights from a local snapshot of the official Meta checkpoint
  (`models/sam3.1/sam3.1_multiplex.pt`), produced by `huggingface-cli snapshot_download facebook/sam3.1`.
- Build the model via Meta's `sam3` package (not HuggingFace `Sam3Model`), preserving the
  Object Multiplex shared-memory pathway in the loaded weights even though v0 training
  uses only the image/detector forward path.
- Expose the model behind a thin wrapper that returns Meta's native output dict
  unchanged, with an adapter module (`matching.py`) that translates into a canonical
  shape the loss code can consume.
- Apply bf16 dtype + gradient checkpointing per `ModelConfig`.
- Leave parameter freezing and PEFT adapter attachment to the downstream
  `peft_adapters/` subsystem.

## 2. File layout

| File | Responsibility |
| --- | --- |
| `src/esam3/models/sam3.py` | `load_sam31(cfg)` + `Sam3Wrapper(nn.Module)`. |
| `src/esam3/models/matching.py` | `meta_to_canonical(outputs)` adapter, `CanonicalOutputs` dataclass, `HungarianMatcher`. |
| `src/esam3/models/losses.py` | `mask_loss`, `box_loss`, `objectness_loss`, `class_loss`, `total_loss`. |
| `src/esam3/config/schema.py` | `ModelConfig` extensions, new `LossConfig`, new `MatcherWeights`. |
| `configs/examples/coco_text_lora.yaml`, `coco_bbox_qlora.yaml` | Mismatch fixes (image size, normalization, loss block). |
| `tests/fixtures/tiny_sam3_stub.py` | Rewritten to mimic Meta output dict shape. |
| `tests/unit/test_*.py` | New tests for config / matcher / adapter / losses; updates to imports + stub-raises. |
| `tests/integration/test_load_sam31_real.py` | New, gated by checkpoint presence marker. |

The parent spec listed only `sam3.py` + `losses.py`. We add `matching.py` so each file keeps a
single purpose (load/forward vs. match/adapt vs. reduce). Total LOC budget is roughly the same.

## 3. `load_sam31(cfg: ModelConfig) -> Sam3Wrapper`

### 3.1 Algorithm

1. **Resolve source.** If `cfg.local_dir` is set and exists, use it; else attempt
   `cfg.name` as a HuggingFace repo id. If neither resolves, raise `FileNotFoundError`
   with a message naming `huggingface-cli download facebook/sam3.1 --local-dir models/sam3.1`.
2. **Build the runtime model via Meta's `sam3` package.** Import the build helper
   (exact entry-point name pinned at implementation time by inspecting the repo —
   reserved here as `sam3.build_sam3`). Construct a **full SAM 3.1** model with the
   tracker / Object Multiplex modules instantiated, dtype = `cfg.dtype` (bf16 default),
   device = `cfg.device or "cuda"`.
3. **Load the `.pt` state-dict.** `torch.load(checkpoint_path, map_location="cpu",
   weights_only=True)`. The file is a top-level dict with two roots: `{"detector",
   "tracker"}`. Flatten by re-prefixing each root into the single namespace the built
   module tree expects. The exact prefix mapping is pinned at implementation time by
   diffing `model.state_dict().keys()` against the raw checkpoint keys.
4. **`model.load_state_dict(state_dict, strict=False)`**, capture `(missing,
   unexpected)`. Both lists are logged via `esam3.logging`.
   - Raise `RuntimeError` if any missing key starts with the detector backbone prefix
     (catastrophic — the model would silently run with random backbone weights). Based
     on checkpoint inspection, the relevant prefix is rooted at the vision backbone
     (`detector.backbone.vision_backbone.*` in the raw `.pt`); the exact post-flatten
     prefix is pinned at implementation time.
   - Warn but do not raise for other missing/unexpected keys (e.g., renamed buffers).
5. **Gradient checkpointing.** If `cfg.gradient_checkpointing`, prefer Meta's native
   `set_grad_checkpointing(True)` if exposed on the ViT encoder; otherwise wrap encoder
   blocks with `torch.utils.checkpoint.checkpoint_wrapper`. The choice is resolved at
   implementation time by introspecting the built module.
6. **Wrap and return.** Return `Sam3Wrapper(model, image_size=1008, mask_size=288)`.

### 3.2 Signature & defaults

```python
def load_sam31(cfg: ModelConfig) -> Sam3Wrapper: ...
```

The loader sets `requires_grad=True` on all parameters by default. Freezing policy is
owned by the downstream `peft_adapters/` modules (see §10).

### 3.3 Error surface

| Condition | Behavior |
| --- | --- |
| `cfg.local_dir` missing and `cfg.name` not a valid Hub id | `FileNotFoundError` naming the expected path + download command. |
| Meta `sam3` package not importable | `ImportError` naming the pyproject dependency line and the suggested install command. |
| Missing state-dict key intersects backbone prefix | `RuntimeError` listing the offending keys. |
| Any non-fatal missing / unexpected keys | Logged at WARN; not raised. |

### 3.4 Pyproject changes

- Add `sam3 @ git+https://github.com/facebookresearch/sam3@<pinned-sha>` to
  `[project.dependencies]`. The SHA is pinned at implementation time after verifying
  the build helper signature.
- Bump `transformers>=5.0`. We do not import `Sam3Model` (Meta's package owns the
  model), but the data layer still uses `transformers` utilities (CLIP tokenizer, image
  processor) and the current `>=4.50` pin predates SAM 3 support.
- Add `scipy>=1.10` to `[project.dependencies]` for `linear_sum_assignment` in the
  matcher (scipy is transitively present via `pycocotools` but should be explicit).
- Keep `bitsandbytes` in the `qlora` extra (already present, unchanged here).

## 4. `Sam3Wrapper(nn.Module)`

### 4.1 Interface

```python
class Sam3Wrapper(nn.Module):
    def __init__(self, model: nn.Module, image_size: int = 1008, mask_size: int = 288) -> None: ...
    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Any]: ...
```

`Prompts` is the tagged union defined in `src/esam3/data/base.py`:
`TextPrompts(classes: list[str])` or `BoxPrompts(boxes: Tensor, class_ids: Tensor)`.

### 4.2 Forward behavior

> **Superseded for normalization (2026-05-21).** The `mean=std=[0.5, 0.5, 0.5]` claim
> on line 120 and the example-config edits on lines 283-285 are **wrong** for SAM3.1.
> See [`2026-05-21-yaml-config-defaults-audit-design.md`](2026-05-21-yaml-config-defaults-audit-design.md)
> for the corrected ground truth (ImageNet stats) and the three-step resolver.
> Everything else in this spec — image-size 1008, wrapper API, matcher, losses — stands.

- **`images`**: `(B, 3, 1008, 1008)` bf16, normalized with `mean=std=[0.5, 0.5, 0.5]`.
  Normalization is the data layer's responsibility; the wrapper does not re-normalize.
- **`prompts`**: list of length `B`. Every entry **must** be the same variant — mixing
  `TextPrompts` and `BoxPrompts` in one batch is not supported in v0. The collator in
  the data layer already enforces this; the wrapper re-asserts and raises
  `ValueError` with a helpful message if violated.
- **Multiplex cap**: `len(prompt.classes) <= 16` for `TextPrompts` and
  `prompt.boxes.shape[0] <= 16` for `BoxPrompts`. Already enforced upstream; wrapper
  re-asserts.
- **Routing**:
  - `TextPrompts` → tokenize `classes` with `CLIPTokenizer` (`max_length=32`), pass
    `input_ids` + `attention_mask` to Meta's text-prompt forward path.
  - `BoxPrompts` → reshape `boxes (N, 4)` into Meta's geometry-prompt format
    (cx,cy,w,h in normalized coordinates per the processor config), pass `class_ids`
    as label tensor.
- **Output**: Meta's native output dict is returned **unchanged**. The exact key
  names are pinned at implementation time by inspecting the output of a single small
  batch. Expected fields:
  - per-query class logits `(B, Q, C+1)` where `Q=200` and `C` is the number of
    classes inferred at runtime from the prompts;
  - per-query boxes `(B, Q, 4)` in normalized cx,cy,w,h;
  - per-query low-resolution mask logits `(B, Q, 288, 288)`;
  - per-query presence/objectness logit `(B, Q)`.

### 4.3 What the wrapper does not do

- No matching, no loss, no shape flattening — all in `matching.py` / `losses.py`.
- No tokenizer caching across calls (cheap; revisit if profiling shows otherwise).
- No `forward_multiplex(...)` convenience hook in v0 (YAGNI). The multiplex pathway
  is fully present on the underlying model accessible via `wrapper.model`; future
  video work calls Meta's forward directly.

## 5. `models/matching.py`

### 5.1 Canonical output adapter

```python
@dataclass
class CanonicalOutputs:
    class_logits: Tensor   # (B, Q, C+1) — last index is the "no-object" class
    pred_boxes:   Tensor   # (B, Q, 4) — normalized cx,cy,w,h
    pred_masks:   Tensor   # (B, Q, 288, 288) — mask logits at low resolution
    presence:     Tensor   # (B, Q) — objectness logit

def meta_to_canonical(outputs: dict) -> CanonicalOutputs: ...
```

`meta_to_canonical` is the **single point** in the codebase that knows Meta's native key
names. If Meta renames a field, only this function breaks. The number of classes `C` is
derived from the trailing dimension of Meta's class head output, not from config — keeping
the adapter robust to text-prompt-variable `C`.

### 5.2 Hungarian matcher

```python
class HungarianMatcher:
    def __init__(self, lambda_cls: float, lambda_l1: float,
                 lambda_giou: float, lambda_mask: float) -> None: ...

    @torch.no_grad()
    def __call__(self, outputs: CanonicalOutputs,
                 targets: list[list[Instance]]) -> list[tuple[Tensor, Tensor]]:
        """Per-batch (pred_idx, tgt_idx) LongTensor pairs."""
```

Cost terms, per batch element, between `Q=200` predictions and `N` targets:

| Term | Formula |
| --- | --- |
| `cost_cls` | `-softmax(class_logits)[:, target_class_id]` — classic DETR trick; focal weighting is applied only inside the loss, not inside the matcher. |
| `cost_l1` | `||pred_box - tgt_box||_1` in normalized cx,cy,w,h. |
| `cost_giou` | `-giou(pred_box_xyxy, tgt_box_xyxy)`. |
| `cost_mask` | `dice_cost(pred_mask_lowres, tgt_mask_resized_288)`. Target masks are bilinear-downsampled to 288×288 inside the matcher under `no_grad`. |

Total cost: `λ_cls·cost_cls + λ_l1·cost_l1 + λ_giou·cost_giou + λ_mask·cost_mask`.
Assignment is performed per batch element via `scipy.optimize.linear_sum_assignment`.
Empty `targets[i]` returns `(empty_tensor, empty_tensor)`.

Matching is non-differentiable; gradients flow only through post-match loss terms
applied to the matched subset.

## 6. `models/losses.py`

### 6.1 Per-component loss functions

```python
def mask_loss(pred_masks: Tensor, tgt_masks: Tensor) -> Tensor: ...
def box_loss(pred_boxes: Tensor, tgt_boxes: Tensor) -> Tensor: ...
def objectness_loss(presence: Tensor, matched_mask: Tensor) -> Tensor: ...
def class_loss(class_logits: Tensor, target_class_ids: Tensor) -> Tensor: ...
```

| Loss | Definition |
| --- | --- |
| `mask_loss` | `0.5 * dice(pred_hi, tgt_hi) + 0.5 * BCE(pred_hi, tgt_hi)` where `pred_hi` is bilinear-upsampled from 288×288 to the resolution at which ground-truth masks are stored in `Instance.mask` (i.e., the data layer's post-resize image resolution, typically 1008×1008). Reduced as mean over all matched instances across the batch (DETR convention). |
| `box_loss` | `λ_l1 * smoothL1(pred, tgt) + λ_giou * (1 - giou(pred, tgt))` in normalized cx,cy,w,h. Matched pairs only. |
| `objectness_loss` | Focal CE (γ=`focal_gamma`, α=`focal_alpha`) on the presence logit for **every** query. Positive label = matched query, negative = unmatched. Teaches the model to abstain on unmatched queries. |
| `class_loss` | Focal CE over `class_logits` for **every** query. Target = matched class for matched queries, target = "no-object" (extra class index `C`) for unmatched. |

### 6.2 `total_loss`

```python
def total_loss(outputs: dict, targets: list[list[Instance]],
               cfg: LossConfig) -> dict[str, Tensor]:
    canonical = meta_to_canonical(outputs)
    indices = HungarianMatcher(**cfg.matcher_weights.model_dump())(canonical, targets)
    matched = _gather(canonical, targets, indices)

    losses = {
        "mask": mask_loss(matched.pred_masks_hi, matched.tgt_masks_hi),
        "box":  box_loss(matched.pred_boxes,     matched.tgt_boxes),
        "obj":  objectness_loss(canonical.presence,
                                _matched_query_mask(canonical, indices)),
        "cls":  class_loss(canonical.class_logits,
                           _full_class_targets(canonical, targets, indices)),
    }
    losses["total"] = (cfg.w_mask * losses["mask"]
                       + cfg.w_box  * losses["box"]
                       + cfg.w_obj  * losses["obj"]
                       + cfg.w_cls  * losses["cls"])
    return losses
```

The dict return shape gives the trainer per-component logging without re-running any
computation. The `_gather`, `_matched_query_mask`, and `_full_class_targets` helpers are
module-private utilities.

## 7. Config schema additions (`src/esam3/config/schema.py`)

```python
class ModelConfig(_Strict):
    name: str = "facebook/sam3.1"
    local_dir: str | None = "models/sam3.1"          # NEW
    checkpoint_file: str = "sam3.1_multiplex.pt"      # NEW
    revision: str | None = None
    gradient_checkpointing: bool = True
    dtype: Dtype = "bfloat16"
    device: str | None = None                         # NEW

class MatcherWeights(_Strict):                        # NEW
    lambda_cls:  float = 2.0
    lambda_l1:   float = 5.0
    lambda_giou: float = 2.0
    lambda_mask: float = 5.0

class LossConfig(_Strict):                            # NEW
    w_mask: float = 1.0
    w_box:  float = 5.0
    w_obj:  float = 1.0
    w_cls:  float = 2.0
    matcher_weights: MatcherWeights = MatcherWeights()
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
```

`TrainConfig` gains `loss: LossConfig = LossConfig()`. All existing fields remain `_Strict`
and untouched. Default loss weights mirror DETR / Mask2Former public configs; tuning
happens in the training-loop spec, not here.

## 8. Example config fixes

`configs/examples/coco_text_lora.yaml` and `configs/examples/coco_bbox_qlora.yaml`:

- `data.image_size: 1024` → `1008` (SAM 3.1's native resolution).
- `data.normalize.mean: [0.485, 0.456, 0.406]` → `[0.5, 0.5, 0.5]`.
- `data.normalize.std:  [0.229, 0.224, 0.225]` → `[0.5, 0.5, 0.5]`.
- Add `model.local_dir: models/sam3.1` and `model.checkpoint_file: sam3.1_multiplex.pt`.
- Add a commented `train.loss:` block exposing `LossConfig` defaults so users see the
  knobs without having to read the schema.

These fixes are in scope for this spec because shipping the loader without them would
make the example configs produce silently wrong inputs on day 1.

## 9. Testing

### 9.1 Unit tests (CPU-only, no real checkpoint)

| Test file | Asserts |
| --- | --- |
| `tests/unit/test_model_config.py` | `ModelConfig` parses defaults; extra fields rejected (`_Strict`); `local_dir`/`checkpoint_file` resolve. |
| `tests/unit/test_loss_config.py` | `LossConfig` + `MatcherWeights` defaults and strict parsing. |
| `tests/unit/test_matching.py` | `HungarianMatcher` on synthetic `CanonicalOutputs` and hand-built targets returns the expected `(pred_idx, tgt_idx)` for trivial cases: (a) one query with zero cost matches one target, (b) empty targets returns empty index pairs, (c) `N > Q` targets get truncated cleanly. |
| `tests/unit/test_meta_to_canonical.py` | Updated `TinySam3Stub` returns Meta-shaped dict; `meta_to_canonical` converts to a `CanonicalOutputs` with the expected shapes and dtype. |
| `tests/unit/test_losses.py` | Each per-component loss on synthetic matched pairs returns a scalar `> 0`; `total_loss` returns a dict with all 5 keys and `total ≈ Σ w_i · loss_i` within floating-point tolerance. |
| `tests/unit/test_stubs_raise.py` | **Remove** the existing assertion that `load_sam31` raises `NotImplementedError`. |
| `tests/unit/test_imports.py` | Still passes — `matching.py` added to the import sweep. |

### 9.2 Integration test (gated)

`tests/integration/test_load_sam31_real.py`, marked `@pytest.mark.requires_checkpoint`,
skipped unless `models/sam3.1/sam3.1_multiplex.pt` exists. Asserts:

- `load_sam31(ModelConfig())` returns a `Sam3Wrapper`.
- `(missing, unexpected)` lists are small and contain no backbone keys.
- A `(1, 3, 1008, 1008)` zero-tensor input with `TextPrompts(["cat"])` produces a dict
  with the keys `meta_to_canonical` expects, and `meta_to_canonical(outputs)` returns
  a `CanonicalOutputs` with the documented shapes.

This test is excluded from default `pytest` runs; the CI lane is CPU-only without Meta
weights. CONTRIBUTING gets a one-paragraph note on how to opt in.

### 9.3 Test fixture update

`tests/fixtures/tiny_sam3_stub.py`: rewrite so `TinySam3Stub.forward(image, prompts)`
returns a dict matching Meta's output dict keys (resolved at implementation time), not
the previously assumed flat `{masks, boxes, objectness, class_logits}` shape. The stub
stays small (≤2 queries, ≤32×32 masks) to keep loss tests fast on CPU.

## 10. Frozen-base policy

The loader does **not** freeze any parameters. All `requires_grad` remain `True` after
`load_sam31` returns. Freezing is the responsibility of `peft_adapters/`:

- LoRA: freeze the whole base, then attach LoRA adapters via PEFT (which marks the
  adapters trainable).
- QLoRA: post-load module swap of target `nn.Linear` → `bnb.nn.Linear4bit`, then LoRA
  attach. Quantization is **not** done in `load_sam31`.
- Full finetune: leave everything trainable; `peft_adapters/none.py` is a no-op.

This boundary keeps the loader's responsibility tight and lets the PEFT spec own all
trainable-parameter policy.

## 11. Out of scope / deferred

| Item | Deferred to |
| --- | --- |
| `apply_lora` / `apply_qlora` implementation (post-load module swap strategy is locked in). | `spec/peft-adapters` |
| Optimizer / scheduler / `Trainer.fit()` loop. | `spec/training-loop` |
| Video / tracker forward path (`forward_multiplex`). Weights are loaded; pathway is unused in v0 train forward. | `spec/video-finetuning` |
| Distributed training (FSDP / DDP). v0 is single-GPU. | `spec/distributed` |
| Evaluation metrics (mAP, mask AP). | `spec/evaluation` |
| VRAM profiling of the multiplex-capable loaded model on 12–16 GB GPUs. | Followed up after first end-to-end train run; spec ships with bf16 + grad checkpointing defaults. |

## 12. Risks / open items pinned at implementation time

These do not block spec approval; each has a defined resolution path during
implementation:

- **Meta `sam3` build-helper entry-point name** and **pinned commit SHA**: resolved by
  inspecting `github.com/facebookresearch/sam3` at implementation time. The spec
  reserves `sam3.build_sam3` as a placeholder.
- **Meta output dict key names**: pinned by `print(outputs.keys())` against a tiny batch.
  `meta_to_canonical` is the single point of contact for the resolved names.
- **Exact state-dict prefix mapping** between the `.pt` file's `{detector, tracker}`
  roots and Meta's built module tree: resolved by diffing `model.state_dict().keys()`
  against the checkpoint keys.
- **Native grad-checkpointing hook**: prefer Meta's `set_grad_checkpointing(True)` if
  exposed on the ViT encoder; fall back to `torch.utils.checkpoint.checkpoint_wrapper`
  on encoder blocks if not.
- **Licensing**: `models/sam3.1/README.md` is `license: other` with extra gated terms.
  CONTRIBUTING gains a paragraph that contributors must accept Meta's gated terms on
  Hugging Face before `snapshot_download` succeeds.

## 13. Acceptance criteria

A correct implementation of this spec satisfies:

1. `load_sam31(ModelConfig())` returns a `Sam3Wrapper` on a machine where
   `models/sam3.1/sam3.1_multiplex.pt` and the Meta `sam3` package are both available;
   on a machine missing either, it raises with a message that points at the fix.
2. `wrapper(images, prompts)` returns Meta's native output dict; `meta_to_canonical`
   produces the documented `CanonicalOutputs` shape.
3. `total_loss(outputs, targets, LossConfig())` returns `{"total", "mask", "box",
   "obj", "cls"}` with `total ≈ w_mask·mask + w_box·box + w_obj·obj + w_cls·cls`
   within floating-point tolerance, on a synthetic batch from the rewritten
   `TinySam3Stub`.
4. All unit tests in §9.1 pass on CPU without the real checkpoint.
5. The integration test in §9.2 passes when run with the real checkpoint on a GPU.
6. The example configs in §8 load via `parse_train_config(...)` without errors and
   produce the corrected image-size / normalization values.
7. Linting and formatting (`ruff`, `ruff format`) pass on all touched files.
