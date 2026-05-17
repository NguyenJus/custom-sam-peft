# Colab GPU Integration Fix Design (spec/colab-gpu-integration-fix)

**Status:** ready for implementation
**Parent specs:**
- [`2026-05-16-model-loading-design.md`](2026-05-16-model-loading-design.md) §4, §11 (the `_Sam3ImageAdapter.forward` stub was deferred to implementation time)
- [`2026-05-17-peft-lora-design.md`](2026-05-17-peft-lora-design.md) §5.3 (`SCOPE_TARGETS` patterns were left as placeholders pinned at implementation time)
**Sibling spec:** [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md)
**Scope (this spec):** `src/esam3/models/sam3.py` (`_Sam3ImageAdapter.forward`), `src/esam3/peft_adapters/lora.py` (`SCOPE_TARGETS`), `tests/fixtures/tiny_sam3_lora_stub.py` (rename of dummy subtrees), `tests/unit/test_peft_lora.py` (rename of asserted substrings), `tests/integration/test_load_sam31_real.py` + `test_peft_lora_real.py` + `test_peft_qlora_real.py` (rename of asserted substrings only — no logic changes).

Branch: `worktree-fix+colab-bpe-gzip` (PR #13 already open). This spec adds two commits on top of `517ff6a fix(models): drop bpe_path override so sam3 uses its bundled gzipped vocab` — the gzip fix is verified passing on Colab T4 and must not be touched.

---

## 1. Purpose

After the gzip-vocab fix in `517ff6a`, the integration suite under `bash scripts/run_gpu_tests.sh` runs end-to-end on Colab T4 but only 1 of 9 tests passes. The remaining 8 failures have two independent pre-existing root causes that were left as `IMPLEMENTOR` placeholders in earlier specs:

- **Root cause A (1 failure)** — `_Sam3ImageAdapter.forward` is a stub that raises `NotImplementedError`.
- **Root cause B (7 failures)** — The regex patterns in `SCOPE_TARGETS` don't match any module names in the real loaded SAM 3.1, so `apply_lora` and `apply_qlora` raise `ValueError: no nn.Linear modules matched ...`.

This spec resolves both, completes the two `IMPLEMENTOR` placeholders, and aligns the unit fixtures + integration test assertions with the real SAM 3.1 module naming. After implementation, all 9 tests under `requires_compatible_gpu and requires_checkpoint` pass on Colab T4; the 200 unit tests stay green.

## 2. Constraints

### 2.1 Branch and commit policy

- Stay on branch `worktree-fix+colab-bpe-gzip`.
- Add NEW commits on top of `517ff6a`; do not amend or rebase that commit.
- No new dependencies (`peft`, `torch`, `sam3`, `scipy` etc. are already pinned).
- No emojis in source, comments, or commit messages.
- Logging: append-only to `logs/log.md` after each meaningful step using `[TIMESTAMP] [ROLE] action` (create the file at task 1; do not read it during execution).

### 2.2 Existing wrapper contract (Root cause A)

`src/esam3/models/sam3.py:27-74` defines `Sam3Wrapper`:

- `forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Any]` accepts `(B, 3, H, W)` images and a list of B prompts.
- `_validate_prompts` enforces:
  - `images.ndim == 4`,
  - `len(prompts) == B`,
  - every prompt is the same variant (`TextPrompts` XOR `BoxPrompts`),
  - for `TextPrompts`, `len(p.classes) == 1` per forward call (the trainer loops over the vocabulary externally).
- `Sam3Wrapper.forward` calls `self.model(images, prompts)`, where `self.model` is the `_Sam3ImageAdapter` instance. The adapter's `forward(images, prompts)` is responsible for translating into Meta's `Sam3Image` API.
- The wrapper returns the adapter's dict unchanged. Downstream `meta_to_canonical` (`src/esam3/models/matching.py:39-60`) reads keys `pred_logits`, `pred_boxes`, `pred_masks`, `presence_logit_dec` and asserts shapes:
  - `pred_logits: (B, Q, 1)` (squeezed to `obj_logits: (B, Q)`).
  - `pred_boxes:  (B, Q, 4)` in normalized cxcywh.
  - `pred_masks:  (B, Q, H, W)` with `H = W = 288`.
  - `presence_logit_dec: (B, 1)` (squeezed to `img_presence: (B,)`).

The implementer MUST preserve this dict contract.

### 2.3 SAM 3.1 (`sam3` package) API surface

Inspected against the installed package at
`/home/justin/projects/Efficient-SAM3-Finetuning/.venv/lib/python3.13/site-packages/sam3/`.

#### 2.3.1 `Sam3Image` public forward-ish methods

From `sam3/model/sam3_image.py`:

- `forward(self, input: BatchedDatapoint)` (line 555). Builds a "find" workflow over a `BatchedDatapoint` (one or more text queries against one or more images). Calls `self.backbone.forward_image(input.img_batch)`, `self.backbone.forward_text(input.find_text_batch, device=device)`, constructs a `geometric_prompt` from `find_inputs[0]`, then loops `forward_grounding(...)` for each interactive step. Returns a `SAM3Output` object, not a plain dict.
- `forward_grounding(self, backbone_out, find_input, find_target, geometric_prompt: Prompt, **kwargs)` (line 440). The core training-mode forward. Returns the dict shape `meta_to_canonical` expects (sets `pred_logits`, `pred_boxes`, `pred_boxes_xyxy`, `pred_masks`, `presence_logit_dec`, plus `aux_outputs` when training).
- `predict_inst(self, inference_state, **kwargs)` and `predict_inst_batch(...)` (lines 624+). Inference-only paths driven by the SAM2 instance-interactivity predictor; they consume an `inference_state` dict built outside the model and do not accept `(images, prompts)` directly. Require `enable_inst_interactivity=True` at build time, which our loader sets to `False` (`sam3.py:143`).
- `_get_dummy_prompt(self, num_prompts=1)` (line 547). Returns a `Prompt` with zero box embeddings — used by `Sam3Processor` when no geometric prompt is given. Public-enough to call.

There is no method that takes `(images, list[str])` directly. The closest reference implementation is `sam3.model.sam3_image_processor.Sam3Processor` (file `sam3/model/sam3_image_processor.py`), whose `set_image` + `set_text_prompt` chain produces exactly the dict shape we need.

#### 2.3.2 `Sam3Processor.set_image` + `set_text_prompt` recipe (the blueprint)

Annotated trace from `sam3_image_processor.py`:

```python
# set_image(image, state):
state["backbone_out"] = self.model.backbone.forward_image(image)        # image: (B, 3, R, R) bf16

# set_text_prompt(prompt, state):
text_outputs = self.model.backbone.forward_text([prompt], device=device)
state["backbone_out"].update(text_outputs)                              # adds language_features, language_mask
state["geometric_prompt"] = self.model._get_dummy_prompt()              # null geometry

# _forward_grounding(state):
outputs = self.model.forward_grounding(
    backbone_out=state["backbone_out"],
    find_input=self.find_stage,                                         # see below
    geometric_prompt=state["geometric_prompt"],
    find_target=None,                                                   # eval-mode path
)
# outputs has keys: pred_logits, pred_boxes, pred_boxes_xyxy, pred_masks, presence_logit_dec.
```

`self.find_stage` (lines 31-39 of `sam3_image_processor.py`):

```python
FindStage(
    img_ids=torch.tensor([0], device=device, dtype=torch.long),         # shape: (P,)
    text_ids=torch.tensor([0], device=device, dtype=torch.long),        # shape: (P,)
    input_boxes=None,
    input_boxes_mask=None,
    input_boxes_label=None,
    input_points=None,
    input_points_mask=None,
)
```

The dataclass `FindStage` is defined in `sam3/model/data_misc.py` (line 109). `Prompt` is in `sam3/model/geometry_encoders.py` (line 83). The processor's `find_stage` cardinality is one image, one text prompt — exactly our per-image-per-class call shape.

#### 2.3.3 Real module naming in a built `Sam3Image`

Hierarchy of the `Sam3Image` instance returned by `sam3.build_sam3_image_model(enable_inst_interactivity=False, enable_segmentation=True, ...)`:

- `backbone: SAM3VLBackbone` (`sam3/model/vl_combiner.py:19`)
  - `backbone.vision_backbone: Sam3DualViTDetNeck` (`sam3/model/necks.py:15`)
    - `backbone.vision_backbone.trunk: ViT` (`sam3/model/vitdet.py:743`)
      - `backbone.vision_backbone.trunk.patch_embed.proj` (Conv2d — not LoRA-target)
      - `backbone.vision_backbone.trunk.blocks.{i}: Block` (`vitdet.py:635`), for i = 0 .. N-1
        - `backbone.vision_backbone.trunk.blocks.{i}.attn: Attention` (`vitdet.py:386`)
          - `backbone.vision_backbone.trunk.blocks.{i}.attn.qkv: nn.Linear(dim, 3*dim)` (`vitdet.py:433`)
          - `backbone.vision_backbone.trunk.blocks.{i}.attn.proj: nn.Linear(dim, dim)` (`vitdet.py:434`)
        - `backbone.vision_backbone.trunk.blocks.{i}.mlp: Mlp`
          - `backbone.vision_backbone.trunk.blocks.{i}.mlp.fc1: nn.Linear` (`vitdet.py:61`)
          - `backbone.vision_backbone.trunk.blocks.{i}.mlp.fc2: nn.Linear` (`vitdet.py:67`)
  - `backbone.language_backbone: VETextEncoder` (text encoder; many `nn.Linear`s)
- `geometry_encoder: SequenceGeometryEncoder` (`sam3/model/geometry_encoders.py:470`)
- `transformer: TransformerWrapper` (`sam3/model/model_misc.py`)
  - `transformer.encoder: TransformerEncoderCrossAttention` (`sam3/model/decoder.py:616`) — note: `TransformerEncoderCrossAttention` is the encoder for SAM 3.1's two-stream architecture, despite its class name.
    - `transformer.encoder.layers.{i}: TransformerDecoderLayerv2` (`sam3/model/decoder.py:888`), one of two layer variants per `model_builder.py`.
      - `transformer.encoder.layers.{i}.self_attn: MultiheadAttentionWrapper` (`model_misc.py:453`)
        - `self_attn.out_proj: NonDynamicallyQuantizableLinear` (subclass of `nn.Linear`)
        - `self_attn.in_proj_weight`, `self_attn.q_proj_weight`, `self_attn.k_proj_weight`, `self_attn.v_proj_weight` are **bare Parameters**, not Linears — LoRA cannot target them.
      - `transformer.encoder.layers.{i}.cross_attn_image: RoPEAttention` (`sam3/sam/transformer.py:267`)
        - `cross_attn_image.q_proj: nn.Linear` (line 211)
        - `cross_attn_image.k_proj: nn.Linear` (line 212)
        - `cross_attn_image.v_proj: nn.Linear` (line 213)
        - `cross_attn_image.out_proj: nn.Linear` (line 214)
      - `transformer.encoder.layers.{i}.linear1: nn.Linear`, `transformer.encoder.layers.{i}.linear2: nn.Linear` (FFN)
  - `transformer.decoder: TransformerDecoder` (`sam3/model/decoder.py:192`)
    - `transformer.decoder.layers.{i}: TransformerDecoderLayer` (`sam3/model/decoder.py:33`)
      - `transformer.decoder.layers.{i}.self_attn: MultiheadAttentionWrapper`
        - only `out_proj: nn.Linear` is LoRA-targetable
      - `transformer.decoder.layers.{i}.cross_attn: MultiheadAttentionWrapper`
        - only `out_proj: nn.Linear` is LoRA-targetable
      - `transformer.decoder.layers.{i}.ca_text: MultiheadAttentionWrapper` (if `use_text_cross_attention`)
        - only `out_proj: nn.Linear` is LoRA-targetable
      - `transformer.decoder.layers.{i}.linear1`, `linear2`: nn.Linear (FFN)
    - `transformer.decoder.bbox_embed: MLP` (`sam3.model.model_misc.MLP`, several Linears)
    - `transformer.decoder.query_embed: nn.Embedding`
- `segmentation_head: UniversalSegmentationHead` (`maskformer_segmentation.py:234`)
  - mostly `Conv2d` and `MLP`s; no attention-style projections
- `dot_prod_scoring: DotProductScoring` (Linear-bearing head used to compute `pred_logits`)
- `class_embed: nn.Linear(hidden_dim, 1)` (only present when `use_dot_prod_scoring=False`; absent in our build)

**Key observations driving design decisions in §4:**

1. The strings `vision_encoder` and `mask_decoder` **do not appear anywhere** in the real `Sam3Image.named_modules()` tree. The current `SCOPE_TARGETS` patterns, written against those assumed names, match zero `nn.Linear` modules.
2. The structural analog of "vision encoder" is `backbone.vision_backbone.trunk.blocks.*.attn.*`. The structural analog of "(mask) decoder cross-attention" is `transformer.decoder.layers.*.{self_attn,cross_attn}.out_proj` plus `transformer.encoder.layers.*.cross_attn_image.{q,k,v,out}_proj`.
3. The `MultiheadAttentionWrapper` instances in `transformer.{encoder,decoder}` only expose `out_proj` as a Linear. The fused `in_proj_weight` is a bare Parameter and is unreachable by PEFT LoRA. This narrows the decoder LoRA surface to `out_proj` (+ FFN `linear1`/`linear2` if we choose to include them, which we do NOT — only attention).

### 2.4 Hardware constraint

- The dev box is GTX 1080 (compute capability 6.1). It hits `requires_compatible_gpu` and skips all 9 integration tests.
- Verification of the fix happens on Colab T4 (compute capability 7.5) via `notebooks/colab_gpu_tests.ipynb` → `bash scripts/run_gpu_tests.sh`.
- The implementer cannot run pytest on the integration tier locally. CPU-only unit tests must give high confidence; the Colab notebook is the final gate.

## 3. File layout

| File | Change |
| --- | --- |
| `src/esam3/models/sam3.py` | Replace `_Sam3ImageAdapter.forward` body (lines 108-124) with the recipe in §4. Keep the rest of the file untouched. |
| `src/esam3/peft_adapters/lora.py` | Replace `SCOPE_TARGETS` dict (lines 32-43) with the patterns in §5. Drop the `TODO(task-7)` block. No changes to `_resolve_targets`, `apply_lora`, `save_lora`, `load_lora`, `merge_lora`. |
| `tests/fixtures/tiny_sam3_lora_stub.py` | Rename the two dummy subtrees (`vision_encoder` -> `vision_trunk`, `mask_decoder` -> `transformer_decoder`) so the fixture mirrors the real-name vocabulary; the negative controls and adapter wrapping stay unchanged. See §6.2. |
| `tests/unit/test_peft_lora.py` | Update asserted substrings (`"vision_encoder"` -> `"vision_trunk"`, `"mask_decoder"` -> `"transformer_decoder"`) and the explicit `target_modules` override path used in `test_target_modules_overrides_scope` and `test_resolve_targets_*`. Logic unchanged. |
| `tests/integration/test_load_sam31_real.py` | No changes. The existing `test_load_sam31_forward_to_canonical` test is the contract Root cause A must satisfy. |
| `tests/integration/test_peft_lora_real.py` | Update asserted LoRA name substrings: `"vision_encoder"` -> `"vision_backbone"` and `"mask_decoder"` -> `"transformer.decoder"`. Same number of assertions. |
| `tests/integration/test_peft_qlora_real.py` | Same substring updates as `test_peft_lora_real.py`. |
| `logs/log.md` | Create (new); append entries per repo convention. |

No production code outside `sam3.py` and `lora.py` changes. No schema changes. No `pyproject.toml` changes. No example-config changes (the configs are still semantically valid since they don't pin `target_modules`).

## 4. Design decisions for `_Sam3ImageAdapter.forward`

### 4.1 Chosen sam3 entry point: `forward_grounding`

`_Sam3ImageAdapter.forward(images, prompts)` will:

1. Validate `prompts` (TextPrompts only; one class per call; same length as `images`).
2. Run `self.model.backbone.forward_image(images)` -> `backbone_out` dict (vision features).
3. Build `class_names = [p.classes[0] for p in prompts]`. **All entries in this list must be the same string** because the wrapper enforces one class per forward call AND because `forward_grounding` runs a single decoder pass per (image, text-prompt) pair. The adapter asserts this and raises `ValueError` if violated. The B images are jointly processed with the single class name.
4. Run `text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)` (single-element list, matching `Sam3Processor.set_text_prompt`).
5. Merge `text_outputs` into `backbone_out` via `backbone_out.update(text_outputs)`. This adds `language_features`, `language_mask`, `language_embeds`.
6. Build a `FindStage` whose `img_ids = [0, 1, ..., B-1]` (so every image in the batch is processed) and `text_ids = [0] * B` (every image points at the same single text prompt at index 0). All other fields (`input_boxes`, `input_boxes_mask`, `input_boxes_label`, `input_points`, `input_points_mask`) are `None`, matching `Sam3Processor.find_stage`. The `FindStage` is constructed via the dataclass constructor.
7. Build a dummy geometric prompt via `geometric_prompt = self.model._get_dummy_prompt(num_prompts=1)`. The processor uses `num_prompts=1`; we follow suit because there is exactly one text prompt in `find_text_batch`.
8. Call `outputs = self.model.forward_grounding(backbone_out=backbone_out, find_input=find_input, find_target=None, geometric_prompt=geometric_prompt)`.
9. Return `outputs` (Meta's native dict; `meta_to_canonical` consumes it unchanged).

### 4.2 Rationale — why `forward_grounding`, not `forward(BatchedDatapoint)` or `predict_inst`

| Option | Cost | Verdict |
| --- | --- | --- |
| **Drive `forward_grounding` directly** (chosen) | ~25-30 lines: replicates `Sam3Processor`'s pre-computed `find_stage` + `_get_dummy_prompt` pattern. Stays in training mode (gradients flow). Returns the exact dict `meta_to_canonical` expects. Zero glue between adapter and downstream loss code. | Selected. |
| Build a `BatchedDatapoint` and call `Sam3Image.forward(input)` | Requires constructing `BatchedFindTarget` and `BatchedInferenceMetadata` dataclasses with COCO-shaped placeholders even though we have none at adapter time; the dataclasses have many required fields (`num_boxes`, `boxes`, `boxes_padded`, `is_exhaustive`, `repeated_boxes`, `object_ids`, `object_ids_padded`, etc., per `data_misc.py:142`). `Sam3Image.forward` returns a `SAM3Output` object (not a dict), so the adapter would still have to unwrap a stage output. Strictly more code for no benefit. | Rejected. |
| Use `predict_inst` / `predict_inst_batch` | Requires `enable_inst_interactivity=True` at build time (our loader passes `False`, see `sam3.py:143`). Returns NumPy arrays for the SAM2 interactive predictor's per-class masks, not Meta's per-query dict. Wrong data shape and wrong model build. | Rejected. |
| Wrap `Sam3Processor` itself | `Sam3Processor.set_image` and `set_text_prompt` are `@torch.inference_mode()`, which disables gradients — incompatible with training. Also re-runs `backbone.forward_image` per call and packs results into a `state` dict we'd then have to unwrap. Doubles the work. | Rejected. |

`forward_grounding` is also the function whose docstring directly produces the keys (`pred_logits`, `pred_boxes`, `pred_boxes_xyxy`, `pred_masks`, `presence_logit_dec`) that `matching.py:meta_to_canonical` already reads. The choice keeps `meta_to_canonical` correct without changes.

### 4.3 Function body (illustrative — not the final source, but pins the structure)

```python
def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Tensor]:
    if not all(isinstance(p, TextPrompts) for p in prompts):
        raise ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")
    class_names = [p.classes[0] for p in prompts]
    if len(set(class_names)) > 1:
        raise ValueError(
            "All prompts in a batch must share the same class name "
            "(SAM 3.1 forward_grounding runs one text prompt per call); "
            f"got {class_names}"
        )

    device = images.device
    backbone_out = self.model.backbone.forward_image(images)
    text_outputs = self.model.backbone.forward_text([class_names[0]], device=device)
    backbone_out.update(text_outputs)

    b = images.shape[0]
    find_input = FindStage(
        img_ids=torch.arange(b, device=device, dtype=torch.long),
        text_ids=torch.zeros(b, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = self.model._get_dummy_prompt(num_prompts=1)
    outputs = self.model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        find_target=None,
        geometric_prompt=geometric_prompt,
    )
    return outputs
```

`FindStage` import is `from sam3.model.data_misc import FindStage` and goes at the top of `sam3.py`, alongside the existing `import sam3` and `import torch`. The function body stays at ~25 lines.

### 4.4 Open behavior questions, resolved here

- **What about LoRA wrapping?** After `apply_lora`, `wrapper.model.model` becomes a `PeftModel` wrapping the `Sam3Image`. PEFT's `PeftModel.__getattr__` proxies unknown attributes to the wrapped base, so `self.model.backbone`, `self.model.backbone.forward_image`, `self.model.backbone.forward_text`, `self.model._get_dummy_prompt`, and `self.model.forward_grounding` all resolve. LoRA-adapted Linears under `backbone.vision_backbone.trunk` and `transformer.decoder` see traffic naturally because they are still wired into the same module tree. No adapter changes needed when LoRA is applied.
- **Why `text_ids=[0]*B`?** All B images share the same class prompt (one class per forward call). `forward_grounding` interprets `text_ids` as indices into `language_features` (see `sam3_image.py:179-181`); a single-element `language_features` plus `text_ids=[0]*B` means every image attends to the same single text prompt. This mirrors how `Sam3Processor` handles `set_image_batch` followed by a single `set_text_prompt`.
- **Gradient flow.** `Sam3Processor._forward_grounding` is wrapped in `@torch.inference_mode()`. The adapter does NOT wrap in inference_mode — we call `forward_grounding` directly so autograd is preserved when `self.training` is True. The wrapper inherits training mode from `nn.Module`.
- **No tokenizer caching.** `forward_text` re-tokenizes per call. Profiling-driven optimization is deferred (model-loading spec §4.3 took the same call).
- **Output dict mutation.** `outputs` from `forward_grounding` may include extra keys (`aux_outputs`, `queries`, `presence_feats`, `pred_boxes_xyxy`, `pred_logits_o2m`, etc.) depending on `self.training`. The adapter returns the dict as-is; `meta_to_canonical` indexes the four canonical keys only and ignores the rest. This matches the wrapper docstring at `sam3.py:38`.

### 4.5 Error surface

| Condition | Behavior |
| --- | --- |
| `prompts` contains a non-`TextPrompts` variant | `ValueError("_Sam3ImageAdapter only supports TextPrompts in v0")` |
| `prompts` contains more than one distinct class name | `ValueError("All prompts in a batch must share the same class name ...")` |
| Any other failure (e.g. CUDA OOM, sam3 internal assert) | Propagated as-is — these are real SAM 3.1 errors and the wrapper does not mask them. |

### 4.6 What stays the same in `Sam3Wrapper`

- `Sam3Wrapper.forward` body (`sam3.py:48-51`) is unchanged. It still calls `self._validate_prompts(...)` then `self.model(images, prompts)`. The existing validations remain the first guard.
- `Sam3Wrapper._validate_prompts` (`sam3.py:53-74`) is unchanged.
- `load_sam31` (`sam3.py:127-162`) is unchanged.
- `_resolve_checkpoint_path` is unchanged.
- The `_Sam3ImageAdapter.__init__` (`sam3.py:104-106`) is unchanged.

## 5. Design decisions for `SCOPE_TARGETS`

### 5.1 New patterns

```python
SCOPE_TARGETS: dict[str, list[str]] = {
    # ViT vision trunk (fused qkv + output projection per block).
    "vision": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
    ],
    # Vision trunk attention + transformer decoder attention output projections.
    # MultiheadAttentionWrapper exposes only `out_proj` as nn.Linear; its
    # in_proj_weight is a bare Parameter and is unreachable by PEFT.
    "vision_decoder": [
        r"backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
        r"transformer\.decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text)\.out_proj$",
    ],
    # Every nn.Linear in the tree (including FFN, segmentation MLPs, text-encoder
    # projections). TODO(future): narrow once SAM 3.1 attention surface is
    # exhaustively profiled. Left as `.*` per the existing comment in lora.py.
    "all": [r".*"],
}
```

Each pattern is anchored at end-of-name (`$`) so it matches only the leaf Linear, not parent modules whose names happen to end the same way.

### 5.2 What each pattern matches (verified against sam3 source citations in §2.3.3)

| Scope | Matches | Approximate count for SAM 3.1 main build |
| --- | --- | --- |
| `vision` | `backbone.vision_backbone.trunk.blocks.{i}.attn.qkv` and `.attn.proj` for every ViT block i | 2 × N_blocks; for the released SAM 3.1 ViT-B trunk, N_blocks=12 -> 24 Linears |
| `vision_decoder` | All `vision` matches plus `transformer.decoder.layers.{i}.{self_attn,cross_attn,ca_text}.out_proj` for every decoder layer i | 24 (vision) + 3 × N_decoder_layers (typically 6 layers per `model_builder.py:184` -> 18 decoder Linears) ≈ 42 |
| `all` | Every `nn.Linear` in the loaded module tree (~hundreds, including text encoder, geometry encoder, FFN, MLP heads) | several hundred — left as the existing trade-off; the per-spec TODO in `lora.py` remains |

### 5.3 What each pattern intentionally excludes

- **MLP feedforward (`mlp.fc1`, `mlp.fc2`)** in `backbone.vision_backbone.trunk.blocks.*`: trainable-ratio budget. The spec sibling (`peft-lora-design.md` §5.4) caps the LoRA trainable ratio at <5% on the real model; including MLP doubles the count of adapted Linears per block and would breach the budget that `test_apply_lora_on_real_sam31_under_trainable_budget` enforces.
- **`MultiheadAttentionWrapper.q_proj_weight / k_proj_weight / v_proj_weight / in_proj_weight`**: these are bare `nn.Parameter` instances, not Linears. PEFT LoRA cannot inject `nn.Linear`-shaped adapters into them. (RoPE attention modules in `transformer.encoder.layers.*.cross_attn_image` DO expose `q_proj`/`k_proj`/`v_proj` as Linears, but those live in the encoder, not the decoder. The `"vision_decoder"` scope's name refers to the model-loading-spec convention of "adapt vision + decoder cross-attention"; the natural place for that on this build is `transformer.decoder.layers.*.{...}.out_proj`. Encoder-RoPE expansion is left out of `"vision_decoder"` because including it would also breach the 5% budget; it is reachable via `"all"`.)
- **`segmentation_head.*`**: mostly Conv2d + small MLPs; not the standard LoRA-target surface and not justified by ablation evidence.
- **`backbone.language_backbone.*`**: text-encoder adaptation is reserved for `"all"` (and a future text-encoder-specific scope). The default workload (novel taxonomy of English class words) does not justify it in `"vision_decoder"`.

### 5.4 Test fixtures alignment

The unit-test fixture `tests/fixtures/tiny_sam3_lora_stub.py` currently constructs dummy subtrees named `vision_encoder.*` and `mask_decoder.*` that match the OLD regex patterns. The fixture will be renamed to `vision_trunk.*` and `transformer_decoder.*` so the unit tests exercise patterns shaped the same way as the new SCOPE_TARGETS (i.e., regexes anchored on the leaf-Linear path, with an indexed block / layer segment). This keeps unit tests as a meaningful regression guard against future SCOPE_TARGETS rewrites.

The exact fixture rename:

| Old fixture name | New fixture name |
| --- | --- |
| `vision_encoder` (subtree) | `vision_trunk` |
| `vision_encoder.block0`, `.block1` | `vision_trunk.blocks.0`, `vision_trunk.blocks.1` (use `nn.ModuleList`) |
| `mask_decoder` (subtree) | `transformer_decoder` |
| `mask_decoder.layer0` | `transformer_decoder.layers.0` (`nn.ModuleList`) |
| `_DecoderAttn` with `q_proj/k_proj/v_proj/out_proj` | unchanged class internals, but only `out_proj` is targeted by the new fixture-scope regex below |
| `neg_control_a`, `neg_control_b` | unchanged |

Unit-test SCOPE_TARGETS still drives matching in the fixture because the fixture's attribute path now matches the real-model patterns shape-for-shape. The unit test file uses dedicated test-only patterns when needed; see §6.3.

### 5.5 Sanity guard

The existing `_resolve_targets` (`lora.py:46-67`) preserves the helpful "first 50 Linear names" error message on miss. No change.

## 6. Test plan

### 6.1 Unit tests, CPU only (no GPU, no checkpoint)

All existing 200 unit tests must remain green. The tests below either stay as-is or get their asserted substrings renamed:

| Test | Change | Why |
| --- | --- | --- |
| `tests/unit/test_peft_lora.py::test_apply_lora_default_scope_freezes_base` | none | Body uses generic `"lora_"` substring; unaffected by fixture rename. |
| `test_apply_lora_vision_scope_matches_only_vision` | rename `"vision_encoder"` -> `"vision_trunk"`; rename `"mask_decoder"` -> `"transformer_decoder"` in assertions | Fixture rename. |
| `test_apply_lora_vision_decoder_scope` | same substring renames | Fixture rename. |
| `test_apply_lora_all_scope_includes_negative_controls` | none | tests `"neg_control"` substring; unchanged. |
| `test_target_modules_overrides_scope` | rename `target_modules=["vision_encoder.block0.attn.qkv"]` -> `target_modules=["vision_trunk.blocks.0.attn.qkv"]`; update the `qkv_lora` substring filter accordingly | Fixture rename. |
| `test_apply_lora_no_match_raises` | rename `"vision_encoder" in msg` -> `"vision_trunk" in msg` | Fixture rename. |
| `test_apply_lora_idempotent_guard` | none | unaffected. |
| `test_apply_lora_trainable_ratio_under_default_scope` | none | The ratio threshold (<20% on tiny stub) is robust to the rename. |
| `test_apply_lora_preserves_forward_signature` | none | signature-only. |
| `test_apply_lora_sets_peft_model_handle` | none | unaffected. |
| `test_scope_targets_keys_match_lora_scope_literal` | none | LoraScope literals unchanged. |
| `test_save_load_lora_roundtrip` | none | Substring `"lora_"` only. |
| `test_load_lora_idempotent_guard` | none | unaffected. |
| `test_save_lora_without_apply_raises` | none | unaffected. |
| `test_merge_lora_unwraps_and_clears_handle` | rename two `vision_encoder.block0.attn.qkv` references to `vision_trunk.blocks.0.attn.qkv` | Fixture rename. |
| `test_merge_lora_without_apply_raises` | none | unaffected. |
| `test_apply_lora_registered_under_peft_lora` | none | unaffected. |
| `test_resolve_targets_supports_custom_linear_types` | rename `vision_encoder.block0.attn.qkv` -> `vision_trunk.blocks.0.attn.qkv` and ensure the inline `Base` class uses `vision_trunk` and `nn.ModuleList` so the new `"vision"` regex (`backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$`) does NOT match. Use the test-only Base with the matching real-model naming AND scope `"vision"`. | The inline `Base` class in this test must be modified to use the new real-model naming (`backbone.vision_backbone.trunk.blocks.0.attn.qkv/proj`) so the real production regex applies. |
| `test_resolve_targets_default_still_filters_to_nn_linear` | same restructure as above | Same reason. |

Additionally, add ONE new unit test in `tests/unit/test_peft_lora.py`:

- **`test_scope_targets_match_realistic_sam3_module_names`** (NEW): build a minimal `nn.Module` tree mirroring the real SAM 3.1 naming (`backbone.vision_backbone.trunk.blocks.{0,1}.attn.{qkv,proj}` + `transformer.decoder.layers.{0,1}.{self_attn,cross_attn,ca_text}.out_proj` + `transformer.decoder.layers.{0,1}.linear1` as a negative control). Run `_resolve_targets(stub, PEFTConfig(method="lora", scope=s))` for each `s in {"vision", "vision_decoder", "all"}`. Assert:
  - `"vision"` returns exactly the 4 `attn.{qkv,proj}` names (2 blocks × 2 projections).
  - `"vision_decoder"` returns the 4 vision names plus exactly 6 `out_proj` names (2 layers × 3 attention sub-modules).
  - `"vision_decoder"` does NOT include `linear1` (FFN), the negative control.
  - `"all"` returns the strict superset.
  This is a regression guard against future SCOPE_TARGETS drift.

### 6.2 Unit-fixture rewrite (`tests/fixtures/tiny_sam3_lora_stub.py`)

```python
class TinySam3LoraStub(nn.Module):
    """Fake SAM 3.1 inner base with REAL-MODEL attention naming.

    Subtree shapes intentionally mirror sam3/model/{vitdet.py, decoder.py}:
      vision_trunk.blocks.{i}.attn.{qkv,proj}
      transformer_decoder.layers.{i}.{self_attn,cross_attn,ca_text}.out_proj
      neg_control_{a,b}                                  (Linears outside any scope)
    """
```

The two test-only SCOPE_TARGETS override patterns used in `_resolve_targets_supports_custom_linear_types` and `_resolve_targets_default_still_filters_to_nn_linear` must be retargeted to use the new fixture path (`vision_trunk.blocks.0.attn.{qkv,proj}`). Concretely the test patches `cfg.target_modules` or `cfg.scope` and asserts against the renamed leaves.

Important detail: the fixture's subtrees use `vision_trunk` and `transformer_decoder` (no nested `backbone.vision_backbone` or `transformer.decoder` chain), so unit tests cannot exercise the EXACT real-model regex. That's intentional — the unit fixture verifies the PEFT pipeline (freeze/apply/save/load/merge) end-to-end on a small graph; the new `test_scope_targets_match_realistic_sam3_module_names` test verifies the EXACT real-model regex against a different inline tree that DOES use the real-name prefixes. The two tests together cover both concerns.

### 6.3 Integration tests (Colab T4, gated)

All three integration files (`test_load_sam31_real.py`, `test_peft_lora_real.py`, `test_peft_qlora_real.py`) are marked `requires_compatible_gpu and requires_checkpoint` and run via `bash scripts/run_gpu_tests.sh`.

`test_load_sam31_real.py`:

- `test_load_sam31_returns_wrapper`: already passing post-gzip-fix. No change.
- `test_load_sam31_forward_to_canonical`: must pass post-Root-cause-A fix. No source change needed (the test asserts shape contracts that Root cause A's chosen forward path produces).

`test_peft_lora_real.py`:

- `test_apply_lora_on_real_sam31_under_trainable_budget`: update the asserted LoRA param-name substrings:
  - `assert any("vision_encoder" in n for n in lora_names)` -> `assert any("vision_backbone" in n for n in lora_names)` (matches `backbone.vision_backbone.trunk.blocks.*.attn.{qkv,proj}.lora_*`).
  - `assert any("mask_decoder" in n for n in lora_names)` -> `assert any("transformer.decoder" in n for n in lora_names)` (matches `transformer.decoder.layers.*.{self_attn,cross_attn,ca_text}.out_proj.lora_*`).
- `test_save_load_roundtrip_on_real_sam31`: no change.
- `test_merge_lora_on_real_sam31`: no change.

`test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora`: identical two-substring update as `test_peft_lora_real.py`.

### 6.4 What is NOT tested at the unit tier

- Numerical correctness of `forward_grounding` outputs (covered by the Colab integration tests).
- Cross-platform CUDA kernels behind the bf16 path (covered on Colab).
- Token-pinning of `_get_dummy_prompt` and `FindStage` semantics (we rely on the `Sam3Processor` reference blueprint matching production usage).

### 6.5 Verification gates

Local (dev box, GTX 1080):

```bash
ruff check src/esam3/models/sam3.py src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
ruff format --check src/esam3/models/sam3.py src/esam3/peft_adapters/lora.py tests/fixtures/tiny_sam3_lora_stub.py tests/unit/test_peft_lora.py tests/integration/test_peft_lora_real.py tests/integration/test_peft_qlora_real.py
uv run pytest tests/unit -q     # must stay at the existing 200-passing baseline (or 201 with the new regex test)
```

Colab T4 (final gate; user-driven):

```bash
bash scripts/run_gpu_tests.sh   # all 9 tests must pass
```

The user runs `notebooks/colab_gpu_tests.ipynb` end-to-end and reports back. The implementer does not have a path to run integration locally.

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| `forward_grounding` requires keyword args we missed (e.g. `is_instance_prompt`, `**kwargs`). | Low — the `Sam3Processor._forward_grounding` reference uses exactly the keyword set in §4.3; we replicate it 1:1. | Mirror the processor verbatim; do not add extra kwargs. |
| `_get_dummy_prompt(num_prompts=1)` returns a tensor on the wrong device (the sam3 code does `device = self.device`). | Low — `Sam3Image.device` resolves from `next(self.parameters()).device` (`sam3_image.py:106-108`). The loader places the model on `cfg.device` before the adapter is constructed. | Implementer asserts in a CPU-only smoke check that the prompt's device matches the model's device. Surfaces in Colab integration test on miss. |
| `text_ids=[0]*B` shares the same text features across all B images — could this break gradient flow when B>1? | Low — Meta's own batched `Sam3Processor.set_image_batch` followed by a single `set_text_prompt` uses the same pattern (lines 76-125 of `sam3_image_processor.py`). The decoder's text-cross-attention is invariant to repeated text rows. | Covered by integration test. |
| `outputs` from `forward_grounding` is missing `presence_logit_dec` when `dec_presence_out is None`. | Plausible — `sam3_image.py:339-342` only writes `presence_logit_dec` when `dec_presence_out is not None`. | The SAM 3.1 main build uses `LinearPresenceHead` (per `model_builder.py`), so `dec_presence_out` is always present. If the integration test fails on this, defer to a follow-up issue and update `meta_to_canonical` to default-zero on absence. Out of scope here. |
| `SCOPE_TARGETS["vision_decoder"]` produces zero matches if Meta renames `transformer.decoder.layers` in a future release. | Low for the pinned SAM 3.1 release. | `_resolve_targets` already raises `ValueError` listing the first 50 Linear names. The error message is debuggable. |
| Trainable-ratio assertion (<5%) in `test_apply_lora_on_real_sam31_under_trainable_budget` fails because the new patterns adapt too many params. | Medium — depends on hidden_dim, n_blocks, n_decoder_layers. Rough math: 24 vision Linears + 18 decoder Linears across ViT-B equivalents adds < 1% trainable params at r=16. | If breached, narrow the `"vision_decoder"` scope to skip `ca_text.out_proj`. Cited in §5.3. |
| MultiheadAttentionWrapper's `out_proj` is `NonDynamicallyQuantizableLinear`, a subclass of `nn.Linear`. PEFT may or may not handle the subclass. | Low — PEFT checks `isinstance(m, nn.Linear)`; subclasses pass. | If PEFT 0.13 rejects, fall back to explicit `target_modules` listing in a follow-up. Out of scope. |
| Renaming fixture subtrees breaks tests we missed. | Medium — `vision_encoder` and `mask_decoder` substrings could appear in places besides the listed tests. | The implementer runs `rg -n "vision_encoder|mask_decoder" tests/` BEFORE renaming and includes any remaining matches in the diff. The error is loud — tests would fail on the next pytest run. |

## 8. Out of scope / deferred

| Item | Deferred to |
| --- | --- |
| QLoRA-side regex updates beyond simple substring renames in `test_peft_qlora_real.py`. The `qlora.py` module uses `_resolve_targets(linear_types=(bnb.nn.Linear4bit,))` and shares `SCOPE_TARGETS`; the new patterns work for both. | Already in scope; no follow-up needed. |
| Optimization of `forward_text` re-tokenization per call. | Future profiling pass. |
| Sweep of `"all"` scope to narrow to attention-only Linears (the existing `lora.py` TODO). | Out of scope; the TODO stays. |
| Box-prompt path through `_Sam3ImageAdapter` (`BoxPrompts`). | Future spec; this adapter v0 supports `TextPrompts` only. |
| Multi-class-per-batch support (lifting the one-class-per-forward-call constraint). | Future spec; the wrapper contract enforces this today. |

## 9. Acceptance criteria

A correct implementation of this spec satisfies:

1. `bash scripts/run_gpu_tests.sh` on Colab T4 passes all 9 tests under `requires_compatible_gpu and requires_checkpoint`.
2. `uv run pytest tests/unit -q` passes; count is ≥ 201 (200 baseline + 1 new regex regression test) with no regressions.
3. `ruff check` and `ruff format --check` pass on every touched file.
4. The branch `worktree-fix+colab-bpe-gzip` contains the gzip-fix commit `517ff6a` untouched plus 1-2 new commits (one per work item is acceptable; squashing into one is also acceptable). PR #13 is updated with the new commits.
5. No new dependencies in `pyproject.toml`.
6. No emojis anywhere in the diff.
7. `logs/log.md` contains an append-only trail of `[TIMESTAMP] [ROLE] action` entries for each work item.
