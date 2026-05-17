# Training-Loop Notes: Meta `geometric_prompt` Layout (Task 0 Output)

Pinned 2026-05-17 against installed package `sam3==<as-installed>` at
`/home/justin/projects/Efficient-SAM3-Finetuning/.venv/lib/python3.13/site-packages/sam3/`.
All line numbers below refer to that installation.

## Status: COMPATIBLE

Meta's text-forward path exposes a box-hint slot. The training loop can feed
GT-box localization hints through `forward_grounding(..., geometric_prompt=Prompt(...))`
without spec renegotiation. Proceed to Task 1.

## Where the slot lives

- `SAM3Image.forward_grounding(self, backbone_out, find_input, find_target,
  geometric_prompt: Prompt, **kwargs)` — `sam3/model/sam3_image.py:440-446`.
- The prompt is consumed by `SAM3Image._encode_prompt(...)` at
  `sam3/model/sam3_image.py:167-210`, which passes
  `geo_prompt=geometric_prompt` directly to `self.geometry_encoder(...)` at
  `sam3/model/sam3_image.py:189-194`.
- The geometry encoder is `SequenceGeometryEncoder` in
  `sam3/model/geometry_encoders.py` (instantiated as `self.geometry_encoder`
  on `SAM3Image`); its `forward(geo_prompt: Prompt, img_feats, img_sizes,
  img_pos_embeds=None)` lives at `sam3/model/geometry_encoders.py:717`.
- The `Prompt` class itself is defined at
  `sam3/model/geometry_encoders.py:83-238`.

The same plumbing is used by Meta's own training entry point
(`SAM3Image.forward` builds the initial `Prompt` from `find_input.input_boxes`
at `sam3/model/sam3_image.py:576-580` and then calls `self.forward_grounding(...,
geometric_prompt=geometric_prompt.clone())` at line 592-597), so we are
using the exact slot Meta trains through, not a side path.

## The Four Facts

### 1. Tensor shape

`Prompt` is **not** a single tensor. It is a multi-field container; the
box-hint slot is two coupled fields (plus an optional labels field):

| Field            | Shape                       | Dtype          | Notes                                  |
| ---------------- | --------------------------- | -------------- | -------------------------------------- |
| `box_embeddings` | `(N_boxes, B, 4)`           | `torch.float`  | **Sequence-first, batch-second.**      |
| `box_mask`       | `(B, N_boxes)`              | `torch.bool`   | **Batch-first.** Key-padding convention. |
| `box_labels`     | `(N_boxes, B)` *(optional)* | `torch.long`   | Defaults to all-ones (positive) if `None`. |

Verbatim from the `Prompt` docstring (`geometry_encoders.py:86-99`):

> We expect the sequences in pytorch convention, that is sequence first,
> batch second. The dimensions are expected as follows:
> `box_embeddings shape: N_boxes x B x C_box`
> `box_mask shape: B x N_boxes. Can be None if nothing is masked out`
> ...
> `box_labels: long tensor of shape N_boxes x B`

The shape is enforced by asserts at `geometry_encoders.py:150-179`:
`box_embeddings.shape[:2] == [box_seq_len, bs]`,
`box_mask.shape == [bs, box_seq_len]`,
`box_labels.shape == [box_seq_len, bs]`.

`N_boxes` is the **padded** max-hints-per-image for the batch. `C_box = 4`
(asserted at `geometry_encoders.py:750`: `assert boxes.shape[-1] == 4`).

### 2. Coordinate space

**Normalized CxCyWH in `[0, 1]`.** Verbatim from the
`SequenceGeometryEncoder` docstring at `geometry_encoders.py:473`:

> It assumes boxes are passed in the "normalized CxCyWH" format, and points
> in normalized xy

Confirmed by the encoder body at `geometry_encoders.py:644-650`:

```python
# boxes are [Num_boxes, bs, 4], normalized in [0, 1]
# We need to denormalize, and convert to [x, y, x, y]
boxes_xyxy = box_cxcywh_to_xyxy(boxes)
scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
...
boxes_xyxy = boxes_xyxy * scale
```

And again at `geometry_encoders.py:746-756` (the `encode_boxes_as_points`
branch — Meta's own SAM3 checkpoint config takes this branch; see Note A):

```python
assert boxes.shape[-1] == 4
boxes_xyxy = box_cxcywh_to_xyxy(boxes)
top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)
```

The reference image size is the **per-image post-resize size** at which the
backbone runs (i.e., the same `H, W` the dataloader normalizes ground-truth
boxes against — see how Meta builds it in `SAM3Image.forward` at
`sam3_image.py:576-580`, where `find_input.input_boxes` is already the
normalized tensor coming out of the data pipeline; the encoder later
multiplies back by `(W, H)` of the feature map for ROI ops).

**Implementer must**: convert COCO-style absolute xyxy → normalized cxcywh
in `[0, 1]` before stuffing into `box_embeddings`. The helper
`sam3.model.box_ops.box_cxcywh_to_xyxy` exists as the inverse; an
`xyxy → cxcywh` flavor is also available in the same module.

### 3. Padding convention

**PyTorch key-padding convention: `True` means "this slot is padding (ignore)",
`False` means "this slot is a real hint".** Confirmed by:

- `concat_padded_sequences` comment at `geometry_encoders.py:28-29`:
  > Following pytorch's convention, tensors are sequence first, and the mask
  > are batch first, with 1s for padded values.
- `is_right_padded` at `geometry_encoders.py:17-20` and the default
  initialization in `Prompt._init_box` (`geometry_encoders.py:283-290`),
  which fills `box_mask = torch.zeros(bs, box_seq_len, dtype=torch.bool)`
  — i.e., all-zeros = nothing padded = all hints valid.
- The encoder computes valid counts as `(~mask).sum(dim=-1)` at
  `geometry_encoders.py:51-52`.

**Padding must be right-padded** (enforced by `torch._assert_async(is_right_padded(mask))`
at `geometry_encoders.py:48-49`).

#### Per-image hint-count patterns

To express "image *i* has 0 hints, image *j* has 3 hints" with `N_boxes = max=3`:

| Image | `box_embeddings[:, i, :]`           | `box_mask[i, :]`         |
| ----- | ----------------------------------- | ------------------------ |
| 0 hints | any (e.g., zeros)                 | `[True, True, True]`     |
| 1 hint  | `[hint, *, *]`                    | `[False, True, True]`    |
| 3 hints | `[h1, h2, h3]`                    | `[False, False, False]`  |

The actual values in padded slots are unused, but they must be finite (the
encoder still does linear projections / ROI ops on the full tensor before the
attention mask filters them downstream). Using zeros is safe and matches
Meta's own `_get_dummy_prompt` pattern (`sam3_image.py:547-553`).

### 4. None-sentinel / empty-batch handling

`forward_grounding` does **not** accept `geometric_prompt=None`. It expects a
`Prompt` instance. There are two "no hints" idioms:

a. **Whole-batch zero hints** (preferred for the curriculum's `p_box=0` tail):
   Use Meta's own `_get_dummy_prompt` shape — a zero-length sequence dim:
   ```python
   # sam3_image.py:547-553
   Prompt(
       box_embeddings=torch.zeros(0, B, 4, device=device),
       box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
   )
   ```
   This is well-defined: the encoder sees `N_boxes = 0`, asserts pass, and
   the geometry-token contribution to the prompt sequence is empty.

b. **Mixed batch (some images have hints, some don't)**: build a full
   `(N_max, B, 4)` / `(B, N_max)` pair and set the entire `box_mask[i, :]`
   row to `True` for any zero-hint image (pattern from §3 above).

Note: technically you can also construct `Prompt()` with all-None inputs
(the "Check for null prompt" branch at `geometry_encoders.py:114-130`), but
this produces a `Prompt` whose `.box_embeddings is None`, which the
`SequenceGeometryEncoder.forward` happy-path does not handle uniformly
(it routes through `_init_box` defaults only when at least one of box /
point / mask is provided). **Do not** rely on the all-None branch from the
training loop; always supply a (possibly zero-length) `box_embeddings`.

## Note A: encoder branch selection

`SequenceGeometryEncoder` has two box-encoding modes selected by
`self.encode_boxes_as_points` (constructor flag, set from checkpoint config).
Both consume the **same** `Prompt.box_embeddings` / `box_mask` layout
documented above — the difference is internal: one ROI-pools image features
inside each box, the other splits each box into TL/BR corner points and
runs them through the point encoder. **The builder we ship in Task 4 does
not need to know which branch is active**; the contract is the same.

## Note B: where the layout lands in Task 4

`src/esam3/models/sam3.py` will gain
`Sam3Wrapper._build_geometric_prompt(box_hints_per_image: list[Tensor | None],
device: torch.device, image_size: tuple[int, int]) -> Prompt`. It must:

1. Pad to `N_max = max(len(h) for h in box_hints_per_image)` (or 0 if all
   empty → use the §4(a) dummy form).
2. Convert each per-image absolute xyxy → normalized cxcywh in `[0, 1]`
   (divide by `(W, H, W, H)`, then `xyxy_to_cxcywh`).
3. Stack to `(N_max, B, 4)`, transpose-or-permute as needed so the seq dim
   is first.
4. Build `box_mask` of shape `(B, N_max)` with `True` for padded slots,
   right-padded.
5. Construct `Prompt(box_embeddings=..., box_mask=..., box_labels=None)`
   (labels default to all-positive, which is the correct semantics for a
   localization hint).

The four facts above are the *complete* contract — no further reading of
Meta's source is required to implement step 4.

## Sources (verbatim, for reproducibility)

All paths resolved via:
```bash
python -c "import sam3.model.sam3_image as m; print(m.__file__)"
# → /home/justin/projects/Efficient-SAM3-Finetuning/.venv/lib/python3.13/site-packages/sam3/model/sam3_image.py
```

Key cited spans:
- `sam3/model/sam3_image.py:440-451` — `forward_grounding` signature & first use.
- `sam3/model/sam3_image.py:167-210` — `_encode_prompt` → `geometry_encoder`.
- `sam3/model/sam3_image.py:547-553` — `_get_dummy_prompt` (the zero-hints canon).
- `sam3/model/sam3_image.py:576-580` — Meta's own `Prompt` construction from `find_input`.
- `sam3/model/geometry_encoders.py:17-80` — `is_right_padded`, `concat_padded_sequences` (mask convention).
- `sam3/model/geometry_encoders.py:83-238` — `Prompt` class (shapes, dtypes).
- `sam3/model/geometry_encoders.py:283-290` — `_init_box` (default mask = all-zeros = no padding).
- `sam3/model/geometry_encoders.py:470-489` — `SequenceGeometryEncoder` docstring ("normalized CxCyWH").
- `sam3/model/geometry_encoders.py:644-650` — coordinate-space confirmation in encoder body.
- `sam3/model/geometry_encoders.py:717-756` — `forward(geo_prompt, ...)`, the consumer.
- `sam3/model/data_misc.py:108-126` — `FindStage` dataclass showing `input_boxes` dtype is `float`, `input_boxes_mask` is `bool`.
