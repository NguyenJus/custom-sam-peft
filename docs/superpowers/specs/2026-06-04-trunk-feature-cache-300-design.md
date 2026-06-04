# Trunk feature cache: replay frozen ViT-trunk features across epochs

## Motivation / context

Issue [#300](https://github.com/justin/custom-sam-peft/issues/300): when the
ViT vision trunk is fully frozen and deterministic, `forward_image` recomputes
identical features every epoch. This spec specifies a pure REPLAY cache:
compute trunk features once on epoch 0, replay on epochs 1+, and skip the trunk
forward entirely. The lever is wall-clock for compute-constrained runs; net
saving per replayed step is approximately `(trunk_fwd_time - H2D_copy_time)`,
summed over `(epochs - 1)` epochs.

This feature REQUIRES an adapter-free trunk scope, which has landed:
[#304](https://github.com/justin/custom-sam-peft/pull/304) added `decoder_concept`
as the new default scope. `decoder_concept` is `vision_decoder_concept` MINUS the
ViT-trunk pattern, so the trunk carries no LoRA and all its base params keep
`requires_grad=False` (`SCOPE_TARGETS` / `SCOPE_MHA_MODULES` at
`src/custom_sam_peft/peft_adapters/lora.py:39-86`;
`LoraScope = Literal["vision","vision_decoder","vision_decoder_concept","decoder_concept","all"]`
at `src/custom_sam_peft/config/schema.py:106`, default
`scope = "decoder_concept"` at `schema.py:583`). The fully-frozen-trunk
precondition is therefore the project default. Guard 1 (Section 2) remains the
hard backstop: if any future scope or override leaves the trunk trainable, the
cache hard-errors rather than miscaching.

This spec is SPIKE-FIRST: production wiring is conditional on a feasibility
spike on the real SAM3.1 model (see Spike-first plan). Cache RESIDENCE
(RAM / disk / hybrid) is deliberately left to the spike's evidence.

## 1. Regime and shape of the win

A pure REPLAY cache. Under a fully-frozen, deterministic trunk with a fixed
input, `forward_image` returns identical features each epoch. Compute once
(epoch 0), replay epochs 1+.

The image size is fixed at `SAM3_IMAGE_SIZE = 1008`
(`src/custom_sam_peft/presets.py:51`). Because `vision_pos_enc` is
image-CONTENT-independent (it depends only on the fixed spatial grid), it is
computed once and EXCLUDED from the per-image cache. Only the content-dependent
tensors are stored: `backbone_fpn` / `vision_features` (and, when active, the
`sam2_backbone_out` pyramid). `forward_image` (defined in
`.venv/.../sam3/model/vl_combiner.py`) returns:

```text
{
  "vision_features": sam3_features[-1],
  "vision_pos_enc":  sam3_pos,          # excluded: content-independent
  "backbone_fpn":    sam3_features,     # cached
  "sam2_backbone_out": <dict | None>,   # cached when present
}
```

Net saving per replayed step is approximately
`(trunk_fwd_time - H2D_copy_time)`. The spike confirms this is positive on the
target box before any wiring lands.

## 2. Correctness gate: three independent guards, ALL required

Activation is opt-in via the config flag `cache_trunk_features`
(cited default: `false` — a wall-clock optimization, off until validated; see
the user's "cite new hyperparams" rule). When set, the adapter performs a
FAIL-FAST hard-error at build time unless ALL THREE preconditions hold. Each
failure must name the offending condition AND the config key to change.

1. **Trunk frozen.** Zero `requires_grad` params under the trunk AND no LoRA
   module attached to it. This is also the no-op backstop: if the trunk is
   trainable, error rather than silently miscache. The default `decoder_concept`
   scope satisfies this (no trunk-attached LoRA); legacy scopes still attach
   trunk LoRA via `SCOPE_TARGETS` / `SCOPE_MHA_MODULES` (`lora.py:39-86`), and
   this guard rejects them.
2. **RGB input.** `channel_adapter is None`, i.e. `channel_semantics == "rgb"`.
   `_build_channel_adapter` (`src/custom_sam_peft/models/sam3.py:243-270`)
   returns `None` for RGB and otherwise a fully-trainable
   `nn.Conv2d(channels, 3, 1)` applied UPSTREAM of the trunk
   (`sam3.py:324-330`). A trainable channel adapter drifts the trunk input
   every step, so caching is invalid.
3. **Aug-off.** No trunk-input-affecting train augmentation
   (geometric / photometric / resize / jitter). Asserted against the BUILT
   train transform. Augmentation is stochastic per-epoch via the albumentations
   global RNG (`A.*` with `p=0.5`, `np.random.uniform`) in
   `src/custom_sam_peft/data/transforms.py`; the per-image `rng` in
   `src/custom_sam_peft/data/coco.py:344` only seeds PROMPT sampling, not image
   pixels. With augmentation on, the trunk input would differ across epochs,
   which is exactly why this guard is mandatory.

## 3. Cache boundary, key, contents

### Boundary

Wrap exactly `self.model.backbone.forward_image(images)` at
`src/custom_sam_peft/models/sam3.py:332`, BEFORE the
`backbone_out.update(text_outputs)` call at `sam3.py:336`. Text outputs are
prompt-dependent and cheap (`forward_text` at `sam3.py:333-335`) and are NEVER
cached.

### Key

A stable per-SAMPLE uid, NOT `image_id` alone. Tiling expands one image into
`(image_id, window)` samples with distinct trunk inputs
(`self._samples: list[tuple[int, Window]]` in
`src/custom_sam_peft/data/coco.py`). Introduce a `sample_uid`
(e.g. `f"{image_id}:{window}"`) on the `Example` / collate path and thread it
down to the adapter. The batch dict already carries `image_ids`
(`src/custom_sam_peft/data/collate.py:32`); `sample_uid` is threaded alongside
it. The uid is stable across epochs even with shuffle because aug-off fixes the
`index -> pixels` mapping.

The key namespace also includes a trunk-config FINGERPRINT (trunk identity:
checkpoint id, scope, dtype, image size) so a stale cache cannot be replayed
against a different trunk.

### Stored value

The `forward_image` return dict (minus `vision_pos_enc`), batch-unbound into
per-image entries, `detach()`ed, kept on CPU (VRAM stays free), and PINNED for
fast non-blocking H2D on replay. Prior art: the pinned-copy path in #288 /
`transfer_binarize` (the only survivor of the #273 algo/CUDA audit).

## 4. Batch policy

- **Epoch 0 (all-miss):** run the trunk on the full batch; store each image's
  unbound entry.
- **Epochs 1+ (all-hit):** if EVERY image in the batch hits, assemble
  `backbone_out` from cache and skip the trunk entirely.
- **Any miss (only possible under eviction):** recompute the WHOLE batch and
  refresh the cache.

No per-image scatter/gather inside the trunk: the trunk runs on either the full
batch or none of it.

## 5. Spike-first plan

Residence is deferred to evidence. The spike is the gate before any production
wiring.

### Part A: feasibility spike (GPU box, real SAM3.1)

1. Per-image feature bytes in fp16 — measured WITH and WITHOUT the
   `sam2_backbone_out` path.
2. Trunk-forward wall-clock as a fraction of the full `train_step`
   (`src/custom_sam_peft/train/loop.py:227-475`), plus H2D copy time for cached
   features. Confirm net win `> 0`. Reuse the permanent profiling harness
   (`CSP_PROFILE=1` + `csp profile`).
3. Break-even table: dataset size x per-image bytes vs 16 GB host RAM and vs
   disk-I/O headroom.
4. Recommendation: go / no-go plus residence choice (RAM-cap / disk / hybrid).
   Disk is explicitly weighed against the documented HDD-saturation
   session-crash risk on this box (sessions crash from disk-I/O saturation, not
   RAM/VRAM OOM).

### Part B: conditional implementation (only if the spike says go)

- The cache module (residence filled in from Part A).
- The three correctness guards (Section 2).
- The `cache_trunk_features` config flag (default `false`).
- `_Sam3ImageAdapter` integration at the boundary (Section 3), including the
  `sample_uid` threading on the `Example` / collate path.

## 6. Testing

CPU stub model, following the existing shape-probe pattern in
`tests/unit/test_sam3_wrapper.py`. Real-model byte / timing numbers live in the
spike (Part A), NOT in CI.

- **Guard matrix:** each precondition violation (trainable trunk / trunk-LoRA,
  non-RGB channel adapter, aug-on) produces the correct hard-error with the
  right message and the right config key named.
- **Key stability:** `sample_uid` is stable across simulated epochs and across
  shuffle; tiling windows of one image map to distinct uids.
- **Epoch-0-store / epoch-1-replay equivalence:** the replayed `backbone_out`
  is bit-identical to a fresh recompute (modulo the excluded `vision_pos_enc`,
  which is recomputed).
- **Eviction -> recompute:** a forced miss recomputes the whole batch and
  refreshes the cache.

## 7. Out of scope

- The adapter-free trunk scope itself (already landed as `decoder_concept` in
  #304). Guard 1 hard-errors when a trunk-trainable scope is used.
- Feature-space augmentation.
- A multi-run persistent disk cache, unless the spike picks disk residence.
- QLoRA / bnb interactions.

## Open questions resolved

- **(a) Scope dependency:** satisfied — the adapter-free `decoder_concept` scope
  landed in #304 and is the new default. Guard 1 remains the hard backstop for
  any trunk-trainable scope/override.
- **(b) Augmentation:** hard-require aug-off via a fail-fast build-time guard
  (Guard 3) asserted against the built train transform — not a silent
  best-effort.
- **(c) Residence (RAM / disk / hybrid):** deferred to the Part A spike;
  disk is weighed against the known HDD-saturation crash risk on this box.
- **(d) Activation:** opt-in flag `cache_trunk_features` (cited default `false`)
  with lazy populate — epoch 0 fills the cache, epochs 1+ replay.
