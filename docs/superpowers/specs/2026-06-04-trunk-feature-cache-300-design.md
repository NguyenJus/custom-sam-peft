# Trunk feature cache: replay frozen ViT-trunk features across epochs

## Motivation / context

Issue [#300](https://github.com/justin/custom-sam-peft/issues/300): when the
ViT vision trunk is fully frozen and deterministic, `forward_image` recomputes
identical features every epoch. This spec specifies a pure REPLAY cache:
compute trunk features once on epoch 0, replay on epochs 1+, and skip the trunk
forward entirely. The lever is wall-clock for compute-constrained runs; net
saving per replayed step is approximately `(trunk_fwd_time - replay_time)`,
where under the resolved single-run on-disk (SSD) residence
`replay_time = disk_read + pin + H2D` (≈ 21.3 ms cold, or ≈ 0 with the 1-step
prefetch in §3.5), summed over `(epochs - 1)` epochs. Measured: +70 ms/step
cold, ~+91 ms with prefetch
(`docs/spikes/2026-06-04-trunk-feature-cache-disk-respike.md`).

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

This spec is SPIKE-FIRST: production wiring was conditional on a feasibility
spike on the real SAM3.1 model (see Spike-first plan). That spike has now RUN
(twice): Part A is DONE and RESIDENCE is RESOLVED to **single-run on-disk (SSD)**
(§5 Part A, Open question (c)). The original HDD-era spike
(`docs/spikes/2026-06-04-trunk-feature-cache-spike.md`, NO-GO on disk) was
reversed by an SSD re-spike after WSL moved onto an SSD; see
`docs/spikes/2026-06-04-trunk-feature-cache-disk-respike.md`.

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
`(trunk_fwd_time - replay_time)`. The spike confirmed this is positive on the
target box (+70 ms/step cold, ~+91 ms with prefetch) before any wiring lands.

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
per-image entries, `detach()`ed, cast to fp16, and — under the resolved
single-run on-disk residence (§3.5) — SERIALIZED to the on-disk cache dir
(`torch.save`, one file per `sample_uid`). VRAM stays free; host RAM is not held
either (this reverses the original RAM-residence plan, which kept PINNED CPU
tensors). On replay the entry is COLD-READ from disk, pinned, then non-blocking
H2D. Prior art for the pin + H2D tail: the pinned-copy path in #288 /
`transfer_binarize` (the only survivor of the #273 algo/CUDA audit). The
trunk-config fingerprint in the Key (above) is written into / checked on every
on-disk entry, so a corrupt or foreign blob is rejected rather than replayed.

## 3.5 Disk residence, activation guards, prefetch, cleanup

Residence is RESOLVED: **single-run on-disk (SSD)**. The original HDD-era spike
was NO-GO on disk (HDD-saturation crash + 193 GiB ≫ 16 GB RAM); after WSL moved
onto an SSD the re-spike flipped it to GO — the 3,720-image / ~193 GiB DFC roof
working set fits the 918 GB free SSD with ~4.7× headroom, and cold reads sustain
2.91 GB/s
(`docs/spikes/2026-06-04-trunk-feature-cache-disk-respike.md`).

### Single-run, NOT cross-run persistent

The cache lives under the run's output directory — default
`<output_dir>/.trunk_cache/` (cited default path: a hidden subdir of the run's
own `output_dir`, so it is namespaced per-run and swept with the run) — and is
DELETED at run end in a `finally` / teardown (see Cleanup below). It is NOT a
multi-run persistent cache.

Rationale: a 160-epoch run already replays on epochs 1–159; the one-time
epoch-0 build is ~88 ms × 3,720 ≈ **5.5 min**, negligible against the run. Cross-run
persistence would cost ~193 GiB on disk PER distinct trunk fingerprint and
accumulate (only ~4 such caches fit the 918 GB SSD), for a benefit that only
materializes on frequent fresh restarts with byte-identical configs — not worth
the storage bloat and the cross-run invalidation/cleanup policy it would require.
This keeps §7's "no multi-run persistent disk cache" stance, now as a DELIBERATE
choice rather than a deferral.

### Serialization

Per-image entry via `torch.save`, fp16. `torch.save` is self-describing with
~zero size overhead; the measured 17.86 ms cold read is dominated by the disk
read, not unpickling, so deserialization cost is not on the critical path.
Raw-bytes / `safetensors` is noted as a LATER tuning knob only — adopt it only if
deserialize ever shows up in a profile (it does not today; re-spike Step 2d).

### Three-layer activation guard (in addition to §2's three correctness guards)

§2's three correctness guards (trunk-frozen / RGB / aug-off) are UNCHANGED and
still all required. The residence decision adds three further activation gates.
**None may gate on the kernel `rotational` flag** — the re-spike proved it
unreliable under WSL2/VHDX: this SSD reports `rotational=1` (would falsely flag
as HDD) while RAM-backed `ram*` devices report `rotational=0`
(`...disk-respike.md`, "Device assessment").

- **(a) Opt-in flag** — `cache_trunk_features` (cited default `false`,
  unchanged from §2). Caching never turns on implicitly.
- **(b) Free-disk fit-check** (build-time, fail-fast) — compute projected cache
  size `n_samples × per_image_bytes` from the ACTUAL post-tiling sample count
  (per-`sample_uid`, so tiling-window expansion is counted) and per-image bytes
  (53.16 MiB fp16 measured; re-spike Step 1). Refuse to enable, naming the flag,
  when the projected size exceeds **70%** of FREE disk on the cache volume.
  Cited default `0.70`: leaves ≳30% headroom on the cache volume for checkpoints,
  logs, DataLoader spill, and OS, while still admitting the 193 GiB DFC set on
  the 918 GB SSD (193/918 ≈ 21% ≪ 70%); `# tbd:` if the user wants a tighter
  number. This REPLACES the original spec's implicit 16 GB host-RAM fit concern
  (residence is disk now, not RAM).
- **(c) Throughput auto-guard** (build-time probe of the cache dir) — cold-read a
  representative blob (page-cache evicted, as prototyped in
  `scripts/spike_trunk_cache_feasibility.py` Step 2d) and measure throughput. The
  break-even is DERIVED, not hardcoded: `feature_bytes / trunk_fwd` — below this,
  a cold read costs more than just recomputing the trunk. With the measured
  53.16 MiB / 91.4 ms this is ≈ **0.57 GB/s**; the spec stores the derivation, and
  the guard recomputes it from the live `per_image_bytes` and `trunk_fwd` at probe
  time. If measured throughput is below break-even, refuse / fall back to
  recompute UNLESS the explicit override `cache_allow_slow_disk` (cited default
  `false`) is set. The measured SSD cold read (2.91 GB/s) clears the bar by ~5×;
  the probe is the gate that auto-PASSES this SSD and would auto-FAIL a genuine
  ~0.15 GB/s HDD — by measured speed, not by the unreliable device label.

### Replay path + 1-step prefetch

Replay is **cold read → pin → non-blocking H2D**, WITH a depth-1 prefetch. A
background reader prefetches the NEXT step's blobs during the current step's
compute, hiding the ~17.86 ms read behind the ~91 ms prior-step compute and
lifting the win from **+70 ms** (cold, no prefetch) to **~+91 ms/step**
(re-spike Net-win table). This requires lookahead into the (possibly shuffled)
sampler order plus a background reader thread (or a dedicated CUDA stream for the
H2D tail).

Prefetch DEPTH = 1 (cited): one step's ~91 ms compute already exceeds one blob's
~18 ms read, so depth 1 fully hides the read; deeper prefetch only adds pinned-
memory pressure for marginal gain.

### Cleanup / teardown

The cache dir is DELETED at run end in a `finally` / teardown block, so an
aborted or crashed run does not leak ~193 GiB. Because residence is single-run,
there is no cross-run reuse to preserve — teardown unconditionally removes
`<output_dir>/.trunk_cache/`.

## 4. Batch policy

- **Epoch 0 (all-miss):** run the trunk on the full batch; store each image's
  unbound entry.
- **Epochs 1+ (all-hit):** if EVERY image in the batch hits, assemble
  `backbone_out` from cache and skip the trunk entirely.
- **Any miss (only possible under eviction):** recompute the WHOLE batch and
  refresh the cache.

No per-image scatter/gather inside the trunk: the trunk runs on either the full
batch or none of it.

The batch-policy LOGIC is unchanged by the disk-residence decision. Under
single-run disk residence the whole working set fits (the §3.5(b) fit-check
guarantees it), so after epoch 0 a "miss" only happens on a missing or corrupt
on-disk blob — caught by the trunk-config fingerprint on read (§3 Stored value).
Such a miss falls into the existing any-miss path: recompute that whole batch and
rewrite its entries.

## 5. Spike-first plan

### Part A: feasibility spike (GPU box, real SAM3.1) — DONE

DONE and GO. Ran twice. Verdict: **GO on the mechanics, GO on single-run on-disk
(SSD) residence** for the full ~160-epoch DFC roof run.

- Original (HDD-era):
  `docs/spikes/2026-06-04-trunk-feature-cache-spike.md` — GO on mechanics, NO-GO
  on residence (HDD-saturation crash rule + 193 GiB ≫ 16 GB RAM).
- SSD re-spike:
  `docs/spikes/2026-06-04-trunk-feature-cache-disk-respike.md` — after WSL moved
  to an SSD, residence FLIPPED to GO. Snapshot:
  `docs/spikes/snapshots/2026-06-04-trunk-feature-cache-disk-respike-snapshot.json`.

Measured (RTX 5070 Ti, real SAM 3.1, B=1 bf16): 53.16 MiB/image fp16;
`trunk_fwd` 91.4 ms; cold disk read (page-cache evicted) 17.86 ms (2.91 GB/s);
full replay 21.31 ms; net win +70 ms/step cold, ~91 ms with 1-step prefetch;
193 GiB fits 918 GB free SSD; break-even read throughput
`feature_bytes / trunk_fwd` ≈ 0.57 GB/s.

### Part B: implementation (spike says GO)

- The cache module, residence = **single-run on-disk (SSD)** under
  `<output_dir>/.trunk_cache/` (§3.5): `torch.save`/fp16 per-`sample_uid` entry,
  cold-read → pin → non-blocking H2D on replay.
- The three correctness guards (Section 2) — UNCHANGED.
- The three-layer activation guard (§3.5): opt-in `cache_trunk_features`
  (default `false`); build-time free-disk fit-check (refuse > 70% of free disk);
  build-time throughput auto-guard (probe cold read vs the derived
  `feature_bytes / trunk_fwd` break-even, override `cache_allow_slow_disk`,
  default `false`). NONE gates on the kernel `rotational` flag.
- The 1-step prefetch (§3.5): background reader / CUDA stream hides the read
  behind prior-step compute; depth 1.
- Cleanup / teardown (§3.5): delete the cache dir in a `finally` / teardown.
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
- **Free-disk fit-check fail-fast (§3.5b):** a projected cache size over the
  70% free-disk fraction hard-errors at build time, naming the flag; a size under
  it passes. Drive with a stubbed free-disk value and synthetic
  `n_samples × per_image_bytes`.
- **Throughput auto-guard fail / override (§3.5c):** a probed cold-read
  throughput below the derived `feature_bytes / trunk_fwd` break-even refuses (or
  falls back to recompute) unless `cache_allow_slow_disk=true`; above break-even
  passes. The guard must NOT read the kernel `rotational` flag (assert it is
  derived from measured throughput).
- **Cleanup on teardown (§3.5):** the cache dir is removed on normal exit AND on
  a simulated mid-run exception (the `finally` / teardown fires).
- **Prefetch correctness (§3.5):** replayed `backbone_out` is identical with the
  1-step prefetch ON vs OFF (prefetch must not change the replayed features,
  including under a shuffled sampler order).

## 7. Out of scope

- The adapter-free trunk scope itself (already landed as `decoder_concept` in
  #304). Guard 1 hard-errors when a trunk-trainable scope is used.
- Feature-space augmentation.
- A MULTI-RUN persistent disk cache. Residence is single-run on-disk (SSD) and
  swept at teardown (§3.5) — this is now a DELIBERATE choice, not a deferral: see
  §3.5 "Single-run, NOT cross-run persistent" for the storage-bloat /
  cross-run-invalidation rationale.
- QLoRA / bnb interactions.

## Open questions resolved

- **(a) Scope dependency:** satisfied — the adapter-free `decoder_concept` scope
  landed in #304 and is the new default. Guard 1 remains the hard backstop for
  any trunk-trainable scope/override.
- **(b) Augmentation:** hard-require aug-off via a fail-fast build-time guard
  (Guard 3) asserted against the built train transform — not a silent
  best-effort.
- **(c) Residence (RAM / disk / hybrid):** RESOLVED — **single-run on-disk
  (SSD)** under `<output_dir>/.trunk_cache/`, deleted at run end (§3.5). The
  HDD-era spike was NO-GO on disk (HDD-saturation crash + 193 GiB ≫ 16 GB RAM);
  after WSL moved onto an SSD the re-spike
  (`docs/spikes/2026-06-04-trunk-feature-cache-disk-respike.md`) flipped it to
  GO — 193 GiB fits the 918 GB free SSD, cold reads sustain 2.91 GB/s ≫ the
  0.57 GB/s break-even, and the old HDD-saturation crash mode no longer exists.
  RAM residence is rejected (the 193 GiB working set is ~12× host RAM, and a
  capped RAM-LRU thrashes to ~0% hit rate since intra-epoch reuse is zero).
- **(d) Activation:** opt-in flag `cache_trunk_features` (cited default `false`)
  with lazy populate — epoch 0 fills the cache, epochs 1+ replay.
