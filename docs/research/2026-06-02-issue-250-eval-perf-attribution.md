# Eval Performance Attribution — Issue #250 (Phase 1)

> Research write-up for [issue #250](https://github.com/NguyenJus/custom-sam-peft/issues/250)
> Date: 2026-06-02 (profile executed 2026-06-03)
>
> **Phase 1 deliverable.** This report's six contract values are the sole input to
> Phase 2 (Task 2.0 reads them and applies the decision gate). Phase 2 fills the
> `Before/After` subsection (Task 2.6).

## TL;DR

Eval wall-time is **CPU-postprocess bound, not GPU-forward bound.** On the RTX 5070
Ti, the serial single-threaded per-query RLE encode (`rle_encode`) alone is **78.2%**
of timed eval wall-time; the full postprocess (upsample + transfer-binarize +
RLE-encode) is **87.7%**. The GPU forward is only **12.3%**. The model emits
**N = 200** queries per forward, and the eval forward already runs in **bfloat16**
(the `Runtime(dtype=torch.float32)` label is cosmetic — Cause #1 was a misread).

All three Phase-2 levers are **GO**:

```text
LEVER_2a_top100_filter: GO        # postprocess dominates (87.7%) AND N=200 > 100
LEVER_3_batched_transfer_rle: GO  # transfer+RLE = 86.5% of eval
LEVER_1_dtype_label_cosmetic: GO  # pure cosmetic
```

---

## §1 — Run environment

| field | value |
| --- | --- |
| GPU | RTX 5070 Ti, sm_120 (CC 12.0), **native bf16** |
| model | `facebook/sam3.1`, `sam3.1_multiplex.pt`, `dtype: bfloat16`, baseline (no adapter / zero-shot) |
| dataset | COCO `val2017`, 8-image subset (`instances_val2017_subset8.json`), 80 dense classes, 55 GT instances |
| eval config | `eval.mode: full`, `visualize: false`, `save_predictions: false`, `batch_size: auto` |
| model input HW | 1008 × 1008 |
| mask logit HW | 288 × 288 |
| commit | `8628395` (branch `worktree-eval-perf-spike-250`) |
| harness | `scripts/profile_eval_250.py` under `CSP_EVAL_PROFILE=1`, CUDA-synchronized bucket timers (`eval/_profile.py`) |

The profile is a **representative full-mode run** (spec §4.2): real model, real COCO
images, real 1008² input, all 80 class-prompts evaluated. The 8-image count was
chosen only to bound wall-clock and memory on a 16 GB box — per-image cost is
invariant to image count, so the bucket shares are representative. The COCO 80-class
case is the **worst case** for postprocess (5 forwards/image); DataFusionContest's
12 classes is ~5× lighter (1 forward/image). Per the user, this report covers COCO
only.

---

## §2 — Per-bucket breakdown

CUDA-synchronized timers, summed across 8 images (40 forwards). `% of timed` is the
share of `TOTAL(timed)`, which is the sum of the five instrumented buckets.

| bucket | total (s) | per-image (ms) | % of timed |
| --- | ---: | ---: | ---: |
| `forward` (GPU) | 13.978 | 1 747 | 12.3% |
| `mask_upsample` | 1.369 | 171 | 1.2% |
| `transfer_binarize` | 9.442 | 1 180 | 8.3% |
| `rle_encode` | 89.100 | 11 138 | **78.2%** |
| `coco_aggregate` | 0.077 | 10 | 0.1% |
| **TOTAL(timed)** | **113.966** | **14 246** | **100.0%** |

- **Postprocess total** (`mask_upsample` + `transfer_binarize` + `rle_encode`) =
  99.91 s = **87.7%**.
- **`transfer_binarize` + `rle_encode`** = 98.54 s = **86.5%**.
- The headline pain: `rle_encode` is **~11.1 s/image** of serial, single-threaded
  `pycocotools mask_utils.encode` — paid per query, on the CPU, while the GPU idles.

This holds for in-training validation too: `queries_to_coco_results` is called
**unconditionally** in `Evaluator._predict`, so lite-mode periodic validation pays
the same per-query RLE tail (it only skips per-class breakdown, prediction saving,
and caps images at 64). That is the user's actual validation-slowness complaint.

---

## §3 — Measured N

```text
N: 200
```

`N` = `pred_logits.shape[1]`, the SAM 3.1 multiplex query count — a model-architecture
constant (not pinned in the repo; CPU stubs use N=4, spec §3.3). Measured on the real
model at 200. This clears the `N > 100` bar that gates LEVER_2a (the top-`max(maxDets)`
filter): with 200 queries per forward but COCO mAP capped at `maxDets=100`, ≥100
queries per forward are discarded *after* paying their full RLE cost.

---

## §4 — bf16-already-on confirmation (Cause #1)

```text
eval_forward_dtype: torch.bfloat16
```

The eval forward outputs are **bfloat16**, measured directly. There is **no slow
fp32 forward to fix.**

- Weights are cast to bf16 unconditionally at load: `_apply_dtype`
  (`models/sam3.py:569`) → `model.to(dtype=torch.bfloat16)` (`:576`), no capability
  branch.
- The `Runtime(device=param_device, dtype=torch.float32)` label at
  `evaluator.py:152` is **cosmetic** — it is never applied to forward math (`to_device`
  moves the device only). The empirical bf16 output dtype proves the forward runs bf16
  regardless of the label. Removing the misleading label is the (cosmetic) LEVER_1.

---

## §5 — CC 7.5 (T4) finding (Cause #1b)

**Reasoned from code, not measured** — no T4 hardware is available, and the repo's
`gpu_t4` tier was only ever validated on the 5070 Ti superset. Stated explicitly per
spec §3.2:

- **Eval weights are bf16 even on CC 7.5.** `_apply_dtype` (`models/sam3.py:569`)
  casts to bf16 **unconditionally** — there is no capability coercion at load.
- **Training coerces; eval does not.** Training's autocast `_autocast_ctx`
  (`train/loop.py:200`) routes through `coerce_dtype_for_capability`
  (`runtime/_runtime.py`) which downgrades bf16 → fp16 on CC < 8.0. Eval has **no
  autocast**, so eval weights stay bf16 on CC 7.5 — but bf16 on sub-Ampere is
  **non-native / emulated**, and there is a potential **TRAIN (fp16-via-autocast) vs
  EVAL (bf16-via-weights) precision divergence** on such cards.
- **Crash-risk cross-link.** If eval precision ever collapsed to fp16, fp16's ~65504
  max can overflow to inf/NaN, and the postprocess finite-guards
  (`postprocess.py:90, :98, :108`) **raise `RuntimeError`** — a hard crash, not silent
  drift. On the bf16 eval path above the guards should not trip from range overflow
  (bf16 shares fp32's exponent range), but the guards remain the failure mode if a
  future change coerces eval to fp16.
- **Caveat.** The "bf16 is already on / faithful" claim is validated **only on the RTX
  5070 Ti (CC 12.0, native bf16)** and **must not be assumed to generalize** to CC 7.5.

---

## §6 — Forwards-per-image (Cause #4)

```text
forwards_per_image: 5    # = forwards / n_images = 40 / 8 = ceil(n_classes / 16)
```

Per-image cost scales with `ceil(n_classes / MULTIPLEX_CAP)`, `MULTIPLEX_CAP = 16`.
COCO's 80 classes → 5 forwards/image; DataFusionContest's 12 classes → 1
forward/image. **Reported, not optimized** — reducing forwards/image is out of scope
(spec §9).

---

## §7 — GO / NO-GO per lever

```text
LEVER_2a_top100_filter: GO        # postprocess dominates (87.7%) AND N=200 > 100
LEVER_3_batched_transfer_rle: GO  # transfer+RLE = 86.5% of eval
LEVER_1_dtype_label_cosmetic: GO  # pure cosmetic, no measurement dependency
```

- **LEVER_2a** — GO. Postprocess buckets dominate (87.7% ≫ forward's 12.3%) **and**
  N = 200 > 100. The top-`max(maxDets)=100` filter discards ≥100 queries/forward
  *before* RLE, mAP-exactly.
- **LEVER_3** — GO. `transfer_binarize` + `rle_encode` = 86.5%, a non-trivial share
  (independent of N). Batched device→host transfer + batched RLE over survivors.
- **LEVER_1** — GO (always). Drop the misleading `dtype=torch.float32` Runtime label.

No surprises: the decision gate's GO condition is met. Phase 2 proceeds.

---

## §8 — Phase 2 plan

Phase 2 (spec §4.4) consumes **only** the contract values above and lands three
mAP-exact changes — the tie-safe top-`max(maxDets)` query filter (#2a), the batched
device→host transfer + batched RLE over survivors (#3), and the cosmetic dtype-label
removal (#1) — then removes the Phase-1 instrumentation (`eval/_profile.py`,
`scripts/profile_eval_250.py`) and re-profiles.

### Before/After

Re-profiled on the same RTX 5070 Ti, same 8-image COCO `val2017` subset, same
zero-shot baseline (`checkpoint=None`), same instrumented profiler driver — the
**after** run uses the post-Phase-2 code (top-`max(maxDets)` filter + batched
transfer/RLE + dtype-label fix) with the Phase-1 timers temporarily restored for an
apples-to-apples bucket comparison, then reverted (uncommitted).

| bucket (timed) | before (s) | after (s) | after/before |
| --- | ---: | ---: | ---: |
| `forward` (GPU) | 13.978 | 13.587 | 0.97× (unchanged) |
| `mask_upsample` | 1.369 | 0.696 | 0.51× |
| `transfer_binarize` | 9.442 | 5.805 | 0.61× |
| `rle_encode` | 89.100 | 48.513 | 0.54× |
| `coco_aggregate` | 0.077 | 0.049 | 0.64× |
| **TOTAL(timed)** | **113.966** | **68.650** | **0.60× → 1.66× speedup** |

**Eval timed wall-time: 113.97 s → 68.65 s — a 1.66× speedup (−40%)** on this
config, with the GPU `forward` bucket unchanged (as expected — the levers touch only
postprocess). Per-image timed cost drops 14.25 s → 8.58 s.

**mAP proven unchanged (the optimization is free).** Full `MetricsReport` is
bit-identical before vs after:

| metric | before | after |
| --- | ---: | ---: |
| `mAP` | 0.0 | 0.0 |
| `mAP_50` | 0.0 | 0.0 |
| `mAP_75` | 0.0 | 0.0 |
| `per_class` | `{}` | `{}` |

The mAP is 0.0 because this is the **zero-shot** baseline (no PEFT adapter), so the
before/after equality is empirically trivial here. The substantive mAP-exactness
proof is therefore **analytic + unit-tested**, not from this run: pycocotools'
COCOeval truncates to `max(params.maxDets)` = 100 detections by score per
`(image, category)`, so dropping strictly-lower-scored survivors cannot move the
metric. This is locked in by `tests/unit/test_eval_postprocess.py`
(`test_filter_keeps_top_cap_by_score`, `test_filter_boundary_ties_keep_superset`,
`test_filter_no_op_when_n_le_cap`, `test_filter_n_zero_returns_empty`) and the
batched-RLE characterization test (`test_batched_rle_decodes_identically`).

**Filter is live and tie-safe.** Emitted detections drop **128 000 → 64 635**
(8 images × 80 classes × N): the filter caps each `(image, category)` group at the
top-100 by score, and the +635 over a hard `8 × 80 × 100 = 64 000` is the **tie-safe
superset** — boundary ties at the 100th-highest score are kept (`score >= kth`), never
truncated below the cap. mAP is unaffected because COCOeval re-truncates to 100 by
score anyway.

**Per-lever realized contribution.**

- **Top-`max(maxDets)` filter (#2a) — dominant.** Halving the survivor count
  (N=200 → ~100) roughly halves every postprocess bucket: `mask_upsample` 0.51×,
  `transfer_binarize` 0.61×, `rle_encode` 0.54×. This is the bulk of the 1.66×.
- **Batched transfer + RLE (#3) — marginal here.** Per-mask, RLE cost is essentially
  flat (before 89.100 s / 128 000 ≈ 696 µs/mask; after 48.513 s / 64 635 ≈ 750 µs/mask).
  `pycocotools mask_utils.encode` is **C-encode-bound by mask area**, not Python-loop
  bound, so collapsing the per-query loop into one batched call removes little wall-time
  on full-resolution masks (the `(M,H,W) → Fortran (H,W,M)` copy offsets the loop saving).
  The change remains valuable as a bitwise-identical simplification and avoids per-query
  Python overhead that would matter more for many small masks, but on this config the
  realized win comes from the filter.
- **Dtype-label removal (#1) — cosmetic.** Zero runtime effect (confirmed: eval
  `forward` bucket unchanged); the eval forward was already bf16.
