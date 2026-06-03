# Algorithmic + CUDA Performance Audit — Issue #273

> Triage record for [issue #273](https://github.com/NguyenJus/custom-sam-peft/issues/273)
> (algo/CUDA perf survey) and the first live consumer of
> [issue #256](https://github.com/NguyenJus/custom-sam-peft/issues/256)
> (repeatable profiler-run attribution workflow).
> Date: 2026-06-03 · branch `worktree-audit-273-algo-cuda` · base `6a8c863`
>
> **No optimization code lands under #273** — this is a triage record. The
> measurement infrastructure that lands as code (the §3b profiling buckets and
> the #256 attribution reader) is analysis, not optimization (same justification
> as #265's permanent `eval.dataset_load` timer).

## TL;DR

Profiled the four surfaces on the real SAM3.1 checkpoint + LoRA adapter against a
small DataFusionContest subset, ran every snapshot through the new #256
attribution reader, and re-ranked #273's six candidates against the **post-#276**
distribution. **Only candidates #4 and #6 clear the 5%-of-surface-wall triage
threshold; #1, #2, #3, #5 are retired with the measurement that killed them.**

```text
#1 batch GT-side RLE        RETIRE  eval.gt_rle_encode = 0.7% of full-eval wall
#2 matcher single-sync      RETIRE  train.matcher = 2.2%; bs=1 ⇒ 1 sync/step anyway
#3 semantic confusion→GPU   RETIRE  transfer+confusion = 1.8% at K=4 (K-scaling noted)
#4 mask transfer-binarize   FILE    eval.transfer_binarize = 14.6%; pinned-copy 8.3× EXACT
#5 per-example IoU (viz)    RETIRE  eval.pair_iou = 1.1%; viz-only, no metric impact
#6 forward levers           FILE    train.forward 33% / eval.forward 15%; folded into #4 issue
```

The single biggest full-eval bucket — `eval.rle_encode` at **62.6%** — is the
*already-optimized* pred-side path (batched in #257, top-100 query filter already
applied at `evaluator.py:502`). No further "do-less" lever remains there in the
candidate set. The lite path is RLE-free and GPU-forward-bound (negative control,
confirmed). **One filed issue** covers #4 (primary, exact ~12–13% eval win) with
#6 as a documented secondary lever requiring its own deeper spike.

---

## §1 — Run environment

| field | value |
| --- | --- |
| GPU | RTX 5070 Ti, sm_120 (CC 12.0), native bf16, 16 GB |
| model | `facebook/sam3.1`, `sam3.1_multiplex.pt`, `dtype: bfloat16` |
| adapter | `runs/test-dfc-20260603-035435/adapter` (LoRA r=64 α=32, scope `vision_decoder_concept`) — **validated end-to-end** (loads + runs full eval; the recently-hand-fixed config is sound) |
| dataset | DataFusionContest roof subset (`runs/test-dfc-20260603-035435/subset.json`) |
| harness | permanent `CSP_PROFILE` bucket timers (#255/#263) **including the §3b buckets added by this work**; `csp profile` (eval) and `CSP_PROFILE=1 csp train` |
| attribution | `scripts/attribute_profile.py` / `src/custom_sam_peft/profiling_report.py` (#256, this work) |
| commit | `6a8c863` |

**Caveat — image count.** `data.limit.val` does **not** gate the eval val path
(it derives from `val_split`), so eval-full ran on the full **355**-image val
split, not a cap. It completed on the 16 GB box (predictions list peaked safely)
but took ~13 min. The capped surfaces below (`eval-pairiou`, semantic) use
explicit small val sets. Train stopped at **step 18** on the host-RAM floor
(`num_workers=4` + model on the 16 GB box); 18 steps is sufficient for *relative*
bucket attribution.

---

## §2 — Per-surface attribution

### eval-full (exact COCO, 355 images, visualize=false) — wall 777.4 s

| Bucket | Seconds | % of wall | Kind |
| --- | --- | --- | --- |
| `eval.rle_encode` | 486.36 | **62.6%** | cpu (already batched #257, post top-100) |
| `eval.forward` | 116.34 | 15.0% | gpu |
| `eval.transfer_binarize` | 113.59 | **14.6%** | sync ← **candidate #4** |
| `eval.dataset_load` | 11.55 | 1.5% | io |
| `eval.gt_rle_encode` | 5.37 | 0.7% | cpu ← candidate #1 |
| `eval.mask_upsample` | 4.73 | 0.6% | gpu |
| `eval.box_transfer` | 0.88 | 0.1% | sync |
| `eval.coco_aggregate` | 0.40 | 0.1% | cpu |
| (residual) | 38.14 | 4.9% | — |

Structural facts: N=200 queries/forward, 355 forwards (1.0/image), forward dtype
bf16, mask logits 288×288. CPU 63.3% / GPU 15.6% / sync 14.7% / IO 1.5%.

### eval-lite (post-#276 proxy, 64 images, visualize=false) — wall 25.7 s · negative control

| Bucket | Seconds | % of wall | Kind |
| --- | --- | --- | --- |
| `eval.forward` | 20.63 | **80.3%** | gpu |
| `eval.dataset_load` | 1.57 | 6.1% | io |
| `eval.mask_upsample` | 0.80 | 3.1% | gpu |
| `eval.proxy_iou` | 0.29 | 1.1% | gpu |
| (residual) | 2.40 | 9.3% | — |

**Confirms the spec's premise**: the lite path emits **zero** `rle_encode` /
`transfer_binarize` / `gt_rle_encode` buckets — it is RLE-free and
GPU-forward-bound. Candidates #1/#4/#5 do not apply on this path.

### eval-pairiou (full + visualize, 15-image explicit val) — wall 24.4 s · attributes #5

| Bucket | Seconds | % of wall | Kind |
| --- | --- | --- | --- |
| `eval.rle_encode` | 19.57 | 80.3% | cpu |
| `eval.forward` | 4.98 | 20.4% | gpu |
| `eval.transfer_binarize` | 2.31 | 9.5% | sync |
| `eval.gt_rle_encode` | 1.43 | 5.9% | cpu |
| `eval.pair_iou` | 0.26 | **1.1%** | cpu ← **candidate #5** |
| `train.matcher` | 0.30 | 1.2% | cpu (viz pred↔GT pairing — see note) |
| `eval.coco_aggregate` / `eval.box_transfer` | <0.1 | 0.2% / 0.2% | — |

`eval.pair_iou` (the request-gated per-example-IoU `mask_utils.iou` for viz
sample-picking) is **1.1%** of even this tiny run. `eval.gt_rle_encode` reads 5.9%
here vs **0.7% on the representative 355-image full run** — it is a one-time
GT-build cost amortized over image count, so the full-run figure is the one that
governs candidate #1.

### train (instance, 18 steps, batch_size=1) — wall 31.0 s

| Bucket | Seconds | % of wall | Kind |
| --- | --- | --- | --- |
| `train.backward` | 18.66 | 60.1% | gpu |
| `train.forward` | 10.24 | **33.0%** | gpu ← candidate #6 |
| `train.loss` | 1.23 | 3.9% | gpu |
| `train.matcher` | 0.68 | **2.2%** | cpu/sync ← candidate #2 |
| `train.optim_step` | 0.23 | 0.7% | gpu |

Training is **GPU-bound: 97.8% GPU / 2.2% CPU**. The only CPU/sync cost is the
matcher at 2.2%.

### semantic-eval (8-image rasterized DFC set, K=4) — wall 1.89 s (parent)

| Bucket | Seconds | % of total | Kind |
| --- | --- | --- | --- |
| `semantic_eval.forward` | 1.749 | **92.4%** | gpu |
| `semantic_eval.upsample` | 0.032 | 1.7% | gpu |
| `semantic_eval.confusion` | 0.022 | 1.2% | cpu ← candidate #3 |
| `semantic_eval.transfer` | 0.012 | 0.6% | sync ← candidate #3 |

Candidate #3's target = `transfer` + `confusion` = **1.8%** of the semantic-eval
span (measured against `semantic_eval.forward` as the GPU baseline). K=4 here;
the confusion/`bincount` cost scales with K and pixel count, so the share grows
with K — but `forward` scales too, and at K=4 the per-image host confusion matrix
is firmly below threshold. (Semantic data was synthesized by rasterizing DFC
instance masks into PNG label maps; mIoU is meaningless — only the timing
profile is used.)

---

## §3 — Candidate verdicts

### #4 — Mask upsample → host binarize transfer · **FILE** (primary)

`eval.transfer_binarize` = **14.6%** of full-eval wall. Microbench
(`scratch/proto_4_transfer.py`, synthetic masks at the real shapes) shows the span
is **pageable-PCIe-bound, not compute-bound**: threshold-only is 0.75 ms but
`(masks_up > thr).cpu().numpy()` is ~22 ms for M=100 × 1008².

| Variant (M=100, 1008²) | ms | speedup | exact? |
| --- | --- | --- | --- |
| baseline `(>thr).cpu().numpy()` | 22.20 | 1.00× | — (current) |
| `.contiguous()` | 23.09 | 0.96× | yes (neutral) |
| `.to(uint8)` | 45.17 | 0.49× | yes (**pessimizes**) |
| **pinned buffer + `non_blocking`** | **2.68** | **8.27×** | **yes (bit-identical)** |

At M=200 the pinned copy is 7.2×. **Achievable EXACT win:** an ~8× cut on a
14.6%-of-wall bucket recovers **~12–13% of total exact-eval wall, with
bit-identical masks** (asserted in the prototype) — no metric perturbation, no
faithfulness gate needed for the exact pinned-copy variant. The spec's perturbing
low-res variant is **not** needed and is not pursued. Implementation note for the
filed issue: real masks vary in M and native HxW, so production wants a sized
pinned-buffer pool rather than a per-call `pin_memory=True` alloc.

### #6 — Forward levers (`torch.compile` / `channels_last` / CUDA-graph) · **FILE** (folded into #4 issue)

`train.forward` is 33% of train wall / `eval.forward` is 15–20% — a real ceiling.
Time-boxed spike (`scratch/proto_6_forward.py`, real model + adapter):

| Lever | forward ms | result |
| --- | --- | --- |
| eager (baseline) | 318.0 | — |
| `channels_last` | 375.5 | **0.85× — pessimization** |
| `torch.compile(reduce-overhead)` | — | **blocked** (fails against the monkeypatch stack) |

Neither cheap lever helps: `channels_last` is slower on this model, and
`torch.compile` does not survive the dtype/RoPE/attention monkeypatch stack.
A real win needs CUDA-graph capture and/or making the patched modules
compile-safe, interacting with bf16 autocast (`train/loop.py`) and the VRAM
K-autosize OOM ladder (#203/#204). **Folded into #4's issue as a secondary lever
flagged for a dedicated deeper spike** (per spec §7: #6 "may spawn its own
dedicated spike rather than resolve here").

### #1 — Batch GT-side RLE encode · **RETIRE**

`eval.gt_rle_encode` = **0.7%** of full-eval wall (one-time GT build, amortized
over 355 images). The pred-side encode is already batched (#257). Below threshold.

### #2 — HungarianMatcher single-sync · **RETIRE**

`train.matcher` = **2.2%** of train wall. Below threshold, and the candidate's
premise (collapse *B* per-image `linear_sum_assignment` syncs → 1) only bites at
batch_size > 1 — the validated regime is **batch_size=1**, where there is exactly
one matcher sync per step with nothing to collapse. Training is 97.8% GPU; the
matcher is not the lever.

### #3 — Semantic confusion matrix on GPU · **RETIRE**

`semantic_eval.transfer` + `semantic_eval.confusion` = **1.8%** of the
semantic-eval span at K=4; `semantic_eval.forward` is 92%. Below threshold.
**Caveat:** the host confusion/`bincount` scales with K (and pixels); a re-measure
is warranted if a semantic config with large K (≥16) becomes the working regime.

### #5 — Per-example IoU for viz sample-picking · **RETIRE**

`eval.pair_iou` = **1.1%** of even the tiny viz run. It is request-gated
(visualize / lite per-example-IoU only) and ranks *which examples get
visualized* — it never touches reported metrics. Below threshold; not worth a
shared GPU kernel.

---

## §4 — Method / tooling notes (feed #256 + future audits)

1. **`*.total` denominator generalization.** `cli/profile_cmd.py` special-cases
   the literal `eval.total` as the wall denominator, so it mislabels
   `semantic_eval.total` as a summed leaf. The #256 attribution reader generalizes
   to any `.total` suffix and attributes the semantic surface correctly. Worth a
   one-line `profile_cmd` follow-up to match.
2. **GO/NO-GO heuristic is lever-aware, not state-aware.** The reader flags the
   top-100 query-filter as **GO** on eval-full (RLE dominant, N=200>100) — but
   that filter is *already applied* (#257). The heuristic correctly identifies the
   lever; it cannot know it is already pulled. Read GO verdicts as "this lever is
   relevant," then check whether it is already implemented.
3. **`train.matcher` is not train-exclusive.** It fired (1.2%) in the
   `eval-pairiou` run because the visualization path pairs predictions to GT via
   the same `HungarianMatcher`. The bucket measures matcher cost wherever it runs;
   on the train surface it is the candidate-#2 figure.
4. **`limit.val` does not gate eval.** Use an explicit small `data.val` COCO (or
   accept the full val split) for box-constrained full-eval profiling.

---

## §5 — Acceptance (spec §8)

- [x] §3b profiling instrumentation landed (8 buckets + 4 meta keys, CPU unit
      tests, analysis-only) — commit `4427fd8`.
- [x] #256 attribution tool delivered (reader + report + 60 CPU tests;
      regression-compare path covered) — commit `6a8c863`.
- [x] Checkpoint validated end-to-end (eval-full ran clean on the
      recently-hand-fixed config; recorded in §1).
- [x] Profiler run on all four surfaces with the small DFC subset; snapshots
      captured and run through the attribution tool (§2).
- [x] All six candidates re-ranked against the post-#276 per-surface distribution;
      RLE-adjacent #1/#4/#5 explicitly re-evaluated with the lite-vs-full split
      noted (§2, §3).
- [x] Each candidate filed (#4 + #6 folded) or retired with the killing
      measurement (§3).
- [x] Triage doc committed; no optimization code landed under #273.
