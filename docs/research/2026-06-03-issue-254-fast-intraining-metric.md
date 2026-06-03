# Fast In-Training Validation Metric — Issue #254

> Spike write-up for [issue #254](https://github.com/NguyenJus/custom-sam-peft/issues/254) — Date: 2026-06-03. Decision analysis on decoupling the in-training (lite) validation metric from the exact pycocotools COCO-mAP pipeline by computing a fast GPU dense-IoU AP proxy in-loop, while reserving exact COCO mAP for the final / standalone report.

## TL;DR

In-training (lite) validation today pays the full pycocotools COCO-mAP cost, whose dominant term is the serial single-threaded per-query RLE encode (70.7% of timed eval wall-time on the post-#250 baseline; 78.2% pre-#250). That exact value is wasted in-loop: the single `mAP` scalar lite eval produces drives only three control consumers — best-checkpoint selection, `ReduceLROnPlateau`, and early-stop — and every one is an ordering / threshold comparison against the metric's own history, never the absolute COCO number. So a **monotone-faithful** GPU dense-IoU AP proxy is sufficient in-loop, and the standalone / final report keeps exact COCO mAP. Because only the trainer forces `mode="lite"` and the standalone path runs the default `mode="full"`, an **implicit `lite=fast / full=exact` split needs no new config knob** — the correct outcome under the project priority (accuracy > simplicity >> speed). Recomputed self-consistently against the post-#250 baseline, the proxy is estimated to cut lite-validation timed wall-time to ~25–29% of today's (~3.4×–4.0×) and fits 16 GB with large headroom. No empirical proxy-vs-exact divergence was measured (no non-zero-mAP checkpoint available); the GO is conditional on a pre-enablement rank-correlation gate.

```text
PROXY_DENSE_IOU_INLOOP:   GO  (conditional on validation gate §8)
CONFIG_SURFACE:           implicit  lite=fast-dense-IoU / full=exact-COCO  (NO new knob)
FAITHFULNESS:             monotone-faithful (ordering), NOT bit-equivalent  — REASONED, not measured
EXPECTED_SPEEDUP:         ~3.4x conservative .. ~4.0x optimistic  (est., post-#250 baseline)
MEMORY/NUMERICS:          fits 16GB; matmul path ~8-25x cheaper than naive 4-D; masks bit-identical; fp32 accum
COLD_START_ZERO (#197):   inherited, not regressed, not fixed  (continuous-signal fix = separate decision)
VALIDATION_GATE:          Spearman rho >= 0.95 over ~5-8 non-zero-mAP checkpoints  (NOT run this spike)
```

## §1 — Context: three consumers, two contexts

The in-training `mAP` scalar produced by lite-mode eval drives exactly **three control consumers**, and every one is an ordering / threshold comparison against the metric's own history — not a consumer of the absolute COCO value:

- best-checkpoint selection — strict `metric > self._best_metric_value` (`train/trainer.py:402`),
- `ReduceLROnPlateau(mode="max", threshold=min_delta, threshold_mode="abs")` stepped with `scheduler.step(mAP)` (`train/trainer.py:84-93`, `train/ladder.py:82`),
- rung-2 early-stop, which increments `evals_without_improvement` under the same `mAP > best + min_delta` test (`train/ladder.py:70-96`).

The scalar is read at `train/trainer.py:360` as `report.overall.get(self._best_metric_key)` with `_best_metric_key="mAP"` (`train/trainer.py:204`). It is also logged verbatim to the tracker (`train/trainer.py:358`), but that is informational telemetry, not a control input.

The **two contexts** that consume the metric differ structurally:

- **In-training (lite).** `_eval_epoch` force-overrides the mode to `"lite"` for every periodic validation via `model_copy(update={"mode":"lite","save_predictions":False})` and runs `Evaluator(lite_cfg).evaluate` (`train/trainer.py:343,352,357`). This path writes no artifact — it is pure compute feeding the three consumers.
- **Standalone / final.** `run_eval` builds `Evaluator(eval_cfg)` from `cfg.eval` with **no mode override** (`eval/runner.py:152-173`); `EvalMode` defaults to `"full"` (`config/schema.py:636`). The eval CLI passes `cfg` straight through (`cli/eval_cmd.py:93-114`; its local `mode` is the progress-renderer mode, not `EvalMode`). The final close-out report is built in full mode (`train/close_out.py:72-76`).

Because `mode="lite"` is used **exclusively** in-training and nothing depends on lite producing a true COCO number, decoupling the in-loop metric from exact COCO mAP is safe: only the in-training signal switches to the proxy. The in-loop metric needs only to **preserve the cross-epoch ordering** of checkpoints (plus a stable absolute scale for the fixed `min_delta`; see §7). Bit-equivalence with pycocotools buys nothing the three consumers can use.

## §2 — COCOeval rules: required vs optional for a ranking signal

`compute_coco_map` builds `COCOeval(iouType="segm")`, overrides only `params.iouThrs` to `cfg.iou_thresholds`, and reads `precision[T,R,K,A,M]` at area=all (A=0) and maxDets=100 (M=-1) (`eval/metrics.py:78-81,87,93,98,106`). The `K` axis (index 2) is per-category, so the repo's `mAP` is the mean of **per-category** AP over the threshold sweep, with no-GT categories masked by the `> -1` filter (`eval/metrics.py:87-89`). The proxy must replicate the rules whose omission can **reorder checkpoints** and may skip the rules that only **shift the absolute AP** uniformly:

| COCOeval rule | Replicate? | Why | Citation |
| --- | --- | --- | --- |
| iouThrs sweep over `cfg.iou_thresholds`, mean over T | **REQUIRED** | proxy `mAP` is defined as this mean; the threshold set determines the score | `eval/metrics.py:80,87-89` |
| Per-category pooling across images: pool all dets for a category over the lite subset into ONE PR curve, then mean over categories | **REQUIRED** | COCO AP is per-category (K axis), NOT per-image-then-mean; a per-image mean reorders checkpoints under lite partial coverage | `eval/metrics.py:87,104-107` |
| Score-descending greedy match per (image, category); unmatched dets = FP, unmatched GT = FN | **REQUIRED** | FP/FN bookkeeping is the heart of AP; without the FP penalty the metric cannot punish flooding low-confidence masks, and ordering diverges | pycocotools via `eval/metrics.py:81` |
| Minus-one ("no-GT category") masking before the mean | **REQUIRED** | absent categories otherwise drag the mean toward 0; under lite partial-GT coverage (`lite_max_images=64`) this changes ranking | `eval/metrics.py:88,94,99,107-109` |
| `maxDets=100` cap per (image, category) | **REQUIRED — proxy must re-truncate** | postprocess keeps a *superset* (all queries scoring ≥ the 100th, M ≥ max_dets); the proxy must re-truncate to top-100 by score before matching, else it scores extra dets COCOeval drops | `eval/postprocess.py:95-105`; `eval/metrics.py:17-29` |
| area=all (index 0), maxDets=last only | OPTIONAL | repo never reads S/M/L or maxDets 1/10 slices | `eval/metrics.py:87,93,98,106` |
| 101-point recall interpolation; mergesort equal-score tie-break | OPTIONAL | for continuous `sigmoid(p_obj)*sigmoid(presence)` scores exact ties are ~never; trapezoidal PR-area shifts only the absolute number, monotone in the ≤100-det lite regime | `eval/metrics.py:83-89`; scores at `eval/postprocess.py:86-88` |
| iscrowd modified-IoU branch | OPTIONAL | repo GT is always `iscrowd=0` (hardcoded), so the branch never runs | `eval/evaluator.py:91` |

**Net:** a faithful proxy must (a) compute the dense pred×gt IoU matrix, (b) re-truncate survivors to top-100 by score, (c) do score-ordered greedy matching with the FP penalty, (d) **pool detections per category across the lite image subset into one PR curve**, (e) sweep the same `iou_thresholds`, and (f) mask no-GT categories before averaging. It may use trapezoidal PR-area, ignore area splits and maxDets 1/10, and ignore iscrowd. These optional skips shift only the absolute value, not ordering.

## §3 — Proposed fast path (GPU dense-IoU)

### §3.1 — Where the seam sits

The in-training eval loop calls `queries_to_coco_results` once per `(image, category-in-group)` row, not once per image (`eval/evaluator.py:210-227`). The post-#250 top-`max(maxDets)`=100 filter runs **before** upsample/RLE, capping survivors to `m <= ~100` (a superset of the exact top-100; see §2) and producing `masks_up` at the **original** `H×W` resolution (`eval/postprocess.py:101-129`; upsample at `33-40,129`). The binarized masks already exist on-device on the line that currently transfers them off the GPU: `masks_bin = (masks_up > mask_threshold).cpu().numpy()` (`eval/postprocess.py:130`) — the proxy taps `masks_up > mask_threshold` **before** the `.cpu()`. The fast path replaces everything downstream of that on-device boolean (RLE encode, transfer, COCOeval) with an on-GPU IoU matmul + AP sweep. GT masks are `(H, W)` bool tensors at original resolution on `Instance.mask` (`data/base.py:42`), stackable to `(M, H·W)`.

### §3.2 — Area-sum IoU via matmul (never the 4-D tensor)

Never materialize the naive `(m × M × H × W)` boolean intersection. Flatten and use a matmul, mirroring the semantics of the existing CPU/RLE precedent `mask_utils.iou` at `eval/evaluator.py:384-388`:

```python
pred_f = masks_bin.flatten(1).float()        # (m, H*W)  -- from masks_up > mask_threshold, on-device
gt_f   = gt_masks.flatten(1).float()         # (M, H*W)
inter  = pred_f @ gt_f.T                      # (m, M)
pred_a = pred_f.sum(1)                        # (m,)
gt_a   = gt_f.sum(1)                          # (M,)
union  = pred_a[:, None] + gt_a[None, :] - inter
iou    = inter / union.clamp(min=1)           # (m, M)
```

Only `(m, H·W)`, `(M, H·W)`, and `(m, M)` are ever materialized.

### §3.3 — AP-from-IoU math

On top of the IoU matrix, replicate the §2 REQUIRED rules. Scores already exist as `sigmoid(pred_logits.float()) * sigmoid(presence.float())` (`eval/postprocess.py:86-88`). The aggregation is **per category, pooled across the lite image subset** — not per image. For each category, collect every survivor's `(score, IoU-row)` across all lite images, re-truncate to the top-100 by score (§2), then for each IoU threshold `t` in `cfg.iou_thresholds`:

1. sort the pooled preds by score descending,
2. greedily match each pred to its highest-IoU unmatched GT with `IoU >= t` (one GT per pred); matched = TP, unmatched pred = FP, unmatched GT = FN. Matching runs **per image** (a det can only match GT in its own image) but TP/FP/FN accumulate into the category's pooled cumulative arrays,
3. sweep cumulative precision/recall over the pooled score order, take the PR-area (trapezoidal is acceptable; 101-pt is optional — §2).

Steps 1–3 are repeated **independently per IoU threshold `t`** — the matched set changes with `t` (a det matched at 0.5 may be unmatched at 0.9), so a single match must not be reused across thresholds. Then average the per-category AP over thresholds, mask out no-GT categories, and mean over categories → `mAP`. This reuses the matrix mechanics of `_compute_per_example_iou` but, critically, **adds** the score-ordered greedy matching, FP penalty, and per-category pooling it lacks (see §4.2).

## §4 — Faithfulness: proxy vs exact

### §4.1 — What faithfulness means here

Because the three in-loop consumers (§1) are ordering / threshold comparisons against the metric's own history, the proxy needs only to **preserve cross-epoch checkpoint ranking** — not reproduce the exact COCO value. The proxy MUST replicate the five ordering-critical COCOeval rules (§2: iouThr sweep, per-category pooling across images, score-ordered greedy matching with FP penalty, no-GT-category masking, maxDets=100 re-truncation) and MAY skip the three absolute-value-only rules.

### §4.2 — Why the existing precedent is not enough

`_compute_per_example_iou` (`eval/evaluator.py:351-404`) already builds an `(n_pred, n_gt)` IoU matrix per image — a real dense-IoU precedent — but it is **recall-only and per-image**: it takes the best-pred IoU per GT, thresholds, and means, ignoring false positives, score order, and per-category pooling entirely. Reusing it verbatim would **not be AP-faithful** and could rank a mask-flooding checkpoint above a precise one — an ordering inversion. The proxy reuses the matrix mechanics but must add §3.3's greedy matching, FP penalty, and per-category cross-image pooling.

### §4.3 — Cold-start-zero (load-bearing caveat)

COCO `mAP` sits at **exactly 0** until predictions clear IoU 0.5; a threshold-swept AP proxy **inherits this dead zone** verbatim. With `best` starting at `-inf`, the first eval makes it `0.0`, and while `mAP` stays at `0` the `mAP > best + min_delta` test never fires, so `evals_without_improvement` ticks up and can early-stop a healthy cold run (`train/ladder.py:48-96`, `config/schema.py:553-554`) — the documented #197 failure. This is **not a faithfulness defect**: proxy and exact behave identically here, so the proxy is a clean drop-in that does **not regress** behavior. It also does **not fix** #197. A continuous sub-0.5-IoU signal (e.g. mean best-per-GT IoU) would, but that is a behavior change beyond a speedup — keep it out of scope and flag it to the #197 thread.

### §4.4 — Honesty: reasoned, not measured

No proxy-vs-exact divergence was measured. No trained checkpoint with non-zero mAP is available, and full-dataset GPU evals are crash-risky on the 16 GB sm_120 box. All faithfulness claims above are **analytic**, derived from COCOeval semantics. The validation gate (§8) must clear before the proxy drives any control consumer.

## §5 — Expected speedup

Reasoned from the **post-#250 "after" column** bucket shares (`docs/research/2026-06-02-issue-250-eval-perf-attribution.md:185-196`); **not re-measured this spike**. The post-#250 baseline is the correct anchor because the proxy stacks on the already-landed #250 work (top-100 filter + batched transfer/RLE). Numbers marked *(est.)* are analytic projections.

### §5.1 — What the proxy eliminates vs keeps

Post-#250 timed shares (derived from doc `:185-191` "after" column, TOTAL 68.650 s):

| bucket | % of timed (post-#250) | proxy fate | why |
| --- | ---: | --- | --- |
| `forward` (GPU) | 19.8% | **kept** | model forward unchanged; proxy only swaps the metric backend |
| `mask_upsample` | 1.0% | **kept** | proxy still needs masks at `original_hw` to IoU against GT (`eval/postprocess.py:33-40,129`) |
| `transfer_binarize` | 8.5% | **partly kept** | binarize stays on-GPU; the `.cpu()` half (`eval/postprocess.py:130`) vanishes |
| `rle_encode` | **70.7%** | **eliminated** | no `mask_utils.encode`; IoU is a matmul (mirroring `eval/evaluator.py:388`) |
| `coco_aggregate` | 0.1% | **eliminated** | no `COCOeval(iouType="segm")` pass (`eval/metrics.py:78-81`) |

### §5.2 — Cost of the added GPU IoU + AP compute

Survivors are capped at `m <= ~100` (`eval/postprocess.py:101-105`), GT is `M ≈ tens`. The IoU matmul `(m, HW) @ (HW, M)` for `m=100, M=20, HW≈640·480` is `2·m·HW·M ≈ 1.2 GFLOP`, versus the SAM 3.1 multiplex forward at **~1.70 s/image** of GPU time (post-#250 `forward` 13.587 s / 8 images, doc `:187`). A ~1.2-GFLOP matmul plus the O(m·M)-per-threshold greedy match/PR sweep is **well under 1% of `forward`** *(est.)* — it does not move the projection at the reported precision.

### §5.3 — Speedup factor

| scenario | residual timed share | speedup factor |
| --- | ---: | ---: |
| conservative (whole `transfer_binarize` 8.5% kept) | 19.8 + 1.0 + 8.5 = **29.3%** | **~3.4×** *(est.)* |
| optimistic (transfer half of `transfer_binarize` dropped, ~4.3% kept) | 19.8 + 1.0 + 4.3 ≈ **25.0%** | **~4.0×** *(est.)* |

Net: lite-validation timed wall-time shrinks to roughly **25–29%** of today's → a **~3.4×–4.0× speedup** *(est.)*, with ~3.4× as the headline conservative figure. The proxy makes eval **GPU-forward-bound** rather than CPU-RLE-bound — the architectural floor.

### §5.4 — Per-epoch and full-run implication

Anchoring to the #250 post-baseline per-image timed cost of **8.58 s/image** (`docs/research/2026-06-02-issue-250-eval-perf-attribution.md:196`) — the correct baseline, since the proxy stacks on the already-landed top-100 + batched-RLE work — at `lite_max_images=64` on the COCO 80-class worst case (5 forwards/image):

| quantity | post-#250 baseline | dense-IoU proxy *(est.)* |
| --- | ---: | ---: |
| per-image timed | 8.58 s | ~2.1–2.5 s |
| per-lite-eval (64 img) | ~549 s (~9.2 min) | ~137–161 s (~2.3–2.7 min) |
| saved per lite eval | — | **~6.5–7 min** |

Over a 160-epoch SAMed run with one lite eval per epoch on this COCO-80 worst case, the whole validation budget drops from **~24.4 h → ~6.1–7.2 h** — **~17–18 h of wall-time returned per run** *(est.)*. (The earlier draft's "~24 min" run-budget figure was a minutes/hours scale error: 549 s/eval × 160 evals is ~24 *hours*, not minutes.) The total scales linearly with eval cadence and class count: COCO-80 is the 5-forwards/image worst case, while a 12-class dataset is ~5× lighter in absolute terms (1 forward/image), though the ~3.4×–4.0× *factor* holds regardless.

## §6 — Memory & numerical fit

### §6.1 — Operand sizes

The peak is per-`(image, category)`, never aggregated across 80 categories (`eval/evaluator.py:210-227`). Naive 4-D is `m·M·H·W` bytes (1 B/bool); the matmul path is `m·H·W·4 + M·H·W·4 + m·M·4` with **fp32 operands** (matching the §6.3 recommendation):

| Case | H×W | m | M | naive 4-D bool | matmul path (fp32) | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Representative COCO | 640×480 | 100 | 20 | ~614 MB | ~147 MB | ~4× cheaper |
| Square mid | 640×640 | 100 | 20 | ~819 MB | ~197 MB | ~4× cheaper |
| Worst-case COCO | 640×640 | 100 | 100 | ~4.1 GB | ~328 MB | ~12× cheaper |

All cases fit trivially in 16 GB (worst-case ~328 MB → ~50× headroom). If a pathological image ever exceeded budget, tile the matmul over the GT axis so only `(m, gt_chunk)` is materialized; the naive 4-D tensor is never built on any path. (With fp16 operands + fp32 accumulation the table roughly halves on the operand terms — ~74/98/164 MB, ~8–25× cheaper than naive — also fine; see §6.3.)

### §6.2 — Masks are bit-identical; divergence is AP-only

This is the load-bearing numerical claim. The exact path RLE-encodes exactly `masks_up > mask_threshold` (`eval/postprocess.py:130`); the proxy binarizes the **same** `masks_up` at the **same** `mask_threshold` after the **same** bilinear upsample to `original_hw`. RLE is a lossless run-length encoding of that grid, so `mask_utils.iou` on the RLE and the matmul IoU on the dense bool compute the same integer intersection/union. **Therefore the proxy's divergence from exact COCO mAP originates ONLY in the AP scoring rules (§2) — never from the masks.** This removes mask fidelity from the risk surface. (Caveat: the guarantee holds only if the proxy reuses the exact `masks_up` / `mask_threshold` / upsample path; binarizing at logit resolution would reintroduce mask-level divergence.)

### §6.3 — Numerics

- **Scores** are already fp32 before the sigmoids (`eval/postprocess.py:86-88`), so the score-ordering that drives greedy matching is unaffected by the model's bf16 forward dtype.
- **IoU matmul** must **accumulate in fp32**: for 0/1 masks `inter` and areas are integer counts, exact in fp32 up to `H·W ≈ 16.7M` pixels (far above COCO resolutions); fp16 *accumulation* would lose precision once a mask area exceeds ~2048 pixels. fp16 *operands* are exact for 0/1 values, so the safe choices are **(a) fp32 operands outright** (simplest; §6.1 table) or **(b) fp16 operands with fp32 accumulation** (cheaper memory). Either is fine; the requirement is fp32 *accumulation*.

## §7 — Config surface recommendation

**Option A — implicit split (RECOMMENDED): `lite` ⇒ fast dense-IoU proxy, `full` ⇒ exact COCO mAP, with no new config field or enum value.** This is sound because:

- the trainer already hard-codes `mode="lite"` for in-loop eval (`train/trainer.py:343,352`) and the standalone path runs default `mode="full"` (`eval/runner.py:152`, `config/schema.py:636`) — the axis the proxy needs already exists,
- a metric-backend branch already exists in spirit (`_aggregate_metrics` gates `include_per_class` on `mode=="full"`, `eval/evaluator.py:244-269`); the change is to also branch the now-unconditional `queries_to_coco_results` RLE hotspot (`eval/evaluator.py:219`) on mode,
- the in-loop contract is one key (`mAP`); `mAP_50/mAP_75/per_class` are optional in lite (per-class already empty there), and `log_scalars` logs whatever keys exist (`train/trainer.py:204,358,360,399-402`),
- nothing depends on lite producing true COCO mAP: periodic lite eval writes no artifact (`train/trainer.py:357`), and the final report is built in full mode (`train/close_out.py:72-76`).

**Option B — new enum value / metric-backend knob: REJECT.** It overloads an already-conflated `EvalMode` enum for a speed-only gain. Per the project priority (accuracy > simplicity >> speed), a speed-only benefit is a weak reason to add user-facing surface, and the implicit split achieves the same speedup with zero new knobs.

Option A has two distinct costs, which must not be conflated:

- **(a) Dashboard / comparability (documentation concern).** The in-training logged `mAP` curve switches to proxy units, so it is no longer literally comparable to the standalone exact report. Document the unit change (or log the proxy under a distinct telemetry key).
- **(b) `min_delta` scale transfer (CONTROL concern).** `min_delta=0.001` (`config/schema.py:553`) is tuned for COCO-mAP units and feeds BOTH the `ReduceLROnPlateau` threshold (`train/trainer.py:89`) AND the early-stop test (`train/ladder.py:70`). If the proxy's absolute scale differs from COCO mAP (it may, since absolute-value-only rules are skipped per §2), the same `min_delta` becomes a different fraction of the proxy's dynamic range — shifting plateau/early-stop firing. This is a control-behavior change, not a dashboard concern, and it is a **gating check** in §8.2 (step 4), not an optional recalibration.

## §8 — GO / NO-GO + recommended implementation plan & validation gate

```text
DECISION: GO  — implement the GPU dense-IoU proxy for in-loop (lite) validation,
                via an implicit lite=proxy / full=exact split (no new config knob),
                CONDITIONAL on clearing the §8.2 pre-enablement gate before the proxy
                drives best-checkpoint / ReduceLROnPlateau / early-stop in production.
```

GO is justified on both axes: faithfulness (the three consumers need only monotone ranking, and the proxy replicates the five ordering-critical COCOeval rules — §2, §4) and speed (~3.4×–4.0× est., reclaiming ~17–18 h per 160-epoch COCO-80 run — §5), at trivial memory cost with bit-identical masks (§6). The config recommendation honors accuracy > simplicity >> speed (§7).

### §8.1 — Implementation plan

1. **GPU IoU helper.** Add the flattened area-sum matmul (§3.2) operating on the on-device `masks_up > mask_threshold` boolean and stacked GT masks (`data/base.py:42`), fp32 accumulation. Reuse the matrix shape of `_compute_per_example_iou` but on-GPU.
2. **AP-from-IoU.** Implement re-truncation to top-100 by score, per-threshold-independent score-ordered greedy matching + FP/FN bookkeeping, **per-category pooling across the lite image subset** into one PR curve, threshold sweep, and no-GT-category masking (§3.3, §2 REQUIRED rules); trapezoidal PR-area is acceptable.
3. **Mode branch.** In `queries_to_coco_results` / `_iter_predictions` (`eval/evaluator.py:219`), branch on `mode`: `lite` ⇒ accumulate `(IoU-matrix, scores)` per `(image, category)` and aggregate to a proxy `mAP` by per-category pooling (decide aggregator placement: a lite-aggregator in the evaluator vs. raw-tensor return from postprocess); `full` ⇒ the existing RLE + COCOeval path unchanged.
4. **Telemetry.** Emit the proxy under `report.overall["mAP"]` so the three consumers are untouched; document the unit change (or a distinct key) per §7.

### §8.2 — Pre-enablement validation gate (NOT run this spike)

This gate runs **after** the §8.1 proxy is implemented (it requires proxy code to compute proxy `mAP`) and **before** the proxy is allowed to drive control consumers in production. It is therefore a pre-*enablement* gate, not a pre-*implementation* one. To run once a non-zero-mAP checkpoint exists (no checkpoint is available now, and heavy GPU evals are crash-risky on the 16 GB sm_120 box):

1. Take ≈5–8 checkpoints spanning a real trajectory (cold → converged).
2. For each, compute both exact pycocotools lite `mAP` and proxy `mAP` on the same lite val subset.
3. Compute **Spearman rank-correlation** across the sweep, **restricted to checkpoints with non-zero exact mAP** (the dead zone of §4.3 yields tied zeros in both series, where rank-correlation is degenerate; the consumers only act on the non-zero ordering region). Gate: **ρ ≥ 0.95**, with **no adjacent-checkpoint inversion within `min_delta`**.
4. **`min_delta` scale check (gating, per §7b).** Measure the proxy's absolute scale and dynamic range against exact COCO mAP. If `min_delta=0.001` (`config/schema.py:553`, `threshold_mode="abs"`, `train/trainer.py:89-92`) maps to a materially different fraction of the proxy's range, recalibrate the default before enabling the proxy as a control input — do not ship a silently rescaled plateau/early-stop sensitivity.
5. Re-profile the proxy path on real data to confirm the §5 speedup estimate, and capture a real-checkpoint memory profile to confirm §6.

The proxy must clear steps 3–4 before it is allowed to drive best-checkpoint selection / plateau / early-stop in production runs. The cold-start-zero interaction (#197) is explicitly **out of scope** for this proxy: it is inherited identically from exact COCO mAP and is a separate behavior-change decision.
