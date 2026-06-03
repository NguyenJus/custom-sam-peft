# Spec: #273 algorithmic + CUDA performance audit (+ #256 repeatable attribution workflow)

**Issues:** closes #273 (algo/CUDA triage record) **and** #256 (repeatable profiler-run
attribution workflow).
**Date:** 2026-06-03
**Type:** research / measurement protocol + analysis tooling. **No optimization code
lands under #273.** Analysis tooling lands as code: the #256 attribution reader (§3)
**and** permanent profiling instrumentation on the candidate spans (§3b). Both are
measurement infrastructure — not optimization — so neither violates the "no
optimization code" rule (same justification as #265's permanent `eval.dataset_load`
timer).

---

## 1. Background and corrected premises

#273 surveys the codebase for **algorithmic** (do less / batch calls / better
complexity) and **CUDA / GPU-resident** (cut host↔device syncs) performance
opportunities, as the complement to the #253/#266 multithreading audit (which retired
threading: pycocotools holds the GIL).

Two premises in the issue text are **wrong as written** and are corrected here:

1. **"GPU RLE change / PR #274"** — the issue's hard gate ("take all baselines AFTER the
   GPU RLE change merges") names PR #274. PR #274 is actually the prefetch **NO-GO**
   measurement (#265); it shipped no RLE work. The GPU change the gate refers to is
   **PR #276** (`feat(#269)`, merged 2026-06-03), which is **not a GPU RLE encoder** —
   it is a GPU dense-IoU **AP proxy** that *bypasses* pycocotools RLE **only on the
   `lite` (in-training) validation path**. The exact/`full` eval path is untouched and
   still calls `mask_utils.encode` (`eval/postprocess.py:147`, `eval/evaluator.py:50`).
2. **Bucket re-ranking is path-specific, not global.** After #276:
   - **lite eval** → the `rle_encode` bucket is gone (proxy, GPU-forward-bound).
     RLE-adjacent candidates (#1/#4/#5) **no longer apply on this path**.
   - **full/exact eval** → `rle_encode` still dominates (~61% pre-#276 attribution).
     #1/#4/#5 **still apply here**.

The worktree is branched fresh off `origin/main`, which includes #276 — so the gate
("baseline after the GPU change merges") **is satisfied now**. No further dependency
blocks the audit.

**Accuracy-first constraint** (project design priority): every candidate must preserve
metric correctness. mAP-exactness and bit-identical masks are non-negotiable; any change
that could perturb a metric carries a faithfulness gate before it ships.

---

## 2. Goals / non-goals

**Goals**
- Deliver #256's **repeatable attribution workflow**: a tool that turns a `CSP_PROFILE`
  JSON snapshot into a standard, ranked attribution report with minimal manual effort.
- Use that tool to profile the four surfaces and **re-rank #273's six candidates against
  the post-#276 distribution**.
- For each non-trivial candidate, estimate the **achievable** win (not just the ceiling)
  with a throwaway prototype and confirm exactness.
- File worthwhile candidates as their own implementation issues; retire the rest with the
  measurement that killed them.

**Non-goals**
- No optimization code lands under #273 (triage record only).
- No production enablement of any candidate.
- Not re-litigating the threading audit (#253/#266) — closed, threading is a dead end.

---

## 3. Deliverable 1 — #256 repeatable attribution workflow (lands as code)

A small attribution reader plus a report template. **Exact module home is decided in the
implementation plan** (candidate: `scripts/attribute_profile.py` as a standalone reader,
to avoid expanding the `csp` CLI surface). It consumes the harness's JSON snapshot
(emitted by `CSP_PROFILE=1` + `csp profile`, the permanent harness from #255/#263) and
produces:

- **Bucket ranking** by share of timed wall-time.
- **Dominant-path identification** and the **CPU-vs-GPU split**.
- **Structural facts**: N (queries/forward), forwards-per-image, forward dtype,
  image/mask sizes, image count.
- **Lever GO/NO-GO heuristics**: apply the documented rules established in #250 (e.g.
  "postprocess dominates AND N > 100 → top-100 filter is GO").
- **Regression detection**: diff against a stored baseline snapshot; flag buckets that
  grew.
- **Report skeleton**: emit a `docs/research/`-style attribution report pre-filled from
  the data.

Tested with unit tests over synthetic snapshot JSON (no GPU needed for the tool itself —
the tool is pure snapshot-in / report-out, so its tests run on the CPU suite). The four
real snapshots from §5 are the tool's first live consumer and exercise the
regression-compare path (surface-over-surface or against a stored baseline).

This tooling is **analysis**, not optimization — landing it does not violate #273's
"no optimization code" rule.

---

## 3b. Deliverable 1b — profiling instrumentation (lands as code)

**Amendment (2026-06-03).** The audit measures *per-candidate* bucket share, but
several candidate spans have **no permanent profiling bucket today** — a stock
`CSP_PROFILE` snapshot cannot attribute them. The semantic evaluator emits **zero**
buckets; the instance-eval and train surfaces lack candidate-granular buckets for
#1/#2/#5. Rather than measure these with throwaway/uncommitted timers, **add the
missing buckets to the permanent harness.** This is permanent measurement
infrastructure (analysis, not optimization — parallels #265's `eval.dataset_load`), so
it lands as code without violating #273's no-optimization rule, and it strengthens
#256's "repeatable attribution" goal: future audits get these buckets for free.

**Buckets added** (each a `with profiling.bucket(...)` around the named span):

| Bucket | Span | Class | Candidate |
|---|---|---|---|
| `eval.gt_rle_encode` | `_mask_to_rle` body (`evaluator.py:154`) — all GT-side COCO encode | CPU | #1 |
| `train.matcher` | `HungarianMatcher.__call__` per-image cost-build + `linear_sum_assignment(cost.cpu().numpy())` (`matching.py`) | CPU/sync | #2 |
| `semantic_eval.forward` | `model(...)` call in `_iter_semantic_predictions` | GPU | #3 (baseline) |
| `semantic_eval.upsample` | `F.interpolate` + `semantic_argmax` (on-device) | GPU | #3 (baseline) |
| `semantic_eval.transfer` | `pred/gt.cpu().numpy()` host move (`semantic_evaluator.py:195-196`) | sync | **#3** |
| `semantic_eval.confusion` | `bincount` + accumulate + per-example metrics (`:199-207`) | CPU | **#3** |
| `semantic_eval.total` | whole forward loop (parent / wall denominator) | — | #3 |
| `eval.pair_iou` | `mask_utils.iou` in `_compute_per_example_iou` (`evaluator.py:1080`) | CPU | #5 |

Semantic meta via `note()`/`incr()`: `n_images`, `K` (classes), `sem_forward_dtype`,
`semantic_eval.forwards`. **Candidate #3's target share** = `semantic_eval.transfer` +
`semantic_eval.confusion`, measured against `semantic_eval.forward` as the GPU
baseline. Already-covered spans need no new bucket: **#4** → existing
`eval.mask_upsample` + `eval.transfer_binarize`; **#6** → existing `train.forward`.

**Testing.** TDD against the existing convention (`tests/unit/test_profiling.py`,
`tests/cli/test_profile_surface.py`): enable the profiler, drive each span on tiny CPU
inputs (matcher fixtures from `test_matching.py`; `TinySam3Stub` + the synthetic
semantic dataset from `test_semantic_evaluator.py`; a hand-built mask for
`_mask_to_rle`), then assert the new bucket appears in `snapshot()`. All run on the CPU
suite — no GPU. The lint gate (ruff/ruff-format/mypy/markdownlint) applies.

---

## 4. Deliverable 2 — #273 triage record (no optimization code)

- A triage doc at `docs/research/2026-06-03-issue-273-algo-cuda-audit.md` (matches the
  existing `docs/research/2026-06-0x-issue-*.md` pattern). Contents: per-surface bucket
  tables (from Deliverable 1), per-candidate verdict (file `#NNN` / retire + the number),
  prototype results, and faithfulness notes.
- One spawned implementation issue per **surviving** candidate
  (`gh issue create --assignee @me --label performance --label algo|cuda --label
  priority:*`), each carrying its measured ceiling **and** achievable win, plus a
  faithfulness gate where the change could move a metric.
- **#273 and #256 both close** on this work.

---

## 5. Profiling protocol ("let data decide")

One broad pass with the permanent harness (`CSP_PROFILE=1` + `csp profile`, **JSON dump,
not print**) across the surfaces, on real-checkpoint GPU runs:

| Surface | Why | Candidates it informs |
|---|---|---|
| **eval-full** (exact COCO) | still RLE-bound | #1, #4, #5 |
| **eval-lite** (post-#276 proxy) | confirm RLE-free; #1/#4/#5 should drop out here | (negative control) |
| **train** (instance) | forward-dominated | #2, #6 |
| **semantic-eval** (+ quick semantic-train confirm) | per-image host confusion matrix | #3 |

**Precondition.** The candidate-granular and semantic buckets land first (§3b); this
pass consumes the permanent harness *including* those new buckets — no temporary timers.

**Dataset (small — required).** All runs use a small/capped **DataFusionContest** subset
(the `runs/.../subset.json` path), to fit the 16 GB box and keep each run short
(session-crash guard).

**Box-constraint guards** (from prior profiling experience — see memory
`reference_eval_profiling_gotchas`):
- Cap full-eval image count (full mode holds ~12 GB of predictions on the 16 GB box).
- Set `eval.visualize=False` (it defaults True and injects extra RLE + memory that skews
  attribution).
- Keep each run short; dump JSON, don't print.
- **`eval.pair_iou` is request-gated.** `_compute_per_example_iou` runs only when
  per-example IoU is requested (lite-val `return_per_example_iou` or viz sample-picking).
  The `visualize=False` attribution run will show `eval.pair_iou` ≈ 0 — that is correct,
  not a miss. Attribute #5 from a *separate*, tiny run that forces the per-example-IoU
  path (`return_per_example_iou=True` / `visualize=True`).

**Checkpoint policy.**
- Use the existing `runs/test-dfc-20260603-035435/adapter/` adapter if it loads cleanly.
  ⚠️ **Implementation must double-check it actually works** — its config was *recently
  hand-fixed for staleness*; verify it loads and runs end-to-end before trusting any
  profile taken with it.
- If the config is still stale: **do not add a loader shim/migration** (per the
  no-legacy-config rule — stale `runs/` configs are gitignored and deliberately
  rejected). Instead **train one epoch on the small subset** to produce a fresh adapter.
- The fresh worktree lacks `runs/`; the execution session points at the main checkout's
  adapter or trains anew.

**GPU gate.** The profiling pass, the one-epoch fallback, and the prototypes are **GPU
runs on the real SAM3.1 checkpoint**. Per project policy, the execution session **asks
before kicking off each GPU run** — it does not launch them autonomously.

---

## 6. Triage rubric (attribution + cheap prototype)

- **Threshold:** a candidate whose bucket holds **≥ 5% of its surface's timed wall-time**
  gets a prototype; below that it is **retired in the doc with the number**.
  `# tbd:` — 5% is a chosen cut; confirm or move during execution.
- **Above threshold:** a **throwaway** micro-benchmark (lives in a scratch dir on the
  worktree, **never committed to `src/`** — only its numbers land in the doc) estimates
  the *achievable* win and checks exactness.
- **File vs retire:** file an implementation issue when the achievable win is meaningful
  **and** the change is either mAP/mask-exact or has a viable faithfulness gate;
  otherwise retire with the measurement that killed it.

---

## 7. Candidate plan (re-ranked against post-#276 reality)

| # | Candidate | Class | Surface(s) | Prototype | Exactness |
|---|---|---|---|---|---|
| 1 | Batch GT-side RLE encode — mirror #257's pred-side batching (`evaluator.py:84` loop → one `mask_utils.encode` over a Fortran `(H,W,N)` stack) | algo | eval-full | batched `encode` vs per-instance loop | **exact** |
| 2 | HungarianMatcher: collapse B per-image `linear_sum_assignment(cost.cpu().numpy())` syncs → 1; batch L1/GIoU/Dice cost build (`models/matching.py:120-176`) | cuda+algo | train | single-sync vs per-image-sync matcher; verify identical assignment | 2a/2b **exact**; 2c (GPU auction/Sinkhorn approx) **perturbing → gate** |
| 3 | Semantic confusion matrix on GPU — `torch.bincount`/`scatter_add_` over valid pixels, transfer once (`semantic_evaluator.py:194-207`) | cuda | semantic-eval | GPU accumulation vs per-image numpy `bincount` | **exact** |
| 4 | Mask upsample→native-res→host binarize transfer (`postprocess.py:136-139`) | algo+cuda | eval-full | measure transfer bucket; test lower-res variant exactness | low-res variant **perturbing → hard gate**; fuse + single contiguous copy **exact** |
| 5 | Per-example IoU for **viz sample-picking** (`evaluator.py:408` `mask_utils.iou` per image) → reuse #276's `eval/proxy_map.py::dense_iou_matrix` | cuda | eval-full | shared GPU kernel vs pycocotools `iou` | **looser bar** — only ranks which examples get *visualized*; does not touch reported metrics |
| 6 | Forward levers: `torch.compile` / `channels_last` / CUDA-graph on the PEFT forward (interacts with bf16 autocast `train/loop.py:215` and the VRAM K-autosize OOM ladder #203/#204) | cuda | train | **time-boxed spike**, not a clean microbench (dynamic shapes can cause recompile thrash) | may spawn its own dedicated spike issue rather than resolve here |

**Instrumentation note (§3b):** #1/#2/#3/#5 are attributed via buckets newly added to
the permanent harness; #4/#6 use existing buckets. `eval.pair_iou` (#5) is
request-gated — see §5.

Out of scope / already tracked (do not duplicate): #266/#253 (threading), #269/#276
(lite proxy — landed), #265 (prefetch — NO-GO), #260 (full-mode native-res memory),
#259 (predict double-forward), #252 (semantic train G× memory).

---

## 8. Acceptance

- [ ] Profiling instrumentation (§3b) landed: `eval.gt_rle_encode`, `train.matcher`,
      `semantic_eval.*`, `eval.pair_iou` buckets added to the permanent harness with CPU
      unit tests; no optimization behavior changed (analysis-only).
- [ ] #256 attribution tool delivered (reader + report template + CPU unit tests over
      synthetic snapshots); regression-compare path exercised.
- [ ] Checkpoint validated end-to-end (or a fresh one-epoch adapter trained); the
      validity double-check on the recently-fixed config is recorded.
- [ ] Profiler run on all four surfaces with the small DataFusionContest subset; snapshots
      captured and run through the attribution tool.
- [ ] All six candidates re-ranked against the post-#276 per-surface distribution; the
      RLE-adjacent ones (#1/#4/#5) explicitly re-evaluated, with the lite-vs-full split
      noted.
- [ ] Each candidate either filed as an implementation issue (with measured ceiling +
      achievable win + faithfulness gate where relevant) or retired with the measurement
      that killed it.
- [ ] Triage doc committed; #273 and #256 both closed; no optimization code landed under
      #273.

---

## 9. Open implementation decisions (deferred to the plan)

- Exact home of the #256 attribution reader (`scripts/` vs a module vs a `csp` subcommand).
- Final small-subset size / image cap for each surface (tune to the 16 GB box).
- Confirm or move the 5% triage threshold once the first real distribution is in hand.
- Whether `eval.pair_iou` (#5) is attributed from its own targeted per-example-IoU run
  or analytically (its bucket is request-gated; see §5).
- Semantic-eval bucket granularity (`forward`/`upsample`/`transfer`/`confusion`/`total`)
  is fixed in §3b; revisit only if a span needs finer split to attribute #3.
