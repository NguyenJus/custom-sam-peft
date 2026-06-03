# Eval Performance Spike Design — Issue #250

**Status:** Approved (2026-06-02)
**Type:** Research spike (labels: `research`, `performance`, `priority:medium`)
**Scope:** Profile and attribute eval wall-time, then land two mAP-exact
speedups in `src/custom_sam_peft/eval/`. Two phases: a profiling deliverable
(Phase 1) gates an implementation (Phase 2). No new permanent surfaces.

---

## 1. Goals & Scope

Eval runs **2–3× slower per step than train**, which is backwards. A train
step is forward + backward + optimizer (~3× a forward); an eval step is
forward-only, so it should be ~2–3× *faster*. The gap is overhead, not
inherent cost.

Issue #250's body lists three suspected causes "reasoned from code, not
measured." Tracing the code corrects the diagnosis substantially (Section 3).
The spike's job is to **profile to attribute eval time, then take the
now-confirmed-free wins.**

| Goal | Deliverable |
|---|---|
| Attribute eval wall-time across forward / mask-upsample / device→host-transfer+binarize / RLE-encode / COCO-aggregate, on real hardware, with measured N and forwards-per-image | A committed attribution report (Phase 1) |
| Land the two confirmed-free speedups (#2a top-100 filter, #3 batched transfer/RLE) with a proof the reported metric is unchanged | Code + tests (Phase 2) |
| Document the CC 7.5 (T4) eval-precision reality | A finding in the report (Phase 1) |

**Not in scope** (see Section 9): aggressive sub-100 filtering (#2b),
adding autocast to eval, changing the per-class-group multiplex forward, and
any permanent `--profile` feature.

---

## 2. Architectural Approach

**Measure first, then optimize.** The spike is two phases with an explicit
decision gate between them:

- **Phase 1 — Profiling & attribution.** Add *temporary*, CUDA-synchronized
  timers around the eval buckets, run on real hardware against a
  representative eval config, and write a committed attribution report. This
  is the research deliverable.
- **Phase 2 — mAP-exact speedups.** Consume Phase 1's numbers to confirm the
  levers, implement the top-100 filter and batched transfer/RLE, remove the
  Phase-1 instrumentation, and re-profile to quantify the realized win.

The boundary discipline of the existing eval subsystem (spec
`docs/superpowers/specs/2026-05-17-eval-design.md`) is preserved: all tensor →
COCO conversion stays in `postprocess.py`; `evaluator.py` only orchestrates;
metric math stays in `metrics.py`.

---

## 3. Corrected Diagnosis

This section is the spec's centerpiece. Each suspected cause was traced
against live code; the anchors below were verified in this worktree.

| # | Issue's claim | Verdict | Action |
|---|---|---|---|
| 1 | Eval runs fp32 (~2× the forward) | **MISREAD** — eval already runs bf16 | Verify in profiling; delete the misleading `dtype=torch.float32` Runtime label (cosmetic) |
| 1b | (implicit) bf16 is faithful on all hardware | **UNVERIFIED on CC 7.5** | Measure & document eval-forward dtype on a sub-Ampere card |
| 2 | Postprocess runs on all N queries | **REAL** — reframed into a free top-100 filter | Filter to top-100 by score before upsample/transfer/RLE (mAP-exact) |
| 3 | Per-query `.cpu()` sync + RLE loop | **REAL** | Batch the device→host transfer + RLE over survivors (bitwise-identical) |
| 4 | One model call per class-group per image | **INHERENT** — train multiplexes the same way | Measure & report forwards-per-image; do **not** optimize |

### 3.1 Cause #1 — "eval runs fp32": MISREAD. Eval already runs bf16

- Model weights are cast to bf16 at load. `_apply_dtype`
  (`src/custom_sam_peft/models/sam3.py:569`) does `model.to(dtype=torch.bfloat16)`
  when `cfg.model.dtype == "bfloat16"` — the default
  (`src/custom_sam_peft/config/schema.py:117`).
- The eval loop builds `Runtime(device=param_device, dtype=torch.float32)`
  (`src/custom_sam_peft/eval/evaluator.py:151`), but `to_device`
  (`src/custom_sam_peft/runtime/_device.py:18`) only does `obj.to(runtime.device)`
  — it moves the *device*, never casts dtype. So the `float32` label never
  reaches the forward math.
- Even if an fp32 input slipped in, it is re-cast to the bf16 param dtype: the
  channel-adapter cast (`src/custom_sam_peft/models/sam3.py:329`) and the
  generic `module_input_dtype` forward-pre-hook
  (`src/custom_sam_peft/models/_patches/module_input_dtype.py:40`) cast module
  inputs to `next(module.parameters()).dtype`.

**Conclusion:** the eval forward already runs in bf16 — same as training, and
arguably *purer* bf16, since eval has no autocast upcasting select ops to fp32.
There is **no slow fp32 forward to fix.**

**Action:** verify in profiling; delete the misleading `dtype=torch.float32`
Runtime label at `evaluator.py:151` (cosmetic, no behavior change).

### 3.2 Cause #1b — CC 7.5 / T4 precision-faithfulness check (REQUIRED)

This check is required by the user and must not be omitted.

- `_apply_dtype` casts weights to bf16 **unconditionally** — no capability
  coercion (`src/custom_sam_peft/models/sam3.py:569`).
- But training's autocast **does** coerce: `_autocast_ctx`
  (`src/custom_sam_peft/train/loop.py:200`) calls `coerce_dtype_for_capability`
  (`src/custom_sam_peft/runtime/_runtime.py:61`), which downgrades bf16 → fp16
  on compute capability < 8.0 (T4 is CC 7.5).
- So on a sub-Ampere card there is a potential **TRAIN (fp16-via-autocast) vs
  EVAL (bf16-via-weights, or degraded) precision divergence**, and bf16 may be
  non-native / emulated.

**Cross-link (crash risk, not just metric drift):** if eval precision ever
collapses to fp16 on such hardware, fp16's small dynamic range (max ~65504,
vs bf16/fp32's full range) can overflow to inf/NaN, and the postprocess
finite-guards **raise `RuntimeError`** on non-finite values
(`src/custom_sam_peft/eval/postprocess.py:88`, `:96`, `:105`) — i.e., a crash,
not just a metric shift.

**Requirement:** Phase 1 must **measure and document** the actual eval-forward
dtype on CC 7.5 (T4) — does it stay bf16, collapse to fp16, or run emulated? —
**confirm the finite-guards don't trip**, and surface any train/eval precision
divergence as a finding. The "bf16 is already on / faithful" claim is validated
only on the **RTX 5070 Ti** (CC 12.0, native bf16) and **must not be assumed to
generalize**. (Repo context: the `gpu_t4` tier was only ever validated on the
5070 Ti superset, not on real T4 hardware.)

### 3.3 Cause #2 — "postprocess on all N queries": REAL, reframed into a free win

- The COCO scorer never overrides maxDets. `compute_coco_map`
  (`src/custom_sam_peft/eval/metrics.py`) reads `precision[:, :, :, 0, -1]`
  (`src/custom_sam_peft/eval/metrics.py:72`) — the **last** maxDets slice — and
  `coco_eval.params.maxDets` is never set, so it stays at the pycocotools
  default `[1, 10, 100]`. The reported mAP / mAP_50 / mAP_75 therefore use the
  **top-100 detections by score, per (image, category)**. Inside pycocotools'
  `COCOeval.evaluateImg`, detections for an `(imgId, catId)` pair are sorted by
  score descending and truncated to `maxDet=100`.
- Meanwhile `queries_to_coco_results`
  (`src/custom_sam_peft/eval/postprocess.py:51`) postprocesses **every** one of
  the model's N queries ("All queries are returned; no filtering or NMS
  applied", docstring `postprocess.py:68`): it upsamples all N mask logits to
  original resolution (`postprocess.py:110`), binarizes + transfers to CPU
  (`postprocess.py:111`), and RLE-encodes each query in a Python loop
  (`postprocess.py:116-125`, via `_logits_to_rle`).
- So queries ranked 101…N are fully postprocessed and then **discarded** by the
  scorer. Pure waste.

**Action (#2a):** filter to the **top-100 by score per (image, category)**
*before* upsample / transfer / RLE. This is **mAP-exact** (zero metric change),
because the scorer already truncates to 100 — citation: pycocotools
`maxDets=100` semantics. Win magnitude = `(N − 100) / N`, where N is measured in
Phase 1.

**N is not pinned in the repo.** It is a SAM 3.1 internal: the real-model
integration test `tests/integration/test_load_sam31_real.py` only asserts that
the `(B, Q)` shape exists (`obj_logits.dim() == 2`) and the multiplex
`shape[0] == B*K`, **not** Q's value; stubs use N=4 as test scaffolding
(`tests/fixtures/tiny_sam3_stub.py:30`, `num_queries: int = 4`). So **N must be
measured in Phase 1** (from `pred_logits.shape[1]`). The filter is exact
regardless of N:

- **N > 100** → real win.
- **N ≤ 100** → harmless no-op (the survivor entries are identical), and the
  profile redirects effort to the transfer/RLE cost.

**Tie-handling (correctness subtlety).** Keep **all queries whose score ≥ the
100th-highest score** (a *threshold*, not exactly-100). This guarantees the
survivor set is a **superset** of whatever 100 detections `COCOeval` would pick
under its own tie-break, so the metric is provably unchanged even when scores
tie at the 100/101 boundary. The filter must rank by the **same score COCOeval
uses** — the emitted
`score = sigmoid(pred_logits) * sigmoid(presence_logit_dec)`
(`postprocess.py:85-87`).

**Derive the cap from `max(coco_eval.params.maxDets)`** rather than hardcoding
100, so the filter and the scorer cannot drift.

### 3.4 Cause #3 — "per-query `.cpu()` sync + RLE loop": REAL

- Per postprocess call (i.e., per image × class-group) there are **3
  device→host syncs**: the `(N, H, W)` bool mask transfer
  (`postprocess.py:111`), boxes (`postprocess.py:114`), and scores
  (`postprocess.py:115`); plus **N pycocotools RLE encodes in a Python loop**
  (`postprocess.py:116-125`). The repeated `.cpu()` calls serialize GPU and CPU
  so neither overlaps.

**Action (#3):** keep masks on GPU through the filter, then do a **single
batched device→host transfer** for survivors and **batch the RLE encode**.
**Bitwise-identical** results — pure perf, no metric change.

### 3.5 Issue's 4th cause — one model call per class-group per image

The forward loop runs `model(...)` per `(image-chunk, class-group)` pair, in
groups of `MULTIPLEX_CAP = 16` (`src/custom_sam_peft/eval/evaluator.py`, lines
~166–223). This is **inherent to multiplexing** — train multiplexes the same
way. Phase 1 will **measure and report forwards-per-image**, but the structure
is **not** optimized. Out of scope (Section 9).

---

## 4. Phase 1 — Profiling & Attribution

The research deliverable.

### 4.1 Instrumentation

- Add **temporary** CUDA-synchronized timers around each bucket: call
  `torch.cuda.synchronize()` around each segment and measure with
  `time.perf_counter`, accumulating across an eval run. Reference the existing
  timing style in `src/custom_sam_peft/predict/runner.py:395`
  (`t_start = time.perf_counter()`) and the existing `torch.cuda.synchronize()`
  use in `src/custom_sam_peft/cli/calibrate_cmd.py:140`.
- **Buckets:** forward / mask-upsample / device→host-transfer+binarize /
  RLE-encode / COCO-aggregate.
- **Also capture:** N (query count from `pred_logits.shape[1]`), `n_classes`,
  forwards-per-image, model-input image size, `original_hw`, mask-logit spatial
  size, `n_images`.
- **Gate the instrumentation behind a temporary flag/env** so normal runs pay
  nothing and it is cleanly removable. This is **spike-only** — do **not**
  design it as a permanent `--profile` feature (Section 9).

### 4.2 Run environment

- Run **locally on the RTX 5070 Ti** (sm_120, native bf16 — the real training
  hardware) against a **representative eval config**: a real run's eval — full
  mode, real val split, real image size.
- Use **GPU-test isolation** (the `scripts/run_gpu_tests.sh` pattern), **not**
  freeze-prone bare `pytest tests/` (a single-process real-model GPU run risks
  freezing the 16 GB box).
- **Separately** reason about / measure the **CC 7.5 path** per Cause #1b
  (Section 3.2).

### 4.3 Deliverable: attribution report

A committed markdown report. **Proposed path:**
`docs/research/2026-06-02-issue-250-eval-perf-attribution.md` (matches the
`docs/research/YYYY-MM-DD-issue-NNN-<slug>.md` convention of that directory).

The report contains:

- The **per-bucket time breakdown** (absolute + % of eval wall-time).
- The **measured N**.
- **Confirmation that bf16 is already on** (Cause #1).
- The **CC 7.5 precision finding** (Cause #1b) — measured eval-forward dtype,
  whether the finite-guards trip, and any train/eval divergence.
- **Forwards-per-image.**
- An **explicit GO / NO-GO per Phase-2 lever**.

### 4.4 Decision gate (between phases)

**Confirm postprocess dominates and N > 100 before building the top-100
filter.** If the profile surprises us — forward dominates, or N ≤ 100 — Phase 2
scope adjusts. This is a deliberate measure-then-decide gate; it is a point
where the **plan may need amendment** (the orchestrator escalates rather than
blindly proceeding — Section 8).

---

## 5. Phase 2 — mAP-Exact Speedups

Consumes Phase 1's report.

1. **#2a top-100 filter** (threshold/tie-safe, cap derived from
   `max(coco_eval.params.maxDets)`) in/around `src/custom_sam_peft/eval/postprocess.py`.
2. **#3 batched device→host transfer + batched RLE** over survivors.
3. **#1 cosmetic:** drop the misleading `dtype=torch.float32` Runtime label in
   `src/custom_sam_peft/eval/evaluator.py:151`.
4. **Remove the Phase-1 instrumentation.**
5. **Re-profile** to quantify the realized win; record before/after numbers in
   the report.

---

## 6. Correctness & Testing — "the score didn't move" proof

### 6.1 Unit (CPU, fast)

Mirror the existing postprocess/evaluator unit tests in
`tests/unit/test_evaluator.py` (or add a dedicated postprocess test):

- Construct **> 100 queries** with known distinct scores → assert the filter
  keeps exactly the top-100 (plus any boundary ties) and that the resulting
  **COCO entry set and computed mAP equal the unfiltered baseline.**
- Add a **tie-at-boundary** case (scores tied at the 100/101 boundary) → the
  ≥-threshold keeps the superset; entry set and mAP unchanged.

### 6.2 Real-run regression

On the representative eval, assert the **full `MetricsReport`** (overall mAP /
mAP_50 / mAP_75 + per-class AP) is **identical before vs after Phase 2** within
float tolerance — **ideally bit-exact for kept entries**. This is the proof the
optimization is free.

### 6.3 Edge cases

- **N ≤ 100** — no-op; identical entries.
- **N = 0** — already returns `[]` (`src/custom_sam_peft/eval/postprocess.py:81`).
- **Score ties straddling the 100/101 boundary** — the ≥-threshold keeps the
  superset, so the metric is unchanged.

---

## 7. Files Affected

| File | Rationale |
|---|---|
| `src/custom_sam_peft/eval/postprocess.py` | Top-100 filter + batched transfer/RLE |
| `src/custom_sam_peft/eval/evaluator.py` | Cosmetic dtype-label removal; thread the maxDets cap to postprocess |
| `src/custom_sam_peft/eval/metrics.py` | Expose the maxDets-cap source (so the filter derives it, not hardcodes) |
| Temporary Phase-1 timing hooks (postprocess + the evaluator loop) | Added Phase 1, **removed in Phase 2** |
| `tests/unit/test_evaluator.py` (or a new postprocess test) | Top-100 exactness + tie + edge cases |
| `docs/research/2026-06-02-issue-250-eval-perf-attribution.md` | The attribution report (Phase 1), updated with before/after (Phase 2) |

---

## 8. Phasing & Interface Contract

This spec yields a **2-phase plan**.

**Phase boundary.** Phase 1 **exposes** the measured attribution report
(per-bucket %, N, forwards-per-image, CC 7.5 finding) plus a **GO / NO-GO per
lever**. Phase 2 **consumes** those to confirm the levers, implement, and
quantify the realized win. If Phase 1 **invalidates a lever**, the orchestrator
**escalates** (plan amendment) rather than blindly proceeding.

---

## 9. Out of Scope (YAGNI)

- **No #2b** (sub-100 / aggressive score-threshold filtering): it **would move
  mAP** and need an accuracy gate + a cited threshold hyperparam. Dropped.
  Revivable as a follow-up issue **only if** Phase-1 numbers show top-100 RLE
  still dominates **and** there is measured accuracy headroom.
- **No autocast added to eval** (Cause #1 dissolved — Section 3.1).
- **No change** to the per-class-group multiplex forward structure (Cause #4 —
  Section 3.5).
- **Profiling instrumentation is spike-only** (temporary), removed in Phase 2 —
  per the user's explicit choice. Do **not** design it as a permanent
  `--profile` feature.

---

## 10. Open Questions / Risks

- **N is unknown until measured;** the top-100 win's magnitude is contingent on
  N > 100.
- **CC 7.5 eval precision is unverified;** the spec requires measuring it
  (Section 3.2).
- **The decision gate after Phase 1** means Phase 2's exact scope is
  measurement-contingent by design (Section 4.4).

---

## Appendix A — Verified Anchors

All `file:line` claims above were checked against this worktree's live source.
Minor notes where the original brief's line numbers were paraphrased rather
than exact:

- Cause #4's loop "`evaluator.py:166-223`" — the per-`(image-chunk,
  class-group)` forward loop spans roughly lines 166–223 of
  `src/custom_sam_peft/eval/evaluator.py`; `MULTIPLEX_CAP = 16` is imported from
  `src/custom_sam_peft/models/sam3.py` and used at `evaluator.py:158`.
- All other anchors (`sam3.py:569 / :329`, `schema.py:117`, `evaluator.py:151`,
  `_device.py:18`, `module_input_dtype.py:40`, `loop.py:200`, `_runtime.py:61`,
  `postprocess.py:51/:68/:81/:85-87/:88/:96/:105/:110/:111/:114/:115/:116-125`,
  `metrics.py:72`, `predict/runner.py:395`, `calibrate_cmd.py:140`) verified exact
  or within ±1 line drift.
