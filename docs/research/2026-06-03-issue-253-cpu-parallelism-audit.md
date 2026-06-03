# CPU-Bound Parallelism Audit — Where Serial Work Remains After PR #257, and Whether Parallelizing Pays

> Research write-up for [issue #253](https://github.com/NguyenJus/custom-sam-peft/issues/253)
> Date: 2026-06-03
> Method: SPIKE (scope + de-risk, no implementation). Static audit grounded at HEAD
> `ebf0fba` with PR #257 merged, cross-checked against the compiled
> `pycocotools/_mask.abi3.so` symbol table (GIL microbench + `nm -D` probe) and the
> #250 attribution doc (`docs/research/2026-06-02-issue-250-eval-perf-attribution.md`).
> MEASURED facts and INFERRED estimates are tagged inline.

## TL;DR

The motivating bottleneck for this spike — the serial single-threaded per-query RLE
postprocess — was **already addressed by PR #257** (merged at `ebf0fba`): a mAP-exact
top-100 query filter plus a batched single-call `mask_utils.encode` over all survivors,
reducing eval timed wall-time from **113.97 s → 68.65 s (1.66×, MEASURED)**. This
spike confirms that and audits the residual CPU-parallelism surface on current main.

**The single highest-leverage remaining action** is adding bounded DataLoader prefetch
workers for the eval path (CONDITIONAL, ~1.1–1.4× end-to-end, memory-safe). Every other
candidate is NO-GO: the GIL (encode holds it — threads are useless) and the 16 GB WSL2
memory bound (process pools reproduce the documented swap → 100% HDD crash) close the
door on the deeper wins.

---

## §1 — Eval RLE postprocess: the batched-encode residual (post-#257)

**Serial today:** no longer the per-query loop. `postprocess.py:135–157` now performs
a single batched `mask_utils.encode` over a Fortran-order `(H, W, M)` uint8 array;
the `for i in range(m)` loop at `postprocess.py:144` only assembles output dicts — no
encode call inside. The per-query loop is gone.

**Wall-time dominance:** was dominant pre-#257 (`rle_encode` = 78.2%, MEASURED). After
the top-100 filter halved survivor count (128 k → 64.6 k predictions) and collapsed the
Python loop, the `rle_encode` bucket dropped from 89.100 s to 48.513 s (MEASURED,
`docs/research/2026-06-02-issue-250-eval-perf-attribution.md`, §8 Before/After table).
The batched call itself is one GIL-holding C invocation per `(image, class-group)` row.

**Speedup ceiling:** ~1.0× for any further loop collapse (MEASURED, near-zero). The
#250 attribution doc (§8 per-lever analysis) shows per-mask RLE cost essentially flat
before vs after batching: 89.100 s / 128 000 ≈ 696 µs/mask → 48.513 s / 64 635 ≈
750 µs/mask. `pycocotools mask_utils.encode` is **C-encode-bound by mask area, not
Python-loop bound.** The 1.66× gain came from the query-count filter (lever #2a), not
batching. Crediting further batching-style optimizations with that gain double-counts.

**Recommended approach:** none as a standalone. Batching is already merged; no further
loop-level cleanup remains.

**Memory guardrail:** the current batched path is low-risk (one transposed `(H, W, M)`
uint8 array, no process multiplier). The only path to real additional speedup —
process-pool fan-out over the `(image, class-group)` loop — is HIGH risk: full-mode
eval's predictions list holds ~12 GB on this box; pools multiply host RAM → the
documented swap → 100% HDD crash. No safe memory-bounded scope exists without a hard
worker cap + chunked pickle transfer of binarized (not logit) arrays — high cost for a
sub-1.3× remaining payoff.

**Determinism guardrail:** none for the current batched path — `mask_utils.encode`
returns RLEs in input order, `postprocess.py:144–156` reassembles in that order, and
`test_batched_rle_decodes_identically` locks in bit-identical output. A pool variant
would need an explicit re-sort by `(image_id, category_id, query_idx)` before `loadRes`
because COCOeval tie-breaking is resolution-order sensitive (#257 saw +635 boundary
ties).

**Verdict: NO-GO / low.** The dominant lever is merged. Residual is C-area-bound;
further parallelism needs a process-pool that is memory-unsafe on this box.

---

## §2 — Eval forward loop: process-pool fan-out over `(image, class-group)` rows

**Serial today:** yes. `_iter_predictions` in `evaluator.py:170–242` iterates strictly
serially over image chunks and class-group rows, calling `queries_to_coco_results` at
`evaluator.py:219–226` once per `(image, class-group)` pair. Three separate
GIL-held encode stages exist: prediction RLE (`postprocess.py:141`), GT RLE
(`evaluator.py:52–54` via `_mask_to_rle`, called from `evaluator.py:83`), and
COCOeval's own `computeIoU` (`metrics.py:45–47` via `COCOeval.evaluate/accumulate`).

**Wall-time dominance:** significant but no longer singularly dominant (INFERRED).
Post-#257, the filter halved the `rle_encode` share. GPU forward (~12.3%, MEASURED
pre-#257), transfer/binarize (~8.3%), and COCOeval now occupy a larger relative
fraction; no single serial CPU stage clearly dominates.

**Speedup ceiling:** ~1.3–1.8× of eval wall-time (INFERRED), Amdahl-bounded by
the now-significant non-encode fraction. Three caps apply: (1) the query-count lever
is already spent at 1.66× (MEASURED); (2) residual encode is C-area-bound; (3) the GT
RLE and COCOeval serial stages are unparallelized and would cap any Amdahl bound.

**Recommended approach:** process-pool is the only GIL-free option — but see verdict.

**Memory guardrail:** HIGH and deciding. Thread-pool is ruled out (GIL — see §Cross-cutting).
Each process-pool worker duplicates upsampled mask buffers (N × H × W float, pre-binarize)
plus SAM overhead. Mitigation requires hard worker caps (≤2), passing already-binarized
uint8 masks across the pickle boundary, and bounded chunking — high engineering cost for
sub-1.8× payoff on a 16 GB box that has crashed before under pool fan-out.

**Determinism guardrail:** MEDIUM. COCOeval tie-breaking is resolution-order sensitive;
pool results return out of completion order. An explicit re-sort by stable key
`(image_id, category_id, query_idx)` before `loadRes` (`metrics.py:78`) is required
for bit-exact mAP — not free.

**Verdict: NO-GO / low.** Thread-pool fails the GIL precondition. Process-pool is the
16 GB WSL2 swap hazard. The dominant lever is merged. Redirect cost attack to #254
(dense-IoU GPU proxy that removes RLE from the in-training metric path entirely).

---

## §3 — Predict path: batched-encode residual

**Serial today:** yes. `runner.py:486–492` calls `queries_to_coco_results` with
`max_dets=None` (no top-100 filter, by design — PNG/visualize need every query). With
#257 merged into HEAD, predict already inherits the batched single-call
`mask_utils.encode` via the shared `queries_to_coco_results` function
(`postprocess.py:135–157`). No per-query encode loop remains in the predict path.

**Wall-time dominance:** significant but unquantified (INFERRED from the 78.2%
pre-#257 eval figure; no measured predict-only profile exists on current main). The
predict path uses the identical function and does not apply the top-100 filter, so it
encodes all ~200 queries/forward. Per-mask cost is C-area-bound at ~696–750 µs/mask
(MEASURED in eval; predict masks are the same resolution).

**Speedup ceiling:** modest. With no Python loop remaining, the only speedup lever is
process-pool fan-out over the image loop — but that is Amdahl-capped because the
dominant cost is already one batched GIL-holding C call per forward, not a reducible
Python overhead.

**Recommended approach:** none. Inherit current batched-encode state and re-measure
predict wall-time on a representative predict run before scoping any pool.

**Memory guardrail:** thread-pool = low memory, ~0 speedup (GIL). Process-pool = HIGH
risk: each worker duplicates SAM 3.1 bf16 weights (~300–500 MB) plus intermediates; 4
workers ≈ 3–4 GB overhead → the documented swap → 100% HDD crash (NOT VRAM OOM).

**Determinism guardrail:** moot under batching. The batched encode preserves query
order natively; output JSON is byte-exact with no re-sort. A pool would reintroduce
completion-order nondeterminism requiring re-sort by `image_id` before extending
`all_predictions` (`runner.py:503`).

**Verdict: NO-GO (absent a post-measurement showing residual dominance) / low.**
Predict already inherits the batched encode. No meaningful parallelism lever remains
without a process-pool that is memory-unsafe on this box.

---

## §4 — Eval prefetch: serial dataset load before the forward loop

**Serial today:** yes. `evaluator.py:339` loads all eval examples synchronously:
`examples = [dataset[i] for i in range(n)]`. This is a single-threaded sequential
list comprehension — no DataLoader, no prefetch workers.

**Wall-time dominance:** moderate (INFERRED; no profiler bucket was assigned to the
prefetch phase in the #250 profile). For COCO 80 classes / 8 images the 40-forward
loop dominates, but on larger eval sets the prefetch proportion grows linearly with
image count.

**Speedup ceiling:** ~1.1–1.4× end-to-end (INFERRED), Amdahl-capped by the remaining
forward + encode cost. Down from the prior 2–4× estimate. Note: encode and COCOeval
`computeIoU` are GIL-held, so threads help nothing on those buckets (MEASURED:
4-thread encode microbench = 0.99×, 4-thread iou microbench = 1.00×). Only overlapping
I/O-bound prefetch with the GPU forward yields real gain.

**Recommended approach:** bounded DataLoader-style prefetch workers for the `examples`
load (e.g., a `ThreadPoolExecutor` over `dataset.__getitem__`, or restructuring the
evaluator to use a PyTorch DataLoader with `num_workers > 0`). Scope to prefetch only;
do not extend to RLE or COCOeval.

**Memory guardrail:** prefetch workers carry no mask-buffer multiplier — they load and
decode images, not float upsampled tensors. A conservative cap of ≤4 workers is safe
on the 16 GB box (each worker holds one decoded image at a time). The RLE process-pool
variant (§2) is out of scope here and remains NO-GO.

**Determinism guardrail:** low. mAP is order-invariant via COCOeval's internal score
sort (`metrics.py:79–81` loads `coco_dt` via `ground_truth.loadRes`, which sorts by
score). Prefetch reorders only input batches, not predictions.

**Verdict: CONDITIONAL / medium.** This is the one memory-safe, positive-payoff
parallelism lever found by this audit. Gate it on a profiler measurement confirming
prefetch is a non-trivial idle slice in the current main eval path.

---

## §5 — Training DataLoader configuration

**Serial today:** partial. The training DataLoader at `trainer.py:617–626` is already
configured with `num_workers=cfg.train.num_workers` (default `min(4, cpu_count)` from
`schema.py:623–625`), `pin_memory` on CUDA, and `persistent_workers`. Multiprocessing
is already on; the question is tuning.

**Wall-time dominance:** negligible. GPU forward/backward dominates training.
Collation is trivial (`collate.py` — `torch.stack` only). `batch_size=1` micro-batched
via `grad_accum_steps=8` means the GPU consumes prefetched batches fast; no evidence
the loader starves the GPU.

**Speedup ceiling:** ~1.0–1.2× if the GPU is I/O-bound (INFERRED); likely 1.0× since
it is compute-bound on the 5070 Ti.

**Recommended approach:** none. The 4-worker default is conservative but unvalidated
(`docs/defaults-provenance.md` marks `num_workers` as `# tbd: #191`). Retuning would
require proving GPU I/O-starvation first.

**Memory guardrail:** each worker forks dataset state (~5× footprint at 4 workers).
Conservative cap in place; no known crash from this path.

**Determinism guardrail:** none — seeded `worker_init_fn` (`trainer.py:625`); SGD is
order-independent per epoch.

**Verdict: NO-GO / low.** No evidence the loader starves the GPU; skip.

---

## §6 — torch intra-op + BLAS/OMP thread configuration

**Serial today:** partial. There is no explicit thread configuration anywhere in the
codebase — no `torch.set_num_threads`/`set_num_interop_threads` call in
`_bootstrap.py`, `cli/main.py`, conftest, or `run_gpu_tests.sh`; zero grep matches for
`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` across `src/`. torch
defaults to `os.cpu_count()` intra-op threads, so tensor ops already get default
parallelism.

**Wall-time dominance:** negligible. Torch ops are the ~12.3% forward slice (MEASURED);
the eval tensor work (`F.interpolate`, sigmoid in `postprocess.py:129–130`) already
benefits from torch ≥ 2.4 default intra-op threading.

**Speedup ceiling:** ~1.05× (INFERRED), Amdahl-bounded by the RLE encode share.

**Recommended approach:** none today. Thread config is absent and unnecessary for the
bottleneck.

**Memory guardrail:** currently no risk (no pool active). **Mandatory if any
process-pool lands elsewhere:** `torch.set_num_threads(1)` + `set_num_interop_threads(1)`
per worker to prevent oversubscription on the 16 GB box.

**Determinism guardrail:** negligible (intra-op threading does not reorder matrix op
outputs).

**Verdict: NO-GO / low.** Absent and unnecessary; revisit only as a bounding measure
if a process-pool ever lands.

---

## Cross-cutting constraints

### The GIL: encode holds it — threads are useless

`nm -D` on `.venv/lib/python3.12/site-packages/pycocotools/_mask.abi3.so` finds **no**
`PyEval_SaveThread`, `PyEval_RestoreThread`, or `Py_BEGIN_ALLOW_THREADS` symbols
(MEASURED: `nm -D` exits non-zero on those symbols; `strings` likewise finds none).
No `.pyx` or `maskApi.c` source ships with the installed package. If a `with nogil:`
block existed in the Cython source, the compiler would emit those libpython call sites;
their total absence is **strong inference** (INFERRED, high confidence) that
`mask_utils.encode` does not release the GIL.

A 4-thread microbench confirms this: 4-thread parallel encode yields **~0.99×**
(MEASURED) versus single-threaded — consistent with GIL serialization. The same holds
for `mask_utils.iou` (MEASURED: 4 threads = 1.00×).

Consequence: **Python threads cannot fan out the encode.** The only GIL-free option
is a process-pool, which is the WSL2 16 GB memory hazard (§2 / §3).

### The 16 GB memory bound: process pools reproduce the swap-crash

The 16 GB WSL2 box has a documented failure mode: process fan-out on full-mode eval
holds ~12 GB in the predictions list and adds per-worker buffer duplicates → swap fills
→ WSL2's spinning HDD saturates → session crash (NOT VRAM OOM). This has occurred
previously on broad `find`/`du` scans and full-dataset GPU eval fan-out. Any process-pool
recommendation for eval or predict must include a hard worker cap (≤2) + pre-binarized
uint8 arrays across the pickle boundary + chunked transfer — cost that is not justified
by the sub-1.8× payoff remaining post-#257.

---

## Prioritized follow-up issues

| # | Area (file:line) | Priority | Verdict | Approach | One-line payoff |
| --- | --- | --- | --- | --- | --- |
| 1 | Eval prefetch (`evaluator.py:339`) — tracked as [#265](https://github.com/NguyenJus/custom-sam-peft/issues/265) | medium | CONDITIONAL | Bounded prefetch workers (I/O overlap with GPU forward) | ~1.1–1.4× eval wall-time; memory-safe, the only positive-payoff lever |

**Closed / NO-GO:**

- **§1 batched-RLE residual** — already merged in #257; no further loop to collapse.
- **§2 eval process-pool fan-out** — GIL blocks threads; process-pool is the 16 GB
  swap hazard; dominant lever already spent. Redirect to #254 (GPU dense-IoU proxy).
- **§3 predict process-pool** — same GIL + memory constraints; predict already inherits
  batched encode from #257; no residual serial Python loop to attack.
- **§5 training DataLoader tuning** — GPU is compute-bound; no evidence of loader
  starvation.
- **§6 torch intra-op threads** — absent and unnecessary for the bottleneck; revisit
  only as a bounding measure if a process-pool lands.
