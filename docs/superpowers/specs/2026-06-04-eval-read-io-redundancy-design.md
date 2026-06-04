# Eval-read I/O redundancy: exact GT counts + eval-scoped read cache

Status: locked design, implementation-ready.
Issues: #315 (Problem A) + periodic-eval val re-decode (Problem B).

Eval is heavy on disk I/O. This spec removes two distinct redundant-read
problems, each caused by reading and decoding the same val/eval images more
often than necessary. The two parts are independent and can land in either
order, but are specified together because they share the same root cause and
the same affected modules.

## Problem

### Problem A — viz selection re-decodes the entire eval slice (#315)

With `eval.visualize: true` (the default; `config/schema.py:818`), the
visualization pass runs after metrics. Its sample selector `pick_samples`
(`src/custom_sam_peft/eval/visualize.py:75`) builds its candidate pool with:

```python
candidates = [i for i in range(len(per_example_iou)) if len(dataset[i].instances) > 0]
```

(`visualize.py:90`). This calls `dataset[i]` over the **entire** eval slice
purely to count GT instances, in order to pick `visualize_count` (default 10;
`config/schema.py:819`) images. Every concrete `__getitem__`
(`data/coco.py:376`, `data/hf.py:348`, `data/mask_png.py:121`) performs a full
disk read + image decode + mask decode + albumentations transform with **no
caching**. This is a second full decode pass: the metrics pass already
materialized every example once at `eval/evaluator.py:948`
(`examples = [dataset[i] for i in range(n)]`).

The redundant pass fires in standalone `csp eval`
(`eval/runner.py:208`, `eval/runner.py:262`) and at training close-out
(`train/close_out.py:111` → `:115`). `save_predictions` defaults `False`, so
`predictions.json` is not a contributing factor — the double decode is the
whole cost.

The `visualize_count` (10) selected images still decode once inside
`render_eval_pair` (`visualize.py:491`) — that decode is **necessary** (the
renderer needs pixels) and is **not** redundant. Only the full-slice count scan
is wasteful.

### Problem B — periodic validation re-decodes the same val images every epoch

The in-loop periodic validation runs at `train/trainer.py:378`:

```python
report = Evaluator(lite_cfg).evaluate(self.model, self.val_ds)
```

This fires every `eval_every` (defaults to `steps_per_epoch` → roughly once per
epoch → ~160 evals on a 160-epoch run). Each call materializes
`examples = [dataset[i] for i in range(n)]` at `evaluator.py:948` with
`n = min(cfg.lite_max_images, n_total)` (default `lite_max_images = 64`;
`config/schema.py:814`), re-reading and re-decoding the **same 64 val images**
from disk every epoch. The val set is fixed and eval transforms are
deterministic, so the disk read + decode is redundant across epochs — only the
forward pass legitimately changes between evals.

## Goals

- **Part 1**: eliminate the full-slice decode in viz selection (Problem A) by
  feeding `pick_samples` exact GT counts the evaluator already computed for
  free, keeping selection byte-identical to today.
- **Part 2**: eliminate the cross-epoch re-decode of the fixed val set
  (Problem B) with a bounded, eval-scoped image-read cache in the data I/O seam.

## Non-goals

- No change to which images get visualized, the ranking, or rendered output.
- No change to the metric values, the per-example IoU values, or any control
  signal (best-checkpoint selection, early stop).
- No caching of training reads or HF/Arrow reads (see scoping rationale below).
- No change to the close-out / standalone full-read pass beyond threading the
  new return element.

## Design — Part 1: exact GT counts from the evaluator

Chosen approach ("Approach 1"): return the GT counts the evaluator already has.

### Evaluator return shape

`Evaluator.evaluate(..., return_per_example_iou=True)` already materializes
`examples` at `evaluator.py:948`. Compute, for free:

```python
gt_counts = [len(ex.instances) for ex in examples]
```

and return it as a **third** tuple element from the
`return_per_example_iou=True` branch (today returns
`(report, per_example_iou)` at `evaluator.py:991`). The new return shape is:

```python
tuple[MetricsReport, list[float], list[int] | None]
```

Update:

- the `@overload` for `return_per_example_iou: Literal[True]`
  (`evaluator.py:882-890`) to the 3-tuple;
- the impl's union return annotation (`evaluator.py:892-898`) to
  `MetricsReport | tuple[MetricsReport, list[float], list[int] | None]`;
- the return statement at `evaluator.py:991` to include `gt_counts`.

`gt_counts[i]` is exactly `len(dataset[i].instances)` by construction (same
`examples` list the metrics pass built), so any selection driven by it is
byte-identical to the current decode-based filter.

### SemanticEvaluator — uniform shape, `None` counts

`SemanticEvaluator.evaluate` (`src/custom_sam_peft/eval/semantic_evaluator.py:263`)
has its own `return_per_example_iou` overloads (`semantic_evaluator.py:245-261`)
and returns `(report, per_example_ious)` at `semantic_evaluator.py:298`. Keep
the return shape **uniform** across both evaluators: semantic has no instance
concept, so it returns `gt_counts = None` (instance-viz selects nothing
meaningful for semantic; `None` routes the consumer to the existing fallback).
Update its overloads and impl to the 3-tuple with `None` as the third element:

```python
return report, per_example_ious, None
```

### `pick_samples` and `write_eval_visualizations` gain `gt_counts`

Both `pick_samples` (`visualize.py:75`) and `write_eval_visualizations`
(`visualize.py:445`) gain a parameter:

```python
gt_counts: Sequence[int] | None = None
```

Contract for the candidate filter in `pick_samples` (`visualize.py:90`):

- when `gt_counts` is provided, filter with `gt_counts[i] > 0` — **zero
  decode**;
- when `None`, fall back to the existing `len(dataset[i].instances) > 0`
  decode path (preserves semantic and any other caller).

`write_eval_visualizations` forwards `gt_counts` into its `pick_samples` call
(`visualize.py:463`). The selected `count` images still decode once via
`dataset[idx]` inside the render loop (`visualize.py:491`) — unchanged.

### Threading through call sites

The three call sites that request `return_per_example_iou=True` now unpack a
3-tuple and pass `gt_counts` into the corresponding
`write_eval_visualizations(...)`:

- `eval/runner.py:231` (unpack) → `_run_viz` →
  `write_eval_visualizations(...)` at `runner.py:208`. `_run_viz`
  (`runner.py:202`) gains a `gt_counts` parameter (or closes over it) so the
  count flows into the viz call.
- `eval/runner.py:262` (unpack) → the second viz path that also lands in
  `_run_viz`. Both runner unpack sites return the 3-tuple where they currently
  return `(report, per_example_iou)` (`runner.py:251` and the `runner.py:262`
  branch tail) — update those return statements to carry `gt_counts` if the
  runner's own signature propagates it; otherwise consume it locally for viz.
- `train/close_out.py:83-85` (unpack) → `write_eval_visualizations(...)` at
  `close_out.py:115`. Add `gt_counts=gt_counts` to that call.

Where a runner branch re-emits the per-example tuple to its own caller, keep the
3-tuple shape consistent so static typing holds; the `gt_counts` element is the
only addition.

## Design — Part 2: main-process, eval-scoped image-read cache

Add a bounded, long-lived image-read cache in `src/custom_sam_peft/data/io.py`,
consulted by `read_image` (`io.py:110`).

`read_image` is the disk-read seam for COCO and mask_png (`data/coco.py:250`,
`data/mask_png.py:127` both call `read_image`). `HFDataset` reads via Arrow mmap
through `_decode_image` (`data/hf.py:206`, called at `hf.py:354`) and never
touches `read_image`, so it is **intentionally not covered** — Arrow mmap reads
are already cheap.

### Cache mechanics

- **Key**: `(resolved_path_str, channels)` where `resolved_path_str =
  str(Path(path))` (the same normalization `read_image` already applies at
  `io.py:112`).
- **Value**: the decoded `uint8` ndarray with `flags.writeable = False` set
  before storing. This is a mutation canary: consumers only slice (→ view) and
  transform (→ copy via albumentations), never write in place. A read-only array
  makes any accidental in-place write raise immediately rather than corrupting a
  shared buffer.
- **Hit**: return the cached read-only array directly (no copy — callers must
  not mutate, enforced by the read-only flag).

### Activation via context manager

Provide a context manager, e.g. `cached_image_reads(maxsize)`, that toggles the
cache **active**. `read_image` consults and populates the cache only while
active; when inactive it behaves exactly as today (no consult, no populate).

The underlying store is **long-lived** (module-level, persists across calls), so
the hot ~64 val images stay resident across all ~160 periodic evals. The context
manager toggles active state only — it does **not** clear the store on exit.
Clearing per-eval would defeat cross-epoch reuse, which is the entire point.

Bound the store with an LRU; `maxsize = lite_max_images` (default 64). LRU
eviction keeps it bounded even if a larger index range ever passes through it.

Suggested shape (illustrative, not prescriptive):

```python
@contextmanager
def cached_image_reads(maxsize: int) -> Iterator[None]:
    # set module-active flag + ensure the LRU store's capacity is >= maxsize
    # yield
    # restore the active flag (do NOT clear the store)
```

`read_image` (`io.py:110`) gains a guarded fast path at the top: when active,
compute the key, return a cached read-only hit if present; otherwise decode via
the existing dispatch, set `writeable = False`, insert into the LRU, and return.

### Scope — wrap only periodic eval

Wrap **only** the periodic eval call at `train/trainer.py:378`:

```python
with cached_image_reads(maxsize=lite_cfg.lite_max_images):
    report = Evaluator(lite_cfg).evaluate(self.model, self.val_ds)
```

Do **not** wrap close-out (`train/close_out.py:83`) — it is a one-shot full read
with no repeat benefit, and wrapping it would cache the whole (potentially large)
eval set. Standalone `csp eval` is likewise unwrapped.

**Scoping rationale (state explicitly):** periodic eval runs synchronously in
the **main process** (no DataLoader workers). Training image reads happen only
in worker processes. Because the cache is activated only around the
main-process eval call — never during DataLoader iteration or across a worker
fork — training workers never see the active flag and never populate the cache.
This keeps the cache strictly an eval-read optimization with zero interaction
with the training read path.

### Memory bound

Worst case ~`maxsize` raw `uint8` images resident (~190 MB at 64 typical-size
images). Tiled/oversized images that invoke `read_image` with the same resolved
path across windows share **one** cache entry (bonus dedup across tile windows).
LRU eviction caps residency even if an index range larger than `maxsize` passes
through (e.g. if close-out were ever run under the context manager — which this
spec does not do).

## Affected call sites (files + functions to change)

Part 1:

- `src/custom_sam_peft/eval/evaluator.py` — `Evaluator.evaluate`: overload
  (`:882-890`), impl union return (`:892-898`), compute `gt_counts` from
  `examples` (`:948`), return 3-tuple (`:991`).
- `src/custom_sam_peft/eval/semantic_evaluator.py` —
  `SemanticEvaluator.evaluate`: overloads (`:245-261`), impl return
  (`:263`/`:298`) → 3-tuple with `None`.
- `src/custom_sam_peft/eval/visualize.py` — `pick_samples` (`:75`, filter at
  `:90`) and `write_eval_visualizations` (`:445`, `pick_samples` call at
  `:463`): add `gt_counts: Sequence[int] | None = None`.
- `src/custom_sam_peft/eval/runner.py` — unpack 3-tuple at `:231` and `:262`;
  thread `gt_counts` into `_run_viz` (`:202`) →
  `write_eval_visualizations(...)` (`:208`).
- `src/custom_sam_peft/train/close_out.py` — unpack 3-tuple at `:83-85`; pass
  `gt_counts=gt_counts` into `write_eval_visualizations(...)` (`:115`).

Part 2:

- `src/custom_sam_peft/data/io.py` — add the LRU store + `cached_image_reads`
  context manager; add the active-cache fast path to `read_image` (`:110`).
- `src/custom_sam_peft/train/trainer.py` — wrap the periodic eval call (`:378`)
  in `cached_image_reads(maxsize=lite_cfg.lite_max_images)`.

## Testing

### Part 1

- `gt_counts` returned by `evaluate(..., return_per_example_iou=True)` equals
  `[len(ex.instances) for ex in examples]` on a fixture dataset.
- `pick_samples(..., gt_counts=...)` returns indices **identical** to the
  decode-fallback path (`gt_counts=None`) on the same fixture.
- `gt_counts=None` exercises the fallback and matches current behavior.
- A spy / counter on `dataset.__getitem__` asserts it is **not** called over the
  full range during viz selection when `gt_counts` is provided — only the
  `count` selected images get decoded (via `render_eval_pair` at
  `visualize.py:491`).
- `SemanticEvaluator.evaluate(..., return_per_example_iou=True)` returns a
  3-tuple with `None` as the third element; the semantic viz path takes the
  `None` fallback without error.

### Part 2

- A counting fake `read_image` proves a second eval over the same indices, while
  the cache is active, triggers **zero** new reads.
- The store is **not** consulted or populated when inactive (reads outside the
  context manager always hit the decode path).
- The read-only flag rejects in-place mutation (writing to a returned array
  raises).
- The training read path is unaffected: reads issued while the cache is inactive
  behave exactly as before.
- Cross-eval persistence: reads on eval #2..N over the same indices are served
  from the cache (store survives context-manager exit).

## Verification gates

- Full pytest suite over the real CPU dirs:
  `unit/ config/ cli/ eval/ train/ predict/ integration/`.
- `ruff check`
- `ruff format --check`
- `mypy src/custom_sam_peft` — ndarray annotations need full type args
  (`np.ndarray[Any, Any]`), matching the existing `io.py` style.

## Out of scope

- Unbounded checkpoint accumulation (`train/trainer.py:528`, no rotation) —
  tracked as a separate issue; not addressed here.
- The 10 selected-image decodes in `render_eval_pair` (`visualize.py:491`) are
  necessary (pixels are required to render) and are **not** redundant — no
  change.
- HF/Arrow read caching — Arrow mmap is already cheap; `read_image` is not on
  the HF path.

## References

- Issue #315 — "Eval visualize=True triggers a redundant full-dataset decode
  pass (double disk I/O)" (Problem A).
- Problem B (periodic-eval val re-decode) — no standalone issue; specified here
  alongside #315 because both share the eval-read I/O root cause and the same
  modules.
