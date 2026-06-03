# Eval Performance Spike (#250): profile, then land two mAP-exact speedups — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute eval wall-time across forward / mask-upsample /
device→host-transfer+binarize / RLE-encode / COCO-aggregate on real hardware
(Phase 1), then — gated on that attribution — land two **mAP-exact** speedups
(a tie-safe top-100 query filter and batched device→host transfer + RLE) plus a
cosmetic dtype-label fix, with a proof the reported metric is unchanged (Phase 2).

**Architecture:** Two phases with an explicit decision gate between them. Phase 1
adds *temporary*, CUDA-synchronized bucket timers gated behind an env flag, runs a
representative eval on the RTX 5070 Ti, documents the CC 7.5 (T4) eval-precision
reality, and writes a committed attribution report. Phase 2 consumes ONLY the
report's per-bucket numbers + measured N + GO/NO-GO per lever; it filters queries
to the top-`max(maxDets)` by score *before* upsample/transfer/RLE (mAP-exact
because pycocotools already truncates to 100 per (image, category)), batches the
survivors' device→host transfer + RLE, drops the misleading `float32` Runtime
label, removes the Phase-1 instrumentation, and re-profiles. The eval subsystem's
boundary discipline is preserved: all tensor → COCO conversion stays in
`postprocess.py`; `evaluator.py` only orchestrates; metric math stays in
`metrics.py`.

**Tech Stack:** Python 3.12, PyTorch (CUDA, bf16), pycocotools (COCOeval segm,
RLE), pytest (CPU unit subset + GPU-isolated real-model run via
`scripts/run_gpu_tests.sh`), ruff.

---

## Phasing overview

Two sequential phases, each an independently-reviewable feature block. Phase 1 is
a research deliverable (instrumentation + a committed report); Phase 2 is the
code change (filter + batched transfer/RLE + cleanup + re-profile). A **decision
gate** sits between them (spec §4.4): Phase 2 only proceeds if the profile
confirms postprocess dominates AND measured N > 100. If the profile surprises
(forward dominates, or N ≤ 100), the orchestrator **escalates for a plan
amendment** rather than proceeding (spec §8).

- **Phase 1 — Profiling & attribution (research deliverable).** Temporary,
  CUDA-synchronized bucket timers + metadata capture gated behind a temporary env
  flag; a representative eval run on the 5070 Ti; the CC 7.5 eval-dtype finding;
  the committed attribution report with per-bucket %, measured N, bf16-already-on
  confirmation, forwards-per-image, and an explicit GO/NO-GO per Phase-2 lever.
- **Phase 2 — mAP-exact speedups (code).** The tie-safe top-`max(maxDets)` filter
  (cap derived from `max(coco_eval.params.maxDets)`, ranking by the emitted
  `score`); batched device→host transfer + batched RLE over survivors; the
  cosmetic `dtype=torch.float32` label removal; removal of the Phase-1
  instrumentation; a re-profile recorded as before/after in the report.

---

## Phase 1 → Phase 2 interface contract

> **Phase 1 EXPOSES (and Phase 2 CONSUMES ONLY) these — Phase 2 must NOT
> re-derive the profiling.** A later Phase-2 session reads this contract + the
> report; it does not re-read Phase-1 instrumentation code (which is removed in
> Phase 2 anyway).

The committed report `docs/research/2026-06-02-issue-250-eval-perf-attribution.md`
exposes, as machine-greppable values:

1. **Per-bucket breakdown** (absolute ms + % of eval wall-time) for the five
   buckets: `forward`, `mask_upsample`, `transfer_binarize`, `rle_encode`,
   `coco_aggregate`.
2. **Measured N** — the model's query count, read from `pred_logits.shape[1]` on
   the real model. (N is a SAM 3.1 internal, not pinned in the repo; stubs use
   N=4 — spec §3.3.)
3. **`bf16-already-on` confirmation** (Cause #1) — the eval forward runs bf16, not
   fp32; there is no slow fp32 forward to fix.
4. **CC 7.5 (T4) finding** (Cause #1b) — measured/reasoned eval-forward dtype on a
   sub-Ampere card, whether the postprocess finite-guards trip, any train/eval
   precision divergence.
5. **Forwards-per-image** (Cause #4) — reported, not optimized.
6. **GO / NO-GO per Phase-2 lever**, exactly:
   - `LEVER_2a_top100_filter: GO|NO-GO` (GO iff postprocess buckets dominate AND
     N > 100).
   - `LEVER_3_batched_transfer_rle: GO|NO-GO` (GO iff the transfer+RLE buckets are
     a non-trivial share, independent of N).
   - `LEVER_1_dtype_label_cosmetic: GO` (always — pure cosmetic, no measurement
     dependency).

**Decision-gate rule (spec §4.4 / §8):** Phase 2 Task 2.0 reads these six values
from the report. If `LEVER_2a_top100_filter` is NO-GO (forward dominates, or
N ≤ 100), the orchestrator **HALTS and escalates for a plan amendment** — it does
NOT silently skip or silently build the filter. `LEVER_1` and `LEVER_3` can each
proceed independently of `LEVER_2a` per their own GO flags.

---

## Conventions every phase follows

**CPU unit-test subset (bypass the global `--cov-fail-under=80` gate):**

```bash
uv run pytest tests/unit/test_eval_postprocess.py tests/unit/test_evaluator.py -o "addopts=" -q
```

The `-o "addopts="` clears the global pytest addopts (which include
`--cov-fail-under=80`); `--no-cov` does NOT work in this repo. Substitute the
task's test path. **Do NOT run `pytest --cov` locally — it segfaults torch's
C-extension import on this WSL2/sm_120 box; coverage is CI-only.**

**GPU run isolation.** Never run a bare `pytest tests/` here — a single-process
real-model GPU run risks freezing the 16 GB box. Phase-1 profiling and the
Phase-2 real-run regression run via the `scripts/run_gpu_tests.sh` per-file
isolation pattern (one pytest process per file; CUDA + host memory reclaimed at
process exit), or via a standalone one-file profiling script invoked once.

**Lint gate before every commit (CI runs both separately):**

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

If `ruff format --check` reports diffs, run `uv run ruff format src tests scripts`
and re-stage.

**Eager-import guard.** `src/custom_sam_peft/__init__.py` eagerly imports the
train chain. After any step that changes a public signature or removes a symbol
in `eval/`, verify the package still imports before claiming success:

```bash
uv run python -c "import custom_sam_peft"
uv run python -m py_compile src/custom_sam_peft/eval/postprocess.py src/custom_sam_peft/eval/evaluator.py src/custom_sam_peft/eval/metrics.py
```

**Blast-radius rule (signature changes).** `queries_to_coco_results` has THREE
production callers — `eval/evaluator.py:215`, `predict/runner.py:481`,
`eval/visualize.py:264` — plus unit-test callers in
`tests/unit/test_eval_postprocess.py` and the contract stub in
`tests/predict/test_runner_smoke.py`. Any signature change MUST: (a) be added as
an **optional keyword-only parameter with a default that preserves existing
behavior** (so predict/visualize, which legitimately need ALL queries, are
untouched); (b) be verified by grepping ALL callers; (c) be verified by running
the FULL relevant CPU test set (`test_eval_postprocess.py` + `test_evaluator.py`
+ `tests/predict/`), not just the named file, before "done".

**Cited-constants rule.** The only "constant" introduced is the top-detections
cap, and it is **DERIVED from `max(coco_eval.params.maxDets)`** (the pycocotools
default `[1, 10, 100]` → 100), NOT hardcoded. Deriving it from that source (with
the pycocotools citation in a comment) is mandatory. No other new hyperparameter
is introduced; a score *threshold* is the out-of-scope #2b (spec §9) and must be
rejected if a task is tempted to add one.

---

## Phase 1 — Profiling & attribution

**Goal:** Add temporary, CUDA-synchronized bucket timers + metadata capture gated
behind a temporary env flag, run a representative eval on the RTX 5070 Ti, reason
about/measure the CC 7.5 eval-forward dtype, and write the committed attribution
report exposing the Phase-1→Phase-2 contract values.

**Files:**

- Create (temporary, removed in Phase 2):
  `src/custom_sam_peft/eval/_profile.py` — the gated timer harness.
- Modify (temporary hooks, reverted in Phase 2):
  `src/custom_sam_peft/eval/evaluator.py` (wrap the forward + aggregate buckets),
  `src/custom_sam_peft/eval/postprocess.py` (wrap upsample / transfer+binarize /
  RLE buckets).
- Create: `scripts/profile_eval_250.py` — a one-file standalone runner that
  builds a representative eval and prints the bucket table + metadata.
- Create: `docs/research/2026-06-02-issue-250-eval-perf-attribution.md` — the
  report.

**Anchors (current line numbers):** `evaluator.py` forward call at `:180-183`,
postprocess call at `:215`, `_aggregate_metrics` at `:239`; `postprocess.py`
upsample at `:110`, binarize+`.cpu()` at `:111`, boxes/scores `.cpu()` at
`:114-115`, RLE loop at `:116-125`. `runtime/_runtime.py:61`
`coerce_dtype_for_capability` (cc<8.0 → fp16). Reference timing style:
`predict/runner.py:395` (`time.perf_counter()`) and
`cli/calibrate_cmd.py:140` (`torch.cuda.synchronize()`).

---

### Task 1.1: Temporary gated profiler module (`_profile.py`)

A self-contained, env-gated accumulator so the hooks in evaluator/postprocess are
one-line calls and the whole thing is a clean revert in Phase 2. Gated on
`CSP_EVAL_PROFILE=1`; when unset, every hook is a near-zero no-op (no
`torch.cuda.synchronize()`, no timing) so normal runs pay nothing.

**Files:**

- Create: `src/custom_sam_peft/eval/_profile.py`

- [ ] **Step 1: Write the profiler module**

```python
"""TEMPORARY eval profiler (issue #250, Phase 1). REMOVED in Phase 2.

Env-gated, CUDA-synchronized bucket timer + metadata capture. When
CSP_EVAL_PROFILE is unset/0, every public call is a no-op so normal eval runs
pay nothing. This is spike-only instrumentation — NOT a permanent --profile
feature (spec §9). All call sites and this file are reverted in Phase 2.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from collections.abc import Iterator

import torch

_ENABLED = os.environ.get("CSP_EVAL_PROFILE", "0") not in ("", "0", "false", "False")

# Bucket name -> accumulated seconds.
_BUCKETS: dict[str, float] = defaultdict(float)
# Free-form metadata (N, n_classes, image sizes, forwards count, ...).
_META: dict[str, object] = {}


def enabled() -> bool:
    return _ENABLED


@contextmanager
def bucket(name: str) -> Iterator[None]:
    """CUDA-synchronized timer for one bucket. No-op when disabled.

    Synchronizes BEFORE starting and BEFORE stopping so async CUDA kernels are
    attributed to the bucket that launched them, not the next one.
    """
    if not _ENABLED:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _BUCKETS[name] += time.perf_counter() - t0


def note(**kwargs: object) -> None:
    """Record metadata (last value wins per key). No-op when disabled."""
    if not _ENABLED:
        return
    _META.update(kwargs)


def incr(key: str, by: int = 1) -> None:
    """Increment an integer counter (e.g. forwards). No-op when disabled."""
    if not _ENABLED:
        return
    _META[key] = int(_META.get(key, 0)) + by  # type: ignore[arg-type]


def snapshot() -> tuple[dict[str, float], dict[str, object]]:
    """Return (buckets_seconds, metadata) copies for reporting."""
    return dict(_BUCKETS), dict(_META)


def reset() -> None:
    _BUCKETS.clear()
    _META.clear()
```

- [ ] **Step 2: Verify it imports and is a no-op when disabled**

```bash
uv run python -c "
import os; os.environ.pop('CSP_EVAL_PROFILE', None)
from custom_sam_peft.eval import _profile
assert _profile.enabled() is False
with _profile.bucket('x'):
    pass
_profile.note(n=4); _profile.incr('forwards')
assert _profile.snapshot() == ({}, {})  # disabled -> nothing recorded
print('OK no-op')
"
```

Expected: `OK no-op`.

- [ ] **Step 3: Verify it records when enabled**

```bash
CSP_EVAL_PROFILE=1 uv run python -c "
from custom_sam_peft.eval import _profile
with _profile.bucket('forward'):
    pass
_profile.note(N=999); _profile.incr('forwards', by=3)
b, m = _profile.snapshot()
assert 'forward' in b and m['N'] == 999 and m['forwards'] == 3
print('OK enabled')
"
```

Expected: `OK enabled`.

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/_profile.py
uv run ruff format --check src/custom_sam_peft/eval/_profile.py
git add src/custom_sam_peft/eval/_profile.py
git commit -m "spike(#250): temporary env-gated eval profiler (Phase 1; removed in Phase 2)"
```

---

### Task 1.2: Wire bucket timers into postprocess (upsample / transfer+binarize / RLE)

Wrap the three postprocess buckets and record N. These hooks call `_profile`,
which is a no-op unless `CSP_EVAL_PROFILE=1`, so default behavior is unchanged.

**Files:**

- Modify: `src/custom_sam_peft/eval/postprocess.py:103-126`

- [ ] **Step 1: Add the import**

At the top of `postprocess.py`, after the existing `from torch import Tensor`:

```python
from custom_sam_peft.eval import _profile  # TEMP #250 Phase 1 — removed in Phase 2
```

- [ ] **Step 2: Record N and wrap the upsample bucket**

Replace the `# --- masks ---` block (`postprocess.py:103-111`):

```python
    # --- masks ---
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (N, H, W)
    masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (N, H, W) bool
```

with (record N + mask-logit spatial size; wrap upsample and transfer+binarize as
SEPARATE buckets — `.cpu()` is the device→host sync, kept distinct from upsample):

```python
    # --- masks ---
    _profile.note(N=int(n), mask_logit_hw=tuple(pred_masks.shape[-2:]))  # TEMP #250
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    with _profile.bucket("mask_upsample"):  # TEMP #250
        masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (N, H, W)
    with _profile.bucket("transfer_binarize"):  # TEMP #250
        masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (N, H, W) bool
```

- [ ] **Step 3: Wrap the boxes/scores transfer + RLE loop**

Replace the entries-building block (`postprocess.py:113-126`):

```python
    entries: list[dict[str, object]] = []
    boxes_list = boxes_xywh.cpu().tolist()
    scores_list = scores.cpu().tolist()
    for i in range(n):
        entries.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(v) for v in boxes_list[i]],
                "score": float(scores_list[i]),
                "segmentation": _logits_to_rle(masks_bin[i]),
            }
        )
    return entries
```

with (fold the box/score `.cpu()` syncs into the transfer bucket; isolate the RLE
loop in its own bucket):

```python
    entries: list[dict[str, object]] = []
    with _profile.bucket("transfer_binarize"):  # TEMP #250 (box/score device->host)
        boxes_list = boxes_xywh.cpu().tolist()
        scores_list = scores.cpu().tolist()
    with _profile.bucket("rle_encode"):  # TEMP #250
        for i in range(n):
            entries.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in boxes_list[i]],
                    "score": float(scores_list[i]),
                    "segmentation": _logits_to_rle(masks_bin[i]),
                }
            )
    return entries
```

- [ ] **Step 4: Verify default behavior unchanged (profiler off)**

```bash
uv run pytest tests/unit/test_eval_postprocess.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
```

Expected: all PASS (the hooks are no-ops with `CSP_EVAL_PROFILE` unset).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/postprocess.py
uv run ruff format --check src/custom_sam_peft/eval/postprocess.py
git add src/custom_sam_peft/eval/postprocess.py
git commit -m "spike(#250): postprocess bucket timers + N capture (Phase 1; removed in Phase 2)"
```

---

### Task 1.3: Wire bucket timers into the evaluator (forward / aggregate / metadata)

Wrap the forward call and the COCO-aggregate call, and capture forwards-per-image
metadata + the eval-forward dtype.

**Files:**

- Modify: `src/custom_sam_peft/eval/evaluator.py:24-25` (import),
  `:180-183` (forward), `:206-222` (per-row meta), `:239-264` (aggregate).

- [ ] **Step 1: Add the import**

After the existing `from custom_sam_peft.eval.postprocess import ...` line:

```python
from custom_sam_peft.eval import _profile  # TEMP #250 Phase 1 — removed in Phase 2
```

- [ ] **Step 2: Wrap the forward call + count forwards + record dtype**

Replace the `try:` forward block (`evaluator.py:179-183`):

```python
                        try:
                            outputs = cast(
                                "dict[str, torch.Tensor]",
                                model(images_t, prompts_g, support=None),
                            )
```

with:

```python
                        try:
                            with _profile.bucket("forward"):  # TEMP #250
                                outputs = cast(
                                    "dict[str, torch.Tensor]",
                                    model(images_t, prompts_g, support=None),
                                )
                            _profile.incr("forwards")  # TEMP #250
                            _profile.note(  # TEMP #250 — eval-forward dtype + sizes
                                eval_forward_dtype=str(
                                    outputs["pred_masks"].dtype
                                    if isinstance(outputs.get("pred_masks"), torch.Tensor)
                                    else "unknown"
                                ),
                                n_classes=int(n_classes),
                                model_input_hw=tuple(images_t.shape[-2:]),
                            )
```

- [ ] **Step 3: Record n_images and wrap the aggregate bucket**

In `_aggregate_metrics` (`evaluator.py:239-264`), wrap the `compute_coco_map`
call. Replace:

```python
        report = compute_coco_map(
            predictions=predictions,
            ground_truth=gt,
            iou_thresholds=cfg.iou_thresholds,
            include_per_class=(cfg.mode == "full"),
        )
```

with:

```python
        _profile.note(n_images=len(gt.imgs))  # TEMP #250
        with _profile.bucket("coco_aggregate"):  # TEMP #250
            report = compute_coco_map(
                predictions=predictions,
                ground_truth=gt,
                iou_thresholds=cfg.iou_thresholds,
                include_per_class=(cfg.mode == "full"),
            )
```

- [ ] **Step 4: Verify default behavior unchanged (profiler off)**

```bash
uv run pytest tests/unit/test_evaluator.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
uv run python -m py_compile src/custom_sam_peft/eval/evaluator.py
```

Expected: all PASS (hooks are no-ops with `CSP_EVAL_PROFILE` unset).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/evaluator.py
uv run ruff format --check src/custom_sam_peft/eval/evaluator.py
git add src/custom_sam_peft/eval/evaluator.py
git commit -m "spike(#250): evaluator forward/aggregate timers + forwards/dtype meta (Phase 1; removed Phase 2)"
```

---

### Task 1.4: Standalone profiling runner script

A one-file runner that builds a representative eval (full mode, real val split,
real image size), runs `Evaluator.evaluate` once with `CSP_EVAL_PROFILE=1`, and
prints the bucket table + metadata. Standalone (NOT a pytest test) so it runs in
its own process — the GPU-isolation discipline (one real-model load per process).

**Files:**

- Create: `scripts/profile_eval_250.py`

- [ ] **Step 1: Write the runner**

```python
#!/usr/bin/env python
"""TEMPORARY standalone eval profiler driver (issue #250, Phase 1).

Runs ONE representative eval with CSP_EVAL_PROFILE=1 and prints the per-bucket
breakdown + metadata for the attribution report. Removed in Phase 2.

Usage (on the RTX 5070 Ti, in its own process for GPU memory isolation):
    CSP_EVAL_PROFILE=1 uv run python scripts/profile_eval_250.py --config <eval.yaml> [--checkpoint <ckpt>]

The --config must point at a real eval config (full mode, real val split, real
image size) — a representative run, not a stub.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    if os.environ.get("CSP_EVAL_PROFILE", "0") in ("", "0"):
        raise SystemExit("set CSP_EVAL_PROFILE=1 before running this profiler")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    args = ap.parse_args()

    from custom_sam_peft.config.loader import load_config
    from custom_sam_peft.eval import _profile
    from custom_sam_peft.eval.runner import run_eval

    _profile.reset()
    cfg = load_config(args.config)
    run_eval(cfg, checkpoint=args.checkpoint, split=args.split)

    buckets, meta = _profile.snapshot()
    total = sum(buckets.values()) or 1.0
    print("\n=== issue #250 eval profile ===")
    print(f"metadata: {meta}")
    print(f"{'bucket':<22}{'seconds':>12}{'% of timed':>14}")
    order = [
        "forward",
        "mask_upsample",
        "transfer_binarize",
        "rle_encode",
        "coco_aggregate",
    ]
    for name in order:
        s = buckets.get(name, 0.0)
        print(f"{name:<22}{s:>12.4f}{100 * s / total:>13.1f}%")
    print(f"{'TOTAL(timed)':<22}{total:>12.4f}{100.0:>13.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses + imports (no GPU needed for the smoke check)**

```bash
uv run python scripts/profile_eval_250.py --help
```

Expected: argparse help text, exit 0. (Running for real requires the GPU + a real
config — done in Task 1.5.)

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check scripts/profile_eval_250.py
uv run ruff format --check scripts/profile_eval_250.py
git add scripts/profile_eval_250.py
git commit -m "spike(#250): standalone eval profiling runner (Phase 1; removed in Phase 2)"
```

---

### Task 1.5: Run the profile on the RTX 5070 Ti + reason about CC 7.5

**GPU-REQUIRED** for the 5070 Ti run. CPU-reason the CC 7.5 path (no T4 hardware;
the repo's `gpu_t4` tier was only ever validated on the 5070 Ti superset).

**Files:** none modified — this task produces the raw numbers for the report.

- [ ] **Step 1: Identify a representative eval config**

Use a real eval config (full mode, real val split, real image size — spec §4.2),
e.g. a config under `src/custom_sam_peft/cli/templates/config_full.yaml` adapted
to a local dataset, or an existing run's config. Confirm `eval.mode: full`. If no
real dataset is locally available, the orchestrator escalates (the profile MUST be
a representative run, not a stub) per spec §4.2.

- [ ] **Step 2: Run the profiler in its own process**

```bash
CSP_EVAL_PROFILE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/profile_eval_250.py --config <representative-eval.yaml>
```

Capture the printed bucket table + metadata (N, n_classes, forwards,
model_input_hw, original_hw via mask sizes, mask_logit_hw, n_images,
eval_forward_dtype). Record the raw output for the report.

- [ ] **Step 3: Confirm bf16-already-on (Cause #1)**

From the captured `eval_forward_dtype`, confirm the eval forward outputs are bf16
(not fp32). Cross-check the trace: weights cast to bf16 at load
(`models/sam3.py:569`); the `Runtime(dtype=torch.float32)` label at
`evaluator.py:151` is never applied to forward math (`to_device` moves device
only — `runtime/_device.py:18`). Note the confirmation for the report.

- [ ] **Step 4: Establish the CC 7.5 (T4) eval-dtype finding (spec §3.2 — REQUIRED)**

This finding is required by the user. Without T4 hardware, reason it from code and
state it explicitly:

- `_apply_dtype` (`models/sam3.py:569`) casts weights to bf16 **unconditionally**
  — no capability coercion. So at load, eval weights are bf16 even on CC 7.5.
- Training's autocast **does** coerce: `_autocast_ctx` (`train/loop.py:200`) →
  `coerce_dtype_for_capability` (`runtime/_runtime.py:61`) downgrades bf16 → fp16
  on cc < 8.0. Eval has NO autocast, so eval weights stay bf16 on CC 7.5 — but
  bf16 on sub-Ampere is **non-native / emulated**, and there is a potential
  **TRAIN (fp16-via-autocast) vs EVAL (bf16-via-weights) precision divergence**.
- Crash-risk cross-link: if eval precision ever collapses to fp16, fp16's ~65504
  max can overflow to inf/NaN, and the postprocess finite-guards
  (`postprocess.py:88, :96, :105`) **raise RuntimeError** — a crash, not just
  drift. Note whether the guards could trip on CC 7.5 given the above.
- State explicitly: the "bf16 is already on / faithful" claim is validated only on
  the **RTX 5070 Ti (CC 12.0, native bf16)** and **must not be assumed to
  generalize** to CC 7.5.

Record this as the report's CC 7.5 finding.

- [ ] **Step 5: Derive the GO/NO-GO per lever**

From the bucket table + measured N:

- `LEVER_2a_top100_filter` → **GO** iff the postprocess buckets
  (`mask_upsample` + `transfer_binarize` + `rle_encode`) dominate eval wall-time
  AND N > 100; else **NO-GO**.
- `LEVER_3_batched_transfer_rle` → **GO** iff `transfer_binarize` + `rle_encode`
  are a non-trivial share (independent of N); else **NO-GO**.
- `LEVER_1_dtype_label_cosmetic` → **GO** (always).

(No code in this task — these feed Task 1.6's report.)

---

### Task 1.6: Write the committed attribution report

**Files:**

- Create: `docs/research/2026-06-02-issue-250-eval-perf-attribution.md`

- [ ] **Step 1: Write the report**

Follow the `docs/research/YYYY-MM-DD-issue-NNN-<slug>.md` house style of the
existing reports in that directory. The report MUST contain, with the
contract values greppable (Phase 1→Phase 2 interface contract above):

1. **Run environment** — GPU (RTX 5070 Ti, sm_120, native bf16), driver, the
   representative eval config + dataset + image size, commit SHA, date.
2. **Per-bucket breakdown** — a table of `forward / mask_upsample /
   transfer_binarize / rle_encode / coco_aggregate` with absolute ms + % of eval
   wall-time (from Task 1.5 Step 2).
3. **Measured N** — `N: <value>` (from `pred_logits.shape[1]`).
4. **bf16-already-on confirmation** (Cause #1) — eval forward dtype + the
   `evaluator.py:151` label-is-cosmetic note (from Task 1.5 Step 3).
5. **CC 7.5 (T4) finding** (Cause #1b) — the full §3.2 reasoning from Task 1.5
   Step 4: measured/reasoned eval dtype, finite-guard risk, train/eval divergence,
   the "validated only on 5070 Ti" caveat.
6. **Forwards-per-image** (Cause #4) — `forwards / n_images`; reported, NOT
   optimized (out of scope, spec §9).
7. **GO / NO-GO per lever** — the three greppable lines:

   ```text
   LEVER_2a_top100_filter: GO        # postprocess dominates AND N=<N> > 100
   LEVER_3_batched_transfer_rle: GO  # transfer+RLE = <pct>% of eval
   LEVER_1_dtype_label_cosmetic: GO  # pure cosmetic
   ```

   (Substitute the real verdicts/numbers; if NO-GO, state the surprise.)
8. A **"Phase 2 plan"** section noting the report leaves a `Before/After`
   subsection to be filled by Phase 2 Task 2.6.

- [ ] **Step 2: Markdown-lint the report (CI lints tracked .md)**

Run the project's markdown linter (the exact tool CI's lint job uses — discover it
from the workflow; do not assume) and fix findings before committing. Reference:
the repo runs markdownlint-cli2 via uvx + nodejs-bin (no system node).

- [ ] **Step 3: Commit**

```bash
git add docs/research/2026-06-02-issue-250-eval-perf-attribution.md
git commit -m "docs(#250): eval perf attribution report (Phase 1 deliverable)"
```

---

### Phase 1 DECISION GATE (spec §4.4 / §8) — explicit checkpoint

> **This is a gate, not a footnote.** Before any Phase-2 work, the orchestrator
> reads the report's six contract values and applies the rule:

- [ ] Confirm `LEVER_2a_top100_filter`: **GO** requires postprocess buckets
  dominate AND measured **N > 100**.
- [ ] If `LEVER_2a` is **NO-GO** (forward dominates, OR N ≤ 100): **HALT and
  escalate for a plan amendment** (spec §8). Do NOT silently skip the filter task
  and do NOT silently build it against a profile that contradicts it. `LEVER_1`
  (cosmetic) and `LEVER_3` (batched transfer/RLE) may still proceed per their own
  GO flags.
- [ ] If `LEVER_2a` is **GO**: proceed to Phase 2.

---

## Phase 2 — mAP-exact speedups

**Goal:** Consume the Phase-1 report's contract values; implement the tie-safe
top-`max(maxDets)` filter (#2a), the batched device→host transfer + batched RLE
(#3), and the cosmetic dtype-label removal (#1); REMOVE the Phase-1
instrumentation; re-profile and record before/after in the report. mAP is
provably unchanged.

**Files:**

- Modify: `src/custom_sam_peft/eval/metrics.py` (expose the maxDets-cap source).
- Modify: `src/custom_sam_peft/eval/postprocess.py` (top-N filter + batched
  transfer/RLE; remove Phase-1 hooks).
- Modify: `src/custom_sam_peft/eval/evaluator.py` (thread the cap; drop the
  `float32` Runtime label; remove Phase-1 hooks).
- Delete: `src/custom_sam_peft/eval/_profile.py`, `scripts/profile_eval_250.py`.
- Modify: `tests/unit/test_eval_postprocess.py` (filter exactness + tie + edge
  cases).
- Modify: `docs/research/2026-06-02-issue-250-eval-perf-attribution.md`
  (before/after).

**Score-ranking fact (used throughout):** the filter MUST rank by the SAME score
COCOeval uses — the emitted
`score = sigmoid(pred_logits) * sigmoid(presence_logit_dec)` (`postprocess.py:85-87`).

---

### Task 2.0: Read the gate; expose the maxDets cap source in `metrics.py`

**Files:**

- Modify: `src/custom_sam_peft/eval/metrics.py`
- Test: a new CPU unit test (`tests/unit/test_eval_postprocess.py` or
  `tests/unit/test_metrics.py` if present — use whichever exists for metrics; this
  plan adds it to `test_eval_postprocess.py` for locality with the filter tests).

- [ ] **Step 0: Re-affirm the decision gate**

Read the report's `LEVER_2a_top100_filter` line. If NO-GO, STOP and escalate (see
the Phase-1 Decision Gate). Only proceed when GO.

- [ ] **Step 1: Write the failing test for the cap helper**

The cap is DERIVED from `max(coco_eval.params.maxDets)` (pycocotools default
`[1, 10, 100]` → 100), not hardcoded. Expose a tiny helper so the filter and the
scorer cite ONE source. Add to `tests/unit/test_eval_postprocess.py`:

```python
from custom_sam_peft.eval.metrics import coco_max_dets_cap


def test_coco_max_dets_cap_is_pycocotools_default_100():
    # pycocotools segm Params default maxDets == [1, 10, 100]; the scorer reads the
    # LAST (max) slice, so the cap the postprocess filter must match is 100.
    assert coco_max_dets_cap() == 100
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_eval_postprocess.py::test_coco_max_dets_cap_is_pycocotools_default_100 -o "addopts=" -q
```

Expected: FAIL — `coco_max_dets_cap` does not exist (ImportError).

- [ ] **Step 3: Implement `coco_max_dets_cap` in `metrics.py`**

Add at module level in `metrics.py` (after the imports):

```python
def coco_max_dets_cap() -> int:
    """The max detections-per-(image, category) the COCO scorer keeps.

    Derived from pycocotools' COCOeval params, NOT hardcoded, so the postprocess
    top-N filter and the scorer cannot drift. compute_coco_map never overrides
    maxDets, so it stays the pycocotools segm default [1, 10, 100]; the scorer
    reads the LAST (max) maxDets slice (see precision[..., -1] below), i.e. it
    keeps the top-100 detections by score per (image, category). Citation:
    pycocotools COCOeval / Params default maxDets = [1, 10, 100].
    """
    from pycocotools.cocoeval import Params

    return int(max(Params(iouType="segm").maxDets))
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/unit/test_eval_postprocess.py::test_coco_max_dets_cap_is_pycocotools_default_100 -o "addopts=" -q
```

Expected: PASS.

- [ ] **Step 5: Lint + import-guard + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/metrics.py tests/unit/test_eval_postprocess.py
uv run ruff format --check src/custom_sam_peft/eval/metrics.py tests/unit/test_eval_postprocess.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/eval/metrics.py tests/unit/test_eval_postprocess.py
git commit -m "feat(#250): expose coco_max_dets_cap() — derive top-N cap from pycocotools maxDets"
```

---

### Task 2.1: TDD the tie-safe top-N filter in `postprocess.py` (test-first)

The mAP-exactness safety net (spec §6) is written WITH the filter. The filter is
added as an **optional keyword-only `max_dets: int | None`** so the other two
callers (predict/visualize, which need ALL queries) keep `None` and are unchanged.
Tie-handling: keep **all queries whose score ≥ the `max_dets`-th-highest score**
(a threshold, not exactly-`max_dets`) so the survivor set is a SUPERSET of whatever
the scorer would pick under its own tie-break — provably mAP-unchanged.

**Files:**

- Modify: `src/custom_sam_peft/eval/postprocess.py`
- Test: `tests/unit/test_eval_postprocess.py`

- [ ] **Step 1: Write the failing filter unit tests (exactness + tie + edges)**

Add to `tests/unit/test_eval_postprocess.py`:

```python
def _outputs_with_scores(scores: list[float], h: int = 4, w: int = 4) -> dict[str, torch.Tensor]:
    # Encode target post-sigmoid scores via pred_logits with presence fixed so
    # sigmoid(presence)=1 (large positive). score = sigmoid(logit) * ~1.
    n = len(scores)
    logits = torch.tensor(
        [[[torch.logit(torch.tensor(min(max(s, 1e-6), 1 - 1e-6))).item()] for s in scores]]
    )
    return {
        "pred_logits": logits,  # (1, n, 1)
        "pred_boxes": torch.full((1, n, 4), 0.5),
        "pred_masks": torch.full((1, n, h, w), -10.0),
        "presence_logit_dec": torch.full((1, 1), 20.0),  # sigmoid≈1
    }


def test_filter_no_op_when_n_le_cap():
    # N=3 <= cap=100 -> identical entries to unfiltered.
    out = _outputs_with_scores([0.9, 0.5, 0.1])
    base = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))
    filt = queries_to_coco_results(
        out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100
    )
    assert filt == base


def test_filter_keeps_top_cap_by_score():
    # 105 queries, distinct descending scores; cap=100 -> exactly 100 survivors,
    # and they are the 100 highest scores.
    scores = [i / 200.0 for i in range(105, 0, -1)]  # 105 distinct, descending
    out = _outputs_with_scores(scores)
    filt = queries_to_coco_results(
        out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100
    )
    assert len(filt) == 100
    kept = sorted((e["score"] for e in filt), reverse=True)
    expected = sorted(scores, reverse=True)[:100]
    assert kept == pytest.approx(expected, abs=1e-5)


def test_filter_boundary_ties_keep_superset():
    # 102 queries; scores tie exactly at the cap boundary (positions 100,101,102
    # all equal). The >= threshold keeps the SUPERSET (all 3 tied), never < cap.
    scores = [0.9 - i * 0.001 for i in range(99)] + [0.3, 0.3, 0.3]  # 99 distinct + 3 ties
    out = _outputs_with_scores(scores)
    filt = queries_to_coco_results(
        out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100
    )
    # 99 above the tie + all 3 tied at 0.3 = 102 survivors (superset, not truncated to 100).
    assert len(filt) == 102


def test_filter_n_zero_returns_empty():
    out = {
        "pred_logits": torch.zeros(1, 0, 1),
        "pred_boxes": torch.zeros(1, 0, 4),
        "pred_masks": torch.zeros(1, 0, 4, 4),
        "presence_logit_dec": torch.zeros(1, 1),
    }
    assert (
        queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4), max_dets=100)
        == []
    )


def test_filter_none_is_no_filter():
    # max_dets=None (default) preserves the predict/visualize contract: ALL queries.
    scores = [i / 200.0 for i in range(105, 0, -1)]
    out = _outputs_with_scores(scores)
    entries = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))
    assert len(entries) == 105  # unfiltered
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/unit/test_eval_postprocess.py -k "filter" -o "addopts=" -q
```

Expected: FAIL — `queries_to_coco_results` has no `max_dets` parameter (TypeError).

- [ ] **Step 3: Add the `max_dets` parameter + threshold filter (rank by score)**

In `queries_to_coco_results` (`postprocess.py:51-57`), add the keyword-only
parameter:

```python
def queries_to_coco_results(
    outputs: dict[str, Tensor],
    image_id: int,
    category_id: int,
    original_hw: tuple[int, int],
    mask_threshold: float = 0.0,
    *,
    max_dets: int | None = None,
) -> list[dict[str, object]]:
```

Update the docstring's "All queries are returned" line to:

```python
    When ``max_dets`` is given, keep only queries whose score is >= the
    ``max_dets``-th-highest score (a threshold, NOT exactly ``max_dets`` —
    boundary ties are kept as a superset). This is mAP-EXACT: pycocotools'
    COCOeval already truncates to ``max(params.maxDets)`` (=100) detections by
    score per (image, category), so dropping the strictly-lower-scored remainder
    cannot change the metric. Citation: pycocotools maxDets=[1,10,100] semantics.
    ``max_dets=None`` (default) returns ALL queries unchanged (predict/visualize
    need every query).
```

Then, immediately AFTER the scores finite-guard (`postprocess.py:88-92`) and
BEFORE the boxes/masks work (so the expensive upsample/transfer/RLE only run on
survivors), insert the filter:

```python
    # --- top-N filter (mAP-exact; spec §3.3) ---
    # Rank by the SAME score COCOeval uses; keep all queries with score >= the
    # max_dets-th-highest score (threshold, not exactly max_dets) so the survivor
    # set is a SUPERSET of whatever 100 the scorer would keep under its own
    # tie-break. Done BEFORE upsample/transfer/RLE so those costs only touch
    # survivors.
    keep_idx: Tensor | None = None
    if max_dets is not None and n > max_dets:
        kth = torch.topk(scores, max_dets).values.min()  # the max_dets-th-highest score
        keep_idx = (scores >= kth).nonzero(as_tuple=False).squeeze(-1)  # (M,), M >= max_dets
        scores = scores[keep_idx]
```

And apply `keep_idx` to boxes and masks where they are computed. Replace the boxes
block (`postprocess.py:94-101`):

```python
    # --- boxes ---
    boxes_norm = pred_boxes.float().squeeze(0)  # (N, 4)
    if not torch.isfinite(boxes_norm).all():
        raise RuntimeError(
            "non-finite box coordinates in postprocess; check model outputs "
            "(pred_boxes contains NaN/Inf)"
        )
    boxes_xywh = _denorm_cxcywh_to_xywh(boxes_norm, original_hw)  # (N, 4)
```

with (slice survivors after the finite-guard so the guard still checks all
queries' validity):

```python
    # --- boxes ---
    boxes_norm = pred_boxes.float().squeeze(0)  # (N, 4)
    if not torch.isfinite(boxes_norm).all():
        raise RuntimeError(
            "non-finite box coordinates in postprocess; check model outputs "
            "(pred_boxes contains NaN/Inf)"
        )
    if keep_idx is not None:
        boxes_norm = boxes_norm[keep_idx]
    boxes_xywh = _denorm_cxcywh_to_xywh(boxes_norm, original_hw)  # (M, 4)
```

And the masks block. **Important:** after Task 1.2 this region is
**`_profile`-wrapped** — the actual current text is:

```python
    # --- masks ---
    _profile.note(N=int(n), mask_logit_hw=tuple(pred_masks.shape[-2:]))  # TEMP #250
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    with _profile.bucket("mask_upsample"):  # TEMP #250
        masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (N, H, W)
    with _profile.bucket("transfer_binarize"):  # TEMP #250
        masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (N, H, W) bool
```

Insert the survivor-slice between the finite-guard and the upsample wrapper (do
NOT remove the `_profile` wrappers — Task 2.5 removes them later):

```python
    # --- masks ---
    _profile.note(N=int(n), mask_logit_hw=tuple(pred_masks.shape[-2:]))  # TEMP #250
    masks_logits = pred_masks.float().squeeze(0)  # (N, H_m, W_m)
    if not torch.isfinite(masks_logits).all():
        raise RuntimeError(
            "non-finite mask logits in postprocess; check model outputs "
            "(pred_masks contains NaN/Inf)"
        )
    if keep_idx is not None:
        masks_logits = masks_logits[keep_idx]
    with _profile.bucket("mask_upsample"):  # TEMP #250
        masks_up = _upsample_mask_logits(masks_logits, original_hw)  # (M, H, W)
    with _profile.bucket("transfer_binarize"):  # TEMP #250
        masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (M, H, W) bool
```

Finally, the entries loop iterates over survivors. Replace `for i in range(n):`
with a survivor count. Compute it once after the filter:

```python
    m = scores.shape[0]  # survivor count (== n when no filter / n <= max_dets)
```

and change the loop bound from `range(n)` to `range(m)`. (The `boxes_list`,
`scores_list`, and `masks_bin` are all already sliced to survivors, so they are
length `m`.)

- [ ] **Step 4: Run the filter tests + the FULL postprocess suite**

```bash
uv run pytest tests/unit/test_eval_postprocess.py -o "addopts=" -q
```

Expected: all PASS — the new filter tests AND every pre-existing postprocess test
(the default `max_dets=None` path is byte-identical to before).

- [ ] **Step 5: Grep all callers; confirm predict/visualize are untouched**

```bash
grep -rn "queries_to_coco_results" src/ tests/
```

Confirm `predict/runner.py:481` and `eval/visualize.py:264` pass NO `max_dets`
(default `None` → no filter, preserving their need for all queries). They require
NO change.

- [ ] **Step 6: Lint + import-guard + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/postprocess.py tests/unit/test_eval_postprocess.py
uv run ruff format --check src/custom_sam_peft/eval/postprocess.py tests/unit/test_eval_postprocess.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/eval/postprocess.py tests/unit/test_eval_postprocess.py
git commit -m "feat(#250): tie-safe top-max_dets query filter (mAP-exact, opt-in via max_dets)"
```

---

### Task 2.2: Thread the cap from the evaluator into postprocess

**Files:**

- Modify: `src/custom_sam_peft/eval/evaluator.py`
- Test: `tests/unit/test_evaluator.py`

- [ ] **Step 1: Write the failing test that the evaluator passes a cap**

Add to `tests/unit/test_evaluator.py` — a spy confirms `queries_to_coco_results`
is called with `max_dets == coco_max_dets_cap()` (100):

```python
def test_iter_predictions_passes_max_dets_cap(stub_model, tiny_text_dataset):
    """Evaluator must thread the COCO maxDets cap into postprocess so >100-query
    models are filtered mAP-exactly."""
    from unittest.mock import patch

    from custom_sam_peft.eval.metrics import coco_max_dets_cap

    cfg = EvalConfig(mode="lite", lite_max_images=1, iou_thresholds=[0.5], batch_size=1)
    ev = Evaluator(cfg)
    examples = [tiny_text_dataset[0]]
    with patch(
        "custom_sam_peft.eval.evaluator.queries_to_coco_results",
        wraps=__import__(
            "custom_sam_peft.eval.postprocess", fromlist=["queries_to_coco_results"]
        ).queries_to_coco_results,
    ) as spy:
        ev._iter_predictions(stub_model, examples, tiny_text_dataset)
    assert spy.called
    for _args, kwargs in spy.call_args_list:
        assert kwargs.get("max_dets") == coco_max_dets_cap()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_evaluator.py::test_iter_predictions_passes_max_dets_cap -o "addopts=" -q
```

Expected: FAIL — the evaluator does not yet pass `max_dets`.

- [ ] **Step 3: Thread the cap**

In `evaluator.py`, add the import (after the metrics import):

```python
from custom_sam_peft.eval.metrics import MetricsReport, coco_max_dets_cap, compute_coco_map
```

Compute the cap once in `_iter_predictions` (after `n_classes` is set, ~`:155`):

```python
        max_dets_cap = coco_max_dets_cap()  # top-N cap, derived from COCOeval maxDets
```

and pass it to the `queries_to_coco_results` call (`evaluator.py:215`):

```python
                            entries = queries_to_coco_results(
                                _row_outputs(outputs, r),
                                int_id,
                                cat_idx + 1,
                                original_hw,
                                cfg.mask_threshold,
                                max_dets=max_dets_cap,
                            )
```

- [ ] **Step 4: Run to verify pass + full evaluator suite**

```bash
uv run pytest tests/unit/test_evaluator.py -o "addopts=" -q
```

Expected: all PASS.

- [ ] **Step 5: Lint + import-guard + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/evaluator.py tests/unit/test_evaluator.py
uv run ruff format --check src/custom_sam_peft/eval/evaluator.py tests/unit/test_evaluator.py
uv run python -c "import custom_sam_peft"
git add src/custom_sam_peft/eval/evaluator.py tests/unit/test_evaluator.py
git commit -m "feat(#250): thread coco_max_dets_cap into eval postprocess (filter is now live in eval)"
```

---

### Task 2.3: Batched device→host transfer + batched RLE over survivors (#3)

Bitwise-identical results — pure perf. Keep masks on GPU through the filter, then
do a SINGLE batched device→host transfer for survivors and batch the RLE encode.
The current per-call code already transfers the whole `(M, H, W)` bool tensor in
one `.cpu()` (`postprocess.py:111`); the win is (a) ensuring boxes+scores+masks
transfer together without interleaved syncs and (b) a batch-RLE encode instead of
the per-query Python `mask_utils.encode` loop.

**Files:**

- Modify: `src/custom_sam_peft/eval/postprocess.py`
- Test: `tests/unit/test_eval_postprocess.py`

- [ ] **Step 1: Write the failing test that RLE output is identical (batched == loop)**

Add to `tests/unit/test_eval_postprocess.py` — asserts the decoded masks +
entries are unchanged by batching (the regression net for #3):

```python
def test_batched_rle_decodes_identically():
    # Distinct mask patterns per query; batched RLE must decode bit-identically.
    n = 5
    masks = torch.full((1, n, 4, 4), -10.0)
    for i in range(n):
        masks[0, i, : i + 1, : i + 1] = 10.0  # growing top-left square
    out = {
        "pred_logits": torch.zeros(1, n, 1),
        "pred_boxes": torch.full((1, n, 4), 0.5),
        "pred_masks": masks,
        "presence_logit_dec": torch.zeros(1, 1),
    }
    entries = queries_to_coco_results(out, image_id=1, category_id=1, original_hw=(4, 4))
    assert len(entries) == n
    for i, e in enumerate(entries):
        decoded = mask_utils.decode(e["segmentation"])
        assert decoded.sum() == (i + 1) * (i + 1)  # the growing square area
```

- [ ] **Step 2: Run — confirm it passes against the current loop (this is the invariant to preserve)**

```bash
uv run pytest tests/unit/test_eval_postprocess.py::test_batched_rle_decodes_identically -o "addopts=" -q
```

Expected: PASS (the current loop already produces this). This test is the
**invariant** Step 3 must NOT break — it is a characterization test, written
before the refactor so the batched form is proven identical.

- [ ] **Step 3: Replace the per-query RLE loop with a batched encode**

`pycocotools.mask.encode` accepts a Fortran-ordered `(H, W, K)` uint8 array and
returns a LIST of K RLE dicts in one call. Replace the entries block (which
currently has the Phase-1 `_profile` buckets from Task 1.2 — those are removed in
Task 2.5; here, keep them and just batch the encode inside the `rle_encode`
bucket):

```python
    entries: list[dict[str, object]] = []
    with _profile.bucket("transfer_binarize"):  # TEMP #250 (box/score device->host)
        boxes_list = boxes_xywh.cpu().tolist()
        scores_list = scores.cpu().tolist()
    with _profile.bucket("rle_encode"):  # TEMP #250
        # Batched RLE: encode all survivor masks in ONE pycocotools call.
        # masks_bin is (M, H, W) bool; encode wants Fortran (H, W, M) uint8.
        if m:
            masks_fortran = np.asfortranarray(
                np.ascontiguousarray(masks_bin).transpose(1, 2, 0).astype(np.uint8)
            )
            rles = mask_utils.encode(masks_fortran)  # list[M] of RLE dicts
        else:
            rles = []
        for i in range(m):
            rle = rles[i]
            counts = rle["counts"]
            rle["counts"] = counts.decode("ascii") if isinstance(counts, bytes) else counts
            entries.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in boxes_list[i]],
                    "score": float(scores_list[i]),
                    "segmentation": rle,
                }
            )
    return entries
```

`_logits_to_rle` becomes unused after this. Leave it in place for Task 2.5 to
decide (it may still be referenced elsewhere — grep before removing).

- [ ] **Step 4: Run the FULL postprocess suite (batched output must be identical)**

```bash
uv run pytest tests/unit/test_eval_postprocess.py -o "addopts=" -q
```

Expected: all PASS — especially `test_batched_rle_decodes_identically`,
`test_rle_roundtrip`, `test_mask_upsample_and_threshold_at_zero`, and the filter
tests (the batched path produces byte-identical RLE).

- [ ] **Step 5: Grep `_logits_to_rle` usage; remove if now dead**

```bash
grep -rn "_logits_to_rle" src/ tests/
```

If the only definition is in `postprocess.py` and it has no remaining callers,
remove it (and its now-unused state). If anything else references it, leave it.
Re-run the suite + import guard after any removal:

```bash
uv run pytest tests/unit/test_eval_postprocess.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
```

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/postprocess.py tests/unit/test_eval_postprocess.py
uv run ruff format --check src/custom_sam_peft/eval/postprocess.py tests/unit/test_eval_postprocess.py
git add src/custom_sam_peft/eval/postprocess.py tests/unit/test_eval_postprocess.py
git commit -m "perf(#250): batched device->host transfer + batched RLE over survivors (bitwise-identical)"
```

---

### Task 2.4: Cosmetic — drop the misleading `dtype=torch.float32` Runtime label (#1)

No behavior change: `to_device` (`runtime/_device.py:18`) moves device only, never
casts; the `float32` label never reaches forward math (Cause #1, spec §3.1).

**Files:**

- Modify: `src/custom_sam_peft/eval/evaluator.py:151`

- [ ] **Step 1: Inspect the Runtime construction + `to_device` usage**

```bash
grep -n "eval_runtime\|Runtime(" src/custom_sam_peft/eval/evaluator.py
```

Confirm `eval_runtime` is used only by `to_device(...)` (device move). The `dtype`
field is inert here.

- [ ] **Step 2: Remove the misleading dtype label**

Replace `evaluator.py:151`:

```python
        eval_runtime = Runtime(device=param_device, dtype=torch.float32)
```

with (drop the dtype kwarg if `Runtime.dtype` has a default; otherwise set it to
the actual eval dtype). First check the `Runtime` signature:

```bash
grep -n "class Runtime\|dtype" src/custom_sam_peft/runtime/_runtime.py | head
```

If `dtype` has a default, use:

```python
        eval_runtime = Runtime(device=param_device)
```

If `dtype` is required (no default), set it to the model's actual param dtype so
the label is truthful rather than misleadingly `float32`:

```python
        eval_runtime = Runtime(device=param_device, dtype=param_dtype)
```

where `param_dtype` is derived next to `param_device`:

```python
        try:
            _p = next(model.parameters())
            param_device = _p.device
            param_dtype = _p.dtype
        except (StopIteration, AttributeError):
            param_device = torch.device("cpu")
            param_dtype = torch.float32
```

Pick whichever branch matches the actual `Runtime` signature; do NOT leave a
hardcoded `float32` that contradicts the bf16 forward.

- [ ] **Step 3: Run the evaluator suite + import guard**

```bash
uv run pytest tests/unit/test_evaluator.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
```

Expected: all PASS.

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check src/custom_sam_peft/eval/evaluator.py
uv run ruff format --check src/custom_sam_peft/eval/evaluator.py
git add src/custom_sam_peft/eval/evaluator.py
git commit -m "fix(#250): drop misleading eval Runtime dtype=float32 label (eval forward is bf16)"
```

---

### Task 2.5: Remove the Phase-1 instrumentation (clean revert)

The instrumentation was gated/isolated specifically so this is a clean removal
with no residue. After this task, `grep -rn "_profile\|CSP_EVAL_PROFILE" src/`
returns nothing.

**Files:**

- Delete: `src/custom_sam_peft/eval/_profile.py`, `scripts/profile_eval_250.py`
- Modify: `src/custom_sam_peft/eval/postprocess.py`,
  `src/custom_sam_peft/eval/evaluator.py` (strip all `_profile` calls + import)

- [ ] **Step 1: Strip `_profile` from postprocess**

Remove the `from custom_sam_peft.eval import _profile` import and unwrap the
`with _profile.bucket(...)` / `_profile.note(...)` calls — restore the plain
statements they wrapped (the filter + batched-RLE logic from Tasks 2.1/2.3 stays;
only the timing wrappers go). After removal, the `mask_upsample`,
`transfer_binarize`, and `rle_encode` work runs unwrapped.

- [ ] **Step 2: Strip `_profile` from the evaluator**

Remove the import and the `with _profile.bucket("forward")`,
`_profile.incr("forwards")`, `_profile.note(...)`, and
`with _profile.bucket("coco_aggregate")` calls — restore the plain forward call
and the plain `compute_coco_map` call.

- [ ] **Step 3: Delete the profiler module + the runner script**

```bash
git rm src/custom_sam_peft/eval/_profile.py scripts/profile_eval_250.py
```

- [ ] **Step 4: Verify no residue + suites green + import guard**

```bash
grep -rn "_profile\|CSP_EVAL_PROFILE\|profile_eval_250" src/ scripts/ tests/ || echo "no residue"
uv run pytest tests/unit/test_eval_postprocess.py tests/unit/test_evaluator.py -o "addopts=" -q
uv run python -c "import custom_sam_peft"
uv run python -m py_compile src/custom_sam_peft/eval/postprocess.py src/custom_sam_peft/eval/evaluator.py
```

Expected: `no residue`; all PASS; import clean.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
git add -A
git commit -m "spike(#250): remove Phase-1 profiling instrumentation (clean revert)"
```

---

### Task 2.6: Full-suite regression + GPU real-run before/after + report update

The proof the optimization is free: the full relevant CPU suite is green, the
blast-radius callers are exercised, and (GPU) the full `MetricsReport` is identical
before vs after within float tolerance.

**Files:**

- Modify: `docs/research/2026-06-02-issue-250-eval-perf-attribution.md`

- [ ] **Step 1: Run the FULL relevant CPU test set (blast radius, not just one file)**

```bash
uv run pytest tests/unit/test_eval_postprocess.py tests/unit/test_evaluator.py tests/predict/ -o "addopts=" -q
```

Expected: all PASS — predict/visualize callers (`max_dets` defaulted to `None`)
behave exactly as before; eval filters mAP-exactly.

- [ ] **Step 2: GPU real-run before/after regression (GPU-REQUIRED)**

On the RTX 5070 Ti, in process-isolated runs (spec §6.2): evaluate the SAME
representative config on the commit BEFORE Phase 2 (`git stash`/checkout or a saved
baseline `metrics.json`) and on the post-Phase-2 HEAD. Assert the full
`MetricsReport` — overall mAP / mAP_50 / mAP_75 + per-class AP — is identical
within float tolerance (ideally bit-exact for kept entries). Use
`scripts/run_gpu_tests.sh`-style per-file isolation; do NOT run a bare
`pytest tests/`. Capture the before/after metrics + the re-profiled bucket table.

If a real dataset is unavailable locally for the regression, the orchestrator
escalates (the "score didn't move" proof requires a representative run) — do NOT
claim the optimization free without it.

- [ ] **Step 3: Re-run the profiler (post-Phase-2) for the realized-win numbers**

The profiler was removed in Task 2.5; for the re-profile, run the Phase-2 GPU
regression with a lightweight wall-clock timer around `Evaluator.evaluate` (or
temporarily re-add and immediately re-remove the timer in a throwaway local edit —
do NOT commit it). Record the realized eval wall-time before vs after.

- [ ] **Step 4: Fill the report's Before/After section**

Update `docs/research/2026-06-02-issue-250-eval-perf-attribution.md`:

- Before/after eval wall-time (absolute + speedup factor).
- The mAP-identical proof (overall + per-class deltas == 0 within tolerance).
- Per-lever realized contribution (top-N filter win ≈ `(N − cap)/N`; batched
  transfer/RLE win from the re-profile).
- Markdown-lint the file (same tool as Task 1.6 Step 2) before committing.

- [ ] **Step 5: Commit**

```bash
git add docs/research/2026-06-02-issue-250-eval-perf-attribution.md
git commit -m "docs(#250): record before/after + mAP-identical proof (Phase 2 close-out)"
```

---

## Self-review — spec coverage map (§1–§10)

| Spec section | Covered by |
|---|---|
| §1 Goals & Scope (attribute eval time; land #2a + #3; document CC 7.5) | Phase 1 (report) + Phase 2 (filter + batched transfer/RLE) + Task 1.5 Step 4 / 1.6 |
| §2 Architectural Approach (measure-first, two phases, decision gate, boundary discipline preserved) | Phasing overview + Phase 1→Phase 2 contract + Decision Gate; postprocess-only conversion preserved (Tasks 2.1/2.3) |
| §3.1 Cause #1 bf16-already-on (MISREAD) | Task 1.5 Step 3 (confirm) + Task 2.4 (cosmetic label removal) |
| §3.2 Cause #1b CC 7.5 precision (REQUIRED) | Task 1.5 Step 4 + report item 5 (Task 1.6 Step 1) |
| §3.3 Cause #2 top-100 filter (REAL; tie-safe; cap from `max(maxDets)`; rank by emitted score; N measured) | Task 2.0 (cap helper) + Task 2.1 (threshold filter, tie-safe, score-ranked) + Task 1.5 (measured N) |
| §3.4 Cause #3 batched transfer + RLE (REAL; bitwise-identical) | Task 2.3 |
| §3.5 Cause #4 forwards-per-image (INHERENT; report only) | Task 1.3 Step 2 (count) + Task 1.6 Step 1 item 6 (report, not optimized) |
| §4.1 Instrumentation (temporary, CUDA-sync, buckets, metadata, env-gated, spike-only) | Tasks 1.1–1.3 (`_profile`, env gate, five buckets, N/n_classes/forwards/sizes/n_images) |
| §4.2 Run environment (5070 Ti, representative config, GPU isolation, CC 7.5 separately) | Task 1.4 (runner) + Task 1.5 (run + CC 7.5 reasoning) |
| §4.3 Deliverable: attribution report (per-bucket %, N, bf16-on, CC 7.5, forwards/img, GO/NO-GO) | Task 1.6 |
| §4.4 Decision gate (postprocess dominates AND N>100; else escalate) | Phase-1 Decision Gate checkpoint + Task 2.0 Step 0 |
| §5 Phase 2 steps 1–5 (filter, batched, cosmetic, remove instrumentation, re-profile) | Tasks 2.1/2.2 (1), 2.3 (2), 2.4 (3), 2.5 (4), 2.6 (5) |
| §6.1 Unit (>100 queries exact + boundary tie; COCO entry set + mAP == baseline) | Task 2.1 Step 1 (`test_filter_keeps_top_cap_by_score`, `test_filter_boundary_ties_keep_superset`) + Task 2.6 Step 2 (mAP-equal real run) |
| §6.2 Real-run regression (full MetricsReport identical before/after) | Task 2.6 Step 2 |
| §6.3 Edge cases (N≤100 no-op; N=0 → []; boundary ties) | Task 2.1 Step 1 (`test_filter_no_op_when_n_le_cap`, `test_filter_n_zero_returns_empty`, `test_filter_boundary_ties_keep_superset`) |
| §7 Files Affected (postprocess, evaluator, metrics, temp hooks added/removed, test_evaluator/postprocess, report) | metrics (2.0), postprocess (2.1/2.3), evaluator (2.2/2.4), temp hooks added P1 (1.1–1.4) / removed P2 (2.5), tests (2.1/2.2/2.3), report (1.6/2.6) |
| §8 Phasing & interface contract (P1 exposes report+GO/NO-GO; P2 consumes; escalate if lever invalidated) | Phase 1→Phase 2 interface contract + Decision Gate |
| §9 Out of scope (no #2b score-threshold; no eval autocast; no multiplex change; instrumentation spike-only) | Cited-constants rule (rejects #2b); no autocast added; Cause #4 not optimized (1.3/1.6); instrumentation removed (2.5) |
| §10 Open questions/risks (N unknown until measured; CC 7.5 unverified; gate makes P2 scope contingent) | Task 1.5 (measures N + CC 7.5) + Decision Gate (contingent scope) |
