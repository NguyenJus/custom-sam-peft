# Eval Subsystem Design (architecture step 6 — `spec/eval`)

**Status:** Approved (2026-05-17)
**Scope:** `src/esam3/eval/` — COCO mAP and per-class AP for SAM 3.1 instance
segmentation, plus the integration points with `Trainer.fit()` and the
`esam3 eval` CLI command. Depends on `spec/data-loading` (1) and
`spec/model-loading` (2). Implements the contracts declared in the
architecture doc, Section 5 (`MetricsReport`, `Evaluator.evaluate`).

---

## 1. Goals & v0 Scope

A small evaluation subsystem that turns SAM 3.1's per-class-prompt forward
outputs into a `MetricsReport` of COCO-style mAP, mAP@.5, mAP@.75, and
per-class AP. Same `Evaluator` class is used for in-training "lite" ticks and
for the standalone `esam3 eval` CLI; the difference is two preset behaviors
controlled by `EvalConfig.mode`.

| Dimension | v0 | Deferred |
|---|---|---|
| Metrics | mAP, mAP@.5, mAP@.75, per-class AP | F1/recall sweeps, custom metrics |
| Backend | pycocotools | torchmetrics, handrolled |
| Prompt mode at eval | text-only (vocab loop per image) | bbox-prompted eval |
| Batching | one image × one class per forward | batched same-class eval |
| Mask resolution | mask logits upsampled to model-input H×W | upsample to original camera resolution (export-time concern) |
| Coord space | resized image coordinates | original camera coordinates |
| Mid-run eval | "lite" preset (capped images, overall mAP only) | online F1, qualitative panels (already handled by tracker) |

---

## 2. Architectural Approach

Three files, each with one job. The only piece with real algorithmic risk
(turning Meta's per-class forward output into COCO results entries) is
isolated into a pure-function module that can be unit-tested against the
existing `tiny_sam3_stub` without touching the dataset or pycocotools.

```
src/esam3/eval/
  __init__.py
  metrics.py        # MetricsReport (already defined) + compute_coco_map(...)
  postprocess.py    # NEW — model output dict → COCO results entries (pure)
  evaluator.py      # Evaluator class; orchestrates dataset × vocab loop
```

**Boundary rules:**

- `evaluator.py` does no tensor math beyond `model.eval()` / `torch.no_grad()`.
- All tensor → COCO conversion lives in `postprocess.py`. No I/O there.
- All metric math lives in `metrics.py` (delegated to pycocotools).
- `Evaluator` depends on the `Dataset` protocol (Section 5 of the
  architecture doc) — never on a concrete adapter.

---

## 3. Public Surfaces

### 3.1 `metrics.py`

`MetricsReport` is already defined; unchanged. New function:

```python
def compute_coco_map(
    predictions: list[dict],            # COCO results entries
    ground_truth: COCO,                 # pycocotools.coco.COCO loaded in-memory
    iou_thresholds: list[float],
    include_per_class: bool,
) -> MetricsReport
```

- `include_per_class=True` populates `MetricsReport.per_class`.
- `include_per_class=False` leaves it `{}` and skips the per-class sweep
  (the most expensive pycocotools step).
- Classes with no GT instances in the eval split are omitted from
  `per_class` (pycocotools' `AP=-1` is not surfaced).
- Empty `predictions` returns `MetricsReport(overall={"mAP":0.0,
  "mAP_50":0.0, "mAP_75":0.0}, per_class={}, n_images=N,
  n_predictions=0)` and logs a warning.

### 3.2 `postprocess.py`

Pure functions, no model, no dataset, no I/O.

```python
def queries_to_coco_results(
    outputs: dict[str, Tensor],          # Meta's per-class forward output
    image_id: int,                       # already-coerced int (Evaluator's job)
    category_id: int,                    # 1-indexed COCO category id
    original_hw: tuple[int, int],        # (H, W) the image was passed in at
    mask_threshold: float = 0.0,         # logit threshold for binarization
) -> list[dict]                          # COCO results entries (one per query)
```

Required keys in `outputs`: `pred_logits` `(1, N, 1)`,
`pred_boxes` `(1, N, 4)` normalized cxcywh, `pred_masks` `(1, N, 288, 288)`
logits, `presence_logit_dec` `(1, 1)`.

Private helpers: `_denorm_cxcywh_to_xywh`, `_upsample_mask_logits`,
`_logits_to_rle`.

### 3.3 `evaluator.py`

```python
class Evaluator:
    def __init__(self, cfg: EvalConfig) -> None: ...
    def evaluate(self, model: Any, dataset: Dataset) -> MetricsReport: ...
    def evaluate_and_save(
        self, model: Any, dataset: Dataset, output_dir: Path
    ) -> MetricsReport: ...
```

`evaluate_and_save` calls `evaluate`, then writes `output_dir/metrics.json`
always and `output_dir/predictions.json` when `cfg.save_predictions and
cfg.mode == "full"`.

---

## 4. Config Schema Additions

Three new fields on `EvalConfig`, plus one one-line addition to
`DataConfig`. All defaults preserve existing YAMLs without edits.

```python
EvalMode = Literal["full", "lite"]

class EvalConfig(_Strict):
    # existing
    metrics: list[str] = Field(default_factory=lambda: [...])
    iou_thresholds: list[float] = Field(default_factory=lambda: [...])
    # NEW
    mode: EvalMode = "full"
    lite_max_images: PositiveInt = 64
    mask_threshold: float = 0.0
    save_predictions: bool = False

class DataConfig(_Strict):
    # existing
    train: DataSplit
    val: DataSplit
    test: DataSplit | None = None   # NEW — enables CLI --split test
    ...
```

**Behavior table:**

| Field             | `mode="full"`           | `mode="lite"`                          |
|-------------------|-------------------------|----------------------------------------|
| Images iterated   | all of `dataset`        | `dataset[:lite_max_images]` in order   |
| Per-class AP      | computed                | skipped (`per_class={}`)               |
| Overall mAP/50/75 | computed                | computed                               |
| Predictions JSON  | per `save_predictions`  | never (forced off)                     |

---

## 5. Data Flow

```
Evaluator.evaluate(model, dataset):
  1. Resolve image set:
       n = len(dataset) if mode=="full" else min(lite_max_images, len(dataset))
       indices = range(n)

  2. Build in-memory COCO ground truth once:
       gt = _build_coco_gt(dataset, indices)
       # image_id = stable int hash of Example.image_id (blake2s, 8-byte digest)
       # category_id = enumerate(dataset.class_names, start=1)
       # error on hash collision (P ≈ 7e-15 at 5k images on 2^64; surfaced loudly if it ever hits)

  3. model.eval(); with torch.no_grad():
       predictions: list[dict] = []
       for i in indices:
           ex = dataset[i]
           original_hw = ex.image.shape[-2:]
           for cat_id, class_name in enumerate(dataset.class_names, start=1):
               outputs = model(
                   ex.image.unsqueeze(0),
                   [TextPrompts(classes=[class_name])],
                   box_hints=None,
               )
               entries = queries_to_coco_results(
                   outputs, int_image_id(ex.image_id), cat_id,
                   original_hw, cfg.mask_threshold,
               )
               predictions.extend(entries)

  4. report = compute_coco_map(
         predictions, gt, cfg.iou_thresholds,
         include_per_class=(cfg.mode == "full"),
     )

  5. Restore model's prior training/eval state; return report.
```

**Coordinate space invariant.** GT masks/boxes from `Example.instances` are
already in the resized image coordinate space. The model predicts in that
same space. pycocotools needs only internal consistency. v0 does not
upsample to original camera resolution — that is an export-time concern.

**Per-image, per-class loop.** Matches the model's "one class prompt per
forward" contract. No same-class batching across images in v0 (deferred —
see `logs/TODO.md`).

**Cost model & lite-preset tuning.** Total forwards per `evaluate()` call =
`n_images × len(dataset.class_names)`. For a vocabulary of 80 classes on a
12GB GPU at ~100 ms / forward, the defaults imply:

| Mode | n_images       | Classes | Forwards | Wall time (rough) |
|------|----------------|---------|----------|-------------------|
| lite | 64             | 80      | 5,120    | ~8.5 min          |
| full | len(val)≈5,000 | 80      | 400,000  | ~11 h             |

The full-preset cost is acceptable end-of-run only; the lite-preset
default of 64 is **only sensible for small vocabularies**. Users with
larger vocabularies should drop `lite_max_images` (e.g., to 8) or raise
`eval_every` so mid-training eval doesn't dominate the schedule. The
trainer logs the wall-clock of each lite eval so users notice.

---

## 6. Postprocess Algorithm (`postprocess.py`)

```
1. p_obj      = sigmoid(pred_logits).squeeze(-1).squeeze(0)        # (N,)
   p_presence = sigmoid(presence_logit_dec).squeeze()              # scalar
   scores     = p_obj * p_presence                                 # (N,)
   assert torch.isfinite(scores).all()

2. boxes_norm = pred_boxes.squeeze(0)                              # (N, 4) cxcywh in [0,1]
   boxes_xywh = _denorm_cxcywh_to_xywh(boxes_norm, original_hw)   # (N, 4) absolute xywh
   # clamps to [0, H/W] before returning

3. masks_logits = pred_masks.squeeze(0).float()                    # (N, 288, 288)
   assert torch.isfinite(masks_logits).all()
   masks_up    = F.interpolate(masks_logits.unsqueeze(1), size=original_hw,
                               mode="bilinear", align_corners=False).squeeze(1)
   masks_bin   = (masks_up > mask_threshold).cpu().numpy()         # (N, H, W) bool

4. for n in range(N):
       rle = pycocotools.mask.encode(np.asfortranarray(masks_bin[n]))
       rle["counts"] = rle["counts"].decode("ascii")
       entries.append({
           "image_id":     image_id,
           "category_id":  category_id,
           "bbox":         boxes_xywh[n].tolist(),
           "score":        float(scores[n]),
           "segmentation": rle,
       })
   return entries
```

**Key choices & invariants:**

- **No filtering.** Per the locked decision (score = obj × presence,
  keep-all-queries), every query becomes an entry. pycocotools' PR sweep
  does the ranking. Top-k can be added later via a kwarg.
- **No NMS.** SAM 3.1's query design is already de-duplicating;
  pycocotools tolerates duplicates with a penalty.
- **`float()` before `interpolate`.** Avoids fp16/bf16 underflow.
- **`np.asfortranarray`** is required by `pycocotools.mask.encode`.
- **Batch == 1.** Enforced by `ValueError` for safety; Evaluator
  guarantees it.

**Failure modes:**

- Missing key in `outputs` → `KeyError` (postprocess is a contract).
- `pred_logits.shape[0] != 1` → `ValueError("postprocess expects batch=1")`.
- `len(original_hw) != 2` → `ValueError`.
- NaN/Inf in scores or mask logits → `RuntimeError` (eval halts, not silently masked).

---

## 7. Edge Cases & Numerical Stability

- **Zero GT for a class.** Class omitted from `per_class`; INFO log notes
  count: `eval: skipped 3/80 classes with no GT instances`.
- **Zero predictions for an image-class pair.** `[]` is just absent from
  the predictions list; pycocotools handles it.
- **Zero predictions overall.** Caught; `MetricsReport` with zeroed overall
  metrics and a warning log.
- **NaN/Inf in `pred_logits`, `presence_logit_dec`, or `pred_masks`.**
  Raise `RuntimeError`. Surfacing a NaN is more useful than reporting a
  misleading number.
- **Score saturation.** None needed — `sigmoid` is safe and
  `p_obj * p_presence ∈ [0, 1]` by construction.
- **Mixed precision.** All postprocess math after `.float()` on raw
  tensors; runs in fp32.
- **Determinism.** No augmentation, no dropout, no seed touched. Output is
  fully determined by model state + dataset order.

---

## 8. Trainer Integration (changes to `train/trainer.py`)

- **Mid-run eval** at every `eval_every` step:
  ```python
  lite_cfg = self.cfg.eval.model_copy(update={"mode": "lite",
                                              "save_predictions": False})
  try:
      report = Evaluator(lite_cfg).evaluate(self.model, self.val_ds)
      self.tracker.log_scalars(step, report.overall)
  except Exception as e:
      logger.warning("lite eval failed at step %d: %s", step, e)
  ```
  A flaky lite eval does not kill a training run.
- **End-of-run eval**:
  ```python
  report = Evaluator(self.cfg.eval).evaluate(self.model, self.val_ds)
  # ...
  return RunResult(..., final_metrics=report)
  ```
  Not wrapped — if it fails, `final_metrics` stays `None` and the
  exception propagates. The user wants to know.
- **`metrics.json`** in `run_dir` is rewritten as:
  ```json
  {
    "overall": {"mAP": ..., "mAP_50": ..., "mAP_75": ...},
    "per_class": {...},
    "n_images": N,
    "n_predictions": M,
    "global_step": ...,
    "epoch": ...,
    "box_hint_p_final": ...
  }
  ```
  i.e., `MetricsReport` fields at top level plus the three run-context
  fields the trainer was previously writing. `evaluate_and_save` writes
  the report fields; the trainer overlays the run-context fields after
  the report is computed.

---

## 9. CLI Integration (`cli/eval_cmd.py`)

```
esam3 eval --config X.yaml --checkpoint runs/foo/adapter
           [--split val|test] [--output PATH] [--save-predictions/--no-save-predictions]
```

- Load config; resolve `--split` to `data.val` or `data.test`; build
  dataset; load model; apply adapter checkpoint;
  call `Evaluator(cfg.eval).evaluate_and_save(model, ds, output_dir)`.
- `--output` defaults to the checkpoint's parent dir (i.e., the run dir).
- `--save-predictions` / `--no-save-predictions` override
  `cfg.eval.save_predictions`.
- `--split test` with `data.test = None` → exit 2 with message:
  `"--split test requires data.test in config; got None"`.

---

## 10. Testing Strategy

Three tiers matching the architecture doc.

### Unit (CPU, every commit)

**`tests/unit/test_eval_postprocess.py`** — pure-function tests, no model.

- `test_shapes_and_keys` — synthetic inputs; assert list length, keys per
  entry, dtypes.
- `test_score_formula` — `pred_logits=0` (sigmoid=0.5),
  `presence=0` (sigmoid=0.5) → all scores 0.25.
- `test_box_denorm` — cxcywh `(0.5, 0.5, 1.0, 1.0)` on `(100, 200)` image
  → xywh `[0, 0, 200, 100]`.
- `test_mask_upsample_and_threshold` — 288×288 logits with a known
  positive region; assert upsampled binary mask covers proportional area
  ±1 pixel.
- `test_mask_threshold_param` — different thresholds, different masks.
- `test_rle_roundtrip` — encode then `pycocotools.mask.decode` returns
  input.
- `test_empty_queries` — `N=0` → `[]`.
- `test_batch_assert` — batch=2 → `ValueError`.
- `test_nonfinite_raises` — NaN in `pred_logits` → `RuntimeError`.
- `test_bfloat16_inputs` — feed bf16 tensors; no exception, finite
  scores.

**`tests/unit/test_eval_metrics.py`** — pycocotools wrapper, no model.

- `test_perfect_predictions` — synthetic GT + identical preds (score=1.0)
  → mAP ≈ 1.0.
- `test_zero_predictions` — empty list → zeroed overall + warning log.
- `test_zero_gt_class_filtered` — GT has only class A; preds span A and
  B → `per_class` contains only A.
- `test_include_per_class_flag` — `include_per_class=False` →
  `per_class == {}`.
- `test_iou_threshold_passthrough` — alternate thresholds reflected in
  the .5/.75 slices.

**`tests/unit/test_evaluator.py`** — orchestration with stub model +
`tiny_coco_dataset`.

- `test_evaluate_full_returns_report` — stub outputs;
  `n_images == len(dataset)`, `per_class` non-empty.
- `test_evaluate_lite_caps_images` — `mode="lite"`, `lite_max_images=1`,
  2-image fixture → `n_images == 1`, `per_class == {}`.
- `test_evaluate_does_not_mutate_training_state` — `model.train()`
  before; `model.training is True` after.
- `test_evaluate_and_save_writes_files` — full mode +
  `save_predictions=True` → `metrics.json` + `predictions.json` exist;
  lite mode + `save_predictions=True` → only `metrics.json`.
- `test_image_id_collision_detected` — patched GT-build → error raised.

CLI flag test in `tests/unit/test_cli.py`:
`test_eval_split_test_missing_errors` — `--split test` without
`data.test` exits non-zero.

### Integration (`@pytest.mark.integration`)

**`tests/integration/test_train_then_eval.py`** —

- Stub model + `tiny_coco/`. Run `Trainer.fit()` for 2 steps with
  `eval_every=1`. Assert: no exceptions, tracker received scalars,
  `RunResult.final_metrics` is a `MetricsReport`, `run_dir/metrics.json`
  is valid JSON matching `final_metrics.overall`.

### GPU smoke

Not in this spec. The architecture doc's step 9 (`spec/smoke-test`) owns
end-to-end real-SAM3.1 mAP assertions.

### Fixture extension

`tests/fixtures/tiny_sam3_stub.py` is extended so its forward returns the
four required keys with correct shapes. Used by both training and eval
tests.

### Coverage gate

Unchanged: 80% on `src/esam3`. New files are well-covered by the unit
tests above.

### Explicitly NOT tested

- pycocotools internals.
- Bilinear upsample correctness beyond ±1 pixel — torch's contract.
- Real SAM 3.1 outputs — owned by step 9 (`spec/smoke-test`).

---

## 11. Out of Scope (deferred to v1+ or other specs)

- Bbox-prompted eval (today eval is text-only).
- Same-class batching across images for throughput.
- Mask upsample to original camera resolution (export-time concern).
- F1/recall sweeps, custom metrics.
- Streaming predictions writer (for very large eval splits).
- mAP-based early stopping (architecture doc's `early_stop_p_threshold`
  hook in `BoxHintSchedule` is left for a future spec to consume).
