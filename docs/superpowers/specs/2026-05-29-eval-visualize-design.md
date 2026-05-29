# `eval --visualize` — GT vs Pred qualitative panels

**Issue:** none — purely additive eval feature; no GitHub issue required (predict's
multi-class rendering is already complete — see §11 non-goals).
**Release:** pre-1.0 minor bump (new user-visible `eval.visualize` knob + CLI flags → MINOR).
**Status:** locked design, single PR, no back-compat shims.
**Sibling spec:** [`2026-05-28-eval-predict-interactive-helpers-design.md`](2026-05-28-eval-predict-interactive-helpers-design.md)
(the eval CLI surface this work extends — `eval_cmd.py` already carries `--interactive`).

The `eval` command emits `metrics.json` (+ optional `predictions.json`) and nothing visual.
This spec adds a qualitative visualization, modeled on `predict --visualize`, but
tailored to evaluation: for a small, variety-weighted sample of eval images it renders
a **two-panel side-by-side composite — `Ground Truth | Prediction` — handling multiple
classes per image**, so a user can eyeball where the model agrees with and diverges from
ground truth. Predictions drawn in the right panel are the **matched 1:1** set (one pred
per GT, per class), so the two panels are directly comparable.

---

## §1 Goals and non-goals

### Goals

- A `eval.visualize` config knob (default **True** everywhere) plus per-command CLI
  overrides on `csp eval` and `csp run`, so the final/standalone eval path writes
  qualitative composites by default.
- One composite PNG per selected image: `Ground Truth | Prediction`, hstacked, with
  panel titles and a shared per-class color legend. The **same class is the same color
  in both panels**.
- A **variety-weighted sample** of `visualize_count` images (default 10), biased toward
  good results but always including a couple of the worst, ranked by the per-image IoU
  that the metrics pass already computes.
- Predictions are **matched 1:1 to GT** per class via the existing `HungarianMatcher`
  (mask-only Dice), aggregated across all classes in the image. Only matched preds are
  drawn — no unmatched/extra detections.
- **Reuse, not fork**: generalize `predict/visualize.py::render_overlay` to make the score
  optional (GT labels carry no score; pred labels do), share `color_for_class`/`PALETTE`,
  and reuse the eval postprocess helpers for pred mask/box/score conversion.
- Wired into the **standalone / final-eval path only** (`run_eval` →
  `Evaluator.evaluate_and_save`), so it covers `csp eval`, `csp run`'s eval phase, and
  `csp train --eval`, while leaving the periodic in-loop training eval untouched.

### Non-goals (one line each)

- **No predict change.** `predict --visualize` already renders multiple classes per image
  (§11); this spec touches `render_overlay` only to add the optional-score parameter,
  which is backward-compatible for predict's existing call.
- **No in-loop training-eval visualization change.** The trainer's periodic sample panel
  (`train/visualize.py::render_mask_panel`) stays exactly as-is (§3).
- **No new `train` CLI flag.** `csp train --eval` is config-driven and picks up
  `eval.visualize` (default True) automatically (§4).
- **No unmatched/extra-prediction rendering.** The Pred panel draws only the Hungarian-matched
  1:1 set, mirroring the GT count per class. Surfacing false positives is a future follow-up.
- **No new metrics.** Visualization is qualitative only; `metrics.json` is unchanged.
- **No GitHub issue.** This is additive and self-contained.

---

## §2 Architecture and data flow

The key design decision is a **two-phase, bounded** structure:

| Phase | What runs | Over how many images | Cost |
|-------|-----------|----------------------|------|
| **Phase 1 — metrics** | the existing `Evaluator.evaluate(..., return_per_example_iou=True)` pass | the whole split | unchanged |
| **Phase 2 — viz** | a NEW bounded pass: re-forward selected images one class at a time, run the matcher, render, write PNGs | only the ~`visualize_count` selected images | ≈ N extra forwards — trivial |

**Why a second pass is required (not a tap on Phase 1):**

1. The Phase-1 metrics path converts each multiplex forward straight into COCO-flat result
   dicts via `queries_to_coco_results` (`eval/postprocess.py:51`) and discards the raw
   `CanonicalOutputs` (logits, normalized boxes, per-query mask logits) that the Hungarian
   matcher needs. Re-deriving matcher inputs from the flattened COCO dicts is not possible.
2. The **"worst-N" selection is not known until Phase 1 finishes** — ranking needs the full
   per-image IoU list. So the viz pass cannot be fused into the metrics loop without
   buffering raw outputs for every image (memory-unbounded on large splits).

Phase 2 is **memory-bounded**: it processes and frees one image at a time, running one
class prompt per forward (K=1) so each forward yields a clean per-class `CanonicalOutputs`
for the matcher. The extra compute is `≈ N × (#classes)` single-class forwards — negligible
against the whole-split Phase-1 pass.

### Placement seam (critical)

Visualization is wired into the **standalone / final-eval path**:

- `run_eval` (`eval/runner.py:58`) is the orchestrator. It already requests
  `return_per_example_iou=True` on the `run` path; for the visualize path it requests the
  same list and threads it (plus the loaded model, dataset, and resolved output dir) into
  the new top-level entry point.
- It must **NOT** run inside `Evaluator.evaluate` (`eval/evaluator.py:332`), because the
  periodic **in-loop** training eval (`train/trainer.py::_eval_epoch` →
  `Evaluator.evaluate`) shares that method and already has its own sample-panel viz
  (`train/visualize.py::render_mask_panel`). Hanging eval-visualization off `evaluate`
  would fire it every epoch during training and double up with the in-loop panel.

Correct placement (`run_eval` / `evaluate_and_save`, never `evaluate`) **automatically**
covers all three final-eval entries — `csp eval`, `csp run`'s eval phase, and
`csp train --eval` — while leaving in-loop eval untouched. See §6 for the exact wiring.

### Data flow (pseudocode)

```
# run_eval, when visualize is on:
report, per_example_iou = evaluator.evaluate(model, dataset, return_per_example_iou=True)
persist metrics.json (+ predictions.json)        # unchanged
if visualize_resolved:
    write_eval_visualizations(
        model, dataset, out_dir,
        per_example_iou=per_example_iou,
        count=cfg.eval.visualize_count,
        mask_threshold=cfg.eval.mask_threshold,
        model_name=cfg.model.name,
        normalize=cfg.data.normalize,
        channel_semantics=cfg.data.channel_semantics,
    )

# inside write_eval_visualizations:
selected = pick_samples(per_example_iou, dataset, count)     # §5 variety-weighted
for idx in selected:
    try:
        render_eval_pair(model, dataset[idx], dataset.class_names, mask_threshold, …)
        save <out_dir>/visualizations/<sanitized image_id>.png
    except Exception:
        log warning, continue                                # §8 robustness
```

---

## §3 Why not the in-loop path (rationale, locked)

`Evaluator.evaluate` is called from two places:

1. **Final / standalone eval** — via `Evaluator.evaluate_and_save`
   (`eval/evaluator.py:423`), invoked by `run_eval` (`eval/runner.py:202`).
2. **In-loop training eval** — `train/trainer.py::_eval_epoch` calls `Evaluator.evaluate`
   directly each eval epoch.

These share the same `evaluate` method. The in-loop case already renders a per-step sample
panel (`train/visualize.py::render_mask_panel`, a `image | GT-overlay | pred-overlay` strip)
for TensorBoard image logging. Putting the new GT-vs-Pred composite inside `evaluate` would:

- fire it on every training eval epoch (unwanted overhead + clutter), and
- duplicate the in-loop panel's purpose.

Wiring at `run_eval` / `evaluate_and_save` confines the new viz to the final-eval surface,
covering `csp eval`, `csp run`, and `csp train --eval` (all of which reach `run_eval` /
`evaluate_and_save`) but never the in-loop `_eval_epoch`. **`render_mask_panel` and the
trainer are not modified by this PR.**

---

## §4 Configuration & CLI surface

### §4.1 `EvalConfig` field additions

`EvalConfig` (`config/schema.py:592`, a pydantic `_Strict` model) gains two fields:

```python
class EvalConfig(_Strict):
    # ... existing fields ...
    batch_size: PositiveInt | Literal["auto"] = "auto"   # existing
    visualize: bool = True
    visualize_count: PositiveInt = 10
```

- `visualize: bool = True` — default **True** everywhere; the standalone/final-eval path
  writes composites unless explicitly disabled.
- `visualize_count: PositiveInt = 10` — number of images to sample (must be ≥ 1; `PositiveInt`
  matches the existing `lite_max_images`/`nan_abort_after` convention in the schema).

Both are **advanced** fields (the section default is usable as-is), placed after
`batch_size` in the `EvalConfig` body.

### §4.2 `csp eval` CLI override

`eval_cmd.py::evaluate` (`cli/eval_cmd.py:21`) gains a tri-state override flag that mirrors
the existing `save_predictions` override exactly (`eval_cmd.py:34-38`, threaded at
`eval_cmd.py:87-93`):

```python
visualize: bool | None = typer.Option(
    None,
    "--visualize/--no-visualize",
    help="Override cfg.eval.visualize (write GT-vs-Pred composite panels).",
),
```

`None` means "use the config value"; an explicit `True`/`False` overrides it. It is passed
into the `run_eval(...)` call (`eval_cmd.py:87`) as `visualize=visualize` alongside the
existing `save_predictions=save_predictions`.

### §4.3 `csp run` CLI flag

`run_cmd.py::run` (`cli/run_cmd.py:177`) gains a two-state flag defaulting **True** (so a
full `train → eval → export` run shows results immediately):

```python
visualize: bool = typer.Option(
    True,
    "--visualize/--no-visualize",
    help="Write GT-vs-Pred composite panels in the eval phase.",
),
```

It is threaded through `_orchestrate` (`cli/run_cmd.py:70`) as a new parameter and forwarded
to the eval-phase `run_eval(...)` call (`cli/run_cmd.py:115`) as `visualize=visualize`. The
`run` command body (`run_cmd.py:226`) calls `_orchestrate(cfg, resume_path, mode, visualize=visualize)`.

`run` defaults visualize **on** (not tri-state) because `run` is the all-in-one happy-path
command — the user wants to see qualitative results at the end without extra flags. The
underlying `cfg.eval.visualize` default is also True, so the `run` flag and config agree;
the flag exists so a user can pass `--no-visualize` to a `run` without editing the config.

### §4.4 `csp train --eval` — no new flag

`train_cmd.py` calls `run_eval(cfg, artifacts=result)` (`cli/train_cmd.py:94`) with no
`visualize` kwarg, so it picks up `cfg.eval.visualize` (default True) via the config. **No
new flag is added to `train`** — it stays config-driven.

### §4.5 `run_eval` signature addition

`run_eval` (`eval/runner.py:58`) gains a keyword-only parameter mirroring `save_predictions`:

```python
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path | None = None,
    artifacts: EvalArtifacts | None = None,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    visualize: bool | None = None,          # NEW — None means "use cfg.eval.visualize"
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: bool = False,
) -> MetricsReport | tuple[MetricsReport, list[float]]:
```

`None` → use `cfg.eval.visualize`; explicit `True`/`False` overrides. The resolution is:
`visualize_resolved = cfg.eval.visualize if visualize is None else visualize`. Both `@overload`
stubs (`runner.py:28-55`) gain the same `visualize: bool | None = None` parameter so the typed
surface stays in sync.

### §4.6 Documentation & example configs

- **`docs/config-schema.md`** is the project's "published schema" referenced by the README
  (README.md:14, :122) — a markdown table doc, **not** a committed JSON-schema artifact (no
  `*.schema.json` exists in the repo; verified). Add two rows to its `## eval` table
  (currently `docs/config-schema.md:142-152`), modeled on the existing `eval.save_predictions`
  row: `eval.visualize` (bool, default `true`, advanced) and `eval.visualize_count` (int >0,
  default `10`, advanced). State in the spec that no JSON-schema artifact needs regenerating.
- **`configs/examples/*.yaml`** — the five examples with an `eval:` block (`coco_text_lora.yaml`,
  `coco_text_qlora.yaml`, `coco_text_auto_split.yaml`, `coco_text_lora_subset.yaml`,
  `coco_text_no_val.yaml`) currently list only `eval.iou_thresholds`. Because `visualize`
  defaults True and is an advanced override, the examples are **not required** to set it. The
  implementer MAY add a commented `# visualize: true` / `# visualize_count: 10` line to the
  primary `coco_text_lora.yaml` `eval:` block for discoverability, but this is optional and
  not load-bearing. Do not add uncommented entries (they would be redundant with the default).
- **`CHANGELOG.md`** — add an `### Added` (or feature) entry under `## [Unreleased]`
  describing the new `eval.visualize` / `eval.visualize_count` knobs, the `csp eval
  --visualize/--no-visualize` and `csp run --visualize/--no-visualize` flags, and the
  GT-vs-Pred composite output location.

---

## §5 Image selection — variety weighted toward good

### §5.1 Candidate set

Candidates are eval images with **≥ 1 GT instance**. No-GT images are excluded — there is
nothing to compare in a `Ground Truth | Prediction` panel when GT is empty. "Has GT" is
`len(dataset[idx].instances) > 0`.

> The per-image IoU list (`Evaluator._compute_per_example_iou`, `eval/evaluator.py:368-421`)
> assigns a **vacuous 1.0** to no-GT images and **0.0** to no-GT-but-has-preds images, so
> raw IoU rank alone would surface uninformative panels. The candidate filter (`has GT`)
> removes both degenerate cases before ranking. Selection ranks candidates only.

### §5.2 Ranking metric

Rank candidates by the per-image mean IoU already produced by the metrics pass:
`Evaluator.evaluate(..., return_per_example_iou=True)` returns `(MetricsReport, list[float])`,
a list aligned to dataset indices (`eval/evaluator.py:312-366`). The metric is, per image,
the threshold-averaged "best-pred IoU per GT" over `cfg.iou_thresholds`
(`_compute_per_example_iou`, `eval/evaluator.py:368-421`). `NaN` entries (examples skipped
during inference) are treated as `-inf` for ranking — they sort to the bottom and are
eligible only as "worst" picks.

> **Ranking vs render-pairing mismatch is intentional, not a bug.** The selection metric is
> the greedy best-pred-IoU-per-GT used by metrics; the render pairing (§7) is the Hungarian
> mask-only matcher. They differ, but both are mask-IoU-driven, so the qualitative ordering
> agrees: a high-ranked image still renders well-overlapping matched masks. State this
> explicitly so it is not misread as a discrepancy.

### §5.3 Variety-weighted spread

For `N = visualize_count`, partition the slots into three IoU bands:

- `good = round(0.5 · N)` — highest-IoU candidates.
- `worst = min(2, max(1, round(0.2 · N)))` — lowest-IoU candidates, **capped at 2**.
- `median = N − good − worst` — candidates around the middle of the ranking.

For `N = 10`: `good = 5`, `worst = min(2, max(1, 2)) = 2`, `median = 10 − 5 − 2 = 3` →
**5 good / 3 median / 2 worst**. The extra (second) worst slot guards against an
unpresentable bottom image (e.g. an all-zero or pathological case) — having two worst picks
means one informative bad example survives even if the very bottom one is degenerate.

**Within-band spread (dedup):** within each band, spread the picks across the band's index
range rather than taking adjacent items, so the chosen images are not near-duplicates in
IoU. Concretely, after sorting candidates by IoU descending: the `good` band is the top
`good` slice's *evenly spaced* indices, `median` is evenly spaced indices around the middle,
`worst` is the bottom `worst` slice's evenly spaced indices. An image is never selected twice
across bands (selection sets are disjoint by construction; when bands would overlap because
the candidate pool is small, dedup by preferring the higher band and back-filling from the
next band).

**Small-pool rule:** if the candidate count `≤ N`, take **all** candidates (no banding
needed). The output then has fewer than `N` composites — that is correct, not an error.

### §5.4 Selection function signature

```python
def pick_samples(
    per_example_iou: Sequence[float],
    dataset: Dataset,
    count: int,
) -> list[int]:
    """Return up to `count` dataset indices, variety-weighted toward high IoU.

    Filters to candidates with >=1 GT instance (excludes no-GT images), ranks by
    per_example_iou (NaN -> -inf, eligible only as 'worst'), and picks a
    good/median/worst spread per §5.3. Returns <= count indices when the candidate
    pool is smaller than count. Indices are returned in descending-IoU order so the
    written composites are filename-stable and roughly best-to-worst.
    """
```

`per_example_iou` is index-aligned with `dataset` (Phase-1 contract). `pick_samples` reads
`len(dataset[idx].instances)` for the GT filter; it does not run the model.

---

## §6 `run_eval` / `evaluate_and_save` wiring

`run_eval` (`eval/runner.py:58`) is the single integration point. After Phase-1 metrics are
computed and persisted, it conditionally runs Phase 2.

### §6.1 Two existing return branches

`run_eval` has two persistence branches today:

1. **`return_per_example_iou=True`** branch (`runner.py:180-200`, used by `csp run`): already
   calls `evaluator.evaluate(..., return_per_example_iou=True)`, writes `metrics.json` (and
   `predictions.json`) into `out`, and has `per_example_iou` in hand.
2. **default** branch (`runner.py:202`, used by `csp eval` / `csp train --eval`): calls
   `evaluator.evaluate_and_save(model, dataset, out)`, which does NOT compute per-image IoU.

### §6.2 Resolution and dispatch

Add near the top of `run_eval` (after `eval_cfg` resolution at `runner.py:149-166`):

```python
visualize_resolved = cfg.eval.visualize if visualize is None else visualize
```

Then:

- **Branch 1 (`return_per_example_iou=True`):** after persisting metrics, when
  `visualize_resolved` is True call `write_eval_visualizations(...)` (§7.4) with the
  `per_example_iou` already in hand, the loaded `wrapper`, `dataset`, and `out`. Return as
  today.
- **Branch 2 (default):** when `visualize_resolved` is True, this branch must obtain
  per-image IoU. **Promote the call to the `return_per_example_iou=True` shape** so it gets
  both the report and the IoU list, then mirror `evaluate_and_save`'s persistence
  (`metrics.json` + `_maybe_save_predictions`) and call `write_eval_visualizations(...)`.
  When `visualize_resolved` is False, keep the existing `evaluator.evaluate_and_save(model,
  dataset, out)` call unchanged (no IoU needed, no behavior change).

> **Implementer note (refactor, not behavior change):** the cleanest implementation routes
> both branches through one internal helper that (a) calls `evaluate` with
> `return_per_example_iou = (return_per_example_iou or visualize_resolved)`, (b) writes
> `metrics.json` + predictions exactly as `evaluate_and_save` does today (same JSON keys, same
> `_maybe_save_predictions` gate: `save_predictions and mode == "full"`), and (c) runs Phase 2
> when `visualize_resolved`. The public `run_eval` return type is unchanged: it still returns
> `tuple[MetricsReport, list[float]]` only when the caller passed `return_per_example_iou=True`,
> and `MetricsReport` otherwise. Visualization needing the IoU list internally does **not**
> change what the caller receives.

### §6.3 Output directory

The viz output dir is the **same resolved `out`** `run_eval` already computes
(`runner.py:171-178`: explicit `output_dir` → `artifacts.run_dir` → `checkpoint.parent` →
`cfg.run.output_dir` → cwd). PNGs land under `<out>/visualizations/` (§7.5). No new
output-dir logic.

### §6.4 Robustness gate

`write_eval_visualizations` is called inside a `try/except` in `run_eval` so a total viz-pass
failure (e.g. selection raised) logs a `WARNING` and never aborts the eval run after metrics
are already persisted. Per-image failures are additionally caught inside the viz pass (§8).

---

## §7 The eval visualization module

New module: **`src/custom_sam_peft/eval/visualize.py`**. It owns selection, the per-class
matched-pred extraction, the per-image render-pair, the compositor, and the top-level
`write_eval_visualizations`. It reuses `predict/visualize.py` for the shared single-panel
renderer, palette, and color map.

### §7.1 Source image — denormalize the `Example` tensor

There is **no file path on `Example`** (`data/base.py:58`: fields are `image`, `image_id`,
`prompts`, `instances`), and HF-dataset sources may have no file on disk. So the universal
approach is to **denormalize `Example.image`** (a normalized `(3, H, W)` float tensor) back
to a uint8 RGB image. Because GT masks/boxes and pred masks are all already expressed in this
tensor's pixel space (the eval transform's resized/padded canvas), overlays align with **zero
coordinate remapping**.

Denormalization uses the **config's actual mean/std**, resolved via
`resolve_normalization(model_name, normalize, channel_semantics=...)`
(`data/transforms.py:182`), **not hardcoded ImageNet** — the project supports n-channel /
custom normalization (`data/channel_semantics.py`, `NormalizeConfig`). The denorm is:
`pixel = (normalized · std + mean)`, then clamp to `[0, 1]`, scale to `[0, 255]`, cast to
uint8, and transpose `(C, H, W) → (H, W, C)`.

**n-channel (>3) rule:** for inputs with more than 3 channels, render using the **first 3
channels** of the denormalized tensor as RGB (drop the remainder). `resolve_normalization`
returns per-channel mean/std lists of the full channel count; index the first 3 for the
denorm of the rendered channels. State this rule in the module docstring. (The common case is
3-channel RGB; n-channel rendering is a best-effort preview, not a faithful multi-spectral
visualization.)

```python
def denormalize_to_rgb(
    image: torch.Tensor,        # (C, H, W) normalized float
    mean: Sequence[float],      # length C
    std: Sequence[float],       # length C
) -> Image.Image:
    """Invert normalization and return a PIL RGB image (first 3 channels when C>3)."""
```

### §7.2 GT `Instance` → render entry dict

`render_overlay` (§7.3) consumes COCO-flat entry dicts: `category_id` (1-indexed), `bbox`
`[x, y, w, h]`, optional `score`, optional `segmentation` (RLE dict with ASCII counts). GT
`Instance` (`data/base.py:45`) is `{mask: (H,W) bool, class_id: int (0-indexed), box: (4,)
xyxy pixel}`. Convert per instance:

- `category_id = class_id + 1` (1-indexed; matches `_build_coco_gt_from_examples`,
  `eval/evaluator.py:132`).
- `bbox`: xyxy → xywh, i.e. `[x1, y1, x2 − x1, y2 − y1]` from `inst.box.tolist()` (matches
  `eval/evaluator.py:127,134`).
- `segmentation`: `inst.mask` → RLE via the same encode used by the metrics path
  (`pycocotools.mask.encode(np.asfortranarray(mask.astype(uint8)))`, ASCII-decoded counts —
  mirrors `eval/postprocess.py::_logits_to_rle`). The implementer SHOULD reuse the existing
  `_mask_to_rle` (`eval/evaluator.py:91`) or `_logits_to_rle` rather than re-implement.
- **no `score` key** (GT carries no score; the optional-score renderer labels it with the
  class name only — §7.3).

```python
def gt_instances_to_entries(instances: list[Instance]) -> list[dict[str, object]]:
    """Convert GT Instances to render_overlay entry dicts (no score key)."""
```

### §7.3 Generalize `render_overlay` — optional score

`predict/visualize.py::render_overlay` (`predict/visualize.py:65-131`) currently requires a
`score` on every entry (`score = float(cast(float, entry["score"]))`, line 89) and always
labels `f"{class_name} {score:.2f}"` (line 128). Generalize so the score is **optional**:

```python
def render_overlay(
    image: Image.Image,
    entries: list[dict[str, object]],
    *,
    prompts: list[str],
) -> Image.Image:
    """... entries' `score` is OPTIONAL. When an entry has no `score` (or None),
    the label is the class name only (GT panel). When present, the label is
    `"<class> <score:.2f>"` (Pred panel). All other behavior unchanged."""
```

Behavior change (backward-compatible): read score with `entry.get("score")`; when it is
`None`/absent, the label is `class_name`; when present, the label is
`f"{class_name} {score:.2f}"`. Predict's existing call always supplies `score`, so its labels
are unchanged. The shared `color_for_class` / `PALETTE` (`predict/visualize.py:25,45`) and the
RLE mask-overlay + box-outline path are unchanged — a class therefore gets the **same color in
both panels** because both panels call the same `color_for_class(class_name)`.

> This is the only edit to `predict/visualize.py`. The `write_visualization` function there
> (which reads from a file path) is **not** used by eval and is left untouched.

### §7.4 Per-image matched render pair

For one selected `Example`, build the GT panel and the matched-Pred panel, then composite.

**GT panel:** `denormalize_to_rgb(ex.image, mean, std)` → `gt_instances_to_entries(ex.instances)`
→ `render_overlay(img, gt_entries, prompts=dataset.class_names)`. (`prompts` is the 1-indexed
class-name list; `category_id = class_id + 1` indexes it correctly.)

**Matched-Pred panel — per class, multi-class aware:** for each class in
`dataset.class_names`:

1. Run a **single-class (K=1) forward**: build a one-element prompt list
   `[TextPrompts(classes=[class_name])]` for this image and call
   `model(images_1, prompts, support=None)` — the wrapper's forward signature is
   `forward(images, prompts, support=SupportPrompts | None)` (`models/sam3.py:225`; the
   removed `box_hints=` kwarg is now `support=`). `images_1` is `ex.image` moved to the
   model's device with a leading batch dim (`Runtime` / `to_device`, as the evaluator does at
   `eval/evaluator.py:191,205`).
2. `canonical = meta_to_canonical(outputs)` (`models/matching.py:39`) → `CanonicalOutputs` for
   this class (B=1).
3. Build the per-class GT target list: the image's `Instance`s whose `class_id` equals this
   class's dense index (`dataset.class_names.index(class_name)`), as `list[Instance]`.
4. Call `matcher(canonical, [targets])` (`models/matching.py:120`, returns a per-image list of
   `(query_idx, target_idx)` tensors) → take element `[0]`. This is **1:1 with the GT masks of
   this class** (one query assigned per GT).
5. For each matched `(query_idx, target_idx)` pair, convert that query's outputs into a render
   entry: mask logits → threshold at `cfg.mask_threshold` and **upsample to image
   resolution**, box → xywh, score → `sigmoid(obj_logit) · sigmoid(presence)`. Reuse the eval
   postprocess helpers: `queries_to_coco_results(_row_outputs(outputs, …), image_id, cat_idx+1,
   original_hw, mask_threshold)` produces COCO-flat entries for **all** queries of this
   class in exactly the score/box/mask form the renderer needs; then **select only the matched
   query rows** (by `query_idx`). The score formula is identical to the metrics path
   (`eval/postprocess.py:85-87`: `sigmoid(pred_logits) · sigmoid(presence_logit_dec)`).

Aggregate the matched entries across all classes → the full multi-class set for the image,
1:1 with GT masks. **Draw matched predictions ONLY** (no unmatched/extra detections).

**Matcher construction:** instantiate with `MatcherWeights()` defaults
(`config/_internal.py:32`: `lambda_l1=0.0`, `lambda_giou=0.0`, `lambda_mask=5.0`) →
**mask-only Dice matching**:

```python
from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.models.matching import HungarianMatcher
w = MatcherWeights()
matcher = HungarianMatcher(lambda_l1=w.lambda_l1, lambda_giou=w.lambda_giou, lambda_mask=w.lambda_mask)
```

(`HungarianMatcher.__init__` takes the three lambdas positionally/by-keyword;
`losses/compose.py:133` is the existing construction precedent.) Because only `lambda_mask`
is nonzero, **the absolute lambda scale is irrelevant** — the cost is `lambda_mask · dice`,
and the assignment `argmin` is invariant to the positive scalar `lambda_mask`. State this.

> **GT box representation is irrelevant to matching here (locked note).** The matcher reads
> target boxes as normalized cxcywh (`models/matching.py:148-153`), but GT `Instance.box` is
> xyxy pixel (`data/base.py:55`). In training the collator normalizes boxes before the matcher;
> the viz pass passes raw `Instance`s. This is **safe** because the box cost terms
> (`lambda_l1`, `lambda_giou`) are both `0`, so the box representation never enters the
> matching cost — only the mask Dice term matters. State this so the unconverted GT box is not
> read as a bug. (The matcher does still read `t.mask` for the Dice term, which is correct
> `(H, W)` bool.)

Per-image render-pair signature:

```python
def render_eval_pair(
    model: Any,
    example: Example,
    class_names: list[str],
    *,
    mask_threshold: float,
    mean: Sequence[float],
    std: Sequence[float],
    matcher: HungarianMatcher,
) -> Image.Image:
    """Return the hstacked `Ground Truth | Prediction` composite for one image."""
```

### §7.5 Compositor & output

The compositor renders the GT panel and the Pred panel via the shared `render_overlay`, then
**hstacks** them into one image with:

- panel titles `"Ground Truth"` (left) and `"Prediction"` (right), drawn above/within each
  panel (PIL `ImageDraw.text`, `ImageFont.load_default()`, consistent with
  `predict/visualize.py:127`), and
- a **shared per-class color legend** (the union of classes present in either panel), each
  legend row a `color_for_class(class_name)` swatch + class name, so the reader can map color
  → class. The legend uses the same palette/color map as both panels.

Output path: `<out>/visualizations/<image_id>.png` — one composite per selected image,
mirroring predict's `visualizations/` convention (`predict/visualize.py:155-162`). Eval has no
file stem, so use the `Example.image_id` string; **sanitize** it for the filesystem (replace
path separators and characters illegal in filenames with `_`) so an `image_id` like a URL or
nested path yields a valid single-segment filename. The `visualizations/` subdir is created if
absent (`mkdir(parents=True, exist_ok=True)`).

### §7.6 Top-level entry point

```python
def write_eval_visualizations(
    model: Any,
    dataset: Dataset,
    output_dir: Path,
    *,
    per_example_iou: Sequence[float],
    count: int,
    mask_threshold: float,
    model_name: str,
    normalize: NormalizeConfig | None,
    channel_semantics: str,
) -> list[Path]:
    """Phase-2 viz pass. Selects `count` variety-weighted images (§5), renders a
    GT-vs-Pred composite per image (§7.4-7.5), writes PNGs under
    output_dir/visualizations/, and returns the written paths. Memory-bounded:
    processes and frees one image at a time. Per-image failures are caught and
    logged (§8); never raises for a single bad image."""
```

It resolves `(mean, std)` once via `resolve_normalization(model_name, normalize,
channel_semantics=channel_semantics)`, constructs the mask-only `HungarianMatcher` once, runs
`pick_samples(per_example_iou, dataset, count)` (§5), and loops the selected indices, calling
`render_eval_pair` and saving each PNG under `output_dir/visualizations/`. The model is
already loaded and (for the standalone path) adapter-loaded by `run_eval`; the viz pass puts
it in eval mode and runs under `torch.no_grad()`.

---

## §8 Error handling & edge cases

| Condition | Behavior |
|-----------|----------|
| Matcher or render fails on a **single** image | Caught and logged at `WARNING`, the loop continues to the next image. Never crashes the eval run — mirrors predict's defensive mask render (`predict/visualize.py:113`, `except Exception: logger.debug`). Eval visualization uses `WARNING` (not `debug`) for whole-image failures so the user notices a missing panel. |
| Whole viz pass fails (e.g. selection raises) | Caught in `run_eval`'s `try/except` (§6.4); `WARNING` logged; metrics already persisted; eval returns normally. |
| Image has a GT class but **no good query** | The matcher still assigns the best available query per GT (`linear_sum_assignment` always returns a 1:1 assignment when targets exist). That (possibly poor) pred is drawn — this is **informative** (it shows where the model's best guess for that GT is, even if low-quality). Not skipped. |
| Class with **no GT** in an image | Its per-class target list is empty; `matcher(canonical, [[]])` returns empty index tensors (`models/matching.py:131-138`). No matched preds are drawn for that class. (Correct: the panel compares against GT only.) |
| `visualize_count` ≥ candidate count | Take all candidates (§5.3 small-pool rule). Fewer than `count` composites written — not an error. |
| **Zero** candidates (no image has GT) | `pick_samples` returns `[]`; the viz pass writes no PNGs and logs an INFO note ("no GT-bearing images to visualize"). Not an error. |
| `eval.mode == "lite"` | Phase 1 already restricts to the first `lite_max_images` examples (`eval/evaluator.py:355`), so `per_example_iou` is length `lite_max_images` and aligned to those indices. `pick_samples` selects from that subset — consistent and correct; no special-casing. |
| Baseline (zero-shot) eval (`checkpoint=None`) | Works unchanged — the model is loaded without an adapter (`eval/runner.py:132-145`) and the viz pass forwards it the same way. Panels show baseline preds vs GT. |
| `NaN` per-image IoU (skipped example) | Ranked as `-inf` (§5.2); eligible only as a "worst" pick. |
| n-channel (>3) input | Render first 3 channels as RGB (§7.1). |

---

## §9 Interfaces (for the downstream planner)

### New module `src/custom_sam_peft/eval/visualize.py`

| Symbol | Signature | Role |
|--------|-----------|------|
| `pick_samples` | `(per_example_iou: Sequence[float], dataset: Dataset, count: int) -> list[int]` | §5 variety-weighted selection over GT-bearing candidates. |
| `denormalize_to_rgb` | `(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> Image.Image` | §7.1 config-aware denorm; first-3-channel rule for C>3. |
| `gt_instances_to_entries` | `(instances: list[Instance]) -> list[dict[str, object]]` | §7.2 GT → render entry dicts (no score). |
| `render_eval_pair` | `(model, example, class_names, *, mask_threshold, mean, std, matcher) -> Image.Image` | §7.4 per-image matched composite. |
| `write_eval_visualizations` | see §7.6 | top-level Phase-2 entry, called by `run_eval`. |

(Internal helpers — the compositor, the matched-pred extraction, image-id sanitization — are
the implementer's to structure; the table above is the module's public surface.)

### Generalized shared renderer

- `predict/visualize.py::render_overlay` — score becomes optional (§7.3). Signature is
  unchanged in arity; the behavior change is `entry.get("score")` instead of
  `entry["score"]`. Shared by predict (unchanged call) and eval (no-score GT entries).
- `predict/visualize.py::color_for_class` and `PALETTE` — reused as-is by both panels and the
  legend.

### Config additions (§4.1)

- `EvalConfig.visualize: bool = True`
- `EvalConfig.visualize_count: PositiveInt = 10`

### CLI / runner additions

- `cli/eval_cmd.py::evaluate` — `visualize: bool | None` `--visualize/--no-visualize` (§4.2).
- `cli/run_cmd.py::run` — `visualize: bool = True` `--visualize/--no-visualize`, threaded
  through `_orchestrate` to the eval `run_eval` call (§4.3).
- `eval/runner.py::run_eval` — new `visualize: bool | None = None` keyword (both overloads +
  impl), resolved against `cfg.eval.visualize` and dispatched per §6.

---

## §10 Testing strategy

CPU-only. Every case here is CPU-testable against the tiny SAM3 stub / tiny multi-class
dataset; no GPU tests are added.

### Unit tests (new)

| Coverage | Assertions |
|----------|-----------|
| **`pick_samples` band sizes** | For `N ∈ {1, 2, 5, 10, 20}` against a synthetic `per_example_iou` + a GT-filter stub: `good = round(0.5N)`, `worst = min(2, max(1, round(0.2N)))`, `median = N − good − worst`; for `N=10` → exactly 5/3/2. |
| **`pick_samples` worst-cap rule** | For large `N` (e.g. 20) the worst band stays capped at 2. |
| **`pick_samples` within-band spread / dedup** | Picks within a band are spread (not adjacent), and no index appears in two bands. |
| **`pick_samples` GT filter** | No-GT images (empty `instances`) are never selected, even with high (vacuous-1.0) IoU. |
| **`pick_samples` N ≥/≤ candidate count** | `N ≤ candidates` → exactly the band split; `N ≥ candidates` (small pool) → all candidates returned, `len ≤ N`. |
| **`pick_samples` NaN handling** | A `NaN` IoU sorts to the bottom and is only ever a "worst" pick. |
| **`render_overlay` score-optional** | An entry WITH `score` → label `"<class> 0.42"`; an entry WITHOUT `score` (GT) → label `"<class>"`; predict's existing scored call still labels with score (regression guard). |
| **`denormalize_to_rgb` round-trip** | Normalize a known uint8 image with config-aware (mean, std), denorm, assert pixel round-trip within rounding tolerance; assert C>3 input renders first 3 channels. |
| **`gt_instances_to_entries` conversion** | `category_id == class_id + 1`; `bbox == [x1, y1, x2−x1, y2−y1]`; `segmentation` is a valid RLE decoding back to the input mask; no `score` key present. |
| **compositor** | Output image width ≈ left + right panel widths (hstacked); both panel titles present; legend lists the union of classes with the same `color_for_class` colors as the panels. |

### Integration test (new, CPU smoke)

- A tiny **multi-class** dataset (≥ 2 classes, a couple of GT-bearing images) through
  `run_eval(..., visualize=True)` (or `visualize` left to the default) using the tiny SAM3
  stub: assert that **N composite PNGs** (capped at candidate count) land in
  `<out>/visualizations/`, named by sanitized `image_id`, and that each is a readable image.
- **In-loop guard:** assert that calling `Evaluator.evaluate(...)` directly (the in-loop path)
  produces **no** `visualizations/` directory and writes no composite — i.e. the viz pass is
  wired only to the `run_eval` / `evaluate_and_save` surface (§3).
- `--no-visualize` (CLI override / `visualize=False`) writes **no** `visualizations/`.

---

## §11 Non-goals already satisfied — `predict` is done

`predict --visualize` **already renders multiple classes in one image** (verified):

- `predict/visualize.py::render_overlay` loops over **all** entries, coloring each by its
  class via `color_for_class(class_name)` (`predict/visualize.py:87-95`) — multi-class in one
  panel.
- `predict/runner.py` groups **all** predictions per image before rendering
  (`predict/runner.py:517-525`: `by_image.setdefault(iid, []).append(entry)` then one
  `write_visualization` per image with all that image's entries).

So **no predict behavior change and no GitHub issue.** The only `predict/visualize.py` edit is
the additive optional-score parameter on `render_overlay` (§7.3), which is backward-compatible
with predict's always-scored call.

---

## §12 Acceptance criteria

Concrete and checkable.

1. `EvalConfig` has `visualize: bool = True` and `visualize_count: PositiveInt = 10`
   (`config/schema.py:592`).
2. `csp eval` has `--visualize/--no-visualize` (`bool | None`, default `None`) mirroring
   `save_predictions`, threaded into `run_eval`.
3. `csp run` has `--visualize/--no-visualize` (default `True`), threaded through `_orchestrate`
   into the eval-phase `run_eval`.
4. `csp train --eval` adds **no** new flag and picks up `cfg.eval.visualize` (default True).
5. `run_eval` has `visualize: bool | None = None` (both overloads + impl), resolved against
   `cfg.eval.visualize`; visualization runs only on the `run_eval` / `evaluate_and_save`
   surface, never inside `Evaluator.evaluate`.
6. New `src/custom_sam_peft/eval/visualize.py` exposes `pick_samples`, `denormalize_to_rgb`,
   `gt_instances_to_entries`, `render_eval_pair`, `write_eval_visualizations` with the §9
   signatures.
7. `pick_samples` filters to GT-bearing candidates, ranks by per-image IoU (NaN → -inf), and
   yields the 5/3/2 good/median/worst spread at `N=10` with within-band spacing and the
   worst-cap-at-2 rule; returns all candidates when the pool ≤ N.
8. `render_overlay` (`predict/visualize.py`) makes `score` optional; GT entries (no score)
   label the class name only; scored entries label `"<class> <score>"`; predict's existing
   call is unchanged.
9. Each composite is `Ground Truth | Prediction` hstacked, with panel titles and a shared
   per-class color legend; a class is the **same color in both panels**.
10. The Pred panel draws the **Hungarian mask-only matched 1:1** set per class (built with
    `MatcherWeights()` defaults), aggregated across all classes; no unmatched detections.
11. Source images are **denormalized from `Example.image`** using config-aware (mean, std) via
    `resolve_normalization`; C>3 renders first 3 channels.
12. PNGs land at `<out>/visualizations/<sanitized image_id>.png`; the in-loop eval path writes
    none.
13. Single-image matcher/render failures are caught + logged and never crash the eval run; a
    whole-pass failure is caught in `run_eval` after metrics persist.
14. `docs/config-schema.md` gains `eval.visualize` and `eval.visualize_count` rows; `CHANGELOG.md`
    has the feature entry; no JSON-schema artifact exists to regenerate.
15. CPU unit + integration tests in §10 exist and pass, including the in-loop-no-viz guard and
    the `--no-visualize` guard.
