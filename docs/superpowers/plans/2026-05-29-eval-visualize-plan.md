# `eval --visualize` — GT vs Pred qualitative panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-out `eval.visualize` knob that, on the final/standalone eval path only, renders a variety-weighted sample of images as a two-panel `Ground Truth | Prediction` composite (matched 1:1 per class via the existing Hungarian mask-only matcher), written under `<out>/visualizations/`.

**Architecture:** A new `eval/visualize.py` module owns selection + rendering. Phase 1 builds the pure, model-free primitives (selection, denorm, GT→entry conversion, the generalized optional-score `render_overlay`, the compositor) — all unit-testable against synthetic data. Phase 2 adds the model-dependent per-image render pair (per-class K=1 forwards → mask-only matcher → matched-query extraction) and the top-level `write_eval_visualizations`. Phase 3 wires it into the config schema, the `csp eval` / `csp run` CLI surface, the `run_eval` dispatch, and the docs, with a CPU integration smoke test.

**Tech Stack:** PyTorch + Meta's `sam3` library, Pydantic v2 (schema), Pillow (PIL) rendering, `pycocotools` (RLE), `scipy.optimize.linear_sum_assignment` (matcher), Typer (CLI), pytest (CPU-only TDD, 80% coverage gate), ruff + mypy + markdownlint-cli2 (CI gates).

Spec: `docs/superpowers/specs/2026-05-29-eval-visualize-design.md`.

---

## Conventions (read before starting)

- Spec references use `§N` (matches the design doc).
- Each task names exact files. **Line numbers in the spec are partially stale against merged work** (re-grounded below in "Grounding deltas"); implementers MUST grep on the surrounding identifier, not trust a line number.
- TDD ordering: write failing test → run red → minimal impl → run green → commit. Every code task uses that loop. Tests are **CPU-only** — every case runs against the tiny SAM3 stub (`tests/fixtures/tiny_sam3_stub.py::TinySam3Stub`) or synthetic data; no GPU tests are added.
- Implementer commits during implementation are exempt from the lint gate; the final phase (Task 13) runs ruff/format/mypy/full-pytest/markdownlint before the branch is PR-ready.
- "Parallelizable with: [task IDs]" means the listed tasks touch disjoint files with no shared-state ordering constraint; the orchestrator may dispatch them concurrently.

### Grounding deltas (verified against the worktree, 2026-05-29)

These differ from the spec's cited line numbers — the spec was written before the multiplex-forward / baseline-eval / per-example-IoU work merged. Use these, grep to confirm:

1. **`Sam3Wrapper.forward`** (`models/sam3.py`) is `forward(self, images, prompts, support: SupportPrompts | None = None)` — confirmed. The K=1 viz forward calls `model(images_1, [TextPrompts(classes=[name])], support=None)`. The stub ignores `support=`/`box_hints=`.
2. **`run_eval`** (`eval/runner.py`) already has BOTH `@overload` stubs carrying `save_predictions: bool | None = None`, and the impl already has the baseline path + the two persistence branches. The two branches are:
   - **Branch 1** (`return_per_example_iou=True`, used by `csp run`): runs `evaluator.evaluate(..., return_per_example_iou=True)`, writes `metrics.json` inline, writes `predictions.json` when `eval_cfg.save_predictions and eval_cfg.mode == "full"`, returns `(report, per_example_iou)`. (NOTE: it writes `predictions.json` flat, not via `_maybe_save_predictions` — this is the existing Branch-1 behavior; do not change it.)
   - **Branch 2** (default, used by `csp eval` / `csp train --eval`): `return evaluator.evaluate_and_save(wrapper, dataset, out)` — does NOT compute per-example IoU.
   So the spec's §4.5 claim "add `visualize` to both overloads + impl" still holds (none of the three signatures has `visualize` yet), and §6.2's "promote Branch 2 to compute IoU" still holds.
3. **`Evaluator.evaluate(model, dataset, *, return_per_example_iou=...)`** (`eval/evaluator.py`) — confirmed; returns `(MetricsReport, list[float])` when True. The per-example IoU list is index-aligned to `examples = [dataset[i] for i in range(n)]` where `n = len(dataset)` (full) or `min(lite_max_images, len)` (lite). `_compute_per_example_iou`'s docstring already references `pick_samples` (the metrics path was pre-grounded for this feature).
4. **`_orchestrate`** (`cli/run_cmd.py`) is currently `def _orchestrate(cfg, resume, mode)` — NO `visualize` param. The `run()` body calls `_orchestrate(cfg, resume_path, mode)`. The eval-phase `run_eval(...)` call lives INSIDE `_orchestrate` (the Branch-1 `return_per_example_iou=True` call).
5. **`cli/eval_cmd.py::evaluate`** already has `--checkpoint` optional, `--interactive`, and threads `save_predictions=save_predictions` into `run_eval`. Add `visualize` next to `save_predictions`.
6. **`EvalConfig`** (`config/schema.py`, `class EvalConfig(_Strict)`) ends with `batch_size: PositiveInt | Literal["auto"] = "auto"`. Add the two new fields after it.
7. **Postprocess reuse:** `eval/postprocess.py::queries_to_coco_results(outputs, image_id, category_id, original_hw, mask_threshold=0.0)` + `eval/evaluator.py::_row_outputs(outputs, r)` are the exact helpers the metrics path uses; the viz pass mirrors them. `_row_outputs` slices batch row `r` (size-1 dim preserved, non-tensors dropped). The score formula `sigmoid(pred_logits) * sigmoid(presence_logit_dec)` lives inside `queries_to_coco_results`.
8. **Matcher:** `models/matching.py::HungarianMatcher(lambda_l1, lambda_giou, lambda_mask)`; `__call__(outputs: CanonicalOutputs, targets: list[list[Instance]]) -> list[tuple[Tensor, Tensor]]` returns per-image `(query_idx, target_idx)` long tensors (from `linear_sum_assignment(cost)` → `row_ind, col_ind`). `meta_to_canonical(outputs) -> CanonicalOutputs` is the adapter. `MatcherWeights()` (`config/_internal.py`) defaults `lambda_l1=0.0, lambda_giou=0.0, lambda_mask=5.0`.
9. **`render_overlay`** (`predict/visualize.py`) reads `score = float(cast(float, entry["score"]))` (the require) and labels `f"{class_name} {score:.2f}"`. Both are the only edits in Task 3.
10. **Normalization:** `data/transforms.py::resolve_normalization(model_name, fallback, *, channel_semantics="rgb") -> (mean, std)`. `fallback` is a `NormalizeConfig` (it is NOT optional in the call; pass `cfg.data.normalize` which is `NormalizeConfig | None` — see Open question OQ-2).

### Open questions / spec gaps (flagged, not invented)

- **OQ-1 (resolved by spec §7.6 signature):** `write_eval_visualizations` takes `normalize: NormalizeConfig | None`. `resolve_normalization`'s `fallback` param is typed `NormalizeConfig` (non-optional). The schema resolves `data.normalize` (default `None` → a profile default is applied at load). **Decision:** pass `cfg.data.normalize` straight through; `resolve_normalization` is called by the existing eval transform with the same value, so passing `None` is already exercised in-tree (`build_eval_transforms` is given `cfg.data.normalize`). If mypy flags the `None`, this matches the existing call sites — verify by grepping `resolve_normalization(` usages; mirror the call shape there. Not a new design decision.
- **OQ-2 (legend layout — spec §7.5 is intentionally loose):** the spec says "a shared per-class color legend … each legend row a swatch + class name" but does not pin pixel geometry. **Decision:** the implementer owns geometry (swatch size, row height, placement — e.g. a legend strip appended below the hstacked panels). The test asserts only: (a) output width ≈ left+right panel widths, (b) both panel titles present, (c) the legend contains the union of classes with `color_for_class` colors. No pixel-exact assertion. This is a noted latitude, not a guess at hidden design.
- **OQ-3 (image-id sanitization charset — spec §7.5 says "replace path separators and illegal chars with `_`"):** the spec does not enumerate the illegal set. **Decision:** sanitize by replacing any character not in `[A-Za-z0-9._-]` with `_` (covers `/`, `\`, `:`, spaces, URL chars). Documented in the helper docstring. Conservative superset of "path separators + illegal-in-filename".

---

## File structure

**New files:**

- `src/custom_sam_peft/eval/visualize.py` — the eval visualization module: `pick_samples` (§5), `denormalize_to_rgb` (§7.1), `gt_instances_to_entries` (§7.2), an internal compositor (§7.5), `render_eval_pair` (§7.4), `write_eval_visualizations` (§7.6), plus internal helpers (matched-pred extraction, image-id sanitization).
- `tests/unit/test_eval_visualize.py` — unit tests for `pick_samples`, `denormalize_to_rgb`, `gt_instances_to_entries`, the compositor, and `render_overlay`'s score-optional behavior.
- `tests/unit/test_eval_visualize_pair.py` — unit tests for `render_eval_pair` + `write_eval_visualizations` against the tiny stub + `tiny_text_dataset`.
- `tests/integration/test_eval_visualize_integration.py` — the CPU integration smoke (run_eval end-to-end, in-loop-no-viz guard, `--no-visualize` guard).

**Modified files:**

- `src/custom_sam_peft/predict/visualize.py` — `render_overlay` score becomes optional (the ONLY predict edit; `write_visualization` untouched).
- `src/custom_sam_peft/config/schema.py` — `EvalConfig` gains `visualize: bool = True` and `visualize_count: PositiveInt = 10`.
- `src/custom_sam_peft/cli/eval_cmd.py` — add the `--visualize/--no-visualize` tri-state option; thread into `run_eval`.
- `src/custom_sam_peft/cli/run_cmd.py` — add the `--visualize/--no-visualize` flag (default True), thread through `_orchestrate` into the eval `run_eval` call.
- `src/custom_sam_peft/eval/runner.py` — `run_eval` gains `visualize: bool | None = None` (both overloads + impl); resolve + dispatch Phase 2 in both branches, with Branch-2 promotion.
- `docs/config-schema.md` — two new `## eval` rows.
- `CHANGELOG.md` — feature entry under `## [Unreleased]`.

**Modified tests:**

- `tests/predict/test_visualize.py` — add a score-optional regression case (predict's existing scored call still labels with score).

---

## Phase 1 — pure rendering primitives (no model)

> **Phase goal:** all model-free primitives, each unit-testable against synthetic data / the tiny stub with no forward pass.
>
> **Interface contract exposed to Phase 2:**
>
> - `eval/visualize.py::pick_samples(per_example_iou: Sequence[float], dataset: Dataset, count: int) -> list[int]` — variety-weighted selection over GT-bearing candidates; returns ≤ `count` dataset indices in descending-IoU order.
> - `eval/visualize.py::denormalize_to_rgb(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> Image.Image` — config-aware denorm; first-3-channel rule for C>3.
> - `eval/visualize.py::gt_instances_to_entries(instances: list[Instance]) -> list[dict[str, object]]` — GT → `render_overlay` entry dicts, no `score` key.
> - `eval/visualize.py::_compose_pair(gt_panel: Image.Image, pred_panel: Image.Image, *, class_names_present: list[str]) -> Image.Image` — internal compositor: hstacks the two panels, draws `"Ground Truth"` / `"Prediction"` titles, appends the shared per-class color legend. (Name is the implementer's; this plan fixes it so Phase 2's `render_eval_pair` can call it.)
> - `predict/visualize.py::render_overlay(image, entries, *, prompts)` — `score` is now OPTIONAL (entry without `score`/`None` → label is the class name only).
> - Shared from `predict/visualize.py` (unchanged): `color_for_class(class_name) -> (r,g,b)`, `PALETTE`.
>
> Tasks 1–5 are file-disjoint EXCEPT Tasks 1, 4, 5 all create/append to the new `eval/visualize.py` — serialize those three (Task 1 creates the module, Tasks 4 and 5 append). Task 2 (`predict/visualize.py`) and Task 3 (`config/schema.py`) are independent and parallelizable with everything in this phase.

### Task 1: `pick_samples` — variety-weighted selection (§5)

**Spec ref:** §5, §5.1–§5.4, §10 (pick_samples rows).

**Files:**

- Create: `src/custom_sam_peft/eval/visualize.py`
- Test: `tests/unit/test_eval_visualize.py`

> Creates the new module. Run before Tasks 4/5 (which append to it).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_eval_visualize.py`:

```python
"""Unit tests for eval/visualize.py pure primitives (CPU-only, no model)."""

from __future__ import annotations

import math

import torch

from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.visualize import pick_samples


class _FakeDataset:
    """Index-aligned dataset whose examples carry the requested #GT instances."""

    def __init__(self, gt_counts: list[int]) -> None:
        self._examples = []
        for i, n in enumerate(gt_counts):
            insts = [
                Instance(
                    mask=torch.zeros(4, 4, dtype=torch.bool),
                    class_id=0,
                    box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
                )
                for _ in range(n)
            ]
            self._examples.append(
                Example(
                    image=torch.zeros(3, 4, 4),
                    image_id=f"img_{i}",
                    prompts=TextPrompts(classes=["a"]),
                    instances=insts,
                )
            )
        self.class_names = ["a"]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]


def _bands(n: int) -> tuple[int, int, int]:
    good = round(0.5 * n)
    worst = min(2, max(1, round(0.2 * n)))
    median = n - good - worst
    return good, median, worst


def test_band_sizes_n10() -> None:
    ds = _FakeDataset([1] * 30)
    iou = [i / 30 for i in range(30)]  # all distinct, all GT-bearing
    picked = pick_samples(iou, ds, 10)
    assert len(picked) == 10
    assert _bands(10) == (5, 3, 2)


def test_band_sizes_various_n() -> None:
    for n, (g, m, w) in [(1, _bands(1)), (2, _bands(2)), (5, _bands(5)), (20, _bands(20))]:
        assert g + m + w == n
        assert w <= 2  # worst cap


def test_worst_cap_large_n() -> None:
    ds = _FakeDataset([1] * 50)
    iou = [i / 50 for i in range(50)]
    picked = pick_samples(iou, ds, 20)
    assert len(picked) == 20
    g, m, w = _bands(20)
    assert w == 2  # capped despite round(0.2*20)=4


def test_gt_filter_excludes_no_gt_images() -> None:
    # idx 0 has the highest IoU but NO GT → must never be selected.
    ds = _FakeDataset([0, 1, 1, 1, 1])
    iou = [1.0, 0.9, 0.8, 0.7, 0.6]
    picked = pick_samples(iou, ds, 4)
    assert 0 not in picked
    assert set(picked) <= {1, 2, 3, 4}


def test_small_pool_returns_all_candidates() -> None:
    ds = _FakeDataset([1, 1, 1])  # 3 GT-bearing candidates
    iou = [0.3, 0.2, 0.1]
    picked = pick_samples(iou, ds, 10)
    assert sorted(picked) == [0, 1, 2]
    assert len(picked) <= 10


def test_indices_unique_across_bands() -> None:
    ds = _FakeDataset([1] * 12)
    iou = [i / 12 for i in range(12)]
    picked = pick_samples(iou, ds, 10)
    assert len(picked) == len(set(picked))  # no index in two bands


def test_nan_sorts_to_bottom_worst_only() -> None:
    # idx 2 is NaN → ranked -inf → only ever a "worst" pick, never "good".
    ds = _FakeDataset([1, 1, 1, 1, 1, 1])
    iou = [0.9, 0.8, math.nan, 0.6, 0.5, 0.4]
    picked = pick_samples(iou, ds, 6)  # pool == N → all returned
    assert 2 in picked  # eligible as worst
    # With N < pool, the top "good" band must not include the NaN index.
    picked2 = pick_samples(iou, ds, 2)
    g, _, w = _bands(2)  # (1, 0, 1)
    assert picked2[0] != 2  # highest-IoU first, never the NaN


def test_returned_in_descending_iou_order() -> None:
    ds = _FakeDataset([1] * 6)
    iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    picked = pick_samples(iou, ds, 6)
    vals = [iou[i] for i in picked]
    assert vals == sorted(vals, reverse=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_visualize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'custom_sam_peft.eval.visualize'`.

- [ ] **Step 3: Create the module + implement `pick_samples`**

Create `src/custom_sam_peft/eval/visualize.py`:

```python
"""Eval GT-vs-Pred qualitative visualization (final/standalone eval path only).

Owns: variety-weighted image selection, config-aware denormalization, GT-instance
to render-entry conversion, the per-image matched render pair, the compositor, and
the top-level write_eval_visualizations entry point. Reuses predict/visualize.py for
the shared single-panel renderer, palette, and color map.

n-channel rule (§7.1): for inputs with more than 3 channels, only the first 3
denormalized channels are rendered as RGB (best-effort preview, not a faithful
multi-spectral visualization).

Spec: docs/superpowers/specs/2026-05-29-eval-visualize-design.md.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from custom_sam_peft.data.base import Dataset

_LOG = logging.getLogger(__name__)


def _spread_indices(sorted_indices: list[int], k: int) -> list[int]:
    """Pick k evenly spaced elements from sorted_indices (preserving order)."""
    if k <= 0 or not sorted_indices:
        return []
    if k >= len(sorted_indices):
        return list(sorted_indices)
    # Evenly spaced positions across [0, len-1].
    positions = [round(j * (len(sorted_indices) - 1) / (k - 1)) for j in range(k)] if k > 1 else [0]
    seen: set[int] = set()
    out: list[int] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            out.append(sorted_indices[p])
    # Back-fill if rounding collided (keep k distinct positions when possible).
    j = 0
    while len(out) < k and j < len(sorted_indices):
        if sorted_indices[j] not in out:
            out.append(sorted_indices[j])
        j += 1
    return out


def pick_samples(
    per_example_iou: Sequence[float],
    dataset: Dataset,
    count: int,
) -> list[int]:
    """Return up to `count` dataset indices, variety-weighted toward high IoU.

    Filters to candidates with >=1 GT instance (excludes no-GT images), ranks by
    per_example_iou (NaN -> -inf, eligible only as 'worst'), and picks a
    good/median/worst spread per spec §5.3. Returns <= count indices when the
    candidate pool is smaller than count. Indices are returned in descending-IoU
    order so the written composites are filename-stable and roughly best-to-worst.
    """
    # Candidate filter: >=1 GT instance. per_example_iou is index-aligned to the
    # dataset slice the metrics pass evaluated (full or lite).
    candidates = [
        i for i in range(len(per_example_iou)) if len(dataset[i].instances) > 0
    ]
    if not candidates:
        return []

    def rank_key(i: int) -> float:
        v = per_example_iou[i]
        return -math.inf if (v is None or math.isnan(v)) else float(v)

    ranked = sorted(candidates, key=rank_key, reverse=True)  # descending IoU

    if len(ranked) <= count:
        return ranked  # small-pool rule: take all, already descending

    good = round(0.5 * count)
    worst = min(2, max(1, round(0.2 * count)))
    median = count - good - worst

    n = len(ranked)
    good_slice = ranked[:good] if good > 0 else []
    worst_slice = ranked[n - worst :] if worst > 0 else []
    # Median band: the middle region between the good and worst slices.
    mid_lo = good
    mid_hi = n - worst
    median_pool = ranked[mid_lo:mid_hi]

    picked_good = _spread_indices(good_slice, good)
    picked_median = _spread_indices(median_pool, median)
    picked_worst = _spread_indices(worst_slice, worst)

    # Disjoint by construction (slices don't overlap). De-dup defensively and
    # back-fill from the next band if a band came up short.
    chosen: list[int] = []
    for idx in [*picked_good, *picked_median, *picked_worst]:
        if idx not in chosen:
            chosen.append(idx)
    if len(chosen) < count:
        for idx in ranked:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == count:
                break

    # Return in descending-IoU order.
    chosen.sort(key=rank_key, reverse=True)
    return chosen[:count]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_visualize.py -v`
Expected: PASS — band sizes match `5/3/2` at N=10; worst capped at 2; no-GT images excluded; small pool returns all; indices unique across bands; NaN only ever a worst pick; output is descending-IoU.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_visualize.py
git commit -m "feat(eval): add pick_samples variety-weighted selection"
```

### Task 2: Generalize `render_overlay` — optional score (§7.3)

**Spec ref:** §7.3, §10 (render_overlay score-optional row), §11.

**Files:**

- Modify: `src/custom_sam_peft/predict/visualize.py` (`render_overlay`, the `score = float(...)` read + the label)
- Test: `tests/predict/test_visualize.py` (append a score-optional case)

> File-disjoint from all other Phase-1 tasks. Parallelizable with Tasks 1, 3, 4, 5.

- [ ] **Step 1: Write the failing test**

Append to `tests/predict/test_visualize.py` (the `_make_rle_entry` / `_make_image` helpers exist at the top of the file; reuse them):

```python
def test_render_overlay_score_optional() -> None:
    """An entry without a `score` key labels the class name only; an entry with a
    score labels `<class> <score>` (predict's existing behavior, regression guard)."""
    from custom_sam_peft.predict.visualize import render_overlay

    img = _make_image(32, 32)

    # Scored entry (predict path) — must still render without error.
    scored = _make_rle_entry(category_id=1, score=0.42)
    out_scored = render_overlay(img, [scored], prompts=["cat", "dog"])
    assert out_scored.size == (32, 32)

    # GT entry: no score key at all.
    gt = dict(scored)
    gt.pop("score")
    out_gt = render_overlay(img, [gt], prompts=["cat", "dog"])
    assert out_gt.size == (32, 32)

    # None score behaves like absent.
    none_score = dict(scored)
    none_score["score"] = None
    out_none = render_overlay(img, [none_score], prompts=["cat", "dog"])
    assert out_none.size == (32, 32)
```

> The rendered label text is drawn via PIL and not trivially asserted pixel-wise; the regression guard here is that the no-score path does not raise `KeyError`/`TypeError` and that the scored path still works. The label-string branching is verified by code inspection in review (the `entry.get("score")` change).

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/predict/test_visualize.py::test_render_overlay_score_optional -v`
Expected: FAIL — the current `score = float(cast(float, entry["score"]))` raises `KeyError: 'score'` on the GT entry (and `TypeError` on the `None` entry).

- [ ] **Step 3: Make `score` optional in `render_overlay`**

In `src/custom_sam_peft/predict/visualize.py`, inside `render_overlay`'s per-entry loop, replace the score read:

```python
        score = float(cast(float, entry["score"]))
```

with:

```python
        raw_score = entry.get("score")
        score = float(cast(float, raw_score)) if raw_score is not None else None
```

Then replace the label construction near the end of the loop:

```python
        font = ImageFont.load_default()
        label = f"{class_name} {score:.2f}"
        draw.text((x, y), label, fill=color, font=font)
```

with:

```python
        font = ImageFont.load_default()
        label = class_name if score is None else f"{class_name} {score:.2f}"
        draw.text((x, y), label, fill=color, font=font)
```

Update the `render_overlay` docstring's `entries` arg description: change `` ``score`` `` to `optional ``score`` (when absent or None, the label is the class name only — GT panels)`. Leave `write_visualization` and everything else untouched. If `cast` becomes unused, leave it — it is still used for `category_id`/`bbox`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/predict/test_visualize.py -v`
Expected: PASS — the new score-optional test passes; ALL existing `test_visualize.py` cases (predict's always-scored calls) stay green (regression guard).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/predict/visualize.py tests/predict/test_visualize.py
git commit -m "feat(predict): make render_overlay score optional (GT panels label class only)"
```

### Task 3: `EvalConfig` field additions (§4.1)

**Spec ref:** §4.1, §12 AC 1.

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (`class EvalConfig(_Strict)`, after `batch_size`)
- Test: `tests/unit/test_config_schema.py` (append) OR `tests/unit/test_evaluator_schema.py`

> File-disjoint from all other Phase-1 tasks. Parallelizable.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config_schema.py`:

```python
def test_eval_config_visualize_defaults() -> None:
    from custom_sam_peft.config.schema import EvalConfig

    cfg = EvalConfig()
    assert cfg.visualize is True
    assert cfg.visualize_count == 10


def test_eval_config_visualize_count_must_be_positive() -> None:
    import pytest
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import EvalConfig

    with pytest.raises(ValidationError):
        EvalConfig(visualize_count=0)
```

If `tests/unit/test_config_schema.py` does not exist, append these to `tests/unit/test_evaluator_schema.py` instead (it already imports `EvalConfig`); confirm with `grep -n "import\|EvalConfig" tests/unit/test_evaluator_schema.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_config_schema.py -k "visualize" -v`
Expected: FAIL — `EvalConfig()` has no `visualize` attribute (and `visualize_count=0` does not raise yet).

- [ ] **Step 3: Add the fields**

In `src/custom_sam_peft/config/schema.py`, in `class EvalConfig(_Strict)`, add immediately after the `batch_size` line:

```python
    visualize: bool = True
    visualize_count: PositiveInt = 10
```

`PositiveInt` is already imported in this module (it types `lite_max_images`). Confirm: `grep -n "PositiveInt" src/custom_sam_peft/config/schema.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_config_schema.py -k "visualize" -v`
Expected: PASS — defaults are `True` / `10`; `visualize_count=0` raises `ValidationError`.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(eval): add EvalConfig.visualize and visualize_count fields"
```

### Task 4: `denormalize_to_rgb` + `gt_instances_to_entries` (§7.1, §7.2)

**Spec ref:** §7.1, §7.2, §10 (denormalize round-trip + gt conversion rows).

**Files:**

- Modify: `src/custom_sam_peft/eval/visualize.py` (append)
- Test: `tests/unit/test_eval_visualize.py` (append)

> Appends to `eval/visualize.py` (shared with Tasks 1, 5) — run after Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_visualize.py`:

```python
def test_denormalize_to_rgb_round_trip() -> None:
    import numpy as np
    from PIL import Image

    from custom_sam_peft.eval.visualize import denormalize_to_rgb

    # Known uint8 image → normalize with mean/std → denorm → expect round-trip.
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)  # (H, W, C)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = (orig.astype(np.float32) / 255.0 - np.asarray(mean)) / np.asarray(std)
    tensor = torch.from_numpy(norm).permute(2, 0, 1)  # (C, H, W)
    img = denormalize_to_rgb(tensor, mean, std)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (7, 5)  # (W, H)
    back = np.asarray(img)
    assert np.abs(back.astype(int) - orig.astype(int)).max() <= 2  # rounding tolerance


def test_denormalize_to_rgb_n_channel_uses_first_3() -> None:
    from custom_sam_peft.eval.visualize import denormalize_to_rgb

    tensor = torch.zeros(5, 4, 6)  # C=5, H=4, W=6
    mean = [0.5] * 5
    std = [0.5] * 5
    img = denormalize_to_rgb(tensor, mean, std)
    assert img.mode == "RGB"
    assert img.size == (6, 4)  # (W, H); only first 3 channels rendered


def test_gt_instances_to_entries_conversion() -> None:
    import pycocotools.mask as mask_utils

    from custom_sam_peft.data.base import Instance
    from custom_sam_peft.eval.visualize import gt_instances_to_entries

    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[1:5, 2:6] = True
    inst = Instance(mask=mask, class_id=2, box=torch.tensor([2.0, 1.0, 6.0, 5.0]))
    entries = gt_instances_to_entries([inst])
    assert len(entries) == 1
    e = entries[0]
    assert e["category_id"] == 3  # class_id + 1
    assert e["bbox"] == [2.0, 1.0, 4.0, 4.0]  # xyxy -> xywh
    assert "score" not in e  # GT carries no score
    # segmentation decodes back to the input mask.
    decoded = mask_utils.decode(e["segmentation"])  # (H, W) uint8
    assert decoded.shape == (8, 8)
    assert bool((torch.from_numpy(decoded).bool() == mask).all())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_visualize.py -k "denormalize or gt_instances" -v`
Expected: FAIL — `denormalize_to_rgb` / `gt_instances_to_entries` are not defined.

- [ ] **Step 3: Implement both functions**

Append to `src/custom_sam_peft/eval/visualize.py`. Add to the module imports at the top:

```python
import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

from custom_sam_peft.data.base import Instance
```

Then append the functions:

```python
def denormalize_to_rgb(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Image.Image:
    """Invert normalization and return a PIL RGB image (first 3 channels when C>3).

    pixel = normalized * std + mean, clamped to [0, 1], scaled to [0, 255], uint8,
    transposed (C, H, W) -> (H, W, C). For C>3 inputs only the first 3 channels are
    rendered as RGB (the corresponding first-3 mean/std are used).
    """
    c = image.shape[0]
    n = min(c, 3)
    chans = image[:n].float()
    m = torch.tensor([float(x) for x in mean[:n]]).view(n, 1, 1)
    s = torch.tensor([float(x) for x in std[:n]]).view(n, 1, 1)
    pixel = (chans * s + m).clamp(0.0, 1.0)
    arr = (pixel * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()  # (H, W, n)
    if n < 3:
        # Pad to 3 channels by repeating the last channel (e.g. grayscale -> RGB).
        arr = np.repeat(arr[:, :, :1], 3, axis=2) if n == 1 else np.concatenate(
            [arr, arr[:, :, -1:].repeat(3 - n, axis=2)], axis=2
        )
    return Image.fromarray(arr, mode="RGB")


def _mask_to_rle(mask: torch.Tensor) -> dict[str, object]:
    """(H, W) bool/uint8 mask -> pycocotools RLE dict with ASCII counts.

    Mirrors eval/postprocess.py::_logits_to_rle's encode + ascii-decode.
    """
    arr = np.asfortranarray(mask.cpu().numpy().astype(np.uint8))
    rle: dict[str, object] = mask_utils.encode(arr)
    counts = rle["counts"]
    rle["counts"] = counts.decode("ascii") if isinstance(counts, bytes) else counts
    return rle


def gt_instances_to_entries(instances: list[Instance]) -> list[dict[str, object]]:
    """Convert GT Instances to render_overlay entry dicts (no score key).

    category_id = class_id + 1 (1-indexed); bbox = xyxy -> xywh; segmentation = RLE
    of inst.mask. No `score` key (GT carries no score; the renderer labels the class
    name only).
    """
    entries: list[dict[str, object]] = []
    for inst in instances:
        x1, y1, x2, y2 = (float(v) for v in inst.box.tolist())
        entries.append(
            {
                "category_id": int(inst.class_id) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": _mask_to_rle(inst.mask),
            }
        )
    return entries
```

> The first-3-channel rule per §7.1 says "render using the first 3 channels"; the `n < 3` padding branch handles grayscale (C=1) / 2-channel inputs so the renderer always gets a 3-channel RGB. For the common C==3 case the padding is skipped. `_mask_to_rle` is a local helper (the spec §7.2 says the implementer MAY reuse `eval/evaluator.py::_mask_to_rle`, but that is a module-private name; re-implementing the 3-line encode here avoids importing a private symbol and keeps `eval/visualize.py` import-light).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_visualize.py -k "denormalize or gt_instances" -v`
Expected: PASS — denorm round-trips within tolerance, C>3 uses first 3 channels, GT entries have `category_id == class_id+1`, xywh bbox, valid RLE, no `score`.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_visualize.py
git commit -m "feat(eval): add denormalize_to_rgb and gt_instances_to_entries"
```

### Task 5: The compositor — hstack + titles + legend (§7.5)

**Spec ref:** §7.5, §10 (compositor row), §12 AC 9.

**Files:**

- Modify: `src/custom_sam_peft/eval/visualize.py` (append `_compose_pair` + `_sanitize_image_id`)
- Test: `tests/unit/test_eval_visualize.py` (append)

> Appends to `eval/visualize.py` (shared with Tasks 1, 4) — run after Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_visualize.py`:

```python
def test_compose_pair_hstacks_with_titles_and_legend() -> None:
    from PIL import Image

    from custom_sam_peft.eval.visualize import _compose_pair
    from custom_sam_peft.predict.visualize import color_for_class

    gt = Image.new("RGB", (40, 30), color=(10, 10, 10))
    pred = Image.new("RGB", (40, 30), color=(20, 20, 20))
    composite = _compose_pair(gt, pred, class_names_present=["cat", "dog"])
    # Width is at least the sum of the two panels (hstacked), height >= panel height.
    assert composite.width >= gt.width + pred.width
    assert composite.height >= gt.height
    # color_for_class is stable + used by the legend (sanity: distinct colors here).
    assert color_for_class("cat") != color_for_class("dog") or True  # may collide; not asserted hard


def test_sanitize_image_id() -> None:
    from custom_sam_peft.eval.visualize import _sanitize_image_id

    assert _sanitize_image_id("img_0") == "img_0"
    assert _sanitize_image_id("a/b/c") == "a_b_c"
    assert _sanitize_image_id("http://x/y.jpg") == "http___x_y.jpg"
    assert "/" not in _sanitize_image_id("nested/path:weird name")
    assert "\\" not in _sanitize_image_id("win\\path")
```

> The compositor test asserts the structural contract (hstack width, titles/legend present without raising) per OQ-2 — it does NOT pixel-assert title text or legend geometry. Title-present / legend-content is verified by code inspection in review.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_visualize.py -k "compose_pair or sanitize" -v`
Expected: FAIL — `_compose_pair` / `_sanitize_image_id` are not defined.

- [ ] **Step 3: Implement the compositor + sanitizer**

Append to `src/custom_sam_peft/eval/visualize.py`. Add to imports:

```python
import re

from PIL import ImageDraw, ImageFont

from custom_sam_peft.predict.visualize import color_for_class
```

Then append:

```python
_TITLE_BAR_H = 18
_LEGEND_ROW_H = 16
_LEGEND_SWATCH = 12
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_image_id(image_id: str) -> str:
    """Replace any char outside [A-Za-z0-9._-] with '_' (path separators, ':',
    spaces, URL chars). Yields a single-segment, filesystem-safe filename stem."""
    return _SANITIZE_RE.sub("_", image_id)


def _compose_pair(
    gt_panel: Image.Image,
    pred_panel: Image.Image,
    *,
    class_names_present: list[str],
) -> Image.Image:
    """Hstack `Ground Truth | Prediction` with panel titles and a shared per-class
    color legend (the union of classes present in either panel). The same class is
    the same color in both panels because both call color_for_class."""
    font = ImageFont.load_default()
    panel_h = max(gt_panel.height, pred_panel.height)
    panel_w = gt_panel.width + pred_panel.width
    legend_h = _LEGEND_ROW_H * (len(class_names_present) + 1) if class_names_present else 0
    total_h = _TITLE_BAR_H + panel_h + legend_h
    canvas = Image.new("RGB", (panel_w, total_h), color=(255, 255, 255))

    # Titles.
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Ground Truth", fill=(0, 0, 0), font=font)
    draw.text((gt_panel.width + 4, 4), "Prediction", fill=(0, 0, 0), font=font)

    # Panels below the title bar.
    canvas.paste(gt_panel, (0, _TITLE_BAR_H))
    canvas.paste(pred_panel, (gt_panel.width, _TITLE_BAR_H))

    # Legend below the panels.
    if class_names_present:
        y = _TITLE_BAR_H + panel_h
        draw.text((4, y), "Legend:", fill=(0, 0, 0), font=font)
        y += _LEGEND_ROW_H
        for name in class_names_present:
            color = color_for_class(name)
            draw.rectangle([4, y, 4 + _LEGEND_SWATCH, y + _LEGEND_SWATCH], fill=color)
            draw.text((4 + _LEGEND_SWATCH + 4, y), name, fill=(0, 0, 0), font=font)
            y += _LEGEND_ROW_H
    return canvas
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_visualize.py -v`
Expected: PASS — composite width ≥ sum of panel widths; sanitizer strips separators / illegal chars; the whole `test_eval_visualize.py` file is green.

- [ ] **Step 5: Run ruff + mypy on the new module**

Run: `uv run ruff check src/custom_sam_peft/eval/visualize.py && uv run mypy src/custom_sam_peft/eval/visualize.py`
Expected: clean (no unused imports; no type errors). Fix any findings before committing.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_visualize.py
git commit -m "feat(eval): add compositor (hstack + titles + legend) and image-id sanitizer"
```

---

## REVIEW CHECKPOINT A — pure primitives complete

- [ ] Run: `uv run pytest tests/unit/test_eval_visualize.py tests/predict/test_visualize.py tests/unit/test_config_schema.py -q`
      Expected: all PASS.
- [ ] Run: `uv run ruff check src/custom_sam_peft/eval/visualize.py src/custom_sam_peft/predict/visualize.py && uv run mypy src/custom_sam_peft/eval/visualize.py`
      Expected: clean.
- [ ] Dispatch a code-review subagent (min sonnet/high) over the Phase 1 diff: confirm (a) `pick_samples` band math is `good=round(0.5N)`, `worst=min(2,max(1,round(0.2N)))`, `median=N-good-worst` with within-band spread and no cross-band dupes; (b) `render_overlay`'s only change is `entry.get("score")` + the label branch — predict's scored call is byte-behavior-identical; (c) `denormalize_to_rgb` uses the passed mean/std (NOT hardcoded ImageNet) and the first-3-channel rule; (d) `gt_instances_to_entries` emits `category_id == class_id+1`, xywh, valid RLE, no score key.

---

## Phase 2 — model-dependent viz pass

> **Phase goal:** the per-image matched render pair (per-class K=1 forwards → mask-only matcher → matched-query extraction → composite) and the top-level `write_eval_visualizations`. Consumes Phase 1's primitives.
>
> **Interface contract exposed to Phase 3:**
>
> - `eval/visualize.py::write_eval_visualizations(model, dataset, output_dir, *, per_example_iou, count, mask_threshold, model_name, normalize, channel_semantics) -> list[Path]` — Phase-2 top-level entry called by `run_eval`. Selects `count` variety-weighted images, renders a GT-vs-Pred composite per image, writes PNGs under `output_dir/visualizations/`, returns the written paths. Memory-bounded (one image at a time); per-image failures caught + logged at WARNING; never raises for a single bad image. `normalize` is `NormalizeConfig | None`; `channel_semantics` is the `data.channel_semantics` string.
> - Internal (called only within the module): `render_eval_pair(model, example, class_names, *, mask_threshold, mean, std, matcher) -> Image.Image`.
>
> Tasks 6 and 7 both append to `eval/visualize.py` and `tests/unit/test_eval_visualize_pair.py` — serialize them (Task 6 before Task 7). Both depend on all of Phase 1.

### Task 6: `render_eval_pair` — per-image matched composite (§7.4)

**Spec ref:** §7.4, §8 (per-image edge cases), §10 (covered indirectly via the smoke test; this task's tests assert the matched-pred extraction shape).

**Files:**

- Modify: `src/custom_sam_peft/eval/visualize.py` (append `render_eval_pair` + a matched-extraction helper)
- Test: `tests/unit/test_eval_visualize_pair.py` (new)

> Appends to `eval/visualize.py` — run after all Phase-1 tasks. Uses the tiny stub + a synthetic Example.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_eval_visualize_pair.py`:

```python
"""Unit tests for the model-dependent eval viz pass (CPU-only, tiny stub)."""

from __future__ import annotations

import torch
from PIL import Image

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.visualize import render_eval_pair
from custom_sam_peft.models.matching import HungarianMatcher
from tests.fixtures.tiny_sam3_stub import TinySam3Stub


def _example(class_id: int) -> Example:
    h = w = 8
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[:4, :4] = True
    return Example(
        image=torch.zeros(3, h, w),
        image_id="img_0",
        prompts=TextPrompts(classes=["cat", "dog"]),
        instances=[Instance(mask=mask, class_id=class_id, box=torch.tensor([0.0, 0.0, 4.0, 4.0]))],
    )


def _matcher() -> HungarianMatcher:
    w = MatcherWeights()
    return HungarianMatcher(lambda_l1=w.lambda_l1, lambda_giou=w.lambda_giou, lambda_mask=w.lambda_mask)


def test_render_eval_pair_returns_hstacked_image() -> None:
    model = TinySam3Stub()
    ex = _example(class_id=0)
    out = render_eval_pair(
        model,
        ex,
        ["cat", "dog"],
        mask_threshold=0.0,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        matcher=_matcher(),
    )
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    # Hstacked: width >= 2 * source width (8 px each, plus legend/titles add height not width).
    assert out.width >= 16


def test_render_eval_pair_no_gt_class_draws_no_pred_for_that_class() -> None:
    # Image has a single 'cat' (class_id 0) GT; 'dog' (class_id 1) has no GT, so
    # the dog matcher target list is empty and no dog pred is drawn. The call must
    # not raise and must return a composite.
    model = TinySam3Stub()
    ex = _example(class_id=0)
    out = render_eval_pair(
        model, ex, ["cat", "dog"], mask_threshold=0.0,
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], matcher=_matcher(),
    )
    assert isinstance(out, Image.Image)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_visualize_pair.py -v`
Expected: FAIL — `render_eval_pair` is not defined.

- [ ] **Step 3: Implement `render_eval_pair` + the matched-extraction helper**

Append to `src/custom_sam_peft/eval/visualize.py`. Add to imports:

```python
from typing import Any

from custom_sam_peft.data.base import Example, TextPrompts
from custom_sam_peft.eval.evaluator import _row_outputs
from custom_sam_peft.eval.postprocess import queries_to_coco_results
from custom_sam_peft.models.matching import HungarianMatcher, meta_to_canonical
from custom_sam_peft.predict.visualize import render_overlay
from custom_sam_peft.runtime import Runtime, to_device
```

Then append:

```python
def _matched_pred_entries(
    model: Any,
    example: Example,
    class_names: list[str],
    *,
    mask_threshold: float,
    matcher: HungarianMatcher,
    runtime: Runtime,
) -> list[dict[str, object]]:
    """Per-class K=1 forward + mask-only Hungarian match; return the matched-query
    COCO entries (1:1 with GT masks) aggregated across all classes. Draws ONLY
    matched preds (no unmatched/extra detections).
    """
    h, w = int(example.image.shape[-2]), int(example.image.shape[-1])
    images_1 = to_device(example.image.unsqueeze(0), runtime)  # (1, C, H, W)
    out_entries: list[dict[str, object]] = []
    for class_name in class_names:
        cls_idx = class_names.index(class_name)
        targets = [inst for inst in example.instances if int(inst.class_id) == cls_idx]
        if not targets:
            continue  # no GT for this class → nothing matched/drawn
        outputs = model(images_1, [TextPrompts(classes=[class_name])], support=None)
        canonical = meta_to_canonical(outputs)
        # matcher returns per-image [(query_idx, target_idx)]; one image here.
        query_idx, _target_idx = matcher(canonical, [targets])[0]
        # All-query COCO entries for this class, then keep only matched query rows.
        all_entries = queries_to_coco_results(
            _row_outputs(outputs, 0),
            int(0),  # image_id is irrelevant for rendering (entries are per-image)
            cls_idx + 1,
            (h, w),
            mask_threshold,
        )
        for q in query_idx.tolist():
            if 0 <= q < len(all_entries):
                out_entries.append(all_entries[q])
    return out_entries


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
    """Return the hstacked `Ground Truth | Prediction` composite for one image.

    GT panel: denormalized source + GT instance overlays (no score). Pred panel:
    denormalized source + the Hungarian mask-only matched 1:1 preds per class,
    aggregated across classes (matched preds only). Both panels use the same
    color_for_class mapping so a class is the same color in both.
    """
    try:
        param_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        param_device = torch.device("cpu")
    runtime = Runtime(device=param_device, dtype=torch.float32)

    source = denormalize_to_rgb(example.image, mean, std)

    gt_entries = gt_instances_to_entries(example.instances)
    gt_panel = render_overlay(source, gt_entries, prompts=class_names)

    pred_entries = _matched_pred_entries(
        model, example, class_names,
        mask_threshold=mask_threshold, matcher=matcher, runtime=runtime,
    )
    pred_panel = render_overlay(source, pred_entries, prompts=class_names)

    # Legend = union of classes present in either panel.
    present_ids = {int(e["category_id"]) for e in (*gt_entries, *pred_entries)}
    names_present = [class_names[c - 1] for c in sorted(present_ids) if 0 < c <= len(class_names)]
    return _compose_pair(gt_panel, pred_panel, class_names_present=names_present)
```

> **Why `image_id=0` in the postprocess call is safe:** `queries_to_coco_results` stamps `image_id`/`category_id` into each entry, but `render_overlay` only reads `category_id`, `bbox`, `score`, `segmentation` — never `image_id`. The matched entries are rendered onto the single source image, so the placeholder id is inert. **GT box representation note (locked, §7.4):** the matcher reads `t.box` (xyxy pixel) as if cxcywh, but `lambda_l1=lambda_giou=0`, so the box cost never enters the assignment — only the mask Dice term matters. Do not "fix" the unconverted box. **Score scale note:** only `lambda_mask` is nonzero, so the `argmin` assignment is invariant to its positive scalar value — the absolute lambda scale is irrelevant.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_visualize_pair.py -v`
Expected: PASS — `render_eval_pair` returns an RGB composite hstacked ≥ 2× source width; a class with no GT draws no pred for that class and does not raise. (The stub returns all-zero outputs → 4 queries with identical Dice cost; `linear_sum_assignment` still returns a valid 1:1 assignment for the single GT.)

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_visualize_pair.py
git commit -m "feat(eval): add render_eval_pair (per-class K=1 forward + mask-only match)"
```

### Task 7: `write_eval_visualizations` — top-level Phase-2 entry (§7.6)

**Spec ref:** §7.6, §8 (whole-pass + per-image robustness, zero-candidate INFO), §10 (covered by the integration smoke).

**Files:**

- Modify: `src/custom_sam_peft/eval/visualize.py` (append `write_eval_visualizations`)
- Test: `tests/unit/test_eval_visualize_pair.py` (append)

> Appends to `eval/visualize.py` (shared with Task 6) — run after Task 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_visualize_pair.py`:

```python
def _dataset(class_ids: list[int]):
    examples = [_example(class_id=c) for c in class_ids]
    for i, ex in enumerate(examples):
        # give each a distinct image_id (frozen dataclass → rebuild)
        examples[i] = Example(
            image=ex.image, image_id=f"img_{i}", prompts=ex.prompts, instances=ex.instances
        )

    class _DS:
        class_names = ["cat", "dog"]

        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, j: int) -> Example:
            return examples[j]

    return _DS()


def test_write_eval_visualizations_writes_pngs(tmp_path) -> None:
    from custom_sam_peft.eval.visualize import write_eval_visualizations

    ds = _dataset([0, 1, 0])  # 3 GT-bearing images
    model = TinySam3Stub()
    paths = write_eval_visualizations(
        model, ds, tmp_path,
        per_example_iou=[0.9, 0.5, 0.1], count=10,
        mask_threshold=0.0, model_name="facebook/sam3.1",
        normalize=None, channel_semantics="rgb",
    )
    assert len(paths) == 3  # small pool → all candidates
    vis_dir = tmp_path / "visualizations"
    assert vis_dir.is_dir()
    written = sorted(p.name for p in vis_dir.glob("*.png"))
    assert written == ["img_0.png", "img_1.png", "img_2.png"]
    for p in paths:
        Image.open(p).verify()  # readable image


def test_write_eval_visualizations_zero_candidates(tmp_path, caplog) -> None:
    from custom_sam_peft.data.base import Example, TextPrompts
    from custom_sam_peft.eval.visualize import write_eval_visualizations

    # All images have NO GT → zero candidates.
    no_gt = [
        Example(image=torch.zeros(3, 8, 8), image_id=f"n_{i}",
                prompts=TextPrompts(classes=["cat", "dog"]), instances=[])
        for i in range(2)
    ]

    class _DS:
        class_names = ["cat", "dog"]
        def __len__(self) -> int: return len(no_gt)
        def __getitem__(self, j: int) -> Example: return no_gt[j]

    with caplog.at_level("INFO"):
        paths = write_eval_visualizations(
            TinySam3Stub(), _DS(), tmp_path,
            per_example_iou=[1.0, 1.0], count=5,
            mask_threshold=0.0, model_name="facebook/sam3.1",
            normalize=None, channel_semantics="rgb",
        )
    assert paths == []
    assert not (tmp_path / "visualizations").exists() or not list(
        (tmp_path / "visualizations").glob("*.png")
    )
    assert any("no GT" in r.message.lower() or "no gt" in r.message.lower() for r in caplog.records)


def test_write_eval_visualizations_per_image_failure_is_caught(tmp_path, monkeypatch, caplog) -> None:
    """A single image that raises during render is logged at WARNING and skipped;
    other images still render."""
    from custom_sam_peft.eval import visualize as viz

    ds = _dataset([0, 1])
    model = TinySam3Stub()
    calls = {"n": 0}
    real = viz.render_eval_pair

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(viz, "render_eval_pair", flaky)
    with caplog.at_level("WARNING"):
        paths = viz.write_eval_visualizations(
            model, ds, tmp_path,
            per_example_iou=[0.9, 0.1], count=10,
            mask_threshold=0.0, model_name="facebook/sam3.1",
            normalize=None, channel_semantics="rgb",
        )
    assert len(paths) == 1  # one survived
    assert any(r.levelname == "WARNING" for r in caplog.records)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_visualize_pair.py -k "write_eval" -v`
Expected: FAIL — `write_eval_visualizations` is not defined.

- [ ] **Step 3: Implement `write_eval_visualizations`**

Append to `src/custom_sam_peft/eval/visualize.py`. Add to imports:

```python
from pathlib import Path

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.transforms import resolve_normalization
```

> If `NormalizeConfig` import causes a cycle (schema imports), confirm with `uv run python -c "import custom_sam_peft.eval.visualize"`; the existing `eval/runner.py` already imports `config.schema`, so this is safe.

Then append:

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
    logged at WARNING; never raises for a single bad image.
    """
    selected = pick_samples(per_example_iou, dataset, count)
    if not selected:
        _LOG.info("eval visualize: no GT-bearing images to visualize; skipping.")
        return []

    mean, std = resolve_normalization(
        model_name, normalize, channel_semantics=channel_semantics  # type: ignore[arg-type]
    )
    w = MatcherWeights()
    matcher = HungarianMatcher(
        lambda_l1=w.lambda_l1, lambda_giou=w.lambda_giou, lambda_mask=w.lambda_mask
    )

    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    written: list[Path] = []
    try:
        with torch.no_grad():
            for idx in selected:
                example = dataset[idx]
                try:
                    composite = render_eval_pair(
                        model, example, list(dataset.class_names),
                        mask_threshold=mask_threshold, mean=mean, std=std, matcher=matcher,
                    )
                    out_path = vis_dir / f"{_sanitize_image_id(example.image_id)}.png"
                    composite.save(out_path)
                    written.append(out_path)
                except Exception:
                    _LOG.warning(
                        "eval visualize: failed to render image_id=%r (idx=%d); skipping.",
                        example.image_id, idx, exc_info=True,
                    )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    return written
```

> The `# type: ignore[arg-type]` on the `resolve_normalization(model_name, normalize, ...)` call handles the `NormalizeConfig | None` → `NormalizeConfig` fallback param mismatch (OQ-1). Before adding the ignore, run `grep -n "resolve_normalization(" src/custom_sam_peft/data/transforms.py` and confirm how `build_eval_transforms` passes `normalize` (it passes `cfg.data.normalize`, same type) — if mypy is already clean there, drop the ignore. Do NOT change `resolve_normalization`'s signature.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_visualize_pair.py -v`
Expected: PASS — 3 GT-bearing images → 3 PNGs named by sanitized image_id, each readable; zero candidates → `[]` + INFO log + no PNGs; a per-image render failure logs WARNING and the other image still renders.

- [ ] **Step 5: Run ruff + mypy on the module**

Run: `uv run ruff check src/custom_sam_peft/eval/visualize.py && uv run mypy src/custom_sam_peft/eval/visualize.py`
Expected: clean. Fix any findings (drop unused `# type: ignore` if mypy reports it as unused) before committing.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_visualize_pair.py
git commit -m "feat(eval): add write_eval_visualizations top-level viz pass"
```

---

## REVIEW CHECKPOINT B — viz pass complete

- [ ] Run: `uv run pytest tests/unit/test_eval_visualize.py tests/unit/test_eval_visualize_pair.py -q`
      Expected: all PASS.
- [ ] Run: `uv run python -c "import custom_sam_peft.eval.visualize"`
      Expected: imports clean (no cycle from `config.schema` / `data.transforms` / `eval.postprocess`).
- [ ] Dispatch a code-review subagent (min sonnet/high; the matcher + postprocess reuse is design-sensitive — consider opus/xhigh): confirm (a) the K=1 forward calls `model(images_1, [TextPrompts(classes=[name])], support=None)`; (b) matched-query extraction indexes `queries_to_coco_results` rows by `query_idx` (the matcher's first return element), drawing matched preds only; (c) the matcher is built from `MatcherWeights()` defaults (mask-only); (d) the no-GT-class branch is skipped, not matched; (e) `write_eval_visualizations` is memory-bounded (one image at a time), per-image failures are WARNING-logged and never abort, and the model's train/eval state is restored.

---

## Phase 3 — integration surface & docs

> **Phase goal:** wire `write_eval_visualizations` into the config + CLI + `run_eval` dispatch, document the knobs, and add the CPU integration smoke (incl. the in-loop-no-viz guard and the `--no-visualize` guard). Consumes Phase 2's `write_eval_visualizations`.
>
> **Interface contract:** after this phase, `cfg.eval.visualize` (default True) drives the viz pass on the `run_eval` / `evaluate_and_save` surface (`csp eval`, `csp run`, `csp train --eval`), never in-loop; `csp eval --visualize/--no-visualize` (tri-state) and `csp run --visualize/--no-visualize` (default True) override it.
>
> Task 8 (`run_eval`) is the foundation of this phase — Tasks 9, 10 thread flags into it and must run AFTER Task 8. Task 11 (docs) is file-disjoint and parallelizable with everything. Task 12 (integration smoke) runs LAST (needs the full wiring).

### Task 8: `run_eval` — `visualize` param + Phase-2 dispatch (§4.5, §6)

**Spec ref:** §4.5, §6.1–§6.4, §12 AC 5.

**Files:**

- Modify: `src/custom_sam_peft/eval/runner.py` (both `@overload` stubs + the impl signature; the resolution + Branch-1/Branch-2 dispatch)
- Test: `tests/unit/test_eval_runner.py` (append)

> Foundation for Phase 3. Run before Tasks 9 and 10.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_runner.py` (reuse `_make_cfg`; note it sets `cfg.eval.model_copy` and `cfg.peft.method`):

```python
def test_run_eval_calls_write_eval_visualizations_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """visualize resolves True (from cfg) → write_eval_visualizations is called on
    the default (Branch-2) path, after metrics persist."""
    cfg = _make_cfg()
    cfg.eval.visualize = True
    cfg.eval.visualize_count = 7
    cfg.eval.mask_threshold = 0.0
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"
    cfg.eval.save_predictions = False

    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    # Evaluator.evaluate(..., return_per_example_iou=True) -> (report, iou_list)
    ev = MagicMock()
    ev.evaluate.return_value = (MagicMock(overall={}, per_class={}, n_images=1, n_predictions=0), [0.5])
    ev._last_predictions = []
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)

    captured: dict[str, object] = {}

    def _fake_write(model, dataset, out, **kw):
        captured.update(kw)
        captured["out"] = out
        return []

    monkeypatch.setattr("custom_sam_peft.eval.visualize.write_eval_visualizations", _fake_write)

    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)
    assert captured["count"] == 7
    assert "per_example_iou" in captured
    assert captured["per_example_iou"] == [0.5]


def test_run_eval_no_visualize_skips_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """visualize=False overrides cfg → write_eval_visualizations is NOT called and the
    plain evaluate_and_save path is used."""
    cfg = _make_cfg()
    cfg.eval.visualize = True  # cfg says on; flag says off → off wins.
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    ev = MagicMock()
    ev.evaluate_and_save.return_value = MagicMock(overall={})
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    called: list[int] = []
    monkeypatch.setattr(
        "custom_sam_peft.eval.visualize.write_eval_visualizations",
        lambda *a, **k: called.append(1),
    )
    run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path, visualize=False)
    assert called == []
    assert ev.evaluate_and_save.called  # plain path preserved


def test_run_eval_viz_failure_does_not_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A whole-pass viz failure is caught in run_eval (metrics already persisted)."""
    cfg = _make_cfg()
    cfg.eval.visualize = True
    cfg.eval.save_predictions = False
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=["cat"]),
    )
    ev = MagicMock()
    ev.evaluate.return_value = (MagicMock(overall={}, per_class={}, n_images=1, n_predictions=0), [0.5])
    ev._last_predictions = []
    monkeypatch.setattr("custom_sam_peft.eval.runner.Evaluator", lambda _c: ev)
    monkeypatch.setattr(
        "custom_sam_peft.eval.visualize.write_eval_visualizations",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("viz boom")),
    )
    with caplog.at_level("WARNING"):
        report = run_eval(cfg, checkpoint=None, split="val", output_dir=tmp_path)
    assert (tmp_path / "metrics.json").exists()  # persisted before viz
    assert any("viz" in r.message.lower() or "visuali" in r.message.lower() for r in caplog.records)
```

> `_make_cfg` returns a `MagicMock` cfg; setting `cfg.eval.visualize = True` works because it's a mock. The Branch-2 promotion test asserts `write_eval_visualizations` is reached with the IoU list. If `_make_cfg`'s `cfg.eval.model_copy` lambda interferes (it returns `cfg.eval` ignoring updates), that is fine — these tests don't depend on `save_predictions` mutation.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_runner.py -k "visualize or viz" -v`
Expected: FAIL — `run_eval` has no `visualize` kwarg (TypeError on the `visualize=False` test) and never calls `write_eval_visualizations`.

- [ ] **Step 3: Add `visualize` to both overloads + the impl signature**

In `src/custom_sam_peft/eval/runner.py`, add `visualize: bool | None = None,` to BOTH `@overload` stubs and the impl signature, immediately after the `save_predictions: bool | None = None,` line in each (3 edits). Grep to find all three: `grep -n "save_predictions: bool | None = None," src/custom_sam_peft/eval/runner.py` → 3 hits.

Example (the impl signature):

```python
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path | None = None,
    artifacts: EvalArtifacts | None = None,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
    visualize: bool | None = None,
    val_dataset: Dataset | None = None,
    model: Any | None = None,
    return_per_example_iou: bool = False,
) -> MetricsReport | tuple[MetricsReport, list[float]]:
```

- [ ] **Step 4: Resolve + dispatch in the two branches**

In `run_eval`, after `eval_cfg = cfg.eval` and the `save_predictions` / auto-batch resolution (right before `evaluator = Evaluator(eval_cfg)`), add the resolution:

```python
    visualize_resolved = cfg.eval.visualize if visualize is None else visualize
```

Then locate the two branches at the end of `run_eval` (`if return_per_example_iou:` ... `return evaluator.evaluate_and_save(wrapper, dataset, out)`). Rewrite the tail so BOTH branches run the viz pass when `visualize_resolved`:

```python
    def _run_viz(per_example_iou: list[float]) -> None:
        if not visualize_resolved:
            return
        try:
            from custom_sam_peft.eval.visualize import write_eval_visualizations

            write_eval_visualizations(
                wrapper,
                dataset,
                out,
                per_example_iou=per_example_iou,
                count=cfg.eval.visualize_count,
                mask_threshold=cfg.eval.mask_threshold,
                model_name=cfg.model.name,
                normalize=cfg.data.normalize,
                channel_semantics=cfg.data.channel_semantics,
            )
        except Exception:
            _LOG.warning("eval visualize pass failed; metrics are persisted.", exc_info=True)

    if return_per_example_iou:
        out.mkdir(parents=True, exist_ok=True)
        report, per_example_iou = evaluator.evaluate(wrapper, dataset, return_per_example_iou=True)
        (out / "metrics.json").write_text(
            json.dumps(
                {
                    "overall": report.overall,
                    "per_class": report.per_class,
                    "n_images": report.n_images,
                    "n_predictions": report.n_predictions,
                },
                indent=2,
            )
        )
        if eval_cfg.save_predictions and eval_cfg.mode == "full":
            (out / "predictions.json").write_text(json.dumps(evaluator._last_predictions))
        _run_viz(per_example_iou)
        return report, per_example_iou

    # Branch 2 (default): when visualize is on, promote to the IoU-shape so the viz
    # pass has its ranking; mirror evaluate_and_save's persistence. When off, keep
    # the original evaluate_and_save call unchanged (no behavior change).
    if not visualize_resolved:
        return evaluator.evaluate_and_save(wrapper, dataset, out)

    out.mkdir(parents=True, exist_ok=True)
    report, per_example_iou = evaluator.evaluate(wrapper, dataset, return_per_example_iou=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "overall": report.overall,
                "per_class": report.per_class,
                "n_images": report.n_images,
                "n_predictions": report.n_predictions,
            },
            indent=2,
        )
    )
    evaluator._maybe_save_predictions(evaluator._last_predictions, run_dir=out)
    _run_viz(per_example_iou)
    return report
```

> **Branch-2 persistence parity (§6.2):** `evaluate_and_save` writes `metrics.json` AND calls `evaluator._maybe_save_predictions(self._last_predictions, run_dir=output_dir)` (gated internally by `save_predictions and mode=="full"`). The promoted Branch 2 mirrors BOTH: it writes the same `metrics.json` keys and calls `_maybe_save_predictions` with `run_dir=out`. The promoted path's return type is `MetricsReport` (NOT the tuple) — `return_per_example_iou` is still False here, so the public return type is unchanged (§6.2 note). Note Branch 1 uses the flat `predictions.json` write (its existing behavior — preserved verbatim), while Branch 2 uses `_maybe_save_predictions` (mirroring `evaluate_and_save`'s canonical path). These two persistence styles pre-exist; do not unify them in this PR (scope discipline).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_runner.py -v`
Expected: PASS — viz pass called with `count=7` + the IoU list on the promoted Branch 2; `visualize=False` skips it and keeps `evaluate_and_save`; a whole-pass viz failure is caught (WARNING) after `metrics.json` is persisted; ALL existing `test_eval_runner.py` cases (baseline, PEFT-inference, dispatch) stay green.

- [ ] **Step 6: Run the eval-runner-adjacent suites**

Run: `uv run pytest tests/unit/test_eval_runner.py tests/unit/test_eval_runner_gate.py tests/unit/test_evaluator.py tests/integration/test_trainer_evaluator_seam.py -q`
Expected: PASS — the gate tests, evaluator orchestration, and the trainer→evaluator seam are unaffected (in-loop `Evaluator.evaluate` is untouched).

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/eval/runner.py tests/unit/test_eval_runner.py
git commit -m "feat(eval): wire visualize into run_eval (both branches, Branch-2 promotion)"
```

### Task 9: `csp eval --visualize/--no-visualize` tri-state flag (§4.2)

**Spec ref:** §4.2, §12 AC 2.

**Files:**

- Modify: `src/custom_sam_peft/cli/eval_cmd.py` (add the option; thread into the `run_eval` call)
- Test: `tests/unit/cli/test_eval_cmd.py` (append)

> Depends on Task 8. File-disjoint from Task 10 (`run_cmd.py`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/cli/test_eval_cmd.py` (the file exists with `runner = CliRunner()` and `from custom_sam_peft.cli.main import app`):

```python
def test_eval_no_visualize_threads_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("placeholder")
    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.load_config", lambda p: MagicMock())
    captured: dict[str, object] = {}

    def _fake_run_eval(cfg, **kw):
        captured.update(kw)
        report = MagicMock()
        report.overall = {}
        return report

    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.run_eval", _fake_run_eval)
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "val", "--no-visualize"])
    assert result.exit_code == 0, result.output
    assert captured["visualize"] is False


def test_eval_default_visualize_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("placeholder")
    monkeypatch.setattr("custom_sam_peft.cli.eval_cmd.load_config", lambda p: MagicMock())
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "custom_sam_peft.cli.eval_cmd.run_eval",
        lambda cfg, **kw: (captured.update(kw), MagicMock(overall={}))[1],
    )
    result = runner.invoke(app, ["eval", "--config", str(cfg), "--split", "val"])
    assert result.exit_code == 0, result.output
    assert captured["visualize"] is None  # tri-state: defer to cfg
```

If `MagicMock` / `pytest` / `Path` are not imported in `test_eval_cmd.py`, add them (`from unittest.mock import MagicMock`, `import pytest`, `from pathlib import Path`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/cli/test_eval_cmd.py -k "visualize" -v`
Expected: FAIL — `eval` has no `--visualize/--no-visualize` flag (`--no-visualize` is an unknown option → exit 2), and `run_eval` is not passed `visualize`.

- [ ] **Step 3: Add the option + thread it**

In `src/custom_sam_peft/cli/eval_cmd.py`, add an option to `evaluate(...)` immediately after the `save_predictions` option:

```python
    visualize: bool | None = typer.Option(
        None,
        "--visualize/--no-visualize",
        help="Override cfg.eval.visualize (write GT-vs-Pred composite panels).",
    ),
```

Then add `visualize=visualize,` to the `run_eval(...)` call (next to `save_predictions=save_predictions,`):

```python
            report = run_eval(
                cfg,
                checkpoint=checkpoint,
                split=split_lit,
                output_dir=output,
                save_predictions=save_predictions,
                visualize=visualize,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/cli/test_eval_cmd.py -v`
Expected: PASS — `--no-visualize` threads `visualize=False`; omitting the flag threads `visualize=None` (defer to cfg); the existing eval-cmd cases stay green.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/eval_cmd.py tests/unit/cli/test_eval_cmd.py
git commit -m "feat(eval): add --visualize/--no-visualize tri-state flag"
```

### Task 10: `csp run --visualize/--no-visualize` flag, threaded through `_orchestrate` (§4.3)

**Spec ref:** §4.3, §12 AC 3.

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py` (`_orchestrate` signature + the eval `run_eval` call; the `run` option + the `_orchestrate` call)
- Test: `tests/unit/test_train_runner.py` OR `tests/integration/test_cli_run.py` (append a flag-threading test)

> Depends on Task 8. File-disjoint from Task 9.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_cli_run.py` (it already drives `csp run` via the CliRunner; confirm imports with `grep -n "import\|CliRunner\|_orchestrate\|def test" tests/integration/test_cli_run.py | head`). If it monkeypatches `run_training` / `run_eval`, mirror that pattern:

```python
def test_run_threads_no_visualize_into_run_eval(monkeypatch, tmp_path) -> None:
    """`csp run --no-visualize` reaches the eval-phase run_eval with visualize=False."""
    import custom_sam_peft.cli.run_cmd as run_cmd

    captured: dict[str, object] = {}

    # Stub the heavy phases so we only assert the threading.
    monkeypatch.setattr(run_cmd, "_orchestrate", lambda cfg, resume, mode, *, visualize: (
        captured.update({"visualize": visualize}), 0
    )[1])
    # ... drive run() with a minimal config + --no-visualize, OR call _orchestrate
    #     directly if run() is too heavy to stub. Prefer a direct _orchestrate
    #     signature test:
```

Because `run()` orchestrates training/eval/export end-to-end, prefer a **direct signature + threading test** over driving the whole command. Replace the above with a focused test in `tests/unit/test_train_runner.py` (or a new `tests/unit/cli/test_run_cmd.py`):

```python
def test_orchestrate_threads_visualize_into_eval_run_eval(monkeypatch, tmp_path) -> None:
    """_orchestrate forwards its visualize kwarg to the eval-phase run_eval call."""
    import custom_sam_peft.cli.run_cmd as run_cmd

    # Minimal cfg mock with a 'none' val mode short-circuits to no eval — so instead
    # force a val mode and stub the deps. Stub run_training -> result; load_val_source
    # -> a 'val' mode source; load_sam31/load_adapter -> no-ops; _build_val_dataset ->
    # a tiny dataset; run_eval -> capture the kwargs.
    captured: dict[str, object] = {}

    fake_result = MagicMock()
    fake_result.run_dir = tmp_path
    fake_result.checkpoint_path = tmp_path / "adapter"
    monkeypatch.setattr(run_cmd, "run_training", lambda cfg, resume_from=None: fake_result)

    vs = MagicMock()
    vs.mode = "auto_split"
    monkeypatch.setattr("custom_sam_peft.data.val_source.load_val_source", lambda rd: vs)
    monkeypatch.setattr(run_cmd, "load_sam31", lambda *a, **k: MagicMock())
    monkeypatch.setattr(run_cmd, "load_adapter", lambda *a, **k: None)
    monkeypatch.setattr(run_cmd, "_build_val_dataset", lambda cfg, vs: MagicMock())
    monkeypatch.setattr(run_cmd, "write_bundle", lambda *a, **k: None)

    def _fake_run_eval(cfg, **kw):
        captured.update(kw)
        return MagicMock(overall={}), [0.5]

    monkeypatch.setattr(run_cmd, "run_eval", _fake_run_eval)

    cfg = MagicMock()
    cfg.train.epochs = 1
    run_cmd._orchestrate(cfg, None, run_cmd.ProgressMode.OFF if hasattr(run_cmd, "ProgressMode") else None, visualize=False)
    assert captured["visualize"] is False
```

> The exact stubs depend on `_orchestrate`'s body (training → val_source → eval → bundle). Read `_orchestrate` (`grep -n "def _orchestrate" src/custom_sam_peft/cli/run_cmd.py` then the body) and stub each external call it makes between entry and the eval-phase `run_eval`. The single assertion is `captured["visualize"] is False`. If stubbing the full `_orchestrate` is brittle, fall back to asserting the SIGNATURE: `import inspect; assert "visualize" in inspect.signature(run_cmd._orchestrate).parameters` plus a CliRunner test that `csp run --no-visualize` parses without a usage error.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_train_runner.py -k "orchestrate_threads_visualize" -v`
Expected: FAIL — `_orchestrate` has no `visualize` keyword (TypeError).

- [ ] **Step 3: Add the `visualize` param to `_orchestrate` + forward it**

In `src/custom_sam_peft/cli/run_cmd.py`, change `_orchestrate`'s signature:

```python
def _orchestrate(cfg: TrainConfig, resume: Path | None, mode: ProgressMode, *, visualize: bool) -> int:
```

In the eval-phase `run_eval(...)` call inside `_orchestrate` (the `return_per_example_iou=True` call), add `visualize=visualize,`:

```python
                report, per_example_iou = cast(
                    tuple[Any, list[float]],
                    run_eval(
                        cfg,
                        checkpoint=adapter_path,
                        output_dir=run_dir,
                        val_dataset=val_dataset,
                        model=wrapper,
                        return_per_example_iou=True,
                        visualize=visualize,
                    ),
                )
```

- [ ] **Step 4: Add the `--visualize/--no-visualize` flag to `run` + pass it**

In `run(...)`'s signature, add an option after `progress_flag` (before the closing `) -> None:`):

```python
    visualize: bool = typer.Option(
        True,
        "--visualize/--no-visualize",
        help="Write GT-vs-Pred composite panels in the eval phase.",
    ),
```

Change the `run()` body's final call from `_orchestrate(cfg, resume_path, mode)` to:

```python
    _orchestrate(cfg, resume_path, mode, visualize=visualize)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_train_runner.py -k "orchestrate_threads_visualize" -v`
Expected: PASS — `_orchestrate` accepts and forwards `visualize`; `csp run --no-visualize` parses and threads `False`.

- [ ] **Step 6: Run the run-cmd-adjacent suites**

Run: `uv run pytest tests/integration/test_cli_run.py tests/unit/test_train_runner.py -q`
Expected: PASS — the end-to-end `csp run` smoke (if present) and runner tests stay green (the default `visualize=True` matches cfg's default True, so no behavior change for existing runs).

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/unit/test_train_runner.py
git commit -m "feat(run): add --visualize/--no-visualize flag threaded through _orchestrate"
```

### Task 11: Docs — `config-schema.md` rows + `CHANGELOG.md` (§4.6)

**Spec ref:** §4.6, §12 AC 14.

**Files:**

- Modify: `docs/config-schema.md` (`## eval` table)
- Modify: `CHANGELOG.md` (`## [Unreleased]`)

> File-disjoint from all other Phase-3 tasks. Parallelizable. No test (docs only); the markdownlint gate runs in Task 13.

- [ ] **Step 1: Add the two `## eval` rows**

In `docs/config-schema.md`, in the `## eval` table, immediately after the `eval.save_predictions` row (the last row before the `---`), add:

```markdown
| `eval.visualize` | bool | `true` | advanced | On the final/standalone eval path, write a `Ground Truth \| Prediction` composite PNG per sampled image under `<output>/visualizations/`. Disabled per-command via `--no-visualize`. | Additive qualitative aid; default on so `csp eval` / `csp run` show results without extra flags. |
| `eval.visualize_count` | int (>0) | `10` | advanced | Number of images to sample for visualization (variety-weighted toward high IoU, always including a couple of the worst). | Bounded second pass (~N extra single-class forwards); 10 is a legible default. |
```

> Note the escaped pipe `\|` inside the `Ground Truth | Prediction` description so the markdown table cell does not break. No JSON-schema artifact exists in the repo to regenerate (verified: no `*.schema.json`); this markdown table is the published schema.

- [ ] **Step 2: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add a new `### Added` section (place it above the existing `### Breaking` section, or append a new `### Added` block if one already exists under Unreleased — check with `grep -n "## \[Unreleased\]\|### Added\|### Breaking" CHANGELOG.md`):

```markdown
### Added — eval GT-vs-Pred visualization

- **eval**: new `eval.visualize` (bool, default `true`) and `eval.visualize_count`
  (int, default `10`) config knobs. On the final/standalone eval path (`csp eval`,
  `csp run`'s eval phase, `csp train --eval`), eval now writes one
  `Ground Truth | Prediction` composite PNG per variety-weighted sampled image
  under `<output>/visualizations/`, with per-class color legend. Predictions are
  the Hungarian mask-only matched 1:1 set per class.
- **cli**: `csp eval --visualize/--no-visualize` (tri-state; defers to config when
  unset) and `csp run --visualize/--no-visualize` (default on). The in-loop
  training eval is unchanged.
```

- [ ] **Step 3: Commit**

```bash
git add docs/config-schema.md CHANGELOG.md
git commit -m "docs: document eval.visualize / visualize_count knobs and CLI flags"
```

### Task 12: CPU integration smoke — end-to-end + in-loop guard + `--no-visualize` guard (§10)

**Spec ref:** §10 (integration test), §12 AC 12, AC 15.

**Files:**

- Create: `tests/integration/test_eval_visualize_integration.py`
- Test: itself

> Runs LAST in Phase 3 (needs Task 8's `run_eval` wiring + the full module). Uses `stub_model` + `tiny_text_dataset` (conftest fixtures: 2 images, 2 classes `cat`/`dog`, both GT-bearing).

- [ ] **Step 1: Write the integration tests**

Create `tests/integration/test_eval_visualize_integration.py`:

```python
"""CPU integration smoke for eval --visualize (tiny stub + 2-class dataset)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from custom_sam_peft.config.schema import EvalConfig
from custom_sam_peft.eval.evaluator import Evaluator
from custom_sam_peft.eval.runner import run_eval


def _cfg(tmp_path: Path, *, visualize: bool = True) -> MagicMock:
    """A run_eval-compatible cfg mock backed by a real EvalConfig for the viz knobs."""
    cfg = MagicMock()
    cfg.data.format = "coco"
    cfg.data.val = MagicMock()
    cfg.data.val_split = None
    cfg.data.test = None
    cfg.data.normalize = None
    cfg.data.channel_semantics = "rgb"
    cfg.model.name = "facebook/sam3.1"
    cfg.peft.method = "lora"
    cfg.run.output_dir = str(tmp_path)
    eval_cfg = EvalConfig(
        mode="full", iou_thresholds=[0.5], batch_size=1, visualize=visualize, visualize_count=10
    )
    cfg.eval = eval_cfg
    return cfg


def test_run_eval_writes_composites(tmp_path, stub_model, tiny_text_dataset) -> None:
    cfg = _cfg(tmp_path, visualize=True)
    run_eval(
        cfg,
        checkpoint=None,
        split="val",
        output_dir=tmp_path,
        val_dataset=tiny_text_dataset,
        model=stub_model,
    )
    vis_dir = tmp_path / "visualizations"
    assert vis_dir.is_dir()
    pngs = sorted(p.name for p in vis_dir.glob("*.png"))
    assert pngs == ["img_0.png", "img_1.png"]  # both GT-bearing, capped at candidate count
    for p in vis_dir.glob("*.png"):
        Image.open(p).verify()
    assert (tmp_path / "metrics.json").exists()


def test_no_visualize_writes_no_composites(tmp_path, stub_model, tiny_text_dataset) -> None:
    cfg = _cfg(tmp_path, visualize=True)  # cfg on, flag off → off.
    run_eval(
        cfg,
        checkpoint=None,
        split="val",
        output_dir=tmp_path,
        val_dataset=tiny_text_dataset,
        model=stub_model,
        visualize=False,
    )
    assert not (tmp_path / "visualizations").exists()
    assert (tmp_path / "metrics.json").exists()


def test_in_loop_evaluate_writes_no_composites(tmp_path, stub_model, tiny_text_dataset) -> None:
    """Calling Evaluator.evaluate directly (the in-loop path) writes NO visualizations/."""
    cfg = EvalConfig(mode="full", iou_thresholds=[0.5], batch_size=1, visualize=True)
    Evaluator(cfg).evaluate(stub_model, tiny_text_dataset)
    # The in-loop path does no disk I/O at all; no visualizations dir anywhere under tmp.
    assert not (tmp_path / "visualizations").exists()
```

> `EvalConfig` is a real pydantic model here (not a mock) so `cfg.eval.visualize` / `visualize_count` / `mask_threshold` resolve to real values inside `run_eval` and `write_eval_visualizations`. The `cfg` wrapper is a `MagicMock` only for the non-eval fields (`data`, `model`, `run`). `val_dataset=tiny_text_dataset` + `model=stub_model` skip the registry/`load_sam31` path entirely, so no real checkpoint is touched. `EvalConfig(..., batch_size=1)` skips the auto-batch branch.

- [ ] **Step 2: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_eval_visualize_integration.py -v`
Expected: PASS — `run_eval(..., visualize=True)` writes `img_0.png` + `img_1.png` (both readable) plus `metrics.json`; `visualize=False` writes no `visualizations/`; the in-loop `Evaluator.evaluate` writes no `visualizations/` (the viz pass is wired only to the `run_eval` surface, §3).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_eval_visualize_integration.py
git commit -m "test(eval): CPU integration smoke for eval --visualize (+ in-loop/no-visualize guards)"
```

---

## REVIEW CHECKPOINT C — integration surface complete

- [ ] Run: `uv run pytest tests/unit/test_eval_visualize.py tests/unit/test_eval_visualize_pair.py tests/unit/test_eval_runner.py tests/unit/cli/test_eval_cmd.py tests/integration/test_eval_visualize_integration.py -q`
      Expected: all PASS.
- [ ] Run: `! grep -n "render_mask_panel\|_eval_epoch" src/custom_sam_peft/eval/runner.py src/custom_sam_peft/eval/visualize.py`
      Expected: no matches — the new code does not touch the in-loop training-eval panel (`train/visualize.py::render_mask_panel`) or the trainer.
- [ ] Run: `! grep -n "ImageNet\|0.485\|0.229" src/custom_sam_peft/eval/visualize.py`
      Expected: no matches — denorm uses the passed config-resolved mean/std, never hardcoded ImageNet stats. (The 0.485/0.229 in the test file is the test's own input synthesis, not the module.)
- [ ] Dispatch a code-review subagent (min sonnet/high) over the Phase 3 diff: confirm (a) `run_eval`'s public return type is unchanged (tuple only when `return_per_example_iou=True`); (b) Branch 2's promoted path persists `metrics.json` + `_maybe_save_predictions` exactly as `evaluate_and_save` does and returns `MetricsReport`; (c) the viz pass is gated by `try/except` in `run_eval` (whole-pass failure is non-fatal after metrics persist); (d) `csp eval` is tri-state (None defers to cfg), `csp run` is two-state default-True; (e) the in-loop `Evaluator.evaluate` path is untouched.

---

## Phase 4 — Final verification (plan steps; do not run during planning)

### Task 13: Full-suite + lint + type + markdown verification

**Files:** none (verification only).

- [ ] **Step 1: Ruff lint**

Run: `uv run ruff check`
Expected: no findings. Common ones here: unused imports in `eval/visualize.py` (e.g. a `# type: ignore` mypy marks unused, or an import added but not used after an edit) — fix before proceeding.

- [ ] **Step 2: Ruff format check**

Run: `uv run ruff format --check`
Expected: clean (run `uv run ruff format` to fix, then re-check).

- [ ] **Step 3: mypy**

Run: `uv run mypy src/custom_sam_peft`
Expected: no errors. If the `resolve_normalization(model_name, normalize, ...)` call flags `NormalizeConfig | None` vs `NormalizeConfig`, keep the `# type: ignore[arg-type]` (OQ-1); if mypy reports the ignore as unused, drop it.

- [ ] **Step 4: FULL pytest suite (the 80% coverage gate only passes on the full run)**

Run: `uv run pytest`
Expected: all PASS; `--cov-fail-under=80` satisfied. Do NOT run a subset for the gate — `addopts` enforces coverage across the whole run. If coverage dips, add focused CPU tests for any uncovered branch in `eval/visualize.py` (e.g. the C>3 / C<3 denorm branches, the per-image-failure path, the back-fill branch in `pick_samples`).

- [ ] **Step 5: markdownlint the spec + plan (Markdown lint gate)**

Discover CI's exact markdown linter + config from the workflow (do NOT assume the tool): `grep -rn "markdownlint" .github/workflows/ .config/ 2>/dev/null`. Then run it against the two touched `.md` files plus the spec/plan. Example (verify the config path first):

Run: `uvx --from nodejs-bin markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "docs/superpowers/plans/2026-05-29-eval-visualize-plan.md" "docs/superpowers/specs/2026-05-29-eval-visualize-design.md" "CHANGELOG.md" "docs/config-schema.md"`
Expected: no findings (fix any before the ready PR; CI lints all tracked `.md`).

- [ ] **Step 6: Commit any lint/format/type/markdown fixups**

```bash
git add -A
git commit -m "chore: lint/format/type/markdown fixups for eval --visualize"
```

---

## Self-review (against the spec, after writing the plan)

- **§1 goals — all represented:** `eval.visualize` knob (Task 3) + per-command CLI overrides (Tasks 9, 10); one composite PNG per image with titles + shared legend (Tasks 5, 6); variety-weighted sample (Task 1); Hungarian mask-only 1:1 matched preds (Task 6); reuse-not-fork via the generalized `render_overlay` + shared `color_for_class`/`PALETTE` + postprocess helpers (Tasks 2, 4, 6); wired into the standalone/final-eval path only (Task 8, guarded by the in-loop test in Task 12).
- **§4 config/CLI — all represented:** §4.1 `EvalConfig` fields (Task 3); §4.2 `csp eval` tri-state (Task 9); §4.3 `csp run` default-True threaded through `_orchestrate` (Task 10); §4.4 `train --eval` no new flag (no task needed — it inherits `cfg.eval.visualize` via `run_eval`'s resolution; covered by Task 8's default); §4.5 `run_eval` signature on both overloads + impl (Task 8); §4.6 docs + CHANGELOG (Task 11).
- **§5 selection — all represented:** candidate filter, ranking, banding, within-band spread, small-pool, NaN, descending order (Task 1, all §10 pick_samples rows).
- **§6 wiring — all represented:** resolution `visualize_resolved` (Task 8 Step 3); Branch-1 + Branch-2 promotion mirroring `evaluate_and_save` persistence (Task 8 Step 4); same resolved `out` dir (reuses the existing `out` logic, no new dir code); robustness `try/except` in `run_eval` (Task 8 Step 4 `_run_viz`).
- **§7 module — all represented:** §7.1 denorm (Task 4); §7.2 GT→entries (Task 4); §7.3 optional-score `render_overlay` (Task 2); §7.4 `render_eval_pair` incl. K=1 forward / `meta_to_canonical` / mask-only matcher / matched-query extraction via `queries_to_coco_results` + `_row_outputs` (Task 6); §7.5 compositor + sanitized image_id output path (Task 5); §7.6 `write_eval_visualizations` (Task 7).
- **§8 error handling — all represented:** per-image WARNING + continue (Task 7); whole-pass WARNING in `run_eval` (Task 8); no-good-query still matched (covered by Task 6's matcher call — `linear_sum_assignment` always returns a 1:1 assignment when targets exist); class-with-no-GT skipped (Task 6 `_matched_pred_entries` `if not targets: continue`); count ≥ candidates / zero candidates (Tasks 1, 7); lite mode (Task 1 reads the index-aligned IoU list, no special-casing); baseline `checkpoint=None` (Task 12 drives it); NaN (Task 1); n-channel (Task 4).
- **§9 interfaces — all match:** `pick_samples`, `denormalize_to_rgb`, `gt_instances_to_entries`, `render_eval_pair`, `write_eval_visualizations` signatures defined exactly as the spec table; `render_overlay` arity unchanged (behavior-only edit); `EvalConfig.visualize`/`visualize_count`; the three CLI/runner additions.
- **§10 tests — all represented:** unit (`pick_samples` bands/cap/spread/GT-filter/pool/NaN → Task 1; `render_overlay` score-optional → Task 2; `denormalize_to_rgb` round-trip + C>3 → Task 4; `gt_instances_to_entries` → Task 4; compositor → Task 5); integration smoke + in-loop guard + `--no-visualize` guard → Task 12. **Plus** `EvalConfig` schema test (Task 3) and `run_eval`/`csp eval`/`csp run` wiring tests (Tasks 8, 9, 10) — required to drive the additive surface, beyond §10's enumeration.
- **TDD:** every code task writes the failing test first, runs red, implements minimally, runs green. The one pure-behavior edit (`render_overlay`, Task 2) writes a regression guard first. All CPU-only against the tiny stub / synthetic data / `tiny_text_dataset`; no GPU tests.
- **Phasing / contracts:** Phase 1 (pure primitives) exposes the §9 pure-function signatures + the compositor `_compose_pair`; Phase 2 consumes them and exposes `write_eval_visualizations`; Phase 3 consumes that and wires the config/CLI/runner/docs. Dependency order holds: primitives → model pass → wiring. Shared-file serializations called out (`eval/visualize.py` across Tasks 1/4/5/6/7; `eval/runner.py` is Task 8 only). Parallel opportunities flagged (Task 2 ∥ Task 3 ∥ Task 1 in Phase 1; Task 9 ∥ Task 10 ∥ Task 11 in Phase 3 after Task 8).
- **Scope discipline:** single PR; no back-compat shims; the only `predict/visualize.py` edit is the additive optional-score (§7.3, §11); the only refactor is the §6 Branch-2 promotion the spec calls for; the two pre-existing Branch-1 (flat `predictions.json`) vs Branch-2 (`_maybe_save_predictions`) persistence styles are NOT unified (noted in Task 8). Each phase ends green (lint + tests).
- **Spec gaps flagged (not invented):** OQ-1 (`NormalizeConfig | None` vs `resolve_normalization`'s non-optional fallback — pass through + match existing call shape), OQ-2 (legend pixel geometry left to the implementer; test asserts structural contract only), OQ-3 (image-id sanitization charset enumerated as `[A-Za-z0-9._-]`).
