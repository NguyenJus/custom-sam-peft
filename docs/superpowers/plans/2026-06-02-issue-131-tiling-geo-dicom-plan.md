# Large-image Tiling + Georeferencing + DICOM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When an input's longest edge exceeds the fixed SAM 3.1 model size (1008px), process it at native resolution via overlapping sliding-window tiles — and for prediction transparently restitch ONE full-extent, full-detail mask via cross-tile fragment-merge — while carrying geospatial (CRS/affine) and DICOM (3D affine) spatial metadata end-to-end through to GeoTIFF / NIfTI writers.

**Architecture:** One shared sliding-window utility (`data/tiling.py`) is the single primitive — window generation, per-window run callback, and the §4 union-find fragment-merge. A `SpatialMeta` frozen dataclass (`data/spatial_meta.py`) is an optional pixels-first sidecar returned by a new `read_image_with_meta`, carried `read → dataset → predict/writers` for output reconstruction only (never reaches the model). `rasterio` replaces `tifffile` as the base TIFF reader; `pydicom` + `nibabel` sit behind an optional `[dicom]` extra (lazy-imported). Tiling auto-engages by input size with zero new user knobs; overlap and association thresholds are internal cited constants.

**Tech Stack:** PyTorch, NumPy, rasterio (new base dep, replaces tifffile), pydicom + nibabel (new `[dicom]` extra), Pydantic v2, pytest. Source spec: `docs/superpowers/specs/2026-06-02-issue-131-tiling-geo-dicom-design.md` (anchors verified against worktree HEAD on branch `131-georeferencing-dicom`, 2026-06-02).

**Branch:** `131-georeferencing-dicom` — **one branch, one PR.** Three phases land sequentially on this branch; the final phase opens the PR.

---

## Anchor verification notes (read before starting)

Re-verified against the live worktree (2026-06-02):

- `read_image(path, channels) -> np.ndarray` lives at `data/io.py:53`; `_coerce_to_channels` at `io.py:15`; `.tif/.tiff` routes through `tifffile` at `io.py:67-71`. The `C == channels` validation is `io.py:43-49`. **#111 has landed** — `data.channels` / `data.channel_semantics` thread end-to-end.
- `SAM3_IMAGE_SIZE = 1008` is at `models/sam3.py:111`. Import it; never hardcode `1008`.
- Predict forward loop: read `predict/runner.py:424` (via local `from ...io import read_image as _read_image`), `orig_h, orig_w` at `:429`, transform `:435`, model forward `:456`, `queries_to_coco_results` `:481`, score filter `:488`, top_k `:490`, `run.json` writer `:572-592`.
- Writers PNG resize-to-original is `predict/writers.py:111-117`.
- COCO `_decode_image` → `read_image` at `coco.py:203-209`; `__getitem__` `:311`; `__len__` `:189-190`; `_fetch_raw` `:196`; `_image_ids` list `:162-164`; `channels` ctor param `:133`/`:136`.
- Train transforms `BboxParams(min_area=0.0, min_visibility=0.0)` at `transforms.py:224-228`; pad `fill=0` at `transforms.py:217`.
- Eval postprocess accumulation `eval/evaluator.py:215` (`queries_to_coco_results`), `:291` (`evaluate`); eval Hungarian matcher imported `eval/visualize.py:35`, constructed `:354`; `write_eval_visualizations` `:325`; `run_eval` `:29`/`_run_viz` `:185`.
- `pyproject.toml`: base deps `:9-29` (`tifffile>=2024.1` at `:21`); optional groups `:31-44`.

## Pinned `# tbd:` resolutions (spec §13 — all 7 resolved here; carry these comments into the code verbatim)

Every NEW default below ships with a citation OR an explicit `# tbd:` tag (CI no-uncited-default hook, spec §1.2.3). Place these as **named module constants** in `data/tiling.py` (or `pyproject.toml` for floors), not silent literals.

1. **Tile overlap fraction = `0.25`.** Cite MONAI `sliding_window_inference` documented default (`overlap=0.25`). Comment:
   `# Overlap fraction of the ROI/tile size. Cited: MONAI sliding_window_inference default overlap=0.25 (https://docs.monai.io/en/stable/inferers.html#monai.inferers.sliding_window_inference).`
2. **Association metric = intersection-over-min-area (within the overlap band).** Pinned over IoU-of-overlap-band: a small seam sliver of a large object must still link, which IoU-of-band penalizes when fragment sizes differ greatly. Comment:
   `# Cross-tile fragment association metric. Pinned: intersection-over-min-fragment-area (robust to dissimilar fragment sizes; spec §4.3). # tbd: no published canonical metric for DETR-fragment association — revisit against C2/C4 if over/under-merge observed.`
3. **`mask_overlap_threshold = 0.50`.** No published canonical value for DETR-fragment association. Comment:
   `# tbd: mask_overlap_threshold for cross-tile fragment linking. 0.50 chosen so genuine same-object seam overlap (≥half the smaller fragment lies in the band) links while incidental adjacency does not (spec §4.3). No published canonical value — tune against the C2/C3/C4 synthetic seam fixtures.`
4. **Score aggregation = area-weighted mean.** Pinned over `max` (spec §4.5): `max` lets a thin high-scoring sliver dominate; area-weighting tracks dominant evidence. Comment:
   `# Merged-fragment score aggregation. Pinned: area-weighted mean of fragment scores over plain max (spec §4.5) — area-weighting reflects the object's dominant evidence; max lets a tiny sliver dominate.`
5. **Eval tiling = non-overlapping** for metric accumulation. Pinned (spec §5.4) to avoid double-counting in the overlap band; documented as differing from predict's overlapping tiling. Comment:
   `# Eval metric tiling uses overlap=0.0 (non-overlapping) to avoid double-counting objects in the overlap band. Deliberately differs from predict's overlapping tiling (spec §5.4); visualization restitch (§4) still uses the predict overlap.`
6. **nodata fill value = `0`.** Citation-free; matches existing pad fill `fill=0` (`transforms.py:217`). Comment:
   `# nodata pixels are zero-filled before the model — matches the existing PadIfNeeded fill=0 (transforms.py:217); spec §6.3. A non-zero fill would be a NEW hyperparameter needing a citation.`
7. **Version floors:** `rasterio>=1.3` (first series with reliably bundled manylinux GDAL wheels — no system GDAL, spec §1.2.4/§10), `pydicom>=2.4` (stable `apply_modality_lut`/`apply_voi_lut` in `pydicom.pixel_data_handlers`, spec §8.1), `nibabel>=5.2` (stable `nib.affines` + Nifti1 affine API, spec §7.3). Each carries a one-line justification comment in `pyproject.toml`. **Confirm `tifffile` removal breaks nothing** by grep before removing the dep (Task 2.0).

---

## Phasing & interface contracts (one PR, three sequential phases)

A fresh session resuming a later phase can rely ONLY on the contracts below — it need not re-read earlier-phase implementation.

### Phase 1 — Tiling core (the accuracy fix, format-agnostic). Tasks 1.x

Operates on plain pixels only — no `SpatialMeta`. **EXPOSES (contract consumed by Phases 2–3 and by predict/train/eval):**

- `data/tiling.py`:
  - `Window` (frozen dataclass): `y0: int, x0: int, h: int, w: int` — origin + clamped size (`h,w ≤ tile`).
  - `iter_windows(h: int, w: int, tile: int, overlap: float) -> list[Window]` — MONAI flush-clamp; `max(h,w) <= tile` ⟹ exactly one full-image window.
  - `tiling_engaged(h: int, w: int, tile: int = SAM3_IMAGE_SIZE) -> bool` — the **single shared auto-engage rule** `max(h,w) > tile`. Predict/train/eval all call this — one rule.
  - `Fragment` (dataclass): `mask: np.ndarray (H,W bool, full-canvas)`, `score: float`, `category_id: int`, `window_id: int`.
  - `merge_fragments(fragments: list[Fragment], canvas_hw: tuple[int,int], *, threshold: float = MASK_OVERLAP_THRESHOLD) -> list[MergedInstance]` — the §4 union-find merge; `MergedInstance` has `mask: np.ndarray`, `score: float`, `category_id: int`.
  - `run_windows(image, windows, fn)` — applies `fn(crop, window) -> list[Fragment]` per window, collects fragments. (Predict/eval pass the real per-tile forward.)
  - Module constants: `DEFAULT_OVERLAP = 0.25`, `MASK_OVERLAP_THRESHOLD = 0.50`, `EVAL_OVERLAP = 0.0` (each with the §13 comment above).
- Predict tiling path wired into `predict/runner.py` (auto-engage by size; small images byte-for-byte unchanged); `run.json` gains an additive `"tiling"` provenance record.
- Train window-gen path in `coco.py` (large rasters expand into independent tile samples; `__len__` reflects expansion).
- Eval per-tile metric accumulation (non-overlapping) in `eval/evaluator.py`; eval-viz restitch via `merge_fragments`.

### Phase 2 — Georeferencing. Tasks 2.x

Introduces the `SpatialMeta` seam and promotes `rasterio`. **EXPOSES (consumed by Phase 3 + writers):**

- `data/spatial_meta.py`: `SpatialMeta` frozen dataclass (tagged union, spec §6.2 table) with `kind: Literal["geo","dicom"]` and all per-backend fields (`crs, affine, nodata, nodata_mask, pixel_spacing, orientation, position, frame_of_reference_uid, rescale, voi_window, series_uid, sop_uid`), all defaulting to `None`. Frozen.
- `data/io.py`: `read_image_with_meta(path, channels) -> tuple[np.ndarray, SpatialMeta | None]`; existing `read_image` becomes a thin wrapper returning `[0]` (signature unchanged → existing callers untouched). **Observable contract: pixels-first, `SpatialMeta=None` for plain images, model path byte-for-byte unchanged.**
- `data/tiling.py`: `tile_affine(parent_affine, window) -> affine` (parent affine offset by window origin; native-res, translation-only).
- `predict/writers.py`: `write_geotiff_mask(...)` keyed on `SpatialMeta.kind == "geo"`, re-marking `nodata_mask`.

### Phase 3 — DICOM. Tasks 3.x

Consumes the Phase 2 `SpatialMeta` seam + Phase 1 tiling. **EXPOSES (terminal — feeds writers only):**

- `[dicom]` optional extra (`pydicom` + `nibabel`); lazy-imported; missing-extra ⟹ actionable `pip install custom-sam-peft[dicom]` error on first `.dcm` access.
- `data/io.py`: `.dcm` dispatch in `read_image_with_meta` → per-slice decode (Modality LUT always; signed/bits; MONOCHROME1; VOI-if-present + override) → `(H,W,C)` pixels + `SpatialMeta(kind="dicom")`.
- DICOM series grouping (by `SeriesInstanceUID`, sorted by `ImagePositionPatient`, 3D affine).
- `config/schema.py`: `data.dicom_voi_window: tuple[float,float] | None = None` (the only new user-facing config).
- `predict/writers.py`: `write_nifti_volume(...)` keyed on `SpatialMeta.kind == "dicom"`.

---

## Conventions for every task

- **TDD:** failing test → run-fails → minimal impl → run-passes → commit. CPU tests run with `uv run pytest <path> -o "addopts=" -q` to bypass the global `--cov-fail-under=80` on CPU subsets (repo convention). NEVER `uv run pytest tests/` bare (loads the full GPU suite in one process). GPU tests (G1/G2) run only via `scripts/run_gpu_tests.sh`.
- **Lint gate per commit:** implementer commits gate on BOTH `uv run ruff check <files>` AND `uv run ruff format --check <files>` (separate; CI runs both). `assert isinstance(...)` is forbidden in `src/` (ruff S101) — narrow structurally.
- **Cite-defaults hook:** every new default literal needs the cited/`# tbd:` comment from the section above, or the pre-commit hook fails.

---

## PHASE 1 — TILING CORE

---

## Task 1.1: `Window` + `iter_windows` + `tiling_engaged` (C1, C5)

**Files:**

- Create: `src/custom_sam_peft/data/tiling.py`
- Test: `tests/unit/test_tiling_windows.py`

**Difficulty:** medium. **Gating tests:** C1 (window generation), C5 (auto-engage threshold). **Blast radius:** none (new module).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tiling_windows.py
import numpy as np

from custom_sam_peft.data.tiling import Window, iter_windows, tiling_engaged
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE


def test_small_image_single_window():
    # C1/C5: max(edge) <= tile -> exactly one full-image window (direct-path equivalent)
    ws = iter_windows(800, 1008, tile=SAM3_IMAGE_SIZE, overlap=0.25)
    assert ws == [Window(y0=0, x0=0, h=800, w=1008)]
    assert tiling_engaged(800, 1008) is False
    assert tiling_engaged(1008, 1008) is False  # boundary: exactly 1008 -> direct


def test_oversized_engages_and_covers():
    # C5: > 1008 engages tiling
    assert tiling_engaged(1500, 1500) is True
    ws = iter_windows(1500, 1500, tile=1008, overlap=0.25)
    # Cover the whole image: every pixel is inside at least one window.
    cover = np.zeros((1500, 1500), bool)
    for w in ws:
        assert w.h <= 1008 and w.w <= 1008
        cover[w.y0 : w.y0 + w.h, w.x0 : w.x0 + w.w] = True
    assert cover.all()


def test_edge_windows_clamp_flush():
    # C1: last window flush to the edge, no margin dropped
    ws = iter_windows(1009, 1009, tile=1008, overlap=0.25)
    assert max(w.x0 + w.w for w in ws) == 1009
    assert max(w.y0 + w.h for w in ws) == 1009
    # last column window is flush-right (x0 + w == 1009)
    assert any(w.x0 + w.w == 1009 and w.x0 > 0 for w in ws)


def test_overlap_band_width_matches_fraction():
    # C1: step = round(tile * (1 - overlap)); adjacent windows overlap by ~ tile*overlap
    ws = iter_windows(2000, 1008, tile=1008, overlap=0.25)
    ys = sorted({w.y0 for w in ws})
    step = ys[1] - ys[0]
    assert step == round(1008 * (1 - 0.25))  # 756
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tiling_windows.py -o "addopts=" -q`
Expected: FAIL — `custom_sam_peft.data.tiling` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/custom_sam_peft/data/tiling.py
"""Shared sliding-window tiling utility (spec §5.1). Window generation, the
auto-engage rule, a per-window run callback, and the §4 cross-tile fragment-merge.
Operates on plain pixels; geo/DICOM SpatialMeta is threaded by callers (spec §6.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

# Overlap fraction of the ROI/tile size. Cited: MONAI sliding_window_inference
# default overlap=0.25 (https://docs.monai.io/en/stable/inferers.html#monai.inferers.sliding_window_inference).
DEFAULT_OVERLAP: float = 0.25
# Eval metric tiling uses overlap=0.0 (non-overlapping) to avoid double-counting
# objects in the overlap band. Deliberately differs from predict's overlapping
# tiling (spec §5.4); the visualization restitch still uses DEFAULT_OVERLAP.
EVAL_OVERLAP: float = 0.0


@dataclass(frozen=True)
class Window:
    """A clamped sliding window: origin (y0, x0) + size (h, w), each <= tile."""

    y0: int
    x0: int
    h: int
    w: int


def tiling_engaged(h: int, w: int, tile: int = SAM3_IMAGE_SIZE) -> bool:
    """The single shared auto-engage rule (spec §5.2/§5.3/§5.4): tile iff an edge
    exceeds the model size. max(h, w) == tile takes the direct path."""
    return max(h, w) > tile


def _axis_starts(extent: int, tile: int, overlap: float) -> list[int]:
    if extent <= tile:
        return [0]
    step = max(1, round(tile * (1.0 - overlap)))
    starts = list(range(0, extent - tile + 1, step))
    last = extent - tile
    if starts[-1] != last:
        starts.append(last)  # flush-clamp the final window to the edge (MONAI convention)
    return starts


def iter_windows(h: int, w: int, tile: int = SAM3_IMAGE_SIZE, overlap: float = DEFAULT_OVERLAP) -> list[Window]:
    """Generate covering sliding windows. A <= tile image yields exactly one
    full-image window (direct-path equivalent). Edge windows clamp flush so no
    margin is dropped (spec §5.1)."""
    ys = _axis_starts(h, tile, overlap)
    xs = _axis_starts(w, tile, overlap)
    out: list[Window] = []
    for y0 in ys:
        for x0 in xs:
            out.append(Window(y0=y0, x0=x0, h=min(tile, h - y0), w=min(tile, w - x0)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tiling_windows.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/tiling.py tests/unit/test_tiling_windows.py
git commit -m "feat(tiling): window generation + shared auto-engage rule (spec §5.1; C1,C5)"
```

---

## Task 1.2: `Fragment` + union-find `merge_fragments` — the §4 crux (C2, C3, C4)

**Files:**

- Modify: `src/custom_sam_peft/data/tiling.py` (add `Fragment`, `MergedInstance`, `merge_fragments`, constants)
- Test: `tests/unit/test_tiling_merge.py`

**Difficulty:** medium (algorithm-heavy; the highest-risk task per spec §14.1). **Gating tests:** C2 (seam merge), C3 (3-tile transitive), C4 (distinct/cross-category don't merge). **Blast radius:** none (new symbols); but C2–C4 are the load-bearing guards.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tiling_merge.py
import numpy as np

from custom_sam_peft.data.tiling import Fragment, merge_fragments


def _box(canvas, y0, y1, x0, x1, cat=1, score=0.9, wid=0):
    m = np.zeros(canvas, bool)
    m[y0:y1, x0:x1] = True
    return Fragment(mask=m, score=score, category_id=cat, window_id=wid)


def test_C2_seam_object_merges_to_one():
    canvas = (40, 40)
    # one object split across a seam at x=20, overlapping in [18,22)
    left = _box(canvas, 10, 30, 5, 22, score=1.0, wid=0)
    right = _box(canvas, 10, 30, 18, 35, score=0.6, wid=1)
    merged = merge_fragments([left, right], canvas)
    assert len(merged) == 1
    ref = left.mask | right.mask
    assert np.array_equal(merged[0].mask, ref)  # logical OR within the component
    # area-weighted mean of scores (left larger -> closer to 1.0)
    la, ra = left.mask.sum(), right.mask.sum()
    expected = (1.0 * la + 0.6 * ra) / (la + ra)
    assert abs(merged[0].score - expected) < 1e-6


def test_C3_three_tile_transitive_merge():
    canvas = (20, 90)
    a = _box(canvas, 5, 15, 0, 35, wid=0)
    b = _box(canvas, 5, 15, 30, 65, wid=1)  # overlaps a in [30,35)
    c = _box(canvas, 5, 15, 60, 90, wid=2)  # overlaps b in [60,65); NOT a
    merged = merge_fragments([a, b, c], canvas)
    assert len(merged) == 1  # transitive A-B-C via union-find


def test_C4_distinct_and_cross_category_stay_separate():
    canvas = (20, 60)
    # same category, NO band overlap -> separate
    o1 = _box(canvas, 5, 15, 0, 20, cat=1, wid=0)
    o2 = _box(canvas, 5, 15, 30, 50, cat=1, wid=1)
    assert len(merge_fragments([o1, o2], canvas)) == 2
    # overlapping but DIFFERENT category -> never merge
    a = _box(canvas, 5, 15, 0, 30, cat=1, wid=0)
    b = _box(canvas, 5, 15, 20, 50, cat=2, wid=1)
    assert len(merge_fragments([a, b], canvas)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tiling_merge.py -o "addopts=" -q`
Expected: FAIL — `Fragment` / `merge_fragments` not defined.

- [ ] **Step 3: Write the implementation**

Append to `tiling.py`:

```python
# tbd: mask_overlap_threshold for cross-tile fragment linking. 0.50 chosen so
# genuine same-object seam overlap (>= half the smaller fragment lies in the band)
# links while incidental adjacency does not (spec §4.3). No published canonical
# value for DETR-fragment association — tune against C2/C3/C4 synthetic fixtures.
MASK_OVERLAP_THRESHOLD: float = 0.50


@dataclass
class Fragment:
    """A per-tile instance placed on the full-image canvas (spec §4.2)."""

    mask: np.ndarray  # (H, W) bool, full-canvas coordinates
    score: float
    category_id: int
    window_id: int


@dataclass
class MergedInstance:
    mask: np.ndarray
    score: float
    category_id: int


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, a: int) -> int:
        while self._p[a] != a:
            self._p[a] = self._p[self._p[a]]
            a = self._p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self._p[self.find(a)] = self.find(b)


def _intersection_over_min(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    smaller = min(int(a.sum()), int(b.sum()))
    # Association metric. Pinned: intersection-over-min-fragment-area (robust to
    # dissimilar fragment sizes; spec §4.3). # tbd: no published canonical metric
    # for DETR-fragment association — revisit against C2/C4 if over/under-merge.
    return inter / smaller if smaller else 0.0


def merge_fragments(
    fragments: list[Fragment],
    canvas_hw: tuple[int, int],
    *,
    threshold: float = MASK_OVERLAP_THRESHOLD,
) -> list[MergedInstance]:
    """Cross-tile fragment MERGE (spec §4): union-find over the overlap graph,
    per category, fragments from different windows only. Logical-OR each component;
    score = area-weighted mean of fragment scores."""
    n = len(fragments)
    uf = _UnionFind(n)
    for i in range(n):
        for k in range(i + 1, n):
            fi, fk = fragments[i], fragments[k]
            if fi.category_id != fk.category_id:
                continue  # different categories never merge (spec §4.1)
            if fi.window_id == fk.window_id:
                continue  # fragments within one tile are already distinct instances
            if _intersection_over_min(fi.mask, fk.mask) > threshold:
                uf.union(i, k)

    comps: dict[int, list[int]] = {}
    for idx in range(n):
        comps.setdefault(uf.find(idx), []).append(idx)

    out: list[MergedInstance] = []
    for members in comps.values():
        mask = np.zeros(canvas_hw, bool)
        weighted, total_area = 0.0, 0
        for m in members:
            frag = fragments[m]
            mask |= frag.mask  # logical OR within a confirmed object (spec §4.5)
            area = int(frag.mask.sum())
            weighted += frag.score * area
            total_area += area
        score = weighted / total_area if total_area else 0.0  # area-weighted mean
        out.append(MergedInstance(mask=mask, score=score, category_id=fragments[members[0]].category_id))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tiling_merge.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/tiling.py tests/unit/test_tiling_merge.py
git commit -m "feat(tiling): union-find cross-tile fragment merge — the §4 crux (C2,C3,C4)"
```

---

## Task 1.3: `run_windows` per-window callback (C1 coverage helper)

**Files:**

- Modify: `src/custom_sam_peft/data/tiling.py` (add `run_windows`)
- Test: `tests/unit/test_tiling_run.py`

**Difficulty:** easy-medium. **Gating tests:** new C1-adjacent unit (collect-fragments-with-origin). **Blast radius:** none.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tiling_run.py
import numpy as np

from custom_sam_peft.data.tiling import Fragment, iter_windows, run_windows


def test_run_windows_offsets_fragments_to_canvas():
    img = np.zeros((1500, 1500, 3), np.uint8)
    windows = iter_windows(1500, 1500, tile=1008, overlap=0.25)

    def fn(crop, window):
        # A fake per-tile "forward": one fragment, a 5x5 box at tile-local (0,0).
        m = np.zeros(crop.shape[:2], bool)
        m[0:5, 0:5] = True
        return [Fragment(mask=m, score=1.0, category_id=1, window_id=window.y0 * 99999 + window.x0)]

    frags = run_windows(img, windows, fn)
    assert len(frags) == len(windows)
    # Each returned fragment mask must be full-canvas sized, offset by window origin.
    for f, win in zip(frags, windows):
        assert f.mask.shape == (1500, 1500)
        assert f.mask[win.y0, win.x0]  # box landed at the window origin on the canvas
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tiling_run.py -o "addopts=" -q`
Expected: FAIL — `run_windows` not defined.

- [ ] **Step 3: Write the implementation**

Append to `tiling.py`:

```python
from collections.abc import Callable


def run_windows(
    image: np.ndarray,
    windows: list[Window],
    fn: Callable[[np.ndarray, Window], list[Fragment]],
) -> list[Fragment]:
    """Apply `fn(crop, window)` to each window's crop and collect fragments,
    re-placing each tile-local fragment mask onto the full-image canvas at the
    window origin (spec §5.1). `fn` returns fragments whose masks are tile-local
    (H_win, W_win); this offsets them to full-canvas coordinates."""
    h, w = image.shape[0], image.shape[1]
    collected: list[Fragment] = []
    for win in windows:
        crop = image[win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w]
        for frag in fn(crop, win):
            canvas = np.zeros((h, w), bool)
            canvas[win.y0 : win.y0 + win.h, win.x0 : win.x0 + win.w] = frag.mask[: win.h, : win.w]
            collected.append(
                Fragment(mask=canvas, score=frag.score, category_id=frag.category_id, window_id=frag.window_id)
            )
    return collected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tiling_run.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/tiling.py tests/unit/test_tiling_run.py
git commit -m "feat(tiling): run_windows per-window callback + canvas re-placement (spec §5.1)"
```

---

## Task 1.4: PREDICT tiling path — tile → run → merge → ONE full-extent mask (C5)

**Files:**

- Modify: `src/custom_sam_peft/predict/runner.py` — per-image branch around the read/transform/forward (`:420-491`); `run.json` record (`:572-592`)
- Test: `tests/unit/test_predict_tiling_unit.py`

**Difficulty:** medium. **Gating tests:** C5 (direct path unchanged for small images via a unit harness); G1 gates the real-model end-to-end (Phase-1 GPU). **Blast radius:** touches the predict forward loop — the small-image path MUST stay byte-for-byte (C5). Run the predict unit suite (`tests/unit/test_predict*`) before "done", not just the new test.

**Design note (per spec §5.2):** factor the existing single-tile body (transform → model forward → `queries_to_coco_results` → score/top_k filter, `:435-491`) into a reusable `_predict_one_tile(crop, ...) -> list[Fragment]` so both the direct path (one whole-image "tile") and the tiling path call the SAME forward. Auto-engage on `tiling_engaged(orig_h, orig_w)`. On the tiling path: `iter_windows` → `run_windows(_predict_one_tile)` → `merge_fragments` → convert merged instances to the same per-(image, category) entry dicts (RLE-encoded), then **skip** the writers' resize step (canvas is already full-extent; spec §5.2.3). The direct path keeps the resize (`writers.py:111-117`). Add the `"tiling"` provenance dict to `run_meta`.

- [ ] **Step 1: Write the failing test (C5 — direct path unchanged + tiling engages)**

```python
# tests/unit/test_predict_tiling_unit.py
"""CPU-only: assert the auto-engage decision and that the tiling path produces
merged full-canvas entries WITHOUT loading the real SAM model. We exercise the
tiling helper that predict calls, with a stub per-tile forward."""
import numpy as np

from custom_sam_peft.data.tiling import Fragment, iter_windows, merge_fragments, run_windows, tiling_engaged


def test_small_image_takes_direct_path():
    assert tiling_engaged(900, 1008) is False


def test_oversized_image_tiles_and_merges_seam_object():
    img = np.zeros((1500, 1500, 3), np.uint8)
    assert tiling_engaged(1500, 1500) is True
    windows = iter_windows(1500, 1500, tile=1008, overlap=0.25)

    # stub forward: emit a fragment covering the left/right halves of a seam object
    def fake_forward(crop, window):
        m = np.zeros(crop.shape[:2], bool)
        m[700:760, :] = True  # a horizontal bar crossing every vertical seam
        return [Fragment(mask=m, score=0.9, category_id=1, window_id=id(window))]

    frags = run_windows(img, windows, fake_forward)
    merged = merge_fragments(frags, (1500, 1500))
    assert all(mi.mask.shape == (1500, 1500) for mi in merged)  # ONE full-extent canvas
```

- [ ] **Step 2: Run test to verify it fails (then becomes the regression guard)**

Run: `uv run pytest tests/unit/test_predict_tiling_unit.py -o "addopts=" -q`
Expected: initially PASS at the tiling-utility level (utilities exist). The IMPLEMENTATION work is wiring `predict/runner.py` to call them — verify by running the existing predict unit suite after wiring (Step 4).

- [ ] **Step 3: Wire `predict/runner.py`**

Refactor the per-image inner body (`:435-491`) into `_predict_one_tile(crop_np, model, prompts, opts, rcfg, image_id, orig_hw, ladder) -> list[Fragment]` returning tile-local fragments (mask at the tile's own resolution + its `category_id` + `score`). In the per-image loop:

```python
from custom_sam_peft.data.tiling import iter_windows, merge_fragments, run_windows, tiling_engaged, DEFAULT_OVERLAP

orig_h, orig_w = img_np.shape[0], img_np.shape[1]
if not tiling_engaged(orig_h, orig_w):
    entries = _entries_for_image_direct(img_np, ...)  # existing path: transform 1008² + resize-back in writer
    n_windows = 1
    tiled = False
else:
    windows = iter_windows(orig_h, orig_w, tile=SAM3_IMAGE_SIZE, overlap=DEFAULT_OVERLAP)
    frags = run_windows(img_np, windows, lambda crop, win: _predict_one_tile(crop, win, ...))
    merged = merge_fragments(frags, (orig_h, orig_w))
    # Re-apply per-(image, category) score_threshold + top_k on MERGED instances (spec §4.6),
    # then RLE-encode each at full-canvas extent (NO writer resize on this path).
    entries = _merged_to_entries(merged, image_id, opts.score_threshold, opts.top_k)
    n_windows = len(windows)
    tiled = True
```

Mark per-image whether it tiled so `write_predictions` skips the resize for tiled entries (full-extent already) — simplest: tiled entries' RLE mask is already `(orig_h, orig_w)`, so the existing `if mask_arr.shape != (h, w)` guard at `writers.py:113` is a no-op. **No writer change needed** — confirm the guard short-circuits.

Add to `run_meta` (`:572-591`):

```python
"tiling": {"engaged": any_tiled, "tile": SAM3_IMAGE_SIZE, "overlap": DEFAULT_OVERLAP, "n_windows_total": total_windows},
```

- [ ] **Step 4: Run the predict unit suite + the new test**

Run: `uv run pytest tests/unit/test_predict_tiling_unit.py tests/unit/ -o "addopts=" -q -k "predict or tiling"`
Expected: PASS — small-image path produces the same entries as before (regression), tiling path emits full-extent merged entries.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/predict/runner.py tests/unit/test_predict_tiling_unit.py
git commit -m "feat(predict): tiling path — tile->run->merge->one full-extent mask + run.json provenance (spec §5.2; C5)"
```

---

## Task 1.5: TRAIN tile expansion — large rasters → independent tile samples (C6)

**Files:**

- Modify: `src/custom_sam_peft/data/coco.py` — `__init__` (`:124-187`), `__len__` (`:189`), `_fetch_raw` (`:196`), `__getitem__` (`:311`), `_decode_targets` (`:211`)
- Test: `tests/unit/test_coco_tiling.py`

**Difficulty:** medium-hard (index-space change with subtle invariants). **Gating tests:** C6 (tile expansion + clip + empty-negative + deterministic `__len__`). **Blast radius (HIGH):** changes dataset `__len__` / index mapping. Per spec §14.5 + memory #245: the **`data.limit` subset cap and the no-val auto-split must see the POST-EXPANSION index space**. Run the FULL relevant suite (`tests/unit/test_data_coco.py`, `test_data_subset_limit*`, `test_no_val_auto_split*`) before "done".

**Design note:** at construction, pre-enumerate `(image_idx, window)` pairs (deterministic; supports `data.limit` + shuffling per spec §5.3). For each kept image, read only its dimensions cheaply (use COCO `width`/`height` record fields if present; else read the image header — DO NOT decode full pixels at construction on this I/O-fragile box). If `tiling_engaged(h, w)`, expand into one entry per `iter_windows`; else one entry (whole-image window). `__len__` returns the expanded count. `_fetch_raw(i)` maps `i → (image_id, window)`; `__getitem__` decodes the image, crops to the window, and clips annotations to the window before transforms. Reuse `BboxParams(min_area=0.0, min_visibility=0.0)` (`transforms.py:224-228`) clip semantics; an empty post-clip tile is a valid negative.

- [ ] **Step 1: Write the failing test (C6)**

```python
# tests/unit/test_coco_tiling.py
"""CPU-only: construct a COCODataset over a synthetic oversized raster + COCO
annotations; assert tile-expanded __len__, per-window clipping, empty-tile negatives."""
import json

import numpy as np
import pytest
from PIL import Image

from custom_sam_peft.config.schema import TextPromptConfig
from custom_sam_peft.data.coco import COCODataset


@pytest.fixture
def oversized_coco(tmp_path):
    img = (np.random.rand(1500, 1500, 3) * 255).astype(np.uint8)
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir()
    Image.fromarray(img).save(imgs_dir / "big.png")
    coco = {
        "images": [{"id": 1, "file_name": "big.png", "width": 1500, "height": 1500}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0,
             "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]},
        ],
        "categories": [{"id": 1, "name": "thing"}],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    return str(ann), str(imgs_dir)


def test_C6_oversized_raster_expands_into_tiles(oversized_coco, _eval_transforms):
    ann, imgs = oversized_coco
    ds = COCODataset(annotations=ann, images=imgs, transforms=_eval_transforms,
                     text_prompt=TextPromptConfig(), channels=3)
    # 1500x1500 @ tile 1008 overlap 0.25 -> 2x2 = 4 windows -> len == 4 (one image)
    assert len(ds) == 4
    # tile containing the top-left object yields >=1 instance; an empty tile is valid.
    n_with, n_empty = 0, 0
    for k in range(len(ds)):
        ex = ds[k]
        (n_with := n_with + 1) if len(ex.instances) else (n_empty := n_empty + 1)
    assert n_with >= 1 and n_empty >= 1  # empty tiles are valid negatives
```

(Provide a `_eval_transforms` fixture in the test building `build_eval_transforms(1008, model_name="<test>", normalize=NormalizeConfig())`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_coco_tiling.py -o "addopts=" -q`
Expected: FAIL — `__len__` returns 1 (no tile expansion yet).

- [ ] **Step 3: Implement tile expansion**

In `__init__`, after `self._image_ids` is finalized (`:164`), build `self._samples: list[tuple[int, Window]]` by reading each image's `width`/`height` from the COCO record (fall back to a header-only PIL read; never full-decode). Use `iter_windows` when `tiling_engaged`, else a single whole-image `Window(0,0,h,w)`. Set `__len__` to `len(self._samples)`. `_fetch_raw(i)` resolves `self._samples[i] = (image_id, window)`. In `__getitem__`, crop `np_img` to the window and clip boxes/masks to the window (offset by `-x0,-y0`, intersect, drop sub-floor boxes). Keep the `data.limit`/auto-split callers operating on `len(self._samples)` (verify their index source is `len(dataset)`, not `len(self._image_ids)`).

- [ ] **Step 4: Run C6 + the blast-radius suite**

Run: `uv run pytest tests/unit/test_coco_tiling.py tests/unit/test_data_coco.py -o "addopts=" -q`
Then: `uv run pytest tests/unit/ -o "addopts=" -q -k "subset_limit or auto_split or coco"`
Expected: PASS — tile expansion is deterministic; limit/split see the expanded index space.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/coco.py tests/unit/test_coco_tiling.py
git commit -m "feat(train): window large rasters into independent tile samples (spec §5.3; C6)"
```

---

## Task 1.6: EVAL per-tile metric accumulation (non-overlapping) + viz restitch

**Files:**

- Modify: `src/custom_sam_peft/eval/evaluator.py` — per-image postprocess (`:206-228`); engage tiling per `tiling_engaged`
- Modify: `src/custom_sam_peft/eval/visualize.py` — `write_eval_visualizations` (`:325`) restitch via `merge_fragments`
- Test: `tests/unit/test_eval_tiling_unit.py`

**Difficulty:** medium. **Gating tests:** new CPU unit (non-overlap windows for accumulation; viz uses merge); G2 gates the real-model accumulation. **Blast radius:** eval forward path — keep the small-image direct path byte-for-byte. Run `tests/unit/test_eval*` before "done".

**Design note (spec §5.4):** when `tiling_engaged`, tile BOTH prediction and GT with `iter_windows(..., overlap=EVAL_OVERLAP)` (non-overlapping — the §13.5 pin); run the per-tile forward; Hungarian-match per tile (matcher already in eval); accumulate metrics across tiles **without** materializing a stitched mask. Restitch (`merge_fragments`) is invoked **only** by `write_eval_visualizations` (uses `DEFAULT_OVERLAP`, not `EVAL_OVERLAP`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_tiling_unit.py
from custom_sam_peft.data.tiling import EVAL_OVERLAP, iter_windows, tiling_engaged


def test_eval_uses_non_overlapping_tiling():
    assert EVAL_OVERLAP == 0.0
    ws = iter_windows(2016, 2016, tile=1008, overlap=EVAL_OVERLAP)
    # non-overlapping: exactly 4 disjoint 1008x1008 windows, no shared band
    assert len(ws) == 4
    starts = sorted({w.y0 for w in ws})
    assert starts == [0, 1008]


def test_small_eval_image_direct_path():
    assert tiling_engaged(700, 700) is False
```

- [ ] **Step 2: Run test to verify it fails / confirm constant**

Run: `uv run pytest tests/unit/test_eval_tiling_unit.py -o "addopts=" -q`
Expected: PASS at the constant level; implementation work is wiring `evaluator.py` + `visualize.py` (verify via Step 4 suite).

- [ ] **Step 3: Wire eval**

In `evaluator.py` per-image postprocess, branch on `tiling_engaged(orig_h, orig_w)`: direct path unchanged; tiling path generates non-overlapping windows, runs the per-tile forward, accumulates per-tile entries against per-tile GT (no stitch). In `visualize.py:write_eval_visualizations`, when the source tiled, call `merge_fragments(..., threshold=MASK_OVERLAP_THRESHOLD)` on per-tile prediction fragments (overlap=DEFAULT_OVERLAP) to render one coherent overlay.

- [ ] **Step 4: Run eval suite**

Run: `uv run pytest tests/unit/test_eval_tiling_unit.py tests/unit/ -o "addopts=" -q -k "eval"`
Expected: PASS — small images unchanged; tiled accumulation is non-overlapping; viz restitches.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/evaluator.py src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_tiling_unit.py
git commit -m "feat(eval): per-tile metric accumulation (non-overlap) + viz restitch (spec §5.4)"
```

---

## Task 1.7: GPU end-to-end — tiled predict (G1) + tiled eval accumulation (G2)

**Files:**

- Create: `tests/gpu/test_tiling_gpu.py`

**Difficulty:** medium. **Gating tests:** G1, G2 (real SAM 3.1; run ONLY via `scripts/run_gpu_tests.sh`). **Blast radius:** none (test-only).

- [ ] **Step 1: Write the GPU tests**

```python
# tests/gpu/test_tiling_gpu.py
import pytest

pytestmark = pytest.mark.gpu


def test_G1_tiled_predict_one_full_extent_mask(tmp_path):
    """run_predict on an oversized image with the real model: exactly one
    full-extent mask per object, no tiles leak, a seam-crossing object is ONE instance."""
    # build an oversized synthetic image with one wide object straddling a seam;
    # run run_predict; assert outputs are at original extent and the seam object is single.
    ...


def test_G2_tiled_eval_accumulates_without_stitch(tmp_path):
    """run_eval on an oversized eval sample accumulates per-tile metrics without
    materializing a stitched mask; the visualize path renders one stitched overlay."""
    ...
```

(Implementer fills the bodies using the existing GPU-test harness patterns in `tests/gpu/`.)

- [ ] **Step 2: Run via the GPU harness**

Run: `scripts/run_gpu_tests.sh tests/gpu/test_tiling_gpu.py`
Expected: PASS on the 5070 Ti.

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_tiling_gpu.py
git commit -m "test(gpu): G1 tiled predict + G2 tiled eval accumulation (spec §12.2)"
```

---

## PHASE 2 — GEOREFERENCING

> **Contract consumed from Phase 1:** `data/tiling.py` (`Window`, `iter_windows`, `tiling_engaged`, `merge_fragments`, `run_windows`, constants). This phase adds `tile_affine` to it and the `SpatialMeta` seam.

---

## Task 2.0: Confirm `tifffile` removal is safe (blast-radius grep)

**Files:**

- Inspect only (no code change yet).

**Difficulty:** easy. **Gating tests:** none (investigation). **Blast radius (HIGH):** removing a base dep.

- [ ] **Step 1: Grep all `tifffile` importers**

Run: `grep -rn "tifffile" src/ tests/`
Expected: the ONLY `src/` import is `data/io.py:68` (verified). If any other `src/` site imports it, list it in the PR description and migrate it in Task 2.2. Test-only usages (fixtures writing `.tif`) may switch to rasterio or stay if dev-only — note them.

- [ ] **Step 2: Record findings** in the PR description / Task 2.2 notes. No commit.

---

## Task 2.1: `SpatialMeta` dataclass (C13 scaffold)

**Files:**

- Create: `src/custom_sam_peft/data/spatial_meta.py`
- Test: `tests/unit/test_spatial_meta.py`

**Difficulty:** easy-medium. **Gating tests:** C13 (None for plain). **Blast radius:** none (new module).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_spatial_meta.py
import dataclasses

from custom_sam_peft.data.spatial_meta import SpatialMeta


def test_geo_kind_carries_crs_affine_nodata():
    m = SpatialMeta(kind="geo", crs="EPSG:4326", affine=(1, 0, 0, 0, 1, 0), nodata=0.0)
    assert m.kind == "geo"
    assert m.crs == "EPSG:4326"
    assert m.orientation is None  # dicom-only field defaults None


def test_is_frozen():
    m = SpatialMeta(kind="geo")
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        m.crs = "EPSG:3857"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_spatial_meta.py -o "addopts=" -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the dataclass (spec §6.2 table)**

```python
# src/custom_sam_peft/data/spatial_meta.py
"""Optional pixels-first spatial-metadata sidecar (spec §6.2). Tagged union by
source `kind`. NEVER reaches the model — carried read->dataset->writers for
output reconstruction only. Default None for plain images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SpatialMeta:
    kind: Literal["geo", "dicom"]
    # geo (rasterio)
    crs: Any = None
    affine: Any = None
    nodata: float | None = None
    nodata_mask: Any = None  # bool ndarray | None
    # dicom (pydicom)
    pixel_spacing: Any = None
    orientation: Any = None  # ImageOrientationPatient
    position: Any = None  # ImagePositionPatient
    frame_of_reference_uid: str | None = None
    rescale: tuple[float, float] | None = None  # (slope, intercept)
    voi_window: tuple[float, float] | None = None  # (center, width)
    series_uid: str | None = None
    sop_uid: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_spatial_meta.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/spatial_meta.py tests/unit/test_spatial_meta.py
git commit -m "feat(data): SpatialMeta optional spatial-metadata sidecar (spec §6.2)"
```

---

## Task 2.2: Promote `rasterio` to base, replacing `tifffile`; `read_image_with_meta` (C7, C13, C14)

**Files:**

- Modify: `pyproject.toml:21` (replace `tifffile` with `rasterio`)
- Modify: `src/custom_sam_peft/data/io.py` — `.tif/.tiff` branch (`:67-71`); add `read_image_with_meta`
- Test: `tests/unit/test_data_io.py` (append), `tests/unit/test_io_geo.py`

**Difficulty:** medium. **Gating tests:** C7 (CRS/affine round-trip read side), C13 (None for plain), C14 (rasterio replaces tifffile; band!=channels error; no tifffile import). **Blast radius (HIGH):** signature seam. `read_image` keeps its exact signature (wrapper) so `coco.py:209` / `predict/runner.py:424` / HF loader are untouched. Verify with the full data-io suite.

- [ ] **Step 1: Replace the dependency**

In `pyproject.toml`, replace line `:21`:

```toml
  # rasterio replaces tifffile as the base TIFF reader (spec §6.3/§10): adds CRS +
  # affine + nodata. Floor 1.3: first series with reliably bundled manylinux GDAL
  # wheels (no system GDAL required).
  "rasterio>=1.3",
```

Run: `uv lock && uv run python -c "import rasterio; print(rasterio.__version__)"`
Expected: prints `>= 1.3`.

- [ ] **Step 2: Write the failing tests (C7 read-side + C13 + C14)**

```python
# tests/unit/test_io_geo.py
import numpy as np
import pytest

from custom_sam_peft.data.io import read_image, read_image_with_meta


def _write_geotiff(path, arr_hwc, crs="EPSG:32633", nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    h, w, c = arr_hwc.shape
    transform = from_origin(500000, 4600000, 10, 10)  # 10m pixels
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=c,
                       dtype=arr_hwc.dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        for b in range(c):
            dst.write(arr_hwc[:, :, b], b + 1)
    return transform


def test_C7_geotiff_read_carries_crs_affine(tmp_path):
    arr = (np.random.rand(20, 24, 3) * 255).astype(np.uint8)
    p = tmp_path / "geo.tif"
    transform = _write_geotiff(p, arr)
    pixels, meta = read_image_with_meta(p, 3)
    assert pixels.shape == (20, 24, 3)
    assert meta is not None and meta.kind == "geo"
    assert "32633" in str(meta.crs)
    assert tuple(meta.affine)[:6] == tuple(transform)[:6]


def test_C13_plain_tiff_returns_none_meta(tmp_path):
    import tifffile  # dev fixture only — plain non-geo TIFF

    arr = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, arr)
    pixels, meta = read_image_with_meta(p, 3)
    assert pixels.shape == (8, 10, 3)
    assert meta is None  # non-geo TIFF -> SpatialMeta None (behaviour identical to old path)


def test_C14_band_count_mismatch_raises(tmp_path):
    arr = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = tmp_path / "rgb.tif"
    _write_geotiff(p, arr)
    with pytest.raises(ValueError, match=r"channels=4"):
        read_image(p, 4)
```

- [ ] **Step 3: Implement the rasterio branch + `read_image_with_meta`**

```python
def read_image_with_meta(path, channels):
    """Read pixels + optional SpatialMeta (spec §6.2). Pixels-first; meta is None
    for plain images. `read_image` is a thin wrapper returning pixels only."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        return _read_tiff_rasterio(path, channels)
    if ext == ".dcm":
        from custom_sam_peft.data.dicom_io import read_dcm_with_meta  # Phase 3

        return read_dcm_with_meta(path, channels)
    return read_image(path, channels), None  # plain raster/npy: meta None


def _read_tiff_rasterio(path, channels):
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()  # (C, H, W)
        pixels = _coerce_to_channels(arr, channels)  # preserves C == channels validation
        if src.crs is None and src.transform.is_identity:
            return pixels, None  # plain non-geo TIFF -> behaviour identical to old tifffile path
        nodata = src.nodata
        nodata_mask = None
        if nodata is not None:
            nodata_mask = np.any(arr == nodata, axis=0)
            # nodata pixels zero-filled before the model — matches PadIfNeeded fill=0
            # (transforms.py:217); spec §6.3.
            pixels = pixels.copy()
            pixels[nodata_mask] = 0
        from custom_sam_peft.data.spatial_meta import SpatialMeta

        return pixels, SpatialMeta(kind="geo", crs=src.crs, affine=src.transform,
                                   nodata=nodata, nodata_mask=nodata_mask)
```

Change the existing `.tif/.tiff` branch in `read_image` to call `_read_tiff_rasterio(...)[0]` (drops meta). Remove the `import tifffile`.

- [ ] **Step 4: Run tests + the full data-io suite**

Run: `uv run pytest tests/unit/test_io_geo.py tests/unit/test_data_io.py -o "addopts=" -q`
Then: `grep -rn "import tifffile" src/` (expected: zero hits in `src/`).
Expected: PASS; no `tifffile` import remains in `src/`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/custom_sam_peft/data/io.py tests/unit/test_io_geo.py tests/unit/test_data_io.py
git commit -m "feat(io): rasterio replaces tifffile + read_image_with_meta + nodata zero-fill (spec §6.3,§10; C7,C13,C14,C9)"
```

---

## Task 2.3: Tiling × geo affine composition — `tile_affine` (C8)

**Files:**

- Modify: `src/custom_sam_peft/data/tiling.py` (add `tile_affine`)
- Test: `tests/unit/test_tiling_affine.py`

**Difficulty:** medium. **Gating tests:** C8 (per-tile affine = parent offset by origin; stitched affine == parent). **Blast radius:** none (new function on Phase-1 module).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tiling_affine.py
from rasterio.transform import from_origin

from custom_sam_peft.data.tiling import Window, tile_affine


def test_C8_tile_affine_is_parent_offset_by_origin():
    parent = from_origin(500000, 4600000, 10, 10)  # 10m pixels
    win = Window(y0=100, x0=200, h=300, w=400)
    child = tile_affine(parent, win)
    # world coord of tile pixel (0,0) == world coord of parent pixel (x0, y0)
    assert child * (0, 0) == parent * (win.x0, win.y0)
    # native-res: no scale change (a, e components identical)
    assert (child.a, child.e) == (parent.a, parent.e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tiling_affine.py -o "addopts=" -q`
Expected: FAIL — `tile_affine` not defined.

- [ ] **Step 3: Implement**

```python
def tile_affine(parent_affine, window):
    """Per-tile affine = parent affine offset by the window origin (y0, x0) in
    pixel space — a pure pixel translation, no scale change (tiles are native-res;
    spec §6.4). The stitched output keeps the parent affine."""
    from affine import Affine

    return parent_affine * Affine.translation(window.x0, window.y0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tiling_affine.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/tiling.py tests/unit/test_tiling_affine.py
git commit -m "feat(tiling): tile_affine — parent affine offset by window origin (spec §6.4; C8)"
```

---

## Task 2.4: GeoTIFF mask writer (C7 write-side round-trip, C9 nodata re-mark)

**Files:**

- Modify: `src/custom_sam_peft/predict/writers.py` (add `write_geotiff_mask`)
- Modify: `src/custom_sam_peft/predict/runner.py` — select GeoTIFF when `SpatialMeta.kind == "geo"` (thread meta through the per-image loop)
- Test: `tests/unit/test_writers_geotiff.py`

**Difficulty:** medium. **Gating tests:** C7 (full round-trip: write → re-read → CRS+affine exact), C9 (nodata re-marked). **Blast radius:** writer selection now depends on `SpatialMeta`; the predict loop must carry meta from `read_image_with_meta` (switch its read from `read_image` to `read_image_with_meta` at `runner.py:424`). Run the predict suite.

- [ ] **Step 1: Write the failing test (C7 + C9)**

```python
# tests/unit/test_writers_geotiff.py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from custom_sam_peft.data.spatial_meta import SpatialMeta
from custom_sam_peft.predict.writers import write_geotiff_mask


def test_C7_C9_geotiff_mask_roundtrips_crs_affine_and_nodata(tmp_path):
    transform = from_origin(500000, 4600000, 10, 10)
    nodata_mask = np.zeros((20, 24), bool)
    nodata_mask[0, :] = True  # top row is nodata
    meta = SpatialMeta(kind="geo", crs="EPSG:32633", affine=transform, nodata=0, nodata_mask=nodata_mask)
    mask = np.ones((20, 24), np.uint8)
    out = tmp_path / "mask.tif"
    write_geotiff_mask(mask, meta, out)
    with rasterio.open(out) as src:
        assert "32633" in str(src.crs)
        assert tuple(src.transform)[:6] == tuple(transform)[:6]
        read = src.read(1)
        assert (read[0, :] == 0).all()  # nodata re-marked to 0 in the output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_writers_geotiff.py -o "addopts=" -q`
Expected: FAIL — `write_geotiff_mask` not defined.

- [ ] **Step 3: Implement**

```python
def write_geotiff_mask(mask, meta, out_path):
    """Write a same-extent GeoTIFF mask carrying the source CRS + affine
    (spec §7.1). nodata pixels (meta.nodata_mask) are re-marked to 0 in the output."""
    import numpy as np
    import rasterio

    arr = np.asarray(mask, np.uint8)
    if meta.nodata_mask is not None:
        arr = arr.copy()
        arr[meta.nodata_mask] = 0
    h, w = arr.shape
    with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs=meta.crs, transform=meta.affine,
                       nodata=0 if meta.nodata_mask is not None else None) as dst:
        dst.write(arr, 1)
```

Wire predict: at `runner.py:424` switch to `img_np, spatial_meta = read_image_with_meta(img_path, rcfg.channels)`; carry `spatial_meta` in the per-image meta; in the writer-selection step, when `spatial_meta is not None and spatial_meta.kind == "geo"`, emit a GeoTIFF mask alongside (PNG/RLE remain available per spec §7.1).

- [ ] **Step 4: Run tests + predict suite**

Run: `uv run pytest tests/unit/test_writers_geotiff.py tests/unit/ -o "addopts=" -q -k "writer or predict or geo"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/predict/writers.py src/custom_sam_peft/predict/runner.py tests/unit/test_writers_geotiff.py
git commit -m "feat(writers): GeoTIFF mask carrying source CRS+affine, nodata re-mark (spec §7.1; C7,C9)"
```

---

## PHASE 3 — DICOM

> **Contract consumed from Phase 2:** `SpatialMeta` (`data/spatial_meta.py`), `read_image_with_meta` dispatch (it already routes `.dcm` to `data/dicom_io.read_dcm_with_meta`), the writer-selection seam in `predict/runner.py`. This phase fills the `.dcm` decode, series grouping, NIfTI writer, and the `[dicom]` extra.

---

## Task 3.1: `[dicom]` extra + lazy-import guard (C12)

**Files:**

- Modify: `pyproject.toml:31-44` (add `[dicom]` optional group)
- Create: `src/custom_sam_peft/data/dicom_io.py` (with the missing-extra guard)
- Test: `tests/unit/test_dicom_missing_extra.py`

**Difficulty:** medium. **Gating tests:** C12 (missing extra → actionable error). **Blast radius:** base import must not require pydicom/nibabel — lazy-import inside functions only.

- [ ] **Step 1: Add the extra**

In `pyproject.toml` `[project.optional-dependencies]`:

```toml
# DICOM medical reads (spec §8). pydicom floor 2.4: stable apply_modality_lut /
# apply_voi_lut in pydicom.pixel_data_handlers. nibabel floor 5.2: stable
# affine API for the NIfTI volume writer (spec §7.3).
dicom = ["pydicom>=2.4", "nibabel>=5.2"]
```

Run: `uv lock` (do NOT install into base; the extra stays optional).

- [ ] **Step 2: Write the failing test (C12)**

```python
# tests/unit/test_dicom_missing_extra.py
import builtins

import pytest


def test_C12_missing_pydicom_raises_actionable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pydicom" or name.startswith("pydicom."):
            raise ImportError("No module named 'pydicom'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from custom_sam_peft.data.dicom_io import read_dcm_with_meta

    with pytest.raises(RuntimeError, match=r"pip install custom-sam-peft\[dicom\]"):
        read_dcm_with_meta("x.dcm", 1)
```

- [ ] **Step 3: Implement the guard**

```python
# src/custom_sam_peft/data/dicom_io.py
"""DICOM reads behind the optional [dicom] extra (spec §8). pydicom/nibabel are
lazy-imported so base install/import never requires them."""

from __future__ import annotations

_MISSING = "DICOM support requires the optional extra: pip install custom-sam-peft[dicom]"


def _require_pydicom():
    try:
        import pydicom  # noqa: F401
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError(_MISSING) from exc
    import pydicom

    return pydicom


def read_dcm_with_meta(path, channels):
    pydicom = _require_pydicom()
    ...  # filled in Task 3.2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_dicom_missing_extra.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/custom_sam_peft/data/dicom_io.py tests/unit/test_dicom_missing_extra.py
git commit -m "feat(dicom): [dicom] extra + lazy-import actionable error (spec §8,§10; C12)"
```

---

## Task 3.2: Per-slice decode — Modality LUT, signed/bits, MONOCHROME1, VOI (C10)

**Files:**

- Modify: `src/custom_sam_peft/data/dicom_io.py` (`read_dcm_with_meta`)
- Modify: `src/custom_sam_peft/config/schema.py` — `DataConfig.dicom_voi_window` (the only new user config)
- Test: `tests/unit/test_dicom_decode.py`

**Difficulty:** medium. **Gating tests:** C10 (Modality LUT applied — signed CT decodes negative HU; MONOCHROME1 inverted; VOI only when file carries a window; override wins). **Blast radius:** new `DataConfig` field — additive, defaults `None`; the override must thread to the read path (predict passes it down).

- [ ] **Step 1: Add the config field**

In `DataConfig` (schema.py):

```python
    dicom_voi_window: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Optional explicit DICOM VOI (center, width) override applied to ALL "
            "slices. None (default) uses each file's own WindowCenter/WindowWidth, "
            "or no VOI if absent. An explicit user choice — no default hyperparameter "
            "is shipped (spec §9)."
        ),
    )
```

- [ ] **Step 2: Write the failing test (C10)**

```python
# tests/unit/test_dicom_decode.py
import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from custom_sam_peft.data.dicom_io import read_dcm_with_meta


def _make_ct(tmp_path, stored, slope=1.0, intercept=-1024.0, photometric="MONOCHROME2",
             signed=0, window=None, name="ct.dcm"):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.Rows, ds.Columns = stored.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = signed
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    dtype = np.int16 if signed else np.uint16
    ds.PixelData = stored.astype(dtype).tobytes()
    p = tmp_path / name
    ds.save_as(p, enforce_file_format=True)
    return p


def test_C10_modality_lut_decodes_negative_hu(tmp_path):
    stored = np.full((4, 4), 24, np.int16)  # 24*1 + (-1024) = -1000 HU (air)
    p = _make_ct(tmp_path, stored, signed=1)
    pixels, meta = read_dcm_with_meta(p, 1)
    assert meta.kind == "dicom"
    assert meta.rescale == (1.0, -1024.0)
    assert pixels.min() < 0  # signed CT decodes negative HU


def test_C10_monochrome1_inverted(tmp_path):
    stored = np.array([[0, 100], [200, 300]], np.uint16)
    p1 = _make_ct(tmp_path, stored, slope=1.0, intercept=0.0, photometric="MONOCHROME1", name="m1.dcm")
    p2 = _make_ct(tmp_path, stored, slope=1.0, intercept=0.0, photometric="MONOCHROME2", name="m2.dcm")
    a1, _ = read_dcm_with_meta(p1, 1)
    a2, _ = read_dcm_with_meta(p2, 1)
    # MONOCHROME1 inverted relative to MONOCHROME2: argmin/argmax flip
    assert np.unravel_index(a1.argmax(), a1.shape[:2]) == np.unravel_index(a2.argmin(), a2.shape[:2])
```

- [ ] **Step 3: Implement the decode (spec §8.1 order)**

In `read_dcm_with_meta`: read via pydicom; `arr = apply_modality_lut(ds.pixel_array, ds)` (ALWAYS); honor signed/bits via `ds.pixel_array`'s dtype (pydicom handles `PixelRepresentation`); if `PhotometricInterpretation == "MONOCHROME1"` invert (`arr = arr.max() - arr`); VOI ONLY if `voi_window` override OR the file carries `WindowCenter/WindowWidth` (`apply_voi_lut`, override wins). Coerce to `(H,W,C)` via `_coerce_to_channels`. Build `SpatialMeta(kind="dicom", pixel_spacing=..., orientation=..., position=..., frame_of_reference_uid=..., rescale=(slope,intercept), voi_window=..., series_uid=..., sop_uid=...)`. Thread the override param from predict's resolved config.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_dicom_decode.py -o "addopts=" -q`
Expected: PASS (or SKIP cleanly if pydicom absent — `importorskip`).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/dicom_io.py src/custom_sam_peft/config/schema.py tests/unit/test_dicom_decode.py
git commit -m "feat(dicom): per-slice decode — Modality LUT, signed, MONOCHROME1, VOI + override (spec §8.1; C10)"
```

---

## Task 3.3: Series grouping + 3D affine + NIfTI volume writer (C11)

**Files:**

- Modify: `src/custom_sam_peft/data/dicom_io.py` (add `group_series`, `series_affine`)
- Modify: `src/custom_sam_peft/predict/writers.py` (add `write_nifti_volume`)
- Modify: `src/custom_sam_peft/predict/runner.py` — when source `SpatialMeta.kind == "dicom"`, group per series + emit one NIfTI per series
- Test: `tests/unit/test_dicom_series_nifti.py`

**Difficulty:** medium-hard (3D affine construction + multi-file grouping). **Gating tests:** C11 (group by `SeriesInstanceUID`, sort by `ImagePositionPatient`, stack to ONE NIfTI of correct dims, 3D affine re-reads correctly). **Blast radius:** predict output now potentially multiple volumes per run (spec §11.5); document one-input-dir → potentially-several-volumes.

**Design note:** build the 3D affine from `PixelSpacing` + `ImageOrientationPatient` + `ImagePositionPatient` using nibabel's verified DICOM→world convention (cite `nibabel.nifti1` / the nibabel "DICOM orientation" doc in a comment). Sort slices by projecting `ImagePositionPatient` onto the slice-normal (cross product of the two `ImageOrientationPatient` direction cosines). Missing geometry tags → clear error for a series; a single slice w/o geometry degrades to a 2D mask (spec §11.4).

- [ ] **Step 1: Write the failing test (C11)**

```python
# tests/unit/test_dicom_series_nifti.py
import numpy as np
import pytest

pytest.importorskip("pydicom")
pytest.importorskip("nibabel")

from custom_sam_peft.data.dicom_io import group_series, series_affine
from custom_sam_peft.predict.writers import write_nifti_volume


def test_C11_series_groups_sorts_stacks_with_affine(tmp_path):
    import nibabel as nib
    # build 3 single-series slices at z=0,2,4 (out of order on disk)
    from tests.unit.test_dicom_decode import _make_ct  # reuse the fixture builder

    series = "1.2.3"
    paths = []
    for z in (4.0, 0.0, 2.0):  # deliberately unsorted
        p = _make_ct(tmp_path, np.full((4, 4), 10, np.int16), signed=1, name=f"s{z}.dcm")
        import pydicom
        ds = pydicom.dcmread(p)
        ds.SeriesInstanceUID = series
        ds.ImagePositionPatient = [0, 0, z]
        ds.save_as(p, enforce_file_format=True)
        paths.append(p)

    groups = group_series(paths)
    assert len(groups) == 1
    ordered = groups[series]
    zs = [float(ds.ImagePositionPatient[2]) for ds in ordered]
    assert zs == [0.0, 2.0, 4.0]  # sorted by position

    affine = series_affine(ordered)
    masks = [np.ones((4, 4), np.uint8) for _ in ordered]
    out = tmp_path / "vol.nii.gz"
    write_nifti_volume(masks, affine, out)
    vol = nib.load(str(out))
    assert vol.shape == (4, 4, 3)
    assert np.allclose(vol.affine, affine)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_dicom_series_nifti.py -o "addopts=" -q`
Expected: FAIL — `group_series` / `series_affine` / `write_nifti_volume` not defined.

- [ ] **Step 3: Implement grouping, affine, NIfTI writer**

`group_series(paths)`: read each `.dcm`, bucket by `SeriesInstanceUID`, sort each bucket by projection of `ImagePositionPatient` onto the slice-normal. `series_affine(ordered_datasets)`: construct the 4×4 from row/col direction cosines × `PixelSpacing` + slice spacing + first slice `ImagePositionPatient` (nibabel DICOM-orientation convention; cite in comment). `write_nifti_volume(masks, affine, out_path)`: stack masks along axis 2 (geometric order), `nib.Nifti1Image(vol, affine)`, save `.nii.gz`. Missing geometry → raise the §11.4 error. Wire predict to group `.dcm` inputs per series and emit one NIfTI each; single 2D slice w/o geometry → 2D mask.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_dicom_series_nifti.py -o "addopts=" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/dicom_io.py src/custom_sam_peft/predict/writers.py src/custom_sam_peft/predict/runner.py tests/unit/test_dicom_series_nifti.py
git commit -m "feat(dicom): series grouping + 3D affine + NIfTI volume writer (spec §7.3,§8.2; C11)"
```

---

## Task 3.4: File the out-of-scope follow-up issues (spec §15) + PR

**Files:**

- None (GitHub issues + PR).

**Difficulty:** easy. **Gating tests:** full CPU suite + GPU suite green.

- [ ] **Step 1: Final full-suite verification**

Run: `uv run pytest tests/unit/ -o "addopts=" -q` (CPU) and `scripts/run_gpu_tests.sh` (GPU). Then `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`.
Expected: all green; lint + format clean.

- [ ] **Step 2: File §15 follow-up issues**

For each of spec §15.1–§15.6 (DICOM-SEG dropped; CRS reprojection; resample-to-GSD; geospatial prompting; true-3D model input #110; Gaussian-blend overlap merge), open a GitHub issue via `gh issue create --assignee @me --label <existing-or-new>` referencing #131. Record §15.1's drop decision as a comment on #131.

- [ ] **Step 3: Open the PR**

`gh pr create --assignee @me --label <…>` linking the spec + this plan. One PR for all three phases on `131-georeferencing-dicom`.

---

## Self-review — spec coverage map

| Spec section | Task(s) |
| --- | --- |
| §4 fragment-merge (union-find, IoMin, threshold, area-weighted score, re-threshold) | 1.2 |
| §5.1 sliding-window utility (iter_windows, run_windows, merge, overlap const) | 1.1, 1.2, 1.3 |
| §5.2 predict tile→run→merge→one mask + run.json | 1.4 |
| §5.3 train tile expansion (clip, negatives, `__len__`) | 1.5 |
| §5.4 eval per-tile accumulation (non-overlap) + viz restitch | 1.6 |
| §6.1/§6.2 SpatialMeta seam + read_image_with_meta | 2.1, 2.2 |
| §6.3 rasterio replaces tifffile + nodata zero-fill | 2.2 |
| §6.4 tiling × affine composition | 2.3 |
| §7.1 GeoTIFF writer | 2.4 |
| §7.3 NIfTI volume writer | 3.3 |
| §8.1 per-slice DICOM decode | 3.2 |
| §8.2 series grouping + 3D affine | 3.3 |
| §8.3 (note: tiling shared, no DICOM branch) | covered by 1.x shared utility (no extra task) |
| §9 config (dicom_voi_window) | 3.2 |
| §10 deps (rasterio base, [dicom] extra, floors) | 2.2, 3.1 |
| §11 error handling (1 channel-mismatch / 2 missing-extra / 3 non-geo / 4 missing-geometry / 5 mixed-series / 6 degenerate) | 2.2 (1,3), 3.1 (2), 3.3 (4,5), 1.1 (6) |
| §12.1 C1–C14 | C1→1.1, C2/C3/C4→1.2, C5→1.1/1.4, C6→1.5, C7→2.2/2.4, C8→2.3, C9→2.2/2.4, C10→3.2, C11→3.3, C12→3.1, C13→2.1/2.2, C14→2.2 |
| §12.2 G1, G2 | 1.7 |
| §13 tbd pins | resolved in the pinned-resolutions section; carried into code at 1.1 (overlap, eval-overlap), 1.2 (metric, threshold, score-agg), 2.2 (nodata), 2.2/3.1 (floors) |
| §14 risks | 14.1→1.2 (+G1), 14.2 (cost) documented in 1.4, 14.3 (rasterio wheel) verified 2.2, 14.4 (DICOM geometry) 3.3, 14.5 (limit×expansion) 1.5 |
| §15 follow-ups | 3.4 |

**No uncovered spec sections.** §8.3 needs no dedicated task (it asserts the tiling utility is shared, which Tasks 1.x already guarantee). §7.2 is "(reserved)" — intentionally empty.
