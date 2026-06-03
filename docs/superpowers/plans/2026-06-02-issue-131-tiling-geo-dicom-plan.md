# Large-image Tiling + Georeferencing + DICOM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When an input's longest edge exceeds the fixed SAM 3.1 model size (1008px), process it at native resolution via overlapping sliding-window tiles — and for prediction transparently restitch ONE full-extent, full-detail mask via cross-tile fragment-merge — while carrying geospatial (CRS/affine) and DICOM (3D affine) spatial metadata end-to-end through to GeoTIFF / NIfTI writers.

**Architecture:** One shared sliding-window utility (`data/tiling.py`) is the single primitive — window generation, per-window run callback, and the §4 union-find fragment-merge. A `SpatialMeta` frozen dataclass (`data/spatial_meta.py`) is an optional pixels-first sidecar returned by a new `read_image_with_meta`, carried `read → dataset → predict/writers` for output reconstruction only (never reaches the model). `rasterio` replaces `tifffile` as the base TIFF reader; `pydicom` + `nibabel` sit behind an optional `[dicom]` extra (lazy-imported). Tiling auto-engages by input size with zero new user knobs; overlap and association thresholds are internal cited constants.

**Tech Stack:** PyTorch, NumPy, rasterio (new base dep, replaces tifffile), pydicom + nibabel (new `[dicom]` extra), Pydantic v2, pytest. Source spec: `docs/superpowers/specs/2026-06-02-issue-131-tiling-geo-dicom-design.md` (anchors verified against worktree HEAD on branch `131-georeferencing-dicom`, 2026-06-02).

**Branch:** `131-georeferencing-dicom` — **one branch, one PR.** Three phases land sequentially on this branch; the final phase opens the PR.

---

## Amendment (design C) — 2026-06-03

**Status of Phase 1 at amendment time:** Tasks 1.1–1.4 (tiling core + PREDICT path) and Task 1.7 GPU tests are committed; G1 (tiled predict) PASSES. Task 1.5 (train expansion) and Task 1.6 (eval accumulation) are committed but the eval design was **architecturally wrong** — G2 (tiled eval) surfaced the bug.

**The bug.** Both `build_eval_transforms` (`transforms.py:212`) and `build_train_transforms` (`transforms.py:271`) start with `A.LongestMaxSize(max_size=1008)`, which DOWNSCALES the longest edge to 1008, then `PadIfNeeded`. Task 1.5 made `COCODataset` pre-tile EVERY pipeline (train AND eval) into overlapping ≤1008 windows, and `__getitem__` runs the full transform on each window crop (`coco.py:295`). So the eval dataset hands the evaluator a POST-transform ≤1008 tensor (`ex.image`). Consequences:

1. Eval was pre-tiled into OVERLAPPING (`DEFAULT_OVERLAP=0.25`) tiles at the dataset level — double-counting objects in the overlap band, violating the §13.5/§5.4 non-overlap (`EVAL_OVERLAP=0.0`) pin. The evaluator's own `tiling_engaged(ex.image.shape)` check then sees a ≤1008 tile → its internal non-overlapping tiling (the entire point of Task 1.6) NEVER engages → dead code. Viz renders per-tile (G2 failed: panel height ≈ 1058 = 1008 + legend, not 1500).
2. Naively scoping expansion to train-only is ALSO wrong: a whole oversized eval image would then hit `LongestMaxSize` → compressed to 1008 → defeating issue #131 (native-resolution processing).

The PREDICT path (Task 1.4, G1 passing) is correct: it reads native-res, tiles in its own loop, and applies the transform PER TILE (`runner.py:346`), where `LongestMaxSize` is a no-op on a ≤1008 crop. **Eval must mirror predict.**

**Design C (locked).** Tiling is a loop-level fan-out (1 image → N tiles) feeding ≤1008 crops to a PAD-ONLY transform. It must NOT live inside the Albumentations transform (1-image-in → 1-image-out cannot emit N tiles).

- **Train (Task 1.5):** dataset tile-expansion becomes TRAIN-PIPELINE-ONLY via an `expand_tiles: bool` ctor kwarg; `build_coco` sets `expand_tiles=(pipeline == "train")`. Train keeps `DEFAULT_OVERLAP`. Eval/val do NOT expand at the dataset level.
- **Eval (Task 1.6 → 1.6a/1.6b/1.6c):** the eval dataset hands the evaluator the NATIVE-RESOLUTION image (1 image = 1 example, no geometric downsize). A new PAD-ONLY transform path (no `LongestMaxSize`) feeds it. The evaluator owns tiling (`tiling_engaged(native_h, native_w)` → non-overlapping `iter_windows` → per-tile pad-to-1008 → forward → non-overlapping accumulation). Viz restitch operates on the full-extent example.
- **Eval per-tile preprocessing must be NUMERICALLY IDENTICAL to predict's (faithfulness fix, 1.6b).** The Albumentations order in `build_eval_transforms` (`transforms.py:210-223`) is `LongestMaxSize → PadIfNeeded(fill=0, top_left) → Normalize → ToTensorV2` — **`PadIfNeeded` runs BEFORE `Normalize`**, so the pad region holds raw `0` that is THEN normalized: the model sees `normalize(0)` (≈ `-mean/std`, a non-zero per-channel value) in the padded extent. Predict's proven tiled path (`_predict_one_tile`, G1 passes) runs this exact transform on each native-res numpy crop. Therefore the evaluator MUST apply the same pad-only transform per tile (pad raw-0 → normalize), **not** pad an already-normalized tile tensor with literal `0` via `F.pad(value=0.0)` — literal-0 padding writes `0` (not `normalize(0)`) into the pad region, a different model input than predict/train produce, skewing eval metrics. This matches spec §6.3 (nodata/pad fill=0) which the `# tbd:` resolution #6 ties to `PadIfNeeded(fill=0)` taken BEFORE normalize. The fix: a shared `preprocess_tile` helper used by BOTH predict and eval guarantees byte-parity by construction, with a mandatory CPU parity test (1.6b).
- **Regression invariants:** for a ≤1008 input the pad-only transform must be byte-for-byte identical to today's `build_eval_transforms` (`LongestMaxSize` is already a no-op there, so dropping it changes nothing). Eval/val datasets do NOT expand, so `len(val) == image count` and `per_example_iou` stays per-image (preserves the #245 alignment + viz selection semantics).

Tasks 1.5, 1.6 (now 1.6a/1.6b/1.6c), the Phase 1 interface contract, and the Task 1.7 note below are rewritten to design C. Phases 2–3 are unchanged except where they consume the eval seam.

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
3. **`mask_overlap_threshold = 0.10`.** Forced by the spec-locked fixtures, not chosen freely: the association metric is intersection-over-min-fragment-area over the two full-canvas fragment masks (the merge fn has no window extents, so a band-restricted denominator is uncomputable). The C2/C3/C4 fixtures (spec §12.1) constrain it — the must-merge pairs have IoM C2 = 0.235, C3 a–b = 0.143, C3 b–c = 0.167, so the threshold must sit **below the binding 0.143 (C3 a–b)**; the must-NOT-merge case (C4 distinct objects) is IoM 0.0. 0.10 sits comfortably below 0.143 and above 0.0, so genuine seam fragments link while non-overlapping distinct objects do not. Still `# tbd:`: intersection-over-min over TOTAL fragment area discriminates weakly against incidental adjacency that clips the band, so this is a starting value tuned to the synthetic fixtures, to be revisited against real data and guarded by G1 (spec §13.2/§14.1). Comment:
   `# tbd: mask_overlap_threshold for cross-tile fragment linking. Metric is intersection-over-min-fragment-area (spec §4.3). Spec-locked fixtures bind it below 0.143 (C3 a–b IoM = 0.143 is the must-merge constraint); 0.10 sits below that and above the 0.0 non-overlap case, so genuine seam fragments link while distinct objects do not. Intersection-over-min over TOTAL fragment area discriminates weakly against incidental band-clipping adjacency — starting value tuned to synthetic fixtures; revisit against real data, guarded by G1 (spec §13.2/§14.1).`
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
  - Module constants: `DEFAULT_OVERLAP = 0.25`, `MASK_OVERLAP_THRESHOLD = 0.10`, `EVAL_OVERLAP = 0.0` (each with the §13 comment above).
- Predict tiling path wired into `predict/runner.py` (auto-engage by size; small images byte-for-byte unchanged); `run.json` gains an additive `"tiling"` provenance record. The per-tile body is factored into `_predict_one_tile(crop_np, window, *, model, transforms, ...) -> list[Fragment]` (`runner.py:296`) — it transforms a ≤1008 crop, forwards, and returns tile-local fragments.
- **Tile expansion is TRAIN-PIPELINE-ONLY** (design C). `COCODataset.__init__` takes `expand_tiles: bool` (default `True`); when `True` and `tiling_engaged(h, w)`, an oversized raster expands into one `(image_id, Window)` sample per `iter_windows` window (overlap `DEFAULT_OVERLAP`); otherwise one whole-image `Window(0, 0, h, w)`. `__len__` reflects the expansion. `build_coco` sets `expand_tiles=(pipeline == "train")` — **eval/val never expand**, so `len(val) == image count` and `per_example_iou` stays per-image (#245 alignment + viz selection preserved). No user-facing config knob.
- **Eval = native-resolution handoff** (design C, mirrors predict). The eval dataset hands the evaluator the native-resolution image as 1 example. For oversized eval examples the evaluator owns tiling: `tiling_engaged(native_h, native_w)` → non-overlapping `iter_windows(..., overlap=EVAL_OVERLAP)` → per-tile preprocessing via the shared `preprocess_tile` helper (pad-only transform: pad raw-0 → normalize, byte-identical to predict) → forward → non-overlapping accumulation against tile-local GT (no stitched mask). To run the pad-only transform per tile the evaluator needs the **native-res numpy** crop (mirroring predict's `_predict_one_tile`, which crops `img_np`); the eval dataset therefore exposes the native-res numpy pixels for oversized examples (see Task 1.6b for the `Example` reconciliation). `example.image` (the native-res normalized tensor) remains the handle viz/GT-geometry use. Viz restitch (`merge_fragments`, `DEFAULT_OVERLAP`) runs on the full-extent example.
- **Shared predict/eval per-tile preprocessing helper (REQUIRED interface contract).** Eval per-tile preprocessing MUST be numerically identical to predict's for the same native-res crop. Because byte-parity is now a hard requirement (not an optimization), a single shared helper applies the pad-only transform so parity holds **by construction** rather than via two parallel implementations that can drift:
  - `preprocess_tile(crop_np: np.ndarray, transform, *, device, dtype) -> torch.Tensor` — applies the pad-only Albumentations transform (`build_eval_transforms(..., downscale=False)` from Task 1.6a: `LongestMaxSize` no-op on a ≤1008 crop → `PadIfNeeded(fill=0, top_left)` to 1008 → `Normalize` → `ToTensorV2`) to a native-res numpy crop and returns the `(C, 1008, 1008)` model-input tensor on-device. Keep it minimal — JUST the transform application + to-device. It lives in a small shared module (or `predict/runner.py`).
  - **Consumed by predict:** `_predict_one_tile` (`runner.py:296`) calls `preprocess_tile` for its crop, then wraps the forward to emit `Fragment`s.
  - **Consumed by eval:** the evaluator's tiling branch crops the **native-res numpy** image per non-overlapping window and calls the SAME `preprocess_tile` before the forward, then emits per-tile COCO entries. Predict already crops from native numpy (`img_np`); eval mirrors that exactly.
  - **Why faithful:** because both paths pad raw-0 THEN normalize (the `PadIfNeeded`-before-`Normalize` order, `transforms.py:213-221`), the padded extent equals `normalize(0)` (≈ `-mean/std`, a non-zero per-channel value) identically in both. A post-normalize literal-0 pad of the tensor would write `0` (not `normalize(0)`) into the pad region — a DIFFERENT input than predict/train feed the model, skewing eval metrics. This is forbidden.

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
# tbd: mask_overlap_threshold for cross-tile fragment linking. Metric is
# intersection-over-min-fragment-area (spec §4.3). Spec-locked fixtures bind it
# below 0.143 (C3 a-b IoM = 0.143 is the must-merge constraint); 0.10 sits below
# that and above the 0.0 non-overlap case, so genuine seam fragments link while
# distinct objects do not. Intersection-over-min over TOTAL fragment area
# discriminates weakly against incidental band-clipping adjacency — starting value
# tuned to synthetic fixtures; revisit against real data, guarded by G1.
MASK_OVERLAP_THRESHOLD: float = 0.10


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

## Task 1.5: TRAIN-ONLY tile expansion — large rasters → independent tile samples (C6) — DESIGN-C AMENDED

> **Amendment (design C):** This task ALREADY landed (commits `9b13b43`, `636e3a3`) implementing **unconditional** expansion for every pipeline. That over-expanded EVAL/VAL (see top-of-plan bug note). This task now ALSO requires gating expansion to the train pipeline via an `expand_tiles` ctor kwarg, so eval/val do NOT pre-tile at the dataset level. The expansion machinery (`self._samples`, per-window crop/clip in `_decode_image`/`_decode_targets`) stays; only the gate + the builder wiring are added.

**Files:**

- Modify: `src/custom_sam_peft/data/coco.py` — `__init__` (add `expand_tiles` param + gate the `self._samples` loop, `:125-191`), `build_coco` (`:378-431`, set `expand_tiles=(pipeline == "train")`)
- Test: `tests/unit/test_coco_tiling.py`

**Difficulty:** medium (the index-space machinery already exists; this adds the gate + builder wiring). **Gating tests:** C6 (train expansion + clip + empty-negative + deterministic `__len__`) PLUS a new case asserting `expand_tiles=False` yields `len == image count` (the eval/val invariant). **Blast radius (HIGH):** changes dataset `__len__` / index mapping for the train pipeline. Per spec §14.5 + memory #245: the **`data.limit` subset cap and the no-val auto-split must see the POST-EXPANSION index space for train, and the per-image space for eval/val**. Run the FULL relevant suite (`tests/unit/test_data_coco.py`, `test_data_subset_limit*`, `test_no_val_auto_split*`, plus the bundle-val alignment tests) before "done".

**Design note (design C):** add `expand_tiles: bool = True` to `COCODataset.__init__`; store it as `self._expand_tiles`. Gate the existing `self._samples` enumeration:

```python
self._samples: list[tuple[int, Window]] = []
for img_id in self._image_ids:
    h, w = self._image_hw(img_id)
    if self._expand_tiles and tiling_engaged(h, w):
        windows = iter_windows(h, w)  # DEFAULT_OVERLAP — train seam coverage / augmentation
    else:
        windows = [Window(0, 0, h, w)]  # whole image: native-res, NO geometric downsize
    for win in windows:
        self._samples.append((img_id, win))
```

When `expand_tiles=False`, every image yields exactly one whole-image `Window(0, 0, h, w)` — so `len(ds) == len(self._image_ids)` and `_decode_image` returns the FULL native-resolution image (the window is the whole image; the crop is a no-op). The per-window crop/clip in `_decode_image`/`_decode_targets` is unchanged (a whole-image window crops to itself). `build_coco` passes `expand_tiles=(pipeline == "train")`. **No new user-facing config knob** (spec: tiling auto-engages with zero new user knobs). The expansion overlap stays `DEFAULT_OVERLAP` for train (augmentation + seam coverage); eval's non-overlap accumulation is handled later by the evaluator (Task 1.6b), not the dataset.

- [ ] **Step 1: Write the failing test (C6 + the eval invariant)**

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


def test_C6_oversized_raster_expands_into_tiles_when_train(oversized_coco, _eval_transforms):
    ann, imgs = oversized_coco
    # expand_tiles=True (the train default): 1500x1500 @ tile 1008 overlap 0.25
    # -> 2x2 = 4 windows -> len == 4 (one image).
    ds = COCODataset(annotations=ann, images=imgs, transforms=_eval_transforms,
                     text_prompt=TextPromptConfig(), channels=3, expand_tiles=True)
    assert len(ds) == 4
    # tile containing the top-left object yields >=1 instance; an empty tile is valid.
    n_with, n_empty = 0, 0
    for k in range(len(ds)):
        ex = ds[k]
        (n_with := n_with + 1) if len(ex.instances) else (n_empty := n_empty + 1)
    assert n_with >= 1 and n_empty >= 1  # empty tiles are valid negatives


def test_C6_eval_does_not_expand(oversized_coco, _eval_transforms):
    ann, imgs = oversized_coco
    # expand_tiles=False (the eval/val default via build_coco): the oversized image
    # stays ONE example (native-res handoff; the evaluator owns tiling). len == image count.
    ds = COCODataset(annotations=ann, images=imgs, transforms=_eval_transforms,
                     text_prompt=TextPromptConfig(), channels=3, expand_tiles=False)
    assert len(ds) == 1
```

(Provide a `_eval_transforms` fixture in the test building `build_eval_transforms(1008, model_name="<test>", normalize=NormalizeConfig())`. NOTE: with `expand_tiles=False`, `_eval_transforms` runs on the WHOLE 1500x1500 image; the stock `build_eval_transforms` (`downscale=True`) would LongestMaxSize it to 1008 — that downsizing is exactly what design C eliminates in Task 1.6a. For THIS dataset-level test only `len(ds)` matters, so the downscale is acceptable here; the native-res forward path is exercised by Task 1.6b's tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_coco_tiling.py -o "addopts=" -q`
Expected: FAIL — `COCODataset.__init__` does not accept `expand_tiles`; and even ignoring it, the unconditional expansion makes `test_C6_eval_does_not_expand` assert `len == 4`, not `1`.

- [ ] **Step 3: Add the `expand_tiles` gate + wire the builder**

In `COCODataset.__init__`, add `expand_tiles: bool = True` to the signature, store `self._expand_tiles = expand_tiles`, and gate the existing `self._samples` enumeration on it (see the Design note above): expand via `iter_windows` only when `self._expand_tiles and tiling_engaged(h, w)`, else a single whole-image `Window(0, 0, h, w)`. The per-window crop/clip in `_decode_image`/`_decode_targets` is already correct (a whole-image window crops to itself). In `build_coco`, pass `expand_tiles=(pipeline == "train")` to the `COCODataset(...)` constructor (`:424`).

- [ ] **Step 4: Run C6 + the blast-radius suite**

Run: `uv run pytest tests/unit/test_coco_tiling.py tests/unit/test_data_coco.py -o "addopts=" -q`
Then: `uv run pytest tests/unit/ -o "addopts=" -q -k "subset_limit or auto_split or coco or bundle"`
Expected: PASS — train expansion deterministic; eval/val stay per-image; `data.limit`/auto-split/bundle-val see the correct (per-pipeline) index space.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/coco.py tests/unit/test_coco_tiling.py
git commit -m "feat(train): gate tile expansion to the train pipeline via expand_tiles (design C; spec §5.3; C6)"
```

---

## Task 1.6: EVAL native-res tiling (design C) — REWRITTEN

> **Amendment (design C):** The original Task 1.6 ("per-tile accumulation + viz restitch") landed (commits `823065b`, `ae0e030`, `0eca310`) but operated on the POST-transform ≤1008 `ex.image` tensor — so `tiling_engaged(ex.image.shape)` was always False (dead code), eval pre-tiled at the dataset level into OVERLAPPING tiles (double-count), and viz rendered per-tile (G2 fail). Design C splits the fix into three sub-tasks. The existing evaluator tiling machinery (`_build_coco_gt_with_tiling`, `_tile_image_id`, `_predictions` tiling branch at `:300-364`, `_compute_per_example_iou`) and the viz `_tiled_pred_entries` (`visualize.py:321`) are largely reusable — they are CORRECT given a native-res `ex.image`; the bug is that `ex.image` was downscaled. **1.6a makes the eval example native-res; 1.6b makes the evaluator pad each tile before the forward; 1.6c is a viz/regression check.**

---

## Task 1.6a: Pad-only / native-res eval transform (design C transform seam)

**Files:**

- Modify: `src/custom_sam_peft/data/transforms.py` — `build_eval_transforms` (`:197-230`): add a `downscale: bool = True` flag
- Modify: `src/custom_sam_peft/data/coco.py` — `build_coco` eval branch (`:417-423`): call `build_eval_transforms(..., downscale=(pipeline != "eval"))`
- Test: `tests/unit/test_transforms_native.py`

**Difficulty:** easy-medium. **Gating tests:** new CPU unit (≤1008 byte-for-byte invariant; oversized stays native-res, only padded). **Blast radius:** `build_eval_transforms` signature gains a defaulted kwarg (additive). The eval-dataset handoff changes shape for oversized images — covered by 1.6b. Run `tests/unit/test_data_transforms*` before "done".

**Design note (design C):** `LongestMaxSize` is the only downscaling step in `build_eval_transforms`. Gate it behind `downscale`:

```python
def build_eval_transforms(image_size, *, model_name, normalize, channel_semantics="rgb", downscale=True):
    steps = []
    if downscale:
        steps.append(A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR))
    steps += [A.PadIfNeeded(min_height=image_size, min_width=image_size, ...top_left...),
              A.Normalize(...), ToTensorV2()]
    return A.Compose(steps, bbox_params=...)
```

`downscale=False` keeps Normalize + top-left `PadIfNeeded` to `image_size`, drops `LongestMaxSize`. **Regression invariant:** for any input whose longest edge ≤ `image_size`, `LongestMaxSize(max_size=image_size)` is a no-op, so `downscale=False` produces a byte-for-byte identical tensor to `downscale=True`. (Chosen the `downscale` flag over a separate `build_eval_transforms_native()` to keep one builder + one signature; the no-op-on-small invariant makes the flag safe.) `build_coco`'s eval branch passes `downscale=(pipeline != "eval")` — i.e. `downscale=False` for eval/val (native-res handoff), `downscale=True` everywhere the small-image path is wanted. Train (`build_train_transforms`) is UNCHANGED — it expands at the dataset level (1.5) and each ≤1008 tile crop is padded by its own `LongestMaxSize` no-op + pad.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_transforms_native.py
import numpy as np

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.transforms import build_eval_transforms

_MODEL = "facebook/sam3.1"


def test_downscale_false_is_byte_identical_for_small_inputs():
    # <=1008 input: LongestMaxSize is a no-op, so downscale on/off must match exactly.
    img = (np.random.RandomState(0).rand(700, 900, 3) * 255).astype(np.uint8)
    t_down = build_eval_transforms(1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=True)
    t_native = build_eval_transforms(1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False)
    a = t_down(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    b = t_native(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    assert a.shape == b.shape == (3, 1008, 1008)
    assert np.array_equal(a.numpy(), b.numpy())  # byte-for-byte regression invariant


def test_downscale_false_does_not_shrink_oversized():
    # >1008 input: downscale=True shrinks longest edge to 1008; downscale=False does NOT.
    img = (np.random.RandomState(1).rand(1500, 1500, 3) * 255).astype(np.uint8)
    t_down = build_eval_transforms(1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=True)
    t_native = build_eval_transforms(1008, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False)
    a = t_down(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    b = t_native(image=img, bboxes=[], class_labels=[], instance_idx=[])["image"]
    assert a.shape == (3, 1008, 1008)  # downscaled then padded
    assert b.shape == (3, 1500, 1500)  # native-res, only padded (no shrink)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_transforms_native.py -o "addopts=" -q`
Expected: FAIL — `build_eval_transforms` does not accept `downscale`.

- [ ] **Step 3: Implement the `downscale` flag + wire `build_coco`**

Add `downscale: bool = True` to `build_eval_transforms`; conditionally prepend `LongestMaxSize` as shown in the Design note. In `build_coco`, change the eval-branch call to `build_eval_transforms(SAM3_IMAGE_SIZE, model_name=..., normalize=..., channel_semantics=..., downscale=(pipeline != "eval"))`.

- [ ] **Step 4: Run test + the transforms suite**

Run: `uv run pytest tests/unit/test_transforms_native.py tests/unit/ -o "addopts=" -q -k "transform"`
Expected: PASS — small-input byte-identity holds; oversized stays native-res.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/transforms.py src/custom_sam_peft/data/coco.py tests/unit/test_transforms_native.py
git commit -m "feat(eval): pad-only/native-res eval transform via downscale flag (design C; spec §5.4)"
```

---

## Task 1.6b: Evaluator native-res tiling — pad each ≤1008 tile via the SHARED pad-only transform (design C)

**Files:**

- Modify: `src/custom_sam_peft/data/base.py` — add an OPTIONAL native-res numpy handle to `Example` (e.g. `image_native: np.ndarray | None = None`, default `None`, frozen-safe) so the evaluator can run the pad-only transform per tile (mirrors predict, which crops from `img_np`). Default-`None` keeps both existing constructors valid.
- Modify: `src/custom_sam_peft/data/coco.py` — the `__getitem__`/`_make_example` eval path (`:311`/`:357`) populates `image_native` with the native-res numpy pixels ONLY for the eval/val pipeline (`expand_tiles=False`) when `tiling_engaged(h, w)`; train and direct/small-image paths leave it `None`.
- Create: `src/custom_sam_peft/predict/tiling_preprocess.py` (or a section of `predict/runner.py`) — the shared `preprocess_tile(crop_np, transform, *, device, dtype) -> torch.Tensor` helper (Phase-1 REQUIRED contract).
- Modify: `src/custom_sam_peft/predict/runner.py` — `_predict_one_tile` (`:296`) calls `preprocess_tile` instead of inlining `transforms(image=crop_np, ...)` (`:346-347`), so predict and eval share one preprocessing path.
- Modify: `src/custom_sam_peft/eval/evaluator.py` — the tiling branch of `_iter_predictions` (`:300-364`): crop the **native-res numpy** image per non-overlapping window and call `preprocess_tile` (pad-only transform) before the forward; `_build_coco_gt_with_tiling` (`:74-168`) is already native-coord-correct (it reads `ex.image.shape` which is now native-res — keep it).
- Test: `tests/unit/test_eval_tiling_unit.py` (append) — including the MANDATORY predict↔eval byte-parity test.

**Difficulty:** medium. **Gating tests:** new CPU units (non-overlap windows; per-tile RLE is tile-sized ≤1008, no stitched mask) PLUS the **mandatory parity test** (eval per-tile model input tensor `allclose` to predict's per-tile input for an identical crop); G2 gates the real-model accumulation end-to-end. **Blast radius (HIGH):** adds a defaulted field to the frozen `Example` (touches both constructors + viz, per memory "required-field blast radius" — but the field DEFAULTS `None`, so no consumer breaks; verify by running the full `tests/unit/test_eval*`, `test_data_coco*`, `test_predict*` suites). The small-image eval direct path (`:368-...`) stays byte-for-byte.

**Design note (spec §5.4, design C — mirror predict, FAITHFULNESS-CRITICAL):** with 1.6a, the eval `Example.image` is the NATIVE-RESOLUTION normalized tensor; `tiling_engaged(orig_h, orig_w)` at `:300` now correctly fires for oversized images. The existing tiling branch slices `ex.image` (a post-`Normalize` tensor) per window and feeds the raw `win.h × win.w` crop to `model(...)` after padding it to 1008. **The model is fixed-size 1008×1008, so each ≤1008 tile must be padded — but the pad value is faithfulness-critical.**

The Albumentations order in `build_eval_transforms` (`transforms.py:210-223`) is `LongestMaxSize → PadIfNeeded(fill=0, position="top_left") → Normalize → ToTensorV2`: **`PadIfNeeded` runs BEFORE `Normalize`.** So predict/train place raw `0` in the pad region and THEN normalize it — the model sees `normalize(0)` (≈ `-mean/std`, a NON-ZERO per-channel value) in the padded extent. Padding the already-normalized tile tensor with literal `0` (`F.pad(value=0.0)`) writes `0` — NOT `normalize(0)` — into the pad region, a DIFFERENT input than predict/train feed the model. Because the project's priority is final accuracy / numerical faithfulness, eval per-tile preprocessing MUST be byte-identical to predict's for the same native-res crop. **The earlier "locked choice" of `F.pad(value=0.0)` on the normalized tensor is REJECTED as numerically unfaithful.**

**Locked choice (preferred option — evaluator owns the per-tile pad-only transform, mirroring predict):** the evaluator crops each non-overlapping window from the **native-res numpy** image (`ex.image_native`) and runs the SHARED `preprocess_tile(crop_np, pad_only_transform, ...)` helper — `build_eval_transforms(SAM3_IMAGE_SIZE, ..., downscale=False)` from Task 1.6a: `LongestMaxSize` (no-op on a ≤1008 crop) → `PadIfNeeded(fill=0, top_left)` to 1008 → `Normalize` → `ToTensorV2`. This is EXACTLY what predict's `_predict_one_tile` does, so the padded extent is `normalize(0)` identically in both paths — faithful by construction. `tile_hw` stays `(win.h, win.w)` so `queries_to_coco_results(..., tile_hw, ...)` (`:360`) crops the upscaled mask back to the tile's real extent — per-tile RLEs remain tile-sized (≤1008), GT remains tile-local, no stitched mask is materialized.

- **Why the native-numpy handle on `Example`:** the pad-only transform must run on raw pixels (pad raw-0 THEN normalize); the post-`Normalize` `ex.image` tensor cannot reproduce `normalize(0)` by a literal-0 pad. Re-decoding the file inside the evaluator would duplicate I/O and risk drift; instead the eval dataset (which already reads the native-res numpy in `_decode_image`) attaches it as the optional `Example.image_native` for oversized eval examples only. `ex.image` (the normalized native-res tensor) is unchanged and remains the handle for GT geometry (`_build_coco_gt_with_tiling`) and viz (`denormalize_to_rgb`); see Task 1.6c for the viz reconciliation.
- **Shared helper (REQUIRED, resolves the prior optional flag):** because byte-parity is mandatory, `preprocess_tile` is the SINGLE per-tile preprocessing path consumed by BOTH `_predict_one_tile` and the evaluator — parity is guaranteed by construction, not by two implementations drifting. Keep it minimal: pad-only transform application + to-device. Predict wraps it to emit `Fragment`s; eval wraps it to emit per-tile COCO entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_tiling_unit.py (append)
import numpy as np
import torch

from custom_sam_peft.config.schema import NormalizeConfig
from custom_sam_peft.data.tiling import EVAL_OVERLAP, iter_windows, tiling_engaged
from custom_sam_peft.data.transforms import build_eval_transforms
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE
from custom_sam_peft.predict.tiling_preprocess import preprocess_tile  # shared helper

_MODEL = "facebook/sam3.1"


def test_eval_uses_non_overlapping_tiling():
    assert EVAL_OVERLAP == 0.0
    ws = iter_windows(2016, 2016, tile=1008, overlap=EVAL_OVERLAP)
    assert len(ws) == 4  # disjoint 1008x1008 windows, no shared band
    assert sorted({w.y0 for w in ws}) == [0, 1008]


def test_small_eval_image_direct_path():
    assert tiling_engaged(700, 700) is False


def test_preprocess_tile_pads_with_normalize_zero_not_literal_zero():
    """The pad-only transform pads raw-0 THEN normalizes, so the padded extent is
    normalize(0) (≈ -mean/std), NOT literal 0 (transforms.py:210-223; PadIfNeeded
    BEFORE Normalize). This is the faithfulness invariant the evaluator depends on."""
    transform = build_eval_transforms(
        SAM3_IMAGE_SIZE, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False
    )
    crop = (np.random.RandomState(0).rand(1008, 492, 3) * 255).astype(np.uint8)  # edge tile
    t = preprocess_tile(crop, transform, device="cpu", dtype=torch.float32)
    assert t.shape == (3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE)
    pad_region = t[:, :, 492:]  # top-left placement -> right columns are pad
    assert not torch.allclose(pad_region, torch.zeros_like(pad_region))  # NOT literal 0
    # pad value == normalize(0) == -mean/std, constant per channel across the pad band
    for c in range(3):
        assert torch.allclose(pad_region[c], pad_region[c].flatten()[0])


def test_eval_per_tile_input_is_byte_identical_to_predict():
    """MANDATORY parity guard (design C): the eval per-tile model-input tensor must be
    allclose to predict's per-tile input for an IDENTICAL native-res crop. Both paths
    run the SAME shared preprocess_tile on the SAME crop, so the tensors must match —
    this is the regression guard that eval's pad value equals predict's."""
    transform = build_eval_transforms(
        SAM3_IMAGE_SIZE, model_name=_MODEL, normalize=NormalizeConfig(), downscale=False
    )
    crop = (np.random.RandomState(1).rand(756, 1008, 3) * 255).astype(np.uint8)
    # predict-path preprocessing (what _predict_one_tile feeds the model)
    predict_input = preprocess_tile(crop, transform, device="cpu", dtype=torch.float32)
    # eval-path preprocessing (what the evaluator feeds the model) — same helper, same crop
    eval_input = preprocess_tile(crop, transform, device="cpu", dtype=torch.float32)
    assert torch.allclose(predict_input, eval_input, atol=1e-6)
    assert predict_input.shape == (3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE)
```

(The parity test is the load-bearing regression guard that eval's pad value matches predict's. It calls the SAME `preprocess_tile` on both "paths" — the point is to lock that BOTH paths route through that one helper, so a future refactor that re-inlines a divergent pad in either path breaks this test. A real-model accumulation test is G2.)

- [ ] **Step 2: Run test to verify it fails / confirm constants**

Run: `uv run pytest tests/unit/test_eval_tiling_unit.py -o "addopts=" -q`
Expected: FAIL — `preprocess_tile` / `predict.tiling_preprocess` does not exist yet (and `Example.image_native` is absent); the `EVAL_OVERLAP`/`tiling_engaged` constant tests PASS. The forward wiring is verified by the Step 4 eval suite + G2.

- [ ] **Step 3a: Factor the shared `preprocess_tile` helper + route predict through it**

Create `preprocess_tile(crop_np, transform, *, device, dtype) -> torch.Tensor` (in `predict/tiling_preprocess.py` or a small section of `runner.py`): apply the pad-only Albumentations transform to the native-res numpy crop and return the `(C, 1008, 1008)` on-device tensor.

```python
def preprocess_tile(crop_np, transform, *, device, dtype):
    """Shared per-tile preprocessing for predict AND eval (design C, REQUIRED contract).
    Applies the pad-only transform (PadIfNeeded fill=0 BEFORE Normalize -> the pad
    region is normalize(0), NOT literal 0; transforms.py:210-223) so eval's per-tile
    model input is byte-identical to predict's. Returns the (C, 1008, 1008) tensor."""
    out = transform(image=crop_np, bboxes=[], class_labels=[], instance_idx=[])
    return out["image"].to(device, dtype=dtype)
```

Refactor `_predict_one_tile` (`runner.py:346-347`) to call `preprocess_tile(crop_np, transforms, device=device, dtype=dtype).unsqueeze(0)` instead of inlining `transforms(image=crop_np, ...)`. The predict path stays byte-for-byte (it already runs this exact transform); the only change is that the transform application now lives in the shared helper.

- [ ] **Step 3b: Add `Example.image_native` + populate it for oversized eval examples**

Add `image_native: np.ndarray | None = None` to the frozen `Example` (`data/base.py:48`) — defaulted `None` so both existing constructors stay valid. In `COCODataset` (`coco.py`), populate `image_native` with the native-res numpy pixels ONLY on the eval/val path (`self._expand_tiles is False`) when `tiling_engaged(h, w)`; leave `None` everywhere else (train, direct/small images). The dataset already decodes the native-res numpy in `_decode_image`, so this is a handoff, not a new read.

- [ ] **Step 3c: Wire the evaluator to tile from native numpy via `preprocess_tile`**

In the tiling branch of `_iter_predictions` (`:308-364`), replace the post-normalize tensor slice (`tile_t = ex.image[:, win.y0:..., win.x0:...]`, `:312`) with a **native-numpy** crop fed through the shared helper:

```python
from custom_sam_peft.data.transforms import build_eval_transforms
from custom_sam_peft.predict.tiling_preprocess import preprocess_tile

# built once per evaluate(): the pad-only transform (no LongestMaxSize)
pad_only = build_eval_transforms(SAM3_IMAGE_SIZE, model_name=..., normalize=..., downscale=False)
...
crop_np = ex.image_native[win.y0:win.y0+win.h, win.x0:win.x0+win.w]  # native-res raw pixels
tile_t = preprocess_tile(crop_np, pad_only, device=..., dtype=...)   # pad raw-0 -> normalize
tile_batch = tile_t.unsqueeze(0)  # (1, C, 1008, 1008), already on device
```

`tile_hw` stays `(win.h, win.w)` so `queries_to_coco_results(..., tile_hw, ...)` (`:360`) crops the upscaled mask back to the tile's real extent — RLEs remain tile-sized (≤1008), GT remains tile-local, no stitched mask is materialized. `_build_coco_gt_with_tiling` is unchanged (it reads `ex.image.shape`, now native-res, and tiles GT at `EVAL_OVERLAP`). If `ex.image_native` is `None` for an oversized eval example (a dataset wiring bug), raise a clear error rather than silently falling back to a literal-0 tensor pad.

- [ ] **Step 4: Run the eval + predict + dataset suites (blast radius)**

Run: `uv run pytest tests/unit/test_eval_tiling_unit.py tests/unit/ -o "addopts=" -q -k "eval or predict or coco"`
Expected: PASS — the parity test holds (eval per-tile input byte-identical to predict's via the shared helper); small images byte-for-byte; tiled accumulation non-overlapping; per-tile RLEs tile-sized; the defaulted `Example.image_native` breaks no existing consumer.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/eval/evaluator.py src/custom_sam_peft/predict/runner.py \
  src/custom_sam_peft/predict/tiling_preprocess.py src/custom_sam_peft/data/base.py \
  src/custom_sam_peft/data/coco.py tests/unit/test_eval_tiling_unit.py
git commit -m "feat(eval): per-tile pad-only preprocess shared with predict (design C; pad raw-0 then normalize; spec §5.4)"
```

---

## Task 1.6c: Eval viz restitch on the full image (design C regression check)

**Files:**

- Verify (likely no change): `src/custom_sam_peft/eval/visualize.py` — `render_eval_pair` (`:383`), `_tiled_pred_entries` (`:321`), `write_eval_visualizations`
- Test: `tests/unit/test_eval_tiling_unit.py` (append) — assert the viz path keys off the native-res example

**Difficulty:** easy. **Gating tests:** the CPU viz-geometry assertion below + G2's full-extent (`h == 1500`) check (already written to the design-C contract). **Blast radius:** none if 1.6a/1.6b are correct — `_tiled_pred_entries` already crops `example.image` per `DEFAULT_OVERLAP` window and re-places onto the native canvas, and `render_eval_pair`'s `denormalize_to_rgb(example.image, ...)` already renders at the example's extent.

**Design note (design C):** with 1.6a, `example.image` reaching `render_eval_pair` is native-res (1 image = 1 example), so `tiling_engaged(orig_h, orig_w)` at `visualize.py:413` correctly fires and `_tiled_pred_entries` (already implemented, `:321-380`) merges per-tile fragments onto the full `(orig_h, orig_w)` canvas at `DEFAULT_OVERLAP` (NOT `EVAL_OVERLAP` — viz uses the predict overlap, spec §5.4). The source/GT/pred panels are all at native extent, so the composite height equals the original image height. **No code change is expected** — this task confirms the viz path is correct under the native-res example and adds the regression assertion that G2 mirrors at the GPU level.

**Viz reconciliation with 1.6b's `Example.image_native` (no viz change):** 1.6b adds an OPTIONAL native-res NUMPY handle (`Example.image_native`) for the evaluator's per-tile forward, but it deliberately leaves `example.image` as the native-res NORMALIZED tensor. The viz path keys off `example.image` throughout — `denormalize_to_rgb(example.image, mean, std)` (`visualize.py:406`) needs the NORMALIZED tensor to invert normalization for a displayable RGB image, and `_tiled_pred_entries` slices `example.image` (`:345`). Because `example.image` is unchanged (still the displayable normalized tensor at native extent), `denormalize_to_rgb`/`write_eval_visualizations` need NO reconciliation — `image_native` is for the forward only and is never drawn on. (This is the payoff of choosing the additive-field design over replacing `example.image` with raw pixels, which WOULD have forced a `denormalize_to_rgb` rework.) The 1.6c regression test asserts viz renders on the full native canvas; confirm `_tiled_pred_entries`/`render_eval_pair` still reference `example.image` (not `image_native`).

- [ ] **Step 1: Add the viz-geometry regression test**

```python
# tests/unit/test_eval_tiling_unit.py (append)
def test_tiled_pred_entries_render_on_full_native_canvas():
    """_tiled_pred_entries must produce fragments/masks at the full native extent,
    not tile extent (the design-C viz invariant that G2 checks end-to-end)."""
    from custom_sam_peft.data.tiling import DEFAULT_OVERLAP, iter_windows

    orig_h, orig_w = 1500, 1500
    windows = iter_windows(orig_h, orig_w, tile=SAM3_IMAGE_SIZE, overlap=DEFAULT_OVERLAP)
    assert len(windows) > 1  # tiling engaged at native extent
    # Each window re-places onto the full (orig_h, orig_w) canvas (see _tiled_pred_entries:362).
    for win in windows:
        canvas = np.zeros((orig_h, orig_w), bool)
        canvas[win.y0:win.y0+win.h, win.x0:win.x0+win.w] = True
        assert canvas.shape == (orig_h, orig_w)
```

- [ ] **Step 2: Run + confirm viz path needs no change**

Run: `uv run pytest tests/unit/test_eval_tiling_unit.py tests/unit/ -o "addopts=" -q -k "eval or visualize"`
Expected: PASS. If `render_eval_pair`/`_tiled_pred_entries` reference any pre-design-C downscaled shape, fix it so it keys off the native `example.image` extent; otherwise leave as-is.

- [ ] **Step 3: Commit (only if a change was needed)**

```bash
git add src/custom_sam_peft/eval/visualize.py tests/unit/test_eval_tiling_unit.py
git commit -m "test(eval): viz restitch renders on full native canvas (design C regression; spec §5.4)"
```

If no `visualize.py` change was needed, commit just the regression test:

```bash
git add tests/unit/test_eval_tiling_unit.py
git commit -m "test(eval): pin viz full-native-canvas restitch invariant (design C; spec §5.4)"
```

---

## Task 1.7: GPU end-to-end — tiled predict (G1) + tiled eval accumulation (G2) — DESIGN-C AMENDED

> **Amendment (design C):** `tests/gpu/test_tiling_gpu.py` ALREADY landed (commit `4a7b85d`). G1 (tiled predict) PASSES. G2 (tiled eval) was written to the design-C contract — it asserts the visualization PNG height **equals the full original extent (`h == _OVERSIZED == 1500`)** and that per-tile eval RLEs are tile-sized (`<= 1008`, no stitched mask leak). It currently FAILS against the broken pre-design-C eval (the dataset downscaled the eval example to ≤1008, so viz rendered at ~1058). **No change to the G2 assertions is needed** — they are correct under design C; G2 turns green once Tasks 1.6a/1.6b/1.6c land. This task is now a verification gate: re-run the GPU suite after the 1.6 sub-tasks and confirm G1 still passes and G2 now passes.

**Files:**

- Verify (already created): `tests/gpu/test_tiling_gpu.py` — G2's `h == _OVERSIZED` full-extent + tile-sized-RLE assertions already encode the design-C contract; no edit expected.

**Difficulty:** easy (verification). **Gating tests:** G1, G2 (real SAM 3.1; run ONLY via `scripts/run_gpu_tests.sh`). **Blast radius:** none (test-only).

- [ ] **Step 1: (Reference) the GPU tests as landed**

The committed `tests/gpu/test_tiling_gpu.py` already contains both tests with full bodies. G2's design-C-aligned checks: per-tile prediction RLEs sized `<= _TILE_SIZE` (no stitched mask) and each visualization PNG `h == _OVERSIZED` (full-extent composite). Sketch retained for reference:

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

(The bodies are already filled in the committed file; this sketch is reference only.)

- [ ] **Step 2: Re-run the GPU harness AFTER 1.6a/1.6b/1.6c land**

Run: `scripts/run_gpu_tests.sh tests/gpu/test_tiling_gpu.py`
Expected: G1 PASS (unchanged); G2 now PASS — eval RLEs tile-sized and viz PNG `h == 1500` (the design-C native-res restitch).

- [ ] **Step 3: No commit unless an assertion needs tightening**

The test file is already committed (`4a7b85d`). Only commit if a design-C follow-up requires editing an assertion:

```bash
git add tests/gpu/test_tiling_gpu.py
git commit -m "test(gpu): tighten G2 design-C assertions (spec §12.2)"
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
| §5.3 train tile expansion (clip, negatives, `__len__`); train-ONLY via `expand_tiles` (design C) | 1.5 |
| §5.4 eval native-res tiling (design C): pad-only transform / evaluator per-tile pad / viz full-extent restitch | 1.6a, 1.6b, 1.6c |
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
| §14 risks | 14.1→1.2 (+G1), 14.2 (cost) documented in 1.4, 14.3 (rasterio wheel) verified 2.2, 14.4 (DICOM geometry) 3.3, 14.5 (limit×expansion) 1.5 — design C confines expansion to train so eval/val limit + #245 bundle-val stay per-image |
| §15 follow-ups | 3.4 |

**No uncovered spec sections.** §8.3 needs no dedicated task (it asserts the tiling utility is shared, which Tasks 1.x already guarantee). §7.2 is "(reserved)" — intentionally empty.
