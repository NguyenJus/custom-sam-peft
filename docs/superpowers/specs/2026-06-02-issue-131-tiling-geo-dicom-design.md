# Large-image tiling + georeferencing (geospatial) + DICOM metadata — Design Specification

**Status:** Locked design (do not redesign; this is the source of truth for the planner and implementers).
**Issue:** [#131](https://github.com/NguyenJus/custom-sam-peft/issues/131)
**Worktree branch:** `131-georeferencing-dicom`
**Scope:** **One spec, one PR.** Three coherent feature blocks, suggested phase order
tiling-core → geo → dicom, but a single deliverable. Follow-up / non-goal from
N-channel input #111 (spec §14.2): #111 reads multi-band TIFF *pixel bands only*
(via `tifffile`), with no georeferencing/CRS and no DICOM metadata. This spec adds
(1) format-agnostic sliding-window **tiling** so inputs larger than the fixed 1008px
model input are no longer destructively downsized, (2) **georeferencing** (CRS +
affine) by promoting `rasterio` to the base TIFF reader, and (3) **DICOM** medical
reads behind an optional `[dicom]` extra.

**Anchors verified against:** worktree HEAD on branch `131-georeferencing-dicom`
(2026-06-02). Line numbers re-confirmed inline below against the live tree; note the
#111 channel work has **already landed** (`data/io.py` exists and routes `.tif` through
`tifffile`; `data.channels` / `data.channel_semantics` are threaded end-to-end including
through predict's `_resolve_config`). This spec builds **on top of** that landed state.

---

## 1. Overview / Goal

### 1.1 The accuracy problem this fixes

SAM 3.1 has a **fixed** input size: `SAM3_IMAGE_SIZE = 1008` (`models/sam3.py:111`),
patch `_SAM3_PATCH = 14` (`presets.py:151`). Every image is forced to that size today
by a longest-edge resize + pad in both the eval and train pipelines:

- `build_eval_transforms` (`data/transforms.py:197`): `A.LongestMaxSize(max_size=image_size)`
  (`transforms.py:212`) → `A.PadIfNeeded(min_height/width=image_size, position="top_left")`
  (`transforms.py:213`).
- `build_train_transforms` (`data/transforms.py:233`): same `LongestMaxSize` + `PadIfNeeded`
  pair (`transforms.py:271-279`).
- Predict reads the full image with `read_image(img_path, channels)` (`predict/runner.py:424`),
  runs the same eval transform (`predict/runner.py:333`), predicts at 1008², and the
  writers upsample each mask back to the original dims via nearest-neighbour PIL resize
  (`predict/writers.py:111-117`).

For a large raster (e.g. a 12000×12000 satellite scene, or a high-res histopathology WSI)
this collapses the whole scene into 1008², destroying the per-object resolution the model
needs. Small objects vanish below a single patch. **This is an accuracy regression that no
hyperparameter can recover — the information is gone before the model sees it.** There is
**no tiling anywhere in the codebase today.**

**Goal:** when an input's longest edge exceeds the model size, process it at **native
resolution** via overlapping sliding-window tiles, and — for prediction — transparently
restitch one full-extent, full-detail mask. Add geospatial (CRS/affine) and medical (DICOM)
metadata carriers so geo/medical workflows round-trip their spatial reference end-to-end.

### 1.2 Governing principles (top priority — from the user; these override everything below on conflict)

1. **Final model accuracy AND user-facing simplicity outrank all else** (>> training speed).
   A speed-only argument is never sufficient to add a user-facing knob.
2. **`predict` returns ONE same-extent, full-detail mask — never tiles, never a "stitch it
   yourself" burden.** The tool absorbs the tiling/stitching complexity internally. The user
   submits an image and receives an output covering the same extent at the same resolution.
3. **Every NEW default hyperparameter needs a rigorous citation OR an explicit `# tbd:`
   tag** — never a silent guess. (This repo has a CI hook enforcing exactly this; see
   `docs/superpowers/specs/2026-06-01-ci-no-uncited-default-hook-design.md`.)
4. **Keep the base dependency tree as light as practical.** `rasterio` is accepted into base
   (ships bundled manylinux GDAL wheels — no system GDAL); DICOM deps are optional.

### 1.3 The DETR set-prediction constraint (why stitching is non-trivial)

SAM 3.1 here is a **DETR-style set-prediction** model: object queries + Hungarian matching
at train time (`models/losses/compose.py:133` constructs `HungarianMatcher`; matched at
`compose.py:149`), **no NMS at inference**. Inference output is per-(image, category)
instance masks via `queries_to_coco_results` (`predict/runner.py:481`), score-thresholded
(`predict/runner.py:488`, `>= opts.score_threshold`) and `top_k`-capped per class
(`predict/runner.py:490`). Each tile produces its **own independent query-set**. An object
straddling a tile seam appears as two (or more) separate fragment instances, one per tile,
with **no shared identity**. Naive canvas placement would therefore emit one real object as
several partial masks. The seam-correct stitch (§4) is the crux of this feature.

---

## 2. Non-goals (state explicitly)

Each is a deliberate exclusion, not an oversight. File a follow-up issue (§12) only where noted.

1. **DICOM-SEG output.** Clinical/PACS-oriented, heavy, wrong audience — this tool targets
   researchers/practitioners, not production PACS. Segmentation output for a DICOM series is a
   **NIfTI volume** (§7.3), which opens aligned in 3D Slicer / ITK-SNAP. DICOM-SEG is dropped.
2. **CRS reprojection / coordinate transforms.** We carry the source CRS + affine through
   unchanged; we never reproject to a different CRS. (Reprojection would change pixels and is
   out of scope.)
3. **Resample-to-target-GSD (ground sample distance).** Ineffective for a fixed-input model
   without fixed-physical-extent tiling, and would introduce an uncited resolution hyperparameter.
   We tile at native pixel resolution, not at a target physical resolution.
4. **Coordinate-based / geospatial prompting** (e.g. "segment within this bounding polygon in
   CRS coordinates"). Prompts remain text/class prompts.
5. **Any user-exposed tiles or user-side stitching.** The user never sees tiles and never
   stitches anything. (Principle 1.2.2.)
6. **Volumetric/temporal *model* input.** DICOM is read as a 3D series for **geometry and
   output stacking** only; the model still consumes 2D slices. True 3D model input is #110,
   disjoint from this work.

---

## 3. Architecture / Design overview

Three layers, sharing one sliding-window utility:

```
                 ┌─────────────────────────────────────────────────────────┐
   read path     │ read_image(path, channels) -> (pixels, SpatialMeta|None) │  §6
  (io.py)        │   rasterio (.tif/.tiff: bands + CRS + affine + nodata)   │
                 │   pydicom  (.dcm: pixels + spatial tags)  [dicom extra]  │
                 │   PIL / np.load (unchanged; SpatialMeta = None)          │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ pixels (H,W,C) + optional SpatialMeta
        ┌────────────────────────┼────────────────────────────┐
        ▼                        ▼                             ▼
  TRAIN (tiles =           PREDICT (tile → run →          EVAL (tile pred + GT;
  independent samples;     fragment-merge stitch →        Hungarian-match per tile;
  window-gen only)  §5.3   ONE full-extent mask)  §5.2    accumulate; no stitch) §5.4
                                 │
                                 ▼
                          WRITERS: GeoTIFF (CRS+affine) / NIfTI (affine) /
                          PNG / COCO-RLE (existing)  §7
```

The **sliding-window utility** (§5.1) is the single shared primitive: window generation,
a per-window run callback, and fragment-merge. Predict and eval-visualization use the full
utility; train uses only window generation.

---

## 4. The seam-correct stitch (the crux) — cross-tile fragment merging

This is the load-bearing algorithm. It is invoked by **predict (always)** and by
**eval-visualization (on request)**; eval metric accumulation does NOT stitch (§5.4).

**It is NOT coverage-OR** (which would smear distinct objects together), **NOT NMS-suppression**
(this model has no NMS and suppressing fragments would delete object parts). It is **cross-tile
fragment MERGING**:

1. **Per category, independently.** Fragments only ever merge within the same `category_id`.
   (Different categories never merge.)
2. **Place fragments on the full-image canvas.** Each tile's per-(image, category) query
   instances are offset by the tile's window origin `(y0, x0)` onto a full-size canvas. Each
   fragment carries: its mask (placed at full-canvas coordinates), its `score`, and its source
   window id.
3. **Build the overlap graph.** Two fragments from **different tiles** are linked when their
   masks overlap in the shared **overlap band** by more than a threshold. The association metric
   and threshold are NEW hyperparameters:
   - association metric: intersection over the *smaller fragment's* area within the overlap band
     (robust to the two fragments being very different total sizes — a small seam sliver of one
     object should still link). `# tbd:` the planner confirms IoU-of-overlap-band vs
     intersection-over-min and pins it with justification.
   - `mask_overlap_threshold` (fraction in `[0,1]`): link iff the metric exceeds it.
     **`# tbd:` cite or tag** — no published canonical value exists for DETR-fragment association;
     planner pins with justification (a reasonable starting point is a moderate fraction such that
     genuine same-object seam overlap links while incidental adjacency does not).
4. **Connected components (union-find) over the overlap graph.** An object spanning 3+ tiles
   transitively merges into one component: if fragment A links B and B links C, then {A,B,C} is
   one object even if A and C never directly touch. This is the reason for a union-find / connected
   components pass rather than pairwise merging.
5. **Union each component into one instance.** The merged mask is the **logical OR** of the
   component's fragment masks on the full canvas (within a component the fragments ARE the same
   object, so OR is correct here — this is union *within* a confirmed object, not coverage-OR
   across distinct objects). The merged score is aggregated by a NEW rule:
   - **score aggregation: area-weighted mean of fragment scores** (a fragment covering more of
     the object contributes more to the confidence than a thin seam sliver). The spec picks
     area-weighted mean over plain `max` because `max` lets a tiny high-scoring sliver dominate;
     area-weighting reflects that the object's confidence should track its dominant evidence.
     **`# tbd:`** planner confirms area-weighted-mean vs `max` and pins it with justification.
6. **Emit merged instances on the full-size canvas.** After component-merge, re-apply the
   per-category `score_threshold` and `top_k` cap (§1.3) on the **merged** instances, so the
   final per-(image, category) instance list matches the single-image contract.

**Overlap is what makes this work.** Tiles must overlap (§5.1) so that an object crossing a
seam has mask area in *both* tiles' shared band; without overlap there is no band to associate
in. The overlap fraction is a NEW hyperparameter (§5.1).

---

## 5. Sliding-window tiling

### 5.1 Shared sliding-window utility

Introduce a single reusable utility (recommended new module, e.g.
`src/custom_sam_peft/data/tiling.py` — planner's final call on location/name) exposing:

- **Window generation** — `iter_windows(h, w, tile, overlap) -> Iterable[Window]`, where each
  `Window` carries its origin `(y0, x0)` and size (≤ `tile`). Edge windows are clamped to the
  image bounds (last row/column may be a partial window placed flush to the edge so no margin is
  dropped — MONAI sliding-window convention). `tile == SAM3_IMAGE_SIZE` (= 1008): tiles are
  native-resolution model-sized crops, so each tile feeds the model at full detail with no
  intra-tile downscale.
- **Per-window run callback** — `run_windows(image, windows, fn)`: applies `fn` (the model
  forward + postprocess) to each window's crop and collects its per-window outputs. Predict/eval
  pass the real forward; this is where the existing OOM ladder (`predict/runner.py:404`) and the
  per-tile transform live.
- **Fragment-merge** — the §4 algorithm: takes per-window instance fragments + their window
  origins, returns merged full-canvas instances.

**Train reuses only window generation** (§5.3). Predict + eval-visualization use the full
utility (generation + run + merge).

**`overlap` is a NEW default hyperparameter.** **`# tbd:` cite or tag.** Reference the MONAI
`sliding_window_inference` convention (`overlap` is a fraction of the ROI size; MONAI's default
is `0.25`). The planner pins the value with a citation to MONAI's documented default (or an
explicit `# tbd:` if the project wants a different fraction justified). Overlap is **internal**
— not a user-facing knob (principle 1.2.1).

### 5.2 PREDICT — tile → run → fragment-merge → ONE full-extent mask

Predict's forward loop today (`predict/runner.py:410-526`) reads each image
(`read_image`, `runner.py:424`), transforms it (`runner.py:435`), batches, and runs the
model emitting per-(image, category) entries (`runner.py:481-491`). The tiling change wraps
this per-image:

1. **Auto-engage decision (per image).** After `read_image` returns `(H, W, C)`
   (`runner.py:429` reads `orig_h, orig_w`), compute `max(H, W)`. If `max(H, W) <= SAM3_IMAGE_SIZE`,
   take the **existing direct path unchanged** (single `LongestMaxSize`+pad forward — byte-for-byte
   as today). If `max(H, W) > SAM3_IMAGE_SIZE`, take the **tiling path**.
2. **Tiling path.** Generate windows (§5.1); for each window crop, run the existing per-tile
   forward (the same model call + `queries_to_coco_results` + score/top_k filtering that runs today,
   but on the tile crop); collect each tile's per-(image, category) fragment instances with the
   window origin. Then run §4 fragment-merge to produce **merged full-canvas instances**.
3. **Output is always one full-size mask** (per merged instance), at the original image extent.
   The writers' existing resize-to-original step (`writers.py:111-117`) is unnecessary on the
   tiling path because the canvas is already at original extent (the planner keeps the resize only
   for the small-image direct path, where the mask comes back at 1008² and must upsample).
4. **`predictions.json` / `run.json` unchanged in shape.** `run.json` (`runner.py:572-591`) gains
   a small `"tiling"` record (engaged: bool; tile, overlap; n_windows) for provenance — additive,
   no breaking change.

**Auto by size; zero new user knobs** (principle 1.2.1). Overlap is internal.

### 5.3 TRAIN — window large labeled rasters into independent tile samples

Training data flows through `CocoDataset` (`data/coco.py`): `__getitem__` (`coco.py:311`)
calls `_decode_image` (`coco.py:203`) → `read_image(img_path, self._channels)` (`coco.py:209`),
then the train transform resize/pads to 1008². For large labeled rasters this loses resolution
identically to predict.

**Train change:** when a sample's source image longest edge exceeds `SAM3_IMAGE_SIZE`, window
**the image and its mask/box annotations together** into independent native-resolution tile
samples — each tile (crop of pixels + the annotations clipped to that window) becomes its **own
training example**. **There is NO restitch in training** — each tile is a standalone sample with
its own queries/targets; the model learns on native-resolution crops. This uses **only the
window-generation half** of §5.1 (no run, no merge).

Planner notes:

- Annotation clipping: boxes are clipped to the window and dropped if their clipped area falls
  below the existing `BboxParams(min_area=0.0, min_visibility=0.0)` floor (`transforms.py:224-228`)
  — preserve current visibility semantics; masks are cropped to the window. An empty tile (no
  annotations after clipping) is still a valid negative sample (consistent with how the model
  trains on negatives today).
- This expands the effective sample count for large rasters (one raster → many tiles). The
  dataset `__len__` / index mapping must account for the tile expansion; the planner picks
  whether to pre-enumerate `(image_idx, window)` pairs at dataset construction (recommended:
  deterministic, supports `data.limit` and shuffling) or lazily.
- The auto-engage threshold is the **same** `max(edge) > SAM3_IMAGE_SIZE` test as predict — one
  rule, shared. Small images keep today's direct path unchanged.

### 5.4 EVAL — tile prediction AND ground truth; match per tile; accumulate; no stitch

Eval runs through `run_eval` (`eval/runner.py:29`) → `Evaluator.evaluate`
(`eval/evaluator.py:291`), which postprocesses per-(image, category) via
`queries_to_coco_results` (`evaluator.py:215`) and accumulates IoU/metrics. Eval does **not**
need a stitched mask for metrics:

- **Tile both prediction and ground truth** with the same windows; run the per-tile forward;
  **Hungarian-match predictions to GT per tile** (the eval matcher already exists —
  `eval/visualize.py:35` imports `HungarianMatcher`, constructed at `visualize.py:353-358`,
  matched at `visualize.py:262`; metric accumulation in `evaluator.py`); **accumulate metrics
  across tiles WITHOUT materializing a stitched full mask.** A seam-crossing object is scored as
  its per-tile fragments against the correspondingly-tiled GT fragments — consistent, and avoids
  the cost/complexity of stitching just to compute a number.
- **Open `# tbd:`** — whether eval uses **overlapping** vs **non-overlapping** tiling for metric
  accumulation. Overlapping tiles double-count objects in the overlap band (inflating/deflating
  metrics depending on direction); non-overlapping avoids double-counting but differs from the
  overlapping tiling predict actually uses. The planner pins this with justification (a defensible
  default is **non-overlapping tiles for metric accumulation** to avoid double-counting, explicitly
  documented as differing from predict's overlapping tiling). **`# tbd:`**
- **Restitch (fragment-merge) ONLY for visualization.** When the user requests an eval
  visualization overlay (`_run_viz`, `eval/runner.py:185`; `write_eval_visualizations`,
  `visualize.py:325`), the overlay path calls the §4 fragment-merge to render one coherent
  full-image overlay. Metrics never trigger a stitch.

---

## 6. Read path & the `SpatialMeta` seam (`src/custom_sam_peft/data/io.py`)

### 6.1 Current read path (landed via #111)

`read_image(path, channels) -> np.ndarray (H, W, C)` (`io.py:53`) dispatches by extension
(`io.py:56-72`): raster exts → PIL; `.npy/.npz` → `np.load`; `.tif/.tiff` → **`tifffile`**
(`io.py:67-71`, pixel bands only). `_coerce_to_channels` (`io.py:15`) normalizes to `(H,W,C)`
and validates `C == channels`. This validation is preserved unchanged.

### 6.2 `SpatialMeta` — an optional sidecar carrier (pixels-first)

Introduce a small frozen dataclass `SpatialMeta` (recommended new module
`src/custom_sam_peft/data/spatial_meta.py` — planner's call) returned **alongside** pixels by
the read path. **Pixels-first, default-None:** plain images (PNG/JPG/npy and tifs with no geo)
return `SpatialMeta = None`, and the model path is **byte-for-byte unchanged**. `SpatialMeta`
never reaches the model — it is carried `read_image → dataset → predict/writers` for output
reconstruction only.

**Read-path signature change.** `read_image` gains an optional metadata return. To avoid
breaking the many existing `read_image(path, channels) -> ndarray` call sites
(`coco.py:209`, `predict/runner.py:424`, and the HF loader path), the planner adds a sibling
that returns both rather than changing the existing return type in place — recommended:

```python
def read_image_with_meta(path, channels) -> tuple[np.ndarray, SpatialMeta | None]: ...
def read_image(path, channels) -> np.ndarray:  # existing; returns just pixels (drops meta)
    return read_image_with_meta(path, channels)[0]
```

so callers that don't need geo/medical metadata are untouched, and predict/writers opt into
`read_image_with_meta`. Planner finalizes the exact shape; the **observable contract** is
"pixels first, optional `SpatialMeta`, `None` for plain images, model path unchanged."

**`SpatialMeta` payload** (a tagged union by source; fields populated per backend):

| Field group | Geo (rasterio) | DICOM (pydicom) | Plain |
| --- | --- | --- | --- |
| `kind` | `"geo"` | `"dicom"` | n/a (meta is `None`) |
| `crs` | source CRS | — | — |
| `affine` | source affine transform (pixel→world) | 3D affine (from `ImagePositionPatient` + `ImageOrientationPatient` + `PixelSpacing`) | — |
| `nodata` | nodata value | — | — |
| `nodata_mask` | bool mask of nodata pixels | — | — |
| `pixel_spacing` | (carried in affine) | `PixelSpacing` | — |
| `orientation` | — | `ImageOrientationPatient` | — |
| `position` | — | `ImagePositionPatient` | — |
| `frame_of_reference_uid` | — | `FrameOfReferenceUID` | — |
| `rescale` | — | (slope, intercept) | — |
| `voi_window` | — | (center, width) or `None` | — |
| `series_uid` / `sop_uid` | — | `SeriesInstanceUID` / `SOPInstanceUID` | — |

### 6.3 `.tif/.tiff` → `rasterio` (REPLACES `tifffile` in base)

`rasterio` becomes the base TIFF reader, **replacing `tifffile`** (`io.py:67-71`). The user
explicitly accepted the GDAL weight; rasterio ships bundled manylinux GDAL wheels (no system
GDAL required).

- **Pixels:** read all bands → `(H, W, C)`; the **same `_coerce_to_channels` / `C == channels`
  validation** as today (`io.py:43-49`) is preserved (band count must equal `data.channels`).
- **Geo metadata (when present):** populate `SpatialMeta(kind="geo", crs=src.crs,
  affine=src.transform, nodata=src.nodata, nodata_mask=...)`. A plain TIFF with no CRS/affine
  returns `SpatialMeta = None` (geo fields absent), keeping behavior identical to the old
  `tifffile` path for non-geo tifs.
- **Preprocess — nodata zero-fill:** where `nodata` is defined, fill those pixels with `0`
  before they reach the model (don't feed fill garbage through the patch-embed) and expose the
  `nodata_mask` in `SpatialMeta` so the writer can re-mark nodata in the output. `# tbd:` the
  planner confirms zero-fill is the right fill value vs per-channel mean (zero is the safe,
  citation-free default — it matches the existing pad fill `fill=0` at `transforms.py:217`; if a
  non-zero fill is chosen it becomes a NEW hyperparameter needing a citation/tag).

### 6.4 Tiling × geo affine composition

Each tile's affine is the **parent affine offset by the window origin** `(y0, x0)` in pixel
space (a pure translation in pixel coordinates composed with the parent affine — no scale change,
since tiles are native-resolution). The **stitched output keeps the parent CRS + affine**: the
full-canvas mask is georeferenced exactly as the source was. The tiling utility threads the
parent `SpatialMeta` and derives per-tile affines internally; this never surfaces to the user.

---

## 7. Writers (`src/custom_sam_peft/predict/writers.py`)

Existing writers (`write_predictions`, `writers.py:64`; RLE encode `writers.py:20`; PNG masks
`writers.py:85-119`) remain available unchanged. Two **new** same-extent output formats are
added, selected by the source `SpatialMeta.kind`:

### 7.1 GeoTIFF mask (geo source)

When the source carried `SpatialMeta(kind="geo")`, write a same-size GeoTIFF mask carrying the
**source CRS + affine** (via `rasterio`'s writer). The mask is the full-extent stitched output
(§5.2). Nodata pixels (`nodata_mask`) are re-marked in the output. PNG / COCO-RLE outputs remain
available alongside (the user can request either).

### 7.2 (reserved)

### 7.3 NIfTI volume (DICOM series source)

A DICOM **series** (multiple `.dcm` slices, grouped per §8) produces **ONE NIfTI volume
(`.nii.gz`)** matching the input volume's dimensions: stack the per-slice masks in geometric
order (sorted by `ImagePositionPatient`), carrying the **3D affine** from `SpatialMeta`. This
opens aligned in 3D Slicer / ITK-SNAP. Stacking is trivial — slices are spatially disjoint along
the through-plane axis, so there is **no NMS, no overlap, no fragment-merge** across slices. A
single 2D `.dcm` produces a same-size **2D mask** (NIfTI 2D, or PNG — planner's call; 2D NIfTI
keeps the affine, PNG drops it).

NIfTI writing uses `nibabel` (the `[dicom]` extra). DICOM-SEG is **not** produced (§2.1).

---

## 8. DICOM read & series handling (`[dicom]` extra)

DICOM support is behind an **optional** `[dicom]` extra (`pydicom` + `nibabel`). Missing extra
at runtime → a clear, actionable install error (e.g.
`"DICOM support requires the optional extra: pip install custom-sam-peft[dicom]"`), raised on the
first `.dcm` read attempt, not at import time (keeps base import light).

### 8.1 Per-slice decode (always)

Read each `.dcm` via `pydicom` → pixel array + spatial tags. Preprocess, **in this order**:

1. **Modality LUT — ALWAYS.** Apply rescale slope/intercept (`RescaleSlope`/`RescaleIntercept`)
   to convert stored values to modality units (e.g. CT Hounsfield). This is mandatory and
   citation-free (it is the DICOM standard's defined transform, applied via
   `pydicom.pixel_data_handlers.apply_modality_lut`).
2. **Correct signed / bits-stored decode.** Honor `PixelRepresentation` (signed vs unsigned),
   `BitsStored`, `BitsAllocated` so values decode correctly (pydicom handles this when the pixel
   array is accessed correctly; the planner verifies signed CT decodes negative HU).
3. **VOI windowing — ONLY if the file carries a window.** If the file has `WindowCenter` /
   `WindowWidth`, apply the **file's own** center/width (`apply_voi_lut`) to the Modality-LUT
   (HU-space) output. **Do NOT invent a window default** — if no window is present, skip VOI (no
   NEW hyperparameter is introduced). An optional **per-dataset config override** lets the user
   specify a center/width explicitly (§9). When the override is set it wins; otherwise the file's
   window is used; otherwise no VOI.
4. **MONOCHROME1 inversion.** When `PhotometricInterpretation == "MONOCHROME1"`, invert so that
   higher stored value = brighter (MONOCHROME1 is inverted relative to MONOCHROME2). Per **DICOM
   PS3.3 §C.11.2**, MONOCHROME1 inversion is a *display-time* (P-Value) step that applies **after**
   VOI windowing — VOI operates on Modality-LUT (HU-space) values, never on inverted ones. (An
   earlier draft of this spec placed inversion before VOI; that corrupts a MONOCHROME1 file carrying
   a window — its window is applied to inverted, non-HU values — so the order was corrected here to
   match the implementation, which cites PS3.3 §C.11.2 in-code.) Standard, citation-anchored.

The decoded slice becomes the `(H, W, C)` pixel array fed to the model (typically `C == 1` for
CT/MR — the `data.channels=1` grayscale path from #111 applies). `SpatialMeta(kind="dicom")` is
populated per §6.2.

### 8.2 Series-aware grouping

Multiple input `.dcm` files are **grouped by `SeriesInstanceUID`**, slices within a series
**sorted by `ImagePositionPatient`** (projected onto the slice-normal derived from
`ImageOrientationPatient`), and a **3D affine** is built from `PixelSpacing` +
`ImageOrientationPatient` + `ImagePositionPatient` (the standard DICOM→world mapping; the
planner uses a verified construction such as nibabel's affine-from-DICOM convention, citing it).
The series + sort order drive the NIfTI stacking order (§7.3).

### 8.3 Tiling rarely triggers for DICOM (note)

Typical CT/MR slices are ≤512² — below `SAM3_IMAGE_SIZE = 1008` — so DICOM input **rarely
triggers tiling**; the auto-engage size test (§5.2/§5.3) simply takes the direct path for sub-1008
slices. The tiling path is **shared** (no DICOM-specific branch), so an unusually large slice
(e.g. some mammography/DX ≥ 1008) would tile correctly via the same utility.

---

## 9. Config / schema changes (`src/custom_sam_peft/config/schema.py`)

**Config surface is deliberately minimal** (principle 1.2.1). Tiling auto-engages by input size
with **no user knob**; overlap is internal. `SpatialMeta` backend is auto-detected by
extension/content. The **only** new user-facing config is the optional DICOM VOI override:

- **`data.dicom_voi_window: tuple[float, float] | None = None`** (or a small nested
  `DicomConfig` with `voi_center` / `voi_width` — planner's call). Default `None` → use the
  file's own window (or no VOI). When set, overrides the file window for **all** slices in the run.
  This is an explicit user choice, not a default guess, so it needs **no citation** (it is `None`
  by default — no hyperparameter is shipped).

No other new config fields. The internal `overlap` and `mask_overlap_threshold` (§4/§5.1) are
**internal constants** (not config), each marked `# tbd:` / cited where defined in code (per the
CI no-uncited-default hook). The planner places them as named module constants with citation/tag
comments, NOT as silent literals.

---

## 10. Dependencies (`pyproject.toml`)

Current base deps include `"tifffile>=2024.1"` (`pyproject.toml:21`); optional-deps groups are
`wandb`, `tensorboard`, `qlora`, `cu130`, `jupyter`, `dev` (`pyproject.toml:31-44`).

- **`rasterio` → base `[project].dependencies`, REPLACING `tifffile`.** Remove the `tifffile`
  line; add `rasterio`. (rasterio's manylinux wheels bundle GDAL — no system GDAL.) **`# tbd:`
  version floor** — planner pins with justification (a recent stable rasterio with bundled GDAL).
  Verify nothing else imports `tifffile` after `io.py` is migrated (grep before removing the dep).
- **`pydicom` + `nibabel` → NEW `[dicom]` optional-dependencies extra.** **`# tbd:` version
  floors** for both — planner pins with justification.
- **Missing `[dicom]` at runtime → clear install error** (§8). Import `pydicom`/`nibabel` lazily
  inside the DICOM read/write code (not at module top) so base install/import never requires them.

All four floors (`rasterio`, `pydicom`, `nibabel`, and confirming `tifffile` removal causes no
breakage) are left as **`# tbd:` for the planner to pin with justification** — version floors must
be pinned with a reason, not guessed (repo convention; #248 / hyperparam-citation rule).

---

## 11. Error handling

1. **Channel-count mismatch (preserved).** rasterio/pydicom reads run through the existing
   `_coerce_to_channels` `C == channels` validation (`io.py:43-49`) — a band/channel count that
   disagrees with `data.channels` raises the clear existing error.
2. **Missing `[dicom]` extra** → actionable install error on first `.dcm` access (§8).
3. **Malformed / non-geo TIFF** → reads pixels fine, `SpatialMeta = None` (not an error; geo is
   optional).
4. **DICOM missing required geometry tags** (no `ImagePositionPatient` / `ImageOrientationPatient`
   for a series) → clear error explaining the series can't be stacked into a NIfTI volume; a single
   slice without geometry still produces a 2D mask (degrade gracefully).
5. **Mixed-series input** (DICOM files from multiple `SeriesInstanceUID`s in one predict run) →
   group and emit one NIfTI per series; document that one input dir → potentially several volumes.
6. **Tiling on a degenerate image** (exactly 1008, or 1009×10) → window generation must produce
   ≥1 valid window and clamp edges; a 1009-wide image yields two overlapping windows, the second
   flush to the right edge.

---

## 12. Testing

**Synthetic, offline, CPU-only fixtures — NO real patient data, NO network.** Real-model GPU
tests run via `scripts/run_gpu_tests.sh` (standing policy); all IO/metadata/tiling-math tests are
CPU-only and must NOT load the real SAM 3.1 model. Use `-o "addopts="` to bypass the global
`--cov-fail-under` on CPU-only test subsets (repo convention).

### 12.1 CPU tests

| # | Case | Assertion |
| --- | --- | --- |
| C1 | Window generation | `iter_windows(H,W,1008,overlap)` covers the full image, edge windows clamp flush to bounds, overlap band width matches the configured fraction; a ≤1008 image yields exactly one window (direct-path equivalent). |
| C2 | **Seam fragment-merge (the crux)** | Build an oversized synthetic raster (e.g. 1500×1500) with a known object straddling a seam → tile with overlap → run fragment-merge over per-tile fragments → assert the stitched output merges the seam-crossing object into **ONE** instance whose mask equals a reference full-canvas mask, and the merged score equals the chosen aggregation rule. |
| C3 | 3+-tile transitive merge | An object spanning three tiles (fragments A–B–C, A and C non-adjacent) merges into one instance via union-find. |
| C4 | Distinct objects don't merge | Two distinct same-category objects in adjacent tiles whose masks do NOT overlap in the band stay **separate**; two different-category overlapping fragments never merge. |
| C5 | Auto-engage threshold | `max(edge) <= 1008` → direct path (no tiling, output identical to pre-feature); `> 1008` → tiling path. Boundary at exactly 1008 takes the direct path. |
| C6 | Train tile expansion | A large labeled raster windows into N independent tile samples; boxes/masks are clipped per window; an empty tile is a valid (negative) sample; dataset `__len__` reflects the expansion deterministically. |
| C7 | **GeoTIFF CRS/affine round-trip** | A CRS-tagged synthetic GeoTIFF read via rasterio → `SpatialMeta(kind="geo")` carries CRS+affine+nodata; write a GeoTIFF mask → re-read → **CRS and affine round-trip exactly**. |
| C8 | Tiling × affine composition | Per-tile affine = parent affine offset by window origin; stitched output's affine == parent affine (native-res, no scale change). |
| C9 | nodata zero-fill | nodata pixels are zero-filled before the model and the `nodata_mask` is exposed and re-applied to the output mask. |
| C10 | DICOM decode correctness | Synthetic single-series `.dcm` slices: Modality LUT (slope/intercept) applied (signed CT decodes negative HU); MONOCHROME1 inverted; VOI applied **only** when the file carries a window; the config override wins when set. |
| C11 | DICOM series → NIfTI affine | A few synthetic single-series slices group by `SeriesInstanceUID`, sort by `ImagePositionPatient`, stack into ONE NIfTI volume of the correct dims, and the **3D affine is correct** (re-read via nibabel matches the constructed affine). |
| C12 | Missing `[dicom]` extra | Simulate the extra absent (monkeypatch the import) → reading a `.dcm` raises the actionable `pip install custom-sam-peft[dicom]` error. |
| C13 | SpatialMeta is None for plain images | PNG/JPG/`.npy` and non-geo TIFF return `SpatialMeta = None`; the model/predict path for these is unchanged. |
| C14 | rasterio replaces tifffile | `.tif/.tiff` route through rasterio; band-count `!= channels` raises the existing error; no remaining `tifffile` import in the codebase. |

### 12.2 GPU-only tests (via `scripts/run_gpu_tests.sh`)

| # | Case | Why GPU-only |
| --- | --- | --- |
| G1 | Real-model tiled predict end-to-end | Run `run_predict` on an oversized image with a real SAM 3.1 model; assert one full-extent mask is emitted, no tiles leak to the user, and a seam-crossing object is a single instance. |
| G2 | Real-model eval tiled accumulation | `run_eval` on an oversized eval sample accumulates per-tile metrics without materializing a stitched mask; the visualize path renders one stitched overlay. |

---

## 13. Open items the planner MUST pin (each `# tbd:` or a citation)

1. **Tile overlap fraction** (§5.1) — cite MONAI `sliding_window_inference` default (0.25) or
   `# tbd:` with justification.
2. **`mask_overlap_threshold`** for cross-tile fragment association (§4.3) — `# tbd:` / cite.
3. **Association metric** (§4.3) — IoU-of-overlap-band vs intersection-over-min-area — pin.
4. **Score-aggregation rule** for merged fragments (§4.5) — area-weighted mean vs `max` — pin.
5. **Eval tiling: overlapping vs non-overlapping** for metric accumulation (§5.4) — pin with
   justification (recommended default: non-overlapping, documented as differing from predict).
6. **nodata fill value** (§6.3) — zero (recommended, citation-free, matches existing pad fill) vs
   per-channel mean — confirm.
7. **`rasterio` / `pydicom` / `nibabel` version floors** (§10) — pin each with justification;
   confirm `tifffile` removal breaks nothing.

---

## 14. Open risks

1. **Stitch correctness (highest risk).** §4 — the fragment-merge is the feature's reason to
   exist; a wrong association threshold either over-merges distinct objects or leaves an object
   fragmented. C2/C3/C4 + G1 are the guards. The threshold/metric/aggregation `# tbd:`s (§13) must
   be pinned with justification and exercised against the synthetic seam fixture.
2. **Tiled inference cost.** N tiles per large image = N× forwards; the existing OOM ladder
   (`predict/runner.py:404`) governs per-tile VRAM, but wall-clock scales with tile count. This is
   an accepted cost (accuracy >> speed, principle 1.2.1); document it, don't optimize it away.
3. **rasterio/GDAL wheel weight.** Accepted by the user; mitigated by bundled manylinux wheels.
   Confirm the CI environment installs the rasterio wheel cleanly (no system GDAL) before relying
   on it in tests; if a CI image lacks the wheel, the geo tests gate on rasterio availability.
4. **DICOM geometry edge cases.** Gantry tilt, non-axial acquisitions, irregular slice spacing
   can make the 3D affine non-trivial. v1 targets regular single-series stacks; §11.4 degrades
   gracefully (2D mask) when geometry is absent. Irregular-spacing handling is a documented sharp
   edge, not a v1 guarantee.
5. **Train tile expansion × `data.limit`.** Windowing multiplies sample count; the `data.limit`
   subset cap (#245 / `2026-05-22-data-subset-limit-design.md`) and the no-val auto-split must see
   the **post-expansion** index space consistently. The planner must verify limit/split apply after
   tile enumeration (mirrors the #245 bundle val-limit alignment lesson).

---

## 15. Out-of-scope / follow-up issues to file

1. **DICOM-SEG output** (clinical/PACS) — explicitly dropped (§2.1); record the decision in #131.
2. **CRS reprojection / coordinate transforms** (§2.2).
3. **Resample-to-target-GSD tiling** (§2.3) — would need fixed-physical-extent tiling + a cited
   resolution hyperparameter.
4. **Coordinate-based / geospatial prompting** (§2.4).
5. **True 3D volumetric model input** (#110, disjoint) — this spec reads DICOM series for geometry
   + output stacking only; the model stays 2D-slice.
6. **Per-channel / learned tiling blend** (e.g. Gaussian-weighted overlap blending of mask
   logits) — v1 uses logical-OR union within a merged component; a softer blend is a future
   refinement.

---

## Appendix A — Verified anchors (worktree `131-georeferencing-dicom`, 2026-06-02)

| Symbol / site | Verified location |
| --- | --- |
| `SAM3_IMAGE_SIZE = 1008` | `models/sam3.py:111` |
| `_SAM3_PATCH = 14` | `presets.py:151` |
| `read_image` / `_coerce_to_channels` | `data/io.py:53` / `io.py:15`; `.tif`→tifffile `io.py:67-71`; `C==channels` validation `io.py:43-49` |
| `build_eval_transforms` LongestMaxSize+pad | `data/transforms.py:197`; `LongestMaxSize` `:212`, `PadIfNeeded` `:213`, `Normalize` `:221` |
| `build_train_transforms` LongestMaxSize+pad | `data/transforms.py:233`; resize/pad `:271-279`; `BboxParams` `:224-228` |
| `resolve_normalization` (channel-semantics aware) | `transforms.py:183`; `_with_path` `:113` |
| COCO `_decode_image` → `read_image` | `data/coco.py:203-209`; `__getitem__` `:311`; `self._channels` `:136` |
| Predict per-image read + transform + forward | `predict/runner.py:424` (read), `:435` (transform), `:456` (model fwd), `:481` (`queries_to_coco_results`), `:488` (score_threshold), `:490` (top_k) |
| Predict `_resolve_config` (channels/semantics) | `predict/runner.py:111`; channels/semantics `:176-185`; `_ResolvedConfig` `:91-104` |
| Predict OOM ladder | `predict/runner.py:404` |
| Predict `run.json` writer call | `predict/runner.py:572-592` |
| Writers: predictions / RLE / PNG resize | `predict/writers.py:64` / `:20` / `:111-117` |
| Predict visualize | `predict/visualize.py:137` (`write_visualization`), `:161` (`Image.open(...).convert("RGB")`) |
| Eval runner + viz hook | `eval/runner.py:29` (`run_eval`), `:185` (`_run_viz`) |
| Evaluator postprocess accumulation | `eval/evaluator.py:291` (`evaluate`), `:215` (`queries_to_coco_results`) |
| Eval Hungarian matcher | `eval/visualize.py:35` (import), `:353-358` (construct), `:262` (match) |
| Train matcher (loss) | `models/losses/compose.py:133` (construct), `:149` (match) |
| `pyproject` base deps / `tifffile` / optional groups | `pyproject.toml:9-29`; `tifffile` `:21`; optional `:31-44` |
