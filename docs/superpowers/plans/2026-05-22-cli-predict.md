# `csp predict` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-22-cli-predict-design.md`](../specs/2026-05-22-cli-predict-design.md)
**Tracking issue:** [#74](https://github.com/NguyenJus/custom-sam-peft/issues/74) — *feat(cli): add `predict`*
**Branch:** `feat/cli-predict`
**Worktree:** `/home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict`

**Goal:** Ship the `csp predict` command end-to-end: pure-function input/prompt/writer/visualize helpers, a `predict/runner.py` library entry point, a thin Typer shell at `cli/predict_cmd.py`, full CPU unit + smoke coverage, GPU integration coverage gated by `@pytest.mark.gpu`, and a short README section. No edits under `eval/`, `train/`, `models/`, `peft_adapters/`, or `cli/run_cmd.py`.

**Architecture:** New top-level package `src/custom_sam_peft/predict/`. Six modules: `inputs.py` (pure), `adapter_load.py` (thin wrappers), `writers.py` (pure), `visualize.py` (pure), `runner.py` (orchestrator + forward loop), and the package `__init__.py` re-exports. CLI surface is a single new file `cli/predict_cmd.py` plus one line in `cli/main.py`. Forward loop reuses `models.sam3.load_sam31`, `data.transforms.build_eval_transforms`, `data.transforms.resolve_normalization`, `peft_adapters.{lora,qlora}.{load_lora,load_qlora,merge_lora}`, and `eval.postprocess.queries_to_coco_results` verbatim. Postprocess is per-image (`pred_logits.shape[0] == 1` is asserted upstream — see spec §9 and §12); v1 honors `--batch-size INT` as a user-visible flag but the loop calls `queries_to_coco_results` one image at a time.

**Tech Stack:** Existing project deps only — `torch`, `numpy`, `pillow`, `pycocotools`, `typer`, `peft`, `bitsandbytes` (QLoRA path), `pytest` / `pytest-cov`, `ruff`, `mypy --strict`. No new runtime dependencies.

---

## Ordering rationale

Order is **minimum-blast-radius first, leaf modules before integration**:

1. Pure-function leaves (`inputs.py`, `writers.py`, `visualize.py`) and the near-pure `adapter_load.py` come first — no torch model on the import path, so failures are cheap to localize.
2. `runner.py` lands once its dependencies exist; the smoke test pins it against a stub `nn.Module` so the integration path can be debugged without a real SAM 3.1 load.
3. CLI shell lands next — it depends on `PredictOptions` being final.
4. Preprocessing parity is a one-file guard that can run any time after `runner.py` exists; it lives alongside Phase 6 to keep latency-sensitive feedback close to Phase 5.
5. GPU tests, docs, and the final lint/typecheck/coverage pass are last.

This mirrors the spec's "reuse existing helpers, do not duplicate" rule (§4) and the train/eval/export shell + runner pattern from `2026-05-18-cli-design.md` §3.

---

## File Map

**New files:**

```
src/custom_sam_peft/cli/predict_cmd.py                NEW
src/custom_sam_peft/predict/__init__.py               NEW
src/custom_sam_peft/predict/runner.py                 NEW
src/custom_sam_peft/predict/inputs.py                 NEW
src/custom_sam_peft/predict/adapter_load.py           NEW
src/custom_sam_peft/predict/writers.py                NEW
src/custom_sam_peft/predict/visualize.py              NEW
tests/predict/__init__.py                             NEW
tests/predict/conftest.py                             NEW   (shared fixtures)
tests/predict/fixtures/                               NEW   (stub adapter dirs, synthetic images)
tests/predict/test_inputs.py                          NEW
tests/predict/test_prompts.py                         NEW
tests/predict/test_config_layering.py                 NEW
tests/predict/test_adapter_detect.py                  NEW
tests/predict/test_writers.py                         NEW
tests/predict/test_visualize.py                       NEW
tests/predict/test_runner_smoke.py                    NEW
tests/predict/test_dry_run.py                         NEW
tests/predict/test_cli_predict.py                     NEW
tests/predict/test_preprocessing_parity.py            NEW
tests/predict/test_gpu_predict.py                     NEW   (@pytest.mark.gpu)
```

**Modified files:**

```
pyproject.toml                         TOUCHED  (+1 line: csp = "custom_sam_peft.cli.main:app" under [project.scripts])
src/custom_sam_peft/cli/main.py        TOUCHED  (+1 import, +1 register line)
README.md                              TOUCHED  ("Run inference on your images" subsection)
```

No deletions, no moves. `src/custom_sam_peft/eval/` is untouched. `src/custom_sam_peft/cli/run_cmd.py` is untouched. Per spec §1.3 / §13.

---

## Parallelization opportunities (for orchestrator dispatch)

Per the CLAUDE.md guidance on `superpowers:dispatching-parallel-agents`, dispatch in parallel only when tasks are file-disjoint and have no shared state.

| Group | Phases | Disjoint? | Dispatch |
| --- | --- | --- | --- |
| A | 1, 2, 3, 4 | Yes — each owns its own `predict/*.py` plus its tests | **Parallel** |
| B | 5 | Depends on A (imports inputs/adapter_load/writers/visualize) | Serial |
| C | 6 | Depends on B (CLI shell builds `PredictOptions` consumed by runner) | Serial |
| D | 7 | Depends on B; can parallel with C (different files) | **Parallel with 6** |
| E | 8 | Depends on C (calls the registered Typer command) | Serial |
| F | 9 | Depends on E (README mentions verified flags) | Serial |
| G | 10 | Last — lint / typecheck / coverage gate | Serial |

Dependency graph:

```
Phase 1 ┐
Phase 2 ├→ Phase 5 → Phase 6 ┐
Phase 3 │             Phase 7 ┴→ Phase 8 → Phase 9 → Phase 10
Phase 4 ┘
```

---

## Pre-flight check

- [ ] **Step 0a: Confirm worktree and branch**

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict rev-parse --show-toplevel
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict rev-parse --abbrev-ref HEAD
```

Expected: worktree path matches and branch is `feat/cli-predict`. If either differs, halt — the orchestrator's safety-check should have caught this.

- [ ] **Step 0b: Confirm `uv sync` is up-to-date**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv sync --all-extras
```

Expected: exits 0, no resolver errors.

- [ ] **Step 0c: Confirm baseline CPU tests are green on `main`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/unit -m 'not gpu' -x -q
```

Expected: all unit tests pass. If anything is red, halt and report — do not start on a broken baseline.

- [ ] **Step 0d: Confirm lint + typecheck baselines are clean**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run ruff check . && uv run mypy --strict src/custom_sam_peft
```

Expected: both exit 0.

- [ ] **Step 0e: Note GPU test policy**

Per `2026-05-19-gpu-test-policy-design.md` (and confirmed by the project memory file `feedback_gpu_vs_cpu_testing.md`): CPU tests are the gate; GPU tests default-skip under `pytest -m 'not gpu'`. The `tests/predict/test_gpu_predict.py` cases (Phase 8) are reserved for **real-only failure modes** (real SAM 3.1 load, real bnb 4-bit quant, real `merge_and_unload` dequant path, VRAM hint log on cuda).

---

## Phase 1 — Inputs + prompts (`predict/inputs.py`)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 2, 3, 4 (file-disjoint).
**Depends on:** none (pure functions, no torch).
**Spec:** §5.1, §5.2, §10 (BadParameter exits).

**Files:**
- Create: `src/custom_sam_peft/predict/__init__.py` (placeholder — final re-exports added in Phase 5)
- Create: `src/custom_sam_peft/predict/inputs.py`
- Create: `tests/predict/__init__.py` (empty)
- Create: `tests/predict/conftest.py` (shared `tmp_path`-based fixtures used in this phase + later phases)
- Create: `tests/predict/test_inputs.py`
- Create: `tests/predict/test_prompts.py`

### Task 1a: Tests first

- [ ] **Step P1-1: Write `tests/predict/test_inputs.py`**

One test per behavior from spec §5.1:

| Test name | Intent |
| --- | --- |
| `test_resolve_images_directory_recursive` | A nested tmpdir with `.jpg`, `.png`, `.txt` → recurses into subdirs, collects only allowed exts. |
| `test_resolve_images_glob_recursive` | `tmp/**/*.jpg` returns sorted absolute paths. |
| `test_resolve_images_single_file` | A single `.png` file path is returned as `[Path(...)]`. |
| `test_resolve_images_txt_manifest` | A `.txt` manifest with comments (`# …`) and blank lines is parsed; relative entries resolve against the manifest's parent dir. |
| `test_resolve_images_json_manifest` | A `.json` manifest decoding to `list[str]` is parsed; non-list contents raise. |
| `test_resolve_images_extension_allowlist` | Files outside the spec §5.1 allowlist (e.g. `.gif`, `.txt` as image) are filtered. |
| `test_resolve_images_rgba_to_rgb_implicit` | The function returns the path; mode normalization happens in the runner — assert the file is in the result and `PIL.Image.open(...).convert("RGB")` succeeds against it. |
| `test_resolve_images_unreadable_warn_and_skip` | A corrupt JPEG (zero-byte file with `.jpg` ext, opened later by the runner) — for `resolve_images` this is **path-level** valid; the WARN-and-skip behavior is exercised by the runner's per-image loop (covered in Phase 5). Assert the corrupt path is still in the resolved list. |
| `test_resolve_images_zero_result_raises` | A tmpdir with no allowed-ext files → `typer.BadParameter`, message includes the spec string. |
| `test_resolve_images_sort_determinism` | Unsorted tmpdir → returned list is sorted by absolute-path string. |

- [ ] **Step P1-2: Write `tests/predict/test_prompts.py`**

| Test name | Intent |
| --- | --- |
| `test_parse_prompts_comma_string` | `"cat,dog,person"` → `["cat", "dog", "person"]`. |
| `test_parse_prompts_one_per_line_file` | UTF-8 file with one class per line → list preserves order. |
| `test_parse_prompts_strips_whitespace` | `"  cat  ,dog   "` → `["cat", "dog"]`. |
| `test_parse_prompts_drops_empty_entries` | `"cat,,dog,"` → `["cat", "dog"]`. |
| `test_parse_prompts_dedupes_first_seen` | `"cat,dog,cat,bird,dog"` → `["cat", "dog", "bird"]`. |
| `test_parse_prompts_empty_raises` | `""` or empty file → `typer.BadParameter` with message from spec §5.2. |

### Task 1b: Implementation

- [ ] **Step P1-3: Implement `src/custom_sam_peft/predict/inputs.py`**

Functions (signatures only — implementer writes the bodies):

```python
ALLOWED_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

def resolve_images(spec: str | Path) -> list[Path]: ...
def parse_prompts(spec: str | Path) -> list[str]: ...
```

Implementation notes:
- Use `pathlib.Path` exclusively; never `os.path`.
- `glob.glob(spec, recursive=True)` only when the literal string contains `*` or `?` (spec §5.1).
- Manifest path resolution uses `manifest.parent / entry` for non-absolute entries (spec §5.1).
- Comment lines are `lstrip().startswith("#")`; the `#` may follow whitespace.
- Sort the final list with `sorted(..., key=lambda p: str(p.resolve()))`.
- Raise `typer.BadParameter("no images resolved from <spec>")` and `typer.BadParameter("--prompts must resolve to at least one non-empty class name")` (verbatim per spec §5.1, §5.2).

- [ ] **Step P1-4: Stub `src/custom_sam_peft/predict/__init__.py`**

Leave the file empty for now (or write only a docstring). Re-exports of `run_predict`, `PredictOptions`, `PredictReport` are added at the end of Phase 5 to avoid circular-import surprises during early development.

### Task 1c: Acceptance gate

- [ ] **Step P1-5: Tests pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_inputs.py tests/predict/test_prompts.py -q
```

Expected: all green. No torch on the import path.

**Reviewer focus:** manifest-relative path resolution (not relative to `cwd`); the exact `typer.BadParameter` strings; sorted determinism; comment-line tolerance for leading whitespace.

---

## Phase 2 — Adapter detection (`predict/adapter_load.py`)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1, 3, 4.
**Depends on:** none (uses existing `peft_adapters` modules, no model load).
**Spec:** §6 (QLoRA detection, adapter pin behavior, merge toggle).

**Files:**
- Create: `src/custom_sam_peft/predict/adapter_load.py`
- Create: `tests/predict/fixtures/lora_adapter/` (stub dir with `adapter_config.json` only)
- Create: `tests/predict/fixtures/qlora_adapter/` (stub dir with `adapter_config.json` + `custom_sam_peft_qlora.json`)
- Create: `tests/predict/test_adapter_detect.py`
- Create: `tests/predict/test_config_layering.py`

### Task 2a: Fixtures

- [ ] **Step P2-1: Build stub adapter dirs under `tests/predict/fixtures/`**

| Fixture | Contents | Notes |
| --- | --- | --- |
| `lora_adapter/adapter_config.json` | `{"base_model_name_or_path": "facebook/sam3.1", "peft_type": "LORA", "r": 8, ...}` | Minimal, no weights. `detect_adapter_kind` only reads filenames. |
| `qlora_adapter/adapter_config.json` | Same as LoRA stub. | |
| `qlora_adapter/custom_sam_peft_qlora.json` | `{"version": 1, "bnb_config": {"load_in_4bit": true, ...}}` | The presence of this file is what flips detection to `"qlora"` (spec §6). |
| `bad_adapter/` (empty dir) | none | For the "no `adapter_config.json` → raises" test. |

Keep payloads minimal — these fixtures are for detection only. Real adapter restore is exercised by Phase 8 GPU tests.

### Task 2b: Tests first

- [ ] **Step P2-2: Write `tests/predict/test_adapter_detect.py`**

| Test name | Intent |
| --- | --- |
| `test_detect_adapter_kind_lora` | `lora_adapter/` fixture → `"lora"`. |
| `test_detect_adapter_kind_qlora` | `qlora_adapter/` fixture → `"qlora"` (presence of `custom_sam_peft_qlora.json`). |
| `test_detect_adapter_kind_missing_adapter_config_raises` | `bad_adapter/` → `typer.BadParameter` (or `ValueError`, matching the spec §10 flag-callback exit 2 chain). |
| `test_load_adapter_dispatches_to_lora` | Monkeypatch `peft_adapters.lora.load_lora` and `peft_adapters.qlora.load_qlora`; assert the right one is called per detected kind. |
| `test_merge_adapter_toggle_off_skips_merge_lora` | Monkeypatch `peft_adapters.lora.merge_lora`; assert it is NOT called when `merge_adapter=False`. |
| `test_merge_adapter_toggle_on_calls_merge_lora` | Same monkeypatch; assert it IS called when `merge_adapter=True`. |

- [ ] **Step P2-3: Write `tests/predict/test_config_layering.py`**

Tests the adapter-pins-model-name rule from spec §6:

| Test name | Intent |
| --- | --- |
| `test_adapter_pin_overrides_config_with_warn` | Adapter stub has `base_model_name_or_path: "facebook/sam3.1"`; supplied config says `"some/other-model"`. Resolution yields `"facebook/sam3.1"` and a single `WARN` log line names both values. |
| `test_no_checkpoint_falls_through_to_config` | When `checkpoint=None`, config-supplied `model.name` wins. |
| `test_no_checkpoint_no_config_uses_builtin_default` | When both are absent, `model.name == "facebook/sam3.1"`. |
| `test_cli_flag_beats_config` | A CLI override (passed via `PredictOptions`) wins over `--config` for non-pinned fields (e.g. `score_threshold`). |

These tests instantiate a `PredictOptions` (frozen dataclass — defined in Phase 5) and a `Resolved` view returned by the layering helper. **If `PredictOptions` is not yet importable at the time Phase 2 lands**, gate this file with a `pytest.importorskip("custom_sam_peft.predict.runner")`-style skip and mark it as needing un-skip in Phase 5. (Recommended path: defer `test_config_layering.py` to Phase 5's tests block; it sits at the top of Phase 5's deliverables.)

### Task 2c: Implementation

- [ ] **Step P2-4: Implement `src/custom_sam_peft/predict/adapter_load.py`**

Signatures:

```python
from typing import Literal

AdapterKind = Literal["lora", "qlora"]

def detect_adapter_kind(checkpoint_dir: Path) -> AdapterKind: ...
def load_adapter(model: nn.Module, checkpoint_dir: Path, kind: AdapterKind) -> nn.Module: ...
def maybe_merge_adapter(model: nn.Module, *, merge: bool) -> nn.Module: ...
def read_adapter_base_model_name(checkpoint_dir: Path) -> str | None: ...
```

Implementation notes:
- `detect_adapter_kind` reads only filenames — `(checkpoint_dir / "custom_sam_peft_qlora.json").is_file()` → `"qlora"`, else assert `(checkpoint_dir / "adapter_config.json").is_file()` and return `"lora"`. If neither, raise `typer.BadParameter("--checkpoint must contain adapter_config.json")` (matches spec §10).
- `load_adapter` is a 4-line dispatch around `peft_adapters.lora.load_lora` / `peft_adapters.qlora.load_qlora`. **Do not duplicate logic** — see spec §4 reuse table.
- `maybe_merge_adapter` calls `peft_adapters.lora.merge_lora(model)` when `merge=True` and returns the model. On QLoRA + merge=True, the function does NOT auto-disable (spec §6 — "the user makes the call").
- `read_adapter_base_model_name` returns `json.load(...)["base_model_name_or_path"]` or `None` if absent. Used by the config-layering helper (which lives in `runner.py` Phase 5).

### Task 2d: Acceptance gate

- [ ] **Step P2-5: Tests pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_adapter_detect.py -q
```

Expected: green. `test_config_layering.py` runs at the Phase 5 gate.

**Reviewer focus:** Detection precedence (QLoRA sentinel beats LoRA fallback even if both files are present — read the spec §6 wording carefully); the `BadParameter` message; that `peft_adapters` is imported lazily so the base-model-only hot path does NOT import it (spec §2 "no `peft_adapters` import on the hot path").

---

## Phase 3 — Writers (`predict/writers.py`)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1, 2, 4.
**Depends on:** none (pure I/O over dicts).
**Spec:** §7.1, §7.2, §7.3, §11.1 (top-K + score-threshold math).

**Files:**
- Create: `src/custom_sam_peft/predict/writers.py`
- Create: `tests/predict/test_writers.py`

### Task 3a: Tests first

- [ ] **Step P3-1: Write `tests/predict/test_writers.py`**

| Test name | Intent |
| --- | --- |
| `test_write_predictions_json_schema` | Synthetic entries → file decodes; required fields per spec §7.1 are present (`image_id`, `category_id`, `bbox`, `score`, `segmentation`). |
| `test_predictions_bbox_is_xywh_in_original_coords` | Caller supplies entries already in original coords (the inverse-transform lives in the runner); writer must NOT mutate. |
| `test_predictions_rle_round_trip_via_pycocotools` | `segmentation.counts` is decoded ASCII; `pycocotools.mask.decode(rle)` returns a 2-D `uint8` mask of `size` (mirrors eval's `_mask_to_rle`). |
| `test_image_id_map_json_shape` | `{str(image_id): str(abs_path)}`; keys are JSON-string ints. |
| `test_run_json_has_all_required_keys` | All keys from spec §7.3 present; `checkpoint` and `adapter_kind` are `None` on the base-model-only path. |
| `test_run_json_git_sha_optional` | When called from a non-git path, `git_sha` is `None` (not the string `"None"`). |
| `test_top_k_and_score_threshold_per_image_class` | Synthetic entries: filter by `score >= threshold` first, then keep highest-K. Per `(image_id, category_id)`, not global. Worst-case row count is `K * num_classes * num_images` (spec §7.1). |
| `test_save_masks_none_omits_segmentation` | With `save_masks="none"`, the field is absent from every entry. |
| `test_save_masks_rle_default_keeps_segmentation` | With `save_masks="rle"`, `segmentation` present and decodable. |
| `test_save_masks_png_writes_files_and_sets_mask_png` | With `save_masks="png"`, PNG files written to `<output>/masks/<stem>_<cat>_<idx>.png`; entry has `mask_png` (relative to `<output>`); `segmentation` dropped. |
| `test_png_mask_dims_match_original_hw` | PNG dimensions == original image H×W (spec §7 layout block). |

### Task 3b: Implementation

- [ ] **Step P3-2: Implement `src/custom_sam_peft/predict/writers.py`**

Signatures (illustrative, ≤ 5 lines):

```python
def select_top_k_per_image_class(entries, *, score_threshold: float, top_k: int) -> list[dict]: ...
def write_predictions(entries, output_dir: Path, *, save_masks: Literal["rle","png","none"], originals: dict[int, tuple[int,int]]) -> None: ...
def write_image_id_map(id_to_path: dict[int, Path], output_dir: Path) -> None: ...
def write_run_json(run_meta: dict, output_dir: Path) -> None: ...
def encode_rle_dict(mask: np.ndarray) -> dict: ...   # uses pycocotools.mask.encode + .decode("ascii")
def decode_rle_to_uint8(rle: dict) -> np.ndarray: ...  # used for PNG emission
```

Implementation notes:
- Use `pycocotools.mask.encode(np.asfortranarray(mask))`; copy eval's `_mask_to_rle` decode-ASCII trick so the JSON is valid (spec §7.1).
- Top-K math: group by `(image_id, category_id)`, sort each group by `-score`, slice `[:top_k]`, flatten — do this BEFORE the writer emits anything (also called from the runner before mask-PNG emission).
- `image_id_map.json` writes `str(image_id)` keys (JSON doesn't permit int keys at the top level for some readers; safer as strings per spec §7.2 example).
- `write_run_json` queries `git rev-parse --short HEAD` via `subprocess.run(..., check=False, capture_output=True)`; on non-zero exit, set `git_sha=None`. Version comes from `custom_sam_peft.__version__`.

### Task 3c: Acceptance gate

- [ ] **Step P3-3: Tests pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_writers.py -q
```

Expected: green; pycocotools round-trips succeed; PNGs written under `tmp_path/masks/`.

**Reviewer focus:** RLE `counts` decode-ASCII step (silent JSON failure otherwise); top-K grouping key is `(image_id, category_id)` not global; `git_sha` path tolerance (non-git checkouts must not crash); PNG filename pattern matches spec §7.

---

## Phase 4 — Visualization (`predict/visualize.py`)

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1, 2, 3.
**Depends on:** none (uses PIL only; consumes synthetic dicts).
**Spec:** §7 layout (`visualizations/<stem>.png`), §11.1 (`test_visualize.py`).

**Files:**
- Create: `src/custom_sam_peft/predict/visualize.py`
- Create: `tests/predict/test_visualize.py`

### Task 4a: Tests first

- [ ] **Step P4-1: Write `tests/predict/test_visualize.py`**

| Test name | Intent |
| --- | --- |
| `test_visualize_writes_png` | Synthetic 32×32 RGB image + one synthetic entry with a small `segmentation` RLE → `<output>/visualizations/<stem>.png` exists and opens via PIL. |
| `test_visualize_color_deterministic_per_class` | Same class name → same RGB triple across two invocations (hash-based palette: `hash(class_name) % palette_size`). |
| `test_visualize_color_differs_per_class` | Two different class names → with overwhelming probability, different colors (palette ≥ 8 entries). |
| `test_visualize_score_label_drawn` | The PNG contains visible pixels in the expected text-rect region (assert a small fraction of pixels in that box are non-image-background). Coarse but adequate — exhaustive visual assertions are out of scope. |
| `test_visualize_handles_empty_entries` | An image with zero entries → still writes a copy of the original PNG; no crash. |
| `test_visualize_skips_when_no_segmentation` | An entry with `save_masks=none` (no `segmentation`) → renders bbox-only; no crash. |

### Task 4b: Implementation

- [ ] **Step P4-2: Implement `src/custom_sam_peft/predict/visualize.py`**

Signatures:

```python
PALETTE: tuple[tuple[int,int,int], ...] = (...)   # 16 distinct colors, hand-picked for legibility

def color_for_class(class_name: str) -> tuple[int, int, int]: ...
def render_overlay(image: PIL.Image.Image, entries: list[dict], *, prompts: list[str]) -> PIL.Image.Image: ...
def write_visualization(image_path: Path, entries: list[dict], output_dir: Path, *, prompts: list[str]) -> Path: ...
```

Implementation notes:
- `color_for_class` uses `hash(class_name) % len(PALETTE)` — `hash` is process-stable but not cross-process-stable. **Use `int(hashlib.blake2s(class_name.encode(), digest_size=4).hexdigest(), 16) % len(PALETTE)`** for cross-run determinism (consistent with the spec's blake2s-based image-id scheme in §7.1).
- Mask overlay: `PIL.Image.blend(original, color_layer, alpha=0.4)` masked by the decoded RLE (use `decode_rle_to_uint8` from writers — import within the function to keep writers a lower-tier dep).
- Bbox + score label: `ImageDraw.rectangle` for the box, `ImageDraw.text` for `f"{class_name} {score:.2f}"`. Use the bundled `ImageFont.load_default()` so we don't introduce a font dependency.
- Output: `<output>/visualizations/<stem>.png` (per spec §7 layout block).

### Task 4c: Acceptance gate

- [ ] **Step P4-3: Tests pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_visualize.py -q
```

Expected: green; PNGs decode; color determinism holds.

**Reviewer focus:** color hash is cross-process-stable (not Python's `hash()`); empty-entries edge case writes a copy not nothing; no font dependency leaks; the `category_id → prompts[category_id - 1]` mapping is 1-indexed (spec §5.2).

---

## Phase 5 — Runner (`predict/runner.py`)

**Model/effort:** sonnet / high.
**Parallel:** No. **Depends on:** Phases 1–4 all committed.
**Spec:** §2 (architecture), §6 (config layering), §9 (forward loop), §10 (errors), §12 (model-batch parallelism).

**Files:**
- Create: `src/custom_sam_peft/predict/runner.py`
- Modify: `src/custom_sam_peft/predict/__init__.py` (re-export `run_predict`, `PredictOptions`, `PredictReport`)
- Create: `tests/predict/test_runner_smoke.py`
- Un-skip: `tests/predict/test_config_layering.py` (defined in Phase 2 but gated on `PredictOptions`)
- Create: `tests/predict/test_dry_run.py`

### Task 5a: Dataclasses

- [ ] **Step P5-1: Define `PredictOptions` and `PredictReport`**

In `predict/runner.py`:

```python
@dataclass(frozen=True)
class PredictOptions:
    images: Path
    prompts: str            # raw spec — resolved by parse_prompts inside run_predict
    output: Path
    checkpoint: Path | None
    merge_adapter: bool     # default True (spec §6)
    config: Path | None
    score_threshold: float  # default 0.3
    top_k: int              # default 100
    save_masks: Literal["rle", "png", "none"]   # default "rle"
    visualize: bool         # default False
    device: Literal["auto", "cuda", "cpu"]      # default "auto"
    dtype: Literal["auto", "bfloat16", "float32"]  # default "auto"
    batch_size: int         # default 1
    seed: int               # default 0
    dry_run: bool           # default False
    verbose: bool           # default False

@dataclass(frozen=True)
class PredictReport:
    n_images: int
    n_predictions: int
    elapsed_sec: float
```

Field defaults are owned by the CLI layer in Phase 6 — `PredictOptions` itself has NO defaults, so test construction is explicit.

### Task 5b: Tests first

- [ ] **Step P5-2: Write `tests/predict/test_runner_smoke.py`**

Setup helpers in `tests/predict/conftest.py`:
- `stub_sam_module()` — a `torch.nn.Module` whose forward returns a dict matching `queries_to_coco_results`'s expected shape: `pred_logits` (1, Q, num_classes), `pred_boxes` (1, Q, 4), `pred_masks` (1, Q, H_low, W_low), `presence_logit_dec` (1, Q, num_classes). Exact shapes match `eval/postprocess.py`'s contract — confirm by reading that file before fabricating.
- `synthetic_image_64()` — a `PIL.Image.Image` size 64×64 saved to tmp.

Tests:

| Test name | Intent |
| --- | --- |
| `test_run_predict_smoke_end_to_end_cpu` | One synthetic 64×64 image + two prompts (`"cat","dog"`) → `predictions.json`, `image_id_map.json`, `run.json` all written under tmp output dir; `PredictReport.n_images==1` and `n_predictions==len(predictions)`. Monkeypatches `load_sam31` to return the stub module. |
| `test_run_predict_base_model_only_no_peft_import` | When `checkpoint=None`, `import custom_sam_peft.peft_adapters` did NOT occur on the hot path. Implementation hint: patch `sys.modules` to a sentinel and assert it's untouched after `run_predict`. (Spec §2 explicit guarantee.) |
| `test_run_predict_warmup_runs_one_forward` | The stub module's `forward` is called exactly once for warmup, before the per-image/per-class loop begins. |
| `test_run_predict_vram_hint_not_logged_on_cpu` | On `device="cpu"`, the spec §6 hint string is NOT logged. (The hint is cuda-only.) |
| `test_run_predict_seed_recorded` | `run.json["seed"] == opts.seed`; `torch.initial_seed()` was set. |
| `test_run_predict_top_k_filtering_applied` | Configure the stub module's outputs to yield more than `top_k` instances; assert the final predictions count is `top_k * num_classes` (or fewer if score-threshold filtered first). |
| `test_run_predict_score_threshold_applied` | All stub scores below threshold → `n_predictions == 0`, exit code 0, file written empty. |
| `test_run_predict_save_masks_none` | With `save_masks="none"`, no `segmentation` field in entries; no `masks/` dir. |
| `test_run_predict_save_masks_png` | With `save_masks="png"`, `<output>/masks/*.png` files exist; entries carry `mask_png`. |
| `test_run_predict_visualize` | With `visualize=True`, `<output>/visualizations/*.png` files exist. |
| `test_run_predict_unreadable_image_warns_and_skips` | One image is corrupt → WARN log + continue; `n_images` reports successful images only. |
| `test_run_predict_every_image_fails_exits_1` | Every image is corrupt → exit 1 (spec §10). |

- [ ] **Step P5-3: Write `tests/predict/test_dry_run.py`**

| Test name | Intent |
| --- | --- |
| `test_dry_run_short_circuits_before_model_load` | Monkeypatch `load_sam31` to raise; with `opts.dry_run=True`, `run_predict` exits 0 and does NOT call it. |
| `test_dry_run_prints_first_10_images_all_prompts` | Captured stdout contains the first 10 resolved image paths and every prompt. |
| `test_dry_run_writes_nothing` | Output dir is empty after dry-run. |
| `test_dry_run_prints_resolved_config` | The resolved config block (model name, device, dtype, image_size, normalize) appears in stdout. |

- [ ] **Step P5-4: Un-skip `tests/predict/test_config_layering.py`**

Remove the Phase 2 skip guard. The tests now import `PredictOptions` from `custom_sam_peft.predict.runner`.

### Task 5c: Implementation

- [ ] **Step P5-5: Implement `src/custom_sam_peft/predict/runner.py`**

Top-level structure (mirrors spec §9 verbatim):

```python
def run_predict(opts: PredictOptions) -> PredictReport:
    # 1. resolve images + prompts
    # 2. preflight log
    # 3. dry-run short-circuit (BEFORE load_sam31)
    # 4. load_sam31 + .to(device, dtype)
    # 5. adapter load + optional merge
    # 6. build transforms
    # 7. VRAM hint (cuda + bs=1 + free > 12 GB)
    # 8. warmup
    # 9. forward loop (per-image postprocess, see spec §9)
    # 10. save_masks branch + visualize branch
    # 11. write predictions.json, image_id_map.json, run.json
    # 12. return PredictReport
```

Implementation notes (cross-referenced to spec):
- **Step 3 short-circuit (spec §9 item 3, §10):** must occur BEFORE `load_sam31`. Stdout-only; no file I/O.
- **Step 5 (spec §6):** `kind = detect_adapter_kind(opts.checkpoint)`; dispatch to `load_qlora` / `load_lora`; if `opts.merge_adapter`, call `maybe_merge_adapter(model, merge=True)`. Note: the spec EXPLICITLY allows merge on the QLoRA path (the user opts in).
- **Step 7 VRAM hint (spec §6, §11.2 GPU test, §12):** only when `device=="cuda"`, `batch_size==1`, and `torch.cuda.mem_get_info()[0] > 12 * 1024**3`. Single INFO line. No auto-bump (locked-in per the user's carve-out).
- **Step 8 warmup:** one forward on `torch.zeros(1, 3, image_size, image_size, device=device, dtype=dtype)`; discard outputs. Catches lazy-init OOMs early.
- **Step 9 forward loop (spec §9 + §12):** per-image postprocess is **non-negotiable** (`queries_to_coco_results` hard-asserts `pred_logits.shape[0] == 1`). The runner may stack a B-image tensor for one model call and slice the outputs, OR call the model B times — either is conforming, but postprocess is always per-image. **v1 calls `queries_to_coco_results` per-image even when `--batch-size > 1`.**
- **Inverse transform (spec §7.1 bbox row):** `build_eval_transforms` does longest-edge resize + top-left pad. Predict must invert: `(x, y, w, h) → (x / scale, y / scale, w / scale, h / scale)` where `scale = image_size / max(original_h, original_w)`. The pad is at top-left, so no offset subtraction is needed. Mask emission must use original H×W (passed as `originals=(H, W)` to `queries_to_coco_results` per spec §9).
- **OOM hint (spec §10):** wrap the forward in a `try/except RuntimeError`; if `"out of memory" in str(e).lower()`, log the hint string `"OOM: consider --no-merge-adapter (QLoRA), --batch-size 1, or --device cpu"` then re-raise.
- **Seed (spec §12 determinism row):** `torch.manual_seed(opts.seed); np.random.seed(opts.seed); random.seed(opts.seed)` at top of function.
- **Verbose log (spec §9 item 9):** when `opts.verbose`, log `f"image {i+1}/{n} {path.name} ({latency_ms:.1f} ms)"` after each image.
- **Hot-path import discipline (spec §2):** `from custom_sam_peft.peft_adapters import ...` happens INSIDE the `if opts.checkpoint is not None:` block, not at module top.

- [ ] **Step P5-6: Update `src/custom_sam_peft/predict/__init__.py` to re-export**

```python
from custom_sam_peft.predict.runner import PredictOptions, PredictReport, run_predict

__all__ = ["PredictOptions", "PredictReport", "run_predict"]
```

### Task 5d: Acceptance gate

- [ ] **Step P5-7: Tests pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_runner_smoke.py tests/predict/test_dry_run.py tests/predict/test_config_layering.py -q
```

Expected: green.

**Reviewer focus:**
1. Per-image postprocess assert is honored — `queries_to_coco_results` is called with B==1 every time. Spec §9, §12, §13.
2. Dry-run short-circuit is BEFORE `load_sam31` (spec §10 exit-code table, last row).
3. Adapter import is lazy (spec §2 — no `peft_adapters` import when `checkpoint=None`).
4. VRAM hint is cuda-only and gated on bs==1 (spec §6).
5. Inverse longest-edge transform math matches `build_eval_transforms` exactly (Phase 7 parity test guards this, but reviewer should spot-check the formula here too).
6. OOM hint log fires before re-raise (spec §10).

---

## Phase 6 — CLI shell (`cli/predict_cmd.py`) + main wiring

**Model/effort:** sonnet / high.
**Parallel:** Can run alongside Phase 7 (different files).
**Depends on:** Phase 5 committed.
**Spec:** §3 (boundary rules), §8 (CLI surface), §10 (BadParameter exits).

**Files:**
- Create: `src/custom_sam_peft/cli/predict_cmd.py`
- Modify: `src/custom_sam_peft/cli/main.py` (+1 import, +1 register line)
- Create: `tests/predict/test_cli_predict.py`

### Task 6a: Tests first

- [ ] **Step P6-1: Write `tests/predict/test_cli_predict.py`**

Use Typer's `CliRunner` (`from typer.testing import CliRunner`).

| Test name | Intent |
| --- | --- |
| `test_predict_help_exit_zero` | `csp predict --help` exits 0 and lists every flag from spec §8. |
| `test_predict_argv_round_trip_to_options` | Monkeypatch `run_predict` to capture its `opts` arg; assert the dataclass matches the CLI inputs (resolved types). |
| `test_score_threshold_out_of_range_rejected` | `--score-threshold 1.5` → exit 2, message names `score-threshold`. |
| `test_score_threshold_negative_rejected` | `--score-threshold -0.1` → exit 2. |
| `test_top_k_zero_rejected` | `--top-k 0` → exit 2. |
| `test_batch_size_zero_rejected` | `--batch-size 0` → exit 2. |
| `test_save_masks_bad_choice_rejected` | `--save-masks foo` → exit 2 (Typer choice validation). |
| `test_device_bad_choice_rejected` | `--device gpu` → exit 2 (must be auto/cuda/cpu). |
| `test_dtype_bad_choice_rejected` | `--dtype fp16` → exit 2 (must be auto/bfloat16/float32). |
| `test_checkpoint_missing_path_rejected` | `--checkpoint /nonexistent/path` → exit 2. |
| `test_checkpoint_lacks_adapter_config_rejected` | A real dir with no `adapter_config.json` → exit 2 (spec §10 third row). |
| `test_merge_adapter_default_on` | No `--no-merge-adapter` flag → captured `opts.merge_adapter == True` (spec §6 — default ON). |
| `test_no_merge_adapter_flips_off` | `--no-merge-adapter` → `opts.merge_adapter == False`. |
| `test_dry_run_short_circuits_at_cli_level` | `--dry-run` → captured `opts.dry_run == True` and `run_predict` is monkeypatched to assert it sees the flag; no model load occurs. |
| `test_zero_images_resolved_propagates_exit_2` | `--images <empty-tmpdir>` → exit 2 (BadParameter from `resolve_images` propagates per spec §10 first row). |
| `test_empty_prompts_propagates_exit_2` | `--prompts ""` → exit 2 (spec §10 second row). |
| `test_summary_line_on_success` | After a successful (stubbed) `run_predict`, stdout contains the `PredictReport` summary one-liner. |

### Task 6b: Implementation

- [ ] **Step P6-2: Implement `src/custom_sam_peft/cli/predict_cmd.py`**

Body target: ≤ 40 lines (spec §3 — relaxed from the train/eval 30-line cap because predict has more flags). Pattern mirrors `cli/train_cmd.py` / `cli/eval_cmd.py`. Signature:

```python
def predict(
    images: Path = typer.Option(..., "--images", help="Dir / glob / manifest / single file."),
    prompts: str = typer.Option(..., "--prompts", help="Comma-separated string or path to one-per-line file."),
    output: Path = typer.Option(..., "--output", help="Output directory (created if missing)."),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", callback=_validate_checkpoint),
    merge_adapter: bool = typer.Option(True, "--merge-adapter/--no-merge-adapter"),
    config: Path | None = typer.Option(None, "--config"),
    score_threshold: float = typer.Option(0.3, "--score-threshold", callback=_validate_unit_interval),
    top_k: int = typer.Option(100, "--top-k", callback=_validate_positive_int),
    save_masks: str = typer.Option("rle", "--save-masks", click_type=click.Choice(["rle","png","none"])),
    visualize: bool = typer.Option(False, "--visualize"),
    device: str = typer.Option("auto", "--device", click_type=click.Choice(["auto","cuda","cpu"])),
    dtype: str = typer.Option("auto", "--dtype", click_type=click.Choice(["auto","bfloat16","float32"])),
    batch_size: int = typer.Option(1, "--batch-size", callback=_validate_positive_int),
    seed: int = typer.Option(0, "--seed"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None: ...
```

Body: configure logging (`_configure_logging(verbose)` from `cli/_logging.py`), build `PredictOptions(...)`, call `run_predict(opts)`, `rich.print` the one-line summary.

Validation callbacks:
- `_validate_unit_interval`: ensure `0.0 <= x <= 1.0`, else `raise typer.BadParameter(f"--score-threshold must be in [0.0, 1.0], got {x}")`.
- `_validate_positive_int`: ensure `x >= 1`.
- `_validate_checkpoint`: ensure `path.exists() and (path / "adapter_config.json").is_file()`. (Spec §10.)

- [ ] **Step P6-3: Wire into `src/custom_sam_peft/cli/main.py`**

Two-line change:

```python
from custom_sam_peft.cli import predict_cmd   # add to the existing import block

app.command("predict", help="Run inference on images with optional adapter.")(predict_cmd.predict)
```

Position the register line immediately after the `eval` registration (logical grouping: train → eval → predict → export → init → doctor → run).

- [ ] **Step P6-3a: Add `csp` script alias to `pyproject.toml`**

Under the existing `[project.scripts]` block, add a sibling line so the shorter `csp` invocation used throughout the spec and README example resolves to the same Typer app:

```toml
[project.scripts]
custom-sam-peft = "custom_sam_peft.cli.main:app"
csp = "custom_sam_peft.cli.main:app"
```

Run `uv sync` after the change so the alias is installed into the venv (`uv run csp --help` should exit 0 from this point on). Do not remove the existing `custom-sam-peft` entry — keep both. This is the only `pyproject.toml` edit in the PR.

### Task 6c: Acceptance gate

- [ ] **Step P6-4: Tests pass**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_cli_predict.py tests/unit/test_cli.py -q
```

Expected: green. The existing `tests/unit/test_cli.py` exercises the global CLI surface; it should still pass since predict is purely additive.

- [ ] **Step P6-5: Manual smoke**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run csp predict --help
```

Expected: exits 0, every flag from spec §8 is listed.

**Reviewer focus:** Body length ≤ 40 lines (spec §3); `_validate_*` callbacks raise `typer.BadParameter` (exit code 2 path); `--merge-adapter/--no-merge-adapter` default is ON; click `Choice` enums match spec §8 exactly.

---

## Phase 7 — Preprocessing parity test

**Model/effort:** sonnet / high.
**Parallel:** Can run alongside Phase 6 (different files).
**Depends on:** Phase 5 committed.
**Spec:** §11.1 (parity test), §16 (#69 guards `resolve_normalization` regression).

**Files:**
- Create: `tests/predict/test_preprocessing_parity.py`

### Task 7a: Test

- [ ] **Step P7-1: Write `tests/predict/test_preprocessing_parity.py`**

| Test name | Intent |
| --- | --- |
| `test_predict_transform_matches_build_eval_transforms_byte_identical` | Construct a synthetic 257×129 RGB image. Apply `build_eval_transforms(image_size=1024, model_name="facebook/sam3.1", normalize=resolve_normalization(...))` directly (eval path). Apply the predict runner's transform path on the same image. Assert `torch.equal(tensor_predict, tensor_eval)` — byte-identical, not just close. |
| `test_predict_normalize_uses_resolved_default_when_no_config` | When `opts.config is None`, `resolve_normalization("facebook/sam3.1", default=NormalizeConfig())` is consulted. The resulting mean/std tensors must match what eval uses on the same model. |

Implementation note: this test does NOT load a model; it only exercises the transform builder. CPU-only.

### Task 7b: Acceptance gate

- [ ] **Step P7-2: Test passes**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_preprocessing_parity.py -q
```

Expected: green.

**Reviewer focus:** `torch.equal` (exact), not `torch.allclose`. Any drift here means predict's outputs will silently diverge from eval's on the same image. This is the test #69 wanted.

---

## Phase 8 — GPU integration tests

**Model/effort:** sonnet / high.
**Parallel:** No. **Depends on:** Phase 6 committed (Typer command must be live).
**Spec:** §11.2, §16 (#74 v1.0 gate).

**Files:**
- Create: `tests/predict/test_gpu_predict.py`

### Task 8a: Tests

- [ ] **Step P8-1: Write `tests/predict/test_gpu_predict.py`**

All tests are decorated with `@pytest.mark.gpu`; they skip cleanly under `pytest -m 'not gpu'` (default). They run in the cuda GHA matrix per `2026-05-19-gpu-test-policy-design.md`.

| Test name | Intent |
| --- | --- |
| `test_predict_base_model_cuda` | Real `facebook/sam3.1` load + warmup + one synthetic 1024×1024 image + two prompts → `predictions.json` written; each `segmentation.counts` decodes via `pycocotools.mask.decode`. |
| `test_predict_lora_adapter_cuda` | Tiny 50-step LoRA fixture (reuse the artifact pattern from `tests/gpu/test_real_train_overfits.py`) → `csp predict` over one image; exercises `merge_lora` on a real PEFT wrapper. |
| `test_predict_qlora_no_merge_cuda` | Real bitsandbytes 4-bit load + `--no-merge-adapter`. This is the only test where the 4-bit dequant-during-merge VRAM regression mode matters (per spec §6 / §11.2). |
| `test_predict_vram_hint_log` | On cuda with sufficient free VRAM and default `--batch-size 1`, the INFO hint string `"free VRAM is >12 GB"` appears in caplog. Skip when `torch.cuda.mem_get_info()[0] <= 12 * 1024**3`. |

Per the project memory rule (`feedback_gpu_vs_cpu_testing.md`): **only** these real-only failure modes belong on GPU. Anything else (top-K math, RLE round-trip, visualization, etc.) lives in the CPU tests (Phases 3–5).

### Task 8b: Acceptance gate

- [ ] **Step P8-2: Tests pass under `-m gpu` (local + CI)**

Local (if GPU available):
```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_gpu_predict.py -m gpu -q
```

CI: triggered automatically by the `cuda` GHA matrix per `2026-05-19-gpu-test-policy-design.md`. Confirm green in the PR's check suite before requesting Phase 9.

- [ ] **Step P8-3: Tests skip cleanly under default**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests/predict/test_gpu_predict.py -m 'not gpu' -q
```

Expected: 0 selected (or all 4 skipped with reason "requires gpu").

**Reviewer focus:** `merge_and_unload` semantics on the QLoRA path — confirm forward still works after merge=True on a bnb 4-bit base (covered by `test_predict_lora_adapter_cuda` exercising the merge path; the QLoRA `--no-merge-adapter` test sidesteps the dequant). VRAM-hint test must skip on low-memory cuda devices, not fail.

---

## Phase 9 — Docs / README

**Model/effort:** haiku / medium.
**Parallel:** No. **Depends on:** Phase 8 green.
**Spec:** §8 (CLI flags), §15 (exit criteria).

**Files:**
- Modify: `README.md` (additive subsection)

### Task 9a: Patch README

- [ ] **Step P9-1: Add "Run inference on your images" subsection**

Add a new H4 subsection immediately after the existing `### CLI` H3 (placement mirrors how the spec §15 phrases "zero-config invocation"). Content:

````markdown
#### Run inference on your images

After installing the package, point `csp predict` at a directory of images and pass class prompts:

```bash
uv run csp predict \
  --images path/to/images/ \
  --prompts "cat,dog,person" \
  --output out/
```

This produces `out/predictions.json` (COCO-flat), `out/image_id_map.json` (id → source path), and `out/run.json` (reproducibility metadata). Pass `--checkpoint path/to/adapter/` to apply a LoRA or QLoRA adapter (auto-detected); add `--visualize` to write per-image overlays. See `csp predict --help` for every flag.
````

- [ ] **Step P9-2: Spot-check no unrelated docs touched**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && git diff README.md | head -40
```

Expected: only the additive subsection appears in the diff. Per the user CLAUDE.md scope rule, do NOT touch unrelated sections.

- [ ] **Step P9-3: Lint the markdown**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && npx --yes markdownlint-cli2 README.md
```

Expected: clean (or at most the same warnings present before the patch — do not introduce new ones).

### Task 9b: Acceptance gate

- [ ] **Step P9-4: Confirm flags shown match `csp predict --help`**

The README snippet must use only flags that appear in `csp predict --help` output. Run:

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run csp predict --help | grep -E -- '--images|--prompts|--output|--checkpoint|--visualize'
```

Expected: all five flags present.

**Reviewer focus:** Stays within the README subsection added by this PR; no edits to unrelated sections (per CLAUDE.md scope rule); no claims the CLI doesn't honor.

---

## Phase 10 — Lint / typecheck / coverage gate

**Model/effort:** sonnet / high (reviewer-pass).
**Parallel:** No. **Depends on:** Phases 1–9 complete.
**Spec:** §11.3 (coverage targets), §15 (exit criteria — last three bullets).

### Task 10a: Lint + format

- [ ] **Step P10-1: Run `ruff check` and `ruff format`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict
uv run ruff format src/custom_sam_peft/predict src/custom_sam_peft/cli/predict_cmd.py tests/predict
uv run ruff check src/custom_sam_peft/predict src/custom_sam_peft/cli/predict_cmd.py tests/predict
```

Expected: both exit 0. Fix any complaints directly (reviewer pass).

### Task 10b: Typecheck

- [ ] **Step P10-2: Run `mypy --strict`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run mypy --strict src/custom_sam_peft
```

Expected: exit 0. Strict mode flags missing return types, missing arg types, etc. — fix at source rather than `# type: ignore`-ing.

### Task 10c: Full CPU test suite + coverage

- [ ] **Step P10-3: Run the full CPU suite**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests -m 'not gpu' -q
```

Expected: all green; no flakes; the `tests/predict/` set runs in well under a minute total.

- [ ] **Step P10-4: Confirm coverage hits per-module targets (spec §11.3)**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run pytest tests -m 'not gpu' --cov=src/custom_sam_peft/predict --cov-report=term-missing
```

Expected coverage:

| Module | Target | Required |
| --- | --- | --- |
| `predict/runner.py` | 90%+ | yes |
| `predict/inputs.py` | 90%+ | yes |
| `predict/writers.py` | 90%+ | yes |
| `predict/adapter_load.py` | 90%+ | yes |
| `predict/visualize.py` | 70%+ | yes |

Repo-wide 80% gate (from `pyproject.toml`) must still pass. If `visualize.py` is below 70%, add a targeted test before claiming green; do NOT lower the gate.

### Task 10d: Final smoke

- [ ] **Step P10-5: Manual `--help` smoke**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/feat-cli-predict && uv run csp predict --help && uv run csp --help
```

Expected: both exit 0; `csp --help` lists `predict` as a subcommand.

**Reviewer focus:** No `# type: ignore` shims introduced (strict mypy must pass natively); coverage targets met without lowering thresholds; `csp predict` appears in the top-level `csp --help` table.

---

## Definition of done

All items below must be checked before the PR can be marked ready for review:

- [ ] `src/custom_sam_peft/predict/` exists with `__init__.py`, `runner.py`, `inputs.py`, `adapter_load.py`, `writers.py`, `visualize.py`.
- [ ] `src/custom_sam_peft/cli/predict_cmd.py` exists; body ≤ 40 lines.
- [ ] `src/custom_sam_peft/cli/main.py` registers `predict` exactly once.
- [ ] `pyproject.toml` `[project.scripts]` has the new `csp` alias alongside the existing `custom-sam-peft` entry; `uv run csp --help` and `uv run csp predict --help` both exit 0.
- [ ] `tests/predict/` exists with all 11 CPU test files + 1 GPU test file from the spec §14 file layout block.
- [ ] `csp predict --help` exits 0 and lists every flag from spec §8.
- [ ] `csp predict --dry-run` short-circuits before model load and writes nothing.
- [ ] `pytest -m 'not gpu' tests/predict` passes locally with coverage meeting spec §11.3 targets.
- [ ] `pytest -m gpu tests/predict/test_gpu_predict.py` passes in the cuda GHA matrix.
- [ ] `ruff check`, `ruff format --check`, `mypy --strict` all clean.
- [ ] README has the "Run inference on your images" subsection; no other docs touched.
- [ ] `src/custom_sam_peft/eval/`, `src/custom_sam_peft/train/`, `src/custom_sam_peft/models/`, `src/custom_sam_peft/peft_adapters/`, `src/custom_sam_peft/cli/run_cmd.py` all UNCHANGED.
- [ ] PR body links spec, plan, and issue #74; lists the locked-in choices from spec §1.2 and §6.

---

## Test fixtures needed

Consolidated list (created across Phases 1, 2, 3, 4 in their respective conftest/fixtures dirs):

| Fixture | Phase that creates it | Used by |
| --- | --- | --- |
| Synthetic 32×32 RGB image (PIL) | Phase 4 conftest | `test_visualize.py` |
| Synthetic 64×64 RGB image (PIL, saved to tmp) | Phase 5 conftest | `test_runner_smoke.py`, `test_dry_run.py`, `test_preprocessing_parity.py` |
| Unsorted directory of mixed-ext files | Phase 1 conftest | `test_inputs.py` |
| Corrupt JPEG (zero-byte `.jpg`) | Phase 5 conftest | `test_runner_smoke.py` unreadable-image tests |
| `.txt` manifest with comments + blanks | Phase 1 (inline) | `test_inputs.py` |
| `.json` manifest with valid list | Phase 1 (inline) | `test_inputs.py` |
| `tests/predict/fixtures/lora_adapter/` (config-only) | Phase 2 | `test_adapter_detect.py`, `test_config_layering.py` |
| `tests/predict/fixtures/qlora_adapter/` (config + qlora sentinel) | Phase 2 | `test_adapter_detect.py` |
| `tests/predict/fixtures/bad_adapter/` (empty dir) | Phase 2 | `test_adapter_detect.py` |
| Stub `nn.Module` whose forward returns `pred_logits`/`pred_boxes`/`pred_masks`/`presence_logit_dec` matching `queries_to_coco_results`'s contract | Phase 5 conftest | `test_runner_smoke.py`, `test_dry_run.py` |
| Synthetic outputs dict (no model) shaped like `queries_to_coco_results` returns | Phase 3, 4 (inline) | `test_writers.py`, `test_visualize.py` |

The stub `nn.Module`'s output shapes MUST be verified against `src/custom_sam_peft/eval/postprocess.py::queries_to_coco_results` before the fixture is written — read that function first. Spec §4 forward signature: `model(img_batch, [TextPrompts(classes=[name])]*B, box_hints=None)`.

---

## Risks and open questions

| # | Risk | Mitigation |
| --- | --- | --- |
| 1 | `model.to(device, dtype=...)` may not recursively move SAM 3.1's text/box/mask heads on the wrapper returned by `load_sam31`. Eval uses bf16; predict should match. | Verify via `models/sam3.py::load_sam31` semantics before Phase 5; if `.to(...)` is insufficient, mirror the eval helper's setup verbatim. Caught in Phase 8 `test_predict_base_model_cuda`. |
| 2 | `merge_lora` on a QLoRA-quantized base may return a non-4-bit module (peft `merge_and_unload` dequantizes). Forward correctness post-merge is unproven for this codebase. | Phase 8 `test_predict_qlora_no_merge_cuda` tests the opt-out path; the spec §6 wording ("the user makes the call") means merge-on-QLoRA is an explicit user opt-in — predict does NOT auto-disable. Add a release note if Phase 8 surfaces a regression. |
| 3 | Visualization color palette: `hash()` is process-stable but NOT cross-process-stable. | Use `blake2s(class_name, digest_size=4)` for determinism. Plan calls this out in Phase 4. Verify legibility against the 16-color palette manually before Phase 9. |
| 4 | `csp predict` slot collisions in `csp --help` (no existing `predict` command, but help text formatting may regress). | Phase 6 acceptance smoke runs `csp --help`; Phase 10 final smoke runs both `csp --help` and `csp predict --help`. |
| 5 | The inverse longest-edge transform formula (Phase 5 implementation note) must match `build_eval_transforms` exactly. | Phase 7 parity test guards this byte-identically. |
| 6 | `git rev-parse --short HEAD` from inside a worktree may behave differently from a regular checkout. | Use `subprocess.run(..., cwd=Path(__file__).resolve().parents[2], check=False, capture_output=True)`; on non-zero exit, set `git_sha=None`. Tested by `test_run_json_git_sha_optional`. |
| 7 | `torch.cuda.mem_get_info()` is unavailable on some older CUDA driver setups. | Guard the call with `try/except RuntimeError`; on failure, skip the VRAM hint silently. CPU code path is unaffected. |

No open questions require escalation back to the user. All design ambiguities surfaced during brainstorming were resolved in the spec (see spec §6, §9, §12).

---

## Final-state acceptance criteria (PR-ready)

Mirrors spec §15, restated as PR-ready criteria:

1. **CLI exit-zero help.** `csp predict --help` exits 0; every flag from spec §8 appears.
2. **Base-model path.** `csp predict --images <dir> --prompts cat,dog --output out/` (no `--checkpoint`) produces `out/predictions.json`, `out/image_id_map.json`, `out/run.json` with the schemas in spec §7.
3. **Adapter paths.** `csp predict --checkpoint <run_dir>/adapter ...` works for both LoRA and QLoRA dirs without code changes.
4. **Dry-run.** `csp predict --dry-run ...` prints the resolved inputs/prompts/config and writes nothing; no model is loaded.
5. **PNG masks.** `--save-masks=png` writes PNGs under `out/masks/` whose dimensions match the original images.
6. **Visualizations.** `--visualize` writes per-image overlays under `out/visualizations/`.
7. **CPU test gate.** `pytest -m 'not gpu' tests/predict` is green with coverage meeting per-module targets (spec §11.3).
8. **GPU test gate.** `pytest -m gpu tests/predict/test_gpu_predict.py` is green in the cuda GHA matrix.
9. **Lint + typecheck clean.** `ruff check`, `ruff format --check`, `mypy --strict` all exit 0.
10. **README documented.** A short "Run inference on your images" subsection exists; no unrelated docs touched.
11. **No churn to eval/train/models/peft_adapters/run_cmd.** Diff is confined to the file map at the top of this plan.

---

## Self-review

**1. Spec coverage:** Every file in spec §14 maps to at least one plan phase:
- `cli/main.py` → Phase 6, Step P6-3.
- `cli/predict_cmd.py` → Phase 6.
- `predict/__init__.py` → Phase 1 (stub), Phase 5 (re-exports).
- `predict/runner.py` → Phase 5.
- `predict/inputs.py` → Phase 1.
- `predict/adapter_load.py` → Phase 2.
- `predict/writers.py` → Phase 3.
- `predict/visualize.py` → Phase 4.
- `tests/predict/test_inputs.py` / `test_prompts.py` → Phase 1.
- `tests/predict/test_adapter_detect.py` / `test_config_layering.py` → Phase 2 (config_layering un-skipped in Phase 5).
- `tests/predict/test_writers.py` → Phase 3.
- `tests/predict/test_visualize.py` → Phase 4.
- `tests/predict/test_runner_smoke.py` / `test_dry_run.py` → Phase 5.
- `tests/predict/test_cli_predict.py` → Phase 6.
- `tests/predict/test_preprocessing_parity.py` → Phase 7.
- `tests/predict/test_gpu_predict.py` → Phase 8.

**2. Locked-in choices preserved verbatim:**
- COCO-flat `predictions.json` + `image_id_map.json` sidecar → spec §7.1, §7.2; Phase 3 tests assert schemas.
- `--batch-size` default 1 + INFO hint on cuda free > 12 GB + NO auto-bump → spec §6, §12; Phase 5 Step P5-5 implementation notes; Phase 8 `test_predict_vram_hint_log`.
- Adapter pins `model.name` with WARN on disagreement → spec §6; Phase 2 `test_adapter_pin_overrides_config_with_warn`.
- `--merge-adapter` default ON → spec §6; Phase 6 `test_merge_adapter_default_on`.
- `--save-masks {rle,png,none}` default `rle` → spec §8; Phase 3 tests; Phase 6 choice validation.
- `--visualize` default off → spec §8; Phase 6 default verified in `test_predict_argv_round_trip_to_options`.
- `--dry-run` short-circuits BEFORE model load → spec §9 item 3, §10; Phase 5 `test_dry_run_short_circuits_before_model_load`.
- Postprocess is per-image (asserts batch=1); v1 calls `queries_to_coco_results` per-image even when `--batch-size > 1` → spec §9, §12, §13; Phase 5 Step P5-5 implementation notes; reviewer-focus row 1.

**3. Tests-first ordering:** Every phase has tests authored BEFORE implementation. Sequence is: tests file written → implementation file written → acceptance gate runs the tests.

**4. Phase ordering:** Phases 1–4 are explicitly file-disjoint (each owns its own `predict/*.py` and test file). Phase 5 has the dependency on 1–4 stated. Phase 6 depends on 5. Phase 7 can run in parallel with 6 (different files). Phase 8 depends on 6. Phase 9 depends on 8. Phase 10 is last. The dependency graph is drawn explicitly under "Parallelization opportunities".

**5. Reviewer focus rows:** Every phase has a "Reviewer focus" one-liner naming the specific scrutiny target. Phases without a natural reviewer angle (Phase 7 — pure parity) still call out the load-bearing assertion (`torch.equal`, not `allclose`).

**6. Placeholder scan:** No "TBD", "TODO later", or "fill in" language. The only deferred decisions are spec §13 follow-ups (auto-batch, eval-share refactor, server mode, etc.) — explicitly out of scope per spec §1.3 and §13. Phase 8's GPU LoRA fixture references a "tiny 50-step LoRA fixture (reuse the artifact pattern from `tests/gpu/test_real_train_overfits.py`)" — that artifact pattern is established in the repo; the implementer follows the same convention.

**7. Scope discipline:** Diff is confined to the file map. Spec §1.3 out-of-scope items (server mode, video, auto-batch, `torch.compile`, eval-share refactor, `csp run` integration, metrics inside predict) are NOT in any phase.
