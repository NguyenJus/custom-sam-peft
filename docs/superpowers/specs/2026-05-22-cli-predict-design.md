# spec/cli-predict — `csp predict` Inference CLI Design

**Status:** Draft (2026-05-22)
**Tracking:** [#74](https://github.com/NguyenJus/custom-sam-peft/issues/74) — *Add `csp predict` for offline inference*
**Scope:** Add a new `csp predict` command for offline inference over arbitrary image inputs (directory, glob, single file, or manifest) using either the base SAM 3.1 model or a trained LoRA / QLoRA adapter. Produces COCO-flat `predictions.json` plus an `image_id_map.json` sidecar and a reproducibility `run.json`, with optional RLE or PNG mask emission and per-image overlay visualizations. Implements the full feature as scoped in #74; no v1 cuts.

**Builds on:**
[`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) (command-shell + runner split — `cli/predict_cmd.py` is a thin Typer shell over a `predict/runner.py` library function, mirroring the train/eval/export pattern);
[`2026-05-17-eval-design.md`](2026-05-17-eval-design.md) (forward-loop signature template — `predict` reuses `eval.postprocess.queries_to_coco_results` and the `model(img_batch, [TextPrompts(classes=[name])]*B, box_hints=None)` call shape);
[`2026-05-17-peft-lora-design.md`](2026-05-17-peft-lora-design.md) / [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md) (adapter restore + merge semantics);
[`2026-05-19-gpu-test-policy-design.md`](2026-05-19-gpu-test-policy-design.md) (GPU-test gating used by the `tests/predict/test_gpu_predict.py` cases below).

---

## 1. Goals & v1 Scope

### 1.1 Motivation

`csp train` produces an adapter; `csp eval` measures its quality against a labeled COCO split. Neither answers the obvious next question: *"give me predictions on these images."* Today the only way to run inference outside training is to write a notebook that reproduces the eval forward loop. Issue #74 calls this out as a v1.0 gate (it is referenced from #70 as one of the v1.0 release criteria). `csp predict` closes the gap with the same configuration surface and the same on-disk output shape (COCO-flat) the eval pipeline already speaks.

### 1.2 In scope

| Deliverable | Where |
| --- | --- |
| Typer command shell | `src/custom_sam_peft/cli/predict_cmd.py` (NEW) |
| `app.command("predict", help=...)` registration | `src/custom_sam_peft/cli/main.py` (+1 line) |
| Runner / library entry point | `src/custom_sam_peft/predict/runner.py` (NEW) |
| Input + prompt resolution | `src/custom_sam_peft/predict/inputs.py` (NEW) |
| Adapter detection + load | `src/custom_sam_peft/predict/adapter_load.py` (NEW) |
| Output writers (predictions / id-map / run / PNG masks) | `src/custom_sam_peft/predict/writers.py` (NEW) |
| Per-image overlay rendering | `src/custom_sam_peft/predict/visualize.py` (NEW) |
| CPU unit + smoke tests | `tests/predict/` (NEW) |
| GPU integration tests | `tests/predict/test_gpu_predict.py` (NEW, `@pytest.mark.gpu`) |
| `csp` script alias | `pyproject.toml` `[project.scripts]` (+1 line: `csp = "custom_sam_peft.cli.main:app"`). All CLI examples in this spec assume `csp` is installed; this row is the change that makes it so. |

### 1.3 Out of scope

| Item | Reason / follow-up |
| --- | --- |
| Server / HTTP API mode | Different surface (auth, request lifecycle, multi-tenant). Tracked separately. |
| Streaming / video inference | Frame-extraction + temporal smoothing is a distinct design. |
| Auto-bumping `--batch-size` from VRAM heuristics | #74 mentioned this aspirationally; explicitly carved out by the user into a separate issue. Predict only emits a one-line INFO hint at default `--batch-size 1` when free VRAM > 12 GB on cuda. |
| `torch.compile`, TensorRT, additional quantization paths | Beyond what QLoRA adapters already provide. |
| Computing metrics (mAP, IoU, …) | That is `csp eval`'s job; predict has no labels. |
| New training-time changes | None. |
| Refactoring `eval/` to share an inference helper with predict | Deferred — current PR keeps `eval/` untouched. Tracked as future cleanup. |
| Adding `predict` to `cli/run_cmd.py`'s train+eval+export bundle | `csp run` keeps its current scope; predict is a standalone command. |

---

## 2. Architectural Approach

`predict` is a new top-level package next to `train/` and `eval/`. It reuses every existing helper that already encodes the right behavior — transforms, normalization resolution, model load, adapter load, and the queries→COCO postprocessor — so the runner is a forward loop plus output plumbing, not a re-implementation of the inference path.

Key constraints (locked in during brainstorming):

> - **`predict_cmd.py` is a thin Typer shell.** It builds a `PredictOptions` frozen dataclass from CLI flags and calls `run_predict(opts) -> PredictReport`. The runner has no Typer dependency and is independently usable from notebooks and tests.
> - **No churn to `eval/`.** Predict consumes `eval.postprocess.queries_to_coco_results` and the `data.transforms` helpers as-is. Eval-vs-predict drift is a known follow-up, deferred.
> - **Output shape is COCO-flat.** The predictions list is the same shape `Evaluator` writes for `--save-predictions`, with two predict-only sidecars (`image_id_map.json`, `run.json`) and one optional per-entry field (`mask_png`).
> - **Base-model-only is a first-class path.** Omitting `--checkpoint` skips adapter loading entirely — same forward loop, same output, no `peft_adapters` import on the hot path.
> - **`predict` never auto-bumps batch size.** On cuda with `torch.cuda.mem_get_info()` free > 12 GB at the default `--batch-size 1`, the runner logs a one-line INFO hint nudging the user to a higher batch size. Auto-bumping is explicitly out of scope.

---

## 3. Module Layout

```text
src/custom_sam_peft/
├── cli/
│   ├── main.py                # CHANGED — +1 line: app.command("predict", help=...)(predict_cmd.predict)
│   └── predict_cmd.py         # NEW — Typer shell; builds PredictOptions; translates exceptions
└── predict/
    ├── __init__.py            # NEW — re-exports run_predict, PredictOptions, PredictReport
    ├── runner.py              # NEW — run_predict(opts: PredictOptions) -> PredictReport
    ├── inputs.py              # NEW — resolve_images + parse_prompts
    ├── adapter_load.py        # NEW — detect_adapter_kind + load (LoRA or QLoRA); merge toggle
    ├── writers.py             # NEW — predictions.json, image_id_map.json, run.json, RLE/PNG mask emit
    └── visualize.py           # NEW — per-image overlay PNGs

tests/predict/                 # NEW — unit + smoke tests (CPU)
└── test_gpu_predict.py        # NEW — @pytest.mark.gpu integration tests
```

Boundary rules (carried from `2026-05-18-cli-design.md` §3):

- `predict_cmd.py` contains only: flag parsing, building `PredictOptions`, calling `run_predict`, formatting output, translating library exceptions to exit codes. Target ≤ 40 lines body (predict has more flags than train, so the 30-line cap from the CLI spec is relaxed).
- `predict/runner.py` is the library seam (notebooks / future drivers call this directly). It takes `PredictOptions` in; it does not parse argv or know about Typer.

---

## 4. Existing helpers reused (do not duplicate)

| Helper | File | Used by predict for |
| --- | --- | --- |
| `build_eval_transforms` | `data/transforms.py` | Resize + pad + normalize; called with `bboxes=[], class_labels=[]` |
| `resolve_normalization` | `data/transforms.py` | Mean/std fallback when `--config` doesn't pin normalize |
| `load_sam31` | `models/sam3.py` | Base model load |
| `load_lora` | `peft_adapters/lora.py` | LoRA adapter restore |
| `merge_lora` | `peft_adapters/lora.py` | `--merge-adapter` (default ON) |
| `load_qlora` | `peft_adapters/qlora.py` | QLoRA adapter restore (detected via `custom_sam_peft_qlora.json` in checkpoint dir) |
| `queries_to_coco_results` | `eval/postprocess.py` | Raw forward output → list of COCO-flat entries (RLE + bbox + score) |
| `_int_image_id` scheme | `eval/evaluator.py` | `blake2s` 8-byte digest of absolute path → stable int id |

Forward signature is identical to eval's:

```python
outputs = model(img_batch.to(device), [TextPrompts(classes=[name])] * B, box_hints=None)
```

---

## 5. Inputs

### 5.1 `resolve_images(spec: str | Path) -> list[Path]`

`spec` may be any of:

| Form | Recognized by | Behavior |
| --- | --- | --- |
| Directory | `Path.is_dir()` | Recursive walk; collect every file with an allowed extension. |
| Glob | `"*"` or `"?"` in the string | `glob.glob(spec, recursive=True)` (so `**/*.jpg` works). |
| Single image file | `Path.is_file()` and ext in allowlist | Return `[Path(spec)]`. |
| Manifest `.txt` | `Path.is_file()` and `.suffix == ".txt"` | One image path per line; comments (`#…`) and blank lines skipped. |
| Manifest `.json` | `Path.is_file()` and `.suffix == ".json"` | Must decode to a JSON list of strings. |

**Allowed image extensions** (case-insensitive): `{.jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff}`.

**Manifest path resolution.** Paths inside a manifest that are not absolute are resolved relative to the *manifest's parent directory*, not `cwd`. This matches the way users typically check manifests into a dataset directory.

**Ordering.** The returned list is sorted lexicographically on the absolute path string. Determinism is required so that `image_id`s (derived from absolute path) line up across runs.

**Mode normalization.** Every resolved image is opened with PIL and `.convert("RGB")`'d before being handed to the transforms. RGBA and palette modes are handled implicitly by `convert`.

**Errors and warnings.**

- Unreadable file (corrupt JPEG, truncated PNG, permission error) → one `WARN` log line, skip, continue.
- Empty result (zero images after filtering) → raise `typer.BadParameter("no images resolved from <spec>")` → exit code 2.

### 5.2 `parse_prompts(spec: str | Path) -> list[str]`

`spec` may be either:

- A **comma-separated string** (e.g., `"cat,dog,person"`).
- A **path to a one-per-line file** (UTF-8). Detection: `Path(spec).is_file()`.

Behavior:

1. Split / read.
2. Strip whitespace from each entry.
3. Drop empty entries.
4. Deduplicate, preserving first-seen order.

Empty result → raise `typer.BadParameter("--prompts must resolve to at least one non-empty class name")` → exit code 2.

The position in the returned list is the prompt's 1-indexed `category_id` in `predictions.json`.

---

## 6. Config layering

Precedence is **CLI flag > `--config` value > builtin default**, with one explicit exception: an adapter's `base_model_name_or_path` PINS `model.name` and warns on disagreement (the adapter was trained against a specific base; using a different one silently is a footgun).

| Field | Source order |
| --- | --- |
| `model.name` | adapter `adapter_config.json:base_model_name_or_path` (when `--checkpoint` given) → `--config.model.name` → `facebook/sam3.1` (builtin default) |
| `image_size` | `--config.data.image_size` → builtin model-native (1024 for SAM 3.1) |
| `normalize` | `resolve_normalization(model_name, --config.data.normalize OR default NormalizeConfig)` |
| `device` | `--device` (default `auto` → `cuda` if `torch.cuda.is_available()` else `cpu`) |
| `dtype` | `--dtype` (default `auto` → `bfloat16` on cuda, `float32` on cpu) |
| `score_threshold` | `--score-threshold` (default `0.3`) |
| `top_k` | `--top-k` (default `100`) per `(image, class)` pair |
| `mask_threshold` | fixed at `0.0` logit (sigmoid > 0.5); matches eval default; NOT a CLI flag |
| `batch_size` | `--batch-size` (default `1`; INFO hint when cuda free VRAM > 12 GB at default) |
| `seed` | `--seed` (default `0`; recorded in `run.json`) |

**Adapter pin behavior.** If `--checkpoint` is given and its `adapter_config.json` has `base_model_name_or_path` that disagrees with `--config.model.name` (or with the builtin default when no `--config`), log one `WARN` line naming both values and use the adapter's value. This is the only field where the adapter wins over an explicit CLI/config value.

**QLoRA detection.** If the checkpoint directory contains a file named `custom_sam_peft_qlora.json`, `detect_adapter_kind` returns `"qlora"` and the runner calls `load_qlora`. Otherwise it returns `"lora"` and the runner calls `load_lora`.

**`--merge-adapter` (default ON).** After loading the adapter, the runner calls `peft_adapters.lora.merge_lora(wrapper)` to fold deltas into the base model. The user opts out with `--no-merge-adapter`. Opt-out is required for QLoRA on tight VRAM because `merge_and_unload` dequantizes the 4-bit base back to bf16/fp32. The CLI does not auto-disable merge for QLoRA — the user makes the call.

**Base-model-only path.** When `--checkpoint` is omitted, the runner skips adapter detection and loading entirely. `model.name` falls through to `--config` or the builtin default; transforms still resolve via the model's `AutoImageProcessor`.

---

## 7. Outputs

`--output DIR` is required and is created if missing. Layout:

```text
<output>/
├── predictions.json     # COCO-flat list — see §7.1
├── image_id_map.json    # {"<int_id>": "<original absolute path>"} predict-only sidecar
├── run.json             # reproducibility record — see §7.2
├── masks/               # only if --save-masks=png
│   └── <stem>_<cat_id>_<inst_idx>.png   # uint8 binary mask, original-image H×W
└── visualizations/      # only if --visualize
    └── <stem>.png       # original image + per-instance overlay
```

### 7.1 `predictions.json` (COCO-flat)

```json
[
  {
    "image_id": 1234567890123456789,
    "category_id": 1,
    "bbox": [x, y, w, h],
    "score": 0.87,
    "segmentation": { "size": [H, W], "counts": "..." },
    "mask_png": "masks/img001_1_0.png"
  }
]
```

| Field | Notes |
| --- | --- |
| `image_id` | `int(blake2s(<absolute_path_str>.encode(), digest_size=8).hexdigest(), 16)` — same scheme as `eval/evaluator._int_image_id`. Stable across runs as long as the absolute path is stable. |
| `category_id` | 1-indexed position of the prompt in the resolved `--prompts` list. |
| `bbox` | `[x, y, w, h]` in **original-image coordinates**. Predict inverts the longest-edge resize and top-left pad applied by `build_eval_transforms` to map back. |
| `score` | Per-instance score from `queries_to_coco_results`. |
| `segmentation` | pycocotools RLE dict. `counts` is `.decode("ascii")`-ed so the JSON is valid (mirrors eval's `_mask_to_rle`). Omitted entirely when `--save-masks=none`. |
| `mask_png` | Relative path under `<output>/`. Present only when `--save-masks=png`. |

**Per-`(image, class)` selection.** Apply `score_threshold` first, then keep the top-K highest-scoring instances. `top_k` is a per-class cap, not a global cap — `K * num_classes * num_images` is the worst-case row count.

### 7.2 `image_id_map.json`

A flat object mapping the stringified `image_id` to the original absolute path:

```json
{
  "1234567890123456789": "/abs/path/to/img001.jpg",
  "9876543210987654321": "/abs/path/to/img002.png"
}
```

Predict-only — eval does not write this because eval already has a COCO dataset JSON.

### 7.3 `run.json`

```json
{
  "model": "facebook/sam3.1",
  "checkpoint": "/abs/path/to/adapter/",
  "adapter_kind": "lora",
  "merge_adapter": true,
  "prompts": ["cat", "dog", "person"],
  "score_threshold": 0.3,
  "top_k": 100,
  "mask_threshold": 0.0,
  "device": "cuda",
  "dtype": "bfloat16",
  "image_size": 1024,
  "batch_size": 1,
  "seed": 0,
  "version": "0.6.0",
  "git_sha": "abc1234",
  "n_images": 42,
  "n_predictions": 137,
  "elapsed_sec": 12.34
}
```

`checkpoint` is `null` for the base-model-only path. `adapter_kind` is `"lora"`, `"qlora"`, or `null`. `git_sha` is the output of `git rev-parse --short HEAD` from the package install root, or `null` if not in a git checkout. `version` is `custom_sam_peft.__version__`.

---

## 8. CLI surface (`csp predict`)

```bash
csp predict \
  --images PATH-OR-GLOB-OR-MANIFEST \
  --prompts STRING-OR-FILE \
  --output DIR \
  [--checkpoint PATH] \
  [--no-merge-adapter | --merge-adapter] \
  [--config PATH] \
  [--score-threshold FLOAT]   # default 0.3
  [--top-k INT]               # default 100 per (image, class)
  [--save-masks {rle,png,none}] # default rle
  [--visualize]               # default off
  [--device {auto,cuda,cpu}]   # default auto
  [--dtype {auto,bfloat16,float32}] # default auto
  [--batch-size INT]          # default 1
  [--seed INT]                # default 0
  [--dry-run]
  [-v|--verbose]
```

**Flag-level validation** (raises `typer.BadParameter` → exit 2):

| Flag | Constraint |
| --- | --- |
| `--score-threshold` | `0.0 <= x <= 1.0` |
| `--top-k` | `x >= 1` |
| `--batch-size` | `x >= 1` |
| `--save-masks` | one of `{rle, png, none}` |
| `--device` | one of `{auto, cuda, cpu}` |
| `--dtype` | one of `{auto, bfloat16, float32}` |
| `--checkpoint` | path must exist and contain `adapter_config.json` |
| `--images` | resolve to ≥ 1 image (see §5.1) |
| `--prompts` | resolve to ≥ 1 non-empty class name (see §5.2) |

`PredictOptions` is a `@dataclass(frozen=True)` with one field per CLI flag (using the *resolved* types — e.g., `images: Path`, `device: Literal["auto","cuda","cpu"]`). `predict_cmd.py` builds the dataclass; `run_predict` consumes it.

---

## 9. Forward loop (`predict/runner.py`)

```text
run_predict(opts: PredictOptions) -> PredictReport:
  1. images = resolve_images(opts.images)
     prompts = parse_prompts(opts.prompts)
  2. preflight log (INFO):
       "predict: model=<name> adapter=<kind|none> device=<dev> dtype=<dt>
        prompts=[N] images=M threshold=<t>"
  3. if opts.dry_run:
       print resolved-inputs preview (first 10 images, all prompts, resolved config)
       to stdout; exit 0; DO NOT load the model.
  4. model = load_sam31(model_cfg)
     resolve dtype; .to(device, dtype=dtype)
  5. if opts.checkpoint is not None:
       kind = detect_adapter_kind(opts.checkpoint)
       if kind == "qlora":  load_qlora(model, opts.checkpoint)
       else:                load_lora(model, opts.checkpoint)
       if opts.merge_adapter:
           merge_lora(model)   # also valid on the QLoRA path; user opted in
  6. transforms = build_eval_transforms(image_size, model_name, normalize)
  7. if device == "cuda" and batch_size == 1
        and torch.cuda.mem_get_info()[0] > 12 * 1024**3:
       log INFO hint: "free VRAM is >12 GB; consider --batch-size 4 or 8."
  8. warmup: one forward on zeros(1, 3, image_size, image_size) on device; discard.
  9. for each batch of B images:
       originals  = [(H, W) for each image in the batch]  # for inverse-transform
       image_ids  = [_int_image_id(path) for path in batch_paths]
       img_tensors = [transform(img) for img in batch_imgs]  # each (3, S, S)
       for class_idx, class_name in enumerate(prompts, start=1):
           # `queries_to_coco_results` is per-image (asserts pred_logits.shape[0] == 1).
           # B-image throughput is an implementer choice (single B-batch model call
           # then slice, OR B per-image model calls) — postprocess is called
           # per-image either way.
           for b in range(B):
               outputs_b = model(
                   img_tensors[b].unsqueeze(0).to(device),
                   [TextPrompts(classes=[class_name])],
                   box_hints=None,
               )
               entries = queries_to_coco_results(
                   outputs_b,
                   image_id=image_ids[b],
                   category_id=class_idx,
                   original_hw=originals[b],
                   mask_threshold=0.0,
               )
               entries = [e for e in entries if e["score"] >= opts.score_threshold]
               entries.sort(key=lambda e: e["score"], reverse=True)
               entries = entries[: opts.top_k]
               predictions.extend(entries)
       if verbose: log per-image latency.
  10. if opts.save_masks == "png":
        for entry in predictions: decode RLE, write masks/<stem>_<cat>_<idx>.png
                                  add entry["mask_png"] = "<rel path>"
                                  drop entry["segmentation"]
      elif opts.save_masks == "none":
        drop entry["segmentation"] from every entry
      # rle (default): leave segmentation in place
      if opts.visualize:
        for image, entries-for-image: render overlay → visualizations/<stem>.png
  11. write predictions.json, image_id_map.json, run.json
  12. return PredictReport(n_images, n_predictions, elapsed_sec)
```

`PredictReport` is a frozen dataclass with the three integer/float fields above. The CLI prints them as a one-line summary on success.

---

## 10. Errors and exit codes

| Condition | Handling | Exit |
| --- | --- | --- |
| Zero images resolved from `--images` | `typer.BadParameter` from `resolve_images` | 2 |
| Empty `--prompts` after parsing | `typer.BadParameter` from `parse_prompts` | 2 |
| `--checkpoint` path missing or lacks `adapter_config.json` | `typer.BadParameter` from `predict_cmd.py` flag callback | 2 |
| Bad enum value on `--save-masks` / `--device` / `--dtype` | Typer choice validation | 2 |
| `--score-threshold` outside `[0.0, 1.0]` | Typer callback | 2 |
| `--top-k` / `--batch-size` < 1 | Typer callback | 2 |
| Unreadable image mid-run | one `WARN`, skip, continue | 0 (if any image succeeded) |
| Every image failed (rare) | log error, exit 1 | 1 |
| CUDA OOM | propagate after logging hint: `"OOM: consider --no-merge-adapter (QLoRA), --batch-size 1, or --device cpu"` | non-zero (torch's `RuntimeError` propagates) |
| bitsandbytes missing on QLoRA path | reraise existing `_import_bnb` `ImportError` | non-zero |
| `--dry-run` | print preview, exit 0, never load model | 0 |

---

## 11. Testing strategy

### 11.1 CPU tests (`tests/predict/`)

All tests are CPU-only; per the [GPU vs CPU testing policy](2026-05-19-gpu-test-policy-design.md), GPU is reserved for real-only failure modes.

| File | Covers |
| --- | --- |
| `test_inputs.py` | Directory recursion; glob (`*` and `**`); single file; `.txt` manifest (comments + blank lines skipped); `.json` manifest; extension allowlist filtering; RGBA → RGB conversion; unreadable file → WARN-and-skip; zero images → `BadParameter` exit 2; sorted determinism (asserting on a hand-crafted unordered tmpdir). |
| `test_prompts.py` | Comma-separated string; one-per-line file; whitespace strip; empty entries dropped; dedupe preserves first-seen order; empty result → exit 2. |
| `test_config_layering.py` | CLI > `--config` > builtin precedence per field; adapter `base_model_name_or_path` pins `model.name` with WARN on disagreement (uses a stub adapter directory containing only `adapter_config.json` — no weights). |
| `test_adapter_detect.py` | `detect_adapter_kind`: LoRA dir, QLoRA dir (with `custom_sam_peft_qlora.json`), invalid dir → raises. |
| `test_writers.py` | `predictions.json` schema (COCO-flat, xywh, pycocotools RLE round-trip); `image_id_map.json` int → path; `run.json` shape and required keys; top-K + score-threshold math (synthetic entries); `--save-masks=none` omits `segmentation`; `--save-masks=png` adds `mask_png` and writes PNG; PNG dimensions match the original-image H×W. |
| `test_visualize.py` | Overlay on a synthetic 32×32 image: PNG written, deterministic per-class color (hash-based palette), score label drawn. Feeds synthetic `queries_to_coco_results`-shaped entries; no model needed. |
| `test_runner_smoke.py` | End-to-end with a stub `nn.Module` whose forward returns hand-crafted outputs matching `queries_to_coco_results`'s expected shape. One synthetic 64×64 image; two prompts; monkeypatch `load_sam31`. Asserts `predictions.json` well-formed, both sidecars written, `PredictReport` fields populated. |
| `test_dry_run.py` | `--dry-run` lists resolved inputs + prompts + resolved config to stdout, writes nothing, exits 0, never calls `load_sam31` (monkeypatched to raise if invoked). |
| `test_cli_predict.py` | Typer-level: argv → `PredictOptions` round-trip; `--score-threshold 1.5` rejected; `--top-k 0` rejected; `--batch-size 0` rejected; `--save-masks foo` rejected; `--device {auto,cuda,cpu}` and `--dtype {auto,bfloat16,float32}` choice validation. |
| `test_preprocessing_parity.py` | Byte-identical tensor between predict's transform path and `build_eval_transforms` for the same input image — guards against silently regressing the preprocessing path #69 warned about. |

### 11.2 GPU tests (`tests/predict/test_gpu_predict.py`, `@pytest.mark.gpu`)

| Test | Rationale (per GPU vs CPU policy: GPU only for real-only failure modes) |
| --- | --- |
| `test_predict_base_model_cuda` | Real `facebook/sam3.1` load + forward on a 1×1024×1024 dummy; assert `predictions.json` written and RLEs decode via pycocotools. |
| `test_predict_lora_adapter_cuda` | Tiny 50-step LoRA fixture (or reused CI artifact) → predict; exercises `merge_lora` path on a real wrapper. |
| `test_predict_qlora_no_merge_cuda` | Real bitsandbytes 4-bit load with `--no-merge-adapter`; the only test where 4-bit dequant-during-merge VRAM behavior matters. |
| `test_predict_vram_hint_log` | On cuda with sufficient free VRAM, the INFO hint string is emitted at default `--batch-size 1`. |

GPU tests skip cleanly under `pytest -m 'not gpu'` (the CI default). They run in the `cuda` GHA matrix per `2026-05-19-gpu-test-policy-design.md`.

### 11.3 Coverage targets

| Module | Target |
| --- | --- |
| `predict/runner.py` | 90%+ |
| `predict/inputs.py` | 90%+ |
| `predict/writers.py` | 90%+ |
| `predict/adapter_load.py` | 90%+ |
| `predict/visualize.py` | 70%+ (rendering paths are hard to assert exhaustively) |

The repo-wide 80% gate in `pyproject.toml` is preserved; new modules clear it comfortably under the per-module targets above.

---

## 12. Cross-cutting

| Concern | Decision |
| --- | --- |
| Bootstrap | `predict_cmd.py` does not need a new bootstrap site; it does not use the plugin registry. It imports `predict.runner` directly. |
| Error model | Argument-style errors → `typer.BadParameter` (exit 2). CUDA OOM and other torch runtime errors propagate after a one-line hint log. Unknown exceptions propagate with traceback intact. |
| Logging | `_configure_logging(verbose)` from `cli/_logging.py` (existing shared helper from the CLI spec). Runner uses Python `logging`; CLI uses `rich.print` only for the final summary. |
| Determinism | `--seed` (default 0) is set on `torch`, `numpy`, and `random` at the start of `run_predict`; the value is recorded in `run.json`. SAM 3.1 inference is not fully deterministic across devices, so the seed is best-effort. |
| Reuse with eval | Predict consumes `queries_to_coco_results` and the transforms unchanged. Sharing an inference helper between predict and eval is deferred — see §13. |
| Model-batch parallelism | `queries_to_coco_results` is per-image (`pred_logits.shape[0] == 1` asserted). v1 honors `--batch-size INT` as a user-visible flag but the forward path post-processes one image at a time after the model call — the implementer may stack a B-image tensor for one model call and slice the outputs, or call the model B times; either is conforming. True batched postprocessing is deferred (see §13). |
| Image-id collisions | `blake2s(absolute_path, digest_size=8)` gives a 64-bit space. Collision-on-a-single-run is negligible. The `image_id_map.json` sidecar means a user can always recover the source path from an id. |

---

## 13. Deferred

| Follow-up | Why deferred |
| --- | --- |
| Auto-bumping `--batch-size` from VRAM heuristics | Mentioned in #74; the user explicitly carved it out into a separate issue. Predict v1 emits an INFO hint only. |
| Batched postprocess (`queries_to_coco_results` for B > 1 images at once) | The postprocess hard-asserts `batch=1`. Reworking it (and validating SAM 3.1's batched forward with replicated per-image `TextPrompts`) is the natural home for the auto-batch follow-up; v1 keeps the per-image postprocess loop. |
| Refactoring `eval/` to share an inference helper with `predict/` | Both eval and predict construct the same forward loop today. Extracting a shared inference helper means touching eval's call sites, which the user wants to keep stable in this PR. |
| Server / HTTP API mode (`csp serve`) | Separate surface; different lifecycle. |
| Streaming / video inference | Frame extraction + temporal smoothing is its own design. |
| `torch.compile`, TRT, additional quant paths | Beyond QLoRA. |
| `csp predict` inside `csp run`'s train+eval+export bundle | `csp run` keeps its current scope. |
| Computing metrics inside predict | That is `csp eval`'s job. |

---

## 14. File layout

```text
pyproject.toml                                       TOUCHED (+1 line: csp script alias under [project.scripts])
src/custom_sam_peft/cli/main.py                      TOUCHED (+1 line: register predict)
src/custom_sam_peft/cli/predict_cmd.py               NEW
src/custom_sam_peft/predict/__init__.py              NEW
src/custom_sam_peft/predict/runner.py                NEW
src/custom_sam_peft/predict/inputs.py                NEW
src/custom_sam_peft/predict/adapter_load.py          NEW
src/custom_sam_peft/predict/writers.py               NEW
src/custom_sam_peft/predict/visualize.py             NEW
tests/predict/__init__.py                            NEW
tests/predict/test_inputs.py                         NEW
tests/predict/test_prompts.py                        NEW
tests/predict/test_config_layering.py                NEW
tests/predict/test_adapter_detect.py                 NEW
tests/predict/test_writers.py                        NEW
tests/predict/test_visualize.py                      NEW
tests/predict/test_runner_smoke.py                   NEW
tests/predict/test_dry_run.py                        NEW
tests/predict/test_cli_predict.py                    NEW
tests/predict/test_preprocessing_parity.py           NEW
tests/predict/test_gpu_predict.py                    NEW (@pytest.mark.gpu)
```

No deletions, no moves. `src/custom_sam_peft/eval/` is untouched. `src/custom_sam_peft/cli/run_cmd.py` is untouched.

---

## 15. Exit criteria

- `csp predict --help` exits 0; all documented flags appear in the help text.
- `csp predict --images <dir> --prompts cat,dog --output out/` on a base model writes `out/predictions.json`, `out/image_id_map.json`, `out/run.json` with the schemas in §7.
- `csp predict --checkpoint <run_dir>/adapter --images <dir> --prompts cat --output out/` works for both LoRA and QLoRA adapter directories without code changes.
- `csp predict --dry-run …` prints the resolved inputs/prompts/config and writes nothing.
- `csp predict --save-masks=png` writes PNGs under `out/masks/` whose dimensions match the original images.
- `csp predict --visualize` writes per-image overlays under `out/visualizations/`.
- `tests/predict/` passes under `pytest -m 'not gpu'` with coverage targets in §11.3 met.
- `tests/predict/test_gpu_predict.py` passes in the `cuda` GHA matrix.
- `ruff check`, `mypy --strict`, `pytest` clean.

---

## 16. Related

- [#74](https://github.com/NguyenJus/custom-sam-peft/issues/74) — Add `csp predict` for offline inference (this feature).
- [#70](https://github.com/NguyenJus/custom-sam-peft/issues/70) — v1.0 criteria; `csp predict` is a v1.0 gate.
- [#69](https://github.com/NguyenJus/custom-sam-peft/issues/69) — YAML defaults / normalization fallback; predict consumes the same `resolve_normalization` path and is guarded by `test_preprocessing_parity.py`.
- [#68](https://github.com/NguyenJus/custom-sam-peft/issues/68) — VRAM floors; `--no-merge-adapter` matters for 8 GB QLoRA.
- [#34](https://github.com/NguyenJus/custom-sam-peft/issues/34) — Docker image; `csp predict` is the natural smoke command inside the published image.
- Existing modules the implementer will need: `peft_adapters/qlora.py::load_qlora` (proves QLoRA-from-disk is supported); `eval/evaluator.py` (forward-loop signature template); `data/transforms.py::build_eval_transforms` + `resolve_normalization` (preprocessing parity).
