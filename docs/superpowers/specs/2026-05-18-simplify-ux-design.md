# Simplify UX Design (`spec/simplify-ux`)

**Status:** Draft (2026-05-18)
**Scope:** A drop-in path for non-technical users to fine-tune SAM 3.1 on Colab or RunPod with minimal ceremony, **without disturbing the existing power-user CLI**. Adds one new top-level command (`esam3 run`), one new orchestration-only Python module (`runs/bundle.py`), one VRAM-tier preset helper, three notebook-side helpers, a user-facing notebook (`notebooks/esam3_train.ipynb`), and a README restructure that puts a "Beginner" section on top of today's "Advanced" content.

**Tracks:** issue [#25](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/25) — *Simplify UX: drop-in model + data, schedule, get results*.

**Builds on:** [`2026-05-15-esam3-architecture-design.md`](2026-05-15-esam3-architecture-design.md) (stable seams; CLI surface; YAML as canonical config; v0 = text-prompts only); [`2026-05-18-cli-design.md`](2026-05-18-cli-design.md) (current `esam3 train|eval|export|init|doctor` shape and runner extraction); [`2026-05-17-eval-design.md`](2026-05-17-eval-design.md) (`MetricsReport` shape; per-example IoU semantics consumed by the bundler); [`2026-05-18-tracking-design.md`](2026-05-18-tracking-design.md) (the notebook does NOT introduce a new tracker backend — it reuses whatever `cfg.tracking.backend` says, defaulting to TensorBoard from the templates).

---

## 1. Goals & v0 Scope

The organizing idea: **two front doors, one spine.**

- **Non-technical users** open a Colab notebook with form widgets, set three values (dataset path, format, run name), and click Run All. The notebook auto-detects environment + VRAM, generates a `config.yaml`, and shells out to `esam3 run` for the actual work.
- **Power users** keep doing exactly what they do today (`esam3 init` → edit YAML → `esam3 train` / `eval` / `export`). They may *also* call `esam3 run` if they want the one-shot orchestrator, but nothing in their flow changes.
- **One spine.** Both doors converge on `esam3 run --config CONFIG`, which composes the existing `train.runner.run_training` → `eval.runner.run_eval` → `train.checkpoint.save_merged` (conditional) → new bundler. The same artefacts land in the same `runs/<id>/` directory the existing CLI already writes.

**In scope:**

| Item | Where |
|---|---|
| `esam3 run` CLI subcommand (train → eval → export → bundle) | `src/esam3/cli/run_cmd.py` (new), `src/esam3/cli/main.py` (+1 registration) |
| VRAM-tier preset patch generator | `src/esam3/presets.py` (new) |
| Notebook-side env / token / checkpoint helpers | `src/esam3/notebook_helpers.py` (new) |
| Results bundler: `summary.md` + `samples/*.png` | `src/esam3/runs/bundle.py` (new) |
| User-facing notebook (separate from existing GPU smoke notebook) | `notebooks/esam3_train.ipynb` (new) |
| README restructure with explicit `## Beginner` / `## Advanced` sections | `README.md` (rewritten in place; no content removed) |
| RunPod walkthrough for laypeople | `cloud/runpod/README.md` (new) |
| Unit + CLI-integration + GPU-smoke tests for all of the above | `tests/` (see §8) |

**Out of scope (filed as follow-ups, see §10):**

- Folder-format dataset adapter (`#33`).
- Published Docker image (`#34`).
- AWS SageMaker / Lambda Labs targets (`#35`).
- Algorithmic preset derivation (`#36`).
- GPU-marker test audit (`#37`).
- Any change to the existing `notebooks/colab_gpu_tests.ipynb` dev-smoke notebook (treated as orthogonal).
- Any new tracker backend, sweep integration, or hosted dashboard — `cfg.tracking.backend` continues to drive logging exactly as `spec/tracking` specified.
- Any change to existing `esam3 train` / `eval` / `export` / `init` / `doctor` behavior.

---

## 2. Architectural Approach

```
NON-TECHNICAL                              POWER USER
─────────────                              ──────────
README "## Beginner" section               README "## Advanced" section
+ Colab badge for esam3_train.ipynb        + existing Quickstart / CLI tables
       │                                          │
       ▼                                          ▼
notebooks/esam3_train.ipynb            esam3 init [--template …]    (unchanged)
  (env-aware: Colab + RunPod)              edit config.yaml
  SETUP    detect_env, pip install              │
           CUDA check, HF_TOKEN check           │
           (skip if local checkpoint present)   │
  FORM     dataset_path, data_format,           │
           run_name                             │
  GENERATE pick_preset() → patch                │
           merge(template, patch, form)         │
           write config.yaml                    │
           subprocess.Popen("esam3 run …")      │
           stream stdout into the cell          │
  RESULTS  render runs/<id>/summary.md          │
           inline-display samples/*.png         │
           print download instructions          │
       │                                         │
       ▼                                         ▼
              esam3 run --config CONFIG ──── NEW orchestrator (run_cmd.py)
                       │
                       ▼
      train (existing runner)
         │   on success
         ▼
      eval (existing runner)
         │   on success
         ▼
      save_merged   (only if cfg.export.merge — existing exporter)
         │   on success or failure (recorded)
         ▼
      write_bundle  (NEW — runs/bundle.py)
                       │
                       ▼
         runs/<run_id>/
           config.yaml        (existing; written by Trainer)
           adapter/           (existing)
           merged/            (only if cfg.export.merge)
           metrics.json       (existing)
           summary.md         (NEW — bundler)
           samples/*.png      (NEW — bundler, ≤ 6 files)
```

**Boundary rules (carried from architecture-design §3 and cli-design §3):**

- CLI files are thin shells. `run_cmd.py` body is ≤ 30 lines (parse → call orchestrator → format exit / output). No training, eval, export, or rendering logic in CLI files.
- `runs/bundle.py` is library-side (no Typer imports). The notebook never imports it directly — it shells out to `esam3 run`, which calls it.
- `presets.py` and `notebook_helpers.py` are independent, importable from both notebook and tests, and do not import from `cli/` or `runs/`.
- Pluggable registries (`dataset`, `peft`, `tracker`) are untouched. `esam3 run` re-uses the same dataset / model / tracker construction code that `esam3 train` and `esam3 eval` use today, by calling the existing runner functions.
- v0 training is text-prompts only (architecture §1). The notebook FORM cell MUST NOT expose a `bbox` prompt-mode option. `esam3 run` re-asserts the same pre-flight check `esam3 train` does (`cfg.data.prompt_mode != "bbox"`).

---

## 3. Deliverable 1 — `esam3 run` (new CLI orchestrator)

### 3.1 Surface

```
esam3 run --config CONFIG [--resume PATH] [-v|--verbose]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--config` | `Path` | required | YAML config (same schema as `esam3 train`) |
| `--resume` | `Path` | `None` | Same semantics as `esam3 train --resume` — checkpoint dir, passed through to `run_training` |
| `-v`/`--verbose` | bool | `False` | DEBUG logging via the existing `cli/_logging.py` helper |

### 3.2 Module

**File:** `src/esam3/cli/run_cmd.py`. **Body ≤ 30 lines** (matches the `cli-design` ≤ 30 LOC rule). Registered in `src/esam3/cli/main.py`:

```python
app.command("run", help="Train + eval + (optional) export + bundle in one shot.")(run_cmd.run)
```

### 3.3 Behavior (in order)

1. `configure_logging(verbose)`.
2. `cfg = load_config(config)` — no `--override`; `esam3 run` is the simple-path entry. Power users with override needs use `esam3 train`/`eval`/`export` directly.
3. Pre-flight: if `cfg.data.prompt_mode == "bbox"`, raise `typer.BadParameter("prompt_mode='bbox' is not supported for training in v0; …", param_hint="--config")` — same string as `esam3 train` uses today.
4. **Phase: train.** `result = run_training(cfg, resume_from=resume)`. On any exception, print the error and the `run_dir` (if available — see §3.4) and exit 1. Do not delete anything.
5. **Phase: eval.** `run_cmd.run` first builds the val dataset and the model wrapper itself (using the same registry calls the runners use), then calls `report, per_example_iou = run_eval(cfg, checkpoint=result.adapter_path, output_dir=result.run_dir, val_dataset=<built>, model=<built>, return_per_example_iou=True)` (split defaults to `"val"`; `save_predictions` defaults to whatever `cfg.eval.save_predictions` is — `esam3 run` does not override). On exception, print error + run_dir, exit 1. No bundle. See §3.4 for the additive `run_eval` kwarg extensions.
6. **Phase: export-merge (conditional).** If `cfg.export.merge` is true: reuse the wrapper built for step 5; `load_adapter(wrapper, result.adapter_path); save_merged(wrapper, result.run_dir / "merged")`. **Failures do not abort the orchestrator** — they're caught, logged at WARNING, recorded as `merged_dir = None` and `merged_export_error = <str(exc)>` on the `BundleContext` (see §6.4), and the orchestrator continues into the bundle phase. The merged-path failure is surfaced in `summary.md`.
7. **Phase: bundle.** Assemble a `BundleContext` (see §6.4) from `result.run_dir`, the config path, the captured `start_ts`/`end_ts`, `os.environ.get("ESAM3_PRESET_LABEL")`, the `per_example_iou` list returned by `run_eval`, and the merge-phase outcome. Call `write_bundle(ctx, report, val_dataset=<the val dataset built in step 5>, model_wrapper=<same wrapper>)`. **Bundle failures re-raise** (unexpected; this is the last phase and partial outputs are already on disk). Exit 1 if the bundle raises.
8. On full success, print `run_dir`, `adapter_path`, `merged_dir` (or "skipped" / "failed"), `summary_path`, and `mAP`. Exit 0.

### 3.4 Re-use boundaries with existing runners

`run_eval` (today) builds the val dataset and the model wrapper internally and discards both on return. `write_bundle` needs the same val dataset (and a loaded wrapper) to re-run inference on the picked sample indices, and we want to avoid paying the val-dataset build cost twice. The resolution is **additive kwarg extensions to `eval/runner.py::run_eval`** (cross-reference `2026-05-17-eval-design.md`):

```python
# eval/runner.py — additive kwargs (backward compatible)
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    val_dataset: Dataset | None = None,        # NEW — if provided, used in place of rebuild
    model: ModelWrapper | None = None,         # NEW — if provided, used in place of rebuild
    return_per_example_iou: bool = False,      # NEW — see §6.3
    # … existing kwargs …
) -> MetricsReport | tuple[MetricsReport, list[float]]:
    ...
```

**Backward-compat guarantee:** every existing caller (the `esam3 eval` CLI subcommand and its tests) continues to pass `val_dataset=None`, `model=None`, `return_per_example_iou=False` (all defaults) and continues to get a `MetricsReport` back. Behavior for those callers is unchanged.

**`esam3 run` usage:** the orchestrator builds the val dataset and the SAM 3.1 wrapper once (via the same registry / `load_sam31` calls the existing runners use), passes them to `run_eval` via the new kwargs, and reuses the same wrapper for both the (optional) merge phase (§3.3 step 6) and `write_bundle` (§3.3 step 7). The val dataset is built exactly once per `esam3 run` invocation. The wrapper is loaded exactly once. The adapter is loaded once via `run_eval` (and again only if the merge phase reloads from disk — implementation may share the in-memory adapted wrapper).

This extension matches `Evaluator.evaluate`'s sibling extension in §6.3.

### 3.5 Exit codes

| Outcome | Exit code |
|---|---|
| Success (all phases completed; merge may have soft-failed) | 0 |
| `prompt_mode='bbox'` pre-flight, bad config | 2 (`BadParameter`) |
| Train failure | 1 |
| Eval failure | 1 |
| Bundle failure | 1 |
| Unknown exception | propagate (traceback preserved) |

`runs/<id>/` is **never** cleaned up after a failure (even partial). The user gets the run_dir path on stderr so they can salvage / inspect.

---

## 4. Deliverable 2 — `src/esam3/presets.py`

### 4.1 Surface

```python
def pick_preset() -> dict:
    """Return a config-patch dict keyed by the current GPU's VRAM.

    Raises:
        RuntimeError: torch.cuda.is_available() is False.
    """
```

### 4.2 Algorithm

1. If `torch.cuda.is_available()` is False → `raise RuntimeError("pick_preset() requires CUDA; got cpu-only torch. In Colab: Runtime → Change runtime type → GPU. On RunPod: deploy a GPU pod.")`.
2. `total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)`.
3. Look up the tier in the table below and return the corresponding patch dict.

### 4.3 Tier table

| VRAM bucket | `peft.method` | `peft.r` | `train.batch_size` | `train.grad_accum_steps` | `model.gradient_checkpointing` | `model.dtype` |
|---|---|---|---|---|---|---|
| `< 12 GB`        | `qlora` | 8  | 1 | 16 | `true`  | `bfloat16` |
| `12 ≤ x < 24 GB` | `qlora` | 16 | 1 | 8  | `true`  | `bfloat16` |
| `24 ≤ x < 48 GB` | `lora`  | 16 | 2 | 4  | `false` | `bfloat16` |
| `x ≥ 48 GB`      | `lora`  | 32 | 4 | 2  | `false` | `bfloat16` |

Bucket boundaries are inclusive-low / exclusive-high. `24.0` GB → "24-48" tier.

### 4.4 Return shape

The dict mirrors a partial `TrainConfig` so the notebook can deep-merge it onto a loaded template:

```python
{
  "peft":  {"method": "qlora", "r": 16},
  "train": {"batch_size": 1, "grad_accum_steps": 8},
  "model": {"gradient_checkpointing": True, "dtype": "bfloat16"},
}
```

Keys not in this patch (e.g. `peft.alpha`, `peft.dropout`, `data.*`, `eval.*`, `tracking.*`, `run.name`) are left to the template + user inputs. The patch never sets `peft.method = "lora"` on top of a QLoRA template (or vice versa); the notebook GENERATE cell picks the template *based on* the patch's `peft.method` (see §6.3).

### 4.5 Tier label

A companion helper for the bundler:

```python
def preset_label(total_bytes: int | None = None) -> str:
    """Return a short tier label like 'auto: 16-24GB tier'. None → reads device 0."""
```

Used only by the notebook to pass a preset label into the bundler. **Plumbing:** the notebook GENERATE cell (§7.3 step 6) sets

```python
os.environ["ESAM3_PRESET_LABEL"] = preset_label()
```

before invoking `esam3 run`. The orchestrator reads `os.environ.get("ESAM3_PRESET_LABEL")` and forwards the value (or `None` if unset) onto `BundleContext.preset_label` (§6.4). When `preset_label` is `None`, `summary.md` records `"manual"`. CLI users who invoke `esam3 run` directly without setting the env var get the `"manual"` rendering — this is the intended fallback.

### 4.6 Provenance + replacement plan

This table is hand-tuned, not derived. **`#36` tracks replacing it with a derived heuristic** (e.g. based on model param count + activation memory estimate). The spec deliberately leaves the table as the only source of truth in v0 — algorithmic derivation is a separate spec, and the table is small enough to audit.

---

## 5. Deliverable 3 — `src/esam3/notebook_helpers.py`

Three small helpers, each unit-tested. CLI never imports them; the notebook + tests do.

### 5.1 `detect_env`

```python
def detect_env() -> Literal["colab", "runpod", "unknown"]:
    """Best-effort environment detection from env vars.

    - "colab" if os.environ.get("COLAB_GPU") is set (any value).
    - "runpod" elif os.environ.get("RUNPOD_POD_ID") is set.
    - "unknown" otherwise.
    """
```

Pure-function; tests inject via `monkeypatch.setenv` / `delenv`.

### 5.2 `check_local_checkpoint`

```python
def check_local_checkpoint(local_dir: Path, checkpoint_file: str) -> bool:
    """Return True iff (local_dir / checkpoint_file).is_file()."""
```

The notebook uses this with the default template's values (`model.local_dir="models/sam3.1"`, `model.checkpoint_file="sam3.1_multiplex.pt"`) to decide whether the HF token requirement applies. A RunPod user mounting a network volume with the checkpoint at the expected path can skip HF auth entirely.

### 5.3 `resolve_hf_token`

```python
def resolve_hf_token(env: Literal["colab", "runpod", "unknown"], local_present: bool) -> str | None:
    """Resolve the HF token according to environment and local-checkpoint state.

    - If local_present is True: print/log "local checkpoint detected — skipping HF auth";
      return None. Caller (notebook) interprets None as 'do not pass token to load_sam31'.
    - Else, fetch the token:
        - "colab"   → google.colab.userdata.get("HF_TOKEN")
        - "runpod"  → os.environ["HF_TOKEN"]
        - "unknown" → os.environ["HF_TOKEN"]
    - If missing, raise RuntimeError with an environment-specific friendly message:
        - "colab"   → "Set HF_TOKEN in Colab Secrets (left sidebar → 🔑)."
        - "runpod"  → "Set HF_TOKEN in your pod's Environment Variables, or mount a
                       network volume containing models/sam3.1/sam3.1_multiplex.pt."
        - "unknown" → "Set HF_TOKEN in your shell environment (export HF_TOKEN=…)."
    """
```

Three-arm error string matters: laypeople copy/paste the fix verbatim, so the wording is part of the contract. Tests assert the substring "Colab Secrets" / "Environment Variables" / "shell environment" per arm.

`google.colab.userdata` is imported inside the function (under `try: from google.colab import userdata except ImportError: …` → if Colab arm requested without `google.colab` available, raise the same Colab error string; the test suite covers this by stubbing `sys.modules`).

---

## 6. Deliverable 4 — `src/esam3/runs/bundle.py`

New module. Three public functions, in dependency order.

### 6.1 `pick_samples`

```python
def pick_samples(
    per_example_iou: list[float],   # one float per val example, in dataset order
    overall_mAP: float,              # from metrics_report.overall["mAP"]
    n_val: int,                      # len(val_dataset); MUST equal len(per_example_iou)
) -> list[int]:                      # indices into the val dataset
    """Pick which val examples to render as sample overlays.

    Returns up to 6 indices (capped at min(6, n_val)), composed by bracket:

      mAP >= 0.7     -> 4 best + 1 median + 1 worst
      0.4 <= mAP <0.7 -> 2 best + 2 median + 2 worst
      mAP < 0.4 or NaN -> 1 best + 1 median + 4 worst

    Ranking score is per_example_iou[i] — defined upstream as the MEAN IoU
    across the eval's IoU thresholds [0.5, 0.55, …, 0.95] for example i.
    The bundler is purely a sorter; it does not compute IoU itself.

    Tie-breaking: by index ascending (deterministic).
    Empty val (n_val == 0): return [].
    n_val < 6: cap and prorate composition proportionally (round down);
      if the prorated total falls short of min(6, n_val), top up with 'worst'
      (failure cases are the most informative).
    """
```

**Bracket composition algorithm (deterministic; testable):**

1. Pick `cap = min(6, n_val)`.
2. Choose the bracket and its (best, median, worst) triple per the docstring.
3. If `cap == 6`: return the triple as-is.
4. Else compute `ratios = [b/6, m/6, w/6]`, multiply by `cap`, floor each → `picks = [bp, mp, wp]`.
5. Top up: while `sum(picks) < cap`, increment `picks[2]` (worst). (Failure cases are most informative; this matches the user's brief.)
6. Convert to indices:
   - `best_idx  = sorted-by-score-desc, take first bp, tie-break by index asc`
   - `worst_idx = sorted-by-score-asc, take first wp, tie-break by index asc`
   - `median_idx`: take the `mp` indices closest to the **median IoU** by absolute distance, excluding any index already picked as best or worst; tie-break by index asc.
7. Return the concatenation in (best…, median…, worst…) order. (Order matters because filenames embed the ordinal — see §6.2.)

**Edge cases the unit tests pin:**

- `n_val == 0` → `[]`. No file I/O implied.
- All-identical IoUs → falls through the tie-break by index; still returns `cap` indices.
- All-zero IoUs → same.
- NaN `overall_mAP` → "poor" bracket (1 best + 1 median + 4 worst, then capped).
- Some `per_example_iou[i]` is NaN (eval skipped that example): treat as `-inf` for ranking — those examples sort as the worst. They are eligible for `worst` picks; they are not eligible for `best` or `median`. The bundler logs `WARNING bundle: N val examples had NaN IoU; treated as worst`.
- `n_val == 1` → `cap = 1`; after proration that lone index lands in the slot dictated by the bracket. In "poor" bracket the topup-with-worst path lands it as `0_worst.png`. (Tested.)

### 6.2 `render_overlay`

```python
def render_overlay(
    image: PIL.Image.Image,                # the resized image as fed to the model
    predicted_mask: np.ndarray,            # bool, shape (H, W) — model output thresholded
    ground_truth_mask: np.ndarray,         # bool, shape (H, W)
    *,
    caption: str,                          # e.g. "best @ IoU=0.83"
) -> PIL.Image.Image:
    """Return a single PNG-able image with prediction overlaid on the source.

    Visual contract (locked so the test can hash the output):
      - Prediction in semi-transparent magenta (RGBA 255, 0, 255, 96).
      - Ground truth in semi-transparent cyan  (RGBA  0, 255, 255, 96).
      - Caption text baked into the image at the bottom-left, white on a
        black 50%-opacity strip, default Pillow font.
      - Output mode is RGB (caller will save as PNG).
    """
```

Implementation notes (non-binding; informs the plan):

- Uses `PIL.ImageDraw` + `PIL.Image.alpha_composite`. No matplotlib (keeps the dep surface clean).
- Caller (`write_bundle`) is responsible for converting `np.ndarray` masks to `bool` and aligning them to the image's H×W before passing in. The bundler asserts `predicted_mask.shape == ground_truth_mask.shape == image.size[::-1]` and raises `ValueError` on mismatch.
- Deterministic given identical inputs. The unit test asserts shape + first-pixel triple, not a hash (font rendering varies across Pillow versions).

### 6.3 `write_bundle`

```python
def write_bundle(
    ctx: BundleContext,                # see §6.4 — bundles run-context fields into one frozen dataclass
    metrics_report: MetricsReport,
    val_dataset: Dataset,              # the same val ds run_eval iterated
    model_wrapper: Any,                # the loaded + adapted SAM3.1 wrapper
) -> None:
    """Write ctx.run_dir/summary.md and ctx.run_dir/samples/*.png. Idempotent: re-runs overwrite."""
```

**Steps:**

1. Read `per_example_iou = ctx.per_example_iou` (already computed during the eval phase — see below). The bundler does NOT re-iterate the val set to compute IoUs; it accepts them as input.
2. `indices = pick_samples(per_example_iou, metrics_report.overall["mAP"], len(val_dataset))`.
3. For each picked index, **re-run inference for that one example only** (eval typically drops predictions to save memory; re-inference is cheap for ≤ 6 examples). Compute the predicted mask, fetch the ground-truth mask from `val_dataset[i]`, and call `render_overlay(...)` with caption `"<bracket> @ IoU=<value:.2f>"`.
4. Write PNGs to `ctx.run_dir / "samples" / f"{ordinal}_{bracket}.png"`, where `ordinal` is 0-based within the bracket: `0_best.png`, `1_best.png`, `0_median.png`, `0_worst.png`, etc. Failed re-inferences are caught per-sample, logged at WARNING, and noted under "Edge cases" in `summary.md`; they do not abort the bundle.
5. Write `ctx.run_dir / "summary.md"` with the sections listed in §6.4.

**Per-example IoU contract (resolved).** `per_example_iou` is produced by the eval phase via the additive `Evaluator.evaluate(..., return_per_example_iou=True)` / `run_eval(..., return_per_example_iou=True)` extensions described below, and is plumbed into the bundler via `BundleContext.per_example_iou`. Cross-reference `2026-05-17-eval-design.md`; the extension is additive and backward-compatible.

**`Evaluator.evaluate` extension** (additive; cross-ref `2026-05-17-eval-design.md`):

```python
class Evaluator:
    def evaluate(
        self,
        model: Any,
        dataset: Dataset,
        *,
        return_per_example_iou: bool = False,
    ) -> MetricsReport | tuple[MetricsReport, list[float]]:
        """When return_per_example_iou=True, also returns a list of per-example
        mean IoU values aligned with `dataset` indices. Each entry is the MEAN
        IoU across the eval's IoU thresholds [0.5, 0.55, …, 0.95] for that
        example. When False (default), behavior is unchanged: returns
        MetricsReport only."""
```

`eval/runner.py::run_eval` grows a matching `return_per_example_iou: bool = False` kwarg and forwards it to `Evaluator.evaluate` (see §3.4 for the full extended signature). When True, `run_eval` returns `(MetricsReport, list[float])`; when False, returns `MetricsReport`.

**Performance rationale.** Computing per-example IoU inside the eval pass costs essentially nothing — the eval is already running inference and computing IoU across the same thresholds; it just needs to retain (instead of drop) the per-example aggregates. Re-deriving the same numbers in the bundle phase would require a second full pass over the val set, which on a real-sized val split costs hours of GPU time. The additive kwarg pattern keeps the eval-design contract intact for existing callers while letting `esam3 run` opt in.

**Backward-compat guarantee.** Every existing caller (the `esam3 eval` CLI subcommand) passes the default `False` and continues to receive only `MetricsReport`. No test or downstream consumer of `Evaluator.evaluate` / `run_eval` needs to change.

### 6.4 `BundleContext` + `summary.md` contract

**`BundleContext`** — a small frozen dataclass that bundles the run-context fields the bundler needs into one argument, replacing a sprawl of `write_bundle` kwargs. Defined alongside `write_bundle` in `src/esam3/runs/bundle.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class BundleContext:
    run_dir: Path
    config_path: Path                # the config.yaml written at run start
    start_ts: datetime               # train phase start (UTC)
    end_ts: datetime                 # eval phase end (UTC); bundler may extend if needed
    preset_label: str | None         # from ESAM3_PRESET_LABEL env var, or None
    per_example_iou: list[float]     # from run_eval(..., return_per_example_iou=True)
    merged_dir: Path | None          # set if export-merge succeeded; None if not requested or failed
    merged_export_error: str | None  # set if export-merge was requested but failed; None otherwise
```

`esam3 run` is the sole assembler of `BundleContext`. It captures `start_ts` immediately before the train phase, captures `end_ts` immediately after the eval phase, reads `os.environ.get("ESAM3_PRESET_LABEL")` for `preset_label` (see §4.5), gets `per_example_iou` from the `run_eval(..., return_per_example_iou=True)` return tuple (see §3.4 / §6.3), and fills `merged_dir` / `merged_export_error` from the export-merge phase outcome (§3.3 step 6). The bundler treats `ctx` as read-only.

**`summary.md`.** The bundler writes the following sections, in this order. Tests assert each section header literal.

```
# <run_name> — <final mAP, 4 decimal places>

## Run
- Start:  <ISO 8601 UTC>
- End:    <ISO 8601 UTC>
- Duration: <H:MM:SS>

## Hardware
- GPU:  <torch.cuda.get_device_properties(0).name>
- VRAM: <total_memory / 1024**3, 1 decimal> GB

## Preset
- Applied: <preset_label or "manual">

## Outputs
- Adapter: <relative path to runs/<id>/adapter/>
- Merged:  <relative path to ctx.merged_dir  |  "skipped (cfg.export.merge=false)"  |  "FAILED — <ctx.merged_export_error> — see logs">
- Config:  <relative path to ctx.config_path>

## Samples
<for each PNG in samples/, a markdown image embed in the order best → median → worst>

## Edge cases
<bulleted list; only included if any of the following triggered>
- empty val: no samples rendered (n_val == 0)
- capped: n_val=<N> < 6 → rendered <K> samples per prorated composition
- NaN mAP: classified as 'poor' bracket
- skipped samples: <i_index> raised <exc class> during inference — see log
```

Times come from a `datetime.now(tz=timezone.utc)` bracket inside `esam3 run` — the orchestrator captures `start_ts` / `end_ts` and plumbs them via `BundleContext` (see the dataclass above). Duration is rendered from `end_ts - start_ts`.

---

## 7. Deliverable 5 — `notebooks/esam3_train.ipynb`

A **separate, new** notebook. The existing `notebooks/colab_gpu_tests.ipynb` (the dev GPU smoke runner) is **not touched** except for the optional addition in §8.4 (adding the new GPU-smoke test to its Run All).

The notebook is intentionally thin glue — all logic lives in `presets.py` / `notebook_helpers.py` / `cli/run_cmd.py`. Cells:

### 7.1 SETUP cell

1. `from esam3.notebook_helpers import detect_env, check_local_checkpoint, resolve_hf_token`
2. `env = detect_env()`
3. `pip install` the repo from GitHub (`pip install "git+https://github.com/NguyenJus/Efficient-SAM3-Finetuning.git[qlora,tensorboard]"`). No Docker image in v0 (deferred to **#34**). The exact extras spec is read from the same line so the notebook and the README stay in sync.
4. `assert torch.cuda.is_available(), "No CUDA detected. In Colab: Runtime → Change runtime type → GPU. On RunPod: deploy a GPU pod."` — explicit error string with the fix.
5. `local_present = check_local_checkpoint(Path("models/sam3.1"), "sam3.1_multiplex.pt")` (paths match the default template).
6. `token = resolve_hf_token(env, local_present)` — handles the local-skip and missing-token error messaging.
7. Print one line: `f"mode: env={env}, local_checkpoint={local_present}, hf_auth={'skipped' if token is None else 'enabled'}"`.

### 7.2 FORM cell (Colab `#@param` widgets)

```python
dataset_path: str = ""        #@param {type:"string"}
data_format: str = "coco"     #@param ["coco", "hf"]
run_name: str = "my-run"      #@param {type:"string"}
```

**Explicit non-option:** there is NO `prompt_mode` widget. v0 is text-only.

Assert `dataset_path` non-empty before continuing.

### 7.3 GENERATE cell

1. `from esam3.presets import pick_preset, preset_label`
2. `patch = pick_preset()`
3. Template selection: `template_name = "coco_text_qlora.yaml" if patch["peft"]["method"] == "qlora" else "coco_text_lora.yaml"`. Load via `importlib.resources.files("esam3.cli.templates") / template_name` (the templates `esam3 init` ships).
4. Deep-merge `(template, patch, user_inputs)` into a `dict`. User inputs map as follows:
   - `data.format = data_format`
   - `data.train.annotations` / `data.train.images` / `data.val.annotations` / `data.val.images`: derived from `dataset_path` per format.
     - **`coco`:** `dataset_path` is a directory containing `train/` and `val/` subdirectories ONLY. The notebook GENERATE cell auto-discovers each split's COCO annotation JSON inside the split directory, using the following preference order (first match wins):
       1. `_annotations.coco.json` (Roboflow's COCO export convention)
       2. `instances.json`
       3. `annotations.json`
       4. First file matching `*.json` (sorted lexically — deterministic)

       If a split directory contains no `*.json` file, the GENERATE cell raises a friendly error naming the searched path and listing the preference order above. Images are taken from the split directory itself (e.g. `<dataset_path>/train/`, `<dataset_path>/val/`); the resolved fields land as `data.train.annotations = "<dataset_path>/train/<resolved.json>"`, `data.train.images = "<dataset_path>/train/"`, and likewise for `val`. **This resolution layer lives in the GENERATE cell — it is a notebook concern, not a change to the COCO dataset adapter.** The COCO adapter continues to receive fully-qualified `annotations` and `images` paths per its existing contract.
     - **`hf`:** `dataset_path` is the HF dataset id and is passed through as-is — no layout discovery is performed. The dataset-format fields fall back to the HF adapter's expected shape (the HF adapter is already registered; see `cli-design` §5.1).
   - `run.name = run_name`
5. Write `./config.yaml` (UTF-8, `yaml.safe_dump(..., sort_keys=False)`).
6. `os.environ["ESAM3_PRESET_LABEL"] = preset_label()` so `esam3 run` can forward it to the bundler.
7. `proc = subprocess.Popen(["esam3", "run", "--config", "config.yaml"], stdout=PIPE, stderr=STDOUT, bufsize=1, text=True)` — line-buffered. Stream each `proc.stdout` line into the cell with `print(line, end="")`.
8. `rc = proc.wait()`; on `rc != 0` print the last 50 lines of captured output, print the run_dir if recoverable from stdout, and `raise SystemExit(rc)` to skip the RESULTS cell.

### 7.4 RESULTS cell (runs only on success)

1. `latest = max(Path("runs").iterdir(), key=lambda p: p.stat().st_mtime)`.
2. Render `latest / "summary.md"` via `IPython.display.Markdown(...)`.
3. For each PNG in `sorted((latest / "samples").glob("*.png"))`, `IPython.display.Image(...)` inline.
4. Print download instructions:
   - Colab: ``from google.colab import files; files.download(f"runs/{run.zip}")`` (with a `shutil.make_archive(...)` line above).
   - RunPod: `scp -P <pod_port> root@<pod_host>:/workspace/runs/<id>.zip ./`.

The notebook itself has **no automated execution test in v0** (see §8.4); helpers are tested, and the implementation plan requires two mandatory manual dry-runs (Colab + RunPod) before the PR is marked ready.

---

## 8. Deliverable 6 — README + RunPod walkthrough

### 8.1 README.md restructure

Two new top-level sections at the top of the file; existing content is **moved, not deleted**.

```
# efficient-sam3-finetuning

<existing 1-paragraph blurb + status callout: unchanged>

## Beginner — train in 3 clicks

(Plain-English language, no jargon. Bullet list of prerequisites.)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenJus/Efficient-SAM3-Finetuning/blob/main/notebooks/esam3_train.ipynb)

1. Open the notebook in Colab via the badge above.
2. In Colab Secrets, set `HF_TOKEN` (Hugging Face token with read
   access to gated `facebook/sam3.1`). If you've already downloaded
   the checkpoint to `models/sam3.1/sam3.1_multiplex.pt` (e.g. on a
   RunPod network volume), skip this step.
3. Either upload a dataset (COCO folder) or paste a HF dataset id,
   then click Runtime → Run All.

When the run finishes, scroll to the bottom of the notebook for a
summary, sample mask overlays, and a one-line download command.

For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).

## Advanced

(Everything from here on is identical content to today, moved verbatim
under a single Advanced header. Anchors preserved where reasonable.)

### Quickstart
…
### CLI
…
### What's supported in v0
…
### v0 Training scope
…
### Repo layout
…
### Development
…
### GPU test automation
…
```

**Explicit content rules:**

- The Beginner section uses the badge for **`esam3_train.ipynb`** (the new user notebook), NOT for `colab_gpu_tests.ipynb` (which stays linked under "GPU test automation" inside Advanced).
- The Advanced header is a literal `## Advanced`. No `<details>` collapse — laypeople scroll past it; power users skim it. Clear is kinder than clever.
- The CLI table inside Advanced gains a new row: `esam3 run --config CONFIG [--resume PATH] [-v]` | Functional. Description: "Train + eval + (optional) export + bundle, in one shot."
- No content is removed. Anchors `#quickstart`, `#cli`, `#whats-supported-in-v0`, `#v0-training-scope`, `#repo-layout`, `#development`, `#gpu-test-automation` continue to exist (they now sit under `## Advanced` rather than at the top level — GitHub renders them as `#quickstart` regardless of nesting).

### 8.2 `cloud/runpod/README.md` (new)

Layperson walkthrough. Approximate section list:

1. **Sign up.** Link to runpod.io. Mention pay-as-you-go vs. spot.
2. **Pick a GPU.** Recommend A40 as the entry tier ($/VRAM sweet spot for SAM 3.1; 48 GB lands in the LoRA tier of `pick_preset`). Note that L4 / 4090 / A100 also work; A40 is just the default suggestion.
3. **Deploy a stock RunPod PyTorch template.** Explicit phrasing: *"We deliberately do not publish or maintain a custom RunPod template — that's tracked in issue #34. Use the stock 'RunPod PyTorch 2.x' template."* Walk through Template → Deploy.
4. **Set `HF_TOKEN`.** Pod → Edit → Environment Variables → add `HF_TOKEN`. Or: skip this step if you've mounted a network volume that contains `models/sam3.1/sam3.1_multiplex.pt`.
5. **Open Jupyter Lab** via the pod's Connect button.
6. **Upload `notebooks/esam3_train.ipynb`.** Two options: drag-and-drop into the Jupyter file browser, or paste the raw GitHub URL into a Jupyter "Open from URL" prompt.
7. **Click Run All.** Same beginner flow as Colab.

Data-upload guidance:

- Small dataset (≤ 1 GB): Jupyter file browser.
- Large dataset: RunPod network volume (one-time upload, persists across pods).
- HF dataset: easiest of all — paste the id into the FORM cell, no upload at all.

---

## 9. Error-Handling Contract (cross-cutting)

| Failure mode | Where caught | User-visible behavior |
|---|---|---|
| No CUDA detected | `pick_preset()` raises `RuntimeError`; notebook SETUP assert | Runtime-type hint message (Colab) / "deploy a GPU pod" hint (RunPod). |
| `HF_TOKEN` missing and no local checkpoint | `resolve_hf_token` raises `RuntimeError` | Env-specific friendly message — "Colab Secrets" / "Environment Variables" / "shell environment". |
| `HF_TOKEN` missing but local checkpoint present | `check_local_checkpoint` short-circuits | Print "local checkpoint detected — skipping HF auth". No error. |
| `cfg.data.prompt_mode == "bbox"` | `esam3 run` pre-flight | `typer.BadParameter` (exit 2); identical message to `esam3 train`. |
| Train phase exception | `esam3 run` | Print error + `run_dir` (if Trainer reached the point of creating one); exit 1. **No eval, no merge, no bundle.** **No cleanup.** |
| Eval phase exception | `esam3 run` | Print error + `run_dir`; exit 1. **No merge, no bundle.** **No cleanup.** |
| Export-merge exception (only if `cfg.export.merge=true`) | `esam3 run` | Log WARNING, set `BundleContext.merged_dir = None` and `BundleContext.merged_export_error = str(exc)` (§6.4). **Bundle still runs.** `summary.md` records `"Merged: FAILED — <error> — see logs"`. `esam3 run` overall exit is still driven by the bundle phase. |
| Bundle exception | `esam3 run` | Re-raise. Exit 1. (Last phase; partial outputs already on disk.) |
| Bundler: empty val (`n_val == 0`) | `write_bundle` | Empty `samples/` directory; "empty val" note in `summary.md`. |
| Bundler: `n_val < 6` | `pick_samples` cap + proration | `samples/` has `min(n_val, 6)` PNGs; "capped" note in `summary.md`. |
| Bundler: per-sample inference exception | `write_bundle` per-sample try/except | That sample skipped; WARNING logged; "skipped samples" note in `summary.md`. Bundle does NOT abort. |
| Bundler: NaN mAP | `pick_samples` bracket selection | "poor" bracket used; "NaN mAP" note in `summary.md`. |
| Notebook subprocess: non-zero exit | GENERATE cell | Print tail of captured output + `run_dir` path (if extractable from stdout); `raise SystemExit(rc)` so RESULTS cell does not run. |

**Partial-output preservation rule (load-bearing):** `esam3 run` NEVER deletes anything in `runs/<id>/`. Train wrote `adapter/`; eval wrote `metrics.json`; the user keeps both even if a later phase blew up. This is the core "no surprises" guarantee for non-technical users.

---

## 10. Out of Scope (filed as follow-up issues)

These are deliberately excluded from this spec. Each must already have a GitHub issue at the time this spec lands; the plan-writer verifies.

| Issue | Title | Why deferred |
|---|---|---|
| **[#33](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/33)** | Folder-format dataset adapter (`data/images/*` + `data/masks/*` + `classes.txt`) | Lowest-friction format for laypeople, but the v0 simplify-UX target supports COCO and HF — those are already implemented. Folder format is additive and slots into the registry without touching this spec. |
| **[#34](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/34)** | Publish Docker image to GHCR | Would replace the notebook SETUP cell's `pip install git+…` with a single `docker pull`. Doable but adds release surface; v0 ships from-source install. |
| **[#35](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/35)** | Investigate AWS SageMaker + Lambda Labs as cloud targets | Stronger PII/security posture than Colab/RunPod. Future work once the Colab/RunPod flow is battle-tested. |
| **[#36](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/36)** | Algorithmically derive preset from VRAM | Replaces the hand-tuned table in `presets.py` (§4.3). The table is the v0 source of truth; algorithmic derivation needs its own brainstorm. |
| **[#37](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/37)** | Audit and minimize the `pytest -m gpu` surface | This spec adds one more GPU-marker test (§8.3); #37 is the place to triage the overall set. |

Also out of scope (no issue needed; just clarifying boundaries):

- New tracker backends (W&B is unchanged; TensorBoard is the default in both templates).
- Sweep / hyperparameter-search integration.
- A hosted dashboard or web UI on top of the notebook.
- Any change to `notebooks/colab_gpu_tests.ipynb` other than the optional one-test addition in §8.4.

---

## 11. Constraints from Upstream Architecture (do not violate)

Pulled forward from `architecture-design` and the existing per-subsystem specs:

- **Additive CLI surface** (`cli-design` §3). `esam3 run` is a new sibling of `train` / `eval` / `export` / `init` / `doctor`. None of those five change behavior.
- **Pluggable registries** (`architecture-design` §11): `dataset`, `peft`, `tracker` remain the only extension surfaces. This spec does not add a new registry.
- **YAML is the canonical config contract** (`architecture-design` §3, `cli-design` §5). Both front doors produce a YAML file; both call the same `load_config` → `TrainConfig` pipeline. The notebook's deep-merge output is just a YAML file written to disk.
- **Reproducibility** (`architecture-design` §Determinism): comes from saved `config.yaml` + RNG-state restore on resume. Bit-identical resume is NOT guaranteed and this spec does not promise it.
- **v0 training is text-prompts only** (`architecture-design` §1, `training-loop-design` §pre-flight). `prompt_mode='bbox'` is rejected at train time. **The notebook FORM cell must NOT expose a bbox option** (§7.2).
- **Image contract for tracking** (`tracking-design` §8): `np.ndarray[uint8]` shape `(H, W, 3)`. The bundler does NOT log to the tracker — it writes PNGs directly to `runs/<id>/samples/`. The image contract therefore does not bind it, but the bundler's PNG bit-depth (RGB 8-bit) coincidentally matches.
- **Eval mode** (`eval-design` §4): the bundler reuses `mode="full"` semantics because `esam3 run` calls `run_eval` with the cfg's default settings — which is `mode="full"` by default. `mode="lite"` is for mid-training ticks only.

---

## 12. Testing Strategy

All tests CPU-only unless explicitly marked `gpu`. Layout follows the existing convention (`tests/unit/test_*.py`, `tests/integration/test_*.py`, `tests/gpu/test_*.py`).

### 12.1 Unit (CPU, every commit)

**`tests/unit/test_presets.py`** (new):

- `test_pick_preset_requires_cuda` — `monkeypatch.setattr(torch.cuda, "is_available", lambda: False)`; assert `RuntimeError` mentioning "CUDA" and the runtime-hint snippet.
- `test_pick_preset_tiers` — parametrize on `(total_bytes, expected_patch)` for each of the four buckets including boundaries (e.g. `11.9 GB → <12 tier`, `12.0 GB → 12-24 tier`, `23.9 GB → 12-24 tier`, `24.0 GB → 24-48 tier`, `47.9 GB → 24-48 tier`, `48.0 GB → ≥48 tier`). Stub `torch.cuda.is_available → True` and `torch.cuda.get_device_properties(0).total_memory → total_bytes`.
- `test_preset_label_format` — basic substring assertions on the four tier labels.

**`tests/unit/test_notebook_helpers.py`** (new):

- `test_detect_env_colab` / `_runpod` / `_unknown` — env-var injection via `monkeypatch.setenv` / `delenv`.
- `test_check_local_checkpoint_present` / `_absent` — `tmp_path` with and without the expected file.
- `test_resolve_hf_token_local_short_circuits` — `local_present=True`, any env → returns `None`, prints/logs the "local checkpoint detected" line, never reads env.
- `test_resolve_hf_token_missing_per_env` — for each of `colab` / `runpod` / `unknown`, with no token present, assert the env-specific error string substring.
- `test_resolve_hf_token_colab_userdata_path` — monkeypatch `sys.modules["google.colab"]` with a stub `userdata.get` returning `"abc"`; assert returned token is `"abc"`. Then re-run with `userdata.get → None` → assert the Colab-arm error string.

**`tests/unit/runs/test_bundle.py`** (new, sub-dir under `tests/unit/`):

- `pick_samples`:
  - parametrized over `(mAP, n_val, ious)` covering all three brackets at `n_val=6`, `n_val=2`, `n_val=1`, `n_val=0`.
  - identical-IoU all-zeros at `n_val=6` → indices `[0,1,2,3,4,5]` partitioned per bracket by index asc.
  - NaN-mAP at `n_val=6` → 1+1+4 layout, sorted by IoU asc for worst.
  - NaN IoUs treated as `-inf`; assert they land in worst, not best/median.
- `render_overlay`:
  - assert output mode == `"RGB"`, output size == input image size.
  - assert prediction pixels and GT pixels are visibly recoloured (sample a pixel where masks differ and assert the channel mix).
  - assert `ValueError` on mismatched mask shapes.
- `write_bundle`:
  - integration-style with stub `MetricsReport`, fake `val_dataset` (returns 3 examples), monkeypatched `render_overlay` to write a deterministic 1-pixel PNG, monkeypatched per-example inference to deterministic IoUs.
  - assert `summary.md` exists, contains the headline mAP token, contains each section header literal, lists PNG files in best→median→worst order.
  - empty val: `samples/` directory present but empty; `summary.md` includes "empty val" line.
  - merge failure path: build a `BundleContext` with `merged_dir=None` and `merged_export_error="<msg>"`; assert `summary.md` includes `"Merged: FAILED — <msg> — see logs"`.

### 12.2 CLI integration (CPU)

**`tests/integration/test_cli_run.py`** (new):

- `test_run_full_success` — monkeypatch `run_training` → returns a fake `RunResult` with a tmp `run_dir`; monkeypatch `run_eval` → returns `(fake_MetricsReport, [0.1, 0.5, 0.9])` (since `esam3 run` invokes it with `return_per_example_iou=True`); `cfg.export.merge=False` so the merge phase is skipped; monkeypatch `write_bundle` → records its args; invoke `esam3 run --config <stub_yaml>`; assert exit 0, assert all three monkeypatched calls were made exactly once in order, and that the `BundleContext` passed to `write_bundle` carries the same `per_example_iou` list returned by `run_eval` and `merged_dir=None`, `merged_export_error=None`.
- `test_run_train_failure_skips_rest` — monkeypatched `run_training` raises `RuntimeError("kaboom")`; assert `run_eval`, `save_merged`, `write_bundle` were NOT called; exit code 1; stderr contains "kaboom".
- `test_run_eval_failure_skips_bundle` — `run_eval` raises; assert `save_merged`/`write_bundle` not called; exit 1.
- `test_run_merge_failure_still_bundles` — `cfg.export.merge=True`; `save_merged` raises; assert `write_bundle` WAS called and that the `BundleContext` it received has `merged_dir=None` and a non-empty `merged_export_error` string matching the raised exception; exit code driven by bundle outcome (here, 0).
- `test_run_bundle_failure_exits_1` — `write_bundle` raises; assert exit 1; train + eval + merge artefacts remain on disk (assert `run_dir` still exists).
- `test_run_rejects_bbox_prompt_mode` — `cfg.data.prompt_mode="bbox"`; assert exit 2 (BadParameter); no other phase called.
- `test_run_help_exits_zero` — `esam3 run --help` exits 0 and mentions "Train + eval".

### 12.3 GPU smoke (gated)

**`tests/gpu/test_run_end_to_end_gpu.py`** (new; uses the existing `-m gpu` marker):

- Invoke `esam3 run --config <existing tiny smoke YAML>` against the existing GPU-smoke fixture dataset.
- Assert:
  - exit code 0
  - `runs/<id>/adapter/` exists and is non-empty
  - `metrics.json` parses and contains an `overall.mAP` numeric value
  - `summary.md` exists and contains the token `mAP`
  - `samples/` exists and contains ≤ 6 `.png` files
  - if `cfg.export.merge=True` in the smoke YAML, `merged/` exists; else does not.

### 12.4 Notebook + cross-cutting

- **No automated notebook execution test in v0.** The notebook is thin glue; helpers are fully unit-tested. The implementation plan includes **two mandatory manual dry-runs before PR-ready**:
  - **Colab dry-run** — fresh Colab T4, badge → Run All on a 10-image fixture dataset. Verify: form widgets render; auto-preset picks the `<12 GB` or `12-24 GB` tier matching T4's 15 GB VRAM; subprocess output streams live; RESULTS cell renders summary + samples; download command works.
  - **RunPod dry-run** — fresh A40 pod, stock PyTorch template, follow `cloud/runpod/README.md` start-to-finish. Verify: env detected as `runpod`; HF token resolved from pod env vars; preset picks the `24-48 GB` tier; bundle samples render with non-trivial overlays.
- **Optional addition to `notebooks/colab_gpu_tests.ipynb`:** add the new `tests/gpu/test_run_end_to_end_gpu.py` to its Run All cell alongside the existing tests, so the dev smoke notebook exercises the orchestrator end-to-end. No other changes to that notebook.

### 12.5 Coverage gate

Unchanged at 80% on `src/esam3`. New modules each have direct unit tests above; `cli/run_cmd.py` is covered by `test_cli_run.py`.

### 12.6 Explicitly NOT tested

- pycocotools / Pillow / IPython internals.
- Real Colab `google.colab.userdata` behavior — stubbed in tests.
- Real W&B / TensorBoard network paths.
- Exact PNG byte hashes for `render_overlay` — Pillow font rendering drifts across versions; we assert shape + channel mixing instead.

---

## 13. File Layout

```
src/esam3/
  presets.py                  # NEW — pick_preset(), preset_label()
  notebook_helpers.py         # NEW — detect_env, check_local_checkpoint, resolve_hf_token
  runs/
    __init__.py               # NEW
    bundle.py                 # NEW — pick_samples, render_overlay, write_bundle
  cli/
    main.py                   # CHANGED — +1 registration for run_cmd.run
    run_cmd.py                # NEW — esam3 run

notebooks/
  esam3_train.ipynb           # NEW — user-facing
  colab_gpu_tests.ipynb       # UNCHANGED (optional: add the gpu smoke test to Run All)

cloud/
  runpod/
    README.md                 # NEW — layperson walkthrough

README.md                     # RESTRUCTURED in place (Beginner section on top; Advanced wraps existing content)

tests/
  unit/
    test_presets.py           # NEW
    test_notebook_helpers.py  # NEW
    runs/
      __init__.py             # NEW
      test_bundle.py          # NEW
  integration/
    test_cli_run.py           # NEW
  gpu/
    test_run_end_to_end_gpu.py  # NEW (gated by -m gpu)
```

No deletions. No moves of existing source files. README is the only existing file whose content is rearranged (not removed).
