<!-- markdownlint-disable MD050 MD058 -->
# Hardening Pass Audit Inventory
**Issue:** #26  
**Date:** 2026-05-21  
**Spec authority:** `docs/superpowers/specs/2026-05-21-hardening-pass-design.md`

---

## Section A — Per-file census

Scope: every `.py` under `src/custom_sam_peft/` (recursively). `tests/` excluded.

**Column notes:**
- *inbound deps*: module-prefix list of importers within `src/custom_sam_peft/`.
- *outbound deps*: module-prefix list of what this file imports within `src/custom_sam_peft/`.
- *duplication notes*: `mcp__token-savior__find_semantic_duplicates` (method=ast) returned **no clusters** across the codebase; "—" for all rows.
- *≥60-line functions*: from `mcp__code-review-graph__find_large_functions_tool` (min_lines=60), functions only (not classes/files).
- *cross-module reach-through*: per OQ5 — import from one `src/custom_sam_peft/<module>/` subdir into a different such subdir, bypassing documented seams. Exempt seams: `runtime/`, `paths/`, `errors.py`, `config/`, `_bootstrap.py`, `_registry.py`, `peft_adapters/__init__.py`.

| path | responsibility (one-liner) | inbound deps | outbound deps | duplication notes | ≥60-line functions | cross-module reach-through |
|---|---|---|---|---|---|---|
| `src/custom_sam_peft/__init__.py` | Package version constant; no logic. | — | — | — | — | — |
| `src/custom_sam_peft/_bootstrap.py` | Triggers all `@register` sites by importing data, peft_adapters, and tracking sub-modules once. | cli, notebook_helpers (indirect) | data, peft_adapters, tracking | — | — | — |
| `src/custom_sam_peft/_registry.py` | Generic (kind, name) factory registry: `@register` decorator + `lookup` + `list_registered` + `reset_registry`. | _bootstrap, data.coco, data.hf, peft_adapters.lora, peft_adapters.qlora, tracking.noop, tracking.tensorboard, tracking.wandb, tracking, eval.runner, train.runner | — | — | — | — |
| `src/custom_sam_peft/cli/__init__.py` | Empty CLI package marker. | cli.main | — | — | — | — |
| `src/custom_sam_peft/cli/_logging.py` | Single `configure_logging(verbose)` function that sets root logger level and format. | cli.train_cmd, cli.eval_cmd, cli.export_cmd, cli.run_cmd | — | — | — | — |
| `src/custom_sam_peft/cli/doctor_cmd.py` | CLI shell for `doctor`: renders `DoctorReport` as a Rich table or JSON. | cli.main | diagnostics | — | — | `cli/doctor_cmd.py` imports from `diagnostics.py` (top-level, not a subdir seam) — **exempt** |
| `src/custom_sam_peft/cli/eval_cmd.py` | CLI shell for `eval`: thin wrapper over `eval.runner.run_eval`. | cli.main | cli._logging, config.loader, eval.runner | — | — | `cli/eval_cmd.py:13` imports `run_eval` from `eval/runner.py` — **cross-module** (cli→eval); no documented seam exists yet |
| `src/custom_sam_peft/cli/export_cmd.py` | CLI shell for `export`: loads model+adapter and saves adapter or merged weights. | cli.main | cli._logging, config.loader, models.sam3, train.checkpoint | — | — | `cli/export_cmd.py:13-14` imports from `models/` and `train/` — **cross-module** (cli→models, cli→train); no seam |
| `src/custom_sam_peft/cli/init_cmd.py` | CLI shell for `init`: writes a starter YAML from a packaged template and optionally downloads weights. | cli.main | config.loader, utils.huggingface | — | — | `cli/init_cmd.py:17` imports from `utils/` — **cross-module** (cli→utils); no seam |
| `src/custom_sam_peft/cli/main.py` | Typer app entry point: registers all subcommands and imports `_bootstrap` for side-effect. | (entry point) | _bootstrap, cli.* | — | — | — |
| `src/custom_sam_peft/cli/run_cmd.py` | CLI shell for `run`: orchestrates train→eval→export-merge→bundle in one shot. | cli.main | _registry, cli._logging, config.loader, config.schema, data.base, eval.runner, models.sam3, runs.bundle, train.checkpoint, train.runner | — | — | `cli/run_cmd.py:20-29` imports from `eval/`, `models/`, `runs/`, `train/` — **cross-module** (cli→eval, cli→models, cli→runs, cli→train); no seam |
| `src/custom_sam_peft/cli/train_cmd.py` | CLI shell for `train`: thin wrapper over `train.runner.run_training`. | cli.main | cli._logging, config.loader, train.runner | — | — | `cli/train_cmd.py:13` imports from `train/` — **cross-module** (cli→train); no seam |
| `src/custom_sam_peft/cli/templates/__init__.py` | Package marker for template YAML files; no logic. | cli.init_cmd (resource access) | — | — | — | — |
| `src/custom_sam_peft/config/__init__.py` | Empty config package marker. | — | — | — | — | — |
| `src/custom_sam_peft/config/loader.py` | YAML load → override application → path resolution → `TrainConfig` validation; raises `ConfigError`. | cli.train_cmd, cli.eval_cmd, cli.run_cmd, cli.init_cmd, diagnostics | config.schema | — | — | — |
| `src/custom_sam_peft/config/schema.py` | Pydantic v2 schema for all config classes (`TrainConfig`, `EvalConfig`, etc.) and all type aliases. | config.loader, train.loop, train.trainer, train.checkpoint, eval.runner, eval.evaluator, peft_adapters.lora, peft_adapters.qlora, data.coco, data.hf, data.transforms, tracking.noop, tracking.tensorboard, tracking.wandb, models.sam3, diagnostics | — | — | — | — |
| `src/custom_sam_peft/data/__init__.py` | Empty data package marker. | _bootstrap | — | — | — | — |
| `src/custom_sam_peft/data/base.py` | Protocol + dataclasses for the data seam: `TextPrompts`, `BoxPrompts`, `Prompts`, `Instance`, `Example`, `Dataset`. | data.coco, data.hf, data.collate, train.loop, train.trainer, eval.evaluator, runs.bundle, models.losses, models.matching | — | — | — | — |
| `src/custom_sam_peft/data/coco.py` | COCO instance-JSON adapter: sparse-id remap, crowd-image drop, segmentation decode, `@register("dataset","coco")`. | _bootstrap, data.hf (imports `_build_text_prompts`, `_decode_segmentation`) | _registry, config.schema, data.base, data.transforms | — | `__getitem__` (107 lines, L168) | `data/hf.py:166` imports `_build_text_prompts` from `data/coco.py` — within `data/` subdir, **intra-module** (not cross-module) |
| `src/custom_sam_peft/data/collate.py` | `collate_batch`: stacks images into `(B,3,H,W)` tensor; keeps prompts/instances as Python lists. | train.trainer | data.base | — | — | — |
| `src/custom_sam_peft/data/hf.py` | HuggingFace datasets adapter with field-map resolution and `@register("dataset","hf")`. | _bootstrap | _registry, config.schema, data.base, data.coco (private `_build_text_prompts`, `_decode_segmentation`), data.transforms | — | `HFDataset.__getitem__` (134 lines, L159) | `data/hf.py:166` imports `_build_text_prompts` from `data/coco.py`; `data/hf.py:206` imports `_decode_segmentation` from `data/coco.py` — intra-module within `data/`; no cross-subdir violation |
| `src/custom_sam_peft/data/transforms.py` | Albumentations train/eval transform pipelines; `resolve_normalization` pulls stats from `AutoImageProcessor`. | data.coco, data.hf | config.schema | — | — | — |
| `src/custom_sam_peft/diagnostics.py` | Cheap-to-run environment audit: GPU, optional deps, SAM weights, HF auth → `DoctorReport`. | cli.doctor_cmd | config.schema (via `ModelConfig()`) | — | — | — |
| `src/custom_sam_peft/eval/__init__.py` | Empty eval package marker. | — | — | — | — | — |
| `src/custom_sam_peft/eval/evaluator.py` | `Evaluator.evaluate`: forward-pass loop over dataset → COCO predictions → `compute_coco_map`; also `evaluate_and_save`. | train.trainer, eval.runner | config.schema, data.base, eval.metrics, eval.postprocess | — | `evaluate` (98 lines, L119); `_compute_per_example_iou` (52 lines, L218) | — |
| `src/custom_sam_peft/eval/metrics.py` | `compute_coco_map` via pycocotools: overall mAP + per-class AP; `MetricsReport` dataclass. | eval.evaluator, runs.bundle, tracking (TYPE_CHECKING) | — | — | — | — |
| `src/custom_sam_peft/eval/postprocess.py` | Pure postprocess: model outputs → COCO results entries (score, bbox, RLE mask). | eval.evaluator | — | — | `queries_to_coco_results` (76 lines, L51) | — |
| `src/custom_sam_peft/eval/runner.py` | `run_eval` end-to-end pipeline: load model/adapter, build dataset, run `Evaluator`. | cli.eval_cmd, cli.run_cmd | _registry, config.schema, data.base, eval.evaluator, eval.metrics, models.sam3, peft_adapters.lora | — | `run_eval` (80 lines, L50) | `eval/runner.py:19` imports `load_lora` from `peft_adapters/lora.py` — **cross-module** (eval→peft_adapters); no documented seam |
| `src/custom_sam_peft/models/__init__.py` | Empty models package marker. | — | — | — | — | — |
| `src/custom_sam_peft/models/losses.py` | Training losses: `mask_loss` (Dice+BCE), `box_loss` (L1+GIoU), `objectness_loss`, `presence_loss`, `total_loss`. | train.loop | config.schema, data.base, models.matching | — | — | — |
| `src/custom_sam_peft/models/matching.py` | `meta_to_canonical` key-name adapter + `HungarianMatcher` (scipy linear_sum_assignment). | models.losses | data.base | — | — | — |
| `src/custom_sam_peft/models/sam3.py` | SAM 3.1 loader (`load_sam31`), wrapper (`Sam3Wrapper`, `_Sam3ImageAdapter`), and the entire `_patch_*` wall (7 patch functions). | cli.export_cmd, cli.run_cmd, eval.runner, train.runner, peft_adapters.lora, peft_adapters.qlora | config.schema, data.base, utils.huggingface | — | `load_sam31` (150 lines, L1054); `_patch_addmm_act_grad_safe` (101 lines, L691); `_patch_roi_align_dtype` (91 lines, L421); `_patch_text_pool_dtype` (87 lines, L571); `_patch_forward_grounding_skip_matching_on_none_target` (85 lines, L794); `_patch_mha_input_dtype` (85 lines, L881); `_patch_module_input_dtype` (69 lines, L984); `_patch_pos_enc_dtype` (56 lines, L363) | — |
| `src/custom_sam_peft/notebook_helpers.py` | Env detection, local-checkpoint check, and HF-token resolution for the Colab notebook. | notebooks (not src) | — | — | — | — |
| `src/custom_sam_peft/peft_adapters/__init__.py` | Empty peft_adapters package marker (documented seam — exempt from cross-module check). | _bootstrap, eval.runner | — | — | — | — |
| `src/custom_sam_peft/peft_adapters/lora.py` | LoRA adapter: `apply_lora`, `save_lora`, `load_lora`, `merge_lora`, `SCOPE_TARGETS`, `@register("peft","lora")`. | _bootstrap, eval.runner, train.checkpoint, peft_adapters.qlora | _registry, config.schema, models.sam3 | — | — | — |
| `src/custom_sam_peft/peft_adapters/qlora.py` | QLoRA adapter: 4-bit quantization + LoRA injection, `apply_qlora`, `save_qlora`, `load_qlora`, `@register("peft","qlora")`. | _bootstrap, train.checkpoint | _registry, config.schema, models.sam3, peft_adapters.lora | — | `apply_qlora` (124 lines, L159) | — |
| `src/custom_sam_peft/presets.py` | VRAM-tier preset table: `pick_preset()` and `preset_label()` for GPU-aware config patching in the notebook. | notebooks (not src) | — | — | — | — |
| `src/custom_sam_peft/runs/__init__.py` | Empty runs package marker. | — | — | — | — | — |
| `src/custom_sam_peft/runs/bundle.py` | `write_bundle`: picks sample indices, re-infers, renders overlays, writes `summary.md` + PNGs. | cli.run_cmd | data.base, eval.metrics | — | `write_bundle` (117 lines, L262) | — |
| `src/custom_sam_peft/tracking/__init__.py` | `build_tracker` factory and `flatten_metrics_report` helper; re-exports `Tracker` protocol. | train.runner | _registry, config.schema, tracking.base | — | — | — |
| `src/custom_sam_peft/tracking/base.py` | `Tracker` Protocol definition and `_validate_image` helper. | tracking.*, train.loop, train.trainer | — | — | — | — |
| `src/custom_sam_peft/tracking/noop.py` | `NoopTracker`: drops all calls; `@register("tracker","none")`. | _bootstrap, tracking.__init__ | _registry, config.schema | — | — | — |
| `src/custom_sam_peft/tracking/tensorboard.py` | `TensorBoardTracker`: writes TensorBoard event files; `@register("tracker","tensorboard")`. | _bootstrap, tracking.__init__ | _registry, config.schema, tracking.base | — | — | — |
| `src/custom_sam_peft/tracking/wandb.py` | `WandBTracker`: wraps wandb.init/log/finish with resume-id persistence; `@register("tracker","wandb")`. | _bootstrap, tracking.__init__ | _registry, config.schema, tracking.base | — | — | — |
| `src/custom_sam_peft/train/__init__.py` | Empty train package marker. | — | — | — | — | — |
| `src/custom_sam_peft/train/checkpoint.py` | Save/load full training state (adapter + optimizer + scheduler + RNG); dispatches LoRA vs QLoRA by `Linear4bit` presence or JSON marker. | train.trainer | config.schema, models.sam3, peft_adapters.lora, peft_adapters.qlora | — | — | `train/checkpoint.py:27-28` imports `load_lora`/`save_lora`/`merge_lora` from `peft_adapters/lora.py` and `load_qlora`/`save_qlora` from `peft_adapters/qlora.py` — **cross-module** (train→peft_adapters); no documented seam |
| `src/custom_sam_peft/train/loop.py` | `train_step` (per-batch class-vocab loop, NaN-skip, grad accum) and `run_epoch` (cadence: log, checkpoint, eval). | train.trainer | config.schema, data.base, models.losses, models.sam3, tracking.base | — | `train_step` (88 lines, L74) | — |
| `src/custom_sam_peft/train/runner.py` | `run_training`: build datasets + model + PEFT + tracker + `Trainer`; returns `RunResult`. | cli.train_cmd, cli.run_cmd | _registry, config.schema, data.base, models.sam3, tracking, train.trainer | — | — | — |
| `src/custom_sam_peft/train/trainer.py` | `Trainer.fit`: run-dir setup, DataLoader construction, optimizer/scheduler build, epoch loop, final eval/export. | train.runner | config.schema, data.base, data.collate, eval.evaluator, eval.metrics, models.sam3, tracking.base, train.checkpoint, train.loop, train.visualize | — | `Trainer.fit` (126 lines, L155) | `train/trainer.py:20-21` imports from `eval/` — **cross-module** (train→eval); no seam for trainer↔evaluator hand-off |
| `src/custom_sam_peft/train/visualize.py` | Pure `render_mask_panel`: returns (H,3W,3) uint8 strip of image/GT/pred panels for `log_images`. | train.trainer | — | — | — | — |
| `src/custom_sam_peft/utils/__init__.py` | Empty utils package marker. | — | — | — | — | — |
| `src/custom_sam_peft/utils/huggingface.py` | `resolve_hf_token` + `download_model` (snapshot_download wrapper with error mapping). | models.sam3, cli.init_cmd | — | — | — | — |

---

## Section B — PEFT method-string leak inventory

**Grep command used:** `rg -n '\.method\s*[=!]=' src/custom_sam_peft/ --type py`

### Hits (4 total, confirming the 4 known leaks from the spec)

**Group 1 — Autocast context selection (train/loop.py)**

```
src/custom_sam_peft/train/loop.py:66:    if cfg.peft.method == "qlora":
```

Context: `_autocast_ctx(cfg)` returns `nullcontext()` for QLoRA (no outer autocast allowed due to sam3's internal `autocast(enabled=False)` regions) and a real `torch.autocast` scope for LoRA. The entire function is a per-method branch.

**Proposed replacement:** `peft_method_instance.disables_outer_autocast() -> bool` — protocol method that returns `True` for QLoRA, `False` for LoRA. The caller becomes `if not peft.disables_outer_autocast(): use_autocast(...)`.

---

**Group 2 — Optimizer selection (train/trainer.py)**

```
src/custom_sam_peft/train/trainer.py:49:    return "adamw8bit" if cfg.peft.method == "qlora" else "adamw"
```

Context: `_resolve_optimizer_name(cfg)` maps `"auto"` to `"adamw8bit"` for QLoRA and `"adamw"` otherwise. This is the "recommended optimizer" concept.

**Proposed replacement:** `peft_method_instance.recommended_optimizer() -> str` — returns `"adamw8bit"` for QLoRA, `"adamw"` for LoRA. The caller becomes `return peft.recommended_optimizer() if cfg.train.optimizer == "auto" else cfg.train.optimizer`.

---

**Group 3 — Checkpoint method detection (train/checkpoint.py)**

```
(no direct .method == hit in checkpoint.py at the grep level)
```

Note: `checkpoint.py` dispatches by `Linear4bit`-presence (`_has_linear4bit`) and by JSON-file presence (`_QLORA_META_FILENAME`), not by `cfg.peft.method`. These are structural probes rather than string branches. However, `save_full_state` at L122 writes `"peft_method": cfg.peft.method` and `load_full_state` at L149-151 reads `detected_method = "qlora" if has_qlora_marker else "lora"` and cross-checks against `saved_method`. Both remain `if method == "qlora"` style at the semantic level.

**Proposed replacement:** `peft_method_instance.detect_method_from_checkpoint(adapter_dir: Path) -> str` — inspects the adapter directory and returns the canonical method string (checking for `_QLORA_META_FILENAME`). This removes the inline detection from `load_full_state`.

---

**Group 4 — Checkpoint loading gate (eval/runner.py)**

```
src/custom_sam_peft/eval/runner.py:76:    if model is None and cfg.peft.method != "lora":
```

Context: `run_eval` raises `ValueError` if asked to load a checkpoint for a non-LoRA method (QLoRA disk loading not yet supported). The guard is a method-string branch.

**Proposed replacement:** `peft_method_instance.supports_checkpoint_load_from_disk() -> bool` — returns `True` for LoRA, `False` for QLoRA (until QLoRA disk-load is implemented). The caller becomes `if model is None and not peft.supports_checkpoint_load_from_disk(): raise ValueError(...)`.

---

### Finalized PEFTMethod Protocol Surface

These names are binding for downstream tasks (Task 4.1 and later):

```python
class PEFTMethod(Protocol):
    """Protocol for PEFT adapter implementations registered via @register("peft", ...).

    Trainers, evaluators, and checkpoint code call these methods instead of
    branching on cfg.peft.method strings. Each registered adapter (lora.py,
    qlora.py) must implement this interface.
    """

    def recommended_optimizer(self) -> str:
        """Return the optimizer name to use when cfg.train.optimizer == 'auto'.

        Returns 'adamw8bit' for QLoRA (requires bitsandbytes), 'adamw' for LoRA.
        """

    def disables_outer_autocast(self) -> bool:
        """Return True if outer torch.autocast must NOT be used during training.

        QLoRA returns True: sam3's internal autocast(enabled=False) regions
        produce bf16/fp32 collisions under an outer autocast scope.
        LoRA returns False: outer autocast is safe.
        """

    def detect_method_from_checkpoint(self, adapter_dir: Path) -> str:
        """Inspect adapter_dir and return the canonical method string.

        QLoRA: checks for custom_sam_peft_qlora.json presence → returns 'qlora'.
        LoRA: absence of the JSON marker → returns 'lora'.
        Raises CheckpointError on ambiguous or corrupted state.
        """

    def supports_checkpoint_load_from_disk(self) -> bool:
        """Return True if this method can load a checkpoint from disk without
        a pre-loaded model wrapper.

        LoRA returns True. QLoRA returns False (requires a live wrapper with
        quantized base; disk-only load is deferred to a follow-up PR).
        """
```

**Summary of candidates resolved:**
- `recommended_optimizer() -> str` — kept as-is (covers Group 2).
- `qlora_aware_train_step_hook(...)` — **renamed** to `disables_outer_autocast() -> bool` (cleaner predicate; covers Group 1; the hook wrapper stays in `loop.py` as `_autocast_ctx` but uses the protocol).
- `detect_method_from_checkpoint(ckpt) -> str` — kept, signature narrowed to `(adapter_dir: Path) -> str` (covers Group 3).
- `supports_checkpoint_load_from_disk() -> bool` — **new** (audit found Group 4 which the spec's three candidates didn't cover).

No `TBD` remains. These four method names and signatures are locked.

---

## Section C — Device-move site inventory

**Grep command:** `rg -n '\.to\(device|\.to\(self\.device|\.cuda\(' src/custom_sam_peft/ --type py`

| file:line | expression | verdict | rationale |
|---|---|---|---|
| `train/loop.py:85` | `batch["images"].to(device)` | **move into `runtime/`** | Batch-level `.to(device)` should be the collator's job (via a `to_device(batch, runtime)` helper); currently duplicated across loop and evaluator. |
| `train/loop.py:107` | `torch.stack([inst.box for inst in targets_c[i]]).to(device)` | **move into `runtime/`** | Per-class box hint tensors are moved inside the training loop; should be handled by the collator or a single `to_device` call. |
| `train/trainer.py:313` | `ex.image.unsqueeze(0).to(device)` | **move into `runtime/`** | Inside `_log_image_panel`; same pattern as `evaluator.py:178`. Consolidate into a shared helper. |
| `models/sam3.py:156` | `h.to(device=device, dtype=torch.float32)` | **stay (model-internal geometry)** | Box hint normalization inside `_build_geometric_prompt`; operates on already-device-resident tensors passed in from the adapter. This is a model-internal tensor op, not a dataset→GPU move. |
| `models/sam3.py:379` (comment only) | `.to(device)` in docstring | not a live call | Documentation reference in `_patch_pos_enc_dtype` comment — no action. |
| `models/sam3.py:1008` (comment only) | `.to(dtype=)` / `.to(device)` in docstring | not a live call | Documentation reference in `_patch_module_input_dtype` comment — no action. |
| `eval/evaluator.py:178` | `ex.image.unsqueeze(0).to(device)` | **move into `runtime/`** | Duplicate of `trainer.py:313`; both should be eliminated by a single `to_device(batch, runtime)` helper called from the collator. |

**Plan:** introduce `runtime.to_device(tensor, runtime: Runtime) -> Tensor` (or `to_device(batch, runtime)` for the full batch dict). The data collator calls it once. The evaluator and trainer receive already-on-device tensors and never call `.to(device)` themselves.

---

## Section D — Path-construction inventory

**Grep commands:**
```bash
rg -n 'runs/.*/checkpoints|os\.path\.join.*checkpoints|Path.*checkpoints' src/custom_sam_peft/ --type py
rg -n '"checkpoints"' src/custom_sam_peft/ --type py
```

| file:line | expression | replacement |
|---|---|---|
| `train/trainer.py:165` | `(run_dir / "checkpoints").mkdir(...)` | `checkpoint_path(run_dir, step).parent.mkdir(...)` — use `paths.checkpoint_path` to derive the checkpoints root |
| `train/trainer.py:208` | `run_dir / "checkpoints" / f"step_{step}"` | `paths.checkpoint_path(run_dir, step)` |
| `train/checkpoint.py:138` | String in error message `<run_dir>/checkpoints/step_N/...` | Update error message to use `checkpoint_path(run_dir, N)` representation |
| `tracking/wandb.py:65` | Comment: `runs/<old>/checkpoints/step_100` finds `wandb_run_id.txt` | No code change needed; update comment to reference `paths.checkpoint_path` |

**Additional path constructions needing `paths/`:**
| file:line | expression | replacement |
|---|---|---|
| `train/trainer.py:163` | `Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"` | `paths.run_dir(cfg.run.output_dir, cfg.run.name, stamp)` |
| `train/runner.py:20` | `Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"` | `paths.run_dir(...)` — duplicate of trainer.py:163 |
| `train/trainer.py:251` | `run_dir / "adapter"` | `paths.artifact_path(run_dir, "adapter")` |
| `train/trainer.py:254` | `run_dir / "merged"` | `paths.artifact_path(run_dir, "merged")` |
| `cli/run_cmd.py:79` | `run_dir / "merged"` | `paths.artifact_path(run_dir, "merged")` |
| `eval/runner.py:104` | `checkpoint.parent` (fallback output dir) | Route through `paths.artifact_path` |
| `runs/bundle.py:275` | `ctx.run_dir / "samples"` | `paths.artifact_path(ctx.run_dir, "samples")` |
| `runs/bundle.py:378` | `ctx.run_dir / "summary.md"` | `paths.artifact_path(ctx.run_dir, "summary.md")` |

Named functions to add to `src/custom_sam_peft/paths/__init__.py`:
- `run_dir(output_dir, name, stamp) -> Path`
- `checkpoint_path(run_dir, step) -> Path`
- `artifact_path(run_dir, name) -> Path`
- `predictions_path(run_dir, split) -> Path`
- `bundle_path(run_dir) -> Path`

---

## Section E — Config field-use census

Fields in all Pydantic config classes under `src/custom_sam_peft/config/`. "hits" means any YAML key or Python attribute reference in the target location. Tests column excludes schema validation tests; includes integration test instantiation.

| class.field | hits in configs/examples/ | hits in notebooks/ | hits in tests/ | non-test hits in src/ | YAGNI verdict |
|---|---|---|---|---|---|
| `RunConfig.name` | 4 | 1 | many | many | keep |
| `RunConfig.output_dir` | 4 | 0 | many | many | keep |
| `RunConfig.seed` | 4 | 0 | many | many | keep |
| `ModelConfig.name` | 4 | 1 | many | many | keep |
| `ModelConfig.local_dir` | 4 | 0 | many | many | keep |
| `ModelConfig.checkpoint_file` | 4 | 0 | many | many | keep |
| `ModelConfig.revision` | 0 | 0 | 1 | 2 | keep-advanced |
| `ModelConfig.gradient_checkpointing` | 4 | 1 (via preset) | some | many | keep |
| `ModelConfig.dtype` | 4 | 1 (via preset) | many | many | keep |
| `ModelConfig.device` | 0 | 0 | 1 | 1 | keep-advanced |
| `DataSplit.annotations` | 4 | 0 | many | 1 (loader) | keep |
| `DataSplit.images` | 4 | 0 | many | 1 (loader) | keep |
| `AugmentationsConfig.hflip` | 4 | 0 | some | 1 | keep |
| `AugmentationsConfig.color_jitter` | 4 | 0 | some | 1 | keep |
| `TextPromptConfig.mode` | 2 | 0 | some | 2 | keep |
| `TextPromptConfig.negatives_per_image` | 2 | 0 | some | 1 | keep-advanced |
| `TextPromptConfig.k` | 0 | 0 | some | 1 | keep-advanced |
| `NormalizeConfig.mean` | 2 | 0 | some | 1 | keep-advanced |
| `NormalizeConfig.std` | 2 | 0 | some | 1 | keep-advanced |
| `HFFieldMap.image` | 0 | 0 | some | 1 | keep-advanced |
| `HFFieldMap.bbox` | 0 | 0 | some | 1 | keep-advanced |
| `HFFieldMap.category` | 0 | 0 | some | 1 | keep-advanced |
| `HFFieldMap.segmentation` | 0 | 0 | some | 1 | keep-advanced |
| `HFFieldMap.categories_feature` | 0 | 0 | some | 1 | keep-advanced |
| `HFFieldMap.bbox_format` | 0 | 0 | some | 1 | keep-advanced |
| `HFDatasetConfig.name` | 0 | 0 | some | 1 | keep-advanced |
| `HFDatasetConfig.split_train` | 0 | 0 | some | 1 | keep-advanced |
| `HFDatasetConfig.split_val` | 0 | 0 | some | 1 | keep-advanced |
| `HFDatasetConfig.field_map` | 0 | 0 | some | 1 | keep-advanced |
| `DataConfig.format` | 4 | 1 | many | many | keep |
| `DataConfig.train` | 4 | 0 | many | many | keep |
| `DataConfig.val` | 4 | 0 | many | many | keep |
| `DataConfig.test` | 0 | 0 | some | 1 | keep-advanced |
| `DataConfig.hf` | 0 | 0 | some | 2 | keep-advanced |
| `DataConfig.prompt_mode` | 4 | 0 | many | many | keep |
| `DataConfig.image_size` | 4 | 0 | many | many | keep |
| `DataConfig.augmentations` | 4 | 0 | many | 2 | keep |
| `DataConfig.text_prompt` | 2 | 0 | some | 2 | keep |
| `DataConfig.normalize` | 2 | 0 | some | 2 | keep |
| `QLoRAConfig.quant_type` | 0 | 0 | some | 3 | keep-advanced |
| `QLoRAConfig.compute_dtype` | 0 | 0 | some | 3 | keep-advanced |
| `PEFTConfig.method` | 4 | 1 | many | many | keep |
| `PEFTConfig.r` | 4 | 1 | many | many | keep |
| `PEFTConfig.alpha` | 2 | 0 | many | many | keep |
| `PEFTConfig.dropout` | 2 | 0 | many | many | keep |
| `PEFTConfig.scope` | 2 | 0 | many | many | keep |
| `PEFTConfig.target_modules` | 0 | 0 | some | 2 | keep-advanced |
| `PEFTConfig.bias` | 0 | 0 | some | 2 | keep-advanced |
| `PEFTConfig.qlora` | 0 | 0 | some | 2 | keep-advanced |
| `MatcherWeights.lambda_l1` | 0 | 0 | some | 1 | demote |
| `MatcherWeights.lambda_giou` | 0 | 0 | some | 1 | demote |
| `MatcherWeights.lambda_mask` | 2 | 0 | some | 1 | keep-advanced |
| `BoxHintSchedule.p_start` | 4 | 0 | some | 1 | keep |
| `BoxHintSchedule.p_end` | 4 | 0 | some | 1 | keep |
| `BoxHintSchedule.decay_steps` | 4 | 0 | some | 1 | keep |
| `BoxHintSchedule.early_stop_p_threshold` | 2 | 0 | some | 0 | demote (future early-stopper reads it but nothing in active src uses it) |
| `LossConfig.w_mask` | 2 | 0 | some | 1 | keep-advanced |
| `LossConfig.w_box` | 0 | 0 | some | 1 | demote |
| `LossConfig.w_obj` | 2 | 0 | some | 1 | keep-advanced |
| `LossConfig.w_presence` | 2 | 0 | some | 1 | keep-advanced |
| `LossConfig.matcher_weights` | 2 | 0 | some | 1 | keep-advanced |
| `LossConfig.focal_gamma` | 0 | 0 | some | 1 | demote |
| `LossConfig.focal_alpha` | 0 | 0 | some | 1 | demote |
| `TrainHyperparams.epochs` | 4 | 0 | many | many | keep |
| `TrainHyperparams.batch_size` | 4 | 1 (via preset) | many | many | keep |
| `TrainHyperparams.grad_accum_steps` | 4 | 1 (via preset) | many | many | keep |
| `TrainHyperparams.optimizer` | 4 | 0 | many | many | keep |
| `TrainHyperparams.lr` | 4 | 0 | many | many | keep |
| `TrainHyperparams.lr_schedule` | 2 | 0 | many | many | keep |
| `TrainHyperparams.warmup_steps` | 4 | 0 | many | many | keep |
| `TrainHyperparams.max_grad_norm` | 2 | 0 | some | 1 | keep-advanced |
| `TrainHyperparams.eval_every` | 2 | 0 | some | 1 | keep-advanced |
| `TrainHyperparams.save_every` | 4 | 0 | many | 1 | keep |
| `TrainHyperparams.loss` | 2 | 0 | some | 1 | keep-advanced |
| `TrainHyperparams.box_hint` | 4 | 0 | some | 2 | keep |
| `TrainHyperparams.log_every` | 4 | 0 | some | 1 | keep |
| `TrainHyperparams.nan_abort_after` | 2 | 0 | some | 1 | keep-advanced |
| `TrainHyperparams.num_workers` | 2 | 0 | some | 1 | keep-advanced |
| `EvalConfig.metrics` | 2 | 0 | some | 0 | demote (field exists in schema but `compute_coco_map` ignores it; metrics are hardcoded in `metrics.py`) |
| `EvalConfig.iou_thresholds` | 2 | 0 | some | 2 | keep-advanced |
| `EvalConfig.mode` | 0 | 0 | some | 3 | keep-advanced |
| `EvalConfig.lite_max_images` | 0 | 0 | some | 1 | keep-advanced |
| `EvalConfig.mask_threshold` | 0 | 0 | some | 1 | keep-advanced |
| `EvalConfig.save_predictions` | 0 | 0 | some | 2 | keep-advanced |
| `WandbConfig.project` | 0 | 0 | some | 1 | keep-advanced |
| `WandbConfig.entity` | 0 | 0 | some | 1 | keep-advanced |
| `TrackingConfig.backend` | 4 | 0 | many | many | keep |
| `TrackingConfig.wandb` | 0 | 0 | some | 1 | keep-advanced |
| `ExportConfig.merge` | 0 | 0 | some | 2 | keep-advanced |

**YAGNI demote candidates (4):**
- `MatcherWeights.lambda_l1` — no config sets it; hardcoded `0.0`; move to a constant in `LossConfig`.
- `MatcherWeights.lambda_giou` — same; hardcoded `0.0`; move to a constant.
- `BoxHintSchedule.early_stop_p_threshold` — no active src consumer; spec §2 says "future early-stopping mechanism."
- `EvalConfig.metrics` — listed in 2 configs but the field is currently not read by `compute_coco_map`; metrics computed are hardcoded.

**Demote candidates (5):**
- `LossConfig.w_box` — `0.0` in defaults, `0.0` in all examples; no non-test src reads it except `total_loss`.
- `LossConfig.focal_gamma` — no config sets it; `2.0` default; demote to constant.
- `LossConfig.focal_alpha` — no config sets it; `0.25` default; demote to constant.
- `MatcherWeights.lambda_l1` / `lambda_giou` — see above.

---

## Section F — Field rename table

Based on audit findings:

| old name | new name | rationale |
|---|---|---|
| `train.lr` | `train.learning_rate` | "lr" is an abbreviation; "learning_rate" matches the concept and is self-documenting. Both currently exist as `lr` only — rename to `learning_rate`. |
| `train.batch_size` | `train.batch_size` | **No rename.** Audit found `batch_size` is consistent throughout schema, configs, tests, and notebooks. The spec's "watch for" list included `batch_size` vs `train_batch_size`; audit finds only `batch_size` is present — no conflict. |
| `run.output_dir` | `run.output_dir` | **No rename.** Consistent everywhere. |
| `tracking.wandb.project` | `tracking.wandb.project` | **No rename needed.** Spec listed `wandb_project` vs `tracking.wandb.project` — audit finds the nested form is already used; no flat `wandb_project` key exists. |
| `config/loader.py::ConfigError` | move to `errors.py::ConfigError` | Currently defined locally in `loader.py`; must move to central `errors.py` taxonomy per §4.4. |

**Note:** The `lr` → `learning_rate` rename is the only field rename surfaced. The schema already uses the nested structure (`tracking.wandb.project`, `train.batch_size`) that the spec flagged as potential inconsistencies; the inconsistency was pre-existing in field name style (`lr` abbreviation) rather than structural nesting.

---

## Section G — Pydantic-vs-dataclass per internal sub-config (OQ2)

Decision rule: Pydantic iff (a) enum fields, (b) constrained ints/floats with bounds, OR (c) ≥3 end-user-set fields. Dataclass otherwise.

| class | enum fields? | constrained fields? | ≥3 end-user-set? | verdict | rationale |
|---|---|---|---|---|---|
| `RunConfig` | No | No | Yes (name, output_dir, seed) | **Pydantic** | 3 user-set fields; top-level |
| `ModelConfig` | Yes (dtype, device is `str\|None`) | No | Yes (name, local_dir, dtype, ...) | **Pydantic** | enum `Dtype` field; ≥3 user-set |
| `DataSplit` | No | Yes (min_length=1 on both) | No (2 fields only) | **Pydantic** | constrained fields |
| `AugmentationsConfig` | No | Yes (color_jitter: ge=0 le=1) | No | **Pydantic** | constrained float |
| `TextPromptConfig` | Yes (TextPromptMode) | Yes (negatives_per_image: ge=0; k: ge=1, le=16) | Yes | **Pydantic** | enum + constrained fields |
| `NormalizeConfig` | No | Yes (mean/std validated in model_validator) | No (2 fields) | **Pydantic** | constrained via validator |
| `HFFieldMap` | Yes (bbox_format: Literal["xywh","xyxy"]) | No | No (internal, no common user-setting) | **Pydantic** | enum field |
| `HFDatasetConfig` | No | Yes (name: min_length=1) | No (internal) | **Pydantic** | constrained field |
| `DataConfig` | Yes (format, prompt_mode) | Yes (image_size: PositiveInt) | Yes | **Pydantic** | enum + constrained + ≥3 user-set |
| `QLoRAConfig` | Yes (quant_type, compute_dtype) | No | No (rarely user-set) | **Pydantic** | enum fields |
| `PEFTConfig` | Yes (method, scope, bias) | Yes (r,alpha: PositiveInt; dropout: ge=0, lt=1) | Yes | **Pydantic** | enum + constrained + ≥3 user-set |
| `MatcherWeights` | No | Yes (ge=0 bounds) | No (0-1 user-set in practice) | **→ dataclass** | no enum; constrained but bounds only matter internally; rarely set by users |
| `BoxHintSchedule` | No | Yes (ge=0, le=1 on p fields; PositiveInt) | Yes (p_start, p_end, decay_steps) | **Pydantic** | constrained + ≥3 user-set |
| `LossConfig` | No | Yes (PositiveFloat; ge=0 on focal_alpha) | No (rarely user-set; mostly defaults) | **→ dataclass** | no enum; constrained but almost all fields are never set by users; promote to `_internal.py` |
| `TrainHyperparams` | Yes (optimizer, lr_schedule) | Yes (PositiveInt, PositiveFloat, ge=0) | Yes (epochs, batch_size, lr, ...) | **Pydantic** | enum + constrained + ≥3 user-set |
| `EvalConfig` | Yes (mode: EvalMode) | Yes (iou_thresholds) | No (rarely set by users) | **Pydantic** | enum field |
| `WandbConfig` | No | No | No (rarely set) | **→ dataclass** | no enum, no constraints, rarely set by user; promote to `_internal.py` |
| `TrackingConfig` | Yes (backend: TrackerBackend) | No | No (backend + nested) | **Pydantic** | enum field |
| `ExportConfig` | No | No | No (1 field) | **→ dataclass** | no enum, no constraints, 1 field |

**Classes to promote to dataclasses (move to `config/_internal.py`):**
- `MatcherWeights` — demote to dataclass; used only internally in `LossConfig`.
- `LossConfig` — demote to dataclass; almost all fields are internal constants.
- `WandbConfig` — demote to dataclass; 2 rarely-set fields, no validation needed.
- `ExportConfig` — demote to dataclass; 1 boolean field.

---

## Section H — Cross-module reach-through findings

Consolidated from Section A column:

| finding | files | proposed fix |
|---|---|---|
| **H1** `cli → eval`: `eval_cmd.py` imports `run_eval` from `eval.runner` | `cli/eval_cmd.py:13` → `eval/runner.py` | Acceptable pattern once §5.4 "CLI commands are thin wrappers over `run_*` library functions" lands — `run_eval` becomes a documented public API. No fix needed beyond §5.4. |
| **H2** `cli → models/train/runs/eval`: `run_cmd.py` imports across 4 modules | `cli/run_cmd.py:20-29` → `eval/runner.py`, `models/sam3.py`, `runs/bundle.py`, `train/checkpoint.py`, `train/runner.py` | After §5.4 all of these become `run_*` library calls. `cli/run_cmd.py` should call `run_training(cfg)`, `run_eval(cfg)`, `write_bundle(ctx)` and nothing else. The inner imports (`load_sam31`, `load_adapter`, `save_merged`) should move behind `run_*` facades. |
| **H3** `cli → models`: `export_cmd.py` imports `load_sam31`, `load_adapter`, `save_adapter`, `save_merged` | `cli/export_cmd.py:13-14` → `models/sam3.py`, `train/checkpoint.py` | Extract `run_export(cfg, checkpoint, ...)` library function; CLI shell calls it. |
| **H4** `cli → utils`: `init_cmd.py` imports `download_model` from `utils.huggingface` | `cli/init_cmd.py:17` → `utils/huggingface.py` | Acceptable; `utils.huggingface` is a utility, not a domain module with a seam boundary. No fix required. |
| **H5** `eval → peft_adapters`: `eval/runner.py` imports `load_lora` directly | `eval/runner.py:19` → `peft_adapters/lora.py` | Route through `train.checkpoint.load_adapter(wrapper, path)` which already dispatches. `eval/runner.py` should call `load_adapter`, not `load_lora` directly. |
| **H6** `train → eval`: `train/trainer.py` imports `Evaluator` and `MetricsReport` from `eval/` | `train/trainer.py:20-21` → `eval/evaluator.py`, `eval/metrics.py` | This is the trainer↔evaluator seam violation. Fix: introduce `EvalArtifacts` value object (§5.3). Trainer stops constructing `Evaluator` internally; instead returns `EvalArtifacts`; caller (`train/runner.py` or `cli/run_cmd.py`) constructs `Evaluator`. |
| **H7** `train → peft_adapters`: `train/checkpoint.py` imports from both `lora.py` and `qlora.py` | `train/checkpoint.py:27-28` → `peft_adapters/lora.py`, `peft_adapters/qlora.py` | `checkpoint.py` already partially abstracts via `load_adapter`/`save_adapter`. After §5.1 PEFTMethod protocol lands, the remaining inline dispatch (`_has_linear4bit`, `_QLORA_META_FILENAME`) moves behind `peft_method.detect_method_from_checkpoint()`. |

---

## Section I — Dead-code candidates

**Methodology:** `find_large_functions_tool` and structural analysis; `find_dead_code` and `refactor_tool` were not invoked via MCP (the `refactor_tool` schema was not available; analysis done by reading all files).

| candidate | location | evidence | verdict |
|---|---|---|---|
| `_autocast_ctx` in `train/loop.py` | L65 | Only called inside `train_step`; will be replaced by `peft_method.disables_outer_autocast()` protocol call | Delete after §5.1 refactor; not dead today |
| `_resolve_optimizer_name` in `train/trainer.py` | L45 | Only called in `Trainer.__init__`; will move to `peft_method.recommended_optimizer()` | Delete after §5.1; not dead today |
| `_has_linear4bit` in `train/checkpoint.py` | L44 | Used to dispatch `save_adapter`; after §5.1 protocol lands, dispatch moves to `peft_method` | Delete after §5.1; not dead today |
| `EvalConfig.metrics` field | `config/schema.py:252` | Field exists but `compute_coco_map` never reads it; the metrics computed are hardcoded in `metrics.py`. The list in examples (`[mAP, mAP_50, mAP_75, per_class_AP]`) has no effect. | **Delete field** (YAGNI); demote to internal constant or inline in `compute_coco_map` |
| `render_mask_panel` in `train/visualize.py` | L31 | Only called in `Trainer._log_image_panel`; well-tested; legitimate code | Keep |
| `Tracker.is_primary` / `world_size` | Not yet implemented (seam scaffolding) | §2 discipline — `is_primary` and `world_size` will be on `Runtime`, not `Tracker`, per §4.3 | **Retain as seam scaffolding** per §10 — do not delete |
| `ResumeState` dataclass in `train/checkpoint.py` | L37 | Used in `load_full_state` return + `Trainer.fit` — live code | Keep |
| `_ScalarWindow` in `train/loop.py` | L164 | Used in `run_epoch` for windowed scalar logging — live code | Keep |
| `_bracket_label` in `runs/bundle.py` | L233 | Only called in `write_bundle` — live code | Keep |
| `BundleContext` in `runs/bundle.py` | L39 | Passed from `cli/run_cmd.py` — live code | Keep |
| `detect_env` / `check_local_checkpoint` / `resolve_hf_token` in `notebook_helpers.py` | — | Used by the Colab notebook; no src imports | Keep (notebook API) |
| `presets.py::pick_preset` / `preset_label` | — | Used by the Colab notebook; no src imports | Keep (notebook API) |
| `reset_registry` in `_registry.py` | L49 | Test-only helper; clearly documented | Keep (test infrastructure) |
| `flatten_metrics_report` in `tracking/__init__.py` | L40 | Not called anywhere in `src/` (only tested directly) | **Dead-code candidate** — no src callers found; confirm before deletion |

**§2 seam scaffolding confirmed retained:**
- `Runtime.is_primary: bool` — will exist when `runtime/` module is created; must not be removed.
- `Runtime.world_size: int` — same; always 1 today but the field is required for DDP-safe future.

---

## Section J — Items deferred to follow-up issues

| description | proposed GitHub issue title | label |
|---|---|---|
| J1. `EvalConfig.metrics` field is present in schema and example configs but is ignored by `compute_coco_map`; metrics computed are hardcoded. Remove field and hardcode, or wire it up. | "hardening: remove or wire EvalConfig.metrics field (currently silently ignored)" | `hardening-followup` |
| J2. `MatcherWeights.lambda_l1` and `.lambda_giou` are exposed as config fields but always `0.0` in every example; box supervision is deferred (v0 is mask-only). Demote to internal constants. | "hardening: demote MatcherWeights.lambda_l1/lambda_giou to internal constants (box supervision deferred)" | `hardening-followup` |
| J3. `LossConfig.focal_gamma` and `.focal_alpha` are configurable but no user ever sets them; `2.0`/`0.25` are the effective constants. Demote to internal constants or embed in `objectness_loss`. | "hardening: demote LossConfig.focal_gamma/focal_alpha to internal constants (never set by users)" | `hardening-followup` |
| J4. `BoxHintSchedule.early_stop_p_threshold` exists for a future early-stopping mechanism with no active consumer in `src/`. Retain field but add a note that it is unused until the early-stopper lands. Issue to track eventual consumption. | "hardening: track early_stop_p_threshold consumption in early-stopping implementation" | `hardening-followup` |
| J5. `flatten_metrics_report` in `tracking/__init__.py` has no callers in `src/`; only tested directly. Either add a caller (e.g., in `run_eval` output path) or delete. | "hardening: wire or delete flatten_metrics_report (no src callers)" | `hardening-followup` |
| J6. `models/sam3.py` `_patch_*` wall is 7 functions totalling ~600 lines in a single file. Each patch has a clear rationale and a "re-evaluate every sam3 version bump" note; they should be one file per patch under `models/_patches/` as specified in §4.3 / §6. This is in-scope for the PR but warrants a follow-up issue to track the sam3 version-bump re-evaluation cadence. | "hardening: establish sam3-version-bump checklist for _patches/* re-evaluation" | `hardening-followup` |
| J7. `utils/huggingface.py` and `notebook_helpers.py` both define HF token resolution (`resolve_hf_token`), each with slightly different semantics (one is for the library, one for the notebook). Consolidate or document the distinction. | "hardening: consolidate HF token resolution (utils vs notebook_helpers duplication)" | `hardening-followup` |
| J8. `eval/runner.py` rebuilds the model and loads the adapter from disk for standalone eval, but does not yet support QLoRA disk-load. `supports_checkpoint_load_from_disk()` is the seam. Implement QLoRA disk-load to close the gap. | "hardening: implement QLoRA checkpoint load from disk in eval/runner.py" | `hardening-followup` |
| J9. `presets.py` and `notebook_helpers.py` have zero `src/` inbound deps; they are notebook-only. Consider whether they belong under `src/` or should live under `notebooks/` as non-installed helpers. | "hardening: evaluate moving presets.py and notebook_helpers.py to notebooks/ (not installed)" | `hardening-followup` |

**Total deferred items: 9**

---

*Inventory generated 2026-05-21 by hardening-pass audit subagent.*
