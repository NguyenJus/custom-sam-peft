# spec/cli — CLI Wiring & Diagnostics Design

**Status:** Draft (2026-05-18)
**Architecture step:** 8 of 9 (see `2026-05-15-esam3-architecture-design.md` §11)
**Scope:** Wire `esam3 train`, `esam3 export`, `esam3 init`, `esam3 doctor` to the implemented library. Extract a thin runner layer so CLI files contain no training/eval logic. Refactor `esam3 eval` for consistency (behavior preserved). Out of scope: tracker backend implementations (spec/tracking), GPU smoke test (spec/smoke-test), `coco-bbox` / `hf-text` templates (deferred).

---

## 1. Current State

| Command | State |
|---|---|
| `esam3 train`   | Stub — parses config, prints "not yet implemented". |
| `esam3 eval`    | Implemented (PR #17). Loads cfg + COCO dataset + LoRA, runs `Evaluator.evaluate_and_save`. Wiring inlined in `cli/eval_cmd.py`. |
| `esam3 export`  | Stub. |
| `esam3 init`    | Stub — validates `--template` against `{coco-text, coco-bbox, hf-text}` but writes nothing. |
| `esam3 doctor`  | Stub. |
| `cli/main.py`   | Typer app wired; all five commands registered. |

Existing tests (`tests/unit/test_cli.py`) assert the "not yet implemented" strings for `train` and `doctor`; those two cases are rewritten by this spec.

---

## 2. Goals & Non-Goals

**Goals.**

- Every CLI command parses config → calls a library function → prints a result. CLI files contain no training/eval/export logic.
- The library functions are independently usable from notebooks and scripts.
- `esam3 doctor` reports environment health without importing heavy/optional deps.
- `esam3 init` produces a config that is valid against `TrainConfig` after the user edits the data paths.
- Plugin registry (`data`, `peft`, `tracker`) is bootstrapped once at CLI entry; no per-command lazy imports.

**Non-goals.**

- Implementing tracker backends (`tensorboard.py`, `wandb.py` still raise `NotImplementedError`; spec/tracking owns them). The `none` backend works today and is the safe default for CLI tests.
- Adding `coco-bbox` or `hf-text` templates — bbox training is rejected by `Trainer.__init__` and the HF dataset adapter is unimplemented in both train and eval. Both are appended to `logs/TODO.md`.
- Live progress bars, global `--json` mode (only `doctor` gets `--json`), `esam3 serve` / `esam3 infer` / `esam3 run-from-checkpoint`.
- GPU integration coverage (the existing `tests/integration/test_train_end_to_end.py` already exercises the library; CLI tests verify wiring only).

---

## 3. Module Layout

```
src/esam3/
  _bootstrap.py            # NEW — imports every @register site once
  diagnostics.py           # NEW — run_doctor() -> DoctorReport
  train/
    runner.py              # NEW — run_training(cfg, *, resume_from) -> RunResult
  eval/
    runner.py              # NEW — run_eval(cfg, ...) -> MetricsReport
  cli/
    templates/             # NEW
      coco_text_lora.yaml  # NEW — placeholder-path version of configs/examples/
      coco_text_qlora.yaml # NEW — placeholder-path version of configs/examples/
    main.py                # CHANGED — `import esam3._bootstrap` at module load
    train_cmd.py           # REWRITTEN — thin shell over run_training
    eval_cmd.py            # CHANGED — thin shell over run_eval
    export_cmd.py          # REWRITTEN
    init_cmd.py            # REWRITTEN
    doctor_cmd.py          # REWRITTEN — formats DoctorReport
```

**Boundary rules (carried from architecture-design §3):**

- CLI files contain only: argument parsing, config loading, calling a library function, formatting output, and translating library exceptions to exit codes. Aim for ≤30 lines per command.
- Library runners (`train/runner.py`, `eval/runner.py`) are the seam other entry points (notebooks, future Ray Train driver) will call. They take `TrainConfig` in; they do not parse YAML or know about Typer.

---

## 4. Trainer Refactor (`run_dir` Lift)

`Trainer.fit` currently computes `run_dir = output_dir / f"{name}-{timestamp}"` internally. The runner needs `run_dir` before constructing the tracker (so TensorBoard / W&B get a per-run log dir). Lift the computation.

**New helper** (`esam3/train/runner.py`):

```python
def make_run_dir(cfg: TrainConfig) -> Path:
    """Compute and create runs/{name}-{UTC-timestamp}. Idempotent on retry only
    within the same second; callers pass the returned path to Trainer.fit."""
```

**Trainer change** (`esam3/train/trainer.py`):

```python
def fit(self, *, run_dir: Path, resume_from: Path | None = None) -> RunResult:
    """Use the caller-provided run_dir; do not compute one internally."""
```

The internal block that creates `run_dir/checkpoints/` and writes `config.yaml` stays, but reads from the parameter instead of computing locally. No other call sites construct a `Trainer` (verified: only the runner will), so the signature change is local.

---

## 5. Public Library API

### 5.1 `esam3.train.runner`

```python
def run_training(
    cfg: TrainConfig,
    *,
    resume_from: Path | None = None,
) -> RunResult:
    """End-to-end: build datasets → load + adapt model → build tracker →
    Trainer.fit → return RunResult.

    Raises:
        NotImplementedError: cfg.data.format != "coco" (only COCO supported in v0).
        ValueError: cfg.data.prompt_mode == "bbox" (bubbles from Trainer.__init__;
            callers should pre-check for a friendlier error).
    """
```

**Implementation outline (no actual code in spec — that's the plan's job):**

1. `run_dir = make_run_dir(cfg)`.
2. Build train + val datasets via the registry: `build = lookup("dataset", cfg.data.format)`; `train_ds = build(cfg.data.model_dump(), model_name=cfg.model.name, pipeline="train")`; same with `pipeline="eval"` for val. This works for both `"coco"` and `"hf"` because both are `@register("dataset", ...)` today; no per-format conditionals in the runner.
3. `model = load_sam31(cfg.model)`; `lookup("peft", cfg.peft.method)(model, cfg.peft)`.
4. `tracker = build_tracker(cfg.tracking, run_dir)` — a tiny dispatcher that looks up the registered factory and supplies the right kwargs per backend (TB needs `log_dir`; W&B needs `project`/`entity`; noop needs nothing).
5. `Trainer(model, train_ds, val_ds, tracker, cfg).fit(run_dir=run_dir, resume_from=resume_from)`.

### 5.2 `esam3.eval.runner`

```python
def run_eval(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: Literal["val", "test"] = "val",
    output_dir: Path | None = None,
    save_predictions: bool | None = None,
) -> MetricsReport:
    """Load model + adapter, build dataset, run Evaluator.evaluate_and_save.

    Raises:
        ValueError: cfg.peft.method != "lora" (current limitation; matches existing
            behavior in cli/eval_cmd.py).
        ValueError: split == "test" and cfg.data.test is None.
    """
```

This is a near-verbatim move of `_run_eval` from `cli/eval_cmd.py`, with two upgrades:

- Dataset construction switches from the hardcoded COCO branch to the same registry lookup used by `run_training` (`lookup("dataset", cfg.data.format)`), so `format: hf` now works end-to-end through eval as well as train. The current `NotImplementedError("Only 'coco' is currently implemented")` is deleted.
- The TODO comment "add HF dataset support (out of scope for Task 6)" in `cli/eval_cmd.py` is resolved.

CLI translates `ValueError` to `typer.BadParameter` to preserve exit-code parity with the current tests.

### 5.3 `esam3.diagnostics`

```python
@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    capability: tuple[int, int]
    total_mib: int
    free_mib: int

@dataclass(frozen=True)
class WeightsInfo:
    path: Path
    exists: bool
    size_bytes: int | None  # None if not exists

@dataclass(frozen=True)
class DoctorReport:
    python_version: str               # sys.version.split()[0]
    platform: str                     # platform.platform()
    torch_version: str
    cuda_build: str | None            # torch.version.cuda
    cuda_available: bool
    gpus: list[GpuInfo]
    optional_deps: dict[str, str | None]   # "bitsandbytes", "wandb", "tensorboard"
    core_versions: dict[str, str]          # "peft", "transformers", "sam3"
    sam3_weights: WeightsInfo
    issues: list[str]                      # human-readable warnings

def run_doctor(*, weights_path: Path | None = None) -> DoctorReport:
    """Cheap-to-run environment audit. No GPU-touching imports."""
```

**Implementation rules:**

- `importlib.util.find_spec` for optional-dep presence (does not import bitsandbytes). Version via `importlib.metadata.version` only if present.
- `torch.cuda.mem_get_info(i)` for VRAM (already requires `torch.cuda.is_available()`).
- Default `weights_path` is `Path(ModelConfig().local_dir) / ModelConfig().checkpoint_file` — equals `models/sam3.1/sam3.1_multiplex.pt` today.
- `issues` collects soft warnings: Python < 3.12, `cuda_available is False`, `sam3_weights.exists is False`, missing `[qlora]` extra when bitsandbytes-relevant configs are detected (skip the last one in v0 — doctor doesn't read configs).

### 5.4 `esam3._bootstrap`

```python
"""Import every @register site once so the registry is populated.

CLI entry imports this module; library callers (notebooks) may import it
explicitly if they need plugin lookup."""

from esam3.data import coco              # noqa: F401  (registers "data", "coco")
from esam3.peft_adapters import lora, qlora  # noqa: F401
from esam3.tracking import noop, tensorboard, wandb  # noqa: F401
```

No `__all__`; the side effect is the point. `cli/main.py` adds `import esam3._bootstrap` at module load (top-level, not inside a function — registry must be ready before any command runs).

---

## 6. Command Behavior

### 6.1 `esam3 train`

```
esam3 train --config PATH [--override key=val]... [--resume PATH] [-v|--verbose]
```

Body (target ≤30 lines, excluding imports):

1. `_configure_logging(verbose)` — `logging.basicConfig(level=DEBUG if verbose else INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")`.
2. `cfg = load_config(config, overrides=override)`.
3. Pre-flight: if `cfg.data.prompt_mode == "bbox"`, raise `typer.BadParameter("prompt_mode='bbox' is not supported for training in v0 ...", param_hint="--config")`.
4. `result = run_training(cfg, resume_from=resume)`.
5. Print run_dir, adapter path, and (if present) `final_metrics.overall["mAP"]`.

**Errors / exit codes:**

| Source | Mapping |
|---|---|
| `ConfigError` from `load_config` | `typer.BadParameter` → exit 2 |
| `NotImplementedError` for non-COCO format | catch in CLI, `typer.Exit(1)` with rich error |
| `ValueError` for bbox mode (defensive double-guard) | catch, `typer.Exit(1)` |
| Unknown exception | propagate (traceback preserved for bug reports) |

### 6.2 `esam3 eval`

```
esam3 eval --config PATH --checkpoint PATH [--split val|test] [--output PATH]
           [--save-predictions/--no-save-predictions]
```

CLI shrinks to a thin wrapper that calls `run_eval`. Behavior identical to today — including the existing `BadParameter` for `--split test` when `data.test is None` and for `peft.method != "lora"`. All current tests in `tests/unit/test_cli.py` must still pass without modification.

### 6.3 `esam3 export`

```
esam3 export --checkpoint PATH [--merge] [--output PATH] [--config PATH]
```

**Config resolution.** If `--config` given, use it; record the parent dir as `run_dir`. Else walk up from `--checkpoint` parents looking for a sibling `config.yaml` (Trainer writes one to `run_dir`). Stop at the filesystem root. If none found, raise `typer.BadParameter("could not auto-discover config.yaml; pass --config", param_hint="--config")`. The discovered `config.yaml`'s parent becomes `run_dir`.

**`--checkpoint` accepts:** any directory that `load_adapter` accepts — i.e., either `run_dir/adapter/` or `run_dir/checkpoints/step_N/`. `load_adapter` (defined in `train/checkpoint.py`) dispatches LoRA vs QLoRA by file presence and is self-sufficient when `wrapper.peft_model is None` — it sets up the PEFT structure itself; **do not call `apply_lora`/`apply_qlora` first**.

**Output rules:**

| Mode | Default `--output` | Explicit `--output` |
|---|---|---|
| no `--merge` (re-emit adapter) | **required** (BadParameter if missing — refuse to clobber source) | use as given |
| `--merge`                     | `run_dir / "merged"` (sibling of `adapter/`, regardless of whether `--checkpoint` pointed at `adapter/` or `checkpoints/step_N/`) | use as given |

**Pipeline:**

1. Resolve config + `run_dir`.
2. `wrapper = load_sam31(cfg.model)`; `load_adapter(wrapper, checkpoint)`. No separate `apply_*` call.
3. If `--merge`: `save_merged(wrapper, output)`.
4. Else: `save_adapter(wrapper, output)`.
5. Print the written path.

QLoRA merged exports dequantize via the existing `save_merged` → `merge_lora` path. No new code needed.

### 6.4 `esam3 init`

```
esam3 init [--template coco-text-lora|coco-text-qlora] [--output ./config.yaml] [--force]
```

- Templates ship as files under `src/esam3/cli/templates/`. Resolved at runtime via `importlib.resources.files("esam3.cli.templates")`.
- Template content = the existing `configs/examples/*.yaml` with concrete paths replaced by placeholders:
  - `data.train.annotations: data/train.json` → user-edit token
  - `data.train.images: data/train/`
  - `data.val.annotations: data/val.json`
  - `data.val.images: data/val/`
  - Leading comment block: "Edit the paths in the `data:` section, then run `esam3 train --config <this file>`."
- Default `--template` value changes from the current `coco-bbox` (which never had a config) to `coco-text-lora`.
- If `--output` exists and `--force` not set: `typer.BadParameter("refusing to overwrite ...; pass --force")`.
- Print the written path on success.

**Validation guard.** After writing the template, the test suite calls `load_config(written_path, overrides=["data.train.annotations=<tmp>", ...])` and asserts no exception — i.e., the placeholders are the only required edits.

### 6.5 `esam3 doctor`

```
esam3 doctor [--weights-path PATH] [--json]
```

- Calls `diagnostics.run_doctor(weights_path=...)`.
- Default output: rich `Table` with sections **Runtime**, **GPU**, **Optional deps**, **Core versions**, **SAM3.1 weights**, **Issues**.
- `--json` outputs `json.dumps(dataclasses.asdict(report), default=str, indent=2)`.
- Exit code: always 0 (informational). A future `--strict` flag is out of scope.

---

## 7. Cross-Cutting

| Concern | Decision |
|---|---|
| Bootstrap | `cli/main.py` has `import esam3._bootstrap  # noqa: F401` at module top. |
| Error model | `ConfigError` / argument-style errors → `typer.BadParameter` (exit 2). Library `ValueError` / `NotImplementedError` from known-bad states → catch in CLI and `typer.Exit(1)` with `rich.print` of the message. Unknown exceptions propagate. |
| Logging | `_configure_logging(verbose)` in `cli/_logging.py` (new shared helper, used by `train`, `eval`, `export`). Trainer & runners use Python `logging`; CLI uses `rprint` only for terminal-facing summaries. |
| `--override` parsing | Already in `load_config`; no change. |
| Tracker construction | `build_tracker(cfg.tracking, run_dir)` lives in `train/runner.py` (not in `tracking/`, to keep that module focused on the protocol + backends). It looks up `lookup("tracker", cfg.tracking.backend)` and dispatches the kwargs. |

---

## 8. Testing

All tests CPU-only. Tier: unit, in `tests/unit/`.

### 8.1 Existing tests to keep / rewrite

| Test | Action |
|---|---|
| `test_root_help_exits_zero` ... `test_init_help_exits_zero` | Keep unchanged. |
| `test_train_with_valid_config_prints_not_implemented` | **Rewrite** — monkeypatch `esam3.train.runner.run_training` to return a fake `RunResult`; assert exit 0 and `"run_dir="` in stdout. |
| `test_doctor_runs_and_prints_not_implemented` | **Rewrite** — assert exit 0 and `"torch"` in stdout (table mode). |
| `test_eval_command_with_split_test_missing_data_test` | Keep — behavior preserved by runner extraction. |
| `test_eval_command_save_predictions_flag_parses` | Keep — flag still parses; CLI now delegates to `run_eval`. |
| `test_eval_command_rejects_qlora_method` | Keep — `run_eval` raises `ValueError`; CLI translates to `BadParameter`. |

### 8.2 New tests

| Test | Mechanism |
|---|---|
| `test_train_rejects_bbox_prompt_mode` | Build cfg with `prompt_mode: bbox`; assert exit ≠ 0 and `"bbox"` in stderr. |
| `test_export_auto_discovers_config` | Create `tmp/run_dir/config.yaml` + `tmp/run_dir/adapter/`; monkeypatch `save_adapter` / `load_adapter` / `load_sam31` / `lookup`; invoke without `--config`; assert success. |
| `test_export_no_merge_requires_output` | Without `--merge` and without `--output`: assert `BadParameter`. |
| `test_export_merge_default_output` | With `--merge` only: assert output defaults to `{run_dir}/merged/`. |
| `test_export_config_not_found` | `--checkpoint` in a dir without `config.yaml` upstream and no `--config`: assert `BadParameter`. |
| `test_init_writes_template_lora` / `_qlora` | Write to `tmp_path / "config.yaml"`; reload via `load_config` (after override-substituting the four data paths to existing files); assert no exception. |
| `test_init_refuses_clobber` | Pre-create target → exit ≠ 0; with `--force` → exit 0. |
| `test_init_unknown_template` | `--template hf-text` → BadParameter (since hf-text isn't shipped). |
| `test_doctor_report_populated` | `run_doctor()` → assert `python_version`, `torch_version`, `optional_deps` keys present. |
| `test_doctor_json_round_trips` | `esam3 doctor --json` → `json.loads(stdout)` returns a dict with expected keys. |
| `test_doctor_weights_path_override` | `--weights-path <tmp/fake.pt>` (file exists) → `sam3_weights.exists` is True in the JSON output. |
| `test_bootstrap_populates_registry` | Call `reset_registry()`; import `esam3._bootstrap`; assert `list_registered("data")`, `("peft")`, `("tracker")` are non-empty. |

Coverage gate (80% on `src/esam3`) is preserved; new modules each have direct unit tests as above.

---

## 9. Deferred (Appended to `logs/TODO.md`)

- `coco-bbox` init template — depends on a future bbox-training spec.
- `hf-text` init template — `format: hf` runs end-to-end today (data/hf.py is registered), but no shipped starter config exercises it. A template (and a tiny fixture-backed test that loads it) is deferred.
- `esam3 doctor --strict` non-zero exit on critical issues.
- HF cache scan in doctor.

---

## 10. Exit Criteria

- All five `esam3 ...` commands exit 0 with valid input and produce the expected artifact / report.
- `tests/unit/test_cli.py` passes (existing + new tests above).
- `cli/*_cmd.py` files each ≤30 lines body (imports + function definition).
- `ruff check`, `mypy --strict`, `pytest` clean.
- No new entries in `logs/TODO.md` beyond §9.
