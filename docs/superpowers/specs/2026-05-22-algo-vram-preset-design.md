# Algorithmic VRAM-tier PEFT preset + per-step OOM auto-retry

**Issue:** [#36 — Algorithmically derive PEFT preset from VRAM tier (replace lookup table)](https://github.com/NguyenJus/custom-sam-peft/issues/36)
**Release:** `v0.7.0` (pre-1.0 breaking → minor bump)
**Status:** locked design, single PR, no back-compat shims.

The current `presets.py` hard-codes four VRAM tiers (11 / 16 / 40 / 80 GiB) in `_tier_for_gb` and emits a fixed `(method, r, batch_size, …)` tuple per tier. That table cannot reason about non-standard cards (24/48 GiB), nor about non-1024 image sizes, nor about the actual cost of LoRA at different ranks. This spec replaces the table with an analytic memory model plus an optional one-shot calibration probe, and folds in per-step OOM auto-retry so that estimator misses do not crash long runs. Both pieces ship together: the preset chooser produces aggressive-but-safe configs and the trainer is the safety net.

---

## §1 Scope & non-goals

### In scope

| File | Change |
|------|--------|
| `src/custom_sam_peft/presets.py` | Internals rewritten. Public surface reduced to `decide_preset()` + `PresetDecision`. `pick_preset()`, `preset_label()`, `_tier_for_gb` removed. |
| `src/custom_sam_peft/cli/calibrate_cmd.py` | **New.** Implements `custom_sam_peft calibrate` subcommand. |
| `src/custom_sam_peft/cli/main.py` | +1 line registering `calibrate` subcommand on the existing Typer app. |
| `src/custom_sam_peft/train/trainer.py` (+ `runner.py`) | OOM retry ladder inside step loop; microbatching; `oom_events` propagated through `run_training`. |
| `src/custom_sam_peft/train/types.py` | **New** module. Houses the `OomEvent` frozen dataclass. |
| `src/custom_sam_peft/runs/bundle.py` | `BundleContext.preset` becomes required `PresetDecision`; `oom_events` field added; `## Preset` block rendered structurally; one new `edge_note` builder for OOM retries. |
| `notebooks/custom_sam_peft_train.ipynb` | New CALIBRATE cell between SETUP and FORM. GENERATE cell rewritten to use `decide_preset()` + sidecar `preset.json`. |
| `tests/unit/test_presets.py` | Rewritten end-to-end. |
| `tests/unit/test_calibrate_cmd.py` | **New.** Heavy-mock CLI test. |
| `tests/unit/test_trainer_oom_retry.py` | **New.** Synthetic OOM injection through the step loop. |
| `tests/gpu/test_calibrate_real.py` | **New.** One GPU-marked sanity test on real activation bytes. |
| `pyproject.toml`, `uv.lock` | Version bump `0.6.0 → 0.7.0`. |

### Out of scope

- No new quantization beyond bnb NF4 / int8 (still only LoRA + QLoRA).
- No dynamic LoRA-rank downgrade during OOM retry — that would invalidate optimizer state.
- No change to `model.dtype` — bf16 is fixed throughout.
- No mid-run dataloader recreation — microbatching stays inside the step.
- No eval-time OOM auto-retry — train-step-only in v1; eval OOM still surfaces as today.

---

## §2 Architectural approach

Two stages, neither of which talks to the other except through `PresetDecision` on disk and `oom_events` in the run result.

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                       PREDICT (before training)                      │
   └────────────────────────────────────────────────────────────────────┘

 notebook CALIBRATE cell        ──── or ────       power-user terminal
   !custom_sam_peft calibrate                       $ custom_sam_peft calibrate
            │                                                  │
            ▼                                                  ▼
   src/custom_sam_peft/cli/calibrate_cmd.py
     load SAM 3.1 + LoRA r=4 stub                ── runs once per (gpu, image_size)
     forward+backward, batch=1, ckpt=off, bf16
     activation = max_memory_allocated() − (model + opt + adapter)
     write ./.custom_sam_peft_calibration.json
            │
            ▼  reads cache if present, else uses analytic constants
 notebook GENERATE cell:  decision = decide_preset(image_size=1008)
                          patch    = decision.config_patch
                          Path("preset.json").write_text(decision.to_json())
            │
            ▼
   custom_sam_peft run --config config.yaml  (subprocess; existing flow)
   run_cmd.py: read preset.json → BundleContext.preset

   ┌────────────────────────────────────────────────────────────────────┐
   │                       RECOVER (during training)                      │
   └────────────────────────────────────────────────────────────────────┘

 train/trainer.py step loop:
   try:
     forward → backward → optimizer.step()
   except torch.cuda.OutOfMemoryError:
     torch.cuda.empty_cache()
     if micro_batch_size > 1:
        micro_batch_size //= 2                  ── ladder rung 1
        oom_events.append(OomEvent("microbatch_halved", step, …))
        retry
     elif not gradient_checkpointing:
        gradient_checkpointing = True            ── ladder rung 2
        oom_events.append(OomEvent("grad_ckpt_enabled", step, …))
        retry
     else:
        raise RuntimeError(diagnostic)           ── ladder rung 3

 runner.run_training returns oom_events in the run result.
 bundle.py renders `- OOM retries: N (final micro_batch=M, …)` in `## Edge cases`.
```

The PREDICT stage is quality-first: it walks a discrete search space (LoRA over QLoRA, then maximize rank, then batch, then prefer ckpt-off) and picks the largest configuration that fits. The RECOVER stage is the safety net for cases where the analytic estimate was wrong (no calibration cache, fragmentation, kernel selection differences). Together they replace the lookup table without sacrificing safety.

---

## §3 Algorithm: memory model + search

### Budget

```
budget_bytes = torch.cuda.get_device_properties(0).total_memory − headroom_bytes
```

Headroom defaults to **1.0 GiB**. Override via env var `CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB` (float). Non-numeric or negative override raises `RuntimeError` immediately at `decide_preset()` entry.

### Per-candidate prediction

```
model_bytes(method)
    = MODEL_PARAMS × 2          # bf16, LoRA — frozen base in bf16
    | MODEL_PARAMS × 0.5 + Q_OVERHEAD  # NF4, QLoRA — 4-bit base + per-block scales

adapter_bytes(r)
    = LORA_LAYERS × r × (D_IN + D_OUT) × 2     # bf16 adapter weights

optimizer_bytes(r)
    = adapter_bytes(r) × 4      # AdamW state on trainable params, fp32 moments
                                # adapter is bf16 (2B/param); state is 8B/param → 4×

activation_bytes(image_size, batch, ckpt)
    = ACTIVATION_PER_EXAMPLE(image_size) × batch × (CKPT_FACTOR if ckpt else 1.0)
                                # CKPT_FACTOR = 0.3 (rule-of-thumb sqrt(num_layers))

predicted_bytes = model_bytes + adapter_bytes + optimizer_bytes
                + activation_bytes + WORKSPACE_BYTES
                                # WORKSPACE_BYTES = 256 MiB constant for cuDNN +
                                # autograd graph + tmp buffers
```

### Constants

The following names live at module level in `presets.py` as UPPER_CASE constants, each with a one-line comment naming its source (e.g., `# from SAM 3.1 checkpoint inspection, 2026-05-22`). The planner subagent fills in the numeric values during implementation — this spec only fixes the contract:

- `MODEL_PARAMS` — total parameter count of the SAM 3.1 base checkpoint.
- `LORA_LAYERS` — count of attention projection layers that receive a LoRA adapter under the project's default `target_modules`.
- `D_IN`, `D_OUT` — input and output feature dimensions averaged across LoRA target layers (used only for adapter byte estimate; exact per-layer sizing is over-engineering for a budget check).
- `Q_OVERHEAD` — per-block fp16 scale + zero-point overhead for bnb NF4 quantization.
- `WORKSPACE_BYTES` — 256 MiB; covers cuDNN workspace, autograd graph, tmp buffers.
- `CKPT_FACTOR` — 0.3; reduction factor on activation memory when `gradient_checkpointing=True`.
- `BASE_ACTIVATION_AT_1024` — analytic-fallback activation cost per example at `image_size=1024`, bf16, LoRA, ckpt-off.

### `ACTIVATION_PER_EXAMPLE(image_size)` resolution order

1. **Calibration cache** — read `./.custom_sam_peft_calibration.json`. Use its `activation_bytes_per_example` if and only if `(gpu_name, image_size, sam3_checkpoint_sha)` all match the current run.
2. **Analytic fallback** — `BASE_ACTIVATION_AT_1024 × (image_size / 1024)²`. Activations scale roughly with token count, which scales with pixel count.

Whichever path is used drives `PresetDecision.provenance` (`"calibrated"` vs `"analytic"`).

### Search space

Exhaustive enumeration over:

- `method ∈ {"lora", "qlora"}` (2)
- `r ∈ {8, 16, 24, 32, 48, 64}` (6)
- `batch ∈ 1..16` (16)
- `ckpt ∈ {False, True}` (2)

= **384 candidates**. Pure arithmetic per candidate, < 1 ms total. No need for a smart solver.

### Sort key

```python
key = (
    0 if c.method == "lora" else 1,     # LoRA before QLoRA (quality)
    -c.r,                                # higher rank first
    -c.batch,                            # bigger batch first
    0 if not c.ckpt else 1,              # ckpt-off before ckpt-on
)
```

Filter to feasible (`predicted_bytes ≤ budget_bytes`), sort by `key`, pick `candidates[0]`.

### No feasible candidate

```
RuntimeError(
    f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
    f"headroom — SAM 3.1 needs ≈{min_needed_gib:.1f} GiB even at QLoRA r=4 "
    f"batch=1 ckpt=on. Use a larger GPU."
)
```

(The function name in the error message intentionally says `pick_preset` because that is the user-facing concept even though the symbol is now `decide_preset` — the message stays grep-stable for ops runbooks.)

### Grad-accum

```python
grad_accum_steps = max(1, 16 // batch_size)
```

Targets an effective batch of 16 regardless of the chosen `batch_size`.

---

## §4 Calibration CLI subcommand

### Surface

```
custom_sam_peft calibrate [--image-size N] [--output PATH] [--force]
```

Defaults:

- `--image-size 1008` — matches both notebook and pyproject template default.
- `--output ./.custom_sam_peft_calibration.json` — relative to cwd.
- `--force` — disables cache-fresh check.

### Procedure (10 steps, in order)

1. Require CUDA. Reuse `_CUDA_HINT` from `presets.py`. Non-CUDA → exit 2 with hint.
2. **Cache-fresh check.** If `--output` exists, `--force` not set, and the cache's `(gpu_name, image_size, sam3_checkpoint_sha)` match the current environment → print `"cache fresh — exiting"`, exit 0.
3. Load SAM 3.1 base in bf16 via `models/sam3.py`.
4. Attach a LoRA stub adapter via `peft_adapters/lora.py` at `r=4`.
5. Build one synthetic batch: input tensor shape `(1, 3, image_size, image_size)`, one text prompt, one GT mask — reuse shape helpers from `tests/fixtures/tiny_sam3_lora_stub.py`.
6. `torch.cuda.reset_peak_memory_stats()`; forward + backward only (no `optimizer.step()`).
7. `peak = torch.cuda.max_memory_allocated()`.
8. Compute:
   ```
   activation_bytes_per_example = peak
                                − model_bytes("lora")
                                − adapter_bytes(4)
                                − optimizer_bytes(4)
                                − WORKSPACE_BYTES
   ```
   Clamp `≥ 0`. Warn (stderr) if the raw value was negative, indicating constants need recalibration.
9. Write the JSON cache atomically (`tmp` + `os.replace`) with `schema_version=1` and these fields:
   - `calibrated_at` (ISO 8601 UTC string)
   - `gpu_name` (`torch.cuda.get_device_name(0)`)
   - `gpu_total_memory_bytes` (`torch.cuda.get_device_properties(0).total_memory`)
   - `image_size`
   - `sam3_checkpoint_sha` (sha256 of the checkpoint file)
   - `torch_version`
   - `custom_sam_peft_version`
   - `activation_bytes_per_example`
   - `peak_memory_bytes_at_probe`
10. Print a 4-line human-readable summary to stdout:
    ```
    GPU:        NVIDIA A100-SXM4-40GB (image_size=1008)
    Peak:       38.2 GiB
    Activation: 0.43 GiB/example
    Cache:      ./.custom_sam_peft_calibration.json
    ```

Failure modes are in §10.

---

## §5 Notebook integration

### New CALIBRATE cell (between SETUP and FORM)

Markdown preamble:

> **CALIBRATE — strongly recommended (~30–60s, one-time per machine).** Probes peak VRAM on this GPU so `decide_preset()` can return a tight, accurate config instead of a conservative analytic estimate. Skip via `CUSTOM_SAM_PEFT_SKIP_CALIBRATE=1` if you know what you are doing.

Cell body (exact):

```python
import os
if os.environ.get("CUSTOM_SAM_PEFT_SKIP_CALIBRATE") != "1":
    !custom_sam_peft calibrate --image-size 1008
else:
    print("CALIBRATE skipped via CUSTOM_SAM_PEFT_SKIP_CALIBRATE=1")
```

### GENERATE cell (rewritten)

Before (current, abbreviated):

```python
from custom_sam_peft.presets import pick_preset, preset_label
preset = pick_preset()
os.environ["CUSTOM_SAM_PEFT_PRESET_LABEL"] = preset_label()
config = deep_merge(template, preset.config_patch)
```

After (new):

```python
from pathlib import Path
from custom_sam_peft.presets import decide_preset

decision = decide_preset(image_size=1008)
config = deep_merge(template, decision.config_patch)
Path("preset.json").write_text(decision.to_json())
print(decision.label())
```

All references to `pick_preset`, `preset_label`, and `CUSTOM_SAM_PEFT_PRESET_LABEL` are removed from the notebook. The sidecar `preset.json` is what the subprocess-spawned `run` command reads (via `run_cmd.py`) to populate `BundleContext.preset`.

---

## §6 Trainer OOM-retry path

### Step-loop pseudocode (in `trainer.py`)

```python
def train_step(state: TrainState, batch: Batch) -> StepResult:
    """One optimizer.step(), possibly multiple microbatches."""
    while True:
        try:
            state.optimizer.zero_grad()
            n_micro = math.ceil(len(batch) / state.micro_batch_size)
            for i in range(n_micro):
                micro = batch.slice(i, state.micro_batch_size)
                loss = _run_step(state.model, micro, state.gradient_checkpointing)
                (loss / (state.grad_accum_steps * n_micro)).backward()
            state.optimizer.step()
            return StepResult(loss=loss.detach(), oom_events=state.pending_oom_events)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if state.micro_batch_size > 1:
                state.micro_batch_size //= 2
                state.pending_oom_events.append(OomEvent(
                    step=state.step,
                    action="microbatch_halved",
                    new_micro_batch_size=state.micro_batch_size,
                    new_gradient_checkpointing=state.gradient_checkpointing,
                ))
                logger.warning(
                    "OOM at step %d — halving micro_batch_size to %d",
                    state.step, state.micro_batch_size,
                )
                continue  # retry this step
            if not state.gradient_checkpointing:
                state.gradient_checkpointing = True
                state.pending_oom_events.append(OomEvent(
                    step=state.step,
                    action="grad_ckpt_enabled",
                    new_micro_batch_size=state.micro_batch_size,
                    new_gradient_checkpointing=True,
                ))
                logger.warning(
                    "OOM at step %d — enabling gradient_checkpointing",
                    state.step,
                )
                continue  # retry this step
            raise RuntimeError(
                f"OOM at step {state.step} after micro_batch=1 + "
                f"gradient_checkpointing=on. Use a larger GPU or smaller image_size."
            )
```

### `_run_step` microbatching helper

```python
def _run_step(model, micro_batch, gradient_checkpointing: bool) -> torch.Tensor:
    """Forward pass for one microbatch. Caller does backward and optimizer.step."""
    if gradient_checkpointing and not model.is_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model(micro_batch).loss
```

### Runner integration

`runner.run_training` accumulates `state.pending_oom_events` across all steps and returns the flat list in the run-result dict under key `"oom_events"`. This is the only additive change to that contract.

### `BundleContext` change

`preset: PresetDecision` is now required (was `preset_label: str | None`). `oom_events: tuple[OomEvent, ...]` is new, also required (no default — callers always pass at least `()`).

### `edge_note` rendering

When `oom_events` is non-empty, `write_bundle` appends one line to the existing `## Edge cases` block:

```
- OOM retries: 3 — final micro_batch=2, gradient_checkpointing enabled at step 412
```

The `gradient_checkpointing enabled at step S` clause is omitted when no `grad_ckpt_enabled` event appears in the list.

### Invariants enforced

- **Microbatch shrink is sticky** — once halved, it stays halved for the rest of the run.
- **`gradient_checkpointing` toggles at most once per run.** Once on, it stays on.
- **`optimizer.zero_grad()` never called mid-microbatch.** It happens once per outer try.
- **Loss division.** Loss is divided by `grad_accum_steps × n_micro` so the effective gradient magnitude is preserved across ladder rungs.
- **Mid-step OOM replay** — when OOM fires inside the microbatch loop, the outer `while True` restarts that step's microbatch loop from `i=0` at the smaller size. `optimizer.step()` only happens at the original `grad_accum_steps` boundary, never mid-replay.

### `OomEvent` shape

```python
@dataclass(frozen=True)
class OomEvent:
    step: int
    action: Literal["microbatch_halved", "grad_ckpt_enabled"]
    new_micro_batch_size: int
    new_gradient_checkpointing: bool
```

Lives in `src/custom_sam_peft/train/types.py` (new module).

---

## §7 Public API & types

### `presets.py` — sole public function

```python
def decide_preset(image_size: int) -> PresetDecision: ...
```

- `image_size` is **required**.
- `pick_preset()`, `preset_label()`, and `_tier_for_gb` are **removed**. No deprecation shims.

### `PresetDecision` dataclass

```python
@dataclass(frozen=True)
class PresetDecision:
    method: Literal["lora", "qlora"]
    r: int
    batch_size: int
    grad_accum_steps: int
    gradient_checkpointing: bool
    dtype: Literal["bfloat16"]
    headroom_bytes: int                       # budget − predicted_bytes(chosen)
    predicted_bytes: int
    budget_bytes: int                         # total VRAM − headroom
    image_size: int
    gpu_name: str
    provenance: Literal["calibrated", "analytic"]
    cache_path: Path | None

    @property
    def config_patch(self) -> dict[str, dict[str, object]]: ...
    def label(self) -> str: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "PresetDecision": ...
```

- `config_patch` returns the existing 3-section shape (`{"model": {...}, "peft": {...}, "train": {...}}`) so deep-merge into templates keeps working — this is the only stability guarantee.
- `label()` produces a one-line summary:
  ```
  auto: LoRA r=32 batch=2 grad_accum=8 ckpt=off bf16 — fits in 38.4/40.0 GiB on NVIDIA A100-SXM4-40GB (calibrated 2026-05-22)
  ```
  When `provenance == "analytic"`, the suffix is `(analytic estimate)` instead of `(calibrated YYYY-MM-DD)`.
- `to_json()` / `from_json()` are the disk contract for the sidecar `preset.json`. Schema is the dataclass fields verbatim, JSON-encoded, with `Path` rendered as string and `cache_path=None` rendered as `null`.

### `train/types.py` (new module)

Single dataclass: `OomEvent` (see §6). No re-exports from `train/__init__.py` — callers import from `custom_sam_peft.train.types` directly to keep the package's public surface minimal.

### `runner.run_training` result dict

Gains `"oom_events": list[OomEvent]`. This is the only additive change to that contract; no existing keys change shape or semantics.

---

## §8 Bundler integration

### `BundleContext` final shape

```python
@dataclass(frozen=True)
class BundleContext:
    run_dir: Path
    config_path: Path
    start_ts: datetime
    end_ts: datetime
    preset: PresetDecision                  # required (replaces preset_label: str | None)
    per_example_iou: list[float]
    merged_dir: Path | None
    merged_export_error: str | None
    oom_events: tuple[OomEvent, ...]        # required, no default
```

### `## Preset` block (rendered verbatim, structurally)

```
## Preset
- Method: LoRA r=32, batch=2, grad_accum=8, gradient_checkpointing=off, bf16
- GPU:    NVIDIA A100-SXM4-40GB (40.0 GiB)
- Budget: 38.4 / 40.0 GiB used (1.6 GiB headroom)
- Source: calibrated 2026-05-22 (cache: .custom_sam_peft_calibration.json)
```

When `provenance == "analytic"`, the `Source:` line reads:

```
- Source: analytic estimate
```

### Edge-cases line

When `oom_events` is non-empty, append to the existing `## Edge cases` block:

```
- OOM retries: 3 — final micro_batch=2, gradient_checkpointing enabled at step 412
```

The `gradient_checkpointing enabled at step S` clause is omitted when no `grad_ckpt_enabled` event appears in the list. `N` is `len(oom_events)`; `M` is the final `micro_batch_size`.

---

## §9 Testing strategy

Per-project rule: CPU-testable cases live on CPU; GPU tests are reserved for real-only failure modes (bnb quant, real-model state_dict, peak VRAM). Honored here — only **one** new GPU test.

Canonical GPU sizes used in stubs across all CPU tests: **11 / 16 / 40 / 80 GiB** (matches today's tier table for behavioral continuity).

### `tests/unit/test_presets.py` (rewritten)

| Test | Stub GPU | Asserts |
|------|----------|---------|
| `test_decide_preset_11gib_chooses_qlora` | 11 GiB | `method == "qlora"`, `r ∈ {8,16}`, fits budget |
| `test_decide_preset_16gib_chooses_lora_low_rank` | 16 GiB | `method == "lora"`, `r ≤ 32`, `batch ≥ 1` |
| `test_decide_preset_40gib_chooses_lora_high_rank` | 40 GiB | `method == "lora"`, `r ≥ 32`, `batch ≥ 2`, `gradient_checkpointing == False` |
| `test_decide_preset_80gib_chooses_max_rank_batch` | 80 GiB | `r == 64`, `batch == 16` (or near max) |
| `test_decide_preset_unfittable_raises` | stub 4 GiB | `RuntimeError` with `"SAM 3.1 needs"` in message |
| `test_decide_preset_grad_accum_targets_16` | any | `batch_size × grad_accum_steps >= 16` |
| `test_decide_preset_prefers_lora_over_qlora_when_both_fit` | 40 GiB | `method == "lora"` even if QLoRA r=64 also fits |
| `test_decide_preset_image_size_scales_activation` | 40 GiB | predicted_bytes for image_size=2048 > image_size=1024 |
| `test_decide_preset_uses_calibration_cache_when_matching` | 40 GiB, mock cache | `provenance == "calibrated"` |
| `test_decide_preset_ignores_stale_cache` | 40 GiB, cache with wrong sha | `provenance == "analytic"` |
| `test_decide_preset_headroom_env_override` | 40 GiB, env=2.0 | `budget_bytes` = total − 2 GiB |
| `test_decide_preset_headroom_env_invalid_raises` | env="not-a-number" | `RuntimeError` |
| `test_preset_decision_label_calibrated` | — | label suffix is `(calibrated 2026-05-22)` |
| `test_preset_decision_label_analytic` | — | label suffix is `(analytic estimate)` |
| `test_preset_decision_to_json_round_trip` | — | `from_json(to_json(d)) == d` |
| `test_preset_decision_config_patch_3_sections` | — | keys are `{"model","peft","train"}` |

### `tests/unit/test_calibrate_cmd.py` (new)

Heavy-mock — no real model load. Monkeypatch `models.sam3.load`, `peft_adapters.lora.attach`, and `torch.cuda.max_memory_allocated` to deterministic values.

| Test | Asserts |
|------|---------|
| `test_calibrate_writes_cache_with_schema_v1` | output JSON has every field listed in §4 step 9, `schema_version=1` |
| `test_calibrate_cache_fresh_exits_zero` | existing matching cache + no `--force` → exit 0, file not rewritten |
| `test_calibrate_force_overwrites_cache` | `--force` rewrites even when fresh |
| `test_calibrate_non_cuda_exits_2` | `torch.cuda.is_available() == False` → exit 2 |
| `test_calibrate_negative_activation_warns` | mocked peak < model+adapter+opt → stderr warning, clamps to 0 |
| `test_calibrate_atomic_write` | partial-failure mid-write leaves prior cache intact (tmp + replace) |

### `tests/unit/test_trainer_oom_retry.py` (new)

Synthetic OOM injection: subclass `torch.nn.Module` whose `forward` raises `torch.cuda.OutOfMemoryError` on the first N calls then succeeds. Runs entirely on CPU (the exception type is importable without CUDA).

| Test | Injected | Asserts |
|------|----------|---------|
| `test_oom_first_attempt_halves_microbatch` | 1 OOM, batch=8 | post-step `micro_batch_size == 4`, 1 `OomEvent("microbatch_halved")` |
| `test_oom_multiple_halvings_until_one` | 3 OOMs, batch=8 | post `micro_batch_size == 1`, 3 events |
| `test_oom_after_microbatch_1_enables_ckpt` | 4 OOMs, batch=8 | last event is `"grad_ckpt_enabled"`, `gradient_checkpointing == True` |
| `test_oom_after_ckpt_enabled_raises` | 5 OOMs, batch=8 | `RuntimeError("OOM at step …")` |
| `test_oom_microbatch_shrink_is_sticky` | OOM step 1 only | step 2 starts at the shrunk size, not original |
| `test_oom_ckpt_toggle_is_once` | 2 separate OOMs that would each enable ckpt | only one `grad_ckpt_enabled` event in list |
| `test_oom_optimizer_zero_grad_called_once_per_step` | 1 OOM | `optimizer.zero_grad` call count == steps (not microbatches × retries) |
| `test_oom_events_propagated_in_run_result` | any | `run_result["oom_events"]` is a list of `OomEvent` |
| `test_oom_events_serialise_into_bundle_edge_cases` | non-empty list | rendered `## Edge cases` block contains `OOM retries: N` |

### `tests/gpu/test_calibrate_real.py` (new, GPU-marked)

Single test:

| Test | Asserts |
|------|---------|
| `test_calibrate_real_activation_in_sane_range` | `5e8 ≤ activation_bytes_per_example ≤ 1e10` (~0.5–10 GiB/example order-of-magnitude bracket) |

### Tests deleted

These existed against `pick_preset` / `preset_label` and are removed outright — **not adapted**:

- `test_pick_preset_requires_cuda`
- `test_pick_preset_tiers`
- `test_preset_label_format`
- `test_preset_label_with_explicit_total_bytes`

They are replaced by the new `test_decide_preset_*` suite plus bundler tests exercising `PresetDecision.label()`.

---

## §10 Error handling

### `presets.py`

| Condition | Behavior |
|-----------|----------|
| CUDA unavailable | `RuntimeError` reusing `_CUDA_HINT` text from the current module |
| `image_size` ≤ 0 or non-int | `ValueError("image_size must be a positive integer")` |
| `CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB` non-numeric or negative | `RuntimeError("CUSTOM_SAM_PEFT_VRAM_HEADROOM_GIB must be a non-negative float")` |
| No feasible candidate | `RuntimeError` with the exact message from §3 ("GPU has X GiB … SAM 3.1 needs ≈Y GiB … Use a larger GPU.") |
| Calibration cache JSON malformed | log warning, fall through to analytic path (no exception) |
| Calibration cache JSON has `schema_version` ≠ 1 | log warning, fall through to analytic path |

### `calibrate_cmd.py`

| Condition | Behavior |
|-----------|----------|
| CUDA unavailable | exit 2, stderr message reusing `_CUDA_HINT` |
| SAM 3.1 checkpoint not found / load fails | exit 3, stderr names the missing path |
| LoRA stub attach fails | exit 4, stderr names the failure |
| Forward / backward OOM at r=4 batch=1 | exit 5, stderr `"calibration probe OOMed at minimum config — GPU too small"` |
| Cache write fails (disk full / permission) | exit 6, leaves prior cache intact (atomic write) |
| `activation_bytes_per_example` computed as negative | stderr warning, clamp to 0, still write cache |

### Trainer step loop

| Condition | Behavior |
|-----------|----------|
| OOM, `micro_batch_size > 1` | halve, record event, retry |
| OOM, `micro_batch_size == 1`, `gradient_checkpointing == False` | enable ckpt, record event, retry |
| OOM, `micro_batch_size == 1`, `gradient_checkpointing == True` | `RuntimeError("OOM at step N after micro_batch=1 + gradient_checkpointing=on. Use a larger GPU or smaller image_size.")` |
| Non-OOM exception inside step | propagate as-is — no retry, no event |

**Note:** Eval-time OOM is **not** caught by this retry path — matches today's behavior. Folding eval into the safety net is deferred to a future issue.

---

## §11 Migration & no-backwards-compat

**0.7.0 is a breaking release. No shims. Users update.**

### What breaks / who is affected

| What | Who | How they notice |
|------|-----|-----------------|
| `from custom_sam_peft.presets import pick_preset, preset_label` removed | Anyone with custom training scripts importing those symbols | `ImportError` |
| `CUSTOM_SAM_PEFT_PRESET_LABEL` env var contract removed | Anyone setting that env var manually | env var read returns `None`; bundler renders the new structural `## Preset` block instead |
| `BundleContext.preset_label: str \| None` → `BundleContext.preset: PresetDecision` (required) | Anyone constructing `BundleContext` directly (currently only `runner.run_training`) | `TypeError` from dataclass init |
| Notebook GENERATE cell now requires the CALIBRATE cell to have run (or env var skip) | All notebook users | If they skip CALIBRATE: `decide_preset()` uses analytic fallback, `label()` says `(analytic estimate)`; no crash, just less tight |

### User-facing changes (3 numbered steps in release notes)

1. **Re-download the notebook.** The GENERATE cell signature changed and there is a new CALIBRATE cell.
2. **Run `custom_sam_peft calibrate` once per machine.** Optional but strongly recommended — gives a tighter, larger preset.
3. **Update any scripts that imported `pick_preset` or `preset_label`** — switch to `decide_preset(image_size=...)` and use `decision.config_patch` / `decision.label()`.

### Version stamp

- `pyproject.toml`: `version = "0.6.0"` → `"0.7.0"`.
- `uv.lock`: regenerated to match.
- Tag `v0.7.0` after merge (orchestrator handles per close-out 5a).

### Tests deleted (not adapted)

(repeated from §9 for release-notes copy-paste)

- `test_pick_preset_requires_cuda`
- `test_pick_preset_tiers`
- `test_preset_label_format`
- `test_preset_label_with_explicit_total_bytes`

### Rollback story

Revert the `presets.py` rewrite and the trainer step-loop change; that restores the lookup-table behavior and removes the auto-retry safety net. The calibration CLI subcommand and the notebook CALIBRATE cell are additive and harmless — leaving them in place after a partial rollback does no damage (the cache file becomes orphaned but unreferenced). The version stamp (`0.7.0`) would need to be reverted alongside the code to keep semver honest. Full rollback is therefore: revert the PR, regenerate `uv.lock`, retag.
