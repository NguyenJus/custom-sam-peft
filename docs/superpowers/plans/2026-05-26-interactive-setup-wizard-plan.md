# Interactive setup wizard + full gradient-checkpointing removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--interactive` setup wizard to `csp init` that renders one unified comprehensive `config_full.yaml`, and fully remove the abandoned gradient-checkpointing (GC) knob across schema, presets, model, trainer, bundle, docs, and configs — shipped as one PR.

**Architecture:** Two bundled workstreams. WS2 (GC removal) lands first so `PresetDecision.config_patch` is already GC-free before the wizard's VRAM step consumes it. WS1 then authors the unified template, retargets the flag-driven `init`, deletes the two legacy templates, and adds `setup_wizard.py` (declarative `WizardStep` registry + render/validate/emit pipeline reusing `load_config` and `_maybe_download_weights`).

**Tech Stack:** Python 3, Typer + rich (CLI/prompts), Pydantic v2 (schema/validation), `string.Template` (templates), pycocotools (class-imbalance counting), pytest + pytest-cov (TDD, 80% gate), ruff + mypy + markdownlint-cli2 + yamllint (CI gates).

---

## Sequencing rationale (read before starting)

1. **WS2 before WS1's VRAM step.** The wizard's `peft_sizing` step applies `decision.config_patch` verbatim with no stripping logic. That only holds once `config_patch` no longer carries a `gradient_checkpointing` key. So `presets.py` (Phase 2) is edited before `setup_wizard.py`'s VRAM step (Phase 4).
2. **All `presets.py` edits in one task.** `PresetDecision`, `config_patch`, `_candidates`, `_sort_key`, `CKPT_FACTOR`, `_activation_bytes`/`_predicted_bytes`, `label()`, `decide_preset`, `decide_eval_batch_size`, `to_json`/`from_json` share the file — serialize them (Phase 2, Task 2).
3. **Schema field removal first within WS2.** Removing `ModelConfig.gradient_checkpointing` (Phase 1) is the breaking pivot; everything downstream (presets, sam3, loop, types, bundle, configs, docs) aligns to the GC-free schema. Do it first so the `extra="forbid"` failure is visible early.
4. **Unified template authored before legacy deletion.** Author `config_full.yaml` and retarget `run_init` to it (Phase 3, Tasks 7–8) while the two legacy templates still exist, then delete the legacy files (Phase 3, Task 9). This avoids a broken intermediate where `run_init` points at a deleted file.
5. **`setup_wizard.py` after the template exists.** The wizard renders `config_full.yaml`; build the template first (Phase 3) then the wizard (Phase 4), then wire `--interactive` into `init` (Phase 5).
6. **Parallelizable groups** are called out per phase. Within a phase, file-disjoint tasks may run in parallel; same-file or chained tasks are serialized.

### Breaking-change note (NOT a migration step — spec §11)

This PR is a clean breaking change with **no shim and no migration**:

- Any config (including the 7 `configs/examples/*.yaml` shipped in-repo) carrying `model.gradient_checkpointing:` now **fails to load** with a Pydantic "extra fields not permitted" error, because `_Strict` sets `extra="forbid"`. The plan strips that line from the shipped example YAMLs (Task 6) so the repo's own configs still load; downstream users must delete the line from their own YAML themselves.
- `preset.json` sidecars from prior runs no longer deserialize (`from_json` gets an unexpected `gradient_checkpointing` key → `TypeError`); a new run regenerates them.
- `OomEvent.action == "grad_ckpt_enabled"` is never produced; any match on that literal is dead code.

These are consequences to document, not tasks to mitigate.

---

## File structure

**Workstream 2 (GC removal) — modify:**

- `src/custom_sam_peft/config/schema.py` — drop `ModelConfig.gradient_checkpointing`.
- `src/custom_sam_peft/presets.py` — drop GC from `PresetDecision`/`config_patch`/search/sort/label/`decide_preset`/`decide_eval_batch_size`; delete `CKPT_FACTOR`; drop `ckpt` param from `_activation_bytes`/`_predicted_bytes`.
- `src/custom_sam_peft/models/sam3.py` — delete the `if cfg.gradient_checkpointing:` block.
- `src/custom_sam_peft/train/loop.py` — drop GC rung + `OomState.gradient_checkpointing`; fix final raise + docstrings.
- `src/custom_sam_peft/train/types.py` — `OomEvent.action` → single literal; drop `new_gradient_checkpointing`.
- `src/custom_sam_peft/runs/bundle.py` — drop ckpt rendering in `_preset_block` + `_oom_edge_note`.
- `docs/config-schema.md` — drop the `model.gradient_checkpointing` row.
- `configs/examples/*.yaml` (7 files) — strip the `gradient_checkpointing:` line (repo's own configs must still load).

**Workstream 1 (wizard + template) — create/modify/delete:**

- `src/custom_sam_peft/cli/templates/config_full.yaml` — **new** unified `string.Template`.
- `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `coco_text_qlora.yaml` — **deleted**.
- `src/custom_sam_peft/cli/init_cmd.py` — retarget `run_init`/`TEMPLATES` to the unified template; add `--interactive`/`-i` + pre-flight + wizard branch.
- `src/custom_sam_peft/cli/setup_wizard.py` — **new**: `Ctx`, `WizardStep`, `_deep_merge`, prompt primitives, `STEPS`, `run_wizard`, `render`, `validate`, `emit`, `infer_class_imbalance`.

**Tests — create/modify:**

- `tests/unit/cli/__init__.py` — **new** (package marker; the `tests/unit/cli/` dir does not yet exist).
- `tests/unit/cli/test_setup_wizard.py` — **new** (wizard + VRAM-step + class-imbalance cases).
- `tests/unit/test_model_config.py`, `test_presets.py`, `test_decide_eval_batch_size.py`, `test_trainer_oom_retry.py`, `test_train_types.py`, `tests/unit/runs/test_bundle.py`, `test_data_transforms.py`, `test_cli_init.py` — **modified**.
- `tests/integration/test_load_sam31_real.py`, `test_cli_run.py` — **modified**.
- `tests/gpu/test_multiplex_vram.py` — **modified** (only GPU file touched; no new GPU test).

---

## Phase 1 — Schema: remove the GC field (WS2)

### Task 1: Remove `ModelConfig.gradient_checkpointing`

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (the `ModelConfig` class, lines ~115-125)
- Test: `tests/unit/test_model_config.py`

- [ ] **Step 1: Rewrite the model-config test for the removed field**

Replace the GC assertions. In `tests/unit/test_model_config.py`, change `test_model_config_defaults` to drop the GC line and `test_model_config_overrides` to drop the `gradient_checkpointing=False` arg, then add a new test asserting the field is now forbidden:

```python
def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.name == "facebook/sam3.1"
    assert cfg.local_dir == "models/sam3.1"
    assert cfg.checkpoint_file == "sam3.1_multiplex.pt"
    assert cfg.revision is None
    assert cfg.dtype == "bfloat16"
    assert cfg.device is None


def test_model_config_overrides() -> None:
    cfg = ModelConfig(local_dir=None, device="cpu")
    assert cfg.local_dir is None
    assert cfg.device == "cpu"


def test_model_config_rejects_gradient_checkpointing() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(gradient_checkpointing=False)  # type: ignore[call-arg]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_model_config.py -v`
Expected: `test_model_config_rejects_gradient_checkpointing` FAILS (the field is still accepted, no `ValidationError` raised).

- [ ] **Step 3: Remove the field from the schema**

In `src/custom_sam_peft/config/schema.py`, delete these lines from `ModelConfig` (the field plus its `TODO(#60)` comment):

```python
    gradient_checkpointing: bool = (
        False  # TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed
    )
```

So `ModelConfig` becomes:

```python
class ModelConfig(_Strict):
    name: str = "facebook/sam3.1"
    local_dir: str | None = "models/sam3.1"
    checkpoint_file: str = "sam3.1_multiplex.pt"
    dtype: Dtype = "bfloat16"
    # --- advanced ---
    revision: str | None = None
    device: str | None = None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_model_config.py -v`
Expected: PASS (all three tests green; `ModelConfig(gradient_checkpointing=...)` now raises `ValidationError`).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_model_config.py
git commit -m "feat(schema)!: remove ModelConfig.gradient_checkpointing (#60)"
```

---

## Phase 2 — Presets: collapse the ckpt search dimension (WS2)

> Single file (`presets.py`); all sub-edits serialized into one task, then its tests.

### Task 2: Remove GC from `presets.py`

**Files:**

- Modify: `src/custom_sam_peft/presets.py`
- Test: `tests/unit/test_presets.py`, `tests/unit/test_decide_eval_batch_size.py`

- [ ] **Step 1: Rewrite preset tests to the GC-free shape**

In `tests/unit/test_presets.py`:

1. In `test_decide_preset_40gib_chooses_lora_high_rank`, delete the line `assert d.gradient_checkpointing is False`.
2. In `test_decide_preset_unfittable_raises`, keep the `match=r"SAM 3\.1 needs"` (the new message still contains it).
3. In `_make_decision(...)`, delete the `gradient_checkpointing=False,` line from the `PresetDecision(...)` call.
4. In `test_preset_decision_config_patch_3_sections`, delete the line `assert patch["model"]["gradient_checkpointing"] is False` and add `assert "gradient_checkpointing" not in patch["model"]`.
5. In `test_predicted_bytes_train_mode_unchanged`, change both `_predicted_bytes(...)` calls to drop the `ckpt=False` argument:

```python
def test_predicted_bytes_train_mode_unchanged() -> None:
    """Existing train-mode callers stay correct after the ckpt param removal."""
    from custom_sam_peft.presets import _predicted_bytes

    n = _predicted_bytes("lora", r=4, batch=1, image_size=1024, cache=None)
    assert n == _predicted_bytes("lora", r=4, batch=1, image_size=1024, cache=None, mode="train")
```

6. Add a label test asserting no `ckpt=` token:

```python
def test_preset_decision_label_has_no_ckpt_token() -> None:
    d = _make_decision()
    assert "ckpt=" not in d.label()
```

In `tests/unit/test_decide_eval_batch_size.py`, in `test_predicted_bytes_eval_mode_excludes_optimizer_and_adapter`, drop `ckpt=False` from both `_predicted_bytes(...)` calls:

```python
    train_bytes = _predicted_bytes("lora", r=4, batch=1, image_size=1024, cache=None, mode="train")
    eval_bytes = _predicted_bytes("lora", r=4, batch=1, image_size=1024, cache=None, mode="eval")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py -v`
Expected: FAIL — `_make_decision()` passes `gradient_checkpointing=` (still a required field) so collection/construction errors, and `_predicted_bytes(...)` without `ckpt=` errors (TypeError: missing arg).

- [ ] **Step 3: Edit `presets.py` — `PresetDecision` + `config_patch` + `label`**

In `src/custom_sam_peft/presets.py`:

Delete the `gradient_checkpointing: bool` field from the `PresetDecision` dataclass (between `grad_accum_steps: int` and `dtype:`).

Rewrite `config_patch` to drop the `gradient_checkpointing` key:

```python
    @property
    def config_patch(self) -> dict[str, dict[str, object]]:
        """The 3-section dict the deep-merge consumer expects."""
        return {
            "model": {"dtype": self.dtype},
            "peft": {"method": self.method, "r": self.r},
            "train": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
        }
```

Rewrite `label()` to drop the `ckpt` local and the `ckpt={ckpt}` token:

```python
    def label(self) -> str:
        method = method_pretty_name(self.method)
        used_gib = self.predicted_bytes / _GB
        total_gib = (self.budget_bytes + self.headroom_bytes) / _GB
        if self.provenance == "calibrated":
            date_str = self.calibrated_at[:10] if self.calibrated_at else "unknown"
            suffix = f"(calibrated {date_str})"
        else:
            suffix = "(analytic estimate)"
        dtype_token = "fp16" if self.dtype == "float16" else "bf16"
        return (
            f"auto: {method} r={self.r} batch={self.batch_size} "
            f"grad_accum={self.grad_accum_steps} {dtype_token} — "
            f"fits in {used_gib:.1f}/{total_gib:.1f} GiB on {self.gpu_name} {suffix}"
        )
```

`to_json`/`from_json` need no edits (they use `asdict`/`**d`; the field is simply gone).

- [ ] **Step 4: Edit `presets.py` — memory model + search space**

Delete the `CKPT_FACTOR = (...)` constant (lines ~49-51).

Rewrite `_activation_bytes` (drop `ckpt` param + factor):

```python
def _activation_bytes(image_size: int, batch: int, cache: dict[str, Any] | None) -> int:
    per = _activation_per_example(image_size, cache)
    return int(per * batch)
```

Rewrite `_predicted_bytes` (drop `ckpt` param; fix both branches):

```python
def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    image_size: int,
    cache: dict[str, Any] | None,
    mode: Literal["train", "eval"] = "train",
) -> int:
    if mode == "train":
        return (
            _model_bytes(method)
            + _adapter_bytes(r)
            + _optimizer_bytes(r)
            + _activation_bytes(image_size, batch, cache)
            + WORKSPACE_BYTES
        )
    activations = int(_activation_bytes(image_size, batch, cache) * forward_only_factor)
    return _model_bytes(method) + activations + WORKSPACE_BYTES
```

Rewrite `_candidates` (drop the ckpt dimension; return `(method, r, batch)` triples):

```python
def _candidates() -> list[tuple[str, int, int]]:
    methods = ("lora", "qlora")
    rs = (8, 16, 24, 32, 48, 64)
    batches = tuple(range(1, 17))
    return [(m, r, b) for m in methods for r in rs for b in batches]
```

Rewrite `_sort_key` (drop the ckpt tiebreaker):

```python
def _sort_key(c: tuple[str, int, int]) -> tuple[int, int, int]:
    method, r, batch = c
    return (
        0 if method == "lora" else 1,
        -r,
        -batch,
    )
```

- [ ] **Step 5: Edit `presets.py` — `decide_preset` + `decide_eval_batch_size`**

In `decide_preset`, rewrite the feasible loop, the "nothing fits" message, the sort + unpack, and the `PresetDecision(...)` construction:

```python
    feasible = []
    for method, r, batch in _candidates():
        pb = _predicted_bytes(method, r, batch, image_size, cache)
        if pb <= budget:
            feasible.append((method, r, batch, pb))

    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1. Use a larger GPU."
        )

    feasible.sort(key=lambda t: _sort_key(t[:3]))
    method, r, batch, predicted = feasible[0]
    grad_accum = max(1, 16 // batch)

    return PresetDecision(
        method=method,  # type: ignore[arg-type]
        r=r,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        dtype=decided_dtype,
        headroom_bytes=headroom,
        predicted_bytes=predicted,
        budget_bytes=budget,
        image_size=image_size,
        gpu_name=gpu_name,
        provenance=provenance,
        cache_path=cache_path,
        calibrated_at=calibrated_at,
    )
```

In `decide_eval_batch_size`, drop `ckpt=False` from both `_predicted_bytes(...)` calls:

```python
    best_bs = 1
    best_predicted = _predicted_bytes(
        "lora", r=4, batch=1, image_size=image_size, cache=cache, mode="eval"
    )
    for batch in range(1, 65):
        pb = _predicted_bytes(
            "lora", r=4, batch=batch, image_size=image_size, cache=cache, mode="eval"
        )
        if pb <= budget:
            best_bs = batch
            best_predicted = pb
    return best_bs, best_predicted, provenance
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py -v`
Expected: PASS (config_patch has no `gradient_checkpointing`; label has no `ckpt=`; `_predicted_bytes` callers without `ckpt=` work; "nothing fits" message present).

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py
git commit -m "feat(presets)!: collapse ckpt search dimension; drop gradient_checkpointing"
```

---

## Phase 3 — Runtime, types, bundle, configs, docs (WS2)

> Tasks 3, 4, 5, 6 touch disjoint files (`sam3.py`+`test_load_sam31_real.py`; `loop.py`+`types.py`+two test files; `bundle.py`+`test_bundle.py`+`test_cli_run.py`; `configs/examples/*`+`docs/config-schema.md`+`test_data_transforms.py`). Tasks 3, 5, 6 are dependency-free and could run in parallel. Task 4 chains `types.py`→`loop.py` internally. Run after Phase 2.

### Task 3: Remove the GC no-op block in `models/sam3.py`

**Files:**

- Modify: `src/custom_sam_peft/models/sam3.py` (lines ~670-683)
- Test: `tests/integration/test_load_sam31_real.py` (GPU/checkpoint-gated; CPU-collectable)

- [ ] **Step 1: Strip GC from the integration test config constructions**

In `tests/integration/test_load_sam31_real.py`, on each of the three `ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")` calls (lines ~26, ~33, ~56), drop the `gradient_checkpointing=False` argument so each reads `ModelConfig(device="cuda", dtype="bfloat16")`.

- [ ] **Step 2: Run collection to verify it fails**

Run: `uv run pytest tests/integration/test_load_sam31_real.py --collect-only -q`
Expected: before the edit, constructing `ModelConfig(gradient_checkpointing=False)` at import/parametrize time would raise; after Step 1 the file collects but the source still references the field. (This test is checkpoint-gated, so we verify via collection + the unit guard below rather than a real run.)

- [ ] **Step 3: Delete the GC block in `sam3.py`**

In `src/custom_sam_peft/models/sam3.py`, delete the entire block (lines ~670-683):

```python
    if cfg.gradient_checkpointing:
        if hasattr(raw_model, "set_grad_checkpointing"):
            raw_model.set_grad_checkpointing(True)
        else:
            # ... (multi-line comment) ...
            logger.warning(
                "Meta sam3 model has no `set_grad_checkpointing`; "
                "gradient_checkpointing=True is a no-op on this revision."
            )
```

Leave `assert isinstance(raw_model, nn.Module)` and `return raw_model` immediately after. Nothing replaces the block.

- [ ] **Step 4: Run collection + a static grep to verify**

Run: `uv run pytest tests/integration/test_load_sam31_real.py --collect-only -q && ! grep -rn "gradient_checkpointing" src/custom_sam_peft/models/sam3.py`
Expected: collection succeeds AND the grep finds nothing (exit 0 overall because `!` inverts the empty grep).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/integration/test_load_sam31_real.py
git commit -m "refactor(sam3): remove gradient_checkpointing no-op block"
```

### Task 4: Collapse the OOM ladder to 2 rungs (`train/types.py` + `train/loop.py`)

**Files:**

- Modify: `src/custom_sam_peft/train/types.py`
- Modify: `src/custom_sam_peft/train/loop.py` (`OomState` ~55-68; `_train_step_with_oom_ladder` ~91-145)
- Test: `tests/unit/test_train_types.py`, `tests/unit/test_trainer_oom_retry.py`

- [ ] **Step 1: Rewrite `test_train_types.py` to the single-literal action**

Replace the file body's GC-bearing parts. Final test file:

```python
"""Tests for src/custom_sam_peft/train/types.py — frozen dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from custom_sam_peft.train.types import OomEvent


def test_oom_event_is_frozen() -> None:
    ev = OomEvent(step=42, action="microbatch_halved", new_micro_batch_size=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.step = 99  # type: ignore[misc]


def test_oom_event_field_order_and_types() -> None:
    fields = {f.name: f.type for f in dataclasses.fields(OomEvent)}
    assert list(fields) == ["step", "action", "new_micro_batch_size"]


def test_oom_event_only_microbatch_halved_action() -> None:
    ev = OomEvent(step=0, action="microbatch_halved", new_micro_batch_size=1)
    assert ev.action == "microbatch_halved"
```

- [ ] **Step 2: Rewrite `test_trainer_oom_retry.py` to the 2-rung ladder**

In `tests/unit/test_trainer_oom_retry.py`:

1. Remove the `gradient_checkpointing: bool = False` field from the `_State` dataclass.
2. Delete `test_oom_after_microbatch_1_enables_ckpt` and `test_oom_ckpt_toggle_is_once` entirely.
3. Replace `test_oom_after_ckpt_enabled_raises` with `test_oom_after_microbatch_1_raises`:

```python
def test_oom_after_microbatch_1_raises() -> None:
    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=4)  # 3 halvings → mb=1, 4th OOM raises
    with pytest.raises(RuntimeError, match="OOM at step"):
        _train_step_with_oom_ladder(model, _make_batch(8), state, forward_call=_fake_forward_call)
```

4. In `test_oom_first_attempt_halves_microbatch` and `test_oom_multiple_halvings_until_one`, drop any `new_gradient_checkpointing` references (there are none in those bodies — they only read `.action`/`.micro_batch_size`; leave them).
5. In `test_oom_events_serialise_into_bundle_edge_cases`, in the `PresetDecision(...)` literal drop `gradient_checkpointing=False,`, and in the `OomEvent(...)` literal drop `new_gradient_checkpointing=False,` so it reads:

```python
            oom_events=(
                OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4),
            ),
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_train_types.py tests/unit/test_trainer_oom_retry.py -v`
Expected: FAIL — `OomEvent(...)` still requires `new_gradient_checkpointing`; field-order test sees the extra field; the new raise test still hits the GC rung instead of raising at mb=1.

- [ ] **Step 4: Edit `train/types.py`**

Rewrite `OomEvent`:

```python
@dataclass(frozen=True)
class OomEvent:
    """One step where the trainer caught OOM and adapted before retrying.

    `action` records the single adaptive rung:
      - "microbatch_halved": `state.micro_batch_size //= 2`, retry same step.

    The fields capture *post*-adaptation state so downstream rendering
    ("OOM retries: N — final micro_batch=M") can reconstruct the run's
    safety-net history without re-traversing the trainer's mutable state.
    """

    step: int
    action: Literal["microbatch_halved"]
    new_micro_batch_size: int
```

- [ ] **Step 5: Edit `train/loop.py` — `OomState` + the ladder**

In `OomState`, delete the `gradient_checkpointing: bool = False` field and fix the docstring sentence "on OOM the helper mutates `micro_batch_size` / `gradient_checkpointing` in place" → "on OOM the helper halves `micro_batch_size` in place".

In `_train_step_with_oom_ladder`'s docstring, replace the bullet `- gradient_checkpointing toggles at most once per run` with nothing (remove that line).

In the `except torch.cuda.OutOfMemoryError` handler: keep the microbatch-halving branch but drop the `new_gradient_checkpointing=...` kwarg from its `OomEvent(...)`; delete the entire `if not state.gradient_checkpointing:` rung; rewrite the final raise. The handler becomes:

```python
        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if state.micro_batch_size > 1:
                state.micro_batch_size //= 2
                state.pending_oom_events.append(
                    OomEvent(
                        step=state.step,
                        action="microbatch_halved",
                        new_micro_batch_size=state.micro_batch_size,
                    )
                )
                _LOG.warning(
                    "OOM at step %d — halving micro_batch_size to %d",
                    state.step,
                    state.micro_batch_size,
                )
                continue
            raise RuntimeError(
                f"OOM at step {state.step} after micro_batch=1. "
                f"Use a larger GPU or smaller image_size."
            ) from oom_err
```

(`trainer.py` constructs `OomState(micro_batch_size=...)` only and never sets `gradient_checkpointing`, so no trainer edit is needed.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_train_types.py tests/unit/test_trainer_oom_retry.py -v`
Expected: PASS (single-literal action; 2-rung ladder; mb=1 OOM raises with no GC clause; stickiness/zero-grad-once/event-propagation/gradient-magnitude tests still green).

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/train/types.py src/custom_sam_peft/train/loop.py tests/unit/test_train_types.py tests/unit/test_trainer_oom_retry.py
git commit -m "feat(train)!: collapse OOM ladder to 2 rungs; drop grad_ckpt rung"
```

### Task 5: Stop rendering ckpt in `runs/bundle.py`

**Files:**

- Modify: `src/custom_sam_peft/runs/bundle.py` (`_preset_block` ~313-331; `_oom_edge_note` ~334-343)
- Test: `tests/unit/runs/test_bundle.py`, `tests/integration/test_cli_run.py`

- [ ] **Step 1: Rewrite the bundle tests**

In `tests/unit/runs/test_bundle.py`:

1. In `_make_decision()` (~157-173), delete `gradient_checkpointing=False,`.
2. In `test_write_bundle_preset_block_structured`, delete `assert "gradient_checkpointing=off" in summary` and add `assert "gradient_checkpointing" not in summary`.
3. Rewrite `test_write_bundle_oom_edge_note_with_ckpt` → `test_write_bundle_oom_edge_note_multiple_halvings`: replace the 3-event tuple (which used `grad_ckpt_enabled` + `new_gradient_checkpointing`) with two `microbatch_halved` events and assert no GC clause:

```python
def test_write_bundle_oom_edge_note_multiple_halvings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = (
        OomEvent(step=10, action="microbatch_halved", new_micro_batch_size=4),
        OomEvent(step=20, action="microbatch_halved", new_micro_batch_size=2),
    )
    ctx = _make_ctx(tmp_path, per_example_iou=[], oom_events=events)
    monkeypatch.setattr("custom_sam_peft.runs.bundle._reinfer_one_example", _fake_reinfer)
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "OOM retries: 2" in summary
    assert "final micro_batch=2" in summary
    assert "gradient_checkpointing" not in summary
```

4. In `test_write_bundle_oom_edge_note_no_ckpt`, drop `new_gradient_checkpointing=False,` from the single `OomEvent(...)` literal; keep the `assert "gradient_checkpointing enabled" not in summary` assertion (it still holds).

In `tests/integration/test_cli_run.py`, in both `PresetDecision(...)` literals (`_make_preset_decision` ~44 and `_write_preset_sidecar` ~63) delete `gradient_checkpointing=False,`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/runs/test_bundle.py -v`
Expected: FAIL — `_make_decision()`/`OomEvent(...)` still require the removed fields, and the renamed test references `_preset_block`/`_oom_edge_note` output that still includes the GC clause.

- [ ] **Step 3: Edit `_preset_block` in `bundle.py`**

Remove the `ckpt_word` local and the `gradient_checkpointing=...` clause from the `- Method:` line:

```python
def _preset_block(preset: PresetDecision) -> str:
    method_pretty = method_pretty_name(preset.method)
    used_gib = preset.predicted_bytes / (1024**3)
    total_gib = (preset.budget_bytes + preset.headroom_bytes) / (1024**3)
    headroom_gib = preset.headroom_bytes / (1024**3)
    if preset.provenance == "calibrated":
        date_str = preset.calibrated_at[:10] if preset.calibrated_at else "unknown"
        cache_name = Path(preset.cache_path).name if preset.cache_path else "(unknown)"
        source_line = f"- Source: calibrated {date_str} (cache: {cache_name})"
    else:
        source_line = "- Source: analytic estimate"
    return (
        f"- Method: {method_pretty} r={preset.r}, batch={preset.batch_size}, "
        f"grad_accum={preset.grad_accum_steps}, bf16\n"
        f"- GPU:    {preset.gpu_name} ({total_gib:.1f} GiB)\n"
        f"- Budget: {used_gib:.1f} / {total_gib:.1f} GiB used ({headroom_gib:.1f} GiB headroom)\n"
        f"{source_line}"
    )
```

- [ ] **Step 4: Edit `_oom_edge_note` in `bundle.py`**

Remove the `ckpt_event` lookup + its clause:

```python
def _oom_edge_note(events: tuple[OomEvent, ...]) -> str | None:
    """Return the OOM-summary line for `## Edge cases`, or None when there were none."""
    if not events:
        return None
    final_mb = events[-1].new_micro_batch_size
    return f"OOM retries: {len(events)} — final micro_batch={final_mb}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/runs/test_bundle.py -v`
Expected: PASS (preset block has no `gradient_checkpointing=`; edge note has no GC clause).

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/runs/bundle.py tests/unit/runs/test_bundle.py tests/integration/test_cli_run.py
git commit -m "refactor(bundle): drop ckpt state from preset block + oom edge note"
```

### Task 6: Strip GC from shipped example configs + schema docs

**Files:**

- Modify: `configs/examples/coco_text_lora.yaml`, `coco_text_qlora.yaml`, `coco_text_no_val.yaml`, `coco_text_auto_split.yaml`, `coco_text_lora_subset.yaml`, `gpu_smoke_lora.yaml`, `gpu_smoke_qlora.yaml` (7 files)
- Modify: `docs/config-schema.md` (row ~35)
- Test: `tests/unit/test_data_transforms.py`, `tests/unit/test_config_examples.py` (no edit needed — it must now pass)

- [ ] **Step 1: Fix `test_data_transforms.py::test_shipped_yamls_match_schema_defaults`**

This test reads `ModelConfig.model_fields["gradient_checkpointing"].default` and asserts `cfg.model.gradient_checkpointing == schema_grad_ckpt`. Both lines now break. Remove the GC parts:

- Delete the line `schema_grad_ckpt = ModelConfig.model_fields["gradient_checkpointing"].default`.
- Delete the line `assert cfg.model.gradient_checkpointing == schema_grad_ckpt, p`.
- Update the docstring's first sentence to drop "/ gradient_checkpointing".

- [ ] **Step 2: Run the affected tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_examples.py tests/unit/test_data_transforms.py::test_shipped_yamls_match_schema_defaults -v`
Expected: FAIL — `test_data_transforms` errors on the missing `model_fields["gradient_checkpointing"]`, and (once that import resolves) `test_config_examples::test_example_config_validates` fails to load all 7 example YAMLs (`extra fields not permitted` on `model.gradient_checkpointing`).

- [ ] **Step 3: Strip the GC line from all 7 example YAMLs**

In each of the 7 `configs/examples/*.yaml` files, delete the line:

```yaml
  gradient_checkpointing: false  # see issue #60 — sam3 activation checkpointing fails under non-reentrant recompute
```

Leave the surrounding `model:` keys (`name`, `local_dir`, `checkpoint_file`, `dtype`) intact and correctly indented. Run `uv run --with yamllint yamllint -c .config/yamllint.yml configs/examples/` afterward to confirm no yamllint regression.

- [ ] **Step 4: Remove the `model.gradient_checkpointing` row from `docs/config-schema.md`**

Delete the table row at ~line 35:

```markdown
| `model.gradient_checkpointing` | bool | `true` | common | Enable gradient checkpointing to trade compute for VRAM during training. | Audit §E: 4/4 examples + notebook (via preset) set it; critical for VRAM-constrained GPUs. |
```

- [ ] **Step 5: Run the tests + yamllint to verify they pass**

Run: `uv run pytest tests/unit/test_config_examples.py tests/unit/test_data_transforms.py -v && uv run --with yamllint yamllint -c .config/yamllint.yml configs/examples/`
Expected: PASS — all 7 examples load; `test_shipped_yamls_match_schema_defaults` passes without the GC assertion; yamllint clean.

- [ ] **Step 6: Markdown-lint the touched doc, then commit**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc docs/config-schema.md`
Expected: no findings (fix any before committing).

```bash
git add configs/examples docs/config-schema.md tests/unit/test_data_transforms.py
git commit -m "chore!: strip gradient_checkpointing from example configs + schema docs"
```

---

## REVIEW CHECKPOINT A — Workstream 2 complete

Before starting WS1, verify GC removal is complete and self-consistent:

- [ ] Run: `! grep -rn "gradient_checkpointing\|grad_ckpt\|CKPT_FACTOR" src/ docs/config-schema.md configs/`
      Expected: no matches in `src/`, `docs/config-schema.md`, or `configs/` (exit 0 via `!`). Note: `cli/templates/coco_text_*.yaml` still carry the line at this point — they are deleted in Task 9, so exclude `src/custom_sam_peft/cli/templates/` from this grep or accept those two hits as expected-pending-deletion.
- [ ] Run: `uv run pytest tests/unit/test_model_config.py tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py tests/unit/test_train_types.py tests/unit/test_trainer_oom_retry.py tests/unit/runs/test_bundle.py tests/unit/test_config_examples.py tests/unit/test_data_transforms.py -q`
      Expected: all PASS.
- [ ] Dispatch a code-review subagent (min sonnet/high) over the WS2 diff: confirm no stray `ckpt`/GC references, the "nothing fits" message reads correctly, and `config_patch` is the GC-free 3-section dict the wizard will consume.

---

## Phase 4 — Unified template + flag-driven `init` retarget (WS1)

> Tasks 7 and 8 share `init_cmd.py` (serialize them). Task 9 (delete legacy) runs after 7+8. The new `config_full.yaml` is authored in Task 7.

### Task 7: Author the unified `config_full.yaml` template

**Files:**

- Create: `src/custom_sam_peft/cli/templates/config_full.yaml`
- Test: covered by Task 8's `test_cli_init.py` updates (the template is exercised through `run_init`).

- [ ] **Step 1: Create the unified template**

Create `src/custom_sam_peft/cli/templates/config_full.yaml`. It is a `string.Template` superset of the two legacy templates. Required placeholders (substituted by both `run_init` and the wizard): `$run_name`, `$peft_method`, `$qlora_block`, `$epochs`, `$aug_preset`, `$loss_preset`, `$aug_intensity`, `$class_imbalance`, `$overrides_block`, `$loss_overrides_block`, `$dataset_block`, `$validation_block`, `$model_block`. No `gradient_checkpointing` line anywhere. `prompt_mode: text` is hardcoded (not a placeholder). Author it so that with the flag-driven defaults from Task 8 it renders to a valid `TrainConfig`:

```yaml
# custom-sam-peft starter config (comprehensive) — text-prompt finetune.
# Edit any section below, then run the launch command shown by `init`.
# -----------------------------------------------------------------------------
# Schema is the source of truth (src/custom_sam_peft/config/schema.py). Keys are
# echoed for discoverability; commented blocks are alternative branches or
# advanced overrides — uncomment and edit to use them.
# -----------------------------------------------------------------------------
run:
  name: $run_name
  output_dir: ./runs
  seed: 42

model:
$model_block
  dtype: bfloat16
  # --- advanced ---
  # revision: null
  # device: null

data:
$dataset_block
$validation_block
  prompt_mode: text
  image_size: 1008
  # --- advanced ---
  # channels: 3
  # channel_semantics: rgb
  augmentations:
    preset: $aug_preset
    intensity: $aug_intensity
    $overrides_block
  text_prompt:
    mode: present_plus_negatives
    negatives_per_image: 4
  # remove the `normalize:` block unless overriding for a non-SAM3 backbone —
  # SAM3.1's Sam3ImageProcessor returns the ImageNet stats below.
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

peft:
  method: $peft_method
  r: 16
  alpha: 32
  dropout: 0.05
  # --- advanced ---
  # scope: vision_decoder
  # target_modules: null
  # bias: none
$qlora_block

train:
  epochs: $epochs
  batch_size: 1
  grad_accum_steps: 8
  optimizer: auto
  learning_rate: 1.0e-4
  lr_schedule: cosine
  warmup_steps: 100
  max_grad_norm: 1.0
  eval_every: 500
  save_every: 1000
  log_every: 50
  nan_abort_after: 20
  box_hint:
    p_start: 1.0
    p_end: 0.0
    decay_steps: 5000
  loss:
    preset: $loss_preset
    class_imbalance: $class_imbalance
    $loss_overrides_block

eval:
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

tracking:
  backend: tensorboard

export:
  merge: false
```

Block-placeholder contracts (the strings `run_init` / the wizard substitute):

- `$model_block` — the `model.name`/`local_dir`/`checkpoint_file` lines, 2-space indented (so they sit under `model:`). Flag-driven default:

  ```text
    name: facebook/sam3.1
    local_dir: models/sam3.1
    checkpoint_file: sam3.1_multiplex.pt
  ```

- `$dataset_block` — the active dataset-format keys (2-space indented under `data:`), followed by the OTHER format as a commented block. COCO-active default:

  ```text
    format: coco
    train:
      annotations: data/train.json
      images: data/train/
    # HuggingFace alternative — set format: hf and uncomment:
    # hf:
    #   name: org/dataset
    #   split_train: train
    #   split_val: validation
  ```

- `$validation_block` — the chosen validation mode active + the other two commented (2-space indented under `data:`). Explicit-val default:

  ```text
    val:
      annotations: data/val.json
      images: data/val/
    # Auto-split alternative (carve data.train into train+val):
    # val_split:
    #   fraction: 0.1
    #   seed: null
    # No-val alternative: omit both val: and val_split:.
  ```

- `$qlora_block` — empty string for LoRA; for QLoRA the `qlora:` sub-block (2-space indented under `peft:`):

  ```text
    qlora:
      quant_type: nf4
      compute_dtype: bfloat16
  ```

- `$overrides_block` / `$loss_overrides_block` — reuse the exact strings `run_init` already builds (the commented-scaffold vs `overrides: {}` forms in `init_cmd._build_loss_overrides_block` and the inline aug-overrides branch).

- [ ] **Step 2: Sanity-check the template renders + reloads (manual probe, no commit yet)**

This template is validated end-to-end by Task 8's tests; do not write a separate template-only test. Proceed to Task 8.

### Task 8: Retarget `run_init` + add `--interactive` flag to `init_cmd.py`

**Files:**

- Modify: `src/custom_sam_peft/cli/init_cmd.py`
- Test: `tests/unit/test_cli_init.py`

- [ ] **Step 1: Update `test_cli_init.py` for the unified template**

Apply these changes to `tests/unit/test_cli_init.py` (the two `--template` values still map onto one template via `$peft_method`):

1. `test_init_writes_qlora_template` — after `cfg.peft.method == "qlora"`, add a body assertion that the qlora sub-block rendered: read `out.read_text()` and `assert "quant_type: nf4" in body`.
2. Add a comprehensiveness test:

```python
def test_init_emits_comprehensive_config(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output
    body = out.read_text()
    for section in ("run:", "model:", "data:", "peft:", "train:", "eval:", "tracking:", "export:"):
        assert section in body
    assert "prompt_mode: text" in body
    # Alternative branches present as comments:
    assert "# val_split:" in body
    assert "# hf:" in body
    cfg = load_config(out)
    assert cfg.run.name == "my-run"
```

3. `test_init_other_fields_parse_identically` — keep as-is (asserts `cfg.run.name == "my-run"`, `model.name`, `train.epochs == 10`).
4. No test may reference `coco_text_lora.yaml` / `coco_text_qlora.yaml` template filenames (the existing file does not — confirm by grepping the test file).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_init.py -v`
Expected: FAIL — `run_init` still renders the legacy per-template files (no `# val_split:`/`# hf:` comment blocks, no qlora-body assertion satisfied through the unified template path) so the new assertions fail.

- [ ] **Step 3: Retarget `run_init` + `TEMPLATES` to the unified template**

In `src/custom_sam_peft/cli/init_cmd.py`:

Change `TEMPLATES` to map the two `--template` values onto the `$peft_method` value (both render the one file):

```python
TEMPLATES: dict[str, str] = {
    "coco-text-lora": "lora",
    "coco-text-qlora": "qlora",
}

UNIFIED_TEMPLATE = "config_full.yaml"
```

Add a helper that builds the flag-driven block defaults (so the wizard can reuse them later by importing):

```python
def _flag_driven_blocks(peft_method: str) -> dict[str, str]:
    """The block-placeholder strings the flag-driven path substitutes.

    Wizard-only branches (dataset format, validation mode, model path) get
    their flag-driven defaults here: local-COCO dataset, explicit-val, default
    model dir.
    """
    model_block = (
        "  name: facebook/sam3.1\n"
        "  local_dir: models/sam3.1\n"
        "  checkpoint_file: sam3.1_multiplex.pt"
    )
    dataset_block = (
        "  format: coco\n"
        "  train:\n"
        "    annotations: data/train.json\n"
        "    images: data/train/\n"
        "  # HuggingFace alternative — set format: hf and uncomment:\n"
        "  # hf:\n"
        "  #   name: org/dataset\n"
        "  #   split_train: train\n"
        "  #   split_val: validation"
    )
    validation_block = (
        "  val:\n"
        "    annotations: data/val.json\n"
        "    images: data/val/\n"
        "  # Auto-split alternative (carve data.train into train+val):\n"
        "  # val_split:\n"
        "  #   fraction: 0.1\n"
        "  #   seed: null\n"
        "  # No-val alternative: omit both val: and val_split:."
    )
    qlora_block = (
        "  qlora:\n    quant_type: nf4\n    compute_dtype: bfloat16" if peft_method == "qlora" else ""
    )
    return {
        "model_block": model_block,
        "dataset_block": dataset_block,
        "validation_block": validation_block,
        "qlora_block": qlora_block,
    }
```

Rewrite `run_init` to render `config_full.yaml`. Keep the existing preset/intensity/class-imbalance validation and the existing `overrides_block`/`loss_overrides_block` construction; add `run_name`/`epochs` with flag-driven defaults and the new blocks:

```python
def run_init(
    template: str,
    output: Path,
    *,
    preset: str = "natural",
    intensity: str = "medium",
    class_imbalance: str = "balanced",
    force: bool = False,
) -> None:
    """Write a starter config (unified config_full.yaml) to *output*."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template '{template}'. Available: {', '.join(TEMPLATES)}")
    valid_presets = set(get_args(Preset))
    valid_intensities = set(get_args(Intensity))
    _CLASS_IMBALANCES = get_args(ClassImbalance)
    if preset not in valid_presets:
        raise ValueError(f"unknown preset '{preset}'. Available: {sorted(valid_presets)}")
    if intensity not in valid_intensities:
        raise ValueError(f"unknown intensity '{intensity}'. Available: {sorted(valid_intensities)}")
    if class_imbalance not in _CLASS_IMBALANCES:
        raise typer.BadParameter(
            f"--class-imbalance must be one of {list(_CLASS_IMBALANCES)}; got {class_imbalance!r}",
            param_hint="--class-imbalance",
        )
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing {output}; pass force=True")

    if preset == "custom":
        overrides_block = (
            "overrides: {}  # fill in knobs: hflip, vflip, rotate90, "
            "rotate_arbitrary, color_jitter, stain_jitter, blur, gauss_noise"
        )
    else:
        overrides_block = (
            "# Override individual knobs here; unset keys inherit from (preset, intensity).\n"
            "    # overrides:\n"
            "    #   hflip: false\n"
            "    #   color_jitter: 0.15"
        )
    loss_overrides_block = _build_loss_overrides_block(preset)

    peft_method = TEMPLATES[template]
    blocks = _flag_driven_blocks(peft_method)
    raw = (files("custom_sam_peft.cli.templates") / UNIFIED_TEMPLATE).read_text()
    body = string.Template(raw).substitute(
        run_name="my-run",
        peft_method=peft_method,
        epochs=10,
        aug_preset=preset,
        loss_preset=preset,
        aug_intensity=intensity,
        class_imbalance=class_imbalance,
        overrides_block=overrides_block,
        loss_overrides_block=loss_overrides_block,
        **blocks,
    )
    output.write_text(body)
```

(`init()`'s flag definitions are unchanged. The `--template` help text already lists both values via `TEMPLATES`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_init.py -v`
Expected: PASS — both templates render the unified file; comprehensiveness + qlora-body assertions hold; output reloads via `load_config`.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/templates/config_full.yaml src/custom_sam_peft/cli/init_cmd.py tests/unit/test_cli_init.py
git commit -m "feat(init): render unified config_full.yaml from flag-driven init"
```

### Task 9: Delete the two legacy templates

**Files:**

- Delete: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `coco_text_qlora.yaml`

- [ ] **Step 1: Delete the legacy template files**

```bash
git rm src/custom_sam_peft/cli/templates/coco_text_lora.yaml src/custom_sam_peft/cli/templates/coco_text_qlora.yaml
```

- [ ] **Step 2: Verify nothing references the deleted files**

Run: `! grep -rn "coco_text_lora.yaml\|coco_text_qlora.yaml" src/ tests/`
Expected: no matches in `src/`/`tests/` for the *template* filenames (the `configs/examples/coco_text_*.yaml` are different paths and are not matched by these basenames in `src/`/`tests/` lookups; if any test still imports the template basenames, fix it). Re-run `uv run pytest tests/unit/test_cli_init.py -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore(init): delete legacy coco_text_{lora,qlora} templates"
```

---

## Phase 5 — The wizard module (WS1)

> One new file `setup_wizard.py` + one new test file. Build incrementally with TDD: helpers/primitives → steps → render → validate → emit → the two smart helpers. Then wire `--interactive` into `init` (Task 16, shares `init_cmd.py`).

### Task 10: Scaffolding — `Ctx`, `WizardStep`, `_deep_merge`, prompt primitives

**Files:**

- Create: `src/custom_sam_peft/cli/setup_wizard.py`
- Create: `tests/unit/cli/__init__.py`
- Create: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/unit/cli/__init__.py` with a single newline (matches the existing `tests/unit/__init__.py` convention).

- [ ] **Step 2: Write failing tests for `_deep_merge` + dataclasses**

Create `tests/unit/cli/test_setup_wizard.py`:

```python
"""Tests for the interactive setup wizard (CPU-only; prompt primitives monkeypatched)."""

from __future__ import annotations

import pytest

from custom_sam_peft.cli import setup_wizard as sw


def test_deep_merge_nested_dicts() -> None:
    dst = {"data": {"format": "coco"}}
    sw._deep_merge(dst, {"data": {"val_split": {"fraction": 0.1}}})
    assert dst == {"data": {"format": "coco", "val_split": {"fraction": 0.1}}}


def test_deep_merge_scalar_overwrites() -> None:
    dst = {"peft": {"method": "lora"}}
    sw._deep_merge(dst, {"peft": {"method": "qlora"}})
    assert dst["peft"]["method"] == "qlora"


def test_ctx_constructs_with_cuda_flag_and_run_mode() -> None:
    ctx = sw.Ctx(answers={}, cuda_available=False)
    assert ctx.answers == {}
    assert ctx.cuda_available is False
    assert ctx.run_mode == "train"  # default
    assert ctx.categories is None
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -v`
Expected: FAIL with `ModuleNotFoundError: custom_sam_peft.cli.setup_wizard`.

- [ ] **Step 4: Implement scaffolding in `setup_wizard.py`**

Create `src/custom_sam_peft/cli/setup_wizard.py`:

```python
"""Interactive `csp init --interactive` wizard.

Declarative WizardStep registry → answers dict → render config_full.yaml →
validate via load_config → emit. See
docs/superpowers/specs/2026-05-26-interactive-setup-wizard-design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import typer

RunMode = Literal["train", "run", "eval"]


@dataclass
class Ctx:
    answers: dict[str, Any]
    cuda_available: bool
    run_mode: RunMode = "train"
    categories: list[str] | None = None
    category_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class WizardStep:
    id: str
    ask: Callable[[Ctx], dict[str, Any]]
    when: Callable[[Ctx], bool] = field(default=lambda ctx: True)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively merge src into dst. Nested dicts merge; scalars/lists overwrite."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def ask_text(prompt: str, *, default: str | None = None,
             validate: Callable[[str], str | None] | None = None) -> str:
    """Free-text prompt; re-asks on validate failure. validate returns an error string or None."""
    while True:
        value = typer.prompt(prompt, default=default) if default is not None else typer.prompt(prompt)
        value = str(value).strip()
        if validate is not None:
            err = validate(value)
            if err is not None:
                typer.echo(err)
                continue
        return value


def ask_choice(prompt: str, choices: list[str], *, default: str | None = None) -> str:
    """Membership-checked choice; re-asks on invalid."""
    rendered = f"{prompt} [{'/'.join(choices)}]"
    while True:
        value = typer.prompt(rendered, default=default) if default is not None else typer.prompt(rendered)
        value = str(value).strip()
        if value in choices:
            return value
        typer.echo(f"choose one of: {', '.join(choices)}")


def ask_confirm(prompt: str, *, default: bool = True) -> bool:
    return typer.confirm(prompt, default=default)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/__init__.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): scaffolding — Ctx, WizardStep, deep_merge, prompt primitives"
```

### Task 11: `infer_class_imbalance` (smart helper §6.2)

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write failing tests for class-imbalance inference**

Add to `tests/unit/cli/test_setup_wizard.py` (synthetic COCO JSONs in `tmp_path`):

```python
import json
from pathlib import Path


def _write_coco(path: Path, per_cat_counts: dict[int, int], *, iscrowd_extra: int = 0) -> None:
    categories = [{"id": cid, "name": f"c{cid}"} for cid in per_cat_counts]
    images, annotations = [], []
    img_id, ann_id = 0, 0
    for cid, count in per_cat_counts.items():
        for _ in range(count):
            images.append({"id": img_id, "file_name": f"{img_id}.jpg", "height": 4, "width": 4})
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cid,
                                 "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0})
            img_id += 1
            ann_id += 1
    for _ in range(iscrowd_extra):
        images.append({"id": img_id, "file_name": f"{img_id}.jpg", "height": 4, "width": 4})
        annotations.append({"id": ann_id, "image_id": img_id, "category_id": next(iter(per_cat_counts)),
                             "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 1})
        img_id += 1
        ann_id += 1
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))


def test_infer_balanced_below_3x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 10, 3: 12})  # R≈1.2
    assert sw.infer_class_imbalance(str(p)) == "balanced"


def test_infer_moderate_3x_to_10x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 40})  # R=4
    assert sw.infer_class_imbalance(str(p)) == "moderate"


def test_infer_severe_at_or_above_10x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 5, 2: 100})  # R=20
    assert sw.infer_class_imbalance(str(p)) == "severe"


def test_infer_thresholds_boundary_exact(tmp_path: Path) -> None:
    p3 = tmp_path / "r3.json"
    _write_coco(p3, {1: 10, 2: 30})  # R=3.0 → moderate
    assert sw.infer_class_imbalance(str(p3)) == "moderate"
    p10 = tmp_path / "r10.json"
    _write_coco(p10, {1: 10, 2: 100})  # R=10.0 → severe
    assert sw.infer_class_imbalance(str(p10)) == "severe"


def test_infer_unreadable_defaults_balanced(tmp_path: Path) -> None:
    assert sw.infer_class_imbalance(str(tmp_path / "missing.json")) == "balanced"


def test_infer_iscrowd_excluded(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    # Without crowd: 10/10 balanced. Crowd instances on cat 1 must NOT tip it.
    _write_coco(p, {1: 10, 2: 10}, iscrowd_extra=50)
    assert sw.infer_class_imbalance(str(p)) == "balanced"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k infer -v`
Expected: FAIL — `sw.infer_class_imbalance` does not exist (AttributeError).

- [ ] **Step 3: Implement `infer_class_imbalance` + threshold constants**

Add to `setup_wizard.py`:

```python
from custom_sam_peft.config.schema import ClassImbalance

IMBALANCE_MODERATE_RATIO = 3.0   # R < 3 → balanced
IMBALANCE_SEVERE_RATIO = 10.0    # 3 <= R < 10 → moderate; R >= 10 → severe


def infer_class_imbalance(annotations: str) -> ClassImbalance:
    """Detect a class-imbalance tier from per-category instance counts.

    Mirrors data/subset.py per-class frequency; uses the pycocotools-backed
    primitives in data/coco.py. On ANY failure (missing/unreadable file, zero
    present categories) returns "balanced".
    """
    try:
        from custom_sam_peft.data.coco import _build_category_remap, _load_coco_index

        coco = _load_coco_index(annotations)
        _sparse, remap, _names = _build_category_remap(coco)
        counts: dict[int, int] = {}
        for img_id in coco.getImgIds():
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            for a in anns:
                if int(a.get("iscrowd", 0)) != 0:
                    continue
                dense = remap.get(int(a["category_id"]))
                if dense is None:
                    continue
                counts[dense] = counts.get(dense, 0) + 1
        present = [c for c in counts.values() if c > 0]
        if not present:
            raise ValueError("no present categories")
        ratio = max(present) / min(present)
    except Exception:  # noqa: BLE001 — any failure defaults to balanced
        return "balanced"

    if ratio < IMBALANCE_MODERATE_RATIO:
        return "balanced"
    if ratio < IMBALANCE_SEVERE_RATIO:
        return "moderate"
    return "severe"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k infer -v`
Expected: PASS (R=3.0 → moderate; R=10.0 → severe; iscrowd excluded; missing file → balanced).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): infer_class_imbalance from per-category counts"
```

### Task 12: The `STEPS` registry + `run_wizard` driver (§4 flow, fragment shapes, gating)

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write failing tests for fragments + gating + driver**

Add to the test file (monkeypatch the prompt primitives to feed deterministic answers):

```python
def _patch_prompts(monkeypatch, *, texts=None, choices=None, confirms=None):
    """Feed scripted answers to the three primitives in call order."""
    t = iter(texts or [])
    c = iter(choices or [])
    cf = iter(confirms or [])
    monkeypatch.setattr(sw, "ask_text", lambda *a, **k: next(t))
    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: next(c))
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: next(cf))


def test_step_fragment_shapes_are_nested_dicts(monkeypatch) -> None:
    # run_mode → train; run_name → my-run; dataset coco + paths; validation none;
    # domain natural/medium; (class_imbalance gated, monkeypatch infer); peft manual lora;
    # epochs 5; weights blank.
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "5", ""],
        choices=["train", "coco", "none", "natural", "medium", "lora"],
    )
    monkeypatch.setattr(sw, "infer_class_imbalance", lambda *a, **k: "balanced")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    answers = sw.run_wizard(ctx)
    assert answers["run"]["name"] == "my-run"
    assert answers["data"]["format"] == "coco"
    assert answers["data"]["train"]["annotations"] == "ann.json"
    assert answers["peft"]["method"] == "lora"
    assert answers["train"]["epochs"] == 5
    assert answers["train"]["loss"]["class_imbalance"] == "balanced"
    assert ctx.run_mode == "train"


def test_when_gating_skips_class_imbalance_in_eval_mode() -> None:
    step = next(s for s in sw.STEPS if s.id == "class_imbalance")
    ctx = sw.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False, run_mode="eval")
    assert step.when(ctx) is False


def test_when_gating_skips_vram_autosize_without_cuda(monkeypatch) -> None:
    _patch_prompts(monkeypatch, choices=["lora"])
    step = next(s for s in sw.STEPS if s.id == "peft_sizing")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    # when() is False without CUDA → driver would skip; ask() reachable directly takes manual branch.
    assert step.when(ctx) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "fragment or gating" -v`
Expected: FAIL — `sw.STEPS` / `sw.run_wizard` not defined.

- [ ] **Step 3: Implement the step `ask` functions, `STEPS`, and `run_wizard`**

Add to `setup_wizard.py`. Each `ask` returns a nested fragment (or `{}` for `run_mode`):

```python
def _ask_run_mode(ctx: Ctx) -> dict[str, Any]:
    ctx.run_mode = ask_choice("Run mode?", ["train", "run", "eval"], default="train")  # type: ignore[assignment]
    return {}


def _ask_run_name(ctx: Ctx) -> dict[str, Any]:
    name = ask_text("Run name?", default="my-run")
    return {"run": {"name": name}}


def _ask_dataset_source(ctx: Ctx) -> dict[str, Any]:
    fmt = ask_choice("Dataset format?", ["coco", "hf"], default="coco")
    if fmt == "coco":
        ann = ask_text("Path to COCO train annotations (.json)?")
        imgs = ask_text("Path to COCO train images dir?")
        return {"data": {"format": "coco", "train": {"annotations": ann, "images": imgs}}}
    name = ask_text("HuggingFace dataset name (org/dataset)?")
    return {"data": {"format": "hf", "hf": {"name": name}}}


def _ask_validation(ctx: Ctx) -> dict[str, Any]:
    fmt = ctx.answers.get("data", {}).get("format", "coco")
    mode = ask_choice("Validation?", ["explicit", "auto-split", "none"], default="auto-split")
    if mode == "none":
        if ctx.run_mode in {"eval", "run"}:
            typer.echo(
                "note: eval/run needs a validation set to score against; "
                "selecting none means eval will have nothing to evaluate."
            )
        return {}
    if mode == "auto-split":
        frac = ask_text("Auto-split fraction (0<f<=0.5)?", default="0.1")
        return {"data": {"val_split": {"fraction": float(frac)}}}
    # explicit
    if fmt == "hf":
        split = ask_text("HF validation split name?", default="validation")
        return {"data": {"hf": {"split_val": split}}}
    ann = ask_text("Path to COCO val annotations (.json)?")
    imgs = ask_text("Path to COCO val images dir?")
    return {"data": {"val": {"annotations": ann, "images": imgs}}}


def _ask_domain(ctx: Ctx) -> dict[str, Any]:
    domain = ask_choice("Domain?", ["natural", "medical", "satellite", "microscopy", "none"],
                        default="natural")
    intensity = ask_choice("Augmentation intensity?", ["safe", "medium", "aggressive"], default="medium")
    return {
        "data": {"augmentations": {"preset": domain, "intensity": intensity}},
        "train": {"loss": {"preset": domain}},
    }


def _coco_train_annotations(ctx: Ctx) -> str | None:
    data = ctx.answers.get("data", {})
    if data.get("format") != "coco":
        return None
    return data.get("train", {}).get("annotations")


def _ask_class_imbalance(ctx: Ctx) -> dict[str, Any]:
    ann = _coco_train_annotations(ctx)
    if ann is None:
        typer.echo("could not auto-detect class imbalance (non-COCO/no annotations); defaulting to balanced")
        detected = "balanced"
    else:
        detected = infer_class_imbalance(ann)
        typer.echo(f"detected class imbalance: {detected}")
    tier = ask_choice("Class imbalance tier?", ["balanced", "moderate", "severe"], default=detected)
    return {"train": {"loss": {"class_imbalance": tier}}}


def _ask_peft_sizing(ctx: Ctx) -> dict[str, Any]:
    from custom_sam_peft.presets import decide_preset

    if ctx.cuda_available and ask_confirm("Auto-size the PEFT config to your GPU's VRAM?", default=True):
        image_size = ctx.answers.get("data", {}).get("image_size", 1008)
        try:
            decision = decide_preset(image_size)
        except RuntimeError as exc:
            typer.echo(f"could not auto-size: {exc}; falling back to manual")
        else:
            typer.echo(decision.label())
            return decision.config_patch
    method = ask_choice("PEFT method?", ["lora", "qlora"], default="lora")
    return {"peft": {"method": method}}


def _ask_epochs(ctx: Ctx) -> dict[str, Any]:
    def _positive_int(s: str) -> str | None:
        try:
            return None if int(s) > 0 else "epochs must be a positive integer"
        except ValueError:
            return "epochs must be a positive integer"

    epochs = ask_text("Number of epochs?", default="10", validate=_positive_int)
    return {"train": {"epochs": int(epochs)}}


def _ask_model_weights(ctx: Ctx) -> dict[str, Any]:
    from pathlib import Path

    def _is_file_or_blank(s: str) -> str | None:
        if s == "":
            return None
        return None if Path(s).is_file() else f"no file at {s}"

    raw = ask_text(
        "Path to an existing SAM 3.1 checkpoint (.pt)? Leave blank to use "
        "`models/sam3.1` and download if missing.",
        default="",
        validate=_is_file_or_blank,
    )
    if raw:
        p = Path(raw)
        return {"model": {"local_dir": str(p.parent), "checkpoint_file": p.name}}
    # Blank: shallow glob for the default checkpoint elsewhere under models/.
    hits = sorted(Path("models").glob("**/sam3.1_multiplex.pt")) if Path("models").is_dir() else []
    if hits:
        return {"model": {"local_dir": str(hits[0].parent)}}
    return {}


STEPS: list[WizardStep] = [
    WizardStep("run_mode", _ask_run_mode),
    WizardStep("run_name", _ask_run_name),
    WizardStep("dataset_source", _ask_dataset_source),
    WizardStep("validation", _ask_validation),
    WizardStep("domain", _ask_domain),
    WizardStep(
        "class_imbalance",
        _ask_class_imbalance,
        when=lambda ctx: ctx.run_mode in {"train", "run"},
    ),
    WizardStep("peft_sizing", _ask_peft_sizing, when=lambda ctx: ctx.cuda_available),
    WizardStep("epochs", _ask_epochs, when=lambda ctx: ctx.run_mode != "eval"),
    WizardStep("model_weights", _ask_model_weights),
]


def run_wizard(ctx: Ctx) -> dict[str, Any]:
    for step in STEPS:
        if step.when(ctx):
            fragment = step.ask(ctx)
            _deep_merge(ctx.answers, fragment)
    return ctx.answers
```

Note on the `peft_sizing` `when`: the spec says the manual branch must be reachable in the same step. `when=lambda ctx: ctx.cuda_available` means with no CUDA the driver SKIPS the step entirely — so `peft.method` would be unset. To honor the spec's "Required (peft.method always set)" guarantee, the driver must still set a method in no-CUDA mode. Resolve this by making `peft_sizing` always run and gating the auto-size offer *inside* `ask` (per §6.1 "the opt-in confirm happens inside ask"):

Change the registry entry to no `when` (always runs):

```python
    WizardStep("peft_sizing", _ask_peft_sizing),
```

and the no-CUDA test asserts the manual branch via the driver, not via `step.when`. Update `test_when_gating_skips_vram_autosize_without_cuda` to assert the fragment instead:

```python
def test_when_gating_skips_vram_autosize_without_cuda(monkeypatch) -> None:
    _patch_prompts(monkeypatch, choices=["lora"])
    step = next(s for s in sw.STEPS if s.id == "peft_sizing")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    assert step.ask(ctx) == {"peft": {"method": "lora"}}
```

And `_ask_epochs` for `eval` mode: the `when=lambda ctx: ctx.run_mode != "eval"` skips it, so the render stage (Task 13) must default `epochs=1` when absent.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "fragment or gating" -v`
Expected: PASS (fragments are nested dicts; class-imbalance gated off in eval; peft manual branch returns `{"peft": {"method": "lora"}}` without CUDA).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): STEPS registry + run_wizard driver (7-prompt flow)"
```

### Task 13: `render` — answers dict → `config_full.yaml` placeholders (§5.2)

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write failing tests for render branches**

```python
import yaml
from custom_sam_peft.config.loader import load_config


def test_render_coco_explicit_val_reloads(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {"format": "coco", "train": {"annotations": "t.json", "images": "t/"},
                 "val": {"annotations": "v.json", "images": "v/"},
                 "augmentations": {"preset": "medical", "intensity": "medium"}},
        "peft": {"method": "lora"},
        "train": {"epochs": 3, "loss": {"preset": "medical", "class_imbalance": "moderate"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "prompt_mode: text" in rendered
    assert "format: coco" in rendered
    assert "# hf:" in rendered           # other format commented
    assert "# val_split:" in rendered    # other validation commented
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.val is not None
    assert cfg.peft.method == "lora"
    assert cfg.train.epochs == 3


def test_render_hf_autosplit_qlora_reloads(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {"format": "hf", "hf": {"name": "org/ds"}, "val_split": {"fraction": 0.2},
                 "augmentations": {"preset": "natural", "intensity": "safe"}},
        "peft": {"method": "qlora"},
        "train": {"epochs": 2, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "name: org/ds" in rendered
    assert "quant_type: nf4" in rendered  # qlora block active
    assert "val_split:" in rendered
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.format == "hf"
    assert cfg.peft.method == "qlora"


def test_render_eval_mode_defaults_epochs_to_1(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {"format": "coco", "train": {"annotations": "t.json", "images": "t/"},
                 "augmentations": {"preset": "natural", "intensity": "medium"}},
        "peft": {"method": "lora"},
        "train": {"loss": {"preset": "natural", "class_imbalance": "balanced"}},  # no epochs
    }
    rendered = sw.render(answers, run_mode="eval")
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.train.epochs == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k render -v`
Expected: FAIL — `sw.render` not defined.

- [ ] **Step 3: Implement `render` (reuse `init_cmd` block builders)**

Add to `setup_wizard.py`. Build the block placeholders from the answers, then substitute into `config_full.yaml`. Reuse `init_cmd._flag_driven_blocks`-style strings but compute the active branch from answers:

```python
import string
from importlib.resources import files

from custom_sam_peft.cli.init_cmd import UNIFIED_TEMPLATE, _build_loss_overrides_block


def _model_block(answers: dict[str, Any]) -> str:
    m = answers.get("model", {})
    local_dir = m.get("local_dir", "models/sam3.1")
    ckpt = m.get("checkpoint_file", "sam3.1_multiplex.pt")
    return (
        "  name: facebook/sam3.1\n"
        f"  local_dir: {local_dir}\n"
        f"  checkpoint_file: {ckpt}"
    )


def _dataset_block(answers: dict[str, Any]) -> str:
    data = answers.get("data", {})
    if data.get("format") == "hf":
        name = data["hf"]["name"]
        return (
            "  format: hf\n"
            "  hf:\n"
            f"    name: {name}\n"
            "  # COCO alternative — set format: coco and uncomment:\n"
            "  # train:\n"
            "  #   annotations: data/train.json\n"
            "  #   images: data/train/"
        )
    train = data.get("train", {})
    ann = train.get("annotations", "data/train.json")
    imgs = train.get("images", "data/train/")
    return (
        "  format: coco\n"
        "  train:\n"
        f"    annotations: {ann}\n"
        f"    images: {imgs}\n"
        "  # HuggingFace alternative — set format: hf and uncomment:\n"
        "  # hf:\n"
        "  #   name: org/dataset\n"
        "  #   split_train: train\n"
        "  #   split_val: validation"
    )


def _validation_block(answers: dict[str, Any]) -> str:
    data = answers.get("data", {})
    explicit_active = auto_active = noval_active = False
    if data.get("val") is not None:
        explicit_active = True
        v = data["val"]
        active = (
            "  val:\n"
            f"    annotations: {v['annotations']}\n"
            f"    images: {v['images']}"
        )
    elif data.get("val_split") is not None:
        auto_active = True
        active = (
            "  val_split:\n"
            f"    fraction: {data['val_split']['fraction']}\n"
            "    seed: null"
        )
    else:
        noval_active = True
        active = "  # no-val mode: neither val: nor val_split: is set."
    # Append the two inactive alternatives as comments.
    alts = []
    if not explicit_active:
        alts.append("  # Explicit-val alternative:\n  # val:\n  #   annotations: data/val.json\n  #   images: data/val/")
    if not auto_active:
        alts.append("  # Auto-split alternative:\n  # val_split:\n  #   fraction: 0.1\n  #   seed: null")
    if not noval_active:
        alts.append("  # No-val alternative: omit both val: and val_split:.")
    return "\n".join([active, *alts])


def _qlora_block(answers: dict[str, Any]) -> str:
    if answers.get("peft", {}).get("method") == "qlora":
        return "  qlora:\n    quant_type: nf4\n    compute_dtype: bfloat16"
    return ""


def _aug_overrides_block() -> str:
    return (
        "# Override individual knobs here; unset keys inherit from (preset, intensity).\n"
        "    # overrides:\n"
        "    #   hflip: false\n"
        "    #   color_jitter: 0.15"
    )


def render(answers: dict[str, Any], *, run_mode: RunMode) -> str:
    data = answers.get("data", {})
    aug = data.get("augmentations", {})
    loss = answers.get("train", {}).get("loss", {})
    epochs = answers.get("train", {}).get("epochs", 1)  # eval defaults to 1
    preset = aug.get("preset", "natural")
    raw = (files("custom_sam_peft.cli.templates") / UNIFIED_TEMPLATE).read_text()
    return string.Template(raw).substitute(
        run_name=answers.get("run", {}).get("name", "my-run"),
        peft_method=answers.get("peft", {}).get("method", "lora"),
        epochs=epochs,
        aug_preset=preset,
        loss_preset=loss.get("preset", "natural"),
        aug_intensity=aug.get("intensity", "medium"),
        class_imbalance=loss.get("class_imbalance", "balanced"),
        overrides_block=_aug_overrides_block(),
        loss_overrides_block=_build_loss_overrides_block(preset),
        model_block=_model_block(answers),
        dataset_block=_dataset_block(answers),
        validation_block=_validation_block(answers),
        qlora_block=_qlora_block(answers),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k render -v`
Expected: PASS — COCO+explicit and HF+auto-split+QLoRA both render and reload; the inactive branches are commented; eval mode defaults epochs to 1.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): render answers onto config_full.yaml placeholders"
```

### Task 14: `validate` + `emit` (§5.3, §5.4)

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write failing tests for validate + emit**

```python
def test_validate_accepts_good_render() -> None:
    answers = {
        "run": {"name": "r"},
        "data": {"format": "coco", "train": {"annotations": "t.json", "images": "t/"},
                 "augmentations": {"preset": "natural", "intensity": "medium"}},
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    sw.validate(rendered)  # must not raise


def test_emit_header_and_launch_command(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {"format": "coco", "train": {"annotations": "t.json", "images": "t/"},
                 "augmentations": {"preset": "natural", "intensity": "medium"}},
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    for mode, verb in [("train", "train"), ("run", "run"), ("eval", "eval")]:
        out = tmp_path / f"{mode}.yaml"
        rendered = sw.render(answers, run_mode=mode)
        sw.emit(rendered, out, force=False, run_mode=mode)
        body = out.read_text()
        lines = body.splitlines()
        assert lines[0].startswith("# Generated by `custom-sam-peft init --interactive`")
        assert lines[1] == f"# Launch: custom-sam-peft {verb} --config {out}"


def test_emit_validated_bytes_reload(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {"format": "coco", "train": {"annotations": "t.json", "images": "t/"},
                 "augmentations": {"preset": "natural", "intensity": "medium"}},
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    out = tmp_path / "c.yaml"
    rendered = sw.render(answers, run_mode="train")
    sw.emit(rendered, out, force=False, run_mode="train")
    cfg = load_config(out)  # the header-prefixed bytes still load
    assert cfg.run.name == "r"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "validate or emit" -v`
Expected: FAIL — `sw.validate` / `sw.emit` not defined.

- [ ] **Step 3: Implement `validate` + `emit`**

Add to `setup_wizard.py`:

```python
from datetime import date
from pathlib import Path

from custom_sam_peft.config.loader import load_config

_LAUNCH_VERB = {"train": "train", "run": "run", "eval": "eval"}


def validate(rendered: str) -> None:
    """Validate the exact bytes via load_config by round-tripping through a temp file."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(rendered)
        tmp = Path(f.name)
    try:
        load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _launch_command(output: Path, run_mode: RunMode) -> str:
    return f"custom-sam-peft {_LAUNCH_VERB[run_mode]} --config {output}"


def emit(rendered: str, output: Path, force: bool, *, run_mode: RunMode) -> str:
    """Write header + rendered config to output. Returns the launch command."""
    if output.exists() and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing {output}; pass --force",
            param_hint="--output",
        )
    launch = _launch_command(output, run_mode)
    header = (
        f"# Generated by `custom-sam-peft init --interactive` on {date.today().isoformat()}\n"
        f"# Launch: {launch}\n\n"
    )
    output.write_text(header + rendered)
    return launch
```

Note: `validate` round-trips through a real temp file because `load_config` takes a path. To validate the *header-prefixed* bytes (spec §5.3 "header included"), the wizard driver (Task 15) will call `validate(header + rendered)` on the same bytes `emit` writes; here `validate(rendered)` is the body-only check used in the happy-path tests, and the driver composes the final bytes. (Resolved choice: the driver builds the final body string once, validates it, then writes it — see Task 15 — so emit's write and validate's input are identical.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "validate or emit" -v`
Expected: PASS — header lines correct per mode; emitted bytes reload.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): validate-via-load_config + emit with header/launch command"
```

### Task 15: Top-level `run_wizard`-to-file orchestration helper (compose render → validate → emit)

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write failing tests for the orchestration entry + backstop + ctrl-c**

```python
def test_generate_config_happy_path_local_coco_autosplit(tmp_path, monkeypatch) -> None:
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "0.1", "7", ""],
        choices=["train", "coco", "auto-split", "natural", "medium", "lora"],
    )
    monkeypatch.setattr(sw, "infer_class_imbalance", lambda *a, **k: "balanced")
    out = tmp_path / "c.yaml"
    sw.generate_config(out, force=False, cuda_available=False)
    cfg = load_config(out)
    assert cfg.run.name == "my-run"
    assert cfg.data.val_split is not None
    assert cfg.train.epochs == 7


def test_validate_backstop_exits_nonzero_no_file(tmp_path, monkeypatch, capsys) -> None:
    # Inject an answers dict that renders invalid (epochs 0 → PositiveInt violation).
    monkeypatch.setattr(sw, "run_wizard", lambda ctx: {
        "run": {"name": "r"},
        "data": {"format": "coco", "train": {"annotations": "t.json", "images": "t/"},
                 "augmentations": {"preset": "natural", "intensity": "medium"}},
        "peft": {"method": "lora"},
        "train": {"epochs": 0, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    })
    out = tmp_path / "c.yaml"
    with pytest.raises(typer.Exit):
        sw.generate_config(out, force=False, cuda_available=False)
    assert not out.exists()


def test_ctrl_c_writes_nothing(tmp_path, monkeypatch) -> None:
    def _boom(ctx):
        raise KeyboardInterrupt
    monkeypatch.setattr(sw, "run_wizard", _boom)
    out = tmp_path / "c.yaml"
    with pytest.raises(KeyboardInterrupt):
        sw.generate_config(out, force=False, cuda_available=False)
    assert not out.exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "generate_config or backstop or ctrl_c" -v`
Expected: FAIL — `sw.generate_config` not defined.

- [ ] **Step 3: Implement `generate_config`**

Add to `setup_wizard.py`. It builds the final body string ONCE (header + rendered), validates those exact bytes, then writes — so validate input == emitted bytes:

```python
import sys

from custom_sam_peft.errors import ConfigError


def generate_config(output: Path, *, force: bool, cuda_available: bool) -> tuple[str, RunMode]:
    """Run the wizard, validate, write. Returns (launch_command, run_mode).

    Raises:
      typer.Exit(1): final validation backstop fired (no file written).
      KeyboardInterrupt: propagates; no file written.
    """
    ctx = Ctx(answers={}, cuda_available=cuda_available)
    answers = run_wizard(ctx)  # KeyboardInterrupt propagates out untouched
    rendered = render(answers, run_mode=ctx.run_mode)
    launch = _launch_command(output, ctx.run_mode)
    header = (
        f"# Generated by `custom-sam-peft init --interactive` on {date.today().isoformat()}\n"
        f"# Launch: {launch}\n\n"
    )
    body = header + rendered
    try:
        validate(body)
    except ConfigError as exc:
        typer.echo(f"error: generated config failed validation:\n{exc}", err=True)
        typer.echo(f"answers: {answers}", err=True)
        raise typer.Exit(code=1) from exc
    if output.exists() and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing {output}; pass --force",
            param_hint="--output",
        )
    output.write_text(body)
    typer.echo(f"wrote {output}")
    typer.echo(launch)
    return launch, ctx.run_mode
```

Note: `emit` (Task 14) remains as a tested unit but `generate_config` is the driver `init` calls (it inlines the same header+write so the validated bytes are byte-identical to what is written). The `test_emit_*` tests still exercise `emit` directly.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "generate_config or backstop or ctrl_c" -v`
Expected: PASS — happy path writes a reloadable file; invalid answers raise `typer.Exit` and write nothing; KeyboardInterrupt writes nothing.

- [ ] **Step 5: Add the VRAM-autosize step tests (§10.3)**

Add to the test file (monkeypatch `decide_preset` at the `setup_wizard` import site):

```python
def test_vram_autosize_applies_config_patch(monkeypatch) -> None:
    from custom_sam_peft.presets import PresetDecision

    decision = PresetDecision(
        method="qlora", r=16, batch_size=2, grad_accum_steps=8, dtype="bfloat16",
        headroom_bytes=0, predicted_bytes=0, budget_bytes=0, image_size=1008,
        gpu_name="StubGPU", provenance="analytic", cache_path=None, calibrated_at=None,
    )
    monkeypatch.setattr("custom_sam_peft.presets.decide_preset", lambda image_size: decision)
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: True)
    ctx = sw.Ctx(answers={}, cuda_available=True)
    frag = sw._ask_peft_sizing(ctx)
    assert frag == decision.config_patch
    assert "gradient_checkpointing" not in frag["model"]


def test_vram_autosize_runtime_error_falls_back_to_manual(monkeypatch) -> None:
    def _boom(image_size):
        raise RuntimeError("nothing fits")
    monkeypatch.setattr("custom_sam_peft.presets.decide_preset", _boom)
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: True)
    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: "qlora")
    ctx = sw.Ctx(answers={}, cuda_available=True)
    frag = sw._ask_peft_sizing(ctx)
    assert frag == {"peft": {"method": "qlora"}}
```

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k vram -v`
Expected: PASS (config_patch applied verbatim, no GC key; RuntimeError → manual prompt). If the patch site differs, ensure `_ask_peft_sizing` does `from custom_sam_peft.presets import decide_preset` at call time (it does — lazy import) so the `setattr` on `custom_sam_peft.presets.decide_preset` takes effect.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): generate_config orchestration + backstop + VRAM-step tests"
```

---

## Phase 6 — Wire `--interactive` into `init` (WS1)

### Task 16: Add `--interactive`/`-i` flag + pre-flight to `init_cmd.init`

**Files:**

- Modify: `src/custom_sam_peft/cli/init_cmd.py`
- Test: `tests/unit/cli/test_setup_wizard.py` (CLI-level cases) or `tests/unit/test_cli_init.py`

- [ ] **Step 1: Write failing CLI-level tests**

Add to `tests/unit/cli/test_setup_wizard.py`:

```python
from typer.testing import CliRunner
from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_non_tty_hard_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.sys.stdin.isatty", lambda: False)
    called = []
    monkeypatch.setattr("custom_sam_peft.cli.setup_wizard.run_wizard", lambda ctx: called.append(1) or {})
    out = tmp_path / "c.yaml"
    result = runner.invoke(app, ["init", "--interactive", "--output", str(out)])
    assert result.exit_code != 0
    assert "TTY" in result.output or "tty" in result.output.lower()
    assert called == []
    assert not out.exists()


def test_output_exists_without_force_errors_before_prompting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.sys.stdin.isatty", lambda: True)
    called = []
    monkeypatch.setattr("custom_sam_peft.cli.setup_wizard.run_wizard", lambda ctx: called.append(1) or {})
    out = tmp_path / "c.yaml"
    out.write_text("existing\n")
    result = runner.invoke(app, ["init", "--interactive", "--output", str(out)])
    assert result.exit_code != 0
    assert called == []
    assert out.read_text() == "existing\n"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "non_tty or output_exists" -v`
Expected: FAIL — `init` has no `--interactive` flag yet (Typer errors on the unknown option, or the flag is ignored and the wizard never branches).

- [ ] **Step 3: Add the flag + branch to `init()`**

In `src/custom_sam_peft/cli/init_cmd.py`, add an `interactive` option and branch at the top of `init()` before the flag-driven `run_init`:

```python
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help=(
            "Run the interactive setup wizard. Ignores --template/--preset/"
            "--intensity/--class-imbalance (collected interactively)."
        ),
    ),
```

At the top of the function body (before the `try: run_init(...)`):

```python
    if interactive:
        import torch

        from custom_sam_peft.cli import setup_wizard

        if not sys.stdin.isatty():
            raise typer.BadParameter(
                "interactive setup needs a TTY; use the flag-driven "
                "`custom-sam-peft init …` instead"
            )
        if output.exists() and not force:
            raise typer.BadParameter(
                f"refusing to overwrite existing {output}; pass --force",
                param_hint="--output",
            )
        setup_wizard.generate_config(
            output, force=force, cuda_available=torch.cuda.is_available()
        )
        _maybe_download_weights(output, download_weights=download_weights, yes=yes)
        return
```

(The existing flag-driven body — `run_init` + `rprint` + `_maybe_download_weights` — stays unchanged below for the non-interactive path.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "non_tty or output_exists" -v`
Expected: PASS — non-TTY errors before prompting (wizard not called, no file); output-exists errors before prompting (existing file untouched).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/init_cmd.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(init): add --interactive/-i flag with TTY + output pre-flight"
```

---

## Phase 7 — GPU test adjustment (WS2)

### Task 17: Adjust the single GPU multiplex VRAM test

**Files:**

- Modify: `tests/gpu/test_multiplex_vram.py`

- [ ] **Step 1: Drop the GC arg from the GPU test's `ModelConfig`**

In `tests/gpu/test_multiplex_vram.py` line ~32, change:

```python
    cfg = ModelConfig(device="cuda", gradient_checkpointing=False, dtype="bfloat16")
```

to:

```python
    cfg = ModelConfig(device="cuda", dtype="bfloat16")
```

(The `decide_eval_batch_size(...)` call already returns the GC-free shape; no other change needed. No new GPU test is added.)

- [ ] **Step 2: Verify it collects on CPU (GPU markers skip the body)**

Run: `uv run pytest tests/gpu/test_multiplex_vram.py --collect-only -q`
Expected: collects without error (the `ModelConfig(...)` is now valid; the test body is GPU-gated and skipped on CPU).

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_multiplex_vram.py
git commit -m "test(gpu): drop gradient_checkpointing from multiplex VRAM test config"
```

---

## REVIEW CHECKPOINT B — both workstreams complete

- [ ] Run: `! grep -rn "gradient_checkpointing\|grad_ckpt\|CKPT_FACTOR\|coco_text_lora.yaml\|coco_text_qlora.yaml" src/`
      Expected: no matches in `src/` (the two legacy template basenames are deleted; GC is gone). `configs/examples/*.yaml` basenames may still appear in tests legitimately — scope the grep to `src/`.
- [ ] Dispatch a code-review subagent (min sonnet/high; opus/xhigh if reviewing the wizard's render/validate seam, which is design-sensitive): confirm (a) the wizard renders the same `config_full.yaml` both paths use, (b) `generate_config` validates the exact bytes it writes, (c) `config_patch` is applied verbatim with no GC stripping, (d) the registry is a one-line-edit extension point, (e) no GC residue anywhere.

---

## Phase 8 — Final verification (do not run during planning; these are plan steps)

### Task 18: Full-suite + lint + type + markdown verification

- [ ] **Step 1: Ruff lint**

Run: `uv run ruff check`
Expected: no findings (fix any before proceeding).

- [ ] **Step 2: Ruff format check**

Run: `uv run ruff format --check`
Expected: clean (run `uv run ruff format` to fix, then re-check).

- [ ] **Step 3: mypy**

Run: `uv run mypy src/custom_sam_peft`
Expected: no errors. (`setup_wizard.py` uses typed signatures; the `# type: ignore` on `decide_preset`'s `method` and the `run_mode` assignment are pre-resolved in the code above.)

- [ ] **Step 4: FULL pytest suite (the 80% coverage gate only passes on the full suite)**

Run: `uv run pytest`
Expected: all PASS; `--cov-fail-under=80` satisfied. (Do NOT run a subset for the gate — `addopts` enforces coverage across the whole run.)

- [ ] **Step 5: yamllint the configs/templates touched**

Run: `uv run --with yamllint yamllint -c .config/yamllint.yml .`
Expected: clean. The entire `src/custom_sam_peft/cli/templates/` directory is already in `.config/yamllint.yml`'s `ignore:` list (templates carry `$...` placeholders and are not standalone-valid YAML), so the new `config_full.yaml` is excluded automatically — no yamllint-config edit needed. The edited `configs/examples/*.yaml` (Task 6) ARE linted, so confirm those are clean.

- [ ] **Step 6: markdownlint the touched docs**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"`
Expected: no findings on `docs/config-schema.md` and this plan file (fix any).

- [ ] **Step 7: Commit any lint/format fixups**

```bash
git add -A
git commit -m "chore: lint/format/type fixups for wizard + GC removal"
```

---

## Phase 9 — Amendment: HF explicit-validation wiring (post-review, spec §12)

> **Why this exists:** Checkpoint-B review found that the wizard's HF + "explicit" validation
> mode is a no-op end-to-end — `resolve_val_source` ignores `hf.split_val`, `eval/runner`
> rejects it, and the wizard `render` drops it. Resolution: wire `hf.split_val` for real
> (spec §12). Sequence: schema (opt-in signal + validator retarget) → resolver + eval gate
> (the runtime wiring) → wizard render → docs. Tasks 19→20→21 are serialized (19 establishes
> the `None` opt-in the rest depend on); Task 22 (docs) is independent. All CPU-testable — no
> real HF dataset is loaded; tests assert the mode decision and gate logic only.

### Task 19: Schema — `split_val` Optional opt-in + validator retarget

**Files:**

- Modify: `src/custom_sam_peft/config/schema.py` (`HFDatasetConfig.split_val` ~line 322; `_check_hf_split_val_compat` ~lines 470-482)
- Test: `tests/unit/test_data_schema_extensions.py`, `tests/unit/test_config_schema.py`

- [ ] **Step 1: Update the failing tests for the `None` default + new opt-in semantics**

In `tests/unit/test_data_schema_extensions.py`, change the default assertion (line ~85):

```python
def test_hf_dataset_config_defaults() -> None:
    cfg = HFDatasetConfig(name="my-org/my-ds")
    assert cfg.name == "my-org/my-ds"
    assert cfg.split_train == "train"
    assert cfg.split_val is None
    assert cfg.field_map.bbox == "objects.bbox"
```

In `tests/unit/test_config_schema.py`, update the comment in `test_hf_split_val_default_with_val_split_validates` (line ~313) and add a new test that HF + `split_val` set (no `val`/`val_split`) validates:

```python
def test_hf_split_val_default_with_val_split_validates() -> None:
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {"name": "tiny/dataset"}  # default split_val=None
    d["data"]["val"] = None  # type: ignore[index]
    d["data"]["val_split"] = {"fraction": 0.1, "seed": 7}  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val_split is not None
    assert cfg.data.val_split.fraction == 0.1
    assert cfg.data.val_split.seed == 7


def test_hf_split_val_set_without_val_split_validates() -> None:
    """spec §12.3: HF + named split_val (no val/val_split) is the explicit opt-in."""
    d = _minimal_dict()
    d["data"]["format"] = "hf"  # type: ignore[index]
    d["data"]["hf"] = {"name": "tiny/dataset", "split_val": "myval"}  # type: ignore[index]
    d["data"]["val"] = None  # type: ignore[index]
    cfg = TrainConfig.model_validate(d)
    assert cfg.data.val_split is None
    assert cfg.data.hf is not None
    assert cfg.data.hf.split_val == "myval"
```

(`test_hf_split_val_custom_with_val_split_rejected` is unchanged — `"custom_val"` is not None, still rejected.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_schema_extensions.py::test_hf_dataset_config_defaults tests/unit/test_config_schema.py -k "split_val or neither_val" -v`
Expected: FAIL — `split_val` still defaults to `"validation"` so `assert cfg.split_val is None` fails; the new opt-in test passes already (default is harmless), but the default-value test is red.

- [ ] **Step 3: Make `split_val` Optional and retarget the validator**

In `src/custom_sam_peft/config/schema.py`, change the `HFDatasetConfig.split_val` field (line ~322):

```python
class HFDatasetConfig(_Strict):
    """HuggingFace dataset specification (used when DataConfig.format == 'hf')."""

    name: str = Field(min_length=1)
    split_train: str = "train"
    split_val: str | None = None
    field_map: HFFieldMap = Field(default_factory=HFFieldMap)
```

Retarget `_check_hf_split_val_compat` (lines ~470-482) from the `"validation"` sentinel to the `None` sentinel:

```python
    @model_validator(mode="after")
    def _check_hf_split_val_compat(self) -> DataConfig:
        if (
            self.format == "hf"
            and self.val_split is not None
            and self.hf is not None
            and self.hf.split_val is not None
        ):
            raise ValueError(
                "data.hf.split_val cannot be customized when data.val_split is set; "
                "auto-split carves the val set from data.hf.split_train. "
                "Remove split_val or remove val_split."
            )
        return self
```

(`_check_val_modes` is unchanged. No guard is added for `val` + `hf.split_val` — they are format-disjoint, per spec §12.3.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_schema_extensions.py tests/unit/test_config_schema.py -v`
Expected: PASS — default is `None`; HF + `split_val` (no val_split) validates; HF + `split_val` + `val_split` still rejected; HF default + `val_split` still validates.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_data_schema_extensions.py tests/unit/test_config_schema.py
git commit -m "feat(schema)!: make hf.split_val Optional as explicit-val opt-in (spec §12)"
```

### Task 20: Resolver + eval gate — honor `hf.split_val` (mode='explicit')

**Files:**

- Modify: `src/custom_sam_peft/data/val_source.py` (`resolve_val_source` ~lines 41-92; optional `_log_val_source` ~line 149)
- Modify: `src/custom_sam_peft/eval/runner.py` (the `--split val` gate ~line 104)
- Test: `tests/unit/test_val_source.py` (existing val-source unit tests; add cases), `tests/unit/test_eval_runner_gate.py` (new, gate-only)

- [ ] **Step 1: Write the failing resolver + gate tests**

Add to `tests/unit/test_val_source.py` (find the module that imports `resolve_val_source`; if it does not exist, create `tests/unit/test_val_source.py` with the import). Use a minimal valid HF `TrainConfig` built via `TrainConfig.model_validate` so no dataset is loaded:

```python
from custom_sam_peft.data.val_source import resolve_val_source


def _hf_cfg(split_val: str | None) -> "TrainConfig":
    from custom_sam_peft.config.schema import TrainConfig

    hf: dict[str, object] = {"name": "tiny/ds"}
    if split_val is not None:
        hf["split_val"] = split_val
    return TrainConfig.model_validate(
        {
            "run": {"name": "r"},
            "model": {},
            "data": {
                "format": "hf",
                "train": {"annotations": "unused", "images": "unused"},
                "val": None,
                "prompt_mode": "text",
                "hf": hf,
            },
            "peft": {"method": "lora"},
            "train": {"epochs": 1},
        }
    )


def test_resolve_hf_split_val_is_explicit() -> None:
    vs = resolve_val_source(_hf_cfg("myval"), run_dir=None)
    assert vs.mode == "explicit"
    assert vs.train_ids is None
    assert vs.val_ids is None


def test_resolve_hf_no_split_val_is_none() -> None:
    vs = resolve_val_source(_hf_cfg(None), run_dir=None)
    assert vs.mode == "none"
```

Create `tests/unit/test_eval_runner_gate.py` to exercise ONLY the gate (short-circuit the build/load so nothing real is loaded):

```python
"""The eval-runner --split val gate accepts HF + hf.split_val (spec §12.5)."""

from __future__ import annotations

import pytest

from custom_sam_peft.config.schema import TrainConfig


def _hf_cfg(split_val: str | None) -> TrainConfig:
    hf: dict[str, object] = {"name": "tiny/ds"}
    if split_val is not None:
        hf["split_val"] = split_val
    return TrainConfig.model_validate(
        {
            "run": {"name": "r"},
            "model": {},
            "data": {
                "format": "hf",
                "train": {"annotations": "unused", "images": "unused"},
                "val": None,
                "prompt_mode": "text",
                "hf": hf,
            },
            "peft": {"method": "lora"},
            "train": {"epochs": 1},
        }
    )


def _gate_only(cfg: TrainConfig, split: str) -> None:
    """Replicate eval/runner's --split val gate in isolation (no model/data load)."""
    _hf_val = cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None
    if split == "val" and cfg.data.val is None and cfg.data.val_split is None and not _hf_val:
        raise ValueError(
            "--split val requires data.val, data.val_split, or data.hf.split_val in config; got none."
        )


def test_gate_accepts_hf_split_val() -> None:
    _gate_only(_hf_cfg("myval"), "val")  # must not raise


def test_gate_rejects_hf_without_split_val() -> None:
    with pytest.raises(ValueError, match="data.hf.split_val"):
        _gate_only(_hf_cfg(None), "val")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_val_source.py -k "hf_split_val or hf_no_split" tests/unit/test_eval_runner_gate.py -v`
Expected: FAIL — `resolve_val_source` returns `mode='none'` for HF + split_val (no HF branch yet); the gate test file's `_gate_only` mirrors the NEW condition, so its tests pass only once the runner matches; the resolver tests fail.

- [ ] **Step 3: Add the HF-explicit branch to `resolve_val_source`**

In `src/custom_sam_peft/data/val_source.py`, insert the new branch between the COCO `val` branch and the final `none` return (after line ~81):

```python
    if cfg.data.val is not None:
        return ValSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            fraction_requested=None,
            seed_used=None,
        )

    if (
        cfg.data.format == "hf"
        and cfg.data.hf is not None
        and cfg.data.hf.split_val is not None
    ):
        return ValSource(
            mode="explicit",
            train_ids=None,
            val_ids=None,
            realized_fraction=None,
            per_class_counts=None,
            missing_in_val=None,
            fraction_requested=None,
            seed_used=None,
        )

    return ValSource(
        mode="none",
        ...
    )
```

Update the dispatch docstring (lines ~44-48) to add the HF-explicit step:

```python
    """Resolve which validation source to use for this run.

    Dispatch (spec §5.2 + §12.4):
      1. run_dir/val_source.json exists → load_val_source(run_dir) (resume).
      2. cfg.data.val_split is not None → enumerate + stratify (auto_split).
      3. cfg.data.val is not None → mode='explicit' (COCO).
      4. cfg.data.format=='hf' and cfg.data.hf.split_val is not None → mode='explicit' (HF).
      5. else → mode='none'.
    """
```

Optionally broaden the `_log_val_source` explicit line (line ~150) to `"val source: explicit (cfg.data.val or data.hf.split_val)"` (cosmetic, log-only).

- [ ] **Step 4: Relax the `eval/runner.py` gate**

In `src/custom_sam_peft/eval/runner.py`, replace the gate at ~line 104:

```python
    _hf_val = (
        cfg.data.format == "hf" and cfg.data.hf is not None and cfg.data.hf.split_val is not None
    )
    if split == "val" and cfg.data.val is None and cfg.data.val_split is None and not _hf_val:
        raise ValueError(
            "--split val requires data.val, data.val_split, or data.hf.split_val in config; "
            "got none."
        )
```

(The `val_dataset is None` build path below is unchanged: HF-explicit has no `_resolved_image_ids` and no `val`, so the HF builder's `else` branch reads `split_val` for `pipeline="eval"`. `data/hf.py` and `train/runner.py` need no edits — spec §12.6.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_val_source.py tests/unit/test_eval_runner_gate.py -v`
Expected: PASS — HF + split_val resolves to `explicit` with `train_ids/val_ids` None; HF without split_val resolves to `none`; gate accepts HF + split_val, rejects HF without it.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/data/val_source.py src/custom_sam_peft/eval/runner.py tests/unit/test_val_source.py tests/unit/test_eval_runner_gate.py
git commit -m "feat(val): resolve+eval honor hf.split_val as explicit validation (spec §12)"
```

### Task 21: Wizard render — emit `split_val:` for HF-explicit; no contradictory no-val line

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py` (`_dataset_block` ~lines 144-173; `_validation_block` ~lines 176-203)
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write failing render tests**

Add to `tests/unit/cli/test_setup_wizard.py`:

```python
def test_render_hf_explicit_emits_split_val(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "hf",
            "hf": {"name": "org/ds", "split_val": "myval"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 2, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "split_val: myval" in rendered
    # No spurious COCO train: block, and no active no-val claim:
    assert "  # no-val mode:" not in rendered  # the active no-val line must not appear
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.format == "hf"
    assert cfg.data.hf is not None
    assert cfg.data.hf.split_val == "myval"


def test_render_hf_none_emits_no_split_val(tmp_path) -> None:
    answers = {
        "run": {"name": "r"},
        "data": {
            "format": "hf",
            "hf": {"name": "org/ds"},
            "augmentations": {"preset": "natural", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1, "loss": {"preset": "natural", "class_imbalance": "balanced"}},
    }
    rendered = sw.render(answers, run_mode="train")
    assert "split_val:" not in rendered.replace("#   split_val", "")  # no ACTIVE split_val line
    out = tmp_path / "c.yaml"
    out.write_text(rendered)
    cfg = load_config(out)
    assert cfg.data.hf is not None
    assert cfg.data.hf.split_val is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "hf_explicit_emits or hf_none_emits" -v`
Expected: FAIL — current `_dataset_block` HF branch drops `split_val` (no `split_val: myval`) and emits a spurious `train:` block; `_validation_block` emits an active no-val line for HF-explicit.

- [ ] **Step 3: Fix `_dataset_block` (drop spurious COCO train; render split_val)**

In `src/custom_sam_peft/cli/setup_wizard.py`, rewrite the HF branch of `_dataset_block`:

```python
def _dataset_block(answers: dict[str, Any]) -> str:
    data = answers.get("data", {})
    if data.get("format") == "hf":
        hf = data.get("hf", {})
        name = hf["name"]
        lines = [
            "  format: hf",
            "  hf:",
            f"    name: {name}",
            "    split_train: train",
        ]
        if hf.get("split_val") is not None:
            lines.append(f"    split_val: {hf['split_val']}")
        lines += [
            "  # COCO alternative — set format: coco and uncomment:",
            "  # train:",
            "  #   annotations: data/train.json",
            "  #   images: data/train/",
        ]
        return "\n".join(lines)
    train = data.get("train", {})
    ann = train.get("annotations", "data/train.json")
    imgs = train.get("images", "data/train/")
    return (
        "  format: coco\n"
        "  train:\n"
        f"    annotations: {ann}\n"
        f"    images: {imgs}\n"
        "  # HuggingFace alternative — set format: hf and uncomment:\n"
        "  # hf:\n"
        "  #   name: org/dataset\n"
        "  #   split_train: train\n"
        "  #   split_val: validation"
    )
```

- [ ] **Step 4: Fix `_validation_block` (HF-explicit must not claim no-val)**

In `_validation_block`, detect HF-explicit (format == "hf" with `hf.split_val` set) and render an active note that points at the dataset block's `split_val`, instead of the no-val line:

```python
def _validation_block(answers: dict[str, Any]) -> str:
    data = answers.get("data", {})
    hf = data.get("hf", {})
    hf_explicit = data.get("format") == "hf" and hf.get("split_val") is not None
    explicit_active = auto_active = noval_active = False
    if data.get("val") is not None:
        explicit_active = True
        v = data["val"]
        active = f"  val:\n    annotations: {v['annotations']}\n    images: {v['images']}"
    elif data.get("val_split") is not None:
        auto_active = True
        active = f"  val_split:\n    fraction: {data['val_split']['fraction']}\n    seed: null"
    elif hf_explicit:
        # Validation comes from data.hf.split_val (rendered in the dataset block).
        active = "  # validation: HF split set via data.hf.split_val above is used as the val set."
    else:
        noval_active = True
        active = "  # no-val mode: neither val: nor val_split: is set."
    alts = []
    if not explicit_active:
        alts.append(
            "  # Explicit-val alternative (COCO):\n"
            "  # val:\n"
            "  #   annotations: data/val.json\n"
            "  #   images: data/val/"
        )
    if not auto_active:
        alts.append(
            "  # Auto-split alternative:\n  # val_split:\n  #   fraction: 0.1\n  #   seed: null"
        )
    if not noval_active:
        alts.append("  # No-val alternative: omit val:, val_split:, and hf.split_val.")
    return "\n".join([active, *alts])
```

(The COCO-branch comment scaffold may keep the illustrative `# split_val: validation` in its commented HF-alternative — it is documentation text, not gating.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "hf_explicit_emits or hf_none_emits or render" -v`
Expected: PASS — HF-explicit renders `split_val: myval`, no active no-val line, reloads with `cfg.data.hf.split_val == "myval"`; HF-none emits no active `split_val:` and reloads with `split_val is None`; existing render tests still green.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "fix(wizard): render hf.split_val for HF explicit validation (spec §12)"
```

### Task 22: Docs — update the `data.hf.split_val` schema row

**Files:**

- Modify: `docs/config-schema.md` (the `data.hf.split_val` row ~line 82)

- [ ] **Step 1: Update the row**

Change the `data.hf.split_val` row to reflect the `null` default and explicit-HF-val behavior:

```markdown
| `data.hf.split_val` | str \| null | `null` | advanced | HF split name used as the validation set. When set (and `data.val_split` is unset), validation runs against this split (mode='explicit'). Null → no HF-driven validation. | Audit §E: explicit HF val opt-in (spec §12). |
```

- [ ] **Step 2: Markdown-lint the touched doc**

Run: `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc docs/config-schema.md`
Expected: no findings (fix any before committing).

- [ ] **Step 3: Commit**

```bash
git add docs/config-schema.md
git commit -m "docs: hf.split_val default null + explicit-HF-val behavior (spec §12)"
```

### REVIEW CHECKPOINT C — amendment complete

- [ ] Run: `uv run pytest tests/unit/test_data_schema_extensions.py tests/unit/test_config_schema.py tests/unit/test_val_source.py tests/unit/test_eval_runner_gate.py tests/unit/cli/test_setup_wizard.py -q`
      Expected: all PASS.
- [ ] Run the FULL suite for the coverage gate: `uv run pytest` → all PASS, `--cov-fail-under=80` satisfied.
- [ ] Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/custom_sam_peft` → clean.
- [ ] Dispatch a code-review subagent (min sonnet/high) over the §12 diff: confirm (a) `split_val is not None` is the single opt-in signal across schema/resolver/eval-gate, (b) the resolver's branch ordering (val_split before hf.split_val) plus the retargeted validator guarantees the two never both fire, (c) the HF builder reads `split_val` only when not None, (d) the wizard render emits `split_val:` for HF-explicit and never an active no-val line, (e) no consumer still relies on the `"validation"` default.

---

## Self-review (against the spec, after writing the plan)

- **§1 in-scope files — all represented:** `setup_wizard.py` (Tasks 10–15), `config_full.yaml` (Task 7), legacy templates deleted (Task 9), `init_cmd.py` (Tasks 8, 16), `schema.py` (Task 1), `presets.py` (Task 2), `models/sam3.py` (Task 3), `train/loop.py` (Task 4), `train/types.py` (Task 4), `runs/bundle.py` (Task 5), `docs/config-schema.md` (Task 6). **Plus** `configs/examples/*.yaml` (Task 6) — not in the spec's §1 table but a required consequence of the schema break (else `test_config_examples.py` fails to load all 7 examples). Flagged explicitly as a breaking-change consequence, handled to keep the repo's own configs loadable.
- **§10 tests — all represented:** §10.1 wizard cases → `tests/unit/cli/test_setup_wizard.py` (Tasks 10, 12, 13, 14, 15, 16); §10.2 `infer_class_imbalance` → Task 11; §10.3 VRAM auto-size → Task 15 Step 5; §10.4 GC removal: `test_model_config.py` (Task 1), `test_presets.py` (Task 2), `test_trainer_oom_retry.py` + `test_train_types.py` (Task 4), `test_bundle.py` (Task 5), `test_data_transforms.py` (Task 6), `test_load_sam31_real.py` (Task 3), `test_cli_run.py` (Task 5); §10.5 GPU `test_multiplex_vram.py` (Task 17); §10.6 flag-driven init `test_cli_init.py` (Task 8). **Plus** `test_decide_eval_batch_size.py` (Task 2) and `test_config_examples.py` (Task 6) — not named in §10 but break under the change; covered.
- **TDD:** every behavioral task writes the failing test first, runs it red, implements, runs it green. Wizard/class-imbalance/VRAM/non-TTY/force/ctrl-c are CPU-only (monkeypatched primitives + `decide_preset`); GC-removal tests are CPU; only the one existing GPU test is adjusted.
- **Sequencing:** WS2 (Phases 1–3) before WS1's VRAM step (Phase 5); all `presets.py` edits in one task; unified template authored before legacy deletion before wizard. Review checkpoints between workstreams (A) and before finish (B).
- **No migration:** the breaking-change note is stated up front, not a task.
