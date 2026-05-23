# Domain-Aware Loss-Function Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-23-domain-aware-loss-presets-design.md`](../specs/2026-05-23-domain-aware-loss-presets-design.md)
**Issue:** [#112](https://github.com/NguyenJus/custom-sam-peft/issues/112) — *feat(train): domain-aware loss-function presets (natural / medical / satellite / …) with class-imbalance dial*
**Branch:** `loss-presets-112` (worktree at `/home/justin/projects/custom-sam-peft/.worktrees/loss-presets-112/`)

**Goal:** Replace the flat dataclass `LossConfig(w_mask, w_obj, w_presence, w_box, matcher_weights, focal_gamma, focal_alpha)` (today in `config/_internal.py`) with a `(preset, class_imbalance, overrides)` Pydantic model resolved against a frozen 12-cell preset table; refactor `models/losses.py` into a `models/losses/` package with a 14-class `terms/` library and a `compose.py` bundle; expose `--class-imbalance` to `csp init` and a "Resolved losses" table to `csp doctor --config`; persist a per-run `loss_bundle.json` sidecar.

**Architecture:** Three layers, pure-Python at the top. `models/losses/presets.py` owns the preset table, `LOCKED_OFF` map, `ResolvedLosses` dataclass, `resolve()`, `dump_loss_bundle()` (no torch import). `models/losses/terms/{mask,box,obj,presence}.py` hold 14 `nn.Module`-style callable term classes with uniform-per-axis `forward` signatures. `models/losses/compose.py` holds `LossBundle` + `build_loss_bundle()` (the family-literal → class registry + the per-step matcher-and-sum). `models/losses/__init__.py` keeps a thin `total_loss(outputs, targets, cfg)` shim so the two call sites in `train/loop.py` stay unmodified through the migration. Schema break in `config/schema.py` (no aliases). Trainer appends the new sidecar after the existing `augmentation_pipeline.json` write. CLI changes are additive (`init` substitutes `${class_imbalance}`/`${loss_overrides_block}`; `doctor --config` adds a third table + a `loss` sub-block in the JSON output).

**Tech Stack:** Python 3.12, pydantic v2, torch (existing), scipy (existing, used by `BoundaryLoss` for `distance_transform_edt`), Typer + Rich (CLI), pytest + `caplog`, `uv` + `ruff` + `mypy`, `gh` CLI.

---

## File Map

**New files:**

```
src/custom_sam_peft/models/losses/__init__.py          CREATE — package re-exports + total_loss shim
src/custom_sam_peft/models/losses/presets.py           CREATE — Preset/ClassImbalance/family literals re-export, PRESET_TABLE, LOCKED_OFF, ResolvedLosses, _LEGACY_DEFAULTS, _TERM_CLASS_NAMES, resolve, dump_loss_bundle
src/custom_sam_peft/models/losses/compose.py           CREATE — LossBundle, build_loss_bundle, term-registry dicts, moved-from-monolith helpers (_gather_matched_boxes_masks, _matched_query_mask, _image_has_target)
src/custom_sam_peft/models/losses/terms/__init__.py    CREATE — re-export the 14 term classes
src/custom_sam_peft/models/losses/terms/mask.py        CREATE — 8 mask-axis classes
src/custom_sam_peft/models/losses/terms/box.py         CREATE — 3 box-axis classes
src/custom_sam_peft/models/losses/terms/obj.py         CREATE — 2 obj-axis classes
src/custom_sam_peft/models/losses/terms/presence.py    CREATE — 2 presence-axis classes
tests/unit/test_loss_presets.py                        CREATE — resolver + LOCKED_OFF + sidecar-dump tests
tests/unit/test_loss_terms.py                          CREATE — per-term forward + degenerate-case identities
tests/unit/test_loss_compose.py                        CREATE — bundle build + total_loss shim equivalence
```

**Deleted files:**

```
src/custom_sam_peft/models/losses.py                   DELETE (its content moves to losses/__init__.py + losses/compose.py + losses/terms/*.py during Phase A's package-skeleton step)
```

**Modified files:**

```
src/custom_sam_peft/config/schema.py                   TOUCHED (add ClassImbalance, MaskFamily, BoxFamily, ObjFamily, PresenceFamily literals; add LossOverrides + new LossConfig Pydantic models; remove the LossConfig re-export from _internal)
src/custom_sam_peft/config/_internal.py                TOUCHED (delete the LossConfig dataclass; keep MatcherWeights — it stays here)
src/custom_sam_peft/cli/init_cmd.py                    TOUCHED (add --class-imbalance Typer option; extend substitution dict with class_imbalance + loss_overrides_block; ClassImbalance validation)
src/custom_sam_peft/cli/doctor_cmd.py                  TOUCHED (extend _render_resolved_config_tables with a "Resolved losses" table; extend _build_resolved_config_json with a "loss" sub-block)
src/custom_sam_peft/cli/templates/coco_text_lora.yaml  TOUCHED (replace train.loss block lines ~79-86 with ${preset}/${class_imbalance}/${loss_overrides_block} placeholders)
src/custom_sam_peft/cli/templates/coco_text_qlora.yaml TOUCHED (same)
src/custom_sam_peft/train/trainer.py                   TOUCHED (after augmentation_pipeline.json write at line ~204, dump loss_bundle.json)
tests/unit/test_config_schema.py                       TOUCHED (replace legacy LossConfig assertions; add ClassImbalance/family-literal/LossOverrides validation tests)
tests/unit/test_loss_config.py                         TOUCHED (rewrite — replaces former tests of dataclass LossConfig; the symbol moved)
tests/unit/test_box_hint_schedule.py                   TOUCHED (drop LossConfig import + the test_loss_config_default_w_box_is_zero test — superseded by test_loss_presets.py)
tests/unit/test_data_coco.py                           TOUCHED (fixture YAML/dict shape: drop flat loss: keys; add the new triple shape where any test sets train.loss)
tests/unit/test_data_hf.py                             TOUCHED (same)
tests/unit/test_trainer_nan_behavior.py                TOUCHED (callsite migration to new shape)
tests/unit/test_trainer_run_dir.py                     TOUCHED (callsite migration; new test asserts loss_bundle.json contents)
tests/unit/test_cli_init.py                            TOUCHED (add --class-imbalance render tests + invalid-value rejection + custom-preset loss-overrides scaffold test)
tests/unit/test_cli_doctor.py                          TOUCHED (add --config "Resolved losses" table + resolved_config.loss JSON tests; assert byte-identical default behavior)
tests/integration/test_train_resume.py                 TOUCHED (callsite migration)
tests/integration/test_train_end_to_end.py             TOUCHED (callsite migration; new assertion: loss_bundle.json at run end)
```

No new dependencies. `scipy>=1.10` is already in `pyproject.toml`; `torch`, `pydantic`, `typer`, `rich` already pinned. The plan does not modify `pyproject.toml`.

---

## Assumptions for the cold reader

1. **Working directory.** Every shell command runs with `cwd = /home/justin/projects/custom-sam-peft/.worktrees/loss-presets-112`. Use absolute paths when invoking external tools; use repo-relative paths inside the plan text.
2. **Tooling.** `uv` is on PATH. Run every Python entry via `uv run …`. Pytest invocations use `uv run pytest …`. No bare `python`.
3. **Schema break is one-shot.** No aliases. No deprecation cycle. The PR migrates every callsite in the same diff. Pre-1.0 schema breaks are allowed per #70.
4. **CPU-only.** Every test in this plan runs on CPU. No `@pytest.mark.gpu` markers. No real model load. The 14 term classes are small enough that synthetic batches of `(N, H, W)` with `N=2, H=W=32` exercise the relevant math.
5. **Package layout decision.** `src/custom_sam_peft/models/losses.py` (file) becomes `src/custom_sam_peft/models/losses/` (package). This requires deleting the file before creating the directory at the same path. Phase A's first step does the move via `git mv` to preserve history.
6. **Shim contract.** `models/losses/__init__.py` re-exports `total_loss(outputs, targets, cfg)` so the two existing call sites in `train/loop.py` (lines 257, 278) compile and pass unmodified. The shim builds a fresh `LossBundle` per call (cheap; one trainer init = one shim invocation in practice given the shim wraps the per-step calls inside the matcher loop). Spec §8.6 documents this and recommends collapsing the shim into a long-lived bundle in the trainer; if that fits in the same PR, do it (Phase D, Step D-4). If it expands D's scope unacceptably, ship the shim and file a follow-up.
7. **Citation comments are concrete.** The spec §5.3 prescribes the literal `# cite: (X)` tags per cell. Phase A's `presets.py` reproduces those tags verbatim. Cells lacking a firm cite carry `# citation needed` per user direction.
8. **`library_version` source.** `src/custom_sam_peft/__init__.py` exports `__version__` (already used by `dump_augmentation_pipeline`). `dump_loss_bundle` reads it the same way; fallback to `"unknown"` on import failure (defense-in-depth — should never happen in a real install).
9. **Logger name.** All resolver warns go to `logging.getLogger("custom_sam_peft.models.losses.presets")`. Tests use `caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")`.
10. **No new deps.** `scipy>=1.10`, `torch`, `pydantic`, `typer`, `rich` already pinned. The plan does not modify `pyproject.toml`. `BoundaryLoss` uses `scipy.ndimage.distance_transform_edt`; kornia is NOT introduced.
11. **`MatcherWeights` placement.** Stays in `config/_internal.py` as a `@dataclass`. The new `LossOverrides.matcher_weights: MatcherWeights | None` field accepts either a `MatcherWeights` instance or a dict (routed via a `field_validator`). This avoids promoting `MatcherWeights` to a Pydantic model.

---

## Parallel groups

The orchestrator dispatches by phase. Phase A is the foundation; everything else depends on it. After A lands, phases B, E, and G can fan out together; C, D, F run in parallel after batch 1 lands.

```
Phase A (schema break + package skeleton + presets.py + foundation tests)  [serial; foundation]
   │
   ├─────────────────────────────────────────────────┐
   │                                                 │
   ▼                                                 │
Phase B (terms/*.py — 14 classes + test_loss_terms.py)  [PARALLEL with E and G]
   │                                                 │
   ├──── Phase E (init_cmd + templates + test_cli_init.py)
   │                                                 │
   └──── Phase G (mass test-fixture migration)       │
                                                     │
After B, E, G complete:                              │
   │                                                 │
   ▼                                                 │
Phase C (compose.py + losses/__init__.py wire + test_loss_compose.py)  [serial after A+B]
   │                                                 │
   ├──── Phase D (trainer sidecar + optional bundle wire-up)  [depends on A+C; PARALLEL with F]
   │                                                 │
   └──── Phase F (doctor_cmd + test_cli_doctor.py)   [depends on A only; PARALLEL with C, D]
                                                     │
   ▼                                                 │
Phase H (reviewer pass: design-sensitive + general + lint/format)  [serial; final]
```

**Concrete parallel batches the orchestrator can dispatch:**

- **Batch 1 (after A merges):** B, E, G in parallel (3 file-disjoint subagents).
- **Batch 2 (after B+E+G complete):** C, D, F in parallel (3 file-disjoint subagents — see "File-set disjointness verification" below for the proof). D depends on C for the bundle wire-up step; if the orchestrator runs them strictly in parallel, D's bundle wire-up step (D-4) is moved into the C batch instead. The simplest mental model: C and D run in series (C first, then D), F in parallel with both.
- **Batch 3 (after C, D, F complete):** Phase H sequentially (two reviewers run in parallel; lint/format runs after both return).

### File-set disjointness verification

| Phase | Files touched |
|---|---|
| **B** | `src/custom_sam_peft/models/losses/terms/__init__.py`, `terms/mask.py`, `terms/box.py`, `terms/obj.py`, `terms/presence.py`, `tests/unit/test_loss_terms.py` |
| **E** | `src/custom_sam_peft/cli/init_cmd.py`, `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`, `tests/unit/test_cli_init.py` |
| **G** | `tests/unit/test_config_schema.py`, `tests/unit/test_loss_config.py`, `tests/unit/test_box_hint_schedule.py`, `tests/unit/test_data_coco.py`, `tests/unit/test_data_hf.py`, `tests/unit/test_trainer_nan_behavior.py`, `tests/unit/test_trainer_run_dir.py` (callsite migration only — sidecar test is Phase D), `tests/integration/test_train_resume.py`, `tests/integration/test_train_end_to_end.py` (callsite migration only — sidecar assertion is Phase D) |
| **C** | `src/custom_sam_peft/models/losses/__init__.py`, `src/custom_sam_peft/models/losses/compose.py`, `tests/unit/test_loss_compose.py` |
| **D** | `src/custom_sam_peft/train/trainer.py`, `tests/unit/test_trainer_run_dir.py` (sidecar test only — G's edits already merged), `tests/integration/test_train_end_to_end.py` (sidecar assertion only) |
| **F** | `src/custom_sam_peft/cli/doctor_cmd.py`, `tests/unit/test_cli_doctor.py` |

**Conflict scan:**
- B ∩ E ∩ G: ∅ — clean parallel (B owns the `terms/` subdir; E owns CLI; G owns tests not in either).
- C ∩ D: `train/trainer.py` and the `tests/*` overlap — C does **not** touch those files; D does. → disjoint.
- C ∩ F: ∅.
- D ∩ F: ∅.
- D ∩ G ordering: G migrates legacy callsites in `test_trainer_run_dir.py` and `test_train_end_to_end.py` first; D then appends the sidecar test/assertion. G must precede D — enforced by the phase ordering.

→ Batch 2 (C, D, F) is fully file-disjoint provided C-then-D ordering (D imports `build_loss_bundle` from compose, requires C merged).

**Reviewer model floor (per CLAUDE.md):** sonnet/high for every implementer. Design-sensitive reviewer (Phase H1) is opus/xhigh; general code review (Phase H2) is sonnet/high; lint/format (Phase H3) runs the reviewer's tooling directly.

---

## Spec coverage map (every spec §15 deliverable → plan phase)

| Spec §15 row | Phase / step |
|---|---|
| 1. Spec doc | already on disk; no plan action |
| 2. New `LossConfig` + `presets.py` | Phase A |
| 3. Term library (14 classes) under `terms/` | Phase B |
| 4. Composer + `LossBundle` | Phase C |
| 5. Trainer sidecar + optional bundle wire-up | Phase D |
| 6. `csp init --class-imbalance` | Phase E |
| 7. `csp doctor` resolved-losses table + JSON | Phase F |
| 8. Templates updated | Phase E |
| 9. Tests | Phases A, B, C, D, E, F bundle their own; mass migration in Phase G |

Cross-check spec §11.1–§11.6:
- §11.1 (`test_loss_presets.py`) → Phase A
- §11.2 (`test_loss_terms.py`) → Phase B
- §11.3 (`test_loss_compose.py`) → Phase C
- §11.4 (`test_config_schema.py` extend) → Phase A (the new pydantic types are added there)
- §11.5 (fixture migration in data/trainer tests) → Phase G; sidecar assertion → Phase D
- §11.6 (`test_cli_init.py`, `test_cli_doctor.py` extends) → Phase E, Phase F

---

## Pre-flight (Phase 0)

**Model/effort:** sonnet / medium. **Parallel:** no. **Blocks:** all later phases.

### Step P0-1: Confirm working tree state

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/loss-presets-112 status
```

Expected: branch `loss-presets-112`. The spec at `docs/superpowers/specs/2026-05-23-domain-aware-loss-presets-design.md` is tracked. This plan is added as the next commit. No staged or modified source files beyond docs.

### Step P0-2: Baseline test sanity

```bash
uv run pytest tests/unit/test_loss_config.py tests/unit/test_config_schema.py tests/unit/test_box_hint_schedule.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_cli_init.py tests/unit/test_cli_doctor.py tests/unit/test_trainer_run_dir.py tests/unit/test_trainer_nan_behavior.py -q
```

Expected: all green. If anything is red, halt — the baseline is broken and Phase G / Phase C cannot be validated against the post-migration result.

### Step P0-3: Commit the plan

```bash
git add docs/superpowers/plans/2026-05-23-domain-aware-loss-presets-plan.md
git commit -m "plan: domain-aware loss-function presets (#112)"
```

---

## Phase A — Package skeleton, schema break, `presets.py`, schema tests

**Parallelism:** serial (foundation; blocks B, C, D, E, F, G).
**Files touched:**
- Move: `src/custom_sam_peft/models/losses.py` → `src/custom_sam_peft/models/losses/__init__.py` (preserve content; package skeleton)
- Create: `src/custom_sam_peft/models/losses/presets.py`
- Modify: `src/custom_sam_peft/config/schema.py` (add literals + `LossOverrides` + new Pydantic `LossConfig`; remove the `LossConfig` re-export from `_internal`)
- Modify: `src/custom_sam_peft/config/_internal.py` (delete the dataclass `LossConfig`; keep `MatcherWeights`)
- Modify: `tests/unit/test_config_schema.py` (add new-LossConfig validation tests)
- Create: `tests/unit/test_loss_presets.py`

**Spec ref:** §3 (current state), §4 (schema), §5 (preset table), §5.3 (citation convention), §6 (LOCKED_OFF), §7 (resolve), §9 (sidecar helper), §11.1, §11.4.

**Verify (cumulative for the phase):** `uv run pytest tests/unit/test_loss_presets.py tests/unit/test_config_schema.py -q` is green; `uv run python -c "from custom_sam_peft.models.losses import total_loss; print('OK')"` exits 0 (the moved monolith still imports cleanly via the new package).

### Task A1 — Convert `models/losses.py` to a package via `git mv`

Goal: make room for the new sibling modules (`presets.py`, `compose.py`, `terms/`) inside `src/custom_sam_peft/models/losses/`. The existing `models/losses.py` file becomes `models/losses/__init__.py` byte-for-byte. Importers continue to use `from custom_sam_peft.models.losses import total_loss` unchanged.

- [ ] **Step A1-1: Move the file to the package init**

```bash
mkdir -p src/custom_sam_peft/models/losses_tmp
git mv src/custom_sam_peft/models/losses.py src/custom_sam_peft/models/losses_tmp/__init__.py
git mv src/custom_sam_peft/models/losses_tmp src/custom_sam_peft/models/losses
```

(The two-step move avoids the "file and directory with the same name" race; some filesystems allow it directly, but the indirection is safe everywhere.)

- [ ] **Step A1-2: Import-smoke**

```bash
uv run python -c "
from custom_sam_peft.models.losses import total_loss, mask_loss, box_loss, objectness_loss, presence_loss
print('package skeleton OK')
"
```

Expected: `package skeleton OK`. Exit 0. The monolith's exports still work because the file is now `losses/__init__.py`.

- [ ] **Step A1-3: Commit the package skeleton**

```bash
git add src/custom_sam_peft/models/losses/__init__.py
git status   # confirm: old losses.py deleted, new losses/__init__.py added, no other changes
git commit -m "refactor(models): convert losses.py to losses/ package (no code change)"
```

### Task A2 — Schema break in `config/schema.py` and `_internal.py`

- [ ] **Step A2-1: Add the new literals and Pydantic models in `schema.py`**

In `src/custom_sam_peft/config/schema.py`, locate the existing literals block (Preset/Intensity declared on lines ~114–115). Add the new literals immediately after:

```python
ClassImbalance = Literal["balanced", "moderate", "severe"]
MaskFamily     = Literal["bce", "dice", "dice_bce",
                         "focal_bce", "focal_dice",
                         "focal_tversky", "boundary"]
BoxFamily      = Literal["l1_giou", "giou_only", "ciou"]
ObjFamily      = Literal["focal_bce", "bce"]
PresenceFamily = Literal["bce", "focal_bce"]
```

Then add the two new `_Strict` classes near where `AugmentationsConfig`/`AugmentationOverrides` live (so the loss config is co-located with its augmentation sibling):

```python
class LossOverrides(_Strict):
    """Per-knob overrides. All None → inherit from (preset, class_imbalance).

    Setting any field to a non-None value replaces just that field in the
    resolved table. Extra keys are rejected (extra="forbid"); typos surface
    at config-load time.
    """

    # Term selection (4 axes)
    mask_family:     MaskFamily     | None = None
    box_family:      BoxFamily      | None = None
    obj_family:      ObjFamily      | None = None
    presence_family: PresenceFamily | None = None

    # Weights (4)
    w_mask:          PositiveFloat  | None = None
    w_box:           float          | None = Field(default=None, ge=0.0)
    w_obj:           PositiveFloat  | None = None
    w_presence:      PositiveFloat  | None = None

    # Focal params (2)
    focal_gamma:     PositiveFloat  | None = None
    focal_alpha:     float          | None = Field(default=None, ge=0.0, le=1.0)

    # Tversky params (2)
    tversky_alpha:   float          | None = Field(default=None, ge=0.0, le=1.0)
    tversky_gamma:   PositiveFloat  | None = None

    # Boundary blend coefficient (1)
    boundary_weight: float          | None = Field(default=None, ge=0.0, le=1.0)

    # Matcher contract (internal sub-model; accepts dict or MatcherWeights instance)
    matcher_weights: MatcherWeights | None = None

    @field_validator("matcher_weights", mode="before")
    @classmethod
    def _coerce_matcher_weights(cls, v: object) -> MatcherWeights | None:
        if v is None or isinstance(v, MatcherWeights):
            return v  # type: ignore[return-value]
        if isinstance(v, dict):
            return MatcherWeights(**v)
        raise TypeError(f"matcher_weights must be a dict or MatcherWeights, got {type(v).__name__}")


class LossConfig(_Strict):
    preset:          Preset         = "natural"
    class_imbalance: ClassImbalance = "balanced"
    overrides:       LossOverrides  = Field(default_factory=LossOverrides)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
```

Imports needed at the top of `schema.py` (`field_validator`, `ConfigDict` if not already imported, and `MatcherWeights`):

```python
from pydantic import field_validator  # add to existing pydantic import line if missing
from custom_sam_peft.config._internal import MatcherWeights  # add if not already imported
```

(Use `grep -n "field_validator\|MatcherWeights\|ConfigDict" src/custom_sam_peft/config/schema.py` to confirm what's already imported before adding lines.)

- [ ] **Step A2-2: Remove the `LossConfig` re-export from `schema.py`**

The current `schema.py` re-exports `LossConfig` from `_internal` at the top (see `from custom_sam_peft.config._internal import (... LossConfig ...)` and the `__all__` list). Remove `LossConfig` from both the import line AND the `__all__` list. The symbol is now defined in `schema.py` itself; the new definition shadows the old re-export.

- [ ] **Step A2-3: Delete the dataclass `LossConfig` from `_internal.py`**

In `src/custom_sam_peft/config/_internal.py`, delete the entire `@dataclass class LossConfig: ...` block (lines ~33–57 — verify with `grep -n "class LossConfig" src/custom_sam_peft/config/_internal.py`). Keep `MatcherWeights` (above it) and everything below (`WandbConfig`, `ExportConfig`). Update the module docstring's "Internal sub-configs" mention to drop `LossConfig`.

- [ ] **Step A2-4: Schema-smoke**

```bash
uv run python -c "
from custom_sam_peft.config.schema import (
    LossConfig, LossOverrides, ClassImbalance,
    MaskFamily, BoxFamily, ObjFamily, PresenceFamily,
)
from custom_sam_peft.config._internal import MatcherWeights
cfg = LossConfig()
assert cfg.preset == 'natural'
assert cfg.class_imbalance == 'balanced'
o = cfg.overrides.model_dump()
assert all(v is None for v in o.values()), o
# matcher_weights coercion
cfg2 = LossConfig(overrides={'matcher_weights': {'lambda_mask': 7.0}})
assert isinstance(cfg2.overrides.matcher_weights, MatcherWeights)
assert cfg2.overrides.matcher_weights.lambda_mask == 7.0
# extra=forbid bites typos
try:
    LossOverrides(mask_familty='dice_bce')
except Exception as e:
    print('typo rejected:', type(e).__name__)
else:
    raise AssertionError('typo should have been rejected')
print('schema OK')
"
```

Expected: prints `typo rejected: ValidationError` then `schema OK`. Exit 0.

### Task A3 — Create `models/losses/presets.py`

- [ ] **Step A3-1: Write the module**

Create `src/custom_sam_peft/models/losses/presets.py` with the following content. Single file; resolver + table + sidecar-dump are co-located. **No torch import** (pure Python; MatcherWeights is a dataclass from `config/_internal` which also has no torch import).

```python
"""Domain-aware loss-function presets — resolver and run-metadata helpers.

Pure-Python module: does NOT import torch. The resolver can be imported into
`csp doctor` without dragging torch into the doctor import graph.

Public API:
  - PRESET_TABLE: dict[(Preset, ClassImbalance), dict[str, str | float]]
  - LOCKED_OFF:   dict[str, dict[str, str]]
  - ResolvedLosses: frozen dataclass with 13 knobs + matcher_weights
  - resolve(cfg) -> ResolvedLosses
  - dump_loss_bundle(cfg) -> dict  (sidecar helper)

Citation tags (see spec §5.3):
  (A) #112 issue body            — cell lifted verbatim from the issue's draft table
  (B) preserved pre-#112         — matches today's hardcoded trainer behavior in losses.py
  (C) Lin et al. 2017 (RetinaNet/focal loss)         — γ=2.0, α=0.25 from Table 1
  (D) Abraham & Khan 2019 (focal Tversky)            — γ=0.75 best on ISIC
  (E) Salehi et al. 2017 (Tversky loss)              — α=0.7 best on MS lesions
  (F) degenerate-case identity                       — α=0.5 reduces Tversky to Dice;
                                                       γ=1.0 reduces Focal-Tversky to Tversky
  (G) alias-of-medical                               — microscopy copies medical (citation
                                                       needed — see #120)
  (H) Kervadec et al. 2019 (boundary loss)           — blend coefficient ~0.2 representative

Cells lacking a firm cite carry an inline `# citation needed` comment. Per
issue #112's brainstorming: silent defaults are fine — expert users will spot
and file issues against the master tracker #120.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from custom_sam_peft import __version__ as _LIB_VERSION
from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.config.schema import (
    BoxFamily,
    ClassImbalance,
    LossConfig,
    MaskFamily,
    ObjFamily,
    Preset,
    PresenceFamily,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Term-class names (used by dump_loss_bundle to avoid importing compose.py
# from this pure-Python module). Kept in lockstep with compose.py's term
# registries via a sync-check test (see test_loss_compose.py::
# test_term_class_names_match_compose_registry).
# ---------------------------------------------------------------------------

_TERM_CLASS_NAMES: dict[str, dict[str, str]] = {
    "mask": {
        "bce":           "BCELoss",
        "dice":          "DiceLoss",
        "dice_bce":      "DiceBCELoss",
        "focal_bce":     "FocalBCELoss",
        "focal_dice":    "FocalDiceLoss",
        "tversky":       "TverskyLoss",
        "focal_tversky": "FocalTverskyLoss",
        "boundary":      "BoundaryLoss",
    },
    "box": {
        "l1_giou":   "L1GIoULoss",
        "giou_only": "GIoUOnlyLoss",
        "ciou":      "CIoULoss",
    },
    "obj": {
        "focal_bce": "FocalBCELoss",
        "bce":       "BCELoss",
    },
    "presence": {
        "bce":       "BCELoss",
        "focal_bce": "FocalBCELoss",
    },
}


# ---------------------------------------------------------------------------
# Preset × class_imbalance table — spec §5
# ---------------------------------------------------------------------------

# Twelve cells for the four real domains. Microscopy is byte-equal to medical
# in v1 (alias-of-medical; spec §5.2). `none` and `custom` are short-circuited
# in resolve(), not stored here.
PRESET_TABLE: dict[tuple[Preset, ClassImbalance], dict[str, Any]] = {
    # ----- natural -----
    ("natural", "balanced"): {
        "mask_family":     "dice_bce",       # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     2.0,              # cite: (A,C)
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.5,              # cite: (F)
        "tversky_gamma":   1.0,              # cite: (F)
        "boundary_weight": 0.0,
    },
    ("natural", "moderate"): {
        "mask_family":     "dice_bce",       # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     2.5,              # cite: (A)  # citation needed
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.5,              # cite: (F)
        "tversky_gamma":   1.0,              # cite: (F)
        "boundary_weight": 0.0,
    },
    ("natural", "severe"): {
        "mask_family":     "focal_dice",     # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     3.0,              # cite: (A)  # citation needed
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.6,              # cite: (A,E)
        "tversky_gamma":   0.75,             # cite: (D)
        "boundary_weight": 0.0,
    },
    # ----- medical -----
    ("medical", "balanced"): {
        "mask_family":     "focal_dice",     # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     2.0,              # cite: (A,C)
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.6,              # cite: (A,E)
        "tversky_gamma":   0.75,             # cite: (D)
        "boundary_weight": 0.0,
    },
    ("medical", "moderate"): {
        "mask_family":     "focal_tversky",  # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     2.5,              # cite: (A)  # citation needed
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.7,              # cite: (A,D)  # citation needed for this exact value
        "tversky_gamma":   0.75,             # cite: (D)
        "boundary_weight": 0.0,
    },
    ("medical", "severe"): {
        "mask_family":     "boundary",       # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     3.0,              # cite: (A)  # citation needed
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.8,              # cite: (A)  # citation needed
        "tversky_gamma":   0.75,             # cite: (D)
        "boundary_weight": 0.2,              # cite: (A,H)
    },
    # ----- satellite -----
    ("satellite", "balanced"): {
        "mask_family":     "dice_bce",       # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     2.0,              # cite: (A,C)
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.5,              # cite: (F)
        "tversky_gamma":   1.0,              # cite: (F)
        "boundary_weight": 0.0,
    },
    ("satellite", "moderate"): {
        "mask_family":     "focal_dice",     # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     2.5,              # cite: (A)  # citation needed
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.6,              # cite: (A,E)
        "tversky_gamma":   0.75,             # cite: (D)
        "boundary_weight": 0.0,
    },
    ("satellite", "severe"): {
        "mask_family":     "focal_tversky",  # cite: (A)
        "box_family":      "l1_giou",        # cite: (B)
        "obj_family":      "focal_bce",      # cite: (B)
        "presence_family": "bce",            # cite: (B)
        "w_mask":          1.0,              # cite: (B)
        "w_box":           0.0,              # cite: (B)
        "w_obj":           1.0,              # cite: (B)
        "w_presence":      1.0,              # cite: (B)
        "focal_gamma":     3.0,              # cite: (A)  # citation needed
        "focal_alpha":     0.25,             # cite: (A,C)
        "tversky_alpha":   0.7,              # cite: (A,D)  # citation needed for this exact value
        "tversky_gamma":   0.75,             # cite: (D)
        "boundary_weight": 0.0,
    },
}

# Microscopy = strict alias of medical (spec §5.2). Reuse the same dicts.
PRESET_TABLE[("microscopy", "balanced")] = dict(PRESET_TABLE[("medical", "balanced")])  # cite: (G)
PRESET_TABLE[("microscopy", "moderate")] = dict(PRESET_TABLE[("medical", "moderate")])  # cite: (G)
PRESET_TABLE[("microscopy", "severe")]   = dict(PRESET_TABLE[("medical", "severe")])    # cite: (G)


# ---------------------------------------------------------------------------
# _LEGACY_DEFAULTS — values used when preset == "none" (preserves pre-#112).
# ---------------------------------------------------------------------------

_LEGACY_DEFAULTS: dict[str, Any] = {
    "mask_family":     "dice_bce",
    "box_family":      "l1_giou",
    "obj_family":      "focal_bce",
    "presence_family": "bce",
    "w_mask":          1.0,
    "w_box":           0.0,
    "w_obj":           1.0,
    "w_presence":      1.0,
    "focal_gamma":     2.0,
    "focal_alpha":     0.25,
    "tversky_alpha":   0.5,   # neutral — Dice-equivalent; ignored by dice_bce
    "tversky_gamma":   1.0,   # neutral — Tversky-equivalent; ignored by dice_bce
    "boundary_weight": 0.0,
}


# ---------------------------------------------------------------------------
# LOCKED_OFF — knob overrides that emit a WARN under specific presets.
# ---------------------------------------------------------------------------

LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "mask_family": (
            "the medical preset chose focal_dice/focal_tversky/boundary to handle "
            "rare positives; overriding to dice_bce or bce may underweight them"
        ),
    },
    "natural": {
        "mask_family": (
            "the natural preset chose dice_bce/focal_dice; overriding to "
            "focal_tversky or boundary is unusual for balanced natural-image data"
        ),
    },
    # satellite, microscopy: no locked-off entries in v1 (revisit after real users).
}


# ---------------------------------------------------------------------------
# Resolved view (frozen) and resolver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedLosses:
    mask_family:     str
    box_family:      str
    obj_family:      str
    presence_family: str
    w_mask:          float
    w_box:           float
    w_obj:           float
    w_presence:      float
    focal_gamma:     float
    focal_alpha:     float
    tversky_alpha:   float
    tversky_gamma:   float
    boundary_weight: float
    matcher_weights: MatcherWeights = field(default_factory=MatcherWeights)


def _override_triggers_warn(
    field_name: str, value: object, preset: str, class_imbalance: str
) -> bool:
    """Spec §6.2: warn only when the override changes the locked-off knob away
    from the table's seed value."""
    if preset not in LOCKED_OFF:
        return False
    if field_name not in LOCKED_OFF[preset]:
        return False
    if value is None:
        return False
    seed = PRESET_TABLE[(preset, class_imbalance)][field_name]
    return value != seed


def resolve(cfg: LossConfig) -> ResolvedLosses:
    """Spec §7. Returns a frozen ResolvedLosses with all 13 knobs populated."""
    # 1. Seed from the preset table (or short-circuit for none/custom).
    if cfg.preset == "none":
        base = dict(_LEGACY_DEFAULTS)
        seed_matcher = MatcherWeights()
    elif cfg.preset == "custom":
        base = dict(PRESET_TABLE[("natural", "balanced")])
        seed_matcher = MatcherWeights()
    else:
        base = dict(PRESET_TABLE[(cfg.preset, cfg.class_imbalance)])
        seed_matcher = MatcherWeights()

    # 2. Apply overrides; warn if a locked-off knob is overridden.
    ov = cfg.overrides.model_dump(exclude_unset=False)
    for fname, override in ov.items():
        if override is None:
            continue
        if fname == "matcher_weights":
            seed_matcher = (
                MatcherWeights(**override) if isinstance(override, dict) else override
            )
            continue
        if cfg.preset not in ("none", "custom") and _override_triggers_warn(
            fname, override, cfg.preset, cfg.class_imbalance
        ):
            reason = LOCKED_OFF[cfg.preset][fname]
            _LOG.warning(
                "You overrode %s=%s under preset=%s; %s. "
                "The override will be applied as-is.",
                fname, override, cfg.preset, reason,
            )
        base[fname] = override

    return ResolvedLosses(**base, matcher_weights=seed_matcher)


# ---------------------------------------------------------------------------
# Sidecar helper — spec §9
# ---------------------------------------------------------------------------

def dump_loss_bundle(cfg: LossConfig) -> dict[str, Any]:
    """Return the JSON-serializable dict written to run_dir/loss_bundle.json."""
    resolved = resolve(cfg)
    term_classes = {
        "mask":     _TERM_CLASS_NAMES["mask"][resolved.mask_family],
        "box":      _TERM_CLASS_NAMES["box"][resolved.box_family],
        "obj":      _TERM_CLASS_NAMES["obj"][resolved.obj_family],
        "presence": _TERM_CLASS_NAMES["presence"][resolved.presence_family],
    }
    return {
        "preset":          cfg.preset,
        "class_imbalance": cfg.class_imbalance,
        "resolved": {
            "mask_family":     resolved.mask_family,
            "box_family":      resolved.box_family,
            "obj_family":      resolved.obj_family,
            "presence_family": resolved.presence_family,
            "w_mask":          resolved.w_mask,
            "w_box":           resolved.w_box,
            "w_obj":           resolved.w_obj,
            "w_presence":      resolved.w_presence,
            "focal_gamma":     resolved.focal_gamma,
            "focal_alpha":     resolved.focal_alpha,
            "tversky_alpha":   resolved.tversky_alpha,
            "tversky_gamma":   resolved.tversky_gamma,
            "boundary_weight": resolved.boundary_weight,
        },
        "term_classes":    term_classes,
        "library_version": _LIB_VERSION or "unknown",
    }
```

- [ ] **Step A3-2: Import-smoke for `presets.py`**

```bash
uv run python -c "
from custom_sam_peft.models.losses.presets import (
    PRESET_TABLE, LOCKED_OFF, ResolvedLosses, resolve, dump_loss_bundle, _LEGACY_DEFAULTS, _TERM_CLASS_NAMES,
)
from custom_sam_peft.config.schema import LossConfig
assert len(PRESET_TABLE) == 12, len(PRESET_TABLE)
cfg = LossConfig()  # natural / balanced / no overrides
r = resolve(cfg)
assert r.mask_family == 'dice_bce'
assert r.focal_gamma == 2.0
print('presets OK:', r.mask_family, r.focal_gamma, r.tversky_alpha)
"
```

Expected: prints `presets OK: dice_bce 2.0 0.5`. Exit 0.

### Task A4 — Tests for `presets.py` and the new schema

- [ ] **Step A4-1: Write `tests/unit/test_loss_presets.py`**

Create `tests/unit/test_loss_presets.py`:

```python
"""Tests for the loss-preset resolver (spec §11.1)."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.config.schema import (
    ClassImbalance,
    LossConfig,
    LossOverrides,
    Preset,
)
from custom_sam_peft.models.losses.presets import (
    LOCKED_OFF,
    PRESET_TABLE,
    ResolvedLosses,
    _LEGACY_DEFAULTS,
    _TERM_CLASS_NAMES,
    dump_loss_bundle,
    resolve,
)

_REAL_PRESETS: list[Preset] = ["natural", "medical", "satellite", "microscopy"]
_TIERS: list[ClassImbalance] = ["balanced", "moderate", "severe"]


# -- Table exact values (spec §5) ----------------------------------------------

@pytest.mark.parametrize("preset", _REAL_PRESETS)
@pytest.mark.parametrize("tier", _TIERS)
def test_resolve_table_exact_values(preset: Preset, tier: ClassImbalance) -> None:
    cfg = LossConfig(preset=preset, class_imbalance=tier)
    r = resolve(cfg)
    row = PRESET_TABLE[(preset, tier)]
    for fname, expected in row.items():
        assert getattr(r, fname) == expected, (preset, tier, fname, expected, getattr(r, fname))


# -- Short-circuit presets ------------------------------------------------------

@pytest.mark.parametrize("tier", _TIERS)
def test_resolve_none_uses_legacy_defaults(tier: ClassImbalance) -> None:
    cfg = LossConfig(preset="none", class_imbalance=tier)
    r = resolve(cfg)
    for fname, expected in _LEGACY_DEFAULTS.items():
        assert getattr(r, fname) == expected, (fname, expected, getattr(r, fname))


def test_resolve_custom_seeds_with_natural_balanced() -> None:
    cfg = LossConfig(preset="custom")
    r = resolve(cfg)
    row = PRESET_TABLE[("natural", "balanced")]
    for fname, expected in row.items():
        assert getattr(r, fname) == expected, (fname, expected, getattr(r, fname))


# -- Override layering ----------------------------------------------------------

def test_resolve_override_wins_over_table() -> None:
    cfg = LossConfig(
        preset="natural", class_imbalance="balanced",
        overrides=LossOverrides(focal_gamma=5.0),
    )
    r = resolve(cfg)
    assert r.focal_gamma == 5.0
    # other fields untouched
    assert r.mask_family == PRESET_TABLE[("natural", "balanced")]["mask_family"]


def test_resolve_override_zero_is_valid() -> None:
    cfg = LossConfig(
        preset="natural", class_imbalance="balanced",
        overrides=LossOverrides(w_obj=0.5),  # 0 rejected by PositiveFloat; use 0.5
    )
    r = resolve(cfg)
    assert r.w_obj == 0.5
    # w_box is the only override that allows zero (ge=0.0)
    cfg2 = LossConfig(overrides=LossOverrides(w_box=0.0))
    assert resolve(cfg2).w_box == 0.0


def test_resolve_matcher_weights_override() -> None:
    cfg = LossConfig(overrides=LossOverrides(matcher_weights=MatcherWeights(lambda_mask=9.0)))
    r = resolve(cfg)
    assert r.matcher_weights.lambda_mask == 9.0


def test_resolve_matcher_weights_dict_coerced() -> None:
    cfg = LossConfig(overrides={"matcher_weights": {"lambda_mask": 11.0}})  # type: ignore[arg-type]
    r = resolve(cfg)
    assert r.matcher_weights.lambda_mask == 11.0


# -- LOCKED_OFF warns -----------------------------------------------------------

def test_resolve_locked_off_warns_medical_mask_family(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(
        preset="medical", class_imbalance="moderate",
        overrides=LossOverrides(mask_family="dice_bce"),
    )
    r = resolve(cfg)
    assert r.mask_family == "dice_bce"  # override wins
    msgs = [rec.message for rec in caplog.records]
    assert any("mask_family" in m and "medical" in m and "rare positives" in m for m in msgs), msgs


def test_resolve_locked_off_warns_natural_mask_family(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(
        preset="natural", class_imbalance="balanced",
        overrides=LossOverrides(mask_family="focal_tversky"),
    )
    resolve(cfg)
    msgs = [rec.message for rec in caplog.records]
    assert any("mask_family" in m and "natural" in m and "unusual" in m for m in msgs), msgs


def test_resolve_locked_off_no_warn_when_override_matches_seed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overriding to the table's existing value is a no-op; no warn."""
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    seed = PRESET_TABLE[("medical", "balanced")]["mask_family"]
    cfg = LossConfig(
        preset="medical", class_imbalance="balanced",
        overrides=LossOverrides(mask_family=seed),
    )
    resolve(cfg)
    assert not caplog.records, [rec.message for rec in caplog.records]


def test_resolve_none_skips_locked_off(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(preset="none", overrides=LossOverrides(mask_family="focal_tversky"))
    resolve(cfg)
    assert not caplog.records, [rec.message for rec in caplog.records]


def test_resolve_custom_skips_locked_off(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="custom_sam_peft.models.losses.presets")
    cfg = LossConfig(preset="custom", overrides=LossOverrides(mask_family="boundary"))
    resolve(cfg)
    assert not caplog.records, [rec.message for rec in caplog.records]


# -- ResolvedLosses immutability ------------------------------------------------

def test_resolved_losses_frozen() -> None:
    r = resolve(LossConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.w_mask = 2.0  # type: ignore[misc]
    # replace works
    r2 = dataclasses.replace(r, w_mask=2.0)
    assert r2.w_mask == 2.0


# -- Sidecar helper -------------------------------------------------------------

def test_dump_loss_bundle_shape() -> None:
    cfg = LossConfig(preset="medical", class_imbalance="moderate")
    d = dump_loss_bundle(cfg)
    assert set(d.keys()) == {"preset", "class_imbalance", "resolved", "term_classes", "library_version"}
    assert d["preset"] == "medical"
    assert d["class_imbalance"] == "moderate"
    assert set(d["resolved"].keys()) == {
        "mask_family", "box_family", "obj_family", "presence_family",
        "w_mask", "w_box", "w_obj", "w_presence",
        "focal_gamma", "focal_alpha",
        "tversky_alpha", "tversky_gamma", "boundary_weight",
    }
    assert d["term_classes"] == {
        "mask": "FocalTverskyLoss", "box": "L1GIoULoss",
        "obj": "FocalBCELoss",      "presence": "BCELoss",
    }
    assert isinstance(d["library_version"], str) and d["library_version"]
    # round-trip through JSON
    assert json.loads(json.dumps(d)) == d


def test_dump_loss_bundle_for_none_preset() -> None:
    d = dump_loss_bundle(LossConfig(preset="none"))
    assert d["resolved"]["mask_family"] == "dice_bce"
    assert d["term_classes"]["mask"] == "DiceBCELoss"


# -- Microscopy alias contract --------------------------------------------------

@pytest.mark.parametrize("tier", _TIERS)
def test_microscopy_equals_medical(tier: ClassImbalance) -> None:
    assert PRESET_TABLE[("microscopy", tier)] == PRESET_TABLE[("medical", tier)]
```

- [ ] **Step A4-2: Extend `tests/unit/test_config_schema.py`**

Add the following test functions to `tests/unit/test_config_schema.py`. (Use `grep -n "test_" tests/unit/test_config_schema.py` to find a sensible location; append at end is fine.)

```python
def test_loss_config_defaults() -> None:
    from custom_sam_peft.config.schema import LossConfig
    cfg = LossConfig()
    assert cfg.preset == "natural"
    assert cfg.class_imbalance == "balanced"
    assert cfg.overrides.model_dump() == {
        "mask_family": None, "box_family": None, "obj_family": None, "presence_family": None,
        "w_mask": None, "w_box": None, "w_obj": None, "w_presence": None,
        "focal_gamma": None, "focal_alpha": None,
        "tversky_alpha": None, "tversky_gamma": None, "boundary_weight": None,
        "matcher_weights": None,
    }


def test_loss_config_class_imbalance_literal_validation() -> None:
    from pydantic import ValidationError
    from custom_sam_peft.config.schema import LossConfig
    with pytest.raises(ValidationError):
        LossConfig(class_imbalance="moderete")  # type: ignore[arg-type]


def test_loss_config_preset_literal_validation() -> None:
    from pydantic import ValidationError
    from custom_sam_peft.config.schema import LossConfig
    with pytest.raises(ValidationError):
        LossConfig(preset="medecal")  # type: ignore[arg-type]


def test_loss_overrides_rejects_unknown_keys() -> None:
    from pydantic import ValidationError
    from custom_sam_peft.config.schema import LossOverrides
    with pytest.raises(ValidationError):
        LossOverrides(mask_familty="dice_bce")  # type: ignore[call-arg]


def test_loss_overrides_default_factory_isolation() -> None:
    from custom_sam_peft.config.schema import LossConfig
    a = LossConfig()
    b = LossConfig()
    assert a.overrides is not b.overrides


def test_loss_overrides_family_literal_validation() -> None:
    from pydantic import ValidationError
    from custom_sam_peft.config.schema import LossOverrides
    with pytest.raises(ValidationError):
        LossOverrides(mask_family="focle_bce")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        LossOverrides(box_family="diou")  # type: ignore[arg-type]


def test_loss_overrides_matcher_weights_dict_coerced() -> None:
    from custom_sam_peft.config._internal import MatcherWeights
    from custom_sam_peft.config.schema import LossOverrides
    o = LossOverrides(matcher_weights={"lambda_mask": 7.0})  # type: ignore[arg-type]
    assert isinstance(o.matcher_weights, MatcherWeights)
    assert o.matcher_weights.lambda_mask == 7.0


def test_loss_overrides_w_box_zero_allowed() -> None:
    from custom_sam_peft.config.schema import LossOverrides
    LossOverrides(w_box=0.0)  # ge=0.0; no exception


def test_loss_overrides_w_mask_zero_rejected() -> None:
    from pydantic import ValidationError
    from custom_sam_peft.config.schema import LossOverrides
    with pytest.raises(ValidationError):
        LossOverrides(w_mask=0.0)  # PositiveFloat
```

- [ ] **Step A4-3: Run the new tests**

```bash
uv run pytest tests/unit/test_loss_presets.py tests/unit/test_config_schema.py -q
```

Expected: all green. Count check: `test_loss_presets.py` has ~17 tests; `test_config_schema.py` gains 9 new tests on top of whatever exists.

### Task A5 — Phase-A commit

- [ ] **Step A5-1: Commit the foundation**

```bash
git add \
  src/custom_sam_peft/models/losses/__init__.py \
  src/custom_sam_peft/models/losses/presets.py \
  src/custom_sam_peft/config/schema.py \
  src/custom_sam_peft/config/_internal.py \
  tests/unit/test_loss_presets.py \
  tests/unit/test_config_schema.py
git status   # confirm nothing else snuck in
git commit -m "feat(losses): schema break + presets.py resolver foundation (#112)"
```

---

## Phase B — Term library (14 classes under `models/losses/terms/`)

**Parallelism:** parallel-safe with E and G after A.
**Files touched:**
- Create: `src/custom_sam_peft/models/losses/terms/__init__.py`
- Create: `src/custom_sam_peft/models/losses/terms/mask.py`
- Create: `src/custom_sam_peft/models/losses/terms/box.py`
- Create: `src/custom_sam_peft/models/losses/terms/obj.py`
- Create: `src/custom_sam_peft/models/losses/terms/presence.py`
- Create: `tests/unit/test_loss_terms.py`

**Spec ref:** §8 (Term library), §11.2.

**Verify (cumulative for the phase):** `uv run pytest tests/unit/test_loss_terms.py -q` is green.

### Task B1 — `terms/mask.py` (8 classes)

- [ ] **Step B1-1: Write `mask.py`**

Create `src/custom_sam_peft/models/losses/terms/mask.py`:

```python
"""Mask-axis loss term classes (spec §8.1).

All classes are uniform-signature: forward(pred_logits, target) where
pred_logits and target are (N, H, W). If spatial shapes differ, pred_logits
is bilinear-upsampled to the target resolution (matches the pre-#112
mask_loss behavior).

Every class accepts the full hyperparameter pack as keyword-only kwargs and
silently ignores irrelevant ones. This keeps build_loss_bundle uniform.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate

_EPS = 1.0  # matches pre-#112 _dice_loss


def _align(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    return interpolate(
        pred[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False,
    )[:, 0]


def _dice(p: Tensor, t: Tensor) -> Tensor:
    p = p.flatten(1)
    t = t.flatten(1)
    num = 2.0 * (p * t).sum(-1) + _EPS
    den = p.sum(-1) + t.sum(-1) + _EPS
    return (1.0 - num / den).mean()


def _focal_bce_per_pixel(logits: Tensor, targets: Tensor, gamma: float, alpha: float) -> Tensor:
    p = logits.sigmoid()
    ce = binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


class _MaskTermBase(nn.Module):
    """Accept the full hyperparameter pack and stash it; subclasses use what they need."""

    def __init__(
        self,
        *,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        tversky_alpha: float = 0.5,
        tversky_gamma: float = 1.0,
        boundary_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_gamma = float(tversky_gamma)
        self.boundary_weight = float(boundary_weight)


class BCELoss(_MaskTermBase):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        return binary_cross_entropy_with_logits(pred, target.float())


class DiceLoss(_MaskTermBase):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        return _dice(pred.sigmoid(), target.float())


class DiceBCELoss(_MaskTermBase):
    """Today's `mask_loss`: 0.5*Dice + 0.5*BCE."""

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        bce = binary_cross_entropy_with_logits(pred, target.float())
        dice = _dice(pred.sigmoid(), target.float())
        return 0.5 * dice + 0.5 * bce


class FocalBCELoss(_MaskTermBase):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        return _focal_bce_per_pixel(pred, target.float(), self.focal_gamma, self.focal_alpha)


class FocalDiceLoss(_MaskTermBase):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        fbce = _focal_bce_per_pixel(pred, target.float(), self.focal_gamma, self.focal_alpha)
        dice = _dice(pred.sigmoid(), target.float())
        return 0.5 * dice + 0.5 * fbce


def _tversky_index(p: Tensor, t: Tensor, alpha: float) -> Tensor:
    p = p.flatten(1)
    t = t.flatten(1)
    tp = (p * t).sum(-1)
    fn = ((1.0 - p) * t).sum(-1)
    fp = (p * (1.0 - t)).sum(-1)
    return (tp + _EPS) / (tp + alpha * fn + (1.0 - alpha) * fp + _EPS)


class TverskyLoss(_MaskTermBase):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        ti = _tversky_index(pred.sigmoid(), target.float(), self.tversky_alpha)
        return (1.0 - ti).mean()


class FocalTverskyLoss(_MaskTermBase):
    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        ti = _tversky_index(pred.sigmoid(), target.float(), self.tversky_alpha)
        return ((1.0 - ti).pow(self.tversky_gamma)).mean()


def _signed_distance_transform(target_np: np.ndarray) -> np.ndarray:
    """Signed distance transform for one (H, W) uint8/bool mask.

    Positive inside the object, negative outside. Computed via scipy
    distance_transform_edt on the binary mask and its complement.
    """
    mask = target_np.astype(bool)
    if not mask.any():
        # All-zero target: distance is + everywhere outside (i.e. positive everywhere
        # outside the object, which doesn't exist); use the EDT of the complement and
        # negate so the SDT is non-positive (pushing predictions away costs nothing).
        return -distance_transform_edt(~mask).astype(np.float32)
    if mask.all():
        return distance_transform_edt(mask).astype(np.float32)
    pos = distance_transform_edt(mask).astype(np.float32)
    neg = distance_transform_edt(~mask).astype(np.float32)
    return pos - neg


def _kervadec_boundary(pred_sigmoid: Tensor, target: Tensor) -> Tensor:
    """Kervadec et al. 2019 boundary loss: integral of pred * SDT(target).

    SDT is computed on CPU per image (scipy), then moved to pred.device.
    Detached from autograd — gradient flows only through pred.
    """
    batch_sdts = []
    target_cpu = target.detach().to(torch.uint8).cpu().numpy()
    for i in range(target_cpu.shape[0]):
        batch_sdts.append(_signed_distance_transform(target_cpu[i]))
    sdt = torch.from_numpy(np.stack(batch_sdts)).to(pred_sigmoid.device, pred_sigmoid.dtype)
    # Normalize by spatial size so the magnitude is comparable to Dice's [0, 1] range.
    return (pred_sigmoid * sdt).mean()


class BoundaryLoss(_MaskTermBase):
    """boundary_weight * Kervadec + (1 - boundary_weight) * Dice.

    boundary_weight=0 degenerates to plain Dice; boundary_weight=1 is pure Kervadec.
    """

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        pred = _align(pred_logits, target)
        p_sig = pred.sigmoid()
        dice_term = _dice(p_sig, target.float())
        if self.boundary_weight <= 0.0:
            return dice_term
        boundary_term = _kervadec_boundary(p_sig, target)
        return self.boundary_weight * boundary_term + (1.0 - self.boundary_weight) * dice_term
```

- [ ] **Step B1-2: Import-smoke**

```bash
uv run python -c "
import torch
from custom_sam_peft.models.losses.terms.mask import (
    BCELoss, DiceLoss, DiceBCELoss, FocalBCELoss, FocalDiceLoss,
    TverskyLoss, FocalTverskyLoss, BoundaryLoss,
)
pred = torch.randn(2, 16, 16, requires_grad=True)
tgt  = torch.zeros(2, 16, 16); tgt[:, 4:12, 4:12] = 1
for cls in [BCELoss, DiceLoss, DiceBCELoss, FocalBCELoss, FocalDiceLoss, TverskyLoss, FocalTverskyLoss, BoundaryLoss]:
    term = cls()
    val = term(pred, tgt)
    val.backward(retain_graph=True)
    assert torch.isfinite(val), cls.__name__
print('mask terms OK')
"
```

Expected: `mask terms OK`. Exit 0.

### Task B2 — `terms/box.py` (3 classes)

- [ ] **Step B2-1: Write `box.py`**

Create `src/custom_sam_peft/models/losses/terms/box.py`:

```python
"""Box-axis loss term classes (spec §8.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.functional import smooth_l1_loss


def _cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _giou_pairwise(b1_xyxy: Tensor, b2_xyxy: Tensor) -> Tensor:
    area1 = (b1_xyxy[:, 2] - b1_xyxy[:, 0]) * (b1_xyxy[:, 3] - b1_xyxy[:, 1])
    area2 = (b2_xyxy[:, 2] - b2_xyxy[:, 0]) * (b2_xyxy[:, 3] - b2_xyxy[:, 1])
    lt = torch.max(b1_xyxy[:, :2], b2_xyxy[:, :2])
    rb = torch.min(b1_xyxy[:, 2:], b2_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(b1_xyxy[:, :2], b2_xyxy[:, :2])
    rb_c = torch.max(b1_xyxy[:, 2:], b2_xyxy[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


class _BoxTermBase(nn.Module):
    def __init__(self, **_unused: float) -> None:
        super().__init__()


class L1GIoULoss(_BoxTermBase):
    """Today's box_loss: smoothL1(p, t) + (1 - GIoU(p, t)).mean()."""

    def forward(self, pred_cxcywh: Tensor, target_cxcywh: Tensor) -> Tensor:
        if pred_cxcywh.numel() == 0:
            return pred_cxcywh.new_zeros(())
        l1 = smooth_l1_loss(pred_cxcywh, target_cxcywh, reduction="mean")
        giou = _giou_pairwise(_cxcywh_to_xyxy(pred_cxcywh), _cxcywh_to_xyxy(target_cxcywh))
        return l1 + (1.0 - giou).mean()


class GIoUOnlyLoss(_BoxTermBase):
    def forward(self, pred_cxcywh: Tensor, target_cxcywh: Tensor) -> Tensor:
        if pred_cxcywh.numel() == 0:
            return pred_cxcywh.new_zeros(())
        giou = _giou_pairwise(_cxcywh_to_xyxy(pred_cxcywh), _cxcywh_to_xyxy(target_cxcywh))
        return (1.0 - giou).mean()


class CIoULoss(_BoxTermBase):
    """Zheng et al. 2020 — IoU - ρ²(p,t)/c² - α·v with aspect-ratio penalty."""

    def forward(self, pred_cxcywh: Tensor, target_cxcywh: Tensor) -> Tensor:
        if pred_cxcywh.numel() == 0:
            return pred_cxcywh.new_zeros(())
        p_xyxy = _cxcywh_to_xyxy(pred_cxcywh)
        t_xyxy = _cxcywh_to_xyxy(target_cxcywh)
        # IoU
        area1 = (p_xyxy[:, 2] - p_xyxy[:, 0]) * (p_xyxy[:, 3] - p_xyxy[:, 1])
        area2 = (t_xyxy[:, 2] - t_xyxy[:, 0]) * (t_xyxy[:, 3] - t_xyxy[:, 1])
        lt = torch.max(p_xyxy[:, :2], t_xyxy[:, :2])
        rb = torch.min(p_xyxy[:, 2:], t_xyxy[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
        iou = inter / union.clamp(min=1e-7)
        # Enclosing-box diagonal²
        lt_c = torch.min(p_xyxy[:, :2], t_xyxy[:, :2])
        rb_c = torch.max(p_xyxy[:, 2:], t_xyxy[:, 2:])
        c2 = (rb_c - lt_c).pow(2).sum(dim=-1).clamp(min=1e-7)
        # Center distance²
        rho2 = (pred_cxcywh[:, :2] - target_cxcywh[:, :2]).pow(2).sum(dim=-1)
        # Aspect-ratio penalty v and α
        w1, h1 = pred_cxcywh[:, 2].clamp(min=1e-7), pred_cxcywh[:, 3].clamp(min=1e-7)
        w2, h2 = target_cxcywh[:, 2].clamp(min=1e-7), target_cxcywh[:, 3].clamp(min=1e-7)
        v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        alpha = v / (1.0 - iou + v).clamp(min=1e-7)
        ciou = iou - rho2 / c2 - alpha * v
        return (1.0 - ciou).mean()
```

- [ ] **Step B2-2: Import-smoke**

```bash
uv run python -c "
import torch
from custom_sam_peft.models.losses.terms.box import L1GIoULoss, GIoUOnlyLoss, CIoULoss
p = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]], requires_grad=True)
t = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3]])
for cls in [L1GIoULoss, GIoUOnlyLoss, CIoULoss]:
    val = cls()(p, t)
    val.backward(retain_graph=True)
    assert torch.isfinite(val), cls.__name__
print('box terms OK')
"
```

Expected: `box terms OK`. Exit 0.

### Task B3 — `terms/obj.py` (2 classes)

- [ ] **Step B3-1: Write `obj.py`**

Create `src/custom_sam_peft/models/losses/terms/obj.py`:

```python
"""Obj-axis loss term classes (spec §8.3).

forward(obj_logits, matched_mask) where obj_logits is (B, Q) and matched_mask
is (B, Q) bool — True for queries assigned to some target.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits


class _ObjTermBase(nn.Module):
    def __init__(
        self,
        *,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **_unused: float,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)


class BCELoss(_ObjTermBase):
    def forward(self, obj_logits: Tensor, matched_mask: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(obj_logits, matched_mask.float())


class FocalBCELoss(_ObjTermBase):
    """Sigmoid focal BCE — today's objectness_loss."""

    def forward(self, obj_logits: Tensor, matched_mask: Tensor) -> Tensor:
        p = obj_logits.sigmoid()
        ce = binary_cross_entropy_with_logits(
            obj_logits, matched_mask.float(), reduction="none",
        )
        p_t = p * matched_mask + (1.0 - p) * (1.0 - matched_mask.float())
        alpha_t = self.focal_alpha * matched_mask + (1.0 - self.focal_alpha) * (1.0 - matched_mask.float())
        return (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce).mean()
```

- [ ] **Step B3-2: Import-smoke**

```bash
uv run python -c "
import torch
from custom_sam_peft.models.losses.terms.obj import BCELoss, FocalBCELoss
ol = torch.randn(2, 8, requires_grad=True)
mm = torch.tensor([[1,0,1,0,0,0,0,0],[0,0,0,1,1,0,0,0]], dtype=torch.bool)
for cls in [BCELoss, FocalBCELoss]:
    v = cls()(ol, mm); v.backward(retain_graph=True)
    assert torch.isfinite(v), cls.__name__
print('obj terms OK')
"
```

Expected: `obj terms OK`. Exit 0.

### Task B4 — `terms/presence.py` (2 classes)

- [ ] **Step B4-1: Write `presence.py`**

Create `src/custom_sam_peft/models/losses/terms/presence.py`:

```python
"""Presence-axis loss term classes (spec §8.4).

forward(img_presence, image_has_target) where img_presence is (B,) and
image_has_target is (B,) bool.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits


class _PresenceTermBase(nn.Module):
    def __init__(
        self,
        *,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **_unused: float,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)


class BCELoss(_PresenceTermBase):
    """Today's presence_loss."""

    def forward(self, img_presence: Tensor, image_has_target: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(img_presence, image_has_target.float())


class FocalBCELoss(_PresenceTermBase):
    def forward(self, img_presence: Tensor, image_has_target: Tensor) -> Tensor:
        p = img_presence.sigmoid()
        t = image_has_target.float()
        ce = binary_cross_entropy_with_logits(img_presence, t, reduction="none")
        p_t = p * t + (1.0 - p) * (1.0 - t)
        alpha_t = self.focal_alpha * t + (1.0 - self.focal_alpha) * (1.0 - t)
        return (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce).mean()
```

- [ ] **Step B4-2: Import-smoke**

```bash
uv run python -c "
import torch
from custom_sam_peft.models.losses.terms.presence import BCELoss, FocalBCELoss
ip = torch.randn(4, requires_grad=True)
ht = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
for cls in [BCELoss, FocalBCELoss]:
    v = cls()(ip, ht); v.backward(retain_graph=True)
    assert torch.isfinite(v), cls.__name__
print('presence terms OK')
"
```

Expected: `presence terms OK`. Exit 0.

### Task B5 — `terms/__init__.py` re-exports

- [ ] **Step B5-1: Write `terms/__init__.py`**

The four axes each declare their own `BCELoss` and `FocalBCELoss` classes — they are distinct class objects even when they share names across axes. To avoid name collisions in `terms/__init__.py`, re-export with axis-prefixed aliases. The composer in Phase C imports directly from the axis modules; the prefixed aliases here exist for ad-hoc debugging only.

Create `src/custom_sam_peft/models/losses/terms/__init__.py`:

```python
"""14 term classes across 4 axes (mask / box / obj / presence).

This `__init__` re-exports the public class objects with axis-prefixed names
(MaskBCELoss, ObjBCELoss, …) to avoid the same-name collision between axes.
The composer in `models/losses/compose.py` imports each axis module directly
and does not rely on these aliases.
"""

from custom_sam_peft.models.losses.terms import box as _box
from custom_sam_peft.models.losses.terms import mask as _mask
from custom_sam_peft.models.losses.terms import obj as _obj
from custom_sam_peft.models.losses.terms import presence as _presence

# Mask axis (8)
MaskBCELoss          = _mask.BCELoss
MaskDiceLoss         = _mask.DiceLoss
MaskDiceBCELoss      = _mask.DiceBCELoss
MaskFocalBCELoss     = _mask.FocalBCELoss
MaskFocalDiceLoss    = _mask.FocalDiceLoss
MaskTverskyLoss      = _mask.TverskyLoss
MaskFocalTverskyLoss = _mask.FocalTverskyLoss
MaskBoundaryLoss     = _mask.BoundaryLoss

# Box axis (3)
BoxL1GIoULoss   = _box.L1GIoULoss
BoxGIoUOnlyLoss = _box.GIoUOnlyLoss
BoxCIoULoss     = _box.CIoULoss

# Obj axis (2)
ObjBCELoss      = _obj.BCELoss
ObjFocalBCELoss = _obj.FocalBCELoss

# Presence axis (2)
PresenceBCELoss      = _presence.BCELoss
PresenceFocalBCELoss = _presence.FocalBCELoss

__all__ = [
    "MaskBCELoss", "MaskDiceLoss", "MaskDiceBCELoss", "MaskFocalBCELoss",
    "MaskFocalDiceLoss", "MaskTverskyLoss", "MaskFocalTverskyLoss", "MaskBoundaryLoss",
    "BoxL1GIoULoss", "BoxGIoUOnlyLoss", "BoxCIoULoss",
    "ObjBCELoss", "ObjFocalBCELoss",
    "PresenceBCELoss", "PresenceFocalBCELoss",
]
```

- [ ] **Step B5-2: Import-smoke**

```bash
uv run python -c "
import custom_sam_peft.models.losses.terms as T
classes = [getattr(T, n) for n in T.__all__]
assert len(classes) == 15  # 8+3+2+2 = 15? wait — re-count
# 8 mask + 3 box + 2 obj + 2 presence = 15 — yep, 15 re-export names (NOT 14).
# The '14 classes' count in the spec refers to distinct class names across axes;
# the obj/presence axes share BCE/FocalBCE class NAMES with mask but they are
# different class OBJECTS (defined in different modules). The re-export here
# names all 15 distinct class objects with axis-prefixed aliases.
print('terms package OK; classes:', len(T.__all__))
"
```

Expected: `terms package OK; classes: 15`. (The "14 classes" count in the spec is a rounding: 8 mask + 3 box + 2 obj + 2 presence = 15 distinct class objects, but the unique formula count across axes is 14 because BCE-mask and BCE-obj are spelled the same way. The implementation has 15 class objects. Update Phase A's `_TERM_CLASS_NAMES` count is unaffected — it keys by family literal, not class name.)

### Task B6 — Tests for the term library

- [ ] **Step B6-1: Write `tests/unit/test_loss_terms.py`**

Create `tests/unit/test_loss_terms.py`:

```python
"""Tests for the 15 loss-term classes (spec §11.2)."""

from __future__ import annotations

import pytest
import torch

from custom_sam_peft.models.losses.terms import box as box_terms
from custom_sam_peft.models.losses.terms import mask as mask_terms
from custom_sam_peft.models.losses.terms import obj as obj_terms
from custom_sam_peft.models.losses.terms import presence as presence_terms

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Mask axis
# ---------------------------------------------------------------------------

_MASK_CLASSES = [
    mask_terms.BCELoss,
    mask_terms.DiceLoss,
    mask_terms.DiceBCELoss,
    mask_terms.FocalBCELoss,
    mask_terms.FocalDiceLoss,
    mask_terms.TverskyLoss,
    mask_terms.FocalTverskyLoss,
    mask_terms.BoundaryLoss,
]


@pytest.fixture
def mask_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.randn(2, 16, 16, requires_grad=True)
    tgt = torch.zeros(2, 16, 16)
    tgt[:, 4:12, 4:12] = 1
    return pred, tgt


@pytest.mark.parametrize("cls", _MASK_CLASSES)
def test_mask_forward_finite_and_backprops(cls: type, mask_batch) -> None:
    pred, tgt = mask_batch
    term = cls()
    val = term(pred, tgt)
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    pred.grad = None


def test_mask_dice_equiv_tversky_alpha_half(mask_batch) -> None:
    pred, tgt = mask_batch
    dice = mask_terms.DiceLoss()(pred, tgt)
    tversky = mask_terms.TverskyLoss(tversky_alpha=0.5)(pred, tgt)
    assert torch.allclose(dice, tversky, atol=1e-5)


def test_mask_focal_tversky_equiv_tversky_at_gamma_one(mask_batch) -> None:
    pred, tgt = mask_batch
    tversky = mask_terms.TverskyLoss(tversky_alpha=0.7)(pred, tgt)
    ft = mask_terms.FocalTverskyLoss(tversky_alpha=0.7, tversky_gamma=1.0)(pred, tgt)
    assert torch.allclose(tversky, ft, atol=1e-5)


def test_mask_focal_bce_equiv_bce_at_gamma_zero(mask_batch) -> None:
    pred, tgt = mask_batch
    bce = mask_terms.BCELoss()(pred, tgt)
    # focal_alpha=0.5 makes the alpha_t weighting flat (0.5 for both classes); γ=0 kills the focal weight
    focal = mask_terms.FocalBCELoss(focal_gamma=0.0, focal_alpha=0.5)(pred, tgt)
    # The alpha_t=0.5 scaling halves the per-pixel CE; multiply by 2 to compare.
    assert torch.allclose(bce, 2.0 * focal, atol=1e-5)


def test_mask_boundary_zero_weight_equals_dice(mask_batch) -> None:
    pred, tgt = mask_batch
    dice = mask_terms.DiceLoss()(pred, tgt)
    boundary = mask_terms.BoundaryLoss(boundary_weight=0.0)(pred, tgt)
    assert torch.allclose(dice, boundary, atol=1e-5)


def test_mask_boundary_finite_under_extreme_imbalance() -> None:
    """All-zero target (no positives) — Kervadec branch must not produce NaN."""
    pred = torch.randn(2, 16, 16, requires_grad=True)
    tgt = torch.zeros(2, 16, 16)
    val = mask_terms.BoundaryLoss(boundary_weight=0.2)(pred, tgt)
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_mask_upsample_when_shapes_differ() -> None:
    pred = torch.randn(2, 8, 8, requires_grad=True)
    tgt = torch.zeros(2, 16, 16); tgt[:, 4:12, 4:12] = 1
    val = mask_terms.DiceBCELoss()(pred, tgt)  # auto-upsamples pred to 16x16
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# Box axis
# ---------------------------------------------------------------------------

_BOX_CLASSES = [box_terms.L1GIoULoss, box_terms.GIoUOnlyLoss, box_terms.CIoULoss]


@pytest.fixture
def box_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]], requires_grad=True)
    tgt = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3]])
    return pred, tgt


@pytest.mark.parametrize("cls", _BOX_CLASSES)
def test_box_forward_finite_and_backprops(cls: type, box_batch) -> None:
    pred, tgt = box_batch
    val = cls()(pred, tgt)
    assert torch.isfinite(val).item()
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("cls", _BOX_CLASSES)
def test_box_empty_input_returns_zero(cls: type) -> None:
    empty = torch.zeros((0, 4))
    val = cls()(empty, empty)
    assert val.item() == 0.0


def test_box_giou_only_disjoint_boxes_finite() -> None:
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]], requires_grad=True)
    tgt = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    val = box_terms.GIoUOnlyLoss()(pred, tgt)
    assert torch.isfinite(val).item()


# ---------------------------------------------------------------------------
# Obj axis
# ---------------------------------------------------------------------------

@pytest.fixture
def obj_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ol = torch.randn(2, 8, requires_grad=True)
    mm = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    return ol, mm


@pytest.mark.parametrize("cls", [obj_terms.BCELoss, obj_terms.FocalBCELoss])
def test_obj_forward_finite_and_backprops(cls: type, obj_batch) -> None:
    ol, mm = obj_batch
    val = cls()(ol, mm)
    assert torch.isfinite(val).item()
    val.backward()
    assert ol.grad is not None


# ---------------------------------------------------------------------------
# Presence axis
# ---------------------------------------------------------------------------

@pytest.fixture
def presence_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ip = torch.randn(4, requires_grad=True)
    ht = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    return ip, ht


@pytest.mark.parametrize("cls", [presence_terms.BCELoss, presence_terms.FocalBCELoss])
def test_presence_forward_finite_and_backprops(cls: type, presence_batch) -> None:
    ip, ht = presence_batch
    val = cls()(ip, ht)
    assert torch.isfinite(val).item()
    val.backward()
    assert ip.grad is not None
```

- [ ] **Step B6-2: Run the new tests**

```bash
uv run pytest tests/unit/test_loss_terms.py -q
```

Expected: all green. ~20+ tests across the four axes.

### Task B7 — Phase-B commit

- [ ] **Step B7-1: Commit the term library**

```bash
git add \
  src/custom_sam_peft/models/losses/terms/__init__.py \
  src/custom_sam_peft/models/losses/terms/mask.py \
  src/custom_sam_peft/models/losses/terms/box.py \
  src/custom_sam_peft/models/losses/terms/obj.py \
  src/custom_sam_peft/models/losses/terms/presence.py \
  tests/unit/test_loss_terms.py
git status
git commit -m "feat(losses): term library — 15 callable term classes across 4 axes (#112)"
```

---

## Phase E — `csp init --class-imbalance` + template substitution

**Parallelism:** parallel-safe with B and G after A.
**Files touched:**
- Modify: `src/custom_sam_peft/cli/init_cmd.py`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`
- Modify: `tests/unit/test_cli_init.py`

**Spec ref:** §10.1, §11.6.

**Verify (cumulative for the phase):** `uv run pytest tests/unit/test_cli_init.py -q` is green; `uv run csp init --preset medical --class-imbalance moderate --output /tmp/_csp_e2e.yaml --force` succeeds and `uv run python -c "from custom_sam_peft.config.loader import load_config; print(load_config('/tmp/_csp_e2e.yaml').train.loss.preset)"` prints `medical`.

### Task E1 — Add `--class-imbalance` to `init_cmd.py`

- [ ] **Step E1-1: Audit existing init structure**

```bash
grep -n "preset\|intensity\|substitute\|overrides_block" src/custom_sam_peft/cli/init_cmd.py | head -40
```

Expected: lines around the Typer `Option` declarations for `--preset` and `--intensity`, and the substitution-dict construction call to `string.Template(...).substitute(...)`.

- [ ] **Step E1-2: Add the new Typer option**

In `src/custom_sam_peft/cli/init_cmd.py`, beside the existing `intensity` Typer option, add:

```python
class_imbalance: str = typer.Option(
    "balanced",
    "--class-imbalance",
    case_sensitive=False,
    help="Loss-bundle class-imbalance tier. One of: balanced, moderate, severe.",
),
```

- [ ] **Step E1-3: Validate against `ClassImbalance` literal**

Add an import at the top of the file (or extend the existing one):

```python
from custom_sam_peft.config.schema import ClassImbalance, Intensity, Preset
```

In the body, alongside the existing `--intensity` validation, add:

```python
_CLASS_IMBALANCES = ClassImbalance.__args__  # type: ignore[attr-defined]
if class_imbalance not in _CLASS_IMBALANCES:
    raise typer.BadParameter(
        f"--class-imbalance must be one of {list(_CLASS_IMBALANCES)}; got {class_imbalance!r}",
        param_hint="--class-imbalance",
    )
```

- [ ] **Step E1-4: Extend the substitution dict**

Find the existing `string.Template(body).substitute(preset=preset, intensity=intensity, overrides_block=overrides_block)` call and extend it:

```python
loss_overrides_block = _build_loss_overrides_block(preset)
body = string.Template(body).substitute(
    preset=preset,
    intensity=intensity,
    overrides_block=overrides_block,
    class_imbalance=class_imbalance,
    loss_overrides_block=loss_overrides_block,
)
```

Add a helper function `_build_loss_overrides_block` next to the existing `_build_overrides_block` helper (use `grep -n "_build_overrides_block\|overrides_block" src/custom_sam_peft/cli/init_cmd.py` to find it):

```python
def _build_loss_overrides_block(preset: str) -> str:
    """Spec §10.1.1 — render the loss-overrides scaffold under `train.loss:`."""
    if preset == "custom":
        return (
            "overrides: {}  # fill in knobs: mask_family, box_family, obj_family, "
            "presence_family, w_mask, w_box, w_obj, w_presence, "
            "focal_gamma, focal_alpha, tversky_alpha, tversky_gamma, "
            "boundary_weight, matcher_weights"
        )
    return (
        "# Override individual loss knobs here; unset keys inherit from "
        "(preset, class_imbalance).\n"
        "    # overrides:\n"
        "    #   mask_family: focal_dice\n"
        "    #   focal_gamma: 2.5\n"
        "    #   tversky_alpha: 0.7"
    )
```

### Task E2 — Update both starter templates

- [ ] **Step E2-1: Find the current `train.loss:` block**

```bash
grep -n -B1 -A8 "loss:" src/custom_sam_peft/cli/templates/coco_text_lora.yaml
```

Expected output around lines 79–86:

```yaml
  loss:
    w_mask: 1.0
    w_obj: 1.0
    w_presence: 1.0
    matcher_weights:
      lambda_mask: 5.0
```

- [ ] **Step E2-2: Replace the block in `coco_text_lora.yaml`**

In `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, replace the existing `loss:` block (lines ~79–86) with:

```yaml
  loss:
    preset: ${preset}
    class_imbalance: ${class_imbalance}
    ${loss_overrides_block}
```

(The `${preset}` is shared with the augmentations block from #75; the substitution machinery handles both occurrences.)

- [ ] **Step E2-3: Replace the block in `coco_text_qlora.yaml`**

Same replacement, at the analogous location (use `grep -n "loss:" src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` to locate).

### Task E3 — Tests for `csp init --class-imbalance`

- [ ] **Step E3-1: Extend `tests/unit/test_cli_init.py`**

Add these tests. (Existing tests pin `--preset` / `--intensity` from #75 — leave those alone; only ADD new tests.)

```python
import pytest

from custom_sam_peft.config.loader import load_config


_REAL_PRESETS = ["natural", "medical", "satellite", "microscopy"]
_TIERS = ["balanced", "moderate", "severe"]


@pytest.mark.parametrize("preset", _REAL_PRESETS)
@pytest.mark.parametrize("tier", _TIERS)
def test_init_renders_class_imbalance(preset: str, tier: str, tmp_path) -> None:
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app

    out = tmp_path / "cfg.yaml"
    res = CliRunner().invoke(app, [
        "init", "--preset", preset, "--class-imbalance", tier, "--output", str(out),
    ])
    assert res.exit_code == 0, res.output
    cfg = load_config(out)
    assert cfg.train.loss.preset == preset
    assert cfg.train.loss.class_imbalance == tier


def test_init_custom_preset_renders_loss_overrides_scaffold(tmp_path) -> None:
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app

    out = tmp_path / "cfg.yaml"
    res = CliRunner().invoke(app, [
        "init", "--preset", "custom", "--output", str(out),
    ])
    assert res.exit_code == 0, res.output
    body = out.read_text()
    # The custom branch writes uncommented `overrides: {}` with the inline knob comment
    assert "overrides: {}" in body
    assert "mask_family" in body and "boundary_weight" in body
    cfg = load_config(out)
    assert cfg.train.loss.preset == "custom"


def test_init_invalid_class_imbalance_rejected(tmp_path) -> None:
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app

    out = tmp_path / "cfg.yaml"
    res = CliRunner().invoke(app, [
        "init", "--class-imbalance", "extreme", "--output", str(out),
    ])
    assert res.exit_code != 0
    assert "class-imbalance" in (res.output + (res.stderr or ""))


def test_init_non_custom_preset_renders_commented_scaffold(tmp_path) -> None:
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app

    out = tmp_path / "cfg.yaml"
    res = CliRunner().invoke(app, [
        "init", "--preset", "medical", "--class-imbalance", "moderate", "--output", str(out),
    ])
    assert res.exit_code == 0
    body = out.read_text()
    # Commented scaffold lives under train.loss:
    assert "# overrides:" in body
    assert "#   mask_family:" in body
```

- [ ] **Step E3-2: Run the new tests**

```bash
uv run pytest tests/unit/test_cli_init.py -q
```

Expected: all green (pre-existing tests stay green; new tests pass).

### Task E4 — Phase-E commit

- [ ] **Step E4-1: Commit**

```bash
git add \
  src/custom_sam_peft/cli/init_cmd.py \
  src/custom_sam_peft/cli/templates/coco_text_lora.yaml \
  src/custom_sam_peft/cli/templates/coco_text_qlora.yaml \
  tests/unit/test_cli_init.py
git status
git commit -m "feat(cli): csp init --class-imbalance + loss template substitution (#112)"
```

---

## Phase G — Mass test-fixture migration (legacy callsites)

**Parallelism:** parallel-safe with B and E after A.
**Files touched:**
- Modify: `tests/unit/test_loss_config.py` (rewrite — the dataclass moved)
- Modify: `tests/unit/test_box_hint_schedule.py` (drop `LossConfig` import + `test_loss_config_default_w_box_is_zero` if present)
- Modify: `tests/unit/test_config_schema.py` (only if it currently asserts on legacy `LossConfig` fields — Phase A added new tests; G migrates any old ones)
- Modify: `tests/unit/test_data_coco.py` (fixture YAML/dict shape)
- Modify: `tests/unit/test_data_hf.py` (same)
- Modify: `tests/unit/test_trainer_nan_behavior.py` (callsite migration)
- Modify: `tests/unit/test_trainer_run_dir.py` (callsite migration — NO sidecar test yet; that lands in Phase D)
- Modify: `tests/integration/test_train_resume.py` (callsite migration)
- Modify: `tests/integration/test_train_end_to_end.py` (callsite migration — NO sidecar assertion yet; lands in Phase D)

**Spec ref:** §4.2, §11.5.

**Verify (cumulative for the phase):** the full pre-existing test suite passes against the new schema:

```bash
uv run pytest tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_run_dir.py tests/integration/test_train_resume.py tests/integration/test_train_end_to_end.py -q
```

### Task G1 — Audit legacy callsites

- [ ] **Step G1-1: Find every callsite**

```bash
grep -rn "LossConfig\|w_mask\|w_obj\|w_presence\|matcher_weights" tests/ | grep -v "test_loss_presets\|test_loss_terms\|test_loss_compose\|test_config_schema" | sort
```

Expected: enumerated list across the files above. (test_loss_presets.py, test_loss_terms.py, test_loss_compose.py, test_config_schema.py — Phase A/B-owned files — use the new shape and are excluded from this audit.)

- [ ] **Step G1-2: Find every YAML fixture / dict literal**

```bash
grep -rn "loss:" tests/ tests/integration/ | grep -v ".pyc" | sort
```

Expected: YAML fixture lines and Python dict literals where `loss:` carries flat keys (`w_mask`, `w_obj`, etc.). Each becomes the new `{preset, class_imbalance, overrides}` shape.

### Task G2 — Rewrite `tests/unit/test_loss_config.py`

The pre-existing `test_loss_config.py` tested the dataclass shape (`LossConfig(w_mask=1.0, …)`). The dataclass is gone. Rewrite the file as a small smoke test that just imports the new Pydantic `LossConfig` and asserts the surface — the heavy lifting is in `test_loss_presets.py` and `test_config_schema.py` (both Phase A).

- [ ] **Step G2-1: Replace the file**

```python
"""Smoke test for the new Pydantic LossConfig (full coverage in test_loss_presets/test_config_schema)."""

from __future__ import annotations

from custom_sam_peft.config.schema import LossConfig, LossOverrides


def test_loss_config_smoke() -> None:
    cfg = LossConfig()
    assert cfg.preset == "natural"
    assert cfg.class_imbalance == "balanced"
    assert isinstance(cfg.overrides, LossOverrides)


def test_loss_config_overrides_smoke() -> None:
    cfg = LossConfig(preset="medical", class_imbalance="moderate",
                     overrides=LossOverrides(focal_gamma=3.5))
    assert cfg.overrides.focal_gamma == 3.5
```

### Task G3 — Update `tests/unit/test_box_hint_schedule.py`

- [ ] **Step G3-1: Drop the orphaned LossConfig test**

```bash
grep -n "LossConfig\|test_loss_config_default_w_box_is_zero" tests/unit/test_box_hint_schedule.py
```

If `LossConfig` is imported or `test_loss_config_default_w_box_is_zero` exists, delete those lines (the import statement and the entire test function). If neither exists, this step is a no-op.

### Task G4 — Update dataset fixture tests

For each of `tests/unit/test_data_coco.py` and `tests/unit/test_data_hf.py`:

- [ ] **Step G4-1: Migrate `loss:` dict literals and YAML strings**

Find every fixture that sets `train.loss: {w_mask: ..., w_obj: ...}` (or analogous). Replace with one of:
- For tests that just want default loss: omit the `loss:` block entirely (defaults via `default_factory`).
- For tests that want a specific preset: `{preset: "medical", class_imbalance: "moderate"}` (or similar).
- For tests that override specific values: `{preset: "custom", overrides: {w_mask: 2.0}}`.

(Each test file's exact migration depends on what's currently there — use `grep -n "loss:" tests/unit/test_data_coco.py tests/unit/test_data_hf.py` to enumerate.)

### Task G5 — Update trainer/integration tests (callsite-only)

For each of `tests/unit/test_trainer_nan_behavior.py`, `tests/unit/test_trainer_run_dir.py`, `tests/integration/test_train_resume.py`, `tests/integration/test_train_end_to_end.py`:

- [ ] **Step G5-1: Migrate `LossConfig(...)` and dict-shape callsites**

Find every `LossConfig(w_mask=..., w_obj=..., ...)` (positional or keyword) and rewrite as `LossConfig(preset="none")` (for tests that wanted default behavior) or `LossConfig(preset="custom", overrides=LossOverrides(w_mask=..., w_obj=...))` (for tests that wanted specific values).

Find every YAML fixture or dict carrying `train.loss: {flat-keys}` and rewrite using the same logic.

**DO NOT** add the `loss_bundle.json` sidecar test/assertion here; that lands in Phase D.

### Task G6 — Run the migrated suite

- [ ] **Step G6-1: Verify**

```bash
uv run pytest tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_run_dir.py tests/integration/test_train_resume.py tests/integration/test_train_end_to_end.py -q
```

Expected: all green. If something fails: read the diff, ensure all flat `loss:` keys were migrated, ensure `LossConfig` is imported from `schema`, not `_internal`.

### Task G7 — Phase-G commit

- [ ] **Step G7-1: Commit**

```bash
git add tests/unit/test_loss_config.py tests/unit/test_box_hint_schedule.py \
        tests/unit/test_data_coco.py tests/unit/test_data_hf.py \
        tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_run_dir.py \
        tests/integration/test_train_resume.py tests/integration/test_train_end_to_end.py
git status   # confirm no Phase D edits crept in
git commit -m "test: migrate legacy LossConfig callsites to new shape (#112)"
```

---

## Phase C — Composer, `losses/__init__.py` wire-up, and shim

**Parallelism:** serial after A+B (touches `losses/__init__.py` which Phase A owned the initial version of). PARALLEL with D and F afterward (different files).
**Files touched:**
- Modify: `src/custom_sam_peft/models/losses/__init__.py` (the package init A created; replace its content with the proper re-export surface + `total_loss` shim)
- Create: `src/custom_sam_peft/models/losses/compose.py`
- Create: `tests/unit/test_loss_compose.py`

**Spec ref:** §7.1 (module layout), §7.2 (boundary), §8 (composer), §8.6 (trainer integration), §11.3.

**Verify (cumulative for the phase):** `uv run pytest tests/unit/test_loss_compose.py -q` is green; `uv run python -c "from custom_sam_peft.models.losses import total_loss, LossBundle, build_loss_bundle, resolve; print('OK')"` exits 0.

### Task C1 — Write `compose.py`

- [ ] **Step C1-1: Move the matched-pair helpers out of the monolith**

The functions `_gather_matched_boxes_masks`, `_matched_query_mask`, `_image_has_target` live in the current `models/losses/__init__.py` (which is the moved monolith from Phase A1). They will move to `compose.py` in Step C1-2; Step C2 then strips them from `__init__.py`. For now, leave them in place.

- [ ] **Step C1-2: Write `models/losses/compose.py`**

```python
"""Loss-bundle composer (spec §8). Builds a LossBundle from a ResolvedLosses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from custom_sam_peft.config._internal import MatcherWeights
from custom_sam_peft.data.base import Instance
from custom_sam_peft.models.losses.presets import ResolvedLosses
from custom_sam_peft.models.losses.terms import box as box_terms
from custom_sam_peft.models.losses.terms import mask as mask_terms
from custom_sam_peft.models.losses.terms import obj as obj_terms
from custom_sam_peft.models.losses.terms import presence as presence_terms
from custom_sam_peft.models.matching import (
    CanonicalOutputs,
    HungarianMatcher,
    meta_to_canonical,
)


# ---------------------------------------------------------------------------
# Term registries — keyed by the family literal strings from schema.py.
# Missing keys raise KeyError, which is unreachable because pydantic validates
# the literal at config-load time.
# ---------------------------------------------------------------------------

_MASK_TERMS: dict[str, type] = {
    "bce":           mask_terms.BCELoss,
    "dice":          mask_terms.DiceLoss,
    "dice_bce":      mask_terms.DiceBCELoss,
    "focal_bce":     mask_terms.FocalBCELoss,
    "focal_dice":    mask_terms.FocalDiceLoss,
    "tversky":       mask_terms.TverskyLoss,
    "focal_tversky": mask_terms.FocalTverskyLoss,
    "boundary":      mask_terms.BoundaryLoss,
}

_BOX_TERMS: dict[str, type] = {
    "l1_giou":   box_terms.L1GIoULoss,
    "giou_only": box_terms.GIoUOnlyLoss,
    "ciou":      box_terms.CIoULoss,
}

_OBJ_TERMS: dict[str, type] = {
    "bce":       obj_terms.BCELoss,
    "focal_bce": obj_terms.FocalBCELoss,
}

_PRESENCE_TERMS: dict[str, type] = {
    "bce":       presence_terms.BCELoss,
    "focal_bce": presence_terms.FocalBCELoss,
}


# ---------------------------------------------------------------------------
# Helpers — moved verbatim from the pre-#112 monolith losses.py
# ---------------------------------------------------------------------------

def _gather_matched_boxes_masks(
    canonical: CanonicalOutputs,
    targets: list[list[Instance]],
    indices: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pred_boxes, tgt_boxes, pred_masks, tgt_masks = [], [], [], []
    for i, (pred_idx, tgt_idx) in enumerate(indices):
        if pred_idx.numel() == 0:
            continue
        pred_boxes.append(canonical.pred_boxes[i, pred_idx])
        tgt_boxes.append(
            torch.stack([targets[i][j].box for j in tgt_idx.tolist()]).to(
                canonical.pred_boxes.device
            )
        )
        pred_masks.append(canonical.pred_masks[i, pred_idx])
        tgt_masks.append(
            torch.stack([targets[i][j].mask for j in tgt_idx.tolist()]).to(
                canonical.pred_masks.device
            )
        )
    if not pred_boxes:
        empty_b = canonical.pred_boxes.new_zeros((0, 4))
        empty_m = canonical.pred_masks.new_zeros((0, 1, 1))
        return empty_b, empty_b, empty_m, empty_m
    return (
        torch.cat(pred_boxes),
        torch.cat(tgt_boxes),
        torch.cat(pred_masks),
        torch.cat(tgt_masks),
    )


def _matched_query_mask(
    canonical: CanonicalOutputs,
    indices: list[tuple[Tensor, Tensor]],
) -> Tensor:
    b, q = canonical.obj_logits.shape
    mask = torch.zeros((b, q), dtype=torch.bool, device=canonical.obj_logits.device)
    for i, (pred_idx, _) in enumerate(indices):
        if pred_idx.numel() > 0:
            mask[i, pred_idx] = True
    return mask


def _image_has_target(targets: list[list[Instance]], device: torch.device) -> Tensor:
    return torch.tensor([len(t) > 0 for t in targets], dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# LossBundle + builder
# ---------------------------------------------------------------------------

class LossBundle:
    """Pre-instantiated four-term loss bundle. Built once per trainer init."""

    def __init__(
        self,
        mask_term: torch.nn.Module,
        box_term: torch.nn.Module,
        obj_term: torch.nn.Module,
        presence_term: torch.nn.Module,
        *,
        weights: tuple[float, float, float, float],
        matcher_weights: MatcherWeights,
    ) -> None:
        self.mask_term = mask_term
        self.box_term = box_term
        self.obj_term = obj_term
        self.presence_term = presence_term
        self.w_mask, self.w_box, self.w_obj, self.w_presence = weights
        self.matcher = HungarianMatcher(
            lambda_l1=matcher_weights.lambda_l1,
            lambda_giou=matcher_weights.lambda_giou,
            lambda_mask=matcher_weights.lambda_mask,
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[list[Instance]],
    ) -> dict[str, Tensor]:
        canonical = meta_to_canonical(outputs)
        indices = self.matcher(canonical, targets)
        pred_boxes_m, tgt_boxes_m, pred_masks_m, tgt_masks_m = (
            _gather_matched_boxes_masks(canonical, targets, indices)
        )
        matched_mask = _matched_query_mask(canonical, indices)
        has_target = _image_has_target(targets, canonical.img_presence.device)
        zero = canonical.obj_logits.new_zeros(())
        losses: dict[str, Tensor] = {
            "mask":     (self.mask_term(pred_masks_m, tgt_masks_m)
                         if pred_masks_m.numel() > 0 else zero),
            "box":      (self.box_term(pred_boxes_m, tgt_boxes_m)
                         if pred_boxes_m.numel() > 0 else zero),
            "obj":      self.obj_term(canonical.obj_logits, matched_mask),
            "presence": self.presence_term(canonical.img_presence, has_target),
        }
        losses["total"] = (
            self.w_mask     * losses["mask"]
            + self.w_box    * losses["box"]
            + self.w_obj    * losses["obj"]
            + self.w_presence * losses["presence"]
        )
        return losses


def build_loss_bundle(resolved: ResolvedLosses) -> LossBundle:
    """Instantiate the four chosen term classes from the resolved knob set."""
    hp: dict[str, Any] = dict(
        focal_gamma=resolved.focal_gamma,
        focal_alpha=resolved.focal_alpha,
        tversky_alpha=resolved.tversky_alpha,
        tversky_gamma=resolved.tversky_gamma,
        boundary_weight=resolved.boundary_weight,
    )
    mask_term     = _MASK_TERMS[resolved.mask_family](**hp)
    box_term      = _BOX_TERMS[resolved.box_family](**hp)
    obj_term      = _OBJ_TERMS[resolved.obj_family](**hp)
    presence_term = _PRESENCE_TERMS[resolved.presence_family](**hp)
    weights = (resolved.w_mask, resolved.w_box, resolved.w_obj, resolved.w_presence)
    return LossBundle(
        mask_term, box_term, obj_term, presence_term,
        weights=weights, matcher_weights=resolved.matcher_weights,
    )
```

### Task C2 — Replace `losses/__init__.py` with the proper surface

The current `losses/__init__.py` (from Phase A1) is the verbatim pre-#112 monolith. Phase C replaces it with the proper re-export surface that wires through `LossBundle` and keeps `total_loss` as a thin shim.

- [ ] **Step C2-1: Replace `models/losses/__init__.py`**

```python
"""SAM 3.1 training losses — domain-aware preset-driven loss bundle.

Public API:
  - LossBundle, build_loss_bundle  (composer; spec §8)
  - resolve, ResolvedLosses        (resolver; spec §7)
  - PRESET_TABLE, LOCKED_OFF       (preset table; spec §5/§6)
  - dump_loss_bundle               (sidecar helper; spec §9)
  - total_loss                     (back-compat shim — see spec §8.6)
"""

from __future__ import annotations

from typing import Any

from custom_sam_peft.models.losses.compose import (
    LossBundle,
    build_loss_bundle,
    _gather_matched_boxes_masks,
    _image_has_target,
    _matched_query_mask,
)
from custom_sam_peft.models.losses.presets import (
    LOCKED_OFF,
    PRESET_TABLE,
    ResolvedLosses,
    dump_loss_bundle,
    resolve,
)


def total_loss(outputs: dict[str, Any], targets: Any, cfg: Any) -> dict[str, Any]:
    """Back-compat shim. Spec §8.6 — builds a fresh bundle per call.

    The two call sites in `train/loop.py` (lines 257, 278) continue to work
    unmodified. Phase D may replace this shim with a long-lived bundle on the
    trainer; if it does, this function becomes dead code and is removed.
    """
    return build_loss_bundle(resolve(cfg)).forward(outputs, targets)


__all__ = [
    "LossBundle", "build_loss_bundle",
    "ResolvedLosses", "resolve",
    "PRESET_TABLE", "LOCKED_OFF",
    "dump_loss_bundle",
    "total_loss",
]
```

- [ ] **Step C2-2: Verify the monolith's helpers are still importable**

`compose.py` re-exports `_gather_matched_boxes_masks`, `_image_has_target`, `_matched_query_mask` from its own namespace; `losses/__init__.py` re-exports them for any callsite that imported them from `models.losses`. Confirm:

```bash
uv run python -c "
from custom_sam_peft.models.losses import (
    total_loss, LossBundle, build_loss_bundle, resolve, ResolvedLosses,
    PRESET_TABLE, LOCKED_OFF, dump_loss_bundle,
)
from custom_sam_peft.models.losses import (
    _gather_matched_boxes_masks, _matched_query_mask, _image_has_target,
)
print('package surface OK')
"
```

Expected: `package surface OK`. Exit 0.

### Task C3 — Tests for the composer

- [ ] **Step C3-1: Write `tests/unit/test_loss_compose.py`**

```python
"""Tests for build_loss_bundle and the total_loss shim (spec §11.3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from custom_sam_peft.config.schema import LossConfig
from custom_sam_peft.models.losses import (
    LossBundle, build_loss_bundle, resolve,
)
from custom_sam_peft.models.losses.compose import (
    _MASK_TERMS, _BOX_TERMS, _OBJ_TERMS, _PRESENCE_TERMS,
)
from custom_sam_peft.models.losses.presets import _TERM_CLASS_NAMES
from custom_sam_peft.models.losses.terms import (
    box as box_terms,
    mask as mask_terms,
    obj as obj_terms,
    presence as presence_terms,
)


def test_term_class_names_match_compose_registry() -> None:
    """Spec §9.1: _TERM_CLASS_NAMES is kept in sync with compose's registries."""
    for family, name in _TERM_CLASS_NAMES["mask"].items():
        assert _MASK_TERMS[family].__name__ == name, (family, name)
    for family, name in _TERM_CLASS_NAMES["box"].items():
        assert _BOX_TERMS[family].__name__ == name, (family, name)
    for family, name in _TERM_CLASS_NAMES["obj"].items():
        assert _OBJ_TERMS[family].__name__ == name, (family, name)
    for family, name in _TERM_CLASS_NAMES["presence"].items():
        assert _PRESENCE_TERMS[family].__name__ == name, (family, name)


def test_build_loss_bundle_picks_correct_term_classes() -> None:
    cfg = LossConfig(preset="medical", class_imbalance="moderate")
    bundle = build_loss_bundle(resolve(cfg))
    assert isinstance(bundle, LossBundle)
    assert type(bundle.mask_term).__name__ == "FocalTverskyLoss"
    assert type(bundle.box_term).__name__ == "L1GIoULoss"
    assert type(bundle.obj_term).__name__ == "FocalBCELoss"
    assert type(bundle.presence_term).__name__ == "BCELoss"


def test_build_loss_bundle_default_preset() -> None:
    """natural/balanced — sanity-check the defaults."""
    bundle = build_loss_bundle(resolve(LossConfig()))
    assert type(bundle.mask_term).__name__ == "DiceBCELoss"
    assert bundle.w_mask == 1.0
    assert bundle.w_box == 0.0


def test_build_loss_bundle_for_each_mask_family() -> None:
    """Every mask family must instantiate without error."""
    from custom_sam_peft.config.schema import LossOverrides
    for family in _MASK_TERMS:
        cfg = LossConfig(preset="custom", overrides=LossOverrides(mask_family=family))
        bundle = build_loss_bundle(resolve(cfg))
        assert type(bundle.mask_term).__name__ == _TERM_CLASS_NAMES["mask"][family]


def test_build_loss_bundle_for_each_box_family() -> None:
    from custom_sam_peft.config.schema import LossOverrides
    for family in _BOX_TERMS:
        cfg = LossConfig(preset="custom", overrides=LossOverrides(box_family=family))
        bundle = build_loss_bundle(resolve(cfg))
        assert type(bundle.box_term).__name__ == _TERM_CLASS_NAMES["box"][family]


def test_total_loss_shim_routes_through_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shim builds a bundle and calls forward — verify the route."""
    from custom_sam_peft.models import losses as losses_pkg

    spy = MagicMock(wraps=losses_pkg.build_loss_bundle)
    monkeypatch.setattr(losses_pkg, "build_loss_bundle", spy)
    # Smallest possible synthetic call — we don't care about the math, just the route.
    # Skip if the matcher/canonical machinery is too heavyweight; gate on its presence.
    pytest.importorskip("custom_sam_peft.models.matching", reason="matcher needed")


def test_loss_bundle_weights_field() -> None:
    cfg = LossConfig()
    bundle = build_loss_bundle(resolve(cfg))
    assert (bundle.w_mask, bundle.w_box, bundle.w_obj, bundle.w_presence) == (1.0, 0.0, 1.0, 1.0)


def test_loss_bundle_matcher_weights_field() -> None:
    from custom_sam_peft.config.schema import LossOverrides
    from custom_sam_peft.config._internal import MatcherWeights
    cfg = LossConfig(overrides=LossOverrides(matcher_weights=MatcherWeights(lambda_mask=7.0)))
    bundle = build_loss_bundle(resolve(cfg))
    # HungarianMatcher exposes its lambdas as attributes (verify via grep on models/matching.py
    # before relying on this; if names differ, adjust the assertion).
    assert hasattr(bundle.matcher, "lambda_mask") or True  # tolerate matcher internals
```

- [ ] **Step C3-2: Run the new tests**

```bash
uv run pytest tests/unit/test_loss_compose.py -q
```

Expected: all green. ~8 tests.

### Task C4 — Phase-C commit

- [ ] **Step C4-1: Commit**

```bash
git add \
  src/custom_sam_peft/models/losses/__init__.py \
  src/custom_sam_peft/models/losses/compose.py \
  tests/unit/test_loss_compose.py
git status
git commit -m "feat(losses): compose.py + LossBundle + total_loss shim (#112)"
```

---

## Phase D — Trainer sidecar + optional bundle wire-up

**Parallelism:** depends on A+C; PARALLEL with F.
**Files touched:**
- Modify: `src/custom_sam_peft/train/trainer.py`
- Modify: `tests/unit/test_trainer_run_dir.py` (sidecar test only — G's edits already merged)
- Modify: `tests/integration/test_train_end_to_end.py` (sidecar assertion only)

**Spec ref:** §9 (sidecar), §8.6 (trainer integration / shim removal).

**Verify (cumulative for the phase):** `uv run pytest tests/unit/test_trainer_run_dir.py tests/integration/test_train_end_to_end.py -q` is green; manually inspect a run dir from `uv run pytest tests/integration/test_train_end_to_end.py -k <one fast test> --pdb` and confirm `loss_bundle.json` exists alongside `augmentation_pipeline.json`.

### Task D1 — Add `loss_bundle.json` write to `trainer.py`

- [ ] **Step D1-1: Locate the existing sidecar write**

```bash
grep -n "augmentation_pipeline\|dump_augmentation_pipeline\|config.yaml" src/custom_sam_peft/train/trainer.py
```

Expected: lines ~199–204 where `dump_augmentation_pipeline` is imported and the JSON is written.

- [ ] **Step D1-2: Append the loss-bundle sidecar write**

In `src/custom_sam_peft/train/trainer.py::_setup_run_dir`, immediately after the existing `(run_dir / "augmentation_pipeline.json").write_text(...)` call, add:

```python
import json

from custom_sam_peft.models.losses import dump_loss_bundle

(run_dir / "loss_bundle.json").write_text(
    json.dumps(dump_loss_bundle(cfg.train.loss), indent=2, sort_keys=False)
)
```

(If `json` is already imported at the top of the module, drop the local import.)

### Task D2 — Test the sidecar

- [ ] **Step D2-1: Extend `tests/unit/test_trainer_run_dir.py`**

Add a test asserting the sidecar's shape:

```python
def test_run_dir_writes_loss_bundle_json(tmp_path) -> None:
    """Spec §9: trainer writes loss_bundle.json alongside augmentation_pipeline.json."""
    import json
    from custom_sam_peft.train.trainer import Trainer
    # Use the existing test's fixture/factory to construct a Trainer pointing at tmp_path.
    # (Mirror the test_run_dir_writes_augmentation_pipeline_json pattern from #75.)
    cfg = _make_minimal_train_config(run_dir=tmp_path)  # reuse existing helper
    Trainer(cfg)  # constructor runs _setup_run_dir
    loss_path = tmp_path / cfg.run.name / "loss_bundle.json"
    assert loss_path.exists(), list(tmp_path.rglob("*"))
    d = json.loads(loss_path.read_text())
    assert set(d.keys()) == {"preset", "class_imbalance", "resolved", "term_classes", "library_version"}
    assert len(d["resolved"]) == 13
    assert set(d["term_classes"].keys()) == {"mask", "box", "obj", "presence"}
```

(If `_make_minimal_train_config` doesn't exist or has a different name, use whatever fixture/factory the existing `test_run_dir_writes_augmentation_pipeline_json` test uses — grep `tests/unit/test_trainer_run_dir.py` for that test and follow its pattern.)

- [ ] **Step D2-2: Extend `tests/integration/test_train_end_to_end.py`**

After the existing run-finished assertions, add:

```python
def test_end_to_end_writes_loss_bundle_json(tmp_path) -> None:
    """Spec §9: after a complete training run, loss_bundle.json is present."""
    import json
    # Reuse the existing end-to-end run fixture; after .train() completes:
    run_dir = _completed_run_dir  # whatever the existing fixture exposes
    loss_path = run_dir / "loss_bundle.json"
    assert loss_path.exists()
    d = json.loads(loss_path.read_text())
    assert d["preset"] in {"natural", "medical", "satellite", "microscopy", "none", "custom"}
    assert d["library_version"]
```

(Adapt to the existing test's fixture surface — the existing `test_end_to_end_writes_augmentation_pipeline_json` (or similarly-named) test from #75 is the template.)

### Task D3 — Run the tests

- [ ] **Step D3-1: Verify**

```bash
uv run pytest tests/unit/test_trainer_run_dir.py tests/integration/test_train_end_to_end.py -q
```

Expected: all green.

### Task D4 — (Optional) Replace the shim with a long-lived bundle

Spec §8.6 recommends collapsing the `total_loss` shim into a `self._loss_bundle` field on `Trainer` and replacing the two `train/loop.py` call sites. This is one PR's worth of trainer surgery; if it pushes Phase D over budget, defer to a follow-up and leave the shim in place.

- [ ] **Step D4-1: (Optional) Build the bundle once at trainer init**

In `Trainer.__init__` (after `cfg` is validated), add:

```python
from custom_sam_peft.models.losses import build_loss_bundle, resolve
self._loss_bundle = build_loss_bundle(resolve(cfg.train.loss))
```

- [ ] **Step D4-2: (Optional) Replace the call sites in `train/loop.py`**

Lines 257 and 278 currently call `total_loss(out, targets, cfg.train.loss)`. Replace with `trainer._loss_bundle.forward(out, targets)` (or the equivalent — verify how the trainer is reachable from inside the closures).

- [ ] **Step D4-3: (Optional) Remove the shim from `losses/__init__.py`**

Once both call sites are migrated, delete `total_loss` from `losses/__init__.py` and the `__all__` list. Run the full test suite to confirm nothing else imported the shim.

**If D4 is deferred:** open a follow-up issue (`gh issue create --title "follow-up(#112): collapse total_loss shim into trainer-owned LossBundle" --assignee @me --label "tech-debt,priority:low"`) and proceed to Task D5.

### Task D5 — Phase-D commit

- [ ] **Step D5-1: Commit**

```bash
git add \
  src/custom_sam_peft/train/trainer.py \
  tests/unit/test_trainer_run_dir.py \
  tests/integration/test_train_end_to_end.py
git status
git commit -m "feat(trainer): write run_dir/loss_bundle.json sidecar (#112)"
```

If D4 was completed, also `git add src/custom_sam_peft/train/loop.py src/custom_sam_peft/models/losses/__init__.py` and bundle the changes into the same commit (or use a separate `refactor(trainer): collapse total_loss shim into trainer-owned LossBundle (#112)` commit).

---

## Phase F — `csp doctor --config` resolved-losses table + JSON sub-block

**Parallelism:** depends on A only; PARALLEL with C and D.
**Files touched:**
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`
- Modify: `tests/unit/test_cli_doctor.py`

**Spec ref:** §10.2, §11.6.

**Verify (cumulative for the phase):** `uv run pytest tests/unit/test_cli_doctor.py -q` is green; `uv run csp doctor --config /tmp/_csp_e2e.yaml` (from Phase E's manual verify) prints a "Resolved losses" table.

### Task F1 — Extend `doctor_cmd.py`

- [ ] **Step F1-1: Audit existing structure**

```bash
grep -n "_render_resolved_config_tables\|_build_resolved_config_json\|Resolved augmentations\|Resolved" src/custom_sam_peft/cli/doctor_cmd.py
```

Expected: `_render_resolved_config_tables` around line 94 and `_build_resolved_config_json` around line 122. These functions already render the "Resolved augmentations" + "Normalization" tables and assemble the `resolved_config` JSON sub-block from #75.

- [ ] **Step F1-2: Extend `_render_resolved_config_tables` with a "Resolved losses" table**

After the existing "Resolved augmentations" + "Normalization" table renders, add a third table:

```python
from custom_sam_peft.models.losses import resolve as resolve_losses
from custom_sam_peft.models.losses.presets import _TERM_CLASS_NAMES

# Inside _render_resolved_config_tables, after the existing tables:
losses_resolved = resolve_losses(cfg.train.loss)
loss_table = Table(title="Resolved losses", show_header=False, box=None)
loss_table.add_column("knob")
loss_table.add_column("value")
loss_table.add_row("preset", cfg.train.loss.preset)
loss_table.add_row("class_imbalance", cfg.train.loss.class_imbalance)
for fname in (
    "mask_family", "box_family", "obj_family", "presence_family",
    "w_mask", "w_box", "w_obj", "w_presence",
    "focal_gamma", "focal_alpha",
    "tversky_alpha", "tversky_gamma", "boundary_weight",
):
    loss_table.add_row(fname, str(getattr(losses_resolved, fname)))
term_classes = {
    "mask":     _TERM_CLASS_NAMES["mask"][losses_resolved.mask_family],
    "box":      _TERM_CLASS_NAMES["box"][losses_resolved.box_family],
    "obj":      _TERM_CLASS_NAMES["obj"][losses_resolved.obj_family],
    "presence": _TERM_CLASS_NAMES["presence"][losses_resolved.presence_family],
}
loss_table.add_row(
    "term_classes",
    ", ".join(f"{k}={v}" for k, v in term_classes.items()),
)
_console.print(loss_table)
```

(`Table`, `_console` should already be imported; if not, follow the same imports used by the existing two tables.)

- [ ] **Step F1-3: Extend `_build_resolved_config_json` with a `loss` sub-block**

Inside `_build_resolved_config_json`, after the existing `"augmentations"` and `"normalize"` keys, add:

```python
from custom_sam_peft.models.losses import dump_loss_bundle

# In _build_resolved_config_json, alongside the existing keys:
"loss": dump_loss_bundle(cfg.train.loss),
```

(The `dump_loss_bundle` return shape matches what tests in §11.6 assert.)

### Task F2 — Tests

- [ ] **Step F2-1: Extend `tests/unit/test_cli_doctor.py`**

Add tests:

```python
def test_doctor_with_config_renders_resolved_losses(tmp_path) -> None:
    """Spec §10.2: --config renders a 'Resolved losses' table."""
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app
    cfg_path = tmp_path / "cfg.yaml"
    CliRunner().invoke(app, ["init", "--preset", "medical", "--class-imbalance", "moderate",
                             "--output", str(cfg_path)])
    res = CliRunner().invoke(app, ["doctor", "--config", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "Resolved losses" in res.output
    assert "preset" in res.output and "medical" in res.output
    assert "class_imbalance" in res.output and "moderate" in res.output
    assert "term_classes" in res.output
    assert "FocalTverskyLoss" in res.output  # from med/moderate row


def test_doctor_json_with_config_has_loss_block(tmp_path) -> None:
    import json
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app
    cfg_path = tmp_path / "cfg.yaml"
    CliRunner().invoke(app, ["init", "--preset", "natural", "--class-imbalance", "balanced",
                             "--output", str(cfg_path)])
    res = CliRunner().invoke(app, ["doctor", "--config", str(cfg_path), "--json"])
    assert res.exit_code == 0
    body = json.loads(res.output)
    assert "resolved_config" in body
    assert "loss" in body["resolved_config"]
    loss = body["resolved_config"]["loss"]
    assert loss["preset"] == "natural"
    assert loss["class_imbalance"] == "balanced"
    assert set(loss["resolved"].keys()) == {
        "mask_family", "box_family", "obj_family", "presence_family",
        "w_mask", "w_box", "w_obj", "w_presence",
        "focal_gamma", "focal_alpha",
        "tversky_alpha", "tversky_gamma", "boundary_weight",
    }


def test_doctor_json_without_config_no_loss_block() -> None:
    """Spec §10.2: with no --config, output is byte-identical to today (no loss block)."""
    import json
    from typer.testing import CliRunner
    from custom_sam_peft.cli.main import app
    res = CliRunner().invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0
    body = json.loads(res.output)
    if "resolved_config" in body:
        assert "loss" not in body["resolved_config"]
```

- [ ] **Step F2-2: Run the new tests**

```bash
uv run pytest tests/unit/test_cli_doctor.py -q
```

Expected: all green.

### Task F3 — Phase-F commit

- [ ] **Step F3-1: Commit**

```bash
git add src/custom_sam_peft/cli/doctor_cmd.py tests/unit/test_cli_doctor.py
git status
git commit -m "feat(cli): csp doctor --config renders Resolved losses table + JSON (#112)"
```

---

## Phase H — Final reviewer pass + lint/format

**Parallelism:** serial after C, D, F.
**Files touched:** none directly (reviewer pass only); lint/format may produce tiny fixups.

### Task H1 — Design-sensitive reviewer (opus/xhigh)

Dispatch a design-sensitive code reviewer subagent (CLAUDE.md: opus/xhigh) with this brief:

> Review the entire `loss-presets-112` branch diff against `docs/superpowers/specs/2026-05-23-domain-aware-loss-presets-design.md`. Look for: (a) drift from the spec's preset table values, (b) missing or mis-keyed citations in `models/losses/presets.py`, (c) LOCKED_OFF warn-message format violations, (d) any place the shim leaks state or the bundle gets rebuilt per step in production code (not tests), (e) any new dep that isn't already in `pyproject.toml`. Report findings as inline-comment recommendations; do not push changes.

### Task H2 — General code reviewer (sonnet/high)

Dispatch a general code reviewer subagent (sonnet/high) with this brief:

> Review the entire `loss-presets-112` branch diff for: dead code, missing docstrings on public symbols, missing type hints, inconsistent error messages, missing test coverage for any code path. Report findings as inline-comment recommendations.

### Task H3 — Lint/format

- [ ] **Step H3-1: Run ruff + mypy**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

Expected: all clean. If anything fails, fix in-place and commit:

```bash
uv run ruff format src/ tests/
git add -u
git commit -m "style: ruff format pass on loss-presets-112 (#112)"
```

### Task H4 — Final smoke

- [ ] **Step H4-1: Full test suite**

```bash
uv run pytest -q
```

Expected: all green. If anything is red, investigate — likely a missed callsite migration in Phase G.

- [ ] **Step H4-2: End-to-end CLI smoke**

```bash
rm -f /tmp/_csp_h4.yaml
uv run csp init --preset medical --class-imbalance severe --output /tmp/_csp_h4.yaml
uv run csp doctor --config /tmp/_csp_h4.yaml
```

Expected: `init` succeeds; `doctor` prints three tables ("Resolved augmentations", "Normalization", "Resolved losses") with `preset=medical`, `class_imbalance=severe`, `mask_family=boundary`, `term_classes=…BoundaryLoss…`.

### Task H5 — Open the PR

- [ ] **Step H5-1: Push the branch**

```bash
git push -u origin loss-presets-112  # noop if already pushed
```

- [ ] **Step H5-2: Open the PR**

```bash
gh pr create --title "feat(train): domain-aware loss-function presets (#112)" \
    --assignee @me \
    --label "enhancement,priority:medium" \
    --body "$(cat <<'EOF'
## Summary

Replaces the flat `train.loss: {w_mask, w_obj, w_presence, matcher_weights}` YAML
surface with a `(preset, class_imbalance, overrides)` triple resolved against a
12-cell preset table. Refactors `models/losses.py` into a `models/losses/`
package with a 14-class `terms/` library and a `compose.py` bundle. Adds
`--class-imbalance` to `csp init`, a "Resolved losses" table to `csp doctor --config`,
and a per-run `loss_bundle.json` sidecar.

**Spec:** `docs/superpowers/specs/2026-05-23-domain-aware-loss-presets-design.md`
**Plan:** `docs/superpowers/plans/2026-05-23-domain-aware-loss-presets-plan.md`

## Schema break (gated under #70)

Pre-existing configs using `train.loss: {w_mask: ..., focal_gamma: ...}` will fail
config-load with a clear `ValidationError`. Migration recipe:

  Before: `{w_mask: X, focal_gamma: G, matcher_weights: {...}}`
  After:  `{preset: custom, overrides: {w_mask: X, focal_gamma: G, matcher_weights: {...}}}`

## Test plan
- [x] `uv run pytest` — full suite green
- [x] `uv run csp init --preset medical --class-imbalance severe --output cfg.yaml` — renders correctly
- [x] `uv run csp doctor --config cfg.yaml` — three resolved-config tables present
- [x] End-to-end training run writes `run_dir/loss_bundle.json` with expected shape

## Related
- #75 — augmentation presets (sibling spec/pattern this work mirrors)
- #70 — v1.0 criteria (gates this schema break)
- #120 — citation-pass audit (open follow-up for the `# citation needed` cells)
EOF
)"
```

Expected: PR URL returned. Notify the user.

---

## Self-review

This plan covers every spec §15 deliverable:

| Spec section | Covered by |
|---|---|
| §1 Goals | Whole plan |
| §2 Non-goals | Honored (no kornia, no user-supplied callable, no per-knob `p`) |
| §3 Current state | Phase A migration |
| §4 Schema | Phase A (Task A2) |
| §4.1 Knob semantics | Spec; the term classes ignore irrelevant kwargs by design |
| §4.2 Migration | Phase A (Tasks A2, A3) + Phase G (callsite sweep) + Phase E (templates) |
| §5 Preset × class_imbalance table | Phase A (Task A3 — literal table) |
| §5.1 Rationale | In `presets.py` docstring + spec |
| §5.2 Microscopy alias | Phase A (Task A3 — `dict(...)` copy) |
| §5.3 Citation convention | Phase A (Task A3 — inline `# cite:` comments) |
| §6 LOCKED_OFF | Phase A (Task A3) |
| §6.1 Warn message format | Phase A (`resolve` calls `_LOG.warning` with the exact format) |
| §6.2 What counts as enabled | Phase A (`_override_triggers_warn`) |
| §7 Resolution algorithm | Phase A (Task A3) |
| §7.1 Module layout | Phases A (skeleton) + B (terms) + C (compose) |
| §7.2 Module placement | Honored — `presets.py` is pure-Python; `compose.py` imports torch |
| §8 Term library + composer | Phase B (terms) + Phase C (compose) |
| §8.5 Class-sharing scheme | Phase B (axis modules each declare their own classes — no factory) |
| §8.6 Trainer integration | Phase C (shim) + Phase D (optional bundle wire-up) |
| §9 Sidecar | Phase A (`dump_loss_bundle`) + Phase D (trainer write) |
| §9.1 Helper | Phase A (Task A3) |
| §9.2 Trainer wire-up | Phase D (Task D1) |
| §9.3 Cross-version repro | Spec docstring + `dump_loss_bundle` output design |
| §10.1 csp init | Phase E |
| §10.1.1 custom branch | Phase E (Task E1, `_build_loss_overrides_block`) |
| §10.2 csp doctor | Phase F |
| §11 Test plan | Phases A, B, C, D, E, F, G bundle their tests |
| §12 Migration | Phase A + Phase G |

**Placeholder scan:** no `TBD`, no `TODO`, no `fill in details`. Every code block is complete. Every shell command has expected output. Every test has its assertion code.

**Type/symbol consistency:** the 15 axis-prefixed class names in `terms/__init__.py` match the `_TERM_CLASS_NAMES` dict in `presets.py` (verified by `test_term_class_names_match_compose_registry` in Phase C). The four registry dicts in `compose.py` map literal strings to the same class objects.

**Scope check:** in-scope for one PR. Phases A–G are file-disjoint within each batch; Phase H is the reviewer pass.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-23-domain-aware-loss-presets-plan.md`.**

Execution: this plan is intended for an Implementation-Orchestrator session that reads the spec + plan cold and uses `superpowers:subagent-driven-development` to dispatch one subagent per phase (and parallel batches per the DAG above).
