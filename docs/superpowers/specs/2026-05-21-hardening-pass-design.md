# Hardening Pass: SOLID / DRY / YAGNI Sweep (issue #26)

**Status:** Draft (2026-05-21)
**Tracking issue:** #26
**Scope:** Single PR (`v0.7.0`, pre-1.0 minor — breaking changes permitted) paying down v0.x debt across config, CLI, training, eval, tracking, and model loading. The objective is twofold: (a) the YAML + CLI surface a researcher touches is small and obvious, and (b) adding a new PEFT method / eval metric / tracking backend / dataset format means one new file behind one protocol, not editing five.

**Frame:** This is a bottom-up sweep. The ordering — audit → shared primitives → seam cleanups → decompose god functions → user-surface redesign → migrate → dead-code sweep — is locked. The user explicitly chose to land everything in one PR rather than stage it across several; the spec must remain tight enough that one implementation plan can carry it without splintering.

---

## 1. Goals & Scope

Resolve issue #26 by executing a deliberate, bottom-up hardening pass that makes the codebase obey SOLID / DRY / YAGNI in the places that actually hurt today. The execution starts with a per-file audit that produces a sibling inventory document, then proceeds through shared primitives, seam cleanups, god-function decomposition, user-surface redesign, in-PR migration of all consumers (example configs, Colab notebook, RunPod scripts, README), and a dead-code sweep, ending in a `v0.7.0` release.

### 1.1 Issue absorbed and closed by this PR

| Issue | Title (short) | Disposition |
| --- | --- | --- |
| #26 | Hardening pass: SOLID/DRY/YAGNI sweep across the codebase | Closed by the merge of this PR. |

### 1.2 In scope

- **Full repo audit** producing a sibling document at `/home/justin/projects/custom-sam-peft/docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md` that, for every file under `src/`, records: (a) responsibility one-liner, (b) inbound dependencies, (c) outbound dependencies, (d) duplication detected by `mcp__token-savior__find_semantic_duplicates`, (e) any function ≥60 lines, (f) any cross-module reach-through. The inventory is created as part of executing this spec, not before it.
- **Aggressive redesign of the YAML schema and CLI surface,** with all example configs under `configs/examples/` and the Colab notebook migrated in the same PR.
- **Centralization** of device handling, dtype resolution, path layout, config loading, and error taxonomy.
- **Elimination of `if peft.method == "lora" / "qlora"` leaks** at the four known sites — `src/custom_sam_peft/train/loop.py:66`, `src/custom_sam_peft/train/trainer.py:49`, `src/custom_sam_peft/train/checkpoint.py:150`, `src/custom_sam_peft/eval/runner.py:76` — plus any others the audit surfaces.
- **Decomposition of god functions:** `load_sam31` (~150 lines, `src/custom_sam_peft/models/sam3.py:1054`), `Trainer.fit` (~121 lines, `src/custom_sam_peft/train/trainer.py:134`), `Evaluator.evaluate` (~98 lines, `src/custom_sam_peft/eval/evaluator.py:119`), `HFDataset.__getitem__` (~134 lines, `src/custom_sam_peft/data/hf.py:159`), `COCODataset.__getitem__` (~106 lines, `src/custom_sam_peft/data/coco.py:168`), `write_bundle` (~117 lines, `src/custom_sam_peft/runs/bundle.py:262`), `apply_qlora` (~124 lines, `src/custom_sam_peft/peft_adapters/qlora.py:159`), and the `_patch_*` wall in `src/custom_sam_peft/models/sam3.py`.
- **Deletion of `src/esam3/`** entirely (audit confirmed 0 source references).
- **Seam-level integration tests** added; full suite ends green on CI.
- **Version bump to `v0.7.0`** across `pyproject.toml`, `uv.lock`, and any other version-carrying manifest the audit surfaces.

### 1.3 Out of scope (explicitly deferred)

- **New PEFT methods, new eval metrics, new CLI commands, new tracking backends, new dataset formats.** This is a refactor PR; capability additions ride later PRs against the cleaner surface.
- **Multi-GPU / DDP / FSDP execution code.** No DDP code lands in this PR. Seams must be *DDP-safe* (§2), but no `torch.distributed` calls, no launchers, no rank-aware logic beyond a single-rank shape.
- **Performance optimization** unless it falls out structurally from the refactor.
- **No `migrate-config` tool or any migration tooling.** Users upgrade by editing their YAML manually against the CHANGELOG rename table. The user explicitly rejected a migrator tool — do not reintroduce it.

---

## 2. Multi-GPU-Safe Seam Discipline (preserved without code)

Even though no DDP code lands, the seams must not preclude DDP/FSDP later. This is a discipline boundary, not a code drop:

- `Trainer` holds a `device` handle injected at construction; it never assumes single-GPU process-wide state.
- Datasets return CPU tensors. The collator (or a dedicated `to_device` helper) is the single device-move site. Nothing downstream re-issues `.to(device)`.
- The `Tracker` protocol stays rank-aware in shape (e.g., carries an `is_primary: bool` or equivalent flag) even though only rank 0 ever exists today.
- Checkpoint identity uses `global_step`, not local step; paths flow through the `paths/` resolver, never string-joined inline.
- Loss / metric reductions don't bake in `world_size == 1` (e.g., averages are computed in a way that would still be correct if a future reducer divided by `world_size`).

The §10 dead-code sweep does **not** remove unused rank-awareness fields; they are seam scaffolding, not dead code.

---

## 3. Success Criteria (principles, not hard numbers)

The user explicitly preferred principles over numeric thresholds for this PR's success criteria — the audit will surface line counts and percentages, but the bar is structural, not metric.

1. **Audit inventory exists.** Every smell it records is either addressed in this PR or has a follow-up GitHub issue labeled `hardening-followup` linking back to the audit line.
2. **No `if .*\.method ==` branches in `src/` outside `src/custom_sam_peft/peft_adapters/`.** Verified by a `rg`-based static guard test (§9).
3. **Device-move sites collapse to a single canonical location** — the data collator, plus at most one runtime boundary helper. Not scattered across the call graph. Verified by a `rg`-based static guard test.
4. **`src/custom_sam_peft/models/sam3.py` substantially shrinks,** with each `_patch_*` extracted to its own file under `src/custom_sam_peft/models/_patches/`. The user explicitly approved one file per patch — do not collapse them.
5. **Every Pydantic config class is either user-facing or internally-marked as such.** A class is user-facing if the end user actually sets at least one of its fields in a real config; internal classes either live under `src/custom_sam_peft/config/_internal.py` or are explicitly marked internal in their docstring.
6. **Test suite green on CI.** New seam-level integration tests cover (a) the trainer ↔ evaluator hand-off and (b) tracker swap-in/swap-out without trainer modification.
7. **The Colab notebook runs end-to-end on the renamed schema.** Verified by the orchestrator after merge; not gated on CI inside this PR (GPU-marked tests stay GPU-marked).

---

## 4. Shared Primitives

Dependency order: `errors` → `paths` → `runtime` → `config` → `_bootstrap`. The audit ordering and the implementation plan must respect this — each layer is the substrate the next leans on.

### 4.1 `src/custom_sam_peft/config/`

- **Single entry point.** `load_config(path, overrides) -> TrainConfig | EvalConfig | ExportConfig` is the *only* config-loading function. All CLI commands, notebook helpers, and tests funnel through it. No downstream module re-parses YAML.
- **Schema collapsed to the minimum user-set surface.** Sub-configs that exist only for internal structuring move to `src/custom_sam_peft/config/_internal.py`, or are demoted to internal dataclasses constructed from the user-facing schema. The audit chooses which approach per class — both are acceptable; the spec does not pre-pick.
- **Loader responsibilities.** YAML reading, `--override key=val` merging, env-var interpolation, and resolving relative paths against the config file's directory at load time. Downstream code receives absolute paths and never has to re-resolve.
- **One validated object per command.** `TrainConfig`, `EvalConfig`, `ExportConfig` — no shape-shifting dicts in flight. Each command receives exactly the one validated object it needs.

### 4.2 `src/custom_sam_peft/paths/`

- **Single module owns the run-dir layout:** `runs/<run_id>/{checkpoints,artifacts,logs,bundle}/`, plus checkpoint paths, artifact paths, predictions paths, dataset paths.
- **Named functions only** — no opaque builders:
  - `checkpoint_path(run_dir, step)`
  - `artifact_path(run_dir, name)`
  - `predictions_path(run_dir, split)`
  - `bundle_path(run_dir)`
- **Replaces ad-hoc string joins** in `src/custom_sam_peft/runs/bundle.py`, `src/custom_sam_peft/train/checkpoint.py`, `src/custom_sam_peft/eval/runner.py`, and the CLI commands.

### 4.3 `src/custom_sam_peft/runtime/`

- **`Runtime` value object.** Fields: `device: torch.device`, `dtype: torch.dtype`, `is_primary: bool`, `world_size: int` (always `1` today; preserved as a seam per §2).
- **Dtype string resolved once.** `Runtime.from_config(...)` is the single place that maps `"bfloat16"` / `"float16"` to a `torch.dtype`. Consumers never re-resolve.
- **One `to_device(x, runtime)` helper.** The data collator becomes the single device-move site; everything downstream trusts batch tensors are on the right device.
- **`Sam3Patches` applier.** The `_patch_*` wall in `src/custom_sam_peft/models/sam3.py` becomes a `Sam3Patches` applier scoped to a `Runtime`: `Sam3Patches.apply(model, runtime)` runs all dtype-correctness patches once at model load. Each `_patch_*` moves to its own file under `src/custom_sam_peft/models/_patches/` — *one file per patch* (user explicitly approved; do not collapse).

### 4.4 `src/custom_sam_peft/errors.py`

- **One module defines the hierarchy.** `CustomSamPeftError` (base) → `ConfigError`, `DataError`, `ModelError`, `CheckpointError`, `EnvironmentError`.
- **Each error carries fix-pointing context.**
  - `ConfigError` includes the YAML field path (e.g., `data.train.path`).
  - `EnvironmentError` includes the failing precondition (HF gating, missing checkpoint, missing GPU, missing extra).
- **CLI catches the base.** `src/custom_sam_peft/cli/main.py::main()` catches `CustomSamPeftError` at the boundary and renders a single paragraph plus a "rerun with `-v` for traceback" hint.
- **Internals never catch typed errors to re-raise as `RuntimeError`.** Internals raise typed exceptions and trust callers. No defensive try/except mid call-graph.

### 4.5 `src/custom_sam_peft/_bootstrap.py`

- **Already exists; refactor so it is the *only* path that registers PEFT adapters and tracking backends, applies model patches, sets seeds, and configures logging.**
- **Every CLI command and the notebook helper call `bootstrap()` once.** Nothing else triggers these side effects.
- **This is what makes "add a new PEFT method = add one file" actually true.** The new file just needs an `@register("peft", "...")` decorator, and `_bootstrap.py` imports it. There is no other registration site to update.

---

## 5. Seam Cleanups

### 5.1 PEFT method-string leaks

The four leaks named in §1.2 (`src/custom_sam_peft/train/loop.py:66`, `src/custom_sam_peft/train/trainer.py:49`, `src/custom_sam_peft/train/checkpoint.py:150`, `src/custom_sam_peft/eval/runner.py:76`), plus any others surfaced by the audit, move behind the existing `@register("peft", ...)` factory.

The `PEFTMethod` protocol grows whatever methods the leaks needed. The audit chooses the exact final names; the candidates implied by the four current branches are:

- `recommended_optimizer() -> str`
- `qlora_aware_train_step_hook(...)` (or equivalent — name TBD by audit)
- `detect_method_from_checkpoint(ckpt) -> str`

Trainer, evaluator, and checkpoint loader receive a `PEFTMethod` instance and call protocol methods. They never branch on method name. After this section lands, the §3 success criterion ("no `if .*\.method ==` outside `peft_adapters/`") is verifiable.

> **Note:** the exact final names of the new `PEFTMethod` protocol methods are TBD by the audit — this is explicitly delegated to audit findings, not a `TODO` placeholder.

### 5.2 Tracking consolidation

The `Tracker` protocol already exists. The audit pass enforces:

- Every direct `import wandb` / `from tensorboardX import ...` outside `src/custom_sam_peft/tracking/` is removed.
- The trainer holds a `Tracker` and never knows the backend.
- `build_tracker` is the single construction site.

### 5.3 Trainer ↔ Evaluator hand-off

Define a small `EvalArtifacts` value object — fields: `checkpoint_path`, `peft_method`, `run_dir` — returned by the trainer and consumed by the evaluator.

- The evaluator stops reaching into trainer internals.
- The trainer stops knowing what metrics the evaluator computes.
- Tests treat `EvalArtifacts` as the seam (see §9).

### 5.4 CLI commands — internals (the structural half)

Each `src/custom_sam_peft/cli/<cmd>_cmd.py` becomes a thin layer: parse args → call one library function → render result.

Library functions (`run_train`, `run_eval`, `run_export`, ...) become the canonical Python API and are what the Colab notebook imports.

> **Sequencing note (user-approved split).** CLI *internals* refactor — making commands thin wrappers over `run_*` library functions — lands in this §5.4. CLI *surface* changes — renames, the new `--eval` / `--export` flags — land in §7.2 (User Surface). So CLI changes occur in two passes within the PR. This split is deliberate and the user explicitly approved it: the internals refactor enables the surface redesign cleanly. Do not collapse the two passes.

---

## 6. God-Function Decomposition

**Convention** (user-approved): private helpers stay `_`-prefixed in the same file. Promote a helper to a sibling module only when a *second* caller appears (Rule of Three). This keeps the file count from exploding while still making each helper independently testable.

Concrete decompositions:

- **`src/custom_sam_peft/models/sam3.py::load_sam31`** → `_locate_weights`, `_construct_raw_model`, `_apply_dtype`, `_apply_patches`, `_freeze_base`. `load_sam31` becomes the orchestrating shell.
- **`src/custom_sam_peft/train/trainer.py::Trainer.fit`** → `_setup_run_dir`, `_build_optimizer`, `_train_epoch`, `_eval_epoch`, `_maybe_checkpoint`.
- **`src/custom_sam_peft/eval/evaluator.py::Evaluator.evaluate`** → `_iter_predictions`, `_aggregate_metrics`, `_maybe_save_predictions`.
- **`src/custom_sam_peft/data/hf.py::HFDataset.__getitem__`** and **`src/custom_sam_peft/data/coco.py::COCODataset.__getitem__`** → `_decode_image`, `_decode_targets`, `_apply_transforms`, `_pack_example`. Each `__getitem__` becomes a four-line pipeline.
- **`src/custom_sam_peft/runs/bundle.py::write_bundle`** → `_collect_artifacts`, `_write_manifest`, `_zip_bundle`. Path construction inside `write_bundle` collapses onto the new `paths/` helpers (§4.2).
- **`src/custom_sam_peft/peft_adapters/qlora.py::apply_qlora`** → `_quantize_base`, `_inject_lora_adapters`, `_freeze_non_adapter`. `apply_qlora` orchestrates and is what `@register("peft", "qlora")` exposes.
- **The `_patch_*` wall in `src/custom_sam_peft/models/sam3.py`** → one file per patch under `src/custom_sam_peft/models/_patches/`, applied by `Sam3Patches.apply(model, runtime)` (§4.3). User explicitly approved the one-file-per-patch decomposition; do not collapse.

After this section, no source file in `src/custom_sam_peft/` should contain a function longer than ~60 lines (audit-verified).

---

## 7. User Surface Redesign (the headline payoff)

### 7.1 YAML schema

- **Required fields shrink to the actual minimum:** `data` (where to load from), `model` (base checkpoint), `peft` (method). Everything else carries a default.
- **Top-level sections stay flat and predictable:** `data`, `model`, `peft`, `train`, `eval`, `tracking`, `export`. Within each section, fields split into "commonly set" and "advanced"; advanced fields are documented as such and grouped at the bottom of each section.
- **Drop any field with zero non-test references.** The audit produces the field-use census (grep across `configs/examples/`, the Colab notebook, the test suite); fields with zero non-test references either become hardcoded defaults or get deleted outright.
- **Field names harmonize: same concept = same name everywhere.** The audit produces the canonical rename table. Common offenders to watch for (the audit may find more):
  - `lr` vs `learning_rate`
  - `batch_size` vs `train_batch_size`
  - `ckpt_dir` vs `checkpoint_dir`
  - `wandb_project` vs `tracking.wandb.project`
- **Schema-doc appendix.** A schema-doc sub-document published either alongside this spec or as an appendix to it lists every surviving field with type, default, layer (common / advanced), and YAGNI-survival rationale. The audit produces the exact contents; the planner decides whether it lives as a separate file or an appendix here. Either way it lands in the same PR.

### 7.2 CLI surface

- **Subcommand semantics.** `train` trains. `eval` evaluates. `export` exports. `run` stays as a documented **alias** for `train --eval --export` so the Colab notebook reads cleanly.
- **Composable bare flags on `train` / `eval` / `export`:** `--eval`, `--export`.
- **Order is fixed.** `train` always implies the fixed `train → eval → export` order; the flags only toggle inclusion, never reorder.
- **`init` keeps its template flag;** templates regenerate from the new schema.
- **`doctor` user surface is unchanged;** internally it reuses the new `EnvironmentError` taxonomy so its checks and failure messages stay in sync with what `run` / `train` actually surface at runtime.
- **Python API parity.** `from custom_sam_peft import run_train, run_eval, run_export, write_bundle, ...` is the same set of functions the CLI calls. The Colab notebook imports these directly.

> **User preference (load-bearing — do not relitigate).** The user explicitly chose **bare flags** (`--eval` / `--export`) over the prefixed alternatives `--with-eval` / `--with-export` and over `--then-eval` / `--then-export`. The rationale: "the dashes should catch human eyes" — the leading dashes provide sufficient visual disambiguation from the subcommand names. Do not switch to `--with-*` or `--then-*` in the spec, the plan, the implementation, or the docs.

### 7.3 Error message UX

Every user-facing error renders as four parts: one-line summary → what was expected → what was found → suggested fix.

Example:

```
ConfigError: data.train.path does not exist.
Expected: an existing directory.
Found: '/foo/bar' (does not exist).
Fix: create the directory or update data.train.path in your config.
```

- The boundary catch in `src/custom_sam_peft/cli/main.py::main()` produces this rendering.
- `-v` re-raises the traceback.
- Internals never catch typed errors just to re-raise as `RuntimeError` (re-stated from §4.4 because it is the rule that makes this UX possible).

---

## 8. Migration (in this PR)

The PR is migration-complete. There is no follow-up PR to update consumers.

- **All files under `configs/examples/`** are rewritten to the new schema.
- **The Colab notebook** (`notebooks/custom_sam_peft_train.ipynb`) is updated cell-by-cell: new CLI command names, new config field names, screenshot / output cells regenerated.
- **`notebooks/` README and the project `README.md` "Beginner — train in Colab" section** are rewritten against the new surface.
- **`cloud/runpod/` and any RunPod / GCP launch scripts** are updated.
- **`CHANGELOG.md`** gets a `v0.7.0` entry containing:
  1. The field rename table.
  2. The command flag changes.
  3. The removed fields / flags.
- **No migrator tool.** The user explicitly rejected one — do not reintroduce it under any name (`migrate-config`, `upgrade-config`, etc.).

---

## 9. Testing Strategy

### 9.1 Seam tests (replace mocks-of-internals)

- **Trainer ↔ evaluator hand-off test.** Runs both ends without mocks; asserts on `EvalArtifacts` shape and that the evaluator consumes nothing else from the trainer.
- **Tracker swap-in/swap-out test.** Parameterizes over `NoopTracker`, a fake recording tracker, and the offline `WandbTracker`; asserts the trainer makes the same protocol calls regardless of backend.
- **PEFT extensibility test (OCP proof).** Registers a stub adapter under `tests/fixtures/` via `@register("peft", "stub")`, runs a tiny `fit`, asserts the trainer accepts it with zero code changes anywhere outside the stub file.

### 9.2 Static guards in CI (just `rg` calls in dedicated tests)

These are cheap and load-bearing — they are how the §3 structural success criteria stay enforced over time:

- **No `if .*\.method ==` in `src/` outside `src/custom_sam_peft/peft_adapters/`.**
- **No `\.to\(device` outside the data collator and the `src/custom_sam_peft/runtime/` module.**
- **No string-joined `runs/.../checkpoints/` paths outside `src/custom_sam_peft/paths/`.**

### 9.3 Follow-up issues

Audit-inventory items not addressed in this PR become follow-up GitHub issues with the `hardening-followup` label (created inline if it does not exist), each linking to the specific audit-inventory line. The label name is locked — do not rename.

### 9.4 GPU-marked tests

Existing GPU-marked tests stay GPU-marked. PR #58 drained them on Colab T4; this PR does not re-litigate that policy. CI without GPU stays green via existing skip markers.

---

## 10. Dead-Code Sweep

- **Delete `src/esam3/` entirely.** The audit confirmed 0 source references.
- **Use `mcp__code-review-graph__refactor_tool`** to surface unreachable functions; delete what the audit confirms is dead.
- **No deprecation warnings, no shims.** Pre-1.0; the README already declares breaking changes.
- **"Just in case" hooks / callbacks with zero callers in `src/`** (excluding tests-of-themselves) get deleted.
- **Exception:** the §2 seam-discipline scaffolding (`is_primary`, `world_size`, etc.) is **not** dead code and is **not** removed, even though only one rank exists today.

---

## 11. Release

- **Version bump to `v0.7.0`** across `pyproject.toml`, `uv.lock`, and any other version-carrying manifest the implementation pass identifies. The discovery command is `rg -l '"?version"?\s*[:=]'` (run from the repo root, scoped to tracked files). `v0.7.0` is a pre-1.0 minor bump and is permitted to ship breaking changes per the project's README.
- **PR description shape.** Leads with the user-visible payoff:
  1. The new YAML schema diff (or a link to the rename table in `CHANGELOG.md`).
  2. The new CLI examples.
  3. The field rename table.
  4. The removed fields / flags.

  The internals refactor (shared primitives, seam cleanups, god-function decomposition, dead-code sweep) appears in the back half of the description, framed as the substrate that made the user-surface changes safe.

---

## 12. Acceptance Criteria

- [ ] **Audit inventory exists** at `/home/justin/projects/custom-sam-peft/docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md` and covers every file under `src/` per §1.2 schema (a)–(f).
- [ ] **No `if .*\.method ==` branches in `src/`** outside `src/custom_sam_peft/peft_adapters/`. Verified by the static guard test in §9.2.
- [ ] **No `\.to\(device` in `src/`** outside the data collator and `src/custom_sam_peft/runtime/`. Verified by the static guard test in §9.2.
- [ ] **No string-joined `runs/.../checkpoints/` paths** outside `src/custom_sam_peft/paths/`. Verified by the static guard test in §9.2.
- [ ] **`src/custom_sam_peft/models/sam3.py` substantially shrinks,** and every `_patch_*` lives in its own file under `src/custom_sam_peft/models/_patches/`.
- [ ] **`src/esam3/` is deleted.**
- [ ] **Each Pydantic config class** is either user-facing (has at least one field set in `configs/examples/` or the Colab notebook) or marked internal (lives under `src/custom_sam_peft/config/_internal.py` or has an explicit "internal" docstring marker).
- [ ] **`EvalArtifacts` value object exists** and is the only object the evaluator consumes from the trainer.
- [ ] **`Sam3Patches.apply(model, runtime)` exists** and is the only application site for the `_patch_*` set.
- [ ] **`CustomSamPeftError` taxonomy exists** with the five subclasses listed in §4.4; `doctor` reuses `EnvironmentError`.
- [ ] **CLI: `train --eval --export` flags exist as bare flags** (not `--with-*`, not `--then-*`). `run` is a documented alias for `train --eval --export`.
- [ ] **Trainer ↔ evaluator seam test exists** and runs without mocking trainer or evaluator internals.
- [ ] **Tracker swap-in/swap-out test exists** and parameterizes over at least `NoopTracker`, a fake recording tracker, and offline `WandbTracker`.
- [ ] **PEFT extensibility test exists,** registering a stub adapter under `tests/fixtures/` and running `fit` without modifying any code outside the stub file.
- [ ] **All files under `configs/examples/`** parse against the new schema.
- [ ] **`notebooks/custom_sam_peft_train.ipynb`** runs end-to-end against the new schema (verified by the orchestrator after merge; not required to gate CI inside this PR).
- [ ] **`cloud/runpod/` and any other launch scripts** are updated to the new CLI surface and schema.
- [ ] **`CHANGELOG.md`** gains a `v0.7.0` entry containing the field rename table, the command flag changes, and the removed fields / flags.
- [ ] **No migrator tool** (`migrate-config`, `upgrade-config`, or any equivalent) exists anywhere in the PR.
- [ ] **Version bumped to `0.7.0`** in `pyproject.toml`, `uv.lock`, and any other version-carrying manifest discovered by `rg -l '"?version"?\s*[:=]'`.
- [ ] **Test suite green on CI** (the GPU-marked subset stays GPU-marked and continues to skip on CI without GPU).
- [ ] **Audit-inventory items not addressed** in this PR each have a `hardening-followup` GitHub issue linking back to the specific audit line.

---

## 13. Out of Scope (Deferred, Tracked Elsewhere)

- **New PEFT methods, new eval metrics, new CLI commands, new tracking backends, new dataset formats.** Capability work rides separate future PRs.
- **Multi-GPU / DDP / FSDP execution code.** No `torch.distributed` calls, no launchers, no rank-aware logic beyond the seam shape from §2.
- **Performance optimization** that does not fall out structurally from the refactor.
- **Any migration tooling** (`migrate-config`, `upgrade-config`, or any other name). User explicitly rejected this; do not reintroduce.
- **Re-litigation of the GPU test policy.** The existing GPU-marker policy from `2026-05-19-gpu-test-policy-design.md` stays as written.
- **Editing archived specs / plans** under `docs/superpowers/{specs,plans}/`. Historical records remain as-is.

---

## 14. Open Questions

These items genuinely could not be resolved while writing the spec. The planner subagent should either resolve them inline or escalate to the user per the design-ambiguity ladder. None of them block the audit phase — the audit can run before any of these are decided.

1. **Exact final names for the new `PEFTMethod` protocol methods (§5.1).** The candidates implied by the four current branches are `recommended_optimizer()`, `qlora_aware_train_step_hook(...)`, and `detect_method_from_checkpoint(ckpt)`. Final names depend on what the audit finds — the audit may discover additional leaks that need their own protocol methods, or merge two of the candidates. *Recommendation:* let the audit finalize, then commit names in the implementation plan.
2. **Internal sub-configs: keep as Pydantic in `_internal.py`, or demote to dataclasses (§4.1)?** Both are acceptable. *Recommendation:* per-class decision by the audit — Pydantic for any class that benefits from validation (e.g., enum fields, constrained ints), dataclass otherwise.
3. **Schema-doc appendix: separate file or inline in this spec (§7.1)?** Either is fine; the audit produces the contents. *Recommendation:* separate file under `docs/` (not under `docs/superpowers/`) so it can be referenced from the README without dragging readers into the design-history tree.
4. **`hardening-followup` label creation.** The label may not exist yet. *Recommendation:* the orchestrator runs `gh label list` before creating follow-up issues and creates the label inline (`gh label create hardening-followup --description "Audit-surfaced items deferred from the hardening pass" --color <hex>`) if missing. Label name is locked at `hardening-followup`.
5. **Coverage of the audit's "cross-module reach-through" check (§1.2 item f).** "Reach-through" is intuitive but not crisply defined here. *Recommendation:* the audit uses a working definition of "a module imports a private symbol (`_`-prefixed) from another module, or accesses an attribute documented as internal." The planner may tighten this further if the audit subagent needs a stricter rule.
