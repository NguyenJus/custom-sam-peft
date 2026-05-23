# Hardening Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-21-hardening-pass-design.md`](../specs/2026-05-21-hardening-pass-design.md)
**Tracking issue:** [#26](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/26) — *Hardening pass: SOLID/DRY/YAGNI sweep across the codebase*
**Branch:** `spec/hardening-pass`
**Worktree:** `/home/justin/projects/custom-sam-peft/.worktrees/spec-hardening-pass/`

**Goal:** Land issue #26 in a single PR that pays down v0.x debt across config, CLI, training, eval, tracking, and model loading — producing a small, obvious user-facing surface and seam-isolated internals — then tag `v0.7.0`.

**Architecture:** Bottom-up sweep. Order is locked: per-file **audit** → **shared primitives** (`errors` → `paths` → `runtime` → `config` → `_bootstrap`) → **seam cleanups** → **god-function decomposition** → **user-surface redesign** → **in-PR migration of consumers** → **dead-code sweep** → **release**. One branch, one PR, audit-first because every downstream task consumes the audit's per-file census, rename table, and protocol-method-name decisions.

**Tech Stack:** Python 3.12, PyTorch 2.4+, Pydantic v2, Typer CLI, pytest, `ripgrep` for static guards, `mcp__code-review-graph__*` MCP tools for audit and refactor surfacing, `mcp__token-savior__find_semantic_duplicates` for duplication detection.

---

## Planner-resolved decisions (locked in this plan)

The spec §14 lists five open questions. Resolved here so subagents do not relitigate:

- **OQ1 — `PEFTMethod` protocol method names (§5.1).** Resolution: the Task 1 audit must include a "PEFT protocol surface" section that lists every method-string branch and proposes a protocol method per branch (or merges related branches). Final names are committed in Task 4.1 (PEFT abstraction extraction) using the audit's recommendation verbatim. The three candidates in the spec (`recommended_optimizer()`, `qlora_aware_train_step_hook(...)`, `detect_method_from_checkpoint(ckpt)`) are starting points the audit may rename.
- **OQ2 — Internal sub-configs (Pydantic vs. dataclass) (§4.1).** Resolution: **default is dataclass for internal-only sub-configs.** Promote to Pydantic only when the audit shows the class has (a) enum fields, (b) constrained ints/floats with bounds, OR (c) three+ end-user-set fields. The audit produces a per-class decision; Task 5.1 (config restructuring) enacts it.
- **OQ3 — Schema-doc appendix location (§7.1).** Resolution: **standalone `docs/config-schema.md`** (NOT under `docs/superpowers/`). Linked from `README.md` "Configuration" section. Lands in Task 7.1.
- **OQ4 — `hardening-followup` label creation.** Resolution: explicit Task 0.4 verifies/creates the label inline using `gh label list` / `gh label create` before any audit-deferred items become issues.
- **OQ5 — "Cross-module reach-through" definition (§1.2 item f).** Resolution: fixed as **"an import from one `src/custom_sam_peft/<module>/` subdir into a different `src/custom_sam_peft/<module>/` subdir that bypasses a documented seam"** — for example, `eval/` importing from `train/` internals, or `trainer.py` importing from `eval/` internals. Imports from `runtime/`, `paths/`, `errors.py`, `config/`, `_bootstrap.py`, `_registry.py`, and `peft_adapters/__init__.py` (the registered-factory entry point) are documented seams and are exempt. The audit applies this definition.

---

## File Structure

This is the post-PR target. Audit (Task 1) may reveal additional sub-modules; if so, amend.

**New files (created during this PR):**

```
src/custom_sam_peft/errors.py                          # CustomSamPeftError + 5 subclasses
src/custom_sam_peft/paths/__init__.py                  # named-function path API
src/custom_sam_peft/paths/_layout.py                   # run-dir layout impl
src/custom_sam_peft/runtime/__init__.py                # Runtime, to_device, Sam3Patches
src/custom_sam_peft/runtime/_runtime.py                # Runtime dataclass + from_config
src/custom_sam_peft/runtime/_device.py                 # to_device helper
src/custom_sam_peft/runtime/_patches.py                # Sam3Patches applier
src/custom_sam_peft/config/_internal.py                # internal-only sub-configs
src/custom_sam_peft/models/_patches/__init__.py        # one file per _patch_*
src/custom_sam_peft/models/_patches/<patch_name>.py    # one per existing _patch_* function
src/custom_sam_peft/eval/_artifacts.py                 # EvalArtifacts dataclass

tests/unit/test_errors.py
tests/unit/test_paths.py
tests/unit/test_runtime.py
tests/unit/test_sam3_patches_applier.py
tests/unit/test_eval_artifacts.py
tests/unit/test_static_guards.py                       # rg-based guards from spec §9.2
tests/integration/test_trainer_evaluator_seam.py
tests/integration/test_tracker_swap.py
tests/integration/test_peft_extensibility.py
tests/fixtures/stub_peft_adapter.py                    # used by test_peft_extensibility

docs/config-schema.md                                  # schema appendix (OQ3)
docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md   # audit output (Task 1)
```

**Modified files (representative — audit will surface more):**

```
src/custom_sam_peft/models/sam3.py                     # decompose load_sam31, extract _patch_*
src/custom_sam_peft/train/trainer.py                   # decompose fit, accept Runtime
src/custom_sam_peft/train/loop.py                      # remove peft.method branches
src/custom_sam_peft/train/checkpoint.py                # use paths/, remove method branches
src/custom_sam_peft/eval/evaluator.py                  # decompose evaluate, EvalArtifacts seam
src/custom_sam_peft/eval/runner.py                     # remove method branches, use paths/
src/custom_sam_peft/data/coco.py                       # decompose __getitem__
src/custom_sam_peft/data/hf.py                         # decompose __getitem__
src/custom_sam_peft/data/collate.py                    # single to_device site
src/custom_sam_peft/runs/bundle.py                     # decompose write_bundle, use paths/
src/custom_sam_peft/peft_adapters/__init__.py          # protocol grows
src/custom_sam_peft/peft_adapters/lora.py              # implement new protocol methods
src/custom_sam_peft/peft_adapters/qlora.py             # decompose apply_qlora + implement protocol
src/custom_sam_peft/cli/main.py                        # catch CustomSamPeftError, render UX
src/custom_sam_peft/cli/{train,eval,export,run,doctor,init}_cmd.py
                                                       # thin wrappers + new --eval/--export flags
src/custom_sam_peft/_bootstrap.py                      # sole registration/patch-apply path
src/custom_sam_peft/config/                            # restructure per audit
configs/examples/*.yaml                                 # rewritten to new schema
notebooks/custom_sam_peft_train.ipynb                  # migrated to new schema + CLI
notebooks/README.md
README.md
cloud/runpod/*                                          # updated to new CLI
CHANGELOG.md                                            # v0.7.0 entry
pyproject.toml                                          # version 0.7.0
uv.lock                                                 # version 0.7.0
```

---

## Parallelization opportunities

- **Task 1 (audit) is the gate** — every later task consumes it. Serial.
- **Task 2 (shared primitives)** has fixed internal order: `errors` → `paths` → `runtime`. `runtime` can be split (Runtime / to_device / Sam3Patches) into parallel subtasks once the audit has run, but each one mutates `src/custom_sam_peft/runtime/`, so they are file-disjoint only at the sub-module level — keep serial unless dispatcher subdivides cleanly.
- **Task 4 (seam cleanups)** subtasks 4.1 (PEFT abstraction), 4.2 (tracking), 4.3 (EvalArtifacts), 4.4 (CLI internals) touch disjoint files and can run in parallel **after** Task 2 lands.
- **Task 5 (god-function decomposition)** subtasks 5.1 (`load_sam31`), 5.2 (`Trainer.fit`), 5.3 (`Evaluator.evaluate`), 5.4 (dataset `__getitem__`s), 5.5 (`write_bundle`), 5.6 (`apply_qlora`), 5.7 (`_patch_*` extraction) touch disjoint files and parallelize cleanly **after** Tasks 2 and 4. Note: 5.7 depends on Task 2's `runtime/_patches.py` Sam3Patches applier.
- **Task 6 (user-surface redesign)** must wait for Task 5 (you can't rename schema fields cleanly while functions are still mid-decomposition). 6.1 (YAML schema) and 6.2 (CLI surface) can run in parallel; 6.3 (error UX) waits for 6.1/6.2.
- **Task 7 (consumer migration)** parallelizes by consumer: 7.1 (configs/examples), 7.2 (notebook), 7.3 (cloud/runpod scripts), 7.4 (README + notebooks/README) — all run after Task 6.
- **Tasks 8 (dead-code sweep) and 9 (release)** are serial and last.

```
Task 0 (pre-flight)
  → Task 1 (audit) [SERIAL — GATE]
    → Task 2 (shared primitives) [SERIAL — errors → paths → runtime → config → _bootstrap]
      → Task 3 (static-guard test scaffold)
        → Task 4 (seam cleanups) [PARALLEL subtasks 4.1–4.4]
          → Task 5 (god-function decomposition) [PARALLEL subtasks 5.1–5.7]
            → Task 6 (user surface) [6.1, 6.2 parallel → 6.3]
              → Task 7 (consumer migration) [PARALLEL subtasks 7.1–7.4]
                → Task 8 (dead-code sweep)
                  → Task 9 (release: version bump, CHANGELOG, follow-up issues, PR)
```

---

## Pre-flight

- [ ] **Step 0.1: Confirm worktree and branch**

```bash
pwd
git rev-parse --abbrev-ref HEAD
```
Expected: working directory ends with `/.worktrees/spec-hardening-pass`; branch `spec/hardening-pass`.

- [ ] **Step 0.2: Confirm spec + plan are committed and tree clean**

```bash
git log --oneline -5
git status
```
Expected: recent commits include the spec (`2026-05-21-hardening-pass-design.md`) and this plan; `git status` clean.

- [ ] **Step 0.3: Confirm v0.6.x baseline**

```bash
grep -E '^version' pyproject.toml
```
Expected: `version = "0.6.x"` (currently `0.6.1`; spec narrative says v0.6.0 just shipped — both are pre-v0.7.0 baselines).

- [ ] **Step 0.4: Verify (or create) the `hardening-followup` label (OQ4)**

```bash
gh label list --limit 200 | grep -E '^hardening-followup\b' || gh label create hardening-followup \
  --description "Audit-surfaced items deferred from the hardening pass" \
  --color BFD4F2
```
Expected: either the label already exists, or `gh label create` returns success. The label name is **locked at `hardening-followup`** — do not rename.

- [ ] **Step 0.5: No commit** — pre-flight is read/check-only.

---

## Task 1: Audit — per-file census + rename table + protocol decisions [GATE]

**Goal:** Produce `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md` containing every input every downstream task needs. This is the single gate that unblocks Tasks 2–8.

**Subagent:** dispatch one audit subagent (sonnet/high). Audit is mostly tool-driven (MCP graph queries + grep), not synthesis-heavy; opus reserved for re-dispatch if the brief is thin.

**Tools the subagent uses:**
- `mcp__code-review-graph__build_or_update_graph_tool` (or `get_architecture_overview_tool`) for the structural skeleton.
- `mcp__code-review-graph__find_large_functions_tool` for the ≥60-line census.
- `mcp__token-savior__find_semantic_duplicates` for duplication detection.
- `mcp__code-review-graph__get_hub_nodes_tool` / `get_bridge_nodes_tool` to surface seam violations.
- `rg` for every method-string leak and `.to(device)` site.

### Files

- Create: `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md`

### Steps

- [ ] **Step 1.1: Dispatch the audit subagent**

Prompt template (literal — orchestrator dispatches this verbatim):

````text
You are the hardening-pass audit subagent for issue #26 in the
custom-sam-peft repo.

Spec authority:
docs/superpowers/specs/2026-05-21-hardening-pass-design.md (§1, §3.1,
§5.1, §7.1, §10, §14).

Produce ONE markdown file at:
docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md

The file MUST contain the following sections, in this order:

## Section A — Per-file census
For EVERY file under src/custom_sam_peft/ (recursively, all .py files
including tests-of-themselves but NOT tests/), one row in a table with
columns:
  | path | responsibility (one-liner) | inbound deps | outbound deps |
  | duplication notes | ≥60-line functions | cross-module reach-through |

- "responsibility" = one sentence; no marketing language.
- "inbound deps" = which modules import from this file (list module
  prefixes like train, eval, data, cli, runtime, _bootstrap; "—" if
  none).
- "outbound deps" = same but for what this file imports.
- "duplication notes" = output from
  mcp__token-savior__find_semantic_duplicates, summarized. List each
  duplicate cluster with the cluster's sibling files. "—" if none.
- "≥60-line functions" = list each function name + line count + start
  line. Run mcp__code-review-graph__find_large_functions_tool with
  min_lines=60. "—" if none.
- "cross-module reach-through" = working definition (planner OQ5): an
  import from one src/custom_sam_peft/<module>/ subdir into a different
  src/custom_sam_peft/<module>/ subdir that bypasses a documented seam.
  Documented seams: runtime/, paths/, errors.py, config/, _bootstrap.py,
  _registry.py, peft_adapters/__init__.py. List violating imports as
  "<src_file>:<line> imports <symbol> from <dst_file>". "—" if none.

## Section B — PEFT method-string leak inventory
List every grep hit for `\.method ==` and `\.method !=` and similar in
src/custom_sam_peft/ outside src/custom_sam_peft/peft_adapters/.
Use: `rg -n '\.method\s*[=!]=' src/custom_sam_peft/ --type py` and
broader patterns as needed. For each hit, propose the PEFTMethod
protocol method that should replace it. Group related hits into the
same protocol method when the conditional logic is the same shape.

End the section with a **proposed PEFTMethod protocol surface**:
the final list of method names, signatures, and one-line docstrings.
The three starting candidates from the spec are:
- recommended_optimizer() -> str
- qlora_aware_train_step_hook(...)
- detect_method_from_checkpoint(ckpt) -> str
Rename or merge them as the audit warrants. THE NAMES YOU COMMIT HERE
ARE THE NAMES THE PLAN WILL USE — do not leave them TBD.

## Section C — Device-move site inventory
Every grep hit for `\.to\(device` and `\.to\(self\.device` and
`\.cuda\(` in src/custom_sam_peft/. For each hit, note whether it
should stay (collator), move into runtime/, or be deleted (caller
trusts batches are already on-device). The output is the contract
Task 4 will enforce.

## Section D — Path-construction inventory
Every grep hit for string-joined `runs/.../checkpoints/`,
`os.path.join.*checkpoints`, `Path.*checkpoints` outside the proposed
src/custom_sam_peft/paths/ module. For each, note which paths/
function should replace it (checkpoint_path / artifact_path /
predictions_path / bundle_path).

## Section E — Config field-use census
For every field in every Pydantic config class under
src/custom_sam_peft/config/, list:
  | class.field | grep hits in configs/examples/ |
  | grep hits in notebooks/ | grep hits in tests/ |
  | non-test grep hits in src/ | YAGNI verdict |
YAGNI verdicts: keep (commonly set), keep-advanced (rarely set but
real), demote (no non-test references — make it a hardcoded internal
default), or delete (no references anywhere).

## Section F — Field rename table
The canonical rename table for v0.7.0. Columns:
  | old name | new name | rationale |
Common offenders from the spec to watch for (audit may add more):
- lr / learning_rate
- batch_size / train_batch_size
- ckpt_dir / checkpoint_dir
- wandb_project / tracking.wandb.project
Each rename must have a one-line rationale ("harmonize with peer
field X" / "match common ML convention").

## Section G — Pydantic-vs-dataclass per internal sub-config (OQ2)
For each sub-config class the audit recommends moving to
config/_internal.py (or marking internal in docstring), choose:
- Pydantic if it has enum fields, constrained ints/floats with bounds,
  OR three+ end-user-set fields.
- dataclass otherwise.
Default to dataclass per planner decision OQ2.

## Section H — Cross-module reach-through findings
The grep+graph output for Section A's last column, consolidated into
one list with proposed fixes ("route through paths/" / "extract X to
runtime/" / "extract EvalArtifacts seam").

## Section I — Dead-code candidates
Output from mcp__code-review-graph__refactor_tool / find_dead_code,
listing functions/classes/files with zero src/ callers (excluding
tests-of-themselves). Mark which ones are documented §2 seam
scaffolding (is_primary, world_size fields) and MUST be retained
even though unused.

## Section J — Items deferred to follow-up issues
Anything the audit surfaces that is too large to fit in this PR.
Each item gets a one-line description + a proposed GitHub issue title
+ a proposed `hardening-followup` label. Task 9 of the plan opens
these issues.

The file is committed as part of this PR.

Return the absolute path of the file you wrote and a one-paragraph
summary of the most consequential findings (so the orchestrator can
brief downstream subagents).
````

- [ ] **Step 1.2: Verify the audit inventory**

Acceptance criteria — the orchestrator checks each before unblocking Task 2:

1. The file exists at `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md`.
2. Section A has a row for every `.py` file under `src/custom_sam_peft/`. Cross-check:

   ```bash
   find src/custom_sam_peft -name '*.py' | wc -l
   ```
   The Section A row count should match within ±1 (off-by-one acceptable for `__init__.py` aggregation).

3. Section B ends with a finalized `PEFTMethod` protocol surface (named methods + signatures, **no `TBD`**).
4. Section F field rename table has at least the four common offenders from the spec (lr, batch_size, ckpt_dir, wandb_project) or an explicit "audit found this field is not used; not renamed" line per offender.
5. Section G assigns Pydantic-or-dataclass to each internal sub-config.
6. Section J lists items deferred to follow-up issues with proposed titles.

If any criterion fails, re-dispatch the audit subagent with feedback naming the failure. Max two re-dispatches before halting and surfacing the gap to the user.

- [ ] **Step 1.3: Commit the audit inventory**

```bash
git add docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md
git commit -m "audit: inventory for hardening pass (issue #26)"
```

- [ ] **Step 1.4: Brief the orchestrator session**

Read the audit's "most consequential findings" paragraph (from the subagent's return value) into the orchestrator session transcript. Downstream subagents will receive both the spec, this plan, AND the relevant audit-inventory sections by reference.

---

## Task 2: Shared primitives [SERIAL — errors → paths → runtime → config → _bootstrap]

The dependency order is locked. Each subtask is small enough to land in one commit.

### Task 2.1: `src/custom_sam_peft/errors.py` — error taxonomy

**Files:**
- Create: `src/custom_sam_peft/errors.py`
- Create: `tests/unit/test_errors.py`

- [ ] **Step 2.1.1: Write the failing tests**

```python
# tests/unit/test_errors.py
import pytest
from custom_sam_peft.errors import (
    CustomSamPeftError,
    ConfigError,
    DataError,
    ModelError,
    CheckpointError,
    EnvironmentError as CSPEnvironmentError,
)


def test_base_class_exists():
    assert issubclass(CustomSamPeftError, Exception)


@pytest.mark.parametrize(
    "subclass",
    [ConfigError, DataError, ModelError, CheckpointError, CSPEnvironmentError],
)
def test_subclasses_inherit_base(subclass):
    assert issubclass(subclass, CustomSamPeftError)


def test_config_error_carries_field_path():
    err = ConfigError("bad value", field_path="data.train.path")
    assert err.field_path == "data.train.path"
    assert "data.train.path" in str(err)


def test_environment_error_carries_precondition():
    err = CSPEnvironmentError("missing checkpoint", precondition="checkpoint_present")
    assert err.precondition == "checkpoint_present"
    assert "checkpoint_present" in str(err)


def test_subclasses_can_be_caught_at_base():
    with pytest.raises(CustomSamPeftError):
        raise ConfigError("x", field_path="a.b")
```

- [ ] **Step 2.1.2: Run tests to confirm failure**

```bash
pytest tests/unit/test_errors.py -v
```
Expected: ImportError / ModuleNotFoundError on `custom_sam_peft.errors`.

- [ ] **Step 2.1.3: Implement `src/custom_sam_peft/errors.py`**

```python
"""Exception taxonomy for custom_sam_peft.

The CLI boundary (cli/main.py::main) catches CustomSamPeftError and
renders a user-facing four-part message. Internals raise typed
exceptions and never catch-and-re-raise as RuntimeError mid call-graph.
"""

from __future__ import annotations


class CustomSamPeftError(Exception):
    """Base class for all user-facing errors raised by this package."""


class ConfigError(CustomSamPeftError):
    """Raised when a config value is missing, malformed, or invalid."""

    def __init__(self, message: str, *, field_path: str) -> None:
        super().__init__(f"{message} (field: {field_path})")
        self.field_path = field_path


class DataError(CustomSamPeftError):
    """Raised for dataset-loading or example-decoding failures."""


class ModelError(CustomSamPeftError):
    """Raised for model construction, patch-application, or adapter failures."""


class CheckpointError(CustomSamPeftError):
    """Raised for checkpoint read/write or resume-state mismatches."""


class EnvironmentError(CustomSamPeftError):
    """Raised when a runtime precondition fails (HF gating, missing GPU, missing extra)."""

    def __init__(self, message: str, *, precondition: str) -> None:
        super().__init__(f"{message} (precondition: {precondition})")
        self.precondition = precondition
```

- [ ] **Step 2.1.4: Run tests to confirm pass**

```bash
pytest tests/unit/test_errors.py -v
```
Expected: all five tests PASS.

- [ ] **Step 2.1.5: Commit**

```bash
git add src/custom_sam_peft/errors.py tests/unit/test_errors.py
git commit -m "feat(errors): add CustomSamPeftError taxonomy"
```

### Task 2.2: `src/custom_sam_peft/paths/` — run-dir layout

**Files:**
- Create: `src/custom_sam_peft/paths/__init__.py`
- Create: `src/custom_sam_peft/paths/_layout.py`
- Create: `tests/unit/test_paths.py`

- [ ] **Step 2.2.1: Write the failing tests**

```python
# tests/unit/test_paths.py
from pathlib import Path
import pytest

from custom_sam_peft.paths import (
    checkpoint_path,
    artifact_path,
    predictions_path,
    bundle_path,
)


def test_checkpoint_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = checkpoint_path(run_dir, step=42)
    assert p.parent == run_dir / "checkpoints"
    assert "42" in p.name
    assert isinstance(p, Path)


def test_artifact_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = artifact_path(run_dir, name="metrics.json")
    assert p == run_dir / "artifacts" / "metrics.json"


def test_predictions_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = predictions_path(run_dir, split="val")
    assert p.parent == run_dir / "artifacts"
    assert "val" in p.name


def test_bundle_path(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    p = bundle_path(run_dir)
    assert p.parent == run_dir / "bundle"


def test_run_dir_layout_constants_exposed():
    from custom_sam_peft.paths import (
        CHECKPOINTS_SUBDIR,
        ARTIFACTS_SUBDIR,
        LOGS_SUBDIR,
        BUNDLE_SUBDIR,
    )

    assert CHECKPOINTS_SUBDIR == "checkpoints"
    assert ARTIFACTS_SUBDIR == "artifacts"
    assert LOGS_SUBDIR == "logs"
    assert BUNDLE_SUBDIR == "bundle"
```

- [ ] **Step 2.2.2: Run tests to confirm failure**

```bash
pytest tests/unit/test_paths.py -v
```
Expected: ImportError on `custom_sam_peft.paths`.

- [ ] **Step 2.2.3: Implement `src/custom_sam_peft/paths/_layout.py`**

```python
"""Single source of truth for run-directory layout.

Layout:
    runs/<run_id>/
        checkpoints/
        artifacts/
        logs/
        bundle/

Never string-join checkpoint paths anywhere else. The §9.2 static
guard test enforces this.
"""

from __future__ import annotations
from pathlib import Path

CHECKPOINTS_SUBDIR = "checkpoints"
ARTIFACTS_SUBDIR = "artifacts"
LOGS_SUBDIR = "logs"
BUNDLE_SUBDIR = "bundle"


def checkpoint_path(run_dir: Path, *, step: int) -> Path:
    """Return the canonical path for the checkpoint at the given global step."""
    return run_dir / CHECKPOINTS_SUBDIR / f"step_{step:08d}.pt"


def artifact_path(run_dir: Path, *, name: str) -> Path:
    """Return the path for a named artifact (metrics.json, schema.json, ...)."""
    return run_dir / ARTIFACTS_SUBDIR / name


def predictions_path(run_dir: Path, *, split: str) -> Path:
    """Return the path for serialized predictions for a given split."""
    return run_dir / ARTIFACTS_SUBDIR / f"predictions_{split}.jsonl"


def bundle_path(run_dir: Path) -> Path:
    """Return the path for the exported run bundle (zip)."""
    return run_dir / BUNDLE_SUBDIR / "bundle.zip"
```

- [ ] **Step 2.2.4: Implement `src/custom_sam_peft/paths/__init__.py`**

```python
"""Run-dir path API. Single seam — do not string-join paths elsewhere."""

from custom_sam_peft.paths._layout import (
    CHECKPOINTS_SUBDIR,
    ARTIFACTS_SUBDIR,
    LOGS_SUBDIR,
    BUNDLE_SUBDIR,
    checkpoint_path,
    artifact_path,
    predictions_path,
    bundle_path,
)

__all__ = [
    "CHECKPOINTS_SUBDIR",
    "ARTIFACTS_SUBDIR",
    "LOGS_SUBDIR",
    "BUNDLE_SUBDIR",
    "checkpoint_path",
    "artifact_path",
    "predictions_path",
    "bundle_path",
]
```

- [ ] **Step 2.2.5: Run tests to confirm pass**

```bash
pytest tests/unit/test_paths.py -v
```
Expected: all five tests PASS.

- [ ] **Step 2.2.6: Commit**

```bash
git add src/custom_sam_peft/paths/ tests/unit/test_paths.py
git commit -m "feat(paths): add named-function path API for run-dir layout"
```

### Task 2.3: `src/custom_sam_peft/runtime/` — Runtime, to_device, Sam3Patches

**Files:**
- Create: `src/custom_sam_peft/runtime/__init__.py`
- Create: `src/custom_sam_peft/runtime/_runtime.py`
- Create: `src/custom_sam_peft/runtime/_device.py`
- Create: `src/custom_sam_peft/runtime/_patches.py`
- Create: `tests/unit/test_runtime.py`
- Create: `tests/unit/test_sam3_patches_applier.py`

- [ ] **Step 2.3.1: Write the failing tests for Runtime + to_device**

```python
# tests/unit/test_runtime.py
import torch
import pytest

from custom_sam_peft.runtime import Runtime, to_device


def test_runtime_fields_default_world_size_1():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    assert rt.world_size == 1
    assert rt.is_primary is True


def test_runtime_from_config_resolves_bfloat16():
    rt = Runtime.from_config(device="cpu", dtype="bfloat16")
    assert rt.dtype is torch.bfloat16


def test_runtime_from_config_resolves_float16():
    rt = Runtime.from_config(device="cpu", dtype="float16")
    assert rt.dtype is torch.float16


def test_runtime_from_config_rejects_unknown_dtype():
    from custom_sam_peft.errors import ConfigError

    with pytest.raises(ConfigError):
        Runtime.from_config(device="cpu", dtype="quadruple")


def test_to_device_moves_tensor():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    x = torch.zeros(3)
    y = to_device(x, rt)
    assert y.device == torch.device("cpu")


def test_to_device_recurses_into_dict():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    batch = {"img": torch.zeros(3), "label": torch.ones(2)}
    out = to_device(batch, rt)
    assert out["img"].device == torch.device("cpu")
    assert out["label"].device == torch.device("cpu")


def test_to_device_recurses_into_list():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    batch = [torch.zeros(3), torch.ones(2)]
    out = to_device(batch, rt)
    assert out[0].device == torch.device("cpu")


def test_to_device_passes_through_non_tensor():
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    assert to_device("hello", rt) == "hello"
    assert to_device(42, rt) == 42
```

- [ ] **Step 2.3.2: Run tests to confirm failure**

```bash
pytest tests/unit/test_runtime.py -v
```
Expected: ImportError.

- [ ] **Step 2.3.3: Implement `src/custom_sam_peft/runtime/_runtime.py`**

```python
"""Runtime value object — single source of device + dtype truth."""

from __future__ import annotations
from dataclasses import dataclass, field
import torch

from custom_sam_peft.errors import ConfigError

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Runtime:
    """Carries device, dtype, and rank-awareness fields.

    `is_primary` and `world_size` are §2 seam scaffolding: they always
    have values (True / 1) today but exist so that future DDP / FSDP
    work has somewhere to plumb rank info without touching every call
    site. The §10 dead-code sweep MUST NOT remove them.
    """

    device: torch.device
    dtype: torch.dtype
    is_primary: bool = True
    world_size: int = 1

    @classmethod
    def from_config(cls, *, device: str, dtype: str) -> "Runtime":
        """Resolve device/dtype strings to torch types once.

        Downstream code receives a Runtime and never re-parses these
        strings.
        """
        try:
            resolved_dtype = _DTYPE_MAP[dtype.lower()]
        except KeyError as e:
            raise ConfigError(
                f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_MAP)}",
                field_path="runtime.dtype",
            ) from e
        return cls(device=torch.device(device), dtype=resolved_dtype)
```

- [ ] **Step 2.3.4: Implement `src/custom_sam_peft/runtime/_device.py`**

```python
"""Single device-move helper. The data collator is the ONLY caller."""

from __future__ import annotations
from typing import Any
import torch

from custom_sam_peft.runtime._runtime import Runtime


def to_device(obj: Any, runtime: Runtime) -> Any:
    """Recursively move tensors in `obj` onto `runtime.device`.

    The §9.2 static guard test enforces that this is the only place
    `.to(device)` runs outside the runtime/ module itself.
    """
    if torch.is_tensor(obj):
        return obj.to(runtime.device)
    if isinstance(obj, dict):
        return {k: to_device(v, runtime) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, runtime) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj
```

- [ ] **Step 2.3.5: Implement `src/custom_sam_peft/runtime/_patches.py` (skeleton)**

The full applier wires up once Task 5.7 extracts the `_patch_*` functions. For now, provide the import-and-apply scaffold.

```python
"""Sam3Patches applier — single application site for all dtype patches.

Each `_patch_*` from src/custom_sam_peft/models/sam3.py is moved to its
own file under src/custom_sam_peft/models/_patches/ in Task 5.7. This
applier imports them all and runs them in deterministic order.
"""

from __future__ import annotations
from typing import Any

from custom_sam_peft.runtime._runtime import Runtime


class Sam3Patches:
    """Aggregates and applies every dtype-correctness patch to a SAM-3 model.

    Usage:
        Sam3Patches.apply(model, runtime)

    This is called exactly once per model-load, from
    models.sam3.load_sam31's `_apply_patches` step (Task 5.1).
    """

    @staticmethod
    def apply(model: Any, runtime: Runtime) -> None:
        # Task 5.7 populates this with imports from models/_patches/.
        # Order is deterministic by file name (sorted).
        from custom_sam_peft.models._patches import _ALL_PATCHES

        for patch in _ALL_PATCHES:
            patch(model, runtime)
```

- [ ] **Step 2.3.6: Implement `src/custom_sam_peft/runtime/__init__.py`**

```python
"""Runtime API. Single seam for device + dtype + rank-awareness."""

from custom_sam_peft.runtime._runtime import Runtime
from custom_sam_peft.runtime._device import to_device
from custom_sam_peft.runtime._patches import Sam3Patches

__all__ = ["Runtime", "to_device", "Sam3Patches"]
```

- [ ] **Step 2.3.7: Write the Sam3Patches placeholder test**

```python
# tests/unit/test_sam3_patches_applier.py
"""Sam3Patches placeholder test — populated by Task 5.7."""
from custom_sam_peft.runtime import Sam3Patches


def test_sam3_patches_class_exists():
    assert hasattr(Sam3Patches, "apply")
```

A richer test is written in Task 5.7 once `_ALL_PATCHES` is populated.

- [ ] **Step 2.3.8: Run tests to confirm pass**

```bash
pytest tests/unit/test_runtime.py tests/unit/test_sam3_patches_applier.py -v
```
Expected: all PASS. Note: `Sam3Patches.apply` cannot be called yet — `_ALL_PATCHES` is empty until Task 5.7. That's intentional.

- [ ] **Step 2.3.9: Stub `src/custom_sam_peft/models/_patches/__init__.py` so the import resolves**

```python
"""Per-patch modules under here. Populated by Task 5.7."""
_ALL_PATCHES: list = []
```

(Create the directory and `__init__.py` so the import in `_patches.py` doesn't fail.)

- [ ] **Step 2.3.10: Commit**

```bash
git add src/custom_sam_peft/runtime/ src/custom_sam_peft/models/_patches/__init__.py \
        tests/unit/test_runtime.py tests/unit/test_sam3_patches_applier.py
git commit -m "feat(runtime): add Runtime, to_device, Sam3Patches scaffolding"
```

### Task 2.4: `src/custom_sam_peft/config/` — restructure per audit

Per audit Sections E + F + G. This is the single subtask that benefits MOST from concrete audit findings — do not start it without Task 1 in hand.

**Files (representative; audit may add/remove):**
- Create: `src/custom_sam_peft/config/_internal.py`
- Modify: `src/custom_sam_peft/config/*.py` (existing schema files)
- Modify: `tests/unit/test_config_schema.py`
- Modify: `tests/unit/test_config_loader.py`

- [ ] **Step 2.4.1: Read audit Sections E, F, G**

The implementer subagent receives the audit-inventory file path as input. Read Sections E (field-use census), F (rename table), G (Pydantic-vs-dataclass per internal class).

- [ ] **Step 2.4.2: Apply Section E verdicts**

For each field with verdict:
- `keep`: leave as-is (still user-facing).
- `keep-advanced`: leave as-is but tag in the schema-doc appendix (Task 7.1).
- `demote`: remove from the user-facing schema, hardcode as default in `_internal.py`.
- `delete`: remove entirely (both schema and any references).

- [ ] **Step 2.4.3: Apply Section F renames**

For each `(old_name, new_name)` in the rename table: rename in `src/custom_sam_peft/config/`, update example configs (Task 7.1 will redo them fully — but ensure tests don't break here), update the test suite.

Use `rg -l <old_name>` to find every caller before renaming. Update them in the same commit.

- [ ] **Step 2.4.4: Apply Section G internal-class decisions**

Each class marked "internal" in the audit either:
- Moves to `src/custom_sam_peft/config/_internal.py`, OR
- Keeps its existing location with an explicit "internal" docstring marker (the first line of the class docstring contains the literal text `Internal config — not user-set.`).

Per OQ2: dataclass by default, Pydantic only when Section G specifies it.

- [ ] **Step 2.4.5: Implement `load_config` single entry point**

Per spec §4.1: ensure `src/custom_sam_peft/config/__init__.py` exposes `load_config(path, overrides) -> TrainConfig | EvalConfig | ExportConfig` as the **only** config-loading function. Audit Section A inbound deps will show every other entry point — remove or redirect them.

Responsibilities (spec §4.1 last bullet):
- YAML reading.
- `--override key=val` merging.
- Env-var interpolation (existing behavior — preserve).
- Resolving relative paths against the config file's directory at load time.
- Downstream code receives absolute paths and never re-resolves.

- [ ] **Step 2.4.6: Update + run config tests**

```bash
pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py \
       tests/unit/test_config_examples.py -v
```
Expected: all PASS. If any example config still references a renamed field, defer the example update to Task 7.1 — but stub the example to parse against the new schema so the test passes here.

- [ ] **Step 2.4.7: Commit**

```bash
git add src/custom_sam_peft/config/ tests/unit/test_config_schema.py \
        tests/unit/test_config_loader.py tests/unit/test_config_examples.py
git commit -m "refactor(config): apply audit verdicts, internal-class split, single load_config seam"
```

### Task 2.5: `src/custom_sam_peft/_bootstrap.py` — sole registration site

**Files:**
- Modify: `src/custom_sam_peft/_bootstrap.py`
- Modify: `tests/unit/test_bootstrap.py`

- [ ] **Step 2.5.1: Read the existing `_bootstrap.py`**

The file already exists per the audit (Section A inbound deps). Identify what it currently does and what needs to centralize into it.

- [ ] **Step 2.5.2: Centralize side effects**

`_bootstrap.bootstrap()` must be the sole path that:
1. Registers PEFT adapters (`peft_adapters.lora`, `peft_adapters.qlora` — imports cause `@register("peft", ...)` to fire).
2. Registers tracking backends (`tracking.noop`, `tracking.wandb`, `tracking.tensorboard`).
3. Applies the seed-setting (`torch.manual_seed`, `random.seed`, etc.) using the config's seed value.
4. Configures `logging` (basic config, level from config).

It does NOT apply Sam3Patches — that runs at model-load time inside `load_sam31` (Task 5.1).

- [ ] **Step 2.5.3: Every CLI command + notebook helper calls `bootstrap()` once**

Audit identifies the existing call sites; ensure they're all there and nothing else triggers registration as a side effect. Specifically, `src/custom_sam_peft/cli/main.py` calls `bootstrap()` before dispatching to any subcommand.

- [ ] **Step 2.5.4: Run bootstrap tests**

```bash
pytest tests/unit/test_bootstrap.py -v
```
Expected: PASS. If existing tests assume the OLD registration path (e.g., import-side-effects in `peft_adapters/__init__.py`), update them to assume `bootstrap()` is the gate.

- [ ] **Step 2.5.5: Commit**

```bash
git add src/custom_sam_peft/_bootstrap.py tests/unit/test_bootstrap.py
git commit -m "refactor(bootstrap): make _bootstrap the sole registration + seed + logging site"
```

---

## Task 3: Static-guard test scaffold

**Goal:** Land the three §9.2 static guards as failing pytest entries. They MUST stay failing until Tasks 4 and 5 do their work; this is the most direct enforcement mechanism for spec §3 success criteria #2, #3, and the no-string-joined-paths rule.

**Files:**
- Create: `tests/unit/test_static_guards.py`

- [ ] **Step 3.1: Write the three guards**

```python
# tests/unit/test_static_guards.py
"""Static guards enforce structural invariants from spec §3.

These are cheap `rg`-based checks. They land FIRST (failing) so that
Tasks 4 and 5 land the refactors that make them pass. After this PR,
they stay green as regression detectors.
"""

from __future__ import annotations
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "custom_sam_peft"


def _rg(pattern: str, *, in_dir: Path, extra_args: list[str] | None = None) -> list[str]:
    cmd = ["rg", "-n", pattern, str(in_dir)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 1):  # 1 = no matches
        raise RuntimeError(f"rg failed: {result.stderr}")
    return [line for line in result.stdout.splitlines() if line]


def test_no_peft_method_branches_outside_peft_adapters():
    """Spec §3 #2: no `if .*\\.method ==` in src/ outside peft_adapters/."""
    hits = _rg(r"\.method\s*==", in_dir=SRC, extra_args=["--type", "py"])
    offenders = [
        h for h in hits
        if "/peft_adapters/" not in h and "test_" not in h
    ]
    assert not offenders, (
        "PEFT method-string branches detected outside peft_adapters/.\n"
        "Move these behind the @register('peft', ...) factory and the\n"
        "PEFTMethod protocol. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_no_to_device_outside_collator_and_runtime():
    """Spec §3 #3: device-move sites collapse to data collator + runtime/."""
    hits = _rg(r"\.to\(device", in_dir=SRC, extra_args=["--type", "py"])
    allowed_substrings = ("/runtime/", "/data/collate.py")
    offenders = [
        h for h in hits
        if not any(allowed in h for allowed in allowed_substrings)
    ]
    assert not offenders, (
        "`.to(device)` outside runtime/ and data/collate.py.\n"
        "Route all device moves through runtime.to_device. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_no_string_joined_checkpoint_paths_outside_paths_module():
    """Spec §3: no `runs/.../checkpoints/` string-joining outside paths/."""
    # Patterns: literal "checkpoints/" in a string, OR
    # f-strings / .format / + concat that build a checkpoints subpath.
    patterns = [
        r'"checkpoints/',          # literal
        r"'checkpoints/",          # literal single-quoted
        r"f\".*checkpoints",       # f-string
        r"f'.*checkpoints",        # f-string single-quoted
    ]
    offenders: list[str] = []
    for pattern in patterns:
        hits = _rg(pattern, in_dir=SRC, extra_args=["--type", "py"])
        for h in hits:
            if "/paths/" in h or "/_patches/" in h:
                continue  # paths/ owns the layout; _patches has no path code
            if "# noqa: paths-guard" in h:
                continue  # explicit opt-out (audit may surface legitimate cases)
            offenders.append(h)
    assert not offenders, (
        "String-joined checkpoint paths outside src/custom_sam_peft/paths/.\n"
        "Use paths.checkpoint_path(run_dir, step=N). Offenders:\n  "
        + "\n  ".join(offenders)
    )
```

- [ ] **Step 3.2: Confirm the guards FAIL today**

```bash
pytest tests/unit/test_static_guards.py -v
```
Expected: **all three FAIL**. This is the correct starting state — Tasks 4 and 5 fix them.

- [ ] **Step 3.3: Commit (with `xfail` markers OFF — we want red CI to keep us honest)**

```bash
git add tests/unit/test_static_guards.py
git commit -m "test(guards): add static guards for method branches, .to(device), checkpoint paths"
```

**Note:** This commit intentionally lands red. CI on this branch will be red until Tasks 4 + 5 land. That is the design — the static guards are the forcing function. If the orchestrator's CI watch shows other unrelated failures, fix those, but DO NOT skip or xfail the three guards.

---

## Task 4: Seam cleanups [PARALLEL subtasks 4.1–4.4]

After Task 2 lands, the four subtasks here touch disjoint file sets and parallelize cleanly. Each subtask is one dispatched implementer.

### Task 4.1: PEFT method-string leaks → protocol

**Files:**
- Modify: `src/custom_sam_peft/peft_adapters/__init__.py` (extend protocol)
- Modify: `src/custom_sam_peft/peft_adapters/lora.py` (implement new methods)
- Modify: `src/custom_sam_peft/peft_adapters/qlora.py` (implement new methods)
- Modify: `src/custom_sam_peft/train/loop.py` (line ~66 — remove branch)
- Modify: `src/custom_sam_peft/train/trainer.py` (line ~49 — remove branch)
- Modify: `src/custom_sam_peft/train/checkpoint.py` (line ~150 — remove branch)
- Modify: `src/custom_sam_peft/eval/runner.py` (line ~76 — remove branch)
- Modify: `tests/unit/test_peft_lora.py`, `tests/unit/test_peft_qlora.py`, `tests/unit/test_registry.py`

- [ ] **Step 4.1.1: Read audit Section B**

The audit finalized the protocol surface (OQ1). Read the exact method names, signatures, and which branches each replaces.

- [ ] **Step 4.1.2: Write failing tests against the new protocol**

For each new protocol method, add a test verifying:
1. The protocol declares it (use `@runtime_checkable` Protocol or ABC — match the existing style in `peft_adapters/__init__.py`).
2. Both `LoraAdapter` and `QloraAdapter` implement it.
3. The trainer / evaluator / checkpoint loader call the protocol method instead of branching.

Example structure (concrete names per audit):

```python
def test_peft_method_protocol_declares_recommended_optimizer():
    from custom_sam_peft.peft_adapters import PEFTMethod
    assert hasattr(PEFTMethod, "recommended_optimizer")


def test_lora_adapter_recommended_optimizer_returns_string():
    from custom_sam_peft.peft_adapters.lora import LoraAdapter
    adapter = LoraAdapter(...)  # minimal init per existing test fixtures
    assert isinstance(adapter.recommended_optimizer(), str)


def test_trainer_does_not_branch_on_method_name():
    """Reads trainer.py source; asserts no `peft.method ==` strings."""
    import inspect
    from custom_sam_peft.train import trainer
    src = inspect.getsource(trainer)
    assert ".method ==" not in src
```

(Each protocol method from audit Section B gets analogous tests.)

- [ ] **Step 4.1.3: Run tests to confirm failure**

```bash
pytest tests/unit/test_peft_lora.py tests/unit/test_peft_qlora.py \
       tests/unit/test_registry.py -v
```
Expected: the new tests FAIL.

- [ ] **Step 4.1.4: Extend the protocol**

In `src/custom_sam_peft/peft_adapters/__init__.py`, add each method named in audit Section B. Signatures verbatim from audit. Use `typing.Protocol` (matching existing style) — the audit may have specified ABCs instead; match what's there.

- [ ] **Step 4.1.5: Implement each new method on `LoraAdapter` and `QloraAdapter`**

For each method:
- `LoraAdapter.<method>` returns whatever the current `if peft.method == "lora": ...` branch returns.
- `QloraAdapter.<method>` returns whatever the current `if peft.method == "qlora": ...` branch returns.
- The implementations should not branch on method name internally.

- [ ] **Step 4.1.6: Remove the four spec-named branches**

For each of:
- `src/custom_sam_peft/train/loop.py:66`
- `src/custom_sam_peft/train/trainer.py:49`
- `src/custom_sam_peft/train/checkpoint.py:150`
- `src/custom_sam_peft/eval/runner.py:76`

…and any other leaks the audit surfaced in Section B, replace the `if .method == "lora": ... elif .method == "qlora": ...` block with a single call to the protocol method on the `PEFTMethod` instance.

- [ ] **Step 4.1.7: Run all tests + the static guard**

```bash
pytest tests/unit/test_static_guards.py::test_no_peft_method_branches_outside_peft_adapters -v
pytest tests/unit/test_peft_lora.py tests/unit/test_peft_qlora.py \
       tests/unit/test_registry.py tests/unit/test_train_step.py \
       tests/unit/test_eval_runner.py tests/unit/test_train_checkpoint.py -v
```
Expected: the static guard now PASSES; all other tests PASS.

- [ ] **Step 4.1.8: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/ src/custom_sam_peft/train/loop.py \
        src/custom_sam_peft/train/trainer.py src/custom_sam_peft/train/checkpoint.py \
        src/custom_sam_peft/eval/runner.py tests/unit/test_peft_lora.py \
        tests/unit/test_peft_qlora.py tests/unit/test_registry.py
git commit -m "refactor(peft): replace method-string branches with PEFTMethod protocol calls"
```

### Task 4.2: Tracking consolidation

**Files:**
- Modify: `src/custom_sam_peft/train/trainer.py` (no direct backend imports)
- Modify: anywhere outside `src/custom_sam_peft/tracking/` that imports `wandb` or `tensorboardX` directly
- Modify: `tests/unit/test_tracking_*.py`

- [ ] **Step 4.2.1: Search for direct backend imports outside tracking/**

```bash
rg -n '^import wandb|^from wandb|^import tensorboardX|^from tensorboardX' \
   src/custom_sam_peft/ --type py
```
Expected post-fix: zero hits outside `src/custom_sam_peft/tracking/`.

- [ ] **Step 4.2.2: Remove every such import; route through the `Tracker` protocol**

For each hit (audit Section A inbound deps surfaces them):
- The caller already has a `Tracker` instance available, or has access to `build_tracker` via `_bootstrap`. Use it.
- Add a method to the `Tracker` protocol if a caller needs a capability the protocol doesn't currently expose. Implement it on `NoopTracker`, `WandbTracker`, `TensorboardTracker`.

- [ ] **Step 4.2.3: Confirm `build_tracker` is the single construction site**

```bash
rg -n 'WandbTracker\(|TensorboardTracker\(|NoopTracker\(' \
   src/custom_sam_peft/ --type py
```
Expected: hits only inside `src/custom_sam_peft/tracking/` (where the backends define themselves) and inside `build_tracker` itself.

- [ ] **Step 4.2.4: Run tracking tests**

```bash
pytest tests/unit/test_tracking_protocol.py tests/unit/test_tracking_noop.py \
       tests/unit/test_tracking_wandb.py tests/unit/test_tracking_tensorboard.py \
       tests/unit/test_tracking_build.py -v
```
Expected: all PASS.

- [ ] **Step 4.2.5: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py src/custom_sam_peft/tracking/ \
        tests/unit/test_tracking_*.py
git commit -m "refactor(tracking): consolidate backend imports behind Tracker protocol"
```

### Task 4.3: `EvalArtifacts` value object — trainer ↔ evaluator seam

**Files:**
- Create: `src/custom_sam_peft/eval/_artifacts.py`
- Modify: `src/custom_sam_peft/train/trainer.py` (return `EvalArtifacts` from `fit`)
- Modify: `src/custom_sam_peft/eval/evaluator.py` (accept `EvalArtifacts`)
- Modify: `src/custom_sam_peft/eval/runner.py`
- Create: `tests/unit/test_eval_artifacts.py`
- Create: `tests/integration/test_trainer_evaluator_seam.py`

- [ ] **Step 4.3.1: Write the failing test for `EvalArtifacts`**

```python
# tests/unit/test_eval_artifacts.py
from pathlib import Path
from custom_sam_peft.eval._artifacts import EvalArtifacts


def test_eval_artifacts_fields():
    art = EvalArtifacts(
        checkpoint_path=Path("/tmp/runs/x/checkpoints/step_00000100.pt"),
        peft_method="lora",
        run_dir=Path("/tmp/runs/x"),
    )
    assert art.checkpoint_path.name == "step_00000100.pt"
    assert art.peft_method == "lora"
    assert art.run_dir == Path("/tmp/runs/x")


def test_eval_artifacts_is_frozen():
    import pytest
    art = EvalArtifacts(
        checkpoint_path=Path("/x"), peft_method="lora", run_dir=Path("/y"),
    )
    with pytest.raises((AttributeError, Exception)):
        art.peft_method = "qlora"  # type: ignore[misc]
```

- [ ] **Step 4.3.2: Run test to confirm failure**

```bash
pytest tests/unit/test_eval_artifacts.py -v
```
Expected: ImportError.

- [ ] **Step 4.3.3: Implement `src/custom_sam_peft/eval/_artifacts.py`**

```python
"""EvalArtifacts — the single value object the evaluator consumes from the trainer."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalArtifacts:
    """Hand-off object returned by Trainer.fit, consumed by Evaluator.

    These are the ONLY fields the evaluator may read from training
    output. The evaluator does not reach into trainer internals beyond
    this object. Tests in tests/integration/test_trainer_evaluator_seam.py
    enforce this.
    """

    checkpoint_path: Path
    peft_method: str
    run_dir: Path
```

- [ ] **Step 4.3.4: Update `Trainer.fit` to return `EvalArtifacts`**

The exact location is `src/custom_sam_peft/train/trainer.py::Trainer.fit` (the function being decomposed in Task 5.2). For this subtask, change the return value only; the decomposition lands in Task 5.2.

- [ ] **Step 4.3.5: Update `Evaluator.evaluate` to accept `EvalArtifacts`**

Drop any reference to trainer fields beyond what `EvalArtifacts` carries. If the evaluator was reading `trainer.checkpoint_dir` or `trainer.peft.method`, it now reads `artifacts.run_dir` / `artifacts.peft_method` instead.

- [ ] **Step 4.3.6: Write the seam integration test**

```python
# tests/integration/test_trainer_evaluator_seam.py
"""Trainer ↔ evaluator seam test (spec §9.1).

Runs both ends without mocking internals. Asserts on EvalArtifacts
shape and that the evaluator consumes nothing else from the trainer.
"""
import pytest
from pathlib import Path

# Use the existing fast-mode fixtures from tests/fixtures/ — the audit
# Section A surfaces which fixture provides a CPU-runnable mini-train.


def test_trainer_returns_eval_artifacts(tmp_path, tiny_train_config):
    from custom_sam_peft import run_train

    artifacts = run_train(tiny_train_config)
    from custom_sam_peft.eval._artifacts import EvalArtifacts

    assert isinstance(artifacts, EvalArtifacts)
    assert artifacts.checkpoint_path.exists()
    assert artifacts.run_dir.is_dir()
    assert artifacts.peft_method in {"lora", "qlora"}


def test_evaluator_consumes_only_eval_artifacts(tmp_path, tiny_eval_config, tiny_train_config):
    from custom_sam_peft import run_train, run_eval

    artifacts = run_train(tiny_train_config)
    # The evaluator receives only EvalArtifacts. If it needed a field
    # from the trainer that EvalArtifacts doesn't carry, this call
    # raises AttributeError.
    metrics = run_eval(tiny_eval_config, artifacts=artifacts)
    assert isinstance(metrics, dict)
```

Use `tiny_train_config` / `tiny_eval_config` fixtures from `tests/conftest.py` (existing) or `tests/fixtures/`. If they don't exist yet at the right shape, add them as part of this commit using the smallest possible config that exercises the trainer/evaluator pair on CPU.

- [ ] **Step 4.3.7: Run all the affected tests**

```bash
pytest tests/unit/test_eval_artifacts.py tests/integration/test_trainer_evaluator_seam.py \
       tests/unit/test_evaluator.py tests/unit/test_eval_runner.py -v
```
Expected: all PASS.

- [ ] **Step 4.3.8: Commit**

```bash
git add src/custom_sam_peft/eval/_artifacts.py src/custom_sam_peft/train/trainer.py \
        src/custom_sam_peft/eval/evaluator.py src/custom_sam_peft/eval/runner.py \
        tests/unit/test_eval_artifacts.py tests/integration/test_trainer_evaluator_seam.py
git commit -m "refactor(eval): add EvalArtifacts seam between Trainer.fit and Evaluator"
```

### Task 4.4: CLI internals — thin wrappers over library API (the structural half of the two-pass split)

**Files:**
- Modify: `src/custom_sam_peft/cli/train_cmd.py`
- Modify: `src/custom_sam_peft/cli/eval_cmd.py`
- Modify: `src/custom_sam_peft/cli/export_cmd.py`
- Modify: `src/custom_sam_peft/cli/run_cmd.py`
- Modify: `src/custom_sam_peft/cli/init_cmd.py`
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`
- Modify: `src/custom_sam_peft/__init__.py` (expose library API)
- Modify: `tests/unit/test_cli.py`

> **Two-pass CLI split (spec §5.4 — user-locked):** this subtask is the **internals** pass. CLI **surface** changes (renames, new `--eval` / `--export` flags) land in Task 6.2. Do NOT change command names or flag surfaces here.

- [ ] **Step 4.4.1: Extract library functions**

For each CLI command, the corresponding library function:
- `cli/train_cmd.py` calls `run_train(config: TrainConfig) -> EvalArtifacts`.
- `cli/eval_cmd.py` calls `run_eval(config: EvalConfig, artifacts: EvalArtifacts | None = None) -> dict[str, float]`.
- `cli/export_cmd.py` calls `run_export(config: ExportConfig, run_dir: Path) -> Path` (returns the bundle path).
- `cli/run_cmd.py` calls `run_train(...)` followed by `run_eval(...)` followed by `run_export(...)` — i.e., it's a documented alias.
- `cli/init_cmd.py` calls `run_init(template_name: str, output_dir: Path)`.
- `cli/doctor_cmd.py` calls `run_doctor()` (internals reused from `diagnostics.py`).

These functions live in their respective modules (e.g., `run_train` in `src/custom_sam_peft/train/runner.py` — it already exists as `runner.run`; rename if needed for clarity).

- [ ] **Step 4.4.2: Make each `_cmd.py` a thin wrapper**

Pattern for every `_cmd.py`:

```python
# src/custom_sam_peft/cli/train_cmd.py (illustrative)
import typer
from pathlib import Path
from custom_sam_peft.config import load_config
from custom_sam_peft.train.runner import run_train

app = typer.Typer()


@app.callback(invoke_without_command=True)
def train(config_path: Path, overrides: list[str] = typer.Option(None, "--override", "-o")):
    """Train a model from a YAML config."""
    config = load_config(config_path, overrides=overrides or [])
    artifacts = run_train(config)
    typer.echo(f"Training complete. Checkpoint: {artifacts.checkpoint_path}")
```

No business logic in `_cmd.py` files. The CLI module is responsible for: argument parsing, calling `load_config`, calling the library function, rendering the result. Nothing else.

- [ ] **Step 4.4.3: Expose the library API**

`src/custom_sam_peft/__init__.py` exports:

```python
from custom_sam_peft.train.runner import run_train
from custom_sam_peft.eval.runner import run_eval
from custom_sam_peft.runs.bundle import run_export, write_bundle
from custom_sam_peft.eval._artifacts import EvalArtifacts
from custom_sam_peft.errors import (
    CustomSamPeftError, ConfigError, DataError, ModelError,
    CheckpointError, EnvironmentError,
)

__all__ = [
    "run_train", "run_eval", "run_export", "write_bundle",
    "EvalArtifacts",
    "CustomSamPeftError", "ConfigError", "DataError", "ModelError",
    "CheckpointError", "EnvironmentError",
]
```

This is what the Colab notebook imports.

- [ ] **Step 4.4.4: Wire `cli/main.py` to catch `CustomSamPeftError`**

```python
# src/custom_sam_peft/cli/main.py (illustrative core)
import typer
from custom_sam_peft._bootstrap import bootstrap
from custom_sam_peft.errors import CustomSamPeftError
# ... command imports ...

app = typer.Typer()
# ... add_typer calls for each subcommand ...


def main():
    bootstrap()
    try:
        app()
    except CustomSamPeftError as e:
        # Task 6.3 expands this to the four-part rendering. For now,
        # just print the message and exit 1.
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        typer.echo("Rerun with -v for full traceback.", err=True)
        raise typer.Exit(code=1)
```

Task 6.3 expands the rendering into the four-part shape (summary / expected / found / fix).

- [ ] **Step 4.4.5: Run CLI tests**

```bash
pytest tests/unit/test_cli.py tests/unit/test_cli_doctor.py \
       tests/unit/test_cli_export.py tests/unit/test_cli_init.py \
       tests/integration/test_cli_run.py -v
```
Expected: all PASS. CLI command names are unchanged here; only internals refactored.

- [ ] **Step 4.4.6: Commit**

```bash
git add src/custom_sam_peft/cli/ src/custom_sam_peft/__init__.py \
        tests/unit/test_cli*.py tests/integration/test_cli_run.py
git commit -m "refactor(cli): thin command wrappers over run_train/run_eval/run_export library API"
```

---

## Task 5: God-function decomposition [PARALLEL subtasks 5.1–5.7]

**Convention reminder (spec §6, user-approved):** private helpers stay `_`-prefixed in the SAME file. Promote a helper to a sibling module only on the second caller (Rule of Three). Single exception: the `_patch_*` wall in `sam3.py` becomes one-file-per-patch under `models/_patches/` (Task 5.7), per explicit user approval.

### Task 5.1: `load_sam31` → orchestrating shell

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py:1054` (`load_sam31`)
- Existing tests under `tests/unit/test_load_sam31_*` and `tests/unit/test_sam3_*` cover behavior; integration: `tests/integration/test_load_sam31_real.py`.

- [ ] **Step 5.1.1: Decompose into the five private helpers per spec §6**

In `src/custom_sam_peft/models/sam3.py`, extract:
- `_locate_weights(config) -> Path` — HF / local / cache resolution.
- `_construct_raw_model(config) -> nn.Module` — instantiate the model class with the raw weights.
- `_apply_dtype(model, runtime: Runtime) -> None` — set dtype based on `runtime.dtype`.
- `_apply_patches(model, runtime: Runtime) -> None` — calls `Sam3Patches.apply(model, runtime)`.
- `_freeze_base(model, peft_method) -> None` — freezes non-adapter params.

`load_sam31` becomes:

```python
def load_sam31(config, runtime: Runtime, peft_method: PEFTMethod) -> nn.Module:
    weights = _locate_weights(config)
    model = _construct_raw_model(config)
    _load_weights_into_model(model, weights)   # if this exists separately; keep
    _apply_dtype(model, runtime)
    _apply_patches(model, runtime)
    _freeze_base(model, peft_method)
    return model
```

- [ ] **Step 5.1.2: Each helper gets a small unit test if it doesn't already**

For helpers without existing coverage, add unit tests using small synthetic configs. The audit Section A surfaces which helpers already have coverage via the existing tests above.

- [ ] **Step 5.1.3: Run all model tests**

```bash
pytest tests/unit/test_load_sam31_missing_keys_filter.py \
       tests/unit/test_sam3_checkpoint_resolve.py \
       tests/unit/test_sam3_wrapper.py -v
```
Expected: PASS. (`tests/integration/test_load_sam31_real.py` is GPU-marked; skips on CPU.)

- [ ] **Step 5.1.4: Verify `load_sam31` is <60 lines**

```bash
awk '/^def load_sam31/,/^def [^_]/' src/custom_sam_peft/models/sam3.py | grep -c '^' || true
```
Manually verify it's ~10 lines (the orchestrating shell).

- [ ] **Step 5.1.5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/
git commit -m "refactor(models): decompose load_sam31 into _locate/_construct/_apply_dtype/_apply_patches/_freeze_base"
```

### Task 5.2: `Trainer.fit` → orchestrating shell

**Files:**
- Modify: `src/custom_sam_peft/train/trainer.py:134` (`Trainer.fit`)
- Modify: `tests/unit/test_train_step.py`, `tests/unit/test_trainer_run_dir.py`, `tests/unit/test_trainer_guards.py`

- [ ] **Step 5.2.1: Decompose into five private helpers per spec §6**

Extract in `Trainer`:
- `_setup_run_dir() -> Path` — uses `paths/` helpers.
- `_build_optimizer() -> Optimizer` — calls `self.peft_method.recommended_optimizer()`.
- `_train_epoch(epoch: int) -> None` — one-epoch loop.
- `_eval_epoch(epoch: int) -> dict[str, float]` — periodic in-loop eval.
- `_maybe_checkpoint(step: int) -> None` — uses `paths.checkpoint_path(self.run_dir, step=step)`.

`Trainer.fit` becomes:

```python
def fit(self) -> EvalArtifacts:
    self.run_dir = self._setup_run_dir()
    self.optimizer = self._build_optimizer()
    for epoch in range(self.config.epochs):
        self._train_epoch(epoch)
        if self._should_eval(epoch):
            self._eval_epoch(epoch)
        self._maybe_checkpoint(self.global_step)
    return EvalArtifacts(
        checkpoint_path=paths.checkpoint_path(self.run_dir, step=self.global_step),
        peft_method=self.peft_method.name,
        run_dir=self.run_dir,
    )
```

- [ ] **Step 5.2.2: Trainer constructor accepts a `Runtime` instance**

Per spec §2: `Trainer.__init__` accepts a `Runtime` and stores it as `self.runtime`. It never reads global device state. Every `self.device` reference inside the trainer becomes `self.runtime.device`.

- [ ] **Step 5.2.3: Run trainer tests**

```bash
pytest tests/unit/test_train_step.py tests/unit/test_trainer_run_dir.py \
       tests/unit/test_trainer_guards.py tests/unit/test_trainer_mp_sharing.py \
       tests/unit/test_train_checkpoint.py tests/unit/test_train_runner.py -v
```
Expected: PASS.

- [ ] **Step 5.2.4: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py tests/unit/test_train*.py tests/unit/test_trainer*.py
git commit -m "refactor(train): decompose Trainer.fit; constructor takes Runtime"
```

### Task 5.3: `Evaluator.evaluate` → orchestrating shell

**Files:**
- Modify: `src/custom_sam_peft/eval/evaluator.py:119`
- Modify: `tests/unit/test_evaluator.py`

- [ ] **Step 5.3.1: Decompose**

Extract:
- `_iter_predictions(model, dataloader) -> Iterable[Prediction]`
- `_aggregate_metrics(predictions) -> dict[str, float]`
- `_maybe_save_predictions(predictions, run_dir) -> None` — uses `paths.predictions_path`.

`evaluate` becomes a four-line pipeline.

- [ ] **Step 5.3.2: Run tests**

```bash
pytest tests/unit/test_evaluator.py tests/unit/test_eval_runner.py -v
```
Expected: PASS.

- [ ] **Step 5.3.3: Commit**

```bash
git add src/custom_sam_peft/eval/evaluator.py tests/unit/test_evaluator.py
git commit -m "refactor(eval): decompose Evaluator.evaluate into iter/aggregate/save helpers"
```

### Task 5.4: Dataset `__getitem__` decompositions

**Files:**
- Modify: `src/custom_sam_peft/data/hf.py:159`
- Modify: `src/custom_sam_peft/data/coco.py:168`
- Modify: `tests/unit/test_data_hf.py`, `tests/unit/test_data_coco.py`

- [ ] **Step 5.4.1: Extract four private helpers from each `__getitem__`**

Per spec §6:
- `_decode_image(raw_example) -> Tensor`
- `_decode_targets(raw_example) -> Targets`
- `_apply_transforms(image, targets) -> tuple[Tensor, Targets]`
- `_pack_example(image, targets) -> Example`

Each `__getitem__` becomes a four-line pipeline:

```python
def __getitem__(self, idx):
    raw = self._fetch_raw(idx)
    image = self._decode_image(raw)
    targets = self._decode_targets(raw)
    image, targets = self._apply_transforms(image, targets)
    return self._pack_example(image, targets)
```

The two classes share helper *signatures* but each owns its own helper bodies (different decoding logic). Don't promote to a shared module — Rule of Three not yet met.

- [ ] **Step 5.4.2: Run dataset tests**

```bash
pytest tests/unit/test_data_hf.py tests/unit/test_data_coco.py \
       tests/unit/test_data_base.py tests/unit/test_data_transforms.py \
       tests/unit/test_data_collate.py -v
```
Expected: PASS.

- [ ] **Step 5.4.3: Commit**

```bash
git add src/custom_sam_peft/data/hf.py src/custom_sam_peft/data/coco.py \
        tests/unit/test_data_*.py
git commit -m "refactor(data): decompose HFDataset/COCODataset __getitem__ into 4-line pipeline"
```

### Task 5.5: `write_bundle` → orchestrating shell + use `paths/`

**Files:**
- Modify: `src/custom_sam_peft/runs/bundle.py:262`
- Modify: `tests/unit/runs/` (existing dir under tests/unit/)

- [ ] **Step 5.5.1: Decompose**

Extract:
- `_collect_artifacts(run_dir) -> list[Path]`
- `_write_manifest(run_dir, artifacts) -> Path`
- `_zip_bundle(run_dir, artifacts, manifest) -> Path`

`write_bundle` becomes a three-line pipeline. All path construction routes through `src/custom_sam_peft/paths/` (no string-joined paths).

- [ ] **Step 5.5.2: Confirm the third static guard passes for `bundle.py`**

```bash
pytest tests/unit/test_static_guards.py::test_no_string_joined_checkpoint_paths_outside_paths_module -v
```
This guard should now pass for `bundle.py` specifically — but won't fully pass until **every** consumer (train/checkpoint.py, eval/runner.py, CLI) is also routed through `paths/`. Other subtasks (4.x, 5.x) handle their own files.

- [ ] **Step 5.5.3: Run bundle tests**

```bash
pytest tests/unit/runs/ -v
```
Expected: PASS.

- [ ] **Step 5.5.4: Commit**

```bash
git add src/custom_sam_peft/runs/bundle.py tests/unit/runs/
git commit -m "refactor(runs): decompose write_bundle into collect/manifest/zip helpers using paths/"
```

### Task 5.6: `apply_qlora` → orchestrating shell

**Files:**
- Modify: `src/custom_sam_peft/peft_adapters/qlora.py:159`
- Modify: `tests/unit/test_peft_qlora.py`

- [ ] **Step 5.6.1: Decompose**

Extract:
- `_quantize_base(model) -> nn.Module`
- `_inject_lora_adapters(model, config) -> nn.Module`
- `_freeze_non_adapter(model) -> None`

`apply_qlora` becomes the orchestrating shell and is what `@register("peft", "qlora")` exposes.

- [ ] **Step 5.6.2: Run QLoRA tests**

```bash
pytest tests/unit/test_peft_qlora.py tests/integration/test_peft_qlora_real.py -v
```
Note: `test_peft_qlora_real.py` is GPU-marked; CPU run skips it. The unit test should PASS.

- [ ] **Step 5.6.3: Commit**

```bash
git add src/custom_sam_peft/peft_adapters/qlora.py tests/unit/test_peft_qlora.py
git commit -m "refactor(peft): decompose apply_qlora into quantize/inject/freeze helpers"
```

### Task 5.7: `_patch_*` wall → one file per patch

**Files:**
- Move: each `_patch_*` function in `src/custom_sam_peft/models/sam3.py` → its own file under `src/custom_sam_peft/models/_patches/<patch_name>.py`
- Modify: `src/custom_sam_peft/models/_patches/__init__.py` (the `_ALL_PATCHES` list)
- Modify: `src/custom_sam_peft/runtime/_patches.py` (already imports `_ALL_PATCHES`)
- Move corresponding tests: `tests/unit/test_sam3_*_patch.py` already exist per-patch; just verify each still imports the patch from its new location

> **User-locked decomposition:** one file per patch. Do NOT collapse multiple patches into one file.

- [ ] **Step 5.7.1: Enumerate `_patch_*` functions in `sam3.py`**

```bash
rg -n '^def _patch_' src/custom_sam_peft/models/sam3.py
```

Expected (from existing per-patch tests in `tests/unit/test_sam3_*_patch.py`):
- `_patch_addmm_act_grad_safe`
- `_patch_encode_prompt`
- `_patch_mha_input_dtype`
- `_patch_module_input_dtype`
- `_patch_pos_enc`
- `_patch_roi_align`
- `_patch_skip_matching`
- `_patch_text_pool`

(Audit Section B / I may surface more; use the audit's count as the source of truth.)

- [ ] **Step 5.7.2: For EACH `_patch_*`, create one file**

Example for `_patch_roi_align`:

```python
# src/custom_sam_peft/models/_patches/roi_align.py
"""Patch: roi_align dtype consistency.

Moved from src/custom_sam_peft/models/sam3.py during the v0.7.0
hardening pass. The applier in src/custom_sam_peft/runtime/_patches.py
runs all patches in deterministic sorted-by-filename order.
"""

from __future__ import annotations
from typing import Any

from custom_sam_peft.runtime._runtime import Runtime


def apply(model: Any, runtime: Runtime) -> None:
    # ... patch body moved verbatim from sam3.py::_patch_roi_align ...
    # Reference: tests/unit/test_sam3_roi_align_patch.py asserts this
    # patch keeps roi_align dtype-correct.
    ...
```

Naming convention: file name is the `_patch_<name>` suffix without the leading underscore (`_patch_roi_align` → `roi_align.py`).

- [ ] **Step 5.7.3: Update `_patches/__init__.py`**

```python
"""Per-patch modules. Each exports an `apply(model, runtime)` function.

Order is deterministic by filename (alphabetical) so behavior is
reproducible and audit-friendly.
"""

from custom_sam_peft.models._patches import (
    addmm_act_grad_safe,
    encode_prompt,
    mha_input_dtype,
    module_input_dtype,
    pos_enc,
    roi_align,
    skip_matching,
    text_pool,
)

_ALL_PATCHES = [
    addmm_act_grad_safe.apply,
    encode_prompt.apply,
    mha_input_dtype.apply,
    module_input_dtype.apply,
    pos_enc.apply,
    roi_align.apply,
    skip_matching.apply,
    text_pool.apply,
]
```

- [ ] **Step 5.7.4: Update existing per-patch tests to import from new locations**

For each `tests/unit/test_sam3_<name>_patch.py`, change the import from `from custom_sam_peft.models.sam3 import _patch_<name>` to `from custom_sam_peft.models._patches.<name> import apply as _patch_<name>` (or similar — preserve the test bodies).

- [ ] **Step 5.7.5: Remove the `_patch_*` functions from `sam3.py`**

After moves, `sam3.py` no longer contains `_patch_*` function definitions. `_apply_patches` (from Task 5.1) calls `Sam3Patches.apply(model, runtime)`, which routes through `_ALL_PATCHES`.

- [ ] **Step 5.7.6: Strengthen the Sam3Patches applier test**

```python
# tests/unit/test_sam3_patches_applier.py (replacing the placeholder)
import torch
from custom_sam_peft.runtime import Runtime, Sam3Patches
from custom_sam_peft.models._patches import _ALL_PATCHES


def test_all_patches_registered():
    assert len(_ALL_PATCHES) >= 8  # audit Section B finalizes the count


def test_apply_runs_every_patch_in_order(monkeypatch):
    calls = []

    def make_recorder(name):
        def _apply(model, runtime):
            calls.append(name)
        return _apply

    fake_patches = [make_recorder(f"p{i}") for i in range(3)]
    monkeypatch.setattr(
        "custom_sam_peft.runtime._patches._ALL_PATCHES", fake_patches,
        raising=False,
    )
    # Re-import or re-run apply path
    rt = Runtime(device=torch.device("cpu"), dtype=torch.float32)
    Sam3Patches.apply(object(), rt)
    assert calls == ["p0", "p1", "p2"]
```

- [ ] **Step 5.7.7: Run all SAM-3 patch tests**

```bash
pytest tests/unit/test_sam3_*_patch.py tests/unit/test_sam3_patches_applier.py -v
```
Expected: PASS.

- [ ] **Step 5.7.8: Verify `sam3.py` substantially shrinks (spec §3 #4)**

```bash
wc -l src/custom_sam_peft/models/sam3.py
```
Compare against pre-task line count from the audit. Expected: substantial reduction (no exact threshold — spec says "substantially shrinks").

- [ ] **Step 5.7.9: Commit**

```bash
git add src/custom_sam_peft/models/_patches/ src/custom_sam_peft/models/sam3.py \
        src/custom_sam_peft/runtime/_patches.py tests/unit/test_sam3_*_patch.py \
        tests/unit/test_sam3_patches_applier.py
git commit -m "refactor(models): extract each _patch_* into its own file under models/_patches/"
```

---

## Task 6: User-surface redesign

### Task 6.1: YAML schema redesign

**Files:**
- Modify: `src/custom_sam_peft/config/` (apply audit Section F renames + Section E demote/delete decisions)
- Create: `docs/config-schema.md` (per OQ3)
- Modify: `tests/unit/test_config_schema.py`, `tests/unit/test_config_examples.py`

- [ ] **Step 6.1.1: Verify required-fields surface**

Per spec §7.1, required fields shrink to: `data` (where to load from), `model` (base checkpoint), `peft` (method). Everything else carries a default. Confirm in the Pydantic root config that these three are the only `Field(...)` (no default) entries.

- [ ] **Step 6.1.2: Verify section flatness**

Top-level sections stay flat: `data`, `model`, `peft`, `train`, `eval`, `tracking`, `export`. Within each, fields split into "commonly set" and "advanced" — group advanced fields at the bottom of each section with an explanatory comment.

- [ ] **Step 6.1.3: Drop / demote fields per audit Section E**

Already applied in Task 2.4. Verify here by re-running:

```bash
pytest tests/unit/test_config_schema.py tests/unit/test_config_examples.py -v
```

- [ ] **Step 6.1.4: Author `docs/config-schema.md` (OQ3)**

For every surviving field, the schema doc lists: section, field name, type, default, layer (common / advanced), one-line description, YAGNI-survival rationale (one sentence).

Template:

```markdown
# Configuration Schema (v0.7.0)

> Generated as part of the v0.7.0 hardening pass (issue #26).
> Re-generate when adding or removing a config field.

## `data`

| Field | Type | Default | Layer | Description |
| --- | --- | --- | --- | --- |
| `train.path` | str | (required) | common | Path to the training dataset (HF id or local dir). |
| ...
```

One section per top-level config section. Link from `README.md` "Configuration" section.

- [ ] **Step 6.1.5: Commit**

```bash
git add src/custom_sam_peft/config/ docs/config-schema.md \
        tests/unit/test_config_schema.py tests/unit/test_config_examples.py
git commit -m "feat(config): finalize v0.7.0 schema + author docs/config-schema.md"
```

### Task 6.2: CLI surface — bare `--eval` / `--export` flags (the second CLI pass)

**Files:**
- Modify: `src/custom_sam_peft/cli/train_cmd.py` (add `--eval`, `--export` flags)
- Modify: `src/custom_sam_peft/cli/eval_cmd.py` (add `--export` flag)
- Modify: `src/custom_sam_peft/cli/run_cmd.py` (becomes a documented alias)
- Modify: `src/custom_sam_peft/cli/init_cmd.py` (template flag preserved; templates regenerate from v0.7.0 schema)
- Modify: `tests/unit/test_cli.py`, `tests/integration/test_cli_run.py`, `tests/integration/test_train_then_eval.py`

> **User-locked (spec §7.2):** the flags are **bare `--eval` / `--export`**. Not `--with-eval` / `--with-export`. Not `--then-eval` / `--then-export`. Do not relitigate.

- [ ] **Step 6.2.1: Write the failing CLI tests**

```python
# tests/unit/test_cli.py (additions)
from typer.testing import CliRunner
from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_train_supports_bare_eval_flag(tmp_path, minimal_train_config_path):
    result = runner.invoke(app, ["train", str(minimal_train_config_path), "--eval", "--help"])
    assert result.exit_code == 0
    assert "--eval" in result.output


def test_train_supports_bare_export_flag(tmp_path, minimal_train_config_path):
    result = runner.invoke(app, ["train", str(minimal_train_config_path), "--export", "--help"])
    assert result.exit_code == 0
    assert "--export" in result.output


def test_run_is_alias_for_train_eval_export(tmp_path, minimal_train_config_path):
    # The `run` subcommand documentation states it is an alias.
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "alias" in result.output.lower() or "train --eval --export" in result.output


def test_no_with_flags_exist():
    """User explicitly rejected --with-eval / --with-export."""
    result = runner.invoke(app, ["train", "--help"])
    assert "--with-eval" not in result.output
    assert "--with-export" not in result.output
    assert "--then-eval" not in result.output
    assert "--then-export" not in result.output
```

- [ ] **Step 6.2.2: Run tests to confirm failure**

```bash
pytest tests/unit/test_cli.py -v
```
Expected: the four new tests FAIL.

- [ ] **Step 6.2.3: Add the bare flags**

In `train_cmd.py`:

```python
@app.callback(invoke_without_command=True)
def train(
    config_path: Path,
    overrides: list[str] = typer.Option(None, "--override", "-o"),
    do_eval: bool = typer.Option(False, "--eval",
        help="After training, run evaluation against the same config's eval section."),
    do_export: bool = typer.Option(False, "--export",
        help="After training (and eval, if --eval), export a run bundle."),
):
    """Train a model. The order is fixed: train → eval → export.

    Flags only toggle inclusion, never reorder.
    """
    config = load_config(config_path, overrides=overrides or [])
    artifacts = run_train(config)
    if do_eval:
        run_eval(config, artifacts=artifacts)
    if do_export:
        run_export(config, run_dir=artifacts.run_dir)
```

Similarly for `eval_cmd.py` (add `--export`).

In `run_cmd.py`:

```python
@app.callback(invoke_without_command=True)
def run(config_path: Path, overrides: list[str] = typer.Option(None, "--override", "-o")):
    """Alias for `train --eval --export`.

    Use this when you want the full pipeline in one command. The
    Colab notebook uses `run` for the canonical end-to-end flow.
    """
    config = load_config(config_path, overrides=overrides or [])
    artifacts = run_train(config)
    run_eval(config, artifacts=artifacts)
    run_export(config, run_dir=artifacts.run_dir)
```

- [ ] **Step 6.2.4: Apply audit Section F renames to CLI surfaces (if any)**

Some renames may also affect CLI flag names (e.g., `--learning-rate` vs `--lr`). Audit Section F identifies them; apply consistently.

- [ ] **Step 6.2.5: Run tests**

```bash
pytest tests/unit/test_cli.py tests/integration/test_cli_run.py \
       tests/integration/test_train_then_eval.py -v
```
Expected: PASS.

- [ ] **Step 6.2.6: Commit**

```bash
git add src/custom_sam_peft/cli/ tests/unit/test_cli.py \
        tests/integration/test_cli_run.py tests/integration/test_train_then_eval.py
git commit -m "feat(cli): bare --eval/--export flags on train/eval; run is documented alias"
```

### Task 6.3: Error message UX — four-part rendering

**Files:**
- Modify: `src/custom_sam_peft/cli/main.py`
- Modify: existing call sites that raise the typed errors (audit Section A surfaces them)
- Create: `tests/unit/test_cli_error_rendering.py`

- [ ] **Step 6.3.1: Write the failing test**

```python
# tests/unit/test_cli_error_rendering.py
from typer.testing import CliRunner
from custom_sam_peft.cli.main import app

runner = CliRunner()


def test_config_error_renders_four_parts(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("data:\n  train:\n    path: /no/such/dir\n")

    result = runner.invoke(app, ["train", str(bad_config)])
    assert result.exit_code == 1
    out = result.output
    # Four-part shape: summary / Expected: / Found: / Fix:
    assert "ConfigError" in out or "Config" in out
    assert "Expected:" in out
    assert "Found:" in out
    assert "Fix:" in out
    assert "-v" in out  # mentions the verbose-flag escape hatch
```

- [ ] **Step 6.3.2: Run tests to confirm failure**

```bash
pytest tests/unit/test_cli_error_rendering.py -v
```
Expected: FAIL (no four-part rendering yet).

- [ ] **Step 6.3.3: Implement the four-part renderer**

Each typed error carries enough info to be rendered. Extend the error classes if needed so the renderer can produce the four parts:

```python
# src/custom_sam_peft/errors.py (extending)

class CustomSamPeftError(Exception):
    """..."""
    @property
    def expected(self) -> str | None:
        return getattr(self, "_expected", None)

    @property
    def found(self) -> str | None:
        return getattr(self, "_found", None)

    @property
    def fix(self) -> str | None:
        return getattr(self, "_fix", None)
```

(Subclass constructors take `expected=`, `found=`, `fix=` kwargs and store them. Audit Section A's inbound-deps + the existing call sites surface every site that raises a CustomSamPeftError — update each to pass these fields when applicable.)

In `cli/main.py`:

```python
def _render_error(e: CustomSamPeftError) -> str:
    parts = [str(e)]
    if e.expected:
        parts.append(f"Expected: {e.expected}")
    if e.found:
        parts.append(f"Found: {e.found}")
    if e.fix:
        parts.append(f"Fix: {e.fix}")
    parts.append("Rerun with -v for full traceback.")
    return "\n".join(parts)


def main():
    bootstrap()
    try:
        app()
    except CustomSamPeftError as e:
        typer.secho(_render_error(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
```

- [ ] **Step 6.3.4: Update existing call sites to populate `expected` / `found` / `fix`**

Audit Section A inbound-deps surfaces every site that raises a typed error. Update each so the user-facing message follows the four-part shape. Example for the test case above:

```python
# src/custom_sam_peft/config/loader.py (illustrative)
raise ConfigError(
    "configured path does not exist",
    field_path="data.train.path",
    expected="an existing directory",
    found=f"{path!r} (does not exist)",
    fix=f"create the directory or update data.train.path in your config",
)
```

- [ ] **Step 6.3.5: Confirm internals do NOT catch typed errors to re-raise as RuntimeError**

```bash
rg -n 'except CustomSamPeftError|except ConfigError|except DataError|except ModelError|except CheckpointError|except EnvironmentError' \
   src/custom_sam_peft/ --type py
```
Expected post-fix: hits ONLY in `src/custom_sam_peft/cli/main.py` (the boundary catch). Internals raise; they do not catch.

- [ ] **Step 6.3.6: Run all tests**

```bash
pytest tests/unit/test_cli_error_rendering.py tests/unit/test_errors.py \
       tests/unit/test_cli*.py -v
```
Expected: PASS.

- [ ] **Step 6.3.7: Commit**

```bash
git add src/custom_sam_peft/errors.py src/custom_sam_peft/cli/main.py \
        src/custom_sam_peft/ tests/unit/test_cli_error_rendering.py \
        tests/unit/test_errors.py
git commit -m "feat(errors): four-part error rendering at CLI boundary"
```

---

## Task 7: Consumer migration [PARALLEL subtasks 7.1–7.4]

After Task 6, all consumers must be updated to the new schema + CLI surface. **No migrator tool** (spec §1.3, §8).

### Task 7.1: `configs/examples/` rewrite

**Files:**
- Modify: `configs/examples/coco_text_lora.yaml`
- Modify: `configs/examples/coco_text_qlora.yaml`
- Modify: `configs/examples/gpu_smoke_lora.yaml`
- Modify: `configs/examples/gpu_smoke_qlora.yaml`
- (Audit may surface more.)

- [ ] **Step 7.1.1: Rewrite each example against the new schema**

Use the v0.7.0 schema from `docs/config-schema.md` as the source of truth. Apply the audit Section F rename table.

- [ ] **Step 7.1.2: Parse each example through `load_config`**

```bash
pytest tests/unit/test_config_examples.py -v
```
Expected: PASS for every file in `configs/examples/`.

- [ ] **Step 7.1.3: Commit**

```bash
git add configs/examples/
git commit -m "feat(configs): rewrite example configs against v0.7.0 schema"
```

### Task 7.2: Colab notebook migration

**Files:**
- Modify: `notebooks/custom_sam_peft_train.ipynb`
- Modify: `notebooks/README.md`

- [ ] **Step 7.2.1: Update each cell**

For each cell in `notebooks/custom_sam_peft_train.ipynb`:
- CLI invocations: change to new command names + flags (`!custom-sam-peft train ... --eval --export` or `!custom-sam-peft run ...`).
- Config field names: apply audit Section F rename table.
- Python API imports: use `from custom_sam_peft import run_train, run_eval, run_export, EvalArtifacts`.

- [ ] **Step 7.2.2: Regenerate output cells**

Outputs (screenshots, training-log snippets, metrics tables) regenerate by running the notebook locally CPU-only (where feasible) or with the GPU-skipped cells annotated as such. The orchestrator will verify end-to-end run on GPU post-merge (spec §3 #7).

- [ ] **Step 7.2.3: Rewrite `notebooks/README.md`**

Reflects the new command names and YAML field names. Mirrors the README's "Beginner — train in Colab" section style.

- [ ] **Step 7.2.4: Commit**

```bash
git add notebooks/custom_sam_peft_train.ipynb notebooks/README.md
git commit -m "docs(notebook): migrate Colab notebook to v0.7.0 schema and CLI surface"
```

### Task 7.3: Cloud launch scripts

**Files:**
- Modify: `cloud/runpod/` (every script that invokes the CLI or references config fields)

- [ ] **Step 7.3.1: Update each script**

```bash
rg -n 'esam3|sam-peft|custom-sam-peft' cloud/runpod/
```
Identify every command invocation and config-field reference. Update to v0.7.0 surface.

- [ ] **Step 7.3.2: Commit**

```bash
git add cloud/runpod/
git commit -m "feat(cloud): migrate RunPod launch scripts to v0.7.0 CLI surface"
```

### Task 7.4: README + top-level docs

**Files:**
- Modify: `README.md`
- Modify: `notebooks/README.md` (if not already done in 7.2)
- Modify: `CHANGELOG.md` (the v0.7.0 entry — full content lands in Task 9)

- [ ] **Step 7.4.1: Update `README.md`**

Sections requiring change:
- "Beginner — train in Colab" — new CLI commands.
- "Configuration" — link to `docs/config-schema.md` (the OQ3 file).
- "Quickstart" / "Usage" — new flag examples (`--eval`, `--export`, `run`).
- Any "v0.6.x" references → "v0.7.0".

- [ ] **Step 7.4.2: Commit**

```bash
git add README.md
git commit -m "docs(readme): update for v0.7.0 schema, CLI surface, config-schema link"
```

---

## Task 8: Dead-code sweep

**Files:**
- Delete: `src/esam3/` (already absent per branch check, but spec §10 calls it out — confirm + no-op or delete if reintroduced)
- Modify: various — driven by audit Section I

- [ ] **Step 8.1: Confirm `src/esam3/` is absent**

```bash
test -d src/esam3 && echo "STILL PRESENT — delete" || echo "OK (already absent)"
```
If still present, `rm -rf src/esam3/` and verify no references via `rg -n esam3 src/ tests/`.

- [ ] **Step 8.2: Run `mcp__code-review-graph__refactor_tool`**

Surface unreachable functions / classes / files. Apply audit Section I — delete what is confirmed dead, EXCEPT:

**Retained (spec §10 + §2 — DO NOT remove):**
- `Runtime.is_primary` and `Runtime.world_size` fields.
- `Tracker.is_primary` (or equivalent rank-awareness field on the Tracker protocol) if present.
- Any other §2 seam scaffolding flagged in audit Section I as "retain — seam scaffolding".

- [ ] **Step 8.3: Delete "just-in-case" hooks with zero callers**

Audit Section I lists them. Remove. No deprecation warnings, no shims (pre-1.0).

- [ ] **Step 8.4: Run the full test suite**

```bash
pytest tests/ -v --ignore=tests/gpu
```
Expected: PASS (CPU path). GPU-marked tests stay skipped on local — they ran during PR #58's GPU pass; this PR does not re-litigate (spec §9.4).

- [ ] **Step 8.5: Commit**

```bash
git add -u src/ tests/
git commit -m "chore: dead-code sweep per audit Section I; retain §2 seam scaffolding"
```

---

## Task 9: Release — version bump, CHANGELOG, follow-up issues, PR

This is the **final pre-PR task** (mandatory per planner instructions).

### Task 9.1: Run the full test suite + static guards

- [ ] **Step 9.1.1: Run everything CPU-only**

```bash
pytest tests/ -v --ignore=tests/gpu
```
Expected: PASS. If any failure, fix in place and re-run before proceeding.

- [ ] **Step 9.1.2: Confirm static guards are all green**

```bash
pytest tests/unit/test_static_guards.py -v
```
Expected: all three guards PASS.

- [ ] **Step 9.1.3: Run linting / formatting**

```bash
# Adjust to whatever the project uses — ruff / black / isort.
ruff check src/ tests/
ruff format src/ tests/ --check
```
Expected: clean. Fix any issues in place.

### Task 9.2: Version bump to v0.7.0

- [ ] **Step 9.2.1: Discover every version-carrying manifest**

```bash
rg -l '"?version"?\s*[:=]'
```
Expected hits at minimum: `pyproject.toml`, `uv.lock`. The audit may surface more.

- [ ] **Step 9.2.2: Update each to 0.7.0**

```bash
# pyproject.toml
sed -i 's/^version = .*/version = "0.7.0"/' pyproject.toml
```

```bash
# uv.lock — regenerate from the updated pyproject.toml
uv lock
```

For any additional manifests (audit-surfaced — e.g., `src/custom_sam_peft/__init__.py` `__version__`, conda `meta.yaml`, etc.): update each.

- [ ] **Step 9.2.3: Verify**

```bash
rg -n '"?version"?\s*[:=]\s*"?(0\.7\.0|0\.6)' --type-add 'lock:*.lock' --type lock --type py --type toml
```
Expected: every match shows `0.7.0`. No leftover `0.6.x`.

### Task 9.3: CHANGELOG.md v0.7.0 entry

- [ ] **Step 9.3.1: Author the CHANGELOG entry**

```markdown
## [0.7.0] — 2026-05-21

### Breaking — v0.x debt paydown ("hardening pass", issue #26)

This release rewrites the YAML schema, CLI surface, and internal seams to
make the user-facing API small and obvious. Upgrade by editing your YAML
manually against the rename table below — there is intentionally no
migration tool.

#### YAML field renames

| Old | New | Notes |
| --- | --- | --- |
| <audit Section F entry 1> | <new> | <one-line rationale> |
| ...

(Populated verbatim from `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md` Section F.)

#### Removed fields

- <list from audit Section E "delete" verdicts>

#### Demoted fields (no longer user-set; hardcoded as internal defaults)

- <list from audit Section E "demote" verdicts>

#### CLI command flag changes

- `train` gains bare `--eval` and `--export` flags.
- `eval` gains a bare `--export` flag.
- `run` is now documented as an alias for `train --eval --export`.

#### Errors

- New error taxonomy: `CustomSamPeftError`, `ConfigError`, `DataError`,
  `ModelError`, `CheckpointError`, `EnvironmentError`.
- CLI renders errors in a four-part shape (summary / expected / found / fix).
  Re-run with `-v` for the full traceback.

#### Internals

- `Runtime` value object centralizes device + dtype + rank-awareness.
- `paths/` module owns the run-dir layout.
- `_bootstrap.py` is the sole site for registration, seeding, and logging.
- Each `_patch_*` lives in its own file under `src/custom_sam_peft/models/_patches/`.
- `EvalArtifacts` is the seam between Trainer and Evaluator.

### See also

- Audit inventory: `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md`
- Config schema reference: `docs/config-schema.md`
- Design spec: `docs/superpowers/specs/2026-05-21-hardening-pass-design.md`
```

- [ ] **Step 9.3.2: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock <other version manifests>
git commit -m "release: bump to v0.7.0; CHANGELOG entry with rename table + removed fields"
```

### Task 9.4: Open `hardening-followup` issues for deferred audit items

- [ ] **Step 9.4.1: Confirm the label exists**

Already verified in Step 0.4. If somehow missing (e.g., orchestrator session restarted), re-run:

```bash
gh label list --limit 200 | grep -E '^hardening-followup\b' || gh label create hardening-followup \
  --description "Audit-surfaced items deferred from the hardening pass" \
  --color BFD4F2
```

- [ ] **Step 9.4.2: Open one issue per audit Section J item**

For each item:

```bash
gh issue create \
  --title "<proposed title from audit Section J>" \
  --body "$(cat <<EOF
Deferred from the v0.7.0 hardening pass (PR #<this PR number>, issue #26).

**Audit reference:** \`docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md\` Section J item N

**Description:** <verbatim from audit>

**Suggested next step:** <verbatim from audit if present, otherwise "Schedule into a future PR.">
EOF
  )" \
  --label hardening-followup \
  --assignee @me
```

(The issue numbers can be recorded in the orchestrator session log; they are NOT committed to the repo.)

- [ ] **Step 9.4.3: Confirm every audit Section J item has an open issue**

```bash
gh issue list --label hardening-followup --state open --json number,title,createdAt
```

Cross-check the count against Section J's item count.

### Task 9.5: Verify Acceptance Criteria

Walk through spec §12 — every checkbox must resolve to "satisfied" or "deferred with `hardening-followup` issue".

- [ ] **Step 9.5.1: Audit inventory exists.** Verify `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md` is committed.
- [ ] **Step 9.5.2: Static guards green.** `pytest tests/unit/test_static_guards.py -v` PASS.
- [ ] **Step 9.5.3: `sam3.py` substantially shrinks.** `wc -l src/custom_sam_peft/models/sam3.py` < pre-task line count (numbers in audit Section A).
- [ ] **Step 9.5.4: `_patch_*` extracted.** `ls src/custom_sam_peft/models/_patches/` shows one file per patch.
- [ ] **Step 9.5.5: `src/esam3/` deleted.** `test ! -d src/esam3 && echo OK`.
- [ ] **Step 9.5.6: Each Pydantic config class** is user-facing or marked internal.
- [ ] **Step 9.5.7: `EvalArtifacts`** is the only object the evaluator consumes from the trainer (verified by `tests/integration/test_trainer_evaluator_seam.py`).
- [ ] **Step 9.5.8: `Sam3Patches.apply`** is the sole application site (verified by `_ALL_PATCHES` membership).
- [ ] **Step 9.5.9: Error taxonomy** has 5 subclasses; `doctor` reuses `EnvironmentError`.
- [ ] **Step 9.5.10: CLI bare flags exist.** `custom-sam-peft train --help` shows `--eval` and `--export` flags; no `--with-*` / `--then-*` strings.
- [ ] **Step 9.5.11: Trainer ↔ evaluator seam test.** `pytest tests/integration/test_trainer_evaluator_seam.py -v` PASS.
- [ ] **Step 9.5.12: Tracker swap test.** `pytest tests/integration/test_tracker_swap.py -v` PASS.
- [ ] **Step 9.5.13: PEFT extensibility test.** `pytest tests/integration/test_peft_extensibility.py -v` PASS.
- [ ] **Step 9.5.14: Configs migrated.** Every `configs/examples/*.yaml` parses through `load_config`.
- [ ] **Step 9.5.15: Notebook migrated.** Manual check — orchestrator will verify GPU end-to-end post-merge.
- [ ] **Step 9.5.16: Cloud scripts migrated.** `rg -n 'esam3' cloud/` returns nothing.
- [ ] **Step 9.5.17: CHANGELOG `v0.7.0` entry present.**
- [ ] **Step 9.5.18: No migrator tool.** `rg -n 'migrate.config|migrate_config|upgrade.config|upgrade_config' src/ cli/ scripts/` returns nothing.
- [ ] **Step 9.5.19: Version stamped 0.7.0.** `rg -n '0\.6' pyproject.toml uv.lock` returns nothing.
- [ ] **Step 9.5.20: Test suite green.** `pytest tests/ --ignore=tests/gpu` PASS.
- [ ] **Step 9.5.21: Audit items not addressed have follow-up issues.** Cross-checked in Step 9.4.3.

### Task 9.6: Open PR

- [ ] **Step 9.6.1: Push the branch**

```bash
git push -u origin spec/hardening-pass
```

- [ ] **Step 9.6.2: Open the PR with the spec-§11 description shape**

```bash
gh pr create --title "v0.7.0: hardening pass — SOLID/DRY/YAGNI sweep (closes #26)" --body "$(cat <<'EOF'
## TL;DR

v0.7.0 ships breaking-by-design YAML and CLI changes that pay down v0.x debt. Upgrade by editing your YAML against the rename table in `CHANGELOG.md`. There is no migration tool — pre-1.0 README already declares this.

## User-visible payoff (lead — spec §11)

1. **New YAML schema.** See `docs/config-schema.md` for the field-by-field reference. Rename table in `CHANGELOG.md`.
2. **New CLI examples.**
   - `custom-sam-peft train cfg.yaml --eval --export` — full pipeline.
   - `custom-sam-peft run cfg.yaml` — documented alias for the above.
   - `custom-sam-peft eval cfg.yaml --export` — eval-then-export.
3. **Field rename table.** In `CHANGELOG.md`.
4. **Removed fields / flags.** In `CHANGELOG.md`.

## Internals refactor (substrate)

These changes are what made the user-facing redesign safe:

- **Shared primitives:** `errors.py`, `paths/`, `runtime/`, `config/_internal.py`, `_bootstrap.py` (centralized).
- **Seam cleanups:** PEFT method-string leaks moved behind `PEFTMethod` protocol; `EvalArtifacts` between Trainer and Evaluator; tracking backends only imported inside `tracking/`; CLI internals are thin wrappers over `run_train` / `run_eval` / `run_export`.
- **God-function decomposition:** `load_sam31`, `Trainer.fit`, `Evaluator.evaluate`, `HFDataset.__getitem__`, `COCODataset.__getitem__`, `write_bundle`, `apply_qlora` all decomposed per spec §6. Each `_patch_*` extracted to its own file under `models/_patches/`.
- **Static guards** in CI enforce: no method-string branches outside `peft_adapters/`, no `.to(device)` outside collator + `runtime/`, no string-joined `runs/.../checkpoints/` outside `paths/`.
- **Dead code removed** per audit Section I; `§2` seam scaffolding (`is_primary`, `world_size`) retained.

## Verification

- Test suite green (CPU path). GPU-marked tests stay GPU-marked per the existing GPU test policy.
- Static guards pass: `pytest tests/unit/test_static_guards.py -v`.
- All audit-deferred items have `hardening-followup` issues opened.

## References

- Design spec: `docs/superpowers/specs/2026-05-21-hardening-pass-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-21-hardening-pass.md`
- Audit inventory: `docs/superpowers/specs/2026-05-21-hardening-audit-inventory.md`
- Config schema reference: `docs/config-schema.md`

Closes #26
EOF
)" --assignee @me
```

- [ ] **Step 9.6.3: Watch CI**

Per CLAUDE.md: use `run_in_background` / Monitor — no polling sleeps. Notify the user only when CI is green. On failure, fix and re-loop.

- [ ] **Step 9.6.4: PR-ready CI trigger (project memory)**

If CI was originally skipped on draft, force a push (or merge from main) to re-fire — `gh pr ready` alone does not. Existing PR was opened with `gh pr create` (non-draft) above, so this should not apply, but check:

```bash
gh pr view --json isDraft -q .isDraft
```

Expected: `false`. If `true` for any reason, mark ready and force-push an empty commit:

```bash
git commit --allow-empty -m "ci: kick"
git push
gh pr ready
```

### Task 9.7: Close-out on merge (orchestrator standard close-out per CLAUDE.md)

Not enumerated here — universal across plans. The orchestrator's close-out:
- Tags the merge (`v0.7.0`).
- Kills background processes.
- Folds the branch log into `logs/logs.md`.
- Removes the worktree.
- Signs off.

---

## Acceptance Criteria (mirrors spec §12; orchestrator verifies before signing off)

Spec §12 has 22 checkboxes. Coverage map — each is implemented by the listed plan task(s):

1. Audit inventory exists at the canonical path → Task 1.
2. No `if .*\.method ==` outside `peft_adapters/` → Tasks 3 (guard scaffold), 4.1 (refactor).
3. No `\.to(device` outside collator + `runtime/` → Tasks 3, 4 (relevant subtasks).
4. No string-joined `runs/.../checkpoints/` outside `paths/` → Tasks 3, 5.5 (and other subtasks that touch paths).
5. `sam3.py` substantially shrinks; one file per `_patch_*` → Tasks 5.1, 5.7.
6. `src/esam3/` deleted → Task 8.1.
7. Pydantic config classes user-facing or internal → Task 2.4 (per audit Section G).
8. `EvalArtifacts` is the sole evaluator-from-trainer object → Task 4.3.
9. `Sam3Patches.apply` is the sole patch-application site → Tasks 2.3.5, 5.7.
10. Error taxonomy exists; doctor reuses `EnvironmentError` → Tasks 2.1, 6.3.
11. CLI bare `--eval` / `--export` flags; `run` is alias → Tasks 4.4 (internals), 6.2 (surface).
12. Trainer ↔ evaluator seam test → Task 4.3.6.
13. Tracker swap test → (added under Task 4.2 — see note below).
14. PEFT extensibility test → (added under Task 4.1 — see note below).
15. `configs/examples/` migrated → Task 7.1.
16. Notebook migrated (verified post-merge by orchestrator) → Task 7.2.
17. `cloud/runpod/` + launch scripts migrated → Task 7.3.
18. CHANGELOG `v0.7.0` entry → Task 9.3.
19. No migrator tool present → Task 9.5.18 verification.
20. Version stamped `0.7.0` everywhere → Task 9.2.
21. Test suite green on CI → Task 9.1, Task 9.6.3.
22. Audit-deferred items have `hardening-followup` issues → Task 9.4.

> **Coverage notes for criteria 13 and 14:** the spec §9.1 also requires "PEFT extensibility test (OCP proof)" and "Tracker swap-in/swap-out test." Author these alongside their respective refactor subtasks. To make this explicit:

- **Tracker swap test (criterion 13):** add `tests/integration/test_tracker_swap.py` during Task 4.2. Parameterize over `NoopTracker`, a fake recording tracker (in-place class for the test), and the offline `WandbTracker`. Assert the trainer makes the same protocol calls regardless of backend.

- **PEFT extensibility test (criterion 14):** add `tests/integration/test_peft_extensibility.py` + `tests/fixtures/stub_peft_adapter.py` during Task 4.1. The fixture registers a stub adapter via `@register("peft", "stub")`; the integration test runs a tiny `fit` against it and asserts the trainer accepts it without any code changes outside `tests/fixtures/`.

Both tests are part of the §9.1 seam-test triad alongside the trainer↔evaluator seam test (Task 4.3.6).
