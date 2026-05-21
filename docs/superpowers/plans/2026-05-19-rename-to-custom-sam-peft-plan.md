# Rename to `custom-sam-peft` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-19-rename-to-custom-sam-peft-design.md` (source of truth; read first).

**Goal:** Rename the in-tree project from `Efficient-SAM3-Finetuning` / `esam3` to `custom-sam-peft` / `custom_sam_peft` end-to-end (Phase 1 of the spec), with no compatibility shims.

**Architecture:** A single keystone `git mv` of the package directory unblocks file-disjoint tasks (pyproject, CI, docs, notebooks, tests). Substitution rules are case-sensitive and applied scoped to active files only -- `docs/superpowers/`, `uv.lock`, `LICENSE`, `.git/`, and `.worktrees/` are off-limits. The plan is parameterized on eight tokens (below) so a later name change requires only updating that block.

**Tech Stack:** `uv`, `hatch`, `ruff`, `mypy`, `pytest`, GitHub Actions, `git mv`, code-review-graph MCP.

**Tooling note:** This repo provides a code-review-graph MCP knowledge graph. Implementers should prefer graph tools (`semantic_search_nodes`, `query_graph`, `get_impact_radius`, `detect_changes`) over Grep/Glob/Read for code exploration and impact analysis. Fall back to Grep only when the graph does not cover the question (e.g. raw text scans across markdown / YAML / notebooks).

---

## Substitution tokens (parameter block)

Updating these eight tokens is the **only** change required if the user picks a different final name later. The rest of this plan references tokens, not literal strings, where natural.

| Token                 | Value                                   |
| --------------------- | --------------------------------------- |
| `${OLD_NAME_KEBAB}`   | `Efficient-SAM3-Finetuning`             |
| `${OLD_NAME_LOWER}`   | `efficient-sam3-finetuning`             |
| `${OLD_PKG}`          | `esam3`                                 |
| `${NEW_NAME_KEBAB}`   | `custom-sam-peft`                       |
| `${NEW_NAME_SNAKE}`   | `custom_sam_peft`                       |
| `${NEW_CLI}`          | `custom-sam-peft`                       |
| `${OLD_GH_PATH}`      | `NguyenJus/Efficient-SAM3-Finetuning`   |
| `${NEW_GH_PATH}`      | `NguyenJus/custom-sam-peft`             |

---

## Out of plan, handled by orchestrator close-out

Phases 2-5 of the spec are deferred to the orchestrator and are **not** decomposed into implementer tasks here:

- **Phase 2 -- Merge** (spec § Phase 2 -- Merge). Orchestrator marks the draft PR ready, watches CI, user merges.
- **Phase 3 -- GitHub rename** (spec § Phase 3 -- GitHub rename (post-merge)). `gh repo rename`, repo description, topics.
- **Phase 4 -- Update local git remote** (spec § Phase 4 -- Update local git remote).
- **Phase 5 -- Local directory rename + worktree metadata fix** (spec § Phase 5 -- Local directory rename (orchestrator close-out, outside worktree)). Must happen from outside the worktree, after killing background processes.

Implementer tasks below cover Phase 1 only.

---

## Review gate

After all Phase 1 tasks are complete and Task 11 (final verification) passes, the orchestrator runs `superpowers:requesting-code-review` before marking the draft PR ready (per CLAUDE.md orchestrator step 3). Linting/formatting is run as the **last** step of the final reviewer pass; Task 11 enforces this.

---

## File-disjointness map (parallelization basis)

After the keystone (Task 1), the following task groups touch disjoint file sets and can run in parallel:

- **Group A -- pyproject + CI configs:** `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/bug_report.yml`. (Tasks 2, 3.)
- **Group B -- Active docs:** `README.md`, `README-dev.md`, `ARCHITECTURE.md`, `CITATION.cff`, `CONTRIBUTING.md`, `RELEASING.md`, `SECURITY.md`, `docs/public-flip-runbook.md`, `docs/testing/gpu-test-policy.md`, `cloud/runpod/README.md`, `scripts/public-flip-bootstrap.sh`, `scripts/run_gpu_tests.sh`. (Tasks 4, 5, 6.)
- **Group C -- Notebooks:** `notebooks/custom_sam_peft_train.ipynb` (renamed in Task 1), `notebooks/colab_gpu_tests.ipynb`. (Task 7.)
- **Group D -- Python source + tests global substitution:** `src/custom_sam_peft/**/*.py`, `tests/**/*.py`, `scripts/**/*.py`, `configs/**`. (Task 8.)

Tasks within a group that share a file must serialize; cross-group tasks can fan out.

---

## Task 1: Keystone -- rename package directory and training notebook

**Model/effort:** sonnet / high.
**Parallel:** No. **Blocks all downstream tasks.**
**Spec:** § Filesystem -- directories and files to rename; § Phase 1 steps 1-2.

**Files:**
- Rename: `src/${OLD_PKG}/` -> `src/${NEW_NAME_SNAKE}/`
- Rename: `notebooks/${OLD_PKG}_train.ipynb` -> `notebooks/${NEW_NAME_SNAKE}_train.ipynb`

**Goal:** Move the package directory and training-notebook filename to their new names using `git mv` so git tracks the rename. No content edits in this task -- only path changes.

**Acceptance criteria:**
- `src/${NEW_NAME_SNAKE}/__init__.py` exists; `src/${OLD_PKG}/` does not.
- `notebooks/${NEW_NAME_SNAKE}_train.ipynb` exists; `notebooks/${OLD_PKG}_train.ipynb` does not.
- `git status` shows both as renames (`R`), not delete+add.
- No file contents modified.

**Steps:**

- [ ] **Step 1: Rename package directory**

```bash
git mv src/esam3 src/custom_sam_peft
```

- [ ] **Step 2: Rename training notebook**

```bash
git mv notebooks/esam3_train.ipynb notebooks/custom_sam_peft_train.ipynb
```

- [ ] **Step 3: Verify rename, not copy**

```bash
git status --short
```

Expected: lines starting with `R  src/esam3/... -> src/custom_sam_peft/...` and `R  notebooks/esam3_train.ipynb -> notebooks/custom_sam_peft_train.ipynb`. No `A` (added) entries for these paths.

- [ ] **Step 4: Commit the keystone**

```bash
git add -A
git commit -m "rename: move src/esam3 -> src/custom_sam_peft (keystone)"
```

---

## Task 2: `pyproject.toml` updates

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 3-8 (file-disjoint).
**Depends on:** Task 1.
**Spec:** § `pyproject.toml`; § Phase 1 step 3.

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore` (one hit: `!src/esam3/runs/` -> `!src/custom_sam_peft/runs/`)

**Goal:** Update every `pyproject.toml` key listed in the spec's `pyproject.toml` table. Align the `version` field to `0.5.0` (matches README; spec § Non-goals clarifies this is an alignment fix, not a release). Rephrase `[project].description` to drop SAM3.1 version-lock language. Update the negated-ignore rule in `.gitignore` to point at the new package dir.

**Acceptance criteria:**
- All keys in spec § `pyproject.toml` reflect new values.
- `[project].version = "0.5.0"`.
- `[project].description` no longer mentions `SAM3.1` (use SAM-family phrasing consistent with the spec's repo-description suggestion).
- `grep -nE '(esam3|efficient-sam3-finetuning)' pyproject.toml .gitignore` returns zero hits.
- `uv sync --all-extras` succeeds and updates `uv.lock` automatically.

**Steps:**

- [ ] **Step 1: Apply the substitutions**

Apply, in order:
1. `[project].name`: `efficient-sam3-finetuning` -> `custom-sam-peft`.
2. `[project].version`: `0.0.1` -> `0.5.0`.
3. `[project].description`: rewrite to drop SAM3.1 version-lock (e.g. `"Closed-vocab finetuning of SAM-family models with LoRA / QLoRA on a single consumer GPU"`).
4. `[project.scripts]`: `esam3 = "esam3.cli.main:app"` -> `custom-sam-peft = "custom_sam_peft.cli.main:app"`.
5. `[tool.hatch.build.targets.wheel].packages`: `["src/esam3"]` -> `["src/custom_sam_peft"]`.
6. `[tool.ruff.lint.per-file-ignores]` keys: `"src/esam3/cli/*_cmd.py"` -> `"src/custom_sam_peft/cli/*_cmd.py"`.
7. `[tool.mypy].files`: `["src/esam3"]` -> `["src/custom_sam_peft"]`.
8. `[tool.pytest.ini_options].addopts`: `--cov=esam3` -> `--cov=custom_sam_peft`.
9. `[tool.coverage.run].source`: `["src/esam3"]` -> `["src/custom_sam_peft"]`.

- [ ] **Step 2: Verify no residue**

```bash
grep -nE '(esam3|efficient-sam3-finetuning)' pyproject.toml .gitignore
```

Expected: no output.

- [ ] **Step 3: Regenerate the lockfile**

```bash
uv sync --all-extras
```

Expected: succeeds; `uv.lock` updates. Do not hand-edit `uv.lock`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "build: rename distribution to custom-sam-peft, align version to 0.5.0"
```

---

## Task 3: CI workflows and GitHub templates

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 2, 4-8 (file-disjoint).
**Depends on:** Task 1.
**Spec:** § CI / workflows.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/PULL_REQUEST_TEMPLATE.md`
- Modify: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Verify (no change expected): `.github/workflows/security.yml`, `.github/workflows/codeql.yml`, `.github/workflows/pr-colab-badge.yml`

**Goal:** Update the one known `mypy src/esam3` invocation in `ci.yml`, the issue/PR templates, and verify the other workflows are clean. `pr-colab-badge.yml` uses `${GITHUB_REPOSITORY}` so it auto-resolves on `gh repo rename` -- do not hard-code a slug.

**Acceptance criteria:**
- `.github/workflows/ci.yml` invokes `mypy src/custom_sam_peft` (no `esam3`).
- `.github/PULL_REQUEST_TEMPLATE.md` and `.github/ISSUE_TEMPLATE/bug_report.yml` reference `${NEW_NAME_KEBAB}` and `${NEW_NAME_SNAKE}` where appropriate.
- `grep -rIi --include='*.yml' --include='*.yaml' --include='*.md' 'esam3\|Efficient-SAM3-Finetuning' .github/` returns zero hits.

**Steps:**

- [ ] **Step 1: Update `ci.yml`**

Replace `mypy src/esam3` with `mypy src/custom_sam_peft`. Sweep any other `esam3` / `Efficient-SAM3-Finetuning` hits in the file using the substitution rules from spec § Phase 1 step 4.

- [ ] **Step 2: Update templates**

Apply the same substitutions to `.github/PULL_REQUEST_TEMPLATE.md` (1 hit per spec) and `.github/ISSUE_TEMPLATE/bug_report.yml` (4 hits per spec).

- [ ] **Step 3: Verify other workflows are clean**

```bash
grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' .github/workflows/security.yml .github/workflows/codeql.yml .github/workflows/pr-colab-badge.yml
```

Expected: no output. If a hit appears, apply the standard substitutions.

- [ ] **Step 4: Verify `.github/` is clean**

```bash
grep -rIE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' .github/
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add .github/
git commit -m "ci: rename references to custom-sam-peft"
```

---

## Task 4: README.md -- rename references + "3 clicks" detune

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 2, 3, 5-8 (file-disjoint).
**Depends on:** Task 1.
**Spec:** § Active docs (non-archived) (README.md row); § Adjacent copy edits (in scope for this PR).

**Files:**
- Modify: `README.md`

**Goal:** Apply the standard substitutions to all 13 hits, **and** replace the "Beginner -- train in 3 clicks" section header (and any matching marketing prose at the top of that section) with neutral phrasing, per spec § Adjacent copy edits.

**Acceptance criteria:**
- `grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' README.md` returns zero hits.
- `grep -nE '3 clicks|in seconds|just |easy ' README.md` returns zero hits (spec constraint: no marketing language, no exclamation marks in the replaced section).
- Section header is factual (e.g. "Beginner -- train in Colab" per spec suggestion), mentions Colab, and notes no local GPU setup required.
- Numbered steps inside the section are unchanged.

**Steps:**

- [ ] **Step 1: Apply substitutions**

Run the spec's substitutions (in order, case-sensitive) over `README.md`:
1. `Efficient-SAM3-Finetuning` -> `custom-sam-peft`
2. `efficient-sam3-finetuning` -> `custom-sam-peft`
3. `esam3` -> `custom_sam_peft`

Review prose hits manually for awkward casing (e.g. heading-initial `esam3`).

- [ ] **Step 2: Detune the "3 clicks" section**

Replace the section header and any matching intro prose with neutral phrasing. Constraints (spec § Adjacent copy edits):
- No `!`, no `just`, no `easy`, no `in seconds`, no `instantly`.
- Factual tone, consistent with the rest of the README.
- Still mentions Colab and that no local GPU setup is required.
- Numbered steps stay as-is.

- [ ] **Step 3: Verify**

```bash
grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning|3 clicks' README.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): rename to custom-sam-peft; detune '3 clicks' framing"
```

---

## Task 5: ARCHITECTURE.md + remaining top-level docs

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 2, 3, 4, 6-8 (file-disjoint with each other; this task is the only one touching these files).
**Depends on:** Task 1.
**Spec:** § Active docs (non-archived).

**Files:**
- Modify: `README-dev.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CITATION.cff`
- Modify: `CONTRIBUTING.md`
- Modify: `RELEASING.md`
- Modify: `SECURITY.md`

**Goal:** Apply the standard substitutions across the remaining top-level docs and `CITATION.cff`. For `ARCHITECTURE.md`, the module-map paths (`src/esam3/...`) need updating too.

**Acceptance criteria:**
- `grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' README-dev.md ARCHITECTURE.md CITATION.cff CONTRIBUTING.md RELEASING.md SECURITY.md` returns zero hits.
- `CITATION.cff` `title` and `repository-code` fields reflect the new name and `${NEW_GH_PATH}`.
- `ARCHITECTURE.md` module-map paths point at `src/${NEW_NAME_SNAKE}/...`.

**Steps:**

- [ ] **Step 1: Apply substitutions**

Run the spec's substitutions over each file (in order, case-sensitive).

- [ ] **Step 2: Verify**

```bash
grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' README-dev.md ARCHITECTURE.md CITATION.cff CONTRIBUTING.md RELEASING.md SECURITY.md
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add README-dev.md ARCHITECTURE.md CITATION.cff CONTRIBUTING.md RELEASING.md SECURITY.md
git commit -m "docs: rename references to custom-sam-peft across top-level docs"
```

---

## Task 6: Runbooks, GPU-test policy, runpod readme, scripts

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 2-5, 7-8 (file-disjoint).
**Depends on:** Task 1.
**Spec:** § Active docs (non-archived) (lower rows).

**Files:**
- Modify: `docs/public-flip-runbook.md`
- Modify: `docs/testing/gpu-test-policy.md`
- Modify: `cloud/runpod/README.md`
- Modify: `scripts/public-flip-bootstrap.sh`
- Modify: `scripts/run_gpu_tests.sh` (one hit: comment string `'esam3'`)

**Goal:** Apply the standard substitutions across the remaining active docs and both in-scope shell scripts.

**Acceptance criteria:**
- `grep -rIE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' docs/public-flip-runbook.md docs/testing/gpu-test-policy.md cloud/runpod/README.md scripts/public-flip-bootstrap.sh scripts/run_gpu_tests.sh` returns zero hits.
- Clone URLs in `cloud/runpod/README.md` point at `${NEW_GH_PATH}`.

**Steps:**

- [ ] **Step 1: Apply substitutions**

Run the spec's substitutions over each file (in order, case-sensitive). Pay attention to URL forms in `cloud/runpod/README.md` (clone commands).

- [ ] **Step 2: Verify**

```bash
grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' docs/public-flip-runbook.md docs/testing/gpu-test-policy.md cloud/runpod/README.md scripts/public-flip-bootstrap.sh scripts/run_gpu_tests.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add docs/public-flip-runbook.md docs/testing/gpu-test-policy.md cloud/runpod/README.md scripts/public-flip-bootstrap.sh scripts/run_gpu_tests.sh
git commit -m "docs: rename references to custom-sam-peft in runbooks and runpod readme"
```

---

## Task 7: Notebooks -- in-cell string substitution

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 2-6, 8 (file-disjoint).
**Depends on:** Task 1.
**Spec:** § Filesystem -- directories and files to rename (notebook note); § Phase 1 step 4 (in-scope `notebooks/`).

**Files:**
- Modify: `notebooks/${NEW_NAME_SNAKE}_train.ipynb` (renamed by Task 1)
- Modify: `notebooks/colab_gpu_tests.ipynb`

**Goal:** Apply the spec's substitutions to in-cell strings in both notebooks. The training notebook's filename was changed in Task 1; this task handles its cell contents.

**Acceptance criteria:**
- `grep -E 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' notebooks/*.ipynb` returns zero hits.
- Notebook JSON remains valid (no broken cell metadata).

**Steps:**

- [ ] **Step 1: Apply substitutions to notebook JSON**

Run the spec's substitutions (in order, case-sensitive) over both notebook files. Treat them as text -- the substitutions only touch string literals inside cell `source` arrays and any inline output paths.

- [ ] **Step 2: Verify JSON validity**

```bash
python -c "import json,sys; [json.load(open(p)) for p in ['notebooks/custom_sam_peft_train.ipynb','notebooks/colab_gpu_tests.ipynb']]; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Verify no residue**

```bash
grep -nE 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' notebooks/*.ipynb
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "notebooks: rename references to custom-sam-peft"
```

---

## Task 8: Python source + tests -- global substitution

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Tasks 2-7 (file-disjoint -- Python files only).
**Depends on:** Task 1.
**Spec:** § Python source -- imports and string literals; § Phase 1 step 4 (in-scope `src/`, `tests/`, `scripts/`, `configs/`); § Phase 1 step 5 (CLI rename in source).

**Files:**
- Modify: `src/${NEW_NAME_SNAKE}/**/*.py` (97 unique files contain `from esam3` / `import esam3`; 317 total import-line occurrences per spec).
- Modify: `tests/**/*.py` (53 test files import `esam3.*` symbols per spec).
- Modify: `scripts/**/*.py` (if any contain `esam3`).
- Modify: `configs/**` (if any contain `esam3`).
- Specifically including:
  - `src/${NEW_NAME_SNAKE}/__init__.py` (docstring + `__version__` import path).
  - `src/${NEW_NAME_SNAKE}/_bootstrap.py` (9-line internal import block).
  - `src/${NEW_NAME_SNAKE}/cli/main.py` (CLI help text + program name -- update hard-coded `esam3` references to `${NEW_CLI}` / `${NEW_NAME_SNAKE}` as appropriate).

**Goal:** Replace every `esam3` identifier and string literal in active Python code with `${NEW_NAME_SNAKE}`. Update CLI help text and program name in `src/${NEW_NAME_SNAKE}/cli/main.py` if hard-coded.

**Acceptance criteria:**
- `grep -rIE 'esam3' src/ tests/ scripts/ configs/` returns zero hits.
- `python -c "import custom_sam_peft; print(custom_sam_peft.__version__)"` succeeds (uses Task 2's `pyproject.toml` version through the package's `__version__` import path).
- `uv run python -m custom_sam_peft.cli.main --help` mentions `custom-sam-peft` and not `esam3`.

**Steps:**

- [ ] **Step 1: Substitute `esam3` -> `custom_sam_peft` in Python source**

Apply substitution rule 3 from spec § Phase 1 step 4 (`esam3` -> `custom_sam_peft`) across `src/`, `tests/`, `scripts/`, `configs/`. Use a text substitution tool; this is purely identifier replacement.

- [ ] **Step 2: Update CLI help / program name**

Inspect `src/custom_sam_peft/cli/main.py` for hard-coded `esam3` strings in help text, `prog_name`, or app config. Replace with `${NEW_CLI}` (`custom-sam-peft`) for the user-facing CLI name and `${NEW_NAME_SNAKE}` for module references.

- [ ] **Step 3: Verify no residue**

```bash
grep -rIE 'esam3' src/ tests/ scripts/ configs/
```

Expected: no output.

- [ ] **Step 4: Smoke-test import**

```bash
uv run python -c "import custom_sam_peft; print(custom_sam_peft.__version__)"
```

Expected: prints `0.5.0` (or whatever Task 2 set).

- [ ] **Step 5: Commit**

```bash
git add src/ tests/ scripts/ configs/
git commit -m "refactor: rename esam3 -> custom_sam_peft across Python source and tests"
```

---

## Task 9: Residual sweep -- catch-all for missed files

**Model/effort:** sonnet / high.
**Parallel:** No. Must run **after** Tasks 2-8 complete.
**Depends on:** Tasks 2-8.
**Spec:** § Phase 1 step 4 (in-scope file list); § Verification checklist.

**Files:**
- Modify: any active files (per spec § Phase 1 step 4 in-scope list) still containing residue.
- Out-of-scope (do not touch): `docs/superpowers/`, `LICENSE`, `uv.lock` (regenerated by `uv sync`), `.git/`, `.worktrees/`.

**Goal:** Catch any active-tree files Tasks 2-8 missed. Common offenders: dotfile configs (`.gitleaks.toml`, `.pre-commit-config.yaml`, etc.) that were not enumerated in the spec's inventory tables but are still in scope per the Phase 1 step 4 sweep rules.

**Acceptance criteria:**
- The full active-tree grep returns zero hits (see Step 1 below).

**Steps:**

- [ ] **Step 1: Run the active-tree grep**

```bash
grep -rIE --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.worktrees --exclude-dir=.code-review-graph --exclude-dir=superpowers 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' .
```

Expected: no output. If output appears, the file is active-tree residue and must be fixed in this task.

- [ ] **Step 2: Apply spec substitutions to any residue**

For each residue hit, apply (in order):
1. `Efficient-SAM3-Finetuning` -> `custom-sam-peft`
2. `efficient-sam3-finetuning` -> `custom-sam-peft`
3. `esam3` -> `custom_sam_peft`

- [ ] **Step 3: Re-run grep until clean**

```bash
grep -rIE --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.worktrees --exclude-dir=.code-review-graph --exclude-dir=superpowers 'esam3|Efficient-SAM3-Finetuning|efficient-sam3-finetuning' .
```

Expected: no output.

- [ ] **Step 4: Filename sweep**

```bash
find . -name "*esam3*" -not -path "./.git/*" -not -path "./.worktrees/*" -not -path "./docs/superpowers/*" -not -path "./.code-review-graph/*"
```

Expected: empty.

- [ ] **Step 5: Commit (only if residue was found)**

```bash
git add -A
git commit -m "chore: sweep residual esam3 references"
```

If Step 1 was clean, skip the commit.

---

## Task 10: Risk-mitigation checks

**Model/effort:** haiku / high (read-only verification, no edits).
**Parallel:** No. Must run **after** Task 9.
**Depends on:** Task 9.
**Spec:** § Risks and mitigations.

**Goal:** Mirror the spec's risks table. Run read-only checks to confirm that nothing was touched that should not have been touched. **No file edits in this task.** If a check fails, surface as a blocker -- do not attempt to fix here.

**Acceptance criteria:** All checks below pass. None modify files.

**Steps:**

- [ ] **Step 1: Confirm `.git/worktrees` metadata is untouched**

```bash
git status --short .git/ 2>/dev/null
ls -la .git
```

Expected: no `.git/` paths in `git status` output (they would be ignored anyway, but verify nothing was accidentally added); `.git` is a pointer file referencing the parent project. **Do not edit these files** -- worktree metadata is rewritten by the orchestrator during Phase 5 close-out.

- [ ] **Step 2: Confirm `docs/superpowers/` is untouched in this branch's diff**

```bash
git diff --name-only main...HEAD -- docs/superpowers/
```

Expected: only files under `docs/superpowers/specs/` and `docs/superpowers/plans/` dated `2026-05-19` for this rename (i.e. the new design + plan). No edits to archived specs/plans.

- [ ] **Step 3: Confirm `uv.lock` was regenerated, not hand-edited**

```bash
git diff main...HEAD -- uv.lock | head -40
```

Expected: changes limited to the package name field (`efficient-sam3-finetuning` -> `custom-sam-peft`) and any hash/version-line updates that `uv sync` produced. If the diff includes unrelated edits (whitespace, reordering not driven by the name change), `uv.lock` was hand-edited -- block and ask the user.

- [ ] **Step 4: Confirm `LICENSE` is untouched**

```bash
git diff --name-only main...HEAD -- LICENSE
```

Expected: empty.

- [ ] **Step 5: Confirm sibling worktree was not touched**

```bash
git diff --name-only main...HEAD -- .worktrees/
```

Expected: empty. (`.worktrees/` is typically gitignored, but verify nothing was accidentally added.)

- [ ] **Step 6: Confirm CLAUDE.md, `RELEASING.md`, and other governance docs only have rename edits (no policy drift)**

```bash
git diff main...HEAD -- CLAUDE.md RELEASING.md
```

Expected: only `esam3` / `Efficient-SAM3-Finetuning` -> new-name substitutions; no rewording.

---

## Task 11: Final verification (spec verification checklist)

**Model/effort:** sonnet / high.
**Parallel:** No. Must run **last**, after Task 10.
**Depends on:** Task 10.
**Spec:** § Verification checklist (post-phase-1 block).

**Goal:** Run the spec's verification checklist verbatim. This is the gate for the orchestrator's `superpowers:requesting-code-review` step. **Linting/formatting runs as the last sub-step**, per CLAUDE.md orchestrator step 3.

**Acceptance criteria:** Every check in the spec's "After phase 1 (pre-merge)" block passes.

**Steps:**

- [ ] **Step 1: Grep checks (zero hits)**

```bash
grep -rIi --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.worktrees --exclude-dir=.code-review-graph --exclude-dir=superpowers "esam3" .
```

Expected: zero hits.

```bash
grep -rIi --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.worktrees --exclude-dir=.code-review-graph --exclude-dir=superpowers "Efficient-SAM3-Finetuning" .
```

Expected: zero hits.

- [ ] **Step 2: Filename sweep**

```bash
find . -name "*esam3*" -not -path "./.git/*" -not -path "./.worktrees/*" -not -path "./docs/superpowers/*" -not -path "./.code-review-graph/*"
```

Expected: empty.

- [ ] **Step 3: `uv sync`**

```bash
uv sync --all-extras
```

Expected: succeeds. No lockfile drift after the run.

- [ ] **Step 4: `mypy`**

```bash
uv run mypy src/custom_sam_peft
```

Expected: clean.

- [ ] **Step 5: `pytest`**

```bash
uv run pytest
```

Expected: passes. (Spec § Verification checklist; respects existing GPU-gating policy.)

- [ ] **Step 6: CLI smoke test**

```bash
uv run custom-sam-peft --help
```

Expected: help text mentions `custom-sam-peft`. If a `doctor` subcommand exists, also run:

```bash
uv run custom-sam-peft doctor
```

Expected: runs without import error.

- [ ] **Step 7 (LAST): Lint + format check**

Run linting/formatting **last** per CLAUDE.md orchestrator step 3:

```bash
uv run ruff check
uv run ruff format --check
```

Expected: both clean. Address any issues directly (apply `ruff format` and re-run; for lint issues, fix in-place and re-run the relevant prior steps).

- [ ] **Step 8: Commit any lint/format fixes**

If Step 7 required edits:

```bash
git add -A
git commit -m "style: ruff format after rename"
```

If Step 7 was clean, skip the commit.

---

## Self-review

**1. Spec coverage:** Each row of spec § Inventory of rename targets maps to a task: filesystem renames -> Task 1; `pyproject.toml` -> Task 2; CI/workflows + templates -> Task 3; README -> Task 4 (incl. adjacent copy edit); ARCHITECTURE + top-level docs -> Task 5; runbooks/runpod/scripts -> Task 6; notebooks -> Task 7; Python source + tests -> Task 8; residual sweep -> Task 9. Spec § Risks and mitigations -> Task 10. Spec § Verification checklist -> Task 11. Phases 2-5 -> Out-of-plan section.

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or "add appropriate X" language. Every task has explicit commands and acceptance criteria.

**3. Type consistency:** Token names are used consistently (`${NEW_NAME_SNAKE}`, `${NEW_NAME_KEBAB}`, `${NEW_CLI}`). Verification commands use the literal `custom-sam-peft` / `custom_sam_peft` (rather than tokens) because shell commands must be executable as-written; this is acceptable because token substitution is a manual step the user performs once if the name changes.

**4. Keystone gating:** Task 1 is explicitly the keystone; Tasks 2-8 list `Depends on: Task 1`; Task 9 depends on 2-8; Task 10 depends on 9; Task 11 depends on 10.

**5. Parallelism:** Tasks 2-8 are file-disjoint as documented in the File-disjointness map; can fan out after Task 1. Tasks 9, 10, 11 serialize.

**6. Resolved decisions only:** No leftover `csp`, `csam`, `v0.0.1`, or `v0.1.0` references in the plan. Version `0.5.0` per spec § Non-goals.
