# Root Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ten tracked root files into `.config/`, `docs/`, and `.github/` subdirectories, update all in-repo references, and verify CI tooling discovers configs at their new paths.

**Architecture:** Three independent `git mv` groups (lint configs, dev docs, community docs), each followed immediately by reference updates so the repo never has dangling links between commits. A final verification task runs `pre-commit`, triggers the three affected hooks individually, and `git grep`s for every old path.

**Tech Stack:** git, pre-commit, yamllint, markdownlint-cli2, gitleaks (download step in security.yml is used locally for smoke-testing only if the binary is available; CI is the authoritative gitleaks check).

---

## File Map

| Action | From | To |
|---|---|---|
| Move | `.gitleaks.toml` | `.config/gitleaks.toml` |
| Move | `.yamllint.yml` | `.config/yamllint.yml` |
| Move | `.markdownlint-cli2.jsonc` | `.config/markdownlint-cli2.jsonc` |
| Move | `.markdownlint.json` | `.config/markdownlint.json` |
| Move | `ARCHITECTURE.md` | `docs/ARCHITECTURE.md` |
| Move | `README-dev.md` | `docs/README-dev.md` |
| Move | `RELEASING.md` | `docs/RELEASING.md` |
| Move | `CONTRIBUTING.md` | `.github/CONTRIBUTING.md` |
| Move | `CODE_OF_CONDUCT.md` | `.github/CODE_OF_CONDUCT.md` |
| Move | `SECURITY.md` | `.github/SECURITY.md` |
| Modify | `.github/workflows/security.yml` | Add `--config .config/gitleaks.toml` (2 occurrences) |
| Modify | `.github/workflows/ci.yml` | Add `-c .config/yamllint.yml` and `--config .config/markdownlint-cli2.jsonc` |
| Modify | `README.md` | Update 3 root-relative references |
| Modify | `README-dev.md` → `docs/README-dev.md` | Update 1 root-relative reference |
| Modify | `CONTRIBUTING.md` → `.github/CONTRIBUTING.md` | Update 1 root-relative reference |

**Note on `.github/ISSUE_TEMPLATE/config.yml`:** This file contains a hard-coded GitHub blob URL
(`…/blob/main/CONTRIBUTING.md`) that points at the old root location. After `CONTRIBUTING.md` moves
to `.github/CONTRIBUTING.md`, GitHub resolves it from `.github/` automatically in its UI banners,
but this blob URL will 404. It must be updated to `…/blob/main/.github/CONTRIBUTING.md`.

---

## Task 1: Move lint configs to `.config/` and update CI invocations

**Files:**
- Create dir: `.config/`
- Move: `.gitleaks.toml` → `.config/gitleaks.toml`
- Move: `.yamllint.yml` → `.config/yamllint.yml`
- Move: `.markdownlint-cli2.jsonc` → `.config/markdownlint-cli2.jsonc`
- Move: `.markdownlint.json` → `.config/markdownlint.json`
- Modify: `.github/workflows/security.yml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1.1: Create `.config/` and `git mv` the four lint configs**

```bash
mkdir -p .config
git mv .gitleaks.toml    .config/gitleaks.toml
git mv .yamllint.yml     .config/yamllint.yml
git mv .markdownlint-cli2.jsonc .config/markdownlint-cli2.jsonc
git mv .markdownlint.json       .config/markdownlint.json
```

Expected: `git status` shows four renames staged (e.g. `renamed: .gitleaks.toml -> .config/gitleaks.toml`). No error output.

- [ ] **Step 1.2: Verify no cross-reference inside `.markdownlint-cli2.jsonc`**

The spec notes that `.markdownlint-cli2.jsonc` does not currently embed a reference to `.markdownlint.json`, but requires the planner to confirm. The current content of `.config/markdownlint-cli2.jsonc` (after the move) is:

```jsonc
// markdownlint-cli2 configuration
{
  "config": {
    "MD013": false,
    "MD018": false,
    "MD029": false
  },
  "ignores": [".venv/**"]
}
```

There is no `configFile` key — no update to the file content is needed. If a future edit adds a `configFile` key pointing at `.markdownlint.json`, that path must be updated to `.config/markdownlint.json`.

- [ ] **Step 1.3: Update `security.yml` — add `--config .config/gitleaks.toml` (2 occurrences)**

Open `.github/workflows/security.yml`. There are two `run:` lines that invoke `./gitleaks detect`:

**Occurrence 1 — PR step (line ~65):**
- Before: `run: ./gitleaks detect --no-banner --redact --verbose`
- After:  `run: ./gitleaks detect --no-banner --redact --verbose --config .config/gitleaks.toml`

**Occurrence 2 — push step (line ~69):**
- Before: `run: ./gitleaks detect --no-banner --redact --verbose --log-opts "${{ github.event.before }}..${{ github.sha }}"`
- After:  `run: ./gitleaks detect --no-banner --redact --verbose --config .config/gitleaks.toml --log-opts "${{ github.event.before }}..${{ github.sha }}"`

- [ ] **Step 1.4: Update `ci.yml` — add `-c .config/yamllint.yml` (1 occurrence)**

Open `.github/workflows/ci.yml`. Find the yamllint step (line ~84):

- Before: `run: uv run --with yamllint yamllint .`
- After:  `run: uv run --with yamllint yamllint -c .config/yamllint.yml .`

- [ ] **Step 1.5: Update `ci.yml` — add `--config .config/markdownlint-cli2.jsonc` (1 occurrence)**

Find the markdownlint step (line ~87):

- Before: `run: npx --yes markdownlint-cli2 "**/*.md" "#node_modules"`
- After:  `run: npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"`

- [ ] **Step 1.6: Verify yamllint and markdownlint-cli2 find their configs locally**

```bash
# yamllint: should exit 0 (config at new path)
uv run --with yamllint yamllint -c .config/yamllint.yml .

# markdownlint-cli2: should exit 0 (config at new path)
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"
```

Expected: both commands exit 0 with no errors.

- [ ] **Step 1.7: Confirm no old config paths remain in tracked files**

```bash
git grep -l '\.gitleaks\.toml\|\.yamllint\.yml\|\.markdownlint-cli2\.jsonc\|\.markdownlint\.json'
```

Expected output: only files where these strings appear as the *new* `.config/` paths (e.g. `ci.yml`, `security.yml`, `.config/` directory itself) — NOT any bare root-relative reference like `".gitleaks.toml"`. If any unexpected hit appears, fix it before committing.

- [ ] **Step 1.8: Commit**

```bash
git add .config/ .github/workflows/security.yml .github/workflows/ci.yml
git commit -m "chore: move lint configs to .config/, update CI invocations"
```

---

## Task 2: Move developer docs to `docs/` and update references

**Files:**
- Move: `ARCHITECTURE.md` → `docs/ARCHITECTURE.md`
- Move: `README-dev.md` → `docs/README-dev.md`
- Move: `RELEASING.md` → `docs/RELEASING.md`
- Modify: `README.md` (3 references)
- Modify: `docs/README-dev.md` after move (1 reference inside the file)

- [ ] **Step 2.1: `git mv` the three developer doc files**

```bash
git mv ARCHITECTURE.md docs/ARCHITECTURE.md
git mv README-dev.md   docs/README-dev.md
git mv RELEASING.md    docs/RELEASING.md
```

Expected: `git status` shows three renames staged.

- [ ] **Step 2.2: Update references in `README.md`**

`README.md` currently references all three files with root-relative paths. Make these three edits:

**Edit 1 — bare reference to ARCHITECTURE.md (line ~86):**
- Before: `See `ARCHITECTURE.md` for the module map and data flow.`
- After:  `See `docs/ARCHITECTURE.md` for the module map and data flow.`

**Edit 2 — link to README-dev.md (line ~91):**
- Before: `[`README-dev.md`](README-dev.md). See`
- After:  `[`README-dev.md`](docs/README-dev.md). See`

**Edit 3 — link to CONTRIBUTING.md (line ~92):**
This link will be updated in Task 3. Leave it for now — it points at root-relative `CONTRIBUTING.md`, which still exists at the root until Task 3 runs. Do not update it here.

- [ ] **Step 2.3: Update the cross-reference inside `docs/README-dev.md`**

`README-dev.md` (now at `docs/README-dev.md`) contains this line (line ~44):

- Before: `See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and data flow.`
- After:  `See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and data flow.`

Both `docs/README-dev.md` and `docs/ARCHITECTURE.md` now live in the same directory, so the bare `ARCHITECTURE.md` relative link is already correct — **no change needed**.

- [ ] **Step 2.4: Verify no stale root-relative references to the moved docs**

```bash
git grep -rn 'ARCHITECTURE\.md\|README-dev\.md\|RELEASING\.md' -- '*.md' '*.yml' '*.yaml' '*.toml' '*.json' '*.jsonc'
```

Examine every hit. Hits inside `docs/superpowers/{specs,plans}/` are historical plan/spec text — they refer to the old root layout and **do not need updating** (those are frozen archival documents). Hits in live files (`README.md`, `docs/README-dev.md`, `docs/ARCHITECTURE.md`, workflow YAML) must all use the new path. If any live file still references a bare root-relative path for these three files, fix it before committing.

- [ ] **Step 2.5: Commit**

```bash
git add docs/ARCHITECTURE.md docs/README-dev.md docs/RELEASING.md README.md
git commit -m "chore: move ARCHITECTURE.md, README-dev.md, RELEASING.md to docs/"
```

---

## Task 3: Move community docs to `.github/` and update references

**Files:**
- Move: `CONTRIBUTING.md` → `.github/CONTRIBUTING.md`
- Move: `CODE_OF_CONDUCT.md` → `.github/CODE_OF_CONDUCT.md`
- Move: `SECURITY.md` → `.github/SECURITY.md`
- Modify: `README.md` (1 reference — the CONTRIBUTING.md link deferred from Task 2)
- Modify: `.github/CONTRIBUTING.md` after move (1 reference inside the file)
- Modify: `.github/ISSUE_TEMPLATE/config.yml` (1 blob URL)

- [ ] **Step 3.1: `git mv` the three community doc files**

```bash
git mv CONTRIBUTING.md   .github/CONTRIBUTING.md
git mv CODE_OF_CONDUCT.md .github/CODE_OF_CONDUCT.md
git mv SECURITY.md        .github/SECURITY.md
```

Expected: `git status` shows three renames staged.

- [ ] **Step 3.2: Update the CONTRIBUTING.md link in `README.md`**

`README.md` line ~92 (the link deferred from Task 2):

- Before: `[`CONTRIBUTING.md`](CONTRIBUTING.md) for the project's contribution`
- After:  `[`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for the project's contribution`

- [ ] **Step 3.3: Update the README-dev.md cross-reference inside `.github/CONTRIBUTING.md`**

`.github/CONTRIBUTING.md` (line ~22) references `README-dev.md` with a root-relative link. After the move in Task 2, `README-dev.md` is now at `docs/README-dev.md`. Since `CONTRIBUTING.md` is now inside `.github/`, the relative path to reach `docs/README-dev.md` from `.github/` is `../docs/README-dev.md`.

- Before: `See [`README-dev.md`](README-dev.md) for the dev loop (uv, ruff, mypy,`
- After:  `See [`README-dev.md`](../docs/README-dev.md) for the dev loop (uv, ruff, mypy,`

- [ ] **Step 3.4: Update blob URL in `.github/ISSUE_TEMPLATE/config.yml`**

`config.yml` line ~4 contains a hard-coded GitHub blob URL pointing at the old root location:

- Before: `url: https://github.com/NguyenJus/Efficient-SAM3-Finetuning/blob/main/CONTRIBUTING.md`
- After:  `url: https://github.com/NguyenJus/Efficient-SAM3-Finetuning/blob/main/.github/CONTRIBUTING.md`

- [ ] **Step 3.5: Verify no stale root-relative references to the moved community docs**

```bash
git grep -rn 'CONTRIBUTING\.md\|CODE_OF_CONDUCT\.md\|SECURITY\.md' -- '*.md' '*.yml' '*.yaml' '*.toml' '*.json' '*.jsonc'
```

Examine every hit. Hits inside `docs/superpowers/{specs,plans}/` are frozen archival documents — no action needed. All live-file hits must reference `.github/CONTRIBUTING.md`, `.github/CODE_OF_CONDUCT.md`, or `.github/SECURITY.md` (or the full blob URL with `.github/`). If any live file still has a bare root-relative reference, fix it.

- [ ] **Step 3.6: Commit**

```bash
git add .github/CONTRIBUTING.md .github/CODE_OF_CONDUCT.md .github/SECURITY.md \
        README.md .github/ISSUE_TEMPLATE/config.yml
git commit -m "chore: move CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md to .github/"
```

---

## Task 4: End-to-end verification

- [ ] **Step 4.1: Run `pre-commit` on all files**

```bash
pre-commit run --all-files
```

Expected: exit 0. The current `.pre-commit-config.yaml` only invokes `ruff` and `nbstripout` — neither touches the moved configs — so this confirms nothing broke in the hook chain.

- [ ] **Step 4.2: Trigger yamllint with the new config path**

```bash
uv run --with yamllint yamllint -c .config/yamllint.yml .
```

Expected: exit 0, no errors or warnings. This confirms config discovery works at the new path (a config-not-found error from yamllint would produce a non-zero exit).

- [ ] **Step 4.3: Trigger markdownlint-cli2 with the new config path**

```bash
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"
```

Expected: exit 0. Confirms `.config/markdownlint-cli2.jsonc` is found and applied.

- [ ] **Step 4.4: Final `git grep` sweep for all old root-relative paths**

Run each of the following and confirm zero hits in live files (hits only in `docs/superpowers/{specs,plans}/` are acceptable and expected — those are frozen archival docs):

```bash
# Developer docs
git grep -n 'ARCHITECTURE\.md'    -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n 'README-dev\.md'      -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n 'RELEASING\.md'       -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'

# Community docs
git grep -n 'CONTRIBUTING\.md'    -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n 'CODE_OF_CONDUCT\.md' -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n 'SECURITY\.md'        -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'

# Lint configs
git grep -n '\.gitleaks\.toml'          -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n '\.yamllint\.yml'           -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n '\.markdownlint-cli2\.jsonc' -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
git grep -n '\.markdownlint\.json'      -- '*.md' '*.yml' '*.yaml' '*.json' '*.jsonc' '*.toml'
```

For each lint-config grep: live hits should only appear inside `.config/` and the workflow files (at their new `.config/` paths). Any hit of the bare root-relative form (e.g. `.gitleaks.toml` without a leading `.config/`) in a live file is a stale reference — fix before marking this task done.

For each doc grep: all live-file hits must use the new paths (`docs/ARCHITECTURE.md`, `.github/CONTRIBUTING.md`, etc.). Only `docs/superpowers/{specs,plans}/` hits may reference old paths.

- [ ] **Step 4.5: Verify root layout matches spec**

```bash
git ls-files | grep -v '/' | sort
```

Expected output (the 8 tracked root files, no more):

```
.gitignore
.pre-commit-config.yaml
.python-version
CITATION.cff
LICENSE
README.md
pyproject.toml
uv.lock
```

If any other tracked file appears in this list, it was missed — move it per the spec or investigate why it was omitted.

- [ ] **Step 4.6: Final commit (if any fixups were made in this task)**

If steps 4.1–4.5 required any fixup edits, stage and commit them:

```bash
git add -p   # review each fixup interactively
git commit -m "chore: fixup stale references found during root-audit verification"
```

If no fixups were needed, skip this step.

---

## Acceptance Criteria (all must hold before marking complete)

1. `pre-commit run --all-files` exits 0.
2. `uv run --with yamllint yamllint -c .config/yamllint.yml .` exits 0.
3. `npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"` exits 0.
4. `git ls-files | grep -v '/' | sort` shows exactly 8 files (`.gitignore`, `.pre-commit-config.yaml`, `.python-version`, `CITATION.cff`, `LICENSE`, `README.md`, `pyproject.toml`, `uv.lock`).
5. The `git grep` sweeps in Step 4.4 show no stale root-relative references in any live (non-archival) file.
6. CI passes on the draft PR (gitleaks security workflow, yamllint, markdownlint steps in `ci.yml` all green).
