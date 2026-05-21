# Root Audit — Design Spec

**Issue:** [#49 — chore: audit root files for moves/deletes](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/49)
**Date:** 2026-05-19
**Status:** Design approved

---

## 1. Problem Statement

The repository root currently holds 18 tracked files (dotfiles + visible) — far more than tooling
and contributor-convention require. Many of these files were deposited at the root as a default of
convenience and have no technical reason to stay there. The excess creates two practical costs:

- **Discoverability:** contributors scanning the root must mentally filter clutter to find what
  matters.
- **Config sprawl:** four lint-tool config files sit at the root solely because that is where their
  generators put them, not because the tools require it.

The goal is to shrink the visible root footprint to only what tooling convention mandates plus the
two files contributors look for first.

---

## 2. Relationship to Issue #49 Checklist

Issue #49 was opened before the public-flip PR (#55) merged. Several items on its checklist are
already resolved:

| Checklist item | Resolution |
|---|---|
| `.coverage` | Already gitignored by the `.gitignore` added in public-flip. |
| `CLAUDE.md` | Already gitignored. |
| `.mcp.json` | Already gitignored. |
| `logs/` | Already gitignored. |
| Cache directories (`.venv/`, `__pycache__/`, etc.) | Already gitignored. |

The open items — which this spec addresses — are the tracked files that legitimately exist on disk
but belong in a different location.

---

## 3. Scope

This is a **pure layout/hygiene change**. No source code logic, training configurations,
notebooks, scripts, or test content is altered. No tracked file that is in active use is deleted.

Explicitly out of scope:

- Any content under `src/`, `tests/`, `cloud/`, `configs/`, `notebooks/`, `scripts/`.
- Inlining lint config into `pyproject.toml`. None of gitleaks, yamllint, markdownlint-cli2, or
  pre-commit read configuration from `pyproject.toml`; inlining is not feasible.
- Moving `.pre-commit-config.yaml`. pre-commit's git hook hard-codes the root path and provides no
  override mechanism.
- Adding new `.gitignore` entries for items that are already gitignored.

---

## 4. What Stays at Root

### 4.1 Tooling-fixed (cannot be moved without breaking the tool)

These tools look for their files at the repo root and have no hook-friendly override:

| File | Reason |
|---|---|
| `.gitignore` | Git mandates root location. |
| `.python-version` | pyenv and uv resolve this from the repo root. |
| `.pre-commit-config.yaml` | pre-commit's git hook hard-codes root discovery. |
| `pyproject.toml` | Python packaging, ruff, and uv all require root placement. |
| `uv.lock` | uv resolves the lockfile relative to `pyproject.toml`. |

### 4.2 Convention (should stay)

| File | Reason |
|---|---|
| `README.md` | Universal contributor landing page; GitHub renders it automatically. |
| `LICENSE` | GitHub license detection reads only from the root. |
| `CITATION.cff` | GitHub's citation widget reads exclusively from the repo root. |

### 4.3 Untouched directories

These directories are verified in active use and are not touched by this work:

| Directory | Active references |
|---|---|
| `.github/` | GitHub-mandated; workflows, templates. |
| `cloud/` | `cloud/runpod/README.md` linked from `README.md` and the simplify-ux spec. |
| `configs/` | `configs/examples/*.yaml` referenced from tests, `README.md`, and multiple specs. |
| `data/` | Contains `.gitkeep`; runtime path expected by code. |
| `models/` | Contains `.gitkeep`; runtime path expected by code. |
| `docs/` | Specs, plans, architecture docs — all in active use. |
| `notebooks/`, `scripts/`, `src/`, `tests/` | All in active use. |

The `.gitkeep` files in `data/` and `models/` are **not** replaced with `README.md` files — they
are minimal and purpose-built, and introducing Markdown there would add markdownlint surface without
benefit.

---

## 5. The Moves

### 5.1 Lint configs → `.config/`

Four lint-tool configuration files move from root to `.config/`. The `.config/` directory is an
established XDG-aligned convention for project-level tool configuration and is already understood by
all four tools when told where to look.

| From | To |
|---|---|
| `.gitleaks.toml` | `.config/gitleaks.toml` |
| `.yamllint.yml` | `.config/yamllint.yml` |
| `.markdownlint-cli2.jsonc` | `.config/markdownlint-cli2.jsonc` |
| `.markdownlint.json` | `.config/markdownlint.json` |

**Tool invocation updates required:**

Each tool that is currently invoked without an explicit `--config` flag relies on automatic root
discovery. After the move, the config path must be supplied explicitly wherever the tool is called:

- **`gitleaks`** — called in `.github/workflows/security.yml` via `./gitleaks detect ...` (two
  steps: PR and push). Both steps must add `--config .config/gitleaks.toml`.
- **`yamllint`** — called in `.github/workflows/ci.yml` via `uv run --with yamllint yamllint .`.
  This step must add `-c .config/yamllint.yml`.
- **`markdownlint-cli2`** — called in `.github/workflows/ci.yml` via
  `npx --yes markdownlint-cli2 "**/*.md" "#node_modules"`. This step must add
  `--config .config/markdownlint-cli2.jsonc`.

**Note on `.markdownlint-cli2.jsonc` and `.markdownlint.json`:** The two files are independent
configs (`.markdownlint-cli2.jsonc` does not embed a reference to `.markdownlint.json`). Both move
as a pair because they serve the same tool family. If markdownlint-cli2 is ever configured to
reference `.markdownlint.json` by path (e.g., via a `configFile` key), that path must be updated to
`.config/markdownlint.json`. The planner should verify the final content of each file and update any
such cross-reference during the move.

**Note on pre-commit:** The current `.pre-commit-config.yaml` only contains `ruff` and `nbstripout`
hooks — it does not invoke gitleaks, yamllint, or markdownlint-cli2. No pre-commit args need
updating.

### 5.2 Developer docs → `docs/`

Three developer-oriented documentation files move into `docs/` (preserving their filenames):

| From | To |
|---|---|
| `ARCHITECTURE.md` | `docs/ARCHITECTURE.md` |
| `README-dev.md` | `docs/README-dev.md` |
| `RELEASING.md` | `docs/RELEASING.md` |

**References to update:** `README.md` currently links to `ARCHITECTURE.md`, `README-dev.md`, and
`CONTRIBUTING.md` (see Section 5.3 below) with root-relative paths. These links must be updated to
the new paths. Any other tracked Markdown that links to these files by their old root paths must
also be updated; the planner should confirm the full reference list via `grep` before the move.

### 5.3 Community/governance docs → `.github/`

GitHub discovers `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` from either the repo
root or the `.github/` directory. Moving them to `.github/` retains full GitHub UI integration
(contributor guidelines banner, security advisories, etc.) while removing them from the root.

| From | To |
|---|---|
| `CONTRIBUTING.md` | `.github/CONTRIBUTING.md` |
| `CODE_OF_CONDUCT.md` | `.github/CODE_OF_CONDUCT.md` |
| `SECURITY.md` | `.github/SECURITY.md` |

**References to update:** `README.md` links to `CONTRIBUTING.md`. Any other tracked files
referencing these by root-relative path must be updated. The planner should confirm the full
reference list via `grep` before the move.

---

## 6. Resulting Root Layout

After all moves, the tracked root contains:

```text
.config/                .github/                cloud/          configs/
data/                   docs/                   models/         notebooks/
scripts/                src/                    tests/
.gitignore              .pre-commit-config.yaml .python-version
CITATION.cff            LICENSE                 README.md
pyproject.toml          uv.lock
```

**Tracked root file count:** 18 → 8 (dotfiles + visible files, excluding directories).
**Visible (non-dotfile) root files:** 11 → 5 (gains one new directory: `.config/`).

---

## 7. Success Criteria

The implementation is complete when all of the following hold:

1. `pre-commit run --all-files` passes on the new layout without errors.
2. Each moved lint tool finds its config at the new path — verified by intentionally triggering each
   hook so that a config-not-found error would surface.
3. No broken relative links remain in `README.md` or any other tracked Markdown file referencing
   the moved files. Verified by `grep`-ing for each old path across the tracked tree.
4. CI passes on the open draft PR (all workflow steps green, including the gitleaks security
   workflow and the yamllint/markdownlint steps in `ci.yml`).

---

## 8. Non-Goals (Explicitly Excluded)

- Touching any code, config, or test logic.
- Merging `.markdownlint.json` into `.markdownlint-cli2.jsonc`.
- Inlining any lint config into `pyproject.toml`.
- Adding, removing, or modifying `.gitignore` entries.
- Moving `CITATION.cff`, `README.md`, or `LICENSE` — all stay at root.
- Renaming any file (only moves, no renames).
