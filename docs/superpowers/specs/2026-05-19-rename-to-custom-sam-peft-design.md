# Rename: `Efficient-SAM3-Finetuning` -> `custom-sam-peft`

## Goal

Rename the project end-to-end -- GitHub repo, Python distribution, import
package, CLI entry point, and every in-tree string -- from
`Efficient-SAM3-Finetuning` / `esam3` to `custom-sam-peft` /
`custom_sam_peft`, with no backwards-compatibility shims. The new name
is provisional and is treated as a parameter so the user can substitute
a different final name later with minimal spec churn.

## Why

- "Efficient" conflates with PEFT and misleads laypersons skimming the
  README; the project's real claim is closed-vocab finetuning on a single
  consumer GPU, not a novel efficiency technique.
- "SAM3" version-locks the name. User plans to support SAM 1/2 later;
  the name should outlive a single backbone.
- "Custom" expresses closed-vocabulary semantics without ML jargon
  (user supplies a fixed class list -- closed by construction).
- "PEFT" is precise for the LoRA / QLoRA implementation actually in the
  repo.
- Accessibility / "easy" framing belongs in the tagline, not the name.
- Solo public project, pre-1.0, no external dependents: a clean break is
  cheaper than maintaining aliases.

## Non-goals

- No feature changes. No behavior changes. No refactors beyond
  identifier renames and string updates.
- No compatibility shims: no aliased `esam3` console script, no
  re-exported `esam3` import package, no deprecation period.
- No git-history rewrites. Old commits keep their original strings.
- No archived spec/plan rewrites. The `docs/superpowers/{specs,plans}/`
  history is a record of how the code got here and stays as written;
  only the *active* docs (README, ARCHITECTURE, CITATION, runbooks,
  templates) get updated.
- No PyPI publish in this change. The distribution rename
  (`efficient-sam3-finetuning` -> `custom-sam-peft`) is purely a local
  package-name change; first publish happens separately.
- No release event. The version field in `pyproject.toml` is currently
  stale (`0.0.1`) while the README states `v0.5.0`; the rename PR
  aligns `pyproject.toml` to the README's stated `0.5.0`. This is an
  alignment fix to record reality, not a release bump.
- Two adjacent in-scope edits accompany the rename: (a) detuning the
  README's "3 clicks" framing (see "Adjacent copy edits" below), and
  (b) the `pyproject.toml` version alignment described above. All
  other prose / feature / refactor changes remain out of scope.

## Name as parameter

Treat these tokens as substitutable so the final name can change with
minimal spec churn:

| Token                 | Value                       | Where it appears                         |
| --------------------- | --------------------------- | ---------------------------------------- |
| `${OLD_NAME_KEBAB}`   | `Efficient-SAM3-Finetuning` | Repo slug, GitHub URLs, badges, prose    |
| `${OLD_NAME_LOWER}`   | `efficient-sam3-finetuning` | `pyproject.toml` distribution name       |
| `${OLD_PKG}`          | `esam3`                     | `src/`, every Python import, CLI script  |
| `${NEW_NAME_KEBAB}`   | `custom-sam-peft`           | New repo slug, distribution name         |
| `${NEW_NAME_SNAKE}`   | `custom_sam_peft`           | New import package (`src/<NEW_SNAKE>`)   |
| `${NEW_CLI}`          | `custom-sam-peft`           | New `[project.scripts]` console entry    |
| `${OLD_GH_PATH}`      | `NguyenJus/Efficient-SAM3-Finetuning` | GitHub URLs, Colab links     |
| `${NEW_GH_PATH}`      | `NguyenJus/custom-sam-peft` | GitHub URLs, Colab links                 |

Substituting the name later means updating these eight tokens in the
spec and plan; the inventory, operation ordering, and verification
sections are written against the tokens, not the literal strings.

## Inventory of rename targets

Counts come from `grep -rIi` over the working tree (excluding `.git`,
`.venv`, `.worktrees`, `.code-review-graph`).

### Filesystem -- directories and files to rename

| Path (current)                          | Path (new)                                    | Notes                                  |
| --------------------------------------- | --------------------------------------------- | -------------------------------------- |
| `src/esam3/`                            | `src/custom_sam_peft/`                        | One `git mv`. All subpackages move with it. |
| `notebooks/esam3_train.ipynb`           | `notebooks/custom_sam_peft_train.ipynb`       | Filename change + in-cell strings.     |

`notebooks/colab_gpu_tests.ipynb` keeps its filename; only in-cell
strings change.

### `pyproject.toml`

| Key                                | Current                          | New                          |
| ---------------------------------- | -------------------------------- | ---------------------------- |
| `[project].name`                   | `efficient-sam3-finetuning`      | `custom-sam-peft`            |
| `[project].version`                | `0.0.1`                          | `0.5.0` (align with README)  |
| `[project].description`            | mentions SAM3.1 specifically     | rephrase, drop version-lock  |
| `[project.scripts]`                | `esam3 = "esam3.cli.main:app"`   | `custom-sam-peft = "custom_sam_peft.cli.main:app"` |
| `[tool.hatch.build.targets.wheel].packages` | `["src/esam3"]`         | `["src/custom_sam_peft"]`    |
| `[tool.ruff.lint.per-file-ignores]` keys   | `"src/esam3/cli/*_cmd.py"` | `"src/custom_sam_peft/cli/*_cmd.py"` |
| `[tool.mypy].files`                | `["src/esam3"]`                  | `["src/custom_sam_peft"]`    |
| `addopts` (pytest)                 | `--cov=esam3`                    | `--cov=custom_sam_peft`      |
| `[tool.coverage.run].source`       | `["src/esam3"]`                  | `["src/custom_sam_peft"]`    |

Approximately 8 hits.

### Python source -- imports and string literals

- 97 unique `.py` files contain `from esam3` / `import esam3`.
- 317 total import-line occurrences.
- All resolve mechanically via package directory rename + global
  `esam3` -> `custom_sam_peft` substitution in Python code.

Notable string-literal sites (non-import):
- `src/esam3/__init__.py` docstring + `__version__` import path.
- `src/esam3/_bootstrap.py` -- the 9-line internal import block.
- `src/esam3/cli/main.py` -- CLI help text and program name.
- Test files: 53 files matching `tests/unit` and `tests/integration`,
  each importing `esam3.*` symbols.

### CI / workflows

| File                                    | Hit                                      | New                              |
| --------------------------------------- | ---------------------------------------- | -------------------------------- |
| `.github/workflows/ci.yml`              | `mypy src/esam3`                         | `mypy src/custom_sam_peft`       |
| `.github/workflows/pr-colab-badge.yml`  | Uses `${GITHUB_REPOSITORY}` -- no change | Auto-updates on `gh repo rename` |
| `.github/workflows/security.yml`        | (none expected; verify)                  | -                                |
| `.github/workflows/codeql.yml`          | (none expected; verify)                  | -                                |
| `.github/PULL_REQUEST_TEMPLATE.md`      | 1 hit                                    | Replace                          |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | 4 hits                                   | Replace                          |

### Active docs (non-archived)

| File                              | Hits | Action                          |
| --------------------------------- | ---- | ------------------------------- |
| `README.md`                       | 13   | Rewrite header, badges, Colab URL, install commands. |
| `README-dev.md`                   | 2    | String replace.                 |
| `ARCHITECTURE.md`                 | 9    | Replace `esam3` references; the module map paths change too. |
| `CITATION.cff`                    | 2    | `title`, `repository-code`.     |
| `CONTRIBUTING.md`                 | 1    | String replace.                 |
| `RELEASING.md`                    | 1    | String replace.                 |
| `SECURITY.md`                     | 1    | String replace.                 |
| `docs/public-flip-runbook.md`     | 4    | String replace; runbook should reflect current name. |
| `docs/testing/gpu-test-policy.md` | 11   | String replace.                 |
| `cloud/runpod/README.md`          | 5    | String replace (clone URLs, commands). |
| `scripts/public-flip-bootstrap.sh`| 1    | String replace.                 |

### Adjacent copy edits (in scope for this PR)

One README copy edit is bundled into this rename PR. Rationale: the
current "Train in 3 clicks" framing reads as marketing clickbait and
does not match the project's tone. Doing it here avoids a separate
trivial PR.

| File        | Change                                                                                  |
| ----------- | --------------------------------------------------------------------------------------- |
| `README.md` | Replace the "Beginner -- train in 3 clicks" header (and any matching prose in that section's intro) with neutral phrasing. Suggested header: "Beginner -- train in Colab" or similar. |

Constraints on the replacement copy:

- No exclamation marks or marketing language ("just", "easy", "in
  seconds", "instantly", etc.).
- Factual tone, consistent with the rest of the README.
- Must still mention Colab + that no local GPU setup is required.
- Numbered steps in that section stay as-is; only the section header
  and any matching prose at the top of the section change.

Exact wording is left to the implementer subject to the constraints
above. This is the **only** non-rename prose change permitted in the
PR.

### Archived specs and plans -- NOT edited

`docs/superpowers/specs/` and `docs/superpowers/plans/` contain ~2000
hits of `esam3` across ~50 files. These are historical records (the
2026-05-15 design that *named* the package, prior plans referencing
modules). They are deliberately left untouched -- editing them would
corrupt the design history without serving any reader.

The spec for *this* rename (the file you are reading) uses the new
names directly.

### GitHub-side

- `gh repo rename custom-sam-peft` -- the only GitHub-side mutation.
  GitHub auto-redirects `NguyenJus/Efficient-SAM3-Finetuning` -> new
  slug for HTTPS and SSH clone/fetch/push. Redirect is documented as
  permanent unless a new repo claims the old slug.
- Open issues, PRs, releases, Actions history, stars: preserved.
- Repo description: set in Phase 3 to "Closed-vocab finetuning of
  SAM-family models with LoRA / QLoRA on a single consumer GPU".
  Topics suggested in Phase 3 (UI edit): `sam`, `peft`, `lora`,
  `qlora`, `segmentation`, `finetuning`.

### Local filesystem and git worktree metadata

- Parent project dir: `/home/justin/projects/Efficient-SAM3-Finetuning`
  -> `/home/justin/projects/custom-sam-peft`.
- Worktree dir: `.worktrees/rename-repo-custom-sam-peft/` -- still valid
  after parent rename so long as worktree metadata is fixed.
- Git worktree metadata stores absolute paths in two places:
  - `<repo>/.git/worktrees/<name>/gitdir` -- currently
    `/home/justin/projects/Efficient-SAM3-Finetuning/.worktrees/rename-repo-custom-sam-peft/.git`.
  - `<worktree>/.git` (a pointer file) -- currently
    `gitdir: /home/justin/projects/Efficient-SAM3-Finetuning/.git/worktrees/rename-repo-custom-sam-peft`.

Both must be rewritten after the parent dir rename, or `git status`
inside the worktree will fail. The `chore-cleanup-public-flip-tooling`
worktree (sibling) needs the same treatment if it still exists at
rename time.

### `uv.lock`

Zero `esam3` hits (the package name `efficient-sam3-finetuning` does
appear). `uv sync` regenerates the lock entry after `pyproject.toml`
updates; no manual edit required.

## Decision: CLI short-name

**Chosen: `custom-sam-peft`.**

| Option                       | Tradeoffs                                                      |
| ---------------------------- | -------------------------------------------------------------- |
| `custom-sam-peft` (chosen)   | Self-documenting; 15 chars and users type it often, but explicitness wins. Deliberate user choice: rejects the three-letter `csam` alternative on acronym-collision grounds (see below), and prefers the explicit long form over a three-letter alternative. Users who want brevity can alias locally (e.g. `alias csp='custom-sam-peft'`). |
| `csp`                        | Three letters, fast to type, mirrors `${NEW_NAME_KEBAB}` initials. Collides with Content Security Policy as a *concept* but not as a common shell command; `which csp` is empty on a default Ubuntu / macOS install. Alternative, not chosen. |
| `csam`                       | Rejected: acronym collides with illegal-material descriptor (CSAM). Not viable for a public-facing CLI. |
| `custom-sam`                 | Drops "PEFT"; sells the project short and risks collision if a separate `custom-sam` project ever emerges. Alternative, not chosen. |

## Decision: Python import package name

**`custom_sam_peft`** (snake_case, per PEP 8).

- Distribution name (`pyproject.toml` `[project].name`): `custom-sam-peft`.
- Import name: `custom_sam_peft`.
- Directory: `src/custom_sam_peft/`.

This matches Python convention (`scikit-learn` distribution ->
`sklearn` import is a pathological exception, not a model).

## Operation ordering

Phases are ordered by dependency. The orchestrator runs phases 1-2;
phases 3-5 are close-out by the orchestrator after merge.

### Phase 1 -- In-tree renames (on `rename-repo-custom-sam-peft` branch)

Done from inside the worktree, while old GitHub slug is still live.

1. `git mv src/esam3 src/custom_sam_peft`.
2. `git mv notebooks/esam3_train.ipynb notebooks/custom_sam_peft_train.ipynb`.
3. Update `pyproject.toml` per the inventory table above.
4. Global text substitution **scoped to active files**:
   - In-scope: `src/`, `tests/`, `scripts/`, `configs/`, `cloud/`,
     `notebooks/`, `.github/`, top-level `*.md`, `*.toml`, `*.cff`,
     `*.yml`, `*.yaml`, `README*`, `ARCHITECTURE.md`,
     `docs/public-flip-runbook.md`, `docs/testing/`.
   - Out-of-scope (untouched): `docs/superpowers/`, `LICENSE`,
     `uv.lock`, `.git/`, `.worktrees/`.
   - Substitutions (in order, case-sensitive):
     1. `Efficient-SAM3-Finetuning` -> `custom-sam-peft`
     2. `efficient-sam3-finetuning` -> `custom-sam-peft`
     3. `esam3` -> `custom_sam_peft`
     4. `NguyenJus/Efficient-SAM3-Finetuning` (already covered by #1
        but verify the URL form survives intact).
   - The `esam3` -> `custom_sam_peft` substitution covers both Python
     identifiers (`esam3.foo`) and prose (`esam3 architecture`); review
     prose hits manually for awkward casing (e.g. `esam3` at sentence
     start in headings).
5. CLI rename in source:
   - `[project.scripts] custom-sam-peft = "custom_sam_peft.cli.main:app"`.
   - Update CLI `--help` strings and program name in
     `src/custom_sam_peft/cli/main.py` if hard-coded.
6. Run `uv sync --all-extras && uv run ruff check && uv run ruff format
   --check && uv run mypy src/custom_sam_peft && uv run pytest
   --no-cov` locally; iterate until green.
7. Commit, push, CI passes on the *old* repo name (GitHub redirect not
   yet involved).

### Phase 2 -- Merge

Mark draft PR ready, wait for CI green, user merges per
`finishing-a-development-branch` override.

### Phase 3 -- GitHub rename (post-merge)

1. `gh repo rename custom-sam-peft -R NguyenJus/Efficient-SAM3-Finetuning`.
2. GitHub redirects old URL for clones, fetches, and pushes. Redirect
   is documented as persisting until a different repo claims the old
   slug.
3. Set the GitHub repo description to:
   *"Closed-vocab finetuning of SAM-family models with LoRA / QLoRA
   on a single consumer GPU"*.
   This can be done via `gh repo edit NguyenJus/custom-sam-peft
   --description "Closed-vocab finetuning of SAM-family models with
   LoRA / QLoRA on a single consumer GPU"` or in the GitHub UI.
4. Topics are user-edited in the GitHub UI. Suggested topics: `sam`,
   `peft`, `lora`, `qlora`, `segmentation`, `finetuning`.

### Phase 4 -- Update local git remote

From any clone or worktree:
```
git remote set-url origin git@github.com:NguyenJus/custom-sam-peft.git
```
The GitHub redirect would make this optional in the short term, but
updating immediately keeps `git remote -v` accurate.

### Phase 5 -- Local directory rename (orchestrator close-out, outside worktree)

This must happen from a shell whose `cwd` is **not** inside the repo
or any worktree, after killing any background processes the orchestrator
spawned.

1. Kill background processes (per CLAUDE.md close-out step 5a).
2. From the parent of the project dir (`/home/justin/projects/`):
   ```
   mv Efficient-SAM3-Finetuning custom-sam-peft
   ```
3. Fix worktree metadata for every worktree (this branch's worktree
   plus any siblings):
   - Rewrite `custom-sam-peft/.git/worktrees/<name>/gitdir` to point at
     the new absolute path of the worktree.
   - Rewrite `custom-sam-peft/.worktrees/<name>/.git` (pointer file) to
     reference the new absolute path of
     `custom-sam-peft/.git/worktrees/<name>`.
   - `sed -i` is acceptable; both files are plain text.
4. From inside the renamed worktree, run `git status` to confirm the
   metadata is consistent.
5. Continue with branch-log fold + worktree removal per CLAUDE.md
   close-out steps 5b-5d.

The directory rename is intentionally deferred so the user's running
processes (a background runner is noted in the dispatch prompt) are
not disrupted mid-implementation.

## Risks and mitigations

| Risk                                                       | Mitigation                                       |
| ---------------------------------------------------------- | ------------------------------------------------ |
| Stale clones / bookmarks pointing at old GitHub URL.       | GitHub auto-redirects; document in PR body for posterity. No action needed. |
| CI run is in-flight when `gh repo rename` fires.           | Order phase 3 strictly after merge + CI green; `pr-colab-badge.yml` uses `${GITHUB_REPOSITORY}` so it picks up the new slug automatically. |
| Published artifacts (Colab badge image cached, blog links). | None published yet. README Colab badge URL gets updated in phase 1. After rename, GitHub redirect keeps any stale links working. |
| Worktree metadata staleness after dir rename.              | Phase 5 step 3 rewrites both files explicitly.   |
| `.gitleaks.toml`, `.pre-commit-config.yaml`, other config files reference old name. | Phase 1 step 4 sweep covers all root-level configs; verification step greps for residue. |
| Accidental references in commit history / archived specs. | Explicitly out of scope. Commit messages and archived specs remain as-is. |
| Sibling worktree (`chore-cleanup-public-flip-tooling`) breaks. | Same metadata-rewrite procedure as phase 5 step 3, applied to its files too. |
| User decides to change the final name later.               | Spec is parameterized on eight tokens; substitute and re-run. |

## Verification checklist

After phase 1 (pre-merge):

- `grep -rIi --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.worktrees --exclude-dir=.code-review-graph --exclude-dir=superpowers "esam3" .`
  returns **zero hits**. (The `--exclude-dir=superpowers` flag matches by
  basename, which is sufficient because the directory name is unique.)
- Same grep for `Efficient-SAM3-Finetuning` returns **zero hits**.
- `uv sync --all-extras` succeeds.
- `uv run ruff check && uv run ruff format --check` clean.
- `uv run mypy src/custom_sam_peft` clean.
- `uv run pytest` passes.
- `uv run custom-sam-peft --help` prints help text mentioning `custom-sam-peft`.
- `uv run custom-sam-peft doctor` (if implemented) runs without import error.
- `find . -name "*esam3*" -not -path "./.git/*" -not -path "./.worktrees/*" -not -path "./docs/superpowers/*"`
  returns **empty**.

After phase 3 (post-rename):

- `git remote -v` shows the new SSH URL.
- `git fetch origin` succeeds.
- CI badge in README renders against new URL.

After phase 5 (post-dir-rename):

- `pwd` from inside the worktree resolves under `custom-sam-peft/`.
- `git status` succeeds inside the worktree.
- The orchestrator's close-out commands (branch-log fold, worktree
  remove) run without metadata errors.

## Open questions

All open questions from the brainstorm have been resolved.
