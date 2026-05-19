# Public-Flip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-19-public-flip-design.md`](../specs/2026-05-19-public-flip-design.md)
**Umbrella issue:** [#52](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/52) — *make repo public*
**Closes:** #52, #31, #47, #48, #50
**Branch:** `tracking/52-public-flip`

**Goal:** Land every artifact required to flip `Efficient-SAM3-Finetuning` from private to public (community-standards files, README split, CodeQL workflow, per-branch PR Colab badge, notebook GPU guard, `--deselect` CI check, idempotent bootstrap script, operator runbook, `RELEASING.md`) and resolve the retroactive-tag SHAs into the runbook. The plan stops at "draft PR ready for review"; the visibility flip itself is operator work, executed via the runbook on flip day.

**Architecture:** All committed artifacts land in one umbrella PR on branch `tracking/52-public-flip`. Two pre-flight orchestrator outputs (issue-body sensitivity audit + retro-tag SHA block) feed into the PR as a comment and a runbook edit respectively, then the PR is marked ready. The bootstrap script is split `pre-flip` / `post-flip` to mirror GitHub API constraints (some endpoints require the repo to already be public); both subcommands are idempotent so partial runs are safely resumable.

**Tech Stack:** Bash + `gh` CLI for the bootstrap script. Markdown for docs / community files. YAML for GitHub workflows. CFF 1.2.0 for `CITATION.cff`. ShellCheck + actionlint + yamllint + markdownlint enforce hygiene via the existing `lint-hygiene` CI job.

---

## File Map

**New files (committed in this PR):**

```
.github/
  ISSUE_TEMPLATE/
    bug_report.yml
    config.yml
  PULL_REQUEST_TEMPLATE.md
  workflows/
    codeql.yml
    pr-colab-badge.yml

docs/
  public-flip-runbook.md

scripts/
  public-flip-bootstrap.sh

CITATION.cff
CODE_OF_CONDUCT.md
CONTRIBUTING.md
README-dev.md
RELEASING.md
SECURITY.md
```

**Modified files:**

```
.github/workflows/ci.yml        # add a job step that fails on stray `--deselect` in run_gpu_tests.sh
README.md                       # status line, badge row, GH_TOKEN caveat removal, README-dev pointer, Colab-badge removal from "GPU test automation"
notebooks/colab_gpu_tests.ipynb # add metadata.colab.accelerator: "GPU" + early guard cell
scripts/run_gpu_tests.sh        # header comment documenting --deselect convention
```

**No deletions.** README content is *moved* into `README-dev.md`, not duplicated.

**Orchestrator-executed pre-flight work (NOT committed code):**

- Issue-body sensitivity audit (Task 10) → posted as a PR comment.
- Retro-tag SHA resolution (Task 11) → pasted into `docs/public-flip-runbook.md` step 6 (this is a code edit; the *output* is not committed code, the SHA block in the runbook is).

---

## Parallelization opportunities (for orchestrator dispatch)

The orchestrator may fan out these task groups in parallel; each group is file-disjoint from the others:

- **Group A (community standards):** Tasks 1, 2, 3, 4, 5.
- **Group B (README split):** Tasks 6, 7.
- **Group C (workflows + GPU test plumbing):** Tasks 8, 9, 12, 13.
- **Group D (bootstrap script + runbook + RELEASING):** Tasks 14, 15, 16.
- **Group E (pre-flight orchestrator output):** Tasks 10, 11. Task 11 *writes into* `docs/public-flip-runbook.md` and so must run **after** Task 15 lands (the runbook file must exist before its step-6 block is filled).

Task 17 (final verification + PR readiness) serializes after every other task.

---

## Pre-flight check

- [ ] **Step 0a: Confirm working tree clean (only this plan + spec untracked is OK)**

```bash
git status
```
Expected: only `docs/superpowers/specs/2026-05-19-public-flip-design.md` and `docs/superpowers/plans/2026-05-19-public-flip.md` shown (the orchestrator will commit these before fan-out).

- [ ] **Step 0b: Confirm baseline CI hygiene passes locally**

```bash
uv run ruff check
uv run ruff format --check
shellcheck scripts/*.sh
uv run --with yamllint yamllint .
npx --yes markdownlint-cli2 "**/*.md" "#node_modules"
```
Expected: all clean. (If anything is already red, halt and report — do not start work on top of a broken baseline.)

- [ ] **Step 0c: Confirm `gh` CLI is authenticated**

```bash
gh auth status
```
Expected: logged in as the project owner. The bootstrap script and Task 8's PR-body inject test rely on `gh`.

---

## Task 1: `CONTRIBUTING.md` — solo-research posture

**Files:**
- Create: `CONTRIBUTING.md`

Solo public project: forks/dev-clones welcome under Apache-2.0, external PRs not currently accepted, bug reports via issues are OK.

- [ ] **Step 1a: Create `CONTRIBUTING.md`**

```markdown
# Contributing

`Efficient-SAM3-Finetuning` is a solo research project. The maintainer is
**not currently accepting external pull requests**, but forks and
dev-clones are welcome under Apache-2.0 — fork it, modify it, ship it.

## What is welcome

- **Bug reports.** Open an issue using the bug-report template.
- **Forks.** The project is Apache-2.0; you do not need permission to fork
  and modify.
- **Questions in issues.** If something is unclear, ask in an issue.

## What is not currently accepted

- **Pull requests from outside contributors.** Feature work is scoped
  internally; PRs from forks will be closed without review. If you need
  a variant of this project, fork it.

## Developer setup

See [`README-dev.md`](README-dev.md) for the dev loop (uv, ruff, mypy,
pytest, GPU test automation, repo layout).
```

- [ ] **Step 1b: Markdownlint check**

```bash
npx --yes markdownlint-cli2 CONTRIBUTING.md
```
Expected: clean.

- [ ] **Step 1c: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md — solo-research posture, no external PRs"
```

---

## Task 2: `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1

**Files:**
- Create: `CODE_OF_CONDUCT.md`

Verbatim Contributor Covenant 2.1. Contact email is the maintainer's GitHub-public address (the spec calls this out — use the address visible on the GitHub profile, not a separate inbox).

- [ ] **Step 2a: Download Contributor Covenant 2.1 verbatim**

The canonical text is published at
<https://www.contributor-covenant.org/version/2/1/code_of_conduct/>. Copy
the body of that page (the version starting *"We as members, contributors,
and leaders pledge…"*) into `CODE_OF_CONDUCT.md`. **Substitute the
`[INSERT CONTACT METHOD]` placeholder** with the maintainer's
GitHub-public email address (`JustinTNguyen64@gmail.com`).

- [ ] **Step 2b: Markdownlint check**

```bash
npx --yes markdownlint-cli2 CODE_OF_CONDUCT.md
```
Expected: clean. If markdownlint flags an MD line in the Covenant text
(unlikely; the upstream text is mature), add a `<!-- markdownlint-disable
MDxxx -->` comment scoped to that paragraph rather than editing the
Covenant verbatim text.

- [ ] **Step 2c: Commit**

```bash
git add CODE_OF_CONDUCT.md
git commit -m "docs: add CODE_OF_CONDUCT.md (Contributor Covenant 2.1)"
```

---

## Task 3: `SECURITY.md` — point at private vulnerability reporting

**Files:**
- Create: `SECURITY.md`

One paragraph. Points reporters at GitHub's private-vulnerability-reporting (PVR). PVR is enabled by `post-flip` step 4 (Task 14), so the link only works after the flip — that's expected and documented.

- [ ] **Step 3a: Create `SECURITY.md`**

```markdown
# Security policy

## Reporting a vulnerability

Please report security vulnerabilities via GitHub's
[private vulnerability reporting](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/security/advisories/new).
This routes the report directly to the maintainer through a private
channel — please do **not** open a public issue for security findings.

There is no separate disclosure inbox and no PGP key; GitHub's PVR is the
canonical channel.
```

- [ ] **Step 3b: Markdownlint check**

```bash
npx --yes markdownlint-cli2 SECURITY.md
```
Expected: clean.

- [ ] **Step 3c: Commit**

```bash
git add SECURITY.md
git commit -m "docs: add SECURITY.md — GitHub private vulnerability reporting"
```

---

## Task 4: `CITATION.cff` — CFF 1.2.0 for academic forkers

**Files:**
- Create: `CITATION.cff`

CFF 1.2.0; `type: software`; author is the project owner; license `Apache-2.0`.

- [ ] **Step 4a: Create `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
type: software
title: "Efficient-SAM3-Finetuning"
abstract: >-
  Parameter-efficient finetuning of SAM 3.1 for instance segmentation on a
  single consumer GPU. Implements LoRA and QLoRA adapters over the
  Meta SAM 3.1 open-vocab head.
authors:
  - family-names: Nguyen
    given-names: Justin
license: Apache-2.0
repository-code: "https://github.com/NguyenJus/Efficient-SAM3-Finetuning"
```

- [ ] **Step 4b: Validate the CFF**

```bash
uv run --with cffconvert cffconvert --validate -i CITATION.cff
```
Expected: `Citation metadata are valid according to schema version 1.2.0`.
If `cffconvert` reports an issue, fix it; the file must validate before
landing.

- [ ] **Step 4c: yamllint check**

```bash
uv run --with yamllint yamllint CITATION.cff
```
Expected: clean.

- [ ] **Step 4d: Commit**

```bash
git add CITATION.cff
git commit -m "docs: add CITATION.cff (CFF 1.2.0) for academic forks"
```

---

## Task 5: Issue + PR templates

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

Form-mode bug template; `blank_issues_enabled: false`; one contact link for "Feature requests" pointing to a *fork-and-modify* note (no `feature_request.yml`). PR template is a short owner-facing checklist.

- [ ] **Step 5a: Create `.github/ISSUE_TEMPLATE/bug_report.yml`**

```yaml
name: Bug report
description: Report a bug in Efficient-SAM3-Finetuning
title: "[bug] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for filing a bug. Please fill out the fields below so the
        maintainer can reproduce the issue quickly.
  - type: textarea
    id: env
    attributes:
      label: Environment
      description: Output of `uv run esam3 doctor` (or equivalent — Python, OS, CUDA, torch version).
      placeholder: |
        Python 3.13.x
        OS: Ubuntu 24.04
        CUDA: 12.4
        torch: 2.4.x
    validations:
      required: true
  - type: textarea
    id: repro
    attributes:
      label: Reproduction steps
      description: Minimal commands to reproduce. Include the config file used.
      placeholder: |
        1. uv run esam3 init --template coco-text-lora
        2. uv run esam3 train --config config.yaml
        3. ...
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
    validations:
      required: true
  - type: textarea
    id: actual
    attributes:
      label: Actual behavior
      description: Include the full error message and traceback if any.
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Logs
      description: Relevant log output. Use ``` blocks for code formatting.
      render: shell
    validations:
      required: false
```

- [ ] **Step 5b: Create `.github/ISSUE_TEMPLATE/config.yml`**

```yaml
blank_issues_enabled: false
contact_links:
  - name: Feature requests
    url: https://github.com/NguyenJus/Efficient-SAM3-Finetuning/blob/main/CONTRIBUTING.md
    about: >-
      Feature work is scoped internally; this project is not currently
      accepting external PRs. If you need a variant, fork the repo and
      modify it — it is Apache-2.0 licensed.
```

- [ ] **Step 5c: Create `.github/PULL_REQUEST_TEMPLATE.md`**

```markdown
## Summary

<!-- 1-3 sentences: what changed and why. -->

## Checklist

- [ ] Tests added/updated and `uv run pytest` passes locally.
- [ ] `uv run ruff check && uv run ruff format --check` clean.
- [ ] `uv run mypy src/esam3` clean.
- [ ] Scope is bounded; no incidental refactors.
- [ ] Linked to the issue this resolves (if any).
```

- [ ] **Step 5d: yamllint + actionlint + markdownlint**

```bash
uv run --with yamllint yamllint .github/ISSUE_TEMPLATE/bug_report.yml .github/ISSUE_TEMPLATE/config.yml
npx --yes markdownlint-cli2 .github/PULL_REQUEST_TEMPLATE.md
```
Expected: all clean. (actionlint only checks `workflows/`; the issue
templates are validated by GitHub at upload time, so yamllint is the
local proxy.)

- [ ] **Step 5e: Commit**

```bash
git add .github/ISSUE_TEMPLATE/bug_report.yml .github/ISSUE_TEMPLATE/config.yml .github/PULL_REQUEST_TEMPLATE.md
git commit -m "docs: add issue + PR templates (bug-report form, no feature-request)"
```

---

## Task 6: Create `README-dev.md` (absorbs #47)

**Files:**
- Create: `README-dev.md`

Contains content moved out of `README.md` in Task 7: dev loop (uv/ruff/mypy/pytest), GPU test automation (without the Colab badge — moved to per-PR injection), and the repo-layout pointer.

- [ ] **Step 6a: Create `README-dev.md`**

```markdown
# Developer guide

[← back to README](README.md)

This file covers the developer-facing surface of
`Efficient-SAM3-Finetuning`: dev loop, GPU test automation, and repo
layout. End-user documentation lives in [`README.md`](README.md).

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/esam3
uv run pytest
```

GPU smoke test (requires CUDA + SAM 3.1 weights):

```bash
uv run pytest -m gpu
```

## GPU test automation

GPU-gated tests run on a free Colab T4 (no local GPU required). The
Colab notebook lives at
[`notebooks/colab_gpu_tests.ipynb`](notebooks/colab_gpu_tests.ipynb).

A per-branch **Open in Colab** badge is injected into the body of every
pull request by `.github/workflows/pr-colab-badge.yml` — open a PR and
the badge will point at the notebook on that PR's branch.

In Colab Secrets, set `HF_TOKEN` (Hugging Face token with read access
to gated `facebook/sam3.1`). Choose a T4 (or better) runtime, then Run
All. See
[`docs/superpowers/specs/2026-05-17-peft-qlora-design.md`](docs/superpowers/specs/2026-05-17-peft-qlora-design.md)
§11 for the test catalog and
[`docs/testing/gpu-test-policy.md`](docs/testing/gpu-test-policy.md)
for the inspection/release tier breakdown.

## Repo layout

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and data flow.
```

Note: the `ARCHITECTURE.md` pointer mirrors what the current `README.md`
already says (Repo layout section). If `ARCHITECTURE.md` does not exist
yet, that's an existing gap — out of scope for this plan; the broken
link is the same broken link `README.md` already has.

- [ ] **Step 6b: Markdownlint check**

```bash
npx --yes markdownlint-cli2 README-dev.md
```
Expected: clean.

- [ ] **Step 6c: Commit**

```bash
git add README-dev.md
git commit -m "docs: add README-dev.md (dev loop, GPU automation, repo layout)"
```

---

## Task 7: Edit `README.md` — status line, badge row, GH_TOKEN cleanup, README-dev pointer

**Files:**
- Modify: `README.md`

Four surgical edits, each independent. Spec §4.2 defines the change set.

- [ ] **Step 7a: Replace the status callout with an explicit WIP banner (lines 7–10)**

Per spec §4.2, the public README must state at the top that the package
is a work in progress and not ready to run. The banner is placed
**immediately after the H1 title and the one-line tagline** (i.e. right
after the existing lines 3–5 description) so the WIP warning is in a
reader's first glance, and it **subsumes** the old `Status:` block (no
separate `Status:` line is kept).

Find and **remove** the existing status block at lines 7–10:

```markdown
> **Status:** v0 scaffolding only. The CLI and library surfaces exist;
> training/eval/data-loading bodies land in subsequent specs. See
> `docs/superpowers/specs/` for design and `docs/superpowers/plans/`
> for the build sequence.
```

In its place (same position — right after the tagline ending on line 5,
before the next section heading), insert the WIP banner:

```markdown
> **⚠️ Work in progress — not ready to run.**
> v0.5.0 is an active development snapshot. The CLI surfaces (`train`, `eval`, `export`, `run`, `init`, `doctor`) exist and exercise real subsystems (LoRA / QLoRA adapters, W&B tracking), but the project has not been validated end-to-end on production workloads. Expect breaking changes. Use at your own risk; pin to a tagged release if you need stability.
```

Implementer may lightly polish the prose, but the banner must (a) lead
with the WIP / not-ready-to-run warning, (b) name `v0.5.0`, and (c)
appear at the top of `README.md` so a first-glance reader sees it. The
banner does **not** reference a `CHANGELOG.md` (none ships).

- [ ] **Step 7b: Add a status-badge row immediately under the H1**

The H1 is `# efficient-sam3-finetuning` on line 1. Insert a blank line
followed by this badge row immediately after the H1 (before the existing
description paragraph):

```markdown
[![CI](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/actions/workflows/ci.yml/badge.svg)](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
```

- [ ] **Step 7c: Remove the `GH_TOKEN` caveat in the "GPU test automation" section**

Find the "GPU test automation" subsection (currently ends around lines
107–109 of the unmodified README). Two changes:

1. Remove the entire `### GPU test automation` subsection from `README.md`
   (it now lives in `README-dev.md`). That includes the Colab badge for
   `colab_gpu_tests.ipynb`, the prereqs paragraph (the `HF_TOKEN` +
   `GH_TOKEN` lines), and the pointer to the qlora spec.
2. The Beginner section's user-facing Colab badge for `esam3_train.ipynb`
   stays in `README.md` — it is the end-user training notebook, not the
   GPU-test runner.

- [ ] **Step 7d: Replace the "Development" subsection with a pointer to `README-dev.md`**

Find the `### Development` subsection (currently around lines 89–98). It
should be removed from `README.md` (it now lives in `README-dev.md`). In
its place, append a single-paragraph pointer immediately above the
`## License` heading:

```markdown
## Developer setup

Dev loop, GPU test automation, and repo layout live in
[`README-dev.md`](README-dev.md). See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the project's contribution
posture (solo research; forks welcome, external PRs not currently
accepted).
```

- [ ] **Step 7e: Verify `README-dev.md` covers everything removed from `README.md`**

Diff-check that the three subsections removed from `README.md`
(`### Development`, `### GPU test automation`, the `### Repo layout`
pointer) all appear in `README-dev.md` (created in Task 6) with
equivalent content. If a section was removed from `README.md` but not
present in `README-dev.md`, port it over now.

- [ ] **Step 7f: Markdownlint check on both files**

```bash
npx --yes markdownlint-cli2 README.md README-dev.md
```
Expected: clean.

- [ ] **Step 7g: Commit**

```bash
git add README.md
git commit -m "docs: README — v0.5.0 status, badge row, move dev/gpu sections to README-dev.md"
```

---

## Task 8: `.github/workflows/pr-colab-badge.yml` — per-branch Colab badge (#48.1)

**Files:**
- Create: `.github/workflows/pr-colab-badge.yml`

Trigger on PR `opened`, `synchronize`, `reopened`. Use `gh pr edit` to
idempotently inject a delimited block into the PR body. SHA-pin every
third-party action; matches the pattern in `.github/workflows/security.yml`.

- [ ] **Step 8a: Create the workflow**

```yaml
name: PR Colab badge

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  pull-requests: write
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  inject-badge:
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd             # v6.0.2

      - name: Inject per-branch Colab badge into PR body
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
          BRANCH: ${{ github.event.pull_request.head.ref }}
        run: |
          set -euo pipefail
          # URL-encode the branch ref so '/' in names like tracking/52-public-flip
          # is preserved (Colab's github-loader accepts unescaped '/', but other
          # ref chars like '#' or '?' must be encoded).
          encoded_branch="$(python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe='/'))" "${BRANCH}")"
          badge_url="https://colab.research.google.com/github/${GITHUB_REPOSITORY}/blob/${encoded_branch}/notebooks/colab_gpu_tests.ipynb"
          new_block=$(printf '<!-- colab-badge-start -->\n[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](%s)\n<!-- colab-badge-end -->\n' "${badge_url}")

          current_body=$(gh pr view "${PR_NUMBER}" --json body --jq .body)
          # If the delimited block exists, splice the new one in over it.
          # Otherwise, append the block to the end of the body with a
          # leading blank line.
          if printf '%s' "${current_body}" | grep -q '<!-- colab-badge-start -->'; then
            new_body=$(printf '%s' "${current_body}" | python3 -c "
          import re, sys
          body = sys.stdin.read()
          replacement = '''${new_block//\'/\'\\\'\'}'''
          new = re.sub(
              r'<!-- colab-badge-start -->.*?<!-- colab-badge-end -->\n?',
              replacement,
              body,
              count=1,
              flags=re.DOTALL,
          )
          sys.stdout.write(new)
          ")
          else
            new_body=$(printf '%s\n\n%s' "${current_body}" "${new_block}")
          fi

          # No-op write if unchanged (saves a noisy PR-edit event).
          if [ "${new_body}" = "${current_body}" ]; then
            echo "PR body already contains the correct badge; no edit."
            exit 0
          fi
          printf '%s' "${new_body}" | gh pr edit "${PR_NUMBER}" --body-file -
```

**Design notes (do not paste into the file):**
- The `if: head.repo.full_name == github.repository` guard skips PRs from
  forks — fork PRs do not get the badge, because the badge would point
  at a fork branch the maintainer has not reviewed. Aligns with the
  solo-project posture in `CONTRIBUTING.md`.
- The Python heredoc is for HTML-delimited splicing; bash sed would
  struggle with multiline DOTALL. Python is already on `ubuntu-latest`.
- `GITHUB_REPOSITORY` is set by GitHub Actions runtime; no need to pass
  via env.
- SHA-pinned action: `actions/checkout@de0fac…` matches the pin in
  `ci.yml` and `security.yml`. If a Dependabot bump rotates either,
  also rotate here in the same PR.

- [ ] **Step 8b: actionlint + yamllint**

```bash
# Install actionlint from the version pinned in ci.yml's lint-hygiene job
bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7
./actionlint .github/workflows/pr-colab-badge.yml
uv run --with yamllint yamllint .github/workflows/pr-colab-badge.yml
```
Expected: both clean. If actionlint complains about the embedded shell,
the `shellcheck` it spawns will name the line — fix in place.

- [ ] **Step 8c: Smoke-test the badge logic locally**

The shell pipeline is tricky enough to warrant a dry run. Extract the
badge-construction logic into a tmp script and run it with a fake
`current_body` to confirm the splice/append branches behave:

```bash
mkdir -p /tmp/colab-badge-test && cd /tmp/colab-badge-test
export GITHUB_REPOSITORY="NguyenJus/Efficient-SAM3-Finetuning"
export BRANCH="tracking/52-public-flip"

# Case 1: empty body → append
current_body=""
encoded_branch="$(python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe='/'))" "${BRANCH}")"
badge_url="https://colab.research.google.com/github/${GITHUB_REPOSITORY}/blob/${encoded_branch}/notebooks/colab_gpu_tests.ipynb"
new_block=$(printf '<!-- colab-badge-start -->\n[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](%s)\n<!-- colab-badge-end -->\n' "${badge_url}")
echo "=== append case ==="
printf '%s\n\n%s\n' "${current_body}" "${new_block}"

# Case 2: existing block → splice
current_body=$(printf 'Some PR body.\n\n<!-- colab-badge-start -->\nold badge\n<!-- colab-badge-end -->\n\nMore text.')
echo "=== splice case ==="
printf '%s' "${current_body}" | python3 -c "
import re, sys
body = sys.stdin.read()
replacement = '''${new_block//\'/\'\\\'\'}'''
new = re.sub(r'<!-- colab-badge-start -->.*?<!-- colab-badge-end -->\n?', replacement, body, count=1, flags=re.DOTALL)
sys.stdout.write(new)
"
```
Expected: case 1 prints the badge appended at the end with a blank line
separator; case 2 prints "Some PR body." then the new badge block then
"More text." — the old `old badge` line is gone, no duplication.

- [ ] **Step 8d: Commit**

```bash
git add .github/workflows/pr-colab-badge.yml
git commit -m "ci: inject per-branch Open-in-Colab badge into PR bodies (#48.1)"
```

- [ ] **Step 8e: Real-PR smoke (deferred to Task 17)**

The workflow can only be observed end-to-end when the draft PR exists
and the workflow has been pushed. Verify in Task 17 that the badge
appears in this PR's body after pushing.

---

## Task 9: `.github/workflows/codeql.yml` — Python static analysis (#31)

**Files:**
- Create: `.github/workflows/codeql.yml`

Standard CodeQL Python workflow. SHA-pin every action. The workflow will
fail-or-no-op on private repos (CodeQL requires GHAS); a leading comment
documents this.

- [ ] **Step 9a: Create the workflow**

```yaml
# CodeQL Python static analysis.
#
# Note: CodeQL on private repos requires GitHub Advanced Security (GHAS),
# which this project does not have. While the repo is private, this
# workflow will fail at the analyze step. That is expected and intentional —
# the workflow is in place so that the moment the repo flips public (and
# CodeQL becomes free for public repos), analysis starts running on the
# next push without a follow-up PR.

name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  schedule:
    # Weekly Monday 06:00 UTC. Matches the project's other scheduled jobs.
    - cron: "0 6 * * 1"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  actions: read
  contents: read
  security-events: write

jobs:
  analyze:
    if: github.event_name != 'pull_request' || github.event.pull_request.draft == false
    name: Analyze (python)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd             # v6.0.2

      - name: Initialize CodeQL
        uses: github/codeql-action/init@4e828ff8d448a8a6e532957b1811f387a63867e8     # v3.29.0
        with:
          languages: python
          # Use the default + security-and-quality query suite. Adjust here
          # if a future security review wants a tighter or looser set.
          queries: security-and-quality

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@4e828ff8d448a8a6e532957b1811f387a63867e8  # v3.29.0
        with:
          category: "/language:python"
```

**Note on SHA pins:** `github/codeql-action/init` and `analyze` are
pinned to the same v3.29.0 SHA above. If a future Dependabot bump
rotates one of these, the other must be rotated to the same SHA in the
same PR (they are part of one upstream release).

- [ ] **Step 9b: actionlint + yamllint**

```bash
./actionlint .github/workflows/codeql.yml
uv run --with yamllint yamllint .github/workflows/codeql.yml
```
Expected: both clean. (If actionlint flags the SHA — the action exists
upstream — that's actionable; otherwise the SHA-pin is the canonical
form per `.github/workflows/security.yml`.)

- [ ] **Step 9c: Commit**

```bash
git add .github/workflows/codeql.yml
git commit -m "ci: add CodeQL Python workflow (effective on public-flip; #31)"
```

---

## Task 10: Pre-flight orchestrator output — issue-body sensitivity audit (#5.1)

**Files:**
- No committed files; output goes to a PR comment.

The implementer runs the audit and posts the result as a PR comment
**before** Task 17 marks the PR ready for review.

- [ ] **Step 10a: List open issues**

```bash
gh issue list --state open --json number,title,body --limit 100 > /tmp/open-issues.json
gh issue list --state open --json number,title --limit 100
```
Expected: a list of open issues. The spec snapshot was #9, #16, #20,
#22, #23, #24, #33, #34, #35, #36, #44, #51, #52, #53 — the set may
have drifted; trust the live `gh` output.

- [ ] **Step 10b: Audit each body**

For each issue body, scan for:

- Personal names, email addresses, or GitHub handles other than the
  project owner (`@NguyenJus` / `JustinTNguyen64@gmail.com`).
- Internal URLs (corporate intranet, private gists, signed S3 links,
  any URL not on the public web).
- Prior-employer references or non-public project names.

A quick triage greppable pass:

```bash
python3 -c "
import json, re
data = json.load(open('/tmp/open-issues.json'))
for issue in data:
    body = issue['body'] or ''
    flags = []
    # Email pattern (excluding the owner's own)
    for m in re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', body):
        if m.lower() != 'justintnguyen64@gmail.com':
            flags.append(('email', m))
    # @-handle pattern (excluding the owner's own)
    for m in re.findall(r'(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]+)', body):
        if m.lower() != 'nguyenjus':
            flags.append(('handle', '@' + m))
    # Private-gist or signed-S3 patterns
    for m in re.findall(r'https?://[^ )\n\"]+', body):
        if any(needle in m for needle in ('gist.github.com', 'amazonaws.com', 'storage.googleapis.com', 'intranet')):
            flags.append(('url', m))
    if flags:
        print(f'#{issue[\"number\"]} {issue[\"title\"]}: {flags}')
"
```

Then read each issue body that the script flags (and a sample of those
it didn't) in the `gh` UI to confirm nothing slipped through.

- [ ] **Step 10c: Remediate (if anything found)**

For each finding:

```bash
gh issue edit <number> --body "<redacted body>"
```

Replace the offending substring (e.g. an email) with a redacted token
like `[redacted]`. **Do not delete the issue** — preserve thread
continuity.

- [ ] **Step 10d: Post the audit result as a PR comment**

If clean:

```bash
gh pr comment --body "## Issue-body sensitivity audit

Scanned all $(gh issue list --state open --json number --limit 100 | python3 -c "import json,sys; print(len(json.load(sys.stdin)))") open issue bodies for:
- emails or @-handles other than the owner
- private/internal URLs
- prior-employer references

**Result: clean.** No remediation required."
```

If anything was remediated, replace the *Result* line with a per-issue
list:

```text
**Result: remediated.**

- #<num>: redacted <what>
- #<num>: redacted <what>
- ...
```

- [ ] **Step 10e: No commit — this task produces a PR comment only.**

---

## Task 11: Pre-flight orchestrator output — retroactive-tag SHA resolution (#5.2)

**Files:**
- Modify: `docs/public-flip-runbook.md` (step 6's SHA block — depends on Task 15)

**Sequencing:** This task runs *after* Task 15 creates
`docs/public-flip-runbook.md`. The runbook's step 6 will have a
placeholder block; this task replaces it with resolved SHAs.

- [ ] **Step 11a: Resolve each milestone SHA from `git log`**

The spec §3.1 table fixes the milestone semantics. The merge commits
landed via PRs; use the merge SHA (the `(#<n>)` commits on `main`).
From a `git log --oneline --all` walk at plan-write time, the
resolutions are:

| Tag | PR | Merge SHA | One-liner |
| --- | --- | --- | --- |
| `v0.1.0` | #14 | `5071c00` | training loop — Trainer, train_step, checkpoint, box-hint curriculum |
| `v0.2.0` | #17 | `cf81dd7` | eval subsystem — Evaluator, postprocess, CLI, Trainer wiring |
| `v0.3.0` | #7  | `25dcc9a` | peft-qlora — apply_qlora + save/load (LoRA #4 had already landed) |
| `v0.4.0` | #32 | `494bbf5` | ci-hardening — security.yml + lock-check + lint-hygiene (W&B tracking #18 had already landed at `f0cbbee`) |
| `v0.5.0` | (this PR) | `<merge-sha>` | public-flip — community standards, CodeQL, README split, runbook |

**Re-verify before pasting** (`HEAD` may have advanced):

```bash
git log --oneline --all | grep -E '5071c00|cf81dd7|25dcc9a|494bbf5'
```
Expected: all four SHAs resolve to the commits described above. If any
have been rebased/squashed away (e.g. by a merge-queue rewrite), re-walk
`git log` and pick the equivalent commit by message.

- [ ] **Step 11b: Replace the runbook's placeholder block**

In `docs/public-flip-runbook.md` (created by Task 15), find the block
under step 6 marked `<!-- retro-tag-block-start -->` … `<!-- retro-tag-block-end -->` and replace its contents with the
resolved SHAs:

````markdown
<!-- retro-tag-block-start -->
```bash
git tag -a v0.1.0 5071c00 -m "v0.1.0: training loop — Trainer + checkpoint + box-hint curriculum (#14)"
git tag -a v0.2.0 cf81dd7 -m "v0.2.0: eval subsystem — Evaluator + postprocess + CLI wiring (#17)"
git tag -a v0.3.0 25dcc9a -m "v0.3.0: LoRA + QLoRA support (#4, #7)"
git tag -a v0.4.0 494bbf5 -m "v0.4.0: W&B tracking (#18) + CI hardening (#32)"
git tag -a v0.5.0 <merge-sha> -m "v0.5.0: public-flip — community standards + CodeQL + runbook"
git push --tags
```
<!-- retro-tag-block-end -->
````

`<merge-sha>` for `v0.5.0` is left as a literal placeholder — the
operator fills it in at runbook step 6 with the merge SHA of *this* PR
(which does not exist yet at plan-write time).

- [ ] **Step 11c: Markdownlint check on the runbook**

```bash
npx --yes markdownlint-cli2 docs/public-flip-runbook.md
```
Expected: clean.

- [ ] **Step 11d: Commit**

```bash
git add docs/public-flip-runbook.md
git commit -m "docs(runbook): resolve retroactive-tag SHAs for v0.1.0–v0.4.0"
```

---

## Task 12: Update `notebooks/colab_gpu_tests.ipynb` — GPU metadata + guard cell (#48.2)

**Files:**
- Modify: `notebooks/colab_gpu_tests.ipynb`

Two edits:
1. Set `metadata.colab.accelerator: "GPU"` so Colab pre-selects GPU on open.
2. Insert a new **first code cell** that asserts CUDA-GPU presence (hard
   fail) and warns (does not fail) if the detected GPU is not a T4.

The existing first code cell is "Cell 0: Config" — the new guard cell
becomes the new first code cell and pushes the existing cells down by
one.

- [ ] **Step 12a: Write the edit script**

The notebook is JSON; mutate it via a small Python script rather than
editing JSON by hand. Save to `/tmp/edit_colab_nb.py`:

```python
"""Mutate notebooks/colab_gpu_tests.ipynb: add metadata.colab.accelerator
and insert a leading GPU-guard code cell. Idempotent: re-running on an
already-edited notebook leaves it unchanged."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/colab_gpu_tests.ipynb")

GUARD_CELL_TAG = "esam3-gpu-guard"
GUARD_SOURCE = """\
# GPU guard. Runs FIRST so a misconfigured runtime fails loudly before any
# slow install or test step. Two assertions:
#   1. nvidia-smi must succeed AND report at least one GPU. If it doesn't,
#      the runtime is CPU-only — change Runtime → Change runtime type → GPU.
#   2. If the GPU is not a T4, print a WARN: the test suite's timing /
#      memory assumptions are calibrated for a Colab T4, and other GPUs
#      (V100, A100, L4, …) may show different characteristics. The cell does
#      NOT fail in that case — other GPUs are usually fine, just unverified.
import subprocess
import sys

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
except FileNotFoundError as e:
    raise RuntimeError(
        "nvidia-smi not found; this runtime is CPU-only. "
        "Runtime → Change runtime type → GPU (T4 recommended) → Save."
    ) from e
except subprocess.CalledProcessError as e:
    raise RuntimeError(
        f"nvidia-smi failed (returncode={e.returncode}). "
        f"stderr: {e.stderr.strip()}"
    ) from e

gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
if not gpus:
    raise RuntimeError(
        "nvidia-smi reported no GPUs. "
        "Runtime → Change runtime type → GPU (T4 recommended) → Save."
    )

print(f"GPU(s) detected: {gpus}")
if not any("T4" in g for g in gpus):
    print(
        f"WARN: detected GPU is not a T4 ({gpus}). The test suite's timing "
        "and memory assumptions are calibrated for a T4; other GPUs are "
        "usually fine but unverified.",
        file=sys.stderr,
    )
"""

nb = json.loads(NB_PATH.read_text())

# Edit 1: metadata.colab.accelerator = "GPU"
nb.setdefault("metadata", {}).setdefault("colab", {})["accelerator"] = "GPU"

# Edit 2: prepend the guard cell (idempotent — tagged so re-runs are no-ops).
first_code = next((c for c in nb["cells"] if c["cell_type"] == "code"), None)
already_present = bool(first_code) and GUARD_CELL_TAG in first_code.get(
    "metadata", {}
).get("tags", [])

if not already_present:
    guard_cell = {
        "cell_type": "code",
        "metadata": {"tags": [GUARD_CELL_TAG]},
        "source": GUARD_SOURCE.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }
    # Insert before the first existing code cell (after any leading markdown).
    first_code_idx = next(
        i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
    )
    nb["cells"].insert(first_code_idx, guard_cell)

NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
print("notebook updated")
```

- [ ] **Step 12b: Run it (from the repo root) and inspect the diff**

```bash
python3 /tmp/edit_colab_nb.py
git diff notebooks/colab_gpu_tests.ipynb | head -120
```
Expected: a `metadata.colab.accelerator: "GPU"` insertion and a new
first code cell with the tag `"esam3-gpu-guard"`.

- [ ] **Step 12c: Re-run the script to verify idempotency**

```bash
python3 /tmp/edit_colab_nb.py
git diff notebooks/colab_gpu_tests.ipynb | wc -l
```
Expected: line count is the same as after Step 12b (i.e. the second
invocation makes no changes — the guard cell tag short-circuits the
insertion, and the `accelerator` value is already `"GPU"`).

- [ ] **Step 12d: Add a documentation markdown cell**

Insert a small markdown cell *between* the existing leading markdown
header cell and the new guard code cell, explaining what the guard does.
Easiest path: extend the script above to also splice a markdown cell, or
manually edit the notebook JSON. Sample markdown source:

```markdown
### GPU guard (runs first)

This first code cell asserts a CUDA GPU is available before any install
or test step. Colab can pin the **accelerator class** (GPU vs. CPU) via
notebook metadata (`metadata.colab.accelerator: "GPU"`), but it cannot
pin the **GPU model** — the most common assignment is a T4, but Colab
may serve a V100, A100, L4, or others depending on availability. The
test suite is calibrated for a T4; the guard warns (does not fail) if a
different GPU is detected.
```

Add this via a small one-shot edit to the notebook JSON, or run
`/tmp/edit_colab_nb.py` again after adding a `_maybe_insert_doc_md`
helper. The implementer's choice; tested by Step 12e.

- [ ] **Step 12e: Quick programmatic sanity check**

```bash
python3 -c "
import json
nb = json.load(open('notebooks/colab_gpu_tests.ipynb'))
assert nb['metadata']['colab']['accelerator'] == 'GPU', nb['metadata']
first_code = next(c for c in nb['cells'] if c['cell_type'] == 'code')
assert 'esam3-gpu-guard' in first_code.get('metadata', {}).get('tags', []), first_code['metadata']
print('notebook OK')
"
```
Expected: `notebook OK`.

- [ ] **Step 12f: Commit**

```bash
git add notebooks/colab_gpu_tests.ipynb
git commit -m "feat(notebook): pre-select GPU + add GPU-presence guard cell (#48.2)"
```

---

## Task 13: `--deselect` convention header + CI check (#48.3)

**Files:**
- Modify: `scripts/run_gpu_tests.sh`
- Modify: `.github/workflows/ci.yml`

Append a header comment block documenting the `--deselect` convention to
`run_gpu_tests.sh`; add a new CI job step that greps for `--deselect`
and fails the PR if any flags remain. Spec §4.6 documents the
convention.

- [ ] **Step 13a: Add the header comment block to `scripts/run_gpu_tests.sh`**

Find the existing header comment block (lines 1–13 of
`scripts/run_gpu_tests.sh`). Insert a new paragraph immediately after the
"Tiers" block (after line 12) and before `set -euo pipefail`:

```bash
#
# Stateful test-skipping convention (--deselect):
#   When iterating on GPU tests, Claude (or any operator) appends
#   `--deselect <nodeid>` flags to the pytest invocation below as
#   individual tests are confirmed passing on real GPU hardware. This lets
#   the GPU runner skip already-green tests on subsequent runs without
#   editing the test files.
#
#   The mandatory FINAL ALL-GREEN PASS strips every `--deselect` flag and
#   re-runs the full suite to prove it is green end-to-end on a real GPU.
#   No PR may merge with `--deselect` flags left in this script; the CI job
#   `gpu-deselect-check` in `.github/workflows/ci.yml` greps for them and
#   fails the PR if any remain.
```

- [ ] **Step 13b: shellcheck the modified script**

```bash
shellcheck scripts/run_gpu_tests.sh
```
Expected: clean.

- [ ] **Step 13c: Add the CI check to `.github/workflows/ci.yml`**

Insert a new job at the end of the `jobs:` block (after `lint-hygiene`,
matching its job structure):

```yaml
  gpu-deselect-check:
    if: github.event_name == 'push' || github.event.pull_request.draft == false
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd             # v6.0.2

      - name: Fail if scripts/run_gpu_tests.sh contains stray --deselect
        # The --deselect convention is documented in the script's header.
        # Operators add --deselect flags as tests pass on real GPU hardware;
        # the FINAL all-green pass strips all such flags. This check enforces
        # that no flags remain at merge time.
        # Grep is intentionally substring-based — `--deselect` should never
        # appear in a comment either (the existing header block uses
        # backticks around it, but those count for grep; we exclude lines
        # that are pure shell-comments via `grep -v "^[[:space:]]*#"`).
        run: |
          set -euo pipefail
          if grep -nE -- '(^|[[:space:]])--deselect([[:space:]]|=)' scripts/run_gpu_tests.sh \
              | grep -v "^[[:space:]]*#"; then
            echo "ERROR: scripts/run_gpu_tests.sh contains a non-comment --deselect flag." >&2
            echo "Strip it and re-run the full GPU suite before merging." >&2
            exit 1
          fi
          echo "OK: no stray --deselect flags."
```

Note the grep is anchored to `(^|\s)--deselect(\s|=)` to avoid matching
the literal `` `--deselect` `` substring in comments / docstrings.

- [ ] **Step 13d: actionlint + yamllint**

```bash
./actionlint .github/workflows/ci.yml
uv run --with yamllint yamllint .github/workflows/ci.yml
```
Expected: both clean.

- [ ] **Step 13e: Verify the check WOULD fail if a `--deselect` were present (intentional break + revert)**

This is a negative test — the spec emphasizes this CI check is small but
easy to break. Run the check logic locally against a copy of
`run_gpu_tests.sh` with an injected `--deselect`:

```bash
cp scripts/run_gpu_tests.sh /tmp/run_gpu_tests.sh.bak
# Inject a fake --deselect flag on the pytest line
sed -i 's|--no-cov $PATHS|--no-cov --deselect tests/gpu/test_real.py::test_x $PATHS|' /tmp/run_gpu_tests.sh.bak

# Run the same grep the CI job runs
if grep -nE -- '(^|[[:space:]])--deselect([[:space:]]|=)' /tmp/run_gpu_tests.sh.bak \
    | grep -v "^[[:space:]]*#"; then
  echo "GOOD: check WOULD fail in CI."
else
  echo "BAD: check missed the injected --deselect flag." >&2
  exit 1
fi
```
Expected: `GOOD: check WOULD fail in CI.`

- [ ] **Step 13f: Verify the check passes against the real (unmodified) script**

```bash
if grep -nE -- '(^|[[:space:]])--deselect([[:space:]]|=)' scripts/run_gpu_tests.sh \
    | grep -v "^[[:space:]]*#"; then
  echo "BAD: check fires on the real script." >&2; exit 1
else
  echo "OK: real script is clean."
fi
```
Expected: `OK: real script is clean.`

- [ ] **Step 13g: Commit**

```bash
git add scripts/run_gpu_tests.sh .github/workflows/ci.yml
git commit -m "ci: enforce no stray --deselect in run_gpu_tests.sh + document convention (#48.3)"
```

---

## Task 14: `scripts/public-flip-bootstrap.sh` — idempotent automation (#4.7)

**Files:**
- Create: `scripts/public-flip-bootstrap.sh`

Two subcommands: `pre-flip` and `post-flip`. Each step detects current
state and reports `already configured` or `applied`. Exit code 0 unless
an API call fails; partial-run recovery is `re-run from the top`.

- [ ] **Step 14a: Create the script**

```bash
#!/usr/bin/env bash
# Idempotent pre-flip / post-flip automation for the public-flip runbook.
# See docs/public-flip-runbook.md for the operator-facing flow.
#
# Subcommands:
#   pre-flip   — runs on the still-PRIVATE repo. Sets description, topics,
#                branch protection.
#   post-flip  — runs after the repo is PUBLIC. Enables secret scanning,
#                push protection, Dependabot security updates, and private
#                vulnerability reporting.
#
# Idempotency contract: re-running either subcommand must complete with
# exit code 0 and produce only `already configured` lines if there is
# nothing to do. Every state-changing call detects current state first.
#
# Usage:
#   scripts/public-flip-bootstrap.sh pre-flip
#   scripts/public-flip-bootstrap.sh post-flip
#   scripts/public-flip-bootstrap.sh --help
set -euo pipefail

REPO="NguyenJus/Efficient-SAM3-Finetuning"
DESCRIPTION="Parameter-efficient finetuning of SAM3.1 for instance segmentation on a single consumer GPU"
TOPICS=(
  sam
  sam3
  segmentation
  instance-segmentation
  peft
  lora
  qlora
  fine-tuning
  pytorch
  huggingface
  computer-vision
  colab
)
REQUIRED_STATUS_CHECKS=(
  "test"           # ci.yml job name
  "lock-check"     # ci.yml
  "lint-hygiene"   # ci.yml
  "pip-audit"     # security.yml
  "gitleaks"      # security.yml
)

usage() {
  cat <<EOF
Usage: $0 <pre-flip|post-flip>

  pre-flip   Set description, topics, branch protection on \$REPO.
  post-flip  Enable secret scanning, push protection, Dependabot security
             updates, and private vulnerability reporting on \$REPO.

Both subcommands are idempotent. Requires gh CLI authenticated as a repo
admin.
EOF
}

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

require_gh() {
  command -v gh >/dev/null || { warn "gh CLI not on PATH"; exit 2; }
  gh auth status >/dev/null 2>&1 || { warn "gh not authenticated; run 'gh auth login'"; exit 2; }
}

# Pretty-print a status line for a single step.
#   step_status applied   "description"
#   step_status configured "description"
step_status() {
  local kind="$1" desc="$2"
  case "$kind" in
    applied)    log "applied:           $desc" ;;
    configured) log "already configured: $desc" ;;
    *) warn "unknown status kind: $kind"; exit 3 ;;
  esac
}

# ----------------------------------------------------------------------------
# pre-flip subcommand steps
# ----------------------------------------------------------------------------

set_description() {
  local current
  current="$(gh repo view "$REPO" --json description --jq .description)"
  if [ "$current" = "$DESCRIPTION" ]; then
    step_status configured "repo description matches"
    return
  fi
  gh repo edit "$REPO" --description "$DESCRIPTION" >/dev/null
  step_status applied "set repo description"
}

set_topics() {
  # `gh repo edit --add-topic` is additive and ignores duplicates, but we want
  # to report which were added vs. already present for human-readable output.
  local current_csv
  current_csv="$(gh repo view "$REPO" --json repositoryTopics \
    --jq '[.repositoryTopics[].name] | join(",")')"

  local to_add=()
  for t in "${TOPICS[@]}"; do
    case ",$current_csv," in
      *",$t,"*) : ;;  # already present
      *) to_add+=("$t") ;;
    esac
  done

  if [ ${#to_add[@]} -eq 0 ]; then
    step_status configured "all ${#TOPICS[@]} topics present"
    return
  fi
  gh repo edit "$REPO" --add-topic "$(IFS=,; echo "${to_add[*]}")" >/dev/null
  step_status applied "added topics: ${to_add[*]}"
}

set_branch_protection() {
  # GitHub's branches/.../protection PUT endpoint is idempotent. We construct
  # the desired-state body, compare against the current state, and only PUT if
  # they differ. The diff is verbose; we print a short summary.
  local checks_json
  checks_json="$(printf '%s\n' "${REQUIRED_STATUS_CHECKS[@]}" \
    | python3 -c "import json,sys; print(json.dumps([{'context': c.strip(), 'app_id': -1} for c in sys.stdin if c.strip()]))")"

  local desired
  desired="$(cat <<EOF
{
  "required_status_checks": {
    "strict": true,
    "checks": ${checks_json}
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false
}
EOF
)"

  # Fetch current state. If protection is not set, the GET returns 404 — treat
  # that as "needs apply".
  local current
  if ! current="$(gh api "repos/$REPO/branches/main/protection" 2>/dev/null)"; then
    gh api "repos/$REPO/branches/main/protection" \
      -X PUT \
      --input - <<<"$desired" >/dev/null
    step_status applied "set branch protection on main (was: unset)"
    return
  fi

  # Compare a normalized projection of the relevant fields. We don't compare
  # the full payload because GitHub returns extra metadata (URLs, etc).
  local current_proj desired_proj
  current_proj="$(echo "$current" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps({
    'strict': d.get('required_status_checks', {}).get('strict'),
    'checks': sorted(c.get('context','') for c in d.get('required_status_checks', {}).get('checks', [])),
    'linear': d.get('required_linear_history', {}).get('enabled'),
    'force': d.get('allow_force_pushes', {}).get('enabled'),
    'deletions': d.get('allow_deletions', {}).get('enabled'),
}, sort_keys=True))
")"
  desired_proj="$(echo "$desired" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps({
    'strict': d['required_status_checks']['strict'],
    'checks': sorted(c['context'] for c in d['required_status_checks']['checks']),
    'linear': d['required_linear_history'],
    'force': d['allow_force_pushes'],
    'deletions': d['allow_deletions'],
}, sort_keys=True))
")"

  if [ "$current_proj" = "$desired_proj" ]; then
    step_status configured "branch protection on main matches"
    return
  fi
  warn "branch-protection drift detected; desired vs. current:"
  diff <(printf '%s\n' "$desired_proj") <(printf '%s\n' "$current_proj") || true
  gh api "repos/$REPO/branches/main/protection" \
    -X PUT \
    --input - <<<"$desired" >/dev/null
  step_status applied "updated branch protection on main"
}

# ----------------------------------------------------------------------------
# post-flip subcommand steps
# ----------------------------------------------------------------------------

# Tolerate 409/422 ("already enabled") as `already configured`. Treat any other
# non-2xx response as a hard failure.
api_idempotent_put() {
  local label="$1" path="$2"
  local out rc=0
  if out="$(gh api "$path" -X PUT 2>&1)"; then
    step_status applied "$label"
    return
  fi
  rc=$?
  # 409 = conflict (already enabled). 422 = unprocessable (already set).
  if printf '%s' "$out" | grep -qE 'HTTP 409|HTTP 422|already enabled|already configured'; then
    step_status configured "$label"
    return
  fi
  warn "$label failed (rc=$rc): $out"
  exit "$rc"
}

set_secret_scanning() {
  # PATCH-based: read current, set both fields, only call if a change is needed.
  local current
  current="$(gh api "repos/$REPO" --jq '.security_and_analysis')"
  local ss psp
  ss="$(echo "$current" | python3 -c "import json,sys; print(json.load(sys.stdin).get('secret_scanning',{}).get('status','disabled'))")"
  psp="$(echo "$current" | python3 -c "import json,sys; print(json.load(sys.stdin).get('secret_scanning_push_protection',{}).get('status','disabled'))")"

  if [ "$ss" = "enabled" ] && [ "$psp" = "enabled" ]; then
    step_status configured "secret scanning + push protection"
    return
  fi
  gh api "repos/$REPO" -X PATCH \
    -F security_and_analysis[secret_scanning][status]=enabled \
    -F security_and_analysis[secret_scanning_push_protection][status]=enabled \
    >/dev/null
  step_status applied "enabled secret scanning + push protection"
}

# ----------------------------------------------------------------------------
# subcommand dispatch
# ----------------------------------------------------------------------------

cmd_pre_flip() {
  require_gh
  log "pre-flip starting on $REPO"
  set_description
  set_topics
  set_branch_protection
  log "pre-flip done."
}

cmd_post_flip() {
  require_gh
  log "post-flip starting on $REPO"
  set_secret_scanning
  api_idempotent_put "Dependabot vulnerability alerts" "repos/$REPO/vulnerability-alerts"
  api_idempotent_put "Dependabot automated security fixes" "repos/$REPO/automated-security-fixes"
  api_idempotent_put "private vulnerability reporting"   "repos/$REPO/private-vulnerability-reporting"
  log "post-flip done."
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

case "${1:-}" in
  pre-flip)  shift; cmd_pre_flip "$@" ;;
  post-flip) shift; cmd_post_flip "$@" ;;
  -h|--help|help|"") usage; exit 0 ;;
  *) warn "unknown subcommand: $1"; usage; exit 2 ;;
esac
```

**Notes for the implementer (do not paste into the file):**
- The script uses **only `gh`, `python3`, and `printf`** — no `jq`,
  because `jq` is not part of the default Ubuntu CI image. (`python3` is
  always present.) If the maintainer's local environment has `jq`,
  it's fine — but the script must not require it.
- `python3 -c` heredocs are used for the few places that need real JSON
  diffing. They are intentionally small and could be ported to `jq` if a
  future task introduces it.
- The `REQUIRED_STATUS_CHECKS` array names the **job IDs** (not display
  names). The current ci.yml has `test`, `lock-check`, `lint-hygiene`;
  security.yml has `pip-audit` and `gitleaks`. If a new job is added in
  a later task (e.g. `gpu-deselect-check` in Task 13), add it here too —
  this list and that workflow must stay in sync.

- [ ] **Step 14b: Make the script executable + shellcheck**

```bash
chmod +x scripts/public-flip-bootstrap.sh
shellcheck scripts/public-flip-bootstrap.sh
```
Expected: shellcheck clean. The script intentionally uses bash idioms;
if shellcheck flags `SC2086` or similar, fix in place (most idioms above
are already quoted to satisfy shellcheck).

- [ ] **Step 14c: `--help` smoke test**

```bash
scripts/public-flip-bootstrap.sh --help
```
Expected: usage banner prints; exit code 0.

- [ ] **Step 14d: Dry-run `pre-flip` against the (still-private) repo**

This is the spec §8.1 "dry-run smoke test where steps are safe to apply"
gate. Running `pre-flip` against the private repo will:

- `set_description` and `set_topics` are safe to apply NOW — those are
  cosmetic changes on a private repo with no audience.
- `set_branch_protection` is also safe to apply — it tightens, does not
  loosen, the existing rules.

```bash
scripts/public-flip-bootstrap.sh pre-flip
```
Expected output: each step prints `applied: …` or `already configured:
…`. Exit code 0.

- [ ] **Step 14e: Re-run `pre-flip` to confirm idempotency**

```bash
scripts/public-flip-bootstrap.sh pre-flip
```
Expected: every line is `already configured: …`. Exit code 0. No
applied-lines.

- [ ] **Step 14f: `post-flip` cannot be run before the flip — skip dry-run; document only**

`post-flip` requires the repo to already be public (PVR endpoint is
404 otherwise). It is verified end-to-end on flip day, not in the PR.
Document this in the PR description (Task 17) so reviewers don't expect
it.

- [ ] **Step 14g: Commit**

```bash
git add scripts/public-flip-bootstrap.sh
git commit -m "feat(scripts): public-flip-bootstrap.sh — idempotent pre-flip/post-flip (#4.7)"
```

---

## Task 15: `docs/public-flip-runbook.md` — operator flip-day script (#4.8)

**Files:**
- Create: `docs/public-flip-runbook.md`

Numbered steps the operator copy-pastes. Each step has a "what success
looks like" sentence and an "if this fails" pointer. The history-rewrite
escalation (§6) is documented under step 3.

- [ ] **Step 15a: Create the runbook**

```markdown
# Public-flip runbook

Operator copy-paste path for flipping `Efficient-SAM3-Finetuning` from
private to public on GitHub. Run **in order**; the bootstrap script is
idempotent so a partial-run failure is safe to resume by re-running the
failed step.

**Pre-requisites:** the maintainer's `gh` CLI is authenticated as a
repo admin; `gitleaks` v8.21.2 is installed locally (`brew install
gitleaks` or download from the upstream releases page).

---

## Step 1: Review and merge this PR

`tracking/52-public-flip` → `main`.

**Success looks like:** PR is merged; CI is green on the merge commit;
the merge SHA is in `git log origin/main`.

**If this fails:** address CI failures in the PR; do not proceed to step
2 until `main` is at the merge commit of this PR.

---

## Step 2: Run `pre-flip` bootstrap

```bash
scripts/public-flip-bootstrap.sh pre-flip
```

Sets repo description, topics, and `main` branch protection.

**Success looks like:** every line printed is either `applied: …` or
`already configured: …`; the script exits 0. Re-running it produces only
`already configured: …` lines.

**If this fails:** read the failing step's `WARN:` line. Common
failures: (a) `gh` not authenticated as admin — re-run `gh auth login`;
(b) required-status-checks names drifted (ci.yml job IDs changed) — sync
the `REQUIRED_STATUS_CHECKS` array in the script with the current jobs.

---

## Step 3: Full-history gitleaks sweep

```bash
gitleaks detect --no-banner --redact --verbose
```

Runs over the entire git history (not just `HEAD`). This is the
authoritative check that no credential is reachable in any historical
commit reachable from `main`.

**Success looks like:** `no leaks found` in the gitleaks summary; exit
code 0.

**If this fails:** **HALT before step 4.** Do not flip visibility with a
credential reachable in history. The owner has pre-authorized a
`git filter-repo` rewrite (the project's only `--force` push to `main`);
see the **History-rewrite escalation** appendix at the bottom of this
runbook.

---

## Step 4: Flip visibility to public

```bash
gh repo edit NguyenJus/Efficient-SAM3-Finetuning \
  --visibility public --accept-visibility-change-consequences
```

The actual flip. Once this returns, the repo is public on github.com.

**Success looks like:** `gh` prints no error; visiting the repo URL in a
private browser session (or logged out) loads the repo without a 404.

**If this fails:** the most common cause is the maintainer's `gh` token
lacking the `delete_repo` / `repo` scope needed for visibility changes —
re-run `gh auth refresh -s repo,admin:org` and retry.

**Rollback:** `gh repo edit NguyenJus/Efficient-SAM3-Finetuning
--visibility private --accept-visibility-change-consequences` flips back.

---

## Step 5: Run `post-flip` bootstrap

```bash
scripts/public-flip-bootstrap.sh post-flip
```

Enables secret scanning, push protection, Dependabot security updates,
and private vulnerability reporting (PVR). These endpoints require the
repo to already be public — running `post-flip` before step 4 will 404
on the PVR call.

**Success looks like:** every line printed is `applied: …` or `already
configured: …`; the script exits 0.

**If this fails:** read the failing step's `WARN:` line. If a single
endpoint 404s, you can re-run the script — every step is idempotent and
the script will skip the already-enabled features.

---

## Step 6: Apply retroactive tags + cut `v0.5.0`

The `<merge-sha>` placeholder below is the merge commit SHA of *this*
PR — fill it in from `git log origin/main -1 --format=%h` (or copy from
the PR's "Merged" event on the github.com UI).

<!-- retro-tag-block-start -->
```bash
# PLACEHOLDER — will be replaced by Task 11 with resolved SHAs.
git tag -a v0.1.0 <sha> -m "v0.1.0: <milestone>"
git tag -a v0.2.0 <sha> -m "v0.2.0: <milestone>"
git tag -a v0.3.0 <sha> -m "v0.3.0: <milestone>"
git tag -a v0.4.0 <sha> -m "v0.4.0: <milestone>"
git tag -a v0.5.0 <merge-sha> -m "v0.5.0: public-flip"
git push --tags
```
<!-- retro-tag-block-end -->

Then draft a GitHub Release for `v0.5.0`:

```bash
gh release create v0.5.0 \
  --title "v0.5.0 — public flip" \
  --notes-file <(cat <<'EOF'
First public release. Snapshot of the v0 surface at flip time:
- esam3 CLI: run, train, eval, export, init, doctor
- PEFT: LoRA, QLoRA
- Tracking: TensorBoard, W&B, none
- Data: COCO + HuggingFace datasets
- CI: ruff, mypy, pytest, pip-audit, gitleaks, CodeQL (newly enabled by the flip)

See the retroactive tags v0.1.0–v0.4.0 for the v0 milestone walk.
EOF
)
```

**Success looks like:** `git push --tags` succeeds; the Tags page on
github.com shows all five tags; the Releases page shows `v0.5.0` as the
latest release.

**If this fails:** the most common cause is a tag name already taken
(idempotent for our use — `git push` will skip it). If a SHA was wrong,
delete the local tag (`git tag -d v0.x.x`) and the remote tag
(`git push origin :refs/tags/v0.x.x`), fix the SHA, and re-tag.

---

## Done

Verify against the post-flip checklist in the spec (§8.2):

- Insights → Community Standards → all green (Description, README,
  Code of Conduct, Contributing, License, Security policy, Issue
  templates, Pull request template).
- Settings → Code security: secret scanning, push protection,
  Dependabot security updates, and private vulnerability reporting all
  enabled.
- Actions → CodeQL: first run succeeds.
- Tags page: `v0.1.0`–`v0.4.0` and `v0.5.0` all present.
- Releases page: `v0.5.0` published.

Close the umbrella issue #52 with a comment linking to this runbook and
the merge commit.

---

## Appendix: History-rewrite escalation (if step 3 finds a leak)

The owner has **pre-authorized** a `git filter-repo` rewrite to scrub a
historical credential. This is the project's only `--force` push to
`main`. Follow this path **only** if step 3's gitleaks sweep reports a
hit; **do not** run a rewrite speculatively.

### Consequences (read before running)

- **Force-pushing `main` invalidates every open PR and every local
  clone.** Open PRs must be re-based by their authors (or, in this solo
  project, by the maintainer). All existing clones — including CI
  caches — must be reset (`git fetch --all && git reset --hard
  origin/main`).
- **The credential, if real, is treated as compromised — rotation is
  mandatory regardless of whether the rewrite succeeds.** A scrubbed
  history does not undo any exposure that happened while the secret
  was reachable.
- This step is recommended on a **quiet day with no other open PRs** to
  minimize collateral damage.
- **History rewrites are not reversible** once force-pushed.

### Procedure

1. **HALT** before step 4 (the visibility flip).
2. Document the redacted finding in the umbrella PR thread. Gitleaks
   redacts the secret value by default; copy the gitleaks output as-is.
3. **Rotate the credential.** Treat it as compromised regardless of
   rewrite success.
4. Run the rewrite (choose one):
   - **Substring scrub** (for hardcoded credentials):
     ```bash
     git filter-repo --replace-text <patterns-file>
     ```
     where `<patterns-file>` is a one-per-line list of literal strings
     to redact (one per credential value).
   - **File removal** (for a checked-in secret file):
     ```bash
     git filter-repo --invert-paths --path path/to/file
     ```
5. Force-push the rewritten `main`:
   ```bash
   git push --force-with-lease origin main
   ```
6. Re-base the flip branch onto the rewritten history; force-push the
   branch.
7. Re-request review on the umbrella PR.
8. Resume the runbook at **step 3** (re-run gitleaks to confirm the
   hit is gone), then continue to step 4.
```

- [ ] **Step 15b: Markdownlint check**

```bash
npx --yes markdownlint-cli2 docs/public-flip-runbook.md
```
Expected: clean.

- [ ] **Step 15c: Commit**

```bash
git add docs/public-flip-runbook.md
git commit -m "docs: add public-flip-runbook.md (operator flip-day script)"
```

---

## Task 16: `RELEASING.md` — SemVer + retroactive-tag scheme (#4.9)

**Files:**
- Create: `RELEASING.md`

Short doc covering SemVer, annotated-tag policy, the retroactive-tag
scheme, and how to cut future releases. Points at GitHub Releases as
the changelog surface (no `CHANGELOG.md` until/unless v1.0).

- [ ] **Step 16a: Create the file**

```markdown
# Releasing

This project follows **Semantic Versioning** (`vMAJOR.MINOR.PATCH`).
Releases are **annotated git tags** plus a GitHub Release with
human-readable notes.

## Tag policy

- Tags are **annotated** (`git tag -a`), never lightweight. The message
  is a one-line milestone description.
- Tag the **merge commit** of the release-cut PR (or the latest commit
  on `main` if there is no dedicated cut PR).
- Push tags with `git push --tags` and draft a GitHub Release for each.

## Retroactive tags (v0)

The flip-day release `v0.5.0` was preceded by four retroactive tags
applied to historical milestones. They were applied in a single
`git tag -a ... && git push --tags` block in
[`docs/public-flip-runbook.md`](docs/public-flip-runbook.md) step 6.

| Tag | Milestone |
| --- | --- |
| `v0.1.0` | First working training-loop merge (#14) |
| `v0.2.0` | First eval-pipeline merge — `Evaluator` + `MetricsReport` (#17) |
| `v0.3.0` | LoRA + QLoRA support merged (#4, #7) |
| `v0.4.0` | W&B tracking (#18) + CI hardening (#32) |
| `v0.5.0` | Public-flip merge — community standards, CodeQL, runbook |

## Cutting a future release

1. Decide the new version per SemVer (breaking → major; feature → minor;
   fix → patch).
2. Update the `Status:` line in `README.md` if the new version's surface
   warrants it.
3. Tag and push:
   ```bash
   git tag -a vX.Y.Z <commit-sha> -m "vX.Y.Z: <one-line milestone>"
   git push --tags
   ```
4. Draft a GitHub Release for the tag with notes summarizing changes
   since the previous tag. Use `gh release create vX.Y.Z --generate-notes`
   as a starting point, then edit the body for human readability.

## Changelog

There is no `CHANGELOG.md` in v0. The
[GitHub Releases page](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/releases)
is the canonical changelog surface. A `CHANGELOG.md` may be introduced
at or after v1.0 if it becomes useful.
```

- [ ] **Step 16b: Markdownlint check**

```bash
npx --yes markdownlint-cli2 RELEASING.md
```
Expected: clean.

- [ ] **Step 16c: Commit**

```bash
git add RELEASING.md
git commit -m "docs: add RELEASING.md (SemVer, annotated tags, GitHub Releases as changelog)"
```

---

## Review Checkpoint — after Tasks 1–16

Pause for orchestrator review before Task 17. Reviewer checks:

- Every spec deliverable (§4.1–§4.9) maps to a landed task.
- Pre-flight outputs (Task 10 PR comment, Task 11 runbook edit) are
  posted/landed.
- Bootstrap script's `pre-flip` is idempotent (verified in Step 14e).
- `--deselect` CI check fires correctly under intentional break (Step
  13e) and passes against the real script (Step 13f).
- The retro-tag SHA block in the runbook resolves to the five milestone
  commits.
- No spec content is left as TODO / placeholder.

---

## Task 17: Final verification + draft-PR-ready

**Files:** none (verification + PR-readiness only).

- [ ] **Step 17a: Full local hygiene sweep**

```bash
uv run ruff check
uv run ruff format --check
shellcheck scripts/*.sh
./actionlint -color
uv run --with yamllint yamllint .
npx --yes markdownlint-cli2 "**/*.md" "#node_modules"
```
Expected: every check clean.

- [ ] **Step 17b: Full pytest**

```bash
uv run pytest tests/unit tests/integration -q
```
Expected: full pass; coverage gate (`--cov-fail-under=80`) holds.

- [ ] **Step 17c: Push the branch**

```bash
git push -u origin tracking/52-public-flip
```
Expected: push succeeds; the PR's draft status carries over.

- [ ] **Step 17d: Verify the PR-Colab-badge workflow ran and injected the badge (Task 8 deferred smoke)**

After the push completes, observe:

```bash
gh pr view --json body --jq .body | grep -A1 colab-badge-start
```
Expected: the PR body now contains the delimited badge block pointing
at the `tracking/52-public-flip` branch's notebook. If the block is
absent, check the workflow run:

```bash
gh run list --workflow pr-colab-badge.yml --limit 3
gh run view --log <latest-run-id>
```

- [ ] **Step 17e: Wait for all CI checks green**

```bash
gh pr checks --watch
```
Expected: every required check passes (`test`, `lock-check`,
`lint-hygiene`, `pip-audit`, `gitleaks`, `gpu-deselect-check`, CodeQL).
Note: CodeQL on a private repo with no GHAS may fail at the analyze
step (documented in `codeql.yml`'s leading comment); that's expected
and **does not block** flip authorization. Mark the check as
acknowledged in the PR body if it fails for the documented reason.

- [ ] **Step 17f: Confirm Task 10 PR comment is posted**

```bash
gh pr view --json comments --jq '.comments[].body' | grep -m1 -A1 'Issue-body sensitivity audit'
```
Expected: the audit-result comment from Task 10 is present.

- [ ] **Step 17g: Confirm runbook step 6 has resolved SHAs**

```bash
sed -n '/<!-- retro-tag-block-start -->/,/<!-- retro-tag-block-end -->/p' docs/public-flip-runbook.md
```
Expected: the block contains five `git tag -a vX.Y.Z <sha> -m …` lines,
and only `v0.5.0`'s SHA is a literal `<merge-sha>` placeholder.

- [ ] **Step 17h: Update PR body with reviewer summary**

Edit the PR description to summarize:

- The full deliverable list (one bullet per Task 1–16).
- Pre-flight audit result (link to the PR comment).
- A note that the visibility flip is **not** part of this PR — the
  operator runs `docs/public-flip-runbook.md` after merge.
- The CodeQL-on-private-repo caveat from Step 17e if applicable.

Use:

```bash
gh pr edit --body-file - <<'EOF'
## Summary
<bullets>

## Pre-flight audit
- Issue-body sensitivity audit: see PR comment "Issue-body sensitivity audit"
- Retroactive-tag SHAs resolved in `docs/public-flip-runbook.md` step 6

## Out of scope (operator-day, not this PR)
- The actual `gh repo edit --visibility public` flip
- `post-flip` bootstrap (requires the repo to already be public)
- History-rewrite (only if gitleaks finds a hit — see runbook appendix)

## Closes
Closes #52, #31, #47, #48, #50.
EOF
```

- [ ] **Step 17i: Mark the draft PR ready for review**

```bash
gh pr ready
```
Expected: PR transitions from "Draft" to "Ready for review"; required
CI re-runs on the `ready_for_review` event and stays green.

- [ ] **Step 17j: Final stop**

Stop here. The plan is complete when the PR is ready for review with
all CI green. The visibility flip itself is operator work, executed via
`docs/public-flip-runbook.md`.

---

## Verification commands (cumulative)

```bash
# Per-task (run after each task that touches the relevant surface)
shellcheck scripts/*.sh                                   # Tasks 13, 14
./actionlint .github/workflows/<file>.yml                 # Tasks 8, 9, 13
uv run --with yamllint yamllint <path>                    # Tasks 4, 5, 8, 9, 13
npx --yes markdownlint-cli2 <path-or-glob>                # Tasks 1, 2, 3, 6, 7, 11, 15, 16
uv run --with cffconvert cffconvert --validate -i CITATION.cff  # Task 4

# Bootstrap idempotency (Task 14)
scripts/public-flip-bootstrap.sh pre-flip                 # dry-run on private repo
scripts/public-flip-bootstrap.sh pre-flip                 # second invocation = all "already configured"

# --deselect CI check (Task 13)
# Negative test on a temp copy with an injected --deselect; positive test on the real script.

# Final sweep (Task 17)
uv run ruff check && uv run ruff format --check
shellcheck scripts/*.sh
./actionlint -color
uv run --with yamllint yamllint .
npx --yes markdownlint-cli2 "**/*.md" "#node_modules"
uv run pytest tests/unit tests/integration -q
gh pr checks --watch
```

---

## Risks & Open Questions

| Risk | Mitigation in this plan |
| --- | --- |
| PR-Colab-badge workflow may fail on the first PR run if `gh pr edit` lacks the `pull-requests: write` permission. | Task 8 declares the permission at the workflow level; Task 17e verifies the badge appears in the PR body via `gh pr view` after push. |
| CodeQL workflow will fail on a private repo (no GHAS). | Task 9's leading comment documents this. Step 17e notes that CodeQL failure on the private repo does **not** block PR merge. |
| Bootstrap script's `REQUIRED_STATUS_CHECKS` array drifts from the actual job IDs in `ci.yml` / `security.yml`. | Task 14a comment lists the synchronization requirement; Task 13 adds a new required check (`gpu-deselect-check`) which the planner has already included in `REQUIRED_STATUS_CHECKS`. |
| `gpu-deselect-check` grep regex is too narrow and silently misses `--deselect=foo` style. | The regex `(^|\s)--deselect(\s|=)` covers both space- and `=`-separated forms. Task 13e is a negative test that intentionally injects a `--deselect` to confirm the check fires. |
| Issue-body sensitivity audit script (Step 10b) misses a needle pattern. | The script flags an over-broad superset (every URL, every @-handle); the implementer still reads each issue body in the `gh` UI as the authoritative check. |
| Retroactive-tag SHAs may rebase out of `main` if a merge-queue rewrite happens between plan-write and execution. | Step 11a's "Re-verify before pasting" sub-step re-greps `git log` and instructs the implementer to pick the equivalent merge commit by message if a SHA has shifted. |
| The notebook-edit script (Task 12) is idempotent on the JSON layer but does not detect partial inserts — if an earlier failed run left a half-applied state, the tag-based check could still match. | Step 12c re-runs the script and asserts the line-count is unchanged; Step 12e is a programmatic sanity check that confirms `accelerator: "GPU"` and the guard cell tag are both present exactly once. |
| `docs/public-flip-runbook.md` step 6's `<merge-sha>` placeholder can be missed by the operator. | The placeholder is a literal `<merge-sha>` string (not a SHA); Step 11b leaves it as the only un-resolved token in the block. The runbook's step 6 explanatory text calls out where to find it. |

No open questions remain. Every spec deliverable maps to a landed
task; every operator-day action is covered by the runbook; every CI
check the spec calls for is wired and locally verified before push.

---

## Spec coverage map

| Spec section | Tasks |
| --- | --- |
| §1.1 Issues absorbed (#31, #47, #48, #50, #52) | Tasks 1–17 (closes via Task 17h PR body) |
| §3.1 Retroactive tag plan | Tasks 11, 15, 16 |
| §4.1 Community-standards files | Tasks 1, 2, 3, 4, 5 |
| §4.2 README split | Tasks 6, 7 |
| §4.3 CodeQL workflow (#31) | Task 9 |
| §4.4 PR Colab badge workflow (#48.1) | Task 8 |
| §4.5 Colab notebook GPU metadata + guard cell (#48.2) | Task 12 |
| §4.6 `--deselect` convention header + CI check (#48.3) | Task 13 |
| §4.7 `scripts/public-flip-bootstrap.sh` (pre-flip + post-flip + idempotency) | Task 14 |
| §4.8 `docs/public-flip-runbook.md` | Tasks 15, 11 (SHA block) |
| §4.9 `RELEASING.md` | Task 16 |
| §5.1 Issue-body sensitivity audit | Task 10 |
| §5.2 Retroactive-tag SHA resolution | Task 11 |
| §6 History-rewrite escalation | Task 15 (runbook appendix) |
| §7 Risks and rollback | Risks section above + runbook step-level "if this fails" pointers |
| §8.1 Mergeable PR | Task 17 (verification + ready-for-review) |
| §8.2 Post-flip operator-confirms | Runbook "Done" section (Task 15) |
| §8.3 Explicitly NOT validated | Documented in runbook + PR body (Task 17h) |
| §9 File layout | File Map above |
