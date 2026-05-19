# Public-Flip Design (umbrella issue #52)

**Status:** Draft (2026-05-19)
**Scope:** Everything required to flip the `Efficient-SAM3-Finetuning` repository from private to public on GitHub. Covers community-standards files, README split, CodeQL workflow, Colab GPU testing follow-ups, an idempotent bootstrap script, an operator runbook, and the retroactive SemVer tag plan culminating in `v0.5.0`. Does *not* perform the visibility flip itself — the PR lands the artifacts; the operator runs the runbook.

**Builds on:** [`2026-05-18-ci-hardening-design.md`](2026-05-18-ci-hardening-design.md) (security workflow, SHA pinning); [`2026-05-19-gpu-test-policy-design.md`](2026-05-19-gpu-test-policy-design.md) (Colab GPU test posture); [`2026-05-18-simplify-ux-design.md`](2026-05-18-simplify-ux-design.md) (current CLI surface that justifies the v0.5.0 framing).

---

## 1. Goals & Scope

Take the repo from private to public with the community-standards, security, and release scaffolding a public Apache-2.0 research project is expected to ship. Keep the "solo public project" posture explicit: forks welcome, external PRs not currently accepted. Land everything in a single umbrella PR so the flip is one atomic operator action.

### 1.1 Issues absorbed and closed by this PR

| Issue | Title (short) | Disposition |
| --- | --- | --- |
| #52 | Umbrella — make repo public | Closed by this PR |
| #31 | Add CodeQL workflow once public | New `.github/workflows/codeql.yml` |
| #47 | Create `README-dev.md` | README split; dev content moved |
| #48 | Easier GPU testing via Colab (all three sub-tasks) | Per-branch PR badge workflow, notebook metadata + guard cell, `--deselect` convention + CI check |
| #50 | Retroactively tag commits | Spec defines the tag plan; planner resolves SHAs; runbook applies them |

### 1.2 In scope

- Community-standards files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`).
- README split — `README-dev.md` plus edits to `README.md`.
- `.github/workflows/codeql.yml`.
- `.github/workflows/pr-colab-badge.yml`.
- Colab notebook metadata + GPU guard cell in `notebooks/colab_gpu_tests.ipynb`.
- `scripts/run_gpu_tests.sh` `--deselect` convention header + CI grep check.
- `scripts/public-flip-bootstrap.sh` — idempotent `pre-flip` / `post-flip` automation.
- `docs/public-flip-runbook.md` — operator flip-day script.
- `RELEASING.md` — SemVer + retroactive tag scheme.
- Orchestrator-executed pre-flip work (issue-body sensitivity audit, retroactive tag SHA resolution) — recorded in the PR thread / runbook, not as committed code.

### 1.3 Out of scope (explicitly deferred)

- New feature work (datasets, adapters, training modes).
- Cloud-provider integration audits (#34, #35, #53).
- Repo hardening pass (#26).
- Root-file audit (#49).
- `CHANGELOG.md` — `RELEASING.md` + GitHub Releases cover this for v0.
- Re-opening the project to external contributions.

---

## 2. Architectural Approach

The flip splits cleanly into three phases, all driven from this single PR:

1. **Land artifacts (the PR).** All committed files (community standards, workflows, README split, bootstrap script, runbook, `RELEASING.md`) go in one umbrella draft PR on branch `tracking/52-public-flip`. PR body links to this spec and the plan.
2. **Pre-flight (orchestrator, before merge).** Two checks the implementer runs and reports as PR comments before requesting flip authorization:
   - Issue-body sensitivity audit (every open issue scanned for PII/internal references).
   - Retroactive-tag SHA resolution (planner walks `git log`; SHAs land in the runbook as copy-pasteable `git tag -a` commands).
3. **Operator flip-day (post-merge).** Operator runs `docs/public-flip-runbook.md` end-to-end. The runbook is the only flip-day surface the operator touches; it invokes `scripts/public-flip-bootstrap.sh pre-flip`, then `gh repo edit --visibility public`, then `scripts/public-flip-bootstrap.sh post-flip`, then applies and pushes the retroactive tags.

`scripts/public-flip-bootstrap.sh` is idempotent so that a partial run can be resumed without damage. The split between `pre-flip` (works on a private repo) and `post-flip` (requires the repo to already be public, e.g. for private-vulnerability-reporting) reflects GitHub API constraints, not a workflow preference.

---

## 3. Version Selection — `v0.5.0` as the Flip-Day Release

The flip-day release is **`v0.5.0`**, not `v0.1.0`. Rationale captured here so the planner doesn't relitigate it:

- `README.md` still frames the project as "v0 scaffolding only," so `v1.x` would be too forward — the public model-loading + training-loop story is real but not yet validated against the full SA-1B-scale workloads a v1 implies.
- However, the CLI table (`README.md` lines 51–58) shows `run`, `train`, `eval`, `export`, `init`, `doctor` all functional, with LoRA + QLoRA adapters, W&B tracking, CI hardening, GPU-test policy, and HF utilities all designed and implemented across ~12 spec-driven features in ~5 days (279 commits over May 15–19, 2026).
- `v0.1.0` would understate that surface area. `v0.5.0` reads as "substantial v0 surface, room to grow to v1.0."

### 3.1 Retroactive tag plan

Retroactive annotated tags walk back from `v0.5.0`. The planner resolves exact SHAs from `git log --oneline --all`; this spec fixes the milestones.

| Tag | Milestone (target commit semantics) |
| --- | --- |
| `v0.1.0` | First working training-loop merge (training subsystem lands end-to-end on stub model) |
| `v0.2.0` | First eval pipeline merge (`Evaluator` + `MetricsReport` shipped) |
| `v0.3.0` | LoRA + QLoRA support merged (both PEFT specs landed) |
| `v0.4.0` | W&B tracking + CI hardening merged (tracking subsystem + `.github/workflows/security.yml`) |
| `v0.5.0` | Public-flip merge commit (the merge SHA for this PR) |

Tags are **annotated** (`git tag -a`), not lightweight. Message format: `<tag>: <one-line milestone description>`. All tags are pushed in a single `git push --tags` during runbook step 6.

---

## 4. Deliverables

### 4.1 Community-standards files

Posture: solo research project, owner not currently accepting external contributions, forks and dev-clones welcome under Apache-2.0, bug reports via issues are OK.

| File | Content summary |
| --- | --- |
| `CONTRIBUTING.md` | Short, explicit: solo research project; PRs not currently accepted; forks/dev-clones welcome under Apache-2.0; bug reports via issues are OK; points at `README-dev.md` for the dev loop and (future) `ARCHITECTURE.md` for the module map. |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 2.1 verbatim, applies to issue-thread visitors. Contact email is the user's GitHub-public address. |
| `SECURITY.md` | Points reporters at GitHub's private-vulnerability-reporting (enabled by `post-flip`). One paragraph; no PGP key, no separate inbox. |
| `CITATION.cff` | CFF 1.2.0; `type: software`; title `Efficient-SAM3-Finetuning`; author: project owner; license `Apache-2.0`; repository-code URL. For academic forkers. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Form-mode template: title prefix `[bug]`, fields for env (Python, OS, CUDA), repro steps, expected vs. actual, logs. |
| `.github/ISSUE_TEMPLATE/config.yml` | `blank_issues_enabled: false`. `contact_links:` one entry, "Feature requests" → short note: *feature work is scoped internally; fork-and-modify if you need a variant*. **No** `feature_request.yml`. |
| `.github/PULL_REQUEST_TEMPLATE.md` | Short checklist: tests, lint, scope; used for the owner's own PRs. |

### 4.2 README split (absorbs #47 + flip cleanup)

**New file: `README-dev.md`** at repo root. Opens with a back-link to `README.md`. Contains content moved out of `README.md`:

- `### Development` section (uv run, ruff, mypy, pytest invocations).
- `### GPU test automation` section — note the Colab badge is being removed from `README.md` per #48.1 and now lives only on PRs.
- `### Repo layout` pointer (or short tree, matching the current README's level of detail).

**Edits to `README.md`:**

- Remove the `GH_TOKEN` caveat at lines 107–109 (obsolete once the repo is public and the Colab badge moves to per-PR injection).
- Add a status-badge row near the top: CI build status, license=Apache-2.0, Python version.
- Add a one-line pointer to `README-dev.md` near the bottom (e.g. *Developer setup, GPU test loop, and repo layout live in `README-dev.md`*).
- Replace the existing `> **Status:** v0 scaffolding only.` block (currently lines 7–10) with an **explicit WIP banner** at the top of the README — placed immediately after the H1 title and the one-line tagline (i.e. right after the existing lines 3–5 description), so the WIP warning is in a reader's first glance. The banner combines version, WIP warning, and the "not ready to run" framing in one callout-style block quote. Proposed wording (implementer may polish):

  ```markdown
  > **⚠️ Work in progress — not ready to run.**
  > v0.5.0 is an active development snapshot. The CLI surfaces (`train`, `eval`, `export`, `run`, `init`, `doctor`) exist and exercise real subsystems (LoRA / QLoRA adapters, W&B tracking), but the project has not been validated end-to-end on production workloads. Expect breaking changes. Use at your own risk; pin to a tagged release if you need stability.
  ```

  The new banner subsumes the old `Status:` block; no separate `Status:` line is kept. The banner does **not** reference a `CHANGELOG.md` (none ships); release history is discoverable via GitHub Releases / tags without an inline pointer here.

### 4.3 CodeQL workflow (#31)

**New file: `.github/workflows/codeql.yml`**

- `language: python`.
- Schedule: weekly cron + on `push`/`pull_request` to `main`.
- All third-party actions SHA-pinned, matching the pattern in `.github/workflows/security.yml`.
- The workflow will not run effectively until the repo is public (CodeQL on private repos requires GHAS); that is expected and documented in a leading comment.

### 4.4 Per-branch PR Colab badge workflow (#48.1)

**New file: `.github/workflows/pr-colab-badge.yml`**

- Trigger: `pull_request: [opened, synchronize, reopened]`.
- Permission: `pull-requests: write`.
- Action: idempotently inject a delimited block into the PR body via `gh pr edit --body`, pointing at the Colab notebook for the *PR's branch*:

  ```text
  <!-- colab-badge-start -->
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenJus/Efficient-SAM3-Finetuning/blob/<branch>/notebooks/colab_gpu_tests.ipynb)
  <!-- colab-badge-end -->
  ```

- Idempotency: detect existing delimiters and replace the block; never duplicate. Survives force-pushes and re-runs.
- Implementation: bash + `gh` CLI, all third-party actions SHA-pinned.

### 4.5 Colab GPU notebook updates (#48.2)

**Edit: `notebooks/colab_gpu_tests.ipynb`**

- Set `metadata.colab.accelerator: "GPU"` so Colab pre-selects GPU on open.
- Add an **early guard cell** (first code cell) that:
  - Runs `!nvidia-smi`.
  - Asserts a CUDA GPU is present (fail loudly if `nvidia-smi` errors or no GPU is reported).
  - **Warns (does not fail)** if the detected GPU is not a T4 — print a clear message that the test suite is calibrated for T4 and other GPUs may produce different timing/memory characteristics.
- In-cell markdown documents: Colab can pin accelerator *class* (GPU vs. CPU) but not GPU *model*; T4 is the common-but-not-guaranteed assignment.

### 4.6 Stateful test-skipping convention (#48.3)

**Edit: `scripts/run_gpu_tests.sh`** — add a header comment block documenting the `--deselect` convention:

- Claude (or any operator) appends `--deselect <nodeid>` flags to the pytest invocation as tests are confirmed passing on GPU.
- The mandatory **final all-green pass** strips all `--deselect` flags and re-runs to prove the full suite is green on a real GPU before the PR can land.
- CI enforces that no `--deselect` flags remain at merge time.

**New CI check** in `.github/workflows/ci.yml`: a job step that greps `scripts/run_gpu_tests.sh` for the literal `--deselect` and **fails the PR** if any are found. Implementation: a few lines of bash, no new action dependency.

### 4.7 `scripts/public-flip-bootstrap.sh` — idempotent automation

Two subcommands: `pre-flip` and `post-flip`. Each step detects current state and either no-ops or reports `already configured`.

#### 4.7.1 `pre-flip` (works on a private repo)

1. `gh repo edit --description "Parameter-efficient finetuning of SAM3.1 for instance segmentation on a single consumer GPU"`
2. `gh repo edit --add-topic sam,sam3,segmentation,instance-segmentation,peft,lora,qlora,fine-tuning,pytorch,huggingface,computer-vision,colab`
3. `gh api repos/{owner}/{repo}/branches/main/protection -X PUT` with body requiring:
   - Pull request before merge.
   - Required status checks: `ci.yml`, `security.yml`.
   - No force-push.
   - No deletion.
   - Linear history.

#### 4.7.2 `post-flip` (requires repo to already be public)

1. `gh api repos/{owner}/{repo} -X PATCH -f security_and_analysis[secret_scanning][status]=enabled -f security_and_analysis[secret_scanning_push_protection][status]=enabled`
2. `gh api repos/{owner}/{repo}/vulnerability-alerts -X PUT`
3. `gh api repos/{owner}/{repo}/automated-security-fixes -X PUT`
4. `gh api repos/{owner}/{repo}/private-vulnerability-reporting -X PUT`

#### 4.7.3 Idempotency contract

- Description: read existing description; skip if exact match.
- Topics: read existing topic list; only add missing ones (`--add-topic` is already additive, but the script reports which were added vs. already present).
- Branch protection: PUT is idempotent by GitHub semantics; the script confirms the resulting policy matches the desired state and prints a diff on mismatch.
- Each `post-flip` API call: PUT/PATCH endpoints are idempotent; script tolerates `409`/`422` "already enabled" responses and reports them as `already configured`.

A second invocation of `pre-flip` or `post-flip` must complete with exit code 0 and produce only `already configured` lines.

### 4.8 `docs/public-flip-runbook.md` — operator flip-day script

Numbered steps the operator copy-pastes:

1. Review and merge this PR (`tracking/52-public-flip` → `main`).
2. `scripts/public-flip-bootstrap.sh pre-flip` — description, topics, branch protection.
3. **Full-history gitleaks sweep**: `gitleaks detect --no-banner --redact --verbose` against the whole history. See §6 for the rewrite escalation path if a hit is found.
4. `gh repo edit --visibility public --accept-visibility-change-consequences` — the actual flip.
5. `scripts/public-flip-bootstrap.sh post-flip` — secret scanning, Dependabot, private vulnerability reporting.
6. Apply retroactive tags (planner-resolved SHAs as a copy-pasteable block) + `v0.5.0` on the merge commit. `git push --tags`. Draft a GitHub Release for `v0.5.0`.

Each step has a short "what success looks like" sentence and a "if this fails" pointer (to the rollback section in §7 or to the rewrite escalation in §6).

### 4.9 `RELEASING.md`

Short doc covering:

- SemVer scheme: `vMAJOR.MINOR.PATCH`.
- **Annotated tags only** (`git tag -a`), with a one-line message describing the milestone.
- Retroactive tag scheme used at flip time (the table from §3.1).
- How to cut a release going forward: tag the commit, push the tag, draft a GitHub Release with release notes summarizing changes since the previous tag.
- Pointer to GitHub Releases as the changelog surface; no `CHANGELOG.md` until/unless v1.0.

---

## 5. Pre-flip Orchestrator Work (not committed files)

These tasks the implementer executes *during* the PR, with results recorded as PR comments / runbook content. They are not committed code.

### 5.1 Issue-body sensitivity audit

Re-run `gh issue list --state open --json number,title` at execution time (since the open-issue set drifts). Currently open: #9, #16, #20, #22, #23, #24, #33, #34, #35, #36, #44, #51, #52, #53. Skim every body for:

- Personal names, email addresses, GitHub handles other than the project owner.
- Internal URLs (corporate intranet, private gists, signed S3 links).
- Prior-employer references or non-public project names.

Report findings (or `clean`) as a PR comment **before** the operator runs the flip. If anything is found, edit the issue body in place; do not delete the issue.

### 5.2 Retroactive-tag SHA resolution

Planner walks `git log --oneline --all` and resolves the milestone commit for each tag in the §3.1 table. Output is a copy-pasteable block in `docs/public-flip-runbook.md` step 6, of the form:

```bash
git tag -a v0.1.0 <sha> -m "v0.1.0: first working training loop"
git tag -a v0.2.0 <sha> -m "v0.2.0: eval pipeline + MetricsReport"
git tag -a v0.3.0 <sha> -m "v0.3.0: LoRA + QLoRA support"
git tag -a v0.4.0 <sha> -m "v0.4.0: W&B tracking + CI hardening"
git tag -a v0.5.0 <merge-sha> -m "v0.5.0: public flip"
git push --tags
```

`<merge-sha>` for `v0.5.0` is the merge commit of this PR and is filled in by the operator at runbook step 6, not by the planner.

---

## 6. History-Sweep Policy (Pre-Authorized Rewrite)

The owner has **pre-authorized** a `git filter-repo` rewrite if the full-history gitleaks sweep (runbook step 3) flags anything. The spec encodes the workflow so the operator does not need to ask:

1. Implementer/operator runs `gitleaks detect --no-banner --redact --verbose` against full history during runbook step 3.
2. If clean: continue to step 4 (visibility flip).
3. If a hit is reported:
   - **Halt** before the visibility flip.
   - Document the redacted finding in the PR thread (gitleaks output redacts the secret value by default).
   - Run `git filter-repo --replace-text <patterns>` (string substitution for credentials) or `--invert-paths` (file removal) to scrub the offending content.
   - Force-push the rewritten `main`.
   - Rebase the flip branch onto the rewritten history.
   - Re-request review.
   - Resume the runbook at step 4.

### 6.1 Consequences (called out explicitly)

- **Force-pushing `main` invalidates all open PRs and local clones.** Open PRs need to be re-based by their authors (or, in this solo project, the owner). All existing clones — including CI caches — must be reset.
- **The credential, if real, is treated as compromised.** Rotation is mandatory regardless of rewrite success. A scrubbed history does not undo any exposure that happened while the secret was reachable.
- The spec **recommends doing the sweep on a quiet day with no other open PRs** to minimize collateral damage from the force-push.
- History rewrites are **not reversible** (see §7).

---

## 7. Risk and Rollback

| Action | Reversible? | How |
| --- | --- | --- |
| `gh repo edit --visibility public` | Yes | `gh repo edit --visibility private` |
| Branch protection rule | Yes | Settings UI or `gh api repos/{owner}/{repo}/branches/main/protection -X DELETE` |
| Description / topics | Yes | Edit via `gh repo edit` |
| Secret scanning, Dependabot, private vulnerability reporting | Yes | Disable via Settings UI or corresponding `gh api ... -X DELETE` |
| Retroactive tags | Yes (cosmetically) | `git tag -d <tag> && git push origin :refs/tags/<tag>` (best avoided post-release) |
| **`git filter-repo` history rewrite** | **No** | Once force-pushed and other clones have fetched, no clean rollback |
| `scripts/public-flip-bootstrap.sh` mid-run failure | Yes | Idempotent — re-run from the top |

---

## 8. Testing / Done Definition

The PR is mergeable when, and the flip is complete when:

### 8.1 Mergeable PR

- All CI green on the PR — both `ci.yml` and `security.yml`.
- New `--deselect` grep check (§4.6) is green.
- `scripts/public-flip-bootstrap.sh pre-flip` and `post-flip` are each idempotent — a second invocation completes with exit code 0 and produces only `already configured` output. Validated in PR by running `pre-flip` against the (still-private) repo as a dry-run smoke test where steps are safe to apply (description, topics) and reporting back what would change for the branch protection step.
- Issue-body sensitivity audit reported (clean or remediated) in a PR comment.
- Retroactive-tag SHA block is present in `docs/public-flip-runbook.md` step 6.

### 8.2 Post-flip operator-confirms

- GitHub repo Insights → Community Standards page shows the full checklist green: Description, README, Code of Conduct, Contributing, License, Security policy, Issue templates, Pull request template.
- First `codeql.yml` run succeeds (CodeQL only works on public repos with GHAS; flip enables it).
- Repo Settings → Code security shows: secret scanning, push protection, private vulnerability reporting, and Dependabot security updates all enabled.
- Retroactive tags `v0.1.0`–`v0.4.0` and `v0.5.0` are visible on the Tags page; `git push --tags` succeeded.
- A `v0.5.0` GitHub Release exists (drafted in runbook step 6, published by the operator).
- The runbook copy-paste path actually worked end-to-end — operator confirms in a closing PR comment / issue close.

### 8.3 Explicitly NOT validated by this PR

- That CodeQL finds zero alerts on first run (it might find advisories — those are followed up in separate issues).
- That gitleaks finds zero hits — if it does, §6 escalates rather than fails the PR.
- External-contribution flow (PRs from forks running CI) — out of scope; the project is not accepting contributions.

---

## 9. File Layout

```text
.github/
  ISSUE_TEMPLATE/
    bug_report.yml          # new
    config.yml              # new — blank_issues_enabled: false; feature-request contact link
  PULL_REQUEST_TEMPLATE.md  # new
  workflows/
    codeql.yml              # new (#31)
    pr-colab-badge.yml      # new (#48.1)
    ci.yml                  # edit — add --deselect grep check (#48.3)

docs/
  public-flip-runbook.md    # new — operator flip-day script
  superpowers/specs/
    2026-05-19-public-flip-design.md  # this spec

notebooks/
  colab_gpu_tests.ipynb     # edit — accelerator: "GPU" + guard cell (#48.2)

scripts/
  public-flip-bootstrap.sh  # new — pre-flip / post-flip idempotent automation
  run_gpu_tests.sh          # edit — header comment documenting --deselect convention (#48.3)

CITATION.cff                # new
CODE_OF_CONDUCT.md          # new
CONTRIBUTING.md             # new
README-dev.md               # new (#47)
README.md                   # edit — status line, badge row, GH_TOKEN caveat removal, README-dev pointer
RELEASING.md                # new
SECURITY.md                 # new
```

No deletions of existing files. README content is **moved** into `README-dev.md`, not duplicated.
