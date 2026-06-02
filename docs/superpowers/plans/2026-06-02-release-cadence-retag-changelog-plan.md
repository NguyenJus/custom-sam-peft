# Release-cadence retag + CHANGELOG revival + stale-ref fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **READ THIS FIRST — this is an unusual plan.** It mixes a **destructive, human-supervised ops runbook (Phase 2)** with **normal doc PR work (Phases 1 & 3)**. Phase 2 is **NOT implementer-codeable**: it deletes tags on origin and purges a container registry (both irreversible), and requires interactive auth and explicit user map-approval. A subagent must **never** autonomously execute Phase 2 — the human-supervised orchestrator runs it step-by-step behind the confirmation gates below.

**Spec:** [`docs/superpowers/specs/2026-06-02-release-cadence-retag-changelog-design.md`](../specs/2026-06-02-release-cadence-retag-changelog-design.md)

**Issues:**

- [#235](https://github.com/NguyenJus/custom-sam-peft/issues/235) — *Revive the CHANGELOG (stale since 0.12.0); re-link from README.* (Phase 3, Part C)
- [#241](https://github.com/NguyenJus/custom-sam-peft/issues/241) — *Stale doc reference: `data.prompt_mode` documented in `notebooks/README.md` after removal.* (Phase 1, Part D)

**Goal:** Replace the 64-tag per-PR noise with a clean, milestone-based `v*` tag history starting at `v0.1.0` and a purged GHCR registry (Part B, runbook), then revive `CHANGELOG.md` clean against that new baseline and re-link it from `README.md` (Part C), bundled with a one-row stale-reference deletion in `notebooks/README.md` (Part D).

**Architecture:** Three phases mapping to spec §4. **Phase 1** is a one-row Markdown deletion (independently shippable, no retag dependency). **Phase 2** is a gated destructive ops runbook (git tags on origin + GHCR purge + a temporary `docker.yml` pause) — human-supervised, not subagent work. **Phase 3** is a normal doc PR sequenced **after** Phase 2 so CHANGELOG entries mirror the new milestone tags; Part D folds into this PR for file-area cohesion.

**Tech Stack:** `git` (tag/push), `gh` CLI (`gh workflow disable/enable`, `gh auth refresh`, GHCR package API or web UI), `hatch-vcs` (version derived from tags — no manual edit), `markdownlint-cli2` (CI lint gate).

**Out of scope (do NOT plan or write — spec §2):**

- **Part A** — the go-forward tagging policy. It lives in the user's **private global workflow instructions** and is handled interactively. Do not write policy text into this repo.
- **`docs/RELEASING.md`** — the user keeps the release process in private instructions only. Do **not** add a release-process doc.

---

## Phasing and dependencies

Three phases, mapping 1:1 to spec §4. Dependency arrows are real: Phase 3's CHANGELOG entries mirror Phase 2's new milestone tags, and Part D folds into Phase 3's PR.

| Phase | Part | Nature | Depends on | Subagent-codeable? |
| --- | --- | --- | --- | --- |
| 1 | D (#241) | One-row `.md` deletion | nothing | Yes |
| 2 | B | Destructive ops runbook (tags + GHCR + workflow pause) | final `main`; GATE 1; GATE 2 | **No — human-supervised** |
| 3 | C (#235) + D bundled | Normal doc PR | Phase 2 (mirrors new tags) | Yes (Part C/D), keyed to Phase 2 output |

**Why Phase 1 is its own block even though it ships inside Phase 3's PR.** Part D has zero dependency on the retag runbook and is independently shippable. Carving it out as its own phase guarantees it is **never blocked** by the Phase 2 gates. In practice the orchestrator lands the Phase 1 edit on the Phase 3 doc branch (so all three Markdown files ride one PR), but the edit itself can be made and verified at any time, independent of Phase 2.

### Phase boundary interface contracts

- **Phase 1 → consumers:** `grep -rn 'prompt_mode\|box_hint' README.md notebooks/ docs/ARCHITECTURE.md` returns **no live/stale reference** — only the accurate `docs/ARCHITECTURE.md:6` "it was removed" prose remains. (Historical specs under `docs/superpowers/specs/` are left as-is.)
- **Phase 2 → Phase 3 (what Phase 3 consumes):** a clean `v*` tag set on origin starting at `v0.1.0` (the **GATE-1-approved milestone map**); a purged GHCR `custom-sam-peft` package showing a **single clean current image**; and a hatch-vcs version reflecting the **new top milestone**. Phase 3's CHANGELOG entries are keyed to exactly these milestone tags — the approved map is the source of truth for which `## [vX.Y.Z]` headings exist.
- **Phase 3 → done:** `CHANGELOG.md` revived clean from the new baseline (no 19-release backfill); the prior `[Unreleased]` content promoted into the appropriate milestone entry; two version-agnostic CHANGELOG pointers in `README.md` (banner area + Configuration section, **no pinned version**); all touched `.md` pass markdownlint-cli2.

---

## Repo-state facts (verified at planning time against this worktree)

These anchor the tasks; re-verify at execution time (especially the tag set, which Phase 2 mutates):

- **64 `v*` tags** present (`v0.1.0` … `v0.33.1`), plus one non-`v*` tag `safety/pre-reset-main` (leave it — it is not a release tag).
- **0 GitHub Releases** (`gh release list` empty) — deleting tags orphans no releases, which is what makes the cleanup safe.
- **`.github/workflows/docker.yml:5`** triggers on `push: tags: ["v*"]` — every `v*` push publishes a GHCR image. This is why the bulk historical retag must run with the workflow **paused**.
- **`gh` token scopes:** `gist, read:org, repo` — **lacks** `delete:packages` / `read:packages`. Phase 2 GATE 2 acquires them.
- **Stale ref:** only `notebooks/README.md:42` is live/stale. `README.md` has **zero** `prompt_mode`/`box_hint` refs (fixed upstream). `docs/ARCHITECTURE.md:6` is accurate "removed" prose — **stays**.
- **`README.md`:** the WIP banner blockquote is around line 10–11 (`> **⚠️ Work in progress.**`); the `### Configuration` section is at line 133 (body line 135). README currently has **no** CHANGELOG link.
- **`CHANGELOG.md`:** populated `## [Unreleased]` block (lines ~10–50) with three sections — "Added — eval GT-vs-Pred visualization", "Breaking — text-primary prompt invariant (#126)", "Removed — box_hint localization-hint curriculum (#88)" — followed by `## [0.12.0]`, `## [0.11.0]`, etc. (167 lines total).

---

## Markdown-lint gate (applies to every commit landing on a ready PR in Phases 1 & 3)

CI lints Markdown via `.github/workflows/ci.yml:100`:

```text
npx --yes markdownlint-cli2 --config .config/markdownlint-cli2.jsonc "**/*.md" "#node_modules"
```

Config `.config/markdownlint-cli2.jsonc` disables MD013/MD018/MD029/MD060 and ignores `.venv/**`; all other default rules are active. This dev box has **no system node/npx**, so run the linter via Python-bundled Node (pin `@0.14.0` for node-18 compat; copy the config to an accepted temp filename first):

```bash
cp .config/markdownlint-cli2.jsonc /tmp/x.markdownlint-cli2.jsonc
uv run --no-project --with nodejs-bin python -c "
from nodejs import node, npx
import os, sys
os.environ['PATH'] = os.path.dirname(node.path) + os.pathsep + os.environ['PATH']
sys.exit(npx.run(['--yes','markdownlint-cli2@0.14.0','--config','/tmp/x.markdownlint-cli2.jsonc', *sys.argv[1:]]).returncode)
" CHANGELOG.md README.md notebooks/README.md
```

**Caveat:** local `@0.14.0` lags CI's latest; a clean local run can still fail CI on a newer rule (e.g. MD060 historically). If CI's `lint-hygiene` job fails on a code that can't be reproduced locally, fetch the job log for the exact `MD0xx` and fix the construct or disable the rule in config. Run this gate **before any commit** landing on a ready PR (Phases 1 & 3). Phase 2 touches no tracked files, so the gate does not apply there.

---

## Phase 1 — Part D (#241): delete the stale `data.prompt_mode` row

**Objective:** Remove the single stale config-table row documenting the removed `data.prompt_mode` field, leaving the grep-clean invariant. Independently shippable; no dependency on Phase 2.

**Files:**

- Modify: `notebooks/README.md:42`

**Boundary interface contract (consumed downstream):** `grep -rn 'prompt_mode\|box_hint' README.md notebooks/ docs/ARCHITECTURE.md` returns no live/stale ref — only `docs/ARCHITECTURE.md:6` accurate "removed" prose remains.

**Routing note (for the orchestrator dispatching this):** easy/trivial single-line Markdown deletion — haiku or sonnet, effort high. Bundle the resulting edit onto the Phase 3 doc branch.

### Task 1.1: Delete the stale table row

**Files:**

- Modify: `notebooks/README.md` (delete line 42)

- [ ] **Step 1.1-1: Confirm the exact stale line is present and unique**

Run:

```bash
grep -n 'data.prompt_mode' notebooks/README.md
```

Expected: exactly one match at line 42:

```text
42:| `data.prompt_mode` | `"text"` | Text-class prompting (v0 only; `"bbox"` is planned). |
```

- [ ] **Step 1.1-2: Delete the row**

Delete this entire line (and only this line) from `notebooks/README.md`:

```text
| `data.prompt_mode` | `"text"` | Text-class prompting (v0 only; `"bbox"` is planned). |
```

It sits at the end of the config table — the line above it is the `data.format` row. Do not touch the `data.format` row or the table header. Use the native Edit tool with the full line as `old_string` so the match is unambiguous.

- [ ] **Step 1.1-3: Verify the grep-clean invariant (boundary contract)**

Run:

```bash
grep -rn 'prompt_mode\|box_hint' README.md notebooks/ docs/ARCHITECTURE.md
```

Expected: **exactly one** line — `docs/ARCHITECTURE.md:6` (the accurate "the `box_hint` localization-hint curriculum was removed in #88" prose). No `notebooks/README.md` match, no `README.md` match. (If the orchestrator excludes `docs/superpowers/specs/` from the grep, that one ARCHITECTURE line is the sole result.)

- [ ] **Step 1.1-4: Run the Markdown-lint gate on the touched file**

Run the markdownlint command from the gate section above against `notebooks/README.md`. Expected: clean (exit 0) for that file. Fix any finding before proceeding.

- [ ] **Step 1.1-5: Commit (folds onto the Phase 3 doc branch)**

This edit lands on the **same branch as Phase 3** so all three Markdown files ride one PR. If executing Phase 1 in isolation first, commit on the doc branch:

```bash
git add notebooks/README.md
git commit -m "docs(#241): remove stale data.prompt_mode config-table row"
```

---

## Phase 2 — Part B: retag + GHCR purge (GATED DESTRUCTIVE OPS RUNBOOK)

> **STOP. This phase is not implementer-codeable.** It is a human-supervised runbook of irreversible git/`gh` operations against **origin** and the **GHCR registry**. A subagent must **not** execute it autonomously. The orchestrator runs each step interactively, pausing at every gate for explicit user confirmation. No TDD, no commits to the repo (Part B changes tags on origin and the registry — **not tracked files**; the docker workflow is paused via `gh workflow disable/enable`, leaving no committed change to revert).

**Objective:** Replace the 64-tag per-PR noise with a clean milestone-based `v*` tag history starting at `v0.1.0`, purge the GHCR image flood, and publish exactly one clean current image — all behind confirmation gates.

**Touched (not files):** git tags on origin; the GHCR `custom-sam-peft` package (owner `nguyenjus`); the `Docker` workflow enabled/disabled state (transient).

**Preconditions (gate on these before starting):**

- **Final `main`:** the user's in-flight PRs are merged to `main`, so the milestone map is computed against final `main`. Confirm with the user before step 1.
- **GATE 1 (map approval):** explicit user approval of the proposed tag→commit map — obtained at step 1, **before any destructive step**.
- **GATE 2 (token scope):** `gh` token carries `delete:packages` + `read:packages` (or the user will purge via the GHCR web UI) — acquired at step 2, **before the purge (step 3)**.

**Boundary interface contract (what Phase 3 consumes):** a clean `v*` tag set on origin starting at `v0.1.0` (the approved map); a purged GHCR package with one clean current image; a hatch-vcs version at the new top milestone. The approved map is the authoritative list of milestone tags Phase 3's CHANGELOG mirrors.

### Runbook steps (ordered; gates and irreversibility called out inline)

- [ ] **Step 2-1: Propose the milestone tag map — GATE 1**

Confirm `main` is final (all in-flight PRs merged):

```bash
git fetch origin
git log --oneline origin/main | head -40
```

Re-read the full commit history of final `main` and propose a **small** set of milestone tags starting at `v0.1.0`, placed only at **genuine feature-block boundaries** — **not** one per PR. Milestone rule (spec §3.1 step 1): a tag marks a release you'd point someone at — a breaking change landed, a coherent feature block completed, meaningful user-facing surface accumulated, or notable count/time elapsed since the prior milestone. Produce the concrete `tag → commit-sha` mapping **at execution time** against final `main`.

**GATE 1 (USER CONFIRMATION REQUIRED — blocking).** Present the proposed `tag → commit` map to the user and obtain **explicit approval**. **Do not proceed past this step without it.** Everything after this point is destructive on origin/registry.

> **Optional safety snapshot (recommended before any destruction):** the spec notes a `safety/pre-reset-main` tag may exist as a pre-cleanup snapshot. If one is created here, name it explicitly (e.g. `git tag safety/pre-reset-main <current-HEAD>`) and **leave it in place** — step 4 deletes only the 64 `v*` release tags, never a named safety tag.

- [ ] **Step 2-2: Acquire GHCR token scope — GATE 2**

The current token (`gist, read:org, repo`) lacks the package-delete scopes. The **user** runs this one-time interactive refresh (interactive auth; no token is stored in the repo):

```sh
gh auth refresh -s delete:packages,read:packages
```

…**or** elects to perform the registry purge (step 3) via the **GHCR web UI** instead.

Confirm the scope landed:

```sh
gh auth status
```

Expected: scopes now include `delete:packages` and `read:packages`.

**GATE 2 (CAPABILITY — blocking).** Do **not** proceed to step 3 until the scope is present (or the user has committed to the web-UI purge path).

- [ ] **Step 2-3: Purge GHCR (DESTRUCTIVE — IRREVERSIBLE on the registry)**

Delete the image versions for the `custom-sam-peft` container package (owner `nguyenjus`), clearing the ~64-image flood from the old per-tag publishes.

Enumerate first (read-only), then delete. Via the API:

```sh
# List versions (read:packages)
gh api -H "Accept: application/vnd.github+json" \
  "/users/nguyenjus/packages/container/custom-sam-peft/versions" --paginate

# Delete each version id (delete:packages) — IRREVERSIBLE
gh api --method DELETE \
  "/users/nguyenjus/packages/container/custom-sam-peft/versions/<VERSION_ID>"
```

**USER CONFIRMATION REQUIRED** before issuing any DELETE. Alternatively the user purges via the GHCR web UI. Step 6 republishes one clean current image, so it is fine to clear all existing versions here.

- [ ] **Step 2-4: Delete all 64 `v*` tags, local and origin (DESTRUCTIVE — IRREVERSIBLE on origin)**

This is **safe** (0 GitHub Releases → no orphaned releases). Target **only** the 64 `v*` release tags; **leave** `safety/pre-reset-main` (or any named safety tag).

Capture the exact list, then delete:

```sh
# Snapshot the v* tag list (sanity-check the count is ~64 before deleting)
git tag -l 'v*' | sort -V | tee /tmp/old_v_tags.txt
wc -l /tmp/old_v_tags.txt

# Local delete
git tag -d $(git tag -l 'v*')

# Origin delete (IRREVERSIBLE) — batched
git push origin --delete $(cat /tmp/old_v_tags.txt)
```

**USER CONFIRMATION REQUIRED** before the `git push origin --delete`. Verify afterward:

```sh
git ls-remote --tags origin 'refs/tags/v*'
```

Expected: empty (no `v*` tags on origin).

- [ ] **Step 2-5: Create the new milestone tags locally (per the GATE-1-approved map)**

For each milestone in the approved map:

```sh
git tag v<version> <commit-sha>
```

Verify the local set matches the approved map exactly:

```sh
git tag -l 'v*' | sort -V
```

Expected: only the approved milestone tags, starting at `v0.1.0`. Nothing is pushed yet.

- [ ] **Step 2-6: Push the new tags with `docker.yml` paused, then publish one clean image (IRREVERSIBLE on origin)**

`docker.yml` fires on every `v*` push (trigger `push: tags: ["v*"]`), so pushing tags pointing at **old** commits would build images against **stale Dockerfile states** — wasteful and possibly failing. Use the **temporary-pause** option (spec §3.1 step 6, recommended — least invasive, no committed workflow change to revert):

```sh
# 1. Pause the Docker workflow for the bulk historical retag
gh workflow disable docker.yml

# 2. Push ALL milestone tags while paused (no image builds fire) — IRREVERSIBLE on origin
git push origin $(git tag -l 'v*')

# 3. Re-enable the workflow
gh workflow enable docker.yml
```

Then publish **exactly one** clean image from the **current Dockerfile state**: re-push only the **current baseline (HEAD-adjacent / top) milestone tag** with the workflow enabled so the `Docker` workflow fires once. If that top tag was already pushed in step 2 above (while paused, so no build fired), force the trigger by re-pushing it after re-enable:

```sh
# Delete + re-push ONLY the top milestone tag so docker.yml fires once
git push origin --delete v<TOP_MILESTONE>
git push origin v<TOP_MILESTONE>
```

**USER CONFIRMATION REQUIRED** before each push to origin (GATE 3 is the same approval as GATE 1 — no separate map confirmation, but these pushes are the irreversible origin steps, so they run only after the map is approved and the workflow is confirmed paused/re-enabled appropriately).

- [ ] **Step 2-7: Verify the Phase 2 boundary contract**

```sh
# Origin carries only the approved milestone tags, starting at v0.1.0
git ls-remote --tags origin 'refs/tags/v*'

# Docker workflow is re-enabled
gh workflow view docker.yml

# Exactly one Docker run fired for the baseline tag (not ~64)
gh run list --workflow=docker.yml --limit 5

# hatch-vcs version reflects the new top milestone (untagged HEAD → devN+gSHA off it)
git describe --tags --abbrev=0
```

Expected: origin `v*` set equals the approved map (top = the new milestone, far below the old `v0.33.1`); `docker.yml` enabled; a single recent Docker run; `git describe` reports the new top milestone. **Record the approved map** — Phase 3 keys its CHANGELOG headings to it.

**Documented consequence (expected, not a bug):** the hatch-vcs version **drops** from `v0.33.1` to the new top milestone (≈ `v0.x`). Acceptable and intended — registry purged, 0 GitHub Releases, no external version pins. Untagged commits continue to get honest `devN+g<sha>` versions between milestones.

---

## Phase 3 — Part C (#235): CHANGELOG revival + README re-link, bundled with Part D

> **Sequenced AFTER Phase 2.** The new milestone tags from the GATE-1-approved map are the source of truth for which `## [vX.Y.Z]` headings the revived CHANGELOG carries. Do not start Phase 3 until Phase 2's boundary contract is verified and the approved map is recorded.

**Objective:** Revive `CHANGELOG.md` clean from the new milestone baseline (no 19-release backfill), promote the existing `[Unreleased]` content into the appropriate new milestone entry, and re-add two **version-agnostic** CHANGELOG pointers to `README.md` (banner area + Configuration section, **no pinned version**). Phase 1's `notebooks/README.md` edit rides this same PR.

**Files:**

- Modify: `CHANGELOG.md`
- Modify: `README.md` (two pointers)
- (Already modified in Phase 1, riding this branch/PR: `notebooks/README.md`)

**Boundary interface contract (done state):** CHANGELOG entries mirror exactly the approved milestone tags; the prior `[Unreleased]` content is promoted into the appropriate milestone entry; README carries two version-agnostic CHANGELOG pointers with no pinned version; all touched `.md` pass markdownlint-cli2.

**Routing note (for the orchestrator dispatching this):** medium (Markdown edits keyed to a known map) — sonnet, effort high. Part C and Part D ship as **one doc PR**.

### Task 3.1: Revive `CHANGELOG.md` clean from the new baseline

**Files:**

- Modify: `CHANGELOG.md`

**Pre-read:** the orchestrator must supply this task the **GATE-1-approved milestone map** from Phase 2 (the list of `v<version>` tags + their dates/scope). The headings written here mirror that map exactly.

- [ ] **Step 3.1-1: Confirm the approved milestone map is available**

The task cannot proceed without it. The map (e.g. `v0.1.0 → <date>/<scope>`, …, `v0.<top> → <date>/<scope>`) determines every `## [vX.Y.Z]` heading. If absent, halt and request it from the orchestrator — do **not** invent milestone numbers.

- [ ] **Step 3.1-2: Rewrite the version-entry section to mirror the new tags**

Keep the existing header preamble unchanged (lines 1–8): the `# Changelog` title, the "All notable changes…" line, the Keep-a-Changelog + SemVer links, and the `---` rule.

Replace everything from `## [Unreleased]` to end-of-file with a clean entry set whose **headings are exactly the approved milestone tags** (newest first), plus a top `## [Unreleased]` placeholder for go-forward use. **Do not backfill** the 19-release per-PR noise — the old `## [0.12.0]` / `## [0.11.0]` entries are part of the noise being escaped; carry forward only content that maps to a real new milestone (see Step 3.1-3 for promotion).

Target structure (illustrative — substitute the actual approved tags/dates):

```markdown
## [Unreleased]

<!-- Add entries for the next milestone here. -->

## [v0.<TOP>] — <YYYY-MM-DD>

### Added

- ...

### Breaking

- ...

### Removed

- ...

## [v0.1.0] — <YYYY-MM-DD>

- Initial milestone baseline.
```

Each heading must correspond to a tag the user approved in Phase 2. No heading may reference a tag outside the approved map.

- [ ] **Step 3.1-3: Promote the current `[Unreleased]` content into the appropriate milestone entry**

The pre-existing `[Unreleased]` block (currently CHANGELOG lines ~10–50) holds three sections that describe **real shipped work** and must be **preserved** by moving them under the milestone entry that the approved map assigns them to (typically the new top milestone):

1. **"Added — eval GT-vs-Pred visualization"** — the `eval.visualize` / `eval.visualize_count` knobs and the `csp eval/run --visualize` flags.
2. **"Breaking — text-primary prompt invariant (#126)"** — removed `data.prompt_mode`; removed `Sam3Wrapper.forward(box_hints=)`; removed `BoxPrompts`/`PromptMode`; removed the three `prompt_mode == "bbox"` guards.
3. **"Removed — box_hint localization-hint curriculum (#88)"** — removed the `box_hint` curriculum + `BoxHintSchedule`; `SupportPrompts` retained as a no-op seam; resume tolerates stale `box_hint_p`.

Move these three sections verbatim (preserving their bullet content) under the assigned milestone heading. After the move, `## [Unreleased]` is empty except the go-forward placeholder comment. Do not drop any of the three sections' substance.

- [ ] **Step 3.1-4: Verify no 19-release backfill leaked in**

Run:

```bash
grep -n '^## \[' CHANGELOG.md
```

Expected: only `## [Unreleased]` plus the approved milestone headings — **no** stray `## [0.12.0]` / `## [0.11.0]` / etc. from the old noise unless one of those exact versions is in the approved map. Confirm every listed heading is in the approved map.

### Task 3.2: Re-add two version-agnostic CHANGELOG pointers to `README.md`

**Files:**

- Modify: `README.md` (banner area near top ~line 11; Configuration section ~line 135)

**Context:** #200 (PR #236) removed two CHANGELOG pointers because they were framed around a stale hardcoded `v0.7.0`. The README now deliberately carries **no pinned version** (memory: *README: no pinned version*). Re-add a pointer in **each** of the two original locations, but **version-agnostic** — link to `CHANGELOG.md` without naming any version.

- [ ] **Step 3.2-1: Add a version-agnostic pointer in the banner area**

The banner area is the WIP blockquote near the top (around line 11, `> **⚠️ Work in progress.**`). Add a version-agnostic CHANGELOG link adjacent to it (a new line within/after the blockquote, or a short line right after it). Example prose — **do not** name a version:

```markdown
> See the [CHANGELOG](CHANGELOG.md) for release history.
```

Do **not** re-introduce a "What's new in v0.7.0" (or any pinned-version) banner.

- [ ] **Step 3.2-2: Add a version-agnostic pointer in the Configuration section**

In the `### Configuration` section (body around line 135, after the `docs/config-schema.md` sentence), add a version-agnostic line linking to the CHANGELOG, framed around release history / the field-rename history rather than a pinned version. Example:

```markdown
See the [CHANGELOG](CHANGELOG.md) for release history and config field-rename notes.
```

Do **not** write "v0.7.0 rename" or any pinned version.

- [ ] **Step 3.2-3: Confirm no pinned version was reintroduced**

Run:

```bash
grep -n -iE 'what.?s new|v0\.[0-9]+\.[0-9]+' README.md
```

Expected: **no** match introduced by this change (no `What's new`, no `vX.Y.Z`). Both new lines must be version-agnostic. Confirm both `CHANGELOG.md` links are present:

```bash
grep -n 'CHANGELOG.md' README.md
```

Expected: two matches (banner area + Configuration section).

### Task 3.3: Lint, verify, commit, PR

- [ ] **Step 3.3-1: Run the Markdown-lint gate on all touched `.md`**

Run the markdownlint command from the gate section against the three touched files:

```bash
cp .config/markdownlint-cli2.jsonc /tmp/x.markdownlint-cli2.jsonc
uv run --no-project --with nodejs-bin python -c "
from nodejs import node, npx
import os, sys
os.environ['PATH'] = os.path.dirname(node.path) + os.pathsep + os.environ['PATH']
sys.exit(npx.run(['--yes','markdownlint-cli2@0.14.0','--config','/tmp/x.markdownlint-cli2.jsonc', *sys.argv[1:]]).returncode)
" CHANGELOG.md README.md notebooks/README.md
```

Expected: exit 0. Fix every finding before committing. (Remember the local-vs-CI version caveat: if CI's `lint-hygiene` later fails on a code not seen locally, fetch the job log for the `MD0xx` and fix/disable accordingly.)

- [ ] **Step 3.3-2: Verify all acceptance criteria (spec §6) that this PR owns**

```bash
# Criterion 4: CHANGELOG headings == approved milestone map, [Unreleased] promoted
grep -n '^## \[' CHANGELOG.md

# Criterion 5: two version-agnostic CHANGELOG pointers, no pinned version
grep -n 'CHANGELOG.md' README.md
grep -n -iE 'what.?s new|v0\.[0-9]+\.[0-9]+' README.md   # expect: no new match

# Criterion 6: stale-ref grep clean (Part D)
grep -rn 'prompt_mode\|box_hint' README.md notebooks/ docs/ARCHITECTURE.md
# expect only docs/ARCHITECTURE.md:6 (accurate "removed" prose)
```

(Criteria 1–3 are owned by Phase 2's runbook verification, step 2-7; criterion 7 is the lint gate, step 3.3-1.)

- [ ] **Step 3.3-3: Commit**

```bash
git add CHANGELOG.md README.md notebooks/README.md
git commit -m "docs(#235,#241): revive CHANGELOG from new baseline, re-link README, drop stale prompt_mode row"
```

- [ ] **Step 3.3-4: Open the PR**

Open a ready PR linking the spec, this plan, and issues #235 + #241. Per the project workflow, pass `--assignee @me` and at least one `--label` (pick an existing label via `gh label list` or create one inline). Notify the user.

---

## Self-review against the spec

- **Part A & `docs/RELEASING.md`:** explicitly excluded — no task plans either (spec §2). ✓
- **Spec §3.1 (Part B runbook):** Phase 2 reproduces all six runbook steps with the three gates (GATE 1 map approval, GATE 2 token scope, the docker-pause/re-enable around the bulk retag) and marks every destructive/irreversible origin/registry step as user-confirmation-required. Exact commands from §3.1 included. ✓
- **Spec §3.2 (Part C):** Task 3.1 revives clean from the new baseline with no backfill and promotes the three `[Unreleased]` sections; Task 3.2 re-adds two version-agnostic pointers with no pinned version. ✓
- **Spec §3.3 (Part D):** Phase 1 deletes only `notebooks/README.md:42`; leaves `docs/ARCHITECTURE.md:6`; README already clean. ✓
- **Spec §4 phasing:** exactly three phases with explicit boundary interface contracts; Phase 1 carved out as independently shippable yet folded into the Phase 3 PR. ✓
- **Spec §5 constraints:** Markdown-lint gate documented and required before Phases 1 & 3 commits; destructive-step confirmation gates honored; no secrets (interactive auth only). ✓
- **Spec §6 acceptance criteria 1–7:** mapped to verification steps (2-7 for 1–3; 3.1-4/3.2-3 for 4–5; 1.1-3/3.3-2 for 6; 3.3-1 for 7). ✓
