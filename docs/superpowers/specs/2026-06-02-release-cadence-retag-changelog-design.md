# Release-cadence redesign: one-time tag/registry cleanup + CHANGELOG revival + stale-ref fix

**Status:** Draft (2026-06-02)

**Tracking:**

- [#235](https://github.com/NguyenJus/custom-sam-peft/issues/235) — *Revive the
  CHANGELOG (stale since 0.12.0); re-link from README.*
- [#241](https://github.com/NguyenJus/custom-sam-peft/issues/241) — *Stale doc
  reference: `data.prompt_mode` documented in `notebooks/README.md` after removal.*

**Scope:** A one-time tag-history and container-registry cleanup that replaces 64
per-PR tags with a small milestone-based tag history starting at `v0.1.0` and a clean
GHCR package (Part B), then revives `CHANGELOG.md` clean from the new baseline and
re-links it from the README (Part C, #235), bundled with a one-row stale-reference
deletion in `notebooks/README.md` (Part D, #241).

---

## 1. Motivation and context

This is a **personal/research repo** (a custom SAM PEFT training tool) with **no
external release consumers** — no downstream packages pin its version, and there are
**0 GitHub Releases** (`gh release list` is empty). That posture is what makes the
destructive cleanup below safe.

Three grounded facts drive the redesign:

1. **Tag-history noise.** The repo accumulated **64 git tags in ~16 days**
   (`v0.1.0` → `v0.33.1`), essentially one tag per merged PR. Because parallel
   sessions each bump independently, tags landed even out of merge order, so the
   semver ordering carries no real meaning. There is **no repo automation** that
   tags — every tag came from a per-PR close-out workflow *policy* that lives outside
   this repo (see [§2 Exclusions](#2-exclusions)). Version is derived from git tags via
   **hatch-vcs** (`pyproject.toml`: `[tool.hatch.version] source = "vcs"`); untagged
   commits already get honest dev versions like `0.33.x.devN+g<sha>`, so tagging every
   merge was never required for versioning to work.

2. **Registry flood.** `.github/workflows/docker.yml` triggers on `tags: ["v*"]`
   (line 5), so **every `v*` tag pushed publishes a GHCR container image**. 64 tags
   produced ~64 image versions. `.github/workflows/ghcr-retention.yml` exists to
   garbage-collect that flood, but the clean fix is to stop minting the noise in the
   first place.

3. **Stale CHANGELOG + doc ref.** `CHANGELOG.md`'s last versioned entry is
   `[0.12.0]` (2026-05-23); ~19+ releases have no entries. README's two CHANGELOG
   pointers were removed in #200 (PR #236) because pointing newcomers at a
   19-release-stale changelog was misleading (#235). Separately, `notebooks/README.md`
   line 42 still documents a removed `data.prompt_mode` config field (removed in #126,
   the text-only invariant; any config carrying it now fails Pydantic
   `extra_forbidden`) (#241).

The redesign escapes the 64-tag noise by replacing it with a clean, milestone-based
tag history, purges the registry, and then revives the changelog against that clean
baseline so it stays maintainable going forward.

---

## 2. Exclusions

These are stated explicitly so a reader does not expect them in this repo's PRs:

| Excluded item | Why it is out of scope |
| --- | --- |
| **Part A — the go-forward tagging policy** (stop per-PR auto-tagging; replace with milestone tagging that self-assesses at close-out and reaches out to the user when a release is warranted, then diffs-since-last-tag → infers semver → drafts a changelog entry → tags → pushes) | This policy lives in the user's **private global workflow instructions**, is not project-specific, and is handled interactively by the user. It is mentioned here only as context: the new cadence is what keeps the revived changelog maintainable, which is why the cleanup is safe and coherent. **Do not write the policy text.** |
| **`docs/RELEASING.md`** | The user chose to keep the release process in private instructions only. No release-process doc is added to this repo. |

---

## 3. Design

The work is three parts: **B** (one-time history/registry cleanup — the core), **C**
(#235 CHANGELOG revival, depends on B), and **D** (#241 stale-ref fix, folds into the
C doc PR). Part A is excluded (§2).

### 3.1 Part B — One-time history & registry cleanup (gated ops runbook)

**Goal:** Replace the 64-tag noise with a clean, milestone-based tag history starting
at `v0.1.0`, and a clean container registry.

This part is a destructive, outward-facing ops runbook. It is **gated** on two
preconditions:

- The user's in-flight PRs being **merged to main** (they are, as of this spec being
  written), so the milestone map is computed against final `main`.
- **User approval of the proposed tag→commit map** before any destructive step.

#### Runbook (ordered, with explicit safety gates)

1. **Propose a clean milestone tag sequence.** Re-read the full commit history of
   final `main` and propose a small set of tags starting at `v0.1.0`, placed only at
   **genuine feature-block boundaries** — *not* one per PR. The milestone rule: a tag
   marks a release you'd point someone at — i.e. a breaking change landed, a coherent
   feature block completed, meaningful user-facing surface accumulated, or notable
   count/time elapsed since the prior milestone. The concrete tag→commit mapping is
   produced **at execution time** against final `main`.
   **GATE 1 (approval):** present the proposed map to the user and obtain **explicit
   approval before any destructive step**. The runbook does not proceed past this point
   without it.

2. **Prerequisite — token scope.** Purging GHCR requires `delete:packages` +
   `read:packages` scopes. The current `gh` token **lacks** them (it carries
   `gist, read:org, repo` — confirmed via `gh auth status`). The user must run a
   one-time interactive refresh:

   ```sh
   gh auth refresh -s delete:packages,read:packages
   ```

   …**or** perform the registry purge via the GHCR web UI. **GATE 2 (capability):**
   the runbook cannot proceed past the purge step (step 3) without this scope (or the
   web-UI alternative).

3. **Purge GHCR.** Delete the image versions for the `custom-sam-peft` container
   package (owner `nguyenjus`). This clears the ~64-image flood produced by the old
   per-tag publishes.

4. **Delete all 64 existing `v*` tags**, locally and on origin:

   ```sh
   git tag -d <tag>...                 # local
   git push origin --delete <tag>...   # remote
   ```

   This is **safe**: there are **0 GitHub Releases**, so deleting tags orphans no
   releases. (Note: a non-`v*` `safety/pre-reset-main` tag exists; the deletion targets
   only the 64 `v*` release tags — leave any explicitly-named safety tag in place if
   one is created as a pre-cleanup snapshot.)

5. **Create the new milestone tags** locally, per the GATE-1-approved map
   (`git tag v<version> <commit>` for each milestone).

6. **Push the new tags — with the docker workflow paused.**
   **Critical interaction:** `docker.yml` fires on every `v*` push, so pushing tags
   that point at **old** commits would trigger image builds against **stale Dockerfile
   states** — wasteful and possibly failing. Two viable resolutions:

   | Option | Mechanism | Trade-off |
   | --- | --- | --- |
   | **(Recommended) Temporary pause** | Disable the `Docker` workflow run during the bulk historical retag (e.g. `gh workflow disable docker.yml`), push all milestone tags, then re-enable (`gh workflow enable docker.yml`). | **Least invasive** — no committed workflow change; nothing to revert in git. |
   | Build-guard | Add a condition to `docker.yml` so the build only runs for the latest/HEAD-adjacent tag. | Requires a committed workflow change (and likely a follow-up revert); more churn. |

   **Use the temporary-pause option.** Then push **only** the current baseline
   (HEAD-adjacent) tag with the workflow re-enabled, so exactly **one clean image** is
   published from the current Dockerfile state. **GATE 3** is the same approval as
   GATE 1 — no separate confirmation, but the push is the irreversible step on origin,
   so it executes only after the map is approved and the workflow is confirmed paused.

#### Documented consequence

The hatch-vcs version **drops** from `v0.33.1` to the new top milestone (≈ `v0.x`).
This is **acceptable and intended**, given the registry purge, 0 GitHub Releases, and
no external version pins. Untagged commits continue to receive honest
`devN+g<sha>` versions between milestones.

### 3.2 Part C — #235 CHANGELOG revival (depends on Part B)

A normal repo doc PR, sequenced **after** Part B so it can mirror the new baseline.

- **Revive `CHANGELOG.md` clean from the new baseline.** Entries mirror the **new
  milestone tags from Part B**, *not* the old 64. **Do not backfill** the 19-release
  per-PR noise.
- **Promote the current `[Unreleased]` content** into the appropriate new milestone
  entry/entries. (`CHANGELOG.md` currently has a populated `[Unreleased]` block: an
  "Added — eval GT-vs-Pred visualization" section and the "Breaking — text-primary
  prompt invariant (#126)" / "Removed — box_hint curriculum (#88)" sections.)
- **Re-add the two README CHANGELOG pointers** that #200 (PR #236) removed. The #236
  diff removed:
  1. a top "What's new in v0.7.0" banner that linked to `CHANGELOG.md`, and
  2. a Configuration-section "Quick reference … v0.7.0 rename" line that linked to
     `CHANGELOG.md`.
  Re-add a pointer in each of those two locations (the banner area near the top and the
  Configuration section). **Constraint:** the originals were framed around a stale,
  hardcoded `v0.7.0`; the README deliberately carries **no pinned version** (memo:
  *README: no pinned version*; #200/PR #236). The re-added pointers must therefore be
  **version-agnostic** prose that links to `CHANGELOG.md` (e.g. "see the
  [CHANGELOG](CHANGELOG.md) for release history / the field-rename table") — do **not**
  re-introduce a `v0.7.0` (or any pinned-version) banner.
- **Keep `CHANGELOG.md` honest going forward:** one entry per real milestone release
  (this is what the excluded Part A cadence sustains).

### 3.3 Part D — #241 stale-ref fix (trivial; folds into the Part C PR)

- **Delete the `data.prompt_mode` table row** at `notebooks/README.md` line 42:

  ```text
  | `data.prompt_mode` | `"text"` | Text-class prompting (v0 only; `"bbox"` is planned). |
  ```

- **Scope clarification (confirmed against the tree):** #241 collapses to this single
  table-row deletion.
  - The originally-cited `README.md:116` ref was **already fixed upstream** — README
    now has **zero** `prompt_mode` / `box_hint` references (confirmed by grep).
  - `docs/ARCHITECTURE.md` line 6 mentions `box_hint` only as accurate "it was removed"
    prose — **leave it**.

---

## 4. Phasing and dependencies

Each phase is an independently reviewable unit; the dependency arrows are real (C
mirrors B's tags; D folds into C's PR).

### Phase 1 — Part D (#241), no main dependency

The stale-row deletion has no dependency on Part B and could ship **independently**.
In practice it folds into the Phase 3 doc PR (file-overlap rationale below), but it is
called out as independently shippable so it is never blocked by the retag gates.

**Interface at boundary:** `grep -rn 'prompt_mode\|box_hint' README.md notebooks/
docs/ARCHITECTURE.md` returns no **live/stale** reference — only the accurate
`docs/ARCHITECTURE.md:6` "it was removed" prose remains. (Historical specs under
`docs/superpowers/specs/` may be left as-is.)

### Phase 2 — Part B retag + GHCR purge (gated)

**Gated on:** final `main` (in-flight PRs merged) **and** user approval of the proposed
tag→commit map (GATE 1), plus the GHCR token scope (GATE 2). Destructive, outward-facing.

**Interface at boundary (what Phase 3 consumes):** a clean `v*` tag set on origin
starting at `v0.1.0` (the approved milestone map), a purged GHCR package with a single
clean current image, and a hatch-vcs version reflecting the new top milestone. Phase 3's
CHANGELOG entries are keyed to exactly these milestone tags.

### Phase 3 — Part C CHANGELOG revival + README re-link, bundled with Part D

**Depends on Phase 2** (entries mirror the new milestone tags). Ships as **one doc PR**.

**File-overlap rationale:** Part C touches `CHANGELOG.md` and `README.md`; Part D
touches `notebooks/README.md`. All three live in the docs/README area, so C and D ship
as **one doc PR sequenced after B** rather than as separate PRs racing on the same area.

---

## 5. Constraints (repo conventions)

- **Markdown lint.** All touched `.md` files (`CHANGELOG.md`, `README.md`,
  `notebooks/README.md`, and this spec) are CI-linted with **markdownlint-cli2**. Run
  the project's markdown linter and fix findings before committing.
- **Destructive-step confirmation gates.** Tag deletion on origin (runbook step 4) and
  the GHCR purge (step 3) are **irreversible on origin**. Even for a research repo,
  every outward-facing destructive step executes only behind an explicit
  user-confirmation gate (GATE 1 approval of the map; GATE 2 token-scope prerequisite).
- **No secrets in files.** The `gh auth refresh` / web-UI path uses interactive auth;
  no token is stored in the repo.

---

## 6. Acceptance criteria

| # | Criterion | Reference |
| --- | --- | --- |
| 1 | Origin carries a small, milestone-based `v*` tag set starting at `v0.1.0` (the user-approved map), and the prior 64 `v*` tags are gone. | [§3.1](#31-part-b--one-time-history--registry-cleanup-gated-ops-runbook), steps 4–6 |
| 2 | The GHCR `custom-sam-peft` package shows a single clean current image (the ~64-image flood purged); pushing the historical milestone tags did **not** trigger stale image builds (docker workflow was paused). | [§3.1](#31-part-b--one-time-history--registry-cleanup-gated-ops-runbook), steps 3 & 6 |
| 3 | Every destructive/outward-facing step ran only after explicit user confirmation (approved tag map; token scope acquired). | [§5](#5-constraints-repo-conventions) |
| 4 | `CHANGELOG.md` is revived clean from the new baseline — entries mirror the new milestone tags (no 19-release per-PR backfill); the prior `[Unreleased]` content is promoted into the appropriate milestone entry. | [§3.2](#32-part-c--235-changelog-revival-depends-on-part-b) |
| 5 | The README carries two version-agnostic CHANGELOG pointers (banner area + Configuration section); **no pinned version** is reintroduced. | [§3.2](#32-part-c--235-changelog-revival-depends-on-part-b) |
| 6 | `grep -rn 'prompt_mode\|box_hint' README.md notebooks/ docs/ARCHITECTURE.md` returns no live/stale reference (only the accurate `docs/ARCHITECTURE.md:6` "removed" prose remains). | [§3.3](#33-part-d--241-stale-ref-fix-trivial-folds-into-the-part-c-pr) |
| 7 | All touched `.md` files pass markdownlint-cli2. | [§5](#5-constraints-repo-conventions) |

---

## 7. File layout

```text
docs/superpowers/specs/2026-06-02-release-cadence-retag-changelog-design.md   NEW (this spec)
CHANGELOG.md          TOUCHED (Part C — revived clean from new baseline)
README.md             TOUCHED (Part C — two version-agnostic CHANGELOG pointers re-added)
notebooks/README.md   TOUCHED (Part D — data.prompt_mode row deleted, line 42)
```

Part B (retag + GHCR purge) is an **ops runbook**: it changes git tags on origin and
the GHCR registry, not tracked repo files. The docker workflow is paused/re-enabled via
`gh workflow disable/enable` (no committed change to `.github/workflows/docker.yml`).
