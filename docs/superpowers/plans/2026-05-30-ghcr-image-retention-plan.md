# GHCR Image Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-30-ghcr-image-retention-design.md`](../specs/2026-05-30-ghcr-image-retention-design.md)
**Issue:** [#169](https://github.com/NguyenJus/custom-sam-peft/issues/169) — *Add GHCR image retention: prune old/untagged versions of custom-sam-peft*
**Branch:** `ghcr-image-retention`

**Goal:** Add a scheduled GitHub Actions workflow that weekly prunes dangling untagged GHCR manifests and retains only the 25 newest tagged releases of `ghcr.io/nguyenjus/custom-sam-peft`, and document the required one-time setup and PAT fallback in `docs/RELEASING.md`.

**Architecture:** A single new workflow file (`.github/workflows/ghcr-retention.yml`) using the referrer-aware `dataaxiom/ghcr-cleanup-action` at SHA-pinned v1.2.1. A new `## GHCR image retention` section is appended to `docs/RELEASING.md`. No source code, no Dockerfile, no CI matrix, no test surface. The only runtime verification gate is `actionlint` + `yamllint` + `markdownlint`; full end-to-end acceptance (dry run + live run) is operator work after merge.

**Tech Stack:** GitHub Actions (`ubuntu-latest`), `dataaxiom/ghcr-cleanup-action@f092b48ba3b604b2a83690dc4b2bbb3392e1045f  # v1.2.1`, `actionlint` (v1.7.7 per CI), `yamllint` (repo config `.config/yamllint.yml`), `markdownlint-cli2` (per CI lint-hygiene job).

---

## Phasing decision

**Single phase.** This change touches exactly two files — one new YAML file and one appended Markdown section. There is no source code, no test suite, no lock file regeneration, and no multi-step dependency chain between the two files. Splitting into multiple phases would add ceremony with zero benefit. A single phase produces a complete, independently-reviewable PR.

---

## File Map

**New files:**

```
.github/workflows/ghcr-retention.yml   NEW
```

**Modified files:**

```
docs/RELEASING.md                       TOUCHED  (## GHCR image retention section appended)
```

No other files are modified. No source code in `src/` is touched. No tests added.

---

## Task 1: Create `.github/workflows/ghcr-retention.yml`

**Model/effort:** sonnet / high.
**Spec:** §3.2 (action choice + SHA pin), §3.3 (retention policy), §3.4 (workflow shape).

**Files:**
- Create: `.github/workflows/ghcr-retention.yml`

**Goal:** Write the retention workflow exactly as specified in spec §3.4, using the SHA-pin comment style from the repo (inline `# v1.2.1` suffix matching `codeql.yml` and `docker.yml`), with the concurrency block matching the repo convention.

- [ ] **Step 1-1: Create `.github/workflows/ghcr-retention.yml`**

Create `.github/workflows/ghcr-retention.yml` with exactly this content:

```yaml
name: GHCR Retention

on:
  schedule:
    # Weekly Monday 07:00 UTC. Staggered 1 h after CodeQL (0 6 * * 1)
    # so the two scheduled jobs don't compete for the same runner pool.
    - cron: "0 7 * * 1"
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Preview planned deletions without deleting"
        type: boolean
        default: false

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  packages: write

jobs:
  prune:
    runs-on: ubuntu-latest
    steps:
      - uses: dataaxiom/ghcr-cleanup-action@f092b48ba3b604b2a83690dc4b2bbb3392e1045f  # v1.2.1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          owner: nguyenjus
          package: custom-sam-peft
          delete-untagged: true
          keep-n-tagged: 25
          exclude-tags: latest
          validate: true
          dry-run: ${{ inputs.dry_run }}
```

**Key design notes (do not modify):**

- `schedule: cron: "0 7 * * 1"` — Monday 07:00 UTC. CodeQL runs at `"0 6 * * 1"` (one hour earlier, see `codeql.yml` line 19). The 1-hour stagger prevents runner pool contention between the two scheduled jobs.
- `workflow_dispatch` with `dry_run: boolean, default: false` — allows a maintainer to trigger a preview run on demand before the first live scheduled run.
- `permissions: packages: write` — least privilege; the actual delete capability requires the one-time Actions-access Admin grant documented in `docs/RELEASING.md` (see spec §3.1).
- `concurrency: group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true` — matches the convention used in `docker.yml` (lines 7–9) and `codeql.yml` (lines 21–23) exactly.
- `dataaxiom/ghcr-cleanup-action@f092b48ba3b604b2a83690dc4b2bbb3392e1045f  # v1.2.1` — the two-space gap before the comment matches the alignment style in `codeql.yml` (`uses: github/codeql-action/analyze@7211b7c8077ea37d8641b6271f6a365a22a5fbfa  # v3.29.0`).
- `exclude-tags: latest` — defense-in-depth; `:latest` is inherently within the newest 25 but is explicitly excluded to guard against edge-case interactions between the tag-count logic and the floating tag (spec §3.3).
- `validate: true` — triggers the action's post-prune integrity check to confirm all remaining tagged images are intact.
- `dry-run: ${{ inputs.dry_run }}` — wired to the `workflow_dispatch` input; evaluates to `false` on scheduled runs (no `inputs` context) which is the desired live-run behavior.

- [ ] **Step 1-2: Verify with `actionlint`**

Download and run `actionlint` v1.7.7 (the exact version CI's `lint-hygiene` job installs per `.github/workflows/ci.yml` lines 89–90):

```bash
bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7
./actionlint -color /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.github/workflows/ghcr-retention.yml
```

Expected: exits 0 with no errors. If `actionlint` warns about the SHA-pinned action reference being unresolvable (it cannot validate third-party action SHAs at parse time), that is not an error — the SHA format is correct as long as it is the 40-character hex string shown above.

If `actionlint` is already present in the worktree from a prior step, use `./actionlint` directly.

- [ ] **Step 1-3: Verify with `yamllint`**

Run yamllint using the repo's config (`.config/yamllint.yml`) exactly as CI does (see `ci.yml` line 96):

```bash
uv run --with yamllint yamllint -c /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.config/yamllint.yml /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.github/workflows/ghcr-retention.yml
```

Expected: exits 0 with no errors or warnings. The repo's yamllint config (`.config/yamllint.yml`) disables `line-length` and `document-start` globally, and sets `truthy.check-keys: false` to suppress false positives on `on:` triggers — so the YAML above should pass without inline disable comments.

- [ ] **Step 1-4: Spot-check SHA pin is 40-hex characters**

```bash
grep 'uses:' /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.github/workflows/ghcr-retention.yml
```

Expected output:

```
      - uses: dataaxiom/ghcr-cleanup-action@f092b48ba3b604b2a83690dc4b2bbb3392e1045f  # v1.2.1
```

Confirm the SHA `f092b48ba3b604b2a83690dc4b2bbb3392e1045f` is exactly 40 characters and the comment `# v1.2.1` is present on the same line.

- [ ] **Step 1-5: Commit the workflow file**

```bash
git -C /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention add .github/workflows/ghcr-retention.yml
git -C /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention commit -m "ci: add GHCR retention workflow — weekly prune of untagged + keep 25 tagged (#169)"
```

Expected: exits 0. Confirm the commit is on branch `ghcr-image-retention` (not `main`).

---

## Task 2: Append `## GHCR image retention` section to `docs/RELEASING.md`

**Model/effort:** sonnet / high.
**Spec:** §3.1 (auth + PAT fallback), §3.5 (documentation content).

**Files:**
- Modify: `docs/RELEASING.md`

**Goal:** Append a new `## GHCR image retention` section at the end of `docs/RELEASING.md` (after the existing `## Changelog` section) covering: what the workflow does, the package-must-exist prerequisite, the one-time Actions-access Admin grant, and the PAT fallback. Match the file's existing tone: prose with nested lists, `##` headings, inline code for UI paths and command names.

- [ ] **Step 2-1: Append the `## GHCR image retention` section**

Read the current last line of `docs/RELEASING.md` to confirm it ends after the `## Changelog` section, then append the following content. The file currently ends at the `## Changelog` section (ending around line 55 with the GitHub Releases link sentence). Append after the final line:

```markdown

## GHCR image retention

The `.github/workflows/ghcr-retention.yml` workflow runs automatically every
Monday at 07:00 UTC and keeps the GHCR package page tidy by:

- Deleting dangling **untagged** manifests (attestation and SBOM children
  produced by `docker/build-push-action`'s provenance feature). The action is
  referrer-aware, so children of a live tagged image are never removed.
- Retaining only the **25 newest tagged releases**. Tagged releases older than
  the 25th are pruned.
- Never touching the **`:latest`** floating tag (explicitly excluded as
  defence-in-depth).

A `workflow_dispatch` trigger with a `dry_run` boolean input (default `false`)
lets a maintainer preview planned deletions on demand before the weekly
schedule fires.

### Prerequisite: package must exist

The one-time Admin grant below cannot be configured until the GHCR package
`ghcr.io/nguyenjus/custom-sam-peft` has been published at least once (i.e.
at least one `v*` tag has been pushed and the Docker workflow has completed
successfully). Do not attempt the grant on a fresh clone with no published
image.

### One-time Actions-access Admin grant

By default, a `GITHUB_TOKEN` for a user-owned GHCR package cannot delete
package versions. Granting the repository **Admin** access to the package
confers delete capability on the workflow's built-in token, requiring no
stored secret.

Steps:

1. Go to **GitHub.com → your profile → Packages → `custom-sam-peft`**.
2. Click **Package settings**.
3. Under **Manage Actions access**, click **Add repository**.
4. Search for and select **`NguyenJus/custom-sam-peft`**.
5. Set the role to **Admin**.
6. Click **Save**.

The workflow's `GITHUB_TOKEN` can now delete package versions. No secret
needs to be created.

### PAT fallback

GitHub's documentation on user-owned package deletion via Actions access is
ambiguous, and some users report persistent 403 errors on the delete step
even after granting Admin access. If the scheduled workflow fails with a 403:

1. Mint a **classic PAT** with `read:packages` and `delete:packages` scopes
   (GitHub Settings → Developer settings → Personal access tokens → Tokens
   (classic) → Generate new token).
2. Add it as a repository secret named `GHCR_RETENTION_TOKEN`
   (repository Settings → Secrets and variables → Actions → New repository
   secret).
3. Edit `.github/workflows/ghcr-retention.yml`: in the
   `dataaxiom/ghcr-cleanup-action` step, change

   ```yaml
   token: ${{ secrets.GITHUB_TOKEN }}
   ```

   to

   ```yaml
   token: ${{ secrets.GHCR_RETENTION_TOKEN }}
   ```

4. Commit and push. Re-run the workflow (or wait for the next Monday run).
```

- [ ] **Step 2-2: Verify the section is present and correctly placed**

```bash
grep -n "## GHCR image retention\|## Changelog" /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/docs/RELEASING.md
```

Expected: `## Changelog` appears first (at its original line number), followed by `## GHCR image retention` at a later line. Both lines should appear exactly once.

- [ ] **Step 2-3: Verify `docs/RELEASING.md` with `markdownlint-cli2`**

Run the same markdown linter CI uses (see `ci.yml` line 99 — `npx --yes markdownlint-cli2`):

```bash
npx --yes markdownlint-cli2 --config /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.config/markdownlint-cli2.jsonc /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/docs/RELEASING.md
```

Expected: exits 0, no findings. If markdownlint flags any finding in the new section, fix the content (do not suppress with inline disable comments unless the rule is a false positive on a code block or table). Common pitfalls:

- `MD022` / `MD023` — blank lines around headings. Ensure one blank line before and after each `##`/`###` heading.
- `MD031` — fenced code blocks must be surrounded by blank lines. Ensure a blank line before and after each ` ``` ` block.
- `MD047` — file must end with a single newline. Ensure the appended content ends with exactly one trailing newline.

- [ ] **Step 2-4: Commit `docs/RELEASING.md`**

```bash
git -C /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention add docs/RELEASING.md
git -C /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention commit -m "docs: document GHCR image retention workflow and one-time setup (#169)"
```

Expected: exits 0. The branch should now have exactly two new commits on top of `main`.

---

## Task 3: Open the PR

**Model/effort:** sonnet / medium.
**Depends on:** Tasks 1 and 2 both committed.

**Goal:** Run the full lint gate, push the branch, and open a ready (non-draft) PR linking spec, plan, and issue #169.

- [ ] **Step 3-1: Run the full lint gate before pushing**

These are the exact checks CI's `lint-hygiene` job runs on every PR (`.github/workflows/ci.yml` lines 88–99). All must pass before the PR is opened:

```bash
# actionlint — covers the new workflow file plus all existing ones
bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7 2>/dev/null || true
./actionlint -color /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.github/workflows/ghcr-retention.yml

# yamllint — covers the new workflow file
uv run --with yamllint yamllint -c /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.config/yamllint.yml /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.github/workflows/ghcr-retention.yml

# markdownlint — covers the modified docs file
npx --yes markdownlint-cli2 --config /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/.config/markdownlint-cli2.jsonc /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention/docs/RELEASING.md
```

Expected: all three commands exit 0. Fix any issues before proceeding — do not open the PR on a red lint gate.

- [ ] **Step 3-2: Confirm working tree is clean**

```bash
git -C /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention status
```

Expected: `nothing to commit, working tree clean`. Both new commits should be visible in `git log --oneline -5`.

- [ ] **Step 3-3: Push the branch**

```bash
git -C /home/justin/projects/custom-sam-peft/.claude/worktrees/ghcr-image-retention push -u origin ghcr-image-retention
```

Expected: push succeeds and sets the upstream tracking ref.

- [ ] **Step 3-4: Fetch the `ci` label (confirm it exists) and open the PR**

```bash
gh label list --repo NguyenJus/custom-sam-peft
```

Identify the label to use. If a `ci` label exists use it; if not, use `enhancement` (or create `ci` inline in the next command with `gh label create ci --description "CI / workflow changes" --color 0075ca`).

```bash
gh pr create \
  --repo NguyenJus/custom-sam-peft \
  --assignee @me \
  --label ci \
  --title "ci: add GHCR retention workflow — weekly prune of untagged + keep 25 tagged (#169)" \
  --body "$(cat <<'EOF'
## Summary

- Adds `.github/workflows/ghcr-retention.yml`: a weekly scheduled workflow
  (Monday 07:00 UTC, staggered 1 h after CodeQL) that uses the referrer-aware
  `dataaxiom/ghcr-cleanup-action@f092b48  # v1.2.1` (SHA-pinned) to delete
  dangling untagged manifests and retain the 25 newest tagged releases.
  A `workflow_dispatch` with `dry_run: boolean` input enables on-demand preview.
- Appends `## GHCR image retention` to `docs/RELEASING.md` documenting:
  what the workflow does, the package-must-exist prerequisite, the one-time
  Actions-access Admin grant steps, and the PAT fallback procedure.

**Spec:** `docs/superpowers/specs/2026-05-30-ghcr-image-retention-design.md`
**Plan:** `docs/superpowers/plans/2026-05-30-ghcr-image-retention-plan.md`
**Closes:** #169

## Post-merge rollout (operator, not automated)

These steps are performed by the maintainer after merge. They are documented
in `docs/RELEASING.md § GHCR image retention`.

1. **One-time Admin grant:** GitHub.com → Packages → `custom-sam-peft` →
   Package settings → Manage Actions access → add `NguyenJus/custom-sam-peft`
   → role **Admin** → Save. This enables `GITHUB_TOKEN` to delete package
   versions. Requires the package to exist (at least one `v*` tag pushed).
2. **Dry-run verification:** Actions → GHCR Retention → Run workflow →
   enable `dry_run: true` → Run. Review the logged list of versions the
   action would delete. Confirm `:latest` and current release tags are all
   in the "would keep" list.
3. **PAT fallback (if dry run 403s):** Mint classic PAT with
   `read:packages` + `delete:packages`; store as repo secret
   `GHCR_RETENTION_TOKEN`; swap `token:` in the workflow step; repeat dry run.
4. **Acceptance (after first live run):** Untagged manifests removed from
   GHCR package page; ≤ 25 tagged releases visible; `docker pull
   ghcr.io/nguyenjus/custom-sam-peft:latest` succeeds.

## Test plan

- [ ] `lint-hygiene` CI passes: `actionlint` and `yamllint` on
  `ghcr-retention.yml`; `markdownlint` on `docs/RELEASING.md`.
- [ ] Workflow SHA pin is the full 40-char hex for `dataaxiom/ghcr-cleanup-action`
  v1.2.1 (`f092b48ba3b604b2a83690dc4b2bbb3392e1045f`) with inline `# v1.2.1` comment.
- [ ] Cron expression is `"0 7 * * 1"` (Monday 07:00 UTC, 1 h after CodeQL's
  `"0 6 * * 1"`).
- [ ] `workflow_dispatch` input `dry_run` is `boolean`, default `false`.
- [ ] `permissions: packages: write` is the only permission block.
- [ ] `concurrency` group matches repo convention
  (`${{ github.workflow }}-${{ github.ref }}`).
- [ ] `docs/RELEASING.md` new section covers: purpose, prerequisite, Admin
  grant steps, PAT fallback.
EOF
  )"
```

Expected: command exits 0 and prints the PR URL. Record the URL.

---

## Post-merge rollout (operator steps — not implementation tasks)

These steps are performed by the maintainer **after the PR is merged**. They are documented in `docs/RELEASING.md` and listed here for completeness. They are NOT implementation tasks — they cannot be automated from within the workflow and must not be added as CI steps.

1. **Prerequisite check:** Confirm `ghcr.io/nguyenjus/custom-sam-peft` exists (at least one `v*` tag has been pushed and the Docker workflow completed). Do not proceed to step 2 on a fresh repo with no published image.
2. **One-time Actions-access Admin grant:**
   - GitHub.com → profile → Packages → `custom-sam-peft` → Package settings
   - Under **Manage Actions access** → **Add repository** → `NguyenJus/custom-sam-peft` → role **Admin** → Save
3. **Manual dry-run verification:**
   - Actions → GHCR Retention → Run workflow → set `dry_run: true` → Run workflow
   - Review log output: confirm `:latest` and all current release tags appear in the "would keep" list; confirm the untagged attestation manifests appear in the "would delete" list.
4. **If dry run exits with 403 on delete calls:** follow the PAT fallback procedure in `docs/RELEASING.md`.
5. **First live run acceptance checks** (after the first non-dry-run Monday execution):
   - Untagged manifests no longer appear on the GHCR package page.
   - At most 25 tagged releases are visible; releases older than the 25th are gone.
   - `docker pull ghcr.io/nguyenjus/custom-sam-peft:latest` succeeds.
6. **Close issue #169:**
   ```bash
   gh issue close 169 --repo NguyenJus/custom-sam-peft \
     --comment "GHCR retention workflow merged and live. Admin grant applied. Dry run verified. Closes #169."
   ```

---

## Definition of done

All items below must be checked before the PR is marked ready for review:

- [ ] `.github/workflows/ghcr-retention.yml` exists and matches spec §3.4 verbatim.
- [ ] SHA pin is `f092b48ba3b604b2a83690dc4b2bbb3392e1045f` (40-char hex) with `# v1.2.1` comment on the same line.
- [ ] Cron is `"0 7 * * 1"`; `workflow_dispatch` has `dry_run: boolean, default: false`.
- [ ] `permissions: packages: write` is the sole permissions block.
- [ ] `concurrency` block matches `${{ github.workflow }}-${{ github.ref }}` with `cancel-in-progress: true`.
- [ ] All eight action inputs (`token`, `owner`, `package`, `delete-untagged`, `keep-n-tagged`, `exclude-tags`, `validate`, `dry-run`) are present with the specified values.
- [ ] `docs/RELEASING.md` has `## GHCR image retention` section appended after `## Changelog` covering: workflow purpose, prerequisite, Admin grant steps, PAT fallback.
- [ ] `actionlint` passes on `ghcr-retention.yml`.
- [ ] `yamllint -c .config/yamllint.yml` passes on `ghcr-retention.yml`.
- [ ] `markdownlint-cli2` passes on `docs/RELEASING.md`.
- [ ] PR body links spec path, plan path, and issue #169; lists post-merge rollout steps.

---

## Self-review

**1. Spec coverage:**

| Spec section | Covered by |
| --- | --- |
| §3.1 Auth — one-time Admin grant + PAT fallback | Task 2, Step 2-1 (RELEASING.md section) |
| §3.2 Cleanup mechanism — SHA pin, referrer-aware | Task 1, Step 1-1 (`uses:` line + key design notes) |
| §3.3 Retention policy — untagged, 25 tagged, exclude latest, validate | Task 1, Step 1-1 (action inputs) |
| §3.4 Workflow shape — triggers, permissions, concurrency, jobs | Task 1, Step 1-1 (full YAML) |
| §3.5 Documentation — all four required sub-topics | Task 2, Step 2-1 (full section prose) |
| §4 Rollout — post-merge manual steps | Post-merge rollout section (not implementation tasks) |
| §6 File layout — two files only | File Map confirms exactly two files |

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or vague "add appropriate" language anywhere in the plan. All YAML, commands, and expected outputs are fully specified.

**3. Type/name consistency:** The action input names (`delete-untagged`, `keep-n-tagged`, `exclude-tags`, `dry-run`) are used identically in Step 1-1, the Definition of done checklist, and the key design notes. The secret name `GHCR_RETENTION_TOKEN` is used identically in Step 2-1 (RELEASING.md prose) and the PR body.

**4. Phasing:** Single phase — correct for a two-file, no-source-code change. Stated explicitly at the top of the plan.

**5. Lint gate:** All three lint tools (actionlint, yamllint, markdownlint-cli2) are run in both the per-task verification steps AND the pre-push gate in Task 3, Step 3-1. The exact commands match what CI's `lint-hygiene` job runs (`ci.yml` lines 89–99).

**6. Post-merge rollout:** Runtime verification (Admin grant, dry-run, acceptance checks) is placed in the post-merge rollout section — not as implementation tasks or CI steps — consistent with spec §4 and the plan brief's guidance.
