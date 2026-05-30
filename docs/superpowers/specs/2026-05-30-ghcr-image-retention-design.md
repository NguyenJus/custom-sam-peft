# spec/ghcr-image-retention — GHCR image retention: prune old/untagged versions (issue #169)

**Status:** Draft (2026-05-30)
**Tracking:** [#169](https://github.com/NguyenJus/custom-sam-peft/issues/169) — *Add GHCR image retention: prune old/untagged versions of custom-sam-peft*
**Scope:** Add a scheduled GitHub Actions workflow that automatically deletes untagged GHCR manifests and retains only the 25 newest tagged releases of `ghcr.io/nguyenjus/custom-sam-peft`. Adds one new workflow file and one new documentation section. No changes to `docker.yml`, no source-code changes.

---

## 1. Motivation and context

The `Docker` workflow (`.github/workflows/docker.yml`) publishes to `ghcr.io/nguyenjus/custom-sam-peft` on every `v*` tag push. Because `docker/build-push-action` runs with provenance/SBOM attestation enabled by default, each release push produces an **untagged child manifest** (the attestation) alongside the tagged image. These children accumulate in GHCR as dangling untagged versions after every release. Tagged releases from old versions also accumulate indefinitely.

The `docker image prune` in the `Free disk space` step of `docker.yml` operates only on the **runner's local Docker daemon** — it does not touch the GHCR registry. A registry-side retention workflow is needed.

The repo is **public**, so GHCR storage is free. The motivation for retention is **tidiness** — a clean package page with a bounded history — not storage quota. This justifies a generous retention policy (keep 25 tagged releases) that will not destroy useful history.

---

## 2. Non-goals

| Item | Reason |
| --- | --- |
| Modifying `.github/workflows/docker.yml` | Attestations/provenance are intentionally preserved; the referrer-aware cleanup tool handles their untagged children without any change to `docker.yml`. |
| Disabling provenance or SBOM generation | Provenance is useful; disabling it just to avoid child manifests would sacrifice supply-chain metadata. |
| Multi-arch manifest cleanup | Builds are single-platform (`linux/amd64`); no multi-arch index is present. |
| Cleaning any registry other than GHCR | Out of scope. |

---

## 3. Design

### 3.1 Authentication: one-time Actions-access Admin grant + `GITHUB_TOKEN`

For **user-owned** GHCR packages, the workflow's built-in `GITHUB_TOKEN` cannot delete package versions by default — even with `packages: write`. That capability is org-only in the standard model.

The chosen solution is a **one-time manual grant** performed by the package owner after the PR merges:

> GitHub.com → Packages → `custom-sam-peft` → Package settings → **Manage Actions access** → add `NguyenJus/custom-sam-peft` with the **Admin** role.

Granting Admin access to the repository confers delete capability on the workflow's `GITHUB_TOKEN` without requiring any stored secret. This is zero-maintenance and consistent with the repo's security posture (SHA-pinned actions, env-only secrets).

**Honesty caveat:** GitHub's documentation on user-owned package deletion via Actions access is ambiguous, and some users report persistent 403 errors on delete even after granting Admin access. The fallback procedure if scheduled runs 403 is:

1. Mint a **classic PAT** with `read:packages` + `delete:packages` scopes.
2. Store it as a repo secret named `GHCR_RETENTION_TOKEN`.
3. Change the cleanup step's `token:` input from `${{ secrets.GITHUB_TOKEN }}` to `${{ secrets.GHCR_RETENTION_TOKEN }}`.

Both the one-time grant procedure and the PAT fallback must be documented in `docs/RELEASING.md` (see §3.5), since neither can be automated from within the workflow.

### 3.2 Cleanup mechanism: `dataaxiom/ghcr-cleanup-action`, SHA-pinned

Use `dataaxiom/ghcr-cleanup-action` at **v1.2.1**, SHA-pinned:

```
dataaxiom/ghcr-cleanup-action@f092b48ba3b604b2a83690dc4b2bbb3392e1045f  # v1.2.1
```

This action is **referrer/manifest-aware**: it understands that attestation and SBOM child manifests are linked to a parent tagged image via the OCI referrers API. It will not delete child manifests that belong to a live tagged image, and will not delete an untagged manifest that is a child of an image the policy says to keep. A referrer-unaware tool (e.g. `actions/delete-package-versions`) would delete those children and corrupt the current `:latest` pull. This is the primary reason we did not use the official GitHub action, and why we did not disable attestations in `docker.yml`.

**Confirmed input names** (v1.2.1 README):

| Input | Value | Purpose |
| --- | --- | --- |
| `token` | `${{ secrets.GITHUB_TOKEN }}` | Auth (or `GHCR_RETENTION_TOKEN` if fallback) |
| `owner` | `nguyenjus` | Package owner (lowercase) |
| `package` | `custom-sam-peft` | Package name |
| `delete-untagged` | `true` | Remove dangling untagged manifests, referrer-safe |
| `keep-n-tagged` | `25` | Retain the 25 newest tagged releases |
| `exclude-tags` | `latest` | Never delete the floating `latest` tag |
| `validate` | `true` | Post-prune integrity check |
| `dry-run` | `${{ inputs.dry_run }}` | Preview mode, driven by `workflow_dispatch` input |

### 3.3 Retention policy

- **Delete untagged manifests** — catches dangling attestation children. Because the action is referrer-aware, children of a kept image are spared; only truly orphaned untagged manifests are removed.
- **Keep 25 newest tagged releases** — generous buffer covering the full v0.x history and many future releases. Older tagged releases beyond 25 are pruned.
- **Exclude `latest`** — the `latest` floating tag always points to the newest digest and is inherently within the newest-25, but the explicit exclusion is cheap defense-in-depth against an edge case where the tag-count logic and the floating tag interact unexpectedly.
- **Post-prune validation** — the action's `validate: true` input re-checks that all remaining tagged images are intact after cleanup.

### 3.4 Workflow shape (`.github/workflows/ghcr-retention.yml`)

**Triggers:**

- `schedule:` weekly cron `"0 7 * * 1"` — Monday 07:00 UTC. Staggered 1 hour after CodeQL's `"0 6 * * 1"` (see `codeql.yml`) so the two scheduled jobs don't collide on the same runner pool.
- `workflow_dispatch:` with a boolean input `dry_run` (default `false`) so a maintainer can preview planned deletions on demand before the first live run.

**Permissions:** `packages: write` (least privilege). The workflow only needs to manage packages. Note: the actual delete capability comes from the one-time Actions-access Admin grant, not from this line alone.

**Concurrency:** `group: ${{ github.workflow }}-${{ github.ref }}` with `cancel-in-progress: true`, matching the repo's existing pattern (see `docker.yml` and `codeql.yml`).

**Jobs:** Single job `prune`, `runs-on: ubuntu-latest`. Single meaningful step: the cleanup action.

**Pseudo-YAML shape** (implementer writes final YAML, respecting SHA-pin comment style from the repo):

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

The SHA-pin comment style `# v1.2.1` matches the convention used across all existing workflow files (e.g. `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2`).

### 3.5 Documentation change: `docs/RELEASING.md` new section

Add a new `## GHCR image retention` section to `docs/RELEASING.md` after the existing `## Changelog` section. Match the file's existing tone (prose + lists, `##` headings). The section must cover:

1. **What the scheduled workflow does** — keeps the 25 newest tagged releases, deletes dangling untagged manifests (referrer-safe), runs weekly on Monday at 07:00 UTC.
2. **Prerequisite: package must exist** — the Actions-access grant cannot be configured until the GHCR package is published (i.e., at least one `v*` tag has been pushed). Do not attempt the grant on a fresh clone with no published image.
3. **One-time Actions-access Admin grant** — step-by-step: GitHub.com → Packages → `custom-sam-peft` → Package settings → Manage Actions access → add `NguyenJus/custom-sam-peft` → role **Admin** → Save. This enables `GITHUB_TOKEN` to delete package versions.
4. **PAT fallback** — if the scheduled workflow fails with a 403 on the delete step: mint a classic PAT with `read:packages` + `delete:packages` scopes; store it as repo secret `GHCR_RETENTION_TOKEN`; edit the workflow step to use `token: ${{ secrets.GHCR_RETENTION_TOKEN }}` instead of `GITHUB_TOKEN`.

---

## 4. Rollout and verification

### Prerequisites

The GHCR package `ghcr.io/nguyenjus/custom-sam-peft` must already exist (at least one published image) before the one-time Admin grant can be configured.

### Steps after merge

1. Perform the one-time **Actions-access Admin grant** (see §3.1 and `docs/RELEASING.md`).
2. Navigate to **Actions → GHCR Retention → Run workflow**, enable `dry_run: true`, and run. Review the logged list of versions the action would delete. Confirm that `latest` and the current release tags are all in the "would keep" list.
3. If the dry run output looks correct, let the weekly schedule take over. The first live run will execute the following Monday at 07:00 UTC.
4. If the dry run 403s on any call, follow the PAT fallback procedure and repeat the dry run.

### Acceptance verification

After the first live run:

- Untagged manifests removed from the GHCR package page.
- At most 25 tagged releases visible; the oldest beyond 25 are gone.
- `docker pull ghcr.io/nguyenjus/custom-sam-peft:latest` succeeds.

Workflows cannot be unit-tested. The dry-run run plus YAML/actionlint validation is the test plan. The CI `lint-hygiene` job runs `actionlint` (v1.7.7) and `yamllint` on every PR, so the new workflow file will be validated automatically before merge.

---

## 5. Acceptance criteria mapping

| Issue #169 acceptance bullet | How this design satisfies it |
| --- | --- |
| Untagged manifests are pruned automatically | `delete-untagged: true`; action is referrer-aware so children of live images are spared |
| A bounded number of historical tagged releases is retained | `keep-n-tagged: 25`; oldest tagged releases beyond 25 are pruned |
| No impact on `:latest` or the current release tags | `exclude-tags: latest` + newest-25 policy + referrer-awareness together guarantee the current release and floating tags are never touched |

---

## 6. File layout

```
.github/workflows/ghcr-retention.yml   NEW
docs/RELEASING.md                      TOUCHED (## GHCR image retention section added)
```

No other files are modified.
