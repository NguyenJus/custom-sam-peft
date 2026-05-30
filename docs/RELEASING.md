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

The v0 milestones are marked by annotated tags on the merge commits
listed below. None are published as GitHub Releases yet — the v0.5.0
Release that originally accompanied the public flip was withdrawn
pending a release-ready v0 surface. The tags remain as historical
markers and will be revisited when v0.5.0 (or a successor) is cut as
the first published release.

| Tag | Milestone |
| --- | --- |
| `v0.1.0` | First working training-loop merge (#14) |
| `v0.2.0` | First eval-pipeline merge — `Evaluator` + `MetricsReport` (#17) |
| `v0.3.0` | LoRA + QLoRA support merged (#4, #7) |
| `v0.4.0` | W&B tracking (#18) + CI hardening (#32) |
| `v0.5.0` | Public-flip merge — community standards + CodeQL (tag only; Release withdrawn) |

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
[GitHub Releases page](https://github.com/NguyenJus/custom-sam-peft/releases)
is the canonical changelog surface. A `CHANGELOG.md` may be introduced
at or after v1.0 if it becomes useful.

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
