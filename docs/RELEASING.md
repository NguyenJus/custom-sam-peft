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
[GitHub Releases page](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/releases)
is the canonical changelog surface. A `CHANGELOG.md` may be introduced
at or after v1.0 if it becomes useful.
