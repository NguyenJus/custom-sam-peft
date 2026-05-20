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
