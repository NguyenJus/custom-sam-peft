# spec/ci-hardening — CI Security & Hygiene Hardening

**Status:** Draft (2026-05-18)
**Tracking issue:** #27
**Scope:** Add a security/hygiene tier to CI without taking on any external SaaS dashboard or paid tier. Extend `ci.yml`, add a sibling `security.yml`, add tool configs at the repo root, edit `pyproject.toml` for ruff `S` + HTML coverage. No `src/esam3/` changes.

---

## 1. Current State

| Surface | State today | This spec |
| --- | --- | --- |
| `.github/workflows/ci.yml` | One `test` job on `ubuntu-latest`: `setup-uv@v3` → `uv python install 3.13` → `uv sync --all-extras` → ruff check → ruff format --check → mypy strict on `src/esam3` → pytest. Action refs are floating `@v3`/`@v4` tags. No concurrency control. | **Extended.** Adds `lock-check` and `lint-hygiene` jobs, an HTML-coverage upload step on `test`, top-level concurrency, SHA-pinned third-party Actions. |
| `.github/workflows/security.yml` | Does not exist. | **New.** Two jobs (`pip-audit`, `gitleaks`), same concurrency + SHA-pin posture. |
| `.github/dependabot.yml` | Does not exist. | **New.** `pip` + `github-actions` ecosystems, weekly, grouped. |
| `.gitleaks.toml`, `.markdownlint.json`, `.yamllint.yml` | Do not exist. | **New.** Repo-root configs for the corresponding linters. |
| Ruff `S` (bandit-equivalent) family | Not selected. Current select: `E,F,I,B,UP,SIM,RUF`. | **Selected.** Per-file ignores added for tests (`S101`) and, if needed, scripts (`S603,S607`). |
| `pyproject.toml` pytest `addopts` | `-ra --strict-markers --cov=esam3 --cov-report=term-missing --cov-fail-under=80`. 80% gate already enforced locally. | **Extended.** `--cov-report=html` added so `htmlcov/` exists for the upload step. Gate stays at 80%. |
| `.gitignore` | Does not list `htmlcov/`. | **Edited.** Adds `htmlcov/`. |
| `scripts/run_gpu_tests.sh` | Sole shell script; never linted. | **Linted** by `shellcheck` in `lint-hygiene`. |
| `uv.lock` drift | Not enforced in CI. Local-only behavior. | **Enforced** by a `lock-check` job running `uv lock --check`. |
| Coverage HTML report | Generated only when a developer adds `--cov-report=html` locally. | **Uploaded** as the `coverage-html` artifact (3-day retention) on every CI run. |

The repo is currently **private** (going public later). That is the reason CodeQL is deferred (see §8).

---

## 2. Goals & Non-Goals

**Goals.**

- Catch a leaked secret, vulnerable dep, malformed workflow, broken shell script, or stale lockfile **in CI**, not in a post-incident postmortem.
- Stay 100% OSS / free-tier. No tokens beyond the built-in `GITHUB_TOKEN`. No external SaaS dashboards (Codecov, Snyk, SonarCloud, etc.).
- Keep the wall-clock cost low: critical-path CI under **5 minutes** on `ubuntu-latest`.
- Make every third-party Action reference reproducible and auditable: 40-character commit SHA + version comment, with Dependabot keeping them current.
- Adopt a single, consistent "fix-then-block" rollout — never land a `continue-on-error` workaround.

**Non-goals.**

- Any external SaaS dashboard or paid tier: Codecov, Coveralls, Snyk, SonarCloud, DeepSource, Semgrep Cloud. (All explicitly named in #27 and explicitly rejected here.)
- Any check requiring a token beyond the built-in `GITHUB_TOKEN`.
- Raising the coverage gate above 80% (still 80% — surface the HTML report, do not raise the bar).
- CodeQL — see §8 (deferred to issue #31; depends on repo going public for free).
- bandit (dropped — ruff `S` covers the same rule family).
- trufflehog (dropped — gitleaks covers the same secret-scan surface).
- Any `src/esam3/` source change. This is pure CI + config diff.

---

## 3. Files Touched / Module Layout

```text
.github/
  workflows/
    ci.yml              # CHANGED — concurrency, SHA-pins, +lock-check, +lint-hygiene, +coverage upload
    security.yml        # NEW — pip-audit + gitleaks
  dependabot.yml        # NEW — pip + github-actions, weekly, grouped

.gitleaks.toml          # NEW — extends defaults; empty [allowlist]
.markdownlint.json      # NEW — MD013 off; defaults otherwise
.markdownlint-cli2.jsonc          # NEW — markdownlint-cli2 config; sets ignores:[".venv/**"]
docs/superpowers/.markdownlint.json # NEW — directory-scoped relaxation for archival planning docs
.yamllint.yml           # NEW — extends default; line-length off; truthy.check-keys false

pyproject.toml          # EDITED — ruff lint.select gains "S"; per-file ignores;
                        #          pytest addopts gains --cov-report=html
.gitignore              # EDITED — adds htmlcov/
```

No file under `src/esam3/` is modified. The rollout (§6) inserts other commits in between (fixes for newly surfaced violations), but the final delta is the list above.

---

## 4. Job Map

| Workflow | Job | Purpose | Blocking? |
| --- | --- | --- | --- |
| `ci.yml` | `test` | (existing) ruff check, ruff format --check, mypy strict, pytest `--cov-fail-under=80`, upload HTML coverage | yes |
| `ci.yml` | `lock-check` | (new) `uv lock --check` — fail if `uv.lock` drifted from `pyproject.toml` | yes |
| `ci.yml` | `lint-hygiene` | (new) actionlint + yamllint + markdownlint-cli2 + shellcheck | yes |
| `security.yml` | `pip-audit` | (new) `uv run --with pip-audit pip-audit --skip-editable` against the synced env | yes |
| `security.yml` | `gitleaks` | (new) OSS CLI binary downloaded + checksum-verified; full history on PRs, push range on push | yes |

**Parallelism.** All five jobs run in parallel — no `needs:` chains. The serial chain is solely *inside* `test` (lint → format → mypy → pytest), which is unchanged.

**No `continue-on-error` anywhere.** Every job above is a hard gate. The rollout (§6) fixes pre-existing violations before turning a check on, so the first CI run with the new jobs is green by construction.

---

## 5. Per-Check Details

### 5.1 Ruff `S` (bandit-equivalent) — `pyproject.toml`

`[tool.ruff.lint]` gains `"S"` in `select`. The existing `test` job's `uv run ruff check` step picks the new rule family up automatically; no workflow change.

Per-file ignores added under `[tool.ruff.lint.per-file-ignores]`:

```toml
"tests/**/*.py"   = ["S101", "S311"]       # allow `assert` and `random` (test fixtures use seeded RNG)
"notebooks/**"   = ["S101", "S603", "S607"] # Colab notebooks: runtime-guard `assert`, `subprocess(git ...)`
```

The `scripts/**` ignore is conditional on the rollout step (§6.1) actually surfacing a violation; if `scripts/` is clean under `S`, the second line is not added. (Today `scripts/` contains only a shell script, so the `S603/S607` ignore likely will not be needed and the entry should be omitted. Decide during §6.1.)

Inline `# noqa: S311` is allowed on individual `src/` lines that use `random.Random(seed)` or `random.random()` for deterministic sampling (not security-sensitive). Each occurrence carries a rationale comment. The three current sites are `src/esam3/data/coco.py`, `src/esam3/data/hf.py`, and `src/esam3/train/loop.py`.

### 5.2 `uv lock --check` — new `ci.yml: lock-check` job

Job runs `actions/checkout@<sha>` → `astral-sh/setup-uv@<sha>` → `uv lock --check`. No `uv sync`; the check does not need a synced env. Fails CI if `uv.lock` is out of sync with `pyproject.toml`. Pair with Dependabot's `pip` ecosystem (§5.9) — Dependabot opens PRs that update both files together, so the check stays green by construction.

### 5.3 actionlint — `ci.yml: lint-hygiene`

Download the release binary in a `run:` step (no Action wrapper), checksum-verified:

```yaml
- name: Install actionlint
  run: |
    bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7
- name: actionlint
  run: ./actionlint -color
```

Pinned version `1.7.7` (current latest as of 2026-05). The upstream `download-actionlint.bash` performs SHA-256 verification of the downloaded archive against the value baked into the script for the requested tag, so pinning the tag also pins the expected hash.

### 5.4 yamllint — `ci.yml: lint-hygiene`

```yaml
- name: yamllint
  run: uv run --with yamllint yamllint .
```

`.yamllint.yml`:

```yaml
extends: default
ignore: |
  .venv/
rules:
  line-length: disable
  document-start: disable   # project-wide style omits leading ---
  truthy:
    check-keys: false        # GitHub `on:` triggers truthy false-positives by default
```

`ignore: .venv/` excludes the local Python venv from yamllint scope. `document-start: disable` reflects a project-wide style choice — all YAML files in this repo (workflows, dependabot, configs/examples, src/esam3/cli/templates) omit the leading `---`.

### 5.5 markdownlint-cli2 — `ci.yml: lint-hygiene`

```yaml
- name: markdownlint
  run: npx --yes markdownlint-cli2 "**/*.md" "#node_modules"
```

Node is pre-installed on `ubuntu-latest`. `.markdownlint.json`:

```json
{
  "MD013": false
}
```

```jsonc
// .markdownlint-cli2.jsonc — runner config
{
  "config": { "MD013": false },
  "ignores": [".venv/**"]
}
```

```json
// docs/superpowers/.markdownlint.json — directory-scoped relaxation for the
// archival planning subtree (specs/, plans/). Disables 13 cosmetic rules so
// frozen historical docs do not block CI. The repo-root config still applies
// to README.md, ARCHITECTURE.md, and any future Markdown outside this subtree.
{
  "MD004": false,
  "MD013": false,
  "MD022": false,
  "MD024": false,
  "MD025": false,
  "MD031": false,
  "MD032": false,
  "MD033": false,
  "MD034": false,
  "MD036": false,
  "MD038": false,
  "MD040": false,
  "MD056": false,
  "MD060": false
}
```

`MD013` (line-length) is off; everything else stays at default (headings, links, lists). Add ignore globs if the rollout surfaces generated/vendored markdown that shouldn't be linted.

The directory-scoped relaxation reflects a deliberate scope decision: archival planning documents under `docs/superpowers/` are frozen historical artifacts; mass-fixing 700+ cosmetic violations across them would distort the planning record without a clear payoff. Live documentation (README.md, ARCHITECTURE.md, future docs) and any new file outside `docs/superpowers/` is still subject to the default ruleset (minus MD013).

### 5.6 shellcheck — `ci.yml: lint-hygiene`

```yaml
- name: shellcheck
  run: shellcheck scripts/*.sh
```

Pre-installed on `ubuntu-latest`. No config file; default rules. Today the glob matches `scripts/run_gpu_tests.sh`.

### 5.7 pip-audit — `security.yml: pip-audit`

```yaml
- name: pip-audit
  run: uv run --with pip-audit pip-audit --skip-editable
```

Run after `uv sync --all-extras` so the audited environment matches the env tests run against. No `--ignore-vuln` initially. Policy for the first transitive vuln with no upstream fix: open a follow-up PR adding a single `--ignore-vuln <ID>` flag with a comment naming the advisory and linking the upstream issue. The flag is not added preemptively. `--skip-editable` excludes the locally-installed `efficient-sam3-finetuning` package (an editable distribution not on PyPI). `--strict` is intentionally **not** used here: pip-audit treats every skipped distribution (including `--skip-editable` skips) as a collection failure under `--strict`, making the two flags mutually incompatible in this repo. pip-audit's default mode still exits non-zero on real vulnerability findings, which is the gate we want.

### 5.8 gitleaks — `security.yml: gitleaks`

**No `gitleaks-action` wrapper.** That Action's `LICENSE.txt` requires a paid license for org accounts. We use the upstream OSS CLI binary, which is MIT-licensed and free for any use. The job downloads a pinned release tarball, verifies the SHA-256 against a value committed in the workflow file, extracts, and runs:

```yaml
- name: Download gitleaks
  env:
    GITLEAKS_VERSION: "8.21.2"
    # SHA-256 of gitleaks_${VERSION}_linux_x64.tar.gz from the upstream release page.
    # Implementer pastes the official value when writing this workflow; Dependabot's
    # github-actions ecosystem does not bump non-Action versions, so any future
    # GITLEAKS_VERSION bump is a manual edit that also rotates this hash.
    GITLEAKS_SHA256: "<sha256-from-release-page>"
  run: |
    set -euo pipefail
    curl -sSL -o gitleaks.tar.gz \
      "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
    echo "${GITLEAKS_SHA256}  gitleaks.tar.gz" | sha256sum -c -
    tar -xzf gitleaks.tar.gz gitleaks
- name: gitleaks
  run: ./gitleaks detect --no-banner --redact --verbose
```

Checkout step requirements differ by event:

- **`pull_request`:** `actions/checkout` with `fetch-depth: 0` so `gitleaks detect` sees full history of the PR branch.
- **`push`:** `actions/checkout` with default `fetch-depth: 1`; the implementation passes `--log-opts="${{ github.event.before }}..${{ github.sha }}"` to scan only the push range. (Avoids re-scanning history on every push to `main`.)

`.gitleaks.toml`:

```toml
# Extend the bundled default ruleset.
[extend]
useDefault = true

[allowlist]
# Empty by default. Add `paths`, `regexes`, or `commits` only when a verified
# false positive needs to be silenced; comment each entry with the rationale.
```

### 5.9 Coverage HTML artifact — extends existing `test` job

`pyproject.toml` `[tool.pytest.ini_options].addopts` gains `--cov-report=html`, producing `htmlcov/` on every test run. New step at the end of `test`:

```yaml
- name: Upload coverage HTML
  uses: actions/upload-artifact@<sha>   # v4.x.x
  if: always()
  with:
    name: coverage-html
    path: htmlcov/
    retention-days: 3
```

`if: always()` so a coverage report is uploaded even when pytest fails the 80% gate — that report is the most useful one in that case. The 80% gate itself is unchanged. `.gitignore` adds `htmlcov/`.

### 5.10 Dependabot — `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: weekly
    groups:
      dev-deps:
        dependency-type: development
      patch-updates:
        update-types: ["patch"]
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
    groups:
      patch-updates:
        update-types: ["patch"]
```

`pip` ecosystem reads `pyproject.toml` + `uv.lock` (Dependabot has native uv support as of 2025-Q1). The `dev-deps` group batches dev-only updates into a single PR per week; `patch-updates` collapses every patch bump in both ecosystems into one PR each. The `github-actions` ecosystem is the mechanism that keeps the SHA pins (§5.11) current — Dependabot rewrites the SHA + the trailing `# v4.x.x` comment in one commit.

### 5.11 SHA pinning — both workflows

Every third-party Action reference is replaced with a 40-character commit SHA plus a trailing version comment:

```yaml
- uses: actions/checkout@<40-char-sha>                # v4.2.2
- uses: astral-sh/setup-uv@<40-char-sha>              # v3.2.0
- uses: actions/upload-artifact@<40-char-sha>         # v4.4.3
```

The implementer fills in the SHAs by resolving each tag with `gh api repos/{owner}/{repo}/git/ref/tags/v...` at the moment of writing. Dependabot's `github-actions` ecosystem keeps them current after the PR lands. **First-party** Actions (none currently) would be exempt; today every Action used is third-party, so this applies to every `uses:` line in both workflows.

### 5.12 Concurrency

Top-level on **both** `ci.yml` and `security.yml`:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

Cancels superseded runs on the same ref. A rapid push sequence to a PR branch will leave only the last run alive in each workflow.

---

## 6. Rollout Sequence (fix-then-block)

The user explicitly chose fix-then-block. Each step lands as one or more commits on this branch; CI stays green at every commit.

### 6.1 Step 1 — Foundation, no new CI checks yet

One commit. Add the tool config files and the `pyproject.toml` / `.gitignore` edits:

- Add `.gitleaks.toml`, `.markdownlint.json`, `.yamllint.yml`, `.github/dependabot.yml`.
- `pyproject.toml`: add `"S"` to `[tool.ruff.lint].select`; add the `tests/**/*.py = ["S101"]` per-file ignore; add `--cov-report=html` to pytest `addopts`.
- `.gitignore`: add `htmlcov/`.

Existing `ci.yml` is **unchanged** in this step. The existing `uv run ruff check` step will now exercise the `S` family locally and in CI on the next commit — which is why the next step is "fix the violations."

### 6.2 Step 2 — Fix pre-existing violations, one tool per commit

Each sub-step is one commit. Sub-steps are ordered to minimize re-work (lockfile last so it captures any dep pin that earlier steps add).

| Sub-step | Local command | Fix mechanism |
| --- | --- | --- |
| 6.2.a | `uv run ruff check` | Fix new `S` findings in `src/`, `scripts/`. Add `scripts/**/*.py = ["S603","S607"]` per-file ignore *only if violations remain after a real fix attempt*. |
| 6.2.b | `npx --yes markdownlint-cli2 "**/*.md" "#node_modules"` | Edit Markdown files to satisfy default rules (excluding the disabled `MD013`). |
| 6.2.c | `uv run --with yamllint yamllint .` | Edit YAML files. |
| 6.2.d | `shellcheck scripts/*.sh` | Edit `scripts/run_gpu_tests.sh`. |
| 6.2.e | `uv run --with pip-audit pip-audit --skip-editable` | Bump pinned deps in `pyproject.toml` to a fixed version. If a transitive vulnerability has no upstream fix, the rollout halts here: a follow-up PR (post-merge) adds the targeted `--ignore-vuln <ID>` flag to the `security.yml` invocation with a comment naming the advisory. The first CI green of `security.yml` (Step 4) is gated on this sub-step producing a clean `pip-audit --skip-editable` run with **no** ignore flags. |
| 6.2.f | `gitleaks detect --no-banner --redact --verbose` | If any finding is a real secret: rotate, then `git filter-repo` (separate operation, coordinated with the user). If any finding is a verified false positive: add a narrowly-scoped entry to `.gitleaks.toml` `[allowlist]` with a rationale comment. |
| 6.2.g | `uv lock --check` (then `uv lock` if it failed) | Commit updated `uv.lock`. |

### 6.3 Step 3 — Extend `ci.yml`

One commit. Edit `ci.yml` only:

- Add top-level `concurrency` block (§5.12).
- Convert every existing `uses:` reference to a SHA pin + version comment (§5.11).
- Add the `lock-check` job (§5.2).
- Add the `lint-hygiene` job (§5.3, §5.4, §5.5, §5.6).
- Add the "Upload coverage HTML" step (§5.9) as the final step of the existing `test` job.

### 6.4 Step 4 — Add `security.yml`

One commit. New file only. Top-level `concurrency` block, SHA-pinned `actions/checkout`, two jobs (`pip-audit` per §5.7, `gitleaks` per §5.8).

### 6.5 Step 5 — Verify

Push the branch (or open the PR for draft CI). Confirm:

- All five jobs in both workflows pass.
- Zero `continue-on-error` flags anywhere in either workflow (`grep -rn 'continue-on-error' .github/workflows/` returns nothing).
- Every `uses:` in both workflows is a 40-char SHA with a `# v...` comment (`grep -E 'uses: [^@]+@v[0-9]' .github/workflows/` returns nothing).
- `coverage-html` artifact is downloadable from the CI run and renders in a browser.
- `ci.yml` critical-path wall time under 5 minutes (the longest parallel job determines this; today `test` runs in ~3-4 minutes, and the new jobs are all faster than `test`).
- Open the repo's **Insights → Dependency graph → Dependabot** tab; the new `dependabot.yml` is parsed without errors and lists `pip` + `github-actions` ecosystems.

---

## 7. Acceptance Criteria

- [ ] All jobs in both workflows pass on the PR.
- [ ] Zero `continue-on-error` flags anywhere in either workflow.
- [ ] Every third-party Action reference is a 40-character SHA with a trailing `# v...` version comment.
- [ ] `.github/dependabot.yml` validates (renders without errors on the repo's Dependabot Insights tab).
- [ ] The `coverage-html` artifact uploads, downloads, and renders.
- [ ] `ci.yml` critical-path wall time stays under 5 minutes on `ubuntu-latest`.
- [ ] No `src/esam3/` file modified by this PR.

---

## 8. Deferred (Out of Scope, Tracked Elsewhere)

- **CodeQL** — GitHub-native CodeQL is free only on public repos. This repo is currently private. Tracked as issue **#31**; revisit when the repo goes public.
- **bandit** — dropped. Ruff `S` selects the same rule set (bandit's checks are upstreamed in ruff as the `S` family). Re-adding bandit on top would be duplicative and double-report.
- **trufflehog** — dropped. Gitleaks covers the same secret-scan surface with comparable accuracy on this repo's profile. Running both would inflate CI time and require de-duping findings.
- **Codecov / Coveralls / Snyk / SonarCloud / DeepSource / Semgrep Cloud** — out of scope per #27 and per the project's no-SaaS-dashboard stance. The existing 80% coverage gate + uploaded HTML report give a comparable signal locally and in CI without an external dashboard or token.
- **Raising the coverage gate above 80%** — out of scope. The gate stays at 80%; this PR only surfaces the HTML report.

---

## 9. License Audit

Project license: **Apache-2.0** (`pyproject.toml`). All tools added by this PR are OSS and free for personal and commercial use.

| Tool | License |
| --- | --- |
| ruff, mypy, pytest, pytest-cov | MIT |
| uv | Apache-2.0 OR MIT |
| pip-audit | Apache-2.0 |
| gitleaks (OSS CLI binary) | MIT |
| actionlint | MIT |
| markdownlint-cli2 | MIT |
| `actions/checkout`, `actions/upload-artifact`, `astral-sh/setup-uv` | MIT |
| Dependabot | GitHub built-in (no extra license) |
| yamllint | GPL-3.0 |
| shellcheck | GPL-3.0-or-later |

**On the two GPL tools.** `yamllint` and `shellcheck` are GPL but are used as **external CLI tools that lint source files**. They are not linked into, vendored into, or shipped with the `esam3` distribution. Per the FSF GPL FAQ, this is the same posture as using `gcc` to compile your code: the GPL covers the tool, not the output. The project's Apache-2.0 license is unaffected.

**Explicitly avoided.** `gitleaks-action` — its `LICENSE.txt` requires a paid license for org accounts. Replaced with the upstream MIT CLI binary invoked directly in a `run:` step (§5.8), which is free for any use.
