# CI Security & Hygiene Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement [`spec/ci-hardening`](../specs/2026-05-18-ci-hardening-design.md) — add an OSS/free-tier security & hygiene tier to CI (ruff `S`, `uv lock --check`, actionlint, yamllint, markdownlint-cli2, shellcheck, pip-audit, gitleaks) plus Dependabot, an HTML coverage artifact, and SHA-pinned third-party Actions, without taking on any external SaaS dashboard.

**Architecture:** Pure CI + repo-config diff — no `src/esam3/` source change. Rollout is **fix-then-block**: Phase 1 lands config/foundation files (CI still uses the *old* `ci.yml`, stays green); Phase 2 fixes pre-existing violations one tool per commit; Phase 3 extends `ci.yml` to actually run the new checks blocking; Phase 4 adds the new `security.yml`; Phase 5 verifies the final state on the PR. Every third-party Action is pinned to a 40-char SHA with a trailing `# v…` comment so Dependabot's `github-actions` ecosystem keeps them current.

**Tech Stack:** GitHub Actions (`ubuntu-latest`), `uv` (lockfile + Python env), `ruff` (incl. bandit-equivalent `S`), pytest + pytest-cov (HTML), `actionlint` (CLI binary), `yamllint`, `markdownlint-cli2` (via `npx`), `shellcheck` (pre-installed), `pip-audit` (run via `uv run --with`), `gitleaks` OSS CLI binary (checksum-verified, **not** the paid `gitleaks-action` wrapper), Dependabot v2.

---

## File Map

**New files:**

```
.github/
  workflows/
    security.yml         # Phase 4 — pip-audit + gitleaks jobs
  dependabot.yml         # Phase 1 — pip + github-actions, weekly, grouped

.gitleaks.toml           # Phase 1 — extends defaults; empty [allowlist]
.markdownlint.json       # Phase 1 — MD013 off; defaults otherwise
.yamllint.yml            # Phase 1 — extends default; line-length off; truthy.check-keys false
```

**Modified files:**

```
.github/workflows/ci.yml # Phase 3 — concurrency, SHA-pin existing Actions,
                         #           +lock-check, +lint-hygiene, +coverage upload step
pyproject.toml           # Phase 1 — ruff.lint.select gains "S"; tests/** per-file S101 ignore;
                         #           pytest addopts gains --cov-report=html
```

**Possibly modified files** (only if Phase 2 surfaces pre-existing violations):

```
src/esam3/**             # Phase 2.a — ruff S findings (likely scripts/run_gpu_tests.sh-related code is shell, not python)
scripts/run_gpu_tests.sh # Phase 2.d — shellcheck findings
*.md, **/*.md            # Phase 2.b — markdownlint findings
*.yml, **/*.yml          # Phase 2.c — yamllint findings
pyproject.toml + uv.lock # Phase 2.e — pip-audit dep bumps; Phase 2.g — lockfile refresh
.gitleaks.toml           # Phase 2.f — only if verified false positives need allowlisting
```

**Not touched anywhere in this plan:** any file under `src/esam3/` except where Phase 2.a ruff `S` findings demand a real code fix.

> **Note on `.gitignore`:** Spec §3 lists `.gitignore` as edited (adding `htmlcov/`). Inspection of the current `.gitignore` shows `htmlcov/` is **already on line 10**. The plan therefore does not modify `.gitignore`; Task 1.4 includes a verification step that confirms this and skips the edit. If a future change removes the entry, Task 1.4 will add it back.

---

## Pre-flight checks

- [ ] **Step 0a: Confirm worktree and branch**

```bash
pwd && git rev-parse --abbrev-ref HEAD
```
Expected: `/home/justin/projects/Efficient-SAM3-Finetuning/.worktrees/spec-ci-hardening` and `spec/ci-hardening`.

- [ ] **Step 0b: Confirm working tree clean**

```bash
git status
```
Expected: `nothing to commit, working tree clean` (the spec + this plan are already committed in earlier brainstormer-planner commits).

- [ ] **Step 0c: Confirm baseline CI workflow is currently green on this branch**

```bash
gh run list --branch spec/ci-hardening --workflow CI --limit 3
```
Expected: most recent run is `completed` / `success`. If not, halt and investigate before touching anything — Phase 1 must land on a green baseline so we can attribute any new red to our changes.

- [ ] **Step 0d: Note today's pinned tool versions (record in the PR description for auditability)**

Versions the plan pins (record as the canonical reference; bump in a follow-up PR if newer is preferred at implementation time):

| Tool | Pin |
|---|---|
| `actionlint` | `1.7.7` |
| `gitleaks` (OSS CLI) | `8.21.2` |

Action SHAs (filled in at Phase 3, see Task 3.2):

| Action | Tag | SHA |
|---|---|---|
| `actions/checkout` | `v4.2.2` | TBD at Task 3.2 |
| `astral-sh/setup-uv` | `v3.2.0` | TBD at Task 3.2 |
| `actions/upload-artifact` | `v4.4.3` | TBD at Task 3.2 |

---

# Phase 1 — Foundation (config files + pyproject + .gitignore)

**Goal:** Land all the tool config files and the `pyproject.toml` ruff/pytest edits in a small set of commits, without touching `ci.yml` or adding `security.yml`. After Phase 1, CI is unchanged from baseline and must stay green.

**Parallelism:** Tasks 1.1–1.5 are file-disjoint and can be dispatched to parallel implementer subagents.

**Verification at end of phase:** CI run on the post-Phase-1 head is green (same job set as baseline; no new jobs yet).

---

## Task 1.1: Add `.gitleaks.toml`

**Files:**
- Create: `.gitleaks.toml`

Spec §5.8. Extends the bundled default ruleset with an empty allowlist; entries are added later (Phase 2.f) only for verified false positives.

- [ ] **Step 1.1a: Create the file**

Create `.gitleaks.toml`:

```toml
# Extend the bundled default ruleset.
[extend]
useDefault = true

[allowlist]
# Empty by default. Add `paths`, `regexes`, or `commits` only when a verified
# false positive needs to be silenced; comment each entry with the rationale.
```

- [ ] **Step 1.1b: Verify TOML parses**

```bash
python -c "import tomllib; print('ok', sorted(tomllib.loads(open('.gitleaks.toml').read()).keys()))"
```
Expected: `ok ['allowlist', 'extend']`.

- [ ] **Step 1.1c: Commit**

```bash
git add .gitleaks.toml
git commit -m "ci(gitleaks): add .gitleaks.toml (default ruleset, empty allowlist)"
```

---

## Task 1.2: Add `.markdownlint.json`

**Files:**
- Create: `.markdownlint.json`

Spec §5.5. Disables MD013 (line-length); everything else stays at default.

- [ ] **Step 1.2a: Create the file**

Create `.markdownlint.json`:

```json
{
  "MD013": false
}
```

- [ ] **Step 1.2b: Verify JSON parses**

```bash
python -c "import json; print('ok', json.loads(open('.markdownlint.json').read()))"
```
Expected: `ok {'MD013': False}`.

- [ ] **Step 1.2c: Commit**

```bash
git add .markdownlint.json
git commit -m "ci(markdownlint): add .markdownlint.json (disable MD013)"
```

---

## Task 1.3: Add `.yamllint.yml`

**Files:**
- Create: `.yamllint.yml`

Spec §5.4. Extends default rules; turns off `line-length`; sets `truthy.check-keys: false` so GitHub Actions `on:` triggers don't false-positive.

- [ ] **Step 1.3a: Create the file**

Create `.yamllint.yml`:

```yaml
extends: default
rules:
  line-length: disable
  truthy:
    check-keys: false        # GitHub `on:` triggers truthy false-positives by default
```

- [ ] **Step 1.3b: Verify YAML parses**

```bash
uv run python -c "import yaml; print('ok', sorted(yaml.safe_load(open('.yamllint.yml')).keys()))"
```
Expected: `ok ['extends', 'rules']`.

- [ ] **Step 1.3c: Commit**

```bash
git add .yamllint.yml
git commit -m "ci(yamllint): add .yamllint.yml (extend default; disable line-length; truthy.check-keys=false)"
```

---

## Task 1.4: Add `.github/dependabot.yml`

**Files:**
- Create: `.github/dependabot.yml`

Spec §5.10. Two ecosystems (`pip`, `github-actions`), weekly schedule, with `dev-deps` and `patch-updates` groups.

- [ ] **Step 1.4a: Create the file**

Create `.github/dependabot.yml`:

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

- [ ] **Step 1.4b: Verify YAML parses and has both ecosystems**

```bash
uv run python -c "
import yaml
d = yaml.safe_load(open('.github/dependabot.yml'))
assert d['version'] == 2
ecos = sorted(u['package-ecosystem'] for u in d['updates'])
assert ecos == ['github-actions', 'pip'], ecos
print('ok', ecos)
"
```
Expected: `ok ['github-actions', 'pip']`.

- [ ] **Step 1.4c: Commit**

```bash
git add .github/dependabot.yml
git commit -m "ci(dependabot): add weekly grouped updates for pip + github-actions"
```

---

## Task 1.5: Edit `pyproject.toml` (ruff S + pytest HTML coverage) and verify `.gitignore`

**Files:**
- Modify: `pyproject.toml`
- Verify (likely no edit): `.gitignore`

Spec §5.1, §5.9. Adds `"S"` to ruff lint.select; adds `tests/**/*.py = ["S101"]` per-file ignore so test `assert`s remain allowed; extends pytest `addopts` with `--cov-report=html`. The `scripts/**` per-file ignore is **not** added preemptively (see Task 2.1).

`.gitignore` already lists `htmlcov/` (line 10 of current file); the spec's edit is a no-op and is verified, not re-applied.

- [ ] **Step 1.5a: Verify `.gitignore` already excludes `htmlcov/`**

```bash
grep -n '^htmlcov/' .gitignore
```
Expected: a single matching line. If missing, append `htmlcov/` under the `# Python` block and include `.gitignore` in the commit below; otherwise leave the file untouched.

- [ ] **Step 1.5b: Edit `pyproject.toml` — ruff `S` family**

In `[tool.ruff.lint]`, change:

```toml
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
```

to:

```toml
select = ["E", "F", "I", "B", "UP", "SIM", "RUF", "S"]
```

- [ ] **Step 1.5c: Edit `pyproject.toml` — per-file ignore for tests**

In `[tool.ruff.lint.per-file-ignores]`, append a new line (keep existing entries):

```toml
"tests/**/*.py" = ["S101"]
```

So the section reads (verbatim — preserve any other existing entries):

```toml
[tool.ruff.lint.per-file-ignores]
"src/esam3/cli/*_cmd.py" = ["B008"]
"tests/unit/test_data_*.py" = ["E402"]
"tests/**/*.py" = ["S101"]
```

- [ ] **Step 1.5d: Edit `pyproject.toml` — pytest HTML coverage**

In `[tool.pytest.ini_options]`, change:

```toml
addopts = "-ra --strict-markers --cov=esam3 --cov-report=term-missing --cov-fail-under=80"
```

to:

```toml
addopts = "-ra --strict-markers --cov=esam3 --cov-report=term-missing --cov-report=html --cov-fail-under=80"
```

- [ ] **Step 1.5e: Verify pyproject still parses**

```bash
python -c "import tomllib; d = tomllib.loads(open('pyproject.toml').read()); print('ok', 'S' in d['tool']['ruff']['lint']['select'], '--cov-report=html' in d['tool']['pytest']['ini_options']['addopts'])"
```
Expected: `ok True True`.

- [ ] **Step 1.5f: Verify the pytest HTML report now gets generated locally**

```bash
rm -rf htmlcov
uv run pytest -x -q tests/unit -k "not slow"
ls htmlcov/index.html
```
Expected: `htmlcov/index.html` exists. Then clean up: `rm -rf htmlcov`.

- [ ] **Step 1.5g: Commit**

```bash
git add pyproject.toml
git commit -m "ci(pyproject): enable ruff S family; allow assert in tests; add HTML coverage"
```

(If Step 1.5a required an edit to `.gitignore`, include it in this commit: `git add .gitignore pyproject.toml`.)

---

## Phase 1 exit check

- [ ] **Step 1.X: Confirm Phase 1 baseline CI is green**

```bash
git push
gh run watch --exit-status
```
Expected: existing CI passes. The job set is unchanged from before Phase 1 — only the `test` job now exercises ruff `S` against `src/`, `tests/`, `scripts/`. If `test` fails on ruff, **that is Phase 2.a's job**: defer the push of Phase 1 to after the Task 2.1 fix has been applied, then push them together. (In practice, run Task 2.1 locally first if you suspect new findings; commit it second; push both Phase 1 and Task 2.1 together.)

---

# Phase 2 — Fix pre-existing violations (one tool per commit)

**Goal:** Each new tool from Phase 1's config gains a local pass; the fixes land **before** the corresponding new CI job is turned on. After Phase 2, every tool's local invocation exits clean.

**Order matters slightly.** Lockfile (`uv lock --check`) is last because Task 2.5 (pip-audit) may bump deps. Otherwise the tasks are file-disjoint.

**Parallelism:** Tasks 2.1, 2.2, 2.3, 2.4 are file-disjoint (Python source vs. Markdown vs. YAML vs. shell) and can run as parallel implementer subagents. Tasks 2.5 and 2.6 mostly inspect repo state without expected edits; they can also parallelize with 2.1–2.4. Task 2.7 (lockfile refresh) **must** be last so it picks up any dep bump from Task 2.5.

**Halt conditions** are spelled out per-task below. A halt means: stop the rollout, surface a question or follow-up issue to the user, do not paper over with `continue-on-error` or `--ignore-*` flags (those are not part of this PR's scope).

---

## Task 2.1: Fix `ruff check` (S family) findings

**Files:**
- Possibly modify: any file under `src/`, `tests/`, `scripts/` that ruff `S` flags.
- Possibly modify: `pyproject.toml` (only if `scripts/` retains a real `S603`/`S607` violation after a code-level fix attempt; see spec §5.1 conditional clause).

Spec §6.2.a. Run ruff locally, fix each finding at the source. Only fall back to a per-file ignore if the violation reflects a deliberate-and-safe pattern.

- [ ] **Step 2.1a: Run ruff and capture the current finding set**

```bash
uv run ruff check . 2>&1 | tee /tmp/ruff-s.txt | tail -n 40
```
Expected: either `All checks passed!` (rare on a first run after enabling `S`) or a list of `S###` codes with file:line locations.

- [ ] **Step 2.1b: For each finding, apply a real code fix**

Walk the report. Typical `S` rules and fixes:

| Rule | Fix pattern |
|---|---|
| `S101` (assert) | Already ignored in `tests/**`. If a `src/` file uses `assert` for runtime validation, replace with `if not cond: raise ValueError(...)`. |
| `S102` (`exec`) / `S103` (chmod) | Rewrite to avoid `exec`/insecure chmod. |
| `S104` (hardcoded bind 0.0.0.0) | Use a configurable host. |
| `S105`/`S106`/`S107` (hardcoded password) | Pull from env or test fixture. |
| `S108` (insecure /tmp file) | Use `tempfile.NamedTemporaryFile` / `tmp_path` fixture. |
| `S603` (`subprocess` no validation) / `S607` (partial path) | Use absolute path + `shell=False` + literal `args=[...]`. |
| `S311` | If non-security-sensitive (deterministic sampling, fixture seeding), add `# noqa: S311` with rationale OR extend the per-file-ignores. If genuinely security-sensitive, switch to `secrets`. |

Where a finding is in `scripts/**/*.py` and the safe fix would distort the script (e.g. an intentional `subprocess.run(["bash", "scripts/x.sh"])` that legitimately needs `partial path`), the spec permits adding `"scripts/**/*.py" = ["S603", "S607"]` to `[tool.ruff.lint.per-file-ignores]`. Today `scripts/` contains only one shell script, so this is most likely a no-op.

- [ ] **Step 2.1c: Re-run ruff to verify clean**

```bash
uv run ruff check .
```
Expected: `All checks passed!`. If still red, repeat 2.1b. **Halt condition:** if a finding cannot be fixed at the source and is not legitimately ignorable, halt; this likely indicates a real defect worth a separate issue.

- [ ] **Step 2.1d: Re-run the existing test suite to confirm no regression from fixes**

```bash
uv run pytest -x -q tests/unit
```
Expected: full pass.

- [ ] **Step 2.1e: Commit**

```bash
git add -A
git commit -m "fix(ruff-s): clear pre-existing ruff S family findings"
```

(If no files changed — `git status` is clean — skip the commit; record in the PR description that `ruff S` was already clean.)

> **Rollout note (captured 2026-05-18):** Phase 2.1 surfaced 8 `S101` assertions in `src/esam3/{peft_adapters/qlora.py, tracking/tensorboard.py, tracking/wandb.py}` (converted to `if not … raise`), and 3 `S311` `random` usages in `src/esam3/{data/coco.py, data/hf.py, train/loop.py}` (annotated with `# noqa: S311` + rationale per spec §5.1). The `tests/**` per-file-ignore was extended to include `S311`; a new `notebooks/**` per-file-ignore was added for `S101`/`S603`/`S607`. `scripts/**` did not require any per-file-ignore — the only shell script (`scripts/run_gpu_tests.sh`) is not Python.

---

## Task 2.2: Fix `markdownlint-cli2` findings

**Files:**
- Possibly modify: any `*.md` file in the repo not under `node_modules` (none exist in this repo).

Spec §6.2.b. Markdown rules at default (minus MD013).

- [ ] **Step 2.2a: Run markdownlint and capture findings**

```bash
npx --yes markdownlint-cli2 "**/*.md" "#node_modules" 2>&1 | tee /tmp/md.txt | tail -n 40
```
Expected: either clean exit or a list of `MD###` violations.

- [ ] **Step 2.2b: For each finding, edit the offending Markdown file**

Common rules and fixes (default markdownlint rule set, MD013 disabled):

| Rule | Fix |
|---|---|
| `MD009` trailing space | Strip trailing whitespace. |
| `MD012` multiple consecutive blank lines | Collapse to one blank line. |
| `MD022` headings need blank lines around | Add blank line before/after. |
| `MD031` fenced code blocks need blank lines around | Same. |
| `MD034` bare URL | Wrap as `<https://...>` or `[text](url)`. |
| `MD040` fenced code block language | Add a language tag (`` ```python ``). |
| `MD041` first line should be a top-level heading | Add an `# H1` at file start. |

- [ ] **Step 2.2c: Re-run markdownlint to verify clean**

```bash
npx --yes markdownlint-cli2 "**/*.md" "#node_modules"
```
Expected: clean exit. **Halt condition:** if the finding is on a vendored or auto-generated Markdown file that should not be edited, add an `ignores` block to `.markdownlint.json` (e.g. `{"MD013": false, "ignores": ["path/**"]}`) and include `.markdownlint.json` in this commit.

- [ ] **Step 2.2d: Commit**

```bash
git add -A
git commit -m "fix(markdownlint): clear pre-existing markdown violations"
```

(Skip if no files changed.)

> **Rollout note (captured 2026-05-18):** Phase 2.2 surfaced 700+ cosmetic violations in the archival `docs/superpowers/{specs,plans}/` subtree. Resolution per spec §5.6: added `docs/superpowers/.markdownlint.json` (directory-scoped relaxation of 13 cosmetic rules) and `.markdownlint-cli2.jsonc` (sets `ignores: [".venv/**"]`). Live docs (README.md, ARCHITECTURE.md) and the current PR's own `docs/superpowers/specs/2026-05-18-ci-hardening-design.md` received only formatting normalizations (table separator style, code-fence language tags); content unchanged.

---

## Task 2.3: Fix `yamllint` findings

**Files:**
- Possibly modify: any `*.yml` / `*.yaml` file in the repo.

Spec §6.2.c. Default rules, with line-length off and `truthy.check-keys` false.

- [ ] **Step 2.3a: Run yamllint and capture findings**

```bash
uv run --with yamllint yamllint . 2>&1 | tee /tmp/yaml.txt | tail -n 40
```
Expected: clean or a list of `[error]` / `[warning]` lines.

- [ ] **Step 2.3b: For each finding, edit the YAML file**

Common rules and fixes:

| Rule | Fix |
|---|---|
| `indentation` | Align to consistent 2-space indents. |
| `trailing-spaces` | Strip trailing whitespace. |
| `empty-lines` | Collapse multiple blank lines. |
| `new-line-at-end-of-file` | Add a trailing newline. |
| `comments` | Add a space after `#`. |
| `document-start` | Add a leading `---` (or disable via `rules: { document-start: disable }` in `.yamllint.yml` if the repo style is to omit it). |

- [ ] **Step 2.3c: Re-run yamllint to verify clean**

```bash
uv run --with yamllint yamllint .
```
Expected: clean exit. **Halt condition:** if a finding is on the Phase-1 config files themselves (`.yamllint.yml`, `.github/dependabot.yml`), fix in place and amend the relevant Phase 1 commit (`git commit --fixup` then `git rebase --autosquash`). If the finding is on a third-party-provided YAML (none expected in this repo), add the path to a `yamllint` ignore block instead.

- [ ] **Step 2.3d: Commit**

```bash
git add -A
git commit -m "fix(yamllint): clear pre-existing yaml violations"
```

(Skip if no files changed.)

> **Rollout note (captured 2026-05-18):** Phase 2.3 surfaced `too many spaces after colon` warnings in 4 pre-existing YAML config files (`configs/examples/coco_text_lora.yaml`, `coco_text_qlora.yaml`, `src/esam3/cli/templates/coco_text_lora.yaml`, `coco_text_qlora.yaml`) — fixed in place. Added `ignore: .venv/` to `.yamllint.yml` (out-of-tree venv) and `document-start: disable` (project-wide style choice; all YAML omits the leading `---`). Both updates are reflected in spec §5.5.

---

## Task 2.4: Fix `shellcheck` findings

**Files:**
- Possibly modify: `scripts/run_gpu_tests.sh` (currently the only `.sh` file).

Spec §6.2.d. No config — default rules.

- [ ] **Step 2.4a: Run shellcheck and capture findings**

```bash
shellcheck scripts/*.sh 2>&1 | tee /tmp/sh.txt | tail -n 40
```
Expected: either clean exit or a list of `SC####` codes.

- [ ] **Step 2.4b: For each finding, edit the shell script**

Common rules and fixes:

| Rule | Fix |
|---|---|
| `SC2086` unquoted variable | Quote `"$var"`. |
| `SC2046` unquoted command substitution | Quote `"$(cmd)"`. |
| `SC1091` source not followed | Add a literal path or `# shellcheck source=path`. |
| `SC2155` declare-and-assign masks return | Split `local x; x=$(cmd)`. |
| `SC2034` unused variable | Remove or `# shellcheck disable=SC2034` with rationale comment. |

- [ ] **Step 2.4c: Re-run shellcheck to verify clean**

```bash
shellcheck scripts/*.sh
```
Expected: clean exit. **Halt condition:** if a finding requires a behavioral change in the script that affects its existing job (the manual GPU test runner), open a separate issue and disable the specific code with a `# shellcheck disable=SC####` and a rationale comment so this PR doesn't block on script refactoring.

- [ ] **Step 2.4d: Commit**

```bash
git add scripts/
git commit -m "fix(shellcheck): clear pre-existing shellcheck findings in scripts/"
```

(Skip if no files changed.)

> **Rollout note (captured 2026-05-18):** no-op — clean on first run.

---

## Task 2.5: Fix `pip-audit` findings (vulnerable deps)

**Files:**
- Possibly modify: `pyproject.toml` (bumps to vulnerable pinned deps).

Spec §6.2.e, §5.7. The audit runs against the synced env so the audit environment matches CI's test environment.

- [ ] **Step 2.5a: Sync env and run pip-audit**

```bash
uv sync --all-extras
uv run --with pip-audit pip-audit --skip-editable 2>&1 | tee /tmp/audit.txt
```
Expected: either `No known vulnerabilities found` or a table of `Name | Version | ID | Fix Versions`.

- [ ] **Step 2.5b: For each finding with a known fix, bump the dep**

In `pyproject.toml`, raise the pin on the vulnerable dependency to a version listed under `Fix Versions`. Example: if `requests 2.31.0` is flagged with fix `2.32.0`, change `"requests>=2.31"` to `"requests>=2.32"`.

- [ ] **Step 2.5c: Re-sync and re-audit**

```bash
uv sync --all-extras
uv run --with pip-audit pip-audit --skip-editable
```
Expected: `No known vulnerabilities found`. If new vulns surfaced from the bump, repeat 2.5b until clean.

- [ ] **Step 2.5d: HALT condition — unfixable transitive vulnerability**

**This is a halt, not a workaround.** Per spec §5.7: if a finding has no upstream fix version, **do not** add `--ignore-vuln <ID>` in this PR. Stop the rollout, post the advisory + a link to the upstream issue in the draft PR's comments, and instruct the user to:

1. Either accept the risk and open a *separate, post-merge* follow-up PR that adds the single `--ignore-vuln <ID>` flag to the `security.yml` `pip-audit` invocation with an inline comment naming the advisory.
2. Or revisit the dep selection.

In either path, **this** PR cannot proceed past Phase 2.5 until `pip-audit --skip-editable` exits clean **with no ignore flags**.

- [ ] **Step 2.5e: Commit (only if `pyproject.toml` was modified)**

```bash
git add pyproject.toml
git commit -m "fix(deps): bump pinned deps to address pip-audit findings"
```

(Skip if no files changed.)

> **Rollout note (captured 2026-05-18):** `pip-audit --strict` (without `--skip-editable`) always fails on this repo because the local `efficient-sam3-finetuning` package is an editable distribution not on PyPI — pip-audit attempts to look it up and errors. With `--skip-editable` added (see spec §5.7 amendment), the audit runs clean: `No known vulnerabilities found`. No dependency bumps were required. Update (CI run 26075697664, 2026-05-19): `pip-audit --strict --skip-editable` *also* failed in CI — `--strict` treats `--skip-editable` skips as collection failures (the two flags are mutually incompatible in this repo). Final invocation is `pip-audit --skip-editable` alone; spec §5.7 explains the trade-off.

---

## Task 2.6: Fix `gitleaks` findings (secrets)

**Files:**
- Possibly modify: `.gitleaks.toml` (only for verified false positives).
- Possibly modify: source files (only if a real secret needs to be rotated and removed).

Spec §6.2.f. The historic-rewrite path (`git filter-repo`) is **not** part of this plan — if a real leaked secret is found, halt and coordinate with the user.

- [ ] **Step 2.6a: Run gitleaks and capture findings**

If `gitleaks` is not installed locally, install the same OSS CLI binary the workflow will use (matches the pin from Step 0d):

```bash
GITLEAKS_VERSION=8.21.2
if ! command -v gitleaks >/dev/null; then
  curl -sSL -o /tmp/gitleaks.tar.gz "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
  tar -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks
  GITLEAKS=/tmp/gitleaks
else
  GITLEAKS=gitleaks
fi
$GITLEAKS detect --no-banner --redact --verbose 2>&1 | tee /tmp/gitleaks.txt
```

Expected: `no leaks found` or a list of redacted findings with rule IDs.

- [ ] **Step 2.6b: Triage each finding**

For each finding:

| Verdict | Action |
|---|---|
| **Real secret, still valid** | **HALT.** Rotate the secret out-of-band (revoke the key/token/password at the source), then coordinate with the user on history rewrite via `git filter-repo`. This is a multi-step operation that requires a force-push and is explicitly out of scope for this PR. |
| **Real secret, already revoked / inert** | Add a narrowly-scoped `[allowlist]` entry to `.gitleaks.toml` (e.g. specific commit SHA or specific path) with a comment naming the secret and the revocation date. |
| **Verified false positive** (e.g. a fixture string that pattern-matches AWS keys) | Add a narrowly-scoped allowlist entry (path or regex) with a rationale comment. Prefer `paths` over `regexes` whenever possible. |

- [ ] **Step 2.6c: Re-run gitleaks to verify clean**

```bash
$GITLEAKS detect --no-banner --redact --verbose
```
Expected: `no leaks found`.

- [ ] **Step 2.6d: Commit (only if `.gitleaks.toml` was modified)**

```bash
git add .gitleaks.toml
git commit -m "fix(gitleaks): allowlist verified false positives with rationale"
```

(Skip if no files changed.)

> **Rollout note (captured 2026-05-18):** no-op — clean on first run.

---

## Task 2.7: Refresh `uv.lock`

**Files:**
- Possibly modify: `uv.lock`.

Spec §6.2.g. Must run **after** Task 2.5 so any dep bumps land in the lockfile.

- [ ] **Step 2.7a: Check whether the lockfile is in sync**

```bash
uv lock --check
```
Expected: silent / exit 0 if in sync; otherwise a diff and non-zero exit.

- [ ] **Step 2.7b: If out of sync, regenerate**

```bash
uv lock
```

- [ ] **Step 2.7c: Verify sync**

```bash
uv lock --check
```
Expected: exit 0.

- [ ] **Step 2.7d: Re-run full test suite to confirm no regression from the lockfile bump**

```bash
uv sync --all-extras
uv run pytest -x -q
```
Expected: full pass (or at least the same pass set as baseline; record any change).

- [ ] **Step 2.7e: Commit (only if `uv.lock` was modified)**

```bash
git add uv.lock
git commit -m "chore(deps): refresh uv.lock"
```

(Skip if no files changed.)

> **Rollout note (captured 2026-05-18):** no-op — clean on first run.

---

## Phase 2 exit check

- [ ] **Step 2.X: All eight tool invocations exit clean locally**

```bash
uv run ruff check . \
  && npx --yes markdownlint-cli2 "**/*.md" "#node_modules" \
  && uv run --with yamllint yamllint . \
  && shellcheck scripts/*.sh \
  && uv run --with pip-audit pip-audit --skip-editable \
  && gitleaks detect --no-banner --redact --verbose \
  && uv lock --check \
  && uv run pytest -x -q \
  && echo PHASE-2-GREEN
```
Expected: `PHASE-2-GREEN`. If any step fails, return to the corresponding Task 2.x.

- [ ] **Step 2.Y: Push Phase 1 + Phase 2 together, observe baseline CI still green**

```bash
git push
gh run watch --exit-status
```
Expected: existing `test` job passes. The job set is **still** the baseline (Phase 3 hasn't landed yet); we're confirming our Phase-1 pyproject changes and Phase-2 fixes didn't break the existing pipeline.

---

# Phase 3 — Extend `ci.yml`

**Goal:** One commit. Edit `.github/workflows/ci.yml` to:

1. Add the top-level `concurrency` block (spec §5.12).
2. Convert every existing `uses:` reference to a 40-char SHA + version comment (spec §5.11).
3. Add the `lock-check` job (spec §5.2).
4. Add the `lint-hygiene` job (spec §5.3, §5.4, §5.5, §5.6).
5. Add the "Upload coverage HTML" step at the end of the existing `test` job (spec §5.9).

The new jobs are **blocking** (no `continue-on-error:`) — Phase 2 made this safe.

---

## Task 3.1: Resolve the three Action SHAs

**Files:**
- None (read-only step; values are recorded in the Task 3.2 edit).

- [ ] **Step 3.1a: Resolve each tag to a 40-char commit SHA**

```bash
gh api repos/actions/checkout/git/ref/tags/v4.2.2          | jq -r '.object.sha'
gh api repos/astral-sh/setup-uv/git/ref/tags/v3.2.0        | jq -r '.object.sha'
gh api repos/actions/upload-artifact/git/ref/tags/v4.4.3   | jq -r '.object.sha'
```

Expected: three 40-character hex strings. Record each next to the corresponding row in the Step 0d table; you'll paste them in Task 3.2.

**Halt condition:** if any of the listed tags has been moved or yanked between plan-write time and implementation time, halt and ask the user which newer pinned tag to use; bump the table in Step 0d before resuming.

---

## Task 3.2: Edit `.github/workflows/ci.yml`

**Files:**
- Modify: `.github/workflows/ci.yml`

This is one commit. The full target file content is below — replace `<SHA-checkout>`, `<SHA-setup-uv>`, `<SHA-upload-artifact>` with the values resolved in Task 3.1.

- [ ] **Step 3.2a: Replace the entire `ci.yml` with the target content**

Replace `.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<SHA-checkout>             # v4.2.2

      - name: Install uv
        uses: astral-sh/setup-uv@<SHA-setup-uv>           # v3.2.0
        with:
          enable-cache: true

      - name: Set Python version
        run: uv python install 3.13

      - name: Install deps
        run: uv sync --all-extras

      - name: Lint
        run: uv run ruff check

      - name: Format check
        run: uv run ruff format --check

      - name: Type check
        run: uv run mypy src/esam3

      - name: Test
        run: uv run pytest

      - name: Upload coverage HTML
        uses: actions/upload-artifact@<SHA-upload-artifact>   # v4.4.3
        if: always()
        with:
          name: coverage-html
          path: htmlcov/
          retention-days: 3

  lock-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<SHA-checkout>             # v4.2.2

      - name: Install uv
        uses: astral-sh/setup-uv@<SHA-setup-uv>           # v3.2.0
        with:
          enable-cache: true

      - name: uv lock --check
        run: uv lock --check

  lint-hygiene:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<SHA-checkout>             # v4.2.2

      - name: Install uv
        uses: astral-sh/setup-uv@<SHA-setup-uv>           # v3.2.0
        with:
          enable-cache: true

      - name: Install actionlint
        run: |
          bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7

      - name: actionlint
        run: ./actionlint -color

      - name: yamllint
        run: uv run --with yamllint yamllint .

      - name: markdownlint
        run: npx --yes markdownlint-cli2 "**/*.md" "#node_modules"

      - name: shellcheck
        run: shellcheck scripts/*.sh
```

> **Implementer note on actionlint:** spec §5.3 explicitly says the `download-actionlint.bash` script performs SHA-256 verification of the downloaded archive against the value baked into the script for the requested tag. Pinning the tag (`v1.7.7`) therefore also pins the expected hash. **No** literal SHA-256 placeholder appears in this workflow for actionlint — the verification is intrinsic to the upstream script. If at implementation time the upstream release page documents a different recommended invocation, follow the official page rather than this template.

- [ ] **Step 3.2b: Local lint of the new workflow**

```bash
uv run --with yamllint yamllint .github/workflows/ci.yml
```
Expected: clean.

If `actionlint` is installed locally:

```bash
actionlint .github/workflows/ci.yml
```
Expected: clean.

- [ ] **Step 3.2c: Verify the SHA-pin grep produces no `@v…` matches in `ci.yml`**

```bash
grep -E 'uses: [^@]+@v[0-9]' .github/workflows/ci.yml || echo NO-FLOATING-TAGS
```
Expected: `NO-FLOATING-TAGS`.

- [ ] **Step 3.2d: Verify no `continue-on-error` slipped in**

```bash
grep -n 'continue-on-error' .github/workflows/ci.yml || echo CLEAN
```
Expected: `CLEAN`.

- [ ] **Step 3.2e: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: SHA-pin actions, add concurrency, add lock-check + lint-hygiene jobs, upload coverage HTML"
```

- [ ] **Step 3.2f: Push and watch CI**

```bash
git push
gh run watch --exit-status
```
Expected: `test`, `lock-check`, `lint-hygiene` all pass. Critical-path wall time should remain under 5 minutes — note the per-job timings in the PR description.

**Halt condition:** if any of the three jobs fails, the corresponding Phase-2 task missed a violation. Diagnose, fix at the source (not by adding `continue-on-error`), commit the fix, and re-push.

---

# Phase 4 — Add `security.yml`

**Goal:** One commit. New file `.github/workflows/security.yml` with `pip-audit` and `gitleaks` jobs, both blocking, both with SHA-pinned `actions/checkout`, top-level `concurrency`.

---

## Task 4.1: Resolve the gitleaks tarball SHA-256

**Files:**
- None (read-only step; value is recorded in the Task 4.2 edit).

- [ ] **Step 4.1a: Fetch the upstream SHA-256 for the pinned gitleaks release**

The implementer pastes the value from the official gitleaks release page (https://github.com/gitleaks/gitleaks/releases/tag/v8.21.2). Two acceptable sources for the hash, in order of preference:

1. The `checksums.txt` file attached to the release (most authoritative).
2. Recompute locally:

```bash
GITLEAKS_VERSION=8.21.2
curl -sSL -o /tmp/gitleaks.tar.gz \
  "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
sha256sum /tmp/gitleaks.tar.gz
```

Record the 64-character hex string; you'll paste it into Task 4.2's `GITLEAKS_SHA256` env entry.

**Halt condition:** if the official `checksums.txt` and the locally-computed value disagree, halt — this indicates either a tampered download or a moved release artifact. Investigate before proceeding.

---

## Task 4.2: Create `.github/workflows/security.yml`

**Files:**
- Create: `.github/workflows/security.yml`

This is one commit. Replace `<SHA-checkout>` with the value resolved in Task 3.1 (same SHA as in `ci.yml`) and `<gitleaks-sha256>` with the value from Task 4.1.

- [ ] **Step 4.2a: Create the workflow file**

Create `.github/workflows/security.yml`:

```yaml
name: Security

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  pip-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<SHA-checkout>             # v4.2.2

      - name: Install uv
        uses: astral-sh/setup-uv@<SHA-setup-uv>           # v3.2.0
        with:
          enable-cache: true

      - name: Set Python version
        run: uv python install 3.13

      - name: Install deps
        run: uv sync --all-extras

      - name: pip-audit
        run: uv run --with pip-audit pip-audit --skip-editable

  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout (PR — full history)
        if: github.event_name == 'pull_request'
        uses: actions/checkout@<SHA-checkout>             # v4.2.2
        with:
          fetch-depth: 0

      - name: Checkout (push — shallow)
        if: github.event_name != 'pull_request'
        uses: actions/checkout@<SHA-checkout>             # v4.2.2

      - name: Download gitleaks
        env:
          GITLEAKS_VERSION: "8.21.2"
          # SHA-256 of gitleaks_${VERSION}_linux_x64.tar.gz from the upstream release page.
          # Dependabot's github-actions ecosystem does not bump non-Action versions, so any
          # future GITLEAKS_VERSION bump is a manual edit that also rotates this hash.
          GITLEAKS_SHA256: "<gitleaks-sha256>"
        run: |
          set -euo pipefail
          curl -sSL -o gitleaks.tar.gz \
            "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
          echo "${GITLEAKS_SHA256}  gitleaks.tar.gz" | sha256sum -c -
          tar -xzf gitleaks.tar.gz gitleaks

      - name: gitleaks (PR — full)
        if: github.event_name == 'pull_request'
        run: ./gitleaks detect --no-banner --redact --verbose

      - name: gitleaks (push — range)
        if: github.event_name != 'pull_request'
        run: ./gitleaks detect --no-banner --redact --verbose --log-opts "${{ github.event.before }}..${{ github.sha }}"
```

> **Implementer note on gitleaks event-conditional checkout:** spec §5.8 distinguishes `pull_request` (full history needed for the PR-branch scan) from `push` (shallow + push-range scan). The two `actions/checkout` steps above are mutually exclusive via `if:` and only one runs per event. Same for the two `gitleaks detect` invocations.

- [ ] **Step 4.2b: Local lint of the new workflow**

```bash
uv run --with yamllint yamllint .github/workflows/security.yml
```
Expected: clean.

If `actionlint` is installed locally:

```bash
actionlint .github/workflows/security.yml
```
Expected: clean.

- [ ] **Step 4.2c: Verify no floating tag pins**

```bash
grep -E 'uses: [^@]+@v[0-9]' .github/workflows/security.yml || echo NO-FLOATING-TAGS
```
Expected: `NO-FLOATING-TAGS`.

- [ ] **Step 4.2d: Verify no `continue-on-error`**

```bash
grep -n 'continue-on-error' .github/workflows/security.yml || echo CLEAN
```
Expected: `CLEAN`.

- [ ] **Step 4.2e: Commit**

```bash
git add .github/workflows/security.yml
git commit -m "ci(security): add security.yml with pip-audit and gitleaks (CLI binary, SHA-256 verified)"
```

- [ ] **Step 4.2f: Push and watch CI**

```bash
git push
gh run watch --exit-status
```
Expected: both `pip-audit` and `gitleaks` jobs pass. `ci.yml` jobs also still pass.

**Halt condition:** if `pip-audit` fails, it means a new advisory appeared between Phase 2.5 and Phase 4 push. Re-run Task 2.5; commit the fix; re-push. If `gitleaks` fails, re-run Task 2.6.

---

# Phase 5 — Final verification

**Goal:** Confirm the final state matches every acceptance criterion in spec §7.

---

## Task 5.1: Acceptance-criteria sweep on the merged-into-PR state

- [ ] **Step 5.1a: Mark the PR ready (orchestrator step — out of plan scope for an implementer subagent)**

This is handled by the orchestrator per the CLAUDE.md `Implementation-Orchestrator Pipeline` step 4 — listed here only so the implementer doesn't accidentally do it.

- [ ] **Step 5.1b: All five jobs pass on the PR**

```bash
gh pr checks --watch
```
Expected: `test`, `lock-check`, `lint-hygiene`, `pip-audit`, `gitleaks` all pass.

- [ ] **Step 5.1c: Zero `continue-on-error` anywhere in either workflow**

```bash
grep -rn 'continue-on-error' .github/workflows/ || echo NONE
```
Expected: `NONE`.

- [ ] **Step 5.1d: Every third-party `uses:` is a 40-char SHA + version comment**

```bash
grep -E 'uses: [^@]+@v[0-9]' .github/workflows/ || echo NO-FLOATING-TAGS
```
Expected: `NO-FLOATING-TAGS`.

For positive confirmation (each `uses:` is exactly `@<40-hex> # v...`):

```bash
grep -nE 'uses: ' .github/workflows/*.yml
```
Manually eyeball: every line should be `uses: owner/name@<40-hex>` followed by a `# v...` comment.

- [ ] **Step 5.1e: Coverage HTML artifact uploads and renders**

```bash
RUN_ID=$(gh run list --workflow CI --branch spec/ci-hardening --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download "$RUN_ID" --name coverage-html --dir /tmp/htmlcov-pr
ls /tmp/htmlcov-pr/index.html
```
Expected: the file exists. Optionally open in a browser to confirm rendering.

- [ ] **Step 5.1f: `ci.yml` critical-path wall time under 5 minutes**

```bash
RUN_ID=$(gh run list --workflow CI --branch spec/ci-hardening --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN_ID" --json jobs --jq '.jobs[] | {name: .name, started: .startedAt, completed: .completedAt}'
```

Compute `completed - started` for each job; the largest value is the critical path. **Halt condition:** if the critical path exceeds 5 minutes, profile the slowest job. Likely culprits: `uv sync --all-extras` cold-cache (cache should warm in subsequent runs), or `pytest` itself. Investigate before declaring the PR ready.

- [ ] **Step 5.1g: No `src/esam3/` files were modified (except where ruff `S` legitimately required source fixes in Phase 2.1)**

```bash
git diff --name-only origin/main...HEAD -- src/esam3/ | head
```
Expected: either no output (nothing modified), or a small set of files whose diff is **only** the Phase 2.1 ruff `S` fix (no logic change beyond the security fix itself). Anything else indicates scope leak.

- [ ] **Step 5.1h: Dependabot config parses on the repo**

After the PR merges (or even from the open PR, on the **base** branch's Insights tab if available):

1. Open: `https://github.com/<owner>/<repo>/network/updates`
2. Confirm both `pip` and `github-actions` ecosystems are listed with no parse errors.

**Halt condition:** if Dependabot's UI shows a parse error, fix `.github/dependabot.yml`, commit, and re-push. (This may not be testable until the PR merges to `main`; record as a post-merge verification step in the PR description.)

---

## Task 5.2: Final summary in PR description

- [ ] **Step 5.2a: Update the PR description with the final-state checklist**

The PR description should include (copy-paste from spec §7):

```
- [x] All jobs in both workflows pass on the PR.
- [x] Zero `continue-on-error` flags anywhere in either workflow.
- [x] Every third-party Action reference is a 40-character SHA with a trailing `# v...` version comment.
- [ ] `.github/dependabot.yml` validates (post-merge verification on the Dependabot Insights tab).
- [x] The `coverage-html` artifact uploads, downloads, and renders.
- [x] `ci.yml` critical-path wall time stays under 5 minutes on `ubuntu-latest`.
- [x] No `src/esam3/` file modified by this PR (except as required by Phase 2.1 ruff S fixes — listed: <files>).
```

---

## Spec coverage map

| Spec section | Phase / Task |
|---|---|
| §1 Current State — ruff `S` enabled | 1.5 (config), 2.1 (fix) |
| §1 — `uv lock --check` job | 3.2 (job), 2.7 (lockfile-prep) |
| §1 — `lint-hygiene` job | 3.2 (job), 2.2/2.3/2.4 (fixes) |
| §1 — new `security.yml` | 4.2 (file), 2.5/2.6 (fixes) |
| §1 — Dependabot | 1.4 |
| §1 — Repo-root configs (`.gitleaks.toml`, `.markdownlint.json`, `.yamllint.yml`) | 1.1, 1.2, 1.3 |
| §1 — HTML coverage artifact | 1.5 (pytest), 3.2 (upload step) |
| §1 — `.gitignore` `htmlcov/` | 1.5a (verified pre-existing) |
| §3 File map | matches Phase 1–4 |
| §4 Job map — all five jobs blocking | 3.2 + 4.2; verified 5.1c |
| §4 — no `continue-on-error` | verified 5.1c |
| §5.1 ruff `S` (bandit-equiv) | 1.5, 2.1 |
| §5.2 `uv lock --check` | 3.2 (job spec), 2.7 (lockfile-prep) |
| §5.3 actionlint (CLI binary, intrinsic SHA-256 via download script) | 3.2 |
| §5.4 yamllint | 1.3 (config), 3.2 (job), 2.3 (fix) |
| §5.5 markdownlint-cli2 | 1.2 (config), 3.2 (job), 2.2 (fix) |
| §5.6 shellcheck | 3.2 (job), 2.4 (fix) |
| §5.7 pip-audit (`--skip-editable`; **halt** on unfixable vuln, no `--ignore-vuln` in this PR) | 4.2 (job), 2.5 (fix); halt encoded in 2.5d |
| §5.8 gitleaks (OSS CLI binary, NOT `gitleaks-action`; SHA-256 verified; event-conditional checkout & invocation) | 4.1 (hash), 4.2 (job), 2.6 (fix), 1.1 (config) |
| §5.9 Coverage HTML artifact (`retention-days: 3`, `if: always()`) | 1.5 (pytest addopts), 3.2 (upload step) |
| §5.10 Dependabot (pip + github-actions, weekly, `dev-deps` + `patch-updates` groups) | 1.4 |
| §5.11 SHA-pin every third-party Action with `# v...` comment | 3.1 (resolve), 3.2/4.2 (apply), 5.1d (verify) |
| §5.12 Top-level `concurrency` on both workflows | 3.2, 4.2 |
| §6.1 Foundation (no new CI jobs) | Phase 1 |
| §6.2 Violation fixes, one tool per commit | Phase 2 (one task per sub-step a–g) |
| §6.3 Extend `ci.yml` | Phase 3 |
| §6.4 Add `security.yml` | Phase 4 |
| §6.5 Verify | Phase 5 |
| §7 Acceptance criteria | Task 5.1 |
| §8 Deferred (CodeQL, bandit, trufflehog, SaaS, gate-raise) | not implemented |
| §9 License audit | not implemented (audit-only) |
