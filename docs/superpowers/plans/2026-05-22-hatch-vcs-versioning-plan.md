# hatch-vcs Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `custom-sam-peft` from a hard-coded `pyproject.toml` `version` field to `hatch-vcs`, so the git tag is the single source of truth for the package version.

**Architecture:** Remove the literal `version` from `pyproject.toml` and declare it `dynamic`. Add `hatch-vcs` to the build requirements; configure it to derive the version from `git describe --tags` and write a generated `src/custom_sam_peft/_version.py` (untracked). Re-export `__version__` from that generated module in the package `__init__`. CI checkouts fetch full history + tags so editable installs can resolve a dev-version. The Docker build, which has no `.git/` in its context, receives the version via a project-scoped pretend-version build arg / env var that the release workflow derives from the pushed tag.

**Tech Stack:** `hatchling`, `hatch-vcs`, `uv`, `pytest`, `packaging`, GitHub Actions, Docker.

**Reference:** `docs/superpowers/specs/2026-05-22-hatch-vcs-versioning-design.md`

---

## Task ordering & parallelism

Task 1 (pyproject.toml) and Task 2 (package `__init__.py` + `.gitignore`) edit files that are read by Tasks 3, 4, and the verification task. They must complete first, in this order:

1. **Task 1** — pyproject.toml dynamic version (must precede everything else; subsequent tasks rely on the new build config).
2. **Task 2** — `.gitignore` + `src/custom_sam_peft/__init__.py` (depends on Task 1's generated `_version.py` shape).
3. **Task 3** — `uv.lock` regeneration (depends on Task 1's `pyproject.toml` change).
4. **Task 4** — `tests/unit/test_version.py` (depends on Task 2's `__init__.py` re-export).
5. **Tasks 5 & 6** — Dockerfile and `.github/workflows/docker.yml` (file-disjoint, can run in parallel after Task 1).
6. **Task 7** — `.github/workflows/ci.yml` (file-disjoint with 5 & 6, can run in parallel after Task 1).
7. **Task 8** — Final local verification suite (must be last).

Tasks 5, 6, 7 are file-disjoint and have no dependency on Tasks 2, 3, or 4 (they don't touch Python source or the lockfile). They may be executed in parallel once Task 1 is committed.

---

## Task 1: Switch `pyproject.toml` to dynamic version via `hatch-vcs`

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Edit the `[project]` table — replace static `version` with `dynamic = ["version"]`**

In `pyproject.toml`, replace the current header block:

```toml
[project]
name = "custom-sam-peft"
version = "0.8.0"
description = "Closed-vocab finetuning of SAM-family models with LoRA / QLoRA on a single consumer GPU"
```

with:

```toml
[project]
name = "custom-sam-peft"
dynamic = ["version"]
description = "Closed-vocab finetuning of SAM-family models with LoRA / QLoRA on a single consumer GPU"
```

`dynamic` is placed immediately after `name`, in the slot the literal `version` previously occupied.

- [ ] **Step 2: Add `hatch-vcs` to `[build-system].requires`**

Replace:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

with:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"
```

- [ ] **Step 3: Add the `hatch-vcs` configuration tables**

Insert the following three tables immediately after `[tool.hatch.build.targets.wheel]` (so the new tables sit grouped with the other `[tool.hatch.*]` configuration, before `[tool.ruff]`):

```toml
[tool.hatch.version]
source = "vcs"

[tool.hatch.version.raw-options]
fallback_version = "0.0.0+unknown"

[tool.hatch.build.hooks.vcs]
version-file = "src/custom_sam_peft/_version.py"
```

Leave `[tool.hatch.metadata] allow-direct-references = true` untouched — it is required by the `sam3` git+https dependency.

- [ ] **Step 4: Verify `pyproject.toml` parses and the project metadata resolves**

Run:

```bash
uv run --with tomli python -c "import tomllib, pathlib; d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(d['project']['name']); print('dynamic:', d['project'].get('dynamic')); assert 'version' not in d['project'], 'static version still present'; print('build requires:', d['build-system']['requires']); print('vcs source:', d['tool']['hatch']['version']['source']); print('version-file:', d['tool']['hatch']['build']['hooks']['vcs']['version-file'])"
```

Expected output:

```
custom-sam-peft
dynamic: ['version']
build requires: ['hatchling', 'hatch-vcs']
vcs source: vcs
version-file: src/custom_sam_peft/_version.py
```

(No `AssertionError`.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: declare version dynamic and configure hatch-vcs"
```

---

## Task 2: Re-export `__version__` from the generated module; ignore the generated file

**Files:**

- Modify: `src/custom_sam_peft/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Replace the stale literal in `src/custom_sam_peft/__init__.py`**

The current file is:

```python
"""custom_sam_peft — parameter-efficient finetuning of SAM3.1."""

__version__ = "0.0.1"
```

Replace its entire contents with:

```python
"""custom_sam_peft — parameter-efficient finetuning of SAM3.1."""

from ._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 2: Add the generated `_version.py` to `.gitignore`**

`.gitignore` currently has a "Runtime artifacts" section starting at the `# Runtime artifacts` comment. Add a new dedicated section just below the existing `# Python` block (i.e., immediately after the `.ruff_cache/` line and the blank line that follows it, before `# Virtualenv`):

```
# Build artifacts (generated by hatch-vcs)
src/custom_sam_peft/_version.py

```

The final file should contain the new section near the top with a blank line before `# Virtualenv`.

- [ ] **Step 3: Verify `.gitignore` excludes the generated path**

Run from the repo root:

```bash
touch src/custom_sam_peft/_version.py && git check-ignore -v src/custom_sam_peft/_version.py; rm src/custom_sam_peft/_version.py
```

Expected output (the line/pattern numbers may differ; the key is that `git check-ignore` exits 0 and prints a matching line):

```
.gitignore:<N>:src/custom_sam_peft/_version.py	src/custom_sam_peft/_version.py
```

- [ ] **Step 4: Verify `__init__.py` is syntactically valid**

Run:

```bash
uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('src/custom_sam_peft/__init__.py').read_text()); print('ok')"
```

Expected output:

```
ok
```

(We do not import the package yet — that requires Task 3's lockfile sync so `_version.py` gets generated by the build hook.)

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/__init__.py .gitignore
git commit -m "feat: re-export __version__ from generated _version.py"
```

---

## Task 3: Regenerate `uv.lock`

**Files:**

- Modify: `uv.lock`

- [ ] **Step 1: Regenerate the lockfile against the new `pyproject.toml`**

Run from the repo root:

```bash
uv lock
```

This recomputes the dependency lock to reflect the build-system change (`hatch-vcs` added to `[build-system].requires`).

- [ ] **Step 2: Verify the lockfile is consistent with `pyproject.toml`**

Run:

```bash
uv lock --check
```

Expected: exits 0 with no message about being out of date.

- [ ] **Step 3: Verify `hatch-vcs` resolves to a real version in the lockfile**

Run:

```bash
grep -E '^name = "hatch-vcs"' uv.lock
```

Expected: at least one matching line (the exact version pin depends on the resolver). If `uv.lock` does not include build-system requirements in its package list (lock format dependent), skip the assertion and rely on Step 2's `uv lock --check`.

- [ ] **Step 4: Run `uv sync` and confirm `_version.py` is generated**

Run:

```bash
uv sync --all-extras
ls -la src/custom_sam_peft/_version.py
```

Expected: `ls` shows the file exists. Then:

```bash
uv run python -c "import custom_sam_peft; print(custom_sam_peft.__version__)"
```

Expected: prints a PEP 440 version string. From a non-tag working tree this typically looks like `0.8.1.devN+g<sha>` or similar; from a clean checkout at exactly tag `vX.Y.Z` it would be `X.Y.Z`. Either is acceptable here — the test in Task 4 enforces PEP 440 validity.

- [ ] **Step 5: Confirm `_version.py` is untracked**

Run:

```bash
git status --short src/custom_sam_peft/_version.py
```

Expected: empty output (the file is ignored, so `git status` shows nothing).

Also run:

```bash
git ls-files src/custom_sam_peft/_version.py
```

Expected: empty output (the file is not tracked).

- [ ] **Step 6: Commit the lockfile**

```bash
git add uv.lock
git commit -m "build: regenerate uv.lock for hatch-vcs build requirement"
```

---

## Task 4: Add the `_version.py` smoke test

**Files:**

- Create: `tests/unit/test_version.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_version.py`:

```python
"""Smoke test that the hatch-vcs build hook populated __version__."""

from packaging.version import Version

import custom_sam_peft


def test_version_is_valid_pep440() -> None:
    assert isinstance(custom_sam_peft.__version__, str)
    assert custom_sam_peft.__version__, "__version__ must not be empty"
    # Raises InvalidVersion if not parseable.
    Version(custom_sam_peft.__version__)
```

- [ ] **Step 2: Run the test**

Run:

```bash
uv run pytest tests/unit/test_version.py -v
```

Expected: 1 passed. The build hook produces a real PEP 440 string (e.g., `0.8.1.dev3+g<sha>` from a non-tag commit, or `X.Y.Z` from a tagged commit), and `packaging.version.Version` accepts it.

If the test fails with `ImportError: cannot import name '__version__'` or `ModuleNotFoundError: custom_sam_peft._version`, re-run `uv sync --all-extras` (Task 3 Step 4) to regenerate `_version.py` and re-run the test.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_version.py
git commit -m "test: assert __version__ is a valid PEP 440 string"
```

---

## Task 5: Pass the pretend-version build arg through the Dockerfile

**Files:**

- Modify: `Dockerfile`

- [ ] **Step 1: Add the `ARG`/`ENV` pair before any `uv sync` step**

Currently the Dockerfile has this block (lines 9–14):

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface
```

Insert an `ARG`/`ENV` pair between the `COPY --from=...uv...` line and the existing `ENV UV_LINK_MODE=...` block, so the final region reads:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ARG HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT
ENV HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT=${HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT}

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface
```

Do **not** add `.git/` to the build context — the `.dockerignore` exclusion of `.git/` stays as-is. Do not touch any other line of the Dockerfile.

- [ ] **Step 2: Verify the Dockerfile lints**

Run:

```bash
docker buildx build --check -f Dockerfile . 2>&1 | head -40 || true
```

(Some buildkit versions do not support `--check`; if it fails with "unknown flag", skip this step.) Then verify the file content with:

```bash
grep -n 'HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT' Dockerfile
```

Expected: exactly two matches — one `ARG` line and one `ENV` line, both before the first `uv sync` invocation.

Also verify ordering with:

```bash
awk '/uv sync/ { print NR": uv sync"; exit } /HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT/ { print NR": HATCH_VCS_..." }' Dockerfile
```

Expected: the `HATCH_VCS_...` line(s) print first, then a `uv sync` line.

- [ ] **Step 3: Smoke-test the Docker build locally (optional — skip if Docker is not installed)**

Run:

```bash
docker buildx build --build-arg HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT=9.9.9 -t custom-sam-peft:plan-smoke --load . && \
  docker run --rm custom-sam-peft:plan-smoke pip show custom-sam-peft | grep -E '^Version:'
```

Expected: `Version: 9.9.9`.

If Docker is unavailable in the local environment, skip this step — the CI Docker workflow in Task 6 covers it on tag push.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "build(docker): accept hatch-vcs pretend version as build arg"
```

---

## Task 6: Forward the pretend version from the Docker release workflow

**Files:**

- Modify: `.github/workflows/docker.yml`

- [ ] **Step 1: Add a `strip-v` step after the `meta` step and before the first `docker/build-push-action`**

The current workflow has this sequence (lines 29–47 in the file):

```yaml
      - id: meta
        uses: docker/metadata-action@c299e40c65443455700f0fdfc63efafe5b349051       # v5.10.0
        with:
          images: ghcr.io/nguyenjus/custom-sam-peft
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}},enable=${{ !startsWith(github.ref, 'refs/tags/v0.') }}
            type=raw,value=latest

      - uses: docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8   # v6.19.2
        with:
          context: .
          platforms: linux/amd64
          load: true
          push: false
          tags: ghcr.io/nguyenjus/custom-sam-peft:ci-smoke
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

Insert a new step between the `meta` step and the first `docker/build-push-action` so the region becomes:

```yaml
      - id: meta
        uses: docker/metadata-action@c299e40c65443455700f0fdfc63efafe5b349051       # v5.10.0
        with:
          images: ghcr.io/nguyenjus/custom-sam-peft
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}},enable=${{ !startsWith(github.ref, 'refs/tags/v0.') }}
            type=raw,value=latest

      - id: strip-v
        run: echo "version=${GITHUB_REF_NAME#v}" >> "$GITHUB_OUTPUT"

      - uses: docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8   # v6.19.2
        with:
          context: .
          platforms: linux/amd64
          load: true
          push: false
          tags: ghcr.io/nguyenjus/custom-sam-peft:ci-smoke
          build-args: |
            HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT=${{ steps.strip-v.outputs.version }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

(The new `build-args:` block is added to the first `docker/build-push-action` invocation alongside the existing `cache-from`/`cache-to`.)

- [ ] **Step 2: Add the same `build-args` block to the second `docker/build-push-action` invocation**

The current second invocation (lines 54–62):

```yaml
      - uses: docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8   # v6.19.2 (same action, second use)
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
```

becomes:

```yaml
      - uses: docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8   # v6.19.2 (same action, second use)
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          build-args: |
            HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT=${{ steps.strip-v.outputs.version }}
          cache-from: type=gha
```

Do **not** modify the `actions/checkout` step in this workflow — the workflow only runs on tag push, and `github.ref_name` already carries the tag name. The workflow does not run `uv build` or `hatch build` directly.

- [ ] **Step 3: Lint the workflow**

Run:

```bash
./actionlint -color .github/workflows/docker.yml || (bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7 && ./actionlint -color .github/workflows/docker.yml)
```

Expected: exits 0 with no errors. (The fallback `bash <(...)` downloads `actionlint` only if not already in the repo root; the CI `lint-hygiene` job uses the same download.)

Then:

```bash
uv run --with yamllint yamllint -c .config/yamllint.yml .github/workflows/docker.yml
```

Expected: exits 0 with no errors.

- [ ] **Step 4: Confirm both build-push steps got the build-arg and the strip-v step is present**

Run:

```bash
grep -nE 'HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT|strip-v' .github/workflows/docker.yml
```

Expected: at least three matches — one `id: strip-v` line and one `HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT=...` reference per `docker/build-push-action` (so two `HATCH_VCS_...` lines plus the `strip-v` `id` and its `run`, four lines total).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/docker.yml
git commit -m "ci(docker): forward stripped tag as hatch-vcs pretend version"
```

---

## Task 7: Fetch full history + tags in every CI checkout

**Files:**

- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add `fetch-depth: 0` and `fetch-tags: true` to every `actions/checkout`**

There are four `actions/checkout` steps in `.github/workflows/ci.yml` (in jobs `test`, `lock-check`, `lint-hygiene`, and `gpu-deselect-check`). Each currently looks like this single-line form:

```yaml
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd             # v6.0.2
```

Replace each occurrence with the multi-line form:

```yaml
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd             # v6.0.2
        with:
          fetch-depth: 0
          fetch-tags: true
```

Preserve the trailing pin comment (`# v6.0.2`) and the indentation of each line. Apply the change to **all four** occurrences uniformly.

- [ ] **Step 2: Verify all four occurrences are updated**

Run:

```bash
grep -c 'actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd' .github/workflows/ci.yml
```

Expected: `4`.

Then:

```bash
grep -c 'fetch-depth: 0' .github/workflows/ci.yml && grep -c 'fetch-tags: true' .github/workflows/ci.yml
```

Expected: `4` and `4`.

- [ ] **Step 3: Lint the workflow**

Run:

```bash
./actionlint -color .github/workflows/ci.yml || (bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/v1.7.7/scripts/download-actionlint.bash) 1.7.7 && ./actionlint -color .github/workflows/ci.yml)
```

Expected: exits 0 with no errors.

Then:

```bash
uv run --with yamllint yamllint -c .config/yamllint.yml .github/workflows/ci.yml
```

Expected: exits 0 with no errors.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: fetch full history and tags so hatch-vcs can resolve a dev version"
```

---

## Task 8: Final local verification suite

This task runs the full set of checks from the spec's Verification section against the committed state of all prior tasks. No new files are created.

**Files:** none (read-only verification).

- [ ] **Step 1: Clean any stale build artifacts and rebuild the environment**

Run:

```bash
rm -rf dist/ build/ src/custom_sam_peft/_version.py
uv sync --all-extras
```

Expected: `uv sync` succeeds; the build hook recreates `src/custom_sam_peft/_version.py`.

- [ ] **Step 2: Build a wheel and inspect its declared version**

Run:

```bash
uv build --wheel
ls dist/
unzip -p dist/custom_sam_peft-*.whl '*/METADATA' | grep -E '^Version:'
```

Expected: `dist/` contains one `custom_sam_peft-<version>-py3-none-any.whl` file; `Version:` line reports a PEP 440 string consistent with the current git state. From a non-tag commit this looks like `Version: 0.8.1.devN+g<sha>` (exact dev-number and short-sha vary).

- [ ] **Step 3: Import the package and print `__version__`**

Run:

```bash
uv run python -c "import custom_sam_peft; print(custom_sam_peft.__version__)"
```

Expected: prints the same PEP 440 string seen in Step 2 (or an equivalent — the value is regenerated each time the build hook runs).

- [ ] **Step 4: Run the new version test**

Run:

```bash
uv run pytest tests/unit/test_version.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Confirm `_version.py` is untracked**

Run:

```bash
git status --short | grep _version.py || echo 'OK: not in git status'
git ls-files src/custom_sam_peft/_version.py
```

Expected: first command prints `OK: not in git status`; second command prints nothing.

- [ ] **Step 6: Confirm the lockfile is in sync**

Run:

```bash
uv lock --check
```

Expected: exits 0 with no "out of date" message.

- [ ] **Step 7: Run the full lint + type-check + test suite**

Run each of these in turn:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/custom_sam_peft
uv run pytest
```

Expected: each command exits 0. `pytest` reports all tests passing and the project's existing `--cov-fail-under=80` coverage threshold is met.

- [ ] **Step 8: Confirm no static `version =` literal remains in `pyproject.toml` and no stale literal remains in `__init__.py`**

Run:

```bash
grep -nE '^version\s*=' pyproject.toml || echo 'OK: no static version in pyproject.toml'
grep -nE '__version__\s*=\s*"' src/custom_sam_peft/__init__.py || echo 'OK: no literal __version__ in __init__.py'
```

Expected: both commands print their `OK:` message. (The `grep` exits non-zero with no match, the `||` runs the `echo`.)

- [ ] **Step 9: Confirm `actions/checkout` is consistently configured in `ci.yml`**

Run:

```bash
grep -B1 -A3 'actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd' .github/workflows/ci.yml
```

Expected: four blocks shown, each with `fetch-depth: 0` and `fetch-tags: true` immediately below the `uses:` line.

- [ ] **Step 10: Confirm Docker workflow forwards the pretend-version arg**

Run:

```bash
grep -nE 'HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT|id: strip-v' .github/workflows/docker.yml
```

Expected: at least one `id: strip-v` line, one `run: echo "version=${GITHUB_REF_NAME#v}"` line (which the previous grep also catches via `strip-v`), and two `HATCH_VCS_PRETEND_VERSION_FOR_CUSTOM_SAM_PEFT=...` references (one per `docker/build-push-action`).

- [ ] **Step 11: Push the branch and confirm all CI jobs pass**

Run:

```bash
git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
gh pr create --fill --assignee @me --label build --label ci
gh pr checks --watch
```

Expected: `test`, `lock-check`, `lint-hygiene`, and `gpu-deselect-check` all complete with a green status. The Docker workflow does not run on PRs (it only runs on tag pushes); CI green here is the merge gate.

If the `--label` arguments fail because the labels do not exist, list available labels with `gh label list` and substitute appropriate existing labels, or create them inline with `gh label create <name> --description <desc> --color <hex>` before re-running `gh pr edit --add-label <name>`.

At this point the PR is ready to merge.

---

## Coverage map (spec → tasks)

| Spec "Changes" subsection            | Task(s)        |
| ------------------------------------ | -------------- |
| `pyproject.toml`                     | Task 1         |
| `src/custom_sam_peft/__init__.py`    | Task 2         |
| `.gitignore`                         | Task 2         |
| `Dockerfile`                         | Task 5         |
| `.github/workflows/docker.yml`       | Task 6         |
| `.github/workflows/ci.yml`           | Task 7         |
| `tests/unit/test_version.py` (new)   | Task 4         |
| `uv.lock`                            | Task 3         |
| Verification section                 | Task 8         |
