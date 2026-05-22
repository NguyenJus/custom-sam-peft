# Publish Docker Image to GHCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-21-publish-docker-image-design.md`](../specs/2026-05-21-publish-docker-image-design.md)
**Issue:** [#34](https://github.com/NguyenJus/custom-sam-peft/issues/34) — *Containerize esam3 (publish Docker image)*
**Branch:** `spec/publish-docker-image`

**Goal:** Build and publish a CUDA-enabled Docker image of `custom-sam-peft` to GHCR (`ghcr.io/nguyenjus/custom-sam-peft`), triggered on semver tag pushes, with a build-and-push CI workflow, per-provider Docker walkthrough, cross-links in existing docs, and a new `jupyter` optional extras group.

**Architecture:** A single-platform (`linux/amd64`) image built on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` with `uv` for dependency installation. Two-stage layer-cache pattern (deps-only first, then full install with source) maximizes cache reuse. The workflow fires only on semver tag pushes; a smoke test (`--help` + `doctor --json`) runs against the loaded image before any push. All user data lives under a host-mounted `/workspace` volume. The `jupyter` extras group enables JupyterLab for users who prefer the notebook flow. No source code in `src/` is touched.

**Tech Stack:** Docker BuildKit (buildx), `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, `uv` 0.5.11 (copied from `ghcr.io/astral-sh/uv:0.5.11`), GitHub Actions (SHA-pinned: `actions/checkout`, `docker/setup-buildx-action`, `docker/login-action`, `docker/metadata-action`, `docker/build-push-action`), `actionlint`, `yamllint`, `markdownlint`.

---

## File Map

**New files:**

```
Dockerfile                         NEW
.dockerignore                      NEW
.github/workflows/docker.yml       NEW
cloud/docker/README.md             NEW   (new directory: cloud/docker/)
```

**Modified files:**

```
cloud/runpod/README.md             TOUCHED  (blockquote callout prepended at very top)
README.md                          TOUCHED  (Beginner section + Advanced/Quickstart subsection)
pyproject.toml                     TOUCHED  (+1 extras group: jupyter)
uv.lock                            REGENERATED via `uv lock`
```

No deletions, no moves. No source code in `src/` is touched. No tests in `tests/` added or modified.

---

## Parallelization opportunities (for orchestrator dispatch)

**Phase 0** (pyproject + lockfile) blocks **Phase 1** (Dockerfile) because the Dockerfile COPYs `uv.lock` and runs `uv sync --frozen`. Phase 0 must therefore complete and commit before Phase 1 begins.

**Phases 1, 2, and 3** are largely file-disjoint once Phase 0 is committed:

- **Phase 1** touches only `Dockerfile` and `.dockerignore`.
- **Phase 2** touches only `.github/workflows/docker.yml`.
- **Phase 3** touches only `cloud/docker/README.md`, `cloud/runpod/README.md`, and `README.md`.

The orchestrator may fan Phases 1, 2, and 3 out in parallel using `superpowers:dispatching-parallel-agents`. All three land on the same branch/worktree; no `isolation: "worktree"` is needed.

**Phase 4** (PR) serializes after Phases 1–3.
**Phase 5** (post-merge manual steps) is not part of the PR — it is operator work performed after merge.

Dependency graph:

```
Phase 0 → Phase 1 ┐
         → Phase 2 ├→ Phase 4 → Phase 5 (manual, post-merge)
         → Phase 3 ┘
```

---

## Pre-flight check

- [ ] **Step 0a: Confirm working tree is clean**

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image status
```

Expected: only this plan file and the committed spec shown. No unexpected modifications.

- [ ] **Step 0b: Confirm baseline unit test suite is green**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && uv run pytest tests/unit -x -q
```

Expected: all unit tests pass. If anything is red, halt and report — do not start work on a broken baseline.

- [ ] **Step 0c: Confirm Docker and buildx are available**

```bash
docker buildx version
```

Expected: `github.com/docker/buildx vX.Y.Z ...`. If not available, the local build verification in Phase 1 cannot run — note this and skip the local build steps; CI will be the verification gate instead.

---

## Phase 0: Add `jupyter` extras group to `pyproject.toml` and regenerate `uv.lock`

**Model/effort:** haiku / medium.
**Parallel:** No. **Blocks Phase 1** (Dockerfile COPYs `uv.lock` and runs `uv sync --frozen`).
**Spec:** §8 (`pyproject.toml` change), §11 (file layout).

**Files:**
- Modify: `pyproject.toml`
- Regenerate: `uv.lock`

**Goal:** Add `jupyter = ["jupyterlab>=4"]` under `[project.optional-dependencies]`, then run `uv lock` to refresh `uv.lock`. Both files commit together in one atomic commit.

- [ ] **Step P0-1: Add the `jupyter` extras entry to `pyproject.toml`**

In `pyproject.toml`, find the `[project.optional-dependencies]` block. It currently reads:

```toml
[project.optional-dependencies]
wandb = ["wandb>=0.18"]
qlora = ["bitsandbytes>=0.43"]
tensorboard = ["tensorboard>=2.18"]
dev = [
  "ruff>=0.7",
  ...
]
```

Insert the new `jupyter` line immediately before the `dev` group (alphabetically it falls between `qlora`/`tensorboard` and `dev` — order within the group is by convention; matching the spec exactly):

```toml
[project.optional-dependencies]
wandb = ["wandb>=0.18"]
qlora = ["bitsandbytes>=0.43"]
tensorboard = ["tensorboard>=2.18"]
jupyter = ["jupyterlab>=4"]
dev = [
  "ruff>=0.7",
  "mypy>=1.13",
  "pytest>=8",
  "pytest-cov>=5",
  "pre-commit>=4",
  "types-PyYAML>=6",
]
```

No other change to this file.

- [ ] **Step P0-2: Regenerate `uv.lock`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && uv lock
```

Expected: `uv.lock` is updated to include `jupyterlab` and its transitive dependencies. The command exits 0. Do NOT hand-edit `uv.lock`.

- [ ] **Step P0-3: Verify the new extras group is resolvable**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && uv sync --extra jupyter --dry-run 2>&1 | head -20
```

Expected: exits 0 and shows `jupyterlab` in the resolved set (or reports "nothing to install" if the venv already has it). The key check is that `uv sync --extra jupyter` does not error on an unresolvable constraint.

- [ ] **Step P0-4: Verify `dev` group is NOT affected**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && grep -n "jupyterlab\|jupyter" pyproject.toml
```

Expected: only the `jupyter = ["jupyterlab>=4"]` line appears. The `dev` group contains no `jupyter` entry.

- [ ] **Step P0-5: Commit `pyproject.toml` and `uv.lock` together**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && git add pyproject.toml uv.lock && git commit -m "build: add jupyter extras group (jupyterlab>=4) for Docker image"
```

---

## Phase 1: Author `Dockerfile` and `.dockerignore`

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 2 and 3 (file-disjoint). **Depends on:** Phase 0.
**Spec:** §3 (Dockerfile), §4 (`.dockerignore`), §6 (mount convention).

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Goal:** Write the canonical Dockerfile exactly as specified in §3, and the `.dockerignore` as specified in §4. Verify locally with `docker buildx build --load .` and a smoke test `docker run --rm <tag> --help`. No GPU is required for this verification.

### Task 1a: Create `Dockerfile`

- [ ] **Step P1-1: Create `Dockerfile`**

Create `Dockerfile` at the repo root with exactly this content (verbatim from spec §3):

```dockerfile
# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /opt/custom-sam-peft

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
            --extra qlora --extra tensorboard --extra wandb --extra jupyter

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
            --extra qlora --extra tensorboard --extra wandb --extra jupyter

ENV PATH="/opt/custom-sam-peft/.venv/bin:$PATH"

LABEL org.opencontainers.image.source="https://github.com/NguyenJus/custom-sam-peft" \
      org.opencontainers.image.description="Parameter-efficient finetuning of SAM3.1 with LoRA/QLoRA" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["custom-sam-peft"]
CMD ["--help"]
```

**Key design notes (do not modify):**
- `WORKDIR /opt/custom-sam-peft` is where the package is installed; `WORKDIR /workspace` is the runtime working directory for the user's volume mount. The switch is intentional.
- Two-stage `uv sync`: deps-only first (`--no-install-project`), then full install with source. This lets Docker reuse the heavy dependency layer when only source files change.
- `UV_PYTHON_DOWNLOADS=never` prevents uv from pulling a Python version different from the one baked into the base image.
- `--extra dev` is intentionally absent. The `dev` group must never be installed in the published image.

### Task 1b: Create `.dockerignore`

- [ ] **Step P1-2: Create `.dockerignore`**

Create `.dockerignore` at the repo root with exactly this content (verbatim from spec §4):

```
.venv/
.git/
.worktrees/
.mypy_cache/
.ruff_cache/
.pytest_cache/
htmlcov/
runs/
models/
data/
notebooks/
tests/
docs/
**/__pycache__
*.egg-info/
```

### Task 1c: Local verification

- [ ] **Step P1-3: Build the image locally (no GPU required)**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && docker buildx build --load -t custom-sam-peft:local-test .
```

Expected: build completes successfully. If the build fails, diagnose the error before proceeding. Common failure modes:
- `uv sync --frozen` fails because `uv.lock` does not include `jupyterlab` — fix: ensure Phase 0 was committed and the Dockerfile's `COPY pyproject.toml uv.lock README.md ./` picks up the updated lockfile.
- `sam3 @ git+...` clone fails — expected in an offline environment; fix: ensure network access, or note as a CI-only verification and skip.

- [ ] **Step P1-4: Smoke test the built image (CLI resolves, no GPU needed)**

```bash
docker run --rm custom-sam-peft:local-test --help
```

Expected: the `custom-sam-peft` help text is printed and the command exits 0. If `--help` fails with an import error, check the `PATH` env var and the `ENTRYPOINT` in the Dockerfile.

```bash
docker run --rm custom-sam-peft:local-test doctor --json
```

Expected: exits 0 with a JSON object. (The `doctor` command checks package imports and the CLI, not GPU or checkpoint presence — no GPU is needed for this check.)

- [ ] **Step P1-5: Confirm `dev` extras are absent from the image**

```bash
docker run --rm custom-sam-peft:local-test python -c "import pytest" 2>&1
```

Expected: `ModuleNotFoundError: No module named 'pytest'`. If `pytest` is importable, the `dev` extras were accidentally included — check the `uv sync` command in the Dockerfile and fix.

- [ ] **Step P1-6: Confirm `jupyter` extras are present**

```bash
docker run --rm custom-sam-peft:local-test python -c "import jupyterlab; print('jupyterlab ok')"
```

Expected: `jupyterlab ok`.

- [ ] **Step P1-7: Commit `Dockerfile` and `.dockerignore`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && git add Dockerfile .dockerignore && git commit -m "feat: add Dockerfile and .dockerignore for GHCR publish (#34)"
```

---

## Phase 2: Author `.github/workflows/docker.yml`

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1 and 3 (file-disjoint). **Depends on:** Phase 0 (committed).
**Spec:** §5 (build & push workflow).

**Files:**
- Create: `.github/workflows/docker.yml`

**Goal:** Write the GitHub Actions workflow exactly as specified in §5.2, with all `<sha>` placeholders replaced by actual pinned SHAs for the latest stable versions of the five actions. Verify with `actionlint` and `yamllint` before committing.

### SHA pinning instructions

The spec marks five action `<sha>` placeholders. The implementer must look up the current latest stable SHA for each action at PR time and pin it. The five actions and their target versions are:

| Action | Target major version | Comment tag to add (format mirrors `ci.yml`) |
| --- | --- | --- |
| `actions/checkout` | v4 | e.g. `# v4.2.2` |
| `docker/setup-buildx-action` | v3 | e.g. `# v3.7.1` |
| `docker/login-action` | v3 | e.g. `# v3.3.0` |
| `docker/metadata-action` | v5 | e.g. `# v5.6.1` |
| `docker/build-push-action` | v6 | e.g. `# v6.13.0` |

To find the current SHA for a given action version tag, run:

```bash
# Example for docker/metadata-action v5
git ls-remote https://github.com/docker/metadata-action refs/tags/v5 | awk '{print $1}'
# Or look it up on the action's GitHub releases page and use the full SHA of the tagged commit.
```

The SHA must be the full 40-character commit SHA of the tag, not the tag itself — this is the pattern already used in `.github/workflows/ci.yml` and all other workflows in this repo. The comment tag (e.g. `# v4.2.2`) is appended on the same line so the human-readable version is visible.

### Task 2a: Create the workflow file

- [ ] **Step P2-1: Look up current SHAs for the five actions**

Run the following for each action to obtain its pinned SHA. Use the full tag commit SHA (not the annotated tag SHA — check the `^{}` dereference):

```bash
# actions/checkout v4
git ls-remote https://github.com/actions/checkout 'refs/tags/v4' | grep -v '\^{}' | awk '{print $1}'
# If the tag is annotated, dereference:
git ls-remote https://github.com/actions/checkout 'refs/tags/v4^{}' | awk '{print $1}'

# docker/setup-buildx-action v3
git ls-remote https://github.com/docker/setup-buildx-action 'refs/tags/v3^{}' | awk '{print $1}'

# docker/login-action v3
git ls-remote https://github.com/docker/login-action 'refs/tags/v3^{}' | awk '{print $1}'

# docker/metadata-action v5
git ls-remote https://github.com/docker/metadata-action 'refs/tags/v5^{}' | awk '{print $1}'

# docker/build-push-action v6
git ls-remote https://github.com/docker/build-push-action 'refs/tags/v6^{}' | awk '{print $1}'
```

Record the five SHAs. Each will replace a `<sha>` placeholder in the workflow below.

- [ ] **Step P2-2: Create `.github/workflows/docker.yml`**

Create `.github/workflows/docker.yml` with the following content, substituting each `<sha-for-ACTION-vN>` with the SHA obtained in Step P2-1 and each `# vN.Y.Z` comment with the resolved version string. The two `docker/build-push-action` steps use the same SHA — apply it to both.

```yaml
name: Docker

on:
  push:
    tags: ["v*"]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read
  packages: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<sha-for-checkout-v4>             # vN.Y.Z

      - uses: docker/setup-buildx-action@<sha-for-setup-buildx-v3> # vN.Y.Z

      - uses: docker/login-action@<sha-for-login-v3>             # vN.Y.Z
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - id: meta
        uses: docker/metadata-action@<sha-for-metadata-v5>        # vN.Y.Z
        with:
          images: ghcr.io/nguyenjus/custom-sam-peft
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}},enable=${{ !startsWith(github.ref, 'refs/tags/v0.') }}
            type=raw,value=latest

      - uses: docker/build-push-action@<sha-for-build-push-v6>   # vN.Y.Z
        with:
          context: .
          platforms: linux/amd64
          load: true
          push: false
          tags: ghcr.io/nguyenjus/custom-sam-peft:ci-smoke
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Smoke test (CLI resolves, package imports)
        run: |
          docker run --rm ghcr.io/nguyenjus/custom-sam-peft:ci-smoke --help
          docker run --rm ghcr.io/nguyenjus/custom-sam-peft:ci-smoke doctor --json

      - uses: docker/build-push-action@<sha-for-build-push-v6>   # vN.Y.Z (same action, second use)
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
```

**Key design notes (do not modify):**
- The workflow fires ONLY on tag pushes (`tags: ["v*"]`). No PR builds, no push-to-main builds.
- The build-load → smoke-test → build-push pattern ensures the smoke test runs against the exact image that will be pushed (same GHA cache state), so a broken image is never published.
- The second `build-push-action` step replays from GHA cache (`cache-from: type=gha`) — it costs only push network time, not a full rebuild.
- The major-version tag (`1`, `2`, …) is suppressed for pre-1.0 releases (`v0.*`) via the `enable=${{ !startsWith(github.ref, 'refs/tags/v0.') }}` condition. This is intentional; do not remove it.
- The `concurrency` group cancels in-progress runs for the same tag (e.g. if a tag is pushed twice in rapid succession).

### Task 2b: Verify the workflow

- [ ] **Step P2-3: Verify with `actionlint`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && actionlint .github/workflows/docker.yml
```

If `actionlint` is not installed, install it first:

```bash
bash <(curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash)
./actionlint .github/workflows/docker.yml
```

Expected: no errors. If actionlint flags a SHA as unrecognized (it cannot validate external SHAs), that is a false-positive — the SHA format is correct as long as it is 40 hex characters.

- [ ] **Step P2-4: Verify with `yamllint`**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && uv run --with yamllint yamllint .github/workflows/docker.yml
```

Expected: no errors or warnings that aren't already present in other workflow files. If yamllint flags a `line-length` issue on a tag line, suppress with an inline comment `# yamllint disable-line rule:line-length` on that line.

- [ ] **Step P2-5: Spot-check the SHA pins look correct (40-hex characters each)**

```bash
grep -E 'uses:.*@' /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image/.github/workflows/docker.yml
```

Expected: each `uses:` line ends with `@<40-character-hex-sha>  # vN.Y.Z`. If any line still contains `<sha-for-...>` (unreplaced placeholder), the implementer must go back and fill it in.

- [ ] **Step P2-6: Commit the workflow**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && git add .github/workflows/docker.yml && git commit -m "ci: add Docker build-and-push workflow for GHCR publish (#34)"
```

---

## Phase 3: Documentation changes

**Model/effort:** sonnet / high.
**Parallel:** Yes, with Phases 1 and 2 (file-disjoint). **Depends on:** Phase 0 (committed).
**Spec:** §7 (documentation changes).

**Files:**
- Create: `cloud/docker/README.md` (new file in new directory)
- Modify: `cloud/runpod/README.md`
- Modify: `README.md`

**Goal:** Write the per-provider Docker walkthrough (`cloud/docker/README.md`), prepend the blockquote callout to `cloud/runpod/README.md`, and apply the two additive patches to `README.md`. Verify all three with `markdownlint`.

### Task 3a: Create `cloud/docker/README.md`

- [ ] **Step P3-1: Create the `cloud/docker/` directory and `README.md`**

Create `cloud/docker/README.md` with the following content (verbatim from spec §7.1, assembled into a coherent document):

````markdown
# Running custom-sam-peft with Docker

The prebuilt image at
[`ghcr.io/nguyenjus/custom-sam-peft`](https://github.com/NguyenJus/custom-sam-peft/pkgs/container/custom-sam-peft)
packages the `custom-sam-peft` CLI and all runtime extras (LoRA/QLoRA,
TensorBoard, W&B, JupyterLab) so you can skip the `pip install git+…` step
and go straight to training.

## What's in the image

- **Base:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- **Extras installed:** `qlora`, `tensorboard`, `wandb`, `jupyter`
  (`dev` extras are not installed)
- **Entrypoint:** `custom-sam-peft` (the CLI)
- **Default command:** `--help`
- **Mount point:** `/workspace` (all user data lives here)
- **Exposed port:** `8888` (JupyterLab)

## Pick a tag

Browse available tags on the
[GHCR package page](https://github.com/NguyenJus/custom-sam-peft/pkgs/container/custom-sam-peft).

Pin to a semver tag rather than `latest` to avoid unintended upgrades:

```bash
docker pull ghcr.io/nguyenjus/custom-sam-peft:v0.6.0
```

`latest` always points to the most recently published semver release.

## Mount convention

All user data lives under `/workspace`. The image's `WORKDIR` is `/workspace`
so relative paths in configs and notebooks resolve naturally inside the
container. Recommended subdirectory layout:

| Subdirectory | Contents |
| --- | --- |
| `data/` | Training datasets (COCO, HF cache, etc.) |
| `runs/` | Run output (`adapter/`, `metrics.json`, `summary.md`, `samples/`) |
| `models/` | SAM 3.1 checkpoint (`models/sam3.1/sam3.1_multiplex.pt`) |
| `.cache/huggingface/` | HF Hub download cache (mirrors `HF_HOME=/workspace/.cache/huggingface`) |

`HF_HOME` is set in the image to `/workspace/.cache/huggingface` so that
weight downloads land inside your mounted volume and survive container
restarts.

## CLI mode (default)

Run training against a config file:

```bash
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z \
  train --config /workspace/config.yaml
```

Replace `vX.Y.Z` with the tag you picked. `$PWD` should be the directory
where your `config.yaml`, `data/`, `models/`, and `runs/` live.

## Jupyter mode

Launch JupyterLab and override the entrypoint:

```bash
docker run --gpus all --rm -p 8888:8888 \
  -v $PWD:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  --entrypoint jupyter \
  ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z \
  lab --ip=0.0.0.0 --no-browser --allow-root
```

Open the URL printed in the container logs (e.g.
`http://127.0.0.1:8888/lab?token=...`) in your browser.

## Per-provider notes

### RunPod {#runpod}

In the RunPod **Custom Template** form:

| Field | Value |
| --- | --- |
| Container Image | `ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z` |
| Container Disk | ≥ 20 GB |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8888` |
| Environment Variable | `HF_TOKEN` = your HF token |

**CLI mode** (Container Start Command):

```
train --config /workspace/config.yaml
```

**Jupyter mode** (Container Start Command — override entrypoint in the
RunPod template's "Docker Command" field):

```
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

For a step-by-step RunPod walkthrough without Docker (from-source install),
see [`cloud/runpod/README.md`](../runpod/README.md).

### Vast.ai {#vastai}

1. In the **Create Instance** form, set the **Image** field to
   `ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z`.
2. Set **Launch Mode** to "Run" (not SSH only).
3. In the **On-start script** field, add:
   ```bash
   export HF_TOKEN=<your-hf-token>
   ```
4. Set the disk mount path to `/workspace`.
5. Add environment variable `HF_TOKEN` with your Hugging Face token.

For CLI mode, set the start command to `train --config /workspace/config.yaml`.
For Jupyter mode, set it to `jupyter lab --ip=0.0.0.0 --no-browser --allow-root`
and override the entrypoint.

### Lambda Labs / generic {#generic}

If you have a GPU instance with Docker and `nvidia-container-toolkit`
installed, use the CLI and Jupyter snippets from the sections above
directly. Confirm the toolkit is available:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this prints a GPU table, your instance is ready. Then pull and run:

```bash
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z \
  train --config /workspace/config.yaml
```
````

- [ ] **Step P3-2: Run `markdownlint` on the new file**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && npx --yes markdownlint-cli2 cloud/docker/README.md
```

Expected: clean. If markdownlint flags a rule on a code block or table, fix by adding an inline disable comment (`<!-- markdownlint-disable-next-line MDxxx -->`) above the offending line — do not restructure the content.

### Task 3b: Patch `cloud/runpod/README.md`

- [ ] **Step P3-3: Prepend the blockquote callout to `cloud/runpod/README.md`**

Per spec §7.2, prepend a blockquote callout at the **very top** of the file — before the `# Running custom-sam-peft on RunPod` heading. The file currently starts with the `# Running custom-sam-peft on RunPod` heading on line 1. Insert the following two lines before line 1:

```markdown
> **Faster path:** If you're comfortable with Docker, see
> [cloud/docker/README.md#runpod](../docker/README.md#runpod) — it skips
> the pip-install wait and gets you to training in one `docker run` command.

```

(Include the blank line between the blockquote and the `#` heading.) No other content in `cloud/runpod/README.md` is modified.

- [ ] **Step P3-4: Verify `cloud/runpod/README.md` still has its original heading immediately after the callout**

```bash
head -6 /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image/cloud/runpod/README.md
```

Expected:

```
> **Faster path:** If you're comfortable with Docker, see
> [cloud/docker/README.md#runpod](../docker/README.md#runpod) — it skips
> the pip-install wait and gets you to training in one `docker run` command.

# Running custom-sam-peft on RunPod
```

- [ ] **Step P3-5: Run `markdownlint` on the modified file**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && npx --yes markdownlint-cli2 cloud/runpod/README.md
```

Expected: clean.

### Task 3c: Patch `README.md`

The spec calls for two additive patches to `README.md`:

1. Add a sentence at the end of the "For RunPod" line in the Beginner section (after line 33 of the current file).
2. Add a new `#### From the prebuilt image` subsection immediately after the existing `uv sync` block in `### Quickstart`.

- [ ] **Step P3-6: Apply Patch 1 — Beginner section cross-link**

The current line 33 in `README.md` reads:

```
For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).
```

Replace it with (note: two sentences on two separate lines as specified in §7.3 Patch 1):

```
For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).
Already on a GPU pod? Skip the pip-install wait — see [cloud/docker/README.md](cloud/docker/README.md).
```

- [ ] **Step P3-7: Apply Patch 2 — Advanced/Quickstart Docker subsection**

Find the end of the existing `uv sync --all-extras` code block in `### Quickstart`. The block currently ends with the closing triple-backtick of the Quickstart bash block (around line 51 of the current file, after `uv run custom-sam-peft train --config configs/examples/coco_bbox_qlora.yaml`).

Insert the following new subsection immediately after the closing triple-backtick of the Quickstart code block, before `### CLI`:

````markdown

#### From the prebuilt image (no local Python install required)

```bash
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/nguyenjus/custom-sam-peft:latest \
  --help
```

See [cloud/docker/README.md](cloud/docker/README.md) for the full CLI and Jupyter usage.
````

- [ ] **Step P3-8: Verify the two patches are present and correct**

```bash
grep -n "Already on a GPU pod\|From the prebuilt image" /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image/README.md
```

Expected: both strings appear, each on exactly one line.

- [ ] **Step P3-9: Run `markdownlint` on the modified README**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && npx --yes markdownlint-cli2 README.md
```

Expected: clean (or at most the same markdownlint warnings that existed before the patch — do not introduce new ones).

### Task 3d: Commit Phase 3 files

- [ ] **Step P3-10: Commit all documentation changes**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && git add cloud/docker/README.md cloud/runpod/README.md README.md && git commit -m "docs: add Docker walkthrough + cross-links in README and RunPod guide (#34)"
```

---

## Phase 4: Open the PR

**Model/effort:** sonnet / medium.
**Parallel:** No. **Depends on:** Phases 1, 2, and 3 all committed.
**Spec:** §11 (file layout), §10 (one-time first-publish setup — note in PR body).

**Goal:** Open a draft PR linking issue #34, listing the new files, and noting the post-merge manual steps.

- [ ] **Step P4-1: Verify all spec files are committed**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && git status
```

Expected: working tree is clean. All files from spec §11 are committed:
- `Dockerfile` (NEW)
- `.dockerignore` (NEW)
- `.github/workflows/docker.yml` (NEW)
- `cloud/docker/README.md` (NEW)
- `cloud/runpod/README.md` (MODIFIED)
- `README.md` (MODIFIED)
- `pyproject.toml` (MODIFIED)
- `uv.lock` (MODIFIED)

- [ ] **Step P4-2: Confirm `lint-hygiene` CI would pass on the changed files**

Run the checks that `lint-hygiene` in `ci.yml` runs, scoped to the files changed in this branch:

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image
npx --yes markdownlint-cli2 cloud/docker/README.md cloud/runpod/README.md README.md
uv run --with yamllint yamllint .github/workflows/docker.yml
actionlint .github/workflows/docker.yml 2>/dev/null || ./actionlint .github/workflows/docker.yml
```

Expected: all three checks are clean. Fix any issues before opening the PR.

- [ ] **Step P4-3: Push the branch**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/spec-publish-docker-image && git push -u origin spec/publish-docker-image
```

- [ ] **Step P4-4: Open the draft PR**

```bash
gh pr create \
  --assignee @me \
  --title "feat: publish Docker image to GHCR on semver tag push (#34)" \
  --body "$(cat <<'EOF'
## Summary

- Adds `Dockerfile` + `.dockerignore` for a CUDA-enabled `custom-sam-peft` image on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`.
- Adds `.github/workflows/docker.yml`: build-load → smoke test (`--help` + `doctor --json`) → push to `ghcr.io/nguyenjus/custom-sam-peft` on semver tag push only.
- Adds `cloud/docker/README.md`: per-provider Docker walkthrough (RunPod, Vast.ai, Lambda Labs/generic).
- Patches `cloud/runpod/README.md` (Docker fast-path callout) and `README.md` (Beginner cross-link + Advanced Quickstart subsection).
- Adds `jupyter = ["jupyterlab>=4"]` to `pyproject.toml` optional-dependencies; regenerates `uv.lock`.

**Spec:** `docs/superpowers/specs/2026-05-21-publish-docker-image-design.md`
**Plan:** `docs/superpowers/plans/2026-05-21-publish-docker-image.md`
**Closes:** #34

## Post-merge manual steps (not automated)

These two steps must be performed by the operator after the first semver tag push triggers a successful image publish:

1. **Flip GHCR package visibility to public.** GitHub Packages → `custom-sam-peft` → Package settings → Change visibility → Public. GHCR packages default to private on first publish.
2. **Verify source auto-link.** Confirm the `org.opencontainers.image.source` label caused GitHub to auto-link the package to `NguyenJus/custom-sam-peft` (visible on the repo sidebar).
Close issue #34 once the tag publishes successfully and GHCR visibility is public.

## Test plan

- [ ] `lint-hygiene` CI passes (markdownlint, yamllint, actionlint on new files).
- [ ] `lock-check` CI passes (`uv.lock` consistent with `pyproject.toml`).
- [ ] Local `docker buildx build --load .` succeeds and `docker run --rm <tag> --help` + `doctor --json` exit 0.
- [ ] Workflow SHA pins are all 40-hex-character full commit SHAs with version comments.
- [ ] `dev` extras absent from image (`import pytest` fails inside container).
- [ ] `jupyter` extras present in image (`import jupyterlab` succeeds inside container).
EOF
  )" \
  --draft
```

- [ ] **Step P4-5: Note PR URL and watch CI**

After the PR is created, watch for CI to run (the `lint-hygiene` and `lock-check` jobs should trigger on the PR). Note: the `docker.yml` workflow will NOT run on a PR push — it only fires on semver tag pushes. This is expected and intentional.

---

## Phase 5: Post-merge manual steps (not part of the PR)

**Owner:** Operator (Justin Nguyen). **Not automated.**
**Spec:** §10 (one-time first-publish setup).

These steps are performed after the PR is merged and a semver tag is pushed to trigger the first image publish.

- [ ] **Step P5-1: Push a semver tag to trigger the first publish**

After the PR is merged to `main`:

```bash
git checkout main && git pull origin main
git tag v0.7.0   # or whatever the next semver is
git push origin v0.7.0
```

Watch the **Docker** workflow in GitHub Actions → Actions tab → Docker. The workflow should:
1. Build the image.
2. Run smoke tests (`--help` and `doctor --json` both exit 0).
3. Push to `ghcr.io/nguyenjus/custom-sam-peft` with tags `0.7.0`, `0.7`, and `latest`.

- [ ] **Step P5-2: Flip GHCR package visibility to public**

GitHub → your profile → Packages → `custom-sam-peft` → Package settings → Change visibility → Public.

Required because GHCR packages default to private on first publish. Without this step, `docker pull ghcr.io/nguyenjus/custom-sam-peft:...` fails for unauthenticated users.

- [ ] **Step P5-3: Verify source auto-link**

Confirm that the `org.opencontainers.image.source` label (set in the Dockerfile's `LABEL` block) has caused GitHub to auto-link the package to the `NguyenJus/custom-sam-peft` repository. The package should appear in the repo's sidebar under "Packages".

- [ ] **Step P5-4: Close issue #34**

```bash
gh issue close 34 --comment "Docker image published to ghcr.io/nguyenjus/custom-sam-peft. Visibility flipped to public. Source auto-link verified."
```

---

## Definition of done

All items below must be checked before the PR can be marked ready for review:

- [ ] `pyproject.toml` has `jupyter = ["jupyterlab>=4"]` under `[project.optional-dependencies]`.
- [ ] `uv.lock` is consistent with the updated `pyproject.toml` (running `uv lock --check` exits 0).
- [ ] `Dockerfile` exists at the repo root and matches spec §3 verbatim (two-stage `uv sync`, correct `WORKDIR` sequence, `org.opencontainers.image.source` label, `EXPOSE 8888`, `ENTRYPOINT ["custom-sam-peft"]`, `CMD ["--help"]`).
- [ ] `.dockerignore` exists at the repo root and matches spec §4 verbatim.
- [ ] `.github/workflows/docker.yml` exists with all five action SHA pins resolved to full 40-char commit SHAs with version comments.
- [ ] `cloud/docker/README.md` exists and covers all six sections from spec §7.1 (What's in the image, Pick a tag, Mount convention, CLI mode, Jupyter mode, Per-provider notes with RunPod/Vast.ai/generic anchors).
- [ ] `cloud/runpod/README.md` has the blockquote callout prepended at the very top.
- [ ] `README.md` has the Docker cross-link in the Beginner section and the `#### From the prebuilt image` subsection in `### Quickstart`.
- [ ] `markdownlint` passes on all new/modified Markdown files.
- [ ] `yamllint` and `actionlint` pass on `.github/workflows/docker.yml`.
- [ ] Local build + smoke test (`docker run --rm <tag> --help` and `doctor --json`) exits 0 — OR a documented reason why local verification was skipped (e.g. Docker not available on the build host).
- [ ] PR body links issue #34, spec path, and plan path; lists post-merge manual steps.

---

## Self-review

**1. Spec coverage:** Every file in spec §11 maps to at least one plan phase:
- `Dockerfile` → Phase 1, Task 1a.
- `.dockerignore` → Phase 1, Task 1b.
- `.github/workflows/docker.yml` → Phase 2.
- `cloud/docker/README.md` → Phase 3, Task 3a.
- `cloud/runpod/README.md` → Phase 3, Task 3b.
- `README.md` → Phase 3, Task 3c.
- `pyproject.toml` → Phase 0.
- `uv.lock` → Phase 0.

Spec §5.2 intentional SHA placeholders → resolved in Phase 2, Step P2-1 (instructions for the implementer to look up and pin five actions at PR time). Spec §10 (one-time GHCR flip) → Phase 5, Step P5-2. Spec §9 (Colab T4 dry-run) → dropped; CI smoke test (`--help` + `doctor --json` in `docker.yml`) verifies the entrypoint and module imports; no Docker-vs-venv GPU failure mode warrants a manual gate.

**2. Placeholder scan:** No "TBD", "TODO", "implement later", or "fill in details" language. The only remaining `<sha>` references are in Step P2-2 as explicit placeholders the implementer fills in — they are accompanied by exact instructions on how to resolve them (Step P2-1). These are not plan failures; they are intentional delegation points requiring runtime data (current upstream SHAs) that cannot be known at plan-write time.

**3. Type consistency:** `uv sync` flag set is identical in the plan and the spec: `--extra qlora --extra tensorboard --extra wandb --extra jupyter`. The `--no-install-project` flag is included only in the first `uv sync` stage (deps-only), matching the spec exactly.

**4. Parallelism:** Phases 1, 2, 3 are explicitly called out as parallelizable and file-disjoint. Phase 0 is called out as the serial blocker. The dependency graph is drawn explicitly at the top of the plan.

**5. Post-merge steps:** The GHCR visibility flip (spec §10) and source auto-link verification are placed in Phase 5 as operator work — not in the PR's CI. The spec §9 Colab T4 dry-run has been dropped; CI smoke tests (`--help` + `doctor --json`) are sufficient to verify the image entrypoint, and GPU correctness is covered by the forthcoming CI GPU testing work. Issue #34 closes after the tag publishes successfully and GHCR visibility is public.
