# spec/publish-docker-image — Publish Docker image to GHCR (issue #34)

**Status:** Draft (2026-05-21)
**Tracking:** [#34](https://github.com/NguyenJus/custom-sam-peft/issues/34) — *Containerize esam3 (publish Docker image)*
**Scope:** Build and publish a CUDA-enabled Docker image of `custom-sam-peft` to GHCR (`ghcr.io/nguyenjus/custom-sam-peft`), triggered on semver tag pushes. Adds a build-and-push workflow, a per-provider Docker walkthrough, cross-links from the existing RunPod walkthrough and main README, and a new `jupyter` optional extras group. No source code in `src/` is touched.

**Builds on:**
[`2026-05-18-simplify-ux-design.md`](2026-05-18-simplify-ux-design.md) (§10 lists this as a follow-up; §7.1 notebook SETUP cell's `pip install git+…` is the alternative path this image replaces for GPU-pod users);
the existing CI conventions established in `.github/workflows/ci.yml` (SHA-pinned actions, draft-PR skip, concurrency groups, `lint-hygiene` checks).

---

## 1. Goals & v0 Scope

### 1.1 Motivation

Issue #34 was deferred from the simplify-UX brainstorming (#25) with this rationale (§10 of `2026-05-18-simplify-ux-design.md`):

> "Would replace the notebook SETUP cell's `pip install git+…` with a single `docker pull`. Doable but adds release surface; v0 ships from-source install."

Two things changed: the repo flipped public in the public-flip PR (anchored at `v0.5.0`), and the first GPU-tested release (`v0.6.0`) shipped. The image is now worth the release surface because there is a stable, validated tagged version to publish.

### 1.2 In scope

| Deliverable | Where |
| --- | --- |
| Dockerfile | `Dockerfile` (repo root) |
| `.dockerignore` | `.dockerignore` (repo root) |
| Build & push workflow | `.github/workflows/docker.yml` |
| Per-provider Docker walkthrough | `cloud/docker/README.md` (new) |
| Cross-link from RunPod walkthrough | `cloud/runpod/README.md` (top paragraph added) |
| Beginner cross-link in main README | `README.md` (Beginner section + Advanced/Quickstart subsection) |
| `jupyter` extras group | `pyproject.toml` (+1 entry under `[project.optional-dependencies]`) |
| Lockfile refresh | `uv.lock` (regenerated via `uv lock` after the extras group addition) |

### 1.3 Out of scope

| Item | Reason / follow-up |
| --- | --- |
| Switching `notebooks/custom_sam_peft_train.ipynb` to use the image | Image is an alternative path, not a replacement. Colab cannot pull custom Docker images; switching would regress Colab users. |
| RunPod template, Modal app, SageMaker container, Lambda Labs preset | Tracked in issue #35. |
| Multi-arch (`arm64`) build | All NVIDIA GPU pods are `amd64`. |
| SBOM / provenance attestation | No consumer asking; cheap to add later if needed. |
| Baking SAM 3.1 weights into the image | Gated weights + ≈6 GB; pulled at first run via `HF_TOKEN` (existing flow). |
| Renovate config for base image tag bumps | Manual; driven by torch compatibility. |
| Automated GPU testing of the published image | Separately tracked CI GPU testing work will handle it. |

---

## 2. Architectural Approach

The image packages the `custom-sam-peft` CLI and all runtime extras (LoRA/QLoRA, TensorBoard, W&B, Jupyter) into a `pytorch/pytorch` base image so that a GPU pod user can skip the `pip install git+…` step and go straight to training.

Two usage modes, one image:

- **CLI mode** (default) — `docker run … ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z train --config …`. The `ENTRYPOINT` is the `custom-sam-peft` CLI; the default `CMD` is `["--help"]`.
- **Jupyter mode** — `docker run … --entrypoint jupyter … lab --ip=0.0.0.0 --no-browser --allow-root`. Users who prefer the notebook flow can override the entrypoint. Port `8888` is exposed.

The image is **stateless**: all user data (datasets, configs, run artefacts, model weights, HF cache) lives under a single host-mounted volume at `/workspace`. The image's `WORKDIR` is `/workspace` so relative paths in configs and notebooks resolve naturally.

Key constraints:
> - The image is built and pushed **only on semver tag pushes** (`vX.Y.Z`). No PR builds, no push-to-main builds. This keeps GHCR clean and avoids publishing unverified builds.
> - The workflow runs a **smoke test before pushing**: `--help` and `doctor --json` must exit 0. A broken image is never published.
> - The `dev` extras group is **never installed** in the image (`uv sync --extra qlora --extra tensorboard --extra wandb --extra jupyter`). `--all-extras` would pull `dev`, adding test/lint tools to the published image.

---

## 3. Dockerfile

The canonical Dockerfile is committed verbatim. The planner will resolve exact SHA pins for the uv copy (`COPY --from=ghcr.io/astral-sh/uv:0.5.11`) before the implementation PR lands; the version string `0.5.11` is already pinned here because it matches the uv release that introduced reliable `UV_LINK_MODE=copy` behavior on Docker layer caches.

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

**Layer-cache rationale.** The two-stage `uv sync` (deps-only first, then full install with source) lets Docker reuse the heavy dependency layer when only source files change. The `--mount=type=cache` keeps the uv download cache warm across rebuilds. `UV_PYTHON_DOWNLOADS=never` prevents uv from silently pulling a Python version different from the one baked into the base image.

**`WORKDIR /opt/custom-sam-peft` → `/workspace` switch.** The package is installed under `/opt/custom-sam-peft`; the runtime working directory is `/workspace` (the user's mount point). This is intentional: `custom-sam-peft init` writes `config.yaml` relative to `cwd`, which resolves inside the user's volume.

---

## 4. `.dockerignore`

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

Excludes dev artefacts, test data, local run outputs, and notebooks (which would otherwise add multi-MB notebook checkpoints to the build context). The `sam3` git-dependency is fetched from source by uv at build time, not copied from the build context.

---

## 5. Build & push workflow (`.github/workflows/docker.yml`)

### 5.1 Trigger and permissions

The workflow fires **only on tag push** matching `v*`. No PR builds, no push-to-main builds. Using `secrets.GITHUB_TOKEN` for GHCR auth; the token is granted `packages: write` by the job's permissions block.

### 5.2 Workflow YAML

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
      - uses: actions/checkout@<sha>           # planner pins SHA for v4
      - uses: docker/setup-buildx-action@<sha> # planner pins SHA for v3
      - uses: docker/login-action@<sha>        # planner pins SHA for v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - id: meta
        uses: docker/metadata-action@<sha>     # planner pins SHA for v5
        with:
          images: ghcr.io/nguyenjus/custom-sam-peft
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}},enable=${{ !startsWith(github.ref, 'refs/tags/v0.') }}
            type=raw,value=latest

      - uses: docker/build-push-action@<sha>   # planner pins SHA for v6
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

      - uses: docker/build-push-action@<sha>   # planner pins SHA for v6 (same action, second use)
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
```

**Intentional placeholders:** Every `<sha>` in the workflow YAML is a placeholder for the plan-writer / implementer to resolve against the latest pinned SHAs for each action. The action versions in comments (`v4`, `v3`, `v5`, `v6`) give the target major version to pin. These are the only intentional placeholders in the spec; all other decisions are fully resolved.

**Smoke-test rationale.** The build-load → smoke → build-push pattern lets the smoke test run against the image that will be pushed (same cache state, no rebuild), rather than building twice. The second `build-push-action` step replays from cache (`cache-from: type=gha`), so it costs only the push network time.

### 5.3 Tag derivation

| Tag push event | Tags published |
| --- | --- |
| `v0.6.1` | `0.6.1`, `0.6`, `latest` |
| `v1.2.3` | `1.2.3`, `1.2`, `1`, `latest` |

The major-version tag (`1`, `2`, …) is suppressed for pre-1.0 releases (`v0.*`) via the `enable=${{ !startsWith(github.ref, 'refs/tags/v0.') }}` condition on the `type=semver,pattern={{major}}` tag rule.

---

## 6. Mount convention and usage

All user data lives under a single host-mounted volume at `/workspace`. The documented subdir layout:

| Subdirectory | Contents |
| --- | --- |
| `data/` | Training datasets (COCO, HF cache, etc.) |
| `runs/` | Run output (`adapter/`, `metrics.json`, `summary.md`, `samples/`) |
| `models/` | SAM 3.1 checkpoint (`models/sam3.1/sam3.1_multiplex.pt`) |
| `.cache/huggingface/` | HF Hub download cache (mirrors `HF_HOME=/workspace/.cache/huggingface`) |

`HF_HOME` is set in the image to `/workspace/.cache/huggingface` so that weight downloads land inside the user's mounted volume and survive container restarts.

---

## 7. Documentation changes

### 7.1 `cloud/docker/README.md` (new)

A per-provider Docker walkthrough with these sections (in order):

1. **What's in the image** — package versions, extras installed, `ENTRYPOINT`, mount point.
2. **Pick a tag** — links to the GHCR package page; recommends pinning to a semver tag rather than `latest`.
3. **Mount convention** — the `/workspace` layout table from §6.
4. **CLI mode** — the `docker run` snippet for training:

   ```bash
   docker run --gpus all --rm \
     -v $PWD:/workspace \
     -e HF_TOKEN=$HF_TOKEN \
     ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z \
     train --config /workspace/config.yaml
   ```

5. **Jupyter mode** — the `docker run` snippet for launching JupyterLab:

   ```bash
   docker run --gpus all --rm -p 8888:8888 \
     -v $PWD:/workspace \
     -e HF_TOKEN=$HF_TOKEN \
     --entrypoint jupyter \
     ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z \
     lab --ip=0.0.0.0 --no-browser --allow-root
   ```

6. **Per-provider notes** — three subsections, each anchored for direct linking:
   - **RunPod** (`#runpod`): Custom Template fields — Container Image (`ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z`), Container Disk ≥ 20 GB, Volume Mount Path `/workspace`, Expose HTTP Port `8888`, Container Start Command for CLI mode and Jupyter mode, env var `HF_TOKEN`.
   - **Vast.ai** (`#vastai`): image field, launch mode, on-start script, env var `HF_TOKEN`.
   - **Lambda Labs / generic** (`#generic`): use the snippets directly; confirm `nvidia-container-toolkit` is installed.

### 7.2 `cloud/runpod/README.md` (cross-link added)

Prepend a single blockquote callout at the very top of the file — before the `# Running custom-sam-peft on RunPod` heading — pointing to `cloud/docker/README.md#runpod` as the faster path for users who are already familiar with Docker:

```markdown
> **Faster path:** If you're comfortable with Docker, see
> [cloud/docker/README.md#runpod](../docker/README.md#runpod) — it skips
> the pip-install wait and gets you to training in one `docker run` command.
```

No other content in `cloud/runpod/README.md` is modified.

### 7.3 `README.md` (two additive patches)

**Patch 1 — Beginner section.** Add one sentence at the end of the "For RunPod" line in the existing Beginner section:

Before (current line 33):
```
For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).
```

After:
```
For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).
Already on a GPU pod? Skip the pip-install wait — see [cloud/docker/README.md](cloud/docker/README.md).
```

**Patch 2 — Advanced > Quickstart subsection.** Add a new subsection immediately after the existing `uv sync` block in the `### Quickstart` section:

```markdown
#### From the prebuilt image (no local Python install required)

```bash
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/nguyenjus/custom-sam-peft:latest \
  --help
```

See [cloud/docker/README.md](cloud/docker/README.md) for the full CLI and Jupyter usage.
```

---

## 8. `pyproject.toml` change

Add one line under `[project.optional-dependencies]`:

```toml
jupyter = ["jupyterlab>=4"]
```

The `jupyter` group is listed alongside `wandb`, `qlora`, and `tensorboard`. The `dev` group is unchanged and is never installed in the image.

After adding the group, the plan-writer runs `uv lock` to refresh `uv.lock` for the new extra. The updated `uv.lock` is committed alongside `pyproject.toml`.

---

## 9. Verification matrix

| Check | Where | Trigger |
| --- | --- | --- |
| Dockerfile builds cleanly | CI (`docker.yml`) | tag push only |
| `custom-sam-peft --help` exits 0 inside container | CI smoke test | tag push, before publish |
| `custom-sam-peft doctor --json` exits 0 inside container | CI smoke test | tag push, before publish |
| `actionlint` / `yamllint` / `markdownlint` pass on new files | CI (`lint-hygiene`, existing) | every PR |
| GHCR package visibility set to public | Manual one-time | after first publish |
| `org.opencontainers.image.source` label auto-linked package to repo | Manual one-time | after first publish |

The CI smoke test runs `--help` and `doctor --json` without `--gpus` (GitHub-hosted runners have no GPU). This verifies that the CLI is importable and the package graph resolves correctly inside the image. There is no manual GPU verification of the image; CI smoke is the only pre-publish bar. GPU correctness is covered by the venv tests in `ci.yml` and by the forthcoming CI GPU testing in a separate track.

---

## 10. One-time first-publish setup

These two steps are performed once by the operator after the first tagged release publishes the image. They are **not automated** (they require GitHub UI actions that the workflow cannot perform):

1. **Flip GHCR package visibility to public.** GitHub Packages → `custom-sam-peft` → Package settings → Change visibility → Public. Required because GHCR packages default to private on first publish.
2. **Verify source link.** Confirm that the `org.opencontainers.image.source` label (set in the Dockerfile) has caused GitHub to auto-link the package to the `NguyenJus/custom-sam-peft` repo. This makes the package visible on the repo's sidebar.

---

## 11. File layout

```
Dockerfile                                    NEW
.dockerignore                                 NEW
.github/workflows/docker.yml                  NEW
cloud/docker/README.md                        NEW
cloud/runpod/README.md                        TOUCHED (blockquote callout prepended at top)
README.md                                     TOUCHED (Beginner + Advanced/Quickstart patches)
pyproject.toml                                TOUCHED (+1 extras group: jupyter)
uv.lock                                       REGENERATED via `uv lock`
```

No deletions, no moves. No source code in `src/` is touched. No tests in `tests/` added or modified. `cloud/docker/` is a new directory.

---

## 12. Testing strategy

This spec ships no new Python source; all testing is at the CI / manual level.

| Test surface | Method | Notes |
| --- | --- | --- |
| Dockerfile syntax and build | `docker.yml` CI | Build-load step on tag push |
| CLI smoke inside container | `docker run … --help` + `doctor --json` | Both must exit 0 before publish |
| Workflow YAML correctness | `actionlint` in `lint-hygiene` | Catches action version problems, missing permissions |
| Markdown quality | `markdownlint` in `lint-hygiene` | `cloud/docker/README.md` + `README.md` patches |

There are no new Python modules, so no unit tests are added. The 80% coverage gate in `pyproject.toml` is unaffected.

---

## 13. Out of scope (filed as follow-up issues)

| Issue | Title | Why deferred |
| --- | --- | --- |
| [#35](https://github.com/NguyenJus/custom-sam-peft/issues/35) | AWS SageMaker / Lambda Labs / Modal targets | This spec covers Docker + GHCR; provider-managed runtimes (SageMaker, Modal) are a separate surface requiring their own auth and packaging conventions. |
| *(future)* | Multi-arch (`arm64`) image | All NVIDIA GPU pods are `amd64`; no consumer need yet. |
| *(future)* | SBOM / provenance attestation (`attest-build-provenance`) | No consumer asking. One additional workflow step when needed. |
| *(future)* | Renovate/Dependabot config for base image tag bumps | Base image pin (`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`) is driven by torch compatibility; automated bumps could introduce silent regressions. Manual update policy for now. |
| *(future)* | Automated GPU CI for the published image | Tracked under the existing GPU CI testing work; extends GPU correctness coverage to the published image. |
| *(future)* | Switching `notebooks/custom_sam_peft_train.ipynb` to pull from the image | Colab cannot pull custom Docker images; the notebook `pip install git+…` path stays. A separate "image-native notebook" for GPU pods is a future deliverable if a real demand emerges. |
