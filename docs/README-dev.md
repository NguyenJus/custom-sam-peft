# Developer guide

[← back to README](README.md)

This file covers the developer-facing surface of
`Efficient-SAM3-Finetuning`: dev loop, GPU test automation, and repo
layout. End-user documentation lives in [`README.md`](README.md).

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/esam3
uv run pytest
```

GPU smoke test (requires CUDA + SAM 3.1 weights):

```bash
uv run pytest -m gpu
```

## GPU test automation

GPU-gated tests run on a free Colab T4 (no local GPU required). The
Colab notebook lives at
[`notebooks/colab_gpu_tests.ipynb`](notebooks/colab_gpu_tests.ipynb).

A per-branch **Open in Colab** badge is injected into the body of every
pull request by `.github/workflows/pr-colab-badge.yml` — open a PR and
the badge will point at the notebook on that PR's branch.

In Colab Secrets, set `HF_TOKEN` (Hugging Face token with read access
to gated `facebook/sam3.1`). Choose a T4 (or better) runtime, then Run
All. See
[`docs/superpowers/specs/2026-05-17-peft-qlora-design.md`](docs/superpowers/specs/2026-05-17-peft-qlora-design.md)
§11 for the test catalog and
[`docs/testing/gpu-test-policy.md`](docs/testing/gpu-test-policy.md)
for the inspection/release tier breakdown.

## Repo layout

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and data flow.
