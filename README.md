# custom-sam-peft

[![CI](https://github.com/NguyenJus/custom-sam-peft/actions/workflows/ci.yml/badge.svg)](https://github.com/NguyenJus/custom-sam-peft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)

Parameter-efficient finetuning of [SAM3.1](https://huggingface.co/facebook/sam3.1)
on niche image instance-segmentation datasets — runnable on a single
consumer GPU.

> **⚠️ Work in progress.**
> An active development snapshot — **the code runs**, but it hasn't been validated end-to-end on production workloads. The CLI surfaces (`train`, `run`, `eval`, `predict`, `export`, `init`, `doctor`, `calibrate`) exercise real subsystems (LoRA / QLoRA adapters, TensorBoard / W&B tracking). Expect breaking changes; pin to a tagged release if you need stability.

## Train in Colab

Train a custom segmentation model in your browser via Google Colab. No local GPU setup required.

**Prerequisites:** a Hugging Face account (free) with read access to the gated `facebook/sam3.1` checkpoint, and either a custom dataset (a folder with `train/` and `val/` COCO subdirectories) or a Hugging Face dataset ID.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenJus/custom-sam-peft/blob/main/notebooks/custom_sam_peft_train.ipynb)

1. Open the notebook in Colab via the badge above.
2. In Colab Secrets, set `HF_TOKEN` (Hugging Face token with read access
   to gated `facebook/sam3.1`). If you've already downloaded the
   checkpoint to `models/sam3.1/sam3.1_multiplex.pt` (e.g. on a RunPod
   network volume), skip this step.
3. Either upload a dataset (a folder with `train/` and `val/` COCO
   subdirectories) or paste a HF dataset id, then click Runtime → Run All.

When the run finishes, scroll to the bottom of the notebook for a
summary, sample mask overlays, and a one-line download command.

For RunPod, see [cloud/runpod/README.md](cloud/runpod/README.md).
Already on a GPU pod? Skip the pip-install wait — see [cloud/docker/README.md](cloud/docker/README.md).

## Quickstart

New here? The fastest path is the interactive setup wizard: `csp init --interactive` auto-detects your COCO data paths, calibrates VRAM presets for your GPU, and writes a ready-to-train config. Then hand that config to `run`.

The CLI installs under two names — the short `csp` (used throughout this README) and `custom-sam-peft`. They're identical; use whichever you prefer.

```bash
# Install
uv sync --all-extras

# Sanity check the CLI
uv run csp --help
uv run csp doctor

# Generate a config interactively (recommended) — auto-detects your COCO data
# paths, calibrates VRAM presets, and walks you through PEFT method + key knobs
uv run csp init --interactive
# ...or non-interactively from a template:
uv run csp init --template coco-text-qlora --output config.yaml

# Train, then eval and export in one shot (recommended)
uv run csp run --config config.yaml

# Or run steps individually:
uv run csp train --config config.yaml          # train only
uv run csp train --config config.yaml --eval   # train + eval
uv run csp train --config config.yaml --eval --export  # same as `run`
```

`run --config cfg.yaml` is shorthand for `train --config cfg.yaml --eval --export`.

## Advanced

### From the prebuilt image (no local Python install required)

```bash
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/nguyenjus/custom-sam-peft:latest \
  --help
```

<!-- markdownlint-disable-next-line MD013 -->
See [cloud/docker/README.md](cloud/docker/README.md) for the full CLI and Jupyter usage.

### CLI

| Command | Status |
| --- | --- |
| `csp run --config CONFIG [--resume PATH] [-v]` | Functional — shorthand for `train --eval --export` |
| `csp train --config CONFIG [--eval] [--export] [--override key=val]... [--resume PATH] [-v]` | Functional |
| `csp eval --config CONFIG --checkpoint PATH [--split val\|test] [--export] [--output PATH] [--interactive]` | Functional (LoRA + QLoRA adapters) |
| `csp predict --images DIR --prompts "a,b,c" [--checkpoint PATH] [--output PATH] [--visualize] [--interactive]` | Functional |
| `csp export --checkpoint PATH [--merge] [--output PATH] [--config PATH]` | Functional |
| `csp init [--interactive] [--template NAME] [--preset NAME] [--output PATH] [--force]` | Functional |
| `csp calibrate --config CONFIG [--output PATH] [--force]` | Functional |
| `csp doctor [--config PATH] [--weights-path PATH] [--json]` | Functional |

Most commands accept more flags than shown — run `csp <command> --help`
for the full list.

`run --config CONFIG` is equivalent to `train --config CONFIG --eval --export`; use the individual flags when you want only some steps.

`init --interactive` launches the setup wizard (auto-detected data paths, VRAM-calibrated
presets, guided knobs); `eval` and `predict` accept `--interactive` for guided one-off runs.
`coco-bbox` and `hf-text` init templates are deferred (see `logs/TODO.md`).

#### Run inference on your images

After installing the package, point `csp predict` at a directory of images and pass class prompts:

```bash
uv run csp predict \
  --images path/to/images/ \
  --prompts "cat,dog,person" \
  --output out/
```

This produces `out/predictions.json` (COCO-flat), `out/image_id_map.json` (id → source path), and `out/run.json` (reproducibility metadata). Pass `--checkpoint path/to/adapter/` to apply a LoRA or QLoRA adapter (auto-detected); add `--visualize` to write per-image overlays. Not sure of the arguments? `csp predict --interactive` builds the command for you. See `csp predict --help` for every flag.

### What's supported in v0

| | v0 | Deferred |
| --- | --- | --- |
| Model | SAM3.1 | SAM3 |
| Data | static images, COCO + HF datasets | video |
| Output | instance segmentation | semantic segmentation |
| Distribution | single GPU | Ray Train, Argo workflows |
| PEFT | LoRA, QLoRA | other PEFT methods |
| Tracking | TensorBoard, W&B, none | — |

### Testing

Run `pytest -m integration` for end-to-end stub tests (CPU, no checkpoint needed).
GPU test tiers and automation live in [`README-dev.md`](docs/README-dev.md).

### Configuration

Every YAML config field is documented in [`docs/config-schema.md`](docs/config-schema.md). The schema covers all user-settable fields across the `run`, `model`, `data`, `peft`, `training`, `eval`, and `export` sections, with types, defaults, and layer labels (common vs. advanced).

### Repo layout

See `docs/ARCHITECTURE.md` for the module map and data flow.

## Developer setup

Dev loop, GPU test automation, and repo layout live in
[`README-dev.md`](docs/README-dev.md). See
[`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for the project's contribution
posture (solo research; forks welcome, external PRs not currently
accepted).

## License

Apache-2.0. See `LICENSE`.
