# custom-sam-peft

[![CI](https://github.com/NguyenJus/custom-sam-peft/actions/workflows/ci.yml/badge.svg)](https://github.com/NguyenJus/custom-sam-peft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)

Parameter-efficient finetuning of [SAM3.1](https://huggingface.co/facebook/sam3.1)
on niche image instance-segmentation datasets — runnable on a single
consumer GPU.

> **⚠️ Work in progress — not ready to run.**
> v0.5.0 is an active development snapshot. The CLI surfaces (`train`, `eval`, `export`, `run`, `init`, `doctor`) exist and exercise real subsystems (LoRA / QLoRA adapters, W&B tracking), but the project has not been validated end-to-end on production workloads. Expect breaking changes. Use at your own risk; pin to a tagged release if you need stability.

## Beginner — train in Colab

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

## Advanced

### Quickstart

```bash
# Install
uv sync --all-extras

# Sanity check the CLI
uv run custom-sam-peft --help
uv run custom-sam-peft doctor

# Run the (currently stubbed) train command against an example config
uv run custom-sam-peft train --config configs/examples/coco_bbox_qlora.yaml
```

#### From the prebuilt image (no local Python install required)

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
| `custom-sam-peft run --config CONFIG [--resume PATH] [-v]` | Functional |
| `custom-sam-peft train --config CONFIG [--override key=val]... [--resume PATH] [-v]` | Functional |
| `custom-sam-peft eval --config CONFIG --checkpoint PATH [--split val\|test] [--output PATH] [--save-predictions]` | Functional (LoRA adapters only) |
| `custom-sam-peft export --checkpoint PATH [--merge] [--output PATH] [--config PATH]` | Functional |
| `custom-sam-peft init [--template coco-text-lora\|coco-text-qlora] [--output PATH] [--force]` | Functional |
| `custom-sam-peft doctor [--weights-path PATH] [--json]` | Functional |

(`custom-sam-peft run` is "train + eval + (optional) export + bundle in one shot"; the others are unchanged.)

`coco-bbox` and `hf-text` init templates are deferred (see `logs/TODO.md`).

#### Run inference on your images

After installing the package, point `csp predict` at a directory of images and pass class prompts:

```bash
uv run csp predict \
  --images path/to/images/ \
  --prompts "cat,dog,person" \
  --output out/
```

This produces `out/predictions.json` (COCO-flat), `out/image_id_map.json` (id → source path), and `out/run.json` (reproducibility metadata). Pass `--checkpoint path/to/adapter/` to apply a LoRA or QLoRA adapter (auto-detected); add `--visualize` to write per-image overlays. See `csp predict --help` for every flag.

### What's supported in v0

| | v0 | Deferred |
| --- | --- | --- |
| Model | SAM3.1 | SAM3 |
| Prompts | text, bounding boxes | points, masks |
| Data | static images, COCO + HF datasets | video |
| Output | instance segmentation | semantic segmentation |
| Distribution | single GPU | Ray Train, Argo workflows |
| PEFT | LoRA, QLoRA | other PEFT methods |
| Tracking | TensorBoard, W&B, none | — |

### v0 Training scope

v0 trains **text-prompts only**. Ground-truth bounding boxes are used as a curriculum hint during training (increasing probability of box-only forward passes), not as a primary prompt. `prompt_mode='bbox'` is rejected at training time (see logs/TODO.md for deferred bbox-prompt-training spec).

For testing: run `pytest -m integration` for end-to-end stub tests, or `pytest -m gpu` if you have a CUDA GPU and a local SAM 3.1 checkpoint.

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
