# efficient-sam3-finetuning

Parameter-efficient finetuning of [SAM3.1](https://huggingface.co/facebook/sam3.1)
on niche image instance-segmentation datasets — runnable on a single
consumer GPU.

> **Status:** v0 scaffolding only. The CLI and library surfaces exist;
> training/eval/data-loading bodies land in subsequent specs. See
> `docs/superpowers/specs/` for design and `docs/superpowers/plans/`
> for the build sequence.

## Quickstart

```bash
# Install
uv sync --all-extras

# Sanity check the CLI
uv run esam3 --help
uv run esam3 doctor

# Run the (currently stubbed) train command against an example config
uv run esam3 train --config configs/examples/coco_bbox_qlora.yaml
```

## CLI

| Command | Status |
|---|---|
| `esam3 train --config CONFIG [--override key=val]... [--resume PATH] [-v]` | Functional |
| `esam3 eval --config CONFIG --checkpoint PATH [--split val\|test] [--output PATH] [--save-predictions]` | Functional (LoRA adapters only) |
| `esam3 export --checkpoint PATH [--merge] [--output PATH] [--config PATH]` | Functional |
| `esam3 init [--template coco-text-lora\|coco-text-qlora] [--output PATH] [--force]` | Functional |
| `esam3 doctor [--weights-path PATH] [--json]` | Functional |

`coco-bbox` and `hf-text` init templates are deferred (see `logs/TODO.md`).

## What's supported in v0

| | v0 | Deferred |
|---|---|---|
| Model | SAM3.1 | SAM3 |
| Prompts | text, bounding boxes | points, masks |
| Data | static images, COCO + HF datasets | video |
| Output | instance segmentation | semantic segmentation |
| Distribution | single GPU | Ray Train, Argo workflows |
| PEFT | LoRA, QLoRA | other PEFT methods |
| Tracking | TensorBoard, W&B, none | — |

## v0 Training scope

v0 trains **text-prompts only**. Ground-truth bounding boxes are used as a curriculum hint during training (increasing probability of box-only forward passes), not as a primary prompt. `prompt_mode='bbox'` is rejected at training time (see logs/TODO.md for deferred bbox-prompt-training spec).

For testing: run `pytest -m integration` for end-to-end stub tests, or `pytest -m gpu` if you have a CUDA GPU and a local SAM 3.1 checkpoint.

## Repo layout

See `ARCHITECTURE.md` for the module map and data flow.

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src/esam3
uv run pytest
```

GPU smoke test (requires CUDA + SAM3.1 weights):

```bash
uv run pytest -m gpu
```

### GPU test automation

Run the GPU-gated tests on a free Colab T4 (no local GPU required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenJus/Efficient-SAM3-Finetuning/blob/main/notebooks/colab_gpu_tests.ipynb)

In Colab Secrets, set `HF_TOKEN` (Hugging Face token with read access to
gated `facebook/sam3.1`), plus `GH_TOKEN` (GitHub fine-grained PAT with
`Contents: Read`) **if this repo is private**. Choose a T4 (or better)
runtime, then Run All. See [`docs/superpowers/specs/2026-05-17-peft-qlora-design.md`](docs/superpowers/specs/2026-05-17-peft-qlora-design.md) §11 for details.

## License

Apache-2.0. See `LICENSE`.
