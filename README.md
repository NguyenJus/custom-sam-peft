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
uv sync --all-extras --group dev

# Sanity check the CLI
uv run esam3 --help
uv run esam3 doctor

# Run the (currently stubbed) train command against an example config
uv run esam3 train --config configs/examples/coco_bbox_qlora.yaml
```

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

## License

Apache-2.0. See `LICENSE`.
