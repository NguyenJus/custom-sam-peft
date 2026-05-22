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
<!-- markdownlint-disable-next-line MD013 -->
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

```text
train --config /workspace/config.yaml
```

**Jupyter mode** (Container Start Command — override entrypoint in the
RunPod template's "Docker Command" field):

```text
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

For a step-by-step RunPod walkthrough without Docker (from-source install),
see [`cloud/runpod/README.md`](../runpod/README.md).

### Vast.ai {#vastai}

<!-- markdownlint-disable MD031 -->
1. In the **Create Instance** form, set the **Image** field to
   `ghcr.io/nguyenjus/custom-sam-peft:vX.Y.Z`.
2. Set **Launch Mode** to "Run" (not SSH only).
3. In the **On-start script** field, add:
   ```bash
   export HF_TOKEN=<your-hf-token>
   ```
4. Set the disk mount path to `/workspace`.
5. Add environment variable `HF_TOKEN` with your Hugging Face token.
<!-- markdownlint-enable MD031 -->

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
