# Notebooks

## `custom_sam_peft_train.ipynb` — Colab / RunPod training notebook

The canonical beginner notebook for fine-tuning SAM 3.1 with LoRA / QLoRA on a single consumer GPU.
Runs end-to-end in Google Colab (T4 or better) or on a RunPod GPU pod.

### What it does

| Cell | Purpose |
| --- | --- |
| **SETUP** | Installs `custom-sam-peft[qlora,tensorboard]` from GitHub, detects Colab vs RunPod, resolves a HuggingFace token if needed. |
| **FORM** | Colab form widgets for dataset path, data format (`coco` or `hf`), and run name. |
| **GENERATE** | Picks a VRAM-appropriate LoRA / QLoRA preset, merges it with a template config, writes `config.yaml`, and runs `custom-sam-peft run`. |
| **RESULTS** | Renders `summary.md` and sample overlays inline; offers a one-click download zip. |

### CLI surface (v0.7.0)

The notebook uses the `run` alias, which is equivalent to `train --eval --export`:

```bash
custom-sam-peft run --config config.yaml          # train + eval + export (one shot)
custom-sam-peft train --config config.yaml        # train only
custom-sam-peft train --config config.yaml --eval # train + eval
custom-sam-peft train --config config.yaml --eval --export  # same as run
```

### Key config fields (v0.7.0 schema)

The schema is documented fully in `docs/config-schema.md`. Common fields used by the notebook:

| Field | Example | Notes |
| --- | --- | --- |
| `run.name` | `"my-run"` | Unique name; becomes the run-directory prefix. |
| `peft.method` | `"lora"` or `"qlora"` | Auto-selected by VRAM tier preset. |
| `peft.r` | `8`, `16`, `32` | LoRA rank; preset picks based on VRAM. |
| `train.learning_rate` | `1.0e-4` | Peak LR after warm-up. (v0.7.0 rename from `lr`.) |
| `train.epochs` | `10` | Full passes through the training data. |
| `train.batch_size` | `1`–`4` | Micro-batch size; preset adjusts for VRAM. |
| `train.grad_accum_steps` | `2`–`16` | Steps before one optimizer update; preset adjusts. |
| `data.format` | `"coco"` or `"hf"` | Dataset format. |
| `data.prompt_mode` | `"text"` | Text-class prompting (v0 only; `"bbox"` is planned). |

> **v0.7.0 rename note:** the learning-rate field is now `train.learning_rate` (was `lr` in v0.6 and earlier). Update any saved configs before running with v0.7.0.

### VRAM presets

The GENERATE cell calls `pick_preset()` to auto-select PEFT method and batch configuration:

| VRAM tier | method | rank | batch | grad accum |
| --- | --- | --- | --- | --- |
| < 12 GB | qlora | 8 | 1 | 16 |
| 12–24 GB | qlora | 16 | 1 | 8 |
| 24–48 GB | lora | 16 | 2 | 4 |
| ≥ 48 GB | lora | 32 | 4 | 2 |

### Output cells

Output cells are intentionally cleared. The notebook requires a GPU to run end-to-end (SAM 3.1 forward pass), so outputs are not pre-populated. Run the notebook top-to-bottom on a Colab GPU runtime to generate fresh outputs.

---

## `colab_gpu_tests.ipynb` — GPU integration test runner

Runs the GPU-gated integration test suite against a real SAM 3.1 checkpoint.
Intended for CI validation and post-merge smoke tests; not a user-facing training notebook.
