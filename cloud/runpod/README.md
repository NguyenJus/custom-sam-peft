# Running custom-sam-peft on RunPod

A step-by-step guide for non-technical users. If you're on Colab, use the
notebook badge in the [main README](../../README.md) — this file is the
RunPod equivalent.

## 1. Sign up

Go to [runpod.io](https://runpod.io) and create an account. Pay-as-you-go is fine; spot pricing is cheaper but can be interrupted.

## 2. Pick a GPU

**A40 is the recommended entry tier** — 48 GB VRAM lands in the LoRA preset, good $/VRAM ratio. L4, RTX 4090, and A100 also work. Anything with ≥ 12 GB VRAM can run the QLoRA preset.

## 3. Deploy a stock RunPod PyTorch template

We deliberately do **not** publish or maintain a custom RunPod image — see [issue #34](https://github.com/NguyenJus/custom-sam-peft/issues/34). Use the stock **"RunPod PyTorch 2.x"** template from the Templates page. Templates → Deploy → pick the GPU from step 2 → Deploy.

## 4. Set `HF_TOKEN`

Pod → Edit → Environment Variables → add `HF_TOKEN` (your Hugging Face read-access token for gated `facebook/sam3.1`). **Or** skip this if you've mounted a network volume that contains `models/sam3.1/sam3.1_multiplex.pt` — `custom-sam-peft` will detect the local file and skip HF auth.

## 5. Open Jupyter Lab

Click the pod's **Connect** button → Jupyter Lab.

## 6. Upload `notebooks/custom_sam_peft_train.ipynb`

Two options: drag and drop the notebook file into the Jupyter file browser, or in Jupyter, File → Open from URL → paste the raw GitHub URL for `notebooks/custom_sam_peft_train.ipynb`.

## 7. Click Run All

Same beginner flow as Colab — fill in dataset path, format, and run name in the FORM cell, then Runtime → Run All.

## Data upload

**Small dataset (≤ 1 GB):** drag-and-drop into the Jupyter file browser. **Large dataset:** RunPod network volume — one-time upload, persists across pods. **HF dataset:** easiest — paste the dataset id into the FORM cell; no upload needed.

## What you get back

Every run writes a `runs/<id>/` directory with:

- `summary.md` — headline metric, run timing, hardware, sample overlays.
- `samples/*.png` — up to 6 best / median / worst predictions.
- `adapter/` — the LoRA / QLoRA adapter weights.
- `metrics.json` — raw eval numbers.

Download with `scp` or zip + download from Jupyter.
