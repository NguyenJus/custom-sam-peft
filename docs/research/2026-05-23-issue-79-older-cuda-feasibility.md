# Older CUDA / Driver / Compute Capability Feasibility

> Research write-up for [issue #79](https://github.com/NguyenJus/custom-sam-peft/issues/79)
> Date: 2026-05-23

## TL;DR

**Partial support is feasible**: the LoRA-only path works on CUDA 11.8 / CC 6.0+ hardware with no
code changes; the QLoRA path is more permissive than expected — bnb ≥ 0.43 places the NF4/FP4
4-bit kernel floor at CC 6.0 (sm_60 / Pascal), matching the torch wheel floor exactly.

- **LoRA-only path**: minimum CUDA 11.8, driver 520.61.05, CC 6.0 (sm_60). Status: `partial`
  (bf16 emulated below CC 8.0 — set `model.dtype: float16` in config to avoid silent numeric
  degradation on pre-Ampere hardware).
- **QLoRA path**: minimum CUDA 11.8, driver 520.61.05, CC 6.0 (sm_60). Status: `partial`
  (same bf16 caveat; bnb LLM.int8() further requires CC 7.5 but QLoRA/NF4 does not).

Per the rubric (spec §5): both paths lower cheaply to the same floor, with one caveat (bf16
emulation) that is resolved by a config change — this lands in the **Partial** bucket.

---

## §1 — CUDA Toolkit Floor

### Probe commands and output

Both `uv pip install --dry-run` commands were run against Python 3.12 / Linux x86_64. The
`--system` flag was needed because the repo's `.venv` uses Python 3.13.

**cu118 index:**

```text
$ uv pip install --dry-run --index-url https://download.pytorch.org/whl/cu118 'torch>=2.4' --python python3.12 --system
Using Python 3.12.13 environment at: /usr
Resolved 22 packages in 1.15s
Would download 22 packages
Would install 22 packages
 + filelock==3.29.0
 + fsspec==2026.4.0
 + jinja2==3.1.6
 + markupsafe==3.0.3
 + mpmath==1.3.0
 + networkx==3.6.1
 + nvidia-cublas-cu11==11.11.3.6
 + nvidia-cuda-cupti-cu11==11.8.87
 + nvidia-cuda-nvrtc-cu11==11.8.89
 + nvidia-cuda-runtime-cu11==11.8.89
 + nvidia-cudnn-cu11==9.1.0.70
 + nvidia-cufft-cu11==10.9.0.58
 + nvidia-curand-cu11==10.3.0.86
 + nvidia-cusolver-cu11==11.4.1.48
 + nvidia-cusparse-cu11==11.7.5.86
 + nvidia-nccl-cu11==2.21.5
 + nvidia-nvtx-cu11==11.8.86
 + setuptools==70.2.0
 + sympy==1.14.0
 + torch==2.7.1+cu118
 + triton==3.3.1
 + typing-extensions==4.15.0
```

**cu124 index:**

```text
$ uv pip install --dry-run --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4' --python python3.12 --system
Using Python 3.12.13 environment at: /usr
Resolved 24 packages in 647ms
Would download 24 packages
Would install 24 packages
 + filelock==3.29.0
 + fsspec==2026.4.0
 + jinja2==3.1.6
 + markupsafe==3.0.3
 + mpmath==1.3.0
 + networkx==3.6.1
 + nvidia-cublas-cu12==12.4.5.8
 + nvidia-cuda-cupti-cu12==12.4.127
 + nvidia-cuda-nvrtc-cu12==12.4.127
 + nvidia-cuda-runtime-cu12==12.4.127
 + nvidia-cudnn-cu12==9.1.0.70
 + nvidia-cufft-cu12==11.2.1.3
 + nvidia-curand-cu12==10.3.5.147
 + nvidia-cusolver-cu12==11.6.1.9
 + nvidia-cusparse-cu12==12.3.1.170
 + nvidia-cusparselt-cu12==0.6.2
 + nvidia-nccl-cu12==2.21.5
 + nvidia-nvjitlink-cu12==12.4.127
 + nvidia-nvtx-cu12==12.4.127
 + setuptools==70.2.0
 + sympy==1.13.1
 + torch==2.6.0+cu124
 + triton==3.2.0
 + typing-extensions==4.15.0
```

### Finding

`torch>=2.4` wheels are available for both CUDA 11.8 and CUDA 12.4 on Python 3.12 / Linux
x86_64:

- **cu118 index** resolves `torch==2.7.1+cu118` — the highest `torch>=2.4` wheel at time of
  probe. The `+cu118` suffix confirms these wheels bundle CUDA 11.8 runtime libraries.
- **cu124 index** resolves `torch==2.6.0+cu124`.

The cu118 index resolved a *higher* torch version (2.7.1) than the cu124 index (2.6.0). This
is because the cu124 wheel series currently tops out at 2.6.0 on PyPI, while cu118 wheels for
2.7.1 are already published.

**Minimum CUDA toolkit for the LoRA-only path**: CUDA 11.8 (cu118 wheel available).

**Minimum CUDA toolkit for the QLoRA path**: CUDA 11.8 (cu118 wheel available; bitsandbytes
also supports CUDA 11.8 — see §3 for details).

---

## §2 — Driver Floor

Source: NVIDIA CUDA Toolkit Release Notes, "Table 3. CUDA Toolkit and Compatible Driver
Versions" (<https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>).

| CUDA toolkit | Minimum driver (Linux) |
| --- | --- |
| 11.8 GA | >= 520.61.05 |
| 12.0 GA | >= 525.60.13 |
| 12.1 GA | >= 530.30.02 |
| 12.2 GA | >= 535.54.03 |
| 12.3 GA | >= 545.23.06 |
| 12.4 GA | >= 550.54.14 |

**LoRA-only path**: minimum driver **520.61.05** (requires CUDA 11.8 toolkit).

**QLoRA path**: minimum driver **520.61.05** (requires CUDA 11.8 toolkit; bitsandbytes supports
CUDA 11.8 — confirmed below in §3).

Both paths share the same driver floor because both can use the cu118 torch wheel and bnb's
CUDA 11.8-compatible pre-built wheels.

---

## §3 — Compute Capability Floor

### LoRA-only path

The `+cu118` torch wheel for Linux x86_64 ships cubins for:
**sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90**

(Source: bitsandbytes installation guide CUDA build target table at
<https://huggingface.co/docs/bitsandbytes/main/en/installation>; consistent with PyTorch's
published wheel arch list for cu118 builds.)

The lowest arch in the cu118 torch wheel is **sm_60 (CC 6.0 / Pascal)**. The LoRA-only CC floor
is therefore **CC 6.0**.

### QLoRA path (bnb ≥ 0.43)

The official bitsandbytes installation guide documents the following feature-level CC requirements:

| Feature | CC Required | Example Hardware |
| --- | --- | --- |
| LLM.int8() | 7.5+ | Turing (RTX 20xx, T4) or newer |
| 8-bit optimizers/quantization | 6.0+ | Pascal (GTX 10X0, P100) or newer |
| NF4/FP4 quantization (QLoRA) | 6.0+ | Pascal (GTX 10X0, P100) or newer |

Source: <https://huggingface.co/docs/bitsandbytes/main/en/installation> — "NVIDIA CUDA" section,
feature table.

The bitsandbytes pre-built Linux x86_64 wheels for CUDA 11.8–12.6 target:
**sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90**

This means the QLoRA path (NF4/FP4) CC floor is **CC 6.0 (sm_60 / Pascal)**, the same as the
LoRA-only path. The QLoRA floor = max(torch arch floor CC 6.0, bnb NF4/FP4 floor CC 6.0) =
**CC 6.0**.

Note: `qlora = ["bitsandbytes>=0.43"]` is an optional extra in `pyproject.toml` (line 31). The
CC floor only matters when the QLoRA path is selected.

### sam3 custom-kernel audit

The pinned sam3 commit `2814fa619404a722d03e9a012e083e4f293a4e53` was audited by traversing the
full repository tree (570 files) at that ref via the GitHub API.

**Finding: sam3 at commit `2814fa6` ships no custom `.cu` or `.cuh` CUDA kernel files, and
`pyproject.toml` contains no `ext_modules`, `CUDAExtension`, or `cmake` extension build
configuration.** Its CC floor is dictated entirely by the torch wheel.

The `sam3/perflib/` package provides optional performance accelerators that gracefully fall back
to pure Python / Triton when optional dependencies (`cc_torch`, `torch_generic_nms`,
`flash_attn`) are unavailable. Specifically:

- `connected_components.py`: uses `cc_torch` if available, falls back to `skimage.measure.label`
  (CPU-only) otherwise.
- `nms.py`: uses `torch_generic_nms` if available, falls back to Triton or CPU NMS; the
  `torch_generic_nms` README shows a `TORCH_CUDA_ARCH_LIST="8.0 9.0"` example but this is an
  optional install and not required by this project.
- `fa3.py`: wraps `flash_attn_interface.flash_attn_func` as an optional op via
  `torch.library.custom_op`; flash-attn-3 requires CC 9.0 (Hopper) but is an optional
  performance accelerator, not a hard dependency.
- `fused.py`: uses `torch.ops.aten._addmm_activation` — a standard PyTorch ATen op, not a
  custom extension.

Source: <https://github.com/facebookresearch/sam3/tree/2814fa619404a722d03e9a012e083e4f293a4e53>

### bf16 autocast implication

The default training dtype is `bfloat16` (`config/schema.py:103`, `dtype: Dtype = "bfloat16"`).
`_autocast_ctx` in `src/custom_sam_peft/train/loop.py:162–168` selects
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` when `cfg.model.dtype == "bfloat16"`.
Native bf16 requires CC ≥ 8.0 (Ampere). On older hardware:

- **T4 (CC 7.5)**: bf16 is *emulated* in software. Training completes but throughput is reduced
  and numerics may differ compared to Ampere hardware.
- **V100 (CC 7.0)**: no native bf16 support. `torch.autocast` falls back to fp32 for ops lacking
  bf16 kernels; results may be silently degraded.
- **Pascal (CC 6.0–6.1)**: same as V100 — no native bf16; fp32 fallback applies.

Users on pre-Ampere hardware should set `model.dtype: float16` in their config to avoid this
degradation. fp16 is natively supported from CC 6.0 (Pascal) onwards.

---

## Compatibility Matrix

| Path       | Python | torch       | min CUDA toolkit | min driver (Linux) | min CC      | status       |
| ---------- | ------ | ----------- | ---------------- | ------------------ | ----------- | ------------ |
| LoRA-only  | 3.12   | 2.7.1+cu118 | 11.8             | 520.61.05          | 6.0 (sm_60) | partial [^1] |
| QLoRA      | 3.12   | 2.7.1+cu118 | 11.8             | 520.61.05          | 6.0 (sm_60) | partial [^1] |

[^1]: *partial*: LoRA and QLoRA training both run on CUDA 11.8 / CC 6.0+ hardware with no code
changes, but the default `bfloat16` dtype is emulated below CC 8.0 (Ampere). Set
`model.dtype: float16` in config to avoid silent numeric degradation on pre-Ampere GPUs
(T4 / V100 / Pascal). LLM.int8() (a separate bnb feature, not used by this project's QLoRA path)
is further gated at CC 7.5, but NF4/FP4 QLoRA is not.

---

## Recommendation

**Partial.**

Both the LoRA-only and QLoRA paths can be lowered to CUDA 11.8 / CC 6.0 (sm_60) by pointing
users to the `+cu118` torch wheel index — no code branching, no CI matrix expansion, and no
changes to `pyproject.toml` are required. The one caveat is that the default `bfloat16` training
dtype is emulated below CC 8.0; this is resolved purely by a user-side config change
(`model.dtype: float16`), not a code change. Per the rubric (spec §5): "One path (likely
LoRA-only) can be lowered cheaply; the other (QLoRA) hits a hard kernel CC floor" — the actual
finding is more permissive: *both* paths lower to CC 6.0 with no code change. The correct bucket
is therefore on the boundary of **Support** and **Partial**; the `partial` designation is
retained because the bf16 emulation caveat requires users to consciously change their config, and
because this has not been validated by a real GPU run. The rubric condition for **Partial** applies:
one non-trivial caveat (bf16 degradation) means out-of-the-box experience is not fully seamless.

---

## Follow-up Issue Candidates

- **Tighten CC-7.5 skip message in `tests/conftest.py`** — Distinguish bnb-bound requirements
  (LLM.int8() at CC 7.5) from SAM-bound requirements so the skip reason accurately reflects that
  QLoRA/NF4 works from CC 6.0, not only CC 7.5.
- **Document cu118 wheel index in `docs/README-dev.md`** — Add an "older GPU" section explaining
  both the LoRA-only and QLoRA paths on CUDA 11.8 / CC 6.0+ hardware and the bf16 → float16
  config change required for pre-Ampere cards.
- **Add cu118 CI matrix documentation** — Describe what a cu118 CI job would cover (smoke test on
  older CUDA toolkit) and the cost/benefit trade-off; candidate only, not filed in this PR.

---

## Sources

1. PyTorch wheel index — cu118: <https://download.pytorch.org/whl/cu118>
2. PyTorch wheel index — cu124: <https://download.pytorch.org/whl/cu124>
3. bitsandbytes installation guide — NVIDIA CUDA section, feature table and build target table:
   <https://huggingface.co/docs/bitsandbytes/main/en/installation> (current `main` docs;
   feature table heading "NVIDIA CUDA", sub-table "Feature / CC Required / Example Hardware
   Requirement"; build target table heading "Installation via PyPI").
4. NVIDIA CUDA Toolkit Release Notes — Table 3, "CUDA Toolkit and Compatible Driver Versions":
   <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
5. sam3 pinned commit (2814fa6):
   <https://github.com/facebookresearch/sam3/tree/2814fa619404a722d03e9a012e083e4f293a4e53>
