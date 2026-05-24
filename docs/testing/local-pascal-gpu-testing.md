# Local Pascal (GTX 1080) GPU Testing

The dev box holds a GTX 1080 (compute capability 6.1 / sm_61, 8 GB VRAM, ~7 GB
effective after WSL/Xwayland overhead). It is a **GPU-test target**, not a
training/inference platform: it exercises real GPU code paths (sm_61 kernels,
bitsandbytes 4-bit, the gradient-checkpointing fix, float16 dtype handling).

## Provision

The default `uv sync` installs cu130 torch (no sm_61 cubin). To reach the 1080:

```bash
uv sync --extra gpu-pascal   # cu118 torch (sm_60..sm_90 + PTX) + bitsandbytes
```

This extra is isolated via a uv explicit index + extra-scoped source routing, so
the bare `uv sync` and `uv sync --extra dev` paths are unchanged (still cu130).

## Run the gpu_local tier

```bash
bash scripts/run_gpu_tests.sh local
```

Or directly: `uv run pytest -m gpu_local tests/gpu/ tests/integration/ tests/predict/`.

## float16 caveat (Pascal has no fast bf16)

bf16 is **emulated** below compute capability 8.0, so the 1080 trains/runs in
**float16**. A `bfloat16` request is coerced to `float16` with a one-time
warning (see `coerce_dtype_for_capability`). This means numerics validated on
the 1080 do NOT certify the bf16 T4 release path — that confirmation is a
follow-up (gpu_t4 tier).

## Milestone evidence

Recorded 2026-05-24 by task A-2. Full session log:
[`manual-gpu-pass-2026-05-24-gtx1080.md`](manual-gpu-pass-2026-05-24-gtx1080.md).

**Resolution facts (Task A-1, commit eb523ab):**

| env | torch | bitsandbytes |
| --- | --- | --- |
| bare `uv sync` / `uv sync --extra dev` | 2.12.0+cu130 (no sm_61 cubin) | — |
| `uv sync --extra gpu-pascal` | **2.7.1+cu118** (sm_60..sm_90 + PTX) | **0.49.2** |

### Proof 1 — sm_61 CUDA matmul via PTX JIT (PASS)

```text
torch 2.7.1+cu118
device NVIDIA GeForce GTX 1080 cc (6, 1)
matmul max abs err 0.00011444091796875
SM_61 KERNEL OK
```

cu118 torch reaches sm_61 via PTX JIT compiled from `compute_60`. No
`no kernel image is available` error. Matmul error 1.14e-4 < 1e-2.

### Proof 2 — bnb Linear4bit NF4 forward on sm_61, float16 (PASS)

```text
bnb 0.49.2
out (4, 128) torch.float16 finite True
BNB LINEAR4BIT OK
```

bitsandbytes 0.49.2 NF4 4-bit kernel runs on sm_61 under float16. Output
shape (4, 128), dtype float16, all values finite. No CC-rejection error.

**Gate decision: PASS — Pascal track unblocked; downstream B/C/D tasks may proceed.**
