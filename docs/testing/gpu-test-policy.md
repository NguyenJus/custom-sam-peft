# GPU Test Policy

<!-- Three-tier hardware-capability policy for GPU-gated tests. Formerly the gpu_inspection/gpu 2-tier scheme. -->

## 1. Why this exists

GPU-gated tests impose costs that must be managed deliberately. **Wall time**:
loading real SAM 3.1 weights takes seconds even for a structural inspection, and
a training-smoke test that runs 50 gradient updates on a Colab T4 takes many
minutes. **Paid minutes**: Colab free-tier quota is consumed each time a GPU
notebook run is triggered, and any move to a paid cloud runner amplifies that
cost linearly with PR volume. **Flake surface**: GPU tests fail for reasons
unrelated to the code — driver version mismatches, transient OOM from other
sessions sharing a host, and non-deterministic CUDA kernel scheduling.

These costs motivate *tiering*, but the **primary axis is hardware capability**
(VRAM / compute-capability architecture), not cost. Cost guidance is secondary:
it follows naturally from hardware constraints, and tests that require a T4 are
structurally different from tests that fit an entry-level card — not merely
"more expensive". The CPU-only CI workflow (`ci.yml`, `ubuntu-latest`) never
runs GPU-gated tests; the conftest autoskip handles that automatically for any
test carrying `requires_compatible_gpu` or `requires_checkpoint`. See
[#37](https://github.com/NguyenJus/custom-sam-peft/issues/37) for the
original audit that prompted this policy.

## 2. Tier definitions

### Tier 1 — `gpu_local`

- **Marker:** `@pytest.mark.gpu_local`
- **Hardware target:** any CC 6.0+ card with ≤ ~7 GB usable VRAM (e.g. GTX
  1080). NF4 QLoRA and LoRA both work from CC 6.0 / Pascal; only
  bitsandbytes LLM.int8() needs CC 7.5, and it is unused in this project.
- **Cadence guidance:** run on a dev box (or Colab) for any PR touching GPU
  paths. Cheapest GPU gate.
- **Runner invocation:** `bash scripts/run_gpu_tests.sh local`
  (runs one pytest process **per file** so the ~3.3 GB checkpoint is freed
  between files and the card does not OOM on the ~7 GB budget).
- **What's in the tier:** forward-pass and structural inspection tests only;
  no training loops. 17 tests across `tests/integration/`,
  `tests/predict/`, and `tests/gpu/` — see [the per-test inventory](#3-inventory).

Tier 1 tests are appropriate for any PR that touches model loading
(`custom_sam_peft/models/`), PEFT adapter application
(`custom_sam_peft/peft_adapters/`), the configuration schema
(`custom_sam_peft/config/`), or the predict runner
(`custom_sam_peft/predict/`) — changes that can break structural guarantees
(named parameter targets, module type post-conditions, adapter round-trip
fidelity, forward output shapes) without triggering a training-loop
regression.

**Pascal note:** on CC 6.1 (sm_61, e.g. GTX 1080), bf16 is emulated and the
conftest `coerce_dtype_for_capability` coerces it to float16 automatically.
The default `cu130` torch wheel ships no sm_61 cubin, so local runs on a GTX
1080 require the `gpu-pascal` (cu118) uv extra: `uv sync --extra gpu-pascal
--extra dev`.

### Tier 2 — `gpu_t4`

- **Marker:** `@pytest.mark.gpu_t4`
- **Hardware target:** > 8 GB and ≤ 16 GB VRAM, **or** bf16-representative
  numerics, **or** a real training loop. Colab T4 (16 GB).
- **Cadence guidance:** run before a tagged release, or when the training
  runner / tracker / optimizer code changes land.
- **Runner invocation:** `bash scripts/run_gpu_tests.sh t4`
- **What's in the tier:** 10 tests — training overfits, QLoRA training smokes,
  multiplex/VRAM-calibration tests, the end-to-end CLI smoke, and predict
  tests that run a training loop as setup. See [the per-test inventory](#3-inventory).
- **Provisional:** the gpu_t4 tier is provisional pending
  [#137](https://github.com/NguyenJus/custom-sam-peft/issues/137) (whether
  the 8 GB QLoRA training smokes can be made to fit a smaller card without
  gradient checkpointing, which is abandoned).

Tier 2 tests are appropriate as a release gate and for PRs that change the
training runner (`custom_sam_peft/train/`), the tracker interface
(`custom_sam_peft/tracking/`), or the optimizer configuration — changes
where a structural inspection would pass but a broken gradient flow or a
VRAM regression would not be caught until a real training run.

### Tier 3 — `gpu_xl`

- **Marker:** `@pytest.mark.gpu_xl`
- **Hardware target:** > 16 GB VRAM or larger architecture.
- **Cadence guidance:** cloud auto-provision via
  [#124](https://github.com/NguyenJus/custom-sam-peft/issues/124).
- **Runner invocation:** `bash scripts/run_gpu_tests.sh xl`
- **What's in the tier:** currently empty (0 tests). Reserved for future
  tests that exceed a T4's 16 GB budget or require a Turing+ architecture
  beyond what a T4 provides.

### How tiers run

There is no hosted GPU CI runner in this project. All tiers run via
`notebooks/colab_gpu_tests.ipynb` on a manually provisioned runtime, or
directly on a local machine that has the SAM 3.1 checkpoint and
bitsandbytes installed. The tier policy is trigger-agnostic: the same
tests, the same markers, and the same script apply regardless of whether
the trigger is a Colab notebook cell, a local terminal session, or a future
hosted runner.

The runner script maps each tier argument to a pytest marker filter:

| Tier argument | Marker | Tests collected | Typical runtime |
| --- | --- | --- | --- |
| `local` | `gpu_local` | 17 | ~5–10 min on GTX 1080 |
| `t4` | `gpu_t4` | 10 | ~30–40 min on Colab T4 |
| `xl` | `gpu_xl` | 0 (currently empty) | — |

An unknown argument exits with code 2 and prints a usage line on stderr.
The script uses `"${PYTHON:-python}" -m pytest` rather than bare `pytest`
so the runner picks the same interpreter that `pip install -e .` populated
— bare `pytest` on PATH can resolve to a different Python in Colab and
trigger `ModuleNotFoundError: No module named 'custom_sam_peft'`.

## 3. Inventory

**12 GPU-tagged test files, 27 GPU-gated tests: 17 gpu_local, 10 gpu_t4, 0 gpu_xl.**

The authoritative per-test breakdown — including per-test evidence from a
live GTX 1080 run — is in
[`gpu-audit-2026-05-24.md`](gpu-audit-2026-05-24.md).

Key structural notes from the audit:

- **Move-to-CPU: none.** Every GPU-tagged test asserts on real SAM 3.1
  weights, real CUDA kernels, or bitsandbytes quant state that a CPU stub
  cannot reproduce.
- **Two mixed-tier files** carry per-test markers (not a single file-level
  marker):
  - `test_load_sam31_real.py` — 2 gpu_local + 1 gpu_t4 (K=8 multiplex
    forward, which OOMs on a 1080).
  - `test_gpu_predict.py` — 2 gpu_local (base model, VRAM hint log) + 2
    gpu_t4 (LoRA predict, QLoRA predict — both run a training loop as
    setup).
- All other test files are single-tier.

## 4. T4 validation policy

T4 (Tesla T4, 16 GB VRAM, Turing architecture) is the validated ceiling
for training-smoke tests. The VRAM ceilings in the two smoke configs are
calibrated to T4: 14 GB for LoRA (`configs/examples/gpu_smoke_lora.yaml`,
asserted at `tests/gpu/test_real_train_overfits.py:32`) and 10 GB for
QLoRA (`configs/examples/gpu_smoke_qlora.yaml`, asserted at
`tests/gpu/test_real_train_qlora.py:34`). These ceilings are not
aspirational targets — they are the measured peak usage under the current
training configuration on a T4, rounded up by a small margin to absorb
minor non-determinism across Colab sessions.

Larger GPUs (A100, L4, H100) MAY run the suite and are useful for
development iteration when a T4 is not available. However, a green run on
a larger GPU does NOT substitute for a green T4 run when validating a
`gpu_t4` release gate. The asymmetry is fundamental: an A100 with 40 GB
will silently pass the VRAM assertion (`peak_vram_gb <= 14.0`) even if
actual usage is 20 GB; a T4 with 16 GB will fail in that scenario. T4
validation MUST be confirmed before merging any PR that carries `gpu_t4`
tests as a release gate.

The VRAM ceilings MUST NOT be raised to accommodate larger GPUs or to
paper over a VRAM regression. If a future change pushes T4 peak usage
above a ceiling, the correct response is to reduce usage — reduce
micro-batch size, reduce gradient accumulation steps, shrink sequence
length — not to raise the assertion.

## 5. Data-size policy

The smoke configs drive training with `tests/fixtures/tiny_coco/`: 2
images, 50 gradient updates total (epochs=25, batch\_size=1,
grad\_accum\_steps=1; 25 × 2 = 50 gradient updates). This is the minimal
end-to-end overfit shape that exercises the full `run_training` call graph
without growing the GPU run time beyond a single Colab session slot. The
fixture size and step count are confirmed against both
`configs/examples/gpu_smoke_lora.yaml` and
`configs/examples/gpu_smoke_qlora.yaml`. The 50-step window is long
enough to confirm the loss curve is moving (the loss-ratio assertion
verifies it drops by at least 25–30% depending on tier) and short enough
that the entire gpu_t4 suite completes in a reasonable window on a T4.
The two-image fixture intentionally repeats the same images because the
goal is overfit verification — not generalization.

New GPU tests MUST reuse `tiny_coco` (or a smaller fixture). A test that
requires more images is testing data variety, not the training machinery;
it belongs either in a separate tier with explicit cost approval or on CPU
with stubs and mocked data loaders. This policy is enforced by code review
against the Section 6 checklist.

## 6. Adding a new GPU test — criteria checklist

Any test that needs a real GPU must carry at least one of
`@pytest.mark.requires_compatible_gpu` and `@pytest.mark.requires_checkpoint`
so that the autoskip in `tests/conftest.py` (`requires_compatible_gpu`
gates on CC >= 6.0 / Pascal) suppresses it on CPU-only CI runners. It
must also carry **exactly one tier marker** (`gpu_local`, `gpu_t4`, or
`gpu_xl`) so the runner script can filter by tier. The CPU-only CI
workflow on `ubuntu-latest` never runs tests marked
`requires_compatible_gpu` or `requires_checkpoint` — the conftest autoskip
applies at collection time.

Before merging a new GPU-gated test, confirm all five of the following. A
reviewer can paste this list verbatim into a PR comment:

1. Does the test exercise behavior unique to real SAM 3.1 weights (matched
   named params, real forward shapes) or quant kernels (`Linear4bit`, 4-bit
   base + LoRA delta paths)?
2. Is the CPU+stub variant insufficient, AND is that explicitly documented
   in the test docstring?
3. Is the data fixture `tiny_coco` (or smaller)?
4. Does the test fit within the assigned tier's cost envelope?
   **gpu_local:** load time + single forward pass at most (no training).
   **gpu_t4:** ≤ ~20 minutes wall on T4.
5. Is the test tagged with exactly one of `gpu_local`, `gpu_t4`, or
   `gpu_xl`?

If any answer is no, the test belongs out-of-tier (move marker) or needs
redesign (move work to CPU with a stub; file an issue if the work itself is
worth keeping).

### Choosing the right tier

The primary question is hardware capability. If the test only needs to load
the model, apply PEFT, inspect module structure, or run a single forward
pass on a card with ≤ ~7 GB VRAM and CC 6.0+, it belongs in `gpu_local`.
If the test must run multiple gradient steps, requires > 8 GB, relies on
bf16-faithful numerics (Pascal only emulates bf16), or needs bfloat16
precision over float16, it belongs in `gpu_t4`. Tests that exceed T4
capacity belong in `gpu_xl`.

When in doubt, prefer `gpu_local` and document the rationale in the test
docstring. The [audit table](gpu-audit-2026-05-24.md) shows that all
structural and adapter inspection tests fit in `gpu_local`; only the 10
training-loop, multiplex-OOM, and bf16-faithful tests require `gpu_t4`.
