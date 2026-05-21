# GPU Test Policy

<!-- Two-tier policy for GPU-gated tests. See spec/gpu-test-policy (#37). -->

## 1. Why this exists

GPU-gated tests impose three concrete costs that must be managed
deliberately. The first is **wall time**: loading real SAM3.1 weights takes
seconds even for a structural inspection test, and a training-smoke test
that runs 50 gradient updates on a Colab T4 takes minutes — long enough
that triggering the full suite on every PR commit would make contributors
wait for feedback that should arrive in seconds. The second is **paid
minutes**: the Colab free-tier quota is consumed each time a GPU notebook
run is triggered on a PR that touches GPU paths, and any future move to a
paid cloud runner would amplify that cost linearly with PR volume. The
third is **flake surface**: GPU tests fail for reasons that have nothing to
do with the code under review — driver version mismatch between Colab
runtime images, transient OOM from other notebook sessions sharing the
same physical host, and NaN values produced under non-deterministic CUDA
kernel scheduling — and each spurious failure costs a re-run and erodes
contributor trust. This policy tracks all three costs by tiering the
12 GPU-gated tests into a cheap structural tier and an expensive
training-smoke tier, and by constraining the validated hardware target and
the minimum dataset size so future tests cannot silently widen the cost
envelope. The CPU-only CI workflow (`ci.yml`, `ubuntu-latest`) never runs
GPU-gated tests; the conftest autoskip handles that automatically for any
test carrying `requires_compatible_gpu` or `requires_checkpoint`. See
[#37](https://github.com/NguyenJus/custom-sam-peft/issues/37)
for the audit that prompted this work.

## 2. Tier definitions

### Tier 1 — `gpu_inspection`

- **Marker:** `@pytest.mark.gpu_inspection`
- **Cadence guidance:** run on Colab notebook for any PR touching GPU paths.
- **Runner invocation:** `bash scripts/run_gpu_tests.sh inspection`
- **What's in the tier:** nine structural/forward inspection tests in
  `tests/integration/` across three files — `test_load_sam31_real.py` (2
  tests), `test_peft_lora_real.py` (3 tests), and
  `test_peft_qlora_real.py` (4 tests). Each test loads the real SAM3.1
  checkpoint once, performs a structural inspection (parameter names,
  trainable ratio, module types) or at most a single forward pass through
  the model, then exits. The cost per test is dominated by the checkpoint
  load, not the assertion logic.

Tier 1 tests are appropriate for any PR that touches model loading
(`custom_sam_peft/models/`), PEFT adapter application (`custom_sam_peft/peft_adapters/`), or
the configuration schema (`custom_sam_peft/config/`), because those changes can break
the structural guarantees the tier verifies — named parameter targets,
module type postconditions, and adapter round-trip fidelity — without
triggering any training-loop regression.

### Tier 2 — `gpu`

- **Marker:** `@pytest.mark.gpu`
- **Cadence guidance:** run before a tagged release, or when training-loop /
  tracker / optimizer code changes land.
- **Runner invocation:** `bash scripts/run_gpu_tests.sh release`
- **What's in the tier:** three smoke tests in `tests/gpu/` — two training
  overfits (LoRA, QLoRA) plus an end-to-end CLI `custom-sam-peft run` test:
  `test_real_train_overfits.py::test_overfits_in_50_steps`,
  `test_real_train_qlora.py::test_qlora_overfits_in_50_steps`, and
  `test_run_end_to_end_gpu.py::test_run_end_to_end_writes_bundle`. The first
  two drive the full `run_training` call graph: data loading, forward pass,
  loss computation, backward pass, optimizer step, and scalar logging,
  repeated for 50 gradient updates on `tiny_coco`, asserting loss
  convergence, peak VRAM within the T4 ceiling, and finite logged scalars.
  The third drives `custom-sam-peft run` end-to-end via `CliRunner` against a copy of
  `gpu_smoke_lora.yaml` and asserts on the on-disk bundle (adapter files,
  `metrics.json` parses with numeric `overall.mAP`, `summary.md` present,
  `samples/` has ≤ 6 PNGs, no `merged/` since `export.merge=false`).

Tier 2 tests are appropriate as a release gate and for PRs that change the
training runner (`custom_sam_peft/train/`), the tracker interface (`custom_sam_peft/tracking/`),
or the optimizer configuration — changes where a structural inspection would
pass but a broken gradient flow or a VRAM regression would not be caught
until a real training run.

### How tiers run today

There is no hosted GPU CI runner in this project. Both tiers run via the
same `notebooks/colab_gpu_tests.ipynb` notebook on a manually provisioned
Colab T4 runtime. Cell 6 invokes `bash scripts/run_gpu_tests.sh` with no
argument, which defaults to `all` and runs all 12 tests — both tiers — in
one Colab session. A contributor can override Cell 6 to run a single tier:
pass `inspection` to run only the 9 Tier 1 tests (`bash
scripts/run_gpu_tests.sh inspection`), or `release` to run only the 3 Tier
2 tests (`bash scripts/run_gpu_tests.sh release`). The script can also be
invoked directly on any local Turing+ machine that has the SAM3.1 checkpoint
and bitsandbytes installed. The tier policy is trigger-agnostic: the same
tests, the same markers, and the same script apply regardless of whether the
trigger is a Colab notebook cell, a local terminal session, or a future
hosted runner.

The runner script accepts three tier values and maps each to a pytest marker
expression and a path filter:

| Tier argument | Marker expression | Path filter | Tests collected |
| --- | --- | --- | --- |
| `inspection` | `gpu_inspection` | `tests/integration/` | 9 |
| `release` | `gpu` | `tests/gpu/` | 3 |
| `all` (default, no argument) | `gpu or gpu_inspection` | `tests/gpu/ tests/integration/` | 12 |

An unknown argument exits with code 2 and prints a usage line on stderr. The
script uses `"${PYTHON:-python}" -m pytest` rather than bare `pytest` so the
runner picks the same interpreter that `pip install -e .` populated — bare
`pytest` on PATH can resolve to a different Python in Colab and trigger
`ModuleNotFoundError: No module named 'custom_sam_peft'`.

## 3. Inventory

The table below covers all 12 GPU-gated tests as of the
[#37](https://github.com/NguyenJus/custom-sam-peft/issues/37)
audit. Each row is derived from reading the corresponding test file:
its module-level docstring (which names the GPU dependency), the assertions
in the test body (which determine whether a CPU stub could replicate them),
and the tier assignment (from the pytestmark). "CPU-stub viability" uses
three buckets:

- `none — needs real SAM3.1 weights` — the test asserts on SAM3.1-specific
  named parameters (e.g. `"vision_backbone" in n`, `"transformer.decoder"
  in n`), on real forward output shapes that only the loaded model
  produces, or on bitsandbytes quant kernels (`Linear4bit`). A CPU stub
  cannot replicate these.
- `partial — could cover X but loses Y` — a structural assertion (e.g.
  trainable ratio, module type swap) is replicable by a CPU stub that
  mimics the same module-tree shape, but the test loses its end-to-end
  signal (e.g. the real model under real PEFT actually produces a finite
  forward pass).
- `viable — see follow-up #N` — the assertions a `TinySam3Stub` (or
  equivalent CPU mock) can fully satisfy with no real-model dependency. A
  follow-up issue tracks the CPU-stub replacement.

| `file::test` | Tier | What it covers | Why GPU is required | CPU-stub viability |
| --- | --- | --- | --- | --- |
| `tests/integration/test_load_sam31_real.py::test_load_sam31_returns_wrapper` | inspection | `load_sam31` returns a `Sam3Wrapper` instance under a bfloat16 CUDA config. | `load_sam31` loads real SAM3.1 weights from disk; `PositionEmbeddingSine` inside the model hardcodes `device='cuda'`, blocking CPU execution entirely. | none — needs real SAM3.1 weights |
| `tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical` | inspection | Real SAM3.1 forward pass produces `CanonicalOutput` with correct tensor ranks and SAM3.1-specific output dimensions. | Real weights are required for the SAM3.1-specific output dimension (`pred_masks.shape[-1] == 288`); `PositionEmbeddingSine` hardcodes `device='cuda'`. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget` | inspection | LoRA trainable-ratio < 5% and presence of `vision_backbone` + `transformer.decoder` LoRA params after `apply_lora` on real SAM3.1. | Asserts on SAM3.1-specific named parameters; CPU stub does not have these modules. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_lora_real.py::test_save_load_roundtrip_on_real_sam31` | inspection | LoRA adapter saved to disk and reloaded into a fresh model; all `lora_` param names and exact tensor values match. | Asserts on the full set of SAM3.1 `lora_` parameter names produced by real PEFT applied to the real SAM3.1 module tree; these param names are model-specific. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_lora_real.py::test_merge_lora_on_real_sam31` | inspection | `merge_lora` removes the PEFT wrapper; `peft_model` is `None` and the class name no longer contains "Peft". | Requires `load_sam31` with real weights; `PositionEmbeddingSine` hardcodes `device='cuda'`, preventing CPU execution of the setup path. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora` | inspection | Every `nn.Linear` swapped to `Linear4bit`; no plain `nn.Linear` remains; LoRA adapter attached; trainable ratio < 5%; `vision_backbone` + `transformer.decoder` LoRA targets present. | Requires bitsandbytes `Linear4bit` quant kernels (Turing+) and SAM3.1-specific named parameters for the LoRA target assertions. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata` | inspection | `save_qlora` writes `adapter_config.json`, at least one `adapter_model.*` file, and `custom_sam_peft_qlora.json` with exact NF4/bfloat16 metadata. | Requires `apply_qlora` (bitsandbytes Turing+ quant kernels) to produce the PEFT adapter artifacts that the file-existence and JSON-content assertions verify. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` | inspection | QLoRA adapter saved and reloaded into a fresh model; all `lora_` param names and exact tensor values agree between the two model instances. | Asserts on the full set of SAM3.1 `lora_` parameter names from real QLoRA applied to the real module tree; bitsandbytes quant kernels required for `apply_qlora`. | none — needs real SAM3.1 weights |
| `tests/integration/test_peft_qlora_real.py::test_merge_lora_unloads_qlora_wrapper` | inspection | `merge_lora` detaches the PEFT wrapper (`peft_model is None`) while leaving `model.model` accessible after the merge. | Requires `apply_qlora` (bitsandbytes Turing+ quant kernels) and `load_sam31` with real weights; `PositionEmbeddingSine` hardcodes `device='cuda'`. | none — needs real SAM3.1 weights |
| `tests/gpu/test_real_train_overfits.py::test_overfits_in_50_steps` | release | 50-step LoRA overfit on tiny_coco via `run_training(gpu_smoke_lora.yaml)`; asserts loss drops ≥ 30%, peak VRAM ≤ 14 GB, all logged scalars finite. | End-to-end real training: real SAM3.1 weights, real PEFT, real CUDA kernels, real optimizer step. | none — needs real SAM3.1 weights |
| `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps` | release | 50-step QLoRA overfit on tiny_coco via `run_training(gpu_smoke_qlora.yaml)`; asserts loss drops ≥ 25%, peak VRAM ≤ 10 GB, all logged scalars finite. | End-to-end real training with 4-bit base + bf16 LoRA + 8-bit optimizer; requires bitsandbytes quant kernels (Turing+) and real SAM3.1 weights. | none — needs real SAM3.1 weights |
| `tests/gpu/test_run_end_to_end_gpu.py::test_run_end_to_end_writes_bundle` | release | `custom-sam-peft run` CLI end-to-end on tiny_coco: writes adapter, `metrics.json`, `summary.md`, samples; no `merged/`. | End-to-end real training via the CLI Typer entry; real SAM3.1 weights, real PEFT, real optimizer step. | none — needs real SAM3.1 weights |

## 4. T4 validation policy

T4 (Tesla T4, 16 GB VRAM, Turing architecture) is the only validated VRAM
tier for this test suite. The Colab notebook's prereqs cell pins T4 as the
minimum runtime, and the VRAM ceilings in the two smoke configs are
calibrated to T4: 14 GB for LoRA
(`configs/examples/gpu_smoke_lora.yaml`, asserted at
`tests/gpu/test_real_train_overfits.py:32`) and 10 GB for QLoRA
(`configs/examples/gpu_smoke_qlora.yaml`, asserted at
`tests/gpu/test_real_train_qlora.py:34`). These ceilings are not
aspirational targets — they are the measured peak usage under the current
training configuration on a T4, rounded up by a small margin to absorb
minor non-determinism across Colab sessions. The 14 GB LoRA ceiling
leaves headroom below the T4's 16 GB physical limit; the 10 GB QLoRA
ceiling reflects the 4-bit base compression that makes QLoRA viable on T4
at all. Both ceilings are also the minimum hardware requirement for the
bitsandbytes Turing+ quant kernels that `apply_qlora` depends on — a GPU
with compute capability below 7.5 cannot run the QLoRA tier at all,
regardless of VRAM.

Larger GPUs (A100, L4, H100) MAY run the suite and are useful for
development iteration when a T4 is not available. However, a green run on
a larger GPU does NOT substitute for a green T4 run when validating a
release. The asymmetry is fundamental: an A100 with 40 GB or more will
silently pass the VRAM assertion (`peak_vram_gb <= 14.0`) even if the
actual usage is 20 GB, because the test measures real peak usage and a
larger GPU simply has the headroom. A T4 with 16 GB will fail in that same
scenario. Release validation MUST be confirmed on T4 so that contributors
on shared Colab runtimes — where T4 is the common free-tier allocation —
can rely on the ceilings actually protecting them.

The VRAM ceilings MUST NOT be raised to accommodate larger GPUs or to
paper over a regression in training-loop memory usage. If a future change
to the training runner, optimizer, or model configuration pushes T4 peak
usage above a ceiling, the correct response is to reduce VRAM usage:
enable or tune gradient checkpointing, reduce micro-batch size, reduce
gradient accumulation steps, or shrink the sequence length — not to raise
the number in the config or the assertion. Raising the ceiling trades a
test failure today for a silent OOM on the next contributor's T4 tomorrow.

## 5. Data-size policy

The smoke configs drive training with the `tests/fixtures/tiny_coco/`
fixture: 2 images, 50 gradient updates total (epochs=25, batch\_size=1,
grad\_accum\_steps=1 — tiny\_coco has 2 images, so each epoch produces 2
batches and one gradient step per batch, giving 25 × 2 = 50 gradient updates).
This is the minimal
end-to-end overfit shape that exercises the full `run_training` call graph
without growing the GPU run time beyond a single Colab session slot. The
fixture size and step count are confirmed against both
`configs/examples/gpu_smoke_lora.yaml` and
`configs/examples/gpu_smoke_qlora.yaml`. The 50-step window is long
enough to confirm the loss curve is moving (the loss-ratio assertion
verifies it drops by at least 25–30% depending on tier) and short enough
that the entire Tier 2 suite completes in a few minutes on a T4. The
two-image fixture intentionally repeats the same images across all 50 steps
because the goal is overfit verification — confirming the optimizer can
drive loss down on a fixed batch — not generalization.

New GPU tests MUST reuse `tiny_coco` (or a smaller fixture if one exists
or is added). A test that requires more images to make its assertions is
testing data variety or sample coverage rather than the training machinery
itself; it has slipped into an integration test that belongs either in a
separate tier with explicit cost approval or on CPU with appropriate stubs
and mocked data loaders. This policy is enforced by code review against the
Section 6 checklist, not by automated tooling — there is no fixture-size
check in CI. Reviewers should apply the checklist on any PR that adds a new
`@pytest.mark.gpu` or `@pytest.mark.gpu_inspection` test.

## 6. Adding a new GPU test — criteria checklist

Any test that needs a real GPU to run must carry at least one of
`@pytest.mark.requires_compatible_gpu` and `@pytest.mark.requires_checkpoint`
so that the autoskip in `tests/conftest.py` suppresses it on CPU-only CI
runners. It must also carry exactly one tier marker (`gpu_inspection` or
`gpu`) so the runner script can filter by tier. The CPU-only CI workflow on
`ubuntu-latest` never runs tests marked `requires_compatible_gpu` or
`requires_checkpoint` — the conftest autoskip applies at collection time.

Before merging a new GPU-gated test, confirm all five of the following. A
reviewer can paste this list verbatim into a PR comment:

1. Does the test exercise behavior unique to real SAM3.1 weights (matched
   named params, real forward shapes) or quant kernels (`Linear4bit`, 4-bit
   base + LoRA delta paths)?
2. Is the CPU+stub variant insufficient, AND is that explicitly documented
   in the test docstring?
3. Is the data fixture `tiny_coco` (or smaller)?
4. Does the test fit within the assigned tier's cost envelope?
   **Inspection:** ~load time per test (single forward pass at most).
   **Release:** ≤ 2 minutes on T4.
5. Is the test tagged with exactly one of `gpu` or `gpu_inspection`?

If any answer is no, the test belongs out-of-tier (move marker) or needs
redesign (move work to CPU with a stub; file an issue if the work itself is
worth keeping).

### Choosing the right tier

When deciding which tier a new test belongs in, the primary question is
whether the test can produce a useful signal without running a training
loop. If the test only needs to load the model, apply PEFT, inspect module
structure, or run a single forward pass, it belongs in Tier 1
(`gpu_inspection`). If the test must run multiple gradient steps to make a
meaningful assertion — loss convergence, gradient flow, VRAM under a real
optimizer state — it belongs in Tier 2 (`gpu`). When in doubt, prefer
Tier 1 and document the rationale in the test docstring. The Section 3
inventory shows that all current structural and adapter tests fit
comfortably in Tier 1; only the three smoke tests (two training overfits
plus the end-to-end CLI `custom-sam-peft run` test) justify Tier 2.
