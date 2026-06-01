<!-- markdownlint-disable MD013 -->

# GPU evidence — Colab Tesla T4

This is the Colab-T4 companion to `gpu-evidence-5070ti.md`. It records the
real-hardware measurements captured on a free-tier Google Colab **Tesla T4**
that close the two Colab-confirmation-pending issues from PR #211 (#139 bf16
coercion; #193 T4 per-step timing).

**Hardware:** Tesla T4 — CC 7.5, 16 GB VRAM, Turing architecture, free-tier
Google Colab.
**Surface:** `notebooks/colab_gpu_tests.ipynb` → `colab-min` tier
(`scripts/run_gpu_tests.sh colab-min`: load + forward + one QLoRA 50-step smoke).
**Captured:** 2026-06-01 (user Colab run).

## #139 — bf16 → fp16 coercion on the T4 (CC 7.5)

`coerce_dtype_for_capability` (`src/custom_sam_peft/runtime/_runtime.py`) coerces
`bfloat16` → `float16` on any card below CC 8.0, because bf16 is emulated there.
The T4 (CC 7.5) is below that line, so it runs the `colab-min` smoke under fp16,
not faithful bf16. This was confirmed live on the Colab T4 via a bf16-coercion
probe (now reproducible as the `bf16-coercion` capture cell added to
`notebooks/colab_gpu_tests.ipynb` by this PR). Verbatim captured output:

```text
WARNING:custom_sam_peft.runtime._runtime:Requested bfloat16 on a device with compute capability (7, 5) (< 8.0, where bf16 is emulated); coercing to float16. This is expected on Pascal (GTX 1080).
Device      : Tesla T4
Capability  : 7.5
Requested   : torch.bfloat16
Effective   : torch.float16
Coerced     : True  (bf16->fp16 expected on CC < 8.0)
```

The capture was taken against `origin/main`, whose warning still named
`Pascal (GTX 1080)` as the example sub-8.0 card. This PR refreshes that trailing
example in `_runtime.py` to name the **Turing Tesla T4 (CC 7.5)** — Pascal was
dropped as supported hardware in #211 — so a re-run on this branch prints the T4
phrasing. The coercion behaviour itself (the `(7, 5)` capability and the
`bfloat16 → float16` downcast) is unchanged.

The surrounding stack also independently reports the same capability floor — SAM
3.1 disables Flash Attention on the T4 (`UserWarning: Flash Attention is disabled
as it requires a GPU with Ampere (8.0) CUDA capability`). Faithful, non-coerced
bf16 numerics therefore live only in the `gpu_bf16` tier on a CC ≥ 8.0 card (the
RTX 5070 Ti); the `colab-min` T4 surface validates load + forward + one QLoRA
smoke under fp16, which is the correct behaviour on a T4. This closes #139.

## #193 — T4 QLoRA per-step timing (`colab-min` QLoRA smoke)

| Test | Steps | Wall-clock | Per-step |
|------|-------|------------|----------|
| `tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps` (`min_gpu_qlora`) | 50 | 317.1 s | **≈ 6.34 s/step** |

About 8.4× the 5070 Ti QLoRA step (≈ 0.75 s/step; 317.1 / 37.6 = 8.43), as
expected for the T4's fp16 band. The full figure — and its 5070 Ti counterpart — is recorded in the
"Reference Training Profile" section of `docs/defaults-provenance.md`, resolving
the `# tbd: #193` tag there. This is a 50-step smoke-test proxy, **not** the
160-epoch reference-profile wall-clock (which remains unmeasured). This closes
#193.
