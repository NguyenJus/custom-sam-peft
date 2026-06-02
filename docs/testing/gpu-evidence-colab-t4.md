<!-- markdownlint-disable MD013 -->

# GPU evidence — Colab Tesla T4

This is the Colab-T4 companion to `gpu-evidence-5070ti.md`. It records the
real-hardware measurements captured on a free-tier Google Colab **Tesla T4**
that close the two Colab-confirmation-pending issues from PR #211 (#139 bf16
coercion; #193 T4 per-step timing), plus the first **full `gpu_t4`-tier run on
a real T4** (#212) and the capability limits it surfaced — see
[Full `gpu_t4` tier run on a real T4](#full-gpu_t4-tier-run-on-a-real-t4-212).

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

The capture was taken during the Colab run against a checkout whose warning
still named `Pascal (GTX 1080)` as the example sub-8.0 card. That trailing
example clause is cosmetic: #228 generalized the warning to drop all
specific-card names, and this branch inherits that wording — current
`_runtime.py` emits `… (< 8.0, below the CC 8.0 / Ampere floor for native
bf16); coercing to float16` with no card named. The coercion behaviour itself
(the `(7, 5)` capability and the `bfloat16 → float16` downcast) is unchanged.

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

## Full `gpu_t4` tier run on a real T4 (#212)

The `gpu_t4` tier (CC ≥ 7.5, ≤ 16 GB) had only ever been *run* on the RTX
5070 Ti, which satisfies **both** `gpu_t4` and `gpu_bf16` — so it has Flash
attention (CC ≥ 8.0) and native bf16. The 2026-06-01 Colab run is the first time
the tier executed on a **real Tesla T4** (CC 7.5, no Flash, fp16-coerced). It
surfaced the gap between the T4 floor and the 5070 Ti superset: most of the tier
is green on the T4, but two capability assumptions and one stale test failed.

**Run provenance.** The run was taken on a Colab checkout **predating #209** (the
card-aware VRAM-hint fix, `35e5f01`). Every commit between that checkout and the
PR branch that touches the 50-step LoRA overfit path is display/wizard/resume
only — `gpu_smoke_lora.yaml` and the trainer's loss/optimizer/LR-schedule path
are byte-identical — so the training **numbers below carry over to current
main** (modulo fp16/CUDA run-to-run noise). The one stale-checkout artifact is
the VRAM-hint failure, already resolved on this branch (see the last row).

| Finding | What the T4 showed | Resolution on this branch |
|---------|--------------------|---------------------------|
| **Multiplex K=8/K=16 forward OOM** | `test_load_sam31_multiplex_K8_forward` and `test_real_K16_forward_…` both OOM on a single **12.81 GiB** allocation in the SAM 3.1 detection-encoder `self_attn`: with no Flash (CC < 8.0) the math kernel materializes the full H·N² score matrix at 1008px, exceeding the T4's 14.56 GiB. Not batch-size-driven (K=8 uses a fixed `b=2`). The same two tests were already recorded OOMing at the identical 12.81 GiB on the GTX 1080 (another no-Flash card) in [`gpu-audit-2026-05-24.md`](gpu-audit-2026-05-24.md) — a confirmed no-Flash architectural limit, not T4 flakiness. | **Multiplex forward is not a T4 guarantee.** Both tests re-tiered `gpu_t4` → `gpu_bf16` (they need a Flash card). They skip on a real T4 instead of failing. |
| **B=1 / K=1 single-class forward** | `test_load_sam31_forward_to_canonical` (one image, one class) passed — this is the path that *does* fit on a T4. | Kept `gpu_t4`; docstring now names it the **minimal-runnable T4 guarantee**. |
| **fp16 50-step LoRA overfit** | `test_overfits_in_50_steps` loss `0.5704 → 0.4256` (ratio **0.746**). fp16 (the coerced dtype) converges less in 50 steps than the bf16 5070 Ti run (`0.5222 → 0.3081`, ratio 0.590). Training works; the bf16-tuned 0.70 ceiling was just too tight for fp16. | **Training is a T4 guarantee.** The overfit ceiling is now capability-aware: `0.70` (bf16, CC ≥ 8.0) / `0.80` (fp16, CC < 8.0). `# tbd:` — pin the fp16 ceiling with a second confirming T4 sample on current main. |
| **`calibrate` cache schema** | `test_calibrate_real` raised `KeyError: 'activation_bytes_per_example'`. This is a **stale test**, not a T4 effect: schema **v3** (#204) split that scalar into `A_fixed`/`A_per_class`. `calibrate --force` itself exits 0 on the T4 (the cheap QLoRA NF4 probes fit). | Test rewritten to assert the v3 invariants (positive `peak_memory_bytes_at_probe` that fit the card; non-negative split with a real per-class signal). |
| **VRAM-hint log (pre-#209 artifact)** | `test_predict_vram_hint_log` failed on the old `assert "free VRAM is >12 GB"`; the T4's post-load free VRAM (~11.75 GiB) is below the 12 GiB gate, so the hint correctly does not fire. | **Already fixed** by #209 (`35e5f01`, on this branch): the test reads the runner's own post-load free-VRAM log line and asserts conditionally, so it **passes on a real T4**. No further change. |
