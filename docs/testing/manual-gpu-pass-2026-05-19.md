# Manual GPU Test Pass — 2026-05-19 (Colab T4, issue #44)

Operational tracker for the manual GPU verification pass for
[#44](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/44).
**Scope is Colab-only** for this pass (no RunPod A40 dry-run); the cloud
dry-runs from spec/simplify-ux §8.6 will be drained on a separate pass
once non-Colab hardware is available.

This file lives alongside [`gpu-test-policy.md`](gpu-test-policy.md), which
defines the tiering and gating rules these tests follow.

## How to run

Open the GitHub-rendered view of `notebooks/colab_gpu_tests.ipynb` on this
branch and click the Colab badge. The notebook auto-detects the branch
from `document.referrer`, so as long as it's opened from the
`manual-gpu-pass-44` branch URL it will check out the right code. Cell 0
sets `BRANCH = None`; only override if auto-detect fails.

Once the notebook clones and pip-installs, every subsequent cell shells
out to `scripts/run_gpu_tests.sh`. Tier selection:

- `bash scripts/run_gpu_tests.sh inspection` — 9 structural tests in `tests/integration/`
- `bash scripts/run_gpu_tests.sh release`    — 3 training-smoke tests in `tests/gpu/`
- `bash scripts/run_gpu_tests.sh all`        — both (default)

### `--deselect` convention

As individual tests confirm green on real GPU hardware, append
`--deselect <nodeid>` flags to the pytest invocation in
`scripts/run_gpu_tests.sh` so subsequent runs skip them. The
`gpu-deselect-check` CI job blocks merge if any `--deselect` flag remains,
so the **final all-green pass strips every `--deselect`** and re-runs the
full suite end-to-end on real hardware.

## Test checklist

### Inspection tier — `tests/integration/` (9 tests)

- [ ] `test_load_sam31_real.py::test_load_sam31_returns_wrapper`
- [ ] `test_load_sam31_real.py::test_load_sam31_forward_to_canonical`
- [ ] `test_peft_lora_real.py::test_apply_lora_on_real_sam31_under_trainable_budget`
- [ ] `test_peft_lora_real.py::test_save_load_roundtrip_on_real_sam31`
- [ ] `test_peft_lora_real.py::test_merge_lora_on_real_sam31`
- [ ] `test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora`
- [ ] `test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata`
- [ ] `test_peft_qlora_real.py::test_save_load_qlora_roundtrip`
- [ ] `test_peft_qlora_real.py::test_merge_lora_unloads_qlora_wrapper`

### Release tier — `tests/gpu/` (3 tests)

- [ ] `test_run_end_to_end_gpu.py::test_run_end_to_end_writes_bundle` — drives `esam3 run` against `configs/examples/gpu_smoke_lora.yaml`; asserts bundle artefacts (`adapter/`, `metrics.json`, `summary.md`, `samples/*.png`)
- [ ] `test_real_train_overfits.py::test_overfits_in_50_steps` — LoRA overfit; loss must drop to ≤ 0.70 × start loss, VRAM ≤ 14 GB
- [ ] `test_real_train_qlora.py::test_qlora_overfits_in_50_steps` — QLoRA equivalent

### User-facing notebook dry-run — Colab T4

- [ ] Open `notebooks/esam3_train.ipynb` via the README Beginner badge, set `HF_TOKEN` in Colab Secrets, supply a small COCO fixture, Runtime → Run All. Verify: FORM widgets render (`#@param` tokens survived ruff format), GENERATE picks the 12–24 GB tier, RESULTS renders `summary.md` + at least one PNG.

## Session log

Append observations, failures, fix commits, and `--deselect` decisions
below as the pass progresses. Final entry: confirmation of the all-green
pass with every `--deselect` stripped.

<!-- 2026-05-19: pass opened, draft PR created, awaiting Colab handoff -->
