#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# compatible machine with bitsandbytes installed.
#
# Usage:
#   scripts/run_gpu_tests.sh [local|t4|bf16|xl|colab-min|light]
#
# Capability tiers (see docs/testing/gpu-test-policy.md):
#   local     — everything the local dev card runs: gpu_t4 OR gpu_bf16
#               (CC >= 7.5, total VRAM <= 16 GB; e.g. RTX 5070 Ti). DEFAULT.
#   t4        — gpu_t4 only: CC >= 7.5 AND VRAM <= 16 GB (Tesla T4 floor / 5070 Ti).
#   bf16      — gpu_bf16 only: CC >= 8.0 AND VRAM <= 16 GB (native, non-coerced bf16).
#   xl        — gpu_xl only: VRAM > 16 GB. Cloud auto-provision (#124); near-empty.
#   colab-min — minimal Colab T4 surface: load+forward + one short QLoRA smoke.
#               Exactly: test_load_sam31_forward_to_canonical +
#               test_qlora_overfits_in_50_steps. Run one process per node ID.
#   light     — R25 evidence subset: colab-min load+forward + #142 8GB-ceiling
#               QLoRA smoke + predict-fits-8GB probe + predict-budget warning.
#               Run one process per node ID (memory safety).
#
# (Test counts per tier are documented in gpu-test-policy.md, not hardcoded here.)
#
# SEQUENTIAL, NON-OVERLAPPING EXECUTION (memory safety):
#   Every tier runs ONE pytest process PER FILE. Process exit forces the OS to
#   reclaim ALL GPU *and* host memory before the next file starts, so no two
#   files' allocations ever coexist. Real-model GPU tests each load the ~3.3 GB
#   SAM 3.1 checkpoint; on WSL2 (GPU and host share system RAM) running every
#   file in a single process accumulates until the whole machine OOMs and
#   freezes — not a clean CUDA OOM. Per-file isolation prevents that. Within a
#   file, pytest runs tests sequentially and the autouse _free_cuda_after_gpu_test
#   fixture synchronizes + frees CUDA between them, so tests never overlap.
#   No parallelism (no pytest-xdist / `-n`) is used or permitted here.
#
# Stateful test-skipping convention (--deselect):
#   When iterating on GPU tests, Claude (or any operator) appends
#   `--deselect <nodeid>` flags to the pytest invocation below as
#   individual tests are confirmed passing on real GPU hardware. This lets
#   the GPU runner skip already-green tests on subsequent runs without
#   editing the test files.
#
#   The mandatory FINAL ALL-GREEN PASS strips every `--deselect` flag and
#   re-runs the full suite to prove it is green end-to-end on a real GPU.
#   No PR may merge with `--deselect` flags left in this script; the CI job
#   `gpu-deselect-check` in `.github/workflows/ci.yml` greps for them and
#   fails the PR if any remain.
set -euo pipefail
TIER="${1:-local}"

# Use `python -m pytest` (not bare `pytest`) so the test runner picks the
# same interpreter that `pip install -e .` populated. Bare `pytest` on
# PATH can resolve to a different Python (common in Colab) and trigger
# `ModuleNotFoundError: No module named 'custom_sam_peft'`.

# expandable_segments reduces allocator fragmentation within each process.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# _run_node runs a single pytest node ID in its own process.
# Exit code 5 means "no tests collected" — treated as success.
# Any other non-zero exit is a real failure and sets _failed=1.
_run_node() {
  local _node="$1"
  local rc=0
  "${PYTHON:-python}" -m pytest -v --tb=short --no-cov "$_node" || rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
    _failed=1
  fi
}

_failed=0

case "$TIER" in
  local) MARKER_EXPR="gpu_t4 or gpu_bf16" ;;
  t4)    MARKER_EXPR="gpu_t4" ;;
  bf16)  MARKER_EXPR="gpu_bf16" ;;
  xl)    MARKER_EXPR="gpu_xl" ;;

  colab-min)
    # Minimal Colab T4 surface: load+forward + one short QLoRA smoke.
    # One process per node ID for memory safety.
    _run_node "tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical"
    _run_node "tests/gpu/test_real_train_qlora.py::test_qlora_overfits_in_50_steps"
    exit "$_failed"
    ;;

  light)
    # R25 evidence subset: colab-min load+forward + #142 8GB-ceiling QLoRA smoke
    # + predict-fits-8GB probe + predict-budget warning GPU test.
    # One process per node ID for memory safety.
    _run_node "tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical"
    _run_node "tests/gpu/test_qlora_8gb_ceiling.py::test_qlora_8gb_ceiling"
    _run_node "tests/predict/test_predict_fits_8gb.py::test_predict_fits_8gb"
    _run_node "tests/gpu/test_predict_budget_warning.py::test_warning_fires_when_over_budget"
    exit "$_failed"
    ;;

  *) echo "usage: $0 [local|t4|bf16|xl|colab-min|light]" >&2; exit 2 ;;
esac

PATHS="tests/gpu/ tests/integration/ tests/predict/"

# Run one pytest process per file so all GPU + host memory is reclaimed at
# process exit between files (see header). expandable_segments reduces
# allocator fragmentation within each process.

# Collect all test files under the search paths.
# PATHS is a controlled space-separated list; intentional word split.
# shellcheck disable=SC2086
while IFS= read -r _file; do
  # Exit code 5 means "no tests collected" — not a failure (a file may hold
  # no tests for the selected tier). Any other non-zero exit is a real
  # failure and must fail the overall run.
  rc=0
  "${PYTHON:-python}" -m pytest -v --tb=short -m "$MARKER_EXPR" --no-cov "$_file" || rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
    _failed=1
  fi
done < <(find $PATHS -name "test_*.py" | sort)
exit "$_failed"
