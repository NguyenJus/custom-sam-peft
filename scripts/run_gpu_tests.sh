#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# Turing+ machine with bitsandbytes installed.
#
# Usage:
#   scripts/run_gpu_tests.sh [inspection|release|all]
#
# Tiers (see docs/testing/gpu-test-policy.md):
#   inspection — cheap structural/forward tests in tests/integration/ (9 tests)
#   release    — expensive training-smoke tests in tests/gpu/ (2 tests)
#   all        — both tiers; this is the default (11 tests)
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
TIER="${1:-all}"

case "$TIER" in
  inspection) MARKER_EXPR="gpu_inspection" ; PATHS="tests/integration/" ;;
  release)    MARKER_EXPR="gpu"            ; PATHS="tests/gpu/" ;;
  all)        MARKER_EXPR="gpu or gpu_inspection" ; PATHS="tests/gpu/ tests/integration/" ;;
  *) echo "usage: $0 [inspection|release|all]" >&2; exit 2 ;;
esac

# Use `python -m pytest` (not bare `pytest`) so the test runner picks the
# same interpreter that `pip install -e .` populated. Bare `pytest` on
# PATH can resolve to a different Python (common in Colab) and trigger
# `ModuleNotFoundError: No module named 'custom_sam_peft'`.
# PATHS is a controlled space-separated list of paths; intentional word split.
# shellcheck disable=SC2086
"${PYTHON:-python}" -m pytest -v --tb=short \
  -m "$MARKER_EXPR" --no-cov $PATHS
