#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# Turing+ machine with bitsandbytes installed.
set -euo pipefail

# Use `python -m pytest` (not bare `pytest`) so the test runner picks the
# same interpreter that `pip install -e .` populated. Bare `pytest` on
# PATH can resolve to a different Python (common in Colab) and trigger
# `ModuleNotFoundError: No module named 'esam3'`.
# TEMPORARY (PR #15): narrowed to the single previously-failing test while we
# iterate on the Colab notebook (branch auto-detect + no-restart install). Revert
# this `-k` filter to the full `tests/integration/` sweep before merging.
"${PYTHON:-python}" -m pytest -v --tb=short \
  -m "requires_compatible_gpu and requires_checkpoint" \
  --no-cov \
  -k "test_load_sam31_forward_to_canonical" \
  tests/integration/
