#!/usr/bin/env bash
# check_gpu_evidence.sh — non-blocking GPU evidence freshness check.
#
# Usage: check_gpu_evidence.sh <artifact_path> <head_sha>
#
# Freshness signal:
#   current — the artifact file EXISTS and its text CONTAINS <head_sha>.
#   stale   — the artifact file EXISTS but does NOT contain <head_sha>.
#   missing — the artifact file does NOT exist.
#
# Exit behaviour: ALWAYS exits 0 regardless of freshness state.
# This is the non-blocking guarantee; CI jobs running this script always
# succeed. Use the output text (warning annotations for missing/stale,
# green confirmation for current) to observe evidence health without
# gating the build.

set -euo pipefail

ARTIFACT_PATH="${1:?Usage: check_gpu_evidence.sh <artifact_path> <head_sha>}"
HEAD_SHA="${2:?Usage: check_gpu_evidence.sh <artifact_path> <head_sha>}"

if [[ ! -f "${ARTIFACT_PATH}" ]]; then
    echo "::warning::gpu-evidence missing — ${ARTIFACT_PATH} does not exist. Run the GPU test suite and commit the evidence file."
    exit 0
fi

# Use grep's exit code without letting set -e fire on a non-match.
match_count=$(grep -cF "${HEAD_SHA}" "${ARTIFACT_PATH}" || true)

if [[ "${match_count}" -gt 0 ]]; then
    echo "gpu-evidence ok — artifact is current for commit ${HEAD_SHA}."
else
    echo "::warning::gpu-evidence stale — ${ARTIFACT_PATH} exists but does not contain commit ${HEAD_SHA}. Re-run the GPU test suite against this commit."
fi

exit 0
