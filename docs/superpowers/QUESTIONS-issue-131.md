# Open questions / decisions — issue #131 (autonomous session)

This doc collects questions and notable default-decisions made while you were away.
Each entry: what came up, the **default I chose** to keep moving, and whether it needs
your confirmation. Review at your convenience — anything marked **NEEDS CONFIRM** I'd like
you to weigh in on; anything marked **FYI** is a decision I'm confident in.

---

## Status snapshot

- **Design C** (eval native-res tiling, pad-only transform, train-only expansion) is locked
  and the plan is amended. See commits `a48fc06`, `a6ab485`.
- Phase 1 implementation of design C is landing task-by-task (1.5-gate, 1.6a, 1.6b done).

---

## Entries

### 1. G1 GPU test does not exercise the per-tile OOM-retry path — FYI

The Phase-1 handoff asked G1 to exercise the predict per-tile OOM-retry. The real-model OOM
path can't be triggered deterministically on the 16 GB card without unsafe VRAM starvation or
patching `is_cuda_oom` (which bypasses the real forward, faking the test). The retry mechanism
**is** covered by the CPU stub test `test_predict_one_tile_oom_retry_succeeds`.
**Default:** accept the CPU-stub coverage; leave a `# NOTE:` in G1. No GPU OOM test.
Revisit only if you want a CUDA-memory-limited integration test later.

### 2. Deferred follow-up issue (from Task 1.4 review) — FYI, will file

On a direct-path OOM `RETRY_B`, the chunk restarts and already-tiled images in the chunk are
recomputed (double forward). Output is correct, just wasteful. Out of scope for #131.
**Default:** file a `gh issue` before the PR (tracked, not fixed here).

### 3. Full-mode eval memory: `image_native` held per oversized example — NEEDS CONFIRM

Design C attaches the full native-res numpy array (`Example.image_native`) to each oversized
eval example so the evaluator can tile at native res. In `full` eval mode the evaluator
materializes ALL examples up front (`evaluator.py:570`), so every oversized image's native
array is held simultaneously (e.g. ~48 MB for a 4000² image) — on top of the known full-mode
~12 G predictions list, on a 16 GB box. **Lite / in-training eval is bounded** (capped image
count), so the common path is fine.
**Default I chose:** accept for now (lite/in-training is the real path) and file a follow-up
issue to stream/decode `image_native` per-image in full mode rather than holding all at once.
**Confirm:** OK to defer the full-mode streaming fix to a follow-up issue? Or want it in #131?
