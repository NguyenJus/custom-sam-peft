# Spec: Pinned-buffer non-blocking mask transfer (#288)

## Title

Pinned-buffer non-blocking mask transfer for `eval.transfer_binarize`
(bit-identical, ~8x on the bucket / ~12-13% of exact-eval wall-time).

## Context / Background

This work is the **primary lever** of GitHub issue
[#288](https://github.com/NguyenJus/custom-sam-peft/issues/288)
("perf: pinned-buffer non-blocking mask transfer ... + forward-lever spike"),
which was spawned from the #273 algo/CUDA perf audit. The audit triage record
lives at `docs/research/2026-06-03-issue-273-algo-cuda-audit.md` (audit commit
`e5a2d0c`).

The audit measured that the `eval.transfer_binarize` bucket
(`src/custom_sam_peft/eval/postprocess.py`, the
`(masks_up > mask_threshold).cpu().numpy()` line) is **14.6%** of total exact/full
COCO eval wall-time (113.6 s of 777 s on a 355-image DFC run, RTX 5070 Ti, bf16,
adapter). The span is **pageable-PCIe-bound, not compute-bound**: the
threshold-only op is ~0.75 ms, but the pageable `.cpu().numpy()` device-to-host
copy is ~22 ms for M=100 x 1008^2.

The audit prototyped a fix and recorded these microbench results (M=100, 1008^2):

| Variant | ms | speedup | exact? |
| --- | --- | --- | --- |
| baseline `(>thr).cpu().numpy()` | 22.20 | 1.00x | -- (current) |
| `.contiguous()` | 23.09 | 0.96x | yes (neutral) |
| `.to(uint8)` | 45.17 | 0.49x | yes (pessimizes) |
| **pinned + `non_blocking`** | **2.68** | **8.27x** | **yes (bit-identical)** |

At M=200 the pinned copy is 7.2x. An ~8x cut on a 14.6%-of-wall bucket recovers
**~12-13% of total exact-eval wall-time, with bit-identical masks**.

The `.to(uint8)` variant is a **pessimization (0.49x)** and must not be used.

## Goal

Replace the pageable device-to-host mask transfer in
`src/custom_sam_peft/eval/postprocess.py` `queries_to_coco_results` with a
pinned-buffer `non_blocking=True` copy that is **bit-identical** to the current
output and ~8x faster on the `eval.transfer_binarize` bucket.

The change is internalized into a new private module-level helper
`_binarize_to_host(masks_up, mask_threshold)`, still wrapped in the existing
`profiling.bucket("eval.transfer_binarize")` so before/after is directly
comparable via the permanent profiling harness. No changes to the
`queries_to_coco_results` signature and no changes to any call site. The issue
mandates that the work **touches only `postprocess.py`** (plus tests).

## Non-goals / Out-of-scope

- **Forward lever (#6 from the audit, folded into #288 as the secondary lever).**
  `train.forward` (~33% of train wall) / `eval.forward` (~15-20%) is a real
  ceiling, but the cheap levers do not help (`channels_last` is a 0.85x
  pessimization; `torch.compile(reduce-overhead)` is blocked by the
  dtype/RoPE/attention monkeypatch stack). A real win needs CUDA-graph capture
  and/or making the patched modules compile-safe. **This is explicitly deferred
  to a separate, new GitHub issue (CUDA-graph capture / compile-safety spike) and
  must NOT be implemented as part of this work.**
- **The perturbing low-res "binarize before upsample" variant** from the original
  audit. NOT pursued. Only the bit-identical pinned copy is in scope.
- **No GPU-side `.to(uint8)`** anywhere in the transfer path (measured 0.49x
  pessimization).
- **No changes to `eval/evaluator.py`, `eval/visualize.py`, or
  `predict/runner.py`**, nor to `queries_to_coco_results`'s signature.

## Design

### Where the change lands

The current hot line in `queries_to_coco_results` (postprocess.py, ~line 164-165):

```python
with profiling.bucket("eval.transfer_binarize"):
    masks_bin = (masks_up > mask_threshold).cpu().numpy()  # (M, H, W) bool
```

becomes a call to a new private module-level helper, keeping the exact same
profiling bucket:

```python
with profiling.bucket("eval.transfer_binarize"):
    masks_bin = _binarize_to_host(masks_up, mask_threshold)  # (M, H, W) bool
```

`masks_up` is `(M, H, W)` float on the input's device (CUDA in eval, CPU under
the stub/CPU tests). The returned `masks_bin` is an `(M, H, W)` bool numpy array,
identical in dtype and contents to the current path.

### Pinned-buffer pool (module-global singleton)

The pinned buffer is reused across images via module-level mutable state confined
to the helper (a small private class instance or module globals in
`postprocess.py`). Per-call `pin_memory=True` allocation is avoided because it can
be slow/limited; a sized, grow-only pool reused across images is the audit's
recommended approach.

**No locking is needed.** Eval is single-threaded: pycocotools holds the GIL and
CPU-parallelism is a documented dead end in this repo (see the #253 audit).
Document this assumption in a code comment next to the pool state.

The pool tracks:

- `buffer`: a 1-D pinned bool `Tensor` on host (`device="cpu"`, `pin_memory=True`),
  or `None` before first CUDA use.
- `capacity`: `buffer.numel()` (number of bool elements it can hold), or 0.
- `device`: the CUDA device the buffer was last pinned for (re-pin if the source
  device changes; pinned memory is associated with a device context).

Growth is **grow-only**: reallocate (and re-pin) only when the required `numel`
exceeds the current `capacity`, or when the source device differs from the pinned
device. M can shrink between images; on shrink the pool reuses the existing larger
buffer and slices `[:numel]` (see invariant 2 for why the slice prevents stale
tails).

### Concrete code sketch of `_binarize_to_host`

```python
# Module-level pinned-buffer pool. Eval is single-threaded (pycocotools holds the
# GIL; CPU-parallelism is a documented dead end, see #253), so no lock is needed.
class _PinnedHostBuffer:
    """Reusable, grow-only pinned host buffer for D2H mask transfer."""

    def __init__(self) -> None:
        self._buf: Tensor | None = None
        self._device: torch.device | None = None

    def view_for(self, numel: int, device: torch.device) -> Tensor:
        """Return a contiguous 1-D pinned bool slice of length ``numel``,
        pinned for ``device``. Grows (and re-pins) only when required."""
        if (
            self._buf is None
            or self._buf.numel() < numel
            or self._device != device
        ):
            self._buf = torch.empty(numel, dtype=torch.bool, pin_memory=True)
            self._device = device
        return self._buf[:numel]


_PINNED_HOST = _PinnedHostBuffer()


def _binarize_to_host(masks_up: Tensor, mask_threshold: float) -> np.ndarray:
    """Threshold ``masks_up`` to bool and copy to host.

    Bit-identical to ``(masks_up > mask_threshold).cpu().numpy()``. On CUDA, uses a
    reused pinned host buffer with a ``non_blocking=True`` copy + explicit
    synchronize; on CPU, falls back to the plain ``.numpy()`` path (no pinned
    machinery), keeping stub/CPU eval and tests working unchanged.

    WARNING: the CUDA path returns a numpy VIEW into the reused pinned buffer. The
    caller MUST consume it before the next ``_binarize_to_host`` call. See spec
    invariant 2 (the immediately-following RLE block copies it via
    ``np.ascontiguousarray(...).astype(np.uint8)``).
    """
    gpu_bool = masks_up > mask_threshold  # bool, contiguous, on input's device

    if gpu_bool.device.type != "cuda":
        # CPU fallback: bit-identical to the old .cpu().numpy() path.
        return gpu_bool.numpy()

    numel = gpu_bool.numel()
    flat = _PINNED_HOST.view_for(numel, gpu_bool.device)
    view = flat.view(gpu_bool.shape)  # (M, H, W) bool, pinned host
    view.copy_(gpu_bool, non_blocking=True)
    torch.cuda.synchronize()  # required before the host reads the numpy
    return view.numpy()  # zero-copy view of the pinned buffer
```

Notes on the sketch:

- `gpu_bool = masks_up > mask_threshold` produces a contiguous bool tensor on the
  source device (boolean comparison yields a fresh contiguous result), so the
  `copy_` source layout matches the host destination.
- `view = flat.view(gpu_bool.shape)` is the `[:numel]` slice reshaped to
  `(M, H, W)`; `flat` is already `flat[:numel]` from `view_for`.
- The CPU fallback returns `gpu_bool.numpy()` directly. This is bit-identical to
  today's `(masks_up > thr).cpu().numpy()` (on CPU `.cpu()` is a no-op), and uses
  zero pinned machinery so the stub/CPU tests and CPU eval are unaffected.

## Correctness invariants

1. **Output dtype stays bool, never uint8.** The helper returns an `(M, H, W)`
   bool numpy array, exactly as the current path. No GPU-side `.to(uint8)` (the
   issue's measured 0.49x pessimization). Bit-identical to the baseline.
2. **The returned numpy (CUDA path) is a VIEW into the reused pinned buffer and
   must be consumed before the next `_binarize_to_host` call.** It is: the
   immediately-following `eval.rle_encode` block does
   `np.ascontiguousarray(masks_bin).transpose(1, 2, 0).astype(np.uint8)`, which
   copies the data before any next-image call can overwrite the buffer. The
   `[:numel]` slice (via `view_for`) guarantees no stale tail from a previously
   larger M leaks into the result, because the returned view covers exactly
   `M*H*W` elements.
3. **Explicit `torch.cuda.synchronize()` after the `non_blocking=True` copy,
   before the host reads the numpy.** Required for host-side correctness: a
   non-blocking D2H copy may not have completed when control returns, so the host
   must synchronize before reading `view.numpy()`.

## File-by-file changes

### `src/custom_sam_peft/eval/postprocess.py`

- Add the module-level `_PinnedHostBuffer` class (or equivalent module globals)
  and the `_PINNED_HOST` singleton, with the single-threaded / no-lock rationale
  in a comment.
- Add the private `_binarize_to_host(masks_up, mask_threshold)` helper per the
  sketch above (CPU fallback + CUDA pinned path).
- Replace the body of the existing `with profiling.bucket("eval.transfer_binarize"):`
  block (line ~164-165) with a single call to `_binarize_to_host`. The bucket
  wrapper stays so before/after is directly comparable.
- No change to the `queries_to_coco_results` signature, to the surrounding boxes /
  RLE blocks, or to any other function.

### `tests/unit/test_eval_postprocess.py`

- Add a CPU unit test for the helper's CPU fallback (see Testing plan).

### `tests/gpu/test_postprocess_pinned_transfer_gpu.py` (new)

- Add a GPU-gated regression test (see Testing plan), following the repo's
  GPU-test gating convention.

No other files are touched.

## Testing plan

### CPU unit test (runs everywhere, including the CPU stub)

In `tests/unit/test_eval_postprocess.py`, import `_binarize_to_host` and assert
the CPU fallback is bit-identical to the baseline expression:

```python
from custom_sam_peft.eval.postprocess import _binarize_to_host

def test_binarize_to_host_cpu_fallback_matches_baseline():
    masks_up = torch.randn(3, 8, 8)
    thr = 0.0
    got = _binarize_to_host(masks_up, thr)
    expected = (masks_up > thr).numpy()
    assert got.dtype == np.bool_
    assert np.array_equal(got, expected)
```

All existing postprocess tests must stay green — they run on the CPU stub and
exercise `queries_to_coco_results` end-to-end, so they cover the helper's CPU
fallback through the public API as well.

### GPU regression test (gated, runs under `scripts/run_gpu_tests.sh`)

Add `tests/gpu/test_postprocess_pinned_transfer_gpu.py` following the repo's
GPU-test gating convention used by the other `tests/gpu/*.py` files: a
module-level `pytestmark` list applying `pytest.mark.gpu_t4` and
`pytest.mark.requires_compatible_gpu`. (These markers are registered in
`tests/conftest.py` / `pyproject.toml`; `conftest.py`'s collection hook skips
them when no compatible CUDA GPU is present, and `scripts/run_gpu_tests.sh local`
selects `gpu_t4 or gpu_bf16`. This test needs only a CUDA device, not the real
SAM 3.1 checkpoint, so it does NOT carry `requires_checkpoint`.)

The test builds a **real CUDA tensor** and asserts bit-identity against the
baseline across a sequence of shapes that exercise both buffer growth (realloc)
and reuse-without-stale-tail:

```python
pytestmark = [
    pytest.mark.gpu_t4,
    pytest.mark.requires_compatible_gpu,
]

def test_binarize_to_host_cuda_matches_baseline():
    thr = 0.0
    # Sequence: grow M (forces realloc), then shrink M (forces buffer reuse;
    # the [:numel] slice must prevent any stale tail from the larger M leaking).
    for m in (4, 16, 64, 8, 1):
        masks_up = torch.randn(m, 32, 32, device="cuda")
        got = _binarize_to_host(masks_up, thr)
        expected = (masks_up > thr).cpu().numpy()
        assert got.dtype == np.bool_
        assert np.array_equal(got, expected)
```

The M-growth steps exercise the grow-only realloc; the subsequent M-shrink steps
exercise buffer reuse and confirm the `[:numel]` slice prevents a stale tail.

## Acceptance criteria

Mirroring the issue's three acceptance checkboxes:

- [ ] `eval.transfer_binarize` uses a pinned-buffer non-blocking transfer; masks
      are bit-identical to the current path (regression test asserts equality on
      both the CPU fallback and a real CUDA tensor).
- [ ] Before/after `eval.transfer_binarize` bucket timings can be captured via the
      permanent profiling harness on a representative full-eval run (the bucket
      wrapper is preserved, enabling the comparison). The actual GPU capture is
      the Validation step below.
- [ ] Forward (#6) spike is explicitly **deferred to a separate, new GitHub
      issue** (CUDA-graph capture / compile-safety) and is NOT implemented here.

Plus implementation-level criteria:

- [ ] Change is confined to `postprocess.py` (+ tests); no call-site or signature
      changes.
- [ ] No `.to(uint8)` in the transfer path; output dtype stays bool.
- [ ] CPU unit test green in the normal suite; GPU regression test green under
      `scripts/run_gpu_tests.sh` on real hardware.
- [ ] Lint gate green (ruff check + `ruff format --check` + mypy on
      `src/custom_sam_peft`).

## Validation (post-merge, USER-GATED -- NOT part of implementation)

Run a `CSP_PROFILE=1` exact/full eval and compare the `eval.transfer_binarize`
bucket before vs after via the permanent profiling harness (`csp profile`).
Expected: ~8x on the bucket, ~12-13% of total exact-eval wall-time.

**This is a GPU run. Per repo policy, GPU runs require explicit user sign-off
before they are kicked off. Do NOT run it as part of implementation; flag it for
the user to run/approve separately as the final acceptance step.**
