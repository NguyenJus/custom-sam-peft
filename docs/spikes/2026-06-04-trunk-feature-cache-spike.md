# Trunk feature cache: feasibility spike (issue #300, Part A)

Companion to spec
`docs/superpowers/specs/2026-06-04-trunk-feature-cache-300-design.md`.

This spike is the **go/no-go gate** before any production wiring (Part B).
Part B — the cache module, three correctness guards, `cache_trunk_features`
flag, `sample_uid` threading, and `_Sam3ImageAdapter` integration — MUST NOT
land until this spike confirms the net win is positive AND the residence choice
is validated.

---

## How to run

### Full GPU spike (requires real SAM 3.1 + CUDA)

```bash
CSP_PROFILE=1 uv run python scripts/spike_trunk_cache_feasibility.py \
    --dtype bfloat16 \
    --batch 1 \
    --warmup 3 \
    --iters 10 \
    --dataset-sizes 100 500 1000 5000 10000 50000 \
    --snapshot-out /tmp/spike_300_snapshot.json
```

Optionally supply an explicit checkpoint path (skips HF auto-download):

```bash
CSP_PROFILE=1 uv run python scripts/spike_trunk_cache_feasibility.py \
    --checkpoint models/sam3.1/sam3.1_multiplex.pt \
    --dtype bfloat16 \
    --batch 1 \
    --warmup 3 \
    --iters 10
```

The script prints all three measurement blocks and the break-even table.
After the run, inspect the snapshot:

```bash
uv run python scripts/attribute_profile.py /tmp/spike_300_snapshot.json
```

### Pure break-even arithmetic (no GPU)

Once you have per-image byte counts from a real run (or an estimate), rerun
the break-even table standalone:

```bash
uv run python scripts/spike_trunk_cache_feasibility.py \
    --breakeven \
    --bytes-no-sam2 <BYTES> \
    --bytes-with-sam2 <BYTES> \
    --dataset-sizes 100 500 1000 5000 10000 50000 \
    --ram-budget-gb 16 \
    --disk-warn-gb 50
```

Or call the subcommand directly:

```bash
uv run python -c "
import sys; sys.argv = [
    'spike', '--bytes-no-sam2', '<BYTES>', '--dataset-sizes', '100', '1000', '10000'
]
from scripts.spike_trunk_cache_feasibility import _breakeven_main; _breakeven_main()
"
```

### CPU-only smoke test (pure arithmetic, no model)

```bash
uv run pytest tests/unit/test_trunk_cache_breakeven.py -o "addopts=" -p no:cacheprovider
```

---

## Go/no-go decision criteria

ALL three conditions must hold for a GO verdict:

### 1. Net win positive

```text
net_win_per_step = trunk_fwd_mean - h2d_copy_mean > 0
```

The spike prints `GO` or `NO-GO` next to the mean and median net-win. Both
should be positive; a positive mean with a negative median indicates high
variance — investigate before proceeding.

### 2. Fits host-RAM budget

The break-even table marks each (dataset\_size, path) pair as `YES` or `NO`
against the 16 GB host-RAM budget. The spike is a **GO** on RAM only if the
actual dataset fits in RAM with enough headroom for the training process
(model weights in bfloat16 are ~10 GB; the cache competes with that).

### 3. Disk weighed against HDD-saturation crash risk

This box crashes from HDD-I/O saturation, NOT from RAM or VRAM OOM. If the
dataset does NOT fit in RAM:

- Disk residence is only viable if write/read throughput is demonstrated NOT to
  saturate the HDD on a representative run (measure `iostat -x 1` during a
  short training run with cache writes).
- If disk throughput saturates (`%util` near 100% or `await` rises), the spike
  is a **NO-GO on disk residence**. Record and do not proceed.
- The break-even table flags cache sizes above 50 GiB as `WARN: HDD-saturation
  crash risk`. Treat any `WARN` row as requiring explicit disk-I/O validation
  before picking disk residence.

---

## Results

Measured on the RTX 5070 Ti (sm_120, native bf16), real SAM 3.1
(`sam3.1_multiplex.pt`), B=1, warmup 3 / timed 10. Run date 2026-06-04.
Snapshot: `/tmp/spike_300_snapshot.json`.

### Step 1: per-image feature bytes

| Path | Level | Shape (C, H, W) | fp16 bytes |
|------|-------|-----------------|------------|
| backbone\_fpn | 0 | (256, 288, 288) | 40.50 MiB |
| backbone\_fpn | 1 | (256, 144, 144) | 10.12 MiB |
| backbone\_fpn | 2 | (256, 72, 72) | 2.53 MiB |
| **backbone\_fpn total** | | | **53.16 MiB** |
| sam2\_backbone\_out | — | — | N/A (None in this deployment) |
| **Total (no sam2)** | | | **53.16 MiB** |
| **Total (w/ sam2)** | | | **53.16 MiB** |

Level 0 dominates (76% of the per-image bytes); only 3 FPN levels are present.

### Step 2: timing (B=1, bfloat16)

| Metric | Mean | Median |
|--------|------|--------|
| trunk\_fwd | 96.04 ms | 96.81 ms |
| wrapper\_fwd | 151.07 ms | 144.74 ms |
| h2d\_copy (1 image, pinned) | 1.52 ms | 1.48 ms |
| **net\_win = trunk\_fwd - h2d\_copy** | **94.52 ms** | **95.33 ms** |
| trunk\_fwd / wrapper\_fwd | 63.6% | 66.9% |

### Step 3: break-even table (16 GiB host-RAM budget, 50 GiB HDD warn)

| N images | GB (no sam2) | RAM fit? | Disk risk |
|----------|--------------|----------|-----------|
| 100 | 5.191 | YES | — |
| 500 | 25.955 | NO | — |
| 1000 | 51.910 | NO | WARN: HDD-saturation crash risk |
| 5000 | 259.552 | NO | WARN: HDD-saturation crash risk |
| 10000 | 519.104 | NO | WARN: HDD-saturation crash risk |
| 50000 | 2595.520 | NO | WARN: HDD-saturation crash risk |

RAM ceiling is ~300 images in isolation, and realistically ~100–150 once the
training process (model weights + activations + DataLoader) is resident.

### Decision

- [x] Net win positive (mean AND median)? — **YES**, +94.5 ms/replayed step
  (≈64% of the full image forward), low variance (mean≈median).
- [ ] Dataset fits in 16 GB RAM? — **NO** for the real target. The DFC roof
  train set is **3,720 images → ~193 GiB**, 12× over budget (and ≥3,720 cache
  entries before any tiling-window expansion, since the key is per-`sample_uid`).
- [ ] HDD-saturation risk acceptable? — **NO**. 193 GiB is deep in `WARN`
  territory; on this box disk-I/O saturation is the actual session-crash cause,
  so a 193 GiB disk cache is a NO-GO without measured-safe I/O, which it won't be.

**Verdict:** **GO on the mechanics, NO-GO on residence for the full DFC roof run.**

The per-step win is large and real (replaying skips ~96 ms of trunk forward, ~⅔
of the image-encoder cost, for a ~1.5 ms pinned H2D copy — over a ~160-epoch
SAMed regime that is ~159× the saving on every step). But no safe residence
holds the full 3,720-image working set: RAM overflows by 12×, and disk is
excluded by the HDD-saturation crash rule. A capped RAM-LRU does **not** rescue
it — full-dataset epoch iteration touches every sample once per epoch, so an
LRU smaller than the dataset thrashes to a ~0% hit rate (no temporal reuse
within an epoch; reuse is purely across epochs and requires the *whole* set
resident).

**Residence recommendation:** **RAM-cap, small-dataset-only.** The cache is
viable and worthwhile **only when the train set fits RAM with headroom**
(≲ ~150 images at 53 MiB each) — i.e. rapid HP-tuning subsets, overfit-debug,
and smoke runs, NOT the full DFC roof training. Recommend Part B land the cache
behind the existing `cache_trunk_features` flag (default `false`) **plus a
hard fit-check guard**: at build time, compute `n_samples × per_image_bytes`
and refuse to enable (fail-fast, naming the flag) when it exceeds a cited RAM
fraction. That keeps the win available for the small-run case it actually helps,
and prevents it from silently OOM/thrash-crashing the box on a full run.

**Open follow-ups before Part B:**

- Confirm whether DFC images are tiled at `SAM3_IMAGE_SIZE=1008` (would push
  the sample count and bytes above the 3,720-image figure).
- Decide the cited RAM-fraction cap for the fit-check guard (new default →
  needs a citation per the project rule).
- Consider whether the small-run-only scope is worth the Part B implementation
  cost at all, vs. deferring #300 until a larger-RAM host or a cheaper feature
  representation (e.g. storing only level-0 and recomputing the cheap upper
  levels) changes the arithmetic.
