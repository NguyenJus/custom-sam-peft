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

## Results template

Fill in after running on the GPU box.

### Step 1: per-image feature bytes

| Path | Level | Shape (C, H, W) | fp16 bytes |
|------|-------|-----------------|------------|
| backbone\_fpn | 0 | (?, ?, ?) | ? |
| backbone\_fpn | 1 | (?, ?, ?) | ? |
| backbone\_fpn | 2 | (?, ?, ?) | ? |
| backbone\_fpn | 3 | (?, ?, ?) | ? |
| **backbone\_fpn total** | | | **? MiB** |
| sam2\_backbone\_out | — | — | N/A (None) |
| **Total (no sam2)** | | | **? MiB** |
| **Total (w/ sam2)** | | | **? MiB** |

### Step 2: timing (B=1, bfloat16)

| Metric | Mean | Median |
|--------|------|--------|
| trunk\_fwd | ? ms | ? ms |
| wrapper\_fwd | ? ms | ? ms |
| h2d\_copy (1 image, pinned) | ? ms | ? ms |
| **net\_win = trunk\_fwd - h2d\_copy** | **? ms** | **? ms** |
| trunk\_fwd / wrapper\_fwd | ?% | ?% |

### Step 3: break-even table (fill from script output)

<!-- paste the printed table here -->

### Decision

- [ ] Net win positive (mean AND median)?
- [ ] Dataset fits in 16 GB RAM?  (If not: disk required — see HDD risk below.)
- [ ] HDD-saturation risk acceptable?  (No `WARN` rows, OR disk I/O measured safe.)

**Verdict:** GO / NO-GO

**Residence recommendation:** RAM-cap / disk / hybrid

**Rationale:**

<!-- fill in after measurements -->
