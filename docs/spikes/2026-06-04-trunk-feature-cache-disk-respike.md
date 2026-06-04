# Trunk feature cache: disk re-spike (issue #300, Part A — SSD)

Companion to the original spike
`docs/spikes/2026-06-04-trunk-feature-cache-spike.md` and spec
`docs/superpowers/specs/2026-06-04-trunk-feature-cache-300-design.md`.

The original spike returned **GO on the mechanics, NO-GO on residence**: the
per-step win was large, but no safe place held the 3,720-image working set —
RAM overflowed by 12×, and disk was excluded by the **HDD-saturation crash
rule** (this box used to crash from disk-I/O saturation, not RAM/VRAM OOM).

**WSL has since been moved onto an SSD.** That removes the sole disk-residence
blocker, so this re-spike measures the one quantity the original omitted —
**real disk read/write latency for a 53 MiB feature blob** — and re-runs the
go/no-go gate with disk residence on the table.

---

## What changed since the original spike

| Dimension | Original (HDD era) | Re-spike (SSD) |
|-----------|--------------------|----------------|
| WSL backing store | HDD (saturation = crash) | SSD (`/dev/sdd`, ext4 on VHDX) |
| Free disk | (not relevant — excluded) | **918 GB** free vs 193 GiB needed |
| Disk read latency | **never measured** | **measured** (Step 2d, below) |
| Residence verdict | NO-GO (RAM 12× over, disk barred) | **GO** (SSD holds the full set) |

---

## How to run

The disk path is **Step 2d** in the same tool; it runs by default.

```bash
CSP_PROFILE=1 uv run python scripts/spike_trunk_cache_feasibility.py \
    --checkpoint models/sam3.1/sam3.1_multiplex.pt \
    --dtype bfloat16 --batch 1 --warmup 3 --iters 10 \
    --disk-tmp ./.spike_disk_cache_tmp \
    --snapshot-out /tmp/spike_300_disk_snapshot.json
```

`--disk-tmp` must resolve onto the **real disk** under test (the tool prints the
filesystem type and refuses to trust a `tmpfs`/`ramfs` mount). Pass `--skip-disk`
to fall back to the original RAM-only measurement.

Step 2d, for `--iters` distinct blobs: `torch.save` the real fp16 feature entry
to the SSD with `fsync`, **evict the page cache** via
`posix_fadvise(POSIX_FADV_DONTNEED)` (no root needed), cold-read it back, then
pin + non-blocking H2D — and compares the full replay tail to the trunk forward.

---

## Results

Measured on the RTX 5070 Ti (sm_120, native bf16), real SAM 3.1
(`sam3.1_multiplex.pt`), B=1, warmup 3 / timed 10. Run date 2026-06-04.
Snapshot: `docs/spikes/snapshots/2026-06-04-trunk-feature-cache-disk-respike-snapshot.json`.

### Bytes + compute (reproduces the original)

| Metric | Mean | Median |
|--------|------|--------|
| per-image feature bytes (fp16) | 53.16 MiB | — |
| trunk\_fwd | 91.37 ms | 90.57 ms |
| h2d\_copy (1 image, pinned, from RAM) | 1.53 ms | 1.53 ms |
| wrapper\_fwd | 164.45 ms | 139.67 ms |
| trunk\_fwd / wrapper\_fwd | 55.6% | 64.8% |

Feature shapes identical to the original run: 3 FPN levels, level-0
`(256, 288, 288)` = 40.50 MiB (76% of the total). `sam2_backbone_out` is `None`.

### Step 2d: disk-backed replay (SSD, page-cache evicted)

| Metric | Mean | Median |
|--------|------|--------|
| on-disk size / image (`torch.save`) | 53.16 MiB | — |
| disk write (cold, epoch-0 build, one-time) | 88.03 ms | — |
| **disk read (cold, page-cache evicted)** | **17.86 ms** | 16.89 ms |
| pin + H2D | 3.45 ms | — |
| **full replay (read + pin + H2D)** | **21.31 ms** | 20.24 ms |

Measured read throughput **2.91 GB/s** — far above the **0.57 GB/s break-even**
(`feature_bytes / trunk_fwd`, below which a cold read costs more than just
recomputing the trunk). The cold-read figure is the **honest steady state**: the
193 GiB working set is ~12× host RAM, so the OS page cache holds only ~8% of it
and ≈92% of reads are genuinely cold every epoch — exactly the eviction path
measured here.

### Device assessment (the Part-B auto-guard, validated live)

| Signal | Value | Verdict |
|--------|-------|---------|
| backing device | `/dev/sdd` | — |
| kernel `rotational` flag | **1 ("HDD")** | **WRONG — advisory only** |
| measured read throughput | 2.91 GB/s | authoritative |
| break-even threshold | 0.57 GB/s | — |
| **AUTO-GUARD (throughput)** | 2.91 ≥ 0.57 GB/s | **PASS** |

This is the live proof that **metadata-based device detection cannot be
trusted** here: the SSD reports `rotational=1` (would falsely flag it as an HDD),
while the RAM-backed `ram*` devices report `rotational=0`. The throughput probe —
which measures the property we actually care about (fast enough that the cache
is a net win) — correctly PASSES the SSD and would correctly FAIL a real
~0.15 GB/s HDD.

### Net-win

| Residence | Net win / replayed step | Verdict |
|-----------|-------------------------|---------|
| RAM (`trunk_fwd - h2d_copy`) | +89.84 ms (median +89.04) | GO |
| **SSD cold, no prefetch** (`trunk_fwd - replay`) | **+70.06 ms** (median +70.33) | **GO** |
| SSD with 1-step prefetch | ~91.37 ms (read fully hidden) | GO |

---

## Decision

ALL conditions now hold for the **full DFC roof train set** (3,720 images):

- [x] **Net win positive (mean AND median)?** — **YES.** SSD residence saves
  **+70 ms/replayed step** even with the page cache defeated and zero prefetch
  (~77% of the RAM-residence win); ~91 ms with one-step prefetch. Low variance
  (mean ≈ median).
- [x] **Working set fits?** — **YES.** 3,720 images × 53.16 MiB ≈ **193 GiB**,
  which fits the **918 GB** free SSD with ~4.7× headroom. (RAM still does not
  fit — disk is the residence, not RAM.)
- [x] **Disk safe (no saturation)?** — **YES.** SSD sustains 2.91 GB/s read /
  ~1.35 GB/s write; the old HDD-saturation crash mode is gone. The one-time
  epoch-0 build adds ~88 ms × 3,720 ≈ **5.5 min**, amortized over ~160 epochs.

**Verdict:** **GO — disk-backed (SSD) trunk-feature cache is viable for the full
~160-epoch DFC roof run**, not just small subsets. This reverses the original
NO-GO-on-residence, whose sole cause (HDD saturation) no longer exists.

### Caveats / open follow-ups before Part B

- **Sustained throughput under contention.** Step 2d measures the cache read in
  isolation. Real training reads compete with DataLoader I/O, checkpoint writes,
  and logging. There is ~5× headroom over break-even, so a 3–4× degradation
  still clears the bar — but confirm with `iostat -x 1` on a representative run
  and keep the throughput auto-guard as a runtime gate, not just a build-time one.
- **Serialization format.** Measured with `torch.save`/`torch.load` (self-
  describing, near-zero size overhead). A raw-bytes / `safetensors` path is a
  later tuning knob if deserialize ever shows up; it does not here (17.9 ms is
  dominated by the disk read, not unpickling).
- **Tiling.** The cache key is per-`sample_uid`; if DFC images are tiled at
  `SAM3_IMAGE_SIZE`, the sample count (and total bytes) rises above the 3,720
  figure. 918 GB still absorbs a large multiple, but recompute the fit-check
  with the real post-tiling sample count.

### Recommended Part-B guard (updated)

Layer three checks; **none** relies on the kernel `rotational` flag:

1. **Hard gate (opt-in, default off):** `cache_trunk_features: false`. Caching
   never turns on implicitly.
2. **Capacity fit-check** at build time: refuse to enable when
   `n_samples × per_image_bytes` exceeds a cited fraction of **free disk**
   (disk now, not RAM) — fail-fast, naming the flag.
3. **Throughput auto-guard** (prototyped in Step 2d): probe the cache
   directory's cold read throughput; if it is below the
   `feature_bytes / trunk_fwd` break-even (~0.57 GB/s), refuse / fall back to
   recompute unless an explicit `cache_allow_slow_disk` override. This is what
   auto-prevents caching on a genuine HDD — **by measured speed, not by an
   unreliable device label.**
