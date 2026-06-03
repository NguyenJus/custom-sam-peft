"""Shared sliding-window tiling utility (spec §5.1). Window generation, the
auto-engage rule, a per-window run callback, and the §4 cross-tile fragment-merge.
Operates on plain pixels; geo/DICOM SpatialMeta is threaded by callers (spec §6.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE

# Overlap fraction of the ROI/tile size. Cited: MONAI sliding_window_inference
# default overlap=0.25 (https://docs.monai.io/en/stable/inferers.html#monai.inferers.sliding_window_inference).
DEFAULT_OVERLAP: float = 0.25
# Eval metric tiling uses overlap=0.0 (non-overlapping) to avoid double-counting
# objects in the overlap band. Deliberately differs from predict's overlapping
# tiling (spec §5.4); the visualization restitch still uses DEFAULT_OVERLAP.
EVAL_OVERLAP: float = 0.0


@dataclass(frozen=True)
class Window:
    """A clamped sliding window: origin (y0, x0) + size (h, w), each <= tile."""

    y0: int
    x0: int
    h: int
    w: int


def tiling_engaged(h: int, w: int, tile: int = SAM3_IMAGE_SIZE) -> bool:
    """The single shared auto-engage rule (spec §5.2/§5.3/§5.4): tile iff an edge
    exceeds the model size. max(h, w) == tile takes the direct path."""
    return max(h, w) > tile


def _axis_starts(extent: int, tile: int, overlap: float) -> list[int]:
    if extent <= tile:
        return [0]
    step = max(1, round(tile * (1.0 - overlap)))
    starts = list(range(0, extent - tile + 1, step))
    last = extent - tile
    if starts[-1] != last:
        starts.append(last)  # flush-clamp the final window to the edge (MONAI convention)
    return starts


def iter_windows(
    h: int, w: int, tile: int = SAM3_IMAGE_SIZE, overlap: float = DEFAULT_OVERLAP
) -> list[Window]:
    """Generate covering sliding windows. A <= tile image yields exactly one
    full-image window (direct-path equivalent). Edge windows clamp flush so no
    margin is dropped (spec §5.1)."""
    ys = _axis_starts(h, tile, overlap)
    xs = _axis_starts(w, tile, overlap)
    out: list[Window] = []
    for y0 in ys:
        for x0 in xs:
            out.append(Window(y0=y0, x0=x0, h=min(tile, h - y0), w=min(tile, w - x0)))
    return out


# tbd: mask_overlap_threshold for cross-tile fragment linking. 0.10 chosen by
# tuning against the C2/C3/C4 synthetic fixtures (spec §4.3): seam-overlap bands
# achieve IoM ~0.14-0.24 while non-overlapping fragments have IoM=0.0, so 0.10
# is a robust lower bound. No published canonical value for DETR-fragment
# association — revisit if real-data over/under-merge is observed.
MASK_OVERLAP_THRESHOLD: float = 0.10


@dataclass
class Fragment:
    """A per-tile instance placed on the full-image canvas (spec §4.2)."""

    mask: np.ndarray  # (H, W) bool, full-canvas coordinates
    score: float
    category_id: int
    window_id: int


@dataclass
class MergedInstance:
    mask: np.ndarray
    score: float
    category_id: int


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, a: int) -> int:
        while self._p[a] != a:
            self._p[a] = self._p[self._p[a]]
            a = self._p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self._p[self.find(a)] = self.find(b)


def _intersection_over_min(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    smaller = min(int(a.sum()), int(b.sum()))
    # Association metric. Pinned: intersection-over-min-fragment-area (robust to
    # dissimilar fragment sizes; spec §4.3). # tbd: no published canonical metric
    # for DETR-fragment association — revisit against C2/C4 if over/under-merge.
    return inter / smaller if smaller else 0.0


def merge_fragments(
    fragments: list[Fragment],
    canvas_hw: tuple[int, int],
    *,
    threshold: float = MASK_OVERLAP_THRESHOLD,
) -> list[MergedInstance]:
    """Cross-tile fragment MERGE (spec §4): union-find over the overlap graph,
    per category, fragments from different windows only. Logical-OR each component;
    score = area-weighted mean of fragment scores."""
    n = len(fragments)
    uf = _UnionFind(n)
    for i in range(n):
        for k in range(i + 1, n):
            fi, fk = fragments[i], fragments[k]
            if fi.category_id != fk.category_id:
                continue  # different categories never merge (spec §4.1)
            if fi.window_id == fk.window_id:
                continue  # fragments within one tile are already distinct instances
            if _intersection_over_min(fi.mask, fk.mask) > threshold:
                uf.union(i, k)

    comps: dict[int, list[int]] = {}
    for idx in range(n):
        comps.setdefault(uf.find(idx), []).append(idx)

    out: list[MergedInstance] = []
    for members in comps.values():
        mask = np.zeros(canvas_hw, bool)
        weighted, total_area = 0.0, 0
        for m in members:
            frag = fragments[m]
            mask |= frag.mask  # logical OR within a confirmed object (spec §4.5)
            area = int(frag.mask.sum())
            weighted += frag.score * area
            total_area += area
        score = weighted / total_area if total_area else 0.0  # area-weighted mean
        cat = fragments[members[0]].category_id
        out.append(MergedInstance(mask=mask, score=score, category_id=cat))
    return out
