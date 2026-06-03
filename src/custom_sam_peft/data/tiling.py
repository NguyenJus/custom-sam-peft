"""Shared sliding-window tiling utility (spec §5.1). Window generation, the
auto-engage rule, a per-window run callback, and the §4 cross-tile fragment-merge.
Operates on plain pixels; geo/DICOM SpatialMeta is threaded by callers (spec §6.4)."""

from __future__ import annotations

from dataclasses import dataclass

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
