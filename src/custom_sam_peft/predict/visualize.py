"""Per-image overlay visualization for csp predict.

Writes ``<output>/visualizations/<stem>.png`` — the original image with
per-instance mask overlays (alpha-blended), bounding-box outlines, and
``"<class> <score>"`` text labels drawn over each instance.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette — 16 distinct, legible colors (RGB).
# Chosen to be perceptually distinct on both light and dark backgrounds.
# ---------------------------------------------------------------------------

PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),  # red
    (60, 180, 75),  # green
    (255, 225, 25),  # yellow
    (0, 130, 200),  # blue
    (245, 130, 48),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # lime
    (250, 190, 212),  # pink
    (0, 128, 128),  # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),  # brown
    (255, 250, 200),  # beige
    (128, 0, 0),  # maroon
    (0, 60, 100),  # navy
)


def color_for_class(class_name: str) -> tuple[int, int, int]:
    """Return a deterministic RGB color for *class_name* from :data:`PALETTE`.

    Uses ``hashlib.blake2s`` (4-byte digest) so the mapping is stable across
    Python processes and interpreter restarts — ``hash()`` is intentionally
    NOT used here.

    Args:
        class_name: The class/category name string.

    Returns:
        An ``(R, G, B)`` triple from :data:`PALETTE`.
    """
    digest = int(
        hashlib.blake2s(class_name.encode(), digest_size=4).hexdigest(),
        16,
    )
    return PALETTE[digest % len(PALETTE)]


def render_overlay(
    image: Image.Image,
    entries: list[dict[str, object]],
    *,
    prompts: list[str],
) -> Image.Image:
    """Render per-instance mask overlays, bounding boxes, and score labels.

    Args:
        image:   The original PIL RGB image.
        entries: List of COCO-flat prediction dicts for *this image*.
                 Each entry should have ``category_id``, ``bbox``
                 ``[x, y, w, h]``, optional ``score`` (when absent or
                 None, the label is the class name only — GT panels),
                 and optionally ``segmentation`` (COCO RLE dict with
                 ASCII counts).
        prompts: 1-indexed list of class names; ``category_id=1`` maps to
                 ``prompts[0]``.

    Returns:
        A new PIL RGB image with overlays applied.
    """
    result = image.convert("RGB").copy()

    for entry in entries:
        category_id = int(cast(int, entry["category_id"]))
        raw_score = entry.get("score")
        score = float(cast(float, raw_score)) if raw_score is not None else None
        bbox: list[float] = list(cast("list[float]", entry["bbox"]))

        class_name = (
            prompts[category_id - 1] if 0 < category_id <= len(prompts) else str(category_id)
        )
        color = color_for_class(class_name)

        # ------------------------------------------------------------------
        # Mask overlay (only when segmentation is present)
        # ------------------------------------------------------------------
        segmentation = entry.get("segmentation")
        if segmentation is not None:
            try:
                from custom_sam_peft.predict.writers import decode_rle_to_uint8

                mask_arr = decode_rle_to_uint8(segmentation)  # type: ignore[arg-type]

                h, w = mask_arr.shape
                color_layer = Image.new("RGB", (w, h), color)
                mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                blended = Image.blend(result, color_layer, alpha=0.4)
                # Apply blend only inside the mask region.
                result.paste(blended, mask=mask_pil)
            except Exception:
                # Never crash the whole visualization on a bad mask.
                logger.debug("Failed to render mask overlay for entry %r", entry)

        # ------------------------------------------------------------------
        # Bounding box + score label
        # ------------------------------------------------------------------
        draw = ImageDraw.Draw(result)
        x, y, bw, bh = bbox
        draw.rectangle(
            [x, y, x + bw, y + bh],
            outline=color,
            width=2,
        )
        font = ImageFont.load_default()
        label = class_name if score is None else f"{class_name} {score:.2f}"
        draw.text((x, y), label, fill=color, font=font)

    return result


def write_visualization(
    image_path: Path,
    entries: list[dict[str, object]],
    output_dir: Path,
    *,
    prompts: list[str],
) -> Path:
    """Render and write a per-image overlay PNG.

    Writes to ``<output_dir>/visualizations/<stem>.png``.  Creates the
    ``visualizations/`` sub-directory if it does not exist.

    Args:
        image_path: Absolute path to the source image file.
        entries:    COCO-flat prediction entries for this image (may be empty).
        output_dir: The ``--output`` directory supplied by the caller.
        prompts:    1-indexed class-name list (``category_id=1`` → ``prompts[0]``).

    Returns:
        The :class:`~pathlib.Path` of the written PNG file.
    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    rendered = render_overlay(image, entries, prompts=prompts)

    out_path = vis_dir / f"{image_path.stem}.png"
    rendered.save(out_path)
    return out_path
