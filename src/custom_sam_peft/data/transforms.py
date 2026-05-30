"""Image augmentation + normalization pipelines (Albumentations).

Public API:
  - resolve_normalization(model_name, fallback, *, channel_semantics) -> (mean, std)
  - resolve_normalization_with_path(model_name, fallback, *, channel_semantics) -> (mean, std, path)
  - build_eval_transforms(image_size, *, model_name, normalize, channel_semantics) -> A.Compose
  - build_train_transforms(aug_cfg, image_size, *, model_name, normalize,
                           channel_semantics, channels) -> A.Compose
  - StainJitter: HED-space stain jitter Albumentations transform
"""

from __future__ import annotations

import logging
from typing import Literal

import albumentations as A
import numpy as np
from numpy.typing import NDArray

from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig

_LOG = logging.getLogger(__name__)

# One-time warning guards for augmentation regime warnings (spec §8.2, §8.3).
_warned_non3ch_photometric = False
_warned_freeform = False


def _warn_non3ch_photometric_augs_once() -> None:
    global _warned_non3ch_photometric
    if not _warned_non3ch_photometric:
        _LOG.warning(
            "Non-3ch photometric semantic: saturation/hue and StainJitter are "
            "skipped (RGB-3ch-only); brightness/contrast substituted via "
            "A.RandomBrightnessContrast. (spec §8.2)"
        )
        _warned_non3ch_photometric = True


def _warn_freeform_augs_once() -> None:
    global _warned_freeform
    if not _warned_freeform:
        _LOG.warning(
            "freeform (non-photometric) semantic: A.ColorJitter, StainJitter, "
            "A.GaussNoise, and A.GaussianBlur are disabled (they assume photometric "
            "continuity); only geometric augmentations apply. (spec §8.3)"
        )
        _warned_freeform = True


# Known-good (mean, std) per HF model name. Used as the offline fallback
# AND as a divergence sentinel against AutoImageProcessor on path 1.
#
# facebook/sam3.1: ImageNet stats. This matches what
# AutoImageProcessor.from_pretrained("facebook/sam3.1").image_mean/image_std
# returns; consistent with SAM/SAM2-class processors. Ratified by the
# 2026-05-21 config-defaults audit (supersedes the 2026-05-16 model-loading
# spec's [0.5, 0.5, 0.5] claim).
KNOWN_PROCESSOR_STATS: dict[str, tuple[list[float], list[float]]] = {
    # cite: ImageNet stats (HF Sam3ImageProcessor)
    "facebook/sam3.1": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

# Element-wise absolute tolerance for table-vs-processor divergence detection
# on path 1. Loose enough to absorb float-serialization noise; tight enough
# to catch a real change (e.g. [0.5, 0.5, 0.5] diverges by >=0.014 per channel).
_STATS_DIVERGENCE_ATOL = 1e-3  # cite: empirical ([0.5,0.5,0.5] drift; rationale above)

# ---------------------------------------------------------------------------
# StainJitter — HED-space color deconvolution for H&E histopathology
# (Ruifrok & Johnston 2001 / Tellez et al. 2018). Image-only Albumentations
# transform; masks/bboxes/keypoints pass through unchanged.
# ---------------------------------------------------------------------------

# Ruifrok & Johnston 2001 HED basis vectors (rows = stains: H, E, DAB).
_HED_FROM_RGB_MATRIX: NDArray[np.float32] = np.array(  # cite: Ruifrok & Johnston 2001
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float32,
)
_HED_FROM_RGB_INV: NDArray[np.float32] = np.linalg.inv(_HED_FROM_RGB_MATRIX).astype(np.float32)

# Magnitude → Albumentations parameter projection constants — spec §8.1.
_GAUSS_NOISE_MAX_VAR: float = 0.05  # tbd: #191
_GAUSS_BLUR_MAX_SIGMA: float = 3.0  # tbd: #191


def _stats_diverge(
    loaded: tuple[list[float], list[float]],
    table: tuple[list[float], list[float]],
) -> bool:
    """True if loaded and table differ on any channel of either vector beyond tolerance."""
    loaded_mean, loaded_std = loaded
    table_mean, table_std = table
    if len(loaded_mean) != len(table_mean) or len(loaded_std) != len(table_std):
        return True
    for lm, tm in zip(loaded_mean, table_mean, strict=True):
        if abs(lm - tm) > _STATS_DIVERGENCE_ATOL:
            return True
    for ls, ts in zip(loaded_std, table_std, strict=True):
        if abs(ls - ts) > _STATS_DIVERGENCE_ATOL:
            return True
    return False


NormalizationPath = Literal["processor", "table-fallback", "config-fallback"]


def resolve_normalization_with_path(
    model_name: str, fallback: NormalizeConfig, *, channel_semantics: str = "rgb"
) -> tuple[list[float], list[float], NormalizationPath]:
    """Three-step resolution of (mean, std) plus the path that fired.

    Path codes:
      - "processor":       loaded from AutoImageProcessor
      - "table-fallback":  processor unavailable, model in KNOWN_PROCESSOR_STATS
      - "config-fallback": processor unavailable, no table entry, user fallback

    For non-rgb channel semantics, AutoImageProcessor is skipped entirely and
    the config-provided mean/std are returned directly (spec §7.1).

    Logging is unchanged from the legacy ``resolve_normalization``: WARN on
    fallback paths, WARN on table-vs-processor divergence, INFO on the happy
    path.
    """
    # spec §7.1: processor + table are RGB-specific; skip for non-rgb semantics.
    if channel_semantics != "rgb":
        return list(fallback.mean), list(fallback.std), "config-fallback"

    import transformers

    table_entry = KNOWN_PROCESSOR_STATS.get(model_name)

    try:
        proc = transformers.AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_name, local_files_only=True
        )
        mean = list(proc.image_mean)
        std = list(proc.image_std)
    except (OSError, AttributeError, ValueError):
        if table_entry is not None:
            table_mean, table_std = table_entry
            _LOG.warning(
                "AutoImageProcessor unavailable for %r; using known-good stats "
                "(mean=%s, std=%s). Populate the HF cache to silence this warning.",
                model_name,
                table_mean,
                table_std,
            )
            return list(table_mean), list(table_std), "table-fallback"
        _LOG.warning(
            "AutoImageProcessor unavailable for %r AND no known-good entry registered; "
            "using NormalizeConfig fallback (mean=%s, std=%s). Verify these are correct "
            "for this backbone.",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std), "config-fallback"

    if table_entry is not None and _stats_diverge((mean, std), table_entry):
        table_mean, table_std = table_entry
        _LOG.warning(
            "AutoImageProcessor for %r returned stats (mean=%s, std=%s) that diverge "
            "from KNOWN_PROCESSOR_STATS (mean=%s, std=%s) beyond tolerance %g. "
            "Using processor values; update the table if this divergence is expected.",
            model_name,
            mean,
            std,
            table_mean,
            table_std,
            _STATS_DIVERGENCE_ATOL,
        )
    else:
        _LOG.info("Using image_mean/image_std from AutoImageProcessor for %r.", model_name)
    return mean, std, "processor"


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig, *, channel_semantics: str = "rgb"
) -> tuple[list[float], list[float]]:
    """2-tuple wrapper kept for build_eval_transforms / build_train_transforms.

    Equivalent to dropping the path code from
    :func:`resolve_normalization_with_path`.
    """
    mean, std, _path = resolve_normalization_with_path(
        model_name, fallback, channel_semantics=channel_semantics
    )
    return mean, std


def build_eval_transforms(
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
    channel_semantics: str = "rgb",
) -> A.Compose:
    """Deterministic eval pipeline: longest-edge resize -> top-left pad -> normalize -> ToTensor."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    mean, std = resolve_normalization(model_name, normalize, channel_semantics=channel_semantics)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                position="top_left",
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=normalize.max_pixel_value),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels", "instance_idx"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )


def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
    channel_semantics: str = "rgb",
    channels: int = 3,
) -> A.Compose:
    """Train pipeline: resolved presets → ordered Albumentations step list.

    See spec §8 for the canonical step ordering. The Albumentations objects
    appear in the compose iff the corresponding knob is enabled / > 0;
    knob = 0/False omits the step entirely (not p=0).

    Three augmentation regimes (spec §8):
      - rgb (photometric, 3ch): full family — ColorJitter (sat/hue), StainJitter,
        GaussNoise, GaussianBlur, geometry.
      - rgba / grayscale (photometric, non-3ch): brightness/contrast substituted
        for ColorJitter; GaussNoise + GaussianBlur kept; sat/hue + StainJitter
        skipped.
      - freeform (non-photometric): geometry only — all four value-altering augs
        disabled even when knobs > 0.
    """
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    from custom_sam_peft.data.aug_presets import resolve
    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS

    profile = CHANNEL_SEMANTICS[channel_semantics]
    photometric = profile.photometric
    rgb_like = photometric and channels == 3

    resolved = resolve(aug_cfg)
    mean, std = resolve_normalization(model_name, normalize, channel_semantics=channel_semantics)
    steps: list[object] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            position="top_left",
        ),
    ]
    # Geometric steps — identical across all three regimes.
    if resolved.hflip:
        steps.append(A.HorizontalFlip(p=0.5))
    if resolved.vflip:
        steps.append(A.VerticalFlip(p=0.5))
    if resolved.rotate90:
        steps.append(A.RandomRotate90(p=0.5))
    if resolved.rotate_arbitrary > 0.0:
        steps.append(
            A.Affine(
                rotate=(-resolved.rotate_arbitrary, resolved.rotate_arbitrary),
                p=0.5,
                fit_output=False,
                fill=0,
                fill_mask=0,
            )
        )

    # Value-altering steps — gated by regime.
    if not photometric:
        # freeform: geometry only — hard-disable all value-altering augs.
        _warn_freeform_augs_once()
    elif rgb_like:
        # rgb: full family unchanged.
        if resolved.gauss_noise > 0.0:
            # Albumentations 2.x replaced var_limit (variance, value-space) with std_range
            # (std as a fraction of max_pixel_value=255 here, since GaussNoise runs pre-Normalize).
            # We preserve the spec's per-knob scaling intent (magnitude * _GAUSS_NOISE_MAX_VAR) but
            # the numeric semantics differ: a knob of 1 caps std at ~12.75 pixel units, not the
            # sub-LSB variance the literal 1.x reading of the spec would produce. This is the
            # behaviorally meaningful interpretation.
            steps.append(
                A.GaussNoise(
                    std_range=(0.0, resolved.gauss_noise * _GAUSS_NOISE_MAX_VAR),
                    p=0.5,
                )
            )
        if resolved.blur > 0.0:
            steps.append(
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    sigma_limit=(0.0, resolved.blur * _GAUSS_BLUR_MAX_SIGMA),
                    p=0.5,
                )
            )
        if resolved.color_jitter > 0.0:
            v = resolved.color_jitter
            steps.append(
                A.ColorJitter(
                    brightness=v,
                    contrast=v,
                    saturation=v,
                    hue=v * 0.5,
                    p=0.5,
                )
            )
        if resolved.stain_jitter > 0.0:
            steps.append(StainJitter(sigma=resolved.stain_jitter, p=0.5))
    else:
        # rgba / grayscale: substitute ColorJitter → RandomBrightnessContrast;
        # keep GaussNoise + GaussianBlur; skip sat/hue + StainJitter.
        _warn_non3ch_photometric_augs_once()
        if resolved.gauss_noise > 0.0:
            steps.append(
                A.GaussNoise(
                    std_range=(0.0, resolved.gauss_noise * _GAUSS_NOISE_MAX_VAR),
                    p=0.5,
                )
            )
        if resolved.blur > 0.0:
            steps.append(
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    sigma_limit=(0.0, resolved.blur * _GAUSS_BLUR_MAX_SIGMA),
                    p=0.5,
                )
            )
        if resolved.color_jitter > 0.0:
            v = resolved.color_jitter
            # brightness_by_max=False scales the brightness shift to the image MEAN
            # (Albumentations 2.0.8: beta = beta * mean(image)), NOT to dtype-max.
            # Chosen so intensity augs stay range-proportional for arbitrary-range
            # float multi-band data (SAR/height) where dtype-max would be 1.0 and the
            # shift negligible. Trade-off: weaker/content-adaptive perturbation on
            # uint8 rgba/grayscale. Deliberate, confirmed decision (spec §7.2/§8.4).
            steps.append(
                A.RandomBrightnessContrast(
                    brightness_limit=v,
                    contrast_limit=v,
                    brightness_by_max=False,
                    p=0.5,
                )
            )

    steps.append(A.Normalize(mean=mean, std=std, max_pixel_value=normalize.max_pixel_value))
    steps.append(ToTensorV2())
    return A.Compose(
        steps,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels", "instance_idx"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )


class StainJitter(A.ImageOnlyTransform):  # type: ignore[misc]
    """HED-space stain jitter for H&E histopathology images.

    Image-only — masks, bboxes, keypoints pass through unchanged. Implements
    the Tellez et al. (2018) / Ruifrok & Johnston (2001) color deconvolution:
    RGB → optical density → HED basis → per-channel affine perturbation →
    back to RGB.

    Identity at sigma=0 (the implementation short-circuits).

    Note: This class directly subclasses ``albumentations.ImageOnlyTransform``
    at module import time. The plan described a lazy ``__new__`` mixin pattern
    to defer the Albumentations import, but that pattern raises
    ``TypeError: __bases__ assignment: 'ImageOnlyTransform' deallocator differs
    from 'object'`` under Albumentations 2.0.8 (CPython's restriction on
    changing __bases__ for extension-type subclasses). Direct subclassing is
    used instead, which also means ``import albumentations`` is triggered
    whenever this module is imported.
    """

    def __init__(self, sigma: float = 0.0, p: float = 0.5) -> None:
        super().__init__(p=p)
        if sigma < 0:
            raise ValueError(f"StainJitter sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)

    def apply(self, img: NDArray[np.uint8], **params: object) -> NDArray[np.uint8]:
        """Apply HED-space stain perturbation to img (uint8 RGB, HWC)."""
        if self.sigma == 0.0:
            return img
        od = -np.log10((img.astype(np.float32) + 1.0) / 256.0)
        hed = od @ _HED_FROM_RGB_INV
        alpha = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        beta = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        hed = hed * (1.0 + alpha) + beta
        od_back = hed @ _HED_FROM_RGB_MATRIX
        out = 256.0 * np.power(10.0, -od_back) - 1.0
        return np.clip(out, 0.0, 255.0).astype(np.uint8)  # type: ignore[no-any-return]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma",)
