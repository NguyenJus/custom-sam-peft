"""ONNX Runtime session wrappers for the predict path (spec §8.4).

Two layers:

* ``_OrtCore`` — TORCH-FREE. Imports only numpy + onnxruntime + stdlib (with
  ``import onnxruntime`` done lazily inside ``__init__``). Builds the two
  ``InferenceSession`` objects and runs the encoder / decoder graphs, assembling
  the B*K multiplex index arrays and zero box/point prompt embeddings as numpy.
  The torch-free subprocess guard (spec §10.10) loads THIS class.
* ``OnnxSam3Session`` — a drop-in for ``Sam3Wrapper`` in the predict forward
  loop. It bridges numpy<->torch (``torch.from_numpy``) and reuses the shared
  ``validate_forward_inputs`` contract. ``import torch`` is done LAZILY inside
  this class's methods only — never at module top — so loading this module's
  source does not pull torch (keeps ``_OrtCore`` torch-free-loadable).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # torch is a type-only reference here; the runtime import is lazy.
    import torch

_NDArray = np.ndarray[Any, np.dtype[Any]]

ENCODER_FILE = "image_encoder.onnx"
DECODER_FILE = "decoder.onnx"

# Decoder graph output keys, in the order the export side wires them (spec §5.3).
_DECODER_OUTPUT_KEYS = ("pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec")


class _OrtCore:
    """TORCH-FREE ORT core: numpy + onnxruntime only (spec §8.4).

    Builds ``decoder.onnx`` always; ``image_encoder.onnx`` only when the bundle's
    ``model_card.json`` ``include`` is not ``"decoder"``.
    """

    def __init__(self, bundle_dir: Path, providers: list[str]) -> None:
        """Build the encoder (when present) and decoder InferenceSessions."""
        import onnxruntime as ort  # type: ignore[import-untyped]  # lazy; never pull torch

        bundle_dir = Path(bundle_dir)
        # Read model_card.json directly (json + pathlib only) so _OrtCore carries no
        # custom_sam_peft.* top-level import and stays torch-free-loadable (spec §8.4).
        card = json.loads((bundle_dir / "model_card.json").read_text(encoding="utf-8"))
        include = str(card.get("include", "all"))
        self.enc: Any | None = None
        if include != "decoder":
            self.enc = ort.InferenceSession(str(bundle_dir / ENCODER_FILE), providers=providers)
        self.dec: Any = ort.InferenceSession(str(bundle_dir / DECODER_FILE), providers=providers)
        self._dec_input_names: list[str] = [i.name for i in self.dec.get_inputs()]

    def run_encoder(self, np_img: _NDArray) -> dict[str, _NDArray]:
        """Run ``image_encoder.onnx`` on (B, C, 1008, 1008) floats -> named vision arrays."""
        if self.enc is None:
            raise RuntimeError(
                "image_encoder.onnx is not present in this bundle (include=decoder)."
            )
        enc_input = self.enc.get_inputs()[0].name
        out_names = [o.name for o in self.enc.get_outputs()]
        results = self.enc.run(out_names, {enc_input: np_img})
        return dict(zip(out_names, results, strict=True))

    def run_decoder(
        self, vision_feats: dict[str, _NDArray], classes: list[str]
    ) -> dict[str, _NDArray]:
        """Run ``decoder.onnx`` over vision feats + B*K multiplex index + zero prompt arrays.

        Text embeddings are baked into the graph as a constant at export time (spec §5.3),
        so ``classes`` only fixes K (the baked class count) and the multiplex ordering.
        Returns the four-key SAM3-shaped output dict.
        """
        from custom_sam_peft.models._multiplex import multiplex_index_arrays

        # B is recovered from any vision-feature batch dim; K is the prompt class count.
        b = int(next(iter(vision_feats.values())).shape[0])
        k = max(len(classes), 1)
        n_cols = b * k
        img_ids, text_ids = multiplex_index_arrays(b, k)

        # Zero box/point prompt embeddings (no geometric prompts at inference; spec §5.3).
        feed_dtype = next(iter(vision_feats.values())).dtype
        zero_prompts: dict[str, _NDArray] = {
            "img_ids": img_ids,
            "text_ids": text_ids,
            "box_embeddings": np.zeros((0, n_cols, 4), dtype=feed_dtype),
            "box_mask": np.zeros((n_cols, 0), dtype=bool),
            "point_embeddings": np.zeros((0, n_cols, 2), dtype=feed_dtype),
            "point_mask": np.zeros((n_cols, 0), dtype=bool),
        }

        feed: dict[str, _NDArray] = {}
        for name in self._dec_input_names:
            if name in vision_feats:
                feed[name] = vision_feats[name]
            elif name in zero_prompts:
                feed[name] = zero_prompts[name]
            else:
                raise KeyError(
                    f"decoder.onnx expects input {name!r} that is neither a vision feature "
                    f"({sorted(vision_feats)}) nor a known prompt array ({sorted(zero_prompts)})."
                )

        results = self.dec.run(list(_DECODER_OUTPUT_KEYS), feed)
        return dict(zip(_DECODER_OUTPUT_KEYS, results, strict=True))


class OnnxSam3Session:
    """Drop-in for ``Sam3Wrapper`` in the predict loop, backed by an ONNX bundle (spec §8.4).

    ``__call__(images, prompts, support=None) -> dict[str, torch.Tensor]`` returns the
    four-key SAM3-shaped dict (``pred_logits``/``pred_boxes``/``pred_masks``/
    ``presence_logit_dec``) pre-marginalization, so the predict-side semantic reduction
    and ``queries_to_coco_results`` run unchanged over ORT outputs.
    """

    def __init__(self, bundle_dir: Path, *, providers: list[str]) -> None:
        """Build the torch-free ORT core, read expected channels, and set up the encoder LRU."""
        self.bundle_dir = Path(bundle_dir)
        self.core = _OrtCore(self.bundle_dir, providers)
        from custom_sam_peft.predict.onnx_bundle import load_preprocessor

        self.channels = int(load_preprocessor(self.bundle_dir).get("channels", 3))
        self._last_np_img: _NDArray | None = None

        @lru_cache(maxsize=1)
        def _encode(key: tuple[int, tuple[int, ...]]) -> dict[str, _NDArray]:
            """Run the encoder once per unique image batch, keyed (data_ptr, shape) (spec §8.4)."""
            np_img = self._last_np_img
            if np_img is None:
                raise RuntimeError("encoder LRU invoked before image batch was staged")
            return self.core.run_encoder(np_img)

        self._encode = _encode

    def __call__(
        self,
        images: torch.Tensor,
        prompts: list[Any],
        support: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the bundle end-to-end and bridge ORT numpy outputs into a torch-typed dict."""
        import torch  # lazy: keeps module top torch-free so _OrtCore stays torch-free-loadable.

        from custom_sam_peft.models.sam3 import validate_forward_inputs

        validate_forward_inputs(images, prompts, self.channels)

        self._last_np_img = images.detach().cpu().numpy()
        key = (int(images.data_ptr()), tuple(int(s) for s in images.shape))
        vision_feats = self._encode(key)

        classes = list(prompts[0].classes) if prompts else []
        out = self.core.run_decoder(vision_feats, classes)
        return {k: torch.from_numpy(np.ascontiguousarray(out[k])) for k in _DECODER_OUTPUT_KEYS}
