"""End-to-end QLoRA test against the real SAM 3.1 checkpoint.

Skipped automatically unless:
  * the Meta checkpoint is present at models/sam3.1/sam3.1_multiplex.pt
  * a CUDA GPU with compute capability >= 7.5 is available (bnb 4-bit
    requires Turing+); SAM 3.1's PositionEmbeddingSine also hardcodes
    device="cuda".
  * bitsandbytes is importable.

The Colab notebook in notebooks/colab_gpu_tests.ipynb is the primary
trigger for this suite. Local Turing+ machines can also run it directly
via scripts/run_gpu_tests.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from esam3.config.schema import ModelConfig, PEFTConfig
from esam3.models.sam3 import load_sam31
from esam3.peft_adapters.lora import merge_lora
from esam3.peft_adapters.qlora import apply_qlora, load_qlora, save_qlora
from tests.helpers.lora_predicates import has_plain_nn_linear as _has_plain_nn_linear

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_inspection,
]


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def _has_linear4bit_modules(module: nn.Module) -> bool:
    import bitsandbytes as bnb

    return any(isinstance(m, bnb.nn.Linear4bit) for m in module.modules())


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_apply_qlora_swaps_every_linear_and_attaches_lora() -> None:
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora"))

    base = w.model.model
    assert _has_linear4bit_modules(base), "no Linear4bit modules after apply_qlora"
    assert not _has_plain_nn_linear(base), "plain nn.Linear modules remain after swap"
    assert w.peft_model is not None

    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.2%} exceeds 5% budget"

    lora_names = [n for n, _ in base.named_parameters() if "lora_" in n]
    assert any("vision_backbone" in n for n in lora_names), "no vision-trunk LoRA targets"
    assert any("transformer.decoder" in n for n in lora_names), (
        "no transformer-decoder LoRA targets"
    )


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_save_qlora_writes_adapter_and_metadata(tmp_path: Path) -> None:
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora"))
    save_qlora(w, tmp_path)

    # PEFT adapter files present.
    assert (tmp_path / "adapter_config.json").exists()
    adapter_weights = list(tmp_path.glob("adapter_model.*"))
    assert adapter_weights, "no adapter_model.* file written"

    # esam3_qlora.json present with the expected fields.
    meta_path = tmp_path / "esam3_qlora.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta == {
        "format_version": 1,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
    }


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_save_load_qlora_roundtrip(tmp_path: Path) -> None:
    w1 = load_sam31(ModelConfig())
    apply_qlora(w1, PEFTConfig(method="qlora"))
    sd1 = {n: p.detach().clone() for n, p in w1.model.model.named_parameters() if "lora_" in n}
    save_qlora(w1, tmp_path)

    w2 = load_sam31(ModelConfig())
    load_qlora(w2, tmp_path)
    sd2 = {n: p for n, p in w2.model.model.named_parameters() if "lora_" in n}

    assert set(sd1) == set(sd2), f"LoRA param names differ: {set(sd1) ^ set(sd2)}"
    for name, t1 in sd1.items():
        assert torch.allclose(t1, sd2[name], atol=0.0), f"mismatch on {name}"


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_merge_lora_unloads_qlora_wrapper() -> None:
    """merge_lora must unload the LoRA wrapper without crashing.

    peft bnb.Linear4bit.merge() dequants the base, adds the LoRA delta, then
    repacks the result as Params4bit (still quantized).  Therefore Linear4bit
    modules legitimately remain after merge_and_unload — asserting their absence
    was wrong.  The structural contract is: the PeftModel wrapper is removed
    (peft_model is None) and the underlying model is still intact.
    """
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora"))
    merge_lora(w)

    # LoRA wrapper must be detached.
    assert w.peft_model is None
    # The base model must still be accessible after the merge.
    assert w.model.model is not None
