"""End-to-end QLoRA test against the real SAM 3.1 checkpoint.

Skipped automatically unless:
  * the Meta checkpoint is present at models/sam3.1/sam3.1_multiplex.pt
  * a CUDA GPU with compute capability >= 7.5 is available (gpu_t4 tier:
    Tesla T4 / RTX 5070 Ti); bf16 is coerced to fp16 below CC 8.0.
    SAM 3.1's PositionEmbeddingSine also hardcodes device="cuda".
  * bitsandbytes is importable.

The Colab notebook in notebooks/colab_gpu_tests.ipynb is the primary
trigger for this suite. gpu_t4 machines can also run it directly
via scripts/run_gpu_tests.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.models.sam3 import SAM3_IMAGE_SIZE, load_sam31
from custom_sam_peft.peft_adapters.lora import merge_lora
from custom_sam_peft.peft_adapters.qlora import (
    _infer_compute_dtype_from_wrapper,
    apply_qlora,
    load_qlora,
    save_qlora,
)
from tests.helpers.lora_predicates import has_plain_nn_linear as _has_plain_nn_linear

pytestmark = [
    pytest.mark.requires_checkpoint,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.gpu_t4,
]

# #230 Phase 1 spike — go/no-go (recorded 2026-06-02):
#   Mechanism SELECTED: §7.3(a) lora.MultiheadAttention (NOT target_parameters).
#   The target_parameters route (peft 0.19.1 lora.ParamWrapper) hard-raises
#   ValueError: lora.ParamWrapper does not work with lora_dropout != 0,
#   which poisons LoraConfig construction when dropout=0.05 (our default).
#   The concept scope instead names the ca_text/self_attn nn.MultiheadAttention
#   modules in target_modules; peft dispatches them to lora.MultiheadAttention
#   (layer.py:2492), adapting BOTH in_proj_weight AND out_proj, with dropout.
#
#   QLoRA coexistence: GO — GPU-confirmed (§7.2, §7.3a). The MHA stays
#     unquantized (_mha_exclusion_types), so the bf16 in_proj LoRA and the
#     Linear4bit module LoRA attach in one PeftModel with dropout=0.05.
#     merge_and_unload is clean (only a benign NF4-rounding UserWarning).
#   => Production tests below drive apply_qlora / merge_lora via PEFTConfig
#      (scope="vision_decoder_concept"), no inline get_peft_model calls.


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def _has_linear4bit_modules(module: torch.nn.Module) -> bool:
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

    # custom_sam_peft_qlora.json present with the expected fields (format v2).
    # compute_dtype reflects the *effective* quantization dtype: bfloat16 is
    # coerced to float16 below CC 8.0 (bf16 is not faithfully supported there),
    # so derive it from the wrapper rather than hardcoding.
    meta_path = tmp_path / "custom_sam_peft_qlora.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta == {
        "format_version": 2,
        "quant_type": "nf4",
        "compute_dtype": _infer_compute_dtype_from_wrapper(w),
        "use_double_quant": False,
    }


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_save_load_qlora_roundtrip(tmp_path: Path) -> None:
    w1 = load_sam31(ModelConfig())
    apply_qlora(w1, PEFTConfig(method="qlora"))
    sd1 = {
        n: p.detach().cpu().clone() for n, p in w1.model.model.named_parameters() if "lora_" in n
    }
    save_qlora(w1, tmp_path)

    # Forward-output parity setup: capture w1's outputs BEFORE deleting it.
    # Host RAM (~12 GB) cannot hold w1 and w2 simultaneously, so only the small
    # CPU output tensors survive the del. Mirrors evaluator.py's call pattern:
    #   wrapper(images, prompts, support=None) -> dict[str, Tensor]
    # with keys pred_logits / pred_boxes / pred_masks / presence_logit_dec.
    from custom_sam_peft.data.base import TextPrompts

    torch.manual_seed(0)
    w1.eval()
    _images = torch.zeros(
        1, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, device="cuda", dtype=torch.float32
    )
    _prompts = [TextPrompts(classes=["object"])]
    with torch.no_grad():
        _out_w1 = w1(_images, _prompts, support=None)
    _out_w1_cpu = {k: v.detach().cpu() for k, v in _out_w1.items() if isinstance(v, torch.Tensor)}

    # Free w1 before constructing w2 — Colab host RAM (~12 GB) cannot hold
    # two sam31 instances simultaneously, and this test was SIGKILLed (exit
    # 137) before the fix.
    import gc

    del w1
    gc.collect()
    torch.cuda.empty_cache()

    w2 = load_sam31(ModelConfig())
    load_qlora(w2, tmp_path)
    sd2 = {n: p for n, p in w2.model.model.named_parameters() if "lora_" in n}

    assert set(sd1) == set(sd2), f"LoRA param names differ: {set(sd1) ^ set(sd2)}"
    for name, t1 in sd1.items():
        assert torch.allclose(t1.to(sd2[name].device), sd2[name], atol=0.0), f"mismatch on {name}"

    # Forward-output parity: w2 must produce the same outputs as w1 on the same input.
    # atol=1e-4, rtol=1e-4 accommodates 4-bit dequant rounding after the
    # quantize->save->reload cycle; on CC < 8.0 bfloat16 is coerced to float16
    # compute_dtype so tolerances apply to fp16 arithmetic.
    torch.manual_seed(0)
    w2.eval()
    _images2 = torch.zeros(
        1, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, device="cuda", dtype=torch.float32
    )
    _prompts2 = [TextPrompts(classes=["object"])]
    with torch.no_grad():
        _out_w2 = w2(_images2, _prompts2, support=None)
    _out_w2_cpu = {k: v.detach().cpu() for k, v in _out_w2.items() if isinstance(v, torch.Tensor)}
    assert set(_out_w1_cpu) == set(_out_w2_cpu), (
        f"forward output keys differ: {set(_out_w1_cpu) ^ set(_out_w2_cpu)}"
    )
    for _k in _out_w1_cpu:
        assert torch.allclose(_out_w1_cpu[_k], _out_w2_cpu[_k], atol=1e-4, rtol=1e-4), (
            f"forward output mismatch on '{_k}' after load_qlora roundtrip; "
            f"max abs diff={(_out_w1_cpu[_k] - _out_w2_cpu[_k]).abs().max().item():.6f}"
        )


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


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_apply_qlora_vision_scope_targets_only_vision_backbone() -> None:
    """T6 per spec §6.1: mirror of T5 for QLoRA scope='vision'."""
    w = load_sam31(ModelConfig())
    apply_qlora(w, PEFTConfig(method="qlora", scope="vision"))

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]
    assert lora_names, "no lora_ params after apply_qlora(scope='vision')"
    assert any("vision_backbone" in n for n in lora_names), (
        f"no vision-trunk LoRA targets at scope='vision': {lora_names[:5]}"
    )
    assert all("transformer.decoder" not in n for n in lora_names), (
        f"transformer.decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'transformer.decoder' in n][:5]}"
    )
    assert all("mask_decoder" not in n for n in lora_names), (
        f"mask_decoder targets present at scope='vision' (should be excluded): "
        f"{[n for n in lora_names if 'mask_decoder' in n][:5]}"
    )


@pytest.mark.skipif(not _bnb_available(), reason="bitsandbytes not installed")
def test_inproj_qlora_coexists_attaches_merges_vision_decoder_concept() -> None:
    """#230 §7.2, §10.4: under QLoRA the bf16 MHA in_proj LoRA and the
    Linear4bit module LoRA coexist, forward, and merge in ONE PeftModel.

    Mechanism: §7.3(a) SELECTED — apply_qlora with scope="vision_decoder_concept"
    calls _resolve_mha_modules (imported from lora.py), unions the matched
    ca_text/self_attn MHA module names into target_modules, and passes a single
    LoraConfig (dropout=0.05) to get_peft_model. peft dispatches:
      - Linear4bit modules -> lora.Linear (generic module axis)
      - nn.MultiheadAttention ca_text/self_attn -> lora.MultiheadAttention
        (MHA axis; adapting in_proj_weight + out_proj; MHA stays unquantized
        via _mha_exclusion_types, so the MHA LoRA is plain bf16 even in QLoRA)

    Asserts (§10.4):
      1. LoRA params exist for ca_text in_proj (...ca_text.lora_A, no 'out_proj').
      2. LoRA params exist for self_attn in_proj (...self_attn.lora_A, no 'out_proj').
      3. LoRA params exist for ca_text out_proj via MHA wrapper
         (...ca_text.base_layer.out_proj.lora_A).
      4. The LoraConfig carries lora_dropout=0.05 (§7.3a — ParamWrapper crashes
         on dropout != 0; lora.MultiheadAttention supports it).
      5. A forward pass runs without error.
      6. Trainable ratio stays under the 5% budget (§8.3 empirical confirmation).
      7. merge_lora folds both axes without error (QLoRA coexistence requirement,
         §7.2; merge emits only a benign NF4-rounding UserWarning).
    """
    from peft import LoraConfig

    w = load_sam31(ModelConfig())
    cfg = PEFTConfig(method="qlora", scope="vision_decoder_concept")
    apply_qlora(w, cfg)

    lora_names = [n for n, _ in w.model.model.named_parameters() if "lora_" in n]

    # 1. ca_text in_proj LoRA present.
    ca_text_inproj = [n for n in lora_names if "ca_text" in n and "out_proj" not in n]
    assert any("lora_A" in n for n in ca_text_inproj), (
        f"no ca_text in_proj lora_A (qlora): {lora_names[:12]}"
    )

    # 2. self_attn in_proj LoRA present.
    self_attn_inproj = [n for n in lora_names if "self_attn" in n and "out_proj" not in n]
    assert any("lora_A" in n for n in self_attn_inproj), (
        f"no self_attn in_proj lora_A (qlora): {lora_names[:12]}"
    )

    # 3. ca_text out_proj LoRA present via MHA wrapper.
    assert any("ca_text.base_layer.out_proj.lora_A" in n for n in lora_names), (
        f"no ca_text.base_layer.out_proj.lora_A (qlora): "
        f"{[n for n in lora_names if 'ca_text' in n]}"
    )

    # 4. LoraConfig carries dropout=0.05.
    peft_cfg: LoraConfig = w.model.model.peft_config["default"]  # type: ignore[index]
    assert peft_cfg.lora_dropout == 0.05, f"expected lora_dropout=0.05, got {peft_cfg.lora_dropout}"

    # 5. Forward runs.
    from custom_sam_peft.data.base import TextPrompts

    w.eval()
    images = torch.zeros(1, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, device="cuda", dtype=torch.float32)
    prompts = [TextPrompts(classes=["object"])]
    with torch.no_grad():
        out = w(images, prompts, support=None)
    assert out is not None

    # 6. Trainable ratio under 5% budget (§8.3 empirical confirmation).
    trainable = sum(p.numel() for p in w.model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.model.parameters())
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.2%} exceeds 5% budget"

    # 7. Merge folds both axes without error (only a benign NF4-rounding
    # UserWarning is expected; Linear4bit modules legitimately remain after merge).
    merge_lora(w)
    assert w.peft_model is None
    assert w.model.model is not None
