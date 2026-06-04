"""CPU-only tests for the encoder RoPE complex-op swap (spec §Testing, issue #279).

The TinySam3Stub has no real vitdet RoPE attention and structurally cannot
exercise this path.  These tests construct the real ``sam3.model.vitdet.Attention``
on CPU (no checkpoint, no GPU) and cover the four spec-mandated cases:

  (a) Patch correctness / bit-exactness at fp32.
  (b) ONNX export succeeds with no view_as_complex/complex op in the graph.
  (c) VE-RoPE module raises VeRopeUnsupportedError (fail loud).
  (d) Guard not vacuous: corrupted real table triggers RopeEquivalenceError.
"""

from __future__ import annotations

import io

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 64
_NUM_HEADS = 4
_INPUT_SIZE = (8, 8)
# head_dim = 64 // 4 = 16; L = 8*8 = 64
_HEAD_DIM = _DIM // _NUM_HEADS
_L = _INPUT_SIZE[0] * _INPUT_SIZE[1]
_B = 1


def _make_attention(*, use_rope_real: bool = False, use_ve_rope: bool = False):
    """Construct a real sam3 vitdet.Attention on CPU."""
    from sam3.model.vitdet import Attention

    return Attention(
        dim=_DIM,
        num_heads=_NUM_HEADS,
        use_rope=True,
        use_rope_real=use_rope_real,
        use_ve_rope=use_ve_rope,
        input_size=_INPUT_SIZE,
    ).eval()


def _make_qk(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic q, k in (B, H, L, head_dim) shape."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(_B, _NUM_HEADS, _L, _HEAD_DIM, generator=gen)
    k = torch.randn(_B, _NUM_HEADS, _L, _HEAD_DIM, generator=gen)
    return q, k


# ---------------------------------------------------------------------------
# (a) Patch correctness / bit-exactness
# ---------------------------------------------------------------------------


def test_patch_correctness_bit_exact() -> None:
    """Patch flips use_rope_real, registers real buffers, output is bit-exact at fp32.

    Spec: spike showed max|Δ| == 0.0 at fp32; 1e-5 is the durable gate tolerance
    (spec §Spike findings, §Per-module guard).
    """
    from custom_sam_peft.export.onnx import _patch_encoder_rope_for_export

    module = _make_attention()
    assert not module.use_rope_real

    q, k = _make_qk()
    with torch.no_grad():
        q_pre, k_pre = module._apply_rope(q.clone(), k.clone())

    # Wrap in a tiny container so the walker sees it.
    container = torch.nn.Sequential(module)
    n = _patch_encoder_rope_for_export(container)

    assert n == 1, f"expected 1 module patched, got {n}"
    assert module.use_rope_real is True, "use_rope_real should be True after patch"
    assert hasattr(module, "freqs_cis_real"), "freqs_cis_real not registered"
    assert hasattr(module, "freqs_cis_imag"), "freqs_cis_imag not registered"
    # Complex buffer must still be present (not None) — _apply_rope asserts non-None.
    assert module.freqs_cis is not None, "freqs_cis must not be None after patch"

    with torch.no_grad():
        q_post, k_post = module._apply_rope(q.clone(), k.clone())

    max_delta_q = (q_post - q_pre).abs().max().item()
    max_delta_k = (k_post - k_pre).abs().max().item()
    # Spike finding: bit-exact at fp32; 1e-5 is the durable gate (spec §Spike findings).
    assert max_delta_q == 0.0, f"q: expected bit-exact (max|Δ|==0.0), got {max_delta_q}"
    assert max_delta_k == 0.0, f"k: expected bit-exact (max|Δ|==0.0), got {max_delta_k}"
    # Also assert via allclose at the documented tight tolerance (spec §Per-module guard).
    assert torch.allclose(q_post, q_pre, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_post, k_pre, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# (b) ONNX export succeeds with no complex op after patch
# ---------------------------------------------------------------------------


def test_onnx_export_no_complex_op() -> None:
    """After patch, torch.onnx.export succeeds (opset 17) with no view_as_complex in graph.

    Spec: the exact regression #279 is about aten::view_as_complex being
    unsupported in the TorchScript ONNX exporter at any opset.
    """
    from custom_sam_peft.export.onnx import _patch_encoder_rope_for_export

    module = _make_attention()
    container = torch.nn.Sequential(module)
    _patch_encoder_rope_for_export(container)

    # Build a minimal shim that calls _apply_rope then a dummy proj so the graph
    # exercises the RoPE path.  Attention.forward expects a 4-D spatial input, so
    # we trace through it directly.

    class _RopeShim(torch.nn.Module):
        def __init__(self, attn) -> None:
            super().__init__()
            self.attn = attn

        def forward(self, q: torch.Tensor, k: torch.Tensor):
            return self.attn._apply_rope(q, k)

    shim = _RopeShim(module).eval()
    q, k = _make_qk()
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            shim,
            (q, k),
            buf,
            opset_version=17,
            dynamo=False,
            input_names=["q", "k"],
            output_names=["q_out", "k_out"],
        )

    buf.seek(0)
    import onnx

    model_proto = onnx.load(buf)
    graph_ops = {node.op_type for node in model_proto.graph.node}
    assert "view_as_complex" not in graph_ops, (
        f"view_as_complex found in graph after patch; ops present: {graph_ops}"
    )
    # Also guard against any complex-typed node surfacing.
    for node in model_proto.graph.node:
        assert "complex" not in node.op_type.lower(), (
            f"Unexpected complex op in patched graph: {node.op_type}"
        )


# ---------------------------------------------------------------------------
# (c) VE-RoPE fails loud with VeRopeUnsupportedError
# ---------------------------------------------------------------------------


def test_ve_rope_raises_unsupported() -> None:
    """A module with use_ve_rope=True causes _patch_encoder_rope_for_export to raise
    VeRopeUnsupportedError (spec §Error handling).
    """
    from custom_sam_peft.export.onnx import VeRopeUnsupportedError, _patch_encoder_rope_for_export

    module = _make_attention(use_ve_rope=True)
    container = torch.nn.Sequential(module)
    with pytest.raises(VeRopeUnsupportedError):
        _patch_encoder_rope_for_export(container)


# ---------------------------------------------------------------------------
# (d) Guard not vacuous — corrupted real table raises RopeEquivalenceError
# ---------------------------------------------------------------------------


def test_equivalence_guard_fires_on_corrupt_table() -> None:
    """If the regenerated freqs_cis_real is zeroed, the per-module guard fires.

    Spec §Guard is not vacuous: corrupt the regenerated real table so the real path
    diverges from the complex reference → RopeEquivalenceError must be raised.
    """
    import custom_sam_peft.export.onnx as onnx_mod
    from custom_sam_peft.export.onnx import RopeEquivalenceError

    module = _make_attention()
    container = torch.nn.Sequential(module)

    # Monkeypatch _setup_rope_freqs on this specific instance to zero the real table
    # after it has been registered, simulating a corrupt regeneration.
    original_setup = module._setup_rope_freqs

    def _corrupt_setup():
        original_setup()
        # Zero out the real frequencies so the real path diverges from complex.
        module.freqs_cis_real = torch.zeros_like(module.freqs_cis_real)
        module.freqs_cis_imag = torch.zeros_like(module.freqs_cis_imag)

    module._setup_rope_freqs = _corrupt_setup  # type: ignore[method-assign]

    with pytest.raises(RopeEquivalenceError):
        onnx_mod._patch_encoder_rope_for_export(container)
