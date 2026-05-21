"""Pins Meta's `geometric_prompt` Prompt layout (Task-0 adapted).

Layout contract (docs/superpowers/plans/2026-05-17-training-loop-notes.md):
  - box_embeddings: (N_boxes, B, 4) float, normalized cxcywh in [0, 1]
  - box_mask: (B, N_boxes) bool, True = padded (PyTorch key-padding)
  - Returns None when all hints are None (Choice A — adapter substitutes dummy).
"""

from __future__ import annotations

import torch

from custom_sam_peft.models.sam3 import _build_geometric_prompt


def test_all_none_returns_none() -> None:
    out = _build_geometric_prompt([None, None, None], image_size=1008, device=torch.device("cpu"))
    assert out is None


def test_single_image_with_hints_returns_prompt() -> None:
    # batch of 3: images 0 and 2 have no hints, image 1 has 1 box
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])  # (1, 4) xyxy pixel
    out = _build_geometric_prompt([None, boxes, None], image_size=1008, device=torch.device("cpu"))
    assert out is not None

    # box_embeddings: (N_boxes=1, B=3, 4)
    assert out.box_embeddings.shape == (1, 3, 4)
    # box_mask: (B=3, N_boxes=1)
    assert out.box_mask.shape == (3, 1)

    # image 1 (index 1) has a real hint → mask is False (not padded)
    assert out.box_mask[1, 0].item() is False
    # images 0 and 2 are padded → mask is True
    assert out.box_mask[0, 0].item() is True
    assert out.box_mask[2, 0].item() is True

    # pixel xyxy (10, 20, 50, 80) → normalized cxcywh at image_size=1008:
    #   cx = (10+50)/2 / 1008 = 30/1008
    #   cy = (20+80)/2 / 1008 = 50/1008
    #   w  = (50-10)   / 1008 = 40/1008
    #   h  = (80-20)   / 1008 = 60/1008
    expected = torch.tensor([30 / 1008, 50 / 1008, 40 / 1008, 60 / 1008])
    assert torch.allclose(out.box_embeddings[0, 1, :], expected, atol=1e-6)


def test_padding_slots_have_zero_embeddings() -> None:
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    out = _build_geometric_prompt([None, boxes, None], image_size=1008, device=torch.device("cpu"))
    assert out is not None
    # padded images (0 and 2) get zero-filled embeddings
    assert torch.allclose(out.box_embeddings[0, 0, :], torch.zeros(4))
    assert torch.allclose(out.box_embeddings[0, 2, :], torch.zeros(4))


def test_multiple_boxes_per_image() -> None:
    boxes_a = torch.tensor([[0.0, 0.0, 100.0, 100.0], [200.0, 200.0, 400.0, 400.0]])  # 2 boxes
    boxes_b = torch.tensor([[50.0, 50.0, 150.0, 150.0]])  # 1 box
    out = _build_geometric_prompt([boxes_a, boxes_b], image_size=1008, device=torch.device("cpu"))
    assert out is not None

    # N_max = 2, B = 2
    assert out.box_embeddings.shape == (2, 2, 4)
    assert out.box_mask.shape == (2, 2)

    # image 0: 2 real hints → mask all False
    assert out.box_mask[0, 0].item() is False
    assert out.box_mask[0, 1].item() is False
    # image 1: 1 real hint, 1 padded → first False, second True (right-padded)
    assert out.box_mask[1, 0].item() is False
    assert out.box_mask[1, 1].item() is True


def test_device_placement() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Verify the GPU is actually usable with this PyTorch build (sm_61 GPUs
    # fail with "no kernel image" against PyTorch builds targeting sm_75+).
    try:
        torch.zeros(1, device="cuda")
    except Exception:
        pytest.skip("CUDA device not compatible with current PyTorch build")
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = _build_geometric_prompt([boxes], image_size=1008, device=torch.device("cuda"))
    assert out is not None
    assert out.box_embeddings.device.type == "cuda"
    assert out.box_mask.device.type == "cuda"
