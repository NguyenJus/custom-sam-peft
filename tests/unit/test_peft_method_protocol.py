"""TDD tests for the PEFTMethod protocol (Task 4.1).

These tests cover:
  1. Protocol structure — runtime_checkable, has all four methods.
  2. LoraAdapter implements protocol with correct return values.
  3. QloraAdapter implements protocol with correct return values.
  4. detect_method_from_checkpoint path logic.
  5. Source-level assertion that trainer/loop/checkpoint/runner no longer
     branch on cfg.peft.method strings.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from custom_sam_peft.peft_adapters import (
    LoraAdapter,
    PEFTMethod,
    QloraAdapter,
    make_peft_method,
)

# ---------------------------------------------------------------------------
# 1. Protocol structure
# ---------------------------------------------------------------------------


def test_peft_method_is_runtime_checkable() -> None:
    """isinstance() checks must work on PEFTMethod (runtime_checkable Protocol)."""
    lora = LoraAdapter()
    assert isinstance(lora, PEFTMethod)


def test_peft_method_protocol_declares_recommended_optimizer() -> None:
    assert hasattr(PEFTMethod, "recommended_optimizer")


def test_peft_method_protocol_declares_disables_outer_autocast() -> None:
    assert hasattr(PEFTMethod, "disables_outer_autocast")


def test_peft_method_protocol_declares_detect_method_from_checkpoint() -> None:
    assert hasattr(PEFTMethod, "detect_method_from_checkpoint")


def test_peft_method_protocol_declares_supports_checkpoint_load_from_disk() -> None:
    assert hasattr(PEFTMethod, "supports_checkpoint_load_from_disk")


# ---------------------------------------------------------------------------
# 2. LoraAdapter
# ---------------------------------------------------------------------------


def test_lora_adapter_implements_protocol() -> None:
    assert isinstance(LoraAdapter(), PEFTMethod)


def test_lora_adapter_recommended_optimizer() -> None:
    assert LoraAdapter().recommended_optimizer() == "adamw"


def test_lora_adapter_disables_outer_autocast_false() -> None:
    assert LoraAdapter().disables_outer_autocast() is False


def test_lora_adapter_supports_checkpoint_load_from_disk_true() -> None:
    assert LoraAdapter().supports_checkpoint_load_from_disk() is True


def test_lora_adapter_detect_method_no_meta_file(tmp_path: Path) -> None:
    """Without the qlora JSON marker, detect_method_from_checkpoint returns 'lora'."""
    result = LoraAdapter().detect_method_from_checkpoint(tmp_path)
    assert result == "lora"


def test_lora_adapter_detect_method_raises_on_qlora_marker(tmp_path: Path) -> None:
    """If qlora JSON marker is present, LoraAdapter raises CheckpointError."""
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    from custom_sam_peft.errors import CheckpointError

    with pytest.raises(CheckpointError):
        LoraAdapter().detect_method_from_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# 3. QloraAdapter
# ---------------------------------------------------------------------------


def test_qlora_adapter_implements_protocol() -> None:
    assert isinstance(QloraAdapter(), PEFTMethod)


def test_qlora_adapter_recommended_optimizer() -> None:
    assert QloraAdapter().recommended_optimizer() == "adamw8bit"


def test_qlora_adapter_disables_outer_autocast_true() -> None:
    assert QloraAdapter().disables_outer_autocast() is True


def test_qlora_adapter_supports_checkpoint_load_from_disk_false() -> None:
    assert QloraAdapter().supports_checkpoint_load_from_disk() is False


def test_qlora_adapter_detect_method_with_meta_file(tmp_path: Path) -> None:
    """With the qlora JSON marker present, detect_method_from_checkpoint returns 'qlora'."""
    (tmp_path / "custom_sam_peft_qlora.json").write_text("{}")
    result = QloraAdapter().detect_method_from_checkpoint(tmp_path)
    assert result == "qlora"


def test_qlora_adapter_detect_method_raises_without_meta_file(tmp_path: Path) -> None:
    """Without the qlora JSON marker, QloraAdapter raises CheckpointError."""
    from custom_sam_peft.errors import CheckpointError

    with pytest.raises(CheckpointError):
        QloraAdapter().detect_method_from_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# 4. make_peft_method factory
# ---------------------------------------------------------------------------


def test_make_peft_method_lora() -> None:
    adapter = make_peft_method("lora")
    assert isinstance(adapter, LoraAdapter)
    assert isinstance(adapter, PEFTMethod)


def test_make_peft_method_qlora() -> None:
    adapter = make_peft_method("qlora")
    assert isinstance(adapter, QloraAdapter)
    assert isinstance(adapter, PEFTMethod)


def test_make_peft_method_unknown_raises() -> None:
    with pytest.raises(ValueError, match=r"Unknown peft\.method"):
        make_peft_method("unknown")


# ---------------------------------------------------------------------------
# 5. Source-level branch checks — trainer / loop / checkpoint / runner
#    must NOT contain `.method ==` string comparisons.
# ---------------------------------------------------------------------------


def test_trainer_does_not_branch_on_method_name() -> None:
    """train/trainer.py must contain no cfg.peft.method == branch."""
    from custom_sam_peft.train import trainer

    src = inspect.getsource(trainer)
    assert ".method ==" not in src, (
        "train/trainer.py must not branch on cfg.peft.method; "
        "use peft_method_instance.recommended_optimizer() instead."
    )


def test_loop_does_not_branch_on_method_name() -> None:
    """train/loop.py must contain no cfg.peft.method == branch."""
    from custom_sam_peft.train import loop

    src = inspect.getsource(loop)
    assert ".method ==" not in src, (
        "train/loop.py must not branch on cfg.peft.method; "
        "use peft_method_instance.disables_outer_autocast() instead."
    )


def test_checkpoint_does_not_branch_on_method_name() -> None:
    """train/checkpoint.py must contain no cfg.peft.method == comparison."""
    from custom_sam_peft.train import checkpoint

    src = inspect.getsource(checkpoint)
    # `save_full_state` records `cfg.peft.method` as a plain attribute read —
    # that is allowed. What is not allowed is an equality branch on the string.
    assert "cfg.peft.method ==" not in src and "cfg.peft.method !=" not in src, (
        "train/checkpoint.py must not branch on cfg.peft.method ==; "
        "use peft_method_instance.detect_method_from_checkpoint() instead."
    )


def test_eval_runner_does_not_branch_on_method_name() -> None:
    """eval/runner.py must contain no cfg.peft.method == branch."""
    from custom_sam_peft.eval import runner

    src = inspect.getsource(runner)
    assert ".method ==" not in src, (
        "eval/runner.py must not branch on cfg.peft.method; "
        "use peft_method_instance.supports_checkpoint_load_from_disk() instead."
    )
