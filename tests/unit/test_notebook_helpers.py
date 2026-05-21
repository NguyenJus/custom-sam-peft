"""Tests for src/custom_sam_peft/notebook_helpers.py."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from custom_sam_peft.notebook_helpers import (
    check_local_checkpoint,
    detect_env,
    resolve_hf_token,
)

# ---- detect_env ----------------------------------------------------------


def test_detect_env_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_GPU", "1")
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert detect_env() == "colab"


def test_detect_env_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123")
    assert detect_env() == "runpod"


def test_detect_env_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert detect_env() == "unknown"


def test_detect_env_colab_wins_over_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_GPU", "1")
    monkeypatch.setenv("RUNPOD_POD_ID", "abc123")
    assert detect_env() == "colab"


# ---- check_local_checkpoint ----------------------------------------------


def test_check_local_checkpoint_present(tmp_path: Path) -> None:
    (tmp_path / "sam3.1_multiplex.pt").write_bytes(b"x")
    assert check_local_checkpoint(tmp_path, "sam3.1_multiplex.pt") is True


def test_check_local_checkpoint_absent(tmp_path: Path) -> None:
    assert check_local_checkpoint(tmp_path, "sam3.1_multiplex.pt") is False


def test_check_local_checkpoint_dir_not_file(tmp_path: Path) -> None:
    (tmp_path / "sam3.1_multiplex.pt").mkdir()
    assert check_local_checkpoint(tmp_path, "sam3.1_multiplex.pt") is False


# ---- resolve_hf_token ----------------------------------------------------


@pytest.mark.parametrize("env", ["colab", "runpod", "unknown"])
def test_resolve_hf_token_local_short_circuits(env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # The function MUST NOT read env or import google.colab when local_present.
    assert resolve_hf_token(env, local_present=True) is None


def test_resolve_hf_token_missing_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        resolve_hf_token("colab", local_present=False)


def test_resolve_hf_token_missing_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Environment Variables"):
        resolve_hf_token("runpod", local_present=False)


def test_resolve_hf_token_missing_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="shell environment"):
        resolve_hf_token("unknown", local_present=False)


def test_resolve_hf_token_runpod_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_runpod_xyz")
    assert resolve_hf_token("runpod", local_present=False) == "hf_runpod_xyz"


def test_resolve_hf_token_unknown_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_shell_xyz")
    assert resolve_hf_token("unknown", local_present=False) == "hf_shell_xyz"


def test_resolve_hf_token_colab_userdata_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub google.colab.userdata so the colab arm returns the token."""
    fake_userdata = SimpleNamespace(get=lambda _key: "hf_colab_abc")
    fake_colab = SimpleNamespace(userdata=fake_userdata)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(colab=fake_colab))
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    assert resolve_hf_token("colab", local_present=False) == "hf_colab_abc"


def test_resolve_hf_token_colab_userdata_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub userdata.get → None → still surfaces the 'Colab Secrets' message."""

    def _get(_key: str) -> Any:
        return None

    fake_colab = SimpleNamespace(userdata=SimpleNamespace(get=_get))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(colab=fake_colab))
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        resolve_hf_token("colab", local_present=False)
