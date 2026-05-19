"""Direct tests for esam3.utils.huggingface — resolve_hf_token + download_model.

All HuggingFace Hub calls are mocked. No network access in any test.
"""

from __future__ import annotations

import pytest

from esam3.utils.huggingface import resolve_hf_token


def test_resolve_hf_token_explicit_arg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: "cache-token",
    )
    assert resolve_hf_token("explicit") == "explicit"


def test_resolve_hf_token_env_wins_over_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: "cache-token",
    )
    assert resolve_hf_token() == "env-token"


def test_resolve_hf_token_cache_used_when_no_arg_or_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: "cache-token",
    )
    assert resolve_hf_token() == "cache-token"


def test_resolve_hf_token_returns_none_when_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: None,
    )
    assert resolve_hf_token() is None
