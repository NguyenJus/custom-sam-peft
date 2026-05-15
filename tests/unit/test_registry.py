"""Tests for the plugin registry."""

from __future__ import annotations

import pytest

from esam3._registry import (
    RegistryError,
    list_registered,
    lookup,
    register,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_registry()


def test_register_and_lookup_roundtrip() -> None:
    @register("dataset", "fake")
    def factory() -> str:
        return "ok"

    assert lookup("dataset", "fake") is factory
    assert factory() == "ok"


def test_lookup_unknown_raises() -> None:
    with pytest.raises(RegistryError, match="unknown 'dataset' entry 'missing'"):
        lookup("dataset", "missing")


def test_duplicate_name_raises() -> None:
    @register("tracker", "dup")
    def first() -> None:
        return None

    with pytest.raises(RegistryError, match="'tracker' entry 'dup' already registered"):

        @register("tracker", "dup")
        def second() -> None:
            return None


def test_separate_kinds_do_not_collide() -> None:
    @register("dataset", "shared")
    def a() -> str:
        return "dataset"

    @register("tracker", "shared")
    def b() -> str:
        return "tracker"

    assert lookup("dataset", "shared") is a
    assert lookup("tracker", "shared") is b


def test_list_registered_returns_names_for_kind() -> None:
    @register("peft", "lora")
    def _lora() -> None:
        return None

    @register("peft", "qlora")
    def _qlora() -> None:
        return None

    assert set(list_registered("peft")) == {"lora", "qlora"}


def test_list_registered_unknown_kind_returns_empty() -> None:
    assert list_registered("nonexistent") == []
