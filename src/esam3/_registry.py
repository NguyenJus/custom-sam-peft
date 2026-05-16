"""Plugin registry for dataset adapters, PEFT methods, and trackers.

Pluggable surfaces declare themselves via the @register decorator at import
time. The CLI and library look them up by (kind, name). Adding a new
implementation = one file + one @register + one test; no edits to dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable[..., object])


class RegistryError(KeyError):
    """Raised on duplicate registration or unknown lookup."""


_REGISTRY: dict[str, dict[str, Callable[..., object]]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    """Decorator: register `fn` under (kind, name)."""

    def decorator(fn: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise RegistryError(f"'{kind}' entry '{name}' already registered")
        bucket[name] = fn
        return fn

    return decorator


def lookup(kind: str, name: str) -> Callable[..., object]:
    """Return the callable registered under (kind, name)."""
    bucket = _REGISTRY.get(kind, {})
    if name not in bucket:
        raise RegistryError(f"unknown '{kind}' entry '{name}'")
    return bucket[name]


def list_registered(kind: str) -> list[str]:
    """Return the sorted names registered under `kind`. Empty list if no kind."""
    return sorted(_REGISTRY.get(kind, {}).keys())


def reset_registry() -> None:
    """Clear the registry. Test-only helper — do NOT call from production code.

    Clearing the registry wipes all dataset/peft/tracker registrations and
    will break any subsequent lookups until the relevant modules are reloaded.
    """
    _REGISTRY.clear()
