"""`esam3 doctor` — Body deferred to spec/cli."""

from __future__ import annotations

from rich import print as rprint


def doctor() -> None:
    """Report environment + dependency status."""
    rprint("[yellow]not yet implemented[/yellow] — would report CUDA, deps, VRAM, weight cache")
