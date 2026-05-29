"""Re-export shim for the trainer subsystem's OOM types.

`OomEvent` now lives in `custom_sam_peft.oom` (the shared OOM ladder module);
this module re-exports it so existing imports
(`from custom_sam_peft.train.types import OomEvent`) keep working unchanged.

Spec: docs/superpowers/specs/2026-05-29-unified-oom-ladder-design.md §4.
"""

from __future__ import annotations

from custom_sam_peft.oom import OomEvent

__all__ = ["OomEvent"]
