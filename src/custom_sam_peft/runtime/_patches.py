"""Sam3Patches applier — single application site for all dtype patches.

Each `_patch_*` from src/custom_sam_peft/models/sam3.py is moved to its
own file under src/custom_sam_peft/models/_patches/ in Task 5.7. This
applier imports them all and runs them in deterministic order.
"""

from __future__ import annotations

from typing import Any

from custom_sam_peft.runtime._runtime import Runtime


class Sam3Patches:
    """Aggregates and applies every dtype-correctness patch to a SAM-3 model.

    Usage:
        Sam3Patches.apply(model, runtime)

    This is called exactly once per model-load, from
    models.sam3.load_sam31's `_apply_patches` step (Task 5.1).
    """

    @staticmethod
    def apply(model: Any, runtime: Runtime) -> None:
        # Task 5.7 populates this with imports from models/_patches/.
        # Order is deterministic by file name (sorted).
        from custom_sam_peft.models._patches import _ALL_PATCHES

        for patch in _ALL_PATCHES:
            patch(model, runtime)
