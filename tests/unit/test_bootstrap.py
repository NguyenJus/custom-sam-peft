"""custom_sam_peft._bootstrap imports every registrant so the registry is populated."""

from __future__ import annotations

import sys

from custom_sam_peft._registry import _REGISTRY, list_registered, reset_registry


def test_bootstrap_populates_all_kinds() -> None:
    mods_to_evict = (
        "custom_sam_peft.data",
        "custom_sam_peft.data.coco",
        "custom_sam_peft.data.hf",
        "custom_sam_peft.peft_adapters",
        "custom_sam_peft.peft_adapters.lora",
        "custom_sam_peft.peft_adapters.qlora",
        "custom_sam_peft.tracking",
        "custom_sam_peft.tracking.noop",
        "custom_sam_peft.tracking.tensorboard",
        "custom_sam_peft.tracking.wandb",
        "custom_sam_peft._bootstrap",
    )

    # Snapshot state before mutation.
    saved_mods = {m: sys.modules[m] for m in mods_to_evict if m in sys.modules}
    saved_registry: dict[str, dict] = {k: dict(v) for k, v in _REGISTRY.items()}

    # Also capture the package-level submodule attributes that Python sets when
    # sub-packages are imported (e.g. importing custom_sam_peft.data.hf sets
    # custom_sam_peft.data.hf = <module>). When we restore sys.modules, we must
    # also repair these attributes so that `import custom_sam_peft.data.hf` from
    # sibling tests still resolves to the original object.
    _pkg_attr_snapshot: list[tuple[object, str, object]] = []
    for dotted in mods_to_evict:
        parts = dotted.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, attr):
                _pkg_attr_snapshot.append((parent, attr, getattr(parent, attr)))

    try:
        reset_registry()
        for m in mods_to_evict:
            if m in sys.modules:
                del sys.modules[m]

        import custom_sam_peft._bootstrap  # noqa: F401  # triggers @register decorators

        assert set(list_registered("dataset")) >= {"coco", "hf"}
        assert set(list_registered("peft")) >= {"lora", "qlora"}
        assert set(list_registered("tracker")) >= {"tensorboard", "wandb", "none"}
    finally:
        # Restore so sibling tests see the original module/function objects.
        for m in mods_to_evict:
            if m in sys.modules:
                del sys.modules[m]
        sys.modules.update(saved_mods)
        _REGISTRY.clear()
        _REGISTRY.update(saved_registry)
        # Repair package-level submodule attributes so that `import custom_sam_peft.data.hf`
        # in sibling tests resolves through the package attribute chain correctly.
        for parent, attr, orig_val in _pkg_attr_snapshot:
            setattr(parent, attr, orig_val)
