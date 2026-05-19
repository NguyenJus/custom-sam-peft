"""esam3._bootstrap imports every registrant so the registry is populated."""

from __future__ import annotations

import sys

from esam3._registry import _REGISTRY, list_registered, reset_registry


def test_bootstrap_populates_all_kinds() -> None:
    mods_to_evict = (
        "esam3.data",
        "esam3.data.coco",
        "esam3.data.hf",
        "esam3.peft_adapters",
        "esam3.peft_adapters.lora",
        "esam3.peft_adapters.qlora",
        "esam3.tracking",
        "esam3.tracking.noop",
        "esam3.tracking.tensorboard",
        "esam3.tracking.wandb",
        "esam3._bootstrap",
    )

    # Snapshot state before mutation.
    saved_mods = {m: sys.modules[m] for m in mods_to_evict if m in sys.modules}
    saved_registry: dict[str, dict] = {k: dict(v) for k, v in _REGISTRY.items()}

    # Also capture the package-level submodule attributes that Python sets when
    # sub-packages are imported (e.g. esam3.data.hf sets esam3.data.hf = <module>).
    # When we restore sys.modules, we must also repair these attributes so that
    # `import esam3.data.hf` from sibling tests still resolves to the original object.
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

        import esam3._bootstrap  # noqa: F401  # triggers @register decorators

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
        # Repair package-level submodule attributes so that `import esam3.data.hf`
        # in sibling tests resolves through the package attribute chain correctly.
        for parent, attr, orig_val in _pkg_attr_snapshot:
            setattr(parent, attr, orig_val)
