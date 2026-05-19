"""Import every @register site once so the registry is populated.

CLI entry imports this module at load time; library callers (notebooks)
may import it explicitly if they need plugin lookup."""

from esam3.data import (
    coco,  # noqa: F401
    hf,  # noqa: F401
)
from esam3.peft_adapters import (
    lora,  # noqa: F401
    qlora,  # noqa: F401
)
from esam3.tracking import (
    noop,  # noqa: F401
    tensorboard,  # noqa: F401
    wandb,  # noqa: F401
)
