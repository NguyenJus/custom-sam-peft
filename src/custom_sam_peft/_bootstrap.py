"""Import every @register site once so the registry is populated.

CLI entry imports this module at load time; library callers (notebooks)
may import it explicitly if they need plugin lookup."""

from custom_sam_peft.data import (
    coco,  # noqa: F401
    hf,  # noqa: F401
)
from custom_sam_peft.peft_adapters import (
    lora,  # noqa: F401
    qlora,  # noqa: F401
)
from custom_sam_peft.tracking import (
    noop,  # noqa: F401
    tensorboard,  # noqa: F401
    wandb,  # noqa: F401
)
