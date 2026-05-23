"""Bootstrap hook — sole site for plugin registration, seeding, and logging setup.

Call ``bootstrap()`` once at process start (before any plugin lookup).  The
CLI entry-point calls it automatically; notebook / library callers may invoke
it directly::

    from custom_sam_peft._bootstrap import bootstrap
    bootstrap(seed=42, log_level="DEBUG")

Module-level imports below fire every ``@register(...)`` decorator so the
registry is populated the first time this module is loaded.  ``bootstrap()``
therefore only needs to trigger those imports once — additional calls are
idempotent with respect to registration because the modules are cached in
``sys.modules`` after the first import.
"""

from __future__ import annotations

import logging
import random

import torch

# ---------------------------------------------------------------------------
# Registration side-effects
# Each of these imports causes its @register decorator to fire once.
# ---------------------------------------------------------------------------
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

_LOG = logging.getLogger(__name__)


def bootstrap(*, seed: int | None = None, log_level: str = "INFO") -> None:
    """Initialise the runtime environment.

    Safe to call more than once; subsequent calls are no-ops unless *seed*
    or *log_level* differ from the first call (in which case they silently
    win — callers should coordinate a single call site).

    Parameters
    ----------
    seed:
        Integer seed for :func:`random.seed`, :func:`numpy.random.seed`, and
        :func:`torch.manual_seed`.  When *None* the RNG state is left as-is.
    log_level:
        Root-logger level string (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        Passed to :func:`logging.basicConfig` with ``force=True`` so it
        overrides any prior configuration set by test harnesses or other
        libraries.
    """
    # --- Logging -----------------------------------------------------------
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # --- Seeding -----------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np  # optional dependency

            np.random.seed(seed)
        except ModuleNotFoundError:
            # numpy is optional — skip its seed when absent.
            pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _LOG.debug("bootstrap: seeded RNG with seed=%d", seed)

    # Registration was already performed by the module-level imports above.
    _LOG.debug("bootstrap: complete (log_level=%s, seed=%s)", log_level, seed)
