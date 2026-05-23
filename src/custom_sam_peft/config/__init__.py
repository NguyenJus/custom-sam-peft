"""Config package — single entry point for loading and validating YAML configs.

The only config-loading function is `load_config`. All CLI commands, notebook
helpers, and tests should funnel through it. No downstream module re-parses YAML.

Usage:
    from custom_sam_peft.config import load_config
    cfg = load_config("path/to/config.yaml", overrides=["train.epochs=5"])
"""

from custom_sam_peft.config.loader import ConfigError, apply_overrides, load_config
from custom_sam_peft.config.schema import TrainConfig

__all__ = ["ConfigError", "TrainConfig", "apply_overrides", "load_config"]
