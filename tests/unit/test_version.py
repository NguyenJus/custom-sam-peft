"""Smoke test that the hatch-vcs build hook populated __version__."""

from packaging.version import Version

import custom_sam_peft


def test_version_is_valid_pep440() -> None:
    assert isinstance(custom_sam_peft.__version__, str)
    assert custom_sam_peft.__version__, "__version__ must not be empty"
    # Raises InvalidVersion if not parseable.
    Version(custom_sam_peft.__version__)
