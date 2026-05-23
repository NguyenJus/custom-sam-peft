"""Exception taxonomy for custom_sam_peft.

The CLI boundary (cli/main.py::main) catches CustomSamPeftError and
renders a user-facing four-part message. Internals raise typed
exceptions and never catch-and-re-raise as RuntimeError mid call-graph.
"""

from __future__ import annotations


class CustomSamPeftError(Exception):
    """Base class for all user-facing errors raised by this package."""

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        found: str | None = None,
        fix: str | None = None,
    ) -> None:
        super().__init__(message)
        self._expected = expected
        self._found = found
        self._fix = fix

    @property
    def expected(self) -> str | None:
        return self._expected

    @property
    def found(self) -> str | None:
        return self._found

    @property
    def fix(self) -> str | None:
        return self._fix


class ConfigError(CustomSamPeftError):
    """Raised when a config value is missing, malformed, or invalid."""

    def __init__(
        self,
        message: str,
        *,
        field_path: str,
        expected: str | None = None,
        found: str | None = None,
        fix: str | None = None,
    ) -> None:
        super().__init__(
            f"{message} (field: {field_path})",
            expected=expected,
            found=found,
            fix=fix,
        )
        self.field_path = field_path


class DataError(CustomSamPeftError):
    """Raised for dataset-loading or example-decoding failures."""


class ModelError(CustomSamPeftError):
    """Raised for model construction, patch-application, or adapter failures."""


class CheckpointError(CustomSamPeftError):
    """Raised for checkpoint read/write or resume-state mismatches."""


class EnvironmentError(CustomSamPeftError):
    """Raised when a runtime precondition fails (HF gating, missing GPU, missing extra)."""

    def __init__(
        self,
        message: str,
        *,
        precondition: str,
        expected: str | None = None,
        found: str | None = None,
        fix: str | None = None,
    ) -> None:
        super().__init__(
            f"{message} (precondition: {precondition})",
            expected=expected,
            found=found,
            fix=fix,
        )
        self.precondition = precondition
