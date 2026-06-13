"""Parser errors with source provenance."""

from __future__ import annotations


class SpiceParseError(ValueError):
    """Raised when SPICE text cannot be parsed into a stable deck record."""

    def __init__(self, message: str, *, path: str | None = None, line: int | None = None):
        self.message = message
        self.path = path
        self.line = line
        location = _format_location(path, line)
        super().__init__(f"{location}{message}")


class UnsupportedConstructError(SpiceParseError):
    """Raised for known SPICE constructs outside the supported import subset."""


def _format_location(path: str | None, line: int | None) -> str:
    if path and line is not None:
        return f"{path}:{line}: "
    if path:
        return f"{path}: "
    if line is not None:
        return f"line {line}: "
    return ""
