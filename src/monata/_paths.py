"""Shared path/name validation helpers for filesystem-backed APIs."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_path_segment(value: Any, label: str = "name") -> str:
    """Return a safe single path segment or raise a stable ValueError."""

    if value is None:
        raise ValueError(f"{label} must be a single safe path segment: {value}")
    text = str(value)
    path = Path(text)
    if (
        not text
        or _SAFE_PATH_SEGMENT_RE.match(text) is None
        or path.is_absolute()
        or len(path.parts) != 1
        or text in {".", ".."}
    ):
        raise ValueError(f"{label} must be a single safe path segment: {text}")
    return text


def expand_path(path: str | Path) -> Path:
    """Return a path with environment variables and user markers expanded."""

    return Path(os.path.expandvars(str(path))).expanduser().absolute()


def walk_files(path: str | Path, *, follow_symlinks: bool = False) -> Iterator[Path]:
    """Yield files below a path in deterministic traversal order."""

    root = Path(path)
    if root.is_file():
        yield root
        return

    for current_root, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames.sort()
        for filename in sorted(filenames):
            yield Path(current_root) / filename


def find_file(file_name: str | Path, directories: Iterable[str | Path]) -> Path:
    """Return the first matching file name found under the provided directories."""

    name = str(file_name)
    if not name or Path(name).name != name:
        raise ValueError(f"file_name must be a file name: {file_name}")

    roots = tuple(directories)
    for directory in roots:
        for path in walk_files(directory):
            if path.name == name:
                return path

    searched = ", ".join(str(directory) for directory in roots) or "<none>"
    raise FileNotFoundError(f"file {name!r} not found under: {searched}")


def toml_string(value: Any) -> str:
    """Escape a value for a basic TOML string."""

    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
