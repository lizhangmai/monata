"""Shared path validation helpers for data-only cellview loaders."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from monata._paths import resolve_relative_path


def resolve_cell_relative_path(root: str | Path, entry: str, *, label: str = "entry") -> Path:
    """Resolve a cell-local data path after rejecting absolute and parent escapes."""

    return resolve_relative_path(root, entry, label=label, root_label="cell directory")


def read_cell_json_mapping(root: str | Path, entry: str, *, label: str = "entry") -> Mapping[str, Any]:
    """Read a cell-local JSON object after applying data-view path rules."""

    path = resolve_cell_relative_path(root, entry, label=label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


__all__ = ["read_cell_json_mapping", "resolve_cell_relative_path"]
