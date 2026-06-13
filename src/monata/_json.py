"""Shared JSON-safe value coercion helpers."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any, *, strict: bool = False) -> Any:
    """Return a recursively JSON-compatible representation of common Monata values."""

    result = _json_safe(value)
    if strict:
        json.dumps(result)
    return result


def json_safe_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): json_safe(value) for key, value in dict(values).items()}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return json_safe_dict(value)
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return [json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value
