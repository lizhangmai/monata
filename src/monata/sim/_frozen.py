"""Shared immutable public-value helpers for simulation result objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn, SupportsIndex

import numpy as np


class FrozenPayload(dict[str, Any]):
    """A JSON-shaped, recursively read-only mapping."""

    def _readonly(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("mapping is read-only")

    def __setitem__(self, key: str, value: Any) -> NoReturn:
        self._readonly(key, value)

    def __delitem__(self, key: str) -> NoReturn:
        self._readonly(key)

    def clear(self) -> NoReturn:
        self._readonly()

    def pop(self, key: str, default: Any = None) -> NoReturn:
        self._readonly(key, default)

    def popitem(self) -> NoReturn:
        self._readonly()

    def setdefault(self, key: str, default: Any = None) -> NoReturn:
        self._readonly(key, default)

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._readonly(*args, **kwargs)

    def __ior__(self, other: Any) -> "FrozenPayload":
        self._readonly(other)
        return self


class FrozenList(list[Any]):
    """A list-shaped value that preserves equality semantics without mutation."""

    def _readonly(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("list is read-only")

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        self._readonly(key, value)

    def __delitem__(self, key: Any) -> NoReturn:
        self._readonly(key)

    def __iadd__(self, values: Any) -> "FrozenList":
        self._readonly(values)
        return self

    def __imul__(self, value: Any) -> "FrozenList":
        self._readonly(value)
        return self

    def append(self, value: Any) -> NoReturn:
        self._readonly(value)

    def clear(self) -> NoReturn:
        self._readonly()

    def extend(self, values: Any) -> NoReturn:
        self._readonly(values)

    def insert(self, index: SupportsIndex, value: Any) -> NoReturn:
        self._readonly(index, value)

    def pop(self, index: SupportsIndex = -1) -> NoReturn:
        self._readonly(index)

    def remove(self, value: Any) -> NoReturn:
        self._readonly(value)

    def reverse(self) -> NoReturn:
        self._readonly()

    def sort(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._readonly(*args, **kwargs)


def frozen_array(value: Any) -> np.ndarray:
    """Return an owned, read-only numpy array."""

    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def frozen_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a recursively frozen mapping for public result metadata."""

    return FrozenPayload({key: frozen_public_value(value) for key, value in dict(values or {}).items()})


def frozen_public_value(value: Any) -> Any:
    """Freeze JSON-like containers and numpy arrays exposed by public results."""

    if isinstance(value, np.ndarray):
        return frozen_array(value)
    if isinstance(value, Mapping):
        return FrozenPayload({key: frozen_public_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(frozen_public_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(frozen_public_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(frozen_public_value(item) for item in value)
    return value


__all__ = [
    "FrozenList",
    "FrozenPayload",
    "frozen_array",
    "frozen_mapping",
    "frozen_public_value",
]
