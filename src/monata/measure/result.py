"""Structured scalar measure results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from monata._json import json_safe_dict as _json_safe_dict
from monata.units import Quantity, UnitError, quantity as make_quantity


class MeasureNotFoundError(KeyError):
    """Raised when a measure result cannot be resolved by name."""


@dataclass(frozen=True)
class MeasureResult:
    """A scalar simulator-native or derived measure value."""

    name: str
    value: float | None
    unit: str = ""
    source: str = "simulator_measure"
    analysis: str | None = None
    passed: bool | None = None
    reason: str | None = None
    raw: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", _json_safe_dict(self.metadata))
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))
        if self.reason is not None:
            object.__setattr__(self, "reason", str(self.reason))
        if self.analysis is not None:
            object.__setattr__(self, "analysis", str(self.analysis))
        if self.raw is not None:
            object.__setattr__(self, "raw", str(self.raw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "analysis": self.analysis,
            "passed": self.passed,
            "reason": self.reason,
            "raw": self.raw,
            "metadata": dict(self.metadata),
        }

    def value_with_unit(self) -> float | Quantity:
        value = _scalar_measure_value(self)
        if not self.unit:
            return value
        try:
            return make_quantity(value, self.unit)
        except UnitError:
            return value


class MeasureSet(Mapping[str, MeasureResult]):
    """Mapping-style container for scalar measure results."""

    def __init__(self, measures: Mapping[str, MeasureResult | Mapping[str, Any]] | None = None) -> None:
        self._measures: dict[str, MeasureResult] = {}
        for name, measure in (measures or {}).items():
            if isinstance(measure, MeasureResult):
                item = measure
            else:
                item = MeasureResult(**dict(measure))
            self._measures[str(name)] = item

    def __getitem__(self, name: str) -> MeasureResult:
        try:
            return self._measures[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._measures))
            raise MeasureNotFoundError(f"measure not found: {name}; available: {available}") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._measures)

    def __len__(self) -> int:
        return len(self._measures)

    def value(self, name: str) -> float:
        return _scalar_measure_value(self[name])

    def value_with_unit(self, name: str) -> float | Quantity:
        return self[name].value_with_unit()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: measure.to_dict() for name, measure in self._measures.items()}


def _scalar_measure_value(measure: MeasureResult) -> float:
    if measure.value is None:
        reason = measure.reason or "measure has no scalar value"
        raise ValueError(f"measure has no scalar value: {measure.name}; reason: {reason}")
    return measure.value
