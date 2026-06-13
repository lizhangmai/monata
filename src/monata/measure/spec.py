"""Spec evaluation — pass/fail judgment on simulation results."""

from __future__ import annotations

import math
from typing import Any, Callable

from monata.measure.result import MeasureNotFoundError
from monata.units import Quantity, Unit, UnitError, unit as resolve_unit


class _Series:
    """Minimal column/series for SpecDataFrame."""

    def __init__(self, data: list):
        self._data = data

    @property
    def values(self):
        return self._data

    def __eq__(self, other):  # type: ignore[override]
        return _BoolMask([v == other for v in self._data])

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class _BoolMask:
    def __init__(self, data: list[bool]):
        self._data = data


class _Loc:
    def __init__(self, df: "SpecDataFrame"):
        self._df = df

    def __getitem__(self, key):
        mask, col = key
        filtered = [row[col] for row, m in zip(self._df._rows, mask._data) if m]
        return _Series(filtered)


class SpecDataFrame:
    """Lightweight DataFrame returned by SpecTable.evaluate_all."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []
        self.loc = _Loc(self)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, col: str) -> _Series:
        return _Series([row.get(col) for row in self._rows])


class SpecResult:
    def __init__(
        self,
        name: str,
        value: float,
        passed: bool,
        margin: float,
        unit: str = "",
        source: str = "python_metric",
        reason: str | None = None,
    ):
        self.name = name
        self.value = value
        self.passed = passed
        self.margin = margin
        self.unit = unit
        self.source = source
        self.reason = reason


class Spec:
    def __init__(
        self,
        name: str,
        metric_fn: Callable,
        min: Any | None = None,
        max: Any | None = None,
        unit: str = "",
        source: str = "python_metric",
    ):
        self.name = name
        self.metric_fn = metric_fn
        self.min = min
        self.max = max
        self.unit = unit
        self._unit = _spec_unit(unit)
        min_check = _initial_bound_value(self.min, self._unit)
        max_check = _initial_bound_value(self.max, self._unit)
        if min_check is not None and max_check is not None and min_check > max_check:
            raise ValueError(f"spec min must be less than or equal to max: {self.name}")
        self.source = source

    @staticmethod
    def measure(name: str) -> Callable:
        def metric(result):
            return result.measures.value_with_unit(name)

        return metric

    @staticmethod
    def summary(summary: str, key: str) -> Callable:
        def metric(result):
            summary_obj = result.summaries[summary]
            if hasattr(summary_obj, "value_with_unit"):
                return summary_obj.value_with_unit(key)
            if hasattr(summary_obj, "value"):
                return summary_obj.value(key)
            value = summary_obj[key]
            return value

        return metric

    def evaluate(self, result) -> SpecResult:
        value, value_unit, unit_label = _metric_value(self.metric_fn(result), self._unit, self.unit)
        if not math.isfinite(value):
            raise ValueError(f"spec value is not finite: {self.name}")
        passed = True
        margin = float("inf")

        if self.min is not None:
            min_value = _bound_value(self.min, value_unit, "min")
            margin_min = value - min_value
            if margin_min < 0:
                passed = False
            margin = min(margin, margin_min)

        if self.max is not None:
            max_value = _bound_value(self.max, value_unit, "max")
            margin_max = max_value - value
            if margin_max < 0:
                passed = False
            margin = min(margin, margin_max)

        return SpecResult(
            name=self.name,
            value=value,
            passed=passed,
            margin=margin,
            unit=unit_label,
            source=self.source,
        )


class SpecTable:
    def __init__(self):
        self._specs: list[Spec] = []

    def add(self, name: str, metric_fn: Callable, min: Any | None = None, max: Any | None = None, unit: str = ""):
        self._specs.append(Spec(name, metric_fn, min=min, max=max, unit=unit))

    def add_measure(self, name: str, min: Any | None = None, max: Any | None = None, unit: str = ""):
        self._specs.append(
            Spec(
                name,
                Spec.measure(name),
                min=min,
                max=max,
                unit=unit,
                source="simulator_measure",
            )
        )

    def add_summary(
        self,
        name: str,
        summary: str,
        key: str,
        min: Any | None = None,
        max: Any | None = None,
        unit: str = "",
    ):
        self._specs.append(
            Spec(
                name,
                Spec.summary(summary, key),
                min=min,
                max=max,
                unit=unit,
                source="summary",
            )
        )

    def evaluate_all(self, corner_results) -> SpecDataFrame:
        rows = []
        for sim_result in corner_results:
            row = {"corner": sim_result.corner.name if sim_result.corner else None}
            if sim_result.status != "ok":
                for spec in self._specs:
                    row[spec.name] = float("nan")
            else:
                for spec in self._specs:
                    try:
                        sr = spec.evaluate(sim_result)
                    except (MeasureNotFoundError, KeyError):
                        row[spec.name] = float("nan")
                    except ValueError:
                        if spec.source in {"simulator_measure", "summary"}:
                            row[spec.name] = float("nan")
                        else:
                            raise
                    else:
                        row[spec.name] = sr.value
            rows.append(row)
        return SpecDataFrame(rows)

    def evaluate_rows(self, corner_results) -> list[dict]:
        rows = []
        for sim_result in corner_results:
            corner_name = sim_result.corner.name if sim_result.corner else None
            for spec in self._specs:
                if sim_result.status != "ok":
                    rows.append(_spec_row(
                        spec,
                        corner_name,
                        value=None,
                        passed=None,
                        margin=None,
                        source="failed",
                        reason=_failure_reason(sim_result),
                    ))
                    continue
                try:
                    sr = spec.evaluate(sim_result)
                except (MeasureNotFoundError, KeyError) as exc:
                    rows.append(_spec_row(
                        spec,
                        corner_name,
                        value=None,
                        passed=None,
                        margin=None,
                        source="missing",
                        reason=_stable_reason(exc),
                    ))
                except ValueError as exc:
                    source = "missing" if spec.source in {"simulator_measure", "summary"} else spec.source
                    rows.append(_spec_row(
                        spec,
                        corner_name,
                        value=None,
                        passed=None,
                        margin=None,
                        source=source,
                        reason=_stable_reason(exc),
                    ))
                except Exception as exc:
                    rows.append(_spec_row(
                        spec,
                        corner_name,
                        value=None,
                        passed=None,
                        margin=None,
                        source=spec.source,
                        reason=_stable_reason(exc),
                    ))
                else:
                    rows.append(_spec_row(
                        spec,
                        corner_name,
                        value=sr.value,
                        passed=sr.passed,
                        margin=sr.margin,
                        source=sr.source,
                        reason=sr.reason,
                        unit=sr.unit,
                    ))
        return rows

    def worst_corner(self, spec_name: str, corner_results) -> tuple:
        spec = next(s for s in self._specs if s.name == spec_name)
        worst_score = None
        worst_val = None
        worst_corner = None
        for sim_result in corner_results:
            if sim_result.status != "ok":
                continue
            try:
                sr = spec.evaluate(sim_result)
            except (MeasureNotFoundError, KeyError):
                continue
            except ValueError as exc:
                if "not finite" in str(exc) or spec.source in {"simulator_measure", "summary"}:
                    continue
                raise
            if not math.isfinite(sr.value):
                continue
            score = _worst_corner_score(spec, sr)
            if worst_score is None or score < worst_score:
                worst_score = score
                worst_val = sr.value
                worst_corner = sim_result.corner
        return worst_corner, worst_val


def _worst_corner_score(spec: Spec, result: SpecResult) -> float:
    if spec.min is None and spec.max is None:
        return result.value
    return result.margin


def _spec_row(
    spec: Spec,
    corner: str | None,
    *,
    value: float | None,
    passed: bool | None,
    margin: float | None,
    source: str,
    reason: str | None,
    unit: str | None = None,
) -> dict:
    return {
        "name": spec.name,
        "corner": corner,
        "value": value,
        "passed": passed,
        "margin": margin,
        "unit": spec.unit if unit is None else unit,
        "source": source,
        "reason": reason,
    }


def _failure_reason(sim_result) -> str:
    return str(sim_result.metadata.get("reason") or sim_result.error_message or "simulation_failed")


def _stable_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _spec_unit(unit_label: str) -> Unit | None:
    if not unit_label:
        return None
    try:
        return resolve_unit(unit_label)
    except UnitError:
        return None


def _metric_value(value: Any, spec_unit: Unit | None, unit_label: str) -> tuple[float, Unit | None, str]:
    if isinstance(value, Quantity):
        if spec_unit is None and unit_label:
            raise ValueError(f"unknown spec unit: {unit_label}")
        value_unit = spec_unit or value.unit
        try:
            converted = value.to(value_unit)
        except UnitError as exc:
            raise ValueError("spec value must use value-compatible units") from exc
        return converted.value, value_unit, unit_label or value_unit.symbol
    return float(value), spec_unit, unit_label


def _bound_value(bound: Any, value_unit: Unit | None, label: str) -> float:
    if isinstance(bound, Quantity):
        if value_unit is None:
            raise ValueError(f"spec {label} requires a spec unit or quantity metric")
        try:
            return bound.to(value_unit).value
        except UnitError as exc:
            raise ValueError(f"spec {label} must use value-compatible units") from exc
    return float(bound)


def _initial_bound_value(bound: Any | None, spec_unit: Unit | None) -> float | None:
    if bound is None:
        return None
    if isinstance(bound, Quantity):
        if spec_unit is None:
            return None
        try:
            return bound.to(spec_unit).value
        except UnitError as exc:
            raise ValueError("spec bound must use value-compatible units") from exc
    return float(bound)
