"""Statistical measurement utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from monata.units import Quantity, Unit, UnitArray, UnitError, unit_value


def histogram(values: Any, bins: int = 50) -> tuple[np.ndarray | UnitArray, np.ndarray]:
    data = _numeric_vector(values, "values")
    value_unit = _value_unit(values)
    counts, bin_edges = np.histogram(data, bins=bins)
    if value_unit is None:
        return bin_edges, counts
    return UnitArray(bin_edges, value_unit), counts


def sigma_yield(values: Any, spec_min: Any | None = None, spec_max: Any | None = None) -> float:
    data = _numeric_vector(values, "values")
    value_unit = _value_unit(values)
    min_value = None if spec_min is None else _value_scalar(spec_min, value_unit, "spec_min")
    max_value = None if spec_max is None else _value_scalar(spec_max, value_unit, "spec_max")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("spec_min must be less than or equal to spec_max")
    mask = np.ones(len(data), dtype=bool)
    if min_value is not None:
        mask &= data >= min_value
    if max_value is not None:
        mask &= data <= max_value
    return float(np.sum(mask)) / len(data)


def worst_case(values: Any) -> tuple[float | Quantity, float | Quantity]:
    data = _numeric_vector(values, "values")
    value_unit = _value_unit(values)
    return _unit_result(float(np.min(data)), value_unit), _unit_result(float(np.max(data)), value_unit)


def sensitivity(param_values: Any, metric_values: Any) -> float | Quantity:
    params = _numeric_vector(param_values, "param_values")
    metrics = _numeric_vector(metric_values, "metric_values")
    if params.shape != metrics.shape:
        raise ValueError("param_values and metric_values must have the same shape")
    result_unit = _sensitivity_unit(_value_unit(metric_values), _value_unit(param_values))
    if len(params) < 2:
        return _unit_result(0.0, result_unit)
    coeffs = np.polyfit(params, metrics, 1)
    return _unit_result(float(coeffs[0]), result_unit)


def _numeric_vector(values: Any, label: str) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if data.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{label} must be finite")
    return data


def _value_unit(values: Any) -> Unit | None:
    if isinstance(values, UnitArray):
        return values.unit
    return None


def _value_scalar(value: Any, value_unit: Unit | None, label: str) -> float:
    if isinstance(value, Quantity) and value_unit is not None:
        try:
            return value.to(value_unit).value
        except UnitError as exc:
            raise ValueError(f"{label} must use value-compatible units") from exc
    return float(value)


def _unit_result(value: float, value_unit: Unit | None) -> float | Quantity:
    if value_unit is None:
        return value
    return Quantity(value, value_unit)


def _sensitivity_unit(metric_unit: Unit | None, param_unit: Unit | None) -> Unit | None:
    if metric_unit is None and param_unit is None:
        return None
    numerator = metric_unit if metric_unit is not None else unit_value(1).unit
    denominator = param_unit if param_unit is not None else unit_value(1).unit
    return numerator / denominator
