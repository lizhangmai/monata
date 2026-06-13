"""Numerical calculus helpers for sampled waveforms."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import math
from typing import Any, Literal, overload

import numpy as np

from monata.units import Quantity, Unit, UnitArray, UnitError, unit_value

StencilSide = Literal["centered", "centred", "forward", "backward"]


def exact_finite_difference_coefficients(
    derivative_order: int,
    offsets: Sequence[Any],
    *,
    evaluation_offset: Any = 0,
) -> tuple[Fraction, ...]:
    """Return exact rational finite-difference weights for a local stencil."""

    order = _positive_int(derivative_order, "derivative_order")
    offset_values = tuple(_fraction_value(offset, "offset") for offset in offsets)
    if len(offset_values) <= order:
        raise ValueError("offsets must provide more points than derivative_order")
    if len(set(offset_values)) != len(offset_values):
        raise ValueError("offsets must be unique")

    origin = _fraction_value(evaluation_offset, "evaluation_offset")
    shifted_offsets = tuple(offset - origin for offset in offset_values)
    size = len(shifted_offsets)
    matrix = [[offset**power for offset in shifted_offsets] for power in range(size)]
    rhs = [Fraction(0) for _ in range(size)]
    rhs[order] = Fraction(math.factorial(order), 1)
    return _solve_fraction_linear_system(matrix, rhs)


@overload
def finite_difference_stencil(
    derivative_order: int,
    accuracy_order: int,
    *,
    side: StencilSide = "centered",
    exact: Literal[False] = False,
) -> tuple[tuple[int, ...], np.ndarray]: ...


@overload
def finite_difference_stencil(
    derivative_order: int,
    accuracy_order: int,
    *,
    side: StencilSide = "centered",
    exact: Literal[True],
) -> tuple[tuple[int, ...], tuple[Fraction, ...]]: ...


def finite_difference_stencil(
    derivative_order: int,
    accuracy_order: int,
    *,
    side: StencilSide = "centered",
    exact: bool = False,
) -> tuple[tuple[int, ...], tuple[Fraction, ...] | np.ndarray]:
    """Return preset finite-difference offsets and weights for a uniform grid."""

    order = _positive_int(derivative_order, "derivative_order")
    accuracy = _positive_int(accuracy_order, "accuracy_order")
    offsets = _finite_difference_offsets(order, accuracy, side)
    weights = (
        exact_finite_difference_coefficients(order, offsets)
        if exact
        else finite_difference_coefficients(order, offsets)
    )
    return offsets, weights


def finite_difference_coefficients(derivative_order: int, offsets: Sequence[float]) -> np.ndarray:
    """Return finite-difference weights for a derivative at offset zero."""

    order = _positive_int(derivative_order, "derivative_order")
    offset_values = np.asarray(tuple(offsets), dtype=float)
    if offset_values.ndim != 1 or offset_values.size <= order:
        raise ValueError("offsets must provide more points than derivative_order")
    if np.unique(offset_values).size != offset_values.size:
        raise ValueError("offsets must be unique")

    powers = np.arange(offset_values.size, dtype=float)
    matrix = offset_values[np.newaxis, :] ** powers[:, np.newaxis]
    rhs = np.zeros(offset_values.size, dtype=float)
    rhs[order] = math.factorial(order)
    return np.linalg.solve(matrix, rhs)


def simple_derivative(x: Any, values: Any) -> tuple[np.ndarray | UnitArray, np.ndarray | UnitArray]:
    """Return first-order slopes between adjacent samples."""

    x_unit = _unit_metadata(x)
    value_unit = _unit_metadata(values)
    x_values, y_values = _sampled_vectors(x, values)
    dx = np.diff(x_values)
    if np.any(dx == 0):
        raise ValueError("x samples must be unique")
    sample_x = x_values[:-1]
    slope = np.diff(y_values) / dx
    if x_unit is not None:
        sample_x = UnitArray(sample_x, x_unit)
    return sample_x, _derivative_result(slope, value_unit=value_unit, x_unit=x_unit, order=1)


def finite_difference_derivative(
    x: Any,
    values: Any,
    *,
    derivative_order: int = 1,
    accuracy_order: int = 4,
) -> np.ndarray | UnitArray:
    """Differentiate sampled values using local finite-difference stencils."""

    order = _positive_int(derivative_order, "derivative_order")
    accuracy = _positive_int(accuracy_order, "accuracy_order")
    x_unit = _unit_metadata(x)
    value_unit = _unit_metadata(values)
    x_values, y_values = _sampled_vectors(x, values)
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("x samples must be strictly increasing")

    stencil_size = order + accuracy
    if stencil_size % 2 == 0:
        stencil_size += 1
    if x_values.size < stencil_size:
        raise ValueError("not enough samples for requested derivative and accuracy order")

    half_window = stencil_size // 2
    result = np.empty(y_values.shape, dtype=np.result_type(y_values, float))
    for index in range(x_values.size):
        start = min(max(index - half_window, 0), x_values.size - stencil_size)
        indices = np.arange(start, start + stencil_size)
        offsets = x_values[indices] - x_values[index]
        weights = finite_difference_coefficients(order, offsets)
        result[index] = np.dot(weights, y_values[indices])
    return _derivative_result(result, value_unit=value_unit, x_unit=x_unit, order=order)


def cumulative_trapezoidal_integral(
    x: Any,
    values: Any,
    *,
    initial: Any = 0,
) -> np.ndarray | UnitArray:
    """Return cumulative trapezoidal integrals for sampled values."""

    x_unit = _unit_metadata(x)
    value_unit = _unit_metadata(values)
    x_values, y_values = _sampled_vectors(x, values)
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("x samples must be strictly increasing")

    result_unit = _integral_unit(value_unit=value_unit, x_unit=x_unit)
    initial_value = _integral_initial_value(initial, result_unit)
    increments = 0.5 * (y_values[1:] + y_values[:-1]) * np.diff(x_values)
    result = np.empty(y_values.shape, dtype=np.result_type(y_values, increments, initial_value, float))
    result[0] = initial_value
    result[1:] = initial_value + np.cumsum(increments)
    return UnitArray(result, result_unit) if result_unit is not None else result


def trapezoidal_integral(x: Any, values: Any) -> Any:
    """Return the definite trapezoidal integral over sampled values."""

    result = cumulative_trapezoidal_integral(x, values)
    return result[-1]


def _derivative_result(
    values: np.ndarray,
    *,
    value_unit: Unit | None,
    x_unit: Unit | None,
    order: int,
) -> np.ndarray | UnitArray:
    if value_unit is None and x_unit is None:
        return values
    result_unit = value_unit if value_unit is not None else unit_value(1).unit
    if x_unit is not None:
        result_unit = result_unit / (x_unit**order)
    return UnitArray(values, result_unit)


def _integral_unit(*, value_unit: Unit | None, x_unit: Unit | None) -> Unit | None:
    if value_unit is None and x_unit is None:
        return None
    result_unit = value_unit if value_unit is not None else unit_value(1).unit
    if x_unit is not None:
        result_unit = result_unit * x_unit
    return result_unit


def _integral_initial_value(initial: Any, result_unit: Unit | None) -> Any:
    if isinstance(initial, Quantity):
        if result_unit is None:
            raise ValueError("initial integral value has units but x and values are unitless")
        try:
            return initial.to(result_unit).value
        except UnitError as exc:
            raise ValueError("initial integral value must use integral-compatible units") from exc
    if isinstance(initial, UnitArray):
        if result_unit is None:
            raise ValueError("initial integral value has units but x and values are unitless")
        try:
            converted = initial.to(result_unit)
        except UnitError as exc:
            raise ValueError("initial integral value must use integral-compatible units") from exc
        values = np.asarray(converted.values)
        if values.ndim == 0:
            return values.item()
        if values.shape == (1,):
            return values[0]
        raise ValueError("initial integral value must be scalar")
    values = np.asarray(initial)
    if values.ndim == 0:
        return values.item()
    if values.shape == (1,):
        return values[0]
    raise ValueError("initial integral value must be scalar")


def _unit_metadata(value: Any) -> Unit | None:
    if isinstance(value, UnitArray):
        return value.unit
    return None


def _sampled_vectors(x: Any, values: Any) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(values)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("sampled calculus helpers require one-dimensional inputs")
    if x_values.shape != y_values.shape:
        raise ValueError("x and values must have the same shape")
    if x_values.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(x_values)):
        raise ValueError("x samples must be finite")
    return x_values, y_values


def _fraction_value(value: Any, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    if isinstance(value, Fraction):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be finite") from exc


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_difference_offsets(order: int, accuracy: int, side: StencilSide) -> tuple[int, ...]:
    if side in {"centered", "centred"}:
        if accuracy % 2:
            raise ValueError("centered finite-difference accuracy_order must be even")
        window = accuracy // 2
        offsets = tuple(range(-window, window + 1))
    elif side == "forward":
        offsets = tuple(range(order + accuracy))
    elif side == "backward":
        offsets = tuple(reversed(range(-(order + accuracy) + 1, 1)))
    else:
        raise ValueError("finite-difference side must be centered, forward, or backward")
    if len(offsets) <= order:
        raise ValueError("finite-difference stencil must provide more points than derivative_order")
    return offsets


def _solve_fraction_linear_system(
    matrix: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    size = len(rhs)
    rows = [list(matrix[row_index]) + [rhs[row_index]] for row_index in range(size)]
    for column in range(size):
        pivot_index = next(
            (row_index for row_index in range(column, size) if rows[row_index][column] != 0),
            None,
        )
        if pivot_index is None:
            raise ValueError("offsets produce a singular finite-difference system")
        if pivot_index != column:
            rows[column], rows[pivot_index] = rows[pivot_index], rows[column]

        pivot = rows[column][column]
        rows[column] = [entry / pivot for entry in rows[column]]
        pivot_row = rows[column]
        for row_index, row in enumerate(rows):
            if row_index == column:
                continue
            factor = row[column]
            if factor == 0:
                continue
            for entry_index, pivot_entry in enumerate(pivot_row):
                row[entry_index] -= factor * pivot_entry

    return tuple(row[-1] for row in rows)


__all__ = [
    "cumulative_trapezoidal_integral",
    "exact_finite_difference_coefficients",
    "finite_difference_coefficients",
    "finite_difference_derivative",
    "finite_difference_stencil",
    "simple_derivative",
    "trapezoidal_integral",
]
